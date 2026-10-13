import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from live_preview import LivePreview, PreviewQueue


class TranslatorStub:
    def __init__(self, calls=None, gate=None, fail=False):
        self.calls = calls if calls is not None else []
        self.gate = gate
        self.fail = fail
        self._context_turns = 2
        self._history = []
        self.last_usage = (2, 3)
        self._client = SimpleNamespace(copy=lambda **kwargs: self._client)

    def with_target_language(self, target):
        return TranslatorStub(self.calls, self.gate, self.fail)

    def translate_iter(self, text, language):
        self.calls.append(text)
        if self.gate:
            assert self.gate.wait(2)
        if self.fail:
            raise RuntimeError("test failure")
        yield "译文:" + text
        self._append_history(text, "译文:" + text)

    def _append_history(self, text, result):
        if self._context_turns:
            self._history.append((text, result))

    def translate(self, text, language):
        return list(self.translate_iter(text, language))[-1]


@pytest.fixture
def pipeline():
    events, usage = [], []
    p = LivePreview(lambda *args: events.append(args), lambda *args: usage.append(args))
    p.start()
    yield p, events, usage
    p.stop()


def drain(p):
    deadline = time.monotonic() + 3
    while p.jobs.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not p.jobs.unfinished_tasks


def submit(p, engine, text, final=False, enabled=True, language="en"):
    frame = p.frame(np.zeros(16000, dtype=np.float32), final)
    p.submit(frame, text, language, 70, engine, "zh", enabled)
    return frame


def test_final_fifo_and_latest_pending_preview():
    q = PreviewQueue()
    q.put(("preview", 1))
    q.put(("final", "a"))
    q.put(("preview", 2))
    q.put(("final", "b"))
    assert q.qsize() == 3
    assert [q.get(), q.get(), q.get()] == [("final", "a"), ("final", "b"), ("preview", 2)]
    for _ in range(3):
        q.task_done()
    assert q.unfinished_tasks == 0


def test_audio_snapshot_is_independent_and_closes_only_its_utterance(pipeline):
    p, _, _ = pipeline
    audio = np.ones(8)
    frame = p.frame(audio)
    audio[:] = 9
    assert np.all(frame.audio == 1)
    final = p.frame(audio, final=True)
    following = p.frame(audio)
    assert frame.key == final.key and following.utterance == final.utterance + 1
    assert not p.valid(frame) and p.valid(final) and p.valid(following)


@pytest.mark.parametrize("text,language", [("An unfinished sentence", "en"), ("これはまだ途中", "ja")])
def test_single_sentence_preview_and_identical_final_reuse(pipeline, text, language):
    p, events, usage = pipeline
    engine = TranslatorStub()
    submit(p, engine, text, language=language)
    drain(p)
    submit(p, engine, text, language=language)
    drain(p)
    assert engine.calls == [text]
    assert not engine._history
    final = submit(p, engine, text, final=True, language=language)
    drain(p)
    assert engine.calls == [text] and len(usage) == 1
    assert engine._history == [(text, "译文:" + text)]
    assert len([e for e in events if e[-1] == "final"]) == 1
    p.submit(final, text, language, 70, engine, "zh", True)
    drain(p)
    assert len([e for e in events if e[-1] == "final"]) == 1
    assert not p.cache and not p.versions and not p.closed


def test_cloud_preview_requires_explicit_opt_in(pipeline):
    p, events, _ = pipeline
    engine = TranslatorStub()
    submit(p, engine, "hello", enabled=False)
    drain(p)
    assert not engine.calls and events[-1][-1] == "pending"
    submit(p, engine, "hello", final=True, enabled=False)
    drain(p)
    assert engine.calls == ["hello"]


def test_slow_old_result_cannot_overwrite_new_text(pipeline):
    p, events, usage = pipeline
    gate = threading.Event()
    engine = TranslatorStub(gate=gate)
    submit(p, engine, "old")
    deadline = time.monotonic() + 1
    while not engine.calls and time.monotonic() < deadline:
        time.sleep(0.005)
    submit(p, engine, "middle")
    submit(p, engine, "latest")
    assert p.jobs.qsize() == 1
    gate.set()
    drain(p)
    assert engine.calls == ["old", "latest"]
    assert len(usage) == 2
    assert not [e for e in events if e[1] == "old" and e[-1] in ("stream", "preview")]
    assert events[-1][1] == "latest"


def test_invalidation_discards_late_results_and_history(pipeline):
    p, events, usage = pipeline
    gate = threading.Event()
    engine = TranslatorStub(gate=gate)
    submit(p, engine, "old", final=True)
    deadline = time.monotonic() + 1
    while not engine.calls and time.monotonic() < deadline:
        time.sleep(0.005)
    p.invalidate()
    gate.set()
    drain(p)
    assert len(usage) == 1 and not engine._history
    assert not [e for e in events if e[-1] == "final"]


def test_translation_error_has_no_retry_and_does_not_confirm_preview(pipeline):
    p, events, usage = pipeline
    engine = TranslatorStub(fail=True)
    submit(p, engine, "broken", final=True)
    drain(p)
    assert engine.calls == ["broken"] and len(usage) == 1
    assert events[-1][-1] == "final" and events[-1][2] == {}


def test_empty_final_clears_without_translation(pipeline):
    p, events, _ = pipeline
    engine = TranslatorStub()
    submit(p, engine, "", final=True)
    drain(p)
    assert events[-1][1:3] == ("", {}) and events[-1][-1] == "final"
    assert not engine.calls


def test_empty_final_cannot_overtake_an_inflight_final(pipeline):
    p, events, _ = pipeline
    gate = threading.Event()
    engine = TranslatorStub(gate=gate)
    submit(p, engine, "first", final=True)
    deadline = time.monotonic() + 1
    while not engine.calls and time.monotonic() < deadline:
        time.sleep(.005)
    submit(p, engine, "", final=True)
    gate.set()
    drain(p)
    assert [e[1] for e in events if e[-1] == "final"] == ["first", ""]


def test_sdk_retry_policy_is_disabled_only_for_preview_pipeline(pipeline):
    import httpx
    from openai import OpenAI
    from translator import Translator
    requests = []

    def error_response(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "synthetic failure"}})

    engine = Translator("https://test.invalid/v1", "test", "test", "zh", streaming=False)
    engine._client.close()
    engine._client = OpenAI(base_url="https://test.invalid/v1", api_key="test", max_retries=2,
                           http_client=httpx.Client(transport=httpx.MockTransport(error_response)))
    try:
        p, events, _ = pipeline
        submit(p, engine, "test", final=True)
        drain(p)
        assert len(requests) == 1 and events[-1][-1] == "final"
        assert engine._client.max_retries == 2
    finally:
        engine._client.close()
