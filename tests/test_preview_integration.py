import os
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from main import LiveTranslateApp
from live_preview import LivePreview


def app_fixture():
    app = LiveTranslateApp.__new__(LiveTranslateApp)
    events, writes = [], []
    app._overlay = app._subwin = None
    app._target_language = "zh"
    app._msg_id = app._asr_count = app._translate_count = 0
    app._total_prompt_tokens = app._total_completion_tokens = 0
    app._preview = LivePreview(app._publish_caption, app._preview_usage)
    app._preview_translation = False
    app._panel = SimpleNamespace(get_settings=lambda: {"asr_language": "en"})
    app._translator = object()
    app._transcript = SimpleNamespace(
        write_original=lambda *a: writes.append(("original", a)),
        write_translation=lambda *a: writes.append(("translation", a)),
        finalize_no_translation=lambda *a: writes.append(("no_translation", a)))
    app._run_asr = lambda *a, **k: ({"text": "Hello", "language": "en"}, 70)
    app._preview.publish = lambda *a: events.append(a)
    return app, events, writes


def test_preview_does_not_persist_or_count_a_final():
    app, events, writes = app_fixture()
    frame = app._preview.frame(np.zeros(16000))
    app._process_preview_audio(frame)
    assert events[-1][1] == "Hello" and not writes
    assert app._msg_id == app._asr_count == app._translate_count == 0


def test_final_persistence_and_actual_usage_are_separate():
    app, _, writes = app_fixture()
    frame = app._preview.frame(np.zeros(16000), True)
    app._publish_caption(frame, "Hello", {}, 70, 0, "pending")
    app._publish_caption(frame, "Hello", {"zh": "你好"}, 70, 100, "stream")
    assert not writes
    app._publish_caption(frame, "Hello", {"zh": "你好"}, 70, 100, "final")
    app._preview_usage(5, 8)
    assert app._msg_id == app._asr_count == app._translate_count == 1
    assert [kind for kind, _ in writes] == ["original", "translation"]
    assert (app._total_prompt_tokens, app._total_completion_tokens) == (5, 8)


def test_asr_result_after_switch_cannot_enter_translation():
    app, events, _ = app_fixture()
    frame = app._preview.frame(np.zeros(16000))

    def stale(*args):
        app._preview.invalidate()
        return {"text": "late", "language": "en"}, 70

    app._run_asr = stale
    app._process_preview_audio(frame)
    assert not events and not app._preview.jobs.qsize()


def test_language_and_noise_filter_clear_final_preview():
    app, events, writes = app_fixture()
    frame = app._preview.frame(np.zeros(32000), True)
    app._run_asr = lambda *a: ({"text": "The", "language": "en"}, 70)
    app._preview.start()
    try:
        app._process_preview_audio(frame)
        app._preview.jobs.join()
    finally:
        app._preview.stop()
    assert events[-1][1] == "" and events[-1][-1] == "final"
    assert not writes


def test_pause_invalidates_audio_and_late_translation():
    app, _, _ = app_fixture()
    app._live_preview = True
    app._vad_lock = threading.Lock()
    resets = []
    app._vad = SimpleNamespace(_reset=lambda: resets.append(True))
    frame = app._preview.frame(np.ones(16000))
    app.pause()
    assert app._paused and resets and not app._preview.valid(frame)


def test_preview_interval_allows_more_initial_context():
    app, _, _ = app_fixture()
    app._config = {"audio": {"sample_rate": 16000, "chunk_duration": .032}}
    app._running = app._live_preview = app._asr_ready = True
    app._paused = False
    app._vad_lock = threading.Lock()
    app._last_interim_check_time = 0
    app._interim_interval = 1.5
    app._vad = SimpleNamespace(last_confidence=1, process_chunk=lambda x: None)
    durations = iter([1.0, 1.5])
    app._vad.peek_buffer = lambda: (np.zeros(24000), next(durations))
    calls = []

    def audio(**kwargs):
        return np.zeros(512), None

    def enqueue(*args):
        calls.append(args)
        app._running = False

    app._audio = SimpleNamespace(get_audio=audio)
    app._enqueue_asr = enqueue
    app._capture_loop()
    assert len(calls) == 1 and calls[0][0] == "preview"
