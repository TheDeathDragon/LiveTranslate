"""Opt-in provisional captions; final work is FIFO, preview work is replaceable."""
from collections import deque
from dataclasses import dataclass
import logging
import queue
import threading
import time

log = logging.getLogger("LiveTranslate.Preview")


class PreviewQueue(queue.Queue):
    """Keep every final job and at most one pending preview. No audio is dropped."""

    def _init(self, maxsize):
        self.queue = deque()
        self.preview = None

    def _qsize(self):
        return len(self.queue) + (self.preview is not None)

    def _put(self, item):
        if item is not None and item[0] == "preview":
            if self.preview is not None:
                self.unfinished_tasks -= 1
            self.preview = item
        else:
            self.queue.append(item)

    def _get(self):
        if self.queue:
            return self.queue.popleft()
        item, self.preview = self.preview, None
        return item

    def discard_preview(self):
        with self.mutex:
            if self.preview is not None:
                self.preview = None
                self.unfinished_tasks -= 1


@dataclass(frozen=True)
class AudioFrame:
    session: int
    utterance: int
    revision: int
    audio: object
    final: bool
    timestamp: float

    @property
    def key(self):
        return self.session, self.utterance


class LivePreview:
    def __init__(self, publish, usage):
        self.publish = publish
        self.usage = usage
        self.lock = threading.RLock()
        self.session = 0
        self.utterance = 0
        self.revision = 0
        self.closed = set()
        self.versions = {}
        self.cache = {}
        self.last_finalized = -1
        self.jobs = PreviewQueue()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.jobs = PreviewQueue()
        self.thread = threading.Thread(target=self._translate_loop, daemon=True)
        self.thread.start()

    def invalidate(self):
        with self.lock:
            self.session += 1
            self.utterance = 0
            self.revision = 0
            self.closed.clear()
            self.versions.clear()
            self.cache.clear()
            self.last_finalized = -1
            self.jobs.discard_preview()
            return self.session

    def frame(self, audio, final=False):
        with self.lock:
            self.revision += 1
            frame = AudioFrame(self.session, self.utterance, self.revision,
                               audio.copy(), final, time.monotonic())
            if final:
                self.closed.add(frame.key)
                self.utterance += 1
                self.revision = 0
            return frame

    def valid(self, frame, text=None):
        with self.lock:
            if frame.session != self.session:
                return False
            if frame.final:
                return frame.utterance > self.last_finalized
            return (frame.key not in self.closed and
                    (text is None or self.versions.get(frame.key) == (frame.revision, text)))

    def submit(self, frame, text, language, asr_ms, translator, target,
               translate_preview=False, extra_languages=()):
        with self.lock:
            if not self.valid(frame):
                return
            previous = self.versions.get(frame.key)
            if not frame.final and previous and previous[1] == text:
                return
            self.versions[frame.key] = (frame.revision, text)
            self.publish(frame, text, {}, asr_ms, 0, "pending")
            if not text and not frame.final:
                return
            if not frame.final and not translate_preview and language != target:
                return
            job = (frame, text, language, asr_ms, translator, target, tuple(extra_languages))
            self.jobs.put(("final" if frame.final else "preview", job))

    def _finish(self, frame, text, translations, asr_ms, tl_ms):
        with self.lock:
            if not self.valid(frame):
                return
            self.publish(frame, text, translations, asr_ms, tl_ms, "final")
            self.last_finalized = frame.utterance
            self.closed.discard(frame.key)
            self.versions.pop(frame.key, None)
            self.cache.pop(frame.key, None)

    def _translate_loop(self):
        while True:
            item = self.jobs.get()
            try:
                if item is None:
                    return
                self._translate(*item[1])
            except Exception:
                log.exception("Preview translation worker error")
            finally:
                self.jobs.task_done()

    def _translate(self, frame, text, language, asr_ms, translator, target, extras):
        if not self.valid(frame, text):
            return
        if not text:
            self._finish(frame, text, {}, asr_ms, 0)
            return
        translations = {}
        elapsed = 0
        try:
            if language == target:
                translations[target] = text
            else:
                with self.lock:
                    cached = self.cache.get(frame.key)
                if frame.final and cached and cached[:3] == (text, language, target):
                    translations[target], elapsed = cached[3:]
                else:
                    # Preview calls must never mutate the final translator's context.
                    engine = translator.with_target_language(target)
                    engine._client = engine._client.copy(max_retries=0)
                    if frame.final:
                        engine._context_turns = translator._context_turns
                        engine._history = list(translator._history)
                    started = time.perf_counter()
                    engine._last_prompt_tokens = engine._last_completion_tokens = 0
                    try:
                        translated = ""
                        for partial in engine.translate_iter(text, language):
                            translated = partial
                            with self.lock:
                                if self.valid(frame, text):
                                    self.publish(frame, text, {target: partial}, asr_ms,
                                                 (time.perf_counter() - started) * 1000, "stream")
                        translations[target] = translated or ""
                    finally:
                        self.usage(*engine.last_usage)
                    elapsed = (time.perf_counter() - started) * 1000
            if frame.final:
                for lang in extras:
                    engine = translator.with_target_language(lang)
                    engine._client = engine._client.copy(max_retries=0)
                    engine._last_prompt_tokens = engine._last_completion_tokens = 0
                    try:
                        translations[lang] = engine.translate(text, language)
                    finally:
                        self.usage(*engine.last_usage)
                with self.lock:
                    if self.valid(frame) and language != target and translations.get(target):
                        translator._append_history(text, translations[target])
                    self._finish(frame, text, translations, asr_ms, elapsed)
            else:
                with self.lock:
                    if self.valid(frame, text):
                        if translations.get(target):
                            self.cache[frame.key] = (text, language, target,
                                                     translations[target], elapsed)
                        self.publish(frame, text, translations, asr_ms, elapsed, "preview")
        except Exception:
            log.exception("Caption translation failed (no retry)")
            if frame.final:
                self._finish(frame, text, {}, asr_ms, elapsed)

    def stop(self):
        if not self.thread:
            return
        self.jobs.discard_preview()
        self.jobs.put(None)
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise TimeoutError("Preview translation worker did not stop")
        self.thread = None
