import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from audio_capture import AudioCapture
from vad_processor import VADProcessor


log = logging.getLogger("LiveTranslate.Pipeline")


@dataclass
class AudioLevelEvent:
    rms: float
    vad_confidence: float
    mic_rms: float | None = None


@dataclass
class SpeechSegmentEvent:
    kind: str
    audio_seconds: float


@dataclass
class ASRTextEvent:
    text: str
    source_lang: str
    asr_ms: float
    interim: bool = False


@dataclass
class ASRErrorEvent:
    kind: str
    error: Exception


@dataclass
class PipelineStatsEvent:
    state: str
    queue_size: int = 0


class PipelineController:
    """Owns audio capture, VAD, ASR queueing, and incremental ASR coordination."""

    _pysbd_cache = {}

    def __init__(
        self,
        config: dict,
        asr_runner: Callable[..., tuple[dict | None, float]],
        asr_ready: Callable[[], bool],
        asr_language: Callable[[], str],
        audio_level_callback: Callable[[AudioLevelEvent], None] | None = None,
        speech_segment_callback: Callable[[SpeechSegmentEvent], None] | None = None,
        asr_text_callback: Callable[[ASRTextEvent], None] | None = None,
        asr_error_callback: Callable[[ASRErrorEvent], None] | None = None,
        stats_callback: Callable[[PipelineStatsEvent], None] | None = None,
    ):
        self._config = config
        self._sample_rate = config["audio"]["sample_rate"]
        self._chunk_duration = config["audio"]["chunk_duration"]

        self._asr_runner = asr_runner
        self._asr_ready = asr_ready
        self._asr_language = asr_language
        self._audio_level_callback = audio_level_callback
        self._speech_segment_callback = speech_segment_callback
        self._asr_text_callback = asr_text_callback
        self._asr_error_callback = asr_error_callback
        self._stats_callback = stats_callback

        self._audio = AudioCapture(
            device=config["audio"].get("device"),
            sample_rate=self._sample_rate,
            chunk_duration=self._chunk_duration,
        )
        self._vad = VADProcessor(
            sample_rate=self._sample_rate,
            threshold=config["asr"]["vad_threshold"],
            min_speech_duration=config["asr"]["min_speech_duration"],
            max_speech_duration=config["asr"]["max_speech_duration"],
            chunk_duration=self._chunk_duration,
        )

        self._vad_lock = threading.Lock()
        self._asr_queue = queue.Queue(maxsize=16)
        self._capture_thread = None
        self._asr_thread = None
        self._running = False
        self._paused = False

        self._incremental_enabled = False
        self._interim_interval = 2.0
        self._interim_pending = ""
        self._interim_active = False
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self):
        if self._running:
            return
        self._asr_queue = queue.Queue(maxsize=16)
        self._paused = False
        self._audio.start()
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._capture_thread.start()
        self._asr_thread.start()
        self._emit_stats("started")
        log.info("Pipeline started (capture + ASR threads)")

    def stop(self):
        self._running = False
        self._audio.stop()
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None

        self._asr_queue.put(None)
        if self._asr_thread:
            self._asr_thread.join(timeout=10)
            if self._asr_thread.is_alive():
                log.warning("ASR thread still running after timeout, proceeding")
            self._asr_thread = None

        # Flush after the worker threads have stopped to avoid concurrent VAD access.
        if self._interim_active:
            with self._vad_lock:
                remaining = self._vad.force_flush()
            if remaining is not None and self._is_asr_ready():
                self._process_interim_final(remaining)
        else:
            with self._vad_lock:
                remaining = self._vad.flush()
            if remaining is not None and self._is_asr_ready():
                self._process_segment(remaining)

        self._reset_interim_state()
        self._emit_stats("stopped")
        log.info("Pipeline stopped")

    def pause(self):
        self._paused = True
        self._reset_interim_state()
        self._emit_audio_level(0.0, 0.0)
        self._emit_stats("paused")
        log.info("Pipeline paused")

    def resume(self):
        self._paused = False
        self._emit_stats("resumed")
        log.info("Pipeline resumed")

    def apply_settings(self, settings: dict):
        self._vad.update_settings(settings)

        if "audio_device" in settings:
            old_device = self._audio.device_name
            self._audio.set_device(settings["audio_device"])
            if old_device != settings.get("audio_device"):
                with self._vad_lock:
                    self._vad.flush()
                    self._vad.reset()
                self._reset_interim_state()
                self._emit_audio_level(0.0, 0.0)

        if "mic_device" in settings:
            self._audio.set_mic_device(settings["mic_device"])

        if "incremental_asr" in settings:
            self._incremental_enabled = settings["incremental_asr"]
        if "interim_interval" in settings:
            self._interim_interval = settings["interim_interval"]

    def reset_for_asr_switch(self):
        self._reset_interim_state()
        with self._vad_lock:
            self._vad.flush()
            self._vad.reset()

    def buffer_stats(self) -> dict:
        return self._vad.buffer_stats()

    def _capture_loop(self):
        silence_chunk = np.zeros(
            int(self._sample_rate * self._chunk_duration),
            dtype=np.float32,
        )
        while self._running:
            item = self._audio.get_audio(timeout=1.0)
            if item is None:
                if self._vad.is_speaking and not self._paused:
                    n = self._vad.effective_silence_limit_chunks() + 1
                    for _ in range(n):
                        with self._vad_lock:
                            seg = self._vad.process_chunk(silence_chunk)
                        if seg is not None and self._is_asr_ready():
                            self._enqueue_asr("vad_flush", seg)
                            break
                continue

            chunk, mic_rms = item

            if self._paused:
                continue

            rms = float(np.sqrt(np.mean(chunk**2)))
            self._emit_audio_level(rms, self._vad.last_confidence, mic_rms)

            with self._vad_lock:
                speech_segment = self._vad.process_chunk(chunk)

            if speech_segment is None:
                if (
                    self._incremental_enabled
                    and self._is_asr_ready()
                    and self._vad.is_speaking
                ):
                    buf_samples = self._vad.speech_samples
                    total_dur = buf_samples / self._sample_rate
                    elapsed = (
                        buf_samples - self._last_interim_samples
                    ) / self._sample_rate
                    now = time.perf_counter()
                    cooldown = now - self._last_interim_check_time
                    if (
                        total_dur >= self._interim_interval
                        and elapsed >= self._interim_interval
                        and cooldown >= 1.0
                    ):
                        self._last_interim_check_time = now
                        self._enqueue_asr("interim", None)
                continue

            if not self._is_asr_ready():
                log.debug("ASR not ready, dropping segment")
                continue

            self._enqueue_asr("vad_flush", speech_segment)

    def _enqueue_asr(self, seg_type: str, segment):
        try:
            self._asr_queue.put_nowait((seg_type, segment))
        except queue.Full:
            try:
                dropped = self._asr_queue.get_nowait()
                log.warning(f"ASR queue full, dropped {dropped[0]} segment")
            except queue.Empty:
                pass
            try:
                self._asr_queue.put_nowait((seg_type, segment))
            except queue.Full:
                log.warning("ASR queue still full after drop, skipping segment")

    def _asr_loop(self):
        while self._running:
            try:
                item = self._asr_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            seg_type, segment = item

            if seg_type == "vad_flush":
                if self._interim_active:
                    self._process_interim_final(segment)
                else:
                    self._process_segment(segment)
                self._reset_interim_state()
            elif seg_type == "interim":
                self._drain_interim_duplicates()
                self._do_interim_asr()
                with self._vad_lock:
                    self._last_interim_samples = self._vad.speech_samples

    def _process_segment(self, speech_segment):
        seg_len = len(speech_segment) / self._sample_rate
        log.info(f"Speech segment: {seg_len:.1f}s")
        self._emit_speech_segment("segment", seg_len)

        try:
            result, asr_ms = self._run_asr(speech_segment, "segment")
        except Exception as exc:
            log.error(f"ASR error: {exc}", exc_info=True)
            self._emit_asr_error("segment", exc)
            return
        if asr_ms == 0:
            return
        if asr_ms > 10000:
            log.warning(f"ASR took {asr_ms:.0f}ms, possible hang")
        if result is None:
            return

        original_text = result["text"].strip()
        if not original_text or not any(c.isalnum() for c in original_text):
            log.debug(
                f"ASR returned empty/punctuation-only, skipping: '{result['text']}'"
            )
            return

        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(
                f"Noise filter: {seg_len:.1f}s segment produced only "
                f"'{original_text}', skipping"
            )
            return

        source_lang = result["language"]
        if not self._language_allowed(source_lang, original_text):
            return

        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms): {original_text}")
        self._emit_asr_text(original_text, source_lang, asr_ms, interim=False)

    @staticmethod
    def _get_segmenter(lang: str):
        import pysbd

        if lang not in PipelineController._pysbd_cache:
            pysbd_lang = lang if lang in pysbd.languages.LANGUAGE_CODES else "en"
            PipelineController._pysbd_cache[lang] = pysbd.Segmenter(
                language=pysbd_lang, clean=False
            )
        return PipelineController._pysbd_cache[lang]

    def _split_sentences(self, text: str, lang: str = "en") -> list[str]:
        seg = self._get_segmenter(lang)
        parts = [p for p in seg.segment(text) if p.strip()]
        if len(parts) > 1:
            return parts

        min_len = 25 if any(c == "、" for c in text) else 60
        if len(text) > min_len:
            for i in range(len(text) - 8, 5, -1):
                if text[i] in ",，;；、":
                    before = text[: i + 1].strip()
                    after = text[i + 1 :].strip()
                    if before and after and len(before) > 15 and len(after) > 3:
                        return [before, after]

        return parts

    @staticmethod
    def _is_short_utterance(text: str) -> bool:
        alnum = sum(1 for c in text if c.isalnum())
        return alnum <= 8

    def _strip_committed_overlap(self, text: str) -> str:
        if not self._interim_committed_tail:
            return text
        tail = self._interim_committed_tail.lower().rstrip()
        text_lower = text.lower()
        max_check = min(len(tail), len(text_lower))
        for overlap_len in range(max_check, 2, -1):
            if text_lower[:overlap_len] == tail[-overlap_len:]:
                stripped = text[overlap_len:].strip()
                if stripped:
                    log.debug(
                        f"Stripped echo overlap ({overlap_len} chars): "
                        f"'{text[:overlap_len]}...'"
                    )
                    return stripped
                return ""
        return text

    def _do_interim_asr(self) -> bool:
        with self._vad_lock:
            peek = self._vad.peek_buffer()
        if peek is None:
            return False
        audio, duration = peek

        if duration < 1.5:
            return False

        use_word_ts = False

        try:
            if use_word_ts:
                result, asr_ms = self._run_asr(
                    audio, "interim", word_timestamps=use_word_ts
                )
            else:
                result, asr_ms = self._run_asr(audio, "interim")
        except Exception as exc:
            log.error(f"Interim ASR error: {exc}", exc_info=True)
            self._emit_asr_error("interim", exc)
            return False

        if asr_ms == 0 or result is None:
            return False

        full_text = result["text"].strip()
        if not full_text or not any(c.isalnum() for c in full_text):
            return False

        full_text = self._strip_committed_overlap(full_text)
        if not full_text:
            return False

        split_start = time.perf_counter()
        sentences = self._split_sentences(full_text, result["language"])
        split_ms = (time.perf_counter() - split_start) * 1000
        if len(sentences) <= 1:
            return False
        log.debug(
            f"Interim split [{result['language']}] ({split_ms:.1f}ms): "
            f"{len(sentences)} parts -> {sentences}"
        )

        complete = sentences[:-1]
        committed_text = ""
        for sent in complete:
            committed_text += sent

        if not committed_text.strip():
            return False

        total_samples = len(audio)
        if use_word_ts and result.get("words"):
            words = result["words"]
            committed_lower = committed_text.lower().rstrip()
            char_pos = 0
            last_word_end = 0.0
            for word in words:
                word_text = word["word"].strip()
                idx = committed_lower.find(word_text.lower(), char_pos)
                if idx >= 0:
                    char_pos = idx + len(word_text)
                    last_word_end = word["end"]
                if char_pos >= len(committed_lower):
                    break
            trim_samples = int(last_word_end * self._sample_rate)
        else:
            ratio = len(committed_text) / max(len(full_text), 1)
            margin = int(0.3 * self._sample_rate)
            trim_samples = int(ratio * total_samples) + margin
            max_trim = total_samples - int(0.5 * self._sample_rate)
            trim_samples = min(trim_samples, max(max_trim, 0))
            min_trim = int(0.3 * self._sample_rate)
            if 0 < trim_samples < min_trim:
                trim_samples = min(min_trim, total_samples // 2)

        actually_committed = False
        for sent in complete:
            text = sent.strip()
            if not text:
                continue
            if self._is_short_utterance(text):
                self._interim_pending += text
                log.debug(
                    f"Interim short utterance buffered: '{text}', "
                    f"pending='{self._interim_pending}'"
                )
                continue

            if self._interim_pending:
                text = self._interim_pending + text
                self._interim_pending = ""

            self._process_segment_text(text, result["language"], asr_ms)
            actually_committed = True

        if not actually_committed:
            return False

        if trim_samples > 0:
            with self._vad_lock:
                self._vad.trim_front(trim_samples)

        self._interim_committed_tail = (
            committed_text[-50:] if len(committed_text) > 50 else committed_text
        )

        self._interim_active = True
        log.info(
            f"Interim ASR: committed {len(complete)} sentence(s), "
            f"trimmed {trim_samples / self._sample_rate:.2f}s"
        )
        return True

    def _process_segment_text(
        self, text: str, source_lang: str, asr_ms: float = 0.0
    ):
        original_text = text.strip()
        if not original_text or not any(c.isalnum() for c in original_text):
            return

        if not self._language_allowed(source_lang, original_text):
            return

        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms, interim): {original_text}")
        self._emit_asr_text(original_text, source_lang, asr_ms, interim=True)

    def _process_interim_final(self, speech_segment):
        seg_len = len(speech_segment) / self._sample_rate
        log.info(f"Interim final segment: {seg_len:.1f}s")
        self._emit_speech_segment("interim_final", seg_len)

        try:
            result, asr_ms = self._run_asr(speech_segment, "interim_final")
        except Exception as exc:
            log.error(f"Interim final ASR error: {exc}", exc_info=True)
            self._emit_asr_error("interim_final", exc)
            return
        if asr_ms == 0:
            return

        if result is None:
            if self._interim_pending:
                text = self._interim_pending
                self._interim_pending = ""
                lang = self._asr_language()
                if lang == "auto":
                    lang = "unknown"
                self._process_segment_text(text, lang)
            return

        original_text = result["text"].strip()
        original_text = self._strip_committed_overlap(original_text)

        if self._interim_pending:
            original_text = self._interim_pending + original_text
            self._interim_pending = ""

        if not original_text or not any(c.isalnum() for c in original_text):
            return

        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(
                f"Noise filter: {seg_len:.1f}s segment produced only "
                f"'{original_text}', skipping"
            )
            return

        self._process_segment_text(original_text, result["language"], asr_ms)

    def _drain_interim_duplicates(self):
        while True:
            try:
                item = self._asr_queue.get_nowait()
            except queue.Empty:
                break
            if item is None or item[0] != "interim":
                self._asr_queue.put(item)
                break

    def _run_asr(self, audio: np.ndarray, kind: str, **kwargs):
        if not self._is_asr_ready():
            return None, 0.0
        return self._asr_runner(audio, kind, **kwargs)

    def _language_allowed(self, source_lang: str, text: str) -> bool:
        asr_lang_setting = self._asr_language()
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(
                f"Language filter: expected '{asr_lang_setting}' but got "
                f"'{source_lang}', discarding: {text[:60]}"
            )
            return False
        return True

    def _is_asr_ready(self) -> bool:
        try:
            return self._asr_ready()
        except Exception as exc:
            log.warning(f"ASR readiness check failed: {exc}")
            return False

    def _reset_interim_state(self):
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""

    def _emit_audio_level(
        self, rms: float, vad_confidence: float, mic_rms: float | None = None
    ):
        if self._audio_level_callback is not None:
            self._audio_level_callback(AudioLevelEvent(rms, vad_confidence, mic_rms))

    def _emit_speech_segment(self, kind: str, audio_seconds: float):
        if self._speech_segment_callback is not None:
            self._speech_segment_callback(SpeechSegmentEvent(kind, audio_seconds))

    def _emit_asr_text(
        self, text: str, source_lang: str, asr_ms: float, interim: bool
    ):
        if self._asr_text_callback is not None:
            self._asr_text_callback(
                ASRTextEvent(text, source_lang, asr_ms, interim=interim)
            )

    def _emit_asr_error(self, kind: str, error: Exception):
        if self._asr_error_callback is not None:
            self._asr_error_callback(ASRErrorEvent(kind, error))

    def _emit_stats(self, state: str):
        if self._stats_callback is not None:
            self._stats_callback(
                PipelineStatsEvent(state, queue_size=self._asr_queue.qsize())
            )
