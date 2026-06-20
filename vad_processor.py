import logging
import collections

import numpy as np
import torch

torch.set_num_threads(1)

log = logging.getLogger("LiveTranslate.VAD")


class VADProcessor:
    """Voice Activity Detection with multiple modes."""

    def __init__(
        self,
        sample_rate=16000,
        threshold=0.50,
        min_speech_duration=1.0,
        max_speech_duration=15.0,
        chunk_duration=0.032,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.energy_threshold = 0.02
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        self._chunk_duration = chunk_duration
        self.mode = "silero"  # "silero", "firered", "energy", "disabled"
        self._firered = None
        self._firered_model = ""
        self._firered_use_gpu = False
        self._firered_smooth_window_size = 5
        self._firered_frame_aggregation = "max"
        self._firered_failed_key = None
        self._firered_missing_warned_for = None

        # Silero v5 ships its model inside the `silero-vad` PyPI package, so load
        # it from there (zero network). Only fall back to the torch.hub cache
        # (pinned branch -> offline when already cached) if the package is
        # missing, since torch.hub otherwise probes/downloads from GitHub.
        try:
            from silero_vad import load_silero_vad
        except ImportError:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:master",
                model="silero_vad",
                trust_repo=True,
            )
        else:
            self._model = load_silero_vad()
        self._model.eval()

        self._speech_buffer = []
        self._confidence_history = []  # per-chunk confidence, synced with _speech_buffer
        self._speech_samples = 0
        self._is_speaking = False
        self._silence_counter = 0
        self._was_trimmed = False  # True after trim_front (interim ASR active)

        # Pre-speech ring buffer: capture onset consonants before VAD triggers
        self._pre_speech_chunks = 3  # ~96ms at 32ms/chunk
        self._pre_buffer = collections.deque(maxlen=self._pre_speech_chunks)

        # Silence timing
        self._silence_mode = "auto"  # "auto" or "fixed"
        self._fixed_silence_dur = 0.8
        self._silence_limit = self._seconds_to_chunks(0.8)

        # Progressive silence: shorter threshold when buffer is long
        self._progressive_tiers = [
            # (buffer_seconds, silence_multiplier)
            (3.0, 1.0),  # < 3s: use full silence_limit
            (6.0, 0.5),  # 3-6s: use half silence_limit
            (10.0, 0.25),  # 6-10s: use quarter silence_limit
        ]

        # Adaptive silence tracking: recent pause durations (seconds)
        self._pause_history = collections.deque(maxlen=50)
        self._adaptive_min = 0.3
        self._adaptive_max = 2.0

        # Exposed for monitor
        self.last_confidence = 0.0

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def speech_samples(self) -> int:
        return self._speech_samples

    def reset(self):
        self._reset()
        self._reset_firered_adapter()

    def effective_silence_limit_chunks(self) -> int:
        return self._get_effective_silence_limit()

    def buffer_stats(self) -> dict:
        return {
            "chunks": len(self._speech_buffer),
            "samples": self._speech_samples,
            "seconds": self._speech_samples / self.sample_rate,
            "is_speaking": self._is_speaking,
        }

    def _seconds_to_chunks(self, seconds: float) -> int:
        return max(1, round(seconds / self._chunk_duration))

    def _update_adaptive_limit(self):
        if len(self._pause_history) < 3:
            return
        pauses = sorted(self._pause_history)
        # P75 of recent pauses × 1.2
        idx = int(len(pauses) * 0.75)
        p75 = pauses[min(idx, len(pauses) - 1)]
        target = max(self._adaptive_min, min(self._adaptive_max, p75 * 1.2))
        new_limit = self._seconds_to_chunks(target)
        if new_limit != self._silence_limit:
            log.debug(
                f"Adaptive silence: {target:.2f}s ({new_limit} chunks), P75={p75:.2f}s"
            )
            self._silence_limit = new_limit

    def update_settings(self, settings: dict):
        old_mode = self.mode
        old_firered_key = self._firered_config_key()
        if "vad_mode" in settings:
            mode = settings["vad_mode"]
            if mode in ("silero", "firered", "energy", "disabled"):
                self.mode = mode
            else:
                log.warning(f"Unknown VAD mode '{mode}', keeping {self.mode}")
        if "vad_threshold" in settings:
            self.threshold = settings["vad_threshold"]
        if "energy_threshold" in settings:
            self.energy_threshold = settings["energy_threshold"]
        if "firered_vad_model" in settings:
            self._firered_model = str(settings.get("firered_vad_model") or "")
        if "firered_vad_use_gpu" in settings:
            self._firered_use_gpu = bool(settings.get("firered_vad_use_gpu"))
        if "firered_vad_smooth_window_size" in settings:
            try:
                self._firered_smooth_window_size = max(
                    1, int(settings.get("firered_vad_smooth_window_size") or 5)
                )
            except (TypeError, ValueError):
                self._firered_smooth_window_size = 5
        if "firered_vad_frame_aggregation" in settings:
            agg = str(settings.get("firered_vad_frame_aggregation") or "max").lower()
            self._firered_frame_aggregation = (
                agg if agg in ("max", "latest", "mean") else "max"
            )
        if "min_speech_duration" in settings:
            self.min_speech_samples = int(
                settings["min_speech_duration"] * self.sample_rate
            )
        if "max_speech_duration" in settings:
            self.max_speech_samples = int(
                settings["max_speech_duration"] * self.sample_rate
            )
        if "silence_mode" in settings:
            self._silence_mode = settings["silence_mode"]
        if "silence_duration" in settings:
            self._fixed_silence_dur = settings["silence_duration"]
            if self._silence_mode == "fixed":
                self._silence_limit = self._seconds_to_chunks(self._fixed_silence_dur)
        if old_firered_key != self._firered_config_key():
            self._unload_firered_adapter()
        elif old_mode != self.mode:
            self._reset_firered_adapter()
        log.info(
            f"VAD settings updated: mode={self.mode}, threshold={self.threshold}, "
            f"silence={self._silence_mode} "
            f"({self._silence_limit} chunks = {self._silence_limit * self._chunk_duration:.2f}s)"
        )

    def _silero_confidence(self, audio_chunk: np.ndarray) -> float:
        window_size = 512 if self.sample_rate == 16000 else 256
        chunk = audio_chunk[:window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        with torch.inference_mode():
            tensor = torch.from_numpy(chunk).float()
            return float(self._model(tensor, self.sample_rate).item())

    def _energy_confidence(self, audio_chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(audio_chunk**2)))
        return min(1.0, rms / (self.energy_threshold * 2))

    def _firered_config_key(self, resolved_model: str | None = None) -> tuple:
        return (
            resolved_model or self._firered_model,
            self._firered_use_gpu,
            self._firered_smooth_window_size,
            self._firered_frame_aggregation,
        )

    def _reset_firered_adapter(self):
        if self._firered is not None:
            self._firered.reset()

    def _unload_firered_adapter(self):
        if self._firered is not None:
            self._firered.unload()
        self._firered = None
        self._firered_failed_key = None
        self._firered_missing_warned_for = None

    def _fallback_silero_confidence(self, audio_chunk: np.ndarray) -> float:
        return self._silero_confidence(audio_chunk)

    def _firered_confidence(self, audio_chunk: np.ndarray) -> float:
        if not self._firered_model:
            if self._firered_missing_warned_for != "":
                log.warning(
                    "FireRedVAD selected but no Stream-VAD model is configured; "
                    "using Silero VAD confidence"
                )
                self._firered_missing_warned_for = ""
            return self._fallback_silero_confidence(audio_chunk)

        try:
            from model_manager import get_firered_vad_model_path

            model_path = get_firered_vad_model_path(self._firered_model)
        except Exception as exc:
            log.warning(f"FireRedVAD model path resolution failed: {exc}")
            model_path = None

        if not model_path:
            if self._firered_missing_warned_for != self._firered_model:
                log.warning(
                    "FireRedVAD Stream-VAD model is unavailable: %s; "
                    "using Silero VAD confidence",
                    self._firered_model,
                )
                self._firered_missing_warned_for = self._firered_model
            return self._fallback_silero_confidence(audio_chunk)

        config_key = self._firered_config_key(model_path)
        if self._firered is None:
            if self._firered_failed_key == config_key:
                return self._fallback_silero_confidence(audio_chunk)
            try:
                from vad_firered import FireRedVadAdapter

                self._firered = FireRedVadAdapter(
                    model_dir=model_path,
                    threshold=self.threshold,
                    smooth_window_size=self._firered_smooth_window_size,
                    use_gpu=self._firered_use_gpu,
                    frame_aggregation=self._firered_frame_aggregation,
                )
                self._firered_failed_key = None
            except Exception as exc:
                log.warning(f"FireRedVAD unavailable: {exc}")
                self._firered_failed_key = config_key
                self._firered = None
                return self._fallback_silero_confidence(audio_chunk)

        try:
            return self._firered.confidence_for_chunk(audio_chunk)
        except Exception as exc:
            log.warning(f"FireRedVAD confidence failed, falling back to Silero: {exc}")
            self._unload_firered_adapter()
            self._firered_failed_key = config_key
            return self._fallback_silero_confidence(audio_chunk)

    def _get_confidence(self, audio_chunk: np.ndarray) -> float:
        if self.mode == "silero":
            return self._silero_confidence(audio_chunk)
        elif self.mode == "firered":
            return self._firered_confidence(audio_chunk)
        elif self.mode == "energy":
            return self._energy_confidence(audio_chunk)
        else:  # disabled
            return 1.0

    def _effective_threshold(self) -> float:
        return self.threshold if self.mode in ("silero", "firered") else 0.5

    def _get_effective_silence_limit(self) -> int:
        """Progressive silence: accept shorter pauses as split points when buffer is long."""
        buf_seconds = self._speech_samples / self.sample_rate
        multiplier = 1.0
        for tier_sec, tier_mult in self._progressive_tiers:
            if buf_seconds < tier_sec:
                break
            multiplier = tier_mult
        effective = max(1, round(self._silence_limit * multiplier))
        return effective

    def process_chunk(self, audio_chunk: np.ndarray):
        confidence = self._get_confidence(audio_chunk)
        self.last_confidence = confidence

        effective_threshold = self._effective_threshold()
        eff_silence_limit = self._get_effective_silence_limit()

        if confidence >= effective_threshold:
            # Record pause duration for adaptive mode
            if self._is_speaking and self._silence_counter > 0:
                pause_dur = self._silence_counter * self._chunk_duration
                if pause_dur >= 0.1:
                    self._pause_history.append(pause_dur)
                    if self._silence_mode == "auto":
                        self._update_adaptive_limit()

            if not self._is_speaking:
                # Speech onset: prepend pre-speech buffer to capture leading consonants
                # Use threshold as confidence so these chunks don't create false valleys
                for pre_chunk in self._pre_buffer:
                    self._speech_buffer.append(pre_chunk)
                    self._confidence_history.append(effective_threshold)
                    self._speech_samples += len(pre_chunk)
                self._pre_buffer.clear()

            self._is_speaking = True
            self._silence_counter = 0
            self._speech_buffer.append(audio_chunk)
            self._confidence_history.append(confidence)
            self._speech_samples += len(audio_chunk)
        elif self._is_speaking:
            self._silence_counter += 1
            self._speech_buffer.append(audio_chunk)
            self._confidence_history.append(confidence)
            self._speech_samples += len(audio_chunk)
        else:
            # Not speaking: feed pre-speech ring buffer
            self._pre_buffer.append(audio_chunk)

        # Force segment if max duration reached — backtrack to find best split point
        if self._speech_samples >= self.max_speech_samples:
            return self._split_at_best_pause()

        # End segment after enough silence (progressive threshold)
        if self._is_speaking and self._silence_counter >= eff_silence_limit:
            if self._speech_samples >= self.min_speech_samples:
                return self._flush_segment()
            elif self._was_trimmed:
                # Interim ASR trimmed the buffer; return remainder instead of dropping
                log.debug(
                    f"Short segment after trim ({self._speech_samples / self.sample_rate:.1f}s), "
                    f"force flushing for interim final"
                )
                return self.force_flush()
            else:
                # Too short — keep buffer, merge with next speech onset
                log.debug(
                    f"Short segment {self._speech_samples / self.sample_rate:.1f}s "
                    f"< min {self.min_speech_samples / self.sample_rate:.1f}s, "
                    f"keeping for merge"
                )
                self._is_speaking = False
                self._silence_counter = 0
                return None

        return None

    def _find_best_split_index(self) -> int:
        """Find the best chunk index to split at using smoothed confidence.
        A sliding window average reduces single-chunk noise, then we find
        the center of the lowest valley. Works even when the speaker never
        fully pauses (e.g. fast commentary).
        Returns -1 if no usable split point found."""
        n = len(self._confidence_history)
        if n < 4:
            return -1

        # Smooth confidence with a sliding window (~160ms = 5 chunks at 32ms)
        smooth_win = min(5, n // 2)
        smoothed = []
        for i in range(n):
            lo = max(0, i - smooth_win // 2)
            hi = min(n, i + smooth_win // 2 + 1)
            smoothed.append(sum(self._confidence_history[lo:hi]) / (hi - lo))

        # Search in the latter 70% of the buffer (avoid splitting too early)
        search_start = max(1, n * 3 // 10)

        # Find global minimum in smoothed curve
        min_val = float("inf")
        min_idx = -1
        for i in range(search_start, n):
            if smoothed[i] <= min_val:
                min_val = smoothed[i]
                min_idx = i

        if min_idx <= 0:
            return -1

        # Check if this is a meaningful dip
        avg_conf = sum(smoothed[search_start:]) / max(1, n - search_start)
        dip_ratio = min_val / max(avg_conf, 1e-6)

        effective_threshold = self._effective_threshold()
        if min_val < effective_threshold or dip_ratio < 0.8:
            log.debug(
                f"Split point at chunk {min_idx}/{n}: "
                f"smoothed={min_val:.3f}, avg={avg_conf:.3f}, dip_ratio={dip_ratio:.2f}"
            )
            return min_idx

        # Fallback: any point below average is better than hard cut
        if min_val < avg_conf:
            log.debug(
                f"Split point (fallback) at chunk {min_idx}/{n}: "
                f"smoothed={min_val:.3f}, avg={avg_conf:.3f}"
            )
            return min_idx

        return -1

    def _split_at_best_pause(self):
        """When hitting max duration, backtrack to find the best pause point.
        Flushes the first part and keeps the remainder for continued accumulation."""
        if not self._speech_buffer:
            return None

        split_idx = self._find_best_split_index()

        if split_idx <= 0:
            # No good split point — hard flush everything
            log.info(
                f"Max duration reached, no good split point, "
                f"hard flush {self._speech_samples / self.sample_rate:.1f}s"
            )
            return self._flush_segment()

        # Split: emit first part, keep remainder
        first_bufs = self._speech_buffer[:split_idx]
        remain_bufs = self._speech_buffer[split_idx:]
        remain_confs = self._confidence_history[split_idx:]

        first_samples = sum(len(b) for b in first_bufs)
        remain_samples = sum(len(b) for b in remain_bufs)

        log.info(
            f"Max duration split at {first_samples / self.sample_rate:.1f}s, "
            f"keeping {remain_samples / self.sample_rate:.1f}s remainder"
        )

        segment = np.concatenate(first_bufs)

        # Keep remainder in buffer for next segment
        self._speech_buffer = remain_bufs
        self._confidence_history = remain_confs
        self._speech_samples = remain_samples
        self._is_speaking = True
        self._silence_counter = 0

        return segment

    def _flush_segment(self):
        if not self._speech_buffer:
            return None
        # Speech density check: discard segments where most chunks are below threshold
        if len(self._confidence_history) >= 4:
            effective_threshold = self._effective_threshold()
            voiced = sum(
                1 for c in self._confidence_history if c >= effective_threshold
            )
            density = voiced / len(self._confidence_history)
            if density < 0.25:
                dur = self._speech_samples / self.sample_rate
                log.debug(
                    f"Low speech density {density:.0%} ({voiced}/{len(self._confidence_history)}), "
                    f"discarding {dur:.1f}s segment"
                )
                self._reset()
                return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def _reset(self):
        self._speech_buffer = []
        self._confidence_history = []
        self._speech_samples = 0
        self._is_speaking = False
        self._silence_counter = 0
        self._was_trimmed = False

    def peek_buffer(self):
        """Read current speech buffer without flushing. Returns (audio, duration) or None."""
        if not self._speech_buffer or not self._is_speaking:
            return None
        audio = np.concatenate(self._speech_buffer)
        duration = self._speech_samples / self.sample_rate
        return audio, duration

    def trim_front(self, n_samples: int):
        """Remove first n_samples from the speech buffer."""
        if n_samples <= 0:
            return
        removed = 0
        while self._speech_buffer and removed < n_samples:
            chunk = self._speech_buffer[0]
            if removed + len(chunk) <= n_samples:
                self._speech_buffer.pop(0)
                self._confidence_history.pop(0)
                removed += len(chunk)
            else:
                # Partial trim of first chunk
                keep = removed + len(chunk) - n_samples
                self._speech_buffer[0] = chunk[-keep:]
                removed = n_samples
        self._speech_samples = sum(len(b) for b in self._speech_buffer)
        self._was_trimmed = True
        log.debug(f"trim_front: removed {removed} samples, remaining {self._speech_samples / self.sample_rate:.2f}s")

    def force_flush(self):
        """Flush buffer regardless of min_speech_samples."""
        if not self._speech_buffer:
            return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def flush(self):
        if self._speech_samples >= self.min_speech_samples:
            return self._flush_segment()
        self._reset()
        return None
