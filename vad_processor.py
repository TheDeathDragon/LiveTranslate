import logging
import collections

import numpy as np
import torch

torch.set_num_threads(1)

log = logging.getLogger("LiveTrans.VAD")


class VADProcessor:
    """Voice Activity Detection with multiple modes."""

    def __init__(
        self,
        sample_rate=16000,
        threshold=0.5,
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
        self.mode = "silero"  # "silero", "energy", "disabled"

        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()

        self._speech_buffer = []
        self._speech_samples = 0
        self._is_speaking = False
        self._silence_counter = 0

        # Silence timing
        self._silence_mode = "auto"  # "auto" or "fixed"
        self._fixed_silence_dur = 0.8
        self._silence_limit = self._seconds_to_chunks(0.8)

        # Adaptive silence tracking: recent pause durations (seconds)
        self._pause_history = collections.deque(maxlen=50)
        self._adaptive_min = 0.3
        self._adaptive_max = 2.0

        # Exposed for monitor
        self.last_confidence = 0.0

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
            log.debug(f"Adaptive silence: {target:.2f}s ({new_limit} chunks), P75={p75:.2f}s")
            self._silence_limit = new_limit

    def update_settings(self, settings: dict):
        if "vad_mode" in settings:
            self.mode = settings["vad_mode"]
        if "vad_threshold" in settings:
            self.threshold = settings["vad_threshold"]
        if "energy_threshold" in settings:
            self.energy_threshold = settings["energy_threshold"]
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
        log.info(
            f"VAD settings updated: mode={self.mode}, threshold={self.threshold}, "
            f"silence={self._silence_mode} "
            f"({self._silence_limit} chunks = {self._silence_limit * self._chunk_duration:.2f}s)"
        )

    def _silero_confidence(self, audio_chunk: np.ndarray) -> float:
        window_size = 512 if self.sample_rate == 16000 else 256
        max_conf = 0.0
        for start in range(0, len(audio_chunk), window_size):
            window = audio_chunk[start : start + window_size]
            if len(window) < window_size:
                window = np.pad(window, (0, window_size - len(window)))
            tensor = torch.from_numpy(window).float()
            conf = self._model(tensor, self.sample_rate).item()
            max_conf = max(max_conf, conf)
        return max_conf

    def _energy_confidence(self, audio_chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(audio_chunk**2)))
        return min(1.0, rms / (self.energy_threshold * 2))

    def _get_confidence(self, audio_chunk: np.ndarray) -> float:
        if self.mode == "silero":
            return self._silero_confidence(audio_chunk)
        elif self.mode == "energy":
            return self._energy_confidence(audio_chunk)
        else:  # disabled
            return 1.0

    def process_chunk(self, audio_chunk: np.ndarray):
        confidence = self._get_confidence(audio_chunk)
        self.last_confidence = confidence

        effective_threshold = self.threshold if self.mode == "silero" else 0.5

        log.debug(
            f"VAD conf={confidence:.3f} ({self.mode}), speaking={self._is_speaking}, "
            f"buf={self._speech_samples / self.sample_rate:.1f}s, "
            f"silence_cnt={self._silence_counter}, limit={self._silence_limit}"
        )

        if confidence >= effective_threshold:
            # Record pause duration for adaptive mode
            if self._is_speaking and self._silence_counter > 0:
                pause_dur = self._silence_counter * self._chunk_duration
                if pause_dur >= 0.1:
                    self._pause_history.append(pause_dur)
                    if self._silence_mode == "auto":
                        self._update_adaptive_limit()

            self._is_speaking = True
            self._silence_counter = 0
            self._speech_buffer.append(audio_chunk)
            self._speech_samples += len(audio_chunk)
        elif self._is_speaking:
            self._silence_counter += 1
            self._speech_buffer.append(audio_chunk)
            self._speech_samples += len(audio_chunk)

        # Force segment if max duration reached
        if self._speech_samples >= self.max_speech_samples:
            return self._flush_segment()

        # End segment after enough silence
        if self._is_speaking and self._silence_counter >= self._silence_limit:
            if self._speech_samples >= self.min_speech_samples:
                return self._flush_segment()
            else:
                self._reset()
                return None

        return None

    def _flush_segment(self):
        if not self._speech_buffer:
            return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def _reset(self):
        self._speech_buffer = []
        self._speech_samples = 0
        self._is_speaking = False
        self._silence_counter = 0

    def flush(self):
        if self._speech_samples >= self.min_speech_samples:
            return self._flush_segment()
        self._reset()
        return None
