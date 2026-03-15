import logging

import numpy as np
from faster_whisper import WhisperModel

from translator import LANGUAGE_DISPLAY

log = logging.getLogger("LiveTrans.ASR")


LANGUAGE_NAMES = {**LANGUAGE_DISPLAY, "auto": "auto"}


class ASREngine:
    """Speech-to-text using faster-whisper."""

    def __init__(
        self,
        model_size="medium",
        device="cuda",
        compute_type="float16",
        language="auto",
        download_root=None,
    ):
        self.language = language if language != "auto" else None
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        log.info(f"Model loaded: {model_size} on {device} ({compute_type})")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"ASR language: {old} -> {self.language}")

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe audio segment.

        Args:
            audio: float32 numpy array, 16kHz mono

        Returns:
            dict with 'text', 'language', 'language_name' or None if no speech detected.
        """
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())

        full_text = " ".join(text_parts).strip()
        if not full_text:
            return None

        detected_lang = info.language
        return {
            "text": full_text,
            "language": detected_lang,
            "language_name": LANGUAGE_NAMES.get(detected_lang, detected_lang),
        }
