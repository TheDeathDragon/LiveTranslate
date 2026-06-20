import logging
import os
import re
from contextlib import nullcontext

import numpy as np
import torch

log = logging.getLogger("LiveTranslate.SenseVoice")

SAMPLE_RATE = 16000
DEFAULT_PAD_SECONDS = 0.5
PAD_SECONDS_ENV = "LIVETRANS_SENSEVOICE_PAD_SECONDS"

# Language tag mapping from SenseVoice output
LANG_MAP = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|yue|>": "yue",
}


class SenseVoiceEngine:
    """Speech-to-text using FunASR SenseVoice."""

    def __init__(self, model_name=None, device="cuda", hub="ms", pad_seconds=None):
        from funasr import AutoModel
        from model_manager import (
            get_local_model_path,
            asr_model_id,
        )

        local = get_local_model_path("sensevoice", hub=hub)
        model = local or model_name or asr_model_id("sensevoice", hub)
        self._set_precision(device)
        model_kwargs = {
            "model": model,
            "trust_remote_code": True,
            "device": device,
            "hub": hub,
            "disable_update": True,
        }
        if self._use_fp16:
            model_kwargs["fp16"] = True
        self._model = AutoModel(**model_kwargs)
        device = self._model.kwargs.get("device", device)
        self._set_precision(device)
        self._update_runtime_kwargs(device)
        self._set_input_padding(pad_seconds, log_change=False)
        self.language = None  # None = auto detect
        log.info(
            f"SenseVoice loaded: {model} on {device} "
            f"(hub={hub}, precision={self._precision})"
        )
        self._log_input_padding()

    @staticmethod
    def _read_pad_seconds(value=None) -> float:
        if value is None:
            value = os.environ.get(PAD_SECONDS_ENV)
        if value is None:
            return DEFAULT_PAD_SECONDS
        try:
            return float(value)
        except (TypeError, ValueError):
            log.warning(
                f"Invalid SenseVoice pad seconds={value!r}; "
                f"using default {DEFAULT_PAD_SECONDS:g}s"
            )
            return DEFAULT_PAD_SECONDS

    def _set_input_padding(self, pad_seconds=None, log_change=True):
        self._pad_seconds = self._read_pad_seconds(pad_seconds)
        self._pad_quantum = int(round(SAMPLE_RATE * self._pad_seconds))
        if log_change:
            self._log_input_padding()

    def _log_input_padding(self):
        if self._pad_quantum > 0:
            log.info(
                "SenseVoice input padding enabled: "
                f"bucket={self._pad_seconds:g}s, quantum={self._pad_quantum} samples"
            )
        else:
            log.info("SenseVoice input padding disabled")

    @staticmethod
    def _is_cuda_device(device: str) -> bool:
        return str(device).lower().startswith("cuda") and torch.cuda.is_available()

    def _set_precision(self, device: str):
        self._use_fp16 = self._is_cuda_device(device)
        self._precision = "fp16" if self._use_fp16 else "fp32"

    def _apply_model_precision(self):
        model = self._model.model
        if self._use_fp16:
            model.half()
        else:
            model.float()

    def _update_runtime_kwargs(self, device: str):
        self._model.kwargs["device"] = device
        self._model.kwargs["fp16"] = self._use_fp16
        if not self._use_fp16:
            self._model.kwargs.pop("bf16", None)

    def _autocast_context(self):
        if self._use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _prepare_audio_input(self, audio: np.ndarray) -> np.ndarray:
        if self._pad_quantum <= 0 or audio.size == 0:
            return audio

        original_samples = audio.shape[0]
        remainder = original_samples % self._pad_quantum
        if remainder == 0:
            return audio

        padded_samples = original_samples + self._pad_quantum - remainder
        padded = np.pad(audio, (0, padded_samples - original_samples), mode="constant")
        log.debug(
            f"SenseVoice input padded: {original_samples} -> {padded_samples} samples"
        )
        return padded

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"SenseVoice language: {old} -> {self.language}")

    def set_input_padding(self, pad_seconds):
        old_quantum = self._pad_quantum
        self._set_input_padding(pad_seconds, log_change=False)
        if self._pad_quantum != old_quantum:
            self._log_input_padding()

    def to_device(self, device: str):
        self._set_precision(device)
        if self._use_fp16:
            self._apply_model_precision()
            self._model.model.to(device)
        else:
            self._model.model.to(device)
            self._apply_model_precision()
        self._update_runtime_kwargs(device)
        log.info(f"SenseVoice moved to {device} (precision={self._precision})")

    def unload(self):
        if hasattr(self, "_model") and self._model is not None:
            try:
                self._model.model.to("cpu")
            except Exception:
                pass
            self._model = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe audio segment.

        Args:
            audio: float32 numpy array, 16kHz mono

        Returns:
            dict with 'text', 'language', 'language_name' or None.
        """
        cache = {}
        try:
            audio_input = self._prepare_audio_input(audio)
            with torch.inference_mode(), self._autocast_context():
                result = self._model.generate(
                    input=audio_input,
                    cache=cache,
                    language=self.language or "auto",
                    use_itn=True,
                    batch_size_s=0,
                    disable_pbar=True,
                )
        finally:
            cache.clear()

        if not result or not result[0].get("text"):
            return None

        raw_text = result[0]["text"]

        # Parse language tag and clean text
        detected_lang = "auto"
        text = raw_text

        for tag, lang in LANG_MAP.items():
            if tag in text:
                detected_lang = lang
                text = text.replace(tag, "")
                break

        # Remove emotion/event tags like <|HAPPY|>, <|BGM|>, <|Speech|> etc.
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        if not text:
            return None

        log.debug(f"Raw: {raw_text}")
        return {
            "text": text,
            "language": detected_lang,
            "language_name": detected_lang,
        }
