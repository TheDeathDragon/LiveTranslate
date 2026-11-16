import logging

import numpy as np

log = logging.getLogger("LiveTranslate.Qwen3ASR")


class Qwen3ASREngine:
    """Speech-to-text using Qwen3-ASR through transformers' native classes.

    Loaded with `.to(device)` rather than `device_map` because accelerate is not
    installed; `.to()` moves the whole model in one step after the weights have
    landed in host memory.
    """

    def __init__(self, device="cuda", hub="ms", max_new_tokens=512):
        import torch
        from transformers import (
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
        )

        from model_manager import get_local_model_path

        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        model_dir = get_local_model_path("qwen3-asr", hub=hub)
        if not model_dir:
            raise RuntimeError("Qwen3-ASR weights are not cached; download them first")

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self._device = device
        self._max_new_tokens = max_new_tokens
        self.language = None
        self._processor = Qwen3ASRProcessor.from_pretrained(model_dir)
        self._model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype
        ).to(device)
        self._model.eval()
        log.info(f"Qwen3-ASR loaded from {model_dir} on {device}")

    def set_language(self, language: str):
        if language in ("auto", None):
            self.language = None
            return

        from transformers.audio_utils import resolve_language
        from transformers.models.qwen3_asr.processing_qwen3_asr import (
            LANGUAGE_CODE_TO_NAME,
        )

        try:
            self.language = resolve_language(language, LANGUAGE_CODE_TO_NAME)
        except ValueError:
            log.warning(
                f"Qwen3-ASR does not support {language!r}; using auto detection"
            )
            self.language = None

    def to_device(self, device: str):
        try:
            import torch

            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            self._model.to(device, dtype=dtype)
            self._device = device
            log.info(f"Qwen3-ASR moved to {device}")
            return True
        except Exception as e:
            log.warning(f"Qwen3-ASR to_device failed: {e}")
            return False

    def unload(self):
        if getattr(self, "_model", None) is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
            self._model = None
            self._processor = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe audio segment (float32, 16kHz mono)."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen3-ASR engine was unloaded")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        import torch

        from asr_engine import LANGUAGE_NAMES
        from transformers.audio_utils import resolve_language
        from transformers.models.qwen3_asr.processing_qwen3_asr import (
            LANGUAGE_CODE_TO_NAME,
        )

        inputs = self._processor.apply_transcription_request(
            audio, language=self.language
        ).to(self._device)
        # The processor emits float32 features; the model runs in its own dtype.
        inputs["input_features"] = inputs["input_features"].to(self._model.dtype)
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
            )
        parsed = self._processor.decode(
            outputs, return_format="parsed", skip_special_tokens=True
        )
        if isinstance(parsed, list):
            parsed = parsed[0]
        text = (parsed.get("transcription") or "").strip()
        if not text:
            return None

        # The model emits a full language name ("Japanese"); the main process
        # compares against the code shown in the settings combo.
        detected = parsed.get("language")
        if detected:
            try:
                detected = resolve_language(detected, LANGUAGE_CODE_TO_NAME)
            except ValueError:
                detected = None
        lang = detected or self.language or "auto"
        return {
            "text": text,
            "language": lang,
            "language_name": LANGUAGE_NAMES.get(lang, lang),
        }
