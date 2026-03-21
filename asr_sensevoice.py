import logging
import re
import numpy as np

log = logging.getLogger("LiveTrans.SenseVoice")

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

    def __init__(self, model_name="FunAudioLLM/SenseVoiceSmall", device="cuda", hub="ms"):
        from funasr import AutoModel
        from model_manager import get_local_model_path

        local = get_local_model_path("sensevoice", hub=hub)
        model = local or model_name
        self._model = AutoModel(
            model=model,
            trust_remote_code=True,
            device=device,
            hub=hub,
            disable_update=True,
        )
        self.language = None  # None = auto detect
        log.info(f"SenseVoice loaded: {model_name} on {device} (hub={hub})")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"SenseVoice language: {old} -> {self.language}")

    def to_device(self, device: str):
        self._model.model.to(device)
        log.info(f"SenseVoice moved to {device}")

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
        result = self._model.generate(
            input=audio,
            cache={},
            language=self.language or "auto",
            use_itn=True,
            batch_size_s=0,
            disable_pbar=True,
        )

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
