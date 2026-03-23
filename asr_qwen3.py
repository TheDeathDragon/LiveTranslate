import logging
import numpy as np

log = logging.getLogger("LiveTranslate.Qwen3ASR")

# Qwen3-ASR language code mapping (ISO 639-1 -> Qwen3 full name)
_LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}

# Reverse mapping for detected language
_LANG_REV = {v.lower(): k for k, v in _LANG_MAP.items()}


class Qwen3ASREngine:
    """Speech-to-text using Qwen3-ASR (ONNX + GGUF)."""

    def __init__(self, model_dir: str, use_dml: bool = True, chunk_size: float = 10.0):
        from qwen_asr_gguf.inference.schema import ASREngineConfig
        from qwen_asr_gguf.inference.asr import QwenASREngine

        config = ASREngineConfig(
            model_dir=model_dir,
            use_dml=use_dml,
            n_ctx=2048,
            chunk_size=chunk_size,
            memory_num=1,
            verbose=True,
            enable_aligner=False,
            pad_to=int(chunk_size),
        )
        self._engine = QwenASREngine(config)
        self.language = None  # None = auto detect
        self._context = ""  # Rolling context for better accuracy
        log.info(f"Qwen3-ASR loaded: {model_dir} (DML={use_dml})")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"Qwen3-ASR language: {old} -> {self.language}")

    def set_context(self, context: str):
        """Set context text for improved recognition accuracy."""
        self._context = context

    def to_device(self, device: str):
        # Qwen3-ASR uses ONNX+llama.cpp, no PyTorch device migration
        return False

    def unload(self):
        if hasattr(self, "_engine") and self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe audio segment.

        Args:
            audio: float32 numpy array, 16kHz mono

        Returns:
            dict with 'text', 'language', 'language_name' or None.
        """
        if self._engine is None:
            return None

        # Map language code to Qwen3 format
        qwen_lang = None
        if self.language:
            qwen_lang = _LANG_MAP.get(self.language)

        # Truncate to chunk_size to avoid memory overflow
        max_samples = int(self._engine.config.chunk_size * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        audio_dur = len(audio) / 16000
        ctx = self._context
        if ctx:
            ctx_limit = min(int(audio_dur * 20), 200)
            ctx = ctx[-ctx_limit:] if ctx_limit > 0 else None

        audio_embd, enc_time = self._engine.encoder.encode(audio)

        full_embd = self._engine._build_prompt_embd(
            audio_embd=audio_embd,
            prefix_text="",
            context=ctx,
            language=qwen_lang,
        )

        res = self._engine._safe_decode(
            full_embd,
            prefix_text="",
            rollback_num=5,
            is_last_chunk=True,
            temperature=0.4,
        )

        text = res.text.strip()
        if not text:
            return None

        # Update rolling context (keep last 200 chars)
        self._context = (self._context + text)[-200:]

        detected_lang = self.language or self._guess_language(text)

        log.debug(f"Qwen3-ASR result: {text}")
        return {
            "text": text,
            "language": detected_lang,
            "language_name": detected_lang,
        }

    def _guess_language(self, text: str) -> str:
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        jp = sum(
            1 for c in text if "\u3040" <= c <= "\u30ff" or "\u31f0" <= c <= "\u31ff"
        )
        ko = sum(1 for c in text if "\uac00" <= c <= "\ud7af")
        total = len(text)
        if total == 0:
            return "auto"
        if jp > 0:
            return "ja"
        if ko > total * 0.3:
            return "ko"
        if cjk > total * 0.3:
            return "zh"
        return "en"
