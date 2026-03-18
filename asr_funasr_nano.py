import logging
import os
import sys
import tempfile
import wave
import re
import numpy as np

log = logging.getLogger("LiveTrans.FunASR-Nano")

# Add bundled code to path so model.py can resolve its imports (ctc, tools.utils)
_NANO_DIR = os.path.join(os.path.dirname(__file__), "funasr_nano")


class FunASRNanoEngine:
    """Speech-to-text using Fun-ASR-Nano-2512 or Fun-ASR-MLT-Nano-2512."""

    def __init__(self, device="cuda", hub="ms", engine_type="funasr-nano"):
        if _NANO_DIR not in sys.path:
            sys.path.insert(0, _NANO_DIR)

        # Pre-register FunASRNano model class before AutoModel looks it up
        import model as _nano_model  # noqa: F401

        from funasr import AutoModel
        from model_manager import ASR_MODEL_IDS, get_local_model_path

        model_name = ASR_MODEL_IDS[engine_type]
        local = get_local_model_path(engine_type, hub=hub)
        model = local or model_name

        if local:
            self._ensure_qwen_weights(local)

        prev_cwd = os.getcwd()
        if local:
            os.chdir(local)
        try:
            self._model = AutoModel(
                model=model,
                trust_remote_code=True,
                device=device,
                hub=hub,
                disable_update=True,
            )
        finally:
            os.chdir(prev_cwd)
        self.language = None
        log.info(f"{engine_type} loaded: {model_name} on {device} (hub={hub})")

    @staticmethod
    def _ensure_qwen_weights(model_dir: str):
        qwen_dir = os.path.join(model_dir, "Qwen3-0.6B")
        if not os.path.isdir(qwen_dir):
            return
        if any(f.endswith((".safetensors", ".bin")) for f in os.listdir(qwen_dir)):
            return
        log.info("Downloading Qwen3-0.6B weights (one-time)...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            "Qwen/Qwen3-0.6B",
            local_dir=qwen_dir,
            ignore_patterns=["*.gguf"],
        )
        log.info("Qwen3-0.6B weights downloaded")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"Fun-ASR-Nano language: {old} -> {self.language}")

    def to_device(self, device: str):
        self._model.model.to(device)
        log.info(f"Fun-ASR-Nano moved to {device}")

    def unload(self):
        if hasattr(self, "_model") and self._model is not None:
            try:
                self._model.model.to("cpu")
            except Exception:
                pass
            self._model = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe audio segment (float32, 16kHz mono)."""
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            audio_16bit = (audio * 32767).astype(np.int16)
            with wave.open(tmp, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_16bit.tobytes())

            kwargs = {"input": [tmp], "batch_size": 1, "disable_pbar": True}
            if self.language:
                kwargs["language"] = self.language

            result = self._model.generate(**kwargs)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        if not result or not result[0].get("text"):
            return None

        raw_text = result[0]["text"]
        text = result[0].get("text_tn", raw_text) or raw_text

        # Clean special tags
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        if not text or text == "sil":
            return None

        detected_lang = self.language or self._guess_language(text)

        log.debug(f"Raw: {raw_text} | ITN: {text}")
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
