import logging
import numpy as np

log = logging.getLogger("LiveTranslate.AnimeWhisper")

MODEL_ID = "litagin/anime-whisper"


class AnimeWhisperEngine:
    """Speech-to-text using litagin/anime-whisper (kotoba-whisper-v2.0 fine-tune).

    Japanese-only, specialized for anime / galgame speech (sighs, breaths, etc.).
    Loaded via transformers pipeline; no faster-whisper / ctranslate2 path.
    """

    def __init__(self, device="cuda", hub="hf"):
        import torch
        from transformers import pipeline
        from model_manager import get_local_model_path

        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        local = get_local_model_path("anime-whisper", hub=hub)
        model = local or MODEL_ID

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._device = device
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=device,
            dtype=dtype,
            chunk_length_s=30.0,
            batch_size=1,
        )
        self.language = "ja"
        log.info(f"AnimeWhisper loaded from {model} on {device}")

    def set_language(self, language: str):
        # Model is Japanese-only; ignore attempts to change
        if language not in ("auto", "ja", None):
            log.info(f"AnimeWhisper is Japanese-only, ignoring language={language}")
        self.language = "ja"

    def to_device(self, device: str):
        try:
            import torch

            model = self._pipe.model
            if device == "cpu":
                model.to("cpu", dtype=torch.float32)
            else:
                model.to(device, dtype=torch.float16)
            self._pipe.device = model.device
            self._device = device
            log.info(f"AnimeWhisper moved to {device}")
            return True
        except Exception as e:
            log.warning(f"AnimeWhisper to_device failed: {e}")
            return False

    def unload(self):
        if hasattr(self, "_pipe") and self._pipe is not None:
            try:
                self._pipe.model.to("cpu")
            except Exception:
                pass
            self._pipe = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe audio segment (float32, 16kHz mono)."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        import torch

        # anime-whisper README: disable initial_prompt, suppress repetitions
        with torch.inference_mode():
            result = self._pipe(
                audio,
                generate_kwargs={
                    "language": "Japanese",
                    "task": "transcribe",
                    "do_sample": False,
                    "num_beams": 1,
                    "no_repeat_ngram_size": 5,
                    "repetition_penalty": 1.0,
                },
            )

        text = (result or {}).get("text", "").strip()
        if not text:
            return None

        return {
            "text": text,
            "language": "ja",
            "language_name": "ja",
        }
