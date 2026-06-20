import gc
import importlib.metadata
import inspect
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from translator import LANGUAGE_DISPLAY

log = logging.getLogger("LiveTranslate.SherpaOnnx")

SENSE_VOICE_LANGUAGES = {"auto", "zh", "en", "ja", "ko", "yue"}
LANGUAGE_NAMES = {**LANGUAGE_DISPLAY, "auto": "auto", "yue": "Cantonese"}


def sherpa_onnx_cuda_wheel_available() -> bool:
    try:
        version = importlib.metadata.version("sherpa-onnx")
    except importlib.metadata.PackageNotFoundError:
        return False
    return "+cuda" in version.lower()


def _supports_kwarg(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _call_factory(factory, provider: str, kwargs: dict):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if _supports_kwarg(factory, "provider"):
        kwargs["provider"] = provider
    elif provider == "cuda":
        raise RuntimeError(
            "Installed sherpa-onnx recognizer API does not accept provider=; "
            "upgrade sherpa-onnx or use CPU provider"
        )
    else:
        log.warning("sherpa-onnx API has no provider parameter; loading CPU path")

    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        params = {}
    if params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return factory(**kwargs)


def _sense_voice_language(language: str) -> str:
    language = str(language or "auto").lower()
    return language if language in SENSE_VOICE_LANGUAGES else "auto"


def _whisper_language(language: str) -> str:
    language = str(language or "auto").lower()
    if language == "auto":
        return ""
    if language == "ja":
        return "jp"
    return language


class SherpaOnnxEngine:
    """sherpa-onnx recognizer adapter for VAD-segmented audio."""

    def __init__(
        self,
        model_path: str,
        model_info: dict,
        provider: str = "cpu",
        num_threads: int = 2,
        language: str = "auto",
        decoding_method: str = "greedy_search",
        left_padding_seconds: float = 0.3,
        tail_padding_seconds: float = 0.5,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.model_info = dict(model_info or {})
        self.family = self.model_info.get("family")
        self.provider = str(provider or "cpu").lower()
        self.num_threads = int(num_threads or 2)
        self.language = language or "auto"
        self.decoding_method = decoding_method or "greedy_search"
        self.left_padding_seconds = max(0.0, float(left_padding_seconds or 0.0))
        self.tail_padding_seconds = max(0.0, float(tail_padding_seconds or 0.0))
        self.sample_rate = int(self.model_info.get("sample_rate") or 16000)
        self._recognizer = None

        if self.provider not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported sherpa-onnx provider: {self.provider}")
        if self.provider == "cuda" and not sherpa_onnx_cuda_wheel_available():
            raise RuntimeError(
                "sherpa-onnx CUDA provider selected, but the installed package is "
                "not a CUDA wheel. Install sherpa-onnx==...+cuda or select CPU."
            )

        # Imported only in the ASR worker after CUDA_VISIBLE_DEVICES is finalized.
        import sherpa_onnx

        self._sherpa_onnx = sherpa_onnx
        self._load_recognizer()
        log.info(
            "sherpa-onnx loaded: "
            f"family={self.family}, model={self.model_path}, provider={self.provider}, "
            f"threads={self.num_threads}"
        )

    def _load_recognizer(self):
        if self.family == "online_transducer":
            self._load_online_transducer()
            return

        if self.family == "sense_voice":
            factory = self._sherpa_onnx.OfflineRecognizer.from_sense_voice
            self._recognizer = _call_factory(
                factory,
                self.provider,
                {
                    "model": self.model_info["model_file"],
                    "tokens": self.model_info["tokens_file"],
                    "num_threads": self.num_threads,
                    "language": _sense_voice_language(self.language),
                    "use_itn": True,
                    "debug": False,
                },
            )
            return

        if self.family == "paraformer":
            factory = self._sherpa_onnx.OfflineRecognizer.from_paraformer
            self._recognizer = _call_factory(
                factory,
                self.provider,
                {
                    "paraformer": self.model_info["model_file"],
                    "model": self.model_info["model_file"],
                    "tokens": self.model_info["tokens_file"],
                    "num_threads": self.num_threads,
                    "sample_rate": self.sample_rate,
                    "feature_dim": int(self.model_info.get("feature_dim") or 80),
                    "decoding_method": self.decoding_method,
                    "debug": False,
                },
            )
            return

        if self.family == "nemo_ctc":
            factory = self._sherpa_onnx.OfflineRecognizer.from_nemo_ctc
            self._recognizer = _call_factory(
                factory,
                self.provider,
                {
                    "model": self.model_info["model_file"],
                    "tokens": self.model_info["tokens_file"],
                    "num_threads": self.num_threads,
                    "sample_rate": self.sample_rate,
                    "feature_dim": int(self.model_info.get("feature_dim") or 80),
                    "decoding_method": self.decoding_method,
                    "debug": False,
                },
            )
            return

        if self.family == "moonshine":
            factory = self._sherpa_onnx.OfflineRecognizer.from_moonshine
            self._recognizer = _call_factory(
                factory,
                self.provider,
                {
                    "preprocessor": self.model_info["preprocessor_file"],
                    "preprocess": self.model_info["preprocessor_file"],
                    "encoder": self.model_info["encoder_file"],
                    "uncached_decoder": self.model_info["uncached_decoder_file"],
                    "cached_decoder": self.model_info["cached_decoder_file"],
                    "tokens": self.model_info["tokens_file"],
                    "num_threads": self.num_threads,
                    "decoding_method": self.decoding_method,
                    "debug": False,
                },
            )
            return

        if self.family == "whisper":
            factory = self._sherpa_onnx.OfflineRecognizer.from_whisper
            self._recognizer = _call_factory(
                factory,
                self.provider,
                {
                    "encoder": self.model_info["encoder_file"],
                    "decoder": self.model_info["decoder_file"],
                    "tokens": self.model_info["tokens_file"],
                    "num_threads": self.num_threads,
                    "decoding_method": self.decoding_method,
                    "language": _whisper_language(self.language),
                    "debug": False,
                },
            )
            return

        raise ValueError(f"Unsupported sherpa-onnx model family: {self.family}")

    def _load_online_transducer(self):
        factory = getattr(self._sherpa_onnx.OnlineRecognizer, "from_transducer", None)
        if factory is None:
            raise RuntimeError(
                "Installed sherpa-onnx OnlineRecognizer API does not provide "
                "from_transducer(); upgrade sherpa-onnx to use online transducer models"
            )

        self._recognizer = _call_factory(
            factory,
            self.provider,
            {
                "tokens": self.model_info["tokens_file"],
                "encoder": self.model_info["encoder_file"],
                "decoder": self.model_info["decoder_file"],
                "joiner": self.model_info["joiner_file"],
                "num_threads": self.num_threads,
                "sample_rate": self.sample_rate,
                "feature_dim": int(self.model_info.get("feature_dim") or 80),
                "decoding_method": self.decoding_method,
                "model_type": self.model_info.get("model_type"),
                "modeling_unit": self.model_info.get("modeling_unit"),
                "bpe_vocab": self.model_info.get("bpe_vocab"),
                "debug": False,
            },
        )

    def set_language(self, language: str):
        language = language or "auto"
        if language == self.language:
            return
        old = self.language
        self.language = language
        if self.family in ("sense_voice", "whisper"):
            self._load_recognizer()
        log.info(f"sherpa-onnx language: {old} -> {self.language}")

    def unload(self):
        self._recognizer = None
        gc.collect()

    def transcribe(self, audio: np.ndarray, word_timestamps: bool = False) -> dict | None:
        if audio.size == 0:
            return None
        if audio.dtype != np.float32 or not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio, dtype=np.float32)

        stream = self._recognizer.create_stream()
        if self.family == "online_transducer":
            result = self._decode_online_segment(stream, audio)
        else:
            stream.accept_waveform(self.sample_rate, audio)
            self._recognizer.decode_stream(stream)
            result = getattr(stream, "result", None)
        text = self._result_text(result).strip()
        language = self._result_language(result, text)
        text = self._strip_sense_voice_tags(text)
        if not text:
            return None

        normalized = {
            "text": text,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
        }
        if word_timestamps:
            words = self._extract_words(result)
            if words:
                normalized["words"] = words
        return normalized

    def _decode_online_segment(self, stream, audio: np.ndarray):
        set_option = getattr(stream, "set_option", None)
        if callable(set_option):
            language = str(self.language or "auto").strip()
            if language:
                set_option("language", language)

        if self.left_padding_seconds > 0:
            pad = np.zeros(
                int(round(self.sample_rate * self.left_padding_seconds)),
                dtype=np.float32,
            )
            if pad.size:
                audio = np.concatenate((pad, audio))

        if self.tail_padding_seconds > 0:
            pad = np.zeros(
                int(round(self.sample_rate * self.tail_padding_seconds)),
                dtype=np.float32,
            )
            if pad.size:
                audio = np.concatenate((audio, pad))

        stream.accept_waveform(self.sample_rate, audio)
        input_finished = getattr(stream, "input_finished", None)
        if callable(input_finished):
            input_finished()

        max_steps = max(100, int((audio.size / max(self.sample_rate, 1) + 2.0) * 100))
        steps = 0
        while self._recognizer.is_ready(stream) and steps < max_steps:
            self._recognizer.decode_stream(stream)
            steps += 1
        if steps >= max_steps:
            log.warning("sherpa-onnx online decode reached max decode steps")

        get_result_all = getattr(self._recognizer, "get_result_all", None)
        if callable(get_result_all):
            return get_result_all(stream)
        get_result = getattr(self._recognizer, "get_result", None)
        if callable(get_result):
            return get_result(stream)
        return getattr(stream, "result", None)

    def _result_text(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text")
        return str(text or "")

    def _result_language(self, result: Any, text: str) -> str:
        for attr in ("language", "lang"):
            value = getattr(result, attr, None)
            if value:
                return self._normalize_language(str(value))
            if isinstance(result, dict) and result.get(attr):
                return self._normalize_language(str(result[attr]))
        match = re.search(r"<\|([a-z]{2,3})\|>", text or "", re.IGNORECASE)
        if match:
            return self._normalize_language(match.group(1))
        return _sense_voice_language(self.language) if self.family == "sense_voice" else (
            self.language if self.language and self.language != "auto" else "auto"
        )

    def _normalize_language(self, language: str) -> str:
        language = language.strip().lower()
        return "ja" if language == "jp" else language

    def _strip_sense_voice_tags(self, text: str) -> str:
        return re.sub(r"<\|[^|]+?\|>", "", text or "").strip()

    def _extract_words(self, result: Any) -> list[dict]:
        words = []
        raw_words = getattr(result, "words", None)
        tokens = getattr(result, "tokens", None)
        timestamps = getattr(result, "timestamps", None)
        if isinstance(result, dict):
            raw_words = raw_words or result.get("words")
            tokens = tokens or result.get("tokens")
            timestamps = timestamps or result.get("timestamps")

        if raw_words:
            for word in raw_words:
                data = word if isinstance(word, dict) else vars(word)
                text = data.get("word") or data.get("text")
                if not text:
                    continue
                words.append(
                    {
                        "word": str(text),
                        "start": float(data.get("start") or 0.0),
                        "end": float(data.get("end") or 0.0),
                    }
                )
            return words

        if tokens and timestamps and len(tokens) == len(timestamps):
            for token, start in zip(tokens, timestamps, strict=False):
                words.append({"word": str(token), "start": float(start), "end": float(start)})
        return words
