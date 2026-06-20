import inspect
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("LiveTranslate.CrispASR")


def _prepare_crispasr_dll_path():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    try:
        import crispasr
    except Exception:
        return
    package_dir = Path(crispasr.__file__).resolve().parent
    for path in (package_dir, package_dir / "bin"):
        if path.is_dir():
            os.add_dll_directory(str(path))


class CrispASREngine:
    """CrispASR Python binding adapter.

    The binding is intentionally imported only inside the worker process. The
    adapter accepts LiveTranslate's VAD-segmented 16 kHz mono float32 ndarray
    and normalizes CrispASR result variants into the existing ASR result dict.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        gpu_backend: str = "auto",
        device_index: int = 0,
        language: str = "auto",
        punc_model: str | None = "auto",
        n_threads: int | None = None,
        unified_memory: bool = True,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.backend = backend or "auto"
        self._session_backend = None if self.backend in ("", "auto") else self.backend
        self.gpu_backend = gpu_backend or "auto"
        self.device_index = int(device_index or 0)
        self.language = language if language and language != "auto" else None
        self.punc_model = None if punc_model in (None, "", "off", "none") else punc_model
        self._session = None
        self._punc = None

        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"CrispASR model file not found: {self.model_path}")

        os.environ.setdefault("CRISPASR_ARG_DEVICE", str(self.device_index))
        if unified_memory:
            os.environ.setdefault("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1")

        import crispasr
        _prepare_crispasr_dll_path()

        session_cls = getattr(crispasr, "Session", None)
        if session_cls is None:
            raise RuntimeError("crispasr.Session is not available")

        self._session = self._open_session(session_cls, n_threads=n_threads)
        self._punc = self._open_punctuation_model(crispasr)
        log.info(
            "CrispASR loaded: "
            f"model={self.model_path}, backend={self.backend}, "
            f"gpu_backend={self.gpu_backend}, device={self.device_index}"
        )

    def _open_session(self, session_cls, n_threads: int | None):
        kwargs = {
            "model_path": self.model_path,
            "backend": self._session_backend,
            "gpu_backend": self.gpu_backend,
            "device_index": self.device_index,
            "language": self.language,
            "punc_model": self.punc_model,
            "n_threads": n_threads,
        }
        aliases = {
            "model": self.model_path,
            "path": self.model_path,
            "model_file": self.model_path,
            "device": self.device_index,
            "threads": n_threads,
        }

        try:
            params = inspect.signature(session_cls).parameters
        except (TypeError, ValueError):
            params = {}

        attempts = []
        if params:
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in params and value is not None
            }
            accepted.update(
                {
                    key: value
                    for key, value in aliases.items()
                    if key in params and value is not None
                }
            )
            if "model_path" not in accepted and "model" not in accepted and "path" not in accepted:
                attempts.append((self.model_path, accepted))
            else:
                attempts.append((None, accepted))

        explicit = {
            key: value for key, value in kwargs.items() if value is not None
        }
        attempts.extend(
            [
                (self.model_path, {k: v for k, v in explicit.items() if k != "model_path"}),
                (self.model_path, {}),
                (None, {"model_path": self.model_path}),
            ]
        )

        last_error = None
        for arg0, attempt_kwargs in attempts:
            try:
                if arg0 is None:
                    return session_cls(**attempt_kwargs)
                return session_cls(arg0, **attempt_kwargs)
            except TypeError as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to create crispasr.Session: {last_error}")

    def _open_punctuation_model(self, crispasr):
        if not self.punc_model:
            return None
        punc_cls = getattr(crispasr, "PuncModel", None) or getattr(
            crispasr, "PunctuationModel", None
        )
        if punc_cls is None:
            return None
        try:
            return punc_cls(self.punc_model)
        except Exception as exc:
            log.warning(f"CrispASR punctuation model unavailable: {exc}")
            return None

    def set_language(self, language: str):
        self.language = language if language and language != "auto" else None
        for name in ("set_language", "set_lang"):
            fn = getattr(self._session, name, None)
            if fn:
                fn(self.language or "auto")
                return

    def unload(self):
        session = self._session
        self._session = None
        if session is None:
            return
        for name in ("close", "shutdown", "free", "reset"):
            fn = getattr(session, name, None)
            if fn:
                try:
                    fn()
                    break
                except Exception:
                    log.warning(f"CrispASR session {name} failed", exc_info=True)

    def transcribe(self, audio: np.ndarray) -> dict | None:
        if audio.size == 0:
            return None
        if audio.dtype != np.float32 or not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio, dtype=np.float32)

        result = self._call_transcribe(audio)
        normalized = self._normalize_result(result)
        if normalized and self._punc is not None:
            normalized["text"] = self._apply_punctuation(normalized["text"])
        return normalized

    def _call_transcribe(self, audio: np.ndarray):
        for name in ("transcribe", "recognize", "infer"):
            fn = getattr(self._session, name, None)
            if not fn:
                continue
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                params = {}
            kwargs = {}
            if "language" in params and self.language:
                kwargs["language"] = self.language
            if "sample_rate" in params:
                kwargs["sample_rate"] = 16000
            elif "sampling_rate" in params:
                kwargs["sampling_rate"] = 16000
            return fn(audio, **kwargs)
        raise RuntimeError("CrispASR session has no transcribe/recognize/infer method")

    def _apply_punctuation(self, text: str) -> str:
        if not text:
            return text
        for name in ("restore", "punctuate", "process", "apply"):
            fn = getattr(self._punc, name, None)
            if not fn:
                continue
            try:
                value = fn(text)
                return str(value).strip() or text
            except Exception as exc:
                log.warning(f"CrispASR punctuation failed: {exc}")
                return text
        return text

    def _normalize_result(self, result: Any) -> dict | None:
        if result is None:
            return None
        if isinstance(result, str):
            text = result.strip()
            return self._result(text) if text else None

        data = {"segments": result} if isinstance(result, (list, tuple)) else self._to_dict(result)
        text = self._extract_text(data).strip()
        if not text:
            return None

        language = (
            data.get("language")
            or data.get("language_detected")
            or data.get("lang")
            or self.language
            or "auto"
        )
        normalized = self._result(text, str(language))
        words = self._extract_words(data)
        if words:
            normalized["words"] = words
        return normalized

    def _result(self, text: str, language: str = "auto") -> dict:
        return {
            "text": text,
            "language": language,
            "language_name": language,
        }

    def _to_dict(self, value: Any) -> dict:
        if isinstance(value, dict):
            return value
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                pass
        if hasattr(value, "_asdict"):
            try:
                return value._asdict()
            except Exception:
                pass
        data = {}
        for name in (
            "text",
            "transcript",
            "segments",
            "language",
            "language_detected",
            "lang",
            "words",
            "word",
            "start",
            "end",
            "confidence",
            "probability",
        ):
            if hasattr(value, name):
                data[name] = getattr(value, name)
        return data

    def _extract_text(self, data: dict) -> str:
        text = data.get("text") or data.get("transcript")
        if text:
            return str(text)
        segments = data.get("segments") or data.get("result") or []
        if isinstance(segments, dict):
            segments = segments.values()
        parts = []
        for segment in segments:
            if isinstance(segment, str):
                parts.append(segment)
                continue
            seg_data = self._to_dict(segment)
            seg_text = seg_data.get("text") or seg_data.get("transcript")
            if seg_text:
                parts.append(str(seg_text))
        return " ".join(part.strip() for part in parts if part and part.strip())

    def _extract_words(self, data: dict) -> list[dict]:
        raw_words = data.get("words") or []
        segments = data.get("segments") or data.get("result") or []
        if not raw_words and segments:
            if isinstance(segments, dict):
                segments = segments.values()
            for segment in segments:
                seg_data = self._to_dict(segment)
                raw_words.extend(seg_data.get("words") or [])

        words = []
        for word in raw_words:
            word_data = self._to_dict(word)
            text = word_data.get("word") or word_data.get("text")
            if not text:
                continue
            words.append(
                {
                    "word": str(text),
                    "start": float(word_data.get("start") or 0.0),
                    "end": float(word_data.get("end") or 0.0),
                    "probability": float(
                        word_data.get("probability")
                        or word_data.get("confidence")
                        or 1.0
                    ),
                }
            )
        return words
