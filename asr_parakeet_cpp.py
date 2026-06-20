import ctypes
import json
import logging
import os
from pathlib import Path

import numpy as np

from translator import LANGUAGE_DISPLAY

log = logging.getLogger("LiveTranslate.ParakeetCpp")

LANGUAGE_NAMES = {**LANGUAGE_DISPLAY, "auto": "auto"}
_LIBRARY_NAMES = (
    "parakeet.dll",
    "libparakeet.dll",
    "parakeet_capi.dll",
    "libparakeet_capi.dll",
)
_DECODER_IDS = {"auto": 0, "ctc": 1, "tdt": 2, "rnnt": 2}


def _find_library(runtime_dir: Path) -> Path | None:
    for name in _LIBRARY_NAMES:
        candidate = runtime_dir / name
        if candidate.is_file():
            return candidate
    for name in _LIBRARY_NAMES:
        matches = list(runtime_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _parse_device_index(device: str) -> int:
    device = str(device or "cpu").split(" (", 1)[0].strip()
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


class ParakeetCppEngine:
    """parakeet.cpp C API adapter for VAD-segmented 16 kHz mono float32 audio."""

    def __init__(
        self,
        model_path: str,
        runtime_dir: str,
        backend: str = "auto",
        decoder: str = "auto",
        device: str = "cpu",
        language: str = "auto",
        word_timestamps: bool = True,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.runtime_dir = str(Path(runtime_dir).resolve())
        self.backend = str(backend or "auto").lower()
        self.decoder = str(decoder or "auto").lower()
        self.device = str(device or "cpu")
        self.language = str(language or "auto")
        self.word_timestamps = bool(word_timestamps)
        self._dll_handles = []
        self._lib = None
        self._ctx = None
        self._abi_version = 0
        self._json_available = False

        if self.backend not in ("auto", "cpu", "cuda", "vulkan"):
            raise ValueError(f"Unsupported parakeet.cpp backend: {self.backend}")
        if self.decoder not in _DECODER_IDS:
            raise ValueError(f"Unsupported parakeet.cpp decoder: {self.decoder}")
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"parakeet.cpp model not found: {self.model_path}")
        runtime_path = Path(self.runtime_dir)
        if not runtime_path.is_dir():
            raise FileNotFoundError(f"parakeet.cpp runtime dir not found: {runtime_path}")

        library_path = _find_library(runtime_path)
        if not library_path:
            raise FileNotFoundError(
                f"parakeet.cpp shared library not found in runtime dir: {runtime_path}"
            )

        self._configure_backend_env()
        self._add_dll_directories(runtime_path, library_path.parent)
        self._lib = ctypes.CDLL(str(library_path))
        self._bind_api()
        if self._abi_version < 3:
            raise RuntimeError(
                f"parakeet.cpp C API ABI {self._abi_version} is too old; "
                "ABI >= 3 is required"
            )
        self._ctx = self._lib.parakeet_capi_load(self.model_path.encode("utf-8"))
        if not self._ctx:
            raise RuntimeError(f"parakeet.cpp failed to load model: {self._last_error()}")

        log.info(
            "parakeet.cpp loaded: "
            f"model={self.model_path}, runtime={self.runtime_dir}, "
            f"backend={self.backend}, decoder={self.decoder}, abi={self._abi_version}"
        )

    def _configure_backend_env(self):
        backend = self.backend
        device_index = _parse_device_index(self.device)
        if backend == "auto":
            backend = "cuda" if self.device.split(" (", 1)[0].startswith("cuda") else "cpu"
        if backend == "cpu":
            os.environ["PARAKEET_DEVICE"] = "cpu"
        elif backend == "cuda":
            os.environ["PARAKEET_DEVICE"] = f"CUDA{device_index}"
        elif backend == "vulkan":
            os.environ["PARAKEET_DEVICE"] = f"Vulkan{device_index}"

    def _add_dll_directories(self, runtime_path: Path, library_dir: Path):
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return
        dirs = {runtime_path, library_dir}
        for child in runtime_path.rglob("*"):
            if child.is_dir() and any(child.glob("*.dll")):
                dirs.add(child)
        for directory in dirs:
            try:
                self._dll_handles.append(os.add_dll_directory(str(directory)))
            except OSError:
                log.warning(f"Failed to add parakeet.cpp DLL directory: {directory}")

    def _bind_api(self):
        lib = self._lib
        lib.parakeet_capi_abi_version.argtypes = []
        lib.parakeet_capi_abi_version.restype = ctypes.c_int
        self._abi_version = int(lib.parakeet_capi_abi_version())

        lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
        lib.parakeet_capi_load.restype = ctypes.c_void_p
        lib.parakeet_capi_free.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_free.restype = None
        lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_free_string.restype = None

        lib.parakeet_capi_transcribe_pcm_lang.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        lib.parakeet_capi_transcribe_pcm_lang.restype = ctypes.c_void_p

        lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_last_error.restype = ctypes.c_char_p

        batch_json = getattr(lib, "parakeet_capi_transcribe_pcm_batch_json_lang", None)
        if batch_json is not None:
            batch_json.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,
            ]
            batch_json.restype = ctypes.c_void_p
            self._json_available = True

    def set_language(self, language: str):
        self.language = str(language or "auto")

    def unload(self):
        ctx = self._ctx
        self._ctx = None
        if ctx and self._lib:
            try:
                self._lib.parakeet_capi_free(ctx)
            except Exception:
                log.warning("parakeet.cpp context free failed", exc_info=True)
        for handle in self._dll_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._dll_handles.clear()

    def transcribe(
        self, audio: np.ndarray, word_timestamps: bool = False
    ) -> dict | None:
        if audio.size == 0:
            return None
        if audio.dtype != np.float32 or not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio, dtype=np.float32)
        want_words = bool(word_timestamps and self.word_timestamps)
        if want_words and self._json_available:
            try:
                return self._transcribe_json(audio)
            except Exception as exc:
                log.warning(
                    f"parakeet.cpp JSON timestamps failed, falling back to text: {exc}"
                )
        text = self._transcribe_text(audio)
        if not text:
            return None
        language = self.language or "auto"
        if language == "auto":
            language = "en"
        return {
            "text": text,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
        }

    def _target_lang_bytes(self) -> bytes:
        language = self.language or "auto"
        return language.encode("utf-8")

    def _decoder_id(self) -> int:
        return _DECODER_IDS.get(self.decoder, 0)

    def _transcribe_text(self, audio: np.ndarray) -> str:
        ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result_ptr = self._lib.parakeet_capi_transcribe_pcm_lang(
            self._ctx,
            ptr,
            ctypes.c_size_t(audio.size),
            16000,
            self._decoder_id(),
            self._target_lang_bytes(),
        )
        if not result_ptr:
            raise RuntimeError(f"parakeet.cpp transcription failed: {self._last_error()}")
        try:
            return ctypes.string_at(result_ptr).decode("utf-8", errors="replace").strip()
        finally:
            self._lib.parakeet_capi_free_string(result_ptr)

    def _transcribe_json(self, audio: np.ndarray) -> dict | None:
        sample_ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        batch_type = ctypes.POINTER(ctypes.c_float) * 1
        batch = batch_type(sample_ptr)
        lengths_type = ctypes.c_size_t * 1
        lengths = lengths_type(audio.size)
        fn = self._lib.parakeet_capi_transcribe_pcm_batch_json_lang
        result_ptr = fn(
            self._ctx,
            batch,
            lengths,
            1,
            16000,
            self._decoder_id(),
            self._target_lang_bytes(),
        )
        if not result_ptr:
            raise RuntimeError(f"parakeet.cpp JSON transcription failed: {self._last_error()}")
        try:
            raw = ctypes.string_at(result_ptr).decode("utf-8", errors="replace")
        finally:
            self._lib.parakeet_capi_free_string(result_ptr)
        data = json.loads(raw)
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return None
        text = str(data.get("text") or "").strip()
        if not text:
            return None
        language = self.language or "auto"
        if language == "auto":
            language = "en"
        result = {
            "text": text,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
        }
        words = []
        for item in data.get("words") or []:
            if not isinstance(item, dict):
                continue
            word = item.get("word", item.get("w", ""))
            if not word:
                continue
            words.append(
                {
                    "word": str(word),
                    "start": float(item.get("start", 0.0) or 0.0),
                    "end": float(item.get("end", 0.0) or 0.0),
                    "probability": float(item.get("probability", item.get("conf", 0.0)) or 0.0),
                }
            )
        if words:
            result["words"] = words
        return result

    def _last_error(self) -> str:
        if not self._lib:
            return "native library not loaded"
        try:
            value = self._lib.parakeet_capi_last_error(self._ctx)
        except Exception:
            return "unknown native error"
        if not value:
            return "unknown native error"
        return value.decode("utf-8", errors="replace")
