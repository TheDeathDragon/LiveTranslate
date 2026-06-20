import os
import contextlib
import json
import logging
from pathlib import Path

log = logging.getLogger("LiveTranslate.ModelManager")

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@contextlib.contextmanager
def _proxy_env(proxy: str):
    """Temporarily route all download backends through a proxy.

    proxy:
        "system" / "" / None -> leave ambient env & OS proxy untouched
        "none"               -> force-disable any proxy for this download
        a URL                -> send urllib/requests/httpx traffic through it

    Covers torch.hub (urllib), huggingface_hub and modelscope (requests),
    which all honor the *_PROXY env vars; urllib additionally gets an explicit
    opener so a previously cached default opener cannot bypass the setting.
    """
    import urllib.request

    if proxy in ("system", "", None):
        yield
        return
    saved_env: dict = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}
    saved_no_proxy = os.environ.get("NO_PROXY")
    saved_opener = getattr(urllib.request, "_opener", None)
    try:
        if proxy == "none":
            for key in _PROXY_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ["NO_PROXY"] = "*"
            handler = urllib.request.ProxyHandler({})
        else:
            for key in _PROXY_ENV_KEYS:
                os.environ[key] = proxy
            os.environ.pop("NO_PROXY", None)
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        urllib.request.install_opener(urllib.request.build_opener(handler))
        log.info(f"Download proxy active: {proxy}")
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = saved_no_proxy
        urllib.request.install_opener(saved_opener)

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

ASR_MODEL_IDS = {
    "sensevoice": "iic/SenseVoiceSmall",
    "funasr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "funasr-mlt-nano": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    "anime-whisper": "litagin/anime-whisper",
}

FUNASR_MODEL_PROFILES = {
    "sensevoice-small": {
        "display_name": "SenseVoice Small",
        "family": "sensevoice",
        "legacy_engine": "sensevoice",
        "modelscope_id": "iic/SenseVoiceSmall",
        "huggingface_id": "FunAudioLLM/SenseVoiceSmall",
        "estimated_bytes": 940_000_000,
        "supports_padding": True,
        "supports_language": True,
    },
    "funasr-nano-2512": {
        "display_name": "Fun-ASR-Nano",
        "family": "funasr-nano",
        "legacy_engine": "funasr-nano",
        "modelscope_id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "huggingface_id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "estimated_bytes": 1_050_000_000,
        "supports_padding": False,
        "supports_language": True,
    },
    "funasr-mlt-nano-2512": {
        "display_name": "Fun-ASR-MLT-Nano",
        "family": "funasr-nano",
        "legacy_engine": "funasr-mlt-nano",
        "modelscope_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "huggingface_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "estimated_bytes": 1_050_000_000,
        "supports_padding": False,
        "supports_language": True,
    },
}

DEFAULT_FUNASR_MODEL = "sensevoice-small"
DEFAULT_SHERPA_ONNX_MODEL = ""
DEFAULT_FIRERED_VAD_MODEL = ""
DEFAULT_PARAKEET_CPP_MODEL = ""

FUNASR_LEGACY_ENGINE_ALIASES = {
    "sensevoice": "sensevoice-small",
    "funasr-nano": "funasr-nano-2512",
    "funasr-mlt-nano": "funasr-mlt-nano-2512",
}

# HuggingFace repo ids for engines whose namespace differs from ModelScope.
# SenseVoice lives under `iic/` on ModelScope but `FunAudioLLM/` on HuggingFace.
ASR_MODEL_IDS_HF = {
    "sensevoice": "FunAudioLLM/SenseVoiceSmall",
}


def asr_model_id(
    engine_type: str, hub: str = "ms", funasr_model: str | None = None
) -> str:
    """Return the repo id for an engine on the given hub ('ms' or 'hf')."""
    if engine_type == "funasr":
        return funasr_model_id(funasr_model, hub)
    if engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        return funasr_model_id(FUNASR_LEGACY_ENGINE_ALIASES[engine_type], hub)
    if hub == "hf" and engine_type in ASR_MODEL_IDS_HF:
        return ASR_MODEL_IDS_HF[engine_type]
    return ASR_MODEL_IDS[engine_type]

ASR_DISPLAY_NAMES = {
    "funasr": "FunASR",
    "sensevoice": "SenseVoice Small",
    "funasr-nano": "Fun-ASR-Nano",
    "funasr-mlt-nano": "Fun-ASR-MLT-Nano",
    "whisper": "Whisper",
    "anime-whisper": "Anime-Whisper",
    "crispasr": "CrispASR",
    "sherpa-onnx": "sherpa-onnx",
    "parakeet-cpp": "parakeet.cpp",
    "remote-whisper": "Remote-Whisper",
}

_CRISPASR_EXTS = {".gguf", ".bin"}
_CRISPASR_MIN_BYTES = 1_000_000
_PARAKEET_CPP_MIN_BYTES = 1_000_000
_PARAKEET_CPP_MODEL_PREFIXES = (
    "tdt_ctc-110m",
    "tdt_ctc-1.1b",
    "tdt-0.6b-v2",
    "tdt-0.6b-v3",
    "tdt-1.1b",
    "ctc-0.6b",
    "ctc-1.1b",
    "rnnt-0.6b",
    "rnnt-1.1b",
    "realtime_eou_120m-v1",
    "nemotron-3.5-asr-streaming-0.6b",
)
_PARAKEET_CPP_LIBRARY_NAMES = (
    "parakeet.dll",
    "libparakeet.dll",
    "parakeet_capi.dll",
    "libparakeet_capi.dll",
)

_MODEL_SIZE_BYTES = {
    "silero-vad": 2_000_000,
    "firered-vad": 2_200_000,
    "sensevoice": 940_000_000,
    "funasr-nano": 1_050_000_000,
    "funasr-mlt-nano": 1_050_000_000,
    "whisper-tiny": 78_000_000,
    "whisper-base": 148_000_000,
    "whisper-small": 488_000_000,
    "whisper-medium": 1_530_000_000,
    "whisper-large-v3": 3_100_000_000,
    "anime-whisper": 3_100_000_000,
}

_WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

_CACHE_MODELS = [
    ("SenseVoice Small", "funasr", "sensevoice-small"),
    ("Fun-ASR-Nano", "funasr", "funasr-nano-2512"),
    ("Fun-ASR-MLT-Nano", "funasr", "funasr-mlt-nano-2512"),
    ("Anime-Whisper", "anime-whisper"),
]


def normalize_funasr_model_key(model_key: str | None) -> str:
    if model_key in FUNASR_MODEL_PROFILES:
        return model_key
    if model_key in FUNASR_LEGACY_ENGINE_ALIASES:
        return FUNASR_LEGACY_ENGINE_ALIASES[model_key]
    return DEFAULT_FUNASR_MODEL


def normalize_asr_engine_selection(
    engine_type: str | None, funasr_model: str | None = None
) -> tuple[str, str]:
    if engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        return "funasr", FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
    if engine_type == "funasr":
        return "funasr", normalize_funasr_model_key(funasr_model)
    return engine_type or "funasr", normalize_funasr_model_key(funasr_model)


def migrate_funasr_settings(settings: dict | None) -> dict | None:
    if not settings:
        return settings
    engine, model_key = normalize_asr_engine_selection(
        settings.get("asr_engine"), settings.get("funasr_model")
    )
    settings["asr_engine"] = engine
    if engine == "funasr":
        settings["funasr_model"] = model_key
    else:
        settings.setdefault("funasr_model", DEFAULT_FUNASR_MODEL)
    return settings


def funasr_profile(model_key: str | None) -> dict:
    return FUNASR_MODEL_PROFILES[normalize_funasr_model_key(model_key)]


def funasr_model_options() -> list[tuple[str, str]]:
    return [
        (key, profile["display_name"])
        for key, profile in FUNASR_MODEL_PROFILES.items()
    ]


def funasr_display_name(model_key: str | None) -> str:
    return funasr_profile(model_key)["display_name"]


def funasr_supports_padding(model_key: str | None) -> bool:
    return bool(funasr_profile(model_key).get("supports_padding"))


def funasr_model_id(model_key: str | None, hub: str = "ms") -> str:
    profile = funasr_profile(model_key)
    return profile["huggingface_id"] if hub == "hf" else profile["modelscope_id"]


def _custom_parakeet_cpp_path(value) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def _read_parakeet_cpp_sidecar(path: Path) -> dict:
    candidates = [
        path.with_suffix(path.suffix + ".json"),
        path.with_suffix(".json"),
        path.parent / "parakeet_cpp_model.json",
    ]
    for metadata_path in candidates:
        if not metadata_path.is_file():
            continue
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"Invalid parakeet.cpp metadata: {metadata_path}: {exc}")
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _parakeet_cpp_name_hint(path: Path) -> dict | None:
    name = path.name.lower()
    stem = path.stem.lower()
    if not any(stem.startswith(prefix) for prefix in _PARAKEET_CPP_MODEL_PREFIXES):
        return None
    tags = []
    if "nemotron" in stem:
        tags.append("multilingual")
    if "realtime_eou" in stem or "streaming" in stem:
        tags.append("streaming/eou")
    return {
        "display_name": stem.replace("_", " "),
        "decoder": "auto",
        "language": "auto" if "nemotron" in stem else "en",
        "tags": tags,
        "filename": name,
    }


def detect_parakeet_cpp_model_file(path) -> dict | None:
    """Return normalized parakeet.cpp metadata for a known GGUF model file."""
    if not path:
        return None
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        return None
    try:
        if path.stat().st_size < _PARAKEET_CPP_MIN_BYTES:
            return None
        path = path.resolve()
    except OSError:
        return None

    metadata = _read_parakeet_cpp_sidecar(path)
    family = str(metadata.get("family") or "").strip().lower().replace("-", "_")
    architecture = str(
        metadata.get("architecture")
        or metadata.get("gguf.architecture")
        or metadata.get("gguf_architecture")
        or ""
    ).strip().lower()
    model_file = metadata.get("model_file")
    if model_file:
        candidate = Path(str(model_file))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        try:
            if candidate.resolve() != path:
                return None
        except OSError:
            return None

    hint = _parakeet_cpp_name_hint(path)
    if family in ("parakeet_cpp", "parakeet.cpp") or architecture == "parakeet":
        display = metadata.get("display_name") or metadata.get("name")
        return {
            "path": str(path),
            "display_name": str(display).strip() if display else (hint or {}) .get("display_name", path.name),
            "decoder": str(metadata.get("decoder") or (hint or {}).get("decoder") or "auto"),
            "language": str(metadata.get("language") or (hint or {}).get("language") or "auto"),
            "tags": metadata.get("tags") if isinstance(metadata.get("tags"), list) else (hint or {}).get("tags", []),
        }
    if hint:
        return {"path": str(path), **hint}
    return None


def is_parakeet_cpp_model_file(path) -> bool:
    return detect_parakeet_cpp_model_file(path) is not None


def resolve_custom_parakeet_cpp_model(value) -> str | None:
    path = _custom_parakeet_cpp_path(value)
    if path and is_parakeet_cpp_model_file(path):
        return str(path.resolve())
    return None


def list_local_parakeet_cpp_models() -> list[dict]:
    """Scan ./models for recognizable parakeet.cpp GGUF model files."""
    if not MODELS_DIR.exists():
        return []

    entries = []
    name_counts = {}
    seen = set()
    try:
        files = list(MODELS_DIR.rglob("*.gguf"))
    except (OSError, PermissionError):
        return []

    for path in files:
        info = detect_parakeet_cpp_model_file(path)
        if not info:
            continue
        identity = info["path"]
        if identity in seen:
            continue
        seen.add(identity)
        name = str(info.get("display_name") or path.name)
        tags = info.get("tags") or []
        if tags:
            name = f"{name} [{' / '.join(str(tag) for tag in tags)}]"
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{path.stem} ({path.parent.name}){path.suffix}"
        entries.append(
            {
                "name": name,
                "path": identity,
                "decoder": info.get("decoder", "auto"),
                "language": info.get("language", "auto"),
                "info": info,
            }
        )

    entries.sort(key=lambda item: item["name"].lower())
    return entries


def get_parakeet_cpp_model_path(value) -> str | None:
    return resolve_custom_parakeet_cpp_model(value)


def local_parakeet_cpp_display_name(path) -> str | None:
    resolved = resolve_custom_parakeet_cpp_model(path)
    if not resolved:
        return None
    for item in list_local_parakeet_cpp_models():
        if item["path"] == resolved:
            return item["name"]
    info = detect_parakeet_cpp_model_file(resolved)
    return info["display_name"] if info else Path(resolved).name


def _custom_parakeet_cpp_runtime_path(value) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def _find_parakeet_cpp_library(path: Path) -> Path | None:
    for name in _PARAKEET_CPP_LIBRARY_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    for name in _PARAKEET_CPP_LIBRARY_NAMES:
        matches = list(path.rglob(name))
        if matches:
            return matches[0]
    return None


def _parakeet_cpp_runtime_backend_hint(path: Path) -> str:
    lowered = " ".join(part.lower() for part in path.parts)
    if "cuda" in lowered:
        return "cuda"
    if "vulkan" in lowered:
        return "vulkan"
    if "cpu" in lowered:
        return "cpu"
    return "unknown"


def detect_parakeet_cpp_runtime_dir(path) -> dict | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_dir():
        return None
    try:
        path = path.resolve()
    except OSError:
        return None
    library = _find_parakeet_cpp_library(path)
    if not library:
        return None
    backend = _parakeet_cpp_runtime_backend_hint(path)
    missing = []
    if backend == "cuda":
        has_cudart = any(path.rglob("cudart*.dll"))
        if not has_cudart:
            missing.append("cudart*.dll")
    return {
        "path": str(path),
        "library": str(library),
        "backend": backend,
        "display_name": parakeet_cpp_runtime_display_name(path),
        "missing_dependencies": missing,
    }


def resolve_parakeet_cpp_runtime_dir(value, backend: str = "auto") -> str | None:
    path = _custom_parakeet_cpp_runtime_path(value)
    if path:
        info = detect_parakeet_cpp_runtime_dir(path)
        if info and (
            backend in ("", "auto")
            or info["backend"] in ("unknown", backend)
        ):
            return info["path"]
    return None


def list_local_parakeet_cpp_runtimes() -> list[dict]:
    if not MODELS_DIR.exists():
        return []

    runtime_root = MODELS_DIR / "parakeet.cpp" / "runtime"
    candidates = []
    try:
        if any((MODELS_DIR / name).is_file() for name in _PARAKEET_CPP_LIBRARY_NAMES):
            candidates.append(MODELS_DIR)
        candidates.extend(
            path
            for path in MODELS_DIR.iterdir()
            if path.is_dir() and "parakeet" in path.name.lower()
        )
        if runtime_root.exists():
            candidates.append(runtime_root)
            candidates.extend(path for path in runtime_root.rglob("*") if path.is_dir())
    except (OSError, PermissionError):
        return []

    entries = []
    seen = set()
    for path in candidates:
        info = detect_parakeet_cpp_runtime_dir(path)
        if not info or info["path"] in seen:
            continue
        seen.add(info["path"])
        entries.append(
            {
                "name": info["display_name"],
                "path": info["path"],
                "backend": info["backend"],
                "info": info,
            }
        )
    entries.sort(key=lambda item: item["name"].lower())
    return entries


def parakeet_cpp_runtime_display_name(path) -> str:
    path = Path(path)
    backend = _parakeet_cpp_runtime_backend_hint(path)
    name = path.name
    if backend != "unknown" and backend not in name.lower():
        return f"{name} [{backend}]"
    return name


def is_crispasr_model_file(path) -> bool:
    if not path:
        return False
    path = Path(path)
    try:
        return (
            path.is_file()
            and path.suffix.lower() in _CRISPASR_EXTS
            and path.stat().st_size >= _CRISPASR_MIN_BYTES
            and not is_parakeet_cpp_model_file(path)
        )
    except OSError:
        return False


def _custom_crispasr_path(value) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def resolve_custom_crispasr_model(value) -> str | None:
    path = _custom_crispasr_path(value)
    if path and is_crispasr_model_file(path):
        return str(path.absolute())
    return None


def _custom_whisper_path(value) -> Path | None:
    if not value or value in _WHISPER_SIZES:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def is_faster_whisper_model_dir(path) -> bool:
    """True when path looks like a CTranslate2 faster-whisper model directory."""
    if not path:
        return False
    path = Path(path)
    return (
        path.is_dir()
        and (path / "model.bin").is_file()
        and (path / "config.json").is_file()
    )


def resolve_custom_whisper_model(value) -> str | None:
    path = _custom_whisper_path(value)
    if path and is_faster_whisper_model_dir(path):
        return str(path.resolve())
    return None


def _is_builtin_whisper_cache(path: Path) -> bool:
    parts = set(path.parts)
    return any(f"models--Systran--faster-whisper-{s}" in parts for s in _WHISPER_SIZES)


def _is_hf_hub_cache(path: Path) -> bool:
    parts = path.parts
    marker = ("huggingface", "hub")
    return any(parts[i : i + 2] == marker for i in range(len(parts) - 1))


def _hf_snapshot_name(path: Path) -> str | None:
    """Return 'org/repo' for .../models--org--repo/snapshots/<hash>."""
    if path.parent.name != "snapshots":
        return None
    repo_dir = path.parent.parent
    if not repo_dir.name.startswith("models--"):
        return None

    encoded = repo_dir.name.removeprefix("models--")
    parts = encoded.split("--", 1)
    if len(parts) != 2 or not all(parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def list_local_faster_whisper_models() -> list[dict]:
    """Scan ./models for user-provided faster-whisper model directories."""
    if not MODELS_DIR.exists():
        return []

    entries = []
    name_counts = {}
    seen = set()
    try:
        model_bins = list(MODELS_DIR.rglob("model.bin"))
    except (OSError, PermissionError):
        return []

    for model_bin in model_bins:
        model_dir = model_bin.parent
        if _is_builtin_whisper_cache(model_dir):
            continue
        if not is_faster_whisper_model_dir(model_dir):
            continue
        try:
            resolved = str(model_dir.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        name = _hf_snapshot_name(model_dir) or model_dir.name
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{name} ({model_dir.name[:8]})"
        entries.append({"name": name, "path": resolved})

    entries.sort(key=lambda item: item["name"].lower())
    return entries

def local_faster_whisper_display_name(path) -> str | None:
    """Return the same display name used by the local Whisper model selector."""
    resolved = resolve_custom_whisper_model(path)
    if not resolved:
        return None
    for item in list_local_faster_whisper_models():
        if item["path"] == resolved:
            return item["name"]
    return _hf_snapshot_name(Path(resolved)) or Path(resolved).name


def list_local_crispasr_models() -> list[dict]:
    """Scan ./models for user-provided CrispASR single-file models."""
    if not MODELS_DIR.exists():
        return []

    entries = []
    name_counts = {}
    seen = set()
    try:
        files = [
            path
            for ext in _CRISPASR_EXTS
            for path in MODELS_DIR.rglob(f"*{ext}")
        ]
    except (OSError, PermissionError):
        return []

    for path in files:
        if path.name == "model.bin" and is_faster_whisper_model_dir(path.parent):
            continue
        if not is_crispasr_model_file(path):
            continue
        try:
            identity = str(path.resolve())
            model_path = str(path.absolute())
        except OSError:
            continue
        if identity in seen:
            continue
        seen.add(identity)

        name = path.name
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{path.stem} ({path.parent.name}){path.suffix}"
        entries.append({"name": name, "path": model_path})

    entries.sort(key=lambda item: item["name"].lower())
    return entries


def local_crispasr_display_name(path) -> str | None:
    resolved = resolve_custom_crispasr_model(path)
    if not resolved:
        return None
    for item in list_local_crispasr_models():
        if item["path"] == resolved:
            return item["name"]
    return Path(resolved).name


def _custom_sherpa_onnx_path(value) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def _read_sherpa_onnx_metadata(path: Path) -> dict | None:
    metadata_path = path / "sherpa_onnx_model.json"
    if not metadata_path.is_file():
        return None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(f"Invalid sherpa-onnx metadata: {metadata_path}: {exc}")
        return {"_invalid": True}
    return data if isinstance(data, dict) else {"_invalid": True}


def _sherpa_file(path: Path, metadata: dict, *keys: str, default: str | None = None):
    for key in keys:
        value = metadata.get(key)
        if value:
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = path / candidate
            if candidate.is_file():
                return str(candidate.resolve())
    if default:
        candidate = path / default
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _first_glob_file(path: Path, *patterns: str) -> str | None:
    for pattern in patterns:
        matches = sorted(path.glob(pattern))
        for match in matches:
            if match.is_file():
                return str(match.resolve())
    return None


def _sherpa_family_hint(path: Path, metadata: dict) -> str | None:
    family = str(metadata.get("family") or "").strip().lower().replace("-", "_")
    if family:
        aliases = {
            "sensevoice": "sense_voice",
            "sense_voice": "sense_voice",
            "paraformer": "paraformer",
            "moonshine": "moonshine",
            "nemo_ctc": "nemo_ctc",
            "nemo-ctc": "nemo_ctc",
            "whisper": "whisper",
            "online_transducer": "online_transducer",
            "online-transducer": "online_transducer",
        }
        return aliases.get(family)
    return None


def _sherpa_display_name(path: Path, metadata: dict) -> str:
    display_name = metadata.get("display_name") or metadata.get("name")
    if display_name:
        return str(display_name).strip()
    return _hf_snapshot_name(path) or path.name


def _is_sherpa_online_transducer_dir(path: Path, metadata: dict, family: str | None) -> bool:
    encoder = _sherpa_file(
        path, metadata, "encoder", "encoder_file", default="encoder.onnx"
    ) or _first_glob_file(path, "encoder.int8.onnx", "encoder*.onnx")
    decoder = _sherpa_file(
        path, metadata, "decoder", "decoder_file", default="decoder.onnx"
    ) or _first_glob_file(path, "decoder.int8.onnx", "decoder*.onnx")
    joiner = _sherpa_file(
        path, metadata, "joiner", "joiner_file", default="joiner.onnx"
    ) or _first_glob_file(path, "joiner.int8.onnx", "joiner*.onnx")
    tokens = _sherpa_file(path, metadata, "tokens", "tokens_file", default="tokens.txt")
    if encoder and decoder and joiner and tokens:
        return True

    if family == "online_transducer":
        joint = _sherpa_file(path, metadata, "joint", "joint_file", default="joint.onnx")
        return bool(encoder and decoder and joint and tokens)
    return False


def _has_sherpa_prefix(path: Path) -> bool:
    name = path.name.lower().replace("-", "_")
    return name.startswith("sherpa_onnx")


def _is_sherpa_single_model_dir(path: Path, metadata: dict) -> bool:
    model = (
        _sherpa_file(path, metadata, "model", "model_file")
        or _first_glob_file(path, "model.int8.onnx", "model.onnx")
    )
    tokens = _sherpa_file(path, metadata, "tokens", "tokens_file", default="tokens.txt")
    return bool(model and tokens)


def detect_sherpa_onnx_model_dir(path) -> dict | None:
    """Return normalized sherpa-onnx model metadata when a directory is usable."""
    if not path:
        return None
    path = Path(path)
    if not path.is_dir():
        return None
    try:
        path = path.resolve()
    except OSError:
        return None

    metadata = _read_sherpa_onnx_metadata(path) or {}
    if metadata.get("_invalid"):
        return None

    family = _sherpa_family_hint(path, metadata)
    if family is None and _is_sherpa_online_transducer_dir(path, metadata, family):
        family = "online_transducer"
    if family is None and _has_sherpa_prefix(path) and _is_sherpa_single_model_dir(path, metadata):
        family = "nemo_ctc"
    tokens_file = _sherpa_file(path, metadata, "tokens", "tokens_file", default="tokens.txt")
    sample_rate = int(metadata.get("sample_rate") or 16000)
    feature_dim = int(metadata.get("feature_dim") or 80)

    base = {
        "path": str(path),
        "family": family,
        "display_name": _sherpa_display_name(path, metadata),
        "sample_rate": sample_rate,
        "feature_dim": feature_dim,
    }

    if family == "sense_voice":
        model_file = (
            _sherpa_file(path, metadata, "model", "model_file")
            or _first_glob_file(path, "model.int8.onnx", "model.onnx")
        )
        if tokens_file and model_file:
            return {**base, "tokens_file": tokens_file, "model_file": model_file}
        return None

    if family == "paraformer":
        model_file = (
            _sherpa_file(path, metadata, "model", "model_file", "paraformer")
            or _first_glob_file(path, "model.int8.onnx", "model.onnx")
        )
        if tokens_file and model_file:
            return {**base, "tokens_file": tokens_file, "model_file": model_file}
        return None

    if family == "nemo_ctc":
        model_file = (
            _sherpa_file(path, metadata, "model", "model_file")
            or _first_glob_file(path, "model.int8.onnx", "model.onnx")
        )
        if tokens_file and model_file:
            return {**base, "tokens_file": tokens_file, "model_file": model_file}
        return None

    if family == "moonshine":
        preprocessor = (
            _sherpa_file(path, metadata, "preprocessor", "preprocessor_file", "preprocess")
            or _first_glob_file(path, "preprocess.onnx", "preprocessor.onnx")
        )
        encoder = _sherpa_file(path, metadata, "encoder", "encoder_file") or _first_glob_file(
            path, "encode*.onnx", "encoder*.onnx"
        )
        uncached_decoder = _sherpa_file(
            path, metadata, "uncached_decoder", "uncached_decoder_file"
        ) or _first_glob_file(path, "uncached_decode*.onnx", "uncached_decoder*.onnx")
        cached_decoder = _sherpa_file(
            path, metadata, "cached_decoder", "cached_decoder_file"
        ) or _first_glob_file(path, "cached_decode*.onnx", "cached_decoder*.onnx")
        if tokens_file and preprocessor and encoder and uncached_decoder and cached_decoder:
            return {
                **base,
                "tokens_file": tokens_file,
                "preprocessor_file": preprocessor,
                "encoder_file": encoder,
                "uncached_decoder_file": uncached_decoder,
                "cached_decoder_file": cached_decoder,
            }
        return None

    if family == "whisper":
        encoder = _sherpa_file(path, metadata, "encoder", "encoder_file") or _first_glob_file(
            path, "*encoder*.onnx"
        )
        decoder = _sherpa_file(path, metadata, "decoder", "decoder_file") or _first_glob_file(
            path, "*decoder*.onnx"
        )
        tokens = tokens_file or _first_glob_file(path, "*tokens.txt")
        if encoder and decoder and tokens:
            return {**base, "tokens_file": tokens, "encoder_file": encoder, "decoder_file": decoder}
        return None

    if family == "online_transducer":
        encoder = _sherpa_file(
            path, metadata, "encoder", "encoder_file", default="encoder.onnx"
        ) or _first_glob_file(path, "encoder.int8.onnx", "encoder*.onnx")
        decoder = _sherpa_file(
            path, metadata, "decoder", "decoder_file", default="decoder.onnx"
        ) or _first_glob_file(path, "decoder.int8.onnx", "decoder*.onnx")
        joiner = _sherpa_file(
            path, metadata, "joiner", "joiner_file", default="joiner.onnx"
        ) or _first_glob_file(path, "joiner.int8.onnx", "joiner*.onnx")
        if not joiner:
            joiner = _sherpa_file(path, metadata, "joint", "joint_file", default="joint.onnx")
        if encoder and decoder and joiner and tokens_file:
            info = {
                **base,
                "tokens_file": tokens_file,
                "encoder_file": encoder,
                "decoder_file": decoder,
                "joiner_file": joiner,
            }
            for key in ("model_type", "modeling_unit", "bpe_vocab"):
                value = metadata.get(key)
                if value:
                    info[key] = str(value)
            return info
        return None

    return None


def is_sherpa_onnx_model_dir(path) -> bool:
    return detect_sherpa_onnx_model_dir(path) is not None


def resolve_custom_sherpa_onnx_model(value) -> str | None:
    path = _custom_sherpa_onnx_path(value)
    if path and is_sherpa_onnx_model_dir(path):
        return str(path.resolve())
    return None


def list_local_sherpa_onnx_models() -> list[dict]:
    """Scan ./models recursively for recognizable local sherpa-onnx models."""
    if not MODELS_DIR.exists():
        return []

    entries = []
    name_counts = {}
    seen = set()
    try:
        dirs = [MODELS_DIR, *[p for p in MODELS_DIR.rglob("*") if p.is_dir()]]
    except (OSError, PermissionError):
        return []

    for model_dir in dirs:
        info = detect_sherpa_onnx_model_dir(model_dir)
        if not info:
            continue
        identity = info["path"]
        if identity in seen:
            continue
        seen.add(identity)
        name = info["display_name"]
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{name} ({model_dir.parent.name})"
        entries.append(
            {
                "name": name,
                "path": identity,
                "family": info["family"],
                "info": info,
            }
        )

    entries.sort(key=lambda item: item["name"].lower())
    return entries


def local_sherpa_onnx_display_name(path) -> str | None:
    resolved = resolve_custom_sherpa_onnx_model(path)
    if not resolved:
        return None
    for item in list_local_sherpa_onnx_models():
        if item["path"] == resolved:
            return item["name"]
    info = detect_sherpa_onnx_model_dir(resolved)
    return info["display_name"] if info else Path(resolved).name


def get_sherpa_onnx_model_path(value) -> str | None:
    return resolve_custom_sherpa_onnx_model(value)


def _custom_firered_vad_path(value) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def _read_firered_vad_metadata(path: Path) -> dict:
    for name in ("firered_vad_model.json", "model.json"):
        metadata_path = path / name
        if not metadata_path.is_file():
            continue
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"Invalid FireRedVAD metadata: {metadata_path}: {exc}")
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _firered_family_hint(metadata: dict) -> str:
    family = str(metadata.get("family") or "").strip().lower().replace("-", "_")
    if family in ("stream_vad", "streamvad", "firered_stream_vad"):
        return "stream_vad"
    return ""


def _is_firered_stream_files(path: Path) -> bool:
    return (path / "cmvn.ark").is_file() and (path / "model.pth.tar").is_file()


def _is_stream_vad_name(path: Path) -> bool:
    name = path.name.lower().replace("_", "-")
    return "stream-vad" in name or name == "streamvad"


def _firered_display_name(path: Path, metadata: dict) -> str:
    display_name = metadata.get("display_name") or metadata.get("name")
    if display_name:
        return str(display_name).strip()

    repo_name = _hf_snapshot_name(path) or _hf_snapshot_name(path.parent)
    if repo_name:
        return f"{repo_name} / {path.name}"
    if _is_stream_vad_name(path) and path.parent != MODELS_DIR:
        return f"{path.parent.name} / {path.name}"
    return path.name


def detect_firered_vad_model_dir(path) -> dict | None:
    """Return normalized FireRedVAD Stream-VAD metadata for a usable directory."""
    if not path:
        return None
    path = Path(path)
    if not path.is_dir():
        return None
    try:
        path = path.resolve()
    except OSError:
        return None

    root_metadata = _read_firered_vad_metadata(path)
    root_family = _firered_family_hint(root_metadata)

    candidate: Path | None = None
    candidate_metadata: dict = {}

    if _is_firered_stream_files(path) and (
        _is_stream_vad_name(path) or root_family == "stream_vad"
    ):
        candidate = path
        candidate_metadata = root_metadata

    if candidate is None:
        model_dir = root_metadata.get("model_dir")
        if model_dir:
            child = Path(str(model_dir))
            if not child.is_absolute():
                child = path / child
            if child.is_dir() and _is_firered_stream_files(child):
                candidate = child.resolve()
                candidate_metadata = {
                    **root_metadata,
                    **_read_firered_vad_metadata(candidate),
                }

    if candidate is None:
        for child_name in ("Stream-VAD", "stream-vad", "Stream_VAD", "stream_vad"):
            child = path / child_name
            if child.is_dir() and _is_firered_stream_files(child):
                candidate = child.resolve()
                candidate_metadata = {
                    **root_metadata,
                    **_read_firered_vad_metadata(candidate),
                }
                break

    if candidate is None:
        return None

    return {
        "name": _firered_display_name(candidate, candidate_metadata),
        "path": str(candidate),
        "family": "stream_vad",
        "display_name": _firered_display_name(candidate, candidate_metadata),
    }


def is_firered_vad_stream_model_dir(path) -> bool:
    return detect_firered_vad_model_dir(path) is not None


def resolve_custom_firered_vad_model(value) -> str | None:
    path = _custom_firered_vad_path(value)
    if not path:
        return None
    info = detect_firered_vad_model_dir(path)
    if info:
        return info["path"]
    return None


def list_local_firered_vad_models() -> list[dict]:
    """Scan ./models recursively for local FireRedVAD Stream-VAD models."""
    if not MODELS_DIR.exists():
        return []

    candidates: set[Path] = set()
    try:
        for marker in MODELS_DIR.rglob("cmvn.ark"):
            if not marker.is_file():
                continue
            model_dir = marker.parent
            candidates.add(model_dir)
            if _is_stream_vad_name(model_dir):
                candidates.add(model_dir.parent)
    except (OSError, PermissionError):
        return []

    entries = []
    name_counts = {}
    seen = set()
    for candidate in sorted(candidates, key=lambda item: str(item).lower()):
        info = detect_firered_vad_model_dir(candidate)
        if not info:
            continue
        identity = info["path"]
        if identity in seen:
            continue
        seen.add(identity)
        name = info["display_name"]
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{name} ({Path(identity).parent.name})"
        entries.append({"name": name, "path": identity, "family": "stream_vad"})

    entries.sort(key=lambda item: item["name"].lower())
    return entries


def get_firered_vad_model_path(value) -> str | None:
    return resolve_custom_firered_vad_model(value)


def firered_vad_display_name(path) -> str | None:
    resolved = resolve_custom_firered_vad_model(path)
    if not resolved:
        return None
    for item in list_local_firered_vad_models():
        if item["path"] == resolved:
            return item["name"]
    info = detect_firered_vad_model_dir(resolved)
    return info["display_name"] if info else Path(resolved).name


def apply_cache_env():
    """Point all model caches to ./models/."""
    resolved = str(MODELS_DIR.resolve())
    os.environ["MODELSCOPE_CACHE"] = os.path.join(resolved, "modelscope")
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(resolved, "torch")
    log.info(f"Cache env set: {resolved}")


def _has_silero_pkg() -> bool:
    """True when the silero-vad PyPI package (model bundled in wheel) is installed."""
    import importlib.util

    return importlib.util.find_spec("silero_vad") is not None


def is_silero_cached() -> bool:
    if _has_silero_pkg():
        return True
    torch_hub = MODELS_DIR / "torch" / "hub"
    return any(torch_hub.glob("snakers4_silero-vad*")) if torch_hub.exists() else False


def _ms_model_path(org, name):
    """Return the first existing ModelScope cache path, or the default."""
    for sub in (
        MODELS_DIR / "modelscope" / org / name,
        MODELS_DIR / "modelscope" / "hub" / "models" / org / name,
    ):
        if sub.exists():
            return sub
    return MODELS_DIR / "modelscope" / org / name


def _hf_repo_complete(org: str, name: str, min_bytes: int = 50_000_000) -> bool:
    """True if a HuggingFace repo cache exists AND finished downloading.

    A killed/aborted download leaves snapshot entries pointing at missing blobs
    (broken symlinks) or '.incomplete' blobs; treating that as cached makes the
    model load hang. Validate a snapshot where every file resolves (stat follows
    symlinks; a broken link raises) and the resolved bytes are substantial. This
    ignores orphan '.incomplete' blobs left behind by an earlier interrupted run.
    """
    snap_root = MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
    if not snap_root.exists():
        return False
    for snap in snap_root.iterdir():
        if not snap.is_dir():
            continue
        total = 0
        broken = False
        for f in snap.rglob("*"):
            if f.is_dir():
                continue
            try:
                total += f.stat().st_size
            except OSError:
                broken = True
                break
        if not broken and total >= min_bytes:
            return True
    return False


def is_asr_cached(engine_type, model_size="medium", hub="ms") -> bool:
    if engine_type == "crispasr":
        return resolve_custom_crispasr_model(model_size) is not None
    if engine_type == "sherpa-onnx":
        return get_sherpa_onnx_model_path(model_size) is not None
    if engine_type == "parakeet-cpp":
        return get_parakeet_cpp_model_path(model_size) is not None
    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(model_size)
        )
        # Accept cache from either hub to avoid redundant downloads; the repo
        # namespace can differ between ModelScope and HuggingFace (SenseVoice).
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        if _ms_model_path(ms_org, ms_name).exists():
            return True
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
        if _hf_repo_complete(hf_org, hf_name):
            return True
        return False
    if engine_type == "anime-whisper":
        # HF-only (not published to ModelScope). Check that snapshots dir actually
        # contains weight files; an .incomplete blob means a prior run aborted mid-download.
        model_id = ASR_MODEL_IDS[engine_type]
        org, name = model_id.split("/")
        snap_root = (
            MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
        )
        if not snap_root.exists():
            return False
        for snap in snap_root.iterdir():
            if not snap.is_dir():
                continue
            has_weights = any(
                (snap / fn).exists()
                for fn in ("model.safetensors", "pytorch_model.bin")
            )
            has_config = (snap / "config.json").exists()
            if has_weights and has_config:
                return True
        return False
    elif engine_type == "whisper":
        if model_size not in _WHISPER_SIZES:
            return resolve_custom_whisper_model(model_size) is not None
        min_bytes = int(
            _MODEL_SIZE_BYTES.get(f"whisper-{model_size}", 50_000_000) * 0.5
        )
        return _hf_repo_complete(
            "Systran", f"faster-whisper-{model_size}", min_bytes=min_bytes
        )
    return True


def get_missing_models(engine, model_size, hub) -> list:
    missing = []
    if not is_silero_cached():
        missing.append(
            {
                "name": "Silero VAD",
                "type": "silero-vad",
                "estimated_bytes": _MODEL_SIZE_BYTES["silero-vad"],
            }
        )
    if not is_asr_cached(engine, model_size, hub):
        if engine == "whisper" and model_size not in _WHISPER_SIZES:
            return missing
        if engine in ("crispasr", "sherpa-onnx", "parakeet-cpp"):
            return missing
        if engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
            model_key = (
                FUNASR_LEGACY_ENGINE_ALIASES[engine]
                if engine in FUNASR_LEGACY_ENGINE_ALIASES
                else normalize_funasr_model_key(model_size)
            )
            profile = funasr_profile(model_key)
            key = f"funasr:{model_key}"
            display = profile["display_name"]
            estimated_bytes = profile["estimated_bytes"]
        elif engine == "whisper":
            key = engine if engine != "whisper" else f"whisper-{model_size}"
            display = f"Whisper {model_size}"
            estimated_bytes = _MODEL_SIZE_BYTES.get(key, 0)
        else:
            key = engine
            display = ASR_DISPLAY_NAMES.get(engine, engine)
            estimated_bytes = _MODEL_SIZE_BYTES.get(key, 0)
        missing.append(
            {
                "name": display,
                "type": key,
                "estimated_bytes": estimated_bytes,
            }
        )
    return missing


def get_local_model_path(
    engine_type,
    hub="ms",
    funasr_model: str | None = None,
    model_path_or_id: str | None = None,
):
    """Return local snapshot path if model is cached, else None.

    Checks the preferred hub first, then falls back to the other hub.
    """
    if engine_type == "crispasr":
        model_value = funasr_model
        return resolve_custom_crispasr_model(model_value)
    if engine_type == "sherpa-onnx":
        model_value = model_path_or_id if model_path_or_id is not None else funasr_model
        return get_sherpa_onnx_model_path(model_value)
    if engine_type == "parakeet-cpp":
        model_value = model_path_or_id if model_path_or_id is not None else funasr_model
        return get_parakeet_cpp_model_path(model_value)

    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(funasr_model)
        )
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
    elif engine_type in ASR_MODEL_IDS:
        ms_org, ms_name = asr_model_id(engine_type, "ms").split("/")
        hf_org, hf_name = asr_model_id(engine_type, "hf").split("/")
    else:
        return None

    def _try_ms():
        local = _ms_model_path(ms_org, ms_name)
        return str(local) if local.exists() else None

    def _try_hf():
        snap_dir = (
            MODELS_DIR
            / "huggingface"
            / "hub"
            / f"models--{hf_org}--{hf_name}"
            / "snapshots"
        )
        if snap_dir.exists():
            snaps = sorted(snap_dir.iterdir())
            if snaps:
                return str(snaps[-1])
        return None

    if hub == "ms":
        return _try_ms() or _try_hf()
    else:
        return _try_hf() or _try_ms()


def download_silero(proxy: str = "system"):
    if _has_silero_pkg():
        log.info("Silero VAD bundled by silero-vad package, no download needed")
        return
    import torch

    log.info("Downloading Silero VAD...")
    with _proxy_env(proxy):
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:master",
                model="silero_vad",
                trust_repo=True,
            )
        except Exception as exc:
            if "CERTIFICATE_VERIFY" not in str(exc):
                raise
            log.warning("SSL strict verification failed, retrying with relaxed flags")
            model, _ = _load_silero_relaxed_ssl()
    del model
    log.info("Silero VAD downloaded")


def _load_silero_relaxed_ssl():
    # Python 3.13 enables VERIFY_X509_STRICT by default, rejecting certificates
    # without an Authority Key Identifier (common behind SSL-inspecting proxies).
    import ssl

    import torch

    strict = getattr(ssl, "VERIFY_X509_STRICT", 0)
    original = ssl._create_default_https_context

    def relaxed_context(*args, **kwargs):
        ctx = ssl.create_default_context(*args, **kwargs)
        ctx.verify_flags &= ~strict
        return ctx

    ssl._create_default_https_context = relaxed_context
    try:
        return torch.hub.load(
            repo_or_dir="snakers4/silero-vad:master",
            model="silero_vad",
            trust_repo=True,
            force_reload=True,
        )
    finally:
        ssl._create_default_https_context = original


def download_asr(engine, model_size="medium", hub="ms", proxy="system"):
    resolved = str(MODELS_DIR.resolve())
    ms_cache = os.path.join(resolved, "modelscope")
    hf_cache = os.path.join(resolved, "huggingface", "hub")
    with _proxy_env(proxy):
        if engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
            model_key = (
                FUNASR_LEGACY_ENGINE_ALIASES[engine]
                if engine in FUNASR_LEGACY_ENGINE_ALIASES
                else normalize_funasr_model_key(model_size)
            )
            if hub == "ms":
                from modelscope import snapshot_download

                model_id = funasr_model_id(model_key, "ms")
                log.info(f"Downloading {model_id} from ModelScope...")
                snapshot_download(model_id=model_id, cache_dir=ms_cache)
            else:
                from huggingface_hub import snapshot_download

                model_id = funasr_model_id(model_key, "hf")
                log.info(f"Downloading {model_id} from HuggingFace...")
                snapshot_download(repo_id=model_id, cache_dir=hf_cache)
        elif engine == "anime-whisper":
            # HF-only, ignore hub setting
            from huggingface_hub import snapshot_download

            model_id = ASR_MODEL_IDS[engine]
            log.info(f"Downloading {model_id} from HuggingFace...")
            snapshot_download(repo_id=model_id, cache_dir=hf_cache)
        elif engine == "whisper":
            if model_size not in _WHISPER_SIZES:
                raise ValueError(f"Invalid local faster-whisper model: {model_size}")
            from huggingface_hub import snapshot_download

            model_id = f"Systran/faster-whisper-{model_size}"
            log.info(f"Downloading {model_id} from HuggingFace...")
            snapshot_download(repo_id=model_id, cache_dir=hf_cache)
        else:
            raise ValueError(f"Unsupported ASR download engine: {engine}")
    log.info(f"ASR model downloaded: {engine}")


def dir_size(path) -> int:
    path = Path(path)
    if path.is_file():
        try:
            return path.stat().st_size
        except (OSError, PermissionError):
            return 0
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def get_cache_entries():
    """Scan ./models/ for cached models."""
    entries = []
    hf_base = MODELS_DIR / "huggingface" / "hub"
    torch_base = MODELS_DIR / "torch" / "hub"

    for entry in _CACHE_MODELS:
        if len(entry) == 3:
            name, engine, model_key = entry
        else:
            name, engine = entry
            model_key = None
        ms_org, ms_model = asr_model_id(engine, "ms", model_key).split("/")
        hf_org, hf_model = asr_model_id(engine, "hf", model_key).split("/")
        ms_path = _ms_model_path(ms_org, ms_model)
        hf_path = hf_base / f"models--{hf_org}--{hf_model}"
        if ms_path.exists():
            entries.append((f"{name} (ModelScope)", ms_path))
        if hf_path.exists():
            entries.append((f"{name} (HuggingFace)", hf_path))

    for size in _WHISPER_SIZES:
        hf_path = hf_base / f"models--Systran--faster-whisper-{size}"
        if hf_path.exists() and is_asr_cached("whisper", size, "hf"):
            entries.append((f"Whisper {size}", hf_path))

    for item in list_local_faster_whisper_models():
        entries.append((f"Whisper Local: {item['name']}", Path(item["path"])))

    for item in list_local_crispasr_models():
        entries.append((f"CrispASR Local: {item['name']}", Path(item["path"])))

    for item in list_local_sherpa_onnx_models():
        entries.append((f"sherpa-onnx Local: {item['name']}", Path(item["path"])))

    for item in list_local_parakeet_cpp_models():
        entries.append((f"parakeet.cpp Local: {item['name']}", Path(item["path"])))

    for item in list_local_parakeet_cpp_runtimes():
        entries.append((f"parakeet.cpp Runtime: {item['name']}", Path(item["path"])))

    for item in list_local_firered_vad_models():
        entries.append((f"FireRedVAD Local: {item['name']}", Path(item["path"])))

    if torch_base.exists():
        for d in sorted(torch_base.glob("snakers4_silero-vad*")):
            if d.is_dir():
                entries.append(("Silero VAD", d))
                break

    return entries
