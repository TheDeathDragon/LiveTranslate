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
        # includes the separately-fetched Qwen3-0.6B weights (~1.5GB)
        "estimated_bytes": 3_500_000_000,
        "supports_padding": False,
        "supports_language": True,
    },
    "funasr-mlt-nano-2512": {
        "display_name": "Fun-ASR-MLT-Nano",
        "family": "funasr-nano",
        "legacy_engine": "funasr-mlt-nano",
        "modelscope_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "huggingface_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        # includes the separately-fetched Qwen3-0.6B weights (~1.5GB)
        "estimated_bytes": 3_500_000_000,
        "supports_padding": False,
        "supports_language": True,
    },
}

DEFAULT_FUNASR_MODEL = "sensevoice-small"

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

# Engines outside the funasr profile table. `cache_layout` says what the
# download step leaves behind: raw hub weights ("hub") or a CTranslate2
# conversion under models/ct2 ("ct2"). The loader is picked in asr_worker
# from the engine key itself.
ASR_ENGINE_PROFILES = {
    "qwen3-asr": {
        "display_name": "Qwen3-ASR 1.7B",
        # The -hf repo stores `model.*` weights and a chat template that keeps
        # assistant messages. The plain repo targets the qwen-asr package: its
        # `thinker.*` weights do not match transformers' Qwen3ASRModel.
        "modelscope_id": "Qwen/Qwen3-ASR-1.7B-hf",
        "huggingface_id": "Qwen/Qwen3-ASR-1.7B-hf",
        "estimated_bytes": 4_100_000_000,
        "cache_layout": "hub",
        "language_locked": None,
        # UI offers these but the model has no code for them, so it falls back
        # to auto-detect instead of failing.
        "unsupported_languages": ("uk", "no", "he"),
        "supports_padding": False,
    },
    "buzzasr-korean": {
        "display_name": "BuzzASR Korean",
        "modelscope_id": None,
        "huggingface_id": "BuzzASR/korean",
        "estimated_bytes": 3_100_000_000,
        "cache_layout": "ct2",
        "language_locked": "ko",
        "unsupported_languages": (),
        "supports_padding": True,
    },
}

# Converted CTranslate2 models live beside the hub caches, not inside them.
CT2_MODELS_DIR = MODELS_DIR / "ct2"


def ct2_model_dir(engine_type: str):
    """Directory holding the CTranslate2 conversion of `engine_type`."""
    return CT2_MODELS_DIR / engine_type


def asr_model_id(
    engine_type: str, hub: str = "ms", funasr_model: str | None = None
) -> str:
    """Return the repo id for an engine on the given hub ('ms' or 'hf')."""
    profile = ASR_ENGINE_PROFILES.get(engine_type)
    if profile:
        # HF-only engines fall back to the HF id for both hubs so callers can
        # always split the result into org/name.
        if hub == "hf":
            return profile["huggingface_id"]
        return profile["modelscope_id"] or profile["huggingface_id"]
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
    "remote-whisper": "Remote-Whisper",
}

_MODEL_SIZE_BYTES = {
    "silero-vad": 2_000_000,
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
    ("Qwen3-ASR 1.7B", "qwen3-asr"),
    ("BuzzASR Korean", "buzzasr-korean"),
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


def ct2_model_complete(path) -> bool:
    """True when a produced CTranslate2 directory has every file faster-whisper
    reads at load time. A missing tokenizer or preprocessor config sends it to
    the hub and falls back to 80 mels, which breaks large-v3 derived models.
    """
    if not is_faster_whisper_model_dir(path):
        return False
    path = Path(path)
    return all(
        (path / name).is_file()
        for name in ("tokenizer.json", "preprocessor_config.json")
    )


def resolve_custom_whisper_model(value) -> str | None:
    path = _custom_whisper_path(value)
    if path and is_faster_whisper_model_dir(path):
        return str(path.resolve())
    return None


def _is_builtin_whisper_cache(path: Path) -> bool:
    parts = set(path.parts)
    return any(f"models--Systran--faster-whisper-{s}" in parts for s in _WHISPER_SIZES)


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
    """Return the first existing ModelScope cache path, or the default.

    Layouts by SDK version: {org}/{name} = <=1.37 with explicit cache_dir;
    models/{org}/{name} = 1.34~1.37 env-default cache, which >=1.38 keeps
    reusing as legacy even when cache_dir is passed (dots in names written
    as ___ by old SDKs); hub trees = older SDKs; models/{org}--{name}/
    snapshots/{revision} = >=1.38 fresh cache.
    """
    ms_root = MODELS_DIR / "modelscope"
    for sub in (
        ms_root / org / name,
        ms_root / "models" / org / name,
        ms_root / "models" / org / name.replace(".", "___"),
        ms_root / "hub" / "models" / org / name,
        ms_root / "hub" / org / name,
    ):
        if sub.exists():
            return sub
    snap_root = ms_root / "models" / f"{org}--{name}" / "snapshots"
    if snap_root.is_dir():
        snaps = sorted(d for d in snap_root.iterdir() if d.is_dir())
        if snaps:
            return snaps[-1]
    return ms_root / org / name


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


def _weights_complete(model_dir: Path) -> bool:
    """True when every weight shard of a Transformers checkpoint exists.

    A sharded checkpoint lists its files in model.safetensors.index.json;
    checking only "some .safetensors exists" accepts a download that aborted
    after the first shard. A damaged index counts as incomplete, so the
    download step repairs it instead of the loader failing.
    """
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (OSError, ValueError, KeyError):
            return False
        return bool(weight_map) and all(
            (model_dir / fn).is_file() for fn in set(weight_map.values())
        )
    return any(
        (model_dir / fn).is_file()
        for fn in ("model.safetensors", "pytorch_model.bin")
    )


def _complete_model_dir(path: Path) -> bool:
    """True when a model directory holds a config and every weight shard."""
    return (path / "config.json").is_file() and _weights_complete(path)


def _snapshot_complete(model_id: str) -> bool:
    """True when any snapshot of `model_id` holds a complete model."""
    org, name = model_id.split("/")
    snap_root = (
        MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
    )
    if not snap_root.exists():
        return False
    return any(
        snap.is_dir() and _complete_model_dir(snap) for snap in snap_root.iterdir()
    )


def is_asr_cached(engine_type, model_size="medium", hub="ms") -> bool:
    profile = ASR_ENGINE_PROFILES.get(engine_type)
    if profile:
        if profile["cache_layout"] == "ct2":
            # The converted dir is the only thing ASREngine can load; a bare
            # hub snapshot means the conversion step never finished.
            return ct2_model_complete(ct2_model_dir(engine_type))
        # Accept cache from either hub, same rule as the funasr branch below.
        ms_id = profile["modelscope_id"]
        if ms_id:
            org, name = ms_id.split("/")
            if _complete_model_dir(_ms_model_path(org, name)):
                return True
        return _snapshot_complete(profile["huggingface_id"])
    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(model_size)
        )
        # Accept cache from either hub to avoid redundant downloads; the repo
        # namespace can differ between ModelScope and HuggingFace (SenseVoice).
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
        if not (
            _ms_model_path(ms_org, ms_name).exists()
            or _hf_repo_complete(hf_org, hf_name)
        ):
            return False
        # Nano's Qwen3-0.6B weights download separately; require them so the
        # download flow (not the deadline-bound worker) pulls them up-front.
        if funasr_profile(model_key)["family"] == "funasr-nano":
            model_dir = get_local_model_path(engine_type, hub, funasr_model=model_size)
            if not model_dir or not qwen_weights_present(model_dir):
                return False
        return True
    if engine_type == "anime-whisper":
        # HF-only (not published to ModelScope).
        return _snapshot_complete(ASR_MODEL_IDS[engine_type])
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
        if engine in ASR_ENGINE_PROFILES:
            profile = ASR_ENGINE_PROFILES[engine]
            key = engine
            display = profile["display_name"]
            estimated_bytes = profile["estimated_bytes"]
        elif engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
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


def get_local_model_path(engine_type, hub="ms", funasr_model: str | None = None):
    """Return local snapshot path if model is cached, else None.

    Checks the preferred hub first, then falls back to the other hub.
    """
    profile = ASR_ENGINE_PROFILES.get(engine_type)
    if profile and profile["cache_layout"] == "ct2":
        ct2 = ct2_model_dir(engine_type)
        return str(ct2) if ct2_model_complete(ct2) else None
    if profile:
        ms_org, ms_name = asr_model_id(engine_type, "ms").split("/")
        hf_org, hf_name = asr_model_id(engine_type, "hf").split("/")
    elif engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
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

    # Profile engines need every weight shard before they can load; the legacy
    # branches only require the directory to exist.
    check = _complete_model_dir if profile else Path.exists

    def _try_ms():
        local = _ms_model_path(ms_org, ms_name)
        return str(local) if check(local) else None

    def _try_hf():
        snap_dir = (
            MODELS_DIR
            / "huggingface"
            / "hub"
            / f"models--{hf_org}--{hf_name}"
            / "snapshots"
        )
        if snap_dir.exists():
            for snap in sorted(snap_dir.iterdir(), reverse=True):
                if check(snap):
                    return str(snap)
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


def qwen_weights_present(model_dir) -> bool:
    """Whether a nano model's embedded Qwen3-0.6B weights are in place.

    Nano repos ship the Qwen3-0.6B config but not its weights. A variant without
    the subdir needs no Qwen weights, so absence of the subdir counts as present.
    """
    qwen_dir = Path(model_dir) / "Qwen3-0.6B"
    if not qwen_dir.is_dir():
        return True
    return any(f.suffix in (".safetensors", ".bin") for f in qwen_dir.iterdir())


def ensure_qwen_weights(model_dir, hub: str = "ms") -> None:
    """Fetch Qwen3-0.6B weights into a nano model's embedded subdir (one-time).

    Kept off the ASR worker startup path: its 180s ready timeout would otherwise
    kill the process mid-download on slow links.
    """
    qwen_dir = Path(model_dir) / "Qwen3-0.6B"
    if not qwen_dir.is_dir():
        return
    if any(f.suffix in (".safetensors", ".bin") for f in qwen_dir.iterdir()):
        return
    log.info("Downloading Qwen3-0.6B weights (one-time)...")
    if hub == "hf":
        from huggingface_hub import snapshot_download
    else:
        from modelscope import snapshot_download

    snapshot_download(
        "Qwen/Qwen3-0.6B",
        local_dir=str(qwen_dir),
        ignore_patterns=["*.gguf"],
    )
    log.info("Qwen3-0.6B weights downloaded")


def download_asr(engine, model_size="medium", hub="ms", proxy="system"):
    resolved = str(MODELS_DIR.resolve())
    ms_cache = os.path.join(resolved, "modelscope")
    hf_cache = os.path.join(resolved, "huggingface", "hub")
    with _proxy_env(proxy):
        if engine in ASR_ENGINE_PROFILES:
            profile = ASR_ENGINE_PROFILES[engine]
            if profile["cache_layout"] == "ct2":
                from ctranslate2.converters import TransformersConverter
                from huggingface_hub import snapshot_download

                model_id = profile["huggingface_id"]
                log.info(f"Downloading {model_id} from HuggingFace...")
                snapshot_path = snapshot_download(repo_id=model_id, cache_dir=hf_cache)
                target = ct2_model_dir(engine)
                target.parent.mkdir(parents=True, exist_ok=True)
                log.info(f"Converting {model_id} to CTranslate2 float16...")
                # Without these two, faster-whisper fetches the tokenizer from
                # the hub at load time and falls back to an 80-mel extractor,
                # which does not match a large-v3 derived model (128 mels).
                TransformersConverter(
                    snapshot_path,
                    copy_files=["tokenizer.json", "preprocessor_config.json"],
                ).convert(str(target), quantization="float16", force=True)
                log.info(f"CTranslate2 model ready: {target}")
            elif hub == "ms" and profile["modelscope_id"]:
                from modelscope import snapshot_download

                log.info(f"Downloading {profile['modelscope_id']} from ModelScope...")
                snapshot_download(
                    model_id=profile["modelscope_id"], cache_dir=ms_cache
                )
            else:
                from huggingface_hub import snapshot_download

                log.info(f"Downloading {profile['huggingface_id']} from HuggingFace...")
                snapshot_download(
                    repo_id=profile["huggingface_id"], cache_dir=hf_cache
                )
        elif engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
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
            funasr_dir = get_local_model_path("funasr", hub=hub, funasr_model=model_key)
            neutralize_funasr_requirements(funasr_dir)
            if funasr_dir and funasr_profile(model_key)["family"] == "funasr-nano":
                ensure_qwen_weights(funasr_dir, hub=hub)
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
    log.info(f"ASR model downloaded: {engine}")


def neutralize_funasr_requirements(model_dir) -> None:
    """Skip FunASR's load-time `pip install -r requirements.txt`.

    With trust_remote_code=True, FunASR detects requirements.txt in the model
    dir and runs pip in a subprocess whose output is swallowed (PIPE). On a slow
    or proxy-blocked PyPI this hangs indefinitely with no log output, and it can
    pull heavy unused deps (e.g. gradio). All real deps already live in the venv,
    so rename the file out of the way to make the check miss.
    """
    if not model_dir:
        return
    req = Path(model_dir) / "requirements.txt"
    if req.exists():
        try:
            req.replace(req.with_name("requirements.txt.bundled"))
            log.info(f"Skipped FunASR requirements install: {req}")
        except OSError as exc:
            log.warning(f"Failed to neutralize {req}: {exc}")


def dir_size(path) -> int:
    total = 0
    try:
        for f in Path(path).rglob("*"):
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
        if engine in ASR_ENGINE_PROFILES:
            ct2 = ct2_model_dir(engine)
            if ct2_model_complete(ct2):
                entries.append((f"{name} (CTranslate2)", ct2))

    for size in _WHISPER_SIZES:
        hf_path = hf_base / f"models--Systran--faster-whisper-{size}"
        if hf_path.exists() and is_asr_cached("whisper", size, "hf"):
            entries.append((f"Whisper {size}", hf_path))

    for item in list_local_faster_whisper_models():
        entries.append((f"Whisper Local: {item['name']}", Path(item["path"])))

    if torch_base.exists():
        for d in sorted(torch_base.glob("snakers4_silero-vad*")):
            if d.is_dir():
                entries.append(("Silero VAD", d))
                break

    return entries
