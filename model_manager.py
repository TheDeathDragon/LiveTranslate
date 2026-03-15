import os
import logging
from pathlib import Path

log = logging.getLogger("LiveTrans.ModelManager")

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

ASR_MODEL_IDS = {
    "sensevoice": "iic/SenseVoiceSmall",
    "funasr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "funasr-mlt-nano": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
}

ASR_DISPLAY_NAMES = {
    "sensevoice": "SenseVoice Small",
    "funasr-nano": "Fun-ASR-Nano",
    "funasr-mlt-nano": "Fun-ASR-MLT-Nano",
    "whisper": "Whisper",
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
}

_WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

_CACHE_MODELS = [
    ("SenseVoice Small", "iic/SenseVoiceSmall"),
    ("Fun-ASR-Nano", "FunAudioLLM/Fun-ASR-Nano-2512"),
    ("Fun-ASR-MLT-Nano", "FunAudioLLM/Fun-ASR-MLT-Nano-2512"),
]


def apply_cache_env():
    """Point all model caches to ./models/."""
    resolved = str(MODELS_DIR.resolve())
    os.environ["MODELSCOPE_CACHE"] = os.path.join(resolved, "modelscope")
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(resolved, "torch")
    log.info(f"Cache env set: {resolved}")


def is_silero_cached() -> bool:
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


def is_asr_cached(engine_type, model_size="medium", hub="ms") -> bool:
    if engine_type in ("sensevoice", "funasr-nano", "funasr-mlt-nano"):
        model_id = ASR_MODEL_IDS[engine_type]
        org, name = model_id.split("/")
        # Accept cache from either hub to avoid redundant downloads
        if _ms_model_path(org, name).exists():
            return True
        if (MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}").exists():
            return True
        return False
    elif engine_type == "whisper":
        return (
            MODELS_DIR
            / "huggingface"
            / "hub"
            / f"models--Systran--faster-whisper-{model_size}"
        ).exists()
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
        key = engine if engine != "whisper" else f"whisper-{model_size}"
        display = ASR_DISPLAY_NAMES.get(engine, engine)
        if engine == "whisper":
            display = f"Whisper {model_size}"
        missing.append(
            {
                "name": display,
                "type": key,
                "estimated_bytes": _MODEL_SIZE_BYTES.get(key, 0),
            }
        )
    return missing


def get_local_model_path(engine_type, hub="ms"):
    """Return local snapshot path if model is cached, else None.

    Checks the preferred hub first, then falls back to the other hub.
    """
    if engine_type not in ASR_MODEL_IDS:
        return None
    model_id = ASR_MODEL_IDS[engine_type]
    org, name = model_id.split("/")

    def _try_ms():
        local = _ms_model_path(org, name)
        return str(local) if local.exists() else None

    def _try_hf():
        snap_dir = (
            MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
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


def download_silero():
    import torch

    log.info("Downloading Silero VAD...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    del model
    log.info("Silero VAD downloaded")


def download_asr(engine, model_size="medium", hub="ms"):
    resolved = str(MODELS_DIR.resolve())
    ms_cache = os.path.join(resolved, "modelscope")
    hf_cache = os.path.join(resolved, "huggingface", "hub")
    if engine in ("sensevoice", "funasr-nano", "funasr-mlt-nano"):
        model_id = ASR_MODEL_IDS[engine]
        if hub == "ms":
            from modelscope import snapshot_download

            log.info(f"Downloading {model_id} from ModelScope...")
            snapshot_download(model_id=model_id, cache_dir=ms_cache)
        else:
            from huggingface_hub import snapshot_download

            log.info(f"Downloading {model_id} from HuggingFace...")
            snapshot_download(repo_id=model_id, cache_dir=hf_cache)
    elif engine == "whisper":
        from huggingface_hub import snapshot_download

        model_id = f"Systran/faster-whisper-{model_size}"
        log.info(f"Downloading {model_id} from HuggingFace...")
        snapshot_download(repo_id=model_id, cache_dir=hf_cache)
    log.info(f"ASR model downloaded: {engine}")


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

    for name, model_id in _CACHE_MODELS:
        org, model = model_id.split("/")
        ms_path = _ms_model_path(org, model)
        hf_path = hf_base / f"models--{org}--{model}"
        if ms_path.exists():
            entries.append((f"{name} (ModelScope)", ms_path))
        if hf_path.exists():
            entries.append((f"{name} (HuggingFace)", hf_path))

    for size in _WHISPER_SIZES:
        hf_path = hf_base / f"models--Systran--faster-whisper-{size}"
        if hf_path.exists():
            entries.append((f"Whisper {size}", hf_path))

    if torch_base.exists():
        for d in sorted(torch_base.glob("snakers4_silero-vad*")):
            if d.is_dir():
                entries.append(("Silero VAD", d))
                break

    return entries
