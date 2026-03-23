import os
import logging
from pathlib import Path

log = logging.getLogger("LiveTranslate.ModelManager")

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

ASR_MODEL_IDS = {
    "sensevoice": "iic/SenseVoiceSmall",
    "funasr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "funasr-mlt-nano": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    "qwen3-asr": "Qwen3-ASR-1.7B",
}

ASR_DISPLAY_NAMES = {
    "sensevoice": "SenseVoice Small",
    "funasr-nano": "Fun-ASR-Nano",
    "funasr-mlt-nano": "Fun-ASR-MLT-Nano",
    "whisper": "Whisper",
    "qwen3-asr": "Qwen3-ASR",
}

# Expected model files for Qwen3-ASR
QWEN3_ASR_FILES = [
    "qwen3_asr_encoder_frontend.int4.onnx",
    "qwen3_asr_encoder_backend.int4.onnx",
    "qwen3_asr_llm.q4_k.gguf",
]

QWEN3_ASR_DIR_NAME = "qwen3-asr"

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
    "qwen3-asr": 770_000_000,
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


def is_qwen3_asr_ready() -> bool:
    """Check if Qwen3-ASR model files and llama.cpp DLLs are present."""
    model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    if not model_dir.exists():
        return False
    for fn in QWEN3_ASR_FILES:
        if not (model_dir / fn).exists():
            return False
    # Check llama.cpp DLLs
    bin_dir = APP_DIR / "qwen_asr_gguf" / "inference" / "bin"
    if not bin_dir.exists():
        return False
    import sys

    if sys.platform == "win32":
        required_dlls = ["llama.dll", "ggml.dll", "ggml-base.dll"]
    else:
        required_dlls = ["libllama.so", "libggml.so", "libggml-base.so"]
    for dll in required_dlls:
        if not (bin_dir / dll).exists():
            return False
    return True


def get_qwen3_asr_model_dir() -> str:
    """Return the Qwen3-ASR model directory path."""
    return str((MODELS_DIR / QWEN3_ASR_DIR_NAME).resolve())


def is_asr_cached(engine_type, model_size="medium", hub="ms") -> bool:
    if engine_type == "qwen3-asr":
        return is_qwen3_asr_ready()
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
    if engine_type == "qwen3-asr":
        d = MODELS_DIR / QWEN3_ASR_DIR_NAME
        return str(d) if d.exists() else None
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


QWEN3_ASR_MODEL_URL = "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-1.7B-gguf.zip"
LLAMA_CPP_DLL_URL_TEMPLATE = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-win-vulkan-x64.zip"
LLAMA_CPP_LATEST_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress logging."""
    import urllib.request

    log.info(f"Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as e:
        log.error(f"Download failed: {e}")
        raise
    log.info(f"Downloaded: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def download_qwen3_asr():
    """Download and extract Qwen3-ASR model + llama.cpp DLLs."""
    import zipfile
    import json
    import urllib.request

    model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = APP_DIR / "qwen_asr_gguf" / "inference" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and extract model files if missing
    model_missing = any(not (model_dir / fn).exists() for fn in QWEN3_ASR_FILES)
    if model_missing:
        zip_path = MODELS_DIR / "qwen3-asr-1.7b-gguf.zip"
        _download_file(QWEN3_ASR_MODEL_URL, zip_path, "Qwen3-ASR-1.7B model")
        log.info("Extracting model files...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename and basename in QWEN3_ASR_FILES:
                    with (
                        zf.open(member) as src,
                        open(str(model_dir / basename), "wb") as dst,
                    ):
                        import shutil

                        shutil.copyfileobj(src, dst)
        zip_path.unlink(missing_ok=True)
        log.info(f"Model extracted to {model_dir}")

    # 2. Download and extract llama.cpp DLLs if missing
    import sys

    if sys.platform == "win32":
        required_dlls = ["llama.dll", "ggml.dll", "ggml-base.dll"]
    else:
        required_dlls = ["libllama.so", "libggml.so", "libggml-base.so"]

    dlls_missing = any(not (bin_dir / dll).exists() for dll in required_dlls)
    if dlls_missing:
        # Get latest release tag
        log.info("Fetching latest llama.cpp release tag...")
        try:
            req = urllib.request.Request(
                LLAMA_CPP_LATEST_API, headers={"User-Agent": "LiveTranslate"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                tag = json.loads(resp.read())["tag_name"]
        except Exception:
            tag = "b8391"
            log.warning(f"Failed to get latest tag, using fallback: {tag}")

        dll_url = LLAMA_CPP_DLL_URL_TEMPLATE.format(tag=tag)
        zip_path = MODELS_DIR / "llama-cpp-vulkan.zip"
        _download_file(dll_url, zip_path, f"llama.cpp {tag} (Vulkan)")

        log.info("Extracting llama.cpp DLLs...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename in required_dlls:
                    with (
                        zf.open(member) as src,
                        open(str(bin_dir / basename), "wb") as dst,
                    ):
                        import shutil

                        shutil.copyfileobj(src, dst)
        # Also extract ggml-vulkan.dll and other ggml backend DLLs
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.startswith("ggml-") and basename.endswith(".dll"):
                    with (
                        zf.open(member) as src,
                        open(str(bin_dir / basename), "wb") as dst,
                    ):
                        import shutil

                        shutil.copyfileobj(src, dst)
        zip_path.unlink(missing_ok=True)
        log.info(f"DLLs extracted to {bin_dir}")

    log.info("Qwen3-ASR setup complete")


def download_asr(engine, model_size="medium", hub="ms"):
    resolved = str(MODELS_DIR.resolve())
    ms_cache = os.path.join(resolved, "modelscope")
    hf_cache = os.path.join(resolved, "huggingface", "hub")
    if engine == "qwen3-asr":
        download_qwen3_asr()
    elif engine in ("sensevoice", "funasr-nano", "funasr-mlt-nano"):
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

    # Qwen3-ASR model files
    qwen3_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    if qwen3_dir.exists() and any(qwen3_dir.iterdir()):
        entries.append(("Qwen3-ASR 1.7B (GGUF)", qwen3_dir))

    # llama.cpp DLLs
    bin_dir = APP_DIR / "qwen_asr_gguf" / "inference" / "bin"
    if bin_dir.exists() and any(
        f for f in bin_dir.iterdir() if f.suffix in (".dll", ".so")
    ):
        entries.append(("llama.cpp (Vulkan)", bin_dir))

    return entries
