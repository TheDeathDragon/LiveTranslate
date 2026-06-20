import json
import logging
from pathlib import Path
from typing import Any

from model_manager import (
    DEFAULT_FIRERED_VAD_MODEL,
    DEFAULT_FUNASR_MODEL,
    DEFAULT_PARAKEET_CPP_MODEL,
    migrate_funasr_settings,
    normalize_funasr_model_key,
)

log = logging.getLogger("LiveTranslate.Settings")

SETTINGS_FILE = Path(__file__).parent / "user_settings.json"


def normalize_settings(config: dict[str, Any], saved: dict | None = None) -> dict:
    """Return a complete settings dict with migrations and defaults applied."""
    settings = dict(saved or {})
    migrate_funasr_settings(settings)

    asr = config.get("asr", {})
    translation = config.get("translation", {})

    settings.setdefault("vad_mode", "silero")
    settings.setdefault("vad_threshold", asr.get("vad_threshold", 0.5))
    settings.setdefault("energy_threshold", 0.02)
    settings.setdefault(
        "firered_vad_model",
        asr.get("firered_vad_model", DEFAULT_FIRERED_VAD_MODEL),
    )
    settings.setdefault("firered_vad_use_gpu", asr.get("firered_vad_use_gpu", False))
    settings.setdefault(
        "firered_vad_smooth_window_size",
        asr.get("firered_vad_smooth_window_size", 5),
    )
    settings.setdefault(
        "firered_vad_frame_aggregation",
        asr.get("firered_vad_frame_aggregation", "max"),
    )
    settings.setdefault("min_speech_duration", asr.get("min_speech_duration", 1.0))
    settings.setdefault("max_speech_duration", asr.get("max_speech_duration", 8.0))
    settings.setdefault("silence_mode", "auto")
    settings.setdefault("silence_duration", 0.8)
    settings.setdefault("incremental_asr", False)
    settings.setdefault("interim_interval", 2.0)

    settings.setdefault("asr_language", asr.get("language", "auto"))
    settings.setdefault("asr_engine", asr.get("asr_engine", "funasr"))
    settings.setdefault("funasr_model", asr.get("funasr_model", DEFAULT_FUNASR_MODEL))
    settings["funasr_model"] = normalize_funasr_model_key(
        settings.get("funasr_model")
    )
    settings.setdefault("whisper_model_size", asr.get("model_size", "medium"))
    settings.setdefault("crispasr_model", asr.get("crispasr_model", ""))
    settings.setdefault("crispasr_backend", asr.get("crispasr_backend", "auto"))
    settings.setdefault(
        "crispasr_gpu_backend", asr.get("crispasr_gpu_backend", "auto")
    )
    settings.setdefault("crispasr_device_index", asr.get("crispasr_device_index", 0))
    settings.setdefault("crispasr_punc_model", asr.get("crispasr_punc_model", "auto"))
    settings.setdefault(
        "crispasr_unified_memory", asr.get("crispasr_unified_memory", True)
    )
    settings.setdefault("sherpa_onnx_model", asr.get("sherpa_onnx_model", ""))
    settings.setdefault(
        "remote_asr_url", asr.get("remote_asr_url", "http://127.0.0.1:8765")
    )
    settings.setdefault(
        "sherpa_onnx_provider", asr.get("sherpa_onnx_provider", "auto")
    )
    settings.setdefault(
        "sherpa_onnx_num_threads", asr.get("sherpa_onnx_num_threads", 2)
    )
    settings.setdefault(
        "sherpa_onnx_decoding_method",
        asr.get("sherpa_onnx_decoding_method", "greedy_search"),
    )
    settings.setdefault(
        "sherpa_onnx_left_padding_seconds",
        asr.get("sherpa_onnx_left_padding_seconds", 0.3),
    )
    settings.setdefault(
        "sherpa_onnx_tail_padding_seconds",
        asr.get("sherpa_onnx_tail_padding_seconds", 0.5),
    )
    settings.setdefault(
        "parakeet_cpp_model",
        asr.get("parakeet_cpp_model", DEFAULT_PARAKEET_CPP_MODEL),
    )
    settings.setdefault(
        "parakeet_cpp_runtime_dir",
        asr.get("parakeet_cpp_runtime_dir", ""),
    )
    settings.setdefault("parakeet_cpp_backend", asr.get("parakeet_cpp_backend", "auto"))
    settings.setdefault("parakeet_cpp_decoder", asr.get("parakeet_cpp_decoder", "auto"))
    settings.setdefault(
        "parakeet_cpp_word_timestamps",
        asr.get("parakeet_cpp_word_timestamps", True),
    )
    settings.setdefault("asr_device", asr.get("device", "cuda"))
    settings.setdefault(
        "sensevoice_pad_seconds", asr.get("sensevoice_pad_seconds", 0.5)
    )
    settings.setdefault("whisper_pad_seconds", asr.get("whisper_pad_seconds", 0.5))
    settings.setdefault("audio_device", config.get("audio", {}).get("device"))
    settings.setdefault("mic_device", None)
    settings.setdefault("hub", "ms")

    if "models" not in settings:
        model = translation.get("model", "")
        settings["models"] = [
            {
                "name": model or "Local API",
                "api_base": translation.get("api_base", ""),
                "api_key": translation.get("api_key", ""),
                "model": model,
            }
        ]
    settings.setdefault("active_model", 0)
    models = settings.get("models") or []
    if not isinstance(models, list):
        models = []
        settings["models"] = models
    if models:
        active = settings.get("active_model", 0)
        if not isinstance(active, int) or active < 0 or active >= len(models):
            settings["active_model"] = 0

    settings.setdefault("target_language", translation.get("target_language", "zh"))
    settings.setdefault("source_language", translation.get("source_language", "auto"))
    settings.setdefault("context_window", translation.get("context_window", 0))
    settings.setdefault("system_prompt", translation.get("system_prompt", ""))
    settings.setdefault("timeout", 5)
    settings.setdefault("auto_save_transcript", True)

    return settings


def load_settings(config: dict | None = None) -> dict | None:
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            migrate_funasr_settings(data)
            if config is not None:
                data = normalize_settings(config, data)
            log.info(f"Loaded saved settings from {SETTINGS_FILE}")
            return data
    except Exception as e:
        log.warning(f"Failed to load settings: {e}")
    return None


def save_settings(settings: dict):
    try:
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(SETTINGS_FILE)
        log.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        log.warning(f"Failed to save settings: {e}")


def update_settings(patch: dict, config: dict | None = None) -> dict:
    settings = load_settings(config) or {}
    settings.update(patch)
    if config is not None:
        settings = normalize_settings(config, settings)
    save_settings(settings)
    return settings


class SettingsRepository:
    def __init__(self, config: dict):
        self._config = config

    def load(self) -> dict | None:
        return load_settings(self._config)

    def save(self, settings: dict):
        save_settings(settings)

    def update(self, patch: dict) -> dict:
        return update_settings(patch, self._config)
