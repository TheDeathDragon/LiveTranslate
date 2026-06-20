import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from asr_client import ASRClient, ASRWorkerError, ASRWorkerExited, ASRWorkerTimeout
from model_manager import (
    ASR_DISPLAY_NAMES,
    DEFAULT_FUNASR_MODEL,
    MODELS_DIR,
    detect_parakeet_cpp_runtime_dir,
    detect_sherpa_onnx_model_dir,
    get_parakeet_cpp_model_path,
    funasr_display_name,
    funasr_supports_padding,
    get_missing_models,
    get_sherpa_onnx_model_path,
    is_asr_cached,
    local_crispasr_display_name,
    local_faster_whisper_display_name,
    local_parakeet_cpp_display_name,
    local_sherpa_onnx_display_name,
    normalize_asr_engine_selection,
    normalize_funasr_model_key,
    resolve_parakeet_cpp_runtime_dir,
    resolve_custom_crispasr_model,
    resolve_custom_whisper_model,
)

log = logging.getLogger("LiveTranslate.ASRService")


@dataclass
class ASRStatus:
    ready: bool
    engine_type: str | None
    device: str | None
    display_name: str | None
    worker_status: str | None


@dataclass
class ASRSwitchPlan:
    engine_type: str
    device: str
    hub: str
    download_proxy: str
    display_name: str
    cache_model_key: str
    worker_config: dict
    target_state: dict
    missing_models: list[dict]
    already_current: bool = False
    error: str | None = None


@dataclass
class ASRSwitchResult:
    status: str
    target_state: dict | None = None
    restored_state: dict | None = None
    load_error: str | None = None
    restore_error: str | None = None


class ASRService:
    """Owns ASR worker lifecycle and exposes a small runtime API."""

    def __init__(
        self,
        config: dict,
        release_memory_caches: Callable[[], None] | None = None,
        unavailable_callback: Callable[[str], None] | None = None,
    ):
        self._config = config
        self._release_memory_caches = release_memory_caches
        self._unavailable_callback = unavailable_callback

        self._asr_ready = False
        self._asr_type = None
        self._asr: ASRClient | None = None
        self._asr_signature = None
        self._asr_config = None
        self._asr_error_count = 0
        self._asr_device = config["asr"]["device"]
        self._whisper_model_size = config["asr"]["model_size"]
        self._funasr_model_key = normalize_funasr_model_key(
            config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL)
        )
        self._crispasr_model_key = str(config["asr"].get("crispasr_model", "") or "")
        self._sherpa_onnx_model_path = str(
            config["asr"].get("sherpa_onnx_model", "") or ""
        )
        self._parakeet_cpp_model_key = str(
            config["asr"].get("parakeet_cpp_model", "") or ""
        )
        self._parakeet_cpp_runtime_dir = str(
            config["asr"].get("parakeet_cpp_runtime_dir", "") or ""
        )
        self._asr_lock = threading.RLock()

    @property
    def is_ready(self) -> bool:
        with self._asr_lock:
            return (
                self._asr_ready
                and self._asr is not None
                and self._asr.status == "ready"
            )

    @property
    def current_engine_type(self) -> str | None:
        with self._asr_lock:
            return self._asr_type

    @property
    def worker_pid(self) -> int | None:
        with self._asr_lock:
            return getattr(self._asr, "pid", None)

    def status_snapshot(self) -> ASRStatus:
        with self._asr_lock:
            display_name = (self._asr_config or {}).get("display_name")
            worker_status = self._asr.status if self._asr is not None else None
            return ASRStatus(
                ready=self.is_ready,
                engine_type=self._asr_type,
                device=self._asr_device,
                display_name=display_name,
                worker_status=worker_status,
            )

    def start_or_switch(self, engine_type: str, settings: dict) -> ASRSwitchPlan:
        """Build a switch plan. UI code may handle missing downloads before loading."""
        return self.prepare_switch(engine_type, settings)

    def prepare_switch(self, engine_type: str, settings: dict) -> ASRSwitchPlan:
        engine_type, funasr_model = normalize_asr_engine_selection(
            engine_type, settings.get("funasr_model", self._funasr_model_key)
        )
        device = settings.get("asr_device", self._asr_device)
        hub = settings.get("hub", "ms")
        download_proxy = settings.get("download_proxy", "system")

        model_size = settings.get(
            "whisper_model_size", self._config["asr"]["model_size"]
        )
        model_path = None
        cache_model_key = model_size
        crispasr_model = settings.get(
            "crispasr_model",
            self._config["asr"].get("crispasr_model", self._crispasr_model_key),
        )
        crispasr_model = str(crispasr_model or "")
        crispasr_model_path_value = None
        crispasr_backend = settings.get(
            "crispasr_backend", self._config["asr"].get("crispasr_backend", "auto")
        )
        crispasr_gpu_backend = settings.get(
            "crispasr_gpu_backend",
            self._config["asr"].get("crispasr_gpu_backend", "auto"),
        )
        crispasr_device_index = int(
            settings.get(
                "crispasr_device_index",
                self._config["asr"].get("crispasr_device_index", 0),
            )
            or 0
        )
        crispasr_punc_model = settings.get(
            "crispasr_punc_model",
            self._config["asr"].get("crispasr_punc_model", "auto"),
        )
        crispasr_unified_memory = bool(
            settings.get(
                "crispasr_unified_memory",
                self._config["asr"].get("crispasr_unified_memory", True),
            )
        )
        sherpa_onnx_model = str(
            settings.get(
                "sherpa_onnx_model",
                self._config["asr"].get(
                    "sherpa_onnx_model", self._sherpa_onnx_model_path
                ),
            )
            or ""
        )
        sherpa_onnx_provider = str(
            settings.get(
                "sherpa_onnx_provider",
                self._config["asr"].get("sherpa_onnx_provider", "auto"),
            )
            or "auto"
        ).lower()
        sherpa_onnx_num_threads = int(
            settings.get(
                "sherpa_onnx_num_threads",
                self._config["asr"].get("sherpa_onnx_num_threads", 2),
            )
            or 2
        )
        sherpa_onnx_decoding_method = str(
            settings.get(
                "sherpa_onnx_decoding_method",
                self._config["asr"].get(
                    "sherpa_onnx_decoding_method", "greedy_search"
                ),
            )
            or "greedy_search"
        )
        sherpa_onnx_tail_padding_seconds = float(
            settings.get(
                "sherpa_onnx_tail_padding_seconds",
                self._config["asr"].get("sherpa_onnx_tail_padding_seconds", 0.5),
            )
            or 0.0
        )
        sherpa_onnx_left_padding_seconds = float(
            settings.get(
                "sherpa_onnx_left_padding_seconds",
                self._config["asr"].get("sherpa_onnx_left_padding_seconds", 0.3),
            )
            or 0.0
        )
        sherpa_onnx_model_path = None
        sherpa_onnx_model_info = None
        parakeet_cpp_model = str(
            settings.get(
                "parakeet_cpp_model",
                self._config["asr"].get(
                    "parakeet_cpp_model", self._parakeet_cpp_model_key
                ),
            )
            or ""
        )
        parakeet_cpp_runtime_dir_value = str(
            settings.get(
                "parakeet_cpp_runtime_dir",
                self._config["asr"].get(
                    "parakeet_cpp_runtime_dir", self._parakeet_cpp_runtime_dir
                ),
            )
            or ""
        )
        parakeet_cpp_backend = str(
            settings.get(
                "parakeet_cpp_backend",
                self._config["asr"].get("parakeet_cpp_backend", "auto"),
            )
            or "auto"
        ).lower()
        parakeet_cpp_decoder = str(
            settings.get(
                "parakeet_cpp_decoder",
                self._config["asr"].get("parakeet_cpp_decoder", "auto"),
            )
            or "auto"
        ).lower()
        parakeet_cpp_word_timestamps = bool(
            settings.get(
                "parakeet_cpp_word_timestamps",
                self._config["asr"].get("parakeet_cpp_word_timestamps", True),
            )
        )
        parakeet_cpp_model_path = None
        parakeet_cpp_runtime_dir = None
        remote_asr_url = str(
            settings.get(
                "remote_asr_url",
                self._config["asr"].get("remote_asr_url", "http://127.0.0.1:8765"),
            )
            or "http://127.0.0.1:8765"
        ).strip()
        if not remote_asr_url:
            remote_asr_url = "http://127.0.0.1:8765"

        if engine_type == "whisper":
            model_path = resolve_custom_whisper_model(model_size)
            if model_path:
                cache_model_key = model_path
        elif engine_type == "funasr":
            cache_model_key = funasr_model
        elif engine_type == "crispasr":
            if not crispasr_model:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    "CrispASR model is not selected; keeping current ASR worker",
                )
            custom_crispasr_path = resolve_custom_crispasr_model(crispasr_model)
            if custom_crispasr_path:
                cache_model_key = custom_crispasr_path
                crispasr_model_path_value = custom_crispasr_path
            else:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    f"CrispASR model must be a local .gguf/.bin file: {crispasr_model}; "
                    "keeping current ASR worker",
                )
        elif engine_type == "sherpa-onnx":
            if not sherpa_onnx_model:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    "sherpa-onnx model is not selected; keeping current ASR worker",
                )
            sherpa_onnx_model_path = get_sherpa_onnx_model_path(sherpa_onnx_model)
            if not sherpa_onnx_model_path:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    f"sherpa-onnx model directory is unavailable or unrecognized: "
                    f"{sherpa_onnx_model}; keeping current ASR worker",
                )
            sherpa_onnx_model_info = detect_sherpa_onnx_model_dir(
                sherpa_onnx_model_path
            )
            if not sherpa_onnx_model_info:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    f"sherpa-onnx model directory is unavailable or unrecognized: "
                    f"{sherpa_onnx_model_path}; keeping current ASR worker",
                )
            cache_model_key = sherpa_onnx_model_path
        elif engine_type == "parakeet-cpp":
            if not parakeet_cpp_model:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    "parakeet.cpp model is not selected; keeping current ASR worker",
                )
            parakeet_cpp_model_path = get_parakeet_cpp_model_path(parakeet_cpp_model)
            if not parakeet_cpp_model_path:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    f"parakeet.cpp model file is unavailable or unrecognized: "
                    f"{parakeet_cpp_model}; keeping current ASR worker",
                )
            if parakeet_cpp_backend not in ("auto", "cpu", "cuda", "vulkan"):
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    f"Unsupported parakeet.cpp backend: {parakeet_cpp_backend}",
                )
            if parakeet_cpp_decoder not in ("auto", "ctc", "tdt", "rnnt"):
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    f"Unsupported parakeet.cpp decoder: {parakeet_cpp_decoder}",
                )
            parakeet_cpp_runtime_dir = resolve_parakeet_cpp_runtime_dir(
                parakeet_cpp_runtime_dir_value,
                "auto" if parakeet_cpp_backend == "auto" else parakeet_cpp_backend,
            )
            if not parakeet_cpp_runtime_dir:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    f"parakeet.cpp runtime dir is unavailable or unrecognized: "
                    f"{parakeet_cpp_runtime_dir_value}; keeping current ASR worker",
                )
            runtime_info = detect_parakeet_cpp_runtime_dir(parakeet_cpp_runtime_dir)
            if not runtime_info:
                return self._invalid_plan(
                    engine_type,
                    device,
                    hub,
                    download_proxy,
                    f"parakeet.cpp runtime dir is unavailable or unrecognized: "
                    f"{parakeet_cpp_runtime_dir}; keeping current ASR worker",
                )
            cache_model_key = parakeet_cpp_model_path
        elif engine_type == "remote-whisper":
            cache_model_key = remote_asr_url

        compute = self._config["asr"]["compute_type"]
        if engine_type == "whisper":
            signature_model = cache_model_key
        elif engine_type == "funasr":
            signature_model = funasr_model
        elif engine_type == "crispasr":
            signature_model = (
                cache_model_key,
                crispasr_backend,
                crispasr_gpu_backend,
                crispasr_device_index,
                crispasr_punc_model,
                crispasr_unified_memory,
            )
        elif engine_type == "sherpa-onnx":
            signature_model = (
                sherpa_onnx_model_path,
                sherpa_onnx_model_info["family"],
                sherpa_onnx_provider,
                sherpa_onnx_num_threads,
                sherpa_onnx_decoding_method,
                sherpa_onnx_left_padding_seconds,
                sherpa_onnx_tail_padding_seconds,
            )
        elif engine_type == "parakeet-cpp":
            signature_model = (
                parakeet_cpp_model_path,
                parakeet_cpp_runtime_dir,
                parakeet_cpp_backend,
                parakeet_cpp_decoder,
                parakeet_cpp_word_timestamps,
            )
        elif engine_type == "remote-whisper":
            signature_model = remote_asr_url
        else:
            signature_model = engine_type
        language = settings.get(
            "asr_language", self._config["asr"].get("language", "auto")
        )
        signature = (engine_type, signature_model, device, hub, compute, language)

        with self._asr_lock:
            current_asr = self._asr
            current_ready = (
                self._asr_ready
                and current_asr is not None
                and current_asr.status == "ready"
            )
            if current_ready and self._asr_signature == signature:
                return ASRSwitchPlan(
                    engine_type=engine_type,
                    device=device,
                    hub=hub,
                    download_proxy=download_proxy,
                    display_name=(self._asr_config or {}).get("display_name")
                    or engine_type,
                    cache_model_key=str(cache_model_key),
                    worker_config={},
                    target_state={},
                    missing_models=[],
                    already_current=True,
                )
            if not current_ready:
                self._asr_ready = False
            current_type = self._asr_type

        log.info(f"Switching ASR worker: {current_type} -> {engine_type}")

        cached = is_asr_cached(engine_type, cache_model_key, hub)
        display_name = self._display_name(
            engine_type,
            model_size,
            model_path,
            funasr_model,
            crispasr_model,
            sherpa_onnx_model_path,
            parakeet_cpp_model_path,
        )
        worker_config = {
            "engine_type": engine_type,
            "funasr_model": funasr_model,
            "model_size": cache_model_key,
            "device": device,
            "compute_type": compute,
            "hub": hub,
            "language": language,
            "pad_seconds": (
                settings.get(
                    "sensevoice_pad_seconds",
                    self._config["asr"].get("sensevoice_pad_seconds", 0.5),
                )
                if engine_type == "funasr"
                else settings.get(
                    "whisper_pad_seconds",
                    self._config["asr"].get("whisper_pad_seconds", 0.5),
                )
                if engine_type == "whisper"
                else None
            ),
            "download_root": str((MODELS_DIR / "huggingface" / "hub").resolve()),
            "display_name": display_name,
            "remote_asr_url": remote_asr_url,
        }
        if engine_type == "crispasr":
            worker_config.update(
                {
                    "crispasr_model": crispasr_model,
                    "crispasr_model_path": crispasr_model_path_value,
                    "crispasr_backend": crispasr_backend,
                    "crispasr_gpu_backend": crispasr_gpu_backend,
                    "crispasr_device_index": crispasr_device_index,
                    "crispasr_punc_model": crispasr_punc_model,
                    "crispasr_unified_memory": crispasr_unified_memory,
                }
            )
        elif engine_type == "sherpa-onnx":
            worker_config.update(
                {
                    "sherpa_onnx_model": sherpa_onnx_model_path,
                    "sherpa_onnx_model_path": sherpa_onnx_model_path,
                    "sherpa_onnx_model_info": sherpa_onnx_model_info,
                    "sherpa_onnx_provider": sherpa_onnx_provider,
                    "sherpa_onnx_num_threads": sherpa_onnx_num_threads,
                    "sherpa_onnx_decoding_method": sherpa_onnx_decoding_method,
                    "sherpa_onnx_left_padding_seconds": sherpa_onnx_left_padding_seconds,
                    "sherpa_onnx_tail_padding_seconds": sherpa_onnx_tail_padding_seconds,
                }
            )
        elif engine_type == "parakeet-cpp":
            worker_config.update(
                {
                    "parakeet_cpp_model": parakeet_cpp_model,
                    "parakeet_cpp_model_path": parakeet_cpp_model_path,
                    "parakeet_cpp_runtime_dir": parakeet_cpp_runtime_dir,
                    "parakeet_cpp_backend": parakeet_cpp_backend,
                    "parakeet_cpp_decoder": parakeet_cpp_decoder,
                    "parakeet_cpp_word_timestamps": parakeet_cpp_word_timestamps,
                }
            )
        target_state = {
            "type": engine_type,
            "signature": signature,
            "device": device,
            "funasr_model_key": funasr_model
            if engine_type == "funasr"
            else self._funasr_model_key,
            "whisper_model_size": model_size
            if engine_type == "whisper"
            else self._whisper_model_size,
            "crispasr_model_key": crispasr_model
            if engine_type == "crispasr"
            else self._crispasr_model_key,
            "sherpa_onnx_model_path": sherpa_onnx_model_path
            if engine_type == "sherpa-onnx"
            else self._sherpa_onnx_model_path,
            "parakeet_cpp_model_key": parakeet_cpp_model
            if engine_type == "parakeet-cpp"
            else self._parakeet_cpp_model_key,
            "parakeet_cpp_runtime_dir": parakeet_cpp_runtime_dir
            if engine_type == "parakeet-cpp"
            else self._parakeet_cpp_runtime_dir,
            "config": worker_config,
            "display_name": display_name,
            "device_label": remote_asr_url
            if engine_type == "remote-whisper"
            else device,
        }

        missing = []
        if not cached:
            missing = get_missing_models(engine_type, cache_model_key, hub)
            missing = [m for m in missing if m["type"] != "silero-vad"]

        return ASRSwitchPlan(
            engine_type=engine_type,
            device=device,
            hub=hub,
            download_proxy=download_proxy,
            display_name=display_name,
            cache_model_key=str(cache_model_key),
            worker_config=worker_config,
            target_state=target_state,
            missing_models=missing,
        )

    def mark_download_cancelled(self):
        with self._asr_lock:
            self._asr_ready = self._asr is not None and self._asr.status == "ready"

    def switch_worker(self, plan: ASRSwitchPlan) -> ASRSwitchResult:
        with self._asr_lock:
            old_asr = self._asr
            old_config = dict(self._asr_config) if self._asr_config else None
            old_state = {
                "type": self._asr_type,
                "signature": self._asr_signature,
                "device": self._asr_device,
                "funasr_model_key": self._funasr_model_key,
                "whisper_model_size": self._whisper_model_size,
                "crispasr_model_key": self._crispasr_model_key,
                "sherpa_onnx_model_path": self._sherpa_onnx_model_path,
                "parakeet_cpp_model_key": self._parakeet_cpp_model_key,
                "parakeet_cpp_runtime_dir": self._parakeet_cpp_runtime_dir,
                "config": old_config,
                "display_name": (old_config or {}).get("display_name"),
            }
            self._asr = None
            self._asr_ready = False
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0

        new_asr = None
        restored_asr = None
        load_error = None
        restore_error = None

        if old_asr is not None:
            log.info(f"Stopping old ASR worker before switch: pid={old_asr.pid}")
            old_asr.shutdown()
            if self._release_memory_caches is not None:
                self._release_memory_caches()

        try:
            new_asr = self._load_engine_client(plan.worker_config)
        except Exception as exc:
            load_error = str(exc)
            log.error(f"Failed to load ASR worker: {exc}", exc_info=True)
            if old_config:
                try:
                    log.info("Restoring previous ASR worker after switch failure")
                    restored_asr = self._load_engine_client(old_config)
                except Exception as restore_exc:
                    restore_error = str(restore_exc)
                    log.error(
                        f"Failed to restore previous ASR worker: {restore_exc}",
                        exc_info=True,
                    )

        if new_asr is not None:
            self._activate_asr(new_asr, plan.target_state)
            log.info(f"ASR worker ready: {plan.engine_type} on {plan.device}")
            return ASRSwitchResult(status="ready", target_state=plan.target_state)

        if restored_asr is not None:
            self._activate_asr(restored_asr, old_state)
            log.info(
                f"Previous ASR worker restored: "
                f"{old_state.get('type')} on {old_state.get('device')}"
            )
            return ASRSwitchResult(
                status="restored",
                restored_state=old_state,
                load_error=load_error,
            )

        return ASRSwitchResult(
            status="failed", load_error=load_error, restore_error=restore_error
        )

    def shutdown(self):
        with self._asr_lock:
            client = self._asr
            self._asr = None
            self._asr_ready = False
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
        if client is not None:
            log.info(f"Shutting down ASR worker: pid={client.pid}")
            client.shutdown()

    def transcribe(self, audio, **kwargs):
        with self._asr_lock:
            if not self._asr_ready or self._asr is None:
                return None
            client = self._asr
            try:
                result = client.transcribe(audio, **kwargs)
            except (ASRWorkerExited, ASRWorkerTimeout) as exc:
                self._mark_asr_unavailable(str(exc), client)
                raise
            except ASRWorkerError as exc:
                self._asr_error_count += 1
                if not exc.recoverable or self._asr_error_count >= 3:
                    self._mark_asr_unavailable(str(exc), client)
                raise
            self._asr_error_count = 0
            return result

    def set_language(self, language: str):
        with self._asr_lock:
            client = self._asr
            if not self._asr_ready or client is None:
                return
            try:
                client.set_language(language)
            except (ASRWorkerExited, ASRWorkerTimeout) as exc:
                self._mark_asr_unavailable(str(exc), client)
            except ASRWorkerError as exc:
                log.warning(f"ASR language update failed: {exc}")

    def set_padding(self, engine_type: str, pad_seconds):
        with self._asr_lock:
            client = self._asr
            if not self._asr_ready or client is None or self._asr_type != engine_type:
                return
            if engine_type == "funasr" and not funasr_supports_padding(
                self._funasr_model_key
            ):
                return
            try:
                client.set_input_padding(pad_seconds)
            except (ASRWorkerExited, ASRWorkerTimeout) as exc:
                self._mark_asr_unavailable(str(exc), client)
            except ASRWorkerError as exc:
                log.warning(f"ASR padding update failed: {exc}")

    def _invalid_plan(
        self,
        engine_type: str,
        device: str,
        hub: str,
        download_proxy: str,
        message: str,
    ) -> ASRSwitchPlan:
        log.warning(message)
        with self._asr_lock:
            if not (
                self._asr_ready
                and self._asr is not None
                and self._asr.status == "ready"
            ):
                self._asr_ready = False
        return ASRSwitchPlan(
            engine_type=engine_type,
            device=device,
            hub=hub,
            download_proxy=download_proxy,
            display_name=engine_type,
            cache_model_key="",
            worker_config={},
            target_state={},
            missing_models=[],
            already_current=True,
            error=message,
        )

    def _mark_asr_unavailable(self, reason: str, client=None):
        with self._asr_lock:
            current = client or self._asr
            if client is not None and self._asr is not client:
                return
            self._asr_ready = False
            self._asr = None
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
        if current is not None:
            try:
                current.shutdown()
            except Exception:
                try:
                    current.terminate()
                except Exception:
                    pass
        log.warning(f"ASR worker unavailable: {reason}")
        if self._unavailable_callback is not None:
            self._unavailable_callback(reason)

    def _load_asr_client(self, worker_config: dict) -> ASRClient:
        client = ASRClient(worker_config)
        try:
            client.start()
            client.wait_ready()
            return client
        except Exception:
            client.shutdown()
            raise

    def _load_engine_client(self, worker_config: dict):
        if worker_config.get("engine_type") == "remote-whisper":
            from asr_remote import RemoteASREngine

            engine = RemoteASREngine(
                server_url=worker_config.get("remote_asr_url")
                or "http://127.0.0.1:8765"
            )
            language = worker_config.get("language")
            if language:
                engine.set_language(language)
            return engine
        return self._load_asr_client(worker_config)

    def _activate_asr(self, client: ASRClient, state: dict):
        with self._asr_lock:
            self._asr = client
            self._asr_type = state["type"]
            self._asr_signature = state["signature"]
            self._asr_device = state["device"]
            self._asr_config = dict(state["config"]) if state["config"] else None
            self._funasr_model_key = state["funasr_model_key"]
            self._whisper_model_size = state["whisper_model_size"]
            self._crispasr_model_key = state["crispasr_model_key"]
            self._sherpa_onnx_model_path = state.get(
                "sherpa_onnx_model_path", self._sherpa_onnx_model_path
            )
            self._parakeet_cpp_model_key = state.get(
                "parakeet_cpp_model_key", self._parakeet_cpp_model_key
            )
            self._parakeet_cpp_runtime_dir = state.get(
                "parakeet_cpp_runtime_dir", self._parakeet_cpp_runtime_dir
            )
            self._asr_ready = True
            self._asr_error_count = 0

    @staticmethod
    def _display_name(
        engine_type: str,
        model_size: str,
        model_path: str | None,
        funasr_model: str,
        crispasr_model: str,
        sherpa_onnx_model_path: str | None = None,
        parakeet_cpp_model_path: str | None = None,
    ) -> str:
        display_name = ASR_DISPLAY_NAMES.get(engine_type, engine_type)
        if engine_type == "whisper":
            display_model = (
                local_faster_whisper_display_name(model_size)
                if model_path
                else model_size
            ) or Path(model_size).name
            display_name = f"Whisper {display_model}"
        elif engine_type == "funasr":
            display_name = funasr_display_name(funasr_model)
        elif engine_type == "crispasr":
            custom_crispasr_path = resolve_custom_crispasr_model(crispasr_model)
            if custom_crispasr_path:
                display_model = (
                    local_crispasr_display_name(crispasr_model)
                    or Path(crispasr_model).name
                )
                display_name = f"CrispASR {display_model}"
        elif engine_type == "sherpa-onnx" and sherpa_onnx_model_path:
            display_model = (
                local_sherpa_onnx_display_name(sherpa_onnx_model_path)
                or Path(sherpa_onnx_model_path).name
            )
            display_name = f"sherpa-onnx {display_model}"
        elif engine_type == "parakeet-cpp" and parakeet_cpp_model_path:
            display_model = (
                local_parakeet_cpp_display_name(parakeet_cpp_model_path)
                or Path(parakeet_cpp_model_path).name
            )
            display_name = f"parakeet.cpp {display_model}"
        return display_name
