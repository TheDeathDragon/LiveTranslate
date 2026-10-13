"""
LiveTranslate - Phase 0 Prototype
Real-time audio translation using WASAPI loopback + faster-whisper + LLM.
"""

import sys
import signal
import logging
import threading
import queue
import gc
from concurrent.futures import ThreadPoolExecutor
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from model_manager import (
    DEFAULT_FUNASR_MODEL,
    apply_cache_env,
    funasr_display_name,
    funasr_supports_padding,
    get_missing_models,
    is_asr_cached,
    ASR_DISPLAY_NAMES,
    MODELS_DIR,
    local_faster_whisper_display_name,
    migrate_funasr_settings,
    normalize_asr_engine_selection,
    normalize_funasr_model_key,
    resolve_custom_whisper_model,
)

# Set cache env BEFORE importing torch so TORCH_HOME is respected
apply_cache_env()

import os

# torch must be imported before PyQt6 to avoid DLL conflicts on Windows
import torch  # noqa: F401

from audio_capture import AudioCapture
from vad_processor import VADProcessor
from asr_client import ASRClient, ASRWorkerError, ASRWorkerExited, ASRWorkerTimeout
from translator import Translator, RepetitionError
from transcript_writer import TranscriptWriter
from live_preview import LivePreview, PreviewQueue

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QDialog, QMessageBox
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QIcon,
    QPixmap,
    QPainter,
    QColor,
    QFont,
    QFontDatabase,
)
from PyQt6.QtCore import QTimer, Qt

from subtitle_overlay import SubtitleOverlay
from subtitle_window import SubtitleWindow
from log_window import LogWindow
from control_panel import (
    ControlPanel,
    SETTINGS_FILE,
    _load_saved_settings,
    _save_settings,
)
from dialogs import (
    SetupWizardDialog,
    ModelDownloadDialog,
    _ModelLoadDialog,
)
from i18n import t, set_lang, LANGUAGES, COMMON_LANG_CODES

_NO_PENDING = object()


def setup_logging():
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"livetrans_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

    for noisy in (
        "httpcore",
        "httpx",
        "openai",
        "filelock",
        "huggingface_hub",
        "funasr",
        "modelscope",
        "onnxruntime",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"Log file: {log_file}")

    # FunASR/ModelScope spam the root logger; suppress after our own init log
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)

    _logger = logging.getLogger("LiveTranslate")

    def _excepthook(exc_type, exc_value, exc_tb):
        _logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        _logger.critical(
            f"Uncaught exception in thread {args.thread}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    return _logger


log = logging.getLogger("LiveTranslate")


def create_app_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(60, 130, 240))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont("Consolas", 28, QFont.Weight.Bold))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "LT")
    p.end()
    return QIcon(pix)


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LiveTranslateApp:
    def __init__(self, config):
        self._config = config
        self._running = False
        self._paused = False
        self._asr_ready = False  # True when ASR model is loaded

        self._audio = AudioCapture(
            device=config["audio"].get("device"),
            sample_rate=config["audio"]["sample_rate"],
            chunk_duration=config["audio"]["chunk_duration"],
        )
        self._vad = VADProcessor(
            sample_rate=config["audio"]["sample_rate"],
            threshold=config["asr"]["vad_threshold"],
            min_speech_duration=config["asr"]["min_speech_duration"],
            max_speech_duration=config["asr"]["max_speech_duration"],
            chunk_duration=config["audio"]["chunk_duration"],
        )
        self._asr_type = None
        self._asr = None
        self._asr_signature = None
        self._asr_config = None
        self._asr_error_count = 0
        self._asr_device = config["asr"]["device"]
        self._whisper_model_size = config["asr"]["model_size"]
        self._funasr_model_key = normalize_funasr_model_key(
            config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL)
        )
        self._asr_lock = threading.RLock()
        self._vad_lock = threading.Lock()
        # Settings changed from the Qt thread are deferred here and applied by the
        # ASR thread before its next transcribe, so the UI never blocks on the
        # worker pipe (which may be busy with an in-flight cross-process call).
        # Padding is keyed by engine_type because one settings save updates both
        # the funasr and whisper padding and they must not clobber each other.
        self._asr_pending_lock = threading.Lock()
        self._asr_pending_language = _NO_PENDING
        self._asr_pending_padding = {}
        # Auto-restart bookkeeping for a worker that dies mid-session. _asr_generation
        # is bumped on every (de)activation so a slow background (re)start can detect
        # that a newer engine switch superseded it and discard its stale worker.
        self._asr_restart_state = None
        self._asr_restart_count = 0
        self._asr_restart_max = 3
        self._asr_generation = 0
        self._asr_recycling = False
        # Proactively recycle the worker once its RSS grows this far past the
        # post-load baseline, to bound native-side (FunASR/CTranslate2) leaks that
        # accumulate in the long-lived worker process.
        self._asr_worker_baseline_mb = None
        self._asr_recycle_delta_mb = 2048
        self._target_language = config["translation"]["target_language"]
        self._translator = Translator(
            api_base=config["translation"]["api_base"],
            api_key=config["translation"]["api_key"],
            model=config["translation"]["model"],
            target_language=self._target_language,
            max_tokens=config["translation"]["max_tokens"],
            temperature=config["translation"]["temperature"],
            streaming=config["translation"]["streaming"],
            system_prompt=config["translation"].get("system_prompt"),
        )
        self._translator.set_context_turns(
            config["translation"].get("context_window", 0)
        )
        self._overlay = None
        self._subwin = None
        self._panel = None
        self._capture_thread = None
        self._asr_thread = None
        self._asr_queue = queue.Queue(maxsize=16)
        self._tl_executor = ThreadPoolExecutor(max_workers=8)

        self._transcript = TranscriptWriter(Path(__file__).parent / "transcripts")

        # Memory diagnostic state
        import psutil
        self._mem_proc = psutil.Process(os.getpid())
        self._mem_baseline_mb = self._mem_proc.memory_info().rss / 1024 / 1024
        self._mem_last_mb = self._mem_baseline_mb
        self._mem_asr_call_count = 0
        self._mem_periodic_timer = None
        # Memory ceiling: warn once when combined RSS (main + ASR worker) exceeds
        # threshold. The ASR backend now runs in a worker process and keeps
        # native-side workspaces/caches that Python GC cannot always reclaim, so the
        # ceiling must include the worker's RSS (see _mem_snapshot).
        self._mem_threshold_mb = 4096
        self._mem_warned = False
        self._mem_warning_callback = None

        self._asr_count = 0
        self._translate_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._input_price = 0.0
        self._output_price = 0.0
        self._msg_id = 0
        self._last_original = ""
        self._last_msg_id = 0

        # Incremental ASR state
        self._incremental_enabled = False
        self._interim_interval = 2.0
        self._interim_pending = ""
        self._interim_active = False
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        self._live_preview = False
        self._preview_translation = False
        self._preview = LivePreview(self._publish_caption, self._preview_usage)
        self._preview_settings = None

    def set_overlay(self, overlay: SubtitleOverlay):
        self._overlay = overlay
        overlay.cleared.connect(self._invalidate_preview)

    def set_subtitle_window(self, subwin: SubtitleWindow):
        self._subwin = subwin

    def set_panel(self, panel: ControlPanel):
        self._panel = panel
        panel.settings_changed.connect(self._on_settings_changed)
        panel.model_changed.connect(self._on_model_changed)
        panel.models_list_changed.connect(self._on_models_list_changed)

    def _on_models_list_changed(self, models: list, active_idx: int):
        if self._overlay:
            self._overlay.set_models(models, active_idx)

    def _on_settings_changed(self, settings):
        signature = tuple(settings.get(k) for k in (
            "live_preview", "asr_language", "asr_engine", "asr_device",
            "funasr_model", "whisper_model_size", "audio_device", "mic_device",
            "target_language", "system_prompt", "hub",
            "sensevoice_pad_seconds", "whisper_pad_seconds"))
        if self._preview_settings is not None and signature != self._preview_settings:
            self._invalidate_preview()
        self._preview_settings = signature
        self._live_preview = bool(settings.get("live_preview", False))
        self._asr_queue.maxsize = 0 if self._live_preview else 16
        self._vad.update_settings(settings)
        if "style" in settings and self._overlay:
            self._overlay.apply_style(settings["style"])
        if "asr_language" in settings:
            self._set_asr_language(settings["asr_language"])
        if "sensevoice_pad_seconds" in settings:
            self._set_asr_padding("funasr", settings["sensevoice_pad_seconds"])
        if "whisper_pad_seconds" in settings:
            self._set_asr_padding("whisper", settings["whisper_pad_seconds"])
        if any(
            key in settings
            for key in (
                "asr_engine",
                "asr_device",
                "whisper_model_size",
                "funasr_model",
                "hub",
            )
        ):
            self._switch_asr_engine(
                settings.get(
                    "asr_engine",
                    self._asr_type or self._config["asr"].get("asr_engine", "funasr"),
                )
            )
        if "audio_device" in settings:
            old_device = self._audio._device_name
            self._audio.set_device(settings["audio_device"])
            if old_device != settings.get("audio_device"):
                self._vad.flush()
                self._vad._reset()
                if self._overlay:
                    self._overlay.update_monitor(0.0, 0.0)
        if "mic_device" in settings:
            self._audio.set_mic_device(settings["mic_device"])
        if "incremental_asr" in settings:
            self._incremental_enabled = settings["incremental_asr"]
        if "interim_interval" in settings:
            self._interim_interval = settings["interim_interval"]
        if "target_language" in settings:
            self._target_language = settings["target_language"]
            if self._overlay:
                self._overlay.set_target_language(self._target_language)
        if "timeout" in settings and self._translator:
            self._translator.set_timeout(settings["timeout"])
        if "auto_save_transcript" in settings:
            self._transcript.set_enabled(settings["auto_save_transcript"])

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
            self._asr_restart_state = None
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1
        if current is not None:
            try:
                current.shutdown()
            except Exception:
                try:
                    current.terminate()
                except Exception:
                    pass
        log.warning(f"ASR worker unavailable: {reason}")
        if self._overlay:
            self._overlay.update_asr_device("ASR unavailable")

    def _shutdown_asr_worker(self):
        with self._asr_lock:
            client = self._asr
            self._asr = None
            self._asr_ready = False
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
            self._asr_restart_state = None
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1
        if client is not None:
            log.info(f"Shutting down ASR worker: pid={client.pid}")
            client.shutdown()

    def _set_asr_language(self, language: str):
        with self._asr_pending_lock:
            self._asr_pending_language = language

    def _set_asr_padding(self, engine_type: str, pad_seconds):
        with self._asr_pending_lock:
            self._asr_pending_padding[engine_type] = pad_seconds

    def _apply_pending_asr_settings(self, client, asr_type, funasr_key):
        """Apply deferred language/padding on the ASR thread, just before a transcribe.
        A pending value is cleared only once delivered; worker-death exceptions
        propagate with the pending intact so the restarted worker re-applies it. The
        applied value is written back into the restart config so an auto-restart or
        recycle does not revert a runtime override to the engine-switch-time value."""
        with self._asr_pending_lock:
            language = self._asr_pending_language
            pad_seconds = self._asr_pending_padding.get(asr_type, _NO_PENDING)
        if language is not _NO_PENDING:
            try:
                client.set_language(language)
            except ASRWorkerError as exc:
                log.warning(f"ASR language update failed: {exc}")
            self._update_restart_config(language=language)
            self._clear_pending_language(language)
        if pad_seconds is not _NO_PENDING:
            if not (asr_type == "funasr" and not funasr_supports_padding(funasr_key)):
                try:
                    client.set_input_padding(pad_seconds)
                except ASRWorkerError as exc:
                    log.warning(f"ASR padding update failed: {exc}")
                self._update_restart_config(pad_seconds=pad_seconds)
            self._clear_pending_padding(asr_type, pad_seconds)

    def _clear_pending_language(self, language):
        with self._asr_pending_lock:
            if self._asr_pending_language is language:
                self._asr_pending_language = _NO_PENDING

    def _clear_pending_padding(self, asr_type, pad_seconds):
        with self._asr_pending_lock:
            if self._asr_pending_padding.get(asr_type) == pad_seconds:
                del self._asr_pending_padding[asr_type]

    def _update_restart_config(self, **kwargs):
        with self._asr_lock:
            if self._asr_restart_state and self._asr_restart_state.get("config"):
                self._asr_restart_state["config"].update(kwargs)

    def _load_engine_client(self, config: dict):
        """Build the ASR backend for a worker config. Local engines run in an isolated
        worker subprocess (ASRClient); remote-whisper is a thin in-process HTTP client
        that needs no subprocess isolation (no native deps, no GPU model to load)."""
        if config.get("engine_type") == "remote-whisper":
            from asr_remote import RemoteASREngine

            url = config.get("remote_asr_url") or "http://127.0.0.1:8765"
            engine = RemoteASREngine(server_url=url)
            language = config.get("language")
            if language:
                engine.set_language(language)
            return engine
        return self._load_asr_client(config)

    def _load_asr_client(self, worker_config: dict) -> ASRClient:
        # request_timeout bounds how long a hung worker can stall the realtime path
        # before it is killed and auto-restarted. VAD caps segments at a few seconds,
        # so 60s is generous for a healthy transcribe yet far below the old 120s.
        client = ASRClient(worker_config, request_timeout=60.0)
        try:
            client.start()
            client.wait_ready()
            return client
        except Exception:
            client.shutdown()
            raise

    def _on_target_language_changed(self, lang: str):
        if lang != self._target_language:
            self._invalidate_preview()
        self._target_language = lang
        log.info(f"Target language: {lang}")
        if self._translator:
            self._translator.set_target_language(lang)
        if self._panel:
            settings = self._panel.get_settings()
            settings["target_language"] = lang
            from control_panel import _save_settings

            _save_settings(settings)

    def _on_model_changed(self, model_config: dict):
        self._invalidate_preview()
        self._preview_translation = bool(model_config.get("preview_translation", False))
        log.info(
            f"Switching translator: {model_config['name']} ({model_config['model']})"
        )
        prompt = None
        if self._panel:
            prompt = self._panel.get_settings().get("system_prompt")
        if not prompt:
            prompt = self._config["translation"].get("system_prompt")
        timeout = 10
        if self._panel:
            timeout = self._panel.get_settings().get("timeout", 10)
        self._translator = Translator(
            api_base=model_config["api_base"],
            api_key=model_config["api_key"],
            model=model_config["model"],
            target_language=self._target_language,
            max_tokens=self._config["translation"]["max_tokens"],
            temperature=self._config["translation"]["temperature"],
            streaming=model_config.get("streaming", True),
            system_prompt=prompt,
            proxy=model_config.get("proxy", "none"),
            no_system_role=model_config.get("no_system_role", False),
            no_think=model_config.get("no_think", True),
            thinking_style=model_config.get("thinking_style"),
            json_response=model_config.get("json_response", False),
            timeout=timeout,
            overrides=model_config.get("overrides"),
            extra_body=model_config.get("extra_body"),
        )
        context_turns = model_config.get(
            "context_turns", self._config["translation"].get("context_window", 0)
        )
        self._translator.set_context_turns(context_turns)
        self._input_price = model_config.get("input_price", 0)
        self._output_price = model_config.get("output_price", 0)

    def _switch_asr_engine(self, engine_type: str):
        settings = self._panel.get_settings() if self._panel else {}
        engine_type, funasr_model = normalize_asr_engine_selection(
            engine_type, settings.get("funasr_model", self._funasr_model_key)
        )
        device = settings.get("asr_device", self._asr_device)
        hub = "ms"
        download_proxy = "system"
        if self._panel:
            hub = settings.get("hub", "ms")
            download_proxy = settings.get("download_proxy", "system")

        model_size = self._config["asr"]["model_size"]
        if self._panel:
            model_size = settings.get("whisper_model_size", model_size)
        model_path = None
        cache_model_key = model_size
        if engine_type == "whisper":
            model_path = resolve_custom_whisper_model(model_size)
            if model_path:
                cache_model_key = model_path
        elif engine_type == "funasr":
            cache_model_key = funasr_model

        remote_asr_url = settings.get(
            "remote_asr_url",
            self._config["asr"].get("remote_asr_url", "http://127.0.0.1:8765"),
        )

        compute = self._config["asr"]["compute_type"]
        if engine_type == "whisper":
            signature_model = cache_model_key
        elif engine_type == "funasr":
            signature_model = funasr_model
        elif engine_type == "remote-whisper":
            # URL is part of the identity so editing it triggers a reconnect.
            signature_model = remote_asr_url
        else:
            signature_model = engine_type
        signature = (engine_type, signature_model, device, hub, compute)

        with self._asr_lock:
            current_asr = self._asr
            current_ready = (
                self._asr_ready
                and current_asr is not None
                and current_asr.status == "ready"
            )
            if current_ready and self._asr_signature == signature:
                return
            if not current_ready:
                self._asr_ready = False

        log.info(f"Switching ASR worker: {self._asr_type} -> {engine_type}")
        # Reset interim state for the engine boundary. The active worker is
        # stopped before the target worker starts loading.
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        self._vad.flush()
        self._vad._reset()

        cached = is_asr_cached(engine_type, cache_model_key, hub)
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

        parent = (
            self._panel if self._panel and self._panel.isVisible() else self._overlay
        )

        worker_config = {
            "engine_type": engine_type,
            "funasr_model": funasr_model,
            "model_size": cache_model_key,
            "device": device,
            "compute_type": compute,
            "hub": hub,
            "language": settings.get(
                "asr_language", self._config["asr"].get("language", "auto")
            ),
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
            "config": worker_config,
            "display_name": display_name,
            "device_label": (
                remote_asr_url if engine_type == "remote-whisper" else device
            ),
        }

        if not cached:
            missing = get_missing_models(engine_type, cache_model_key, hub)
            missing = [m for m in missing if m["type"] != "silero-vad"]
            if missing:
                dlg = ModelDownloadDialog(
                    missing, hub=hub, proxy=download_proxy, parent=parent
                )
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    log.info(f"Download cancelled/failed: {engine_type}")
                    with self._asr_lock:
                        self._asr_ready = (
                            self._asr is not None and self._asr.status == "ready"
                        )
                    return

        with self._asr_lock:
            old_asr = self._asr
            old_config = dict(self._asr_config) if self._asr_config else None
            old_state = {
                "type": self._asr_type,
                "signature": self._asr_signature,
                "device": self._asr_device,
                "funasr_model_key": self._funasr_model_key,
                "whisper_model_size": self._whisper_model_size,
                "config": old_config,
                "display_name": (old_config or {}).get("display_name"),
                "device_label": (
                    (old_config or {}).get("remote_asr_url")
                    if self._asr_type == "remote-whisper"
                    else self._asr_device
                ),
            }
            self._asr = None
            self._asr_ready = False
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
            self._asr_restart_state = None
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1

        dlg = _ModelLoadDialog(
            t("loading_model").format(name=display_name), parent=parent
        )

        new_asr = [None]
        restored_asr = [None]
        load_error = [None]
        restore_error = [None]

        def _load():
            if old_asr is not None:
                log.info(f"Stopping old ASR worker before switch: pid={old_asr.pid}")
                old_asr.shutdown()
                self._release_memory_caches()
            try:
                new_asr[0] = self._load_engine_client(worker_config)
            except Exception as e:
                load_error[0] = str(e)
                # A remote server that is simply down is an expected, user-actionable
                # condition, not a bug, so skip the noisy traceback for it.
                expected = isinstance(e, ConnectionError)
                log.error(
                    f"Failed to load ASR worker: {e}", exc_info=not expected
                )
                if old_config:
                    try:
                        log.info("Restoring previous ASR worker after switch failure")
                        restored_asr[0] = self._load_engine_client(old_config)
                    except Exception as restore_exc:
                        restore_error[0] = str(restore_exc)
                        log.error(
                            f"Failed to restore previous ASR worker: {restore_exc}",
                            exc_info=True,
                        )

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

        poll_timer = QTimer()

        def _check():
            if not thread.is_alive():
                poll_timer.stop()
                dlg.accept()

        poll_timer.setInterval(100)
        poll_timer.timeout.connect(_check)
        poll_timer.start()

        dlg.exec()
        poll_timer.stop()

        def _activate_asr(client, state):
            with self._asr_lock:
                self._asr = client
                self._asr_type = state["type"]
                self._asr_signature = state["signature"]
                self._asr_device = state["device"]
                self._asr_config = dict(state["config"]) if state["config"] else None
                self._funasr_model_key = state["funasr_model_key"]
                self._whisper_model_size = state["whisper_model_size"]
                self._asr_ready = True
                self._asr_error_count = 0
                self._asr_restart_state = dict(state)
                self._asr_restart_count = 0
                self._asr_worker_baseline_mb = None
                self._asr_generation += 1

        if new_asr[0] is not None:
            _activate_asr(new_asr[0], target_state)
            if self._overlay:
                self._overlay.update_asr_device(
                    f"{display_name} [{target_state['device_label']}]"
                )
            log.info(f"ASR worker ready: {engine_type} on {device}")
            return

        if restored_asr[0] is not None:
            _activate_asr(restored_asr[0], old_state)
            restored_name = old_state.get("display_name") or old_state.get("type")
            if self._overlay:
                self._overlay.update_asr_device(
                    f"{restored_name} [{old_state.get('device_label', old_state['device'])}]"
                )
            QMessageBox.warning(
                parent,
                t("error_title"),
                t("error_load_asr").format(
                    error=(
                        f"{load_error[0] or 'unknown error'}\n"
                        f"{t('asr_restore_succeeded')}"
                    )
                ),
            )
            log.info(
                f"Previous ASR worker restored: "
                f"{old_state.get('type')} on {old_state.get('device')}"
            )
            return

        error = load_error[0] or "unknown error"
        if restore_error[0]:
            error = (
                f"{error}\n"
                f"{t('asr_restore_failed').format(error=restore_error[0])}"
            )
        QMessageBox.warning(
            parent,
            t("error_title"),
            t("error_load_asr").format(error=error),
        )

        if self._overlay:
            self._overlay.update_asr_device("ASR unavailable")

    def _mem_snapshot(self) -> dict:
        rss_mb = self._mem_proc.memory_info().rss / 1024 / 1024
        # The ASR model (and its native-side leak) lives in the worker process now,
        # so sample its RSS too; the main process holds only VAD + Qt.
        worker_rss_mb = 0.0
        client = self._asr
        if client is not None and client.pid is not None:
            try:
                import psutil

                worker_rss_mb = (
                    psutil.Process(client.pid).memory_info().rss / 1024 / 1024
                )
            except Exception:
                worker_rss_mb = 0.0
        gpu_alloc_mb = 0.0
        gpu_reserved_mb = 0.0
        try:
            if torch.cuda.is_available():
                gpu_alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        except Exception:
            pass
        msgs = len(self._overlay._messages) if self._overlay else 0
        vad_buf = len(self._vad._speech_buffer)
        return {
            "rss": rss_mb,
            "worker_rss": worker_rss_mb,
            "total_rss": rss_mb + worker_rss_mb,
            "gpu_alloc": gpu_alloc_mb,
            "gpu_reserved": gpu_reserved_mb,
            "msgs": msgs,
            "vad_buf": vad_buf,
        }

    def _log_mem_after_asr(self, kind: str, audio_seconds: float, asr_ms: float):
        self._mem_asr_call_count += 1
        snap = self._mem_snapshot()
        delta = snap["rss"] - self._mem_last_mb
        total_delta = snap["rss"] - self._mem_baseline_mb
        self._mem_last_mb = snap["rss"]
        log.info(
            f"MEM[asr#{self._mem_asr_call_count}:{kind}] RSS={snap['rss']:.1f}MB "
            f"(Δ{delta:+.2f} since last, {total_delta:+.1f} since start) "
            f"worker_rss={snap['worker_rss']:.0f}MB "
            f"GPU(main alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"audio={audio_seconds:.1f}s asr={asr_ms:.0f}ms "
            f"outputs={self._asr_count} msgs={snap['msgs']} vad_buf={snap['vad_buf']}"
        )
        self._check_memory_threshold(snap["total_rss"])

    def _release_memory_caches(self):
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _run_asr(self, audio: np.ndarray, kind: str, **kwargs):
        audio_seconds = len(audio) / 16000
        asr_start = time.perf_counter()
        # Snapshot the active client under the lock, then release it: the blocking
        # cross-process transcribe must not hold _asr_lock, or a slow/hung worker
        # would freeze the Qt thread on every settings change. ASRClient serializes
        # its own pipe access, and only this (single) ASR thread calls transcribe.
        with self._asr_lock:
            if not self._asr_ready or self._asr is None:
                return None, 0.0
            client = self._asr
            asr_type = self._asr_type
            funasr_key = self._funasr_model_key
        try:
            self._apply_pending_asr_settings(client, asr_type, funasr_key)
            result = client.transcribe(audio, **kwargs)
        except (ASRWorkerExited, ASRWorkerTimeout) as exc:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._log_mem_after_asr(f"{kind}:error", audio_seconds, asr_ms)
            self._recover_asr_worker(client, str(exc))
            raise
        except ASRWorkerError as exc:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._log_mem_after_asr(f"{kind}:error", audio_seconds, asr_ms)
            fatal = False
            with self._asr_lock:
                if self._asr is client:
                    self._asr_error_count += 1
                    fatal = not exc.recoverable or self._asr_error_count >= 3
            if fatal:
                self._mark_asr_unavailable(str(exc), client)
            raise
        except Exception:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._log_mem_after_asr(f"{kind}:error", audio_seconds, asr_ms)
            raise
        with self._asr_lock:
            if self._asr is client:
                self._asr_error_count = 0
                self._asr_restart_count = 0
        asr_ms = (time.perf_counter() - asr_start) * 1000
        self._log_mem_after_asr(kind, audio_seconds, asr_ms)
        return result, asr_ms

    def _start_worker_from_state(self, state: dict, expected_gen: int) -> bool:
        """Load a worker from a saved state and activate it only if no newer engine
        switch happened in the meantime (generation guard). Runs on the ASR thread;
        the load is intentionally done outside _asr_lock. Returns True on activation."""
        try:
            client = self._load_engine_client(state["config"])
        except Exception as e:
            log.error(f"ASR worker (re)start failed: {e}", exc_info=True)
            return False
        stale = None
        with self._asr_lock:
            if self._asr_generation != expected_gen or not self._running:
                stale = client
            else:
                self._asr = client
                self._asr_type = state["type"]
                self._asr_signature = state["signature"]
                self._asr_device = state["device"]
                self._asr_config = dict(state["config"]) if state["config"] else None
                self._funasr_model_key = state["funasr_model_key"]
                self._whisper_model_size = state["whisper_model_size"]
                self._asr_ready = True
                self._asr_error_count = 0
                self._asr_restart_state = dict(state)
                self._asr_worker_baseline_mb = None
                self._asr_generation += 1
        if stale is not None:
            log.info("Discarding superseded ASR worker (newer switch won the race)")
            try:
                stale.shutdown()
            except Exception:
                pass
            return False
        name = state.get("display_name") or state.get("type")
        if self._overlay:
            self._overlay.update_asr_device(
                f"{name} [{state.get('device_label', state['device'])}]"
            )
        return True

    def _recover_asr_worker(self, dead_client, reason: str):
        """Auto-restart a worker that died mid-session. Without this, a single crash
        or transcribe timeout would leave ASR permanently silent for the session."""
        with self._asr_lock:
            if self._asr is not dead_client:
                return  # an engine switch already replaced/cleared it
            state = dict(self._asr_restart_state) if self._asr_restart_state else None
            attempt = self._asr_restart_count + 1
            give_up = (
                state is None
                or not state.get("config")
                or attempt > self._asr_restart_max
            )
            self._asr_restart_count = attempt
            self._asr = None
            self._asr_ready = False
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1
            gen = self._asr_generation
        try:
            dead_client.shutdown()
        except Exception:
            try:
                dead_client.terminate()
            except Exception:
                pass
        if not self._running:
            return  # shutting down; do not spawn a replacement worker
        if give_up:
            log.error(
                f"ASR worker died and auto-restart gave up after "
                f"{self._asr_restart_max} attempts: {reason}"
            )
            if self._overlay:
                self._overlay.update_asr_device("ASR unavailable")
            return
        log.warning(
            f"ASR worker died ({reason}); auto-restart attempt "
            f"{attempt}/{self._asr_restart_max}"
        )
        self._release_memory_caches()
        if self._start_worker_from_state(state, gen):
            log.info(
                f"ASR worker auto-restarted: {state.get('type')} on "
                f"{state.get('device')}"
            )
        elif self._asr is None and self._overlay:
            self._overlay.update_asr_device("ASR unavailable")

    def _maybe_recycle_asr_worker(self):
        """Recycle the worker once its RSS grows well past the post-load baseline, to
        bound native-side leaks that accumulate in the long-lived worker process.
        Called from the ASR thread between segments so the reload gap costs no audio
        beyond what arrives during it."""
        if not self._running:
            return
        with self._asr_lock:
            client = self._asr
            if not self._asr_ready or client is None or self._asr_recycling:
                return
            state = dict(self._asr_restart_state) if self._asr_restart_state else None
        if state is None or not state.get("config") or client.pid is None:
            return
        try:
            import psutil

            rss = psutil.Process(client.pid).memory_info().rss / 1024 / 1024
        except Exception:
            return
        if self._asr_worker_baseline_mb is None:
            self._asr_worker_baseline_mb = rss
            return
        if rss < self._asr_worker_baseline_mb + self._asr_recycle_delta_mb:
            return
        log.warning(
            f"ASR worker RSS={rss:.0f}MB grew "
            f"{rss - self._asr_worker_baseline_mb:.0f}MB over baseline; recycling"
        )
        self._recycle_asr_worker(client, state)

    def _recycle_asr_worker(self, old_client, state: dict):
        # Graceful stop-then-start (no VRAM doubling). The generation guard makes a
        # concurrent engine switch win over this recycle.
        with self._asr_lock:
            if self._asr is not old_client:
                return
            self._asr = None
            self._asr_ready = False
            self._asr_recycling = True
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1
            gen = self._asr_generation
        try:
            old_client.shutdown()
        except Exception:
            try:
                old_client.terminate()
            except Exception:
                pass
        self._release_memory_caches()
        if not self._running:
            with self._asr_lock:
                self._asr_recycling = False
            return
        try:
            started = self._start_worker_from_state(state, gen)
        finally:
            with self._asr_lock:
                self._asr_recycling = False
        if started:
            log.info(f"ASR worker recycled: {state.get('type')} on {state.get('device')}")
        else:
            log.error("ASR worker recycle failed to restart")
            if self._asr is None and self._overlay:
                self._overlay.update_asr_device("ASR unavailable")

    def _check_memory_threshold(self, rss_mb: float):
        if self._mem_warned or rss_mb < self._mem_threshold_mb:
            return
        self._mem_warned = True
        log.warning(
            f"Memory ceiling reached: combined RSS (main+worker)={rss_mb:.0f}MB "
            f"(threshold {self._mem_threshold_mb}MB). "
            f"Recommend restarting LiveTranslate to free C-side allocator caches."
        )
        if self._mem_warning_callback is not None:
            try:
                self._mem_warning_callback(rss_mb)
            except Exception as e:
                log.warning(f"Memory warning callback failed: {e}")

    def set_memory_warning_callback(self, callback):
        self._mem_warning_callback = callback

    def _log_mem_periodic(self):
        snap = self._mem_snapshot()
        total_delta = snap["rss"] - self._mem_baseline_mb
        log.info(
            f"MEM[tick] RSS={snap['rss']:.1f}MB ({total_delta:+.1f} since start) "
            f"worker_rss={snap['worker_rss']:.0f}MB "
            f"GPU(main alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"msgs={snap['msgs']} asr_calls={self._mem_asr_call_count} "
            f"asr_count={self._asr_count} tl_count={self._translate_count}"
        )
        self._check_memory_threshold(snap["total_rss"])

    def _compute_cost(self):
        if self._input_price > 0 or self._output_price > 0:
            return (self._total_prompt_tokens * self._input_price +
                    self._total_completion_tokens * self._output_price) / 1_000_000
        return 0.0

    def _translate_async(self, msg_id, text, source_lang, extra_langs=None):
        """Translate text and update UI with streaming display."""
        try:
            tl_start = time.perf_counter()
            translated = None
            for partial in self._translator.translate_iter(text, source_lang):
                translated = partial
                if self._overlay:
                    self._overlay.update_streaming(msg_id, partial)
            tl_ms = (time.perf_counter() - tl_start) * 1000
            self._translate_count += 1
            pt, ct = self._translator.last_usage
            self._total_prompt_tokens += pt
            self._total_completion_tokens += ct
            cost = self._compute_cost()
            log.info(f"Translate ({tl_ms:.0f}ms): {translated}")
            if translated:
                self._transcript.write_translation(msg_id, translated)
            else:
                self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(msg_id, translated, tl_ms)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    cost,
                )
            if self._subwin and self._subwin.isVisible() and translated:
                tl_dict = {self._target_language: translated}
                if extra_langs:
                    self._translate_extra_langs(text, source_lang, extra_langs, tl_dict)
                self._subwin.update_text(text, tl_dict)
        except RepetitionError:
            log.warning("Repetition loop detected, model may not support structured output well")
            self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(
                    msg_id, f"[{t('error_repetition')}]", 0
                )
        except Exception as e:
            import openai
            if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError,
                              openai.AuthenticationError, openai.APIStatusError,
                              TimeoutError, ConnectionError)):
                log.warning(f"Translate error: {e}")
            else:
                log.error(f"Translate error: {e}", exc_info=True)
            self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(msg_id, f"[error: {e}]", 0)

    def _translate_extra_langs(self, text, source_lang, extra_langs, tl_dict):
        """Translate into additional languages for subtitle window (parallel)."""
        from concurrent.futures import as_completed

        def _do_translate(lang):
            translator = self._translator.with_target_language(lang)
            return lang, translator.translate(text, source_lang)

        futures = []
        for lang in extra_langs:
            futures.append(self._tl_executor.submit(_do_translate, lang))

        for future in as_completed(futures):
            try:
                lang, result = future.result()
                tl_dict[lang] = result
                log.info(f"Extra translate [{lang}]: {result}")
            except Exception as e:
                import openai
                if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError,
                                  openai.AuthenticationError, openai.APIStatusError,
                                  TimeoutError, ConnectionError)):
                    log.warning(f"Extra translate error: {e}")
                else:
                    log.error(f"Extra translate error: {e}", exc_info=True)

    def _translate_subwin_only(self, text, source_lang, extra_langs):
        """Translate only for subtitle window when primary target == source language."""
        tl_dict = {self._target_language: text}  # same language, use original
        self._translate_extra_langs(text, source_lang, extra_langs, tl_dict)
        if self._subwin and self._subwin.isVisible():
            self._subwin.update_text(text, tl_dict)

    def start(self):
        if self._running:
            return
        n = len(self._subwin.get_target_languages()) if self._subwin else 1
        self._tl_executor = ThreadPoolExecutor(max_workers=max(8, n + 1))
        self._asr_queue = PreviewQueue(maxsize=0 if self._live_preview else 16)
        self._preview.start()
        self._invalidate_preview()
        self._running = True
        self._paused = False
        self._audio.start()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._asr_thread = threading.Thread(
            target=self._asr_loop, daemon=True
        )
        self._capture_thread.start()
        self._asr_thread.start()
        # Periodic memory snapshot every 30s
        if self._mem_periodic_timer is None:
            self._mem_periodic_timer = QTimer()
            self._mem_periodic_timer.timeout.connect(self._log_mem_periodic)
            self._mem_periodic_timer.start(30000)
        snap = self._mem_snapshot()
        log.info(
            f"MEM[start] RSS={snap['rss']:.1f}MB "
            f"GPU(alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"(baseline for delta tracking)"
        )
        log.info("Pipeline started (capture + ASR threads)")

    def stop(self):
        self._running = False
        self._audio.stop()
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None
        if isinstance(self._asr_queue, PreviewQueue):
            self._asr_queue.discard_preview()
        self._asr_queue.put(None)
        if self._asr_thread:
            self._asr_thread.join(timeout=10)
            if self._asr_thread.is_alive():
                log.warning("ASR thread still running after timeout, proceeding with cleanup")
            self._asr_thread = None
        # Flush remaining VAD buffer after pipeline threads are done
        if self._live_preview:
            remaining = self._vad.flush()
            if remaining is not None and self._asr_ready:
                self._process_preview_audio(self._preview.frame(remaining, final=True))
        elif self._interim_active:
            remaining = self._vad.force_flush()
            if remaining is not None and self._asr_ready:
                self._process_interim_final(remaining)
        else:
            remaining = self._vad.flush()
            if remaining is not None and self._asr_ready:
                self._process_segment(remaining)
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        self._preview.stop()
        self._invalidate_preview()
        self._tl_executor.shutdown(wait=True)
        self._transcript.close()
        if self._mem_periodic_timer is not None:
            try:
                self._mem_periodic_timer.stop()
            except Exception:
                pass
            self._mem_periodic_timer = None
        snap = self._mem_snapshot()
        total_delta = snap["rss"] - self._mem_baseline_mb
        log.info(
            f"MEM[stop] RSS={snap['rss']:.1f}MB ({total_delta:+.1f} since start) "
            f"GPU(alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"asr_calls={self._mem_asr_call_count} outputs={self._asr_count}"
        )
        self._shutdown_asr_worker()
        log.info("Pipeline stopped")

    def pause(self):
        self._paused = True
        self._invalidate_preview()
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        if self._overlay:
            self._overlay.update_monitor(0.0, 0.0)
        log.info("Pipeline paused")

    def resume(self):
        self._paused = False
        log.info("Pipeline resumed")

    def _process_segment(self, speech_segment):
        """Run ASR + translation on a speech segment. Called from ASR thread and stop()."""
        seg_len = len(speech_segment) / 16000
        log.info(f"Speech segment: {seg_len:.1f}s")

        try:
            result, asr_ms = self._run_asr(speech_segment, "segment")
        except Exception as e:
            log.error(f"ASR error: {e}", exc_info=True)
            return
        if asr_ms == 0:
            return
        if asr_ms > 10000:
            log.warning(f"ASR took {asr_ms:.0f}ms, possible hang")
        if result is None:
            return

        original_text = result["text"].strip()
        # Skip empty or punctuation-only ASR results
        if not original_text or not any(c.isalnum() for c in original_text):
            log.debug(
                f"ASR returned empty/punctuation-only, skipping: '{result['text']}'"
            )
            return

        # Skip suspiciously short text from long segments (likely noise)
        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(
                f"Noise filter: {seg_len:.1f}s segment produced only '{original_text}', skipping"
            )
            return

        source_lang = result["language"]
        asr_lang_setting = self._panel.get_settings().get("asr_language", "auto") if self._panel else "auto"
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(
                f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', "
                f"discarding: {original_text[:60]}"
            )
            return

        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms): {original_text}")

        if self._overlay:
            self._overlay.add_message(
                msg_id, timestamp, original_text, source_lang, asr_ms
            )
        self._transcript.write_original(msg_id, timestamp, original_text)

        # Store for subtitle window (translation will be added later)
        self._last_original = original_text
        self._last_msg_id = msg_id

        target_lang = self._target_language

        # Collect extra languages needed by subtitle window (beyond the primary target)
        extra_langs = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            # Remove primary target and source (no need to translate those)
            extra_langs = subwin_langs - {target_lang, source_lang}

        if source_lang == target_lang:
            log.info(f"Same language ({source_lang}), no translation")
            self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(msg_id, "", 0)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    self._compute_cost(),
                )
            if self._subwin and self._subwin.isVisible():
                # Primary is same language; still need to translate extra langs
                if extra_langs:
                    try:
                        self._tl_executor.submit(
                            self._translate_subwin_only, original_text, source_lang, extra_langs
                        )
                    except RuntimeError:
                        pass
                else:
                    self._subwin.update_text(original_text, {target_lang: original_text})
        else:
            try:
                self._tl_executor.submit(
                    self._translate_async, msg_id, original_text, source_lang,
                    extra_langs or None,
                )
            except RuntimeError:
                log.warning("Translation executor shut down, skipping")

    # ── Incremental ASR ──

    def _invalidate_preview(self):
        if not hasattr(self, "_preview"):
            return
        session = self._preview.invalidate()
        if self._live_preview:
            with self._vad_lock:
                self._vad._reset()
        self._last_interim_check_time = 0.0
        for window in (self._overlay, self._subwin):
            if window:
                window.reset_preview(session)

    def _preview_usage(self, prompt_tokens, completion_tokens):
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

    def _publish_caption(self, frame, original, translations, asr_ms, tl_ms, phase):
        final = phase == "final"
        translation = translations.get(self._target_language, "")
        if final and original:
            self._msg_id += 1
            self._asr_count += 1
            self._translate_count += bool(translation)
            stamp = datetime.fromtimestamp(time.time()).strftime("%H:%M:%S")
            self._transcript.write_original(self._msg_id, stamp, original)
            if translation:
                self._transcript.write_translation(self._msg_id, translation)
            else:
                self._transcript.finalize_no_translation(self._msg_id)
            log.info("Caption final [%s:%s] ASR=%.0fms TL=%.0fms", frame.session,
                     frame.utterance, asr_ms, tl_ms)
        for window in (self._overlay, self._subwin):
            if window:
                window.update_caption(frame.session, frame.utterance, frame.revision,
                                      original, translations, final,
                                      self._msg_id if final and original else 0)
        if self._overlay:
            self._overlay.update_stats(self._asr_count, self._translate_count,
                                      self._total_prompt_tokens, self._total_completion_tokens,
                                      self._compute_cost())

    def _process_preview_audio(self, frame):
        if not self._preview.valid(frame):
            return
        text, language, asr_ms = "", "unknown", 0
        try:
            result, asr_ms = self._run_asr(
                frame.audio, "preview_final" if frame.final else "preview")
            if not self._preview.valid(frame):
                return
            if result:
                text = result["text"].strip()
                language = result["language"]
                expected = self._panel.get_settings().get("asr_language", "auto") if self._panel else "auto"
                if (not any(c.isalnum() for c in text) or
                        (expected != "auto" and language != expected) or
                        (len(frame.audio) / 16000 >= 2 and sum(c.isalnum() for c in text) <= 3)):
                    text = ""
        except Exception:
            log.exception("Preview ASR failed")
        extras = self._subwin.get_target_languages() - {language, self._target_language} if self._subwin and self._subwin.isVisible() else ()
        self._preview.submit(frame, text, language, asr_ms, self._translator,
                             self._target_language, self._preview_translation, extras)

    _segmenter_cache = {}  # lang -> yasbd pysbd_adapter.Segmenter

    @staticmethod
    def _get_segmenter(lang: str):
        from yasbd import get_supported_langs, pysbd_adapter
        if lang not in LiveTranslateApp._segmenter_cache:
            seg_lang = lang if lang in get_supported_langs() else "en"
            LiveTranslateApp._segmenter_cache[lang] = pysbd_adapter.Segmenter(
                language=seg_lang, clean=False
            )
        return LiveTranslateApp._segmenter_cache[lang]

    def _split_sentences(self, text: str, lang: str = "en") -> list[str]:
        """Split text into sentences using yasbd, with comma fallback for long text."""
        seg = self._get_segmenter(lang)
        parts = [p for p in seg.segment(text) if p.strip()]
        if len(parts) > 1:
            return parts

        # Comma fallback for long unsplit text — split at last balanced comma
        # CJK 「、」at 25 chars; all commas at 60 chars (long sentence, reduce latency)
        min_len = 25 if any(c == '、' for c in text) else 60
        if len(text) > min_len:
            for i in range(len(text) - 8, 5, -1):
                if text[i] in ',，;；、':
                    before = text[:i + 1].strip()
                    after = text[i + 1:].strip()
                    if before and after and len(before) > 15 and len(after) > 3:
                        return [before, after]

        return parts

    @staticmethod
    def _is_short_utterance(text: str) -> bool:
        """Check if text has ≤8 alphanumeric chars (likely noise/filler/fragment)."""
        alnum = sum(1 for c in text if c.isalnum())
        return alnum <= 8

    def _strip_committed_overlap(self, text: str) -> str:
        """Remove text that overlaps with previously committed content."""
        if not self._interim_committed_tail:
            return text
        tail = self._interim_committed_tail.lower().rstrip()
        text_lower = text.lower()
        # Check if text starts with a suffix of the committed tail
        max_check = min(len(tail), len(text_lower))
        for overlap_len in range(max_check, 2, -1):
            if text_lower[:overlap_len] == tail[-overlap_len:]:
                stripped = text[overlap_len:].strip()
                if stripped:
                    log.debug(f"Stripped echo overlap ({overlap_len} chars): '{text[:overlap_len]}...'")
                    return stripped
                return ""
        return text

    def _do_interim_asr(self) -> bool:
        """Run ASR on current VAD buffer, output complete sentences, trim consumed audio.
        Returns True if any sentences were committed."""
        with self._vad_lock:
            peek = self._vad.peek_buffer()
        if peek is None:
            return False
        audio, duration = peek

        # Don't bother with very short buffers
        if duration < 1.5:
            return False

        # Word timestamp alignment is expensive for repeated interim passes.
        # The proportional trim path below is less exact but keeps long runs stable.
        use_word_ts = False

        try:
            result, asr_ms = self._run_asr(
                audio, "interim", word_timestamps=use_word_ts
            ) if use_word_ts else self._run_asr(audio, "interim")
        except Exception as e:
            log.error(f"Interim ASR error: {e}", exc_info=True)
            return False

        if asr_ms == 0:
            return False

        if result is None:
            return False

        full_text = result["text"].strip()
        if not full_text or not any(c.isalnum() for c in full_text):
            return False

        # Strip echo from previous commit's overlap
        full_text = self._strip_committed_overlap(full_text)
        if not full_text:
            return False

        split_start = time.perf_counter()
        sentences = self._split_sentences(full_text, result["language"])
        split_ms = (time.perf_counter() - split_start) * 1000
        if len(sentences) <= 1:
            return False
        log.debug(f"Interim split [{result['language']}] ({split_ms:.1f}ms): {len(sentences)} parts -> {sentences}")

        # All but last are complete; last is still being spoken
        complete = sentences[:-1]

        committed_text = ""
        for sent in complete:
            committed_text += sent

        if not committed_text.strip():
            return False

        # Determine trim point
        total_samples = len(audio)
        if use_word_ts and result.get("words"):
            words = result["words"]
            committed_lower = committed_text.lower().rstrip()
            char_pos = 0
            last_word_end = 0.0
            for w in words:
                word_text = w["word"].strip()
                idx = committed_lower.find(word_text.lower(), char_pos)
                if idx >= 0:
                    char_pos = idx + len(word_text)
                    last_word_end = w["end"]
                if char_pos >= len(committed_lower):
                    break
            trim_samples = int(last_word_end * 16000)
        else:
            # Proportional trim with safety margin to reduce echo
            ratio = len(committed_text) / max(len(full_text), 1)
            margin = int(0.3 * 16000)  # 0.3s extra trim to avoid re-recognition
            trim_samples = int(ratio * total_samples) + margin
            # Don't over-trim: keep at least 0.5s for the remaining sentence
            max_trim = total_samples - int(0.5 * 16000)
            trim_samples = min(trim_samples, max(max_trim, 0))
            # Minimum trim to prevent re-recognition loops
            min_trim = int(0.3 * 16000)
            if trim_samples < min_trim and trim_samples > 0:
                trim_samples = min(min_trim, total_samples // 2)

        # Output committed sentences
        actually_committed = False
        for sent in complete:
            text = sent.strip()
            if not text:
                continue
            if self._is_short_utterance(text):
                self._interim_pending += text
                log.debug(f"Interim short utterance buffered: '{text}', pending='{self._interim_pending}'")
                continue

            if self._interim_pending:
                text = self._interim_pending + text
                self._interim_pending = ""

            self._process_segment_text(text, result["language"], asr_ms)
            actually_committed = True

        if not actually_committed:
            return False

        if trim_samples > 0:
            with self._vad_lock:
                self._vad.trim_front(trim_samples)

        # Track committed text tail for echo dedup
        self._interim_committed_tail = committed_text[-50:] if len(committed_text) > 50 else committed_text

        self._interim_active = True
        log.info(f"Interim ASR: committed {len(complete)} sentence(s), trimmed {trim_samples / 16000:.2f}s")
        return True

    def _process_segment_text(self, text: str, source_lang: str, asr_ms: float = 0):
        """Output a text result (from interim or final) — similar to _process_segment but skips ASR."""
        original_text = text.strip()
        if not original_text or not any(c.isalnum() for c in original_text):
            return

        asr_lang_setting = self._panel.get_settings().get("asr_language", "auto") if self._panel else "auto"
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', discarding: {original_text[:60]}")
            return

        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms, interim): {original_text}")

        if self._overlay:
            self._overlay.add_message(msg_id, timestamp, original_text, source_lang, asr_ms)
        self._transcript.write_original(msg_id, timestamp, original_text)

        self._last_original = original_text
        self._last_msg_id = msg_id

        target_lang = self._target_language
        extra_langs = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            extra_langs = subwin_langs - {target_lang, source_lang}

        if source_lang == target_lang:
            log.info(f"Same language ({source_lang}), no translation")
            self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(msg_id, "", 0)
                self._overlay.update_stats(self._asr_count, self._translate_count, self._total_prompt_tokens, self._total_completion_tokens, self._compute_cost())
            if self._subwin and self._subwin.isVisible():
                if extra_langs:
                    try:
                        self._tl_executor.submit(self._translate_subwin_only, original_text, source_lang, extra_langs)
                    except RuntimeError:
                        pass
                else:
                    self._subwin.update_text(original_text, {target_lang: original_text})
        else:
            try:
                self._tl_executor.submit(self._translate_async, msg_id, original_text, source_lang, extra_langs or None)
            except RuntimeError:
                log.warning("Translation executor shut down, skipping")
    def _process_interim_final(self, speech_segment):
        """Handle VAD flush after interim outputs were already made."""
        seg_len = len(speech_segment) / 16000
        log.info(f"Interim final segment: {seg_len:.1f}s")

        try:
            result, asr_ms = self._run_asr(speech_segment, "interim_final")
        except Exception as e:
            log.error(f"Interim final ASR error: {e}", exc_info=True)
            return
        if asr_ms == 0:
            return

        if result is None:
            # Flush any remaining pending
            if self._interim_pending:
                text = self._interim_pending
                self._interim_pending = ""
                lang = self._panel.get_settings().get("asr_language", "auto") if self._panel else "auto"
                if lang == "auto":
                    lang = "unknown"
                self._process_segment_text(text, lang)
            return

        original_text = result["text"].strip()

        # Strip echo from previous commit's overlap
        original_text = self._strip_committed_overlap(original_text)

        # Prepend any remaining pending short utterances
        if self._interim_pending:
            original_text = self._interim_pending + original_text
            self._interim_pending = ""

        if not original_text or not any(c.isalnum() for c in original_text):
            return

        # Apply noise filter like _process_segment
        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(f"Noise filter: {seg_len:.1f}s segment produced only '{original_text}', skipping")
            return

        self._process_segment_text(original_text, result["language"], asr_ms)

    def _capture_loop(self):
        silence_chunk = np.zeros(
            int(
                self._config["audio"]["sample_rate"]
                * self._config["audio"]["chunk_duration"]
            ),
            dtype=np.float32,
        )
        while self._running:
            item = self._audio.get_audio(timeout=1.0)
            if item is None:
                if self._vad._is_speaking and not self._paused:
                    n = self._vad._get_effective_silence_limit() + 1
                    for _ in range(n):
                        with self._vad_lock:
                            seg = self._vad.process_chunk(silence_chunk)
                            frame = self._preview.frame(seg, final=True) if seg is not None and self._live_preview else None
                        if seg is not None and self._asr_ready:
                            self._enqueue_asr("preview_final" if frame else "vad_flush", frame if frame else seg)
                            break
                continue

            chunk, mic_rms = item

            if self._paused:
                continue

            rms = float(np.sqrt(np.mean(chunk**2)))

            if self._overlay:
                self._overlay.update_monitor(rms, self._vad.last_confidence, mic_rms)

            with self._vad_lock:
                speech_segment = self._vad.process_chunk(chunk)
                if self._live_preview and self._asr_ready:
                    if speech_segment is not None:
                        frame = self._preview.frame(speech_segment, final=True)
                        self._enqueue_asr("preview_final", frame)
                        self._last_interim_check_time = 0.0
                    else:
                        peek = self._vad.peek_buffer()
                        now = time.monotonic()
                        if (peek is not None and peek[1] >= max(1.0, self._interim_interval) and
                                now - self._last_interim_check_time >= self._interim_interval):
                            self._last_interim_check_time = now
                            self._enqueue_asr("preview", self._preview.frame(peek[0]))
                    continue

            if speech_segment is None:
                # Still accumulating — check for interim ASR
                if (self._incremental_enabled and self._asr_ready
                        and self._vad._is_speaking):
                    buf_samples = self._vad._speech_samples
                    total_dur = buf_samples / 16000
                    elapsed = (buf_samples - self._last_interim_samples) / 16000
                    now = time.perf_counter()
                    cooldown = now - self._last_interim_check_time
                    if total_dur >= self._interim_interval and elapsed >= self._interim_interval and cooldown >= 1.0:
                        self._last_interim_check_time = now
                        self._enqueue_asr("interim", None)
                continue

            if not self._asr_ready:
                log.debug("ASR not ready, dropping segment")
                continue

            self._enqueue_asr("vad_flush", speech_segment)

    def _enqueue_asr(self, seg_type: str, segment):
        try:
            self._asr_queue.put_nowait((seg_type, segment))
        except queue.Full:
            try:
                dropped = self._asr_queue.get_nowait()
                log.warning(f"ASR queue full, dropped {dropped[0]} segment")
            except queue.Empty:
                pass
            try:
                self._asr_queue.put_nowait((seg_type, segment))
            except queue.Full:
                log.warning("ASR queue still full after drop, skipping segment")

    def _asr_loop(self):
        while self._running or not self._asr_queue.empty():
            try:
                item = self._asr_queue.get(timeout=1.0)
            except queue.Empty:
                # Idle moment: recycle a bloated worker while no audio is waiting.
                # Guarded so an unexpected error can never kill this thread (which
                # would itself silence ASR permanently).
                try:
                    self._maybe_recycle_asr_worker()
                except Exception:
                    log.error("ASR worker recycle check failed", exc_info=True)
                continue

            if item is None:
                break

            seg_type, segment = item

            if seg_type in ("preview", "preview_final"):
                self._process_preview_audio(segment)
                continue
            if self._live_preview:
                continue

            if seg_type == "vad_flush":
                if self._interim_active:
                    self._process_interim_final(segment)
                else:
                    self._process_segment(segment)
                self._interim_active = False
                self._interim_pending = ""
                self._last_interim_samples = 0
                self._last_interim_check_time = 0.0
                self._interim_committed_tail = ""
            elif seg_type == "interim":
                self._drain_interim_duplicates()
                self._do_interim_asr()
                with self._vad_lock:
                    self._last_interim_samples = self._vad._speech_samples

    def _drain_interim_duplicates(self):
        while True:
            try:
                item = self._asr_queue.get_nowait()
            except queue.Empty:
                break
            if item is None or item[0] != "interim":
                self._asr_queue.put(item)
                break


def main():
    setup_logging()
    log.info("LiveTranslate starting...")
    config = load_config()
    config.setdefault("asr", {})
    config["asr"].setdefault("asr_engine", "funasr")
    config["asr"].setdefault("funasr_model", DEFAULT_FUNASR_MODEL)
    saved = _load_saved_settings()
    migrate_funasr_settings(saved)

    # Log actual effective config
    _asr_eng = (saved or {}).get("asr_engine", config["asr"].get("asr_engine", "funasr"))
    _funasr_model = (saved or {}).get(
        "funasr_model", config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL)
    )
    _active_idx = (saved or {}).get("active_model", 0)
    _models = (saved or {}).get("models", [])
    if 0 <= _active_idx < len(_models):
        _m = _models[_active_idx]
        _model_info = f"{_m.get('name', '?')} ({_m.get('model', '?')})"
    else:
        _model_info = f"{config['translation']['model']} (default)"
    if _asr_eng == "funasr":
        log.info(
            f"Config loaded: ASR={_asr_eng}/{_funasr_model}, "
            f"Translator={_model_info}"
        )
    else:
        log.info(f"Config loaded: ASR={_asr_eng}, Translator={_model_info}")

    # Apply UI language before creating any widgets
    if saved and saved.get("ui_lang"):
        set_lang(saved["ui_lang"])

    os.environ["QT_LOGGING_RULES"] = (
        "qt.text.font.db=false;qt.qpa.fonts.warning=false"
    )
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    # Pin a guaranteed TrueType UI font to avoid DirectWrite failures on the
    # legacy "MS Sans Serif" bitmap font Windows may resolve as the default
    if "Segoe UI" in QFontDatabase.families():
        app.setFont(QFont("Segoe UI", 9))
    _app_icon = create_app_icon()
    app.setWindowIcon(_app_icon)

    # First launch → setup wizard (hub + download) → configure translation API
    if not SETTINGS_FILE.exists():
        wizard = SetupWizardDialog()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        saved = _load_saved_settings()
        log.info("Setup wizard completed")

        # Prompt user to configure translation API
        from dialogs import ModelEditDialog

        info = QMessageBox(
            QMessageBox.Icon.Information,
            t("window_setup"),
            t("setup_api_hint"),
        )
        info.exec()

        dlg = ModelEditDialog(None, {
            "name": "hunyuan-mt-chimera-7b",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "sk-lm-tHzDfNGm:dgxlip7eebn3HIMxivqN",
            "model": "hunyuan-mt-chimera-7b",
        })
        dlg.setWindowTitle(t("setup_api_title"))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if data.get("api_key"):
                saved["models"] = [data]
                saved["active_model"] = 0
                _save_settings(saved)
                log.info(f"Translation API configured: {data['name']}")
        # If user skips, ControlPanel will create default placeholder from config.yaml

    # Non-first launch but models missing → download dialog
    else:
        saved = saved or {}
        current_engine = saved.get("asr_engine", config["asr"].get("asr_engine", "funasr"))
        missing = get_missing_models(
            current_engine,
            (
                saved.get("funasr_model", config["asr"].get("funasr_model"))
                if current_engine == "funasr"
                else saved.get("whisper_model_size", config["asr"]["model_size"])
            ),
            saved.get("hub", "ms"),
        )
        if missing:
            log.info(f"Missing models: {[m['name'] for m in missing]}")
            dlg = ModelDownloadDialog(
                missing,
                hub=saved.get("hub", "ms"),
                proxy=saved.get("download_proxy", "system"),
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                sys.exit(0)

    log_window = LogWindow()
    log_handler = log_window.get_handler()
    logging.getLogger().addHandler(log_handler)

    panel = ControlPanel(config, saved_settings=saved)

    overlay = SubtitleOverlay(config["subtitle"])
    if saved:
        ox = saved.get("overlay_x")
        oy = saved.get("overlay_y")
        ow = saved.get("overlay_w")
        oh = saved.get("overlay_h")
        if ox is not None and oy is not None:
            if SubtitleWindow._is_pos_visible(ox, oy):
                overlay.move(ox, oy)
            else:
                screen = QApplication.primaryScreen()
                geo = screen.availableGeometry()
                overlay.move(geo.right() - overlay.width() - 20, geo.bottom() - overlay.height() - 60)
        if ow and oh:
            overlay.resize(ow, oh)
    overlay.show()

    # Subtitle window
    subwin_cfg = (saved or {}).get("subtitle_mode")
    subwin = SubtitleWindow(subwin_cfg)
    subwin_was_enabled = (subwin_cfg or {}).get("enabled", False)

    live_trans = LiveTranslateApp(config)
    live_trans.set_overlay(overlay)
    live_trans.set_subtitle_window(subwin)
    live_trans.set_panel(panel)

    def _deferred_init():
        panel._apply_settings()
        models = panel.get_settings().get("models", [])
        active_idx = panel.get_settings().get("active_model", 0)
        overlay.set_models(models, active_idx)
        target = panel.get_settings().get("target_language", "zh")
        overlay.set_target_language(target)
        asr_lang = panel.get_settings().get("asr_language", "auto")
        overlay.set_source_language(asr_lang)
        style = panel.get_settings().get("style")
        if style:
            overlay.apply_style(style)
        active_model = panel.get_active_model()
        if active_model:
            live_trans._on_model_changed(active_model)

    QTimer.singleShot(100, _deferred_init)

    tray = QSystemTrayIcon()
    tray.setToolTip(t("tray_tooltip"))
    tray.setIcon(_app_icon)

    menu = QMenu()

    # --- Pause / Resume toggle ---
    pause_action = QAction(t("tray_pause"))
    _is_running = [True]  # mutable for closure

    def on_start():
        try:
            live_trans.start()
            overlay.set_running(True)
            _is_running[0] = True
            pause_action.setText(t("tray_pause"))
        except Exception as e:
            log.error(f"Start error: {e}", exc_info=True)

    def on_pause():
        live_trans.pause()
        overlay.set_running(False)
        _is_running[0] = False
        pause_action.setText(t("tray_resume"))

    def on_resume():
        live_trans.resume()
        overlay.set_running(True)
        _is_running[0] = True
        pause_action.setText(t("tray_pause"))

    def on_toggle_pause():
        if _is_running[0]:
            on_pause()
        else:
            on_resume()

    pause_action.triggered.connect(on_toggle_pause)
    menu.addAction(pause_action)
    menu.addSeparator()

    # --- Show/hide overlay ---
    overlay_toggle_action = QAction(t("tray_hide_overlay"))

    _hide_notified = [False]

    def on_toggle_overlay():
        if overlay.isVisible():
            overlay.hide()
            overlay_toggle_action.setText(t("tray_show_overlay"))
            if not _hide_notified[0]:
                _hide_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("hide_tray_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            overlay.show()
            overlay.raise_()
            overlay_toggle_action.setText(t("tray_hide_overlay"))

    overlay_toggle_action.triggered.connect(on_toggle_overlay)
    menu.addAction(overlay_toggle_action)

    # --- Subtitle window toggle ---
    def _save_overlay_pos():
        settings = panel.get_settings()
        pos = overlay.pos()
        size = overlay.size()
        settings["overlay_x"] = pos.x()
        settings["overlay_y"] = pos.y()
        settings["overlay_w"] = size.width()
        settings["overlay_h"] = size.height()
        panel._current_settings.update({
            "overlay_x": pos.x(), "overlay_y": pos.y(),
            "overlay_w": size.width(), "overlay_h": size.height(),
        })
        _save_settings(settings)

    overlay.position_changed.connect(_save_overlay_pos)

    subwin_toggle_action = QAction(t("subwin_show"), checkable=True)

    def _save_subwin_state():
        settings = panel.get_settings()
        sm = settings.get("subtitle_mode") or {}
        sm["enabled"] = subwin.isVisible()
        pos = subwin.pos()
        sm["window_x"] = pos.x()
        sm["window_y"] = pos.y()
        settings["subtitle_mode"] = sm
        panel._current_settings["subtitle_mode"] = sm
        _save_settings(settings)

    _subwin_notified = [False]

    def on_toggle_subwin(checked):
        if checked:
            subwin.show()
            subwin.raise_()
            if not _subwin_notified[0]:
                _subwin_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("subwin_drag_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            subwin.hide()
        overlay.set_subtitle_checked(checked)
        _save_subwin_state()

    subwin_toggle_action.toggled.connect(on_toggle_subwin)
    subwin.position_changed.connect(_save_subwin_state)

    # Sync when subtitle window is manually closed (e.g. Alt+F4)
    def _on_subwin_closed():
        subwin_toggle_action.blockSignals(True)
        subwin_toggle_action.setChecked(False)
        subwin_toggle_action.blockSignals(False)
        overlay.set_subtitle_checked(False)
        _save_subwin_state()

    subwin.window_closed.connect(_on_subwin_closed)

    # Restore subtitle window visibility from saved state
    if subwin_was_enabled:
        subwin_toggle_action.setChecked(True)

    menu.addAction(subwin_toggle_action)

    # Quick toggle for subtitle-window click-through (mirrors the settings checkbox).
    subwin_ct_action = QAction(t("subwin_click_through_tray"), checkable=True)
    _subwin_init = panel.get_settings().get("subtitle_mode") or {}
    subwin_ct_action.setChecked(bool(_subwin_init.get("click_through", False)))

    def on_toggle_subwin_ct(checked):
        subwin.set_click_through(checked)
        settings = panel.get_settings()
        sm = settings.get("subtitle_mode") or {}
        sm["click_through"] = checked
        settings["subtitle_mode"] = sm
        panel._current_settings["subtitle_mode"] = sm
        _save_settings(settings)
        w = panel._subtitle_widget
        w._click_through_check.blockSignals(True)
        w._click_through_check.setChecked(checked)
        w._click_through_check.blockSignals(False)
        w._settings["click_through"] = checked

    subwin_ct_action.toggled.connect(on_toggle_subwin_ct)
    menu.addAction(subwin_ct_action)

    # Connect overlay subtitle button
    def _on_overlay_subtitle_toggle():
        subwin_toggle_action.setChecked(not subwin_toggle_action.isChecked())

    overlay.subtitle_toggled.connect(_on_overlay_subtitle_toggle)

    # Connect panel subtitle settings changes
    def _on_panel_subtitle_changed(s):
        subwin.apply_settings(s)
        subwin_ct_action.blockSignals(True)
        subwin_ct_action.setChecked(bool(s.get("click_through", False)))
        subwin_ct_action.blockSignals(False)

    panel.subtitle_settings_changed.connect(_on_panel_subtitle_changed)

    def _on_reset_positions():
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        subwin.move(100, 100)
        _save_subwin_state()
        ow, oh = overlay.width(), overlay.height()
        overlay.move(geo.right() - ow - 50, geo.bottom() - oh - 100)
        _save_overlay_pos()

    panel.reset_positions.connect(_on_reset_positions)

    menu.addSeparator()

    # --- Show log / panel ---
    log_action = QAction(t("tray_show_log"))
    panel_action = QAction(t("tray_show_panel"))

    def on_toggle_log():
        if log_window.isVisible():
            log_window.hide()
        else:
            log_window.show()
            log_window.raise_()

    def on_toggle_panel():
        if panel.isVisible():
            panel.hide()
        else:
            panel.show()
            panel.raise_()

    log_action.triggered.connect(on_toggle_log)
    panel_action.triggered.connect(on_toggle_panel)
    menu.addAction(panel_action)
    menu.addAction(log_action)
    menu.addSeparator()

    # --- Overlay submenu (click-through, topmost, auto-scroll, taskbar) ---
    overlay_menu = QMenu(t("tray_menu_overlay"))

    ct_action = QAction(t("click_through"), checkable=True)
    topmost_action = QAction(t("top_most"), checkable=True)
    topmost_action.setChecked(True)
    autoscroll_action = QAction(t("auto_scroll"), checkable=True)
    autoscroll_action.setChecked(True)
    taskbar_action = QAction(t("taskbar"), checkable=True)

    # Tray → overlay sync
    ct_action.toggled.connect(lambda v: overlay._handle._ct_check.setChecked(v))
    topmost_action.toggled.connect(
        lambda v: overlay._handle._topmost_check.setChecked(v)
    )
    autoscroll_action.toggled.connect(
        lambda v: overlay._handle._auto_scroll.setChecked(v)
    )
    taskbar_action.toggled.connect(
        lambda v: overlay._handle._taskbar_check.setChecked(v)
    )

    # Overlay → tray sync
    overlay._handle.click_through_toggled.connect(lambda v: ct_action.setChecked(v))
    overlay._handle.topmost_toggled.connect(lambda v: topmost_action.setChecked(v))
    overlay._handle.auto_scroll_toggled.connect(
        lambda v: autoscroll_action.setChecked(v)
    )
    overlay._handle.taskbar_toggled.connect(lambda v: taskbar_action.setChecked(v))

    overlay_menu.addAction(ct_action)
    overlay_menu.addAction(topmost_action)
    overlay_menu.addAction(autoscroll_action)
    overlay_menu.addAction(taskbar_action)
    menu.addMenu(overlay_menu)

    # --- Model submenu ---
    model_menu = QMenu(t("tray_menu_model"))
    model_action_group = QActionGroup(model_menu)
    model_action_group.setExclusive(True)

    def _rebuild_model_menu():
        for a in model_action_group.actions():
            model_action_group.removeAction(a)
        model_menu.clear()
        settings = panel.get_settings()
        models = settings.get("models", [])
        active = settings.get("active_model", 0)
        for i, m in enumerate(models):
            name = m.get("name", m.get("model", "?"))
            action = QAction(name, checkable=True)
            if i == active:
                action.setChecked(True)
            model_action_group.addAction(action)
            action.triggered.connect(lambda checked, idx=i: _on_tray_model_switch(idx))
            model_menu.addAction(action)

    def _on_tray_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            panel._current_settings["active_model"] = index
            _save_settings(settings)
            panel._refresh_model_list()
            live_trans._on_model_changed(models[index])
            overlay.set_models(models, index)

    def on_overlay_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            panel._current_settings["active_model"] = index
            _save_settings(settings)
            panel._refresh_model_list()
            live_trans._on_model_changed(models[index])
        _rebuild_model_menu()

    model_menu.aboutToShow.connect(_rebuild_model_menu)
    menu.addMenu(model_menu)

    # --- Target language submenu ---
    lang_menu = QMenu(t("tray_menu_target_lang"))
    lang_action_group = QActionGroup(lang_menu)
    lang_action_group.setExclusive(True)
    _lang_actions = {}
    lang_more_menu = QMenu(t("tray_more_langs"))

    for code, native in LANGUAGES:
        if code == "auto":
            continue
        action = QAction(f"{code} - {native}", checkable=True)
        lang_action_group.addAction(action)
        action.triggered.connect(lambda checked, lc=code: _on_tray_lang_switch(lc))
        if code in COMMON_LANG_CODES:
            lang_menu.addAction(action)
        else:
            lang_more_menu.addAction(action)
        _lang_actions[code] = action

    lang_menu.addMenu(lang_more_menu)

    current_target = panel.get_settings().get("target_language", "zh")
    if current_target in _lang_actions:
        _lang_actions[current_target].setChecked(True)

    def _on_tray_lang_switch(lang_code):
        overlay.set_target_language(lang_code)
        live_trans._on_target_language_changed(lang_code)
        from control_panel import _save_settings

        settings = panel.get_settings()
        settings["target_language"] = lang_code
        panel._current_settings["target_language"] = lang_code
        _save_settings(settings)

    # Overlay → tray lang sync
    def _on_overlay_lang_changed(lang_code):
        if lang_code in _lang_actions:
            _lang_actions[lang_code].setChecked(True)

    overlay.target_language_changed.connect(_on_overlay_lang_changed)

    menu.addMenu(lang_menu)

    # --- ASR language hint submenu ---
    asr_lang_menu = QMenu(t("tray_menu_asr_lang"))
    asr_lang_action_group = QActionGroup(asr_lang_menu)
    asr_lang_action_group.setExclusive(True)
    _asr_lang_actions = {}
    asr_more_menu = QMenu(t("tray_more_langs"))

    for code, native in LANGUAGES:
        label = t("asr_lang_auto") if code == "auto" else native
        action = QAction(f"{code} - {label}", checkable=True)
        asr_lang_action_group.addAction(action)
        action.triggered.connect(lambda checked, c=code: _on_tray_asr_lang(c))
        if code in COMMON_LANG_CODES:
            asr_lang_menu.addAction(action)
        else:
            asr_more_menu.addAction(action)
        _asr_lang_actions[code] = action

    asr_lang_menu.addMenu(asr_more_menu)

    current_asr_lang = panel.get_settings().get("asr_language", "auto")
    if current_asr_lang in _asr_lang_actions:
        _asr_lang_actions[current_asr_lang].setChecked(True)

    def _on_tray_asr_lang(code):
        from control_panel import _save_settings

        live_trans._set_asr_language(code)
        settings = panel.get_settings()
        settings["asr_language"] = code
        panel._current_settings["asr_language"] = code
        _save_settings(settings)
        # Sync control panel combo
        idx = panel._asr_lang.findData(code)
        if idx >= 0:
            panel._asr_lang.blockSignals(True)
            panel._asr_lang.setCurrentIndex(idx)
            panel._asr_lang.blockSignals(False)

    menu.addMenu(asr_lang_menu)
    menu.addSeparator()

    # --- Export submenu ---
    export_menu = QMenu(t("export_menu"))
    export_orig_action = QAction(t("export_original"))
    export_trans_action = QAction(t("export_translation"))
    export_all_action = QAction(t("export_all"))
    export_orig_action.triggered.connect(lambda: overlay.export_messages("original", parent=panel))
    export_trans_action.triggered.connect(lambda: overlay.export_messages("translation", parent=panel))
    export_all_action.triggered.connect(lambda: overlay.export_messages("both", parent=panel))
    export_menu.addAction(export_orig_action)
    export_menu.addAction(export_trans_action)
    export_menu.addAction(export_all_action)
    menu.addMenu(export_menu)
    menu.addSeparator()

    # --- Quit ---
    quit_action = QAction(t("quit"))

    def on_quit():
        live_trans.stop()
        app.quit()

    quit_action.triggered.connect(on_quit)
    menu.addAction(quit_action)

    # --- Connect overlay signals ---
    overlay.settings_requested.connect(on_toggle_panel)
    overlay.target_language_changed.connect(live_trans._on_target_language_changed)

    def _on_overlay_source_lang(code):
        """Overlay source language combo → sync to panel + ASR engine + tray."""
        _on_tray_asr_lang(code)
        overlay.set_source_language(code)

    def _on_panel_asr_lang_changed(_index):
        """Panel ASR language combo → sync to overlay."""
        code = panel._asr_lang.currentData() or "auto"
        overlay.set_source_language(code)

    overlay.source_language_changed.connect(_on_overlay_source_lang)
    panel._asr_lang.currentIndexChanged.connect(_on_panel_asr_lang_changed)
    overlay.model_switch_requested.connect(on_overlay_model_switch)
    overlay.start_requested.connect(on_resume)
    overlay.stop_requested.connect(on_pause)
    overlay.hide_requested.connect(on_toggle_overlay)
    overlay.quit_requested.connect(on_quit)

    tray.setContextMenu(menu)
    tray.show()

    def _on_memory_warning(rss_mb: float):
        tray.showMessage(
            "LiveTranslate",
            t("mem_warning_msg").format(rss=int(rss_mb)),
            QSystemTrayIcon.MessageIcon.Warning,
            10000,
        )

    live_trans.set_memory_warning_callback(_on_memory_warning)

    QTimer.singleShot(500, on_start)

    signal.signal(signal.SIGINT, lambda *_: on_quit())
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing as _multiprocessing

    _multiprocessing.freeze_support()
    main()
