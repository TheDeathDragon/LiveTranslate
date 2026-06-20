"""
LiveTranslate - Phase 0 Prototype
Real-time audio translation using WASAPI loopback + faster-whisper + LLM.
"""

import sys
import signal
import logging
import threading
import gc
from concurrent.futures import ThreadPoolExecutor
import yaml
import time
from pathlib import Path
from datetime import datetime

from model_manager import (
    DEFAULT_FUNASR_MODEL,
    apply_cache_env,
    get_missing_models,
    migrate_funasr_settings,
    normalize_asr_engine_selection,
)

# Set cache env BEFORE importing torch so TORCH_HOME is respected
apply_cache_env()

import os

# torch must be imported before PyQt6 to avoid DLL conflicts on Windows
import torch  # noqa: F401

from asr_service import ASRService
from pipeline_controller import PipelineController
from translator import Translator, RepetitionError
from transcript_writer import TranscriptWriter

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
from control_panel import ControlPanel
from settings_store import SETTINGS_FILE, load_settings, save_settings
from dialogs import (
    SetupWizardDialog,
    ModelDownloadDialog,
    _ModelLoadDialog,
)
from i18n import t, set_lang, LANGUAGES, COMMON_LANG_CODES


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

        self._asr_service = ASRService(
            config,
            release_memory_caches=self._release_memory_caches,
            unavailable_callback=self._on_asr_unavailable,
        )
        self._pipeline = PipelineController(
            config,
            asr_runner=self._run_asr,
            asr_ready=lambda: self._asr_service.is_ready,
            asr_language=self._current_asr_language,
            audio_level_callback=self._on_audio_level,
            asr_text_callback=self._on_asr_text,
        )
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

    def set_overlay(self, overlay: SubtitleOverlay):
        self._overlay = overlay

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

    def _current_asr_language(self) -> str:
        if self._panel:
            return self._panel.get_settings().get("asr_language", "auto")
        return "auto"

    def _on_audio_level(self, event):
        if self._overlay:
            self._overlay.update_monitor(
                event.rms, event.vad_confidence, event.mic_rms
            )

    def _on_asr_text(self, event):
        self._handle_asr_text(event.text, event.source_lang, event.asr_ms)

    def _on_settings_changed(self, settings):
        self._pipeline.apply_settings(settings)
        if "style" in settings and self._overlay:
            self._overlay.apply_style(settings["style"])
        if "asr_language" in settings:
            self._asr_service.set_language(settings["asr_language"])
        if "sensevoice_pad_seconds" in settings:
            self._asr_service.set_padding("funasr", settings["sensevoice_pad_seconds"])
        if "whisper_pad_seconds" in settings:
            self._asr_service.set_padding("whisper", settings["whisper_pad_seconds"])
        if any(
            key in settings
            for key in (
                "asr_engine",
                "asr_device",
                "whisper_model_size",
                "funasr_model",
                "crispasr_model",
                "crispasr_backend",
                "crispasr_gpu_backend",
                "crispasr_device_index",
                "crispasr_punc_model",
                "crispasr_unified_memory",
                "sherpa_onnx_model",
                "sherpa_onnx_provider",
                "sherpa_onnx_num_threads",
                "sherpa_onnx_decoding_method",
                "parakeet_cpp_model",
                "parakeet_cpp_runtime_dir",
                "parakeet_cpp_backend",
                "parakeet_cpp_decoder",
                "parakeet_cpp_word_timestamps",
                "remote_asr_url",
                "hub",
            )
        ):
            self._switch_asr_engine(
                settings.get(
                    "asr_engine",
                    self._asr_service.current_engine_type
                    or self._config["asr"].get("asr_engine", "funasr"),
                )
            )
        if "target_language" in settings:
            self._target_language = settings["target_language"]
            if self._overlay:
                self._overlay.set_target_language(self._target_language)
        if "timeout" in settings and self._translator:
            self._translator.set_timeout(settings["timeout"])
        if "auto_save_transcript" in settings:
            self._transcript.set_enabled(settings["auto_save_transcript"])

    def _on_asr_unavailable(self, reason: str):
        if self._overlay:
            self._overlay.update_asr_device("ASR unavailable")

    def _on_target_language_changed(self, lang: str):
        self._target_language = lang
        log.info(f"Target language: {lang}")
        if self._translator:
            self._translator.set_target_language(lang)
        if self._panel:
            self._panel.set_target_language(lang)

    def _on_model_changed(self, model_config: dict):
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
        plan = self._asr_service.prepare_switch(engine_type, settings)
        if plan.already_current:
            if plan.error:
                parent = (
                    self._panel
                    if self._panel and self._panel.isVisible()
                    else self._overlay
                )
                QMessageBox.warning(parent, t("error_title"), plan.error)
            return

        # Reset interim state for the engine boundary. The active worker is
        # stopped before the target worker starts loading.
        self._pipeline.reset_for_asr_switch()

        parent = (
            self._panel if self._panel and self._panel.isVisible() else self._overlay
        )

        if plan.missing_models:
            dlg = ModelDownloadDialog(
                plan.missing_models,
                hub=plan.hub,
                proxy=plan.download_proxy,
                parent=parent,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                log.info(f"Download cancelled/failed: {plan.engine_type}")
                self._asr_service.mark_download_cancelled()
                return

        dlg = _ModelLoadDialog(
            t("loading_model").format(name=plan.display_name), parent=parent
        )

        switch_result = [None]

        def _load():
            switch_result[0] = self._asr_service.switch_worker(plan)

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

        result = switch_result[0]
        if result is None:
            return

        if result.status == "ready":
            target_state = result.target_state or plan.target_state or {}
            device_label = target_state.get("device_label", plan.device)
            if self._overlay:
                self._overlay.update_asr_device(
                    f"{plan.display_name} [{device_label}]"
                )
            return

        if result.status == "restored":
            old_state = result.restored_state or {}
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
                        f"{result.load_error or 'unknown error'}\n"
                        f"{t('asr_restore_succeeded')}"
                    )
                ),
            )
            return

        error = result.load_error or "unknown error"
        if result.restore_error:
            error = (
                f"{error}\n"
                f"{t('asr_restore_failed').format(error=result.restore_error)}"
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
        worker_pid = self._asr_service.worker_pid
        if worker_pid is not None:
            try:
                import psutil

                worker_rss_mb = (
                    psutil.Process(worker_pid).memory_info().rss / 1024 / 1024
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
        vad_buf = self._pipeline.buffer_stats()["chunks"]
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

    def _run_asr(self, audio, kind: str, **kwargs):
        audio_seconds = len(audio) / self._config["audio"]["sample_rate"]
        asr_start = time.perf_counter()
        if not self._asr_service.is_ready:
            return None, 0.0
        try:
            result = self._asr_service.transcribe(audio, **kwargs)
        except Exception:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._log_mem_after_asr(f"{kind}:error", audio_seconds, asr_ms)
            raise
        if result is None:
            return None, 0.0
        asr_ms = (time.perf_counter() - asr_start) * 1000
        self._log_mem_after_asr(kind, audio_seconds, asr_ms)
        return result, asr_ms

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
        self._pipeline.start()
        self._running = True
        self._paused = False
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

    def stop(self):
        self._running = False
        self._pipeline.stop()
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
        self._asr_service.shutdown()

    def pause(self):
        self._paused = True
        self._pipeline.pause()

    def resume(self):
        self._paused = False
        self._pipeline.resume()

    def _handle_asr_text(self, original_text: str, source_lang: str, asr_ms: float = 0):
        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")

        if self._overlay:
            self._overlay.add_message(
                msg_id, timestamp, original_text, source_lang, asr_ms
            )
        self._transcript.write_original(msg_id, timestamp, original_text)

        # Store for subtitle window; translation will be added later.
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
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    self._compute_cost(),
                )
            if self._subwin and self._subwin.isVisible():
                if extra_langs:
                    try:
                        self._tl_executor.submit(
                            self._translate_subwin_only,
                            original_text,
                            source_lang,
                            extra_langs,
                        )
                    except RuntimeError:
                        pass
                else:
                    self._subwin.update_text(
                        original_text, {target_lang: original_text}
                    )
        else:
            try:
                self._tl_executor.submit(
                    self._translate_async,
                    msg_id,
                    original_text,
                    source_lang,
                    extra_langs or None,
                )
            except RuntimeError:
                log.warning("Translation executor shut down, skipping")

def main():
    setup_logging()
    log.info("LiveTranslate starting...")
    config = load_config()
    config.setdefault("asr", {})
    config["asr"].setdefault("asr_engine", "funasr")
    config["asr"].setdefault("funasr_model", DEFAULT_FUNASR_MODEL)
    config["asr"].setdefault("sherpa_onnx_model", "")
    config["asr"].setdefault("sherpa_onnx_provider", "auto")
    config["asr"].setdefault("sherpa_onnx_num_threads", 2)
    config["asr"].setdefault("sherpa_onnx_decoding_method", "greedy_search")
    config["asr"].setdefault("parakeet_cpp_model", "")
    config["asr"].setdefault("parakeet_cpp_runtime_dir", "")
    config["asr"].setdefault("parakeet_cpp_backend", "auto")
    config["asr"].setdefault("parakeet_cpp_decoder", "auto")
    config["asr"].setdefault("parakeet_cpp_word_timestamps", True)
    saved = load_settings(config)
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
        saved = load_settings(config)
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
            "api_key": "",
            "model": "hunyuan-mt-chimera-7b",
        })
        dlg.setWindowTitle(t("setup_api_title"))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if data.get("api_key"):
                saved["models"] = [data]
                saved["active_model"] = 0
                save_settings(saved)
                log.info(f"Translation API configured: {data['name']}")
        # If user skips, ControlPanel will create default placeholder from config.yaml

    # Non-first launch but models missing → download dialog
    else:
        saved = saved or {}
        current_engine = saved.get("asr_engine", config["asr"].get("asr_engine", "funasr"))
        current_engine, current_funasr_model = normalize_asr_engine_selection(
            current_engine, saved.get("funasr_model", config["asr"].get("funasr_model"))
        )
        if current_engine == "funasr":
            startup_model_key = current_funasr_model
        elif current_engine == "crispasr":
            startup_model_key = saved.get(
                "crispasr_model",
                config["asr"].get("crispasr_model", ""),
            )
        elif current_engine == "sherpa-onnx":
            startup_model_key = saved.get(
                "sherpa_onnx_model",
                config["asr"].get("sherpa_onnx_model", ""),
            )
        elif current_engine == "parakeet-cpp":
            startup_model_key = saved.get(
                "parakeet_cpp_model",
                config["asr"].get("parakeet_cpp_model", ""),
            )
        else:
            startup_model_key = saved.get("whisper_model_size", config["asr"]["model_size"])
        missing = get_missing_models(
            current_engine,
            startup_model_key,
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
        pos = overlay.pos()
        size = overlay.size()
        panel.update_settings({
            "overlay_x": pos.x(),
            "overlay_y": pos.y(),
            "overlay_w": size.width(),
            "overlay_h": size.height(),
        })

    overlay.position_changed.connect(_save_overlay_pos)

    subwin_toggle_action = QAction(t("subwin_show"), checkable=True)

    def _save_subwin_state():
        pos = subwin.pos()
        panel.update_subtitle_mode({
            "enabled": subwin.isVisible(),
            "window_x": pos.x(),
            "window_y": pos.y(),
        })

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
        save_settings(settings)
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
    ct_action.toggled.connect(overlay.set_click_through_checked)
    topmost_action.toggled.connect(overlay.set_topmost_checked)
    autoscroll_action.toggled.connect(overlay.set_auto_scroll_checked)
    taskbar_action.toggled.connect(overlay.set_taskbar_checked)

    # Overlay → tray sync
    overlay.click_through_toggled.connect(lambda v: ct_action.setChecked(v))
    overlay.topmost_toggled.connect(lambda v: topmost_action.setChecked(v))
    overlay.auto_scroll_toggled.connect(lambda v: autoscroll_action.setChecked(v))
    overlay.taskbar_toggled.connect(lambda v: taskbar_action.setChecked(v))

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
            panel.set_active_model(index)

    def on_overlay_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            panel.set_active_model(index)
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
        live_trans._asr_service.set_language(code)
        panel.set_asr_language(code)
        overlay.set_source_language(code)
        if code in _asr_lang_actions:
            _asr_lang_actions[code].setChecked(True)

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

    def _on_panel_asr_lang_changed(code):
        """Panel ASR language combo → sync to runtime, overlay, and tray."""
        live_trans._asr_service.set_language(code)
        overlay.set_source_language(code)
        if code in _asr_lang_actions:
            _asr_lang_actions[code].setChecked(True)

    overlay.source_language_changed.connect(_on_overlay_source_lang)
    panel.asr_language_changed.connect(_on_panel_asr_lang_changed)
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
