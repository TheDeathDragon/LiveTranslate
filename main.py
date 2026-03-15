"""
LiveTrans - Phase 0 Prototype
Real-time audio translation using WASAPI loopback + faster-whisper + LLM.
"""

import sys
import signal
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from model_manager import (
    apply_cache_env,
    get_missing_models,
    is_asr_cached,
    ASR_DISPLAY_NAMES,
    MODELS_DIR,
)

# Set cache env BEFORE importing torch so TORCH_HOME is respected
apply_cache_env()

# torch must be imported before PyQt6 to avoid DLL conflicts on Windows
import torch  # noqa: F401

from audio_capture import AudioCapture
from vad_processor import VADProcessor
from asr_engine import ASREngine
from translator import Translator

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QDialog, QMessageBox
from PyQt6.QtGui import QAction, QIcon, QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import QTimer, Qt

from subtitle_overlay import SubtitleOverlay
from log_window import LogWindow
from control_panel import (
    ControlPanel,
    SETTINGS_FILE,
    _load_saved_settings,
)
from dialogs import (
    SetupWizardDialog,
    ModelDownloadDialog,
    _ModelLoadDialog,
)
from i18n import t, set_lang


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

    for noisy in ("httpcore", "httpx", "openai", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.info(f"Log file: {log_file}")
    return logging.getLogger("LiveTrans")


log = logging.getLogger("LiveTrans")


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


class LiveTransApp:
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
        self._asr_device = config["asr"]["device"]
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
        self._overlay = None
        self._panel = None
        self._pipeline_thread = None
        self._tl_executor = ThreadPoolExecutor(max_workers=2)

        self._asr_count = 0
        self._translate_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._msg_id = 0

    def set_overlay(self, overlay: SubtitleOverlay):
        self._overlay = overlay

    def set_panel(self, panel: ControlPanel):
        self._panel = panel
        panel.settings_changed.connect(self._on_settings_changed)
        panel.model_changed.connect(self._on_model_changed)

    def _on_settings_changed(self, settings):
        self._vad.update_settings(settings)
        if "asr_language" in settings and self._asr:
            self._asr.set_language(settings["asr_language"])
        # ASR compute device change forces engine reload
        new_device = settings.get("asr_device")
        if new_device and new_device != self._asr_device:
            self._asr_device = new_device
            self._asr_type = None  # force reload
        if "asr_engine" in settings:
            self._switch_asr_engine(settings["asr_engine"])
        if "audio_device" in settings:
            self._audio.set_device(settings["audio_device"])
        if "target_language" in settings:
            self._target_language = settings["target_language"]
            if self._overlay:
                self._overlay.set_target_language(self._target_language)

    def _on_target_language_changed(self, lang: str):
        self._target_language = lang
        log.info(f"Target language: {lang}")
        if self._panel:
            settings = self._panel.get_settings()
            settings["target_language"] = lang
            from control_panel import _save_settings

            _save_settings(settings)
        active = self._panel.get_active_model() if self._panel else None
        if active:
            self._on_model_changed(active)

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
            streaming=self._config["translation"]["streaming"],
            system_prompt=prompt,
            proxy=model_config.get("proxy", "none"),
            no_system_role=model_config.get("no_system_role", False),
            timeout=timeout,
        )

    def _switch_asr_engine(self, engine_type: str):
        if engine_type == self._asr_type:
            return
        log.info(f"Switching ASR engine: {self._asr_type} -> {engine_type}")
        self._asr_ready = False
        device = self._asr_device
        hub = "ms"
        if self._panel:
            hub = self._panel.get_settings().get("hub", "ms")

        model_size = self._config["asr"]["model_size"]
        cached = is_asr_cached(engine_type, model_size, hub)
        display_name = ASR_DISPLAY_NAMES.get(engine_type, engine_type)
        if engine_type == "whisper":
            display_name = f"Whisper {model_size}"

        parent = (
            self._panel if self._panel and self._panel.isVisible() else self._overlay
        )

        if not cached:
            missing = get_missing_models(engine_type, model_size, hub)
            missing = [m for m in missing if m["type"] != "silero-vad"]
            if missing:
                dlg = ModelDownloadDialog(missing, hub=hub, parent=parent)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    log.info(f"Download cancelled/failed: {engine_type}")
                    return

        dlg = _ModelLoadDialog(
            t("loading_model").format(name=display_name), parent=parent
        )

        new_asr = [None]
        load_error = [None]

        def _load():
            try:
                if engine_type == "sensevoice":
                    from asr_sensevoice import SenseVoiceEngine

                    new_asr[0] = SenseVoiceEngine(device=device, hub=hub)
                elif engine_type in ("funasr-nano", "funasr-mlt-nano"):
                    from asr_funasr_nano import FunASRNanoEngine

                    new_asr[0] = FunASRNanoEngine(
                        device=device, hub=hub, engine_type=engine_type
                    )
                else:
                    download_root = str((MODELS_DIR / "huggingface" / "hub").resolve())
                    new_asr[0] = ASREngine(
                        model_size=model_size,
                        device=device,
                        compute_type=self._config["asr"]["compute_type"],
                        language=self._config["asr"]["language"],
                        download_root=download_root,
                    )
            except Exception as e:
                load_error[0] = str(e)
                log.error(f"Failed to load ASR engine: {e}", exc_info=True)

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

        if load_error[0]:
            QMessageBox.warning(
                parent, t("error_title"), t("error_load_asr").format(error=load_error[0])
            )
            return

        self._asr = new_asr[0]
        self._asr_type = engine_type
        if self._panel:
            asr_lang = self._panel.get_settings().get("asr_language", "auto")
            self._asr.set_language(asr_lang)
        self._asr_ready = True
        if self._overlay:
            self._overlay.update_asr_device(f"{display_name} [{device}]")
        log.info(f"ASR engine ready: {engine_type} on {device}")

    def _translate_async(self, msg_id, text, source_lang):
        try:
            tl_start = time.perf_counter()
            translated = self._translator.translate(text, source_lang)
            tl_ms = (time.perf_counter() - tl_start) * 1000
            self._translate_count += 1
            pt, ct = self._translator.last_usage
            self._total_prompt_tokens += pt
            self._total_completion_tokens += ct
            log.info(f"Translate ({tl_ms:.0f}ms): {translated}")
            if self._overlay:
                self._overlay.update_translation(msg_id, translated, tl_ms)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                )
        except Exception as e:
            log.error(f"Translate error: {e}", exc_info=True)
            if self._overlay:
                self._overlay.update_translation(msg_id, f"[error: {e}]", 0)

    def start(self):
        if self._running:
            return
        self._running = True
        self._paused = False
        self._audio.start()
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop, daemon=True
        )
        self._pipeline_thread.start()
        log.info("Pipeline started")

    def stop(self):
        self._running = False
        self._audio.stop()
        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=3)
        self._tl_executor.shutdown(wait=False)
        log.info("Pipeline stopped")

    def pause(self):
        self._paused = True
        log.info("Pipeline paused")

    def resume(self):
        self._paused = False
        log.info("Pipeline resumed")

    def _pipeline_loop(self):
        while self._running:
            chunk = self._audio.get_audio(timeout=1.0)
            if chunk is None:
                continue

            rms = float(np.sqrt(np.mean(chunk**2)))

            if self._overlay:
                self._overlay.update_monitor(rms, self._vad.last_confidence)

            if self._paused:
                continue

            speech_segment = self._vad.process_chunk(chunk)
            if speech_segment is None:
                continue

            if not self._asr_ready:
                log.debug("ASR not ready, dropping segment")
                continue

            seg_len = len(speech_segment) / 16000
            log.info(f"Speech segment: {seg_len:.1f}s")

            asr_start = time.perf_counter()
            try:
                result = self._asr.transcribe(speech_segment)
            except Exception as e:
                log.error(f"ASR error: {e}", exc_info=True)
                continue
            asr_ms = (time.perf_counter() - asr_start) * 1000
            if asr_ms > 10000:
                log.warning(f"ASR took {asr_ms:.0f}ms, possible hang")
            if result is None:
                continue

            self._asr_count += 1
            self._msg_id += 1
            msg_id = self._msg_id
            original_text = result["text"]
            source_lang = result["language"]
            timestamp = datetime.now().strftime("%H:%M:%S")
            log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms): {original_text}")

            if self._overlay:
                self._overlay.add_message(
                    msg_id, timestamp, original_text, source_lang, asr_ms
                )

            target_lang = self._target_language
            if source_lang == target_lang:
                log.info(f"Same language ({source_lang}), no translation")
                if self._overlay:
                    self._overlay.update_translation(msg_id, "", 0)
            else:
                self._tl_executor.submit(
                    self._translate_async, msg_id, original_text, source_lang
                )


def main():
    setup_logging()
    log.info("LiveTrans starting...")
    config = load_config()
    log.info(
        f"Config loaded: ASR={config['asr']['model_size']}, "
        f"API={config['translation']['api_base']}, "
        f"Model={config['translation']['model']}"
    )

    saved = _load_saved_settings()

    # Apply UI language before creating any widgets
    if saved and saved.get("ui_lang"):
        set_lang(saved["ui_lang"])

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    _app_icon = create_app_icon()
    app.setWindowIcon(_app_icon)

    # First launch → setup wizard (hub + download)
    if not SETTINGS_FILE.exists():
        wizard = SetupWizardDialog()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        saved = _load_saved_settings()
        log.info("Setup wizard completed")

    # Non-first launch but models missing → download dialog
    else:
        missing = get_missing_models(
            saved.get("asr_engine", "sensevoice"),
            config["asr"]["model_size"],
            saved.get("hub", "ms"),
        )
        if missing:
            log.info(f"Missing models: {[m['name'] for m in missing]}")
            dlg = ModelDownloadDialog(missing, hub=saved.get("hub", "ms"))
            if dlg.exec() != QDialog.DialogCode.Accepted:
                sys.exit(0)

    log_window = LogWindow()
    log_handler = log_window.get_handler()
    logging.getLogger().addHandler(log_handler)

    panel = ControlPanel(config, saved_settings=saved)

    overlay = SubtitleOverlay(config["subtitle"])
    overlay.show()

    live_trans = LiveTransApp(config)
    live_trans.set_overlay(overlay)
    live_trans.set_panel(panel)

    def _deferred_init():
        panel._apply_settings()
        models = panel.get_settings().get("models", [])
        active_idx = panel.get_settings().get("active_model", 0)
        overlay.set_models(models, active_idx)
        active_model = panel.get_active_model()
        if active_model:
            live_trans._on_model_changed(active_model)

    QTimer.singleShot(100, _deferred_init)

    tray = QSystemTrayIcon()
    tray.setToolTip(t("tray_tooltip"))
    tray.setIcon(_app_icon)

    menu = QMenu()
    start_action = QAction(t("tray_start"))
    stop_action = QAction(t("tray_stop"))
    log_action = QAction(t("tray_show_log"))
    panel_action = QAction(t("tray_show_panel"))
    quit_action = QAction(t("quit"))

    def on_start():
        try:
            live_trans.start()
            overlay.set_running(True)
        except Exception as e:
            log.error(f"Start error: {e}", exc_info=True)

    def on_stop():
        live_trans.stop()
        overlay.set_running(False)

    def on_pause():
        live_trans.pause()
        overlay.set_running(False)

    def on_resume():
        live_trans.resume()
        overlay.set_running(True)

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

    def on_quit():
        live_trans.stop()
        app.quit()

    start_action.triggered.connect(on_start)
    stop_action.triggered.connect(on_stop)
    log_action.triggered.connect(on_toggle_log)
    panel_action.triggered.connect(on_toggle_panel)
    def on_overlay_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            _save_settings(settings)
            live_trans._on_model_changed(models[index])

    overlay.settings_requested.connect(on_toggle_panel)
    overlay.target_language_changed.connect(live_trans._on_target_language_changed)
    overlay.model_switch_requested.connect(on_overlay_model_switch)
    overlay.start_requested.connect(on_resume)
    overlay.stop_requested.connect(on_pause)
    overlay.quit_requested.connect(on_quit)
    quit_action.triggered.connect(on_quit)

    menu.addAction(start_action)
    menu.addAction(stop_action)
    menu.addSeparator()
    menu.addAction(log_action)
    menu.addAction(panel_action)
    menu.addSeparator()
    menu.addAction(quit_action)

    tray.setContextMenu(menu)
    tray.show()

    QTimer.singleShot(500, on_start)

    signal.signal(signal.SIGINT, lambda *_: on_quit())
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
