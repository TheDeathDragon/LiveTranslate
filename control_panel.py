import json
import logging
import os
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from benchmark import run_benchmark
from dialogs import (
    ModelEditDialog,
)
from model_manager import (
    MODELS_DIR,
    dir_size,
    format_size,
    get_cache_entries,
)
from i18n import t

log = logging.getLogger("LiveTrans.Panel")

SETTINGS_FILE = Path(__file__).parent / "user_settings.json"


def _load_saved_settings() -> dict | None:
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            log.info(f"Loaded saved settings from {SETTINGS_FILE}")
            return data
    except Exception as e:
        log.warning(f"Failed to load settings: {e}")
    return None


def _save_settings(settings: dict):
    try:
        SETTINGS_FILE.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        log.warning(f"Failed to save settings: {e}")


class ControlPanel(QWidget):
    """Settings and monitoring panel."""

    settings_changed = pyqtSignal(dict)
    model_changed = pyqtSignal(dict)
    _bench_result = pyqtSignal(str)
    _cache_result = pyqtSignal(list)

    def __init__(self, config, saved_settings=None):
        super().__init__()
        self._config = config
        self.setWindowTitle(t("window_control_panel"))
        self.setMinimumSize(480, 560)
        self.resize(520, 650)

        saved = saved_settings or _load_saved_settings()
        if saved:
            self._current_settings = saved
        else:
            tc = config["translation"]
            self._current_settings = {
                "vad_mode": "silero",
                "vad_threshold": config["asr"]["vad_threshold"],
                "energy_threshold": 0.02,
                "min_speech_duration": config["asr"]["min_speech_duration"],
                "max_speech_duration": config["asr"]["max_speech_duration"],
                "silence_mode": "auto",
                "silence_duration": 0.8,
                "asr_language": config["asr"].get("language", "auto"),
                "asr_engine": "sensevoice",
                "asr_device": "cuda",
                "models": [
                    {
                        "name": f"{tc['model']}",
                        "api_base": tc["api_base"],
                        "api_key": tc["api_key"],
                        "model": tc["model"],
                    }
                ],
                "active_model": 0,
                "hub": "ms",
            }

        if "models" not in self._current_settings:
            tc = config["translation"]
            self._current_settings["models"] = [
                {
                    "name": f"{tc['model']}",
                    "api_base": tc["api_base"],
                    "api_key": tc["api_key"],
                    "model": tc["model"],
                }
            ]
            self._current_settings["active_model"] = 0

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        tabs.addTab(self._create_vad_tab(), t("tab_vad_asr"))
        tabs.addTab(self._create_translation_tab(), t("tab_translation"))
        tabs.addTab(self._create_benchmark_tab(), t("tab_benchmark"))
        self._cache_tab_index = tabs.addTab(self._create_cache_tab(), t("tab_cache"))
        tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(tabs)

        self._bench_result.connect(self._on_bench_result)
        self._cache_result.connect(self._on_cache_result)

        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._do_auto_save)

    # ── VAD / ASR Tab ──

    def _create_vad_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings

        asr_group = QGroupBox(t("group_asr_engine"))
        asr_layout = QGridLayout(asr_group)

        self._asr_engine = QComboBox()
        self._asr_engine.addItems(
            [
                "Whisper (faster-whisper)",
                "SenseVoice (FunASR)",
                "Fun-ASR-Nano (FunASR)",
                "Fun-ASR-MLT-Nano (FunASR, 31 langs)",
            ]
        )
        engine_map_idx = {
            "whisper": 0,
            "sensevoice": 1,
            "funasr-nano": 2,
            "funasr-mlt-nano": 3,
        }
        engine_idx = engine_map_idx.get(s.get("asr_engine"), 0)
        self._asr_engine.setCurrentIndex(engine_idx)
        asr_layout.addWidget(QLabel(t("label_engine")), 0, 0)
        asr_layout.addWidget(self._asr_engine, 0, 1)
        self._asr_engine.currentIndexChanged.connect(self._auto_save)

        self._asr_lang = QComboBox()
        self._asr_lang.addItems(
            ["auto", "ja", "en", "zh", "ko", "fr", "de", "es", "ru"]
        )
        lang = s.get("asr_language", self._config["asr"].get("language", "auto"))
        idx = self._asr_lang.findText(lang)
        if idx >= 0:
            self._asr_lang.setCurrentIndex(idx)
        asr_layout.addWidget(QLabel(t("label_language_hint")), 1, 0)
        asr_layout.addWidget(self._asr_lang, 1, 1)
        self._asr_lang.currentIndexChanged.connect(self._auto_save)

        self._asr_device = QComboBox()
        devices = ["cuda", "cpu"]
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.insert(i, f"cuda:{i} ({name})")
            if torch.cuda.device_count() > 0:
                devices = [d for d in devices if d != "cuda"]
        except Exception:
            pass
        self._asr_device.addItems(devices)
        saved_dev = s.get("asr_device", self._config["asr"].get("device", "cuda"))
        for i in range(self._asr_device.count()):
            if self._asr_device.itemText(i).startswith(saved_dev):
                self._asr_device.setCurrentIndex(i)
                break
        asr_layout.addWidget(QLabel(t("label_device")), 2, 0)
        asr_layout.addWidget(self._asr_device, 2, 1)
        self._asr_device.currentIndexChanged.connect(self._auto_save)

        self._audio_device = QComboBox()
        self._audio_device.addItem(t("system_default"))
        try:
            from audio_capture import list_output_devices

            for name in list_output_devices():
                self._audio_device.addItem(name)
        except Exception:
            pass
        saved_audio = s.get("audio_device")
        if saved_audio:
            idx = self._audio_device.findText(saved_audio)
            if idx >= 0:
                self._audio_device.setCurrentIndex(idx)
        asr_layout.addWidget(QLabel(t("label_audio")), 3, 0)
        asr_layout.addWidget(self._audio_device, 3, 1)
        self._audio_device.currentIndexChanged.connect(self._auto_save)

        self._hub_combo = QComboBox()
        self._hub_combo.addItems([t("hub_modelscope"), t("hub_huggingface")])
        saved_hub = s.get("hub", "ms")
        self._hub_combo.setCurrentIndex(0 if saved_hub == "ms" else 1)
        asr_layout.addWidget(QLabel(t("label_hub")), 4, 0)
        asr_layout.addWidget(self._hub_combo, 4, 1)
        self._hub_combo.currentIndexChanged.connect(self._auto_save)

        self._ui_lang_combo = QComboBox()
        self._ui_lang_combo.addItems(["English", "中文"])
        saved_lang = s.get("ui_lang", "en")
        self._ui_lang_combo.setCurrentIndex(0 if saved_lang == "en" else 1)
        asr_layout.addWidget(QLabel(t("label_ui_lang")), 5, 0)
        asr_layout.addWidget(self._ui_lang_combo, 5, 1)
        self._ui_lang_combo.currentIndexChanged.connect(self._on_ui_lang_changed)

        layout.addWidget(asr_group)

        mode_group = QGroupBox(t("group_vad_mode"))
        mode_layout = QVBoxLayout(mode_group)
        self._vad_mode = QComboBox()
        self._vad_mode.addItems(
            [t("vad_silero"), t("vad_energy"), t("vad_disabled")]
        )
        mode_map = {"silero": 0, "energy": 1, "disabled": 2}
        self._vad_mode.setCurrentIndex(mode_map.get(s.get("vad_mode", "energy"), 1))
        self._vad_mode.currentIndexChanged.connect(self._on_vad_mode_changed)
        self._vad_mode.currentIndexChanged.connect(self._auto_save)
        mode_layout.addWidget(self._vad_mode)
        layout.addWidget(mode_group)

        silero_group = QGroupBox(t("group_silero_threshold"))
        silero_layout = QGridLayout(silero_group)
        self._vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._vad_threshold_slider.setRange(0, 100)
        vad_pct = int(s.get("vad_threshold", 0.3) * 100)
        self._vad_threshold_slider.setValue(vad_pct)
        self._vad_threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._vad_threshold_slider.sliderReleased.connect(self._auto_save)
        self._vad_threshold_label = QLabel(f"{vad_pct}%")
        self._vad_threshold_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        silero_layout.addWidget(QLabel(t("label_threshold")), 0, 0)
        silero_layout.addWidget(self._vad_threshold_slider, 0, 1)
        silero_layout.addWidget(self._vad_threshold_label, 0, 2)
        layout.addWidget(silero_group)

        energy_group = QGroupBox(t("group_energy_threshold"))
        energy_layout = QGridLayout(energy_group)
        self._energy_slider = QSlider(Qt.Orientation.Horizontal)
        self._energy_slider.setRange(1, 100)
        energy_pm = int(s.get("energy_threshold", 0.03) * 1000)
        self._energy_slider.setValue(energy_pm)
        self._energy_slider.valueChanged.connect(self._on_energy_changed)
        self._energy_slider.sliderReleased.connect(self._auto_save)
        self._energy_label = QLabel(f"{energy_pm}\u2030")
        self._energy_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        energy_layout.addWidget(QLabel(t("label_threshold")), 0, 0)
        energy_layout.addWidget(self._energy_slider, 0, 1)
        energy_layout.addWidget(self._energy_label, 0, 2)
        layout.addWidget(energy_group)

        timing_group = QGroupBox(t("group_timing"))
        timing_layout = QGridLayout(timing_group)
        self._min_speech = QDoubleSpinBox()
        self._min_speech.setRange(0.1, 5.0)
        self._min_speech.setSingleStep(0.1)
        self._min_speech.setValue(s.get("min_speech_duration", 1.0))
        self._min_speech.setSuffix(" s")
        self._min_speech.valueChanged.connect(self._on_timing_changed)
        self._min_speech.valueChanged.connect(self._auto_save)
        self._max_speech = QDoubleSpinBox()
        self._max_speech.setRange(3.0, 30.0)
        self._max_speech.setSingleStep(1.0)
        self._max_speech.setValue(s.get("max_speech_duration", 8.0))
        self._max_speech.setSuffix(" s")
        self._max_speech.valueChanged.connect(self._on_timing_changed)
        self._max_speech.valueChanged.connect(self._auto_save)
        self._silence_mode = QComboBox()
        self._silence_mode.addItems([t("silence_auto"), t("silence_fixed")])
        saved_smode = s.get("silence_mode", "auto")
        self._silence_mode.setCurrentIndex(0 if saved_smode == "auto" else 1)
        self._silence_mode.currentIndexChanged.connect(self._on_silence_mode_changed)
        self._silence_mode.currentIndexChanged.connect(self._on_timing_changed)
        self._silence_mode.currentIndexChanged.connect(self._auto_save)

        self._silence_duration = QDoubleSpinBox()
        self._silence_duration.setRange(0.1, 3.0)
        self._silence_duration.setSingleStep(0.1)
        self._silence_duration.setValue(s.get("silence_duration", 0.8))
        self._silence_duration.setSuffix(" s")
        self._silence_duration.setEnabled(saved_smode != "auto")
        self._silence_duration.valueChanged.connect(self._on_timing_changed)
        self._silence_duration.valueChanged.connect(self._auto_save)

        timing_layout.addWidget(QLabel(t("label_min_speech")), 0, 0)
        timing_layout.addWidget(self._min_speech, 0, 1)
        timing_layout.addWidget(QLabel(t("label_max_speech")), 1, 0)
        timing_layout.addWidget(self._max_speech, 1, 1)
        timing_layout.addWidget(QLabel(t("label_silence")), 2, 0)
        timing_layout.addWidget(self._silence_mode, 2, 1)
        timing_layout.addWidget(QLabel(t("label_silence_dur")), 3, 0)
        timing_layout.addWidget(self._silence_duration, 3, 1)
        layout.addWidget(timing_group)

        layout.addStretch()
        return widget

    # ── Translation Tab ──

    def _create_translation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings

        active_group = QGroupBox(t("group_active_model"))
        active_layout = QHBoxLayout(active_group)
        self._active_model_combo = QComboBox()
        self._refresh_model_combo()
        active_idx = s.get("active_model", 0)
        if 0 <= active_idx < self._active_model_combo.count():
            self._active_model_combo.setCurrentIndex(active_idx)
        self._active_model_combo.currentIndexChanged.connect(
            self._on_active_model_changed
        )
        active_layout.addWidget(self._active_model_combo)

        apply_model_btn = QPushButton(t("btn_apply"))
        apply_model_btn.setFixedWidth(60)
        apply_model_btn.clicked.connect(self._apply_active_model)
        active_layout.addWidget(apply_model_btn)
        layout.addWidget(active_group)

        models_group = QGroupBox(t("group_model_configs"))
        models_layout = QVBoxLayout(models_group)

        self._model_list = QListWidget()
        self._model_list.setFont(QFont("Consolas", 9))
        self._model_list.itemDoubleClicked.connect(self._on_model_double_clicked)
        self._refresh_model_list()
        models_layout.addWidget(self._model_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton(t("btn_add"))
        add_btn.clicked.connect(self._add_model)
        btn_row.addWidget(add_btn)
        edit_btn = QPushButton(t("btn_edit"))
        edit_btn.clicked.connect(self._edit_model)
        btn_row.addWidget(edit_btn)
        dup_btn = QPushButton(t("btn_duplicate"))
        dup_btn.clicked.connect(self._dup_model)
        btn_row.addWidget(dup_btn)
        del_btn = QPushButton(t("btn_remove"))
        del_btn.clicked.connect(self._remove_model)
        btn_row.addWidget(del_btn)
        models_layout.addLayout(btn_row)
        layout.addWidget(models_group)

        prompt_group = QGroupBox(t("group_system_prompt"))
        prompt_layout = QVBoxLayout(prompt_group)

        from translator import DEFAULT_PROMPT

        self._prompt_edit = QTextEdit()
        self._prompt_edit.setFont(QFont("Consolas", 9))
        self._prompt_edit.setMaximumHeight(100)
        self._prompt_edit.setPlainText(s.get("system_prompt", DEFAULT_PROMPT))
        prompt_layout.addWidget(self._prompt_edit)

        prompt_btn_row = QHBoxLayout()
        reset_prompt_btn = QPushButton(t("btn_restore_default"))
        reset_prompt_btn.clicked.connect(
            lambda: self._prompt_edit.setPlainText(DEFAULT_PROMPT)
        )
        prompt_btn_row.addWidget(reset_prompt_btn)
        apply_prompt_btn = QPushButton(t("btn_apply_prompt"))
        apply_prompt_btn.clicked.connect(self._apply_prompt)
        prompt_btn_row.addWidget(apply_prompt_btn)
        prompt_btn_row.addStretch()
        prompt_layout.addLayout(prompt_btn_row)
        layout.addWidget(prompt_group)

        timeout_group = QGroupBox(t("group_timeout"))
        timeout_layout = QHBoxLayout(timeout_group)
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(1, 60)
        self._timeout_spin.setValue(s.get("timeout", 5))
        self._timeout_spin.setSuffix(" s")
        self._timeout_spin.valueChanged.connect(
            lambda v: self._current_settings.update({"timeout": v})
        )
        self._timeout_spin.valueChanged.connect(self._auto_save)
        timeout_layout.addWidget(self._timeout_spin)
        timeout_layout.addStretch()
        layout.addWidget(timeout_group)

        layout.addStretch()
        return widget

    # ── Benchmark Tab ──

    def _create_benchmark_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel(t("label_source")))
        self._bench_lang = QComboBox()
        self._bench_lang.addItems(["ja", "en", "zh", "ko", "fr", "de"])
        self._bench_lang.setCurrentIndex(0)
        ctrl_row.addWidget(self._bench_lang)
        ctrl_row.addWidget(QLabel(t("target_label")))
        self._bench_target = QComboBox()
        self._bench_target.addItems(["zh", "en", "ja", "ko", "fr", "de", "es", "ru"])
        ctrl_row.addWidget(self._bench_target)
        ctrl_row.addStretch()
        self._bench_btn = QPushButton(t("btn_test_all"))
        self._bench_btn.clicked.connect(self._run_benchmark)
        ctrl_row.addWidget(self._bench_btn)
        layout.addLayout(ctrl_row)

        self._bench_output = QTextEdit()
        self._bench_output.setReadOnly(True)
        self._bench_output.setFont(QFont("Consolas", 9))
        self._bench_output.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border: 1px solid #444;"
        )
        layout.addWidget(self._bench_output)

        return widget

    # ── Cache Tab ──

    def _create_cache_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        top_row = QHBoxLayout()
        self._cache_total = QLabel("")
        self._cache_total.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        top_row.addWidget(self._cache_total, 1)
        open_btn = QPushButton(t("btn_open_folder"))
        open_btn.clicked.connect(
            lambda: (
                MODELS_DIR.mkdir(parents=True, exist_ok=True),
                os.startfile(str(MODELS_DIR)),
            )
        )
        top_row.addWidget(open_btn)
        delete_all_btn = QPushButton(t("btn_delete_all_exit"))
        delete_all_btn.clicked.connect(self._delete_all_and_exit)
        top_row.addWidget(delete_all_btn)
        layout.addLayout(top_row)

        self._cache_list = QListWidget()
        self._cache_list.setFont(QFont("Consolas", 9))
        self._cache_list.setAlternatingRowColors(True)
        layout.addWidget(self._cache_list, 1)

        self._cache_entries = []
        self._refresh_cache()

        return widget

    def _on_tab_changed(self, index):
        if index == self._cache_tab_index:
            self._refresh_cache()

    def _refresh_cache(self):
        self._cache_list.clear()
        self._cache_total.setText(t("scanning"))

        def _scan():
            entries = get_cache_entries()
            results = []
            for name, path in entries:
                size = dir_size(path)
                results.append((name, str(path), size))
            self._cache_result.emit(results)

        threading.Thread(target=_scan, daemon=True).start()

    def _on_cache_result(self, results):
        self._cache_list.clear()
        self._cache_entries = results
        total = 0
        for name, path, size in results:
            total += size
            self._cache_list.addItem(f"{name}  —  {format_size(size)}")
        if not results:
            self._cache_list.addItem(t("no_cached_models"))
        self._cache_total.setText(
            t("cache_total").format(size=format_size(total), count=len(results))
        )

    def _delete_all_and_exit(self):
        if not self._cache_entries:
            return
        import shutil

        total_size = sum(s for _, _, s in self._cache_entries)
        ret = QMessageBox.warning(
            self,
            t("dialog_delete_title"),
            t("dialog_delete_msg").format(
                count=len(self._cache_entries), size=format_size(total_size)
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        for name, path, _ in self._cache_entries:
            try:
                shutil.rmtree(path)
                log.info(f"Deleted: {path}")
            except Exception as e:
                log.error(f"Failed to delete {path}: {e}")
        QApplication.instance().quit()

    # ── Model Management ──

    def _refresh_model_combo(self):
        self._active_model_combo.blockSignals(True)
        self._active_model_combo.clear()
        for m in self._current_settings.get("models", []):
            self._active_model_combo.addItem(f"{m['name']}  ({m['model']})")
        self._active_model_combo.blockSignals(False)

    def _refresh_model_list(self):
        self._model_list.clear()
        active = self._current_settings.get("active_model", 0)
        for i, m in enumerate(self._current_settings.get("models", [])):
            prefix = ">>> " if i == active else "    "
            proxy = m.get("proxy", "none")
            proxy_tag = f"  [proxy: {proxy}]" if proxy != "none" else ""
            text = (
                f"{prefix}{m['name']}{proxy_tag}\n     {m['api_base']}  |  {m['model']}"
            )
            item = QListWidgetItem(text)
            if i == active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._model_list.addItem(item)

    def _add_model(self):
        dlg = ModelEditDialog(self)
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                self._current_settings.setdefault("models", []).append(data)
                self._refresh_model_list()
                self._refresh_model_combo()
                _save_settings(self._current_settings)

    def _edit_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dlg = ModelEditDialog(self, models[row])
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                models[row] = data
                self._refresh_model_list()
                self._refresh_model_combo()
                _save_settings(self._current_settings)

    def _dup_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dup = dict(models[row])
        dup["name"] = dup["name"] + " (copy)"
        models.append(dup)
        self._refresh_model_list()
        self._refresh_model_combo()
        _save_settings(self._current_settings)

    def _remove_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models) or len(models) <= 1:
            return
        models.pop(row)
        active = self._current_settings.get("active_model", 0)
        if active >= len(models):
            self._current_settings["active_model"] = len(models) - 1
        self._refresh_model_list()
        self._refresh_model_combo()
        self._model_list.setCurrentRow(min(row, len(models) - 1))
        _save_settings(self._current_settings)

    def _on_active_model_changed(self, index):
        if index >= 0:
            self._current_settings["active_model"] = index
            self._refresh_model_list()

    def _on_model_double_clicked(self, item):
        row = self._model_list.row(item)
        models = self._current_settings.get("models", [])
        if 0 <= row < len(models):
            self._active_model_combo.setCurrentIndex(row)
            self._apply_active_model()

    def _apply_active_model(self):
        idx = self._active_model_combo.currentIndex()
        models = self._current_settings.get("models", [])
        if 0 <= idx < len(models):
            self._current_settings["active_model"] = idx
            self._refresh_model_list()
            self.model_changed.emit(models[idx])
            _save_settings(self._current_settings)
            log.info(f"Active model: {models[idx]['name']} ({models[idx]['model']})")

    def _run_benchmark(self):
        models = self._current_settings.get("models", [])
        if not models:
            return

        source_lang = self._bench_lang.currentText()
        target_lang = self._bench_target.currentText()
        timeout_s = self._current_settings.get("timeout", 5)

        self._bench_btn.setEnabled(False)
        self._bench_btn.setText(t("testing"))
        self._bench_output.clear()

        from translator import DEFAULT_PROMPT, LANGUAGE_DISPLAY

        src = LANGUAGE_DISPLAY.get(source_lang, source_lang)
        tgt = LANGUAGE_DISPLAY.get(target_lang, target_lang)
        prompt = self._current_settings.get("system_prompt", DEFAULT_PROMPT)
        try:
            prompt = prompt.format(source_lang=src, target_lang=tgt)
        except (KeyError, IndexError):
            pass

        run_benchmark(
            models, source_lang, target_lang, timeout_s, prompt, self._bench_result.emit
        )

    def _on_bench_result(self, text: str):
        if text == "__DONE__":
            self._bench_btn.setEnabled(True)
            self._bench_btn.setText(t("btn_test_all"))
        else:
            self._bench_output.append(text)

    # ── Shared logic ──

    def _on_silence_mode_changed(self, index):
        self._silence_duration.setEnabled(index == 1)

    def _on_vad_mode_changed(self, index):
        modes = ["silero", "energy", "disabled"]
        self._current_settings["vad_mode"] = modes[index]

    def _on_threshold_changed(self, value):
        t = value / 100.0
        self._current_settings["vad_threshold"] = t
        self._vad_threshold_label.setText(f"{value}%")
        if not self._vad_threshold_slider.isSliderDown():
            self._auto_save()

    def _on_energy_changed(self, value):
        t = value / 1000.0
        self._current_settings["energy_threshold"] = t
        self._energy_label.setText(f"{value}\u2030")
        if not self._energy_slider.isSliderDown():
            self._auto_save()

    def _on_timing_changed(self):
        self._current_settings["min_speech_duration"] = self._min_speech.value()
        self._current_settings["max_speech_duration"] = self._max_speech.value()
        self._current_settings["silence_mode"] = "auto" if self._silence_mode.currentIndex() == 0 else "fixed"
        self._current_settings["silence_duration"] = self._silence_duration.value()

    def _on_ui_lang_changed(self, index):
        lang = "en" if index == 0 else "zh"
        self._current_settings["ui_lang"] = lang
        _save_settings(self._current_settings)
        from i18n import set_lang
        set_lang(lang)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "LiveTrans",
            "Language changed. Please restart the application.\n"
            "语言已更改，请重启应用程序。"
        )

    def _auto_save(self):
        self._save_timer.start()

    def _do_auto_save(self):
        self._apply_settings()
        _save_settings(self._current_settings)

    def _apply_prompt(self):
        text = self._prompt_edit.toPlainText().strip()
        if text:
            self._current_settings["system_prompt"] = text
            self._apply_active_model()
            _save_settings(self._current_settings)
            log.info("System prompt updated")

    def _apply_settings(self):
        self._current_settings["asr_language"] = self._asr_lang.currentText()
        engine_map = {
            0: "whisper",
            1: "sensevoice",
            2: "funasr-nano",
            3: "funasr-mlt-nano",
        }
        self._current_settings["asr_engine"] = engine_map[
            self._asr_engine.currentIndex()
        ]
        dev_text = self._asr_device.currentText()
        self._current_settings["asr_device"] = dev_text.split(" (")[0]
        audio_dev = self._audio_device.currentText()
        self._current_settings["audio_device"] = (
            None if self._audio_device.currentIndex() == 0 else audio_dev
        )
        self._current_settings["hub"] = (
            "ms" if self._hub_combo.currentIndex() == 0 else "hf"
        )
        prompt_text = self._prompt_edit.toPlainText().strip()
        if prompt_text:
            self._current_settings["system_prompt"] = prompt_text
        self._current_settings["timeout"] = self._timeout_spin.value()
        safe = {
            k: v
            for k, v in self._current_settings.items()
            if k not in ("models", "system_prompt")
        }
        log.info(f"Settings applied: {safe}")
        self.settings_changed.emit(dict(self._current_settings))

    def get_settings(self):
        return dict(self._current_settings)

    def get_active_model(self) -> dict | None:
        models = self._current_settings.get("models", [])
        idx = self._current_settings.get("active_model", 0)
        if 0 <= idx < len(models):
            return models[idx]
        return None

    def has_saved_settings(self) -> bool:
        return SETTINGS_FILE.exists()
