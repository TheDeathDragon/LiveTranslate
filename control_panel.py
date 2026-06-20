import logging
import os
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFontComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    DEFAULT_FUNASR_MODEL,
    MODELS_DIR,
    _WHISPER_SIZES,
    dir_size,
    funasr_model_options,
    funasr_supports_padding,
    format_size,
    get_cache_entries,
    list_local_parakeet_cpp_models,
    list_local_parakeet_cpp_runtimes,
    list_local_crispasr_models,
    list_local_faster_whisper_models,
    list_local_firered_vad_models,
    list_local_sherpa_onnx_models,
    migrate_funasr_settings,
    normalize_funasr_model_key,
    resolve_custom_parakeet_cpp_model,
    resolve_parakeet_cpp_runtime_dir,
    resolve_custom_crispasr_model,
    resolve_custom_firered_vad_model,
    resolve_custom_sherpa_onnx_model,
    resolve_custom_whisper_model,
)
from i18n import t, LANGUAGES
from subtitle_settings import SubtitleSettingsWidget
from settings_store import (
    SETTINGS_FILE,
    load_settings,
    normalize_settings,
    save_settings,
)

log = logging.getLogger("LiveTranslate.Panel")

def _load_saved_settings() -> dict | None:
    return load_settings()


def _save_settings(settings: dict):
    save_settings(settings)


class ControlPanel(QWidget):
    """Settings and monitoring panel."""

    settings_changed = pyqtSignal(dict)
    model_changed = pyqtSignal(dict)
    models_list_changed = pyqtSignal(list, int)
    subtitle_settings_changed = pyqtSignal(dict)
    asr_language_changed = pyqtSignal(str)
    _bench_result = pyqtSignal(str)
    _cache_result = pyqtSignal(list)
    reset_positions = pyqtSignal()

    def __init__(self, config, saved_settings=None):
        super().__init__()
        self._config = config
        self.setWindowTitle(t("window_control_panel"))
        self.setMinimumSize(480, 560)
        self.resize(520, 650)

        saved = migrate_funasr_settings(saved_settings) or _load_saved_settings()
        self._current_settings = normalize_settings(config, saved)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        tabs.addTab(self._create_asr_tab(), t("tab_asr"))
        tabs.addTab(self._create_vad_tab(), t("tab_vad"))
        tabs.addTab(self._create_translation_tab(), t("tab_translation"))
        tabs.addTab(self._create_style_tab(), t("tab_style"))
        tabs.addTab(self._create_subtitle_tab(), t("tab_subtitle"))
        tabs.addTab(self._create_benchmark_tab(), t("tab_benchmark"))
        self._cache_tab_index = tabs.addTab(self._create_cache_tab(), t("tab_cache"))
        tabs.addTab(self._create_changelog_tab(), t("tab_changelog"))
        tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(tabs)

        self._bench_result.connect(self._on_bench_result)
        self._cache_result.connect(self._on_cache_result)

        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._do_auto_save)

        # Fit initial height based on whisper group visibility
        QTimer.singleShot(0, lambda: self.resize(self.width(), self.sizeHint().height() + 20))

    # ── ASR Tab ──

    def _create_asr_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings

        asr_group = QGroupBox(t("group_asr_engine"))
        asr_layout = QGridLayout(asr_group)
        asr_layout.setColumnStretch(0, 1)
        asr_layout.setColumnMinimumWidth(1, 180)

        self._asr_engine = QComboBox()
        self._asr_engine.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        for label, key in (
            (f"[{t('asr_accurate')}] Whisper (faster-whisper)", "whisper"),
            (f"[{t('asr_fast')}] FunASR", "funasr"),
            ("Anime-Whisper (ja, anime/galgame)", "anime-whisper"),
            ("Remote Whisper (remote GPU server)", "remote-whisper"),
            ("CrispASR (ggml)", "crispasr"),
            ("sherpa-onnx (ONNX)", "sherpa-onnx"),
            ("parakeet.cpp (GGUF)", "parakeet-cpp"),
        ):
            self._asr_engine.addItem(label, key)
        engine_idx = self._asr_engine.findData(s.get("asr_engine"))
        if engine_idx < 0:
            engine_idx = self._asr_engine.findData("funasr")
        self._asr_engine.setCurrentIndex(engine_idx)
        asr_layout.addWidget(QLabel(t("label_engine")), 0, 0)
        asr_layout.addWidget(self._asr_engine, 0, 1)
        self._asr_engine.currentIndexChanged.connect(self._auto_save)

        self._asr_lang = QComboBox()
        for code, native in LANGUAGES:
            label = t("asr_lang_auto") if code == "auto" else native
            self._asr_lang.addItem(f"{code} - {label}", code)
        lang = s.get("asr_language", self._config["asr"].get("language", "auto"))
        idx = self._asr_lang.findData(lang)
        if idx >= 0:
            self._asr_lang.setCurrentIndex(idx)
        asr_layout.addWidget(QLabel(t("label_language_hint")), 1, 0)
        asr_layout.addWidget(self._asr_lang, 1, 1)
        self._asr_lang.currentIndexChanged.connect(self._auto_save)
        self._asr_lang.currentIndexChanged.connect(
            lambda _idx: self.asr_language_changed.emit(self._get_asr_lang_code())
        )

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

        self._funasr_model_label = QLabel(t("label_funasr_model"))
        self._funasr_model_combo = QComboBox()
        for key, display_name in funasr_model_options():
            self._funasr_model_combo.addItem(display_name, key)
        saved_funasr_model = normalize_funasr_model_key(
            s.get("funasr_model", DEFAULT_FUNASR_MODEL)
        )
        funasr_idx = self._funasr_model_combo.findData(saved_funasr_model)
        if funasr_idx >= 0:
            self._funasr_model_combo.setCurrentIndex(funasr_idx)
        self._funasr_model_combo.currentIndexChanged.connect(
            self._on_funasr_model_changed
        )
        asr_layout.addWidget(self._funasr_model_label, 3, 0)
        asr_layout.addWidget(self._funasr_model_combo, 3, 1)

        self._crispasr_gpu_label = QLabel(t("label_crispasr_gpu_backend"))
        self._crispasr_gpu_backend = QComboBox()
        self._crispasr_gpu_backend.addItems(["Auto", "CUDA", "Vulkan", "CPU"])
        gpu_backend = str(s.get("crispasr_gpu_backend", "auto")).lower()
        gpu_idx = {"auto": 0, "cuda": 1, "vulkan": 2, "cpu": 3}.get(gpu_backend, 0)
        self._crispasr_gpu_backend.setCurrentIndex(gpu_idx)
        self._crispasr_gpu_backend.currentIndexChanged.connect(
            self._on_crispasr_runtime_setting_changed
        )
        asr_layout.addWidget(self._crispasr_gpu_label, 4, 0)
        asr_layout.addWidget(self._crispasr_gpu_backend, 4, 1)

        self._crispasr_device_label = QLabel(t("label_crispasr_device_index"))
        self._crispasr_device_index = QSpinBox()
        self._crispasr_device_index.setRange(0, 16)
        self._crispasr_device_index.setValue(int(s.get("crispasr_device_index", 0) or 0))
        self._crispasr_device_index.valueChanged.connect(
            self._on_crispasr_runtime_setting_changed
        )
        asr_layout.addWidget(self._crispasr_device_label, 5, 0)
        asr_layout.addWidget(self._crispasr_device_index, 5, 1)

        self._crispasr_punc_label = QLabel(t("label_crispasr_punc_model"))
        self._crispasr_punc_model = QComboBox()
        for label, value in (
            ("Auto", "auto"),
            ("Off", "off"),
            ("FireRedPunc", "firered"),
            ("fullstop", "fullstop"),
            ("punctuate-all", "punctuate-all"),
            ("PCS", "pcs"),
        ):
            self._crispasr_punc_model.addItem(label, value)
        punc = str(s.get("crispasr_punc_model", "auto")).lower()
        punc_idx = self._crispasr_punc_model.findData(punc)
        if punc_idx >= 0:
            self._crispasr_punc_model.setCurrentIndex(punc_idx)
        self._crispasr_punc_model.currentIndexChanged.connect(
            self._on_crispasr_runtime_setting_changed
        )
        asr_layout.addWidget(self._crispasr_punc_label, 6, 0)
        asr_layout.addWidget(self._crispasr_punc_model, 6, 1)

        self._crispasr_unified_memory = QCheckBox(t("label_crispasr_unified_memory"))
        self._crispasr_unified_memory.setChecked(
            bool(s.get("crispasr_unified_memory", True))
        )
        self._crispasr_unified_memory.toggled.connect(
            self._on_crispasr_runtime_setting_changed
        )
        asr_layout.addWidget(self._crispasr_unified_memory, 7, 1)

        self._sherpa_onnx_provider_label = QLabel(t("label_sherpa_onnx_provider"))
        self._sherpa_onnx_provider = QComboBox()
        for label, value in (("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")):
            self._sherpa_onnx_provider.addItem(label, value)
        provider = str(s.get("sherpa_onnx_provider", "auto")).lower()
        provider_idx = self._sherpa_onnx_provider.findData(provider)
        if provider_idx >= 0:
            self._sherpa_onnx_provider.setCurrentIndex(provider_idx)
        self._sherpa_onnx_provider.currentIndexChanged.connect(
            self._on_sherpa_onnx_setting_changed
        )
        asr_layout.addWidget(self._sherpa_onnx_provider_label, 8, 0)
        asr_layout.addWidget(self._sherpa_onnx_provider, 8, 1)

        self._sherpa_onnx_threads_label = QLabel(t("label_sherpa_onnx_threads"))
        self._sherpa_onnx_num_threads = QSpinBox()
        self._sherpa_onnx_num_threads.setRange(1, 32)
        self._sherpa_onnx_num_threads.setValue(
            int(s.get("sherpa_onnx_num_threads", 2) or 2)
        )
        self._sherpa_onnx_num_threads.valueChanged.connect(
            self._on_sherpa_onnx_setting_changed
        )
        asr_layout.addWidget(self._sherpa_onnx_threads_label, 9, 0)
        asr_layout.addWidget(self._sherpa_onnx_num_threads, 9, 1)

        self._sherpa_onnx_decoding_label = QLabel(t("label_sherpa_onnx_decoding"))
        self._sherpa_onnx_decoding_method = QComboBox()
        self._sherpa_onnx_decoding_method.addItem("greedy_search", "greedy_search")
        decoding = str(s.get("sherpa_onnx_decoding_method", "greedy_search"))
        decoding_idx = self._sherpa_onnx_decoding_method.findData(decoding)
        if decoding_idx >= 0:
            self._sherpa_onnx_decoding_method.setCurrentIndex(decoding_idx)
        self._sherpa_onnx_decoding_method.currentIndexChanged.connect(
            self._on_sherpa_onnx_setting_changed
        )
        asr_layout.addWidget(self._sherpa_onnx_decoding_label, 10, 0)
        asr_layout.addWidget(self._sherpa_onnx_decoding_method, 10, 1)

        self._parakeet_cpp_backend_label = QLabel(t("label_parakeet_cpp_backend"))
        self._parakeet_cpp_backend = QComboBox()
        for label, value in (("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda"), ("Vulkan", "vulkan")):
            self._parakeet_cpp_backend.addItem(label, value)
        parakeet_backend = str(s.get("parakeet_cpp_backend", "auto")).lower()
        parakeet_backend_idx = self._parakeet_cpp_backend.findData(parakeet_backend)
        if parakeet_backend_idx >= 0:
            self._parakeet_cpp_backend.setCurrentIndex(parakeet_backend_idx)
        self._parakeet_cpp_backend.currentIndexChanged.connect(
            self._on_parakeet_cpp_setting_changed
        )
        asr_layout.addWidget(self._parakeet_cpp_backend_label, 11, 0)
        asr_layout.addWidget(self._parakeet_cpp_backend, 11, 1)

        self._parakeet_cpp_decoder_label = QLabel(t("label_parakeet_cpp_decoder"))
        self._parakeet_cpp_decoder = QComboBox()
        for label, value in (("Auto", "auto"), ("CTC", "ctc"), ("TDT/RNNT", "tdt")):
            self._parakeet_cpp_decoder.addItem(label, value)
        parakeet_decoder = str(s.get("parakeet_cpp_decoder", "auto")).lower()
        parakeet_decoder_idx = self._parakeet_cpp_decoder.findData(parakeet_decoder)
        if parakeet_decoder_idx >= 0:
            self._parakeet_cpp_decoder.setCurrentIndex(parakeet_decoder_idx)
        self._parakeet_cpp_decoder.currentIndexChanged.connect(
            self._on_parakeet_cpp_setting_changed
        )
        asr_layout.addWidget(self._parakeet_cpp_decoder_label, 12, 0)
        asr_layout.addWidget(self._parakeet_cpp_decoder, 12, 1)

        self._parakeet_cpp_word_timestamps = QCheckBox(
            t("label_parakeet_cpp_word_timestamps")
        )
        self._parakeet_cpp_word_timestamps.setChecked(
            bool(s.get("parakeet_cpp_word_timestamps", True))
        )
        self._parakeet_cpp_word_timestamps.toggled.connect(
            self._on_parakeet_cpp_setting_changed
        )
        asr_layout.addWidget(self._parakeet_cpp_word_timestamps, 13, 1)

        self._whisper_pad_label = QLabel(t("label_whisper_padding"))
        self._whisper_pad_seconds = QDoubleSpinBox()
        self._whisper_pad_seconds.setRange(0.0, 5.0)
        self._whisper_pad_seconds.setDecimals(2)
        self._whisper_pad_seconds.setSingleStep(0.1)
        try:
            whisper_pad_seconds = float(s.get("whisper_pad_seconds", 0.5))
        except (TypeError, ValueError):
            whisper_pad_seconds = 0.5
        self._whisper_pad_seconds.setValue(whisper_pad_seconds)
        self._whisper_pad_seconds.setSuffix(" s")
        self._whisper_pad_seconds.setSpecialValueText(t("whisper_padding_off"))
        self._whisper_pad_seconds.setToolTip(t("whisper_padding_tooltip"))
        asr_layout.addWidget(self._whisper_pad_label, 14, 0)
        asr_layout.addWidget(self._whisper_pad_seconds, 14, 1)
        self._whisper_pad_seconds.valueChanged.connect(self._auto_save)

        self._sensevoice_pad_label = QLabel(t("label_sensevoice_padding"))
        self._sensevoice_pad_seconds = QDoubleSpinBox()
        self._sensevoice_pad_seconds.setRange(0.0, 5.0)
        self._sensevoice_pad_seconds.setDecimals(2)
        self._sensevoice_pad_seconds.setSingleStep(0.1)
        try:
            sensevoice_pad_seconds = float(s.get("sensevoice_pad_seconds", 0.5))
        except (TypeError, ValueError):
            sensevoice_pad_seconds = 0.5
        self._sensevoice_pad_seconds.setValue(sensevoice_pad_seconds)
        self._sensevoice_pad_seconds.setSuffix(" s")
        self._sensevoice_pad_seconds.setSpecialValueText(t("sensevoice_padding_off"))
        self._sensevoice_pad_seconds.setToolTip(t("sensevoice_padding_tooltip"))
        asr_layout.addWidget(self._sensevoice_pad_label, 15, 0)
        asr_layout.addWidget(self._sensevoice_pad_seconds, 15, 1)
        self._sensevoice_pad_seconds.valueChanged.connect(self._auto_save)

        self._audio_device = QComboBox()
        self._audio_device.addItem(t("audio_disabled"))
        self._audio_device.addItem(t("system_default"))
        try:
            from audio_capture import list_output_devices

            for name in list_output_devices():
                self._audio_device.addItem(name)
        except Exception:
            pass
        saved_audio = s.get("audio_device")
        if saved_audio == "__disabled__":
            self._audio_device.setCurrentIndex(0)
        elif saved_audio:
            idx = self._audio_device.findText(saved_audio)
            if idx >= 0:
                self._audio_device.setCurrentIndex(idx)
        else:
            self._audio_device.setCurrentIndex(1)  # system default
        asr_layout.addWidget(QLabel(t("label_audio")), 16, 0)
        asr_layout.addWidget(self._audio_device, 16, 1)
        self._audio_device.currentIndexChanged.connect(self._auto_save)

        self._mic_device = QComboBox()
        self._mic_device.addItem(t("mic_disabled"))
        self._mic_device.addItem(t("system_default"))
        try:
            from audio_capture import list_input_devices

            for name in list_input_devices():
                self._mic_device.addItem(name)
        except Exception:
            pass
        saved_mic = s.get("mic_device")
        if saved_mic:
            if saved_mic in ("__default__", "default"):
                self._mic_device.setCurrentIndex(1)
            else:
                idx = self._mic_device.findText(saved_mic)
                if idx >= 0:
                    self._mic_device.setCurrentIndex(idx)
        asr_layout.addWidget(QLabel(t("label_mic")), 17, 0)
        asr_layout.addWidget(self._mic_device, 17, 1)
        self._mic_device.currentIndexChanged.connect(self._auto_save)

        self._hub_combo = QComboBox()
        self._hub_combo.addItems([t("hub_modelscope"), t("hub_huggingface")])
        saved_hub = s.get("hub", "ms")
        self._hub_combo.setCurrentIndex(0 if saved_hub == "ms" else 1)
        asr_layout.addWidget(QLabel(t("label_hub")), 18, 0)
        asr_layout.addWidget(self._hub_combo, 18, 1)
        self._hub_combo.currentIndexChanged.connect(self._auto_save)

        self._ui_lang_combo = QComboBox()
        self._ui_lang_combo.addItems(["English", "中文"])
        from i18n import get_lang

        saved_lang = s.get("ui_lang", get_lang())
        self._ui_lang_combo.setCurrentIndex(0 if saved_lang == "en" else 1)
        asr_layout.addWidget(QLabel(t("label_ui_lang")), 19, 0)
        asr_layout.addWidget(self._ui_lang_combo, 19, 1)
        self._ui_lang_combo.currentIndexChanged.connect(self._on_ui_lang_changed)

        layout.addWidget(asr_group)

        # Whisper model download — only visible when engine is Whisper
        self._whisper_group = QGroupBox(t("group_download_whisper"))
        whisper_layout = QHBoxLayout(self._whisper_group)
        self._whisper_size_combo = QComboBox()
        saved_size = s.get(
            "whisper_model_size", self._config["asr"].get("model_size", "medium")
        )
        self._populate_whisper_models(saved_size)
        self._whisper_size_combo.currentIndexChanged.connect(
            self._on_whisper_size_changed
        )
        whisper_layout.addWidget(self._whisper_size_combo)
        self._whisper_status = QLabel("")
        self._whisper_status.setStyleSheet("color: #888; font-size: 11px;")
        whisper_layout.addWidget(self._whisper_status, 1)
        self._whisper_dl_btn = QPushButton(t("btn_download_whisper"))
        self._whisper_dl_btn.clicked.connect(self._download_whisper)
        whisper_layout.addWidget(self._whisper_dl_btn)
        layout.addWidget(self._whisper_group)
        self._whisper_group.setVisible(self._selected_asr_engine() == "whisper")

        self._crispasr_group = QGroupBox(t("group_crispasr_model"))
        crispasr_layout = QHBoxLayout(self._crispasr_group)
        self._crispasr_model_combo = QComboBox()
        saved_crispasr_model = s.get("crispasr_model", "")
        self._populate_crispasr_models(saved_crispasr_model)
        self._crispasr_model_combo.currentIndexChanged.connect(
            self._on_crispasr_model_changed
        )
        crispasr_layout.addWidget(self._crispasr_model_combo)
        self._crispasr_status = QLabel("")
        self._crispasr_status.setStyleSheet("color: #888; font-size: 11px;")
        crispasr_layout.addWidget(self._crispasr_status, 1)
        layout.addWidget(self._crispasr_group)
        self._crispasr_group.setVisible(self._selected_asr_engine() == "crispasr")

        self._sherpa_onnx_group = QGroupBox(t("group_sherpa_onnx_models"))
        sherpa_layout = QHBoxLayout(self._sherpa_onnx_group)
        self._sherpa_onnx_model_combo = QComboBox()
        saved_sherpa_model = s.get("sherpa_onnx_model", "")
        self._populate_sherpa_onnx_models(saved_sherpa_model)
        self._sherpa_onnx_model_combo.currentIndexChanged.connect(
            self._on_sherpa_onnx_model_changed
        )
        sherpa_layout.addWidget(self._sherpa_onnx_model_combo)
        self._sherpa_onnx_status = QLabel("")
        self._sherpa_onnx_status.setStyleSheet("color: #888; font-size: 11px;")
        sherpa_layout.addWidget(self._sherpa_onnx_status, 1)
        self._sherpa_onnx_refresh_btn = QPushButton(t("btn_refresh_sherpa_onnx_models"))
        self._sherpa_onnx_refresh_btn.clicked.connect(
            self._refresh_sherpa_onnx_models
        )
        sherpa_layout.addWidget(self._sherpa_onnx_refresh_btn)
        layout.addWidget(self._sherpa_onnx_group)
        self._sherpa_onnx_group.setVisible(
            self._selected_asr_engine() == "sherpa-onnx"
        )

        self._parakeet_cpp_model_group = QGroupBox(t("group_parakeet_cpp_model"))
        parakeet_model_layout = QHBoxLayout(self._parakeet_cpp_model_group)
        self._parakeet_cpp_model_combo = QComboBox()
        saved_parakeet_model = s.get("parakeet_cpp_model", "")
        self._populate_parakeet_cpp_models(saved_parakeet_model)
        self._parakeet_cpp_model_combo.currentIndexChanged.connect(
            self._on_parakeet_cpp_model_changed
        )
        parakeet_model_layout.addWidget(self._parakeet_cpp_model_combo)
        self._parakeet_cpp_model_status = QLabel("")
        self._parakeet_cpp_model_status.setStyleSheet("color: #888; font-size: 11px;")
        parakeet_model_layout.addWidget(self._parakeet_cpp_model_status, 1)
        self._parakeet_cpp_model_refresh_btn = QPushButton(
            t("btn_refresh_parakeet_cpp_models")
        )
        self._parakeet_cpp_model_refresh_btn.clicked.connect(
            self._refresh_parakeet_cpp_models
        )
        parakeet_model_layout.addWidget(self._parakeet_cpp_model_refresh_btn)
        layout.addWidget(self._parakeet_cpp_model_group)
        self._parakeet_cpp_model_group.setVisible(
            self._selected_asr_engine() == "parakeet-cpp"
        )

        self._parakeet_cpp_runtime_group = QGroupBox(t("group_parakeet_cpp_runtime"))
        parakeet_runtime_layout = QHBoxLayout(self._parakeet_cpp_runtime_group)
        self._parakeet_cpp_runtime_combo = QComboBox()
        saved_parakeet_runtime = s.get("parakeet_cpp_runtime_dir", "")
        self._populate_parakeet_cpp_runtimes(saved_parakeet_runtime)
        self._parakeet_cpp_runtime_combo.currentIndexChanged.connect(
            self._on_parakeet_cpp_runtime_changed
        )
        parakeet_runtime_layout.addWidget(self._parakeet_cpp_runtime_combo)
        self._parakeet_cpp_runtime_status = QLabel("")
        self._parakeet_cpp_runtime_status.setStyleSheet(
            "color: #888; font-size: 11px;"
        )
        parakeet_runtime_layout.addWidget(self._parakeet_cpp_runtime_status, 1)
        self._parakeet_cpp_runtime_refresh_btn = QPushButton(
            t("btn_refresh_parakeet_cpp_runtimes")
        )
        self._parakeet_cpp_runtime_refresh_btn.clicked.connect(
            self._refresh_parakeet_cpp_runtimes
        )
        parakeet_runtime_layout.addWidget(self._parakeet_cpp_runtime_refresh_btn)
        layout.addWidget(self._parakeet_cpp_runtime_group)
        self._parakeet_cpp_runtime_group.setVisible(
            self._selected_asr_engine() == "parakeet-cpp"
        )

        # Remote ASR server URL — only visible when engine is Remote Whisper
        self._remote_group = QGroupBox("Remote ASR Server")
        remote_layout = QHBoxLayout(self._remote_group)
        remote_layout.addWidget(QLabel("URL"))
        self._remote_url_edit = QLineEdit(
            s.get("remote_asr_url", "http://127.0.0.1:8765")
        )
        self._remote_url_edit.setPlaceholderText("http://127.0.0.1:8765")
        self._remote_url_edit.editingFinished.connect(self._auto_save)
        remote_layout.addWidget(self._remote_url_edit, 1)
        layout.addWidget(self._remote_group)
        self._remote_group.setVisible(
            self._selected_asr_engine() == "remote-whisper"
        )

        self._asr_engine.currentIndexChanged.connect(
            self._on_asr_engine_changed
        )
        self._on_asr_engine_changed(engine_idx)
        self._update_whisper_size_label()
        self._update_crispasr_status()
        self._update_sherpa_onnx_status()
        self._update_parakeet_cpp_model_status()
        self._update_parakeet_cpp_runtime_status()

        layout.addStretch()
        return widget

    # ── VAD Tab ──

    def _create_vad_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings

        mode_group = QGroupBox(t("group_vad_mode"))
        mode_layout = QVBoxLayout(mode_group)
        self._vad_mode = QComboBox()
        for label, key in (
            (t("vad_silero"), "silero"),
            (t("vad_firered"), "firered"),
            (t("vad_energy"), "energy"),
            (t("vad_disabled"), "disabled"),
        ):
            self._vad_mode.addItem(label, key)
        mode_idx = self._vad_mode.findData(s.get("vad_mode", "silero"))
        if mode_idx < 0:
            mode_idx = self._vad_mode.findData("silero")
        self._vad_mode.setCurrentIndex(mode_idx)
        self._vad_mode.currentIndexChanged.connect(self._on_vad_mode_changed)
        self._vad_mode.currentIndexChanged.connect(self._auto_save)
        mode_layout.addWidget(self._vad_mode)
        layout.addWidget(mode_group)

        self._neural_vad_group = QGroupBox(t("group_neural_vad_threshold"))
        silero_layout = QGridLayout(self._neural_vad_group)
        self._vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._vad_threshold_slider.setRange(0, 100)
        vad_pct = int(s.get("vad_threshold", 0.5) * 100)
        self._vad_threshold_slider.setValue(vad_pct)
        self._vad_threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._vad_threshold_slider.sliderReleased.connect(self._auto_save)
        self._vad_threshold_label = QLabel(f"{vad_pct}%")
        self._vad_threshold_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        silero_layout.addWidget(QLabel(t("label_threshold")), 0, 0)
        silero_layout.addWidget(self._vad_threshold_slider, 0, 1)
        silero_layout.addWidget(self._vad_threshold_label, 0, 2)
        layout.addWidget(self._neural_vad_group)

        self._firered_vad_group = QGroupBox(t("group_firered_vad"))
        firered_layout = QGridLayout(self._firered_vad_group)
        firered_layout.setColumnStretch(1, 1)
        self._firered_vad_model_combo = QComboBox()
        saved_firered_model = s.get("firered_vad_model", "")
        self._populate_firered_vad_models(saved_firered_model)
        self._firered_vad_model_combo.currentIndexChanged.connect(
            self._on_firered_vad_model_changed
        )
        self._firered_vad_status = QLabel("")
        self._firered_vad_status.setStyleSheet("color: #888; font-size: 11px;")
        self._firered_vad_refresh_btn = QPushButton(
            t("btn_refresh_firered_vad_models")
        )
        self._firered_vad_refresh_btn.clicked.connect(
            self._refresh_firered_vad_models
        )
        self._firered_vad_smooth_window = QSpinBox()
        self._firered_vad_smooth_window.setRange(1, 30)
        self._firered_vad_smooth_window.setValue(
            int(s.get("firered_vad_smooth_window_size", 5) or 5)
        )
        self._firered_vad_smooth_window.valueChanged.connect(
            self._on_firered_vad_setting_changed
        )
        self._firered_vad_use_gpu = QCheckBox(t("label_firered_vad_use_gpu"))
        self._firered_vad_use_gpu.setChecked(bool(s.get("firered_vad_use_gpu", False)))
        self._firered_vad_use_gpu.toggled.connect(
            self._on_firered_vad_setting_changed
        )
        firered_layout.addWidget(QLabel(t("label_firered_vad_model")), 0, 0)
        firered_layout.addWidget(self._firered_vad_model_combo, 0, 1)
        firered_layout.addWidget(self._firered_vad_refresh_btn, 0, 2)
        firered_layout.addWidget(self._firered_vad_status, 1, 1, 1, 2)
        firered_layout.addWidget(QLabel(t("label_firered_vad_smooth_window")), 2, 0)
        firered_layout.addWidget(self._firered_vad_smooth_window, 2, 1)
        firered_layout.addWidget(self._firered_vad_use_gpu, 3, 1)
        layout.addWidget(self._firered_vad_group)

        self._energy_group = QGroupBox(t("group_energy_threshold"))
        energy_layout = QGridLayout(self._energy_group)
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
        layout.addWidget(self._energy_group)

        timing_group = QGroupBox(t("group_timing"))
        timing_layout = QGridLayout(timing_group)
        timing_layout.setColumnStretch(0, 1)
        timing_layout.setColumnMinimumWidth(1, 180)
        self._min_speech = QDoubleSpinBox()
        self._min_speech.setRange(0.1, 5.0)
        self._min_speech.setSingleStep(0.1)
        self._min_speech.setValue(s.get("min_speech_duration", 2.0))
        self._min_speech.setSuffix(" s")
        self._min_speech.valueChanged.connect(self._on_timing_changed)
        self._min_speech.valueChanged.connect(self._auto_save)
        self._max_speech = QDoubleSpinBox()
        self._max_speech.setRange(2.0, 30.0)
        self._max_speech.setSingleStep(1.0)
        self._max_speech.setValue(s.get("max_speech_duration", 6.0))
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

        self._incremental_asr_cb = QCheckBox(t("label_incremental_asr"))
        self._incremental_asr_cb.setToolTip(t("incremental_asr_tooltip"))
        self._incremental_asr_cb.setChecked(s.get("incremental_asr", False))
        self._incremental_asr_cb.toggled.connect(self._on_timing_changed)
        self._incremental_asr_cb.toggled.connect(self._auto_save)
        timing_layout.addWidget(self._incremental_asr_cb, 4, 0)

        self._interim_interval_spin = QDoubleSpinBox()
        self._interim_interval_spin.setRange(1.0, 10.0)
        self._interim_interval_spin.setSingleStep(0.5)
        self._interim_interval_spin.setValue(s.get("interim_interval", 2.0))
        self._interim_interval_spin.setSuffix(" s")
        self._interim_interval_spin.setEnabled(s.get("incremental_asr", False))
        self._interim_interval_spin.valueChanged.connect(self._on_timing_changed)
        self._interim_interval_spin.valueChanged.connect(self._auto_save)
        self._incremental_asr_cb.toggled.connect(self._interim_interval_spin.setEnabled)
        timing_layout.addWidget(QLabel(t("label_interim_interval")), 5, 0)
        timing_layout.addWidget(self._interim_interval_spin, 5, 1)

        layout.addWidget(timing_group)

        self._update_firered_vad_status()
        self._update_vad_detail_visibility()
        layout.addStretch()
        return widget

    # ── Translation Tab ──

    def _create_translation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings

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

        from translator import DEFAULT_PROMPT, PROMPT_PRESETS

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(t("label_prompt_preset")))
        self._prompt_preset = QComboBox()
        self._prompt_preset.addItem(t("prompt_daily"), "daily")
        self._prompt_preset.addItem(t("prompt_esports"), "esports")
        self._prompt_preset.addItem(t("prompt_anime"), "anime")
        self._prompt_preset.addItem(t("prompt_webid"), "webid")
        self._prompt_preset.addItem(t("prompt_custom"), "custom")

        current_prompt = s.get("system_prompt", DEFAULT_PROMPT)
        preset_idx = 4  # default to custom
        for i, key in enumerate(["daily", "esports", "anime", "webid"]):
            if current_prompt.strip() == PROMPT_PRESETS[key].strip():
                preset_idx = i
                break
        if current_prompt.strip() == DEFAULT_PROMPT.strip():
            preset_idx = 0
        self._prompt_preset.setCurrentIndex(preset_idx)
        self._prompt_preset.currentIndexChanged.connect(self._on_prompt_preset_changed)
        preset_row.addWidget(self._prompt_preset, 1)
        prompt_layout.addLayout(preset_row)

        # Prompt text editor
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setFont(QFont("Consolas", 9))
        self._prompt_edit.setMaximumHeight(100)
        self._prompt_edit.setPlainText(current_prompt)
        self._prompt_debounce = QTimer()
        self._prompt_debounce.setSingleShot(True)
        self._prompt_debounce.setInterval(600)
        self._prompt_debounce.timeout.connect(self._apply_prompt)
        self._prompt_edit.textChanged.connect(self._prompt_debounce.start)
        prompt_layout.addWidget(self._prompt_edit)
        layout.addWidget(prompt_group)

        net_group = QGroupBox(t("group_network"))
        net_layout = QGridLayout(net_group)
        net_layout.setColumnStretch(0, 1)
        net_layout.setColumnMinimumWidth(1, 180)
        net_layout.addWidget(QLabel(t("label_timeout")), 0, 0)
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(1, 60)
        self._timeout_spin.setValue(s.get("timeout", 5))
        self._timeout_spin.setSuffix(" s")
        self._timeout_spin.valueChanged.connect(
            lambda v: self._current_settings.update({"timeout": v})
        )
        self._timeout_spin.valueChanged.connect(self._auto_save)
        net_layout.addWidget(self._timeout_spin, 0, 1)
        layout.addWidget(net_group)

        layout.addStretch()
        return widget

    # ── Style Tab ──

    def _create_style_tab(self):
        from subtitle_overlay import DEFAULT_STYLE

        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings.get("style", dict(DEFAULT_STYLE))

        # Preset group
        preset_group = QGroupBox(t("group_preset"))
        preset_layout = QHBoxLayout(preset_group)
        self._style_preset = QComboBox()
        preset_names = [
            ("default", t("preset_default")),
            ("transparent", t("preset_transparent")),
            ("compact", t("preset_compact")),
            ("light", t("preset_light")),
            ("dracula", t("preset_dracula")),
            ("nord", t("preset_nord")),
            ("monokai", t("preset_monokai")),
            ("solarized", t("preset_solarized")),
            ("gruvbox", t("preset_gruvbox")),
            ("tokyo_night", t("preset_tokyo_night")),
            ("catppuccin", t("preset_catppuccin")),
            ("one_dark", t("preset_one_dark")),
            ("everforest", t("preset_everforest")),
            ("kanagawa", t("preset_kanagawa")),
            ("custom", t("preset_custom")),
        ]
        self._preset_keys = [k for k, _ in preset_names]
        for _, label in preset_names:
            self._style_preset.addItem(label)
        current_preset = s.get("preset", "default")
        if current_preset in self._preset_keys:
            self._style_preset.setCurrentIndex(self._preset_keys.index(current_preset))
        else:
            self._style_preset.setCurrentIndex(5)  # custom
        self._style_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._style_preset, 1)
        reset_btn = QPushButton(t("btn_reset_style"))
        reset_btn.clicked.connect(self._reset_style)
        preset_layout.addWidget(reset_btn)
        reset_pos_btn = QPushButton(t("btn_reset_positions"))
        reset_pos_btn.clicked.connect(self.reset_positions.emit)
        preset_layout.addWidget(reset_pos_btn)
        layout.addWidget(preset_group)

        # Background group
        bg_group = QGroupBox(t("group_background"))
        bg_layout = QGridLayout(bg_group)
        bg_layout.setColumnStretch(0, 1)
        bg_layout.setColumnMinimumWidth(1, 180)

        bg_layout.addWidget(QLabel(t("label_bg_color")), 0, 0)
        self._bg_color_btn = self._make_color_btn(
            s.get("bg_color", DEFAULT_STYLE["bg_color"])
        )
        self._bg_color_btn.clicked.connect(lambda: self._pick_color(self._bg_color_btn))
        bg_layout.addWidget(self._bg_color_btn, 0, 1)

        bg_layout.addWidget(QLabel(t("label_bg_opacity")), 1, 0)
        self._bg_opacity = QSpinBox()
        self._bg_opacity.setRange(0, 100)
        self._bg_opacity.setSuffix("%")
        self._bg_opacity.setValue(round(s.get("bg_opacity", DEFAULT_STYLE["bg_opacity"]) / 255 * 100))
        self._bg_opacity.valueChanged.connect(self._on_style_value_changed)
        self._bg_opacity.valueChanged.connect(self._auto_save)
        bg_layout.addWidget(self._bg_opacity, 1, 1)

        bg_layout.addWidget(QLabel(t("label_header_color")), 2, 0)
        self._header_color_btn = self._make_color_btn(
            s.get("header_color", DEFAULT_STYLE["header_color"])
        )
        self._header_color_btn.clicked.connect(
            lambda: self._pick_color(self._header_color_btn)
        )
        bg_layout.addWidget(self._header_color_btn, 2, 1)

        bg_layout.addWidget(QLabel(t("label_header_opacity")), 3, 0)
        self._header_opacity = QSpinBox()
        self._header_opacity.setRange(0, 100)
        self._header_opacity.setSuffix("%")
        self._header_opacity.setValue(round(s.get("header_opacity", DEFAULT_STYLE["header_opacity"]) / 255 * 100))
        self._header_opacity.valueChanged.connect(self._on_style_value_changed)
        self._header_opacity.valueChanged.connect(self._auto_save)
        bg_layout.addWidget(self._header_opacity, 3, 1)

        bg_layout.addWidget(QLabel(t("label_border_radius")), 4, 0)
        self._border_radius = QSpinBox()
        self._border_radius.setRange(0, 30)
        self._border_radius.setValue(
            s.get("border_radius", DEFAULT_STYLE["border_radius"])
        )
        self._border_radius.setSuffix(" px")
        self._border_radius.valueChanged.connect(self._on_style_value_changed)
        self._border_radius.valueChanged.connect(self._auto_save)
        bg_layout.addWidget(self._border_radius, 4, 1)

        layout.addWidget(bg_group)

        # Text group
        text_group = QGroupBox(t("group_text"))
        text_layout = QGridLayout(text_group)
        text_layout.setColumnStretch(0, 1)
        text_layout.setColumnMinimumWidth(1, 180)

        text_layout.addWidget(QLabel(t("label_original_font")), 0, 0)
        self._orig_font_combo = QFontComboBox()
        self._orig_font_combo.setCurrentFont(
            QFont(s.get("original_font_family", DEFAULT_STYLE["original_font_family"]))
        )
        self._orig_font_combo.currentFontChanged.connect(self._on_style_value_changed)
        self._orig_font_combo.currentFontChanged.connect(self._auto_save)
        text_layout.addWidget(self._orig_font_combo, 0, 1)

        text_layout.addWidget(QLabel(t("label_original_font_size")), 1, 0)
        self._orig_font_size = QSpinBox()
        self._orig_font_size.setRange(6, 24)
        self._orig_font_size.setValue(
            s.get("original_font_size", DEFAULT_STYLE["original_font_size"])
        )
        self._orig_font_size.setSuffix(" pt")
        self._orig_font_size.valueChanged.connect(self._on_style_value_changed)
        self._orig_font_size.valueChanged.connect(self._auto_save)
        text_layout.addWidget(self._orig_font_size, 1, 1)

        text_layout.addWidget(QLabel(t("label_original_color")), 2, 0)
        self._orig_color_btn = self._make_color_btn(
            s.get("original_color", DEFAULT_STYLE["original_color"])
        )
        self._orig_color_btn.clicked.connect(
            lambda: self._pick_color(self._orig_color_btn)
        )
        text_layout.addWidget(self._orig_color_btn, 2, 1)

        text_layout.addWidget(QLabel(t("label_translation_font")), 3, 0)
        self._trans_font_combo = QFontComboBox()
        self._trans_font_combo.setCurrentFont(
            QFont(
                s.get(
                    "translation_font_family", DEFAULT_STYLE["translation_font_family"]
                )
            )
        )
        self._trans_font_combo.currentFontChanged.connect(self._on_style_value_changed)
        self._trans_font_combo.currentFontChanged.connect(self._auto_save)
        text_layout.addWidget(self._trans_font_combo, 3, 1)

        text_layout.addWidget(QLabel(t("label_translation_font_size")), 4, 0)
        self._trans_font_size = QSpinBox()
        self._trans_font_size.setRange(6, 24)
        self._trans_font_size.setValue(
            s.get("translation_font_size", DEFAULT_STYLE["translation_font_size"])
        )
        self._trans_font_size.setSuffix(" pt")
        self._trans_font_size.valueChanged.connect(self._on_style_value_changed)
        self._trans_font_size.valueChanged.connect(self._auto_save)
        text_layout.addWidget(self._trans_font_size, 4, 1)

        text_layout.addWidget(QLabel(t("label_translation_color")), 5, 0)
        self._trans_color_btn = self._make_color_btn(
            s.get("translation_color", DEFAULT_STYLE["translation_color"])
        )
        self._trans_color_btn.clicked.connect(
            lambda: self._pick_color(self._trans_color_btn)
        )
        text_layout.addWidget(self._trans_color_btn, 5, 1)

        text_layout.addWidget(QLabel(t("label_timestamp_color")), 6, 0)
        self._ts_color_btn = self._make_color_btn(
            s.get("timestamp_color", DEFAULT_STYLE["timestamp_color"])
        )
        self._ts_color_btn.clicked.connect(lambda: self._pick_color(self._ts_color_btn))
        text_layout.addWidget(self._ts_color_btn, 6, 1)

        layout.addWidget(text_group)

        # Window group
        win_group = QGroupBox(t("group_window"))
        win_layout = QGridLayout(win_group)
        win_layout.setColumnStretch(0, 1)
        win_layout.setColumnMinimumWidth(1, 180)
        win_layout.addWidget(QLabel(t("label_window_opacity")), 0, 0)
        self._window_opacity = QSpinBox()
        self._window_opacity.setRange(30, 100)
        self._window_opacity.setSuffix("%")
        self._window_opacity.setValue(s.get("window_opacity", DEFAULT_STYLE["window_opacity"]))
        self._window_opacity.valueChanged.connect(self._on_style_value_changed)
        self._window_opacity.valueChanged.connect(self._auto_save)
        win_layout.addWidget(self._window_opacity, 0, 1)
        layout.addWidget(win_group)

        layout.addStretch()
        return widget

    def _make_color_btn(self, color: str) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(60, 24)
        btn.setProperty("hex_color", color)
        btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #888; border-radius: 3px;"
        )
        return btn

    def _pick_color(self, btn: QPushButton):
        from PyQt6.QtGui import QColor as _QColor

        current = _QColor(btn.property("hex_color"))
        color = QColorDialog.getColor(current, self)
        if color.isValid():
            hex_c = color.name()
            btn.setProperty("hex_color", hex_c)
            btn.setStyleSheet(
                f"background-color: {hex_c}; border: 1px solid #888; border-radius: 3px;"
            )
            self._on_style_value_changed()
            self._auto_save()

    def _collect_style(self) -> dict:
        return {
            "preset": self._preset_keys[self._style_preset.currentIndex()],
            "bg_color": self._bg_color_btn.property("hex_color"),
            "bg_opacity": round(self._bg_opacity.value() / 100 * 255),
            "header_color": self._header_color_btn.property("hex_color"),
            "header_opacity": round(self._header_opacity.value() / 100 * 255),
            "border_radius": self._border_radius.value(),
            "original_font_family": self._orig_font_combo.currentFont().family(),
            "translation_font_family": self._trans_font_combo.currentFont().family(),
            "original_font_size": self._orig_font_size.value(),
            "translation_font_size": self._trans_font_size.value(),
            "original_color": self._orig_color_btn.property("hex_color"),
            "translation_color": self._trans_color_btn.property("hex_color"),
            "timestamp_color": self._ts_color_btn.property("hex_color"),
            "window_opacity": self._window_opacity.value(),
        }

    def _apply_style_to_controls(self, s: dict):
        """Update all style controls to match a style dict, without triggering auto-save."""
        self._bg_color_btn.setProperty("hex_color", s["bg_color"])
        self._bg_color_btn.setStyleSheet(
            f"background-color: {s['bg_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._bg_opacity.setValue(round(s["bg_opacity"] / 255 * 100))
        self._header_color_btn.setProperty("hex_color", s["header_color"])
        self._header_color_btn.setStyleSheet(
            f"background-color: {s['header_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._header_opacity.setValue(round(s["header_opacity"] / 255 * 100))
        self._border_radius.setValue(s["border_radius"])
        self._orig_font_combo.setCurrentFont(QFont(s["original_font_family"]))
        self._trans_font_combo.setCurrentFont(QFont(s["translation_font_family"]))
        self._orig_font_size.setValue(s["original_font_size"])
        self._trans_font_size.setValue(s["translation_font_size"])
        self._orig_color_btn.setProperty("hex_color", s["original_color"])
        self._orig_color_btn.setStyleSheet(
            f"background-color: {s['original_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._trans_color_btn.setProperty("hex_color", s["translation_color"])
        self._trans_color_btn.setStyleSheet(
            f"background-color: {s['translation_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._ts_color_btn.setProperty("hex_color", s["timestamp_color"])
        self._ts_color_btn.setStyleSheet(
            f"background-color: {s['timestamp_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._window_opacity.setValue(s["window_opacity"])

    def _on_preset_changed(self, index):
        from subtitle_overlay import STYLE_PRESETS

        key = self._preset_keys[index]
        if key == "custom":
            return
        preset = STYLE_PRESETS.get(key)
        if not preset:
            return
        self._block_style_signals(True)
        self._apply_style_to_controls(preset)
        self._block_style_signals(False)
        self._auto_save()

    def _on_style_value_changed(self, *_args):
        """When any style control changes manually, switch preset to Custom."""
        custom_idx = len(self._preset_keys) - 1
        if self._style_preset.currentIndex() != custom_idx:
            self._style_preset.blockSignals(True)
            self._style_preset.setCurrentIndex(custom_idx)
            self._style_preset.blockSignals(False)
        self._auto_save()

    def _reset_style(self):
        from subtitle_overlay import DEFAULT_STYLE

        self._style_preset.blockSignals(True)
        self._style_preset.setCurrentIndex(0)  # default
        self._style_preset.blockSignals(False)
        self._block_style_signals(True)
        self._apply_style_to_controls(DEFAULT_STYLE)
        self._block_style_signals(False)
        self._auto_save()

    def _block_style_signals(self, block: bool):
        for w in (
            self._bg_opacity,
            self._header_opacity,
            self._border_radius,
            self._orig_font_combo,
            self._trans_font_combo,
            self._orig_font_size,
            self._trans_font_size,
            self._window_opacity,
        ):
            w.blockSignals(block)

    # ── Subtitle Tab ──

    def _create_subtitle_tab(self):
        subtitle_settings = self._current_settings.get("subtitle_mode") or {}
        self._subtitle_widget = SubtitleSettingsWidget(subtitle_settings)
        self._subtitle_widget.settings_changed.connect(self._on_subtitle_settings_changed)
        return self._subtitle_widget

    def _on_subtitle_settings_changed(self, s):
        self._current_settings["subtitle_mode"] = s
        self._auto_save()
        self.subtitle_settings_changed.emit(s)

    def update_subtitle_settings(self, s):
        self._current_settings["subtitle_mode"] = s
        self._subtitle_widget.update_settings(s)

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

    def _create_changelog_tab(self):
        from dialogs import _load_latest_changelog
        widget = QWidget()
        layout = QVBoxLayout(widget)
        _, html = _load_latest_changelog()
        from PyQt6.QtWidgets import QTextBrowser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        browser.setFont(QFont("Microsoft YaHei UI", 10))
        layout.addWidget(browser)
        return widget

    def _create_cache_tab(self):
        from PyQt6.QtWidgets import QCheckBox

        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings

        # Transcript auto-save group
        ts_group = QGroupBox(t("group_transcript"))
        ts_layout = QHBoxLayout(ts_group)
        self._auto_save_transcript_cb = QCheckBox(t("label_auto_save_transcript"))
        self._auto_save_transcript_cb.setToolTip(t("auto_save_transcript_tooltip"))
        self._auto_save_transcript_cb.setChecked(s.get("auto_save_transcript", True))
        self._auto_save_transcript_cb.toggled.connect(self._auto_save)
        ts_layout.addWidget(self._auto_save_transcript_cb, 1)
        ts_open_btn = QPushButton(t("btn_open_transcripts"))
        ts_open_btn.clicked.connect(self._open_transcripts_folder)
        ts_layout.addWidget(ts_open_btn)
        layout.addWidget(ts_group)

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

    def _open_transcripts_folder(self):
        from pathlib import Path
        ts_dir = Path(__file__).parent / "transcripts"
        ts_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(ts_dir))

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
                path_obj = Path(path)
                if path_obj.is_file():
                    path_obj.unlink()
                else:
                    shutil.rmtree(path_obj)
                log.info(f"Deleted: {path}")
            except Exception as e:
                log.error(f"Failed to delete {path}: {e}")
        QApplication.instance().quit()

    def _get_asr_lang_code(self) -> str:
        """Get the language code from the ASR language combo (stored as userData)."""
        return self._asr_lang.currentData() or "auto"

    def _selected_vad_mode(self) -> str:
        value = self._vad_mode.currentData()
        return str(value) if value else "silero"

    def _selected_asr_engine(self) -> str:
        value = self._asr_engine.currentData()
        return str(value) if value else "funasr"

    def _on_asr_engine_changed(self, index):
        engine = self._selected_asr_engine()
        is_whisper = engine == "whisper"
        is_funasr = engine == "funasr"
        is_remote = engine == "remote-whisper"
        is_crispasr = engine == "crispasr"
        is_sherpa_onnx = engine == "sherpa-onnx"
        is_parakeet_cpp = engine == "parakeet-cpp"
        self._whisper_group.setVisible(is_whisper)
        if hasattr(self, "_crispasr_group"):
            self._crispasr_group.setVisible(is_crispasr)
        if hasattr(self, "_sherpa_onnx_group"):
            self._sherpa_onnx_group.setVisible(is_sherpa_onnx)
        if hasattr(self, "_parakeet_cpp_model_group"):
            self._parakeet_cpp_model_group.setVisible(is_parakeet_cpp)
        if hasattr(self, "_parakeet_cpp_runtime_group"):
            self._parakeet_cpp_runtime_group.setVisible(is_parakeet_cpp)
        if hasattr(self, "_funasr_model_combo"):
            self._funasr_model_label.setVisible(is_funasr)
            self._funasr_model_combo.setVisible(is_funasr)
        if hasattr(self, "_crispasr_gpu_backend"):
            self._crispasr_gpu_label.setVisible(is_crispasr)
            self._crispasr_gpu_backend.setVisible(is_crispasr)
            self._crispasr_device_label.setVisible(is_crispasr)
            self._crispasr_device_index.setVisible(is_crispasr)
            self._crispasr_punc_label.setVisible(is_crispasr)
            self._crispasr_punc_model.setVisible(is_crispasr)
            self._crispasr_unified_memory.setVisible(is_crispasr)
        if hasattr(self, "_sherpa_onnx_provider"):
            self._sherpa_onnx_provider_label.setVisible(is_sherpa_onnx)
            self._sherpa_onnx_provider.setVisible(is_sherpa_onnx)
            self._sherpa_onnx_threads_label.setVisible(is_sherpa_onnx)
            self._sherpa_onnx_num_threads.setVisible(is_sherpa_onnx)
            self._sherpa_onnx_decoding_label.setVisible(is_sherpa_onnx)
            self._sherpa_onnx_decoding_method.setVisible(is_sherpa_onnx)
        if hasattr(self, "_parakeet_cpp_backend"):
            self._parakeet_cpp_backend_label.setVisible(is_parakeet_cpp)
            self._parakeet_cpp_backend.setVisible(is_parakeet_cpp)
            self._parakeet_cpp_decoder_label.setVisible(is_parakeet_cpp)
            self._parakeet_cpp_decoder.setVisible(is_parakeet_cpp)
            self._parakeet_cpp_word_timestamps.setVisible(is_parakeet_cpp)
        if hasattr(self, "_whisper_pad_seconds"):
            self._whisper_pad_label.setVisible(is_whisper)
            self._whisper_pad_seconds.setVisible(is_whisper)
        if hasattr(self, "_sensevoice_pad_seconds"):
            show_funasr_pad = is_funasr and funasr_supports_padding(
                self._selected_funasr_model()
            )
            self._sensevoice_pad_label.setVisible(show_funasr_pad)
            self._sensevoice_pad_seconds.setVisible(show_funasr_pad)
        if hasattr(self, "_remote_group"):
            self._remote_group.setVisible(is_remote)
        # Resize window to fit content after whisper group visibility change
        def _fit():
            self.adjustSize()
            h = self.sizeHint().height() + 20
            self.resize(self.width(), max(h, self.minimumHeight()))
        QTimer.singleShot(0, _fit)

    def _selected_funasr_model(self) -> str:
        value = self._funasr_model_combo.currentData()
        return normalize_funasr_model_key(str(value) if value else None)

    def _on_funasr_model_changed(self):
        self._current_settings["funasr_model"] = self._selected_funasr_model()
        self._on_asr_engine_changed(self._asr_engine.currentIndex())
        self._auto_save()

    def _selected_crispasr_model(self) -> str:
        value = self._crispasr_model_combo.currentData()
        return str(value) if value is not None else ""

    def _selected_crispasr_gpu_backend(self) -> str:
        return ["auto", "cuda", "vulkan", "cpu"][self._crispasr_gpu_backend.currentIndex()]

    def _selected_crispasr_punc_model(self) -> str:
        value = self._crispasr_punc_model.currentData()
        return str(value) if value else "auto"

    def _on_crispasr_model_changed(self):
        self._current_settings["crispasr_model"] = self._selected_crispasr_model()
        self._update_crispasr_status()
        self._auto_save()

    def _on_crispasr_runtime_setting_changed(self):
        self._current_settings["crispasr_gpu_backend"] = (
            self._selected_crispasr_gpu_backend()
        )
        self._current_settings["crispasr_device_index"] = (
            self._crispasr_device_index.value()
        )
        self._current_settings["crispasr_punc_model"] = (
            self._selected_crispasr_punc_model()
        )
        self._current_settings["crispasr_unified_memory"] = (
            self._crispasr_unified_memory.isChecked()
        )
        self._auto_save()

    def _selected_sherpa_onnx_model(self) -> str:
        value = self._sherpa_onnx_model_combo.currentData()
        return str(value) if value is not None else ""

    def _selected_sherpa_onnx_provider(self) -> str:
        value = self._sherpa_onnx_provider.currentData()
        return str(value) if value else "auto"

    def _selected_sherpa_onnx_decoding_method(self) -> str:
        value = self._sherpa_onnx_decoding_method.currentData()
        return str(value) if value else "greedy_search"

    def _on_sherpa_onnx_model_changed(self):
        self._current_settings["sherpa_onnx_model"] = (
            self._selected_sherpa_onnx_model()
        )
        self._update_sherpa_onnx_status()
        self._auto_save()

    def _on_sherpa_onnx_setting_changed(self):
        self._current_settings["sherpa_onnx_provider"] = (
            self._selected_sherpa_onnx_provider()
        )
        self._current_settings["sherpa_onnx_num_threads"] = (
            self._sherpa_onnx_num_threads.value()
        )
        self._current_settings["sherpa_onnx_decoding_method"] = (
            self._selected_sherpa_onnx_decoding_method()
        )
        self._auto_save()

    def _selected_parakeet_cpp_model(self) -> str:
        value = self._parakeet_cpp_model_combo.currentData()
        return str(value) if value is not None else ""

    def _selected_parakeet_cpp_runtime_dir(self) -> str:
        value = self._parakeet_cpp_runtime_combo.currentData()
        return str(value) if value is not None else ""

    def _selected_parakeet_cpp_backend(self) -> str:
        value = self._parakeet_cpp_backend.currentData()
        return str(value) if value else "auto"

    def _selected_parakeet_cpp_decoder(self) -> str:
        value = self._parakeet_cpp_decoder.currentData()
        return str(value) if value else "auto"

    def _on_parakeet_cpp_model_changed(self):
        self._current_settings["parakeet_cpp_model"] = (
            self._selected_parakeet_cpp_model()
        )
        self._update_parakeet_cpp_model_status()
        self._auto_save()

    def _on_parakeet_cpp_runtime_changed(self):
        self._current_settings["parakeet_cpp_runtime_dir"] = (
            self._selected_parakeet_cpp_runtime_dir()
        )
        self._update_parakeet_cpp_runtime_status()
        self._auto_save()

    def _on_parakeet_cpp_setting_changed(self):
        self._current_settings["parakeet_cpp_backend"] = (
            self._selected_parakeet_cpp_backend()
        )
        self._current_settings["parakeet_cpp_decoder"] = (
            self._selected_parakeet_cpp_decoder()
        )
        self._current_settings["parakeet_cpp_word_timestamps"] = (
            self._parakeet_cpp_word_timestamps.isChecked()
        )
        self._update_parakeet_cpp_runtime_status()
        self._auto_save()

    def _selected_firered_vad_model(self) -> str:
        value = self._firered_vad_model_combo.currentData()
        return str(value) if value is not None else ""

    def _on_firered_vad_model_changed(self):
        self._current_settings["firered_vad_model"] = (
            self._selected_firered_vad_model()
        )
        self._update_firered_vad_status()
        self._auto_save()

    def _on_firered_vad_setting_changed(self):
        self._current_settings["firered_vad_smooth_window_size"] = (
            self._firered_vad_smooth_window.value()
        )
        self._current_settings["firered_vad_use_gpu"] = (
            self._firered_vad_use_gpu.isChecked()
        )
        self._auto_save()

    def _selected_whisper_model(self) -> str:
        value = self._whisper_size_combo.currentData()
        return str(value) if value else self._whisper_size_combo.currentText()

    def _populate_firered_vad_models(self, saved_value: str):
        self._firered_vad_model_combo.clear()
        self._firered_vad_model_combo.addItem(t("firered_vad_model_placeholder"), "")

        local_prefix = t("firered_vad_local_prefix")
        for item in list_local_firered_vad_models():
            idx = self._firered_vad_model_combo.count()
            self._firered_vad_model_combo.addItem(
                f"{local_prefix}: {item['name']}", item["path"]
            )
            self._firered_vad_model_combo.setItemData(
                idx, item["path"], Qt.ItemDataRole.ToolTipRole
            )

        if not saved_value:
            selected = ""
        else:
            selected = resolve_custom_firered_vad_model(saved_value) or saved_value
        idx = self._firered_vad_model_combo.findData(selected)
        if idx < 0:
            idx = self._firered_vad_model_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('firered_vad_missing_local')}: {Path(str(selected)).name}"
            idx = self._firered_vad_model_combo.count()
            self._firered_vad_model_combo.addItem(label, selected)
            self._firered_vad_model_combo.setItemData(
                idx, str(selected), Qt.ItemDataRole.ToolTipRole
            )
        if idx >= 0:
            self._firered_vad_model_combo.setCurrentIndex(idx)

    def _refresh_firered_vad_models(self):
        saved = self._selected_firered_vad_model()
        self._populate_firered_vad_models(saved)
        self._update_firered_vad_status()

    def _update_firered_vad_status(self):
        model_key = self._selected_firered_vad_model()
        if not model_key:
            if list_local_firered_vad_models():
                self._firered_vad_status.setText(t("firered_vad_select_model"))
            else:
                self._firered_vad_status.setText(t("firered_vad_no_local_models"))
            self._firered_vad_status.setStyleSheet(
                "color: #888; font-size: 11px;"
            )
            return
        if resolve_custom_firered_vad_model(model_key):
            self._firered_vad_status.setText(t("firered_vad_local_ready"))
            self._firered_vad_status.setStyleSheet(
                "color: #4a4; font-size: 11px;"
            )
        else:
            self._firered_vad_status.setText(t("firered_vad_invalid_local"))
            self._firered_vad_status.setStyleSheet(
                "color: #d66; font-size: 11px;"
            )

    def _populate_sherpa_onnx_models(self, saved_value: str):
        self._sherpa_onnx_model_combo.clear()
        self._sherpa_onnx_model_combo.addItem(t("sherpa_onnx_model_placeholder"), "")

        local_prefix = t("sherpa_onnx_local_prefix")
        for item in list_local_sherpa_onnx_models():
            idx = self._sherpa_onnx_model_combo.count()
            family = str(item.get("family") or "").replace("_", " ")
            suffix = f" [{family}]" if family else ""
            self._sherpa_onnx_model_combo.addItem(
                f"{local_prefix}: {item['name']}{suffix}", item["path"]
            )
            self._sherpa_onnx_model_combo.setItemData(
                idx, item["path"], Qt.ItemDataRole.ToolTipRole
            )

        if not saved_value:
            selected = ""
        else:
            selected = resolve_custom_sherpa_onnx_model(saved_value) or saved_value
        idx = self._sherpa_onnx_model_combo.findData(selected)
        if idx < 0:
            idx = self._sherpa_onnx_model_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('sherpa_onnx_missing_local')}: {Path(str(selected)).name}"
            idx = self._sherpa_onnx_model_combo.count()
            self._sherpa_onnx_model_combo.addItem(label, selected)
            self._sherpa_onnx_model_combo.setItemData(
                idx, str(selected), Qt.ItemDataRole.ToolTipRole
            )
        if idx >= 0:
            self._sherpa_onnx_model_combo.setCurrentIndex(idx)

    def _refresh_sherpa_onnx_models(self):
        saved = self._selected_sherpa_onnx_model()
        self._populate_sherpa_onnx_models(saved)
        self._update_sherpa_onnx_status()

    def _update_sherpa_onnx_status(self):
        from model_manager import is_asr_cached

        model_key = self._selected_sherpa_onnx_model()
        if not model_key:
            if list_local_sherpa_onnx_models():
                self._sherpa_onnx_status.setText(t("sherpa_onnx_select_model"))
            else:
                self._sherpa_onnx_status.setText(t("sherpa_onnx_no_local_models"))
            self._sherpa_onnx_status.setStyleSheet("color: #888; font-size: 11px;")
            return
        cached = is_asr_cached("sherpa-onnx", model_key, self._current_settings.get("hub", "ms"))
        if cached:
            self._sherpa_onnx_status.setText(t("sherpa_onnx_local_ready"))
            self._sherpa_onnx_status.setStyleSheet("color: #4a4; font-size: 11px;")
        else:
            self._sherpa_onnx_status.setText(t("sherpa_onnx_invalid_local"))
            self._sherpa_onnx_status.setStyleSheet("color: #d66; font-size: 11px;")

    def _populate_parakeet_cpp_models(self, saved_value: str):
        self._parakeet_cpp_model_combo.clear()
        self._parakeet_cpp_model_combo.addItem(
            t("parakeet_cpp_model_placeholder"), ""
        )

        local_prefix = t("parakeet_cpp_local_prefix")
        for item in list_local_parakeet_cpp_models():
            idx = self._parakeet_cpp_model_combo.count()
            self._parakeet_cpp_model_combo.addItem(
                f"{local_prefix}: {item['name']}", item["path"]
            )
            self._parakeet_cpp_model_combo.setItemData(
                idx, item["path"], Qt.ItemDataRole.ToolTipRole
            )

        if not saved_value:
            selected = ""
        else:
            selected = resolve_custom_parakeet_cpp_model(saved_value) or saved_value
        idx = self._parakeet_cpp_model_combo.findData(selected)
        if idx < 0:
            idx = self._parakeet_cpp_model_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('parakeet_cpp_missing_local')}: {Path(str(selected)).name}"
            idx = self._parakeet_cpp_model_combo.count()
            self._parakeet_cpp_model_combo.addItem(label, selected)
            self._parakeet_cpp_model_combo.setItemData(
                idx, str(selected), Qt.ItemDataRole.ToolTipRole
            )
        if idx >= 0:
            self._parakeet_cpp_model_combo.setCurrentIndex(idx)

    def _populate_parakeet_cpp_runtimes(self, saved_value: str):
        self._parakeet_cpp_runtime_combo.clear()
        self._parakeet_cpp_runtime_combo.addItem(
            t("parakeet_cpp_runtime_placeholder"), ""
        )

        local_prefix = t("parakeet_cpp_local_prefix")
        for item in list_local_parakeet_cpp_runtimes():
            idx = self._parakeet_cpp_runtime_combo.count()
            backend = str(item.get("backend") or "").lower()
            suffix = f" [{backend}]" if backend and backend != "unknown" else ""
            self._parakeet_cpp_runtime_combo.addItem(
                f"{local_prefix}: {item['name']}{suffix}", item["path"]
            )
            self._parakeet_cpp_runtime_combo.setItemData(
                idx, item["path"], Qt.ItemDataRole.ToolTipRole
            )

        if not saved_value:
            selected = ""
        else:
            selected = (
                resolve_parakeet_cpp_runtime_dir(saved_value, "auto")
                or saved_value
            )
        idx = self._parakeet_cpp_runtime_combo.findData(selected)
        if idx < 0:
            idx = self._parakeet_cpp_runtime_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('parakeet_cpp_missing_local')}: {Path(str(selected)).name}"
            idx = self._parakeet_cpp_runtime_combo.count()
            self._parakeet_cpp_runtime_combo.addItem(label, selected)
            self._parakeet_cpp_runtime_combo.setItemData(
                idx, str(selected), Qt.ItemDataRole.ToolTipRole
            )
        if idx >= 0:
            self._parakeet_cpp_runtime_combo.setCurrentIndex(idx)

    def _refresh_parakeet_cpp_models(self):
        saved = self._selected_parakeet_cpp_model()
        self._populate_parakeet_cpp_models(saved)
        self._update_parakeet_cpp_model_status()

    def _refresh_parakeet_cpp_runtimes(self):
        saved = self._selected_parakeet_cpp_runtime_dir()
        self._populate_parakeet_cpp_runtimes(saved)
        self._update_parakeet_cpp_runtime_status()

    def _update_parakeet_cpp_model_status(self):
        from model_manager import is_asr_cached

        model_key = self._selected_parakeet_cpp_model()
        if not model_key:
            if list_local_parakeet_cpp_models():
                self._parakeet_cpp_model_status.setText(
                    t("parakeet_cpp_select_model")
                )
            else:
                self._parakeet_cpp_model_status.setText(
                    t("parakeet_cpp_no_local_models")
                )
            self._parakeet_cpp_model_status.setStyleSheet(
                "color: #888; font-size: 11px;"
            )
            return
        cached = is_asr_cached(
            "parakeet-cpp", model_key, self._current_settings.get("hub", "ms")
        )
        if cached:
            self._parakeet_cpp_model_status.setText(
                t("parakeet_cpp_local_ready")
            )
            self._parakeet_cpp_model_status.setStyleSheet(
                "color: #4a4; font-size: 11px;"
            )
        else:
            self._parakeet_cpp_model_status.setText(
                t("parakeet_cpp_invalid_local")
            )
            self._parakeet_cpp_model_status.setStyleSheet(
                "color: #d66; font-size: 11px;"
            )

    def _update_parakeet_cpp_runtime_status(self):
        from model_manager import detect_parakeet_cpp_runtime_dir

        runtime_dir = self._selected_parakeet_cpp_runtime_dir()
        backend = self._selected_parakeet_cpp_backend()
        if not runtime_dir:
            if list_local_parakeet_cpp_runtimes():
                self._parakeet_cpp_runtime_status.setText(
                    t("parakeet_cpp_select_runtime")
                )
            else:
                self._parakeet_cpp_runtime_status.setText(
                    t("parakeet_cpp_no_local_runtimes")
                )
            self._parakeet_cpp_runtime_status.setStyleSheet(
                "color: #888; font-size: 11px;"
            )
            return
        resolved = resolve_parakeet_cpp_runtime_dir(runtime_dir, "auto")
        info = detect_parakeet_cpp_runtime_dir(resolved) if resolved else None
        if not info:
            self._parakeet_cpp_runtime_status.setText(
                t("parakeet_cpp_invalid_runtime")
            )
            self._parakeet_cpp_runtime_status.setStyleSheet(
                "color: #d66; font-size: 11px;"
            )
            return
        runtime_backend = str(info.get("backend") or "unknown")
        missing = info.get("missing_dependencies") or []
        if backend != "auto" and runtime_backend not in ("unknown", backend):
            self._parakeet_cpp_runtime_status.setText(
                t("parakeet_cpp_invalid_runtime")
            )
            self._parakeet_cpp_runtime_status.setStyleSheet(
                "color: #d66; font-size: 11px;"
            )
            return
        if missing:
            self._parakeet_cpp_runtime_status.setText(
                f"{t('parakeet_cpp_runtime_ready')} ({', '.join(missing)}?)"
            )
            self._parakeet_cpp_runtime_status.setStyleSheet(
                "color: #d99; font-size: 11px;"
            )
            return
        self._parakeet_cpp_runtime_status.setText(
            t("parakeet_cpp_runtime_ready")
        )
        self._parakeet_cpp_runtime_status.setStyleSheet(
            "color: #4a4; font-size: 11px;"
        )

    def _populate_crispasr_models(self, saved_value: str):
        self._crispasr_model_combo.clear()
        self._crispasr_model_combo.addItem(t("crispasr_model_placeholder"), "")

        local_prefix = t("crispasr_local_prefix")
        for item in list_local_crispasr_models():
            idx = self._crispasr_model_combo.count()
            self._crispasr_model_combo.addItem(
                f"{local_prefix}: {item['name']}", item["path"]
            )
            self._crispasr_model_combo.setItemData(
                idx, item["path"], Qt.ItemDataRole.ToolTipRole
            )

        if not saved_value:
            selected = ""
        else:
            selected = resolve_custom_crispasr_model(saved_value) or saved_value
        idx = self._crispasr_model_combo.findData(selected)
        if idx < 0:
            idx = self._crispasr_model_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('crispasr_missing_local')}: {Path(str(selected)).name}"
            idx = self._crispasr_model_combo.count()
            self._crispasr_model_combo.addItem(label, selected)
            self._crispasr_model_combo.setItemData(
                idx, str(selected), Qt.ItemDataRole.ToolTipRole
            )
        if idx >= 0:
            self._crispasr_model_combo.setCurrentIndex(idx)

    def _populate_whisper_models(self, saved_value: str):
        self._whisper_size_combo.clear()
        for size in _WHISPER_SIZES:
            self._whisper_size_combo.addItem(size, size)

        local_prefix = t("whisper_local_prefix")
        for item in list_local_faster_whisper_models():
            idx = self._whisper_size_combo.count()
            self._whisper_size_combo.addItem(
                f"{local_prefix}: {item['name']}", item["path"]
            )
            self._whisper_size_combo.setItemData(
                idx, item["path"], Qt.ItemDataRole.ToolTipRole
            )

        selected = resolve_custom_whisper_model(saved_value) or saved_value
        idx = self._whisper_size_combo.findData(selected)
        if idx < 0:
            idx = self._whisper_size_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('whisper_missing_local')}: {Path(str(selected)).name}"
            idx = self._whisper_size_combo.count()
            self._whisper_size_combo.addItem(label, selected)
            self._whisper_size_combo.setItemData(
                idx, str(selected), Qt.ItemDataRole.ToolTipRole
            )
        if idx >= 0:
            self._whisper_size_combo.setCurrentIndex(idx)

    def _update_whisper_size_label(self):
        from model_manager import is_asr_cached, _MODEL_SIZE_BYTES

        size = self._selected_whisper_model()
        cached = is_asr_cached("whisper", size, self._current_settings.get("hub", "ms"))
        if size not in _WHISPER_SIZES:
            if cached:
                self._whisper_status.setText(t("whisper_local_ready"))
                self._whisper_status.setStyleSheet("color: #4a4; font-size: 11px;")
            else:
                self._whisper_status.setText(t("whisper_invalid_local"))
                self._whisper_status.setStyleSheet("color: #d66; font-size: 11px;")
            self._whisper_dl_btn.setEnabled(False)
            return
        if cached:
            self._whisper_status.setText(t("whisper_already_cached"))
            self._whisper_status.setStyleSheet("color: #4a4; font-size: 11px;")
            self._whisper_dl_btn.setEnabled(False)
        else:
            est = _MODEL_SIZE_BYTES.get(f"whisper-{size}", 0)
            self._whisper_status.setText(f"~{format_size(est)}")
            self._whisper_status.setStyleSheet("color: #888; font-size: 11px;")
            self._whisper_dl_btn.setEnabled(True)

    def _on_whisper_size_changed(self):
        self._current_settings["whisper_model_size"] = (
            self._selected_whisper_model()
        )
        self._update_whisper_size_label()
        # If already cached, switch engine immediately
        from model_manager import is_asr_cached

        size = self._selected_whisper_model()
        if is_asr_cached("whisper", size, self._current_settings.get("hub", "ms")):
            self._auto_save()

    def _download_whisper(self):
        from model_manager import is_asr_cached, get_missing_models

        size = self._selected_whisper_model()
        if size not in _WHISPER_SIZES:
            return
        hub = self._current_settings.get("hub", "ms")
        if is_asr_cached("whisper", size, hub):
            return
        missing = get_missing_models("whisper", size, hub)
        missing = [m for m in missing if m["type"] != "silero-vad"]
        if not missing:
            return
        from dialogs import ModelDownloadDialog

        dlg = ModelDownloadDialog(missing, hub=hub, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self._update_whisper_size_label()
            # Switch to Whisper engine with the downloaded size
            self._auto_save()

    def _update_crispasr_status(self):
        from model_manager import is_asr_cached

        model_key = self._selected_crispasr_model()
        if not model_key:
            self._crispasr_status.setText(t("crispasr_select_model"))
            self._crispasr_status.setStyleSheet("color: #888; font-size: 11px;")
            return
        hub = self._current_settings.get("hub", "ms")
        cached = is_asr_cached("crispasr", model_key, hub)
        if cached:
            self._crispasr_status.setText(t("crispasr_local_ready"))
            self._crispasr_status.setStyleSheet("color: #4a4; font-size: 11px;")
        else:
            self._crispasr_status.setText(t("crispasr_invalid_local"))
            self._crispasr_status.setStyleSheet("color: #d66; font-size: 11px;")

    # ── Model Management ──

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

    def refresh_model_list(self):
        self._refresh_model_list()

    def current_settings(self) -> dict:
        return dict(self._current_settings)

    def set_active_model(self, index: int, save: bool = True, emit: bool = True) -> bool:
        models = self._current_settings.get("models", [])
        if not (0 <= index < len(models)):
            return False
        self._current_settings["active_model"] = index
        self._refresh_model_list()
        if save:
            _save_settings(self._current_settings)
        self._emit_models_list_changed()
        if emit:
            self.model_changed.emit(models[index])
        return True

    def set_target_language(self, code: str, save: bool = True):
        self._current_settings["target_language"] = code
        if save:
            _save_settings(self._current_settings)

    def set_asr_language(self, code: str, save: bool = True, emit: bool = False):
        self._current_settings["asr_language"] = code
        idx = self._asr_lang.findData(code)
        if idx >= 0:
            self._asr_lang.blockSignals(True)
            self._asr_lang.setCurrentIndex(idx)
            self._asr_lang.blockSignals(False)
        if save:
            _save_settings(self._current_settings)
        if emit:
            self.asr_language_changed.emit(code)

    def current_asr_language(self) -> str:
        return self._get_asr_lang_code()

    def update_subtitle_mode(self, patch: dict, save: bool = True) -> dict:
        mode = dict(self._current_settings.get("subtitle_mode") or {})
        mode.update(patch)
        self._current_settings["subtitle_mode"] = mode
        if save:
            _save_settings(self._current_settings)
        return mode

    def update_settings(self, patch: dict, save: bool = True) -> dict:
        self._current_settings.update(patch)
        if save:
            _save_settings(self._current_settings)
        return dict(self._current_settings)

    def _emit_models_list_changed(self):
        models = self._current_settings.get("models", [])
        active_idx = self._current_settings.get("active_model", 0)
        self.models_list_changed.emit(models, active_idx)

    def _add_model(self):
        dlg = ModelEditDialog(self)
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                self._current_settings.setdefault("models", []).append(data)
                self._refresh_model_list()
                _save_settings(self._current_settings)
                self._emit_models_list_changed()

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
                _save_settings(self._current_settings)
                self._emit_models_list_changed()
                # Re-apply if editing the active model
                active = self._current_settings.get("active_model", 0)
                if row == active:
                    self.model_changed.emit(data)

    def _dup_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dup = dict(models[row])
        dup["name"] = dup["name"] + " (copy)"
        models.append(dup)
        self._refresh_model_list()
        _save_settings(self._current_settings)
        self._emit_models_list_changed()

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
        self._model_list.setCurrentRow(min(row, len(models) - 1))
        _save_settings(self._current_settings)
        self._emit_models_list_changed()

    def _on_model_double_clicked(self, item):
        row = self._model_list.row(item)
        models = self._current_settings.get("models", [])
        if 0 <= row < len(models):
            self._model_list.setCurrentRow(row)
            self._edit_model()

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
        self._current_settings["vad_mode"] = self._selected_vad_mode()
        self._update_vad_detail_visibility()

    def _update_vad_detail_visibility(self):
        if not hasattr(self, "_vad_mode"):
            return
        mode = self._selected_vad_mode()
        if hasattr(self, "_neural_vad_group"):
            self._neural_vad_group.setVisible(mode in ("silero", "firered"))
        if hasattr(self, "_firered_vad_group"):
            self._firered_vad_group.setVisible(mode == "firered")
        if hasattr(self, "_energy_group"):
            self._energy_group.setVisible(mode == "energy")

    def _on_threshold_changed(self, value):
        val = value / 100.0
        self._current_settings["vad_threshold"] = val
        self._vad_threshold_label.setText(f"{value}%")
        if not self._vad_threshold_slider.isSliderDown():
            self._auto_save()

    def _on_energy_changed(self, value):
        val = value / 1000.0
        self._current_settings["energy_threshold"] = val
        self._energy_label.setText(f"{value}\u2030")
        if not self._energy_slider.isSliderDown():
            self._auto_save()

    def _on_timing_changed(self):
        self._current_settings["min_speech_duration"] = round(self._min_speech.value(), 2)
        self._current_settings["max_speech_duration"] = round(self._max_speech.value(), 2)
        self._current_settings["silence_mode"] = (
            "auto" if self._silence_mode.currentIndex() == 0 else "fixed"
        )
        self._current_settings["silence_duration"] = round(self._silence_duration.value(), 2)
        self._current_settings["incremental_asr"] = self._incremental_asr_cb.isChecked()
        self._current_settings["interim_interval"] = round(self._interim_interval_spin.value(), 2)

    def _on_ui_lang_changed(self, index):
        lang = "en" if index == 0 else "zh"
        self._current_settings["ui_lang"] = lang
        _save_settings(self._current_settings)
        from i18n import set_lang

        set_lang(lang)
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "LiveTranslate",
            "Language changed. Please restart the application.\n"
            "语言已更改，请重启应用程序。",
        )

    def _auto_save(self):
        self._save_timer.start()

    def _do_auto_save(self):
        self._apply_settings()
        _save_settings(self._current_settings)

    def _on_prompt_preset_changed(self, index):
        from translator import DEFAULT_PROMPT, PROMPT_PRESETS
        key = self._prompt_preset.itemData(index)
        if key == "custom":
            return
        prompt = PROMPT_PRESETS.get(key, DEFAULT_PROMPT)
        self._prompt_edit.setPlainText(prompt)
        self._apply_prompt()

    def _apply_prompt(self):
        text = self._prompt_edit.toPlainText().strip()
        if text:
            self._current_settings["system_prompt"] = text
            active = self.get_active_model()
            if active:
                self.model_changed.emit(active)
            _save_settings(self._current_settings)
            log.info("System prompt updated")
            # Update preset combo to reflect current state
            from translator import PROMPT_PRESETS
            self._prompt_preset.blockSignals(True)
            matched = 4  # custom
            for i, key in enumerate(["daily", "esports", "anime", "webid"]):
                if text.strip() == PROMPT_PRESETS[key].strip():
                    matched = i
                    break
            self._prompt_preset.setCurrentIndex(matched)
            self._prompt_preset.blockSignals(False)

    def _apply_settings(self):
        self._current_settings["asr_language"] = self._get_asr_lang_code()
        self._current_settings["vad_mode"] = self._selected_vad_mode()
        self._current_settings["firered_vad_model"] = (
            self._selected_firered_vad_model()
        )
        self._current_settings["firered_vad_smooth_window_size"] = (
            self._firered_vad_smooth_window.value()
        )
        self._current_settings["firered_vad_use_gpu"] = (
            self._firered_vad_use_gpu.isChecked()
        )
        self._current_settings["asr_engine"] = self._selected_asr_engine()
        self._current_settings["funasr_model"] = self._selected_funasr_model()
        if hasattr(self, "_remote_url_edit"):
            url = self._remote_url_edit.text().strip()
            if url:
                self._current_settings["remote_asr_url"] = url
        self._current_settings["whisper_model_size"] = (
            self._selected_whisper_model()
        )
        self._current_settings["crispasr_model"] = self._selected_crispasr_model()
        self._current_settings["crispasr_backend"] = "auto"
        self._current_settings["crispasr_gpu_backend"] = (
            self._selected_crispasr_gpu_backend()
        )
        self._current_settings["crispasr_device_index"] = (
            self._crispasr_device_index.value()
        )
        self._current_settings["crispasr_punc_model"] = (
            self._selected_crispasr_punc_model()
        )
        self._current_settings["crispasr_unified_memory"] = (
            self._crispasr_unified_memory.isChecked()
        )
        self._current_settings["sherpa_onnx_model"] = (
            self._selected_sherpa_onnx_model()
        )
        self._current_settings["sherpa_onnx_provider"] = (
            self._selected_sherpa_onnx_provider()
        )
        self._current_settings["sherpa_onnx_num_threads"] = (
            self._sherpa_onnx_num_threads.value()
        )
        self._current_settings["sherpa_onnx_decoding_method"] = (
            self._selected_sherpa_onnx_decoding_method()
        )
        self._current_settings["parakeet_cpp_model"] = (
            self._selected_parakeet_cpp_model()
        )
        self._current_settings["parakeet_cpp_runtime_dir"] = (
            self._selected_parakeet_cpp_runtime_dir()
        )
        self._current_settings["parakeet_cpp_backend"] = (
            self._selected_parakeet_cpp_backend()
        )
        self._current_settings["parakeet_cpp_decoder"] = (
            self._selected_parakeet_cpp_decoder()
        )
        self._current_settings["parakeet_cpp_word_timestamps"] = (
            self._parakeet_cpp_word_timestamps.isChecked()
        )
        dev_text = self._asr_device.currentText()
        self._current_settings["asr_device"] = dev_text.split(" (")[0]
        audio_idx = self._audio_device.currentIndex()
        if audio_idx == 0:
            self._current_settings["audio_device"] = "__disabled__"
        elif audio_idx == 1:
            self._current_settings["audio_device"] = None
        else:
            self._current_settings["audio_device"] = self._audio_device.currentText()
        mic_idx = self._mic_device.currentIndex()
        if mic_idx == 0:
            self._current_settings["mic_device"] = None
        elif mic_idx == 1:
            self._current_settings["mic_device"] = "__default__"
        else:
            self._current_settings["mic_device"] = self._mic_device.currentText()
        self._current_settings["hub"] = (
            "ms" if self._hub_combo.currentIndex() == 0 else "hf"
        )
        self._current_settings["sensevoice_pad_seconds"] = round(
            self._sensevoice_pad_seconds.value(), 2
        )
        self._current_settings["whisper_pad_seconds"] = round(
            self._whisper_pad_seconds.value(), 2
        )
        prompt_text = self._prompt_edit.toPlainText().strip()
        if prompt_text:
            self._current_settings["system_prompt"] = prompt_text
        self._current_settings["timeout"] = self._timeout_spin.value()
        if hasattr(self, "_incremental_asr_cb"):
            self._on_timing_changed()
        if hasattr(self, "_auto_save_transcript_cb"):
            self._current_settings["auto_save_transcript"] = (
                self._auto_save_transcript_cb.isChecked()
            )
        if hasattr(self, "_style_preset"):
            self._current_settings["style"] = self._collect_style()
        safe = {
            k: v
            for k, v in self._current_settings.items()
            if k not in ("models", "system_prompt")
        }
        log.info(f"Settings applied: {safe}")
        self.settings_changed.emit(dict(self._current_settings))

    def get_settings(self):
        return self.current_settings()

    def get_active_model(self) -> dict | None:
        models = self._current_settings.get("models", [])
        idx = self._current_settings.get("active_model", 0)
        if 0 <= idx < len(models):
            return models[idx]
        return None

    def has_saved_settings(self) -> bool:
        return SETTINGS_FILE.exists()
