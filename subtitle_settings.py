"""
Subtitle window settings dialog.
Configures text lines, background, alignment for the subtitle window.
"""

import os
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFontDatabase
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from i18n import t, LANGUAGES
from subtitle_window import DEFAULT_SUBTITLE_WIN_SETTINGS

_PROJECT_DIR = Path(__file__).parent


class _ColorButton(QPushButton):
    """Small button that shows a color and opens a picker on click."""

    color_changed = pyqtSignal(str)

    def __init__(self, color="#FFFFFF", parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(28, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()
        self.clicked.connect(self._pick)

    def _update_style(self):
        self.setStyleSheet(
            f"background: {self._color}; border: 1px solid #888; border-radius: 3px;"
        )

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._color), self.window())
        if c.isValid():
            self._color = c.name()
            self._update_style()
            self.color_changed.emit(self._color)

    def color(self):
        return self._color

    def set_color(self, c):
        self._color = c
        self._update_style()


def _make_image_row(current_path: str, on_change):
    """Create a background image selector row. Returns (layout, line_edit)."""
    row = QHBoxLayout()
    row.setSpacing(6)

    line_edit = QLineEdit()
    line_edit.setReadOnly(True)
    line_edit.setText(current_path)
    line_edit.setMinimumWidth(120)
    row.addWidget(line_edit, 1)

    select_btn = QPushButton(t("subwin_bg_image_select"))
    select_btn.setFixedHeight(22)

    def _select():
        path, _ = QFileDialog.getOpenFileName(
            line_edit.window(),
            t("subwin_bg_image_select"),
            "",
            "Images (*.png *.webp *.jpg *.jpeg *.bmp)",
        )
        if path:
            # Store relative path if under project dir
            try:
                rel = os.path.relpath(path, _PROJECT_DIR)
                if not rel.startswith(".."):
                    path = rel.replace("\\", "/")
            except ValueError:
                pass
            line_edit.setText(path)
            on_change()

    select_btn.clicked.connect(_select)
    row.addWidget(select_btn)

    clear_btn = QPushButton(t("subwin_bg_image_clear"))
    clear_btn.setFixedHeight(22)

    def _clear():
        line_edit.setText("")
        on_change()

    clear_btn.clicked.connect(_clear)
    row.addWidget(clear_btn)

    return row, line_edit


class _LineConfigWidget(QWidget):
    """Config panel for a single text line (original or translation)."""

    changed = pyqtSignal()
    remove_requested = pyqtSignal()
    move_up_requested = pyqtSignal()
    move_down_requested = pyqtSignal()

    def __init__(self, cfg: dict, removable=False, parent=None):
        super().__init__(parent)
        self._cfg = dict(cfg)
        self._removable = removable
        self._up_btn = None
        self._down_btn = None
        self._color_controls = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Header row: ▲▼ + enabled + type combo + lang + remove
        hdr = QHBoxLayout()
        hdr.setSpacing(4)

        self._up_btn = QPushButton(t("subwin_move_up"))
        self._up_btn.setFixedSize(24, 22)
        self._up_btn.clicked.connect(self.move_up_requested.emit)
        hdr.addWidget(self._up_btn)

        self._down_btn = QPushButton(t("subwin_move_down"))
        self._down_btn.setFixedSize(24, 22)
        self._down_btn.clicked.connect(self.move_down_requested.emit)
        hdr.addWidget(self._down_btn)

        self._enabled = QCheckBox(t("subwin_enabled"))
        self._enabled.setChecked(self._cfg.get("enabled", True))
        self._enabled.toggled.connect(self._on_change)
        hdr.addWidget(self._enabled)

        self._type_combo = QComboBox()
        self._type_combo.addItem(t("subwin_original"), "original")
        self._type_combo.addItem(t("subwin_translation"), "translation")
        current_type = self._cfg.get("type", "original")
        idx = self._type_combo.findData(current_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        self._type_combo.currentIndexChanged.connect(self._on_type_change)
        hdr.addWidget(self._type_combo)

        self._lang_label = QLabel(t("subwin_target_lang"))
        hdr.addWidget(self._lang_label)
        self._lang_combo = QComboBox()
        self._lang_combo.setMinimumWidth(100)
        for code, native in LANGUAGES:
            if code == "auto":
                continue
            self._lang_combo.addItem(f"{code} - {native}", code)
        current_lang = self._cfg.get("lang", "zh")
        idx = self._lang_combo.findData(current_lang)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        self._lang_combo.currentIndexChanged.connect(self._on_change)
        hdr.addWidget(self._lang_combo)
        self._update_lang_visibility()

        hdr.addStretch()

        if self._removable:
            rm_btn = QPushButton(t("subwin_remove_line"))
            rm_btn.clicked.connect(self.remove_requested.emit)
            hdr.addWidget(rm_btn)

        outer.addLayout(hdr)

        grid = QGridLayout()
        r = 0

        grid.addWidget(QLabel(t("subwin_font")), r, 0)
        self._font_combo = QComboBox()
        families = QFontDatabase.families()
        self._font_combo.addItems(families)
        current_font = self._cfg.get("font_family", "Microsoft YaHei")
        idx = self._font_combo.findText(current_font)
        if idx >= 0:
            self._font_combo.setCurrentIndex(idx)
        self._font_combo.currentTextChanged.connect(self._on_change)
        grid.addWidget(self._font_combo, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_font_size")), r, 0)
        self._size_spin = QSpinBox()
        self._size_spin.setRange(8, 120)
        self._size_spin.setSuffix(" pt")
        self._size_spin.setValue(self._cfg.get("font_size", 24))
        self._size_spin.valueChanged.connect(self._on_change)
        grid.addWidget(self._size_spin, r, 1)
        r += 1

        lbl_color = QLabel(t("subwin_color"))
        grid.addWidget(lbl_color, r, 0)
        self._color_btn = _ColorButton(self._cfg.get("color", "#FFFFFF"))
        self._color_btn.color_changed.connect(self._on_change)
        grid.addWidget(self._color_btn, r, 1)
        self._color_controls.append(lbl_color)
        self._color_controls.append(self._color_btn)
        r += 1

        grid.addWidget(QLabel(t("subwin_opacity")), r, 0)
        self._opacity_spin = QSpinBox()
        self._opacity_spin.setRange(0, 100)
        self._opacity_spin.setSuffix("%")
        self._opacity_spin.setValue(round(self._cfg.get("opacity", 255) / 255 * 100))
        self._opacity_spin.valueChanged.connect(self._on_change)
        grid.addWidget(self._opacity_spin, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_align")), r, 0)
        self._align_combo = QComboBox()
        self._align_combo.addItem(t("subwin_align_left"), "left")
        self._align_combo.addItem(t("subwin_align_center"), "center")
        self._align_combo.addItem(t("subwin_align_right"), "right")
        align = self._cfg.get("align", "center")
        idx = self._align_combo.findData(align)
        if idx >= 0:
            self._align_combo.setCurrentIndex(idx)
        self._align_combo.currentIndexChanged.connect(self._on_change)
        grid.addWidget(self._align_combo, r, 1)
        r += 1

        self._outline_check = QCheckBox(t("subwin_outline"))
        self._outline_check.setChecked(self._cfg.get("outline_enabled", True))
        self._outline_check.toggled.connect(self._on_change)
        grid.addWidget(self._outline_check, r, 0)
        r += 1

        grid.addWidget(QLabel(t("subwin_outline_color")), r, 0)
        self._outline_color_btn = _ColorButton(self._cfg.get("outline_color", "#000000"))
        self._outline_color_btn.color_changed.connect(self._on_change)
        grid.addWidget(self._outline_color_btn, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_outline_width")), r, 0)
        self._outline_width = QSpinBox()
        self._outline_width.setRange(0, 10)
        self._outline_width.setSuffix(" px")
        self._outline_width.setValue(self._cfg.get("outline_width", 2))
        self._outline_width.valueChanged.connect(self._on_change)
        grid.addWidget(self._outline_width, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_bg_image")), r, 0)
        img_row, self._bg_image_edit = _make_image_row(
            self._cfg.get("bg_image", ""), self._on_bg_image_change
        )
        grid.addLayout(img_row, r, 1)
        r += 1

        anim_items = [
            (t("subwin_anim_none"), "none"),
            (t("subwin_anim_fade"), "fade"),
            (t("subwin_anim_slide_left"), "slide_left"),
            (t("subwin_anim_slide_right"), "slide_right"),
            (t("subwin_anim_slide_up"), "slide_up"),
            (t("subwin_anim_slide_down"), "slide_down"),
        ]

        grid.addWidget(QLabel(t("subwin_entry_anim")), r, 0)
        self._entry_anim_combo = QComboBox()
        for label, val in anim_items:
            self._entry_anim_combo.addItem(label, val)
        idx = self._entry_anim_combo.findData(self._cfg.get("entry_animation", "none"))
        if idx >= 0:
            self._entry_anim_combo.setCurrentIndex(idx)
        self._entry_anim_combo.currentIndexChanged.connect(self._on_change)
        grid.addWidget(self._entry_anim_combo, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_exit_anim")), r, 0)
        self._exit_anim_combo = QComboBox()
        for label, val in anim_items:
            self._exit_anim_combo.addItem(label, val)
        idx = self._exit_anim_combo.findData(self._cfg.get("exit_animation", "none"))
        if idx >= 0:
            self._exit_anim_combo.setCurrentIndex(idx)
        self._exit_anim_combo.currentIndexChanged.connect(self._on_change)
        grid.addWidget(self._exit_anim_combo, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_anim_duration")), r, 0)
        self._anim_duration_spin = QSpinBox()
        self._anim_duration_spin.setRange(50, 3000)
        self._anim_duration_spin.setSuffix(" ms")
        self._anim_duration_spin.setValue(self._cfg.get("animation_duration", 300))
        self._anim_duration_spin.valueChanged.connect(self._on_change)
        grid.addWidget(self._anim_duration_spin, r, 1)

        outer.addLayout(grid)
        self._update_color_controls_state()

    def _on_bg_image_change(self):
        self._update_color_controls_state()
        self._on_change()

    def _update_color_controls_state(self):
        has_image = bool(self._bg_image_edit.text())
        for ctrl in self._color_controls:
            ctrl.setEnabled(not has_image)
        self._opacity_spin.setEnabled(not has_image)

    def set_move_enabled(self, up: bool, down: bool):
        if self._up_btn:
            self._up_btn.setEnabled(up)
        if self._down_btn:
            self._down_btn.setEnabled(down)

    def _on_type_change(self, *_):
        self._update_lang_visibility()
        self._on_change()

    def _update_lang_visibility(self):
        is_translation = self._type_combo.currentData() == "translation"
        self._lang_label.setVisible(is_translation)
        self._lang_combo.setVisible(is_translation)

    def _on_change(self, *_):
        self.changed.emit()

    def get_config(self) -> dict:
        cfg = {
            **self._cfg,
            "type": self._type_combo.currentData() or "original",
            "enabled": self._enabled.isChecked(),
            "font_family": self._font_combo.currentText(),
            "font_size": self._size_spin.value(),
            "color": self._color_btn.color(),
            "opacity": round(self._opacity_spin.value() / 100 * 255),
            "align": self._align_combo.currentData() or "center",
            "outline_enabled": self._outline_check.isChecked(),
            "outline_color": self._outline_color_btn.color(),
            "outline_width": self._outline_width.value(),
            "bg_image": self._bg_image_edit.text(),
            "entry_animation": self._entry_anim_combo.currentData() or "none",
            "exit_animation": self._exit_anim_combo.currentData() or "none",
            "animation_duration": self._anim_duration_spin.value(),
        }
        if cfg["type"] == "translation":
            cfg["lang"] = self._lang_combo.currentData() or "zh"
        return cfg


class SubtitleSettingsWidget(QWidget):
    """Embeddable subtitle settings panel (used as a tab in ControlPanel)."""

    settings_changed = pyqtSignal(dict)
    edit_mode_changed = pyqtSignal(bool)

    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        self._settings = {**DEFAULT_SUBTITLE_WIN_SETTINGS, **(current_settings or {})}
        self._line_widgets = []
        self._separators = []
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._emit_settings)
        self._build_ui()

    def update_settings(self, settings: dict):
        self._settings = {**DEFAULT_SUBTITLE_WIN_SETTINGS, **(settings or {})}
        self._spacing_spin.setValue(self._settings.get("line_spacing", 8))
        self._width_spin.setValue(self._settings.get("window_width", 1000))
        self._bg_color_btn.set_color(self._settings.get("bg_color", "#000000"))
        self._bg_opacity_spin.setValue(round(self._settings.get("bg_opacity", 0) / 255 * 100))
        self._win_bg_image_edit.setText(self._settings.get("bg_image", ""))
        self._auto_hide_spin.setValue(self._settings.get("auto_hide_timeout", 0))
        idx = self._hide_anim_combo.findData(self._settings.get("auto_hide_animation", "fade"))
        if idx >= 0:
            self._hide_anim_combo.setCurrentIndex(idx)
        self._hide_duration_spin.setValue(self._settings.get("auto_hide_duration", 300))
        self._rebuild_lines()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        # Top row: edit mode + help
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        self._edit_mode_btn = QPushButton(t("subwin_edit_mode"))
        self._edit_mode_btn.setCheckable(True)
        self._edit_mode_btn.toggled.connect(self._on_edit_mode_toggled)
        top_row.addWidget(self._edit_mode_btn)

        top_row.addStretch()

        reset_btn = QPushButton(t("subwin_reset"))
        reset_btn.clicked.connect(self._on_reset)
        top_row.addWidget(reset_btn)

        hint_btn = QPushButton(t("subwin_help"))
        hint_btn.clicked.connect(lambda: QMessageBox.information(self, t("subwin_help"), t("subwin_hint")))
        top_row.addWidget(hint_btn)

        layout.addLayout(top_row)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(scroll_widget)
        self._scroll_layout.setSpacing(6)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)

        # === Window settings ===
        win_group = QGroupBox(t("subwin_basic"))
        g = QGridLayout(win_group)
        r = 0

        g.addWidget(QLabel(t("subwin_window_width")), r, 0)
        self._width_spin = QSpinBox()
        self._width_spin.setRange(200, 3840)
        self._width_spin.setSuffix(" px")
        self._width_spin.setValue(self._settings.get("window_width", 1000))
        self._width_spin.valueChanged.connect(self._on_change)
        g.addWidget(self._width_spin, r, 1)
        r += 1

        g.addWidget(QLabel(t("subwin_line_spacing")), r, 0)
        self._spacing_spin = QSpinBox()
        self._spacing_spin.setRange(0, 40)
        self._spacing_spin.setSuffix(" px")
        self._spacing_spin.setValue(self._settings.get("line_spacing", 8))
        self._spacing_spin.valueChanged.connect(self._on_change)
        g.addWidget(self._spacing_spin, r, 1)
        r += 1

        self._bg_color_label = QLabel(t("subwin_bg_color"))
        g.addWidget(self._bg_color_label, r, 0)
        self._bg_color_btn = _ColorButton(self._settings.get("bg_color", "#000000"))
        self._bg_color_btn.color_changed.connect(self._on_change)
        g.addWidget(self._bg_color_btn, r, 1)
        r += 1

        self._bg_opacity_label_title = QLabel(t("subwin_bg_opacity"))
        g.addWidget(self._bg_opacity_label_title, r, 0)
        self._bg_opacity_spin = QSpinBox()
        self._bg_opacity_spin.setRange(0, 100)
        self._bg_opacity_spin.setSuffix("%")
        self._bg_opacity_spin.setValue(round(self._settings.get("bg_opacity", 0) / 255 * 100))
        self._bg_opacity_spin.valueChanged.connect(self._on_change)
        g.addWidget(self._bg_opacity_spin, r, 1)
        r += 1

        self._bg_color_controls = [
            self._bg_color_label, self._bg_color_btn,
            self._bg_opacity_label_title, self._bg_opacity_spin,
        ]

        g.addWidget(QLabel(t("subwin_bg_image")), r, 0)
        img_row, self._win_bg_image_edit = _make_image_row(
            self._settings.get("bg_image", ""), self._on_win_bg_image_change
        )
        g.addLayout(img_row, r, 1)
        r += 1

        g.addWidget(QLabel(t("subwin_auto_hide")), r, 0)
        self._auto_hide_spin = QSpinBox()
        self._auto_hide_spin.setRange(0, 120)
        self._auto_hide_spin.setSuffix(" " + t("subwin_auto_hide_sec"))
        self._auto_hide_spin.setValue(self._settings.get("auto_hide_timeout", 0))
        self._auto_hide_spin.valueChanged.connect(self._on_change)
        g.addWidget(self._auto_hide_spin, r, 1)
        r += 1

        g.addWidget(QLabel(t("subwin_hide_animation")), r, 0)
        self._hide_anim_combo = QComboBox()
        for label, val in [(t("subwin_anim_none"), "none"), (t("subwin_anim_fade"), "fade"), (t("subwin_anim_slide_down"), "slide_down")]:
            self._hide_anim_combo.addItem(label, val)
        idx = self._hide_anim_combo.findData(self._settings.get("auto_hide_animation", "fade"))
        if idx >= 0:
            self._hide_anim_combo.setCurrentIndex(idx)
        self._hide_anim_combo.currentIndexChanged.connect(self._on_change)
        g.addWidget(self._hide_anim_combo, r, 1)
        r += 1

        g.addWidget(QLabel(t("subwin_hide_duration")), r, 0)
        self._hide_duration_spin = QSpinBox()
        self._hide_duration_spin.setRange(50, 3000)
        self._hide_duration_spin.setSuffix(" ms")
        self._hide_duration_spin.setValue(self._settings.get("auto_hide_duration", 300))
        self._hide_duration_spin.valueChanged.connect(self._on_change)
        g.addWidget(self._hide_duration_spin, r, 1)

        self._update_win_bg_controls_state()
        self._scroll_layout.addWidget(win_group)

        # === Text lines ===
        self._lines_group = QGroupBox(t("subwin_text_lines"))
        self._lines_layout = QVBoxLayout(self._lines_group)
        self._lines_layout.setSpacing(6)

        self._rebuild_lines()

        self._add_btn = QPushButton(t("subwin_add_line"))
        self._add_btn.clicked.connect(self._add_line)
        self._lines_layout.addWidget(self._add_btn)

        self._scroll_layout.addWidget(self._lines_group)
        self._scroll_layout.addStretch()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def _on_reset(self):
        ret = QMessageBox.question(
            self, t("subwin_reset"), t("subwin_reset_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret == QMessageBox.StandardButton.Yes:
            self._settings = dict(DEFAULT_SUBTITLE_WIN_SETTINGS)
            self.update_settings(self._settings)
            self._schedule_emit()

    def _on_edit_mode_toggled(self, checked):
        self.edit_mode_changed.emit(checked)

    def set_edit_mode(self, enabled: bool):
        self._edit_mode_btn.blockSignals(True)
        self._edit_mode_btn.setChecked(enabled)
        self._edit_mode_btn.blockSignals(False)

    def _on_win_bg_image_change(self):
        self._update_win_bg_controls_state()
        self._on_change()

    def _update_win_bg_controls_state(self):
        has_image = bool(self._win_bg_image_edit.text())
        for ctrl in self._bg_color_controls:
            ctrl.setEnabled(not has_image)

    def _rebuild_lines(self):
        for w in self._line_widgets:
            self._lines_layout.removeWidget(w)
            w.deleteLater()
        for sep in self._separators:
            self._lines_layout.removeWidget(sep)
            sep.deleteLater()
        self._line_widgets = []
        self._separators = []

        lines = self._settings.get("lines", DEFAULT_SUBTITLE_WIN_SETTINGS["lines"])
        n = len(lines)
        for i, cfg in enumerate(lines):
            removable = n > 1
            w = _LineConfigWidget(cfg, removable=removable)
            w.changed.connect(self._on_change)
            w.remove_requested.connect(lambda idx=i: self._remove_line(idx))
            w.move_up_requested.connect(lambda idx=i: self._move_line_up(idx))
            w.move_down_requested.connect(lambda idx=i: self._move_line_down(idx))
            w.set_move_enabled(up=i > 0, down=i < n - 1)
            self._line_widgets.append(w)
            self._lines_layout.insertWidget(self._lines_layout.count() - 1, w)

            if i < n - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("color: #555;")
                self._separators.append(sep)
                self._lines_layout.insertWidget(self._lines_layout.count() - 1, sep)

    def _move_line_up(self, index):
        if index <= 0:
            return
        lines = self._get_lines_config()
        lines[index], lines[index - 1] = lines[index - 1], lines[index]
        self._settings["lines"] = lines
        self._rebuild_lines()
        self._schedule_emit()

    def _move_line_down(self, index):
        lines = self._get_lines_config()
        if index >= len(lines) - 1:
            return
        lines[index], lines[index + 1] = lines[index + 1], lines[index]
        self._settings["lines"] = lines
        self._rebuild_lines()
        self._schedule_emit()

    def _add_line(self):
        new_line = {
            "type": "translation",
            "lang": "en",
            "enabled": True,
            "font_family": "Microsoft YaHei",
            "font_size": 24,
            "color": "#FFFFFF",
            "opacity": 255,
            "outline_enabled": True,
            "outline_color": "#000000",
            "outline_width": 2,
            "align": "center",
            "bg_image": "",
            "scroll_speed": 60,
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        }
        lines = self._get_lines_config()
        lines.append(new_line)
        self._settings["lines"] = lines
        self._rebuild_lines()
        self._schedule_emit()

    def _remove_line(self, index):
        lines = self._get_lines_config()
        if len(lines) > 1 and 0 <= index < len(lines):
            lines.pop(index)
            self._settings["lines"] = lines
            self._rebuild_lines()
            self._schedule_emit()

    def _get_lines_config(self):
        return [w.get_config() for w in self._line_widgets]

    def _on_change(self, *_):
        self._schedule_emit()

    def _schedule_emit(self):
        self._debounce_timer.start()

    def _emit_settings(self):
        s = {
            "line_spacing": self._spacing_spin.value(),
            "window_width": self._width_spin.value(),
            "bg_color": self._bg_color_btn.color(),
            "bg_opacity": round(self._bg_opacity_spin.value() / 100 * 255),
            "bg_image": self._win_bg_image_edit.text(),
            "auto_hide_timeout": self._auto_hide_spin.value(),
            "auto_hide_animation": self._hide_anim_combo.currentData() or "fade",
            "auto_hide_duration": self._hide_duration_spin.value(),
            "lines": self._get_lines_config(),
        }
        self._settings.update(s)
        self.settings_changed.emit(self._settings)

    def get_settings(self) -> dict:
        self._emit_settings()
        return dict(self._settings)


class SubtitleSettingsDialog(QDialog):
    """Standalone dialog wrapper for SubtitleSettingsWidget."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._widget = SubtitleSettingsWidget(current_settings, self)
        self._widget.settings_changed.connect(self.settings_changed.emit)
        layout.addWidget(self._widget)
        self.setWindowTitle(t("subwin_settings"))
        self.setMinimumSize(520, 500)
        self.resize(560, 640)

    def get_settings(self) -> dict:
        return self._widget.get_settings()
