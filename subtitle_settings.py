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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
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


def _make_image_rows(current_path: str, on_change):
    """Create background image selector (2-row layout). Returns (layout, line_edit)."""
    box = QVBoxLayout()
    box.setSpacing(2)

    line_edit = QLineEdit()
    line_edit.setReadOnly(True)
    line_edit.setText(current_path)
    box.addWidget(line_edit)

    btn_row = QHBoxLayout()
    select_btn = QPushButton(t("subwin_bg_image_select"))

    def _select():
        path, _ = QFileDialog.getOpenFileName(
            line_edit.window(),
            t("subwin_bg_image_select"),
            "",
            "Images (*.png *.webp *.jpg *.jpeg *.bmp)",
        )
        if path:
            try:
                rel = os.path.relpath(path, _PROJECT_DIR)
                if not rel.startswith(".."):
                    path = rel.replace("\\", "/")
            except ValueError:
                pass
            line_edit.setText(path)
            on_change()

    select_btn.clicked.connect(_select)
    btn_row.addWidget(select_btn)

    clear_btn = QPushButton(t("subwin_bg_image_clear"))

    def _clear():
        line_edit.setText("")
        on_change()

    clear_btn.clicked.connect(_clear)
    btn_row.addWidget(clear_btn)
    box.addLayout(btn_row)

    return box, line_edit


class LineEditDialog(QDialog):
    """Dialog for editing a single subtitle text line configuration."""

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self._cfg = dict(cfg)
        self._color_controls = []
        self.setWindowTitle(t("subwin_edit_line"))
        self.setMinimumWidth(400)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        r = 0

        grid.addWidget(QLabel(t("subwin_enabled")), r, 0)
        self._enabled = QCheckBox()
        self._enabled.setChecked(self._cfg.get("enabled", True))
        grid.addWidget(self._enabled, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_line_type")), r, 0)
        self._type_combo = QComboBox()
        self._type_combo.addItem(t("subwin_original"), "original")
        self._type_combo.addItem(t("subwin_translation"), "translation")
        idx = self._type_combo.findData(self._cfg.get("type", "original"))
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        self._type_combo.currentIndexChanged.connect(self._update_lang_visibility)
        grid.addWidget(self._type_combo, r, 1)
        r += 1

        self._lang_label = QLabel(t("subwin_target_lang"))
        grid.addWidget(self._lang_label, r, 0)
        self._lang_combo = QComboBox()
        for code, native in LANGUAGES:
            if code == "auto":
                continue
            self._lang_combo.addItem(f"{code} - {native}", code)
        idx = self._lang_combo.findData(self._cfg.get("lang", "zh"))
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        grid.addWidget(self._lang_combo, r, 1)
        self._lang_row = r
        r += 1

        grid.addWidget(QLabel(t("subwin_font")), r, 0)
        self._font_combo = QComboBox()
        self._font_combo.addItems(QFontDatabase.families())
        idx = self._font_combo.findText(self._cfg.get("font_family", "Microsoft YaHei"))
        if idx >= 0:
            self._font_combo.setCurrentIndex(idx)
        grid.addWidget(self._font_combo, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_font_size")), r, 0)
        self._size_spin = QSpinBox()
        self._size_spin.setRange(8, 120)
        self._size_spin.setSuffix(" pt")
        self._size_spin.setValue(self._cfg.get("font_size", 24))
        grid.addWidget(self._size_spin, r, 1)
        r += 1

        lbl_color = QLabel(t("subwin_color"))
        grid.addWidget(lbl_color, r, 0)
        self._color_btn = _ColorButton(self._cfg.get("color", "#FFFFFF"))
        grid.addWidget(self._color_btn, r, 1)
        self._color_controls.append(lbl_color)
        self._color_controls.append(self._color_btn)
        r += 1

        grid.addWidget(QLabel(t("subwin_opacity")), r, 0)
        self._opacity_spin = QSpinBox()
        self._opacity_spin.setRange(0, 100)
        self._opacity_spin.setSuffix("%")
        self._opacity_spin.setValue(round(self._cfg.get("opacity", 255) / 255 * 100))
        grid.addWidget(self._opacity_spin, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_align")), r, 0)
        self._align_combo = QComboBox()
        self._align_combo.addItem(t("subwin_align_left"), "left")
        self._align_combo.addItem(t("subwin_align_center"), "center")
        self._align_combo.addItem(t("subwin_align_right"), "right")
        idx = self._align_combo.findData(self._cfg.get("align", "center"))
        if idx >= 0:
            self._align_combo.setCurrentIndex(idx)
        grid.addWidget(self._align_combo, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_outline")), r, 0)
        self._outline_check = QCheckBox()
        self._outline_check.setChecked(self._cfg.get("outline_enabled", True))
        grid.addWidget(self._outline_check, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_outline_color")), r, 0)
        self._outline_color_btn = _ColorButton(self._cfg.get("outline_color", "#000000"))
        grid.addWidget(self._outline_color_btn, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_outline_width")), r, 0)
        self._outline_width = QSpinBox()
        self._outline_width.setRange(0, 10)
        self._outline_width.setSuffix(" px")
        self._outline_width.setValue(self._cfg.get("outline_width", 2))
        grid.addWidget(self._outline_width, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_bg_image")), r, 0)
        img_row, self._bg_image_edit = _make_image_rows(self._cfg.get("bg_image", ""), lambda: None)
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
        grid.addWidget(self._entry_anim_combo, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_exit_anim")), r, 0)
        self._exit_anim_combo = QComboBox()
        for label, val in anim_items:
            self._exit_anim_combo.addItem(label, val)
        idx = self._exit_anim_combo.findData(self._cfg.get("exit_animation", "none"))
        if idx >= 0:
            self._exit_anim_combo.setCurrentIndex(idx)
        grid.addWidget(self._exit_anim_combo, r, 1)
        r += 1

        grid.addWidget(QLabel(t("subwin_anim_duration")), r, 0)
        self._anim_duration_spin = QSpinBox()
        self._anim_duration_spin.setRange(50, 3000)
        self._anim_duration_spin.setSuffix(" ms")
        self._anim_duration_spin.setValue(self._cfg.get("animation_duration", 300))
        grid.addWidget(self._anim_duration_spin, r, 1)

        layout.addLayout(grid)
        self._update_lang_visibility()

        # OK / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        cancel_btn = QPushButton(t("subwin_cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _update_lang_visibility(self):
        is_translation = self._type_combo.currentData() == "translation"
        self._lang_label.setEnabled(is_translation)
        self._lang_combo.setEnabled(is_translation)

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

    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        self._settings = {**DEFAULT_SUBTITLE_WIN_SETTINGS, **(current_settings or {})}
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
        self._border_radius_spin.setValue(self._settings.get("border_radius", 8))
        self._win_bg_image_edit.setText(self._settings.get("bg_image", ""))
        self._auto_hide_spin.setValue(self._settings.get("auto_hide_timeout", 0))
        idx = self._hide_anim_combo.findData(self._settings.get("auto_hide_animation", "fade"))
        if idx >= 0:
            self._hide_anim_combo.setCurrentIndex(idx)
        self._hide_duration_spin.setValue(self._settings.get("auto_hide_duration", 300))
        self._refresh_lines_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        # === Window settings ===
        win_group = QGroupBox(t("subwin_basic"))
        g = QGridLayout(win_group)
        g.setColumnStretch(0, 1)
        g.setColumnMinimumWidth(1, 180)
        r = 0

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        reset_btn = QPushButton(t("subwin_reset"))
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)
        g.addLayout(btn_row, r, 0, 1, 2)
        r += 1

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

        g.addWidget(QLabel(t("subwin_border_radius")), r, 0)
        self._border_radius_spin = QSpinBox()
        self._border_radius_spin.setRange(0, 30)
        self._border_radius_spin.setSuffix(" px")
        self._border_radius_spin.setValue(self._settings.get("border_radius", 8))
        self._border_radius_spin.valueChanged.connect(self._on_change)
        g.addWidget(self._border_radius_spin, r, 1)
        r += 1

        g.addWidget(QLabel(t("subwin_bg_image")), r, 0)
        img_row, self._win_bg_image_edit = _make_image_rows(
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
        layout.addWidget(win_group)

        # === Text lines (list + edit dialog) ===
        lines_group = QGroupBox(t("subwin_text_lines"))
        lines_layout = QVBoxLayout(lines_group)

        self._lines_list = QListWidget()
        self._lines_list.itemDoubleClicked.connect(self._edit_line)
        self._refresh_lines_list()
        lines_layout.addWidget(self._lines_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton(t("btn_add"))
        add_btn.clicked.connect(self._add_line)
        btn_row.addWidget(add_btn)
        edit_btn = QPushButton(t("btn_edit"))
        edit_btn.clicked.connect(self._edit_current_line)
        btn_row.addWidget(edit_btn)
        del_btn = QPushButton(t("btn_remove"))
        del_btn.clicked.connect(self._remove_line)
        btn_row.addWidget(del_btn)
        up_btn = QPushButton(t("subwin_move_up"))
        up_btn.clicked.connect(self._move_line_up)
        btn_row.addWidget(up_btn)
        down_btn = QPushButton(t("subwin_move_down"))
        down_btn.clicked.connect(self._move_line_down)
        btn_row.addWidget(down_btn)
        lines_layout.addLayout(btn_row)

        layout.addWidget(lines_group)

    def _on_reset(self):
        ret = QMessageBox.question(
            self, t("subwin_reset"), t("subwin_reset_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret == QMessageBox.StandardButton.Yes:
            self._settings = dict(DEFAULT_SUBTITLE_WIN_SETTINGS)
            self.update_settings(self._settings)
            self._schedule_emit()

    def _on_win_bg_image_change(self):
        self._update_win_bg_controls_state()
        self._on_change()

    def _update_win_bg_controls_state(self):
        has_image = bool(self._win_bg_image_edit.text())
        for ctrl in self._bg_color_controls:
            ctrl.setEnabled(not has_image)

    def _refresh_lines_list(self):
        self._lines_list.clear()
        lines = self._settings.get("lines", DEFAULT_SUBTITLE_WIN_SETTINGS["lines"])
        for cfg in lines:
            line_type = cfg.get("type", "original")
            enabled = cfg.get("enabled", True)
            label = t("subwin_original") if line_type == "original" else t("subwin_translation")
            if line_type == "translation":
                label += f" ({cfg.get('lang', 'zh')})"
            font = cfg.get("font_family", "Microsoft YaHei")
            size = cfg.get("font_size", 24)
            color = cfg.get("color", "#FFF")
            align = cfg.get("align", "center")
            outline = t("subwin_outline") if cfg.get("outline_enabled", True) else ""
            entry = cfg.get("entry_animation", "none")
            parts = [
                "✓" if enabled else "✗",
                label,
                f"{font} {size}pt",
                color,
                align,
            ]
            if outline:
                parts.append(outline)
            if entry != "none":
                parts.append(entry)
            text = "  |  ".join(parts)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, cfg)
            self._lines_list.addItem(item)

    def _edit_line(self, item):
        cfg = item.data(Qt.ItemDataRole.UserRole)
        row = self._lines_list.row(item)
        dlg = LineEditDialog(cfg, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_cfg = dlg.get_config()
            lines = self._settings.get("lines", [])[:]
            lines[row] = new_cfg
            self._settings["lines"] = lines
            self._refresh_lines_list()
            self._schedule_emit()

    def _edit_current_line(self):
        item = self._lines_list.currentItem()
        if item:
            self._edit_line(item)

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
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        }
        dlg = LineEditDialog(new_line, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            lines = self._settings.get("lines", [])[:]
            lines.append(dlg.get_config())
            self._settings["lines"] = lines
            self._refresh_lines_list()
            self._schedule_emit()

    def _remove_line(self):
        row = self._lines_list.currentRow()
        lines = self._settings.get("lines", [])[:]
        if len(lines) > 1 and 0 <= row < len(lines):
            lines.pop(row)
            self._settings["lines"] = lines
            self._refresh_lines_list()
            self._schedule_emit()

    def _move_line_up(self):
        row = self._lines_list.currentRow()
        lines = self._settings.get("lines", [])[:]
        if row > 0:
            lines[row], lines[row - 1] = lines[row - 1], lines[row]
            self._settings["lines"] = lines
            self._refresh_lines_list()
            self._lines_list.setCurrentRow(row - 1)
            self._schedule_emit()

    def _move_line_down(self):
        row = self._lines_list.currentRow()
        lines = self._settings.get("lines", [])[:]
        if 0 <= row < len(lines) - 1:
            lines[row], lines[row + 1] = lines[row + 1], lines[row]
            self._settings["lines"] = lines
            self._refresh_lines_list()
            self._lines_list.setCurrentRow(row + 1)
            self._schedule_emit()

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
            "border_radius": self._border_radius_spin.value(),
            "bg_image": self._win_bg_image_edit.text(),
            "auto_hide_timeout": self._auto_hide_spin.value(),
            "auto_hide_animation": self._hide_anim_combo.currentData() or "fade",
            "auto_hide_duration": self._hide_duration_spin.value(),
            "lines": self._settings.get("lines", DEFAULT_SUBTITLE_WIN_SETTINGS["lines"]),
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
