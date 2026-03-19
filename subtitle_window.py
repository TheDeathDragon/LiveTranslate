"""
Subtitle window - clean text-only window for OBS capture.
Uses QPainterPath for outlined text rendering.

Usage:
  - Middle-click drag to move the window
  - Configure via tray menu → Subtitle Mode → Settings
  - OBS: Window Capture → select "LiveTrans Subtitle" → check "Allow Transparency"
"""

from pathlib import Path

import json

from PyQt6.QtCore import (
    Qt, QPoint, pyqtSignal, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QParallelAnimationGroup, QEasingCurve, QTimer,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QPixmap,
    QPolygon,
)
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout


def _resolve_image_path(path: str) -> str:
    """Resolve image path (relative to project dir or absolute)."""
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return str(p) if p.exists() else ""
    resolved = Path(__file__).parent / p
    return str(resolved) if resolved.exists() else ""

# Default subtitle window settings
DEFAULT_SUBTITLE_WIN_SETTINGS = {
    "enabled": False,
    "sentences": 1,
    "window_width": 1000,
    "line_spacing": 8,
    "bg_color": "#000000",
    "bg_opacity": 0,
    "bg_image": "",
    "auto_hide_timeout": 0,
    "auto_hide_animation": "fade",
    "auto_hide_duration": 300,
    "lines": [
        {
            "type": "original",
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
        },
        {
            "type": "translation",
            "lang": "zh",
            "enabled": True,
            "font_family": "Microsoft YaHei",
            "font_size": 28,
            "color": "#FFD700",
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
        },
    ],
}


def _merge_settings(base, override):
    result = {**base}
    for k, v in (override or {}).items():
        if k == "lines" and isinstance(v, list):
            result["lines"] = v
        else:
            result[k] = v
    return result


def _hex_to_rgba(hex_color: str, opacity: int) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


class _SubtitleTextWidget(QWidget):
    """Renders a single line of outlined text using QPainterPath.
    Supports horizontal scrolling, entry/exit animations via custom properties.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._font = QFont("Microsoft YaHei", 24)
        self._color = QColor(255, 255, 255)
        self._outline_enabled = True
        self._outline_color = QColor(0, 0, 0)
        self._outline_width = 2
        self._align = "center"
        self._bg_pixmap = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._render_font = None  # Auto-shrunk font when text exceeds width

        # Scroll state
        self._scroll_offset_val = 0.0
        self._scroll_speed = 60  # px/s
        self._scroll_anim = None
        self._scroll_delay_timer = None

        # Animation state
        self._content_opacity_val = 1.0
        self._slide_offset_x_val = 0.0
        self._slide_offset_y_val = 0.0
        self._entry_animation = "none"
        self._exit_animation = "none"
        self._animation_duration = 300
        self._anim_group = None
        self._pending_text = None

    # --- pyqtProperty for scroll ---
    def _get_scroll_offset(self):
        return self._scroll_offset_val

    def _set_scroll_offset(self, val):
        self._scroll_offset_val = val
        self.update()

    scroll_offset = pyqtProperty(float, _get_scroll_offset, _set_scroll_offset)

    # --- pyqtProperty for content opacity ---
    def _get_content_opacity(self):
        return self._content_opacity_val

    def _set_content_opacity(self, val):
        self._content_opacity_val = val
        self.update()

    content_opacity = pyqtProperty(float, _get_content_opacity, _set_content_opacity)

    # --- pyqtProperty for slide offsets ---
    def _get_slide_offset_x(self):
        return self._slide_offset_x_val

    def _set_slide_offset_x(self, val):
        self._slide_offset_x_val = val
        self.update()

    slide_offset_x = pyqtProperty(float, _get_slide_offset_x, _set_slide_offset_x)

    def _get_slide_offset_y(self):
        return self._slide_offset_y_val

    def _set_slide_offset_y(self, val):
        self._slide_offset_y_val = val
        self.update()

    slide_offset_y = pyqtProperty(float, _get_slide_offset_y, _set_slide_offset_y)

    def set_config(self, cfg: dict):
        self._render_font = None
        self._font = QFont(cfg.get("font_family", "Microsoft YaHei"), cfg.get("font_size", 24))
        c = QColor(cfg.get("color", "#FFFFFF"))
        c.setAlpha(cfg.get("opacity", 255))
        self._color = c
        self._outline_enabled = cfg.get("outline_enabled", True)
        self._outline_color = QColor(cfg.get("outline_color", "#000000"))
        self._outline_width = cfg.get("outline_width", 2)
        self._align = cfg.get("align", "center")
        resolved = _resolve_image_path(cfg.get("bg_image", ""))
        self._bg_pixmap = QPixmap(resolved) if resolved else None
        self._scroll_speed = cfg.get("scroll_speed", 60)
        self._entry_animation = cfg.get("entry_animation", "none")
        self._exit_animation = cfg.get("exit_animation", "none")
        self._animation_duration = cfg.get("animation_duration", 300)
        self._update_height()
        self.update()

    def set_text(self, text: str):
        # If old text exists and exit animation is configured, play exit first
        if self._text and text != self._text and self._exit_animation != "none":
            self._pending_text = text
            self._stop_all_animations()
            self._scroll_offset_val = 0.0
            self.animate_out(callback=self._apply_pending_text)
            return

        self._apply_text_immediate(text)

    def _apply_pending_text(self):
        text = getattr(self, "_pending_text", "")
        self._pending_text = None
        self._apply_text_immediate(text)

    def _apply_text_immediate(self, text: str):
        # Stop any running animations and reset to final state
        self._stop_all_animations()
        self._content_opacity_val = 1.0
        self._slide_offset_x_val = 0.0
        self._slide_offset_y_val = 0.0
        self._scroll_offset_val = 0.0
        self._pending_text = None

        self._text = text
        self._render_font = None
        self._update_height()
        self.update()

        if text:
            self.animate_in(then_scroll=True)

    def _stop_all_animations(self):
        if self._anim_group and self._anim_group.state() != self._anim_group.State.Stopped:
            self._anim_group.stop()
        self._anim_group = None
        if self._scroll_anim and self._scroll_anim.state() != self._scroll_anim.State.Stopped:
            self._scroll_anim.stop()
        self._scroll_anim = None
        if self._scroll_delay_timer:
            self._scroll_delay_timer.stop()
            self._scroll_delay_timer = None

    def animate_in(self, then_scroll=False):
        anim_type = self._entry_animation
        if anim_type == "none":
            self._content_opacity_val = 1.0
            self._slide_offset_x_val = 0.0
            self._slide_offset_y_val = 0.0
            self.update()
            if then_scroll:
                self._start_scroll()
            return

        dur = self._animation_duration
        group = QParallelAnimationGroup(self)

        # Opacity animation (all types fade in)
        opacity_anim = QPropertyAnimation(self, b"content_opacity", self)
        opacity_anim.setDuration(dur)
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)
        opacity_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(opacity_anim)

        w = self.width() or 200
        h = self.height() or 40

        if anim_type == "slide_left":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(dur)
            slide.setStartValue(float(-w))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_right":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(dur)
            slide.setStartValue(float(w))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_up":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(dur)
            slide.setStartValue(float(h))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_down":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(dur)
            slide.setStartValue(float(-h))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)

        self._content_opacity_val = 0.0
        self.update()
        self._anim_group = group

        if then_scroll:
            group.finished.connect(self._start_scroll)

        group.start()

    def animate_out(self, callback=None, anim_type=None, duration=None):
        if anim_type is None:
            anim_type = self._exit_animation
        if duration is None:
            duration = self._animation_duration
        if anim_type == "none":
            self._content_opacity_val = 0.0
            self.update()
            if callback:
                callback()
            return

        self._stop_all_animations()

        group = QParallelAnimationGroup(self)

        opacity_anim = QPropertyAnimation(self, b"content_opacity", self)
        opacity_anim.setDuration(duration)
        opacity_anim.setStartValue(self._content_opacity_val)
        opacity_anim.setEndValue(0.0)
        opacity_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        group.addAnimation(opacity_anim)

        w = self.width() or 200
        h = self.height() or 40

        if anim_type == "slide_left":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(-w))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_right":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(w))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_up":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(-h))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_down":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(h))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)

        if callback:
            group.finished.connect(callback)

        self._anim_group = group
        group.start()

    def _start_scroll(self):
        """Auto-shrink font to fit text within available width instead of scrolling."""
        if not self._text or self._scroll_speed <= 0:
            return
        fm = QFontMetrics(self._font)
        text_w = fm.horizontalAdvance(self._text)
        ow = self._outline_width if self._outline_enabled else 0
        avail_w = self.width() - ow * 2
        if text_w <= avail_w or avail_w <= 0:
            return
        # Shrink font to fit (minimum 50% of original size)
        scale = avail_w / text_w
        scale = max(scale, 0.5)
        new_size = max(int(self._font.pointSize() * scale), 8)
        self._render_font = QFont(self._font)
        self._render_font.setPointSize(new_size)
        self._update_height()
        self.update()

    def desired_height(self) -> int:
        if not self._text:
            return 0
        font = self._render_font or self._font
        fm = QFontMetrics(font)
        ow = self._outline_width if self._outline_enabled else 0
        # Single line only (no wrapping)
        return fm.lineSpacing() + ow * 2 + 4

    def _update_height(self):
        self.setFixedHeight(self.desired_height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Recalculate auto-shrink font on resize
        self._render_font = None
        if self._text:
            self._start_scroll()
        else:
            self._update_height()

    def paintEvent(self, event):
        if not self._text:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._content_opacity_val)

        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self._bg_pixmap)

        font = self._render_font or self._font
        fm = QFontMetrics(font)
        ow = self._outline_width if self._outline_enabled else 0

        # Clip to widget bounds
        painter.setClipRect(self.rect())

        # Apply slide offsets
        offset_x = self._slide_offset_x_val
        offset_y = self._slide_offset_y_val

        y = ow + fm.ascent() + offset_y

        text_w = fm.horizontalAdvance(self._text)
        if self._align == "center":
            lx = (self.width() - text_w) / 2
        elif self._align == "right":
            lx = self.width() - text_w - ow
        else:
            lx = ow

        lx += offset_x

        path = QPainterPath()
        path.addText(lx, y, font, self._text)

        if self._outline_enabled and self._outline_width > 0:
            pen = QPen(self._outline_color, self._outline_width * 2,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._color))
        painter.drawPath(path)

        painter.end()


class SubtitleWindow(QWidget):
    """Clean text-only subtitle window for OBS capture.

    Middle-click anywhere to drag. Window width is fixed (set in settings),
    height auto-fits to text content.
    """

    update_text_signal = pyqtSignal(str, str)  # original, translations_json
    position_changed = pyqtSignal()
    window_closed = pyqtSignal()
    edit_mode_changed = pyqtSignal(bool)
    size_changed = pyqtSignal(int)  # new width

    def __init__(self, settings=None):
        super().__init__()
        self._settings = _merge_settings(DEFAULT_SUBTITLE_WIN_SETTINGS, settings)
        self._text_widgets = []
        self._sentences = []  # [(original, {lang: text, ...}), ...]
        self._drag_pos = None
        self._bg_pixmap = None
        self._edit_mode = False
        self._resize_dragging = False
        self._resize_start_pos = None
        self._resize_start_width = 0

        # Auto-hide state
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self._on_auto_hide_timeout)
        self._is_hidden_by_timeout = False

        self._setup_ui()
        self.update_text_signal.connect(self._on_update_text)

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowTitle("LiveTrans Subtitle")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        s = self._settings
        w = s.get("window_width", 1000)
        saved_x = s.get("window_x")
        saved_y = s.get("window_y")
        if saved_x is not None and saved_y is not None:
            self.move(saved_x, saved_y)
        else:
            screen = QApplication.primaryScreen()
            geo = screen.geometry()
            self.move((geo.width() - w) // 2, geo.height() - 200)
        self.setFixedWidth(w)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # Content area
        self._content = QWidget()
        self._content.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 8, 16, 8)
        self._content_layout.setSpacing(s.get("line_spacing", 8))

        self._rebuild_text_widgets()

        self._main_layout.addWidget(self._content)

        self._apply_background()
        self._fit_height()

    def _rebuild_text_widgets(self):
        for w in self._text_widgets:
            self._content_layout.removeWidget(w)
            w.deleteLater()
        self._text_widgets = []

        for line_cfg in self._settings.get("lines", []):
            if not line_cfg.get("enabled", True):
                continue
            tw = _SubtitleTextWidget()
            tw.set_config(line_cfg)
            self._text_widgets.append(tw)
            self._content_layout.addWidget(tw)

    def _apply_background(self):
        s = self._settings
        resolved = _resolve_image_path(s.get("bg_image", ""))
        if resolved:
            self._bg_pixmap = QPixmap(resolved)
            self._content.setStyleSheet("background: transparent;")
        else:
            self._bg_pixmap = None
            color = s.get("bg_color", "#000000")
            opacity = s.get("bg_opacity", 0)
            if opacity == 0:
                self._content.setStyleSheet("background: transparent;")
            else:
                rgba = _hex_to_rgba(color, opacity)
                self._content.setStyleSheet(f"background: {rgba}; border-radius: 8px;")
        self.update()

    def _fit_height(self):
        """Auto-fit window height to content."""
        margins = self._content_layout.contentsMargins()
        spacing = self._content_layout.spacing()
        total = margins.top() + margins.bottom()
        visible_count = 0
        for tw in self._text_widgets:
            h = tw.desired_height()
            if h > 0:
                total += h
                visible_count += 1
        if visible_count > 1:
            total += spacing * (visible_count - 1)
        total = max(total, 20)
        self.setFixedHeight(total)

    def apply_settings(self, settings: dict):
        self._settings = _merge_settings(DEFAULT_SUBTITLE_WIN_SETTINGS, settings)

        for w in self._text_widgets:
            self._content_layout.removeWidget(w)
            w.deleteLater()
        self._text_widgets = []

        for line_cfg in self._settings.get("lines", []):
            if not line_cfg.get("enabled", True):
                continue
            tw = _SubtitleTextWidget()
            tw.set_config(line_cfg)
            self._text_widgets.append(tw)
            self._content_layout.addWidget(tw)

        self._content_layout.setSpacing(self._settings.get("line_spacing", 8))

        w = self._settings.get("window_width", 1000)
        self.setFixedWidth(w)

        self._apply_background()
        self._refresh_display()

        # Reset auto-hide timer with new settings
        self._restart_auto_hide_timer()

    # --- Auto-hide ---
    def _restart_auto_hide_timer(self):
        timeout = self._settings.get("auto_hide_timeout", 0)
        self._auto_hide_timer.stop()
        if timeout > 0 and self._sentences:
            self._auto_hide_timer.setInterval(timeout * 1000)
            self._auto_hide_timer.start()

    def _on_auto_hide_timeout(self):
        if self._is_hidden_by_timeout:
            return
        self._is_hidden_by_timeout = True
        anim_type = self._settings.get("auto_hide_animation", "fade")
        duration = self._settings.get("auto_hide_duration", 300)
        for tw in self._text_widgets:
            tw.animate_out(anim_type=anim_type, duration=duration)

    def _restore_from_auto_hide(self):
        if not self._is_hidden_by_timeout:
            return
        self._is_hidden_by_timeout = False
        anim_type = self._settings.get("auto_hide_animation", "fade")
        duration = self._settings.get("auto_hide_duration", 300)
        # Reverse the hide animation type for restore
        restore_type = anim_type
        if anim_type == "slide_down":
            restore_type = "slide_up"
        elif anim_type == "slide_up":
            restore_type = "slide_down"
        elif anim_type == "slide_left":
            restore_type = "slide_right"
        elif anim_type == "slide_right":
            restore_type = "slide_left"

        for tw in self._text_widgets:
            tw._stop_all_animations()
            # Set hidden state
            tw._content_opacity_val = 0.0
            if restore_type == "slide_left":
                tw._slide_offset_x_val = float(-(tw.width() or 200))
            elif restore_type == "slide_right":
                tw._slide_offset_x_val = float(tw.width() or 200)
            elif restore_type == "slide_up":
                tw._slide_offset_y_val = float(tw.height() or 40)
            elif restore_type == "slide_down":
                tw._slide_offset_y_val = float(-(tw.height() or 40))
            tw.update()
            # Use the entry animation mechanism with overridden type
            old_entry = tw._entry_animation
            old_dur = tw._animation_duration
            tw._entry_animation = restore_type if restore_type != "none" else "fade"
            tw._animation_duration = duration
            tw.animate_in(then_scroll=False)
            tw._entry_animation = old_entry
            tw._animation_duration = old_dur

    # --- Edit mode ---
    def set_edit_mode(self, enabled: bool):
        if self._edit_mode == enabled:
            return
        self._edit_mode = enabled
        if enabled:
            self.setMinimumWidth(200)
            self.setMaximumWidth(3840)
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            w = self.width()
            self.setFixedWidth(w)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._settings["window_width"] = w
            self.size_changed.emit(w)
        self.update()

    def _in_resize_zone(self, pos):
        """Check if position is in bottom-right resize corner (16x16)."""
        return (self._edit_mode
                and pos.x() >= self.width() - 16
                and pos.y() >= self.height() - 16)

    # --- Mouse events (middle-click drag + edit mode left-click drag/resize) ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
        elif event.button() == Qt.MouseButton.LeftButton and self._edit_mode:
            if self._in_resize_zone(event.pos()):
                self._resize_dragging = True
                self._resize_start_pos = event.globalPosition().toPoint()
                self._resize_start_width = self.width()
            else:
                self._drag_pos = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_dragging:
            dx = event.globalPosition().toPoint().x() - self._resize_start_pos.x()
            new_w = max(200, min(3840, self._resize_start_width + dx))
            self.setFixedWidth(new_w)
            self._fit_height()
        elif self._drag_pos:
            btns = event.buttons()
            if btns & Qt.MouseButton.MiddleButton or (self._edit_mode and btns & Qt.MouseButton.LeftButton):
                self.move(event.globalPosition().toPoint() - self._drag_pos)
        elif self._edit_mode:
            if self._in_resize_zone(event.pos()):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            if self._drag_pos:
                self._drag_pos = None
                self.position_changed.emit()
        elif event.button() == Qt.MouseButton.LeftButton and self._edit_mode:
            if self._resize_dragging:
                self._resize_dragging = False
                self._resize_start_pos = None
                self.size_changed.emit(self.width())
                self.position_changed.emit()
            elif self._drag_pos:
                self._drag_pos = None
                self.position_changed.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._edit_mode:
            self.set_edit_mode(False)
            self.edit_mode_changed.emit(False)
            return
        super().mouseDoubleClickEvent(event)

    def closeEvent(self, event):
        self.window_closed.emit()
        super().closeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self._bg_pixmap)
        if self._edit_mode:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # Draw edit mode border
            pen = QPen(QColor(100, 200, 100, 180), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(1, 1, self.width() - 2, self.height() - 2)
            # Draw resize grip triangle
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(100, 200, 100, 150)))
            grip = QPolygon([
                QPoint(self.width(), self.height() - 12),
                QPoint(self.width() - 12, self.height()),
                QPoint(self.width(), self.height()),
            ])
            painter.drawPolygon(grip)
        painter.end()
        super().paintEvent(event)

    # --- Text updates ---
    def update_text(self, original: str, translations: dict | str):
        """Thread-safe text update.

        translations: dict mapping lang code to translated text,
                      or a plain string (backward compat, treated as primary target).
        """
        if isinstance(translations, str):
            # Backward compat: wrap in dict with empty key
            translations = {"": translations}
        self.update_text_signal.emit(original, json.dumps(translations, ensure_ascii=False))

    @pyqtSlot(str, str)
    def _on_update_text(self, original: str, translations_json: str):
        translations = json.loads(translations_json)
        max_sentences = self._settings.get("sentences", 1)
        self._sentences.append((original, translations))
        if len(self._sentences) > max_sentences:
            self._sentences = self._sentences[-max_sentences:]

        # Restore from auto-hide if needed
        if self._is_hidden_by_timeout:
            self._restore_from_auto_hide()

        self._refresh_display()

        # Reset auto-hide timer
        self._restart_auto_hide_timer()

    def _refresh_display(self):
        if not self._sentences:
            for tw in self._text_widgets:
                tw.set_text("")
            self._fit_height()
            return

        lines_cfg = [ln for ln in self._settings.get("lines", []) if ln.get("enabled", True)]
        wi = 0

        for cfg in lines_cfg:
            if wi >= len(self._text_widgets):
                break
            tw = self._text_widgets[wi]
            line_type = cfg.get("type", "original")

            if line_type == "original":
                texts = [s[0] for s in self._sentences if s[0]]
            else:
                lang = cfg.get("lang", "")
                texts = []
                for _, tl_dict in self._sentences:
                    # tl_dict is {lang_code: text} or {"": text} for backward compat
                    if isinstance(tl_dict, str):
                        if tl_dict:
                            texts.append(tl_dict)
                    elif lang and lang in tl_dict:
                        texts.append(tl_dict[lang])
                    elif "" in tl_dict and tl_dict[""]:
                        # Fallback: use the default (empty key) translation
                        texts.append(tl_dict[""])
                    else:
                        # Fallback: use any available translation
                        for v in tl_dict.values():
                            if v:
                                texts.append(v)
                                break
            tw.set_text(" | ".join(texts) if len(texts) > 1 else (texts[0] if texts else ""))
            wi += 1

        self._fit_height()

    def get_target_languages(self) -> set:
        """Return set of unique target language codes from enabled translation lines."""
        langs = set()
        for cfg in self._settings.get("lines", []):
            if cfg.get("enabled", True) and cfg.get("type") == "translation":
                lang = cfg.get("lang", "")
                if lang:
                    langs.add(lang)
        return langs

    def clear(self):
        self._sentences.clear()
        self._auto_hide_timer.stop()
        self._is_hidden_by_timeout = False
        for tw in self._text_widgets:
            tw._stop_all_animations()
            tw._content_opacity_val = 1.0
            tw._slide_offset_x_val = 0.0
            tw._slide_offset_y_val = 0.0
            tw.set_text("")
        self._fit_height()
