import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, QPointF, QRect, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from subtitle_overlay import SubtitleOverlay, _DragArea, _ResizeGrip


def mouse_event(kind, x, y):
    return QMouseEvent(
        kind,
        QPointF(5, 5),
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


class OverlayGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_screen_limits_prevent_huge_resize_but_allow_manual_resizing(self):
        class Screen:
            def availableGeometry(self):
                return QRect(0, 0, 1000, 700)

        overlay = SubtitleOverlay({})
        overlay._update_size_limits(Screen())
        overlay.resize(20_000, 16_000)
        self.assertEqual((overlay.width(), overlay.height()), (980, 680))

        overlay.resize(620, 500)
        self.assertEqual((overlay.width(), overlay.height()), (620, 500))

    def test_corrupt_saved_geometry_restores_inside_primary_screen(self):
        overlay = SubtitleOverlay({})
        overlay.restore_geometry(-50_000, -50_000, 20_000, 16_000)
        bounds = self.app.primaryScreen().availableGeometry()

        self.assertTrue(bounds.contains(overlay.geometry()))
        self.assertLessEqual(overlay.width(), bounds.width() - 20)
        self.assertLessEqual(overlay.height(), bounds.height() - 20)

    def test_large_saved_size_near_screen_edge_is_clamped(self):
        overlay = SubtitleOverlay({})
        bounds = self.app.primaryScreen().availableGeometry()
        overlay.restore_geometry(
            bounds.right() - 100, bounds.bottom() - 100, 20_000, 16_000
        )

        self.assertTrue(bounds.contains(overlay.geometry()))

    def test_custom_grip_resizes_window(self):
        overlay = SubtitleOverlay({})
        overlay.resize(620, 500)
        grip = overlay._grip
        self.assertIsInstance(grip, _ResizeGrip)

        grip.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 100, 100))
        grip.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 150, 125))
        grip.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, 150, 125))

        self.assertEqual((overlay.width(), overlay.height()), (670, 525))

    def test_drag_keeps_window_inside_cursor_screen(self):
        overlay = SubtitleOverlay({})
        overlay.move(50, 50)
        drag = _DragArea(overlay)
        bounds = self.app.primaryScreen().availableGeometry()

        drag.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 60, 60))
        drag.mouseMoveEvent(
            mouse_event(QEvent.Type.MouseMove, bounds.right() - 5, bounds.bottom() - 5)
        )
        drag.mouseReleaseEvent(
            mouse_event(QEvent.Type.MouseButtonRelease, bounds.right() - 5, bounds.bottom() - 5)
        )

        self.assertTrue(bounds.contains(overlay.geometry()))

    def test_drag_across_three_screens_updates_limits(self):
        class Screen:
            def __init__(self, bounds):
                self._bounds = bounds

            def availableGeometry(self):
                return self._bounds

        large = Screen(QRect(0, 0, 1200, 900))
        small = Screen(QRect(1200, 0, 800, 600))
        raised = Screen(QRect(2000, -200, 1000, 700))
        overlay = SubtitleOverlay({})
        overlay._update_size_limits(large)
        overlay.resize(900, 700)
        overlay.move(50, 50)
        drag = _DragArea(overlay)

        def screen_at(point):
            if point.x() >= 2000:
                return raised
            return small if point.x() >= 1200 else large

        with patch("subtitle_overlay.QApplication") as app_class:
            app_class.screenAt.side_effect = screen_at
            drag.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 60, 60))
            drag.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 1300, 100))
            drag.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, 1300, 100))

            self.assertTrue(
                small.availableGeometry().contains(overlay.geometry()), overlay.geometry()
            )
            self.assertEqual((overlay.maximumWidth(), overlay.maximumHeight()), (780, 580))

            drag.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 1210, 10))
            drag.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 2100, -100))
            drag.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, 2100, -100))

            self.assertTrue(raised.availableGeometry().contains(overlay.geometry()))
            self.assertEqual((overlay.maximumWidth(), overlay.maximumHeight()), (980, 680))

            drag.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 2010, -190))
            drag.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 100, 100))
            drag.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, 100, 100))

        self.assertTrue(large.availableGeometry().contains(overlay.geometry()))
        self.assertEqual((overlay.maximumWidth(), overlay.maximumHeight()), (1180, 880))


if __name__ == "__main__":
    unittest.main()
