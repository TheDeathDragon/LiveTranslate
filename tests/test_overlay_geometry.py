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


class Screen:
    def __init__(self, x, y, width, height):
        self.bounds = QRect(x, y, width, height)

    def availableGeometry(self):
        return self.bounds


class OverlayGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.screens = [
            Screen(0, 0, 1200, 900),
            Screen(1200, 0, 800, 600),
            Screen(2000, -200, 1000, 700),
        ]
        self.overlay = SubtitleOverlay({})
        self.mock_app = patch("subtitle_overlay.QApplication")
        self.app_class = self.mock_app.start()
        self.addCleanup(self.mock_app.stop)
        self.addCleanup(self.overlay.close)
        self.app_class.screens.return_value = self.screens
        self.app_class.primaryScreen.return_value = self.screens[0]
        self.app_class.screenAt.side_effect = self.screen_at
        self.overlay._update_size_limits(self.screens[0])

    def screen_at(self, point):
        for screen in self.screens:
            if screen.bounds.contains(point):
                return screen
        return None

    def test_drag_can_span_three_screens_without_changing_size(self):
        overlay = self.overlay
        overlay._resize_intentionally(900, 700)
        overlay.move(50, 50)
        drag = _DragArea(overlay)

        drag.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 60, 60))
        drag.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 1150, 100))
        self.assertEqual((overlay.x(), overlay.y()), (1140, 90))
        self.assertEqual((overlay.width(), overlay.height()), (900, 700))
        self.assertTrue(overlay.geometry().intersects(self.screens[0].bounds))
        self.assertTrue(overlay.geometry().intersects(self.screens[1].bounds))

        drag.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 1300, 100))
        self.assertEqual((overlay.x(), overlay.y()), (1290, 90))
        self.assertEqual((overlay.width(), overlay.height()), (900, 700))

        drag.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 2100, -100))
        self.assertEqual((overlay.x(), overlay.y()), (2090, -110))
        self.assertEqual((overlay.width(), overlay.height()), (900, 700))

        drag.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 100, 100))
        drag.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, 100, 100))
        self.assertEqual((overlay.x(), overlay.y()), (90, 90))
        self.assertEqual((overlay.width(), overlay.height()), (900, 700))

    def test_restore_keeps_cross_screen_placement(self):
        self.overlay.restore_geometry(1140, 90, 900, 700)
        self.assertEqual((self.overlay.x(), self.overlay.y()), (1140, 90))
        self.assertEqual((self.overlay.width(), self.overlay.height()), (900, 700))
        self.assertTrue(self.overlay.geometry().intersects(self.screens[1].bounds))

        self.overlay.restore_geometry(1290, 90, 900, 700)
        self.assertEqual((self.overlay.x(), self.overlay.y()), (1290, 90))
        self.assertEqual((self.overlay.width(), self.overlay.height()), (900, 700))

    def test_corrupt_saved_size_is_clamped_to_a_screen(self):
        self.overlay.restore_geometry(1300, 100, 20_000, 16_000)
        self.assertTrue(self.screens[1].bounds.contains(self.overlay.geometry()))
        self.assertLessEqual(self.overlay.width(), 640)
        self.assertLessEqual(self.overlay.height(), 480)

    def test_disconnected_saved_position_fits_primary_screen(self):
        self.app_class.primaryScreen.return_value = self.screens[1]
        self.overlay.restore_geometry(50_000, 50_000, 900, 700)
        self.assertTrue(self.screens[1].bounds.contains(self.overlay.geometry()))
        self.assertLessEqual(self.overlay.width(), 640)
        self.assertLessEqual(self.overlay.height(), 480)

    def test_absurd_saved_coordinates_fall_back_to_primary_screen(self):
        self.overlay.restore_geometry(10**12, -(10**12), 620, 500)
        self.assertTrue(self.screens[0].bounds.contains(self.overlay.geometry()))

    def test_restore_reapplies_size_limit_after_temporary_global_cap(self):
        self.overlay._update_size_limits(self.screens[1])
        self.overlay.restore_geometry(1300, 100, 620, 450)
        self.assertEqual((self.overlay.maximumWidth(), self.overlay.maximumHeight()), (640, 480))

    def test_resize_grip_stays_bounded_and_adjustable(self):
        self.overlay.resize(620, 500)
        grip = self.overlay._grip
        self.assertIsInstance(grip, _ResizeGrip)
        grip.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 100, 100))
        grip.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 20_000, 16_000))
        grip.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, 20_000, 16_000))
        self.assertEqual((self.overlay.width(), self.overlay.height()), (960, 720))

        grip.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 100, 100))
        grip.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, -160, -120))
        grip.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, -160, -120))
        self.assertEqual((self.overlay.width(), self.overlay.height()), (700, 500))

    def test_monitor_resize_does_not_replace_manually_chosen_width(self):
        overlay = self.overlay
        overlay.show()
        overlay.move(50, 50)
        grip = overlay._grip
        grip.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 100, 100))
        grip.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 380, 100))
        grip.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, 380, 100))
        chosen_width = overlay.width()

        drag = _DragArea(overlay)
        drag.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, 60, 60))
        drag.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, 1300, 100))
        overlay.setMaximumSize(960, 720)  # Native DPI transition can change constraints.
        overlay.resize(chosen_width + 60, overlay.height())  # Qt DPI resize
        self.assertEqual(overlay.width(), chosen_width)

        drag.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, 1300, 100))
        overlay.setMaximumSize(960, 720)
        overlay.resize(chosen_width + 60, overlay.height())  # Delayed DPI resize.

        self.assertEqual(overlay.width(), chosen_width)


if __name__ == "__main__":
    unittest.main()
