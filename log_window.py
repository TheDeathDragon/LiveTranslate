import logging
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QTextCursor
from i18n import t


class QLogHandler(logging.Handler):
    """Logging handler that emits to a Qt signal."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def emit(self, record):
        msg = self.format(record)
        self._signal.emit(msg, record.levelno)


class LogWindow(QWidget):
    """Real-time log viewer window."""

    log_signal = pyqtSignal(str, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("window_log"))
        self.setMinimumSize(700, 400)
        self.resize(900, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Log display
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Consolas", 9))
        self._text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        layout.addWidget(self._text)

        # Controls
        ctrl = QHBoxLayout()
        self._auto_scroll = QCheckBox(t("auto_scroll"))
        self._auto_scroll.setChecked(True)
        ctrl.addWidget(self._auto_scroll)

        self._show_debug = QCheckBox(t("show_debug"))
        self._show_debug.setChecked(False)
        ctrl.addWidget(self._show_debug)

        ctrl.addStretch()

        clear_btn = QPushButton(t("clear"))
        clear_btn.clicked.connect(self._text.clear)
        ctrl.addWidget(clear_btn)

        layout.addLayout(ctrl)

        # Connect signal
        self.log_signal.connect(self._append_log)

    def get_handler(self):
        handler = QLogHandler(self.log_signal)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        return handler

    @pyqtSlot(str, int)
    def _append_log(self, msg: str, level: int):
        if level < logging.INFO and not self._show_debug.isChecked():
            return

        color = {
            logging.DEBUG: "#808080",
            logging.INFO: "#d4d4d4",
            logging.WARNING: "#dcdcaa",
            logging.ERROR: "#f44747",
            logging.CRITICAL: "#ff0000",
        }.get(level, "#d4d4d4")

        # Highlight ASR and Translate lines
        if "ASR [" in msg:
            color = "#4ec9b0"
        elif "Translate:" in msg:
            color = "#9cdcfe"
        elif "Speech segment" in msg:
            color = "#ce9178"

        self._text.append(f'<span style="color:{color}">{msg}</span>')

        if self._auto_scroll.isChecked():
            self._text.moveCursor(QTextCursor.MoveOperation.End)
