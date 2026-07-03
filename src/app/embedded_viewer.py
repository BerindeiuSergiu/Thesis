from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - depends on optional runtime package
    QWebEngineView = None


WEBENGINE_MISSING_MESSAGE = (
    "Embedded viewer unavailable.\n\n"
    "Install the required dependency:\n"
    "pip install PyQt6-WebEngine"
)


class EmbeddedViewerWidget(QWidget):
    load_succeeded = pyqtSignal(str)
    load_failed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._loaded = False
        self._current_url = ""
        self._load_attempts = 0
        self._max_load_attempts = 12

        self.empty_label = QLabel("No reconstruction viewer loaded. Select a scene and click Open Output.")
        self.empty_label.setObjectName("PreviewBody")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty_label)

        self.web_view = QWebEngineView() if QWebEngineView is not None else None
        if self.web_view is not None:
            self.web_view.loadFinished.connect(self._on_load_finished)
            self.stack.addWidget(self.web_view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

    def load_url(self, url: str) -> bool:
        self._loaded = False
        self._current_url = url
        if self.web_view is None:
            self.empty_label.setText(WEBENGINE_MISSING_MESSAGE)
            self.stack.setCurrentWidget(self.empty_label)
            self.load_failed.emit(WEBENGINE_MISSING_MESSAGE)
            return False
        if not url:
            self.empty_label.setText("Viewer failed to load: URL unavailable.")
            self.stack.setCurrentWidget(self.empty_label)
            self.load_failed.emit("URL unavailable")
            return False

        self.empty_label.setText(f"Loading viewer...\n\n{url}")
        self.stack.setCurrentWidget(self.empty_label)
        self._load_attempts = 1
        self.web_view.load(QUrl(url))
        return True

    def clear(self) -> None:
        self._loaded = False
        self._current_url = ""
        self.empty_label.setText("No reconstruction viewer loaded. Select a scene and click Open Output.")
        self.stack.setCurrentWidget(self.empty_label)

    def show_message(self, message: str) -> None:
        self._loaded = False
        self.empty_label.setText(message)
        self.stack.setCurrentWidget(self.empty_label)

    def is_available(self) -> bool:
        return self.web_view is not None

    def reload(self) -> None:
        if self.web_view is not None and self._current_url:
            self._loaded = False
            self.web_view.reload()

    def is_loaded(self) -> bool:
        return self._loaded

    def _on_load_finished(self, ok: bool) -> None:
        self._loaded = bool(ok)
        if ok:
            self.stack.setCurrentWidget(self.web_view)
            self.load_succeeded.emit(self._current_url)
            return
        if self._current_url and self._load_attempts < self._max_load_attempts:
            self._load_attempts += 1
            QTimer.singleShot(500, self._retry_load)
            return
        self.empty_label.setText("Viewer failed to load.")
        self.stack.setCurrentWidget(self.empty_label)
        self.load_failed.emit(self._current_url or "URL unavailable")

    def _retry_load(self) -> None:
        if self.web_view is not None and self._current_url:
            self.web_view.load(QUrl(self._current_url))
