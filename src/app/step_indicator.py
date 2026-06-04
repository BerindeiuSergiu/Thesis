from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget


STEP_STATES = {"pending", "active", "completed", "failed", "disabled"}


class StepIndicator(QWidget):
    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self._state = "pending"
        self.setObjectName("StepIndicator")
        self.setProperty("state", self._state)
        self.setMinimumHeight(32)
        self.setMinimumWidth(112)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.accent = QFrame()
        self.accent.setObjectName("StepAccent")
        self.accent.setFixedWidth(3)

        self.label = QLabel(label)
        self.label.setObjectName("StepLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(6)
        layout.addWidget(self.accent)
        layout.addWidget(self.label, stretch=1)

    def set_state(self, state: str) -> None:
        normalized = state if state in STEP_STATES else "pending"
        self._state = normalized
        self.setProperty("state", normalized)
        self.accent.setProperty("state", normalized)
        self.label.setProperty("state", normalized)
        self._refresh_style(self)
        self._refresh_style(self.accent)
        self._refresh_style(self.label)

    def state(self) -> str:
        return self._state

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class StepConnector(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state = "pending"
        self.setObjectName("StepConnector")
        self.setProperty("state", self._state)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(2)
        self.setMinimumWidth(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_state(self, state: str) -> None:
        normalized = state if state in STEP_STATES else "pending"
        self._state = normalized
        self.setProperty("state", normalized)
        self.style().unpolish(self)
        self.style().polish(self)

    def state(self) -> str:
        return self._state


class StepSequenceWidget(QWidget):
    def __init__(self, steps: list[tuple[str, str]], parent=None) -> None:
        super().__init__(parent)
        self.steps = steps
        self.indicators: dict[str, StepIndicator] = {}
        self.connectors: list[StepConnector] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for index, (key, label) in enumerate(steps):
            indicator = StepIndicator(label)
            self.indicators[key] = indicator
            layout.addWidget(indicator)
            if index < len(steps) - 1:
                connector = StepConnector()
                self.connectors.append(connector)
                layout.addWidget(connector)

    def set_step_state(self, key: str, state: str) -> None:
        if key not in self.indicators:
            return
        self.indicators[key].set_state(self._normalize_state(state))
        self._refresh_connectors()

    def reset(self) -> None:
        for indicator in self.indicators.values():
            indicator.set_state("pending")
        self._refresh_connectors()

    def set_all(self, state: str) -> None:
        normalized = self._normalize_state(state)
        for indicator in self.indicators.values():
            indicator.set_state(normalized)
        self._refresh_connectors()

    def _refresh_connectors(self) -> None:
        keys = [key for key, _ in self.steps]
        for index, connector in enumerate(self.connectors):
            left_state = self.indicators[keys[index]].state()
            right_state = self.indicators[keys[index + 1]].state()
            if left_state == "failed" or right_state == "failed":
                connector.set_state("failed")
            elif left_state == "completed" and right_state in {"active", "completed"}:
                connector.set_state("completed")
            elif left_state == "active" or right_state == "active":
                connector.set_state("active")
            else:
                connector.set_state("pending")

    @staticmethod
    def _normalize_state(state: str) -> str:
        aliases = {
            "active": "active",
            "running": "active",
            "done": "completed",
            "complete": "completed",
            "error": "failed",
        }
        value = aliases.get(state, state)
        return value if value in STEP_STATES else "pending"
