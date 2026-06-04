from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QLabel, QProgressBar, QVBoxLayout

from src.app.step_indicator import StepSequenceWidget


STAGES = [
    ("frames", "Frames"),
    ("depth", "Fast3R"),
    ("reconstruct", "Geometry"),
    ("outputs", "Output"),
]

class StageProgressWidget(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("StageProgressPanel")
        self.stage_states = {key: "pending" for key, _ in STAGES}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self.stage_sequence = StepSequenceWidget(STAGES)
        layout.addWidget(self.stage_sequence)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("ReconstructionProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.current_stage_label = QLabel("Current Stage: Waiting")
        self.current_stage_label.setObjectName("StageCurrent")
        self.current_operation_label = QLabel("Current Operation: Waiting for reconstruction to start.")
        self.current_operation_label.setObjectName("MutedText")
        self.current_operation_label.setWordWrap(True)
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("MutedText")
        self.eta_label.setVisible(False)
        layout.addWidget(self.current_stage_label)
        layout.addWidget(self.current_operation_label)
        layout.addWidget(self.eta_label)
        self.reset()

    def reset(self) -> None:
        self.stage_states = {key: "pending" for key, _ in STAGES}
        self.stage_sequence.reset()
        self.set_progress(0)
        self.set_current_operation(None, "Waiting for reconstruction to start.")
        self.set_eta("")

    def set_progress(self, percent: int) -> None:
        self.progress_bar.setValue(max(0, min(100, int(percent))))

    def set_stage_status(self, stage_key: str, status: str) -> None:
        normalized = self._normalize_status(status)
        if stage_key not in self.stage_states:
            return
        self.stage_states[stage_key] = normalized
        self.stage_sequence.set_step_state(stage_key, normalized)

    def set_current_operation(self, stage_key: str | None, operation: str) -> None:
        stage_name = self._stage_name(stage_key) if stage_key else "Waiting"
        self.current_stage_label.setText(f"Current Stage: {stage_name}")
        self.current_operation_label.setText(f"Current Operation: {operation}")

    def set_eta(self, eta: str) -> None:
        self.eta_label.setVisible(bool(eta))
        self.eta_label.setText(f"ETA: {eta}" if eta else "")

    @staticmethod
    def _normalize_status(status: str) -> str:
        aliases = {
            "active": "running",
            "done": "completed",
            "complete": "completed",
            "error": "failed",
        }
        value = aliases.get(status, status)
        return value if value in {"pending", "active", "running", "completed", "failed"} else "pending"

    @staticmethod
    def _stage_name(stage_key: str | None) -> str:
        names = dict(STAGES)
        return names.get(stage_key or "", "Waiting")
