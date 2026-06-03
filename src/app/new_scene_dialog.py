from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from src.models.scene import Scene


class NewSceneDialog(QDialog):
    def __init__(
        self,
        pipelines: list[str],
        presets: list[str],
        default_pipeline: str = "default",
        default_preset: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Scene")
        self.setMinimumWidth(520)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Example: Living Room Apartment")
        self.video_edit = QLineEdit()
        self.video_edit.setReadOnly(True)
        self.video_edit.setPlaceholderText("Choose a source video")
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Optional notes about capture conditions or intent")
        self.description_edit.setFixedHeight(90)
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItems(pipelines)
        pipeline_index = self.pipeline_combo.findText(default_pipeline)
        if pipeline_index >= 0:
            self.pipeline_combo.setCurrentIndex(pipeline_index)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(presets)
        preset_index = self.preset_combo.findText(default_preset)
        if preset_index >= 0:
            self.preset_combo.setCurrentIndex(preset_index)
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("Optional, comma separated")

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_video)
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_edit, stretch=1)
        video_row.addWidget(browse_button)

        form = QFormLayout()
        form.addRow("Scene Name *", self.name_edit)
        form.addRow("Source Video *", video_row)
        form.addRow("Description", self.description_edit)
        form.addRow("Pipeline", self.pipeline_combo)
        form.addRow("Reconstruction Preset", self.preset_combo)
        form.addRow("Tags", self.tags_edit)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

    def scene(self) -> Scene:
        tags = [tag.strip() for tag in self.tags_edit.text().split(",") if tag.strip()]
        timestamp = datetime.now().isoformat(timespec="seconds")
        return Scene(
            scene_id=f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
            name=self.name_edit.text().strip(),
            source_video=Path(self.video_edit.text()).resolve(),
            description=self.description_edit.toPlainText().strip(),
            pipeline=self.pipeline_combo.currentText(),
            reconstruction_preset=self.preset_combo.currentText(),
            tags=tags,
            status="Draft",
            created_at=timestamp,
            reconstruction_type="Not reconstructed",
        )

    def _browse_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Source Video",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)",
        )
        if filename:
            self.video_edit.setText(filename)
            if not self.name_edit.text().strip():
                self.name_edit.setText(Path(filename).stem.replace("_", " ").title())

    def _accept_if_valid(self) -> None:
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Scene Name Required", "Please enter a scene name.")
            return
        if not self.video_edit.text().strip():
            QMessageBox.warning(self, "Source Video Required", "Please choose a source video.")
            return
        self.accept()
