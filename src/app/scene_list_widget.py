from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class SceneListWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title_label = QLabel("Scene Library")
        self.title_label.setObjectName("SidebarTitle")
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("SceneList")
        self.list_widget.setSpacing(6)
        self.empty_label = QLabel("No processed scenes yet.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.empty_label)

        self._sync_empty_state()

    def set_scenes(self, scenes: list[dict]) -> None:
        self.list_widget.clear()
        for scene in scenes:
            item = QListWidgetItem(self._scene_label(scene))
            item.setData(Qt.ItemDataRole.UserRole, scene)
            item.setToolTip(str(scene.get("output_dir", "")))
            item.setSizeHint(QSize(220, 76))
            self.list_widget.addItem(item)
        self._sync_empty_state()

    def selected_scene(self) -> dict | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _sync_empty_state(self) -> None:
        has_items = self.list_widget.count() > 0
        self.empty_label.setVisible(not has_items)
        self.list_widget.setVisible(has_items)

    @staticmethod
    def _scene_label(scene: dict) -> str:
        scene_id = scene.get("scene_id", "unknown_scene")
        source_video = Path(str(scene.get("source_video", ""))).name
        metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata", {}), dict) else {}
        outputs = metadata.get("outputs", {}) if isinstance(metadata, dict) else {}
        has_gaussian = bool(outputs.get("gaussian_ply")) or bool(
            (metadata.get("gaussian_splat", {}) if isinstance(metadata, dict) else {}).get("gaussian_ply")
        )
        has_mesh = bool(outputs.get("mesh_ply")) or bool(scene.get("mesh_path"))
        badges = []
        if has_gaussian:
            badges.append("Gaussian")
        if has_mesh:
            badges.append("Mesh")
        status = "Ready" if badges else "Point Cloud"
        output_text = " / ".join(badges) if badges else "PLY"
        if len(source_video) > 28:
            source_video = source_video[:25] + "..."
        return f"{scene_id}\n{status} - {output_text}\n{source_video}"
