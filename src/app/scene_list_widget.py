from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QAbstractItemView, QLabel, QTreeView, QVBoxLayout, QWidget


class SceneListWidget(QWidget):
    selection_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title_label = QLabel("Scene Library")
        self.title_label.setObjectName("SidebarTitle")
        self.tree_view = QTreeView()
        self.tree_view.setObjectName("SceneTree")
        self.tree_view.setRootIsDecorated(False)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.scene_model = QStandardItemModel(0, 4, self)
        self.scene_model.setHorizontalHeaderLabels(["Scene", "Output", "Points", "Source"])
        self.tree_view.setModel(self.scene_model)
        self.tree_view.selectionModel().selectionChanged.connect(lambda *_: self.selection_changed.emit())
        self.empty_label = QLabel("No processed scenes yet.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.tree_view)
        layout.addWidget(self.empty_label)

        self._sync_empty_state()

    def set_scenes(self, scenes: list[dict]) -> None:
        self.scene_model.removeRows(0, self.scene_model.rowCount())
        for scene in scenes:
            row = self._scene_row(scene)
            for item in row:
                item.setData(scene, Qt.ItemDataRole.UserRole)
                item.setToolTip(str(scene.get("output_dir", "")))
                item.setEditable(False)
            self.scene_model.appendRow(row)
        self.tree_view.resizeColumnToContents(0)
        self.tree_view.resizeColumnToContents(1)
        self.tree_view.resizeColumnToContents(2)
        if self.scene_model.rowCount() > 0:
            self.tree_view.setCurrentIndex(self.scene_model.index(0, 0))
        self._sync_empty_state()

    def selected_scene(self) -> dict | None:
        index = self.tree_view.currentIndex()
        if not index.isValid():
            return None
        return index.siblingAtColumn(0).data(Qt.ItemDataRole.UserRole)

    def _sync_empty_state(self) -> None:
        has_items = self.scene_model.rowCount() > 0
        self.empty_label.setVisible(not has_items)
        self.tree_view.setVisible(has_items)

    @staticmethod
    def _scene_row(scene: dict) -> list[QStandardItem]:
        scene_id = scene.get("scene_id", "unknown_scene")
        source_video = Path(str(scene.get("source_video", ""))).name
        metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata", {}), dict) else {}
        outputs = metadata.get("outputs", {}) if isinstance(metadata, dict) else {}
        points = metadata.get("num_points_final") or metadata.get("processed_scaled_points") or ""
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
        return [
            QStandardItem(str(scene_id)),
            QStandardItem(f"{status}: {output_text}"),
            QStandardItem(str(points) if points else "-"),
            QStandardItem(source_video),
        ]
