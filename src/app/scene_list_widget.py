from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QAbstractItemView, QLabel, QTreeView, QVBoxLayout, QWidget

from src.models.scene import Scene


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
        self.scene_model = QStandardItemModel(0, 5, self)
        self.scene_model.setHorizontalHeaderLabels(["Scene", "Status", "Created", "Pipeline", "Reconstruction"])
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
        self.tree_view.resizeColumnToContents(3)
        if self.scene_model.rowCount() > 0:
            self.tree_view.setCurrentIndex(self.scene_model.index(0, 0))
        self._sync_empty_state()

    def selected_scene(self) -> dict | None:
        index = self.tree_view.currentIndex()
        if not index.isValid():
            return None
        return index.siblingAtColumn(0).data(Qt.ItemDataRole.UserRole)

    def select_scene(self, scene_id: str) -> None:
        for row in range(self.scene_model.rowCount()):
            record = self.scene_model.index(row, 0).data(Qt.ItemDataRole.UserRole)
            if isinstance(record, dict) and record.get("scene_id") == scene_id:
                self.tree_view.setCurrentIndex(self.scene_model.index(row, 0))
                return

    def _sync_empty_state(self) -> None:
        has_items = self.scene_model.rowCount() > 0
        self.empty_label.setVisible(not has_items)
        self.tree_view.setVisible(has_items)

    @staticmethod
    def _scene_row(scene: dict) -> list[QStandardItem]:
        scene_entity = Scene.from_record(scene)
        return [
            QStandardItem(f"[Scene] {scene_entity.name}"),
            QStandardItem(scene_entity.status),
            QStandardItem(_format_created_at(scene_entity.created_at)),
            QStandardItem(scene_entity.pipeline),
            QStandardItem(scene_entity.reconstruction_type),
        ]


def _format_created_at(value: str) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value).strftime("%d-%m-%Y")
    except ValueError:
        return value[:10]
