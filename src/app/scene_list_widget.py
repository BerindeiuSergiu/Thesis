from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QLabel, QTreeView, QVBoxLayout, QWidget

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
        self.tree_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tree_view.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree_view.setUniformRowHeights(True)
        self.scene_model = QStandardItemModel(0, 5, self)
        self.scene_model.setHorizontalHeaderLabels(["Scene", "Status", "Created", "Pipeline", "Reconstruction"])
        self.tree_view.setModel(self.scene_model)
        header = self.tree_view.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.tree_view.setColumnWidth(1, 70)
        self.tree_view.setColumnWidth(2, 78)
        self.tree_view.setColumnWidth(3, 68)
        self.tree_view.setColumnWidth(4, 105)
        header.setSectionsMovable(False)
        self.tree_view.selectionModel().selectionChanged.connect(lambda *_: self.selection_changed.emit())
        self.empty_label = QLabel("No processed scenes yet.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.tree_view)
        layout.addWidget(self.empty_label)

        self._sync_empty_state()

    def set_scenes(self, scenes: list[dict]) -> None:
        previous_scene_id = None
        previous = self.selected_scene()
        if isinstance(previous, dict):
            previous_scene_id = previous.get("scene_id")

        self.scene_model.removeRows(0, self.scene_model.rowCount())
        for scene in scenes:
            row = self._scene_row(scene)
            for item in row:
                item.setData(scene, Qt.ItemDataRole.UserRole)
                item.setToolTip(_scene_tooltip(scene))
                item.setEditable(False)
            self.scene_model.appendRow(row)
        if self.scene_model.rowCount() > 0:
            if previous_scene_id:
                self.select_scene(str(previous_scene_id))
            if not self.tree_view.currentIndex().isValid():
                self.tree_view.setCurrentIndex(self.scene_model.index(0, 0))
        self._sync_empty_state()

    def selected_scene(self) -> dict | None:
        selected_rows = self.tree_view.selectionModel().selectedRows(0)
        index = selected_rows[0] if selected_rows else self.tree_view.currentIndex()
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
            QStandardItem(scene_entity.name),
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


def _scene_tooltip(scene: dict) -> str:
    scene_entity = Scene.from_record(scene)
    return "\n".join(
        [
            f"Scene: {scene_entity.name}",
            f"Status: {scene_entity.status}",
            f"Created: {_format_created_at(scene_entity.created_at)}",
            f"Pipeline: {scene_entity.pipeline}",
            f"Type: {scene_entity.reconstruction_type}",
        ]
    )
