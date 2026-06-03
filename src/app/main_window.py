from __future__ import annotations

import copy
import logging
import math
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QHeaderView,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.app.new_scene_dialog import NewSceneDialog
from src.app.scene_list_widget import SceneListWidget
from src.app.worker import PipelineWorker
from src.application.scene_repository import SceneRepository
from src.application.viewer_service import ViewerService
from src.models.scene import Scene
from src.models.scene_result import SceneResult
from src.pipeline.pipeline_factory import list_registered_pipelines


class MainWindow(QMainWindow):
    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_video: Path | None = None
        self.selected_scene_id: str | None = None
        self.running_scene_id: str | None = None
        self.session_presets: dict[str, dict] = {}
        self.worker: PipelineWorker | None = None
        self.stage_rows: dict[str, int] = {}
        self.current_stage_key: str | None = None

        src_root = Path(__file__).resolve().parents[1]
        index_path = Path(config.get("app", {}).get("scenes_index_path", "data/scenes_index.json"))
        scene_index_path = index_path if index_path.is_absolute() else src_root / index_path
        self.scene_repository = SceneRepository(scene_index_path, logger=self.logger)
        self.viewer_service = ViewerService(config)

        self.setWindowTitle(config.get("app", {}).get("name", "Fast3R Desktop Wrapper"))
        self.resize(1180, 760)

        self._init_widgets()
        self._build_layout()
        self._apply_style()
        self._load_scene_index()
        self._sync_action_state()

    def _fast3r_config(self) -> dict:
        return dict(self.config.get("pipeline", {}).get("fast3r", {}))

    def _init_widgets(self) -> None:
        fast3r = self._fast3r_config()

        self.video_label = QLabel("No video selected")
        self.video_label.setObjectName("PathLabel")
        self.video_label.setWordWrap(True)
        self.video_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItems(list_registered_pipelines())
        default_pipeline = self.config.get("app", {}).get("default_pipeline", "default")
        default_index = self.pipeline_combo.findText(default_pipeline)
        if default_index >= 0:
            self.pipeline_combo.setCurrentIndex(default_index)

        self.auto_preset_checkbox = QCheckBox("Auto preset")
        self.auto_preset_checkbox.setChecked(bool(fast3r.get("auto_preset_enabled", True)))
        self.preset_combo = QComboBox()
        preset_names = sorted((fast3r.get("presets") or {}).keys())
        self.preset_combo.addItems(preset_names)
        active_preset = str(fast3r.get("active_preset") or fast3r.get("preset_name") or "")
        active_index = self.preset_combo.findText(active_preset)
        if active_index >= 0:
            self.preset_combo.setCurrentIndex(active_index)

        self.new_scene_action = QAction("New Scene", self)
        self.new_scene_action.setToolTip("Create a named Scene that references a source video.")
        self.calculate_params_action = QAction("Analyze Video", self)
        self.calculate_params_action.setToolTip("Inspect the selected Scene video and create a safe temporary preset.")
        self.run_pipeline_action = QAction("Run Reconstruction", self)
        self.run_pipeline_action.setToolTip("Start the Fast3R reconstruction pipeline.")
        self.view_scene_action = QAction("Open Output", self)
        self.view_scene_action.setToolTip("Open the selected or latest scene in the configured viewer.")
        self.view_scene_button = QPushButton("Open Output")
        self.view_scene_button.setObjectName("SecondaryButton")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(260)
        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("StatusLabel")
        self.scene_list_widget = SceneListWidget()
        self.scene_list_widget.selection_changed.connect(self.on_scene_selection_changed)

        self.preview_title = QLabel("Viewport")
        self.preview_title.setObjectName("PreviewTitle")
        self.preview_body = QLabel(
            "No Scene selected.\n\nUse New Scene to define a scene, choose its source video, and configure reconstruction."
        )
        self.preview_body.setObjectName("PreviewBody")
        self.preview_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_body.setWordWrap(True)
        self.stage_detail_title = QLabel("Pipeline Details")
        self.stage_detail_title.setObjectName("SectionTitle")
        self.stage_detail_body = QLabel("Click a stage to inspect what it does and which settings affect it.")
        self.stage_detail_body.setObjectName("MutedText")
        self.stage_detail_body.setWordWrap(True)

        self.prefilter_enabled_checkbox = QCheckBox("Prefilter")
        self.prefilter_enabled_checkbox.setChecked(bool(fast3r.get("prefilter_enabled", True)))
        self.target_frames_spin = self._spin(1, 300, int(fast3r.get("target_frames", 80)))
        self.scan_stride_spin = self._spin(1, 240, int(fast3r.get("scan_stride", 15)))
        self.min_frame_gap_spin = self._spin(0, 1000, int(fast3r.get("min_frame_gap", 120)))
        self.visual_shortlist_spin = self._spin(1, 500, int(fast3r.get("visual_shortlist_target", 120)))
        self.probe_max_frames_spin = self._spin(1, 300, int(fast3r.get("probe_max_frames", 96)))
        self.selection_mode_combo = QComboBox()
        self.selection_mode_combo.addItems(["geometry_aware", "prefilter"])
        self.selection_mode_combo.setCurrentText(str(fast3r.get("selection_mode", "geometry_aware")))

        self.fast3r_image_size_spin = self._spin(128, 1024, int(fast3r.get("fast3r_image_size", 512)), step=64)
        self.min_confidence_spin = self._double_spin(0.0, 50.0, float(fast3r.get("min_confidence_threshold", 1.0)), 0.1, 2)
        self.confidence_keep_ratio_spin = self._double_spin(0.05, 1.0, float(fast3r.get("confidence_keep_ratio", 1.0)), 0.05, 2)
        self.view_conf_p50_spin = self._double_spin(0.0, 10.0, float(fast3r.get("view_conf_p50_min", 0.0)), 0.05, 2)
        self.view_conf_p90_spin = self._double_spin(0.0, 10.0, float(fast3r.get("view_conf_p90_min", 0.0)), 0.05, 2)
        self.niter_pnp_spin = self._spin(1, 1000, int(fast3r.get("niter_pnp", 100)), step=10)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["float32", "float16", "bfloat16"])
        self.dtype_combo.setCurrentText(str(fast3r.get("dtype", "float32")))

        self.scale_enabled_checkbox = QCheckBox("Scale")
        self.scale_enabled_checkbox.setChecked(bool(fast3r.get("scale_enabled", True)))
        self.scale_reference_real_spin = self._double_spin(0.01, 20.0, float(fast3r.get("scale_reference_real", 2.5)), 0.1, 3)
        self.scale_reference_measured_spin = self._double_spin(
            0.0,
            10000.0,
            float(fast3r.get("scale_reference_measured") or 0.0),
            0.01,
            4,
        )
        self.scale_reference_measured_spin.setSpecialValueText("Auto")
        self.scale_axis_combo = QComboBox()
        self.scale_axis_combo.addItems(["z", "y", "x"])
        self.scale_axis_combo.setCurrentText(str(fast3r.get("scale_reference_axis", "z")))
        self.fail_without_scale_checkbox = QCheckBox("Require Scale")
        self.fail_without_scale_checkbox.setChecked(bool(fast3r.get("fail_without_scale", False)))

        self.radius_percentile_spin = self._double_spin(50.0, 100.0, float(fast3r.get("radius_percentile", 99.0)), 0.25, 2)
        self.voxel_size_spin = self._double_spin(0.0, 0.1, float(fast3r.get("voxel_size", 0.002)), 0.0005, 5)
        self.high_detail_mode_checkbox = QCheckBox("High Detail")
        self.high_detail_mode_checkbox.setChecked(bool(fast3r.get("high_detail_mode", False)))
        self.gaussian_enabled_checkbox = QCheckBox("Gaussian Splat")
        self.gaussian_enabled_checkbox.setChecked(bool(fast3r.get("gaussian_enabled", True)))
        self.gaussian_scale_mult_spin = self._double_spin(0.05, 2.0, float(fast3r.get("gaussian_nn_scale_mult", 0.7)), 0.05, 2)
        self.gaussian_max_scale_spin = self._double_spin(0.0005, 0.1, float(fast3r.get("gaussian_max_scale", 0.02)), 0.0005, 5)
        self.mesh_enabled_checkbox = QCheckBox("Poisson Mesh")
        self.mesh_enabled_checkbox.setChecked(bool(fast3r.get("mesh_enabled", True)))
        self.poisson_depth_spin = self._spin(6, 13, int(fast3r.get("poisson_depth", 10)))
        self.target_triangles_spin = self._spin(0, 5_000_000, int(fast3r.get("target_triangles", 250000)), step=10000)
        self.laplacian_iterations_spin = self._spin(0, 20, int(fast3r.get("laplacian_iterations", 2)))
        self.laplacian_lambda_spin = self._double_spin(0.0, 1.0, float(fast3r.get("laplacian_lambda", 0.25)), 0.05, 3)

        self.new_scene_action.triggered.connect(self.new_scene)
        self.calculate_params_action.triggered.connect(self.calculate_parameters)
        self.run_pipeline_action.triggered.connect(self.run_pipeline)
        self.view_scene_action.triggered.connect(self.view_scene)
        self.view_scene_button.clicked.connect(self.view_scene)
        self.auto_preset_checkbox.toggled.connect(self._update_preset_visibility)
        self._init_stage_model()

    def _spin(self, minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _init_stage_model(self) -> None:
        self.stage_model = QStandardItemModel(0, 4, self)
        self.stage_model.setHorizontalHeaderLabels(["Stage", "Status", "Progress", "Detail"])
        stages = [
            ("frames", "Frames", "Pending", "Waiting for input selection."),
            ("depth", "Fast3R", "Pending", "Model inference not started."),
            ("reconstruct", "Geometry", "Pending", "Filtering and scale normalization pending."),
            ("outputs", "Outputs", "Pending", "Gaussian and mesh export pending."),
        ]
        self.stage_rows.clear()
        for row_index, (key, stage, status, detail) in enumerate(stages):
            row = [
                QStandardItem(stage),
                QStandardItem(status),
                QStandardItem("0%"),
                QStandardItem(detail),
            ]
            for item in row:
                item.setEditable(False)
            self.stage_model.appendRow(row)
            self.stage_rows[key] = row_index

    def _build_layout(self) -> None:
        self._build_toolbar()
        self._build_status_bar()
        self.setCentralWidget(self._build_workspace())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._build_scene_dock())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._build_properties_dock())
        self._update_preset_visibility(self.auto_preset_checkbox.isChecked())

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Reconstruction")
        toolbar.setObjectName("MainToolBar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toolbar.addAction(self.new_scene_action)
        toolbar.addAction(self.calculate_params_action)
        toolbar.addSeparator()
        toolbar.addAction(self.run_pipeline_action)
        toolbar.addAction(self.view_scene_action)
        self.addToolBar(toolbar)

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar()
        status_bar.setObjectName("AppStatusBar")
        status_bar.addWidget(self.progress_label, stretch=1)
        status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(status_bar)

    def _build_scene_dock(self) -> QDockWidget:
        dock = QDockWidget("Scene Library", self)
        dock.setObjectName("SceneLibraryDock")
        dock.setWidget(self.scene_list_widget)
        return dock

    def _build_properties_dock(self) -> QDockWidget:
        dock = QDockWidget("Reconstruction Properties", self)
        dock.setObjectName("PropertiesDock")
        dock.setWidget(self._build_inspector())
        return dock

    def _build_workspace(self) -> QWidget:
        workspace = QFrame()
        workspace.setObjectName("Workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("ViewportSplitter")
        splitter.addWidget(self._build_preview_panel())
        splitter.addWidget(self._build_stage_table())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([620, 160])
        layout.addWidget(splitter, stretch=1)
        return workspace

    def _build_preview_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ViewportPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.addWidget(self.preview_title)
        top.addStretch(1)
        top.addWidget(self.view_scene_button)
        layout.addLayout(top)

        viewport = QFrame()
        viewport.setObjectName("ViewportCanvas")
        canvas_layout = QVBoxLayout(viewport)
        canvas_layout.setContentsMargins(28, 28, 28, 28)
        canvas_layout.addStretch(1)
        canvas_layout.addWidget(self.preview_body)
        canvas_layout.addStretch(1)
        layout.addWidget(viewport, stretch=1)
        return panel

    def _build_stage_table(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("StageTablePanel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stage_table = QTableView()
        self.stage_table.setObjectName("StageTable")
        self.stage_table.setModel(self.stage_model)
        self.stage_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.stage_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.stage_table.verticalHeader().setVisible(False)
        self.stage_table.horizontalHeader().setStretchLastSection(True)
        self.stage_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.stage_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.stage_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.stage_table)
        return frame

    def _build_inspector(self) -> QWidget:
        inspector = QFrame()
        inspector.setObjectName("Inspector")
        inspector.setMinimumWidth(330)
        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        layout.addWidget(self._build_input_group())
        self.stage_detail_title.setVisible(False)
        self.stage_detail_body.setVisible(False)
        self.parameter_tabs = self._build_parameter_tabs()
        layout.addWidget(self.parameter_tabs, stretch=1)
        return inspector

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Scene Setup")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        layout.addWidget(QLabel("Video"), 0, 0)
        layout.addWidget(self.video_label, 0, 1, 1, 2)
        layout.addWidget(QLabel("Pipeline"), 1, 0)
        layout.addWidget(self.pipeline_combo, 1, 1, 1, 2)
        layout.addWidget(self.auto_preset_checkbox, 2, 0)
        layout.addWidget(self.preset_combo, 2, 1, 1, 2)
        return group

    def _build_parameter_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_frames_tab(), "Frames")
        tabs.addTab(self._build_model_tab(), "Model")
        tabs.addTab(self._build_scale_tab(), "Scale")
        tabs.addTab(self._build_outputs_tab(), "Outputs")
        return tabs

    def _build_frames_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addRow("Mode", self.selection_mode_combo)
        layout.addRow("Target frames", self.target_frames_spin)
        layout.addRow("Scan stride", self.scan_stride_spin)
        layout.addRow("Minimum frame gap", self.min_frame_gap_spin)
        layout.addRow("Visual shortlist", self.visual_shortlist_spin)
        layout.addRow("Probe max frames", self.probe_max_frames_spin)
        layout.addRow("", self.prefilter_enabled_checkbox)
        return widget

    def _build_model_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addRow("Image size", self.fast3r_image_size_spin)
        layout.addRow("Confidence min", self.min_confidence_spin)
        layout.addRow("Confidence keep", self.confidence_keep_ratio_spin)
        layout.addRow("View conf p50 min", self.view_conf_p50_spin)
        layout.addRow("View conf p90 min", self.view_conf_p90_spin)
        layout.addRow("PnP iterations", self.niter_pnp_spin)
        layout.addRow("Dtype", self.dtype_combo)
        return widget

    def _build_scale_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addRow("", self.scale_enabled_checkbox)
        layout.addRow("Real dimension m", self.scale_reference_real_spin)
        layout.addRow("Measured units", self.scale_reference_measured_spin)
        layout.addRow("Reference axis", self.scale_axis_combo)
        layout.addRow("", self.fail_without_scale_checkbox)
        return widget

    def _build_outputs_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addRow("Radius percentile", self.radius_percentile_spin)
        layout.addRow("Voxel size", self.voxel_size_spin)
        layout.addRow("", self.high_detail_mode_checkbox)
        layout.addRow("", self.gaussian_enabled_checkbox)
        layout.addRow("Gaussian scale", self.gaussian_scale_mult_spin)
        layout.addRow("Gaussian max", self.gaussian_max_scale_spin)
        layout.addRow("", self.mesh_enabled_checkbox)
        layout.addRow("Poisson depth", self.poisson_depth_spin)
        layout.addRow("Target triangles", self.target_triangles_spin)
        layout.addRow("Smooth iterations", self.laplacian_iterations_spin)
        layout.addRow("Smooth lambda", self.laplacian_lambda_spin)
        return widget

    def _update_preset_visibility(self, enabled: bool) -> None:
        self.preset_combo.setEnabled(bool(enabled))
        if hasattr(self, "parameter_tabs"):
            self.parameter_tabs.setVisible(not bool(enabled))

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #0f1218;
                color: #d8dee9;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }
            QFrame#Sidebar {
                background: #11151d;
                border-right: 1px solid #262c38;
            }
            QFrame#Workspace {
                background: #0f1218;
            }
            QFrame#Inspector {
                background: #151a23;
                border-left: 1px solid #262c38;
            }
            QSplitter::handle {
                background: #262c38;
                width: 1px;
            }
            QLabel#Header {
                font-size: 24px;
                font-weight: 700;
                color: #f4f7fb;
            }
            QLabel#AppTitle {
                font-size: 22px;
                font-weight: 800;
                color: #f4f7fb;
            }
            QLabel#SectionTitle, QLabel#SidebarTitle {
                font-size: 16px;
                font-weight: 700;
                color: #f4f7fb;
            }
            QLabel#PreviewTitle {
                font-size: 17px;
                font-weight: 700;
                color: #f4f7fb;
            }
            QLabel#PreviewBody {
                color: #8f9bad;
                font-size: 15px;
            }
            QLabel#MutedText {
                color: #8f9bad;
            }
            QLabel#PathLabel {
                padding: 8px;
                background: #0f1218;
                border: 1px solid #303848;
                border-radius: 6px;
                color: #aeb7c6;
            }
            QLabel#StatusLabel {
                color: #aeb7c6;
                font-weight: 600;
            }
            QGroupBox {
                background: #181e28;
                border: 1px solid #2c3444;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 14px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #cdd5e3;
            }
            QTabWidget::pane {
                background: #181e28;
                border: 1px solid #2c3444;
                border-radius: 8px;
                top: -1px;
            }
            QTabBar::tab {
                background: #121721;
                color: #8f9bad;
                border: 1px solid #2c3444;
                border-bottom: none;
                padding: 8px 14px;
                margin-right: 3px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #181e28;
                color: #f4f7fb;
            }
            QPushButton {
                border: none;
                border-radius: 6px;
                padding: 9px 14px;
                font-weight: 700;
            }
            QPushButton#PrimaryButton {
                background: #3b82f6;
                color: white;
            }
            QPushButton#PrimaryButton:disabled {
                background: #334155;
                color: #94a3b8;
            }
            QPushButton#SecondaryButton {
                background: #252d3a;
                color: #d8dee9;
            }
            QPushButton#SecondaryButton:hover {
                background: #30394a;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #0f1218;
                border: 1px solid #303848;
                border-radius: 6px;
                padding: 6px;
                min-height: 24px;
                color: #d8dee9;
            }
            QCheckBox {
                spacing: 8px;
                font-weight: 600;
                color: #d8dee9;
            }
            QProgressBar {
                background: #252d3a;
                border: none;
                border-radius: 5px;
                height: 10px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 5px;
            }
            QToolBar#MainToolBar {
                background: #11151d;
                border-bottom: 1px solid #262c38;
                spacing: 6px;
                padding: 4px;
            }
            QStatusBar#AppStatusBar {
                background: #11151d;
                border-top: 1px solid #262c38;
            }
            QDockWidget {
                titlebar-close-icon: none;
                titlebar-normal-icon: none;
            }
            QDockWidget::title {
                background: #151a23;
                padding: 7px;
                border-bottom: 1px solid #262c38;
                font-weight: 700;
            }
            QTreeView#SceneTree, QTableView#StageTable {
                background: #0f1218;
                border: 1px solid #262c38;
                color: #aeb7c6;
                gridline-color: #262c38;
                selection-background-color: #1d4ed8;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background: #151a23;
                color: #cdd5e3;
                border: 1px solid #262c38;
                padding: 5px;
                font-weight: 700;
            }
            QFrame#ViewportPanel, QFrame#StageTablePanel {
                background: #151a23;
                border: 1px solid #262c38;
                border-radius: 8px;
            }
            QFrame#ViewportCanvas {
                background: #0b0e13;
                border: 1px solid #293142;
                border-radius: 8px;
            }
            """
        )

    def show_stage_details(self, stage_key: str) -> None:
        details = {
            "frames": (
                "Frame Extraction",
                "Samples the input video, scores candidate frames, and keeps views that give broad room coverage.",
            ),
            "depth": (
                "Depth Estimation",
                "Runs Fast3R on the selected frames to estimate dense point maps, confidence, and camera poses.",
            ),
            "reconstruct": (
                "Reconstruction",
                "Filters invalid points, normalizes scale, removes extreme outliers, and prepares the shared point cloud.",
            ),
            "outputs": (
                "Outputs",
                "Exports the presentation Gaussian initialization and optional mesh. The Gaussian viewer remains web-based.",
            ),
        }
        title, body = details.get(stage_key, ("Pipeline Details", "Click a stage to inspect it."))
        self.stage_detail_title.setText(title)
        self.stage_detail_body.setText(body)

    def _set_stage_status(self, stage_key: str, status: str) -> None:
        row = self.stage_rows.get(stage_key)
        if row is None:
            return
        labels = {
            "pending": "Pending",
            "active": "Running",
            "done": "Complete",
            "error": "Needs attention",
        }
        progress = {
            "pending": "0%",
            "active": "In progress",
            "done": "100%",
            "error": "Stopped",
        }
        self.stage_model.item(row, 1).setText(labels.get(status, status.title()))
        self.stage_model.item(row, 2).setText(progress.get(status, ""))

    def _reset_stage_statuses(self) -> None:
        for key in self.stage_rows:
            self._set_stage_status(key, "pending")
            row = self.stage_rows[key]
            self.stage_model.item(row, 3).setText("Waiting.")
        self.current_stage_key = None

    def _activate_stage(self, stage_key: str) -> None:
        order = ["frames", "depth", "reconstruct", "outputs"]
        if stage_key not in order:
            return
        active_index = order.index(stage_key)
        for index, key in enumerate(order):
            if index < active_index:
                self._set_stage_status(key, "done")
            elif key == stage_key:
                self._set_stage_status(key, "active")
            else:
                self._set_stage_status(key, "pending")
        self.current_stage_key = stage_key

    def _update_stage_from_progress(self, message: str, failed: bool = False) -> None:
        if failed:
            if self.current_stage_key is not None:
                self._set_stage_status(self.current_stage_key, "error")
                self._set_stage_detail(self.current_stage_key, "Failed. See error details and log.")
            return
        text = message.lower()
        if "selecting input frames" in text:
            self._activate_stage("frames")
        elif "running fast3r" in text or "preparing frames for fast3r" in text or "fast3r reconstruction" in text:
            self._activate_stage("depth")
        elif (
            "filtering invalid" in text
            or "normalizing reconstruction" in text
            or "preparing scaled point cloud" in text
        ):
            self._activate_stage("reconstruct")
        elif "building gaussian" in text or "building poisson" in text or "splat output" in text:
            self._activate_stage("outputs")
        if "complete" in text or "finished" in text:
            for key in self.stage_rows:
                self._set_stage_status(key, "done")

    def _set_stage_detail(self, stage_key: str, detail: str) -> None:
        row = self.stage_rows.get(stage_key)
        if row is not None:
            self.stage_model.item(row, 3).setText(detail)

    def _sync_action_state(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        scene_record = self.scene_list_widget.selected_scene()
        scene = Scene.from_record(scene_record) if scene_record is not None else None
        has_scene = scene is not None
        has_video = scene is not None and bool(str(scene.source_video))
        has_output = scene is not None and scene.output_dir is not None and scene.status == "Ready"
        self.new_scene_action.setEnabled(not running)
        self.calculate_params_action.setEnabled(has_video and not running)
        self.run_pipeline_action.setEnabled(has_video and not running)
        self.view_scene_action.setEnabled(has_output and not running)
        self.view_scene_button.setEnabled(has_output and not running)

    def new_scene(self) -> None:
        fast3r = self._fast3r_config()
        presets = sorted((fast3r.get("presets") or {}).keys())
        dialog = NewSceneDialog(
            pipelines=list_registered_pipelines(),
            presets=presets,
            default_pipeline=self.pipeline_combo.currentText(),
            default_preset=self.preset_combo.currentText(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        scene = dialog.scene()
        records = self.scene_repository.create_scene(scene)
        self.scene_list_widget.set_scenes(records)
        self.scene_list_widget.select_scene(scene.scene_id)
        self.selected_scene_id = scene.scene_id
        self.selected_video = scene.source_video
        self.video_label.setText(str(scene.source_video))
        self.pipeline_combo.setCurrentText(scene.pipeline)
        if scene.reconstruction_preset:
            self.preset_combo.setCurrentText(scene.reconstruction_preset)
        self.progress_label.setText(f"Scene created: {scene.name}")
        self.statusBar().showMessage(f"Scene created: {scene.name}")
        self._show_scene_details(scene)
        self._sync_action_state()

    def calculate_parameters(self) -> None:
        if self.selected_video is None:
            QMessageBox.information(self, "No Scene", "Please create or select a Scene before analyzing video.")
            return
        try:
            preset, report = self._calculate_video_preset(self.selected_video)
        except Exception as exc:
            self.logger.exception("Failed to calculate video parameters")
            QMessageBox.critical(self, "Parameter Calculation Failed", str(exc))
            return

        preset_name = "calculated_for_video"
        self.session_presets[preset_name] = preset
        if self.preset_combo.findText(preset_name) < 0:
            self.preset_combo.addItem(preset_name)
        self.preset_combo.setCurrentText(preset_name)
        self.auto_preset_checkbox.setChecked(True)
        self.progress_label.setText(
            "Calculated: "
            f"{preset['target_frames']} frames, stride {preset['scan_stride']}, "
            f"Fast3R {preset['fast3r_image_size']}"
        )
        self.statusBar().showMessage(self.progress_label.text())
        self.preview_title.setText("Run Estimate")
        self.preview_body.setText(
            "\n".join(
                [
                    f"Video: {self.selected_video.name}",
                    f"Resolution: {report['width']} x {report['height']}",
                    f"Duration: {report['duration_seconds']:.1f}s",
                    f"Total frames: {report['total_frames']}",
                    "",
                    f"Selected frames target: {preset['target_frames']}",
                    f"Scan stride: {preset['scan_stride']}",
                    f"Fast3R image size: {preset['fast3r_image_size']}",
                    f"Confidence min: {preset['min_confidence_threshold']:.2f}",
                    "",
                    "Output: Gaussian initialization; mesh disabled for faster inspection.",
                ]
            )
        )
        QMessageBox.information(
            self,
            "Calculated Parameters",
            (
                f"Temporary preset created for this app session.\n\n"
                f"Video: {report['width']}x{report['height']}, {report['duration_seconds']:.1f}s\n"
                f"Frames: {preset['target_frames']}\n"
                f"Scan stride: {preset['scan_stride']}\n"
                f"Fast3R image size: {preset['fast3r_image_size']}\n"
                f"Confidence min: {preset['min_confidence_threshold']:.2f}\n"
                f"View p50/p90 min: {preset['view_conf_p50_min']:.2f} / {preset['view_conf_p90_min']:.2f}"
            ),
        )

    def _calculate_video_preset(self, video_path: Path) -> tuple[dict, dict]:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = float(total_frames / fps) if fps > 0 else 0.0

        sample_count = min(16, max(total_frames, 1))
        sample_indices = np.linspace(0, max(total_frames - 1, 0), sample_count, dtype=np.int64)
        blur_values: list[float] = []
        clipped_values: list[float] = []
        brightness_values: list[float] = []
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            clipped = np.mean((gray <= 4) | (gray >= 251))
            clipped_values.append(float(clipped))
            brightness_values.append(float(np.mean(gray)))
        cap.release()

        median_blur = float(np.median(blur_values)) if blur_values else 0.0
        mean_clipped = float(np.mean(clipped_values)) if clipped_values else 0.0
        mean_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0

        max_dim = max(width, height)
        if max_dim >= 1280 and median_blur >= 90.0:
            fast3r_image_size = 768
            target_frames = 90 if duration_seconds < 90 else 120
        else:
            fast3r_image_size = 512
            if duration_seconds < 35:
                target_frames = 90
            elif duration_seconds < 110:
                target_frames = 150
            else:
                target_frames = 180

        if total_frames > 0:
            scan_stride = int(round(total_frames / max(target_frames * 4.0, 1.0)))
        else:
            scan_stride = 10
        scan_stride = max(5, min(scan_stride, 30))
        min_frame_gap = max(5, min(int(round(scan_stride * 0.8)), 20))
        max_frame_gap = max(120, int(scan_stride * 18))

        low_texture_or_soft = median_blur < 65.0
        hard_exposure = mean_clipped > 0.10 or mean_brightness < 45.0 or mean_brightness > 210.0
        min_confidence = 1.10 if low_texture_or_soft or hard_exposure else 1.05
        confidence_keep = 0.80 if hard_exposure else 0.85
        view_p50 = 1.25 if hard_exposure else 1.20
        view_p90 = 1.55 if hard_exposure else 1.45

        preset = {
            "target_frames": int(target_frames),
            "selection_mode": "geometry_aware",
            "visual_shortlist_target": int(min(300, max(target_frames + 40, math.ceil(target_frames * 1.25)))),
            "probe_max_frames": int(target_frames),
            "scan_stride": int(scan_stride),
            "min_frame_gap": int(min_frame_gap),
            "prefilter_enabled": True,
            "target_width": 1024,
            "target_height": 768,
            "analysis_width": 384,
            "analysis_height": 288,
            "probe_dtype": "bfloat16",
            "fast3r_image_size": int(fast3r_image_size),
            "dtype": "bfloat16",
            "min_confidence_threshold": float(min_confidence),
            "confidence_keep_ratio": float(confidence_keep),
            "view_conf_p50_min": float(view_p50),
            "view_conf_p90_min": float(view_p90),
            "weak_texture_retention": True,
            "weak_texture_percentile": 35.0,
            "weak_texture_min_conf_thr": 0.55,
            "weak_texture_keep_ratio": 0.35,
            "max_abs_coordinate": 100.0,
            "outlier_method": "radius_percentile",
            "radius_percentile": 99.7,
            "high_detail_mode": False,
            "voxel_size": 0.001,
            "gaussian_enabled": True,
            "gaussian_nn_scale_mult": 0.35,
            "gaussian_min_scale": 0.00025,
            "gaussian_max_scale": 0.008,
            "gaussian_scale_percentile_low": 5.0,
            "gaussian_scale_percentile_high": 85.0,
            "gaussian_density_opacity": True,
            "gaussian_min_alpha": 0.35,
            "gaussian_max_alpha": 0.82,
            "gaussian_surface_aligned": False,
            "mesh_enabled": False,
            "calculated_video_report": {
                "width": int(width),
                "height": int(height),
                "fps": float(fps),
                "total_frames": int(total_frames),
                "duration_seconds": float(duration_seconds),
                "median_laplacian_variance": float(median_blur),
                "mean_clipped_ratio": float(mean_clipped),
                "mean_brightness": float(mean_brightness),
            },
        }
        return preset, preset["calculated_video_report"]

    def run_pipeline(self) -> None:
        scene_record = self.scene_list_widget.selected_scene()
        if scene_record is None:
            QMessageBox.warning(self, "No Scene", "Please create or select a Scene before running reconstruction.")
            return
        scene = Scene.from_record(scene_record)
        if not str(scene.source_video):
            QMessageBox.warning(self, "No Source Video", "The selected Scene does not reference a source video.")
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "The pipeline is already running.")
            return

        self.selected_scene_id = scene.scene_id
        self.running_scene_id = scene.scene_id
        self.selected_video = scene.source_video
        pipeline_name = scene.pipeline or self.pipeline_combo.currentText()
        running_scene = Scene.from_record(scene.to_record())
        running_scene.status = "Running"
        running_scene.pipeline = pipeline_name
        running_scene.reconstruction_preset = self.preset_combo.currentText()
        records = self.scene_repository.update_scene(running_scene)
        self.scene_list_widget.set_scenes(records)
        self.scene_list_widget.select_scene(scene.scene_id)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting pipeline...")
        self.statusBar().showMessage("Starting pipeline...")
        self.preview_title.setText(f"Reconstructing: {scene.name}")
        self.preview_body.setText(
            "The selected Scene is being reconstructed.\n\n"
            f"Video: {scene.source_video.name}\n"
            f"Pipeline: {pipeline_name}\n"
            f"Preset: {self.preset_combo.currentText() or '-'}\n\n"
            "Stage progress is shown below."
        )
        self._reset_stage_statuses()
        self._set_stage_status("frames", "active")
        self._set_stage_detail("frames", "Selecting and ranking input frames.")
        self._sync_action_state()

        run_config = self._config_with_ui_overrides()
        self.worker = PipelineWorker(pipeline_name, run_config, str(scene.source_video))
        self.worker.progress_changed.connect(self.on_progress_changed)
        self.worker.succeeded.connect(self.on_pipeline_succeeded)
        self.worker.failed.connect(self.on_pipeline_failed)
        self.worker.finished.connect(self._sync_action_state)
        self.worker.start()
        self._sync_action_state()

    def _config_with_ui_overrides(self) -> dict:
        run_config = copy.deepcopy(self.config)
        fast3r_config = run_config.setdefault("pipeline", {}).setdefault("fast3r", {})
        if bool(self.auto_preset_checkbox.isChecked()):
            preset_name = self.preset_combo.currentText()
            preset = self.session_presets.get(preset_name)
            if preset is None:
                preset = (fast3r_config.get("presets") or {}).get(preset_name, {})
            if isinstance(preset, dict):
                fast3r_config.update(preset)
                if preset_name in self.session_presets:
                    fast3r_config.setdefault("presets", {})[preset_name] = dict(preset)
            fast3r_config["auto_preset_enabled"] = True
            fast3r_config["active_preset"] = preset_name
            return run_config

        measured = float(self.scale_reference_measured_spin.value())
        fast3r_config.update(
            {
                "auto_preset_enabled": False,
                "prefilter_enabled": bool(self.prefilter_enabled_checkbox.isChecked()),
                "selection_mode": self.selection_mode_combo.currentText(),
                "target_frames": int(self.target_frames_spin.value()),
                "scan_stride": int(self.scan_stride_spin.value()),
                "min_frame_gap": int(self.min_frame_gap_spin.value()),
                "visual_shortlist_target": int(self.visual_shortlist_spin.value()),
                "probe_max_frames": int(self.probe_max_frames_spin.value()),
                "fast3r_image_size": int(self.fast3r_image_size_spin.value()),
                "min_confidence_threshold": float(self.min_confidence_spin.value()),
                "confidence_keep_ratio": float(self.confidence_keep_ratio_spin.value()),
                "view_conf_p50_min": float(self.view_conf_p50_spin.value()),
                "view_conf_p90_min": float(self.view_conf_p90_spin.value()),
                "niter_pnp": int(self.niter_pnp_spin.value()),
                "dtype": self.dtype_combo.currentText(),
                "scale_enabled": bool(self.scale_enabled_checkbox.isChecked()),
                "scale_reference_real": float(self.scale_reference_real_spin.value()),
                "scale_reference_measured": measured if measured > 0 else None,
                "scale_reference_axis": self.scale_axis_combo.currentText(),
                "fail_without_scale": bool(self.fail_without_scale_checkbox.isChecked()),
                "radius_percentile": float(self.radius_percentile_spin.value()),
                "voxel_size": float(self.voxel_size_spin.value()),
                "high_detail_mode": bool(self.high_detail_mode_checkbox.isChecked()),
                "gaussian_enabled": bool(self.gaussian_enabled_checkbox.isChecked()),
                "gaussian_nn_scale_mult": float(self.gaussian_scale_mult_spin.value()),
                "gaussian_max_scale": float(self.gaussian_max_scale_spin.value()),
                "mesh_enabled": bool(self.mesh_enabled_checkbox.isChecked()),
                "poisson_depth": int(self.poisson_depth_spin.value()),
                "target_triangles": int(self.target_triangles_spin.value()),
                "laplacian_iterations": int(self.laplacian_iterations_spin.value()),
                "laplacian_lambda": float(self.laplacian_lambda_spin.value()),
            }
        )
        return run_config

    def on_progress_changed(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
        self._update_stage_from_progress(message)
        if self.current_stage_key:
            self._set_stage_detail(self.current_stage_key, message)

    def on_pipeline_succeeded(self, result: object) -> None:
        scene_result = result if isinstance(result, SceneResult) else None
        if scene_result is None:
            QMessageBox.warning(self, "Unexpected Result", "Pipeline finished but returned an invalid result.")
            return

        scene_id = self.running_scene_id or self.selected_scene_id or scene_result.scene_id
        records = self.scene_repository.update_scene_with_result(scene_id, scene_result)
        self.scene_list_widget.set_scenes(records)
        self.scene_list_widget.select_scene(scene_id)
        scene_record = self.scene_list_widget.selected_scene()
        scene = Scene.from_record(scene_record) if scene_record is not None else Scene.from_record(scene_result.to_record())
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"Finished: {scene.name}")
        self.statusBar().showMessage(f"Finished: {scene.name}")
        for key in self.stage_rows:
            self._set_stage_status(key, "done")
        self._show_scene_details(scene)
        self.running_scene_id = None
        self._sync_action_state()
        QMessageBox.information(self, "Reconstruction Finished", f"Scene ready: {scene.name}")

    def on_pipeline_failed(self, error_details: str) -> None:
        self.logger.error("Pipeline failed:\n%s", error_details)
        if self.running_scene_id:
            scene_record = self.scene_list_widget.selected_scene()
            if scene_record is not None:
                scene = Scene.from_record(scene_record)
                scene.status = "Failed"
                records = self.scene_repository.update_scene(scene)
                self.scene_list_widget.set_scenes(records)
                self.scene_list_widget.select_scene(scene.scene_id)
        self.progress_label.setText("Pipeline failed")
        self.statusBar().showMessage("Pipeline failed")
        self._update_stage_from_progress("failed", failed=True)
        self.preview_title.setText("Run Failed")
        self.preview_body.setText(
            "The pipeline stopped before producing a complete scene.\n\n"
            "Suggested recovery: reduce selected frames, disable mesh/high-detail options, or inspect the log."
        )
        self.running_scene_id = None
        self._sync_action_state()
        QMessageBox.critical(self, "Pipeline Failed", error_details)

    def on_scene_selection_changed(self) -> None:
        scene_record = self.scene_list_widget.selected_scene()
        if scene_record is None:
            return
        scene = Scene.from_record(scene_record)
        self.selected_scene_id = scene.scene_id
        self.selected_video = scene.source_video
        self.video_label.setText(str(scene.source_video))
        self.pipeline_combo.setCurrentText(scene.pipeline)
        if scene.reconstruction_preset:
            self.preset_combo.setCurrentText(scene.reconstruction_preset)
        self._show_scene_details(scene)
        self._sync_action_state()

    def _show_scene_details(self, scene: Scene) -> None:
        metadata = scene.metadata if isinstance(scene.metadata, dict) else {}
        frames = (
            metadata.get("num_selected_frames")
            or metadata.get("selected_frames")
            or metadata.get("frames_selected")
            or metadata.get("frame_count")
            or "-"
        )
        points = metadata.get("num_points_final") or metadata.get("processed_scaled_points") or "-"
        tags = ", ".join(scene.tags) if scene.tags else "-"
        description = scene.description or "-"
        created = self._format_scene_date(scene.created_at)
        video_name = scene.source_video.name if str(scene.source_video) else "-"
        self.preview_title.setText(scene.name)
        self.preview_body.setText(
            "\n".join(
                [
                    f"Name: {scene.name}",
                    f"Video: {video_name}",
                    f"Frames: {frames}",
                    f"Points: {points}",
                    f"Created: {created}",
                    f"Pipeline: {scene.pipeline}",
                    f"Preset: {scene.reconstruction_preset or '-'}",
                    f"Status: {scene.status}",
                    f"Reconstruction: {scene.reconstruction_type}",
                    f"Tags: {tags}",
                    "",
                    f"Description: {description}",
                ]
            )
        )

    @staticmethod
    def _format_scene_date(value: str) -> str:
        if not value:
            return "-"
        try:
            return datetime.fromisoformat(value).strftime("%d-%m-%Y")
        except ValueError:
            return value[:10]

    def view_scene(self) -> None:
        scene_record = self.scene_list_widget.selected_scene()
        if scene_record is None:
            QMessageBox.information(self, "No Scene", "Please select a scene to view.")
            return
        scene = Scene.from_record(scene_record)
        if scene.status != "Ready" or scene.output_dir is None:
            QMessageBox.information(self, "Scene Not Ready", "Run reconstruction before opening this Scene output.")
            return

        scene_result = SceneResult.from_record(scene_record)
        self.viewer_service.open_scene(scene_result)

    def _load_scene_index(self) -> None:
        records = self._read_scene_index()
        self.scene_list_widget.set_scenes(records)

    def _read_scene_index(self) -> list[dict]:
        return self.scene_repository.list_records()

    def _write_scene_index(self, records: list[dict]) -> None:
        self.scene_repository.replace_records(records)
