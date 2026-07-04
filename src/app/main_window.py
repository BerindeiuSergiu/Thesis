from __future__ import annotations

import copy
import logging
import math
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFontMetrics
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.app.embedded_viewer import EmbeddedViewerWidget
from src.app.new_scene_dialog import NewSceneDialog
from src.app.scene_list_widget import SceneListWidget
from src.app.stage_progress_widget import STAGES, StageProgressWidget
from src.app.theme import build_stylesheet
from src.app.worker import PipelineWorker
from src.application.scene_repository import SceneRepository
from src.application.viewer_service import ViewerService
from src.models.scene import Scene
from src.models.scene_result import SceneResult
from src.pipeline.pipeline_factory import list_registered_pipelines


FORM_LABEL_WIDTH = 80
FORM_FIELD_WIDTH = 260
ACTION_PAIR_BUTTON_WIDTH = 127


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
        self.current_stage_key: str | None = None
        self.viewer_launches: dict[str, object] = {}
        self.pending_viewer_scene_id: str | None = None
        self.loaded_viewer_scene_id: str | None = None

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

    def _fast3r_model_profiles(self) -> dict:
        profiles = self._fast3r_config().get("model_profiles", {})
        return profiles if isinstance(profiles, dict) else {}

    def _fast3r_model_profile_items(self) -> list[tuple[str, str]]:
        profiles = self._fast3r_model_profiles()
        if not profiles:
            return [("default", "Default Fast3R")]
        items: list[tuple[str, str]] = []
        for key, value in profiles.items():
            label = value.get("label", key) if isinstance(value, dict) else key
            items.append((str(key), str(label)))
        return items

    def _default_model_profile(self) -> str:
        return str(self._fast3r_config().get("active_model_profile") or "default")

    def _current_model_profile(self) -> str:
        return str(self.model_profile_combo.currentData() or self._default_model_profile())

    def _set_model_profile_combo(self, profile_key: str) -> None:
        index = self.model_profile_combo.findData(profile_key)
        if index < 0:
            index = self.model_profile_combo.findData("default")
        if index >= 0:
            self.model_profile_combo.setCurrentIndex(index)

    def _missing_local_model_profile_path(self, profile_key: str) -> Path | None:
        profile = self._fast3r_model_profiles().get(profile_key, {})
        if not isinstance(profile, dict) or str(profile.get("source_type", "huggingface")).lower() != "local":
            return None
        model_source = str(profile.get("model_name") or "").strip()
        if not model_source:
            return None
        path = Path(model_source)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        path = path.resolve()
        return None if path.exists() else path

    def _model_profile_label(self, profile_key: str) -> str:
        profile = self._fast3r_model_profiles().get(profile_key, {})
        if isinstance(profile, dict):
            return str(profile.get("label") or profile_key)
        return profile_key

    def _init_widgets(self) -> None:
        fast3r = self._fast3r_config()

        self.video_label = QLabel("No video selected")
        self.video_label.setObjectName("PathLabel")
        self.video_label.setWordWrap(True)
        self.video_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.video_name_label = QLabel("No video selected")
        self.video_name_label.setObjectName("PathLabel")
        self.video_name_label.setWordWrap(False)
        self.video_name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.video_name_label.setFixedWidth(FORM_FIELD_WIDTH - 84)
        self.change_video_button = QPushButton("Change...")
        self.change_video_button.setObjectName("SecondaryButton")
        self.change_video_button.setFixedWidth(78)

        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItems(list_registered_pipelines())
        self.pipeline_combo.setFixedWidth(FORM_FIELD_WIDTH)
        default_pipeline = self.config.get("app", {}).get("default_pipeline", "default")
        default_index = self.pipeline_combo.findText(default_pipeline)
        if default_index >= 0:
            self.pipeline_combo.setCurrentIndex(default_index)
        self.model_profile_combo = QComboBox()
        self.model_profile_combo.setFixedWidth(FORM_FIELD_WIDTH)
        for key, label in self._fast3r_model_profile_items():
            self.model_profile_combo.addItem(label, key)
        self._set_model_profile_combo(self._default_model_profile())

        self.auto_preset_checkbox = QCheckBox("Auto preset")
        self.auto_preset_checkbox.setChecked(bool(fast3r.get("auto_preset_enabled", True)))
        self.auto_preset_checkbox.setObjectName("InspectorCheckbox")
        self.preset_combo = QComboBox()
        preset_names = sorted((fast3r.get("presets") or {}).keys())
        self.preset_combo.addItems(preset_names)
        self.preset_combo.setFixedWidth(FORM_FIELD_WIDTH)
        active_preset = str(fast3r.get("active_preset") or fast3r.get("preset_name") or "")
        active_index = self.preset_combo.findText(active_preset)
        if active_index >= 0:
            self.preset_combo.setCurrentIndex(active_index)

        self.new_scene_action = QAction("New Scene", self)
        self.new_scene_action.setToolTip("Create a named Scene that references a source video.")
        self.restore_layout_action = QAction("Restore Default Layout", self)
        self.restore_layout_action.setToolTip("Show the standard Scene Library and Reconstruction Properties docks.")
        self.calculate_params_action = QAction("Analyze Video", self)
        self.calculate_params_action.setToolTip("Inspect the selected Scene video and create a safe temporary preset.")
        self.run_pipeline_action = QAction("Run Reconstruction", self)
        self.run_pipeline_action.setToolTip("Start the Fast3R reconstruction pipeline.")
        self.view_scene_action = QAction("Open Output", self)
        self.view_scene_action.setToolTip("Open the selected or latest scene in the configured viewer.")
        self.analyze_video_button = QPushButton("Analyze")
        self.analyze_video_button.setObjectName("SecondaryButton")
        self.analyze_video_button.setFixedWidth(ACTION_PAIR_BUTTON_WIDTH)
        self.run_reconstruction_button = QPushButton("Run")
        self.run_reconstruction_button.setObjectName("PrimaryButton")
        self.run_reconstruction_button.setFixedWidth(ACTION_PAIR_BUTTON_WIDTH)
        self.view_scene_button = QPushButton("Open Output")
        self.view_scene_button.setObjectName("SecondaryButton")
        self.view_scene_button.setFixedWidth(FORM_FIELD_WIDTH)
        self.next_step_label = QLabel("No video selected. Create a scene first.")
        self.next_step_label.setObjectName("StatusStrip")
        self.next_step_label.setWordWrap(True)
        self.next_step_label.setFixedWidth(FORM_FIELD_WIDTH)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(260)
        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("StatusLabel")
        self.scene_list_widget = SceneListWidget()
        self.scene_list_widget.selection_changed.connect(self.on_scene_selection_changed)
        self.stage_progress_widget = StageProgressWidget()

        self.preview_title = QLabel("Viewport")
        self.preview_title.setObjectName("PreviewTitle")
        self.embedded_viewer = EmbeddedViewerWidget()
        self.embedded_viewer.load_succeeded.connect(self._on_viewer_loaded)
        self.embedded_viewer.load_failed.connect(self._on_viewer_failed)
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
        self.restore_layout_action.triggered.connect(self.restore_default_layout)
        self.change_video_button.clicked.connect(self.new_scene)
        self.calculate_params_action.triggered.connect(self.calculate_parameters)
        self.run_pipeline_action.triggered.connect(self.run_pipeline)
        self.view_scene_action.triggered.connect(self.view_scene)
        self.analyze_video_button.clicked.connect(self.calculate_parameters)
        self.run_reconstruction_button.clicked.connect(self.run_pipeline)
        self.view_scene_button.clicked.connect(self.view_scene)
        self.auto_preset_checkbox.toggled.connect(self._update_preset_visibility)
        self.auto_preset_checkbox.toggled.connect(lambda *_: self._sync_action_state())
        self.model_profile_combo.currentIndexChanged.connect(lambda *_: self._sync_action_state())
        self.preset_combo.currentTextChanged.connect(lambda *_: self._sync_action_state())

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

    def _build_layout(self) -> None:
        self._build_workflow_menu()
        self._build_status_bar()
        self.setCentralWidget(self._build_workspace())
        self.scene_dock = self._build_scene_dock()
        self.properties_dock = self._build_properties_dock()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        self._build_view_menu()
        self._update_preset_visibility(self.auto_preset_checkbox.isChecked())

    def _build_workflow_menu(self) -> None:
        workflow_menu = self.menuBar().addMenu("&Workflow")
        workflow_menu.addAction(self.new_scene_action)
        workflow_menu.addSeparator()
        workflow_menu.addAction(self.calculate_params_action)
        workflow_menu.addAction(self.run_pipeline_action)
        workflow_menu.addAction(self.view_scene_action)

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar()
        status_bar.setObjectName("AppStatusBar")
        status_bar.addWidget(self.progress_label, stretch=1)
        status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(status_bar)

    def _build_view_menu(self) -> None:
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.scene_dock.toggleViewAction())
        view_menu.addAction(self.properties_dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self.restore_layout_action)

    def restore_default_layout(self) -> None:
        self.scene_dock.setFloating(False)
        self.properties_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        self.scene_dock.show()
        self.properties_dock.show()
        self.scene_dock.raise_()
        self.properties_dock.raise_()
        self.statusBar().showMessage("Default layout restored.")

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
        splitter.addWidget(self._build_stage_progress_panel())
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
        layout.addLayout(top)

        viewport = QFrame()
        viewport.setObjectName("ViewportCanvas")
        canvas_layout = QVBoxLayout(viewport)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.embedded_viewer, stretch=1)
        layout.addWidget(viewport, stretch=1)
        return panel

    def _build_stage_progress_panel(self) -> QWidget:
        return self.stage_progress_widget

    def _build_inspector(self) -> QWidget:
        inspector = QFrame()
        inspector.setObjectName("Inspector")
        inspector.setMinimumWidth(330)
        inspector.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        layout.addLayout(self._build_reconstruction_title_row())
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_pipeline_group())
        layout.addWidget(self._build_auto_configuration_group())
        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_actions_group())
        self.stage_detail_title.setVisible(False)
        self.stage_detail_body.setVisible(False)
        self.parameter_tabs = self._build_parameter_tabs()
        layout.addWidget(self.parameter_tabs, stretch=1)
        self.inspector_spacer = QWidget()
        self.inspector_spacer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.inspector_spacer, stretch=1)
        return inspector

    def _build_reconstruction_title_row(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(2)
        title = QLabel("Reconstruction")
        title.setObjectName("CompactPanelTitle")
        layout.addWidget(title)
        return layout

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Input")
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = self._compact_group_layout(group)
        video_row = QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.setSpacing(6)
        video_row.addWidget(self.video_name_label)
        video_row.addWidget(self.change_video_button)
        layout.addLayout(self._form_row("Video", video_row))
        return group

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Pipeline")
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = self._compact_group_layout(group)
        layout.addLayout(self._form_row("Pipeline", self.pipeline_combo))
        layout.addLayout(self._form_row("Weights", self.model_profile_combo))
        return group

    def _build_auto_configuration_group(self) -> QGroupBox:
        group = QGroupBox("Auto Configuration")
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = self._compact_group_layout(group)
        layout.addLayout(self._form_row("", self.auto_preset_checkbox))
        layout.addLayout(self._form_row("Preset", self.preset_combo))
        return group

    def _build_status_group(self) -> QWidget:
        container = QWidget()
        container.setObjectName("StatusRow")
        container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._form_label("Status"))
        layout.addWidget(self.next_step_label)
        layout.addStretch(1)
        return container

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("Actions")
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = self._compact_group_layout(group)
        layout.setSpacing(4)
        first_row = QHBoxLayout()
        first_row.setContentsMargins(0, 0, 0, 0)
        first_row.setSpacing(6)
        first_row.addWidget(self.analyze_video_button)
        first_row.addWidget(self.run_reconstruction_button)
        layout.addLayout(self._form_row("", first_row))
        layout.addLayout(self._form_row("", self.view_scene_button))
        return group

    def _compact_group_layout(self, group: QGroupBox) -> QVBoxLayout:
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(5)
        return layout

    def _form_row(self, label: str, field: QWidget | QHBoxLayout) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self._form_label(label))
        if isinstance(field, QHBoxLayout):
            row.addLayout(field)
        else:
            row.addWidget(field)
        row.addStretch(1)
        return row

    def _form_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("FormLabel")
        label.setFixedWidth(FORM_LABEL_WIDTH)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def _build_parameter_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
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
            manual_mode = not bool(enabled)
            self.parameter_tabs.setVisible(manual_mode)
            if hasattr(self, "inspector_spacer"):
                self.inspector_spacer.setVisible(not manual_mode)

    def _apply_style(self) -> None:
        self.setStyleSheet(build_stylesheet())

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
        self.stage_progress_widget.set_stage_status(stage_key, status)

    def _reset_stage_statuses(self) -> None:
        self.stage_progress_widget.reset()
        self.current_stage_key = None

    def _activate_stage(self, stage_key: str) -> None:
        order = [key for key, _ in STAGES]
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
            else:
                self.stage_progress_widget.set_current_operation(None, "Failed. See error details and log.")
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
            for key, _ in STAGES:
                self._set_stage_status(key, "done")

    def _set_stage_detail(self, stage_key: str, detail: str) -> None:
        self.stage_progress_widget.set_current_operation(stage_key, detail)

    def _sync_action_state(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        scene_record = self.scene_list_widget.selected_scene()
        scene = Scene.from_record(scene_record) if scene_record is not None else None
        has_video = scene is not None and bool(str(scene.source_video))
        has_output = scene is not None and scene.status == "Ready" and self._scene_has_viewable_output(scene)
        self.new_scene_action.setEnabled(not running)
        self.change_video_button.setEnabled(not running)
        self.model_profile_combo.setEnabled(not running)
        self.calculate_params_action.setEnabled(has_video and not running)
        self.analyze_video_button.setEnabled(has_video and not running)
        self.run_pipeline_action.setEnabled(has_video and not running)
        self.run_reconstruction_button.setEnabled(has_video and not running)
        self.view_scene_action.setEnabled(not running)
        self.view_scene_button.setEnabled(not running)
        self._update_reconstruction_panel_state(scene, running=running, has_output=has_output)

    def _update_reconstruction_panel_state(self, scene: Scene | None, running: bool, has_output: bool) -> None:
        if scene is None or not str(scene.source_video):
            self.video_name_label.setText("No video selected")
            self.video_name_label.setToolTip("")
            self.video_label.setText("No video selected")
            self.video_label.setToolTip("")
            self._set_reconstruction_status("No video selected. Create a scene first.")
            return

        self.video_name_label.setText(self._elided_text(scene.source_video.name, self.video_name_label.width()))
        self.video_name_label.setToolTip(str(scene.source_video))
        self.video_label.setText(str(scene.source_video))
        self.video_label.setToolTip(str(scene.source_video))
        if running:
            self._set_reconstruction_status("Reconstruction running. Monitor progress below.")
        elif has_output:
            self._set_reconstruction_status("Reconstruction complete. Open output.")
        elif self.preset_combo.currentText() == "calculated_for_video":
            self._set_reconstruction_status("Analysis complete. Ready to run reconstruction.")
        else:
            self._set_reconstruction_status("Video selected. Analyze video or run reconstruction.")

    def _set_reconstruction_status(self, text: str) -> None:
        self.next_step_label.setText(text)

    def _elided_text(self, text: str, width: int) -> str:
        metrics = QFontMetrics(self.video_name_label.font())
        return metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(80, width - 10))

    def new_scene(self) -> None:
        fast3r = self._fast3r_config()
        presets = sorted((fast3r.get("presets") or {}).keys())
        dialog = NewSceneDialog(
            pipelines=list_registered_pipelines(),
            presets=presets,
            model_profiles=self._fast3r_model_profile_items(),
            default_pipeline=self.pipeline_combo.currentText(),
            default_preset=self.preset_combo.currentText(),
            default_model_profile=self._current_model_profile(),
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
        self._set_model_profile_combo(scene.model_profile)
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
        self.embedded_viewer.show_message(
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
        self._sync_action_state()
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
        model_profile = self._current_model_profile()
        missing_model_path = self._missing_local_model_profile_path(model_profile)
        if missing_model_path is not None:
            QMessageBox.warning(
                self,
                "Weights Profile Missing",
                "The selected weights profile is configured as a local model, but the model folder does not exist yet.\n\n"
                f"Profile: {self._model_profile_label(model_profile)}\n"
                f"Expected folder: {missing_model_path}\n\n"
                "Export or copy the fine-tuned Fast3R weights there, or switch Weights back to Default Fast3R.",
            )
            return
        running_scene = Scene.from_record(scene.to_record())
        running_scene.status = "Running"
        running_scene.pipeline = pipeline_name
        running_scene.model_profile = model_profile
        running_scene.reconstruction_preset = self.preset_combo.currentText()
        records = self.scene_repository.update_scene(running_scene)
        self.scene_list_widget.set_scenes(records)
        self.scene_list_widget.select_scene(scene.scene_id)
        self.progress_bar.setValue(0)
        self.stage_progress_widget.set_progress(0)
        self.progress_label.setText("Starting pipeline...")
        self.statusBar().showMessage("Starting pipeline...")
        self.preview_title.setText(f"Reconstructing: {scene.name}")
        self.loaded_viewer_scene_id = None
        self.embedded_viewer.show_message(
            "The selected Scene is being reconstructed.\n\n"
            f"Video: {scene.source_video.name}\n"
            f"Pipeline: {pipeline_name}\n"
            f"Weights: {self.model_profile_combo.currentText()}\n"
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
        fast3r_config["active_model_profile"] = self._current_model_profile()
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
        bounded_percent = max(0, min(100, percent))
        self.progress_bar.setValue(bounded_percent)
        self.stage_progress_widget.set_progress(bounded_percent)
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
        self.stage_progress_widget.set_progress(100)
        self.progress_label.setText(f"Finished: {scene.name}")
        self.statusBar().showMessage(f"Finished: {scene.name}")
        for key, _ in STAGES:
            self._set_stage_status(key, "done")
        self.stage_progress_widget.set_current_operation("outputs", "Reconstruction complete. Output is ready to inspect.")
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
        self.stage_progress_widget.set_current_operation(self.current_stage_key, "Failed. See error details and log.")
        self._update_stage_from_progress("failed", failed=True)
        self.preview_title.setText("Run Failed")
        self.loaded_viewer_scene_id = None
        self.embedded_viewer.show_message(
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
        self._set_model_profile_combo(scene.model_profile)
        if scene.reconstruction_preset:
            self.preset_combo.setCurrentText(scene.reconstruction_preset)
        self._show_scene_details(scene)
        self._sync_action_state()

    def _show_scene_details(self, scene: Scene) -> None:
        self.preview_title.setText(scene.name)
        if self.loaded_viewer_scene_id != scene.scene_id:
            self.embedded_viewer.show_message("No reconstruction viewer loaded. Select a scene and click Open Output.")

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
        if scene_record is None and self.selected_scene_id:
            scene_record = next(
                (record for record in self.scene_repository.list_records() if record.get("scene_id") == self.selected_scene_id),
                None,
            )
        if scene_record is None:
            self.progress_label.setText("No output selected")
            self.statusBar().showMessage("No output selected")
            QMessageBox.information(
                self,
                "No Scene Selected",
                "Select a Scene first. To construct outputs, choose a Scene and press Run Reconstruction.",
            )
            return
        scene = Scene.from_record(scene_record)
        if scene.status != "Ready":
            self.progress_label.setText("No output selected")
            self.statusBar().showMessage("No output selected")
            QMessageBox.information(
                self,
                "Output Not Constructed",
                "This Scene does not have constructed outputs yet.\n\n"
                "Please construct the outputs by pressing Run Reconstruction.",
            )
            return
        if not self._scene_has_viewable_output(scene):
            self.progress_label.setText("Output missing")
            self.statusBar().showMessage("Output missing")
            self.embedded_viewer.show_message(
                "Output files are missing.\n\nPlease construct the outputs by pressing Run Reconstruction."
            )
            QMessageBox.warning(
                self,
                "Output Missing",
                "The selected Scene output files could not be found.\n\n"
                "Please construct the outputs by pressing Run Reconstruction.",
            )
            return
        if not self.embedded_viewer.is_available():
            self.progress_label.setText("Viewer failed to load")
            self.statusBar().showMessage("Viewer failed to load")
            self.embedded_viewer.show_message(
                "Embedded viewer unavailable.\n\nInstall the required dependency:\npip install PyQt6-WebEngine"
            )
            QMessageBox.warning(
                self,
                "Embedded Viewer Unavailable",
                "Install the required dependency:\n\npip install PyQt6-WebEngine",
            )
            return

        scene_result = SceneResult.from_record(scene_record)
        try:
            self.logger.info(
                "Opening viewer for scene_id=%s name=%s output_dir=%s pointcloud=%s mesh=%s",
                scene.scene_id,
                scene.name,
                scene.output_dir,
                scene.pointcloud_path,
                scene.mesh_path,
            )
            self.progress_label.setText("Starting viewer...")
            self.statusBar().showMessage("Starting viewer...")
            self.preview_title.setText(scene.name)
            self.embedded_viewer.show_message(f"Loading output for:\n\n{scene.name}")
            self._stop_other_viewers(scene.scene_id)
            launch = self._viewer_launch_for_scene(scene.scene_id, scene_result)
            url = str(getattr(launch, "url", ""))
            if not url:
                raise RuntimeError("Viewer URL unavailable.")
            self.pending_viewer_scene_id = scene.scene_id
            if not self.embedded_viewer.load_url(url):
                return
            self.statusBar().showMessage(f"Loading viewer: {url}")
        except Exception as exc:
            self.logger.exception("Failed to start embedded viewer")
            self.progress_label.setText("Viewer failed to load")
            self.statusBar().showMessage("Viewer failed to load")
            self.embedded_viewer.show_message(f"Viewer failed to load.\n\n{exc}")
            QMessageBox.critical(self, "Viewer Failed", str(exc))

    def _viewer_launch_for_scene(self, scene_id: str, scene_result: SceneResult) -> object:
        existing = self.viewer_launches.get(scene_id)
        process = getattr(existing, "process", None)
        if existing is not None and process is not None and process.poll() is None:
            return existing
        launch = self.viewer_service.open_scene(scene_result)
        self.viewer_launches[scene_id] = launch
        return launch

    def _stop_other_viewers(self, scene_id: str) -> None:
        for cached_scene_id, launch in list(self.viewer_launches.items()):
            if cached_scene_id == scene_id:
                continue
            process = getattr(launch, "process", None)
            if process is not None and process.poll() is None:
                process.terminate()
            self.viewer_launches.pop(cached_scene_id, None)

    def _scene_has_viewable_output(self, scene: Scene) -> bool:
        if scene.output_dir is None or not scene.output_dir.exists():
            return False
        candidates: list[Path] = []
        if scene.pointcloud_path is not None:
            candidates.append(scene.pointcloud_path)
        if scene.mesh_path is not None:
            candidates.append(scene.mesh_path)
        metadata = scene.metadata if isinstance(scene.metadata, dict) else {}
        gaussian_report = metadata.get("gaussian_splat", {}) if isinstance(metadata.get("gaussian_splat", {}), dict) else {}
        gaussian_pointcloud = gaussian_report.get("pointcloud_ply")
        if gaussian_pointcloud:
            candidates.append(Path(gaussian_pointcloud))
        candidates.append(scene.output_dir / "gaussian_splat" / "scaled_points_for_gaussian.ply")
        return any(path.exists() for path in candidates)

    def _on_viewer_loaded(self, url: str) -> None:
        self.loaded_viewer_scene_id = self.pending_viewer_scene_id
        self.pending_viewer_scene_id = None
        self.progress_label.setText("Viewer loaded")
        self.statusBar().showMessage(f"Viewer loaded: {url}")

    def _on_viewer_failed(self, detail: str) -> None:
        self.loaded_viewer_scene_id = None
        self.pending_viewer_scene_id = None
        self.progress_label.setText("Viewer failed to load")
        self.statusBar().showMessage("Viewer failed to load")
        self.logger.error("Embedded viewer failed to load: %s", detail)

    def _load_scene_index(self) -> None:
        records = self._read_scene_index()
        self.scene_list_widget.set_scenes(records)

    def _read_scene_index(self) -> list[dict]:
        return self.scene_repository.list_records()

    def _write_scene_index(self, records: list[dict]) -> None:
        self.scene_repository.replace_records(records)
