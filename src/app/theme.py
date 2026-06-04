from __future__ import annotations


THEME = {
    "APP_BG": "#10141c",
    "PANEL_BG": "#151a23",
    "PANEL_BG_ALT": "#111722",
    "GROUP_BG": "#19212c",
    "FIELD_BG": "#1d2634",
    "FIELD_BG_HOVER": "#263247",
    "FIELD_BORDER": "#354055",
    "FIELD_BORDER_FOCUS": "#526b92",
    "TEXT_PRIMARY": "#f4f7fb",
    "TEXT_SECONDARY": "#d8dee9",
    "TEXT_MUTED": "#8f9bad",
    "PRIMARY": "#3b82f6",
    "PRIMARY_HOVER": "#4f8ff8",
    "PRIMARY_PRESSED": "#2563eb",
    "SECONDARY": "#202838",
    "SECONDARY_HOVER": "#2b3549",
    "SECONDARY_PRESSED": "#182033",
    "DISABLED_BG": "#171d28",
    "DISABLED_TEXT": "#687386",
    "WARNING": "#f59e0b",
    "ERROR": "#b45363",
    "SUCCESS": "#3f6f5a",
    "SEPARATOR": "#2c3444",
}


def build_stylesheet() -> str:
    t = THEME
    return f"""
    QWidget {{
        background: {t["APP_BG"]};
        color: {t["TEXT_SECONDARY"]};
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 13px;
    }}
    QFrame#Workspace {{
        background: {t["APP_BG"]};
    }}
    QFrame#Inspector {{
        background: {t["PANEL_BG"]};
        border-left: 1px solid {t["SEPARATOR"]};
    }}
    QSplitter::handle {{
        background: {t["SEPARATOR"]};
        width: 1px;
    }}
    QLabel#SectionTitle, QLabel#SidebarTitle {{
        font-size: 15px;
        font-weight: 700;
        color: {t["TEXT_PRIMARY"]};
    }}
    QLabel#PreviewTitle {{
        font-size: 17px;
        font-weight: 700;
        color: {t["TEXT_PRIMARY"]};
    }}
    QLabel#PreviewBody, QLabel#MutedText {{
        color: {t["TEXT_MUTED"]};
    }}
    QLabel#CompactPanelTitle {{
        font-size: 15px;
        font-weight: 800;
        color: {t["TEXT_PRIMARY"]};
    }}
    QLabel#CompactStatus {{
        color: {t["TEXT_MUTED"]};
        font-size: 12px;
    }}
    QLabel#FormLabel {{
        color: {t["TEXT_MUTED"]};
        background: transparent;
        font-weight: 600;
        padding: 0;
    }}
    QLabel#PathLabel {{
        padding: 4px 7px;
        background: {t["FIELD_BG"]};
        border: 1px solid {t["FIELD_BORDER"]};
        border-radius: 4px;
        color: {t["TEXT_SECONDARY"]};
    }}
    QLabel#StatusLabel {{
        color: {t["TEXT_MUTED"]};
        font-weight: 600;
    }}
    QLabel#StatusStrip {{
        color: {t["TEXT_SECONDARY"]};
        background: {t["FIELD_BG"]};
        border: 1px solid {t["FIELD_BORDER"]};
        border-radius: 4px;
        padding: 4px 7px;
    }}
    QGroupBox {{
        background: {t["GROUP_BG"]};
        border: 1px solid {t["SEPARATOR"]};
        border-radius: 5px;
        margin-top: 4px;
        padding-top: 6px;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {t["TEXT_SECONDARY"]};
    }}
    QTabWidget::pane {{
        background: {t["GROUP_BG"]};
        border: 1px solid {t["SEPARATOR"]};
        border-radius: 5px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: {t["PANEL_BG_ALT"]};
        color: {t["TEXT_MUTED"]};
        border: 1px solid {t["SEPARATOR"]};
        border-bottom: none;
        padding: 6px 12px;
        margin-right: 3px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }}
    QTabBar::tab:selected {{
        background: {t["GROUP_BG"]};
        color: {t["TEXT_PRIMARY"]};
    }}
    QPushButton {{
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 5px 10px;
        font-weight: 700;
        min-height: 24px;
    }}
    QPushButton#PrimaryButton {{
        background: {t["PRIMARY"]};
        color: white;
        border-color: {t["PRIMARY_HOVER"]};
    }}
    QPushButton#PrimaryButton:hover {{
        background: {t["PRIMARY_HOVER"]};
        border-color: {t["FIELD_BORDER_FOCUS"]};
    }}
    QPushButton#PrimaryButton:pressed {{
        background: {t["PRIMARY_PRESSED"]};
        border-color: {t["TEXT_PRIMARY"]};
    }}
    QPushButton#PrimaryButton:disabled {{
        background: {t["DISABLED_BG"]};
        color: {t["DISABLED_TEXT"]};
        border-color: {t["SEPARATOR"]};
    }}
    QPushButton#SecondaryButton {{
        background: {t["SECONDARY"]};
        color: {t["TEXT_SECONDARY"]};
        border-color: {t["FIELD_BORDER"]};
    }}
    QPushButton#SecondaryButton:hover {{
        background: {t["SECONDARY_HOVER"]};
        color: {t["TEXT_PRIMARY"]};
        border-color: {t["FIELD_BORDER_FOCUS"]};
    }}
    QPushButton#SecondaryButton:pressed {{
        background: {t["SECONDARY_PRESSED"]};
        color: {t["TEXT_PRIMARY"]};
        border-color: {t["PRIMARY"]};
    }}
    QPushButton#SecondaryButton:disabled {{
        background: {t["DISABLED_BG"]};
        color: {t["DISABLED_TEXT"]};
        border-color: {t["SEPARATOR"]};
    }}
    QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {t["FIELD_BG"]};
        border: 1px solid {t["FIELD_BORDER"]};
        border-radius: 5px;
        padding: 4px 6px;
        min-height: 22px;
        color: {t["TEXT_SECONDARY"]};
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        background: {t["FIELD_BG_HOVER"]};
        border-color: {t["FIELD_BORDER_FOCUS"]};
    }}
    QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background: {t["DISABLED_BG"]};
        color: {t["DISABLED_TEXT"]};
        border-color: {t["SEPARATOR"]};
    }}
    QCheckBox {{
        spacing: 8px;
        font-weight: 600;
        color: {t["TEXT_SECONDARY"]};
    }}
    QCheckBox:disabled {{
        color: {t["DISABLED_TEXT"]};
    }}
    QProgressBar {{
        background: {t["FIELD_BG"]};
        border: none;
        border-radius: 5px;
        height: 10px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{
        background: {t["PRIMARY"]};
        border-radius: 5px;
    }}
    QMenuBar {{
        background: {t["PANEL_BG_ALT"]};
        color: {t["TEXT_SECONDARY"]};
        border-bottom: 1px solid {t["SEPARATOR"]};
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 4px 10px;
    }}
    QMenuBar::item:selected {{
        background: {t["SECONDARY_HOVER"]};
        color: {t["TEXT_PRIMARY"]};
    }}
    QMenu {{
        background: {t["PANEL_BG"]};
        color: {t["TEXT_SECONDARY"]};
        border: 1px solid {t["SEPARATOR"]};
    }}
    QMenu::item {{
        padding: 5px 24px 5px 20px;
    }}
    QMenu::item:selected {{
        background: {t["SECONDARY_HOVER"]};
        color: {t["TEXT_PRIMARY"]};
    }}
    QStatusBar#AppStatusBar {{
        background: {t["PANEL_BG_ALT"]};
        border-top: 1px solid {t["SEPARATOR"]};
    }}
    QDockWidget::title {{
        background: {t["PANEL_BG"]};
        padding: 7px;
        border-bottom: 1px solid {t["SEPARATOR"]};
        font-weight: 700;
    }}
    QTreeView#SceneTree {{
        background: {t["APP_BG"]};
        border: 1px solid {t["SEPARATOR"]};
        color: {t["TEXT_SECONDARY"]};
        gridline-color: {t["SEPARATOR"]};
        selection-background-color: {t["PRIMARY_PRESSED"]};
        selection-color: {t["TEXT_PRIMARY"]};
    }}
    QHeaderView::section {{
        background: {t["PANEL_BG"]};
        color: {t["TEXT_SECONDARY"]};
        border: 1px solid {t["SEPARATOR"]};
        padding: 5px;
        font-weight: 700;
    }}
    QFrame#ViewportPanel, QFrame#StageProgressPanel {{
        background: {t["PANEL_BG"]};
        border: 1px solid {t["SEPARATOR"]};
        border-radius: 6px;
    }}
    QFrame#ViewportCanvas {{
        background: {t["APP_BG"]};
        border: 1px solid {t["SEPARATOR"]};
        border-radius: 6px;
    }}
    QLabel#StageCurrent {{
        color: {t["TEXT_PRIMARY"]};
        font-weight: 700;
    }}
    QWidget#StepIndicator {{
        background: {t["SECONDARY"]};
        border: 1px solid {t["FIELD_BORDER"]};
        border-radius: 5px;
    }}
    QWidget#StepIndicator[state="active"] {{
        background: {t["PANEL_BG_ALT"]};
        border-color: {t["PRIMARY"]};
    }}
    QWidget#StepIndicator[state="completed"] {{
        background: {t["PANEL_BG_ALT"]};
        border-color: {t["SUCCESS"]};
    }}
    QWidget#StepIndicator[state="failed"] {{
        background: {t["PANEL_BG_ALT"]};
        border-color: {t["ERROR"]};
    }}
    QWidget#StepIndicator[state="disabled"] {{
        background: {t["DISABLED_BG"]};
        border-color: {t["SEPARATOR"]};
    }}
    QFrame#StepAccent {{
        background: {t["FIELD_BORDER"]};
        border-top-left-radius: 5px;
        border-bottom-left-radius: 5px;
    }}
    QFrame#StepAccent[state="active"] {{
        background: {t["PRIMARY"]};
    }}
    QFrame#StepAccent[state="completed"] {{
        background: {t["SUCCESS"]};
    }}
    QFrame#StepAccent[state="failed"] {{
        background: {t["ERROR"]};
    }}
    QFrame#StepAccent[state="disabled"] {{
        background: {t["SEPARATOR"]};
    }}
    QLabel#StepLabel {{
        color: {t["TEXT_SECONDARY"]};
        font-weight: 700;
    }}
    QLabel#StepLabel[state="active"] {{
        color: {t["TEXT_PRIMARY"]};
    }}
    QLabel#StepLabel[state="completed"] {{
        color: {t["TEXT_SECONDARY"]};
    }}
    QLabel#StepLabel[state="failed"] {{
        color: {t["TEXT_SECONDARY"]};
    }}
    QLabel#StepLabel[state="disabled"] {{
        color: {t["DISABLED_TEXT"]};
    }}
    QFrame#StepConnector {{
        background: {t["SEPARATOR"]};
        border: none;
        min-height: 2px;
        max-height: 2px;
    }}
    QFrame#StepConnector[state="active"] {{
        background: {t["PRIMARY"]};
    }}
    QFrame#StepConnector[state="completed"] {{
        background: {t["SUCCESS"]};
    }}
    QFrame#StepConnector[state="failed"] {{
        background: {t["ERROR"]};
    }}
    QProgressBar#ReconstructionProgress {{
        min-height: 14px;
        max-height: 14px;
        color: {t["TEXT_SECONDARY"]};
        text-align: center;
    }}
    """
