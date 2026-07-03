from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from src.app.main_window import MainWindow


def load_config(src_root: Path) -> dict:
    config_path = src_root / "config" / "settings.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    app_config = config.setdefault("app", {})
    for key in ("scenes_index_path", "outputs_root", "log_file"):
        value = Path(app_config[key])
        if not value.is_absolute():
            app_config[key] = str((src_root / value).resolve())
    return config


def configure_logging(config: dict) -> None:
    log_level = getattr(logging, str(config.get("logging", {}).get("level", "INFO")).upper(), logging.INFO)
    log_file = Path(config.get("app", {}).get("log_file"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def ensure_runtime_files(config: dict) -> None:
    scenes_index = Path(config.get("app", {}).get("scenes_index_path"))
    scenes_index.parent.mkdir(parents=True, exist_ok=True)
    if not scenes_index.exists():
        scenes_index.write_text("[]", encoding="utf-8")

    outputs_root = Path(config.get("app", {}).get("outputs_root"))
    outputs_root.mkdir(parents=True, exist_ok=True)


def configure_windows_app_identity(app_name: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        app_id = f"thesis.{app_name.lower().replace(' ', '.')}"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        logging.getLogger(__name__).debug("Could not set Windows app identity.", exc_info=True)


def resolve_app_icon_path(config: dict, src_root: Path) -> Path | None:
    icon_path_value = config.get("app", {}).get("icon_path")
    if not icon_path_value:
        return None

    icon_path = Path(icon_path_value)
    if not icon_path.is_absolute():
        icon_path = src_root / icon_path
    if not icon_path.exists():
        logging.getLogger(__name__).warning("Configured app icon does not exist: %s", icon_path)
        return None

    return icon_path


def load_app_icon(icon_path: Path | None) -> QIcon | None:
    if icon_path is None:
        return None

    icon = QIcon(str(icon_path))
    return icon if not icon.isNull() else None


def apply_windows_window_icon(window: MainWindow, icon_path: Path | None) -> None:
    if sys.platform != "win32" or icon_path is None:
        return

    try:
        import ctypes
        from ctypes import wintypes

        hwnd = wintypes.HWND(int(window.winId()))
        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.SendMessageW.restype = wintypes.LPARAM
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        image_icon = 1
        wm_seticon = 0x0080
        icon_small = 0
        icon_big = 1
        lr_loadfromfile = 0x0010
        lr_shared = 0x8000

        small_size = user32.GetSystemMetrics(49) or 16
        large_size = user32.GetSystemMetrics(11) or 32
        load_flags = lr_loadfromfile | lr_shared
        small_icon = user32.LoadImageW(None, str(icon_path), image_icon, small_size, small_size, load_flags)
        large_icon = user32.LoadImageW(None, str(icon_path), image_icon, large_size, large_size, load_flags)

        if small_icon:
            user32.SendMessageW(hwnd, wm_seticon, icon_small, small_icon.value)
        if large_icon:
            user32.SendMessageW(hwnd, wm_seticon, icon_big, large_icon.value)
        window._native_icon_handles = (small_icon, large_icon)
    except Exception:
        logging.getLogger(__name__).debug("Could not set native Windows window icon.", exc_info=True)


def main() -> int:
    src_root = Path(__file__).resolve().parent
    config = load_config(src_root)
    configure_logging(config)
    ensure_runtime_files(config)

    app_name = str(config.get("app", {}).get("name", "Fast3R Desktop Wrapper"))
    configure_windows_app_identity(app_name)

    app = QApplication(sys.argv)
    app_icon_path = resolve_app_icon_path(config, src_root)
    app_icon = load_app_icon(app_icon_path)
    if app_icon is not None:
        app.setWindowIcon(app_icon)

    window = MainWindow(config=config)
    if app_icon is not None:
        window.setWindowIcon(app_icon)
    apply_windows_window_icon(window, app_icon_path)
    window.show()
    apply_windows_window_icon(window, app_icon_path)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
