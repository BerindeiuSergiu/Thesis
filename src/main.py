from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
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


def main() -> int:
    src_root = Path(__file__).resolve().parent
    config = load_config(src_root)
    configure_logging(config)
    ensure_runtime_files(config)

    app = QApplication(sys.argv)
    window = MainWindow(config=config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
