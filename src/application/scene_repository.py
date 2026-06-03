from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models.scene_result import SceneResult


class SceneRepository:
    """Persist and retrieve scene history records from the local JSON index."""

    def __init__(self, index_path: Path, logger: logging.Logger | None = None) -> None:
        self.index_path = index_path
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def list_records(self) -> list[dict]:
        self._ensure_index_exists()
        try:
            return list(json.loads(self.index_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            self.logger.exception("Invalid scene index JSON: %s", self.index_path)
            return []

    def add_scene(self, scene_result: SceneResult) -> list[dict]:
        records = self.list_records()
        records.insert(0, scene_result.to_record())
        self.replace_records(records)
        return records

    def replace_records(self, records: list[dict]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _ensure_index_exists(self) -> None:
        if self.index_path.exists():
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text("[]", encoding="utf-8")
