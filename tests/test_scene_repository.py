from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from src.application.scene_repository import SceneRepository
from src.models.scene_result import SceneResult


def temporary_workspace():
    temp_root = Path(__file__).resolve().parents[1] / "src" / "data" / "test_scene_repository_tmp"
    temp_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=temp_root)


class SceneRepositoryTests(unittest.TestCase):
    def test_list_records_creates_missing_index_file(self) -> None:
        with temporary_workspace() as tmpdir:
            index_path = Path(tmpdir) / "nested" / "scenes.json"
            repository = SceneRepository(index_path)

            records = repository.list_records()

            self.assertEqual(records, [])
            self.assertEqual(index_path.read_text(encoding="utf-8"), "[]")

    def test_list_records_returns_empty_list_for_invalid_json(self) -> None:
        with temporary_workspace() as tmpdir:
            index_path = Path(tmpdir) / "scenes.json"
            index_path.write_text("{not valid json", encoding="utf-8")
            logger = logging.getLogger("scene_repository_test_invalid_json")
            logger.disabled = True
            repository = SceneRepository(index_path, logger=logger)

            records = repository.list_records()

            self.assertEqual(records, [])
            self.assertEqual(index_path.read_text(encoding="utf-8"), "{not valid json")

    def test_add_scene_prepends_record_and_persists_index(self) -> None:
        with temporary_workspace() as tmpdir:
            index_path = Path(tmpdir) / "scenes.json"
            existing_record = {
                "scene_id": "old_scene",
                "source_video": "old.mp4",
                "output_dir": "old_output",
                "pointcloud_path": "",
                "mesh_path": "",
                "camera_poses": [],
                "metadata": {},
            }
            index_path.write_text(json.dumps([existing_record]), encoding="utf-8")
            repository = SceneRepository(index_path)
            scene_result = SceneResult(
                scene_id="new_scene",
                source_video=Path("new.mp4"),
                output_dir=Path("new_output"),
                pointcloud_path=Path("pointcloud.ply"),
                mesh_path=None,
                camera_poses=[{"frame": 1}],
                metadata={"num_points_final": 42},
            )

            records = repository.add_scene(scene_result)

            persisted = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(records, persisted)
            self.assertEqual([record["scene_id"] for record in records], ["new_scene", "old_scene"])
            self.assertEqual(records[0]["pointcloud_path"], "pointcloud.ply")
            self.assertEqual(records[0]["mesh_path"], "")


if __name__ == "__main__":
    unittest.main()
