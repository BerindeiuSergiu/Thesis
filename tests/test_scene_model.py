from __future__ import annotations

import unittest
from pathlib import Path

from src.models.scene import Scene
from src.models.scene_result import SceneResult


class SceneModelTests(unittest.TestCase):
    def test_scene_record_preserves_user_facing_fields(self) -> None:
        scene = Scene(
            scene_id="scene_internal",
            name="Living Room Apartment",
            source_video=Path("living_room.mp4"),
            description="Evening capture",
            pipeline="default",
            reconstruction_preset="high_quality_detail",
            tags=["apartment", "living room"],
            created_at="2026-06-03T12:00:00",
        )

        record = scene.to_record()
        restored = Scene.from_record(record)

        self.assertEqual(restored.scene_id, "scene_internal")
        self.assertEqual(restored.name, "Living Room Apartment")
        self.assertEqual(restored.status, "Draft")
        self.assertEqual(restored.reconstruction_type, "Not reconstructed")
        self.assertEqual(restored.tags, ["apartment", "living room"])

    def test_with_result_keeps_scene_identity_and_attaches_outputs(self) -> None:
        scene = Scene(
            scene_id="scene_internal",
            name="Living Room Apartment",
            source_video=Path("living_room.mp4"),
            pipeline="default",
            reconstruction_preset="high_quality_detail",
            created_at="2026-06-03T12:00:00",
        )
        result = SceneResult(
            scene_id="scene_20260603_120100",
            source_video=Path("living_room.mp4"),
            output_dir=Path("outputs/scene_20260603_120100"),
            pointcloud_path=Path("outputs/scene_20260603_120100/cloud.ply"),
            metadata={"outputs": {"gaussian_ply": "gaussian.ply"}, "num_points_final": 814638},
        )

        ready_scene = scene.with_result(result)

        self.assertEqual(ready_scene.scene_id, "scene_internal")
        self.assertEqual(ready_scene.name, "Living Room Apartment")
        self.assertEqual(ready_scene.status, "Ready")
        self.assertEqual(ready_scene.reconstruction_type, "Gaussian Reconstruction")
        self.assertEqual(ready_scene.metadata["pipeline_output_id"], "scene_20260603_120100")


if __name__ == "__main__":
    unittest.main()
