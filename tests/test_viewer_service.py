from __future__ import annotations

import unittest
from pathlib import Path

from src.application.viewer_service import ViewerService
from src.models.scene_result import SceneResult


class ViewerServiceTests(unittest.TestCase):
    def test_open_scene_uses_configured_viewer_settings(self) -> None:
        calls = []

        def fake_launcher(scene_result, **kwargs):
            calls.append((scene_result, kwargs))
            return "viewer-process"

        scene_result = SceneResult(
            scene_id="scene",
            source_video=Path("video.mp4"),
            output_dir=Path("output"),
        )
        service = ViewerService(
            {"viewer": {"preferred": "open3d", "geometry_mode": "mesh"}},
            launcher=fake_launcher,
        )

        result = service.open_scene(scene_result)

        self.assertEqual(result, "viewer-process")
        self.assertEqual(
            calls,
            [(scene_result, {"preferred_viewer": "open3d", "geometry_mode": "mesh", "open_browser": False})],
        )

    def test_open_scene_preserves_existing_viewer_defaults(self) -> None:
        calls = []

        def fake_launcher(scene_result, **kwargs):
            calls.append(kwargs)

        scene_result = SceneResult(
            scene_id="scene",
            source_video=Path("video.mp4"),
            output_dir=Path("output"),
        )
        service = ViewerService({}, launcher=fake_launcher)

        service.open_scene(scene_result)

        self.assertEqual(calls, [{"preferred_viewer": "viser", "geometry_mode": "gaussian", "open_browser": False}])


if __name__ == "__main__":
    unittest.main()
