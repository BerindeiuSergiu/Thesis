from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.models.scene_result import SceneResult
from src.viewer.scene_viewer import launch_scene_viewer

ViewerLauncher = Callable[..., Any]


class ViewerService:
    """Resolve viewer settings and launch scene inspection."""

    def __init__(self, config: dict, launcher: ViewerLauncher = launch_scene_viewer) -> None:
        self.config = config
        self.launcher = launcher

    def open_scene(self, scene_result: SceneResult) -> Any:
        viewer_config = self.config.get("viewer", {})
        preferred_viewer = viewer_config.get("preferred", "viser")
        geometry_mode = viewer_config.get("geometry_mode", "gaussian")
        return self.launcher(
            scene_result,
            preferred_viewer=preferred_viewer,
            geometry_mode=geometry_mode,
        )
