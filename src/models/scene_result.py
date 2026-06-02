from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SceneResult:
    scene_id: str
    source_video: Path
    output_dir: Path
    pointcloud_path: Path | None = None
    mesh_path: Path | None = None
    camera_poses: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "source_video": str(self.source_video),
            "output_dir": str(self.output_dir),
            "pointcloud_path": str(self.pointcloud_path) if self.pointcloud_path else "",
            "mesh_path": str(self.mesh_path) if self.mesh_path else "",
            "camera_poses": self.camera_poses,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict) -> "SceneResult":
        return cls(
            scene_id=record["scene_id"],
            source_video=Path(record["source_video"]),
            output_dir=Path(record["output_dir"]),
            pointcloud_path=Path(record["pointcloud_path"]) if record.get("pointcloud_path") else None,
            mesh_path=Path(record["mesh_path"]) if record.get("mesh_path") else None,
            camera_poses=list(record.get("camera_poses", [])),
            metadata=dict(record.get("metadata", {})),
        )
