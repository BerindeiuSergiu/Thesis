from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.models.scene_result import SceneResult


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Scene:
    scene_id: str
    name: str
    source_video: Path
    description: str = ""
    pipeline: str = "default"
    reconstruction_preset: str = ""
    tags: list[str] = field(default_factory=list)
    status: str = "Draft"
    created_at: str = field(default_factory=_now_iso)
    reconstruction_type: str = "Not reconstructed"
    output_dir: Path | None = None
    pointcloud_path: Path | None = None
    mesh_path: Path | None = None
    camera_poses: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "source_video": str(self.source_video),
            "description": self.description,
            "pipeline": self.pipeline,
            "reconstruction_preset": self.reconstruction_preset,
            "tags": self.tags,
            "status": self.status,
            "created_at": self.created_at,
            "reconstruction_type": self.reconstruction_type,
            "output_dir": str(self.output_dir) if self.output_dir else "",
            "pointcloud_path": str(self.pointcloud_path) if self.pointcloud_path else "",
            "mesh_path": str(self.mesh_path) if self.mesh_path else "",
            "camera_poses": self.camera_poses,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict) -> "Scene":
        source_video = Path(str(record.get("source_video", "")))
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        outputs = metadata.get("outputs", {}) if isinstance(metadata.get("outputs", {}), dict) else {}
        name = str(record.get("name") or metadata.get("scene_name") or source_video.stem or record.get("scene_id", "Untitled Scene"))
        status = str(record.get("status") or _status_from_record(record))
        reconstruction_type = str(record.get("reconstruction_type") or _reconstruction_type_from_record(record))
        return cls(
            scene_id=str(record.get("scene_id") or f"scene_{_now_iso().replace(':', '').replace('-', '')}"),
            name=name,
            source_video=source_video,
            description=str(record.get("description", "")),
            pipeline=str(record.get("pipeline", "default")),
            reconstruction_preset=str(record.get("reconstruction_preset", "")),
            tags=list(record.get("tags", [])) if isinstance(record.get("tags", []), list) else [],
            status=status,
            created_at=str(record.get("created_at") or metadata.get("created_at") or ""),
            reconstruction_type=reconstruction_type,
            output_dir=Path(record["output_dir"]) if record.get("output_dir") else None,
            pointcloud_path=Path(record["pointcloud_path"]) if record.get("pointcloud_path") else None,
            mesh_path=Path(record["mesh_path"]) if record.get("mesh_path") else None,
            camera_poses=list(record.get("camera_poses", [])),
            metadata=metadata | {"outputs": outputs} if outputs else metadata,
        )

    def with_result(self, result: SceneResult) -> "Scene":
        metadata = dict(result.metadata)
        metadata.update(
            {
                "scene_name": self.name,
                "source_scene_id": self.scene_id,
                "pipeline_output_id": result.scene_id,
                "description": self.description,
                "tags": self.tags,
                "created_at": self.created_at,
            }
        )
        return Scene(
            scene_id=self.scene_id,
            name=self.name,
            source_video=result.source_video,
            description=self.description,
            pipeline=self.pipeline,
            reconstruction_preset=self.reconstruction_preset,
            tags=self.tags,
            status="Ready",
            created_at=self.created_at,
            reconstruction_type=_reconstruction_type_from_result(result),
            output_dir=result.output_dir,
            pointcloud_path=result.pointcloud_path,
            mesh_path=result.mesh_path,
            camera_poses=result.camera_poses,
            metadata=metadata,
        )


def _status_from_record(record: dict) -> str:
    if record.get("output_dir"):
        return "Ready"
    return "Draft"


def _reconstruction_type_from_record(record: dict) -> str:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    outputs = metadata.get("outputs", {}) if isinstance(metadata.get("outputs", {}), dict) else {}
    has_gaussian = bool(outputs.get("gaussian_ply")) or bool(
        (metadata.get("gaussian_splat", {}) if isinstance(metadata.get("gaussian_splat", {}), dict) else {}).get("gaussian_ply")
    )
    has_mesh = bool(outputs.get("mesh_ply")) or bool(record.get("mesh_path"))
    if has_gaussian and has_mesh:
        return "Gaussian + Mesh Reconstruction"
    if has_gaussian:
        return "Gaussian Reconstruction"
    if has_mesh:
        return "Mesh Reconstruction"
    if record.get("pointcloud_path"):
        return "Point Cloud Reconstruction"
    return "Not reconstructed"


def _reconstruction_type_from_result(result: SceneResult) -> str:
    record = result.to_record()
    return _reconstruction_type_from_record(record)
