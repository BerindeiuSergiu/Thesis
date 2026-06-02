from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.pipeline.fast3r.mesh_reconstruction import build_poisson_mesh

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit(f"Open3D is required: {exc}") from exc


def _load_scene_pointcloud(scene_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    pointcloud_path = scene_dir / "pointcloud.ply"
    if not pointcloud_path.exists():
        raise FileNotFoundError(f"Missing point cloud: {pointcloud_path}")
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) else None
    if points.size == 0:
        raise RuntimeError(f"Point cloud has no points: {pointcloud_path}")
    return points, colors


def _update_scene_metadata(scene_dir: Path, report: dict) -> None:
    metadata_path = scene_dir / "scene_metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["mesh_branch"] = report
    outputs = metadata.setdefault("outputs", {})
    outputs["mesh_ply"] = report.get("mesh_ply", "")
    outputs["mesh_obj"] = report.get("mesh_obj", "")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _update_scene_index(scene_dir: Path, report: dict) -> None:
    scenes_index = Path("src/data/scenes_index.json")
    if not scenes_index.exists():
        return
    records = json.loads(scenes_index.read_text(encoding="utf-8"))
    scene_dir_resolved = str(scene_dir.resolve())
    for record in records:
        if str(Path(record.get("output_dir", "")).resolve()) != scene_dir_resolved:
            continue
        record["mesh_path"] = report.get("mesh_ply", "")
        metadata = record.setdefault("metadata", {})
        metadata["mesh_branch"] = report
        outputs = metadata.setdefault("outputs", {})
        outputs["mesh_ply"] = report.get("mesh_ply", "")
        outputs["mesh_obj"] = report.get("mesh_obj", "")
        break
    scenes_index.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Poisson mesh from an existing src scene point cloud.")
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--mesh-voxel-size", type=float, default=0.008)
    parser.add_argument("--mesh-max-points", type=int, default=500000)
    parser.add_argument("--target-triangles", type=int, default=180000)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    points, colors = _load_scene_pointcloud(scene_dir)
    config = {
        "mesh_enabled": True,
        "mesh_voxel_size": float(args.mesh_voxel_size),
        "mesh_max_points": int(args.mesh_max_points),
        "poisson_depth": int(args.depth),
        "poisson_scale": 1.1,
        "poisson_linear_fit": False,
        "poisson_trim_quantile": 0.01,
        "crop_bbox_margin": 0.03,
        "orient_normals": True,
        "mesh_normal_radius": 0.0,
        "mesh_normal_max_nn": 64,
        "min_component_triangles": 1000,
        "min_component_ratio": 0.01,
        "laplacian_iterations": 1,
        "laplacian_lambda": 0.2,
        "target_triangles": int(args.target_triangles),
        "color_vertices": True,
    }
    report = build_poisson_mesh(
        points,
        colors,
        output_dir=scene_dir / "mesh",
        config=config,
        scale_to_meters=1.0,
    )
    _update_scene_metadata(scene_dir, report)
    _update_scene_index(scene_dir, report)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
