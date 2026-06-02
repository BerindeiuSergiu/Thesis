# NOTE: Do NOT import directly from experiments/.
# All pipeline code must be copied into src/pipeline/fast3r/ before modification.
# Best methods from experiments/ have been audited and integrated — see src/EXPERIMENTS_AUDIT.md

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import open3d as o3d

    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


def radius_percentile_filter(
    points: np.ndarray,
    colors: np.ndarray | None,
    percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    if points.size == 0:
        return points, colors, {"removed_points": 0, "retention_ratio": 1.0}

    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.all():
        points = points[finite_mask]
        colors = colors[finite_mask] if colors is not None and len(colors) == len(finite_mask) else colors
    if points.size == 0:
        return points, colors, {
            "removed_points": int(len(finite_mask)),
            "retention_ratio": 0.0,
            "invalid_points_dropped": int(np.count_nonzero(~finite_mask)),
            "radius_percentile_threshold": None,
        }

    center = np.median(points, axis=0)
    offset = points - center
    finite_offset_mask = np.isfinite(offset).all(axis=1)
    if not finite_offset_mask.all():
        points = points[finite_offset_mask]
        colors = colors[finite_offset_mask] if colors is not None and len(colors) == len(finite_offset_mask) else colors
        offset = points - np.median(points, axis=0)
    dist = np.sqrt(np.sum(offset.astype(np.float64) ** 2, axis=1))
    dist = np.nan_to_num(dist, nan=np.inf, posinf=np.inf, neginf=np.inf)
    finite_dist = dist[np.isfinite(dist)]
    if finite_dist.size == 0:
        return points[:0], colors[:0] if colors is not None and len(colors) == len(points) else colors, {
            "removed_points": int(len(points)),
            "retention_ratio": 0.0,
            "invalid_points_dropped": int(np.count_nonzero(~finite_mask)),
            "radius_percentile_threshold": None,
        }

    threshold = float(np.percentile(finite_dist, percentile))
    mask = dist <= threshold
    filtered_points = points[mask]
    filtered_colors = colors[mask] if colors is not None and len(colors) == len(points) else colors
    return filtered_points, filtered_colors, {
        "removed_points": int(len(finite_mask) - len(filtered_points)),
        "retention_ratio": float(len(filtered_points) / max(len(points), 1)),
        "invalid_points_dropped": int(np.count_nonzero(~finite_mask)),
        "radius_percentile_threshold": threshold,
    }


def numeric_sanity_filter(
    points: np.ndarray,
    colors: np.ndarray | None,
    max_abs_coordinate: float = 100.0,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    if points.size == 0:
        return points, colors, {
            "input_points": 0,
            "finite_points": 0,
            "points_after_max_abs": 0,
            "removed_points": 0,
            "retention_ratio": 1.0,
            "max_abs_coordinate": float(max_abs_coordinate),
        }

    points = np.asarray(points, dtype=np.float32)
    input_count = int(points.shape[0])
    finite_mask = np.isfinite(points).all(axis=1)
    max_abs = float(max_abs_coordinate)
    if max_abs > 0 and np.isfinite(max_abs):
        magnitude_mask = np.max(np.abs(np.nan_to_num(points, nan=np.inf, posinf=np.inf, neginf=np.inf)), axis=1) <= max_abs
    else:
        magnitude_mask = np.ones(input_count, dtype=bool)
    mask = finite_mask & magnitude_mask
    filtered_points = points[mask]
    filtered_colors = colors[mask] if colors is not None and len(colors) == input_count else colors
    stats = {
        "input_points": input_count,
        "finite_points": int(np.count_nonzero(finite_mask)),
        "nonfinite_points_dropped": int(np.count_nonzero(~finite_mask)),
        "points_after_max_abs": int(np.count_nonzero(mask)),
        "large_coordinate_points_dropped": int(np.count_nonzero(finite_mask & ~magnitude_mask)),
        "removed_points": int(input_count - np.count_nonzero(mask)),
        "retention_ratio": float(np.count_nonzero(mask) / max(input_count, 1)),
        "max_abs_coordinate": max_abs,
    }
    return filtered_points, filtered_colors, stats


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not O3D_AVAILABLE or points.size == 0:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    points_ds = np.asarray(pcd.points)
    colors_ds = np.asarray(pcd.colors) if len(pcd.colors) else colors
    return points_ds, colors_ds


def save_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if O3D_AVAILABLE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(str(path), pcd)
        return path

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        if colors is not None and len(colors) == len(points):
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for idx, point in enumerate(points):
            row = [f"{float(point[0]):.6f}", f"{float(point[1]):.6f}", f"{float(point[2]):.6f}"]
            if colors is not None and len(colors) == len(points):
                color = np.clip(colors[idx], 0.0, 1.0) * 255.0
                row.extend([str(int(color[0])), str(int(color[1])), str(int(color[2]))])
            handle.write(" ".join(row) + "\n")
    return path


def create_dummy_pointcloud(path: Path) -> Path:
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2 * np.pi, 1500)
    costheta = rng.uniform(-1.0, 1.0, 1500)
    theta = np.arccos(costheta)
    radius = 0.5 + 0.05 * rng.normal(size=1500)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = np.clip((points - points.min()) / max((points.max() - points.min()), 1e-6), 0.0, 1.0)
    return save_pointcloud_ply(path, points, colors)


def build_mesh_from_pointcloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    output_dir: Path,
    method: str,
    bpa_radii: list[float],
) -> Path | None:
    if not O3D_AVAILABLE or points.size == 0:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if method == "poisson":
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    else:
        radii = o3d.utility.DoubleVector([float(value) for value in bpa_radii])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    mesh.compute_vertex_normals()
    mesh_path = Path(output_dir) / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    return mesh_path
