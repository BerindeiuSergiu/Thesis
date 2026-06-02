"""Reusable point cloud cleanup helpers for beta reconstruction pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import open3d as o3d

    O3D_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime dependency guard
    O3D_AVAILABLE = False
    o3d = None


@dataclass
class PointCloudProcessingConfig:
    """Common point cloud preprocessing settings."""

    high_detail_mode: bool = False
    finite_filter: bool = True
    max_abs_coordinate: float = 0.0
    radius_percentile: float = 0.0
    statistical_outlier_removal: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    voxel_size: float = 0.002
    max_points: Optional[int] = None
    random_seed: int = 42
    detail_metrics_sample_size: int = 200000


def coerce_colors(colors: Optional[np.ndarray], n_points: int) -> np.ndarray:
    """Return Nx3 float colors in [0, 1], defaulting to neutral gray."""

    if colors is None or colors.ndim != 2 or colors.shape[0] != n_points:
        return np.full((n_points, 3), 0.5, dtype=np.float32)

    colors = colors[:, :3] if colors.shape[1] >= 3 else colors
    if colors.shape[1] < 3:
        padded = np.full((n_points, 3), 0.5, dtype=np.float32)
        padded[:, : colors.shape[1]] = colors
        colors = padded

    colors = colors.astype(np.float32, copy=False)
    if colors.size and float(np.nanmax(colors)) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def finite_point_mask(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=1)


def max_abs_coordinate_mask(points: np.ndarray, max_abs_coordinate: float) -> np.ndarray:
    max_abs = float(max_abs_coordinate)
    if max_abs <= 0 or not np.isfinite(max_abs):
        return np.ones(points.shape[0], dtype=bool)
    safe_points = np.nan_to_num(points, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return np.max(np.abs(safe_points), axis=1) <= max_abs


def radius_percentile_mask(points: np.ndarray, percentile: float) -> tuple[np.ndarray, Optional[float]]:
    if points.size == 0:
        return np.ones(points.shape[0], dtype=bool), None
    percentile = float(percentile)
    if percentile <= 0 or percentile >= 100:
        return np.ones(points.shape[0], dtype=bool), None

    center = np.median(points, axis=0)
    dist = np.linalg.norm((points - center).astype(np.float64), axis=1)
    dist = np.nan_to_num(dist, nan=np.inf, posinf=np.inf, neginf=np.inf)
    finite_dist = dist[np.isfinite(dist)]
    if finite_dist.size == 0:
        return np.zeros(points.shape[0], dtype=bool), None
    threshold = float(np.percentile(finite_dist, percentile))
    return dist <= threshold, threshold


def subsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if max_points is None or points.shape[0] <= max_points:
        return points, colors, None
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices], colors[indices], indices


def make_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None):
    if not O3D_AVAILABLE:
        raise RuntimeError("Open3D is required for point cloud operations.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(coerce_colors(colors, len(points)).astype(np.float64))
    return pcd


def nearest_neighbor_stats(points: np.ndarray, sample_size: int, seed: int) -> dict:
    if not O3D_AVAILABLE or points.shape[0] < 2 or sample_size <= 0:
        return {}
    if points.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(points.shape[0], size=int(sample_size), replace=False)
        sample = points[sample_idx]
    else:
        sample = points
    pcd = make_pointcloud(sample)
    distances = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float32)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size == 0:
        return {}
    return {
        "sample_size": int(sample.shape[0]),
        "nn_min": float(np.min(distances)),
        "nn_p10": float(np.percentile(distances, 10)),
        "nn_median": float(np.median(distances)),
        "nn_mean": float(np.mean(distances)),
        "nn_p90": float(np.percentile(distances, 90)),
        "nn_max": float(np.max(distances)),
    }


def process_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    config: PointCloudProcessingConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Clean a model point cloud once for downstream branches."""

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    original_count = int(points.shape[0])
    if config.finite_filter:
        mask = finite_point_mask(points)
        points = points[mask]
        colors = colors[mask] if colors is not None and len(colors) == len(mask) else colors
    else:
        mask = np.ones(original_count, dtype=bool)

    colors_arr = coerce_colors(colors, points.shape[0])
    finite_count = int(points.shape[0])

    effective_radius_percentile = float(config.radius_percentile)
    effective_voxel_size = float(config.voxel_size)
    effective_statistical_outlier_removal = bool(config.statistical_outlier_removal)
    effective_max_points = config.max_points
    if config.high_detail_mode:
        if effective_radius_percentile > 0:
            effective_radius_percentile = max(effective_radius_percentile, 99.9)
        effective_voxel_size = 0.0
        effective_statistical_outlier_removal = False
        effective_max_points = None

    input_nn_stats = nearest_neighbor_stats(points, int(config.detail_metrics_sample_size), int(config.random_seed))

    max_abs_mask = max_abs_coordinate_mask(points, float(config.max_abs_coordinate))
    if not max_abs_mask.all():
        points = points[max_abs_mask]
        colors_arr = colors_arr[max_abs_mask]
    max_abs_count = int(points.shape[0])

    radius_mask, radius_threshold = radius_percentile_mask(points, effective_radius_percentile)
    if not radius_mask.all():
        points = points[radius_mask]
        colors_arr = colors_arr[radius_mask]
    radius_count = int(points.shape[0])

    points, colors_arr, sample_indices = subsample_points(
        points,
        colors_arr,
        max_points=effective_max_points,
        seed=config.random_seed,
    )

    stats = {
        "input_points": original_count,
        "high_detail_mode": bool(config.high_detail_mode),
        "finite_points": finite_count,
        "invalid_points_dropped": int(original_count - finite_count),
        "finite_retention_ratio": float(finite_count / original_count) if original_count else 0.0,
        "max_abs_coordinate": float(config.max_abs_coordinate),
        "points_after_max_abs_coordinate": max_abs_count,
        "points_removed_by_max_abs_coordinate": int(finite_count - max_abs_count),
        "configured_radius_percentile": float(config.radius_percentile),
        "effective_radius_percentile": float(effective_radius_percentile),
        "radius_percentile_threshold": radius_threshold,
        "points_after_radius_percentile": radius_count,
        "points_removed_by_radius_percentile": int(max_abs_count - radius_count),
        "subsample_applied": sample_indices is not None,
        "points_after_subsample": int(points.shape[0]),
        "points_removed_by_subsample": int(radius_count - points.shape[0]),
        "configured_statistical_outlier_removal": bool(config.statistical_outlier_removal),
        "effective_statistical_outlier_removal": bool(effective_statistical_outlier_removal),
        "configured_voxel_size": float(config.voxel_size),
        "effective_voxel_size": float(effective_voxel_size),
        "input_nearest_neighbor_stats": input_nn_stats,
    }

    if not O3D_AVAILABLE:
        stats["open3d_available"] = False
        stats["points_after_outlier_removal"] = int(points.shape[0])
        stats["points_after_voxel"] = int(points.shape[0])
        stats["points_removed_by_outlier_removal"] = 0
        stats["points_removed_by_voxel"] = 0
        stats["overall_retention_ratio"] = float(points.shape[0] / original_count) if original_count else 0.0
        stats["output_nearest_neighbor_stats"] = {}
        return points, colors_arr, stats

    pcd = make_pointcloud(points, colors_arr)

    if effective_statistical_outlier_removal and len(pcd.points) > config.outlier_nb_neighbors:
        before_outlier = int(len(pcd.points))
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(config.outlier_nb_neighbors),
            std_ratio=float(config.outlier_std_ratio),
        )
        stats["points_removed_by_outlier_removal"] = int(before_outlier - len(pcd.points))
    else:
        stats["points_removed_by_outlier_removal"] = 0
    stats["points_after_outlier_removal"] = int(len(pcd.points))

    if effective_voxel_size and effective_voxel_size > 0:
        before_voxel = int(len(pcd.points))
        pcd = pcd.voxel_down_sample(voxel_size=float(effective_voxel_size))
        stats["points_removed_by_voxel"] = int(before_voxel - len(pcd.points))
    else:
        stats["points_removed_by_voxel"] = 0
    stats["points_after_voxel"] = int(len(pcd.points))
    stats["overall_retention_ratio"] = float(len(pcd.points) / original_count) if original_count else 0.0
    stats["output_nearest_neighbor_stats"] = nearest_neighbor_stats(
        np.asarray(pcd.points, dtype=np.float32),
        int(config.detail_metrics_sample_size),
        int(config.random_seed),
    )

    return (
        np.asarray(pcd.points, dtype=np.float32),
        coerce_colors(np.asarray(pcd.colors), len(pcd.points)),
        stats,
    )


def bbox_extent(points: np.ndarray) -> list[float]:
    if points.size == 0:
        return [0.0, 0.0, 0.0]
    extent = points.max(axis=0) - points.min(axis=0)
    return [float(extent[0]), float(extent[1]), float(extent[2])]


def bbox_diagonal(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
