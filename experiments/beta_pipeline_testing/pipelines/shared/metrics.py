"""CSV-based metrics collection for beta reconstruction pipelines."""

from __future__ import annotations

import csv
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch

from .paths import BETA_ROOT

try:
    from scipy.spatial import cKDTree

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def utc_timestamp() -> str:
    """Return a stable UTC timestamp string for logs and run IDs."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def numeric_summary(values: np.ndarray, prefix: str) -> dict:
    """Summarize a numeric vector with robust descriptive statistics."""
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }

    values = values.astype(np.float64, copy=False)
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_median": float(np.percentile(values, 50)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(values.max()),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _grayscale_entropy(gray_frame: np.ndarray) -> float:
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(hist.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def get_peak_gpu_memory_gb() -> float:
    """Return PyTorch peak GPU memory usage for the current process."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def reset_peak_gpu_memory_stats() -> None:
    """Reset PyTorch's per-process peak memory tracker."""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def collect_system_info() -> dict:
    """Collect lightweight runtime metadata for reproducibility."""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "cwd": str(Path.cwd()),
        "executable": sys.executable,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda_device_name": props.name,
                "cuda_total_memory_gb": float(props.total_memory / (1024**3)),
                "cuda_capability_major": int(props.major),
                "cuda_capability_minor": int(props.minor),
            }
        )
    else:
        info.update(
            {
                "cuda_device_name": "cpu",
                "cuda_total_memory_gb": 0.0,
                "cuda_capability_major": 0,
                "cuda_capability_minor": 0,
            }
        )

    return info


class MethodMetricsLogger:
    """Append-only CSV logger for method-specific reconstruction metrics."""

    def __init__(self, method_name: str, data_root: Optional[Path] = None, run_id: Optional[str] = None):
        self.method_name = method_name
        self.data_root = Path(data_root or (BETA_ROOT / f"{method_name}_data"))
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def _csv_path(self, category: str, filename: str) -> Path:
        path = self.data_root / category / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _prepare_row(self, row: dict) -> dict:
        prepared = {"run_id": self.run_id, "recorded_at_utc": utc_timestamp()}
        for key, value in row.items():
            if is_dataclass(value):
                prepared[key] = str(asdict(value))
            elif isinstance(value, Path):
                prepared[key] = str(value)
            elif isinstance(value, (np.floating, np.integer)):
                prepared[key] = value.item()
            elif isinstance(value, np.bool_):
                prepared[key] = bool(value)
            else:
                prepared[key] = value
        return prepared

    def append_row(self, category: str, filename: str, row: dict) -> Path:
        path = self._csv_path(category, filename)
        prepared = self._prepare_row(row)

        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prepared.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(prepared)
        return path

    def append_rows(self, category: str, filename: str, rows: Iterable[dict]) -> Optional[Path]:
        rows = list(rows)
        if not rows:
            return None
        path = self._csv_path(category, filename)
        prepared_rows = [self._prepare_row(row) for row in rows]

        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prepared_rows[0].keys()))
            if not exists:
                writer.writeheader()
            writer.writerows(prepared_rows)
        return path


def compute_frame_metrics(frames: list[np.ndarray], metadata: Optional[dict] = None) -> tuple[list[dict], list[dict]]:
    """Compute no-reference image quality and temporal overlap proxies."""
    metadata = metadata or {}
    fps = float(metadata.get("fps", 0.0))
    frame_indices = metadata.get("frame_indices", list(range(len(frames))))

    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    frame_rows: list[dict] = []
    pair_rows: list[dict] = []
    gray_frames: list[np.ndarray] = []
    descriptors: list[Optional[np.ndarray]] = []
    keypoint_counts: list[int] = []

    for local_idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_mean = float(gray.mean())
        contrast_std = float(gray.std())
        entropy = _grayscale_entropy(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        keypoints, desc = orb.detectAndCompute(gray, None)
        descriptors.append(desc)
        keypoint_count = len(keypoints)
        keypoint_counts.append(keypoint_count)

        frame_index = int(frame_indices[local_idx]) if local_idx < len(frame_indices) else local_idx
        frame_rows.append(
            {
                "frame_local_index": local_idx,
                "source_frame_index": frame_index,
                "timestamp_seconds": _safe_divide(frame_index, fps),
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
                "brightness_mean": brightness_mean,
                "brightness_std": contrast_std,
                "entropy_bits": entropy,
                "laplacian_variance": laplacian_var,
                "edge_density": edge_density,
                "orb_keypoints": keypoint_count,
            }
        )

    for idx in range(max(0, len(frames) - 1)):
        gray_a = gray_frames[idx]
        gray_b = gray_frames[idx + 1]
        diff = cv2.absdiff(gray_a, gray_b)
        mean_abs_diff = float(diff.mean())
        normalized_abs_diff = _safe_divide(mean_abs_diff, 255.0)

        desc_a = descriptors[idx]
        desc_b = descriptors[idx + 1]
        matches = []
        if desc_a is not None and desc_b is not None and len(desc_a) > 0 and len(desc_b) > 0:
            matches = matcher.match(desc_a, desc_b)

        match_distances = np.array([match.distance for match in matches], dtype=np.float64)
        pair_rows.append(
            {
                "frame_local_index_a": idx,
                "frame_local_index_b": idx + 1,
                "source_frame_index_a": int(frame_indices[idx]) if idx < len(frame_indices) else idx,
                "source_frame_index_b": int(frame_indices[idx + 1]) if (idx + 1) < len(frame_indices) else idx + 1,
                "time_gap_seconds": _safe_divide(
                    abs(
                        (frame_indices[idx + 1] if (idx + 1) < len(frame_indices) else idx + 1)
                        - (frame_indices[idx] if idx < len(frame_indices) else idx)
                    ),
                    fps,
                ),
                "mean_abs_diff": mean_abs_diff,
                "normalized_abs_diff": normalized_abs_diff,
                "orb_matches": len(matches),
                "orb_match_ratio_to_keypoints": _safe_divide(
                    len(matches),
                    max(1, min(keypoint_counts[idx], keypoint_counts[idx + 1])),
                ),
                **numeric_summary(match_distances, "match_distance"),
            }
        )

    return frame_rows, pair_rows


def compute_pair_graph_metrics(pairs: list[tuple[dict, dict]]) -> list[dict]:
    """Extract DUSt3R pair-graph structure for analysis."""
    rows: list[dict] = []
    for pair_idx, (img_a, img_b) in enumerate(pairs):
        idx_a = int(img_a["idx"])
        idx_b = int(img_b["idx"])
        rows.append(
            {
                "pair_index": pair_idx,
                "frame_idx_a": idx_a,
                "frame_idx_b": idx_b,
                "frame_gap": abs(idx_a - idx_b),
                "direction": f"{idx_a}->{idx_b}",
                "is_self_pair": idx_a == idx_b,
            }
        )
    return rows


def compute_depth_confidence_metrics(
    pts3d: list,
    confidence: list,
    conf_threshold: float,
) -> list[dict]:
    """Summarize per-frame depth and confidence maps from DUSt3R."""
    rows: list[dict] = []
    for frame_idx, (pts, conf) in enumerate(zip(pts3d, confidence)):
        pts_np = pts.detach().cpu().numpy()
        conf_np = conf.detach().cpu().numpy()
        depth = pts_np[:, :, 2].astype(np.float64)

        valid_depth_mask = np.isfinite(depth) & (depth > 0)
        confident_mask = np.isfinite(conf_np) & (conf_np > conf_threshold)
        valid_depth = depth[valid_depth_mask]
        valid_conf = conf_np[np.isfinite(conf_np)]

        row = {
            "frame_local_index": frame_idx,
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "valid_depth_ratio": float(valid_depth_mask.mean()),
            "confidence_ratio_above_threshold": float(confident_mask.mean()),
            "points_above_conf_threshold": int(confident_mask.sum()),
            "confidence_threshold": float(conf_threshold),
        }
        row.update(numeric_summary(valid_depth, "depth"))
        row.update(numeric_summary(valid_conf, "confidence"))
        rows.append(row)

    return rows


def compute_pose_metrics(
    camera_poses: np.ndarray,
    focals: Optional[np.ndarray] = None,
    principal_points: Optional[np.ndarray] = None,
) -> tuple[list[dict], list[dict]]:
    """Summarize estimated camera trajectory and inter-frame motion."""
    pose_rows: list[dict] = []
    step_rows: list[dict] = []
    cumulative_translation = 0.0

    for idx, pose in enumerate(camera_poses):
        translation = pose[:3, 3]
        focal = focals[idx] if focals is not None and idx < len(focals) else np.nan
        pp = principal_points[idx] if principal_points is not None and idx < len(principal_points) else np.array([np.nan, np.nan])
        pose_rows.append(
            {
                "frame_local_index": idx,
                "tx": float(translation[0]),
                "ty": float(translation[1]),
                "tz": float(translation[2]),
                "translation_norm": float(np.linalg.norm(translation)),
                "focal_x": float(focal[0]) if np.ndim(focal) > 0 else float(focal),
                "focal_y": float(focal[-1]) if np.ndim(focal) > 0 else float(focal),
                "principal_point_x": float(pp[0]),
                "principal_point_y": float(pp[1]),
            }
        )

        if idx == 0:
            continue

        prev_pose = camera_poses[idx - 1]
        relative_rotation = pose[:3, :3] @ prev_pose[:3, :3].T
        trace_value = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_step_deg = float(np.degrees(np.arccos(trace_value)))
        translation_step = float(np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3]))
        cumulative_translation += translation_step

        step_rows.append(
            {
                "frame_local_index_a": idx - 1,
                "frame_local_index_b": idx,
                "translation_step": translation_step,
                "rotation_step_deg": rotation_step_deg,
                "cumulative_translation": cumulative_translation,
            }
        )

    return pose_rows, step_rows


def compute_pointcloud_geometry_metrics(points: np.ndarray, colors: Optional[np.ndarray] = None) -> dict:
    """Summarize reconstructed point cloud geometry and color spread."""
    if points.size == 0:
        return {
            "num_points": 0,
            "bbox_min_x": 0.0,
            "bbox_min_y": 0.0,
            "bbox_min_z": 0.0,
            "bbox_max_x": 0.0,
            "bbox_max_y": 0.0,
            "bbox_max_z": 0.0,
            "bbox_extent_x": 0.0,
            "bbox_extent_y": 0.0,
            "bbox_extent_z": 0.0,
            "bbox_volume": 0.0,
            "point_density_bbox": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "centroid_z": 0.0,
            "mean_radius_from_centroid": 0.0,
            "color_mean_r": 0.0,
            "color_mean_g": 0.0,
            "color_mean_b": 0.0,
            "color_std_r": 0.0,
            "color_std_g": 0.0,
            "color_std_b": 0.0,
        }

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    bbox_volume = float(np.prod(extent)) if np.all(extent > 0) else 0.0
    centroid = points.mean(axis=0)
    radius = np.linalg.norm(points - centroid, axis=1)

    metrics = {
        "num_points": int(points.shape[0]),
        "bbox_min_x": float(bbox_min[0]),
        "bbox_min_y": float(bbox_min[1]),
        "bbox_min_z": float(bbox_min[2]),
        "bbox_max_x": float(bbox_max[0]),
        "bbox_max_y": float(bbox_max[1]),
        "bbox_max_z": float(bbox_max[2]),
        "bbox_extent_x": float(extent[0]),
        "bbox_extent_y": float(extent[1]),
        "bbox_extent_z": float(extent[2]),
        "bbox_volume": bbox_volume,
        "point_density_bbox": _safe_divide(points.shape[0], bbox_volume),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "mean_radius_from_centroid": float(radius.mean()),
    }
    metrics.update(numeric_summary(radius, "radius"))

    if colors is not None and colors.size > 0:
        colors = np.clip(colors, 0.0, 1.0)
        metrics.update(
            {
                "color_mean_r": float(colors[:, 0].mean()),
                "color_mean_g": float(colors[:, 1].mean()),
                "color_mean_b": float(colors[:, 2].mean()),
                "color_std_r": float(colors[:, 0].std()),
                "color_std_g": float(colors[:, 1].std()),
                "color_std_b": float(colors[:, 2].std()),
            }
        )
    else:
        metrics.update(
            {
                "color_mean_r": 0.0,
                "color_mean_g": 0.0,
                "color_mean_b": 0.0,
                "color_std_r": 0.0,
                "color_std_g": 0.0,
                "color_std_b": 0.0,
            }
        )

    return metrics


def compute_pointcloud_nearest_neighbor_metrics(points: np.ndarray, max_sample_points: int = 20000) -> dict:
    """Approximate point density using nearest-neighbor distances."""
    if not SCIPY_AVAILABLE or points.size == 0:
        return {
            "nn_available": bool(SCIPY_AVAILABLE),
            "nn_sample_size": 0,
            "nn_mean": 0.0,
            "nn_std": 0.0,
            "nn_min": 0.0,
            "nn_p25": 0.0,
            "nn_median": 0.0,
            "nn_p75": 0.0,
            "nn_p95": 0.0,
            "nn_max": 0.0,
        }

    if len(points) > max_sample_points:
        indices = np.linspace(0, len(points) - 1, max_sample_points, dtype=int)
        sampled = points[indices]
    else:
        sampled = points

    tree = cKDTree(sampled)
    distances, _ = tree.query(sampled, k=2)
    nn = distances[:, 1]
    metrics = {"nn_available": True, "nn_sample_size": int(len(sampled))}
    metrics.update(numeric_summary(nn, "nn"))
    return metrics


def compute_mesh_metrics(mesh) -> dict:
    """Summarize mesh topology and triangle quality."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if vertices.size == 0 or triangles.size == 0:
        return {
            "num_vertices": int(len(vertices)),
            "num_triangles": int(len(triangles)),
            "surface_area": 0.0,
            "bbox_extent_x": 0.0,
            "bbox_extent_y": 0.0,
            "bbox_extent_z": 0.0,
            "num_connected_components": 0,
            "largest_component_ratio": 0.0,
            "is_watertight": False,
            "is_edge_manifold": False,
            "is_vertex_manifold": False,
        }

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()

    tri_vertices = vertices[triangles]
    edge_lengths = np.stack(
        [
            np.linalg.norm(tri_vertices[:, 0] - tri_vertices[:, 1], axis=1),
            np.linalg.norm(tri_vertices[:, 1] - tri_vertices[:, 2], axis=1),
            np.linalg.norm(tri_vertices[:, 2] - tri_vertices[:, 0], axis=1),
        ],
        axis=1,
    )
    edge_lengths_flat = edge_lengths.reshape(-1)
    min_edges = np.maximum(edge_lengths.min(axis=1), 1e-8)
    max_edges = edge_lengths.max(axis=1)
    aspect_ratios = max_edges / min_edges
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]),
        axis=1,
    )

    metrics = {
        "num_vertices": int(len(vertices)),
        "num_triangles": int(len(triangles)),
        "surface_area": float(mesh.get_surface_area()) if hasattr(mesh, "get_surface_area") else float(triangle_areas.sum()),
        "bbox_extent_x": float(extent[0]),
        "bbox_extent_y": float(extent[1]),
        "bbox_extent_z": float(extent[2]),
        "is_watertight": bool(mesh.is_watertight()) if hasattr(mesh, "is_watertight") else False,
        "is_edge_manifold": bool(mesh.is_edge_manifold()) if hasattr(mesh, "is_edge_manifold") else False,
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()) if hasattr(mesh, "is_vertex_manifold") else False,
        "is_self_intersecting": bool(mesh.is_self_intersecting()) if hasattr(mesh, "is_self_intersecting") else False,
    }

    try:
        _, cluster_triangle_counts, _ = mesh.cluster_connected_triangles()
        cluster_triangle_counts = np.asarray(cluster_triangle_counts, dtype=np.int64)
        metrics["num_connected_components"] = int(cluster_triangle_counts.size)
        metrics["largest_component_ratio"] = _safe_divide(cluster_triangle_counts.max(), len(triangles))
    except Exception:
        metrics["num_connected_components"] = 0
        metrics["largest_component_ratio"] = 0.0

    metrics.update(numeric_summary(edge_lengths_flat, "edge_length"))
    metrics.update(numeric_summary(triangle_areas, "triangle_area"))
    metrics.update(numeric_summary(aspect_ratios, "triangle_aspect_ratio"))
    return metrics
