from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _positive_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric) and numeric > 0:
        return numeric
    return None


def _axis_index(axis: str) -> int:
    axis = str(axis).lower()
    if axis in {"x", "0"}:
        return 0
    if axis in {"y", "1"}:
        return 1
    if axis in {"z", "2", "height"}:
        return 2
    raise ValueError(f"Unsupported scale reference axis: {axis}")


def _metadata_scale(metadata: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    for key in ("scale_to_meters", "meters_per_unit", "absolute_scale", "scene_to_meters"):
        scale = _positive_float(metadata.get(key))
        if scale is not None:
            return scale, {"source": "metadata", "field": key, "scale_to_meters": scale}

    baseline_meters = _positive_float(metadata.get("baseline_meters"))
    baseline_scene_units = _positive_float(metadata.get("baseline_scene_units"))
    if baseline_meters is not None and baseline_scene_units is not None:
        scale = baseline_meters / baseline_scene_units
        return scale, {
            "source": "metadata",
            "field": "baseline_meters/baseline_scene_units",
            "baseline_meters": baseline_meters,
            "baseline_scene_units": baseline_scene_units,
            "scale_to_meters": scale,
        }

    if baseline_meters is not None and metadata.get("camera_poses") is not None:
        poses = np.asarray(metadata["camera_poses"], dtype=np.float64)
        if poses.ndim == 3 and poses.shape[1:] == (4, 4) and poses.shape[0] > 1:
            centers = poses[:, :3, 3]
            steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            steps = steps[np.isfinite(steps) & (steps > 0)]
            if steps.size:
                baseline_scene_units = float(np.median(steps))
                scale = baseline_meters / baseline_scene_units
                return scale, {
                    "source": "metadata",
                    "field": "baseline_meters/camera_pose_translation_steps",
                    "baseline_meters": baseline_meters,
                    "baseline_scene_units": baseline_scene_units,
                    "scale_to_meters": scale,
                    "num_pose_steps": int(steps.size),
                }

    return None, {
        "source": "metadata",
        "reason": "No explicit meter/unit field or paired baseline metadata was available.",
    }


def _geometric_scale(points: np.ndarray, config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float32)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.size == 0:
        raise ValueError("No finite points were available for scale estimation.")

    configured_measurement = _positive_float(config.get("scale_reference_measured"))
    if configured_measurement is not None:
        measured = configured_measurement
        measurement_report = {"method": "configured_measured_dimension"}
    else:
        axis = _axis_index(str(config.get("scale_reference_axis", "z")))
        low_pct = float(config.get("scale_percentile_low", 2.0))
        high_pct = float(config.get("scale_percentile_high", 98.0))
        low = float(np.percentile(points[:, axis], low_pct))
        high = float(np.percentile(points[:, axis], high_pct))
        measured = high - low
        measurement_report = {
            "method": "axis_percentile_extent",
            "axis": str(config.get("scale_reference_axis", "z")),
            "percentile_low": low_pct,
            "percentile_high": high_pct,
            "low_value": low,
            "high_value": high,
        }

    real_world_dimension = _positive_float(config.get("scale_reference_real", 2.5))
    if real_world_dimension is None:
        raise ValueError("scale_reference_real must be a positive finite value.")
    if measured <= 0 or not np.isfinite(measured):
        raise ValueError("Measured scale reference must be positive and finite.")

    scale = real_world_dimension / measured
    return scale, {
        "source": "geometric_reference",
        "reference_name": str(config.get("scale_reference_name", "room_height")),
        "measured_dimension_in_reconstruction_units": float(measured),
        "real_world_dimension_meters": float(real_world_dimension),
        "scale_formula": "real_world_dimension_meters / measured_dimension_in_reconstruction_units",
        "scale_to_meters": float(scale),
        **measurement_report,
    }


def _scale_poses(poses: list[np.ndarray], scale: float) -> list[np.ndarray]:
    scaled_poses: list[np.ndarray] = []
    for pose in poses:
        pose_arr = np.array(pose, dtype=np.float32, copy=True)
        if pose_arr.shape == (4, 4):
            pose_arr[:3, 3] *= float(scale)
        scaled_poses.append(pose_arr)
    return scaled_poses


def scale_normalization(
    points: np.ndarray,
    poses: list[np.ndarray],
    metadata: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Apply one global isotropic scale before point cloud output branches."""

    points = np.asarray(points, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bool(config.get("scale_enabled", True)):
        report = {
            "enabled": False,
            "source": "disabled",
            "scale_to_meters": 1.0,
            "applied_once_globally": True,
            "isotropic": True,
            "point_count": int(points.shape[0]),
        }
        (output_dir / "scale_normalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return points, poses, report

    scale, metadata_attempt = _metadata_scale(metadata)
    if scale is None:
        try:
            scale, scale_report = _geometric_scale(points, config)
        except Exception as exc:
            if bool(config.get("fail_without_scale", False)):
                raise
            scale = 1.0
            scale_report = {
                "source": "identity_unscaled",
                "reason": str(exc),
                "scale_to_meters": 1.0,
            }
    else:
        scale_report = metadata_attempt

    scaled_points = points * float(scale)
    scaled_poses = _scale_poses(poses, float(scale))
    report = {
        **scale_report,
        "enabled": True,
        "scale_to_meters": float(scale),
        "applied_once_globally": True,
        "isotropic": True,
        "point_count": int(points.shape[0]),
        "metadata_attempt": metadata_attempt,
    }
    (output_dir / "scale_normalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return scaled_points, scaled_poses, report
