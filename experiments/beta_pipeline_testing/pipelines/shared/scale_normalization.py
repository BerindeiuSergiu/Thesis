"""Scale normalization stage for real-world reconstruction units."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .reconstruction_data import ReconstructionOutput


@dataclass
class ReferenceMeasurement:
    """Known real-world measurement used when metadata cannot resolve scale."""

    name: str = "room_height"
    real_world_dimension: float = 2.5
    axis: str = "z"
    measured_dimension: Optional[float] = None
    point_a: Optional[tuple[float, float, float]] = None
    point_b: Optional[tuple[float, float, float]] = None
    percentile_low: float = 2.0
    percentile_high: float = 98.0


@dataclass
class ScaleNormalizationConfig:
    """Configuration for the mandatory scale normalization stage."""

    enabled: bool = True
    prefer_metadata: bool = True
    reference: ReferenceMeasurement = field(default_factory=ReferenceMeasurement)
    fail_without_scale: bool = False
    report_filename: str = "scale_normalization_report.json"


def _numeric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if np.isfinite(value) and value > 0:
            return value
    except (TypeError, ValueError):
        return None
    return None


def _metadata_scale(metadata: dict[str, Any]) -> tuple[Optional[float], dict[str, Any]]:
    """Infer scene-units to meters from explicit model/camera metadata."""

    candidates = [
        "scale_to_meters",
        "meters_per_unit",
        "absolute_scale",
        "scene_to_meters",
    ]
    for key in candidates:
        value = _numeric(metadata.get(key))
        if value is not None:
            return value, {"source": "metadata", "field": key, "scale_to_meters": value}

    baseline_meters = _numeric(metadata.get("baseline_meters"))
    baseline_scene_units = _numeric(metadata.get("baseline_scene_units"))
    if baseline_meters is not None and baseline_scene_units is not None:
        scale = baseline_meters / baseline_scene_units
        return scale, {
            "source": "metadata",
            "field": "baseline_meters/baseline_scene_units",
            "baseline_meters": baseline_meters,
            "baseline_scene_units": baseline_scene_units,
            "scale_to_meters": scale,
        }

    if baseline_meters is not None:
        poses = metadata.get("camera_poses")
        if poses is not None:
            pose_arr = np.asarray(poses, dtype=np.float64)
            if pose_arr.ndim == 3 and pose_arr.shape[1:] == (4, 4) and pose_arr.shape[0] > 1:
                centers = pose_arr[:, :3, 3]
                scene_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
                scene_steps = scene_steps[np.isfinite(scene_steps) & (scene_steps > 0)]
                if scene_steps.size:
                    baseline_scene_units = float(np.median(scene_steps))
                    scale = baseline_meters / baseline_scene_units
                    return scale, {
                        "source": "metadata",
                        "field": "baseline_meters/camera_pose_translation_steps",
                        "baseline_meters": baseline_meters,
                        "baseline_scene_units": baseline_scene_units,
                        "num_pose_steps": int(scene_steps.size),
                        "scale_to_meters": scale,
                    }

    real_baselines = metadata.get("baseline_distances_meters")
    scene_baselines = metadata.get("baseline_distances_scene_units")
    if real_baselines is not None and scene_baselines is not None:
        real = np.asarray(real_baselines, dtype=np.float64)
        scene = np.asarray(scene_baselines, dtype=np.float64)
        valid = np.isfinite(real) & np.isfinite(scene) & (real > 0) & (scene > 0)
        if np.any(valid):
            ratios = real[valid] / scene[valid]
            scale = float(np.median(ratios))
            return scale, {
                "source": "metadata",
                "field": "baseline_distances_meters/baseline_distances_scene_units",
                "num_baselines": int(np.count_nonzero(valid)),
                "scale_to_meters": scale,
            }

    failure_report = {
        "source": "metadata",
        "reason": "No explicit absolute scale, meter/unit field, or paired baseline distances were present.",
    }
    poses = metadata.get("camera_poses")
    if poses is not None:
        pose_arr = np.asarray(poses, dtype=np.float64)
        if pose_arr.ndim == 3 and pose_arr.shape[1:] == (4, 4) and pose_arr.shape[0] > 1:
            centers = pose_arr[:, :3, 3]
            scene_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            scene_steps = scene_steps[np.isfinite(scene_steps) & (scene_steps > 0)]
            if scene_steps.size:
                failure_report.update(
                    {
                        "camera_pose_steps_available": True,
                        "camera_pose_step_median_scene_units": float(np.median(scene_steps)),
                        "camera_pose_step_count": int(scene_steps.size),
                    }
                )
    if metadata.get("focals") is not None or metadata.get("estimated_focals") is not None:
        failure_report["focal_metadata_available"] = True
        failure_report["focal_metadata_note"] = (
            "Focal estimates constrain projective geometry, but no physical sensor/object reference "
            "was available to convert scene units to meters."
        )
    return None, failure_report


def _axis_index(axis: str) -> int:
    axis = axis.lower()
    if axis in {"x", "0"}:
        return 0
    if axis in {"y", "1"}:
        return 1
    if axis in {"z", "2", "height"}:
        return 2
    raise ValueError(f"Unsupported reference axis: {axis}")


def _reference_dimension(points: np.ndarray, reference: ReferenceMeasurement) -> tuple[float, dict[str, Any]]:
    if reference.measured_dimension is not None:
        measured = _numeric(reference.measured_dimension)
        if measured is None:
            raise ValueError("reference.measured_dimension must be a positive finite number.")
        return measured, {"method": "configured_measured_dimension"}

    if reference.point_a is not None and reference.point_b is not None:
        point_a = np.asarray(reference.point_a, dtype=np.float64)
        point_b = np.asarray(reference.point_b, dtype=np.float64)
        measured = float(np.linalg.norm(point_a - point_b))
        if measured <= 0 or not np.isfinite(measured):
            raise ValueError("Reference point distance must be positive and finite.")
        return measured, {"method": "configured_point_distance"}

    if points.size == 0:
        raise ValueError("Cannot estimate a reference dimension from an empty point cloud.")

    axis = _axis_index(reference.axis)
    low = float(np.percentile(points[:, axis], reference.percentile_low))
    high = float(np.percentile(points[:, axis], reference.percentile_high))
    measured = high - low
    if measured <= 0 or not np.isfinite(measured):
        raise ValueError("Estimated reference dimension must be positive and finite.")
    return measured, {
        "method": "axis_percentile_extent",
        "axis": reference.axis,
        "percentile_low": float(reference.percentile_low),
        "percentile_high": float(reference.percentile_high),
        "low_value": low,
        "high_value": high,
    }


def _fallback_scale(points: np.ndarray, reference: ReferenceMeasurement) -> tuple[float, dict[str, Any]]:
    measured, measurement_info = _reference_dimension(points, reference)
    real = _numeric(reference.real_world_dimension)
    if real is None:
        raise ValueError("reference.real_world_dimension must be a positive finite number.")
    scale = real / measured
    return scale, {
        "source": "geometric_reference",
        "reference_name": reference.name,
        "measured_dimension_in_reconstruction_units": measured,
        "real_world_dimension_meters": real,
        "scale_formula": "real_world_dimension_meters / measured_dimension_in_reconstruction_units",
        "scale_to_meters": scale,
        **measurement_info,
    }


def _scale_poses(poses: Optional[list[np.ndarray]], scale: float) -> Optional[list[np.ndarray]]:
    if poses is None:
        return None
    scaled: list[np.ndarray] = []
    for pose in poses:
        pose_arr = np.array(pose, dtype=np.float32, copy=True)
        if pose_arr.shape == (4, 4):
            pose_arr[:3, 3] *= scale
        scaled.append(pose_arr)
    return scaled


def scale_normalization(
    reconstruction: ReconstructionOutput,
    config: ScaleNormalizationConfig,
    output_dir: Optional[Path] = None,
) -> tuple[ReconstructionOutput, dict[str, Any]]:
    """Apply one global isotropic scale before any output branches run."""

    if not config.enabled:
        report = {
            "enabled": False,
            "scale_to_meters": 1.0,
            "source": "disabled",
            "applied_once_globally": True,
            "isotropic": True,
        }
        return reconstruction.copy(), report

    points = np.asarray(reconstruction.points, dtype=np.float32)
    scale = None
    metadata_report: dict[str, Any] = {}
    if config.prefer_metadata:
        scale, metadata_report = _metadata_scale(reconstruction.metadata)

    if scale is None:
        try:
            scale, scale_report = _fallback_scale(points, config.reference)
        except Exception as exc:
            if config.fail_without_scale:
                raise
            scale = 1.0
            scale_report = {
                "source": "identity_unscaled",
                "reason": str(exc),
                "scale_to_meters": 1.0,
            }
    else:
        scale_report = metadata_report

    scaled = reconstruction.copy()
    scaled.points = points * float(scale)
    scaled.poses = _scale_poses(reconstruction.poses, float(scale))
    scaled.scale_applied = float(reconstruction.scale_applied) * float(scale)
    scaled.metadata["scale_normalization"] = dict(scale_report)

    report = {
        **scale_report,
        "enabled": True,
        "scale_to_meters": float(scale),
        "cumulative_scale_applied": float(scaled.scale_applied),
        "applied_once_globally": True,
        "isotropic": True,
        "point_count": int(points.shape[0]),
        "metadata_attempt": metadata_report,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / config.report_filename).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return scaled, report
