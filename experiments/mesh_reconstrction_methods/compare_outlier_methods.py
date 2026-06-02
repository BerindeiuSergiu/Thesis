"""Compare outlier-removal methods on raw point-cloud outputs.

This script loads raw points from Fast3R/DUSt3R outputs, applies multiple
outlier-removal strategies, and writes a human-readable TXT report.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = THIS_DIR.parent
BETA_ROOT = EXPERIMENTS_ROOT / "beta_pipeline_testing"
DEFAULT_OUTPUTS_ROOT = BETA_ROOT / "outputs"

ALL_METHODS = [
    "identity",
    "iqr_axis",
    "radius_percentile",
    "mad_axis",
    "zscore_radius",
    "mahalanobis",
    "voxel_min_points",
    "iqr_then_radius",
]


@dataclass
class MethodResult:
    method: str
    status: str
    message: str
    points_before: int
    points_after: int
    removed_points: int
    retention_ratio: float
    elapsed_seconds: float
    bbox_extent_x: float
    bbox_extent_y: float
    bbox_extent_z: float


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _default_paths_for_model(model: str) -> tuple[Path, Optional[Path]]:
    output_dir = DEFAULT_OUTPUTS_ROOT / model
    return output_dir / "raw_points.npy", output_dir / "raw_colors.npy"


def load_arrays(points_path: Optional[Path], colors_path: Optional[Path], npz_path: Optional[Path]) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    source_info: dict = {}
    if npz_path is not None:
        payload = np.load(npz_path)
        if "points" not in payload:
            raise KeyError(f"`points` key not found in npz: {npz_path}")
        points = payload["points"]
        colors = payload["colors"] if "colors" in payload else None
        source_info["points_source"] = str(npz_path)
        source_info["colors_source"] = str(npz_path) if colors is not None else ""
        source_info["input_mode"] = "npz"
        return points, colors, source_info

    if points_path is None:
        raise ValueError("Points path is required when --npz is not provided.")
    points = np.load(points_path)
    colors = np.load(colors_path) if (colors_path is not None and colors_path.exists()) else None
    source_info["points_source"] = str(points_path)
    source_info["colors_source"] = str(colors_path) if colors is not None and colors_path is not None else ""
    source_info["input_mode"] = "npy"
    return points, colors, source_info


def subsample(points: np.ndarray, colors: Optional[np.ndarray], max_points: Optional[int], seed: int) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if max_points is None or points.shape[0] <= max_points:
        return points, colors, None
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    points_sub = points[indices]
    colors_sub = colors[indices] if colors is not None and colors.shape[0] == points.shape[0] else None
    return points_sub, colors_sub, indices


def bbox_extent(points: np.ndarray) -> tuple[float, float, float]:
    if points.size == 0:
        return 0.0, 0.0, 0.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ext = maxs - mins
    return float(ext[0]), float(ext[1]), float(ext[2])


def method_identity(points: np.ndarray) -> np.ndarray:
    return np.ones(points.shape[0], dtype=bool)


def method_iqr_axis(points: np.ndarray, iqr_k: float = 1.5) -> np.ndarray:
    q1 = np.percentile(points, 25, axis=0)
    q3 = np.percentile(points, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    return np.all((points >= lower) & (points <= upper), axis=1)


def method_radius_percentile(points: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    center = np.median(points, axis=0)
    d = np.linalg.norm(points - center, axis=1)
    threshold = float(np.percentile(d, percentile))
    return d <= threshold


def method_mad_axis(points: np.ndarray, mad_k: float = 3.5) -> np.ndarray:
    median = np.median(points, axis=0)
    abs_dev = np.abs(points - median)
    mad = np.median(abs_dev, axis=0)
    scale = 1.4826 * mad + 1e-12
    robust_z = abs_dev / scale
    return np.all(robust_z <= mad_k, axis=1)


def method_zscore_radius(points: np.ndarray, z_k: float = 3.0) -> np.ndarray:
    center = points.mean(axis=0)
    d = np.linalg.norm(points - center, axis=1)
    mu = float(d.mean())
    sigma = float(d.std()) + 1e-12
    return d <= (mu + z_k * sigma)


def method_mahalanobis(points: np.ndarray, md2_threshold: float = 16.0) -> np.ndarray:
    center = np.median(points, axis=0)
    cov = np.cov(points, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    diff = points - center
    md2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return md2 <= md2_threshold


def method_voxel_min_points(points: np.ndarray, voxel_size: float = 0.004, min_points: int = 3) -> np.ndarray:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    vox = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse] >= min_points


def method_iqr_then_radius(points: np.ndarray) -> np.ndarray:
    first_mask = method_iqr_axis(points, iqr_k=1.5)
    reduced = points[first_mask]
    if reduced.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    second_mask_local = method_radius_percentile(reduced, percentile=99.0)
    reduced_indices = np.flatnonzero(first_mask)
    kept_indices = reduced_indices[second_mask_local]
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[kept_indices] = True
    return mask


def run_method(name: str, points: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> MethodResult:
    start = time.perf_counter()
    try:
        mask = fn(points)
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        if mask.shape[0] != points.shape[0]:
            raise ValueError(f"Method returned mask with invalid size: {mask.shape[0]}")
        filtered = points[mask]
        elapsed = time.perf_counter() - start
        ex, ey, ez = bbox_extent(filtered)
        before = int(points.shape[0])
        after = int(filtered.shape[0])
        removed = before - after
        ratio = float(after / max(before, 1))
        return MethodResult(
            method=name,
            status="ok",
            message="",
            points_before=before,
            points_after=after,
            removed_points=removed,
            retention_ratio=ratio,
            elapsed_seconds=float(elapsed),
            bbox_extent_x=ex,
            bbox_extent_y=ey,
            bbox_extent_z=ez,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return MethodResult(
            method=name,
            status="failed",
            message=str(exc),
            points_before=int(points.shape[0]),
            points_after=0,
            removed_points=int(points.shape[0]),
            retention_ratio=0.0,
            elapsed_seconds=float(elapsed),
            bbox_extent_x=0.0,
            bbox_extent_y=0.0,
            bbox_extent_z=0.0,
        )


def write_txt_report(path: Path, header: dict, results: list[MethodResult]) -> None:
    lines: list[str] = []
    lines.append("Outlier Removal Methods Report")
    lines.append("=" * 80)
    for key, value in header.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Methods")
    lines.append("-" * 80)
    for result in results:
        lines.append(f"method={result.method}")
        lines.append(f"  status={result.status}")
        if result.message:
            lines.append(f"  message={result.message}")
        lines.append(f"  points_before={result.points_before}")
        lines.append(f"  points_after={result.points_after}")
        lines.append(f"  removed_points={result.removed_points}")
        lines.append(f"  retention_ratio={result.retention_ratio:.6f}")
        lines.append(f"  elapsed_seconds={result.elapsed_seconds:.4f}")
        lines.append(
            "  bbox_extent_xyz="
            f"({result.bbox_extent_x:.6f}, {result.bbox_extent_y:.6f}, {result.bbox_extent_z:.6f})"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple outlier-removal methods and log results.")
    parser.add_argument("--model", choices=["fast3r", "dust3r"], default="fast3r")
    parser.add_argument("--points", type=Path, default=None, help="Override points .npy path.")
    parser.add_argument("--colors", type=Path, default=None, help="Override colors .npy path.")
    parser.add_argument("--npz", type=Path, default=None, help="Optional .npz containing points/colors.")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_METHODS),
        help=f"Comma-separated methods. Available: {', '.join(ALL_METHODS)}",
    )
    parser.add_argument("--max-points", type=int, default=None, help="Optional random subsample size for faster iteration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --max-points is set.")
    parser.add_argument(
        "--save-filtered",
        type=parse_bool,
        default=False,
        help="Save filtered point arrays as .npz files in the output directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=THIS_DIR / "results",
        help="Directory where report files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    default_points, default_colors = _default_paths_for_model(args.model)
    points_path = args.points if args.points is not None else default_points
    colors_path = args.colors if args.colors is not None else default_colors

    if args.npz is None and not points_path.exists():
        raise FileNotFoundError(f"Points file not found: {points_path}")
    if args.npz is not None and not args.npz.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.npz}")

    points_raw, colors_raw, source_info = load_arrays(points_path, colors_path, args.npz)
    if points_raw.ndim != 2 or points_raw.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3). Got: {points_raw.shape}")

    original_points_count = int(points_raw.shape[0])
    finite_mask = np.isfinite(points_raw).all(axis=1)
    points = points_raw[finite_mask]
    finite_points_count = int(points.shape[0])
    invalid_points_dropped = int(original_points_count - finite_points_count)

    if colors_raw is not None and colors_raw.ndim == 2 and colors_raw.shape[0] == finite_mask.shape[0]:
        colors = colors_raw[finite_mask]
    else:
        colors = None

    points, colors, sampled_indices = subsample(points, colors, args.max_points, args.seed)
    points_used_for_methods = int(points.shape[0])

    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = [name for name in methods if name not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {ALL_METHODS}")

    method_map: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "identity": method_identity,
        "iqr_axis": method_iqr_axis,
        "radius_percentile": method_radius_percentile,
        "mad_axis": method_mad_axis,
        "zscore_radius": method_zscore_radius,
        "mahalanobis": method_mahalanobis,
        "voxel_min_points": method_voxel_min_points,
        "iqr_then_radius": method_iqr_then_radius,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = args.out_dir / f"{stamp}_{args.model}_outlier_report.txt"
    json_path = args.out_dir / f"{stamp}_{args.model}_outlier_report.json"

    results: list[MethodResult] = []
    filtered_store: dict[str, dict[str, str]] = {}

    for method_name in methods:
        result = run_method(method_name, points, method_map[method_name])
        results.append(result)

        if args.save_filtered and result.status == "ok":
            mask = method_map[method_name](points)
            kept_points = points[mask]
            kept_colors = colors[mask] if colors is not None and colors.shape[0] == points.shape[0] else None
            save_path = args.out_dir / f"{stamp}_{args.model}_{method_name}_filtered.npz"
            if kept_colors is not None:
                np.savez_compressed(save_path, points=kept_points, colors=kept_colors)
            else:
                np.savez_compressed(save_path, points=kept_points)
            filtered_store[method_name] = {"npz": str(save_path)}

    header = {
        "timestamp": stamp,
        "model_hint": args.model,
        "dependency_profile": "numpy_only",
        **source_info,
        "raw_points_count_original": original_points_count,
        "finite_points_count": finite_points_count,
        "invalid_points_dropped": invalid_points_dropped,
        "points_used_for_methods": points_used_for_methods,
        "subsample_applied": bool(args.max_points is not None),
        "subsample_target_max_points": args.max_points if args.max_points is not None else "full",
        "subsample_size_actual": int(points.shape[0]) if args.max_points is not None else "full",
        "sample_seed": args.seed if args.max_points is not None else "",
        "methods": ", ".join(methods),
    }
    if sampled_indices is not None:
        header["sampled_from_original_count"] = int(sampled_indices.shape[0])

    write_txt_report(txt_path, header, results)

    payload = {
        "header": header,
        "results": [asdict(item) for item in results],
        "saved_filtered_outputs": filtered_store,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved TXT report: {txt_path}")
    print(f"Saved JSON report: {json_path}")
    if args.save_filtered:
        print(f"Saved filtered files: {len(filtered_store)}")


if __name__ == "__main__":
    main()
