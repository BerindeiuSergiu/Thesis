"""Log available statistics from raw point-cloud outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = THIS_DIR.parent
BETA_ROOT = EXPERIMENTS_ROOT / "beta_pipeline_testing"
DEFAULT_OUTPUTS_ROOT = BETA_ROOT / "outputs"


def _default_paths_for_model(model: str) -> tuple[Path, Optional[Path]]:
    output_dir = DEFAULT_OUTPUTS_ROOT / model
    return output_dir / "raw_points.npy", output_dir / "raw_colors.npy"


def _as_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def summarize_points(points: np.ndarray) -> dict:
    summary: dict = {
        "shape": list(points.shape),
        "dtype": str(points.dtype),
        "memory_mb": float(points.nbytes / (1024**2)),
    }

    if points.ndim != 2 or points.shape[1] != 3:
        summary["warning"] = "Expected points array of shape (N, 3)."
        return summary

    finite_mask = np.isfinite(points).all(axis=1)
    valid = points[finite_mask]
    invalid_count = int(points.shape[0] - valid.shape[0])

    summary["num_points_total"] = int(points.shape[0])
    summary["num_points_valid"] = int(valid.shape[0])
    summary["num_points_invalid"] = invalid_count
    summary["valid_ratio"] = float(valid.shape[0] / max(points.shape[0], 1))

    if valid.size == 0:
        return summary

    mins = valid.min(axis=0)
    maxs = valid.max(axis=0)
    extent = maxs - mins
    centroid = valid.mean(axis=0)
    std = valid.std(axis=0)
    radii = np.linalg.norm(valid - centroid, axis=1)

    summary.update(
        {
            "bbox_min_xyz": mins.tolist(),
            "bbox_max_xyz": maxs.tolist(),
            "bbox_extent_xyz": extent.tolist(),
            "centroid_xyz": centroid.tolist(),
            "std_xyz": std.tolist(),
            "radius_mean": float(radii.mean()),
            "radius_std": float(radii.std()),
            "radius_p50": float(np.percentile(radii, 50)),
            "radius_p95": float(np.percentile(radii, 95)),
            "radius_max": float(radii.max()),
        }
    )
    return summary


def summarize_colors(colors: np.ndarray, expected_rows: int) -> dict:
    summary: dict = {
        "shape": list(colors.shape),
        "dtype": str(colors.dtype),
        "memory_mb": float(colors.nbytes / (1024**2)),
    }

    if colors.ndim != 2 or colors.shape[1] != 3:
        summary["warning"] = "Expected colors array of shape (N, 3)."
        return summary

    finite_mask = np.isfinite(colors).all(axis=1)
    valid = colors[finite_mask]

    summary["num_colors_total"] = int(colors.shape[0])
    summary["num_colors_valid"] = int(valid.shape[0])
    summary["num_colors_invalid"] = int(colors.shape[0] - valid.shape[0])
    summary["row_count_matches_points"] = bool(colors.shape[0] == expected_rows)

    if valid.size == 0:
        return summary

    summary.update(
        {
            "mean_rgb": valid.mean(axis=0).tolist(),
            "std_rgb": valid.std(axis=0).tolist(),
            "min_rgb": valid.min(axis=0).tolist(),
            "max_rgb": valid.max(axis=0).tolist(),
        }
    )
    return summary


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
    source_info["colors_source"] = str(colors_path) if colors_path is not None and colors is not None else ""
    source_info["input_mode"] = "npy"
    return points, colors, source_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log basic statistics from raw point-cloud arrays.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["fast3r", "dust3r"],
        default="fast3r",
        help="Use default paths from beta_pipeline_testing/outputs/<model>/raw_points.npy.",
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=None,
        help="Override path to points .npy file.",
    )
    parser.add_argument(
        "--colors",
        type=Path,
        default=None,
        help="Override path to colors .npy file.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Optional .npz path containing keys: points and optionally colors.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for JSON summary.",
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

    points, colors, source_info = load_arrays(points_path, colors_path, args.npz)

    report = {
        "model_hint": args.model,
        **source_info,
        "points": summarize_points(points),
    }

    if colors is not None:
        report["colors"] = summarize_colors(colors, expected_rows=points.shape[0] if points.ndim >= 1 else 0)
    else:
        report["colors"] = {"available": False}

    print("Raw point data summary")
    print(json.dumps(report, indent=2, default=_as_python))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, default=_as_python), encoding="utf-8")
        print(f"\nSaved JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
