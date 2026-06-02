"""Reprocess saved Fast3R arrays from a completed run, then rerun Gaussian export."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from pipelines.shared.gaussian_splats import GaussianSplatConfig, gaussian_splat_branch
from pipelines.shared.pointcloud_ops import PointCloudProcessingConfig, process_pointcloud
from pipelines.shared.reconstruction_data import ReconstructionOutput


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean saved reconstruction arrays without rerunning Fast3R.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--view-conf-p50-min", type=float, default=1.2)
    parser.add_argument("--view-conf-p90-min", type=float, default=1.45)
    parser.add_argument("--point-conf-min", type=float, default=1.05)
    parser.add_argument("--radius-percentile", type=float, default=99.7)
    parser.add_argument("--voxel-size", type=float, default=0.001)
    parser.add_argument("--max-abs-coordinate", type=float, default=100.0)
    parser.add_argument("--high-detail-mode", type=_parse_bool, default=False)
    parser.add_argument("--gaussian-nn-scale-mult", type=float, default=0.35)
    parser.add_argument("--gaussian-max-scale", type=float, default=0.008)
    parser.add_argument("--gaussian-scale-percentile-high", type=float, default=85.0)
    parser.add_argument("--gaussian-surface-aligned", type=_parse_bool, default=False)
    return parser.parse_args()


def _kept_views_from_report(report_path: Path, p50_min: float, p90_min: float) -> tuple[set[int], list[dict]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    kept: set[int] = set()
    rows = []
    for row in report.get("per_view_confidence_stats", []):
        keep = (
            float(row.get("confidence_p50", 0.0)) >= float(p50_min)
            and float(row.get("confidence_p90", 0.0)) >= float(p90_min)
        )
        rows.append(
            {
                "view_idx": int(row.get("view_idx", -1)),
                "keep": bool(keep),
                "confidence_p50": float(row.get("confidence_p50", 0.0)),
                "confidence_p90": float(row.get("confidence_p90", 0.0)),
                "points_after_filter": int(row.get("points_after_filter", 0)),
            }
        )
        if keep:
            kept.add(int(row.get("view_idx", -1)))
    return kept, rows


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output or run_dir.with_name(run_dir.name + "_recleaned")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points = np.load(run_dir / "scaled_points.npy", mmap_mode="r")
    colors = np.load(run_dir / "raw_colors.npy", mmap_mode="r")
    confidences = np.load(run_dir / "raw_confidences.npy", mmap_mode="r")
    source_view_ids = np.load(run_dir / "source_view_ids.npy", mmap_mode="r")

    kept_views, view_rows = _kept_views_from_report(
        run_dir / "confidence_export_report.json",
        p50_min=float(args.view_conf_p50_min),
        p90_min=float(args.view_conf_p90_min),
    )
    kept_view_array = np.array(sorted(kept_views), dtype=np.int32)
    mask = np.isin(source_view_ids, kept_view_array) & (confidences >= float(args.point_conf_min))

    print(f"Input scaled points: {points.shape[0]:,}")
    print(f"Kept views: {len(kept_views):,} / {len(view_rows):,}")
    print(f"Points after view/conf pruning: {int(np.count_nonzero(mask)):,}")

    filtered_points = np.asarray(points[mask], dtype=np.float32)
    filtered_colors = np.asarray(colors[mask], dtype=np.float32)
    processed_points, processed_colors, process_report = process_pointcloud(
        filtered_points,
        filtered_colors,
        PointCloudProcessingConfig(
            high_detail_mode=bool(args.high_detail_mode),
            max_abs_coordinate=float(args.max_abs_coordinate),
            radius_percentile=float(args.radius_percentile),
            statistical_outlier_removal=False,
            voxel_size=float(args.voxel_size),
        ),
    )
    np.save(output_dir / "processed_scaled_points.npy", processed_points)
    np.save(output_dir / "processed_scaled_colors.npy", processed_colors)
    shutil.copy2(run_dir / "confidence_export_report.json", output_dir / "source_confidence_export_report.json")

    report = {
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "input_points": int(points.shape[0]),
        "view_conf_p50_min": float(args.view_conf_p50_min),
        "view_conf_p90_min": float(args.view_conf_p90_min),
        "point_conf_min": float(args.point_conf_min),
        "kept_views": [int(v) for v in sorted(kept_views)],
        "dropped_views": [int(row["view_idx"]) for row in view_rows if not row["keep"]],
        "points_after_view_conf_pruning": int(np.count_nonzero(mask)),
        "processed_points": int(processed_points.shape[0]),
        "pointcloud_processing": process_report,
    }
    (output_dir / "saved_reconstruction_reprocess_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    gaussian_report = gaussian_splat_branch(
        ReconstructionOutput(points=processed_points, colors=processed_colors, scale_applied=1.0),
        output_root=output_dir,
        config=GaussianSplatConfig(
            enabled=True,
            nn_scale_mult=float(args.gaussian_nn_scale_mult),
            max_scale=float(args.gaussian_max_scale),
            scale_percentile_high=float(args.gaussian_scale_percentile_high),
            surface_aligned=bool(args.gaussian_surface_aligned),
        ),
    )
    print(f"Processed points: {processed_points.shape[0]:,}")
    print(f"Gaussian PLY: {gaussian_report.get('gaussian_ply')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
