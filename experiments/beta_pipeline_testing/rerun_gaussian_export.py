"""Regenerate only the Gaussian export from a completed beta pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from pipelines.shared.gaussian_splats import GaussianSplatConfig, gaussian_splat_branch
from pipelines.shared.reconstruction_data import ReconstructionOutput


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun only Gaussian export from saved processed point arrays.")
    parser.add_argument("run_dir", type=Path, help="Completed pipeline output directory.")
    parser.add_argument("--points-name", default="processed_scaled_points.npy")
    parser.add_argument("--colors-name", default="processed_scaled_colors.npy")
    parser.add_argument("--scale-applied", type=float, default=1.0)
    parser.add_argument("--output-dir-name", default="gaussian_splat")
    parser.add_argument("--output-name", default="gaussian_splats_init.ply")
    parser.add_argument("--save-npz", type=_parse_bool, default=True)
    parser.add_argument("--save-pointcloud-ply", type=_parse_bool, default=True)
    parser.add_argument("--gaussian-voxel-size", type=float, default=0.0)
    parser.add_argument("--gaussian-normal-radius", type=float, default=0.0)
    parser.add_argument("--gaussian-normal-max-nn", type=int, default=64)
    parser.add_argument("--gaussian-nn-scale-mult", type=float, default=0.35)
    parser.add_argument("--gaussian-min-scale", type=float, default=0.00025)
    parser.add_argument("--gaussian-max-scale", type=float, default=0.008)
    parser.add_argument("--gaussian-alpha", type=float, default=0.85)
    parser.add_argument("--gaussian-density-opacity", type=_parse_bool, default=True)
    parser.add_argument("--gaussian-min-alpha", type=float, default=0.35)
    parser.add_argument("--gaussian-max-alpha", type=float, default=0.82)
    parser.add_argument("--gaussian-scale-percentile-low", type=float, default=5.0)
    parser.add_argument("--gaussian-scale-percentile-high", type=float, default=85.0)
    parser.add_argument("--gaussian-surface-aligned", type=_parse_bool, default=False)
    parser.add_argument("--gaussian-tangent-scale-mult", type=float, default=1.0)
    parser.add_argument("--gaussian-normal-scale-mult", type=float, default=0.2)
    parser.add_argument("--gaussian-max-surface-aligned-points", type=int, default=8000000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    points_path = run_dir / args.points_name
    colors_path = run_dir / args.colors_name

    if not points_path.exists():
        raise FileNotFoundError(f"Missing points array: {points_path}")
    if not colors_path.exists():
        raise FileNotFoundError(f"Missing colors array: {colors_path}")

    print(f"Loading points: {points_path}")
    points = np.load(points_path, mmap_mode="r")
    print(f"Loading colors: {colors_path}")
    colors = np.load(colors_path, mmap_mode="r")
    print(f"Points available for Gaussian export: {points.shape[0]:,}")

    reconstruction = ReconstructionOutput(
        points=np.asarray(points, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.float32),
        scale_applied=float(args.scale_applied),
        metadata={"source_run_dir": str(run_dir), "source_points": str(points_path), "source_colors": str(colors_path)},
    )
    config = GaussianSplatConfig(
        enabled=True,
        output_dir_name=args.output_dir_name,
        output_name=args.output_name,
        save_npz=bool(args.save_npz),
        save_pointcloud_ply=bool(args.save_pointcloud_ply),
        voxel_size=float(args.gaussian_voxel_size),
        normal_radius=float(args.gaussian_normal_radius),
        normal_max_nn=int(args.gaussian_normal_max_nn),
        nn_scale_mult=float(args.gaussian_nn_scale_mult),
        min_scale=float(args.gaussian_min_scale),
        max_scale=float(args.gaussian_max_scale),
        alpha=float(args.gaussian_alpha),
        density_opacity=bool(args.gaussian_density_opacity),
        min_alpha=float(args.gaussian_min_alpha),
        max_alpha=float(args.gaussian_max_alpha),
        scale_percentile_low=float(args.gaussian_scale_percentile_low),
        scale_percentile_high=float(args.gaussian_scale_percentile_high),
        surface_aligned=bool(args.gaussian_surface_aligned),
        tangent_scale_mult=float(args.gaussian_tangent_scale_mult),
        normal_scale_mult=float(args.gaussian_normal_scale_mult),
        max_surface_aligned_points=int(args.gaussian_max_surface_aligned_points),
    )
    report = gaussian_splat_branch(reconstruction, output_root=run_dir, config=config)

    rerun_report = {
        "source_run_dir": str(run_dir),
        "source_points": str(points_path),
        "source_colors": str(colors_path),
        "gaussian_report": report,
    }
    (run_dir / args.output_dir_name / "gaussian_rerun_report.json").write_text(
        json.dumps(rerun_report, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Gaussian PLY: {report.get('gaussian_ply')}")
    print(f"Report: {run_dir / args.output_dir_name / 'gaussian_splat_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
