"""Build Gaussian-splat initialization data from raw model point clouds.

This script is experimental and intended for fast iteration:
1) load raw points/colors from Fast3R or DUSt3R outputs
2) optionally filter outliers (iqr_then_radius)
3) estimate normals and local spacing
4) export a Gaussian-style PLY and run report
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Open3D is required. Install it in your active environment first."
    ) from exc


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = THIS_DIR.parent
BETA_ROOT = EXPERIMENTS_ROOT / "beta_pipeline_testing"
DEFAULT_OUTPUTS_ROOT = BETA_ROOT / "outputs"
DEFAULT_TMP_RESULTS = THIS_DIR / "tmp_results"

C0 = 0.28209479177387814  # SH constant used by common Gaussian-splat pipelines


def print_stage(stage_idx: int, total: int, title: str) -> None:
    pct = (stage_idx / max(total, 1)) * 100.0
    print(f"\n[{stage_idx}/{total}] {title} ({pct:.0f}%)", flush=True)


def _default_paths_for_model(model: str) -> tuple[Path, Optional[Path]]:
    output_dir = DEFAULT_OUTPUTS_ROOT / model
    return output_dir / "raw_points.npy", output_dir / "raw_colors.npy"


def _default_poses_path_for_model(model: str) -> Path:
    return DEFAULT_OUTPUTS_ROOT / model / "poses.npy"


def load_poses(poses_path: Path) -> np.ndarray:
    poses = np.load(poses_path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses with shape (N,4,4), got {poses.shape}")
    return poses.astype(np.float32, copy=False)


def extract_camera_centers_auto(poses: np.ndarray, ref_points: np.ndarray) -> tuple[np.ndarray, str]:
    # Candidate 1: translation is already camera center (cam-to-world style)
    centers_t = poses[:, :3, 3]
    # Candidate 2: invert world-to-cam extrinsics center = -R^T t
    r = poses[:, :3, :3]
    t = poses[:, :3, 3]
    centers_inv = -np.einsum("nij,nj->ni", np.transpose(r, (0, 2, 1)), t)

    ref_center = np.median(ref_points, axis=0) if ref_points.size else np.zeros(3, dtype=np.float32)
    med_t = float(np.median(np.linalg.norm(centers_t - ref_center, axis=1)))
    med_inv = float(np.median(np.linalg.norm(centers_inv - ref_center, axis=1)))

    if med_t <= med_inv:
        return centers_t.astype(np.float32), "translation_column"
    return centers_inv.astype(np.float32), "inverse_extrinsic"


def nearest_camera_distance(points: np.ndarray, camera_centers: np.ndarray) -> np.ndarray:
    if points.size == 0 or camera_centers.size == 0:
        return np.zeros(points.shape[0], dtype=np.float32)
    diff = points[:, None, :] - camera_centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist.min(axis=1).astype(np.float32)


def camera_distance_mask(points: np.ndarray, camera_centers: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    dmin = nearest_camera_distance(points, camera_centers)
    threshold = float(np.percentile(dmin, percentile))
    return dmin <= threshold, threshold


def alpha_from_camera_distance(
    camera_dmin: np.ndarray,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    if camera_dmin.size == 0:
        return np.array([], dtype=np.float32)
    d05 = float(np.percentile(camera_dmin, 5))
    d95 = float(np.percentile(camera_dmin, 95))
    denom = max(d95 - d05, 1e-9)
    d_norm = np.clip((camera_dmin - d05) / denom, 0.0, 1.0)
    alpha = float(alpha_max) - d_norm * (float(alpha_max) - float(alpha_min))
    return np.clip(alpha.astype(np.float32), float(alpha_min), float(alpha_max))


def coerce_colors(colors: Optional[np.ndarray], n_points: int) -> np.ndarray:
    if colors is None or colors.ndim != 2 or colors.shape[0] != n_points:
        return np.full((n_points, 3), 0.5, dtype=np.float32)

    if colors.shape[1] > 3:
        colors = colors[:, :3]
    if colors.shape[1] < 3:
        out = np.full((n_points, 3), 0.5, dtype=np.float32)
        out[:, : colors.shape[1]] = colors
        colors = out

    colors = colors.astype(np.float32, copy=False)
    if colors.max(initial=1.0) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def load_arrays(
    points_path: Optional[Path],
    colors_path: Optional[Path],
    npz_path: Optional[Path],
) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    source_info: dict = {}
    if npz_path is not None:
        payload = np.load(npz_path)
        if "points" not in payload:
            raise KeyError(f"`points` key not found in npz: {npz_path}")
        points = payload["points"]
        colors = payload["colors"] if "colors" in payload else None
        source_info["input_mode"] = "npz"
        source_info["points_source"] = str(npz_path)
        source_info["colors_source"] = str(npz_path) if colors is not None else ""
        return points, colors, source_info

    if points_path is None:
        raise ValueError("Points path is required when --npz is not provided.")
    points = np.load(points_path)
    colors = np.load(colors_path) if (colors_path is not None and colors_path.exists()) else None
    source_info["input_mode"] = "npy"
    source_info["points_source"] = str(points_path)
    source_info["colors_source"] = str(colors_path) if colors is not None and colors_path is not None else ""
    return points, colors, source_info


def bbox_extent(points: np.ndarray) -> list[float]:
    if points.size == 0:
        return [0.0, 0.0, 0.0]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ext = maxs - mins
    return [float(ext[0]), float(ext[1]), float(ext[2])]


def bbox_diagonal(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def safe_min(arr: np.ndarray, default: float = 0.0) -> float:
    return float(np.min(arr)) if arr.size else float(default)


def safe_max(arr: np.ndarray, default: float = 0.0) -> float:
    return float(np.max(arr)) if arr.size else float(default)


def method_iqr_axis(points: np.ndarray, iqr_k: float = 1.5) -> np.ndarray:
    q1 = np.percentile(points, 25, axis=0)
    q3 = np.percentile(points, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    return np.all((points >= lower) & (points <= upper), axis=1)


def method_radius_percentile(points: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    threshold = float(np.percentile(dist, percentile))
    return dist <= threshold


def method_iqr_then_radius(points: np.ndarray, iqr_k: float, radius_percentile: float) -> np.ndarray:
    first_mask = method_iqr_axis(points, iqr_k=iqr_k)
    reduced = points[first_mask]
    if reduced.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    second_mask_local = method_radius_percentile(reduced, percentile=radius_percentile)
    reduced_indices = np.flatnonzero(first_mask)
    kept_indices = reduced_indices[second_mask_local]
    final_mask = np.zeros(points.shape[0], dtype=bool)
    final_mask[kept_indices] = True
    return final_mask


def subsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    max_points: Optional[int],
    seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray], bool]:
    if max_points is None or points.shape[0] <= max_points:
        return points, colors, False
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    pts = points[indices]
    cols = colors[indices] if colors is not None and colors.shape[0] == points.shape[0] else None
    return pts, cols, True


def make_pointcloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def write_gaussian_ply(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    sh_dc: np.ndarray,
    sh_rest: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> None:
    n = points.shape[0]
    if n == 0:
        raise ValueError("No points left to write.")

    dtype = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
    ]
    for i in range(sh_rest.shape[1]):
        dtype.append((f"f_rest_{i}", "<f4"))
    dtype.extend(
        [
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )

    arr = np.empty(n, dtype=np.dtype(dtype))
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["nx"] = normals[:, 0].astype(np.float32)
    arr["ny"] = normals[:, 1].astype(np.float32)
    arr["nz"] = normals[:, 2].astype(np.float32)
    arr["f_dc_0"] = sh_dc[:, 0].astype(np.float32)
    arr["f_dc_1"] = sh_dc[:, 1].astype(np.float32)
    arr["f_dc_2"] = sh_dc[:, 2].astype(np.float32)
    for i in range(sh_rest.shape[1]):
        arr[f"f_rest_{i}"] = sh_rest[:, i].astype(np.float32)
    arr["opacity"] = opacity.astype(np.float32)
    arr["scale_0"] = scales[:, 0].astype(np.float32)
    arr["scale_1"] = scales[:, 1].astype(np.float32)
    arr["scale_2"] = scales[:, 2].astype(np.float32)
    arr["rot_0"] = rotations[:, 0].astype(np.float32)
    arr["rot_1"] = rotations[:, 1].astype(np.float32)
    arr["rot_2"] = rotations[:, 2].astype(np.float32)
    arr["rot_3"] = rotations[:, 3].astype(np.float32)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated by build_gaussian_splats.py",
        f"element vertex {n}",
    ]
    for field_name, _ in dtype:
        header_lines.append(f"property float {field_name}")
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    with path.open("wb") as f:
        f.write(header)
        arr.tofile(f)


def write_txt_report(path: Path, report: dict) -> None:
    lines: list[str] = []
    lines.append("Gaussian Splat Build Report")
    lines.append("=" * 80)
    keys = [
        "timestamp",
        "model_hint",
        "input_mode",
        "points_source",
        "colors_source",
        "raw_points_count_original",
        "finite_points_count",
        "invalid_points_dropped",
        "points_used",
        "subsample_applied",
        "subsample_target",
        "filter_enabled",
        "iqr_k",
        "radius_percentile",
        "poses_source",
        "poses_loaded",
        "num_cameras",
        "pose_convention_selected",
        "pose_aware_filter",
        "camera_distance_percentile",
        "camera_distance_threshold",
        "voxel_size",
        "normal_radius_effective",
        "normal_max_nn",
        "nn_scale_mult",
        "min_scale",
        "max_scale",
        "alpha_mode",
        "alpha_by_camera",
        "alpha",
        "alpha_min",
        "alpha_max",
        "output_dir",
    ]
    for key in keys:
        lines.append(f"{key}: {report.get(key)}")

    lines.append("")
    lines.append("Counts")
    lines.append("-" * 80)
    for k, v in report["counts"].items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("Timing (seconds)")
    lines.append("-" * 80)
    for k, v in report["timing_seconds"].items():
        lines.append(f"{k}: {v:.4f}")

    lines.append("")
    lines.append("Scale stats")
    lines.append("-" * 80)
    for k, v in report["scale_stats"].items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("Artifacts")
    lines.append("-" * 80)
    for k, v in report["artifacts"].items():
        lines.append(f"{k}: {v}")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Gaussian-splat initialization PLY from raw point clouds."
    )
    parser.add_argument("--model", choices=["fast3r", "dust3r", "legacy_depth"], default="fast3r")
    parser.add_argument("--points", type=Path, default=None, help="Override points .npy path.")
    parser.add_argument("--colors", type=Path, default=None, help="Override colors .npy path.")
    parser.add_argument("--npz", type=Path, default=None, help="Optional .npz containing points/colors.")
    parser.add_argument("--poses", type=Path, default=None, help="Override poses .npy path (shape Nx4x4).")
    parser.add_argument("--max-points", type=int, default=800000, help="Optional random subsample size.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed when --max-points is used.")
    parser.add_argument("--disable-filter", action="store_true", help="Skip iqr_then_radius filtering.")
    parser.add_argument(
        "--pose-aware-filter",
        action="store_true",
        help="Use camera centers from poses.npy to filter far points.",
    )
    parser.add_argument(
        "--camera-distance-percentile",
        type=float,
        default=99.5,
        help="Keep points within this percentile of nearest-camera distance.",
    )
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--radius-percentile", type=float, default=99.5)
    parser.add_argument("--voxel-size", type=float, default=0.0015, help="Voxel downsample size; 0 disables.")
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.0,
        help="Normal estimation radius; <=0 means auto from scene scale.",
    )
    parser.add_argument("--normal-max-nn", type=int, default=64)
    parser.add_argument("--nn-scale-mult", type=float, default=0.7, help="Scale multiplier on nearest-neighbor distance.")
    parser.add_argument("--min-scale", type=float, default=0.00025)
    parser.add_argument("--max-scale", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.85, help="Used when --alpha-by-density is not set.")
    parser.add_argument("--alpha-by-density", action="store_true", help="Set per-point alpha from local density.")
    parser.add_argument("--alpha-by-camera", action="store_true", help="Set per-point alpha from camera distance.")
    parser.add_argument("--alpha-min", type=float, default=0.55)
    parser.add_argument("--alpha-max", type=float, default=0.95)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_TMP_RESULTS,
        help="Root output folder for dated runs.",
    )
    parser.add_argument("--output-name", type=str, default="gaussian_splats_init.ply")
    parser.add_argument("--save-npz", action="store_true", help="Also save gaussian parameter arrays to NPZ.")
    parser.add_argument("--save-open3d-ply", action="store_true", help="Also save filtered point cloud as classic PLY.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_stages = 8

    default_points, default_colors = _default_paths_for_model(args.model)
    default_poses = _default_poses_path_for_model(args.model)
    points_path = args.points if args.points is not None else default_points
    colors_path = args.colors if args.colors is not None else default_colors
    poses_path = args.poses if args.poses is not None else default_poses

    if args.npz is None and not points_path.exists():
        raise FileNotFoundError(f"Points file not found: {points_path}")
    if args.npz is not None and not args.npz.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.npz}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"{stamp}_{args.model}_gaussian_splatting"
    out_dir.mkdir(parents=True, exist_ok=True)

    print_stage(1, total_stages, "Load arrays + finite filtering")
    t0 = time.perf_counter()
    points_raw, colors_raw, source_info = load_arrays(points_path, colors_path, args.npz)
    if points_raw.ndim != 2 or points_raw.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points_raw.shape}")
    points_raw = points_raw.astype(np.float32, copy=False)
    original_count = int(points_raw.shape[0])
    finite_mask = np.isfinite(points_raw).all(axis=1)
    points = points_raw[finite_mask]
    colors = colors_raw[finite_mask] if colors_raw is not None and len(colors_raw) == len(finite_mask) else None
    finite_count = int(points.shape[0])
    invalid_dropped = int(original_count - finite_count)
    colors = coerce_colors(colors, points.shape[0])
    t_load = time.perf_counter() - t0
    print(f"  finite_points={finite_count:,}", flush=True)

    print_stage(2, total_stages, "Optional subsampling")
    points, colors, subsample_applied = subsample(points, colors, args.max_points, args.seed)
    used_count = int(points.shape[0])
    print(f"  points_used={used_count:,}", flush=True)

    camera_centers: Optional[np.ndarray] = None
    poses_source = ""
    poses_loaded = False
    num_cameras = 0
    pose_convention_selected = ""
    camera_centers_path = ""
    camera_distance_threshold: float | str = ""
    apply_pose_ops = bool(args.pose_aware_filter or args.alpha_by_camera)
    if apply_pose_ops:
        if poses_path.exists():
            poses = load_poses(poses_path)
            camera_centers, pose_convention_selected = extract_camera_centers_auto(poses, points)
            poses_source = str(poses_path)
            poses_loaded = True
            num_cameras = int(camera_centers.shape[0])
            camera_centers_out = out_dir / "camera_centers.npy"
            np.save(camera_centers_out, camera_centers)
            camera_centers_path = str(camera_centers_out)
            print(
                f"  loaded poses={num_cameras} (convention={pose_convention_selected})",
                flush=True,
            )
        else:
            print(
                f"  warning: poses not found at {poses_path}, skipping pose-aware features",
                flush=True,
            )

    print_stage(3, total_stages, "Outlier + pose-aware filtering")
    t1 = time.perf_counter()
    if args.disable_filter:
        filtered_mask = np.ones(points.shape[0], dtype=bool)
    else:
        filtered_mask = method_iqr_then_radius(
            points,
            iqr_k=float(args.iqr_k),
            radius_percentile=float(args.radius_percentile),
        )
    points_filtered = points[filtered_mask]
    colors_filtered = colors[filtered_mask]
    count_after_filter = int(points_filtered.shape[0])

    if args.pose_aware_filter and camera_centers is not None and points_filtered.shape[0] > 0:
        cam_mask, camera_distance_threshold = camera_distance_mask(
            points_filtered,
            camera_centers,
            percentile=float(args.camera_distance_percentile),
        )
        points_filtered = points_filtered[cam_mask]
        colors_filtered = colors_filtered[cam_mask]

    count_after_pose_filter = int(points_filtered.shape[0])
    t_filter = time.perf_counter() - t1
    print(
        f"  points_after_filter={count_after_filter:,} "
        f"points_after_pose_filter={count_after_pose_filter:,}",
        flush=True,
    )

    print_stage(4, total_stages, "Point cloud build + voxel downsample")
    t2 = time.perf_counter()
    pcd = make_pointcloud(points_filtered, colors_filtered)
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel_size))
    points_voxel = np.asarray(pcd.points, dtype=np.float32)
    colors_voxel = np.asarray(pcd.colors, dtype=np.float32)
    t_voxel = time.perf_counter() - t2
    print(f"  points_after_voxel={points_voxel.shape[0]:,}", flush=True)

    print_stage(5, total_stages, "Normal + NN distance estimation")
    t3 = time.perf_counter()
    diag = bbox_diagonal(points_voxel)
    auto_normal_radius = max(float(args.voxel_size) * 3.0, diag * 0.01, 1e-4)
    normal_radius_effective = float(args.normal_radius) if float(args.normal_radius) > 0 else auto_normal_radius
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius_effective),
            max_nn=int(args.normal_max_nn),
        )
    )
    normals = np.asarray(pcd.normals, dtype=np.float32)
    nn_dist = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float32)
    if nn_dist.shape[0] != points_voxel.shape[0]:
        nn_dist = np.full(points_voxel.shape[0], max(float(args.min_scale), 1e-6), dtype=np.float32)
    nn_dist = np.nan_to_num(
        nn_dist,
        nan=float(args.min_scale),
        posinf=float(args.max_scale),
        neginf=float(args.min_scale),
    )
    camera_dmin_voxel = (
        nearest_camera_distance(points_voxel, camera_centers)
        if camera_centers is not None and points_voxel.shape[0] > 0
        else np.array([], dtype=np.float32)
    )
    t_normals = time.perf_counter() - t3

    print_stage(6, total_stages, "Gaussian parameter build")
    t4 = time.perf_counter()
    sh_dc = ((colors_voxel - 0.5) / C0).astype(np.float32)
    sh_rest = np.zeros((points_voxel.shape[0], 45), dtype=np.float32)

    scales_world = np.clip(nn_dist * float(args.nn_scale_mult), float(args.min_scale), float(args.max_scale))
    scales_param = np.log(np.repeat(scales_world[:, None], 3, axis=1)).astype(np.float32)

    if args.alpha_by_density:
        d_min = float(nn_dist.min(initial=0.0))
        d_max = float(nn_dist.max(initial=1.0))
        denom = max(d_max - d_min, 1e-9)
        d_norm = (nn_dist - d_min) / denom
        alpha = float(args.alpha_max) - d_norm * (float(args.alpha_max) - float(args.alpha_min))
        alpha = np.clip(alpha, float(args.alpha_min), float(args.alpha_max))
        alpha_mode = "density"
    else:
        alpha = np.full(points_voxel.shape[0], float(args.alpha), dtype=np.float32)
        alpha_mode = "constant"

    if args.alpha_by_camera and camera_dmin_voxel.size > 0:
        alpha_cam = alpha_from_camera_distance(
            camera_dmin_voxel,
            alpha_min=float(args.alpha_min),
            alpha_max=float(args.alpha_max),
        )
        if alpha_mode == "constant":
            alpha = alpha_cam
            alpha_mode = "camera_distance"
        else:
            alpha = np.minimum(alpha, alpha_cam)
            alpha_mode = "density_and_camera"
    elif args.alpha_by_camera and camera_dmin_voxel.size == 0:
        print("  warning: --alpha-by-camera requested but poses were not available", flush=True)

    alpha = np.clip(alpha, 1e-4, 1.0 - 1e-4)
    opacity = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    rotations = np.zeros((points_voxel.shape[0], 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    t_params = time.perf_counter() - t4

    print_stage(7, total_stages, "Write artifacts")
    t5 = time.perf_counter()
    gaussian_ply = out_dir / args.output_name
    write_gaussian_ply(
        gaussian_ply,
        points=points_voxel,
        normals=normals,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        opacity=opacity,
        scales=scales_param,
        rotations=rotations,
    )

    classic_ply_path = ""
    if args.save_open3d_ply:
        classic_ply = out_dir / "points_filtered_voxel.ply"
        o3d.io.write_point_cloud(str(classic_ply), pcd)
        classic_ply_path = str(classic_ply)

    npz_path = ""
    if args.save_npz:
        npz_path_obj = out_dir / "gaussian_params.npz"
        np.savez_compressed(
            npz_path_obj,
            points=points_voxel,
            colors=colors_voxel,
            normals=normals,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
            opacity=opacity,
            scales=scales_param,
            rotations=rotations,
        )
        npz_path = str(npz_path_obj)
    t_write = time.perf_counter() - t5

    print_stage(8, total_stages, "Write report")
    report = {
        "timestamp": stamp,
        "model_hint": args.model,
        **source_info,
        "output_dir": str(out_dir),
        "raw_points_count_original": original_count,
        "finite_points_count": finite_count,
        "invalid_points_dropped": invalid_dropped,
        "points_used": used_count,
        "subsample_applied": bool(subsample_applied),
        "subsample_target": int(args.max_points) if args.max_points is not None else "full",
        "filter_enabled": not bool(args.disable_filter),
        "iqr_k": float(args.iqr_k),
        "radius_percentile": float(args.radius_percentile),
        "poses_source": poses_source,
        "poses_loaded": bool(poses_loaded),
        "num_cameras": int(num_cameras),
        "pose_convention_selected": pose_convention_selected,
        "pose_aware_filter": bool(args.pose_aware_filter),
        "camera_distance_percentile": float(args.camera_distance_percentile),
        "camera_distance_threshold": camera_distance_threshold,
        "voxel_size": float(args.voxel_size),
        "normal_radius_effective": float(normal_radius_effective),
        "normal_max_nn": int(args.normal_max_nn),
        "nn_scale_mult": float(args.nn_scale_mult),
        "min_scale": float(args.min_scale),
        "max_scale": float(args.max_scale),
        "alpha_mode": alpha_mode,
        "alpha_by_camera": bool(args.alpha_by_camera),
        "alpha": float(args.alpha),
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "counts": {
            "after_filter": int(count_after_filter),
            "after_pose_filter": int(count_after_pose_filter),
            "after_voxel": int(points_voxel.shape[0]),
        },
        "bbox_extent_xyz": {
            "used_input": bbox_extent(points),
            "after_filter": bbox_extent(points[filtered_mask]),
            "after_pose_filter": bbox_extent(points_filtered),
            "after_voxel": bbox_extent(points_voxel),
        },
        "scale_stats": {
            "nn_distance_min": safe_min(nn_dist),
            "nn_distance_median": float(np.median(nn_dist)) if nn_dist.size else 0.0,
            "nn_distance_max": safe_max(nn_dist),
            "camera_distance_min": safe_min(camera_dmin_voxel) if camera_dmin_voxel.size else "",
            "camera_distance_median": float(np.median(camera_dmin_voxel)) if camera_dmin_voxel.size else "",
            "camera_distance_max": safe_max(camera_dmin_voxel) if camera_dmin_voxel.size else "",
            "scale_world_min": safe_min(scales_world),
            "scale_world_median": float(np.median(scales_world)) if scales_world.size else 0.0,
            "scale_world_max": safe_max(scales_world),
            "opacity_logit_min": safe_min(opacity),
            "opacity_logit_max": safe_max(opacity),
        },
        "timing_seconds": {
            "load": float(t_load),
            "filter": float(t_filter),
            "voxel": float(t_voxel),
            "normals_and_nn": float(t_normals),
            "build_gaussian_params": float(t_params),
            "write_outputs": float(t_write),
        },
        "artifacts": {
            "gaussian_ply": str(gaussian_ply),
            "gaussian_npz": npz_path,
            "filtered_open3d_ply": classic_ply_path,
            "camera_centers_npy": camera_centers_path,
        },
    }

    txt_path = out_dir / "gaussian_build_report.txt"
    json_path = out_dir / "gaussian_build_report.json"
    write_txt_report(txt_path, report)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved results to: {out_dir}", flush=True)
    print(f"Gaussian PLY: {gaussian_ply}", flush=True)
    print(f"TXT report: {txt_path}", flush=True)
    print(f"JSON report: {json_path}", flush=True)


if __name__ == "__main__":
    main()
