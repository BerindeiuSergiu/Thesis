"""Build a mesh from raw points using iqr_then_radius filtering.

Outputs are saved under:
    experiments/mesh_reconstrction_methods/tmp_results/<timestamp>_<model>_iqr_then_radius/
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Open3D is required for mesh creation. "
        "Install it in your active environment or run with your thesis env python."
    ) from exc


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = THIS_DIR.parent
BETA_ROOT = EXPERIMENTS_ROOT / "beta_pipeline_testing"
DEFAULT_OUTPUTS_ROOT = BETA_ROOT / "outputs"
DEFAULT_TMP_RESULTS = THIS_DIR / "tmp_results"


def _default_paths_for_model(model: str) -> tuple[Path, Optional[Path]]:
    output_dir = DEFAULT_OUTPUTS_ROOT / model
    return output_dir / "raw_points.npy", output_dir / "raw_colors.npy"


def _default_poses_path_for_model(model: str) -> Path:
    return DEFAULT_OUTPUTS_ROOT / model / "poses.npy"


def load_arrays(points_path: Optional[Path], colors_path: Optional[Path], npz_path: Optional[Path]) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
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


def load_poses(poses_path: Path) -> np.ndarray:
    poses = np.load(poses_path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses with shape (N,4,4), got {poses.shape}")
    return poses.astype(np.float32, copy=False)


def extract_camera_centers_auto(poses: np.ndarray, ref_points: np.ndarray) -> tuple[np.ndarray, str]:
    # Candidate 1: translation column already camera center.
    centers_t = poses[:, :3, 3]
    # Candidate 2: camera center from world-to-camera extrinsics: -R^T t.
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


def subsample(points: np.ndarray, colors: Optional[np.ndarray], max_points: Optional[int], seed: int) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if max_points is None or points.shape[0] <= max_points:
        return points, colors, None
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    pts = points[indices]
    cols = colors[indices] if colors is not None and colors.shape[0] == points.shape[0] else None
    return pts, cols, indices


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


def method_iqr_then_radius(points: np.ndarray, iqr_k: float = 1.5, radius_percentile: float = 99.0) -> np.ndarray:
    first_mask = method_iqr_axis(points, iqr_k=iqr_k)
    reduced = points[first_mask]
    if reduced.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    second_mask_local = method_radius_percentile(reduced, percentile=radius_percentile)
    reduced_indices = np.flatnonzero(first_mask)
    kept_indices = reduced_indices[second_mask_local]
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[kept_indices] = True
    return mask


def make_pointcloud(points: np.ndarray, colors: Optional[np.ndarray]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def parse_radii(text: str) -> list[float]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one BPA radius is required.")
    return [float(v) for v in values]


def mesh_metrics(mesh: o3d.geometry.TriangleMesh, include_self_intersection: bool = False) -> dict:
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    self_intersecting = bool(mesh.is_self_intersecting()) if include_self_intersection else "skipped"
    return {
        "num_vertices": int(len(mesh.vertices)),
        "num_triangles": int(len(mesh.triangles)),
        "surface_area": float(mesh.get_surface_area()),
        "bbox_extent_xyz": [float(extent[0]), float(extent[1]), float(extent[2])],
        "is_watertight": bool(mesh.is_watertight()),
        "is_edge_manifold": bool(mesh.is_edge_manifold()),
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()),
        "is_self_intersecting": self_intersecting,
    }


def write_txt(path: Path, report: dict) -> None:
    lines: list[str] = []
    lines.append("Mesh Build Report (iqr_then_radius)")
    lines.append("=" * 80)

    for key in [
        "timestamp",
        "model_hint",
        "points_source",
        "colors_source",
        "input_mode",
        "raw_points_count_original",
        "finite_points_count",
        "invalid_points_dropped",
        "points_used_for_filtering",
        "subsample_applied",
        "subsample_target_max_points",
        "subsample_size_actual",
        "seed",
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
        "skip_voxel",
        "mesh_method",
        "bpa_radii",
        "poisson_depth",
        "poisson_scale",
        "poisson_linear_fit",
        "orient_normals",
        "normal_radius",
        "normal_radius_effective",
        "normal_max_nn",
        "poisson_trim_quantile",
        "crop_bbox_margin",
        "smooth_taubin_iters",
        "output_dir",
    ]:
        lines.append(f"{key}: {report.get(key)}")

    lines.append("")
    lines.append("Stage counts")
    lines.append("-" * 80)
    counts = report["counts"]
    for name, value in counts.items():
        lines.append(f"{name}: {value}")

    lines.append("")
    lines.append("Retention")
    lines.append("-" * 80)
    ret = report["retention"]
    for name, value in ret.items():
        lines.append(f"{name}: {value:.6f}")

    lines.append("")
    lines.append("Timing (seconds)")
    lines.append("-" * 80)
    for name, value in report["timing_seconds"].items():
        lines.append(f"{name}: {value:.4f}")

    lines.append("")
    lines.append("Extents")
    lines.append("-" * 80)
    for name, value in report["bbox_extent_xyz"].items():
        lines.append(f"{name}: {value}")

    if "camera_distance_stats" in report:
        lines.append("")
        lines.append("Camera distance stats")
        lines.append("-" * 80)
        for name, value in report["camera_distance_stats"].items():
            lines.append(f"{name}: {value}")

    lines.append("")
    lines.append("Mesh metrics")
    lines.append("-" * 80)
    for key, value in report["mesh_metrics"].items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Artifacts")
    lines.append("-" * 80)
    for key, value in report["artifacts"].items():
        lines.append(f"{key}: {value}")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_stage_header(stage_index: int, total_stages: int, name: str) -> None:
    pct = (stage_index / max(total_stages, 1)) * 100.0
    print(f"\n[{stage_index}/{total_stages}] {name} ({pct:.0f}%)", flush=True)


def run_with_spinner(label: str, fn):
    """Run a function while printing a live spinner and elapsed time."""
    done = threading.Event()
    result_holder: dict = {}
    error_holder: dict = {}

    def _worker():
        try:
            result_holder["value"] = fn()
        except Exception as exc:  # pragma: no cover - runtime propagation
            error_holder["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    frames = ["-", "\\", "|", "/"]
    frame_idx = 0
    start = time.perf_counter()
    while not done.wait(0.2):
        elapsed = time.perf_counter() - start
        print(f"\r  {label} {frames[frame_idx % len(frames)]} elapsed={elapsed:7.2f}s", end="", flush=True)
        frame_idx += 1

    elapsed = time.perf_counter() - start
    print(f"\r  {label} done elapsed={elapsed:7.2f}s", flush=True)

    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create mesh from raw points using iqr_then_radius filtering.")
    parser.add_argument("--model", choices=["fast3r", "dust3r"], default="fast3r")
    parser.add_argument("--points", type=Path, default=None, help="Override points .npy path.")
    parser.add_argument("--colors", type=Path, default=None, help="Override colors .npy path.")
    parser.add_argument("--npz", type=Path, default=None, help="Optional .npz containing points/colors.")
    parser.add_argument("--poses", type=Path, default=None, help="Optional poses .npy path (Nx4x4).")
    parser.add_argument("--max-points", type=int, default=None, help="Optional random subsample size.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed when --max-points is used.")
    parser.add_argument("--iqr-k", type=float, default=1.5, help="IQR multiplier for axis-based outlier clipping.")
    parser.add_argument(
        "--radius-percentile",
        type=float,
        default=99.0,
        help="Radius percentile kept after IQR stage (e.g. 99.5 keeps more detail).",
    )
    parser.add_argument(
        "--pose-aware-filter",
        action="store_true",
        help="Optionally filter far points using nearest camera distance from poses.",
    )
    parser.add_argument(
        "--camera-distance-percentile",
        type=float,
        default=99.5,
        help="Keep points within this percentile of nearest-camera distance.",
    )
    parser.add_argument("--voxel-size", type=float, default=0.002, help="Voxel downsample size before meshing.")
    parser.add_argument("--skip-voxel", action="store_true", help="Skip voxel downsampling for maximum detail.")
    parser.add_argument("--mesh-method", choices=["bpa", "poisson"], default="bpa")
    parser.add_argument("--bpa-radii", type=str, default="0.004,0.008,0.016,0.032")
    parser.add_argument("--poisson-depth", type=int, default=10)
    parser.add_argument("--poisson-scale", type=float, default=1.1, help="Poisson scale parameter.")
    parser.add_argument("--poisson-linear-fit", action="store_true", help="Enable Poisson linear fit.")
    parser.add_argument(
        "--poisson-trim-quantile",
        type=float,
        default=0.005,
        help="Remove lowest-density vertices after Poisson (0 disables).",
    )
    parser.add_argument(
        "--crop-bbox-margin",
        type=float,
        default=0.03,
        help="Expand input bbox by this fraction and crop Poisson mesh (0 disables).",
    )
    parser.add_argument(
        "--orient-normals",
        action="store_true",
        help="Run normal orientation consistency before Poisson (recommended).",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.0,
        help="Normal estimation search radius. <=0 uses auto radius from scene scale.",
    )
    parser.add_argument("--normal-max-nn", type=int, default=64, help="Normal estimation max neighbors.")
    parser.add_argument(
        "--smooth-taubin-iters",
        type=int,
        default=0,
        help="Optional Taubin smoothing iterations after mesh reconstruction.",
    )
    parser.add_argument(
        "--poisson-verbosity",
        action="store_true",
        help="Enable Open3D debug verbosity during Poisson reconstruction.",
    )
    parser.add_argument(
        "--self-intersection-check",
        action="store_true",
        help="Enable expensive mesh self-intersection check in final report.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_TMP_RESULTS, help="Root output folder for dated tmp results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    out_dir = args.out_root / f"{stamp}_{args.model}_iqr_then_radius"
    out_dir.mkdir(parents=True, exist_ok=True)

    points_raw, colors_raw, source_info = load_arrays(points_path, colors_path, args.npz)
    if points_raw.ndim != 2 or points_raw.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {points_raw.shape}")

    total_stages = 9

    print_stage_header(1, total_stages, "Load raw points and finite filtering")
    t0 = time.perf_counter()
    original_count = int(points_raw.shape[0])
    finite_mask = np.isfinite(points_raw).all(axis=1)
    points = points_raw[finite_mask]
    finite_count = int(points.shape[0])
    invalid_dropped = int(original_count - finite_count)

    if colors_raw is not None and colors_raw.ndim == 2 and colors_raw.shape[0] == finite_mask.shape[0]:
        colors = colors_raw[finite_mask]
    else:
        colors = None
    t_load = time.perf_counter() - t0

    print_stage_header(2, total_stages, "Optional subsampling")
    points, colors, _ = subsample(points, colors, args.max_points, args.seed)
    used_count = int(points.shape[0])
    print(f"  points_used_for_filtering={used_count:,}", flush=True)

    print_stage_header(3, total_stages, "IQR filter")
    t1 = time.perf_counter()
    iqr_mask = run_with_spinner("iqr_axis", lambda: method_iqr_axis(points, iqr_k=float(args.iqr_k)))
    points_iqr = points[iqr_mask]
    colors_iqr = colors[iqr_mask] if colors is not None else None
    t_iqr = time.perf_counter() - t1

    print_stage_header(4, total_stages, "Radius + optional pose-aware filter")
    t2 = time.perf_counter()
    rad_mask_local = (
        run_with_spinner(
            "radius_percentile",
            lambda: method_radius_percentile(points_iqr, percentile=float(args.radius_percentile)),
        )
        if points_iqr.shape[0] > 0
        else np.array([], dtype=bool)
    )
    points_filtered = points_iqr[rad_mask_local]
    colors_filtered = colors_iqr[rad_mask_local] if colors_iqr is not None else None
    points_after_iqr_radius = int(points_filtered.shape[0])

    poses_source = ""
    poses_loaded = False
    num_cameras = 0
    pose_convention_selected = ""
    camera_distance_threshold: float | str = ""
    camera_centers_npy_path = ""
    camera_dmin = np.array([], dtype=np.float32)

    if args.pose_aware_filter:
        if poses_path.exists():
            poses = load_poses(poses_path)
            camera_centers, pose_convention_selected = extract_camera_centers_auto(poses, points)
            poses_loaded = True
            poses_source = str(poses_path)
            num_cameras = int(camera_centers.shape[0])

            camera_centers_out = out_dir / "camera_centers.npy"
            np.save(camera_centers_out, camera_centers)
            camera_centers_npy_path = str(camera_centers_out)

            if points_filtered.shape[0] > 0:
                cam_mask, camera_distance_threshold = run_with_spinner(
                    "pose_distance_filter",
                    lambda: camera_distance_mask(
                        points_filtered,
                        camera_centers,
                        percentile=float(args.camera_distance_percentile),
                    ),
                )
                points_filtered = points_filtered[cam_mask]
                colors_filtered = colors_filtered[cam_mask] if colors_filtered is not None else None
                camera_dmin = nearest_camera_distance(points_filtered, camera_centers)
            print(
                f"  pose-aware: kept={points_filtered.shape[0]:,} from {points_after_iqr_radius:,} "
                f"with {num_cameras} cameras ({pose_convention_selected})",
                flush=True,
            )
        else:
            print(f"  warning: poses file not found, skipping pose-aware filter: {poses_path}", flush=True)

    t_rad = time.perf_counter() - t2

    print_stage_header(5, total_stages, "Voxel downsample")
    t3 = time.perf_counter()
    pcd_filtered = run_with_spinner("build_pointcloud", lambda: make_pointcloud(points_filtered, colors_filtered))
    if args.skip_voxel:
        pcd_voxel = pcd_filtered
    else:
        pcd_voxel = run_with_spinner(
            "voxel_down_sample",
            lambda: pcd_filtered.voxel_down_sample(voxel_size=float(args.voxel_size)),
        )
    points_voxel = np.asarray(pcd_voxel.points)
    colors_voxel = np.asarray(pcd_voxel.colors) if len(pcd_voxel.colors) == len(pcd_voxel.points) else None
    t_voxel = time.perf_counter() - t3

    print_stage_header(6, total_stages, "Mesh reconstruction")
    t4 = time.perf_counter()
    diag = bbox_diagonal(points_voxel)
    auto_normal_radius = max(float(args.voxel_size) * 3.0, diag * 0.01)
    effective_normal_radius = float(args.normal_radius) if float(args.normal_radius) > 0 else auto_normal_radius
    print(
        f"  normal_radius_effective={effective_normal_radius:.6f} "
        f"(auto={auto_normal_radius:.6f}, diag={diag:.6f})",
        flush=True,
    )
    run_with_spinner(
        "estimate_normals",
        lambda: pcd_voxel.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=effective_normal_radius,
                max_nn=int(args.normal_max_nn),
            )
        ),
    )
    if args.mesh_method == "bpa":
        radii = parse_radii(args.bpa_radii)
        mesh = run_with_spinner(
            "ball_pivoting",
            lambda: o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd_voxel, o3d.utility.DoubleVector(radii)
            ),
        )
    else:
        if args.orient_normals:
            run_with_spinner(
                "orient_normals",
                lambda: pcd_voxel.orient_normals_consistent_tangent_plane(30),
            )

        def _poisson():
            if args.poisson_verbosity:
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
                    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd_voxel,
                        depth=int(args.poisson_depth),
                        scale=float(args.poisson_scale),
                        linear_fit=bool(args.poisson_linear_fit),
                    )
            return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_voxel,
                depth=int(args.poisson_depth),
                scale=float(args.poisson_scale),
                linear_fit=bool(args.poisson_linear_fit),
            )

        mesh, densities = run_with_spinner("poisson_reconstruction", _poisson)

        if args.poisson_trim_quantile > 0:
            density_arr = np.asarray(densities)
            threshold = np.quantile(density_arr, float(args.poisson_trim_quantile))
            keep_vertices = density_arr >= threshold
            run_with_spinner(
                "poisson_trim_low_density",
                lambda: mesh.remove_vertices_by_mask(~keep_vertices),
            )

        if args.crop_bbox_margin > 0:
            bbox = pcd_voxel.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            margin_vec = np.asarray(extent) * float(args.crop_bbox_margin)
            min_b = bbox.get_min_bound() - margin_vec
            max_b = bbox.get_max_bound() + margin_vec
            crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
            mesh = run_with_spinner("poisson_crop_bbox", lambda: mesh.crop(crop_bbox))

    if int(args.smooth_taubin_iters) > 0:
        mesh = run_with_spinner(
            "smooth_taubin",
            lambda: mesh.filter_smooth_taubin(number_of_iterations=int(args.smooth_taubin_iters)),
        )

    run_with_spinner("compute_vertex_normals", lambda: mesh.compute_vertex_normals())
    t_mesh = time.perf_counter() - t4

    print_stage_header(7, total_stages, "Save outputs")
    # Persist intermediates
    filtered_npz = out_dir / "points_filtered_iqr_then_radius.npz"
    if colors_filtered is not None:
        np.savez_compressed(filtered_npz, points=points_filtered, colors=colors_filtered)
    else:
        np.savez_compressed(filtered_npz, points=points_filtered)

    voxel_npz = out_dir / "points_voxel.npz"
    if colors_voxel is not None:
        np.savez_compressed(voxel_npz, points=points_voxel, colors=colors_voxel)
    else:
        np.savez_compressed(voxel_npz, points=points_voxel)

    filtered_ply = out_dir / "points_filtered_iqr_then_radius.ply"
    voxel_ply = out_dir / "points_voxel.ply"
    mesh_ply = out_dir / "mesh.ply"
    mesh_obj = out_dir / "mesh.obj"
    run_with_spinner("write_filtered_ply", lambda: o3d.io.write_point_cloud(str(filtered_ply), pcd_filtered))
    run_with_spinner("write_voxel_ply", lambda: o3d.io.write_point_cloud(str(voxel_ply), pcd_voxel))
    run_with_spinner("write_mesh_ply", lambda: o3d.io.write_triangle_mesh(str(mesh_ply), mesh))
    run_with_spinner("write_mesh_obj", lambda: o3d.io.write_triangle_mesh(str(mesh_obj), mesh))

    print_stage_header(8, total_stages, "Build report")

    count_filtered = int(points_filtered.shape[0])
    count_voxel = int(points_voxel.shape[0])

    report = {
        "timestamp": stamp,
        "model_hint": args.model,
        **source_info,
        "raw_points_count_original": original_count,
        "finite_points_count": finite_count,
        "invalid_points_dropped": invalid_dropped,
        "points_used_for_filtering": used_count,
        "subsample_applied": bool(args.max_points is not None),
        "subsample_target_max_points": args.max_points if args.max_points is not None else "full",
        "subsample_size_actual": used_count if args.max_points is not None else "full",
        "seed": args.seed if args.max_points is not None else "",
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
        "skip_voxel": bool(args.skip_voxel),
        "mesh_method": args.mesh_method,
        "bpa_radii": args.bpa_radii,
        "poisson_depth": int(args.poisson_depth),
        "poisson_scale": float(args.poisson_scale),
        "poisson_linear_fit": bool(args.poisson_linear_fit),
        "output_dir": str(out_dir),
        "orient_normals": bool(args.orient_normals),
        "normal_radius": float(args.normal_radius),
        "normal_radius_effective": float(effective_normal_radius),
        "normal_max_nn": int(args.normal_max_nn),
        "poisson_trim_quantile": float(args.poisson_trim_quantile),
        "crop_bbox_margin": float(args.crop_bbox_margin),
        "smooth_taubin_iters": int(args.smooth_taubin_iters),
        "counts": {
            "after_iqr": int(points_iqr.shape[0]),
            "after_iqr_then_radius": int(points_after_iqr_radius),
            "after_pose_filter": count_filtered,
            "after_voxel": count_voxel,
        },
        "retention": {
            "after_iqr_vs_used": float(points_iqr.shape[0] / max(used_count, 1)),
            "after_iqr_then_radius_vs_used": float(points_after_iqr_radius / max(used_count, 1)),
            "after_pose_filter_vs_used": float(count_filtered / max(used_count, 1)),
            "after_voxel_vs_used": float(count_voxel / max(used_count, 1)),
        },
        "timing_seconds": {
            "load_and_finite_filter": float(t_load),
            "iqr_filter": float(t_iqr),
            "radius_filter": float(t_rad),
            "voxel_downsample": float(t_voxel),
            "mesh_reconstruction": float(t_mesh),
        },
        "bbox_extent_xyz": {
            "used_input": bbox_extent(points),
            "after_iqr": bbox_extent(points_iqr),
            "after_iqr_then_radius": bbox_extent(points_iqr[rad_mask_local]),
            "after_pose_filter": bbox_extent(points_filtered),
            "after_voxel": bbox_extent(points_voxel),
        },
        "camera_distance_stats": {
            "min": float(np.min(camera_dmin)) if camera_dmin.size else "",
            "median": float(np.median(camera_dmin)) if camera_dmin.size else "",
            "max": float(np.max(camera_dmin)) if camera_dmin.size else "",
        },
        "mesh_metrics": run_with_spinner(
            "compute_mesh_metrics",
            lambda: mesh_metrics(mesh, include_self_intersection=bool(args.self_intersection_check)),
        ),
        "artifacts": {
            "filtered_npz": str(filtered_npz),
            "voxel_npz": str(voxel_npz),
            "filtered_ply": str(filtered_ply),
            "voxel_ply": str(voxel_ply),
            "mesh_ply": str(mesh_ply),
            "mesh_obj": str(mesh_obj),
            "camera_centers_npy": camera_centers_npy_path,
        },
    }

    txt_path = out_dir / "mesh_build_report.txt"
    json_path = out_dir / "mesh_build_report.json"
    write_txt(txt_path, report)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_stage_header(9, total_stages, "Finalize")
    print(f"Saved results to: {out_dir}")
    print(f"TXT report: {txt_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
