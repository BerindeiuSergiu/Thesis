from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import open3d as o3d

    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    o3d = None


C0 = 0.28209479177387814


def _coerce_colors(colors: np.ndarray | None, count: int) -> np.ndarray:
    if colors is None or colors.ndim != 2 or colors.shape[0] != count:
        return np.full((count, 3), 0.5, dtype=np.float32)
    colors = colors[:, :3].astype(np.float32, copy=False)
    if colors.size and float(np.nanmax(colors)) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _bbox_diagonal(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _make_pointcloud(points: np.ndarray, colors: np.ndarray | None):
    if not O3D_AVAILABLE:
        raise RuntimeError("Open3D is required for Gaussian splat export.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(_coerce_colors(colors, len(points)).astype(np.float64))
    return pcd


def _normalize_vectors(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-8)
    out = np.tile(fallback.astype(np.float32), (vectors.shape[0], 1))
    out[valid] = vectors[valid] / norms[valid]
    return out.astype(np.float32)


def _rotation_matrices_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    matrices = rotations.astype(np.float32, copy=False)
    quats = np.zeros((matrices.shape[0], 4), dtype=np.float32)
    trace = matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]

    mask = trace > 0.0
    s = np.sqrt(np.maximum(trace[mask] + 1.0, 1e-12)) * 2.0
    quats[mask, 0] = 0.25 * s
    quats[mask, 1] = (matrices[mask, 2, 1] - matrices[mask, 1, 2]) / s
    quats[mask, 2] = (matrices[mask, 0, 2] - matrices[mask, 2, 0]) / s
    quats[mask, 3] = (matrices[mask, 1, 0] - matrices[mask, 0, 1]) / s

    remaining = ~mask
    mask_x = remaining & (matrices[:, 0, 0] > matrices[:, 1, 1]) & (matrices[:, 0, 0] > matrices[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + matrices[mask_x, 0, 0] - matrices[mask_x, 1, 1] - matrices[mask_x, 2, 2], 1e-12)) * 2.0
    quats[mask_x, 0] = (matrices[mask_x, 2, 1] - matrices[mask_x, 1, 2]) / s
    quats[mask_x, 1] = 0.25 * s
    quats[mask_x, 2] = (matrices[mask_x, 0, 1] + matrices[mask_x, 1, 0]) / s
    quats[mask_x, 3] = (matrices[mask_x, 0, 2] + matrices[mask_x, 2, 0]) / s

    mask_y = remaining & ~mask_x & (matrices[:, 1, 1] > matrices[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + matrices[mask_y, 1, 1] - matrices[mask_y, 0, 0] - matrices[mask_y, 2, 2], 1e-12)) * 2.0
    quats[mask_y, 0] = (matrices[mask_y, 0, 2] - matrices[mask_y, 2, 0]) / s
    quats[mask_y, 1] = (matrices[mask_y, 0, 1] + matrices[mask_y, 1, 0]) / s
    quats[mask_y, 2] = 0.25 * s
    quats[mask_y, 3] = (matrices[mask_y, 1, 2] + matrices[mask_y, 2, 1]) / s

    mask_z = remaining & ~mask_x & ~mask_y
    s = np.sqrt(np.maximum(1.0 + matrices[mask_z, 2, 2] - matrices[mask_z, 0, 0] - matrices[mask_z, 1, 1], 1e-12)) * 2.0
    quats[mask_z, 0] = (matrices[mask_z, 1, 0] - matrices[mask_z, 0, 1]) / s
    quats[mask_z, 1] = (matrices[mask_z, 0, 2] + matrices[mask_z, 2, 0]) / s
    quats[mask_z, 2] = (matrices[mask_z, 1, 2] + matrices[mask_z, 2, 1]) / s
    quats[mask_z, 3] = 0.25 * s

    quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    valid = np.isfinite(quat_norms[:, 0]) & (quat_norms[:, 0] > 1e-8)
    quats[valid] /= quat_norms[valid]
    quats[~valid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quats.astype(np.float32)


def _surface_aligned_quaternions(normals: np.ndarray) -> np.ndarray:
    normal = _normalize_vectors(normals, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (normal.shape[0], 1))
    ref[np.abs(normal[:, 2]) > 0.9] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent0 = _normalize_vectors(np.cross(ref, normal), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    tangent1 = _normalize_vectors(np.cross(normal, tangent0), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    matrices = np.stack([tangent0, tangent1, normal], axis=2).astype(np.float32)
    return _rotation_matrices_to_quaternions(matrices)


def _write_gaussian_ply(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    sh_dc: np.ndarray,
    sh_rest: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> None:
    count = points.shape[0]
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
    for index in range(sh_rest.shape[1]):
        dtype.append((f"f_rest_{index}", "<f4"))
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

    arr = np.empty(count, dtype=np.dtype(dtype))
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
    for index in range(sh_rest.shape[1]):
        arr[f"f_rest_{index}"] = sh_rest[:, index]
    arr["opacity"] = opacity
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated by Fast3R app dual-output pipeline",
        f"element vertex {count}",
    ]
    header.extend(f"property float {field_name}" for field_name, _ in dtype)
    header.append("end_header")

    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        arr.tofile(handle)


def build_gaussian_splats(
    points: np.ndarray,
    colors: np.ndarray | None,
    output_dir: Path,
    config: dict[str, Any],
    scale_to_meters: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(config.get("gaussian_enabled", True)):
        return {"enabled": False, "output_dir": str(output_dir)}
    if points.size == 0:
        return {"enabled": False, "output_dir": str(output_dir), "reason": "empty_pointcloud"}
    if not O3D_AVAILABLE:
        raise RuntimeError("Open3D is required for Gaussian splat export.")

    voxel_size = float(config.get("gaussian_voxel_size", 0.0))
    pcd = _make_pointcloud(np.asarray(points, dtype=np.float32), colors)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    points_out = np.asarray(pcd.points, dtype=np.float32)
    colors_out = _coerce_colors(np.asarray(pcd.colors), points_out.shape[0])
    diagonal = _bbox_diagonal(points_out)
    normal_radius = float(config.get("gaussian_normal_radius", 0.0))
    if normal_radius <= 0:
        normal_radius = max(voxel_size * 3.0, diagonal * 0.01, 1e-4)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=int(config.get("gaussian_normal_max_nn", 64)),
        )
    )
    normals = np.asarray(pcd.normals, dtype=np.float32)
    if normals.shape != points_out.shape:
        normals = np.zeros_like(points_out, dtype=np.float32)

    nn_dist = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float32)
    min_scale = float(config.get("gaussian_min_scale", 0.00025))
    max_scale = float(config.get("gaussian_max_scale", 0.02))
    if nn_dist.shape[0] != points_out.shape[0]:
        nn_dist = np.full(points_out.shape[0], min_scale, dtype=np.float32)
    nn_dist = np.nan_to_num(nn_dist, nan=min_scale, posinf=max_scale, neginf=min_scale)

    positive_nn = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
    if positive_nn.size:
        pct_low = float(np.clip(config.get("gaussian_scale_percentile_low", 5.0), 0.0, 100.0))
        pct_high = float(np.clip(config.get("gaussian_scale_percentile_high", 95.0), pct_low, 100.0))
        nn_clip_low = float(np.percentile(positive_nn, pct_low))
        nn_clip_high = float(np.percentile(positive_nn, pct_high))
        if nn_clip_high <= nn_clip_low:
            nn_clip_high = float(max(nn_clip_low, np.median(positive_nn), min_scale))
    else:
        nn_clip_low = min_scale
        nn_clip_high = max_scale
    nn_dist_robust = np.clip(nn_dist, nn_clip_low, nn_clip_high)

    sh_dc = ((colors_out - 0.5) / C0).astype(np.float32)
    sh_rest = np.zeros((points_out.shape[0], 45), dtype=np.float32)
    scales_world = np.clip(
        nn_dist_robust * float(config.get("gaussian_nn_scale_mult", 0.7)),
        min_scale,
        max_scale,
    )
    if bool(config.get("gaussian_surface_aligned", False)):
        tangent_mult = max(float(config.get("gaussian_tangent_scale_mult", 1.15)), 1e-4)
        normal_mult = max(float(config.get("gaussian_normal_scale_mult", 0.25)), 1e-4)
        scales_xyz_world = np.stack(
            [
                np.clip(scales_world * tangent_mult, min_scale, max_scale),
                np.clip(scales_world * tangent_mult, min_scale, max_scale),
                np.clip(scales_world * normal_mult, min_scale, max_scale),
            ],
            axis=1,
        ).astype(np.float32)
    else:
        scales_xyz_world = np.repeat(scales_world[:, None], 3, axis=1).astype(np.float32)
    scales = np.log(np.clip(scales_xyz_world, 1e-8, None)).astype(np.float32)

    if bool(config.get("gaussian_density_opacity", False)) and points_out.shape[0] > 0 and nn_clip_high > nn_clip_low:
        dense_score = 1.0 - (nn_dist_robust - nn_clip_low) / (nn_clip_high - nn_clip_low + 1e-8)
        alpha_min = np.clip(float(config.get("gaussian_min_alpha", 0.35)), 1e-4, 1.0 - 1e-4)
        alpha_max = np.clip(float(config.get("gaussian_max_alpha", 0.85)), alpha_min, 1.0 - 1e-4)
        alpha_values = alpha_min + np.clip(dense_score, 0.0, 1.0) * (alpha_max - alpha_min)
        alpha = float(np.median(alpha_values))
    else:
        alpha = np.clip(float(config.get("gaussian_alpha", 0.85)), 1e-4, 1.0 - 1e-4)
        alpha_values = np.full(points_out.shape[0], alpha, dtype=np.float32)
    opacity = np.log(alpha_values / (1.0 - alpha_values)).astype(np.float32)

    surface_alignment_skipped_reason = ""
    max_aligned = int(config.get("gaussian_max_surface_aligned_points", 8_000_000))
    if bool(config.get("gaussian_surface_aligned", False)) and (max_aligned <= 0 or points_out.shape[0] <= max_aligned):
        rotations = _surface_aligned_quaternions(normals)
        rotations_are_identity = False
    else:
        rotations = np.zeros((points_out.shape[0], 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        rotations_are_identity = True
        if bool(config.get("gaussian_surface_aligned", False)) and points_out.shape[0] > max_aligned > 0:
            surface_alignment_skipped_reason = "too_many_points_for_memory_safe_rotation_export"

    gaussian_ply = output_dir / "gaussian_splats_init.ply"
    _write_gaussian_ply(gaussian_ply, points_out, normals, sh_dc, sh_rest, opacity, scales, rotations)

    pointcloud_ply = output_dir / "scaled_points_for_gaussian.ply"
    o3d.io.write_point_cloud(str(pointcloud_ply), pcd)
    npz_path = output_dir / "gaussian_params.npz"
    np.savez_compressed(
        npz_path,
        points=points_out,
        colors=colors_out,
        normals=normals,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
    )

    report = {
        "enabled": True,
        "output_dir": str(output_dir),
        "gaussian_ply": str(gaussian_ply),
        "gaussian_npz": str(npz_path),
        "pointcloud_ply": str(pointcloud_ply),
        "input_points": int(points.shape[0]),
        "output_gaussians": int(points_out.shape[0]),
        "scale_to_meters_already_applied": float(scale_to_meters),
        "voxel_size": voxel_size,
        "normal_radius_effective": normal_radius,
        "normal_max_nn": int(config.get("gaussian_normal_max_nn", 64)),
        "alpha": alpha,
        "density_opacity": bool(config.get("gaussian_density_opacity", False)),
        "min_alpha": float(config.get("gaussian_min_alpha", 0.35)),
        "max_alpha": float(config.get("gaussian_max_alpha", 0.85)),
        "surface_aligned": bool(config.get("gaussian_surface_aligned", False)),
        "surface_alignment_skipped_reason": surface_alignment_skipped_reason,
        "rotations_are_identity": bool(rotations_are_identity),
        "scale_percentile_low": float(config.get("gaussian_scale_percentile_low", 5.0)),
        "scale_percentile_high": float(config.get("gaussian_scale_percentile_high", 95.0)),
        "nn_scale_mult": float(config.get("gaussian_nn_scale_mult", 0.7)),
        "nn_distance_raw_min": float(positive_nn.min()) if positive_nn.size else 0.0,
        "nn_distance_raw_p10": float(np.percentile(positive_nn, 10)) if positive_nn.size else 0.0,
        "nn_distance_raw_median": float(np.median(positive_nn)) if positive_nn.size else 0.0,
        "nn_distance_raw_p90": float(np.percentile(positive_nn, 90)) if positive_nn.size else 0.0,
        "nn_distance_raw_max": float(positive_nn.max()) if positive_nn.size else 0.0,
        "nn_distance_clip_low": float(nn_clip_low),
        "nn_distance_clip_high": float(nn_clip_high),
        "scale_world_median": float(np.median(scales_world)) if scales_world.size else 0.0,
        "scale_world_p10": float(np.percentile(scales_world, 10)) if scales_world.size else 0.0,
        "scale_world_p90": float(np.percentile(scales_world, 90)) if scales_world.size else 0.0,
        "scale_world_max": float(scales_world.max()) if scales_world.size else 0.0,
        "opacity_alpha_min": float(alpha_values.min()) if alpha_values.size else 0.0,
        "opacity_alpha_median": float(np.median(alpha_values)) if alpha_values.size else 0.0,
        "opacity_alpha_max": float(alpha_values.max()) if alpha_values.size else 0.0,
        "sh_initialization": "dc_only",
        "sh_rest_nonzero": 0,
    }
    (output_dir / "gaussian_splat_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
