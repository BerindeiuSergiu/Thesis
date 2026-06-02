"""Gaussian splat branch for scaled reconstruction outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .pointcloud_ops import O3D_AVAILABLE, bbox_diagonal, coerce_colors, make_pointcloud, o3d
from .reconstruction_data import ReconstructionOutput


C0 = 0.28209479177387814


@dataclass
class GaussianSplatConfig:
    """Controls conversion from scaled point cloud to Gaussian init PLY."""

    enabled: bool = True
    output_dir_name: str = "gaussian_splat"
    output_name: str = "gaussian_splats_init.ply"
    save_npz: bool = True
    save_pointcloud_ply: bool = True
    voxel_size: float = 0.0
    normal_radius: float = 0.0
    normal_max_nn: int = 64
    nn_scale_mult: float = 0.7
    min_scale: float = 0.00025
    max_scale: float = 0.02
    alpha: float = 0.85
    scale_percentile_low: float = 5.0
    scale_percentile_high: float = 95.0
    density_opacity: bool = True
    min_alpha: float = 0.35
    max_alpha: float = 0.85
    surface_aligned: bool = True
    tangent_scale_mult: float = 1.15
    normal_scale_mult: float = 0.25
    max_surface_aligned_points: int = 8000000


def _normalize_vectors(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-8)
    out = np.tile(fallback.astype(np.float32), (vectors.shape[0], 1))
    out[valid] = vectors[valid] / norms[valid]
    return out.astype(np.float32)


def _rotation_matrices_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    """Convert local-to-world rotation matrices to scalar-first quaternions."""

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
    quats[~valid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quats.astype(np.float32)


def _surface_aligned_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build stable local tangent frames with local z aligned to the normal."""

    normal = _normalize_vectors(normals, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (normal.shape[0], 1))
    near_z = np.abs(normal[:, 2]) > 0.9
    ref[near_z] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    tangent0 = np.cross(ref, normal)
    tangent0 = _normalize_vectors(tangent0, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    tangent1 = np.cross(normal, tangent0)
    tangent1 = _normalize_vectors(tangent1, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    matrices = np.stack([tangent0, tangent1, normal], axis=2).astype(np.float32)
    quats = _rotation_matrices_to_quaternions(matrices)
    return matrices, quats


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
    n = points.shape[0]
    if n == 0:
        raise ValueError("No points available for Gaussian splat export.")

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
    for idx in range(sh_rest.shape[1]):
        dtype.append((f"f_rest_{idx}", "<f4"))
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
    for idx in range(sh_rest.shape[1]):
        arr[f"f_rest_{idx}"] = sh_rest[:, idx].astype(np.float32)
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
        "comment Generated by beta dual-output pipeline",
        f"element vertex {n}",
    ]
    for field_name, _ in dtype:
        header_lines.append(f"property float {field_name}")
    header_lines.append("end_header")

    with path.open("wb") as handle:
        handle.write(("\n".join(header_lines) + "\n").encode("ascii"))
        arr.tofile(handle)


def gaussian_splat_branch(
    reconstruction: ReconstructionOutput,
    output_root: Path,
    config: Optional[GaussianSplatConfig] = None,
) -> dict:
    """Generate a real-time friendly Gaussian splat initialization scene."""

    config = config or GaussianSplatConfig()
    output_dir = Path(output_root) / config.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.enabled:
        return {"enabled": False, "output_dir": str(output_dir)}
    if not O3D_AVAILABLE:
        raise RuntimeError("Open3D is required for Gaussian splat normal and spacing estimation.")

    points = np.asarray(reconstruction.points, dtype=np.float32)
    colors = coerce_colors(reconstruction.colors, points.shape[0])
    pcd = make_pointcloud(points, colors)

    if config.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(config.voxel_size))

    points_out = np.asarray(pcd.points, dtype=np.float32)
    colors_out = coerce_colors(np.asarray(pcd.colors), points_out.shape[0])
    diag = bbox_diagonal(points_out)
    normal_radius = (
        float(config.normal_radius)
        if config.normal_radius > 0
        else max(float(config.voxel_size) * 3.0, diag * 0.01, 1e-4)
    )
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(config.normal_max_nn),
        )
    )

    normals = np.asarray(pcd.normals, dtype=np.float32)
    if normals.shape != points_out.shape:
        normals = np.zeros_like(points_out, dtype=np.float32)

    nn_dist = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float32)
    if nn_dist.shape[0] != points_out.shape[0]:
        nn_dist = np.full(points_out.shape[0], max(float(config.min_scale), 1e-6), dtype=np.float32)
    nn_dist = np.nan_to_num(
        nn_dist,
        nan=float(config.min_scale),
        posinf=float(config.max_scale),
        neginf=float(config.min_scale),
    )
    positive_nn = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
    if positive_nn.size:
        pct_low = float(np.clip(config.scale_percentile_low, 0.0, 100.0))
        pct_high = float(np.clip(config.scale_percentile_high, pct_low, 100.0))
        nn_clip_low = float(np.percentile(positive_nn, pct_low))
        nn_clip_high = float(np.percentile(positive_nn, pct_high))
        if nn_clip_high <= nn_clip_low:
            nn_clip_high = float(max(nn_clip_low, np.median(positive_nn), config.min_scale))
    else:
        nn_clip_low = float(config.min_scale)
        nn_clip_high = float(config.max_scale)
    nn_dist_robust = np.clip(nn_dist, nn_clip_low, nn_clip_high)

    sh_dc = ((colors_out - 0.5) / C0).astype(np.float32)
    sh_rest = np.zeros((points_out.shape[0], 45), dtype=np.float32)

    base_scales_world = np.clip(
        nn_dist_robust * float(config.nn_scale_mult),
        float(config.min_scale),
        float(config.max_scale),
    )
    if config.surface_aligned:
        tangent_mult = max(float(config.tangent_scale_mult), 1e-4)
        normal_mult = max(float(config.normal_scale_mult), 1e-4)
        scale_xyz_world = np.stack(
            [
                np.clip(base_scales_world * tangent_mult, float(config.min_scale), float(config.max_scale)),
                np.clip(base_scales_world * tangent_mult, float(config.min_scale), float(config.max_scale)),
                np.clip(base_scales_world * normal_mult, float(config.min_scale), float(config.max_scale)),
            ],
            axis=1,
        ).astype(np.float32)
    else:
        scale_xyz_world = np.repeat(base_scales_world[:, None], 3, axis=1).astype(np.float32)
    scales = np.log(np.clip(scale_xyz_world, 1e-8, None)).astype(np.float32)

    if config.density_opacity and points_out.shape[0] > 0 and nn_clip_high > nn_clip_low:
        dense_score = 1.0 - (nn_dist_robust - nn_clip_low) / (nn_clip_high - nn_clip_low + 1e-8)
        alpha_min = np.clip(float(config.min_alpha), 1e-4, 1.0 - 1e-4)
        alpha_max = np.clip(float(config.max_alpha), alpha_min, 1.0 - 1e-4)
        alpha_values = alpha_min + np.clip(dense_score, 0.0, 1.0) * (alpha_max - alpha_min)
    else:
        alpha = np.clip(float(config.alpha), 1e-4, 1.0 - 1e-4)
        alpha_values = np.full(points_out.shape[0], alpha, dtype=np.float32)
    opacity = np.log(alpha_values / (1.0 - alpha_values)).astype(np.float32)

    surface_alignment_skipped_reason = ""
    if (
        config.surface_aligned
        and points_out.shape[0] > 0
        and (int(config.max_surface_aligned_points) <= 0 or points_out.shape[0] <= int(config.max_surface_aligned_points))
    ):
        _, rotations = _surface_aligned_frames(normals)
        rotations_are_identity = False
    else:
        rotations = np.zeros((points_out.shape[0], 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        rotations_are_identity = True
        if config.surface_aligned and points_out.shape[0] > int(config.max_surface_aligned_points) > 0:
            surface_alignment_skipped_reason = "too_many_points_for_memory_safe_rotation_export"

    gaussian_ply = output_dir / config.output_name
    _write_gaussian_ply(
        gaussian_ply,
        points=points_out,
        normals=normals,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
    )

    pointcloud_ply = ""
    if config.save_pointcloud_ply:
        pointcloud_path = output_dir / "scaled_points_for_gaussian.ply"
        o3d.io.write_point_cloud(str(pointcloud_path), pcd)
        pointcloud_ply = str(pointcloud_path)

    npz_path = ""
    if config.save_npz:
        npz_out = output_dir / "gaussian_params.npz"
        np.savez_compressed(
            npz_out,
            points=points_out,
            colors=colors_out,
            normals=normals,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
        )
        npz_path = str(npz_out)

    report = {
        "enabled": True,
        "output_dir": str(output_dir),
        "gaussian_ply": str(gaussian_ply),
        "gaussian_npz": npz_path,
        "pointcloud_ply": pointcloud_ply,
        "input_points": int(points.shape[0]),
        "output_gaussians": int(points_out.shape[0]),
        "voxel_size": float(config.voxel_size),
        "normal_radius_effective": float(normal_radius),
        "normal_max_nn": int(config.normal_max_nn),
        "nn_scale_mult": float(config.nn_scale_mult),
        "min_scale": float(config.min_scale),
        "max_scale": float(config.max_scale),
        "alpha": float(config.alpha),
        "density_opacity": bool(config.density_opacity),
        "min_alpha": float(config.min_alpha),
        "max_alpha": float(config.max_alpha),
        "surface_aligned": bool(config.surface_aligned),
        "max_surface_aligned_points": int(config.max_surface_aligned_points),
        "surface_alignment_skipped_reason": surface_alignment_skipped_reason,
        "rotations_are_identity": bool(rotations_are_identity),
        "tangent_scale_mult": float(config.tangent_scale_mult),
        "normal_scale_mult": float(config.normal_scale_mult),
        "scale_percentile_low": float(config.scale_percentile_low),
        "scale_percentile_high": float(config.scale_percentile_high),
        "nn_distance_raw_min": float(positive_nn.min()) if positive_nn.size else 0.0,
        "nn_distance_raw_p10": float(np.percentile(positive_nn, 10)) if positive_nn.size else 0.0,
        "nn_distance_raw_median": float(np.median(positive_nn)) if positive_nn.size else 0.0,
        "nn_distance_raw_mean": float(np.mean(positive_nn)) if positive_nn.size else 0.0,
        "nn_distance_raw_p90": float(np.percentile(positive_nn, 90)) if positive_nn.size else 0.0,
        "nn_distance_raw_max": float(positive_nn.max()) if positive_nn.size else 0.0,
        "nn_distance_clip_low": float(nn_clip_low),
        "nn_distance_clip_high": float(nn_clip_high),
        "scale_to_meters_already_applied": float(reconstruction.scale_applied),
        "base_scale_world_min": float(base_scales_world.min()) if base_scales_world.size else 0.0,
        "base_scale_world_p10": float(np.percentile(base_scales_world, 10)) if base_scales_world.size else 0.0,
        "base_scale_world_median": float(np.median(base_scales_world)) if base_scales_world.size else 0.0,
        "base_scale_world_mean": float(np.mean(base_scales_world)) if base_scales_world.size else 0.0,
        "base_scale_world_p90": float(np.percentile(base_scales_world, 90)) if base_scales_world.size else 0.0,
        "base_scale_world_max": float(base_scales_world.max()) if base_scales_world.size else 0.0,
        "scale_x_world_median": float(np.median(scale_xyz_world[:, 0])) if scale_xyz_world.size else 0.0,
        "scale_y_world_median": float(np.median(scale_xyz_world[:, 1])) if scale_xyz_world.size else 0.0,
        "scale_z_world_median": float(np.median(scale_xyz_world[:, 2])) if scale_xyz_world.size else 0.0,
        "anisotropy_ratio_median": float(np.median(np.max(scale_xyz_world, axis=1) / np.maximum(np.min(scale_xyz_world, axis=1), 1e-8))) if scale_xyz_world.size else 1.0,
        "anisotropy_ratio_p90": float(np.percentile(np.max(scale_xyz_world, axis=1) / np.maximum(np.min(scale_xyz_world, axis=1), 1e-8), 90)) if scale_xyz_world.size else 1.0,
        "opacity_alpha_min": float(alpha_values.min()) if alpha_values.size else 0.0,
        "opacity_alpha_median": float(np.median(alpha_values)) if alpha_values.size else 0.0,
        "opacity_alpha_max": float(alpha_values.max()) if alpha_values.size else 0.0,
        "sh_initialization": "dc_only",
        "sh_rest_nonzero": 0,
    }
    (output_dir / "gaussian_splat_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
