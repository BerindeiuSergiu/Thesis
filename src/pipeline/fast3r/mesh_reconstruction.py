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
        raise RuntimeError("Open3D is required for mesh reconstruction.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(_coerce_colors(colors, len(points)).astype(np.float64))
    return pcd


def _cap_pointcloud(pcd, max_points: int, seed: int):
    point_count = len(pcd.points)
    if max_points <= 0 or point_count <= max_points:
        return pcd, False

    rng = np.random.default_rng(seed)
    indices = rng.choice(point_count, size=max_points, replace=False)
    return pcd.select_by_index(indices.tolist()), True


def _remove_small_components(mesh, min_triangles: int, min_ratio: float) -> tuple[object, dict[str, Any]]:
    if len(mesh.triangles) == 0:
        return mesh, {"components_before": 0, "components_removed": 0}
    triangle_clusters, cluster_triangle_counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(triangle_clusters)
    counts = np.asarray(cluster_triangle_counts)
    if counts.size == 0:
        return mesh, {"components_before": 0, "components_removed": 0}

    largest = int(counts.max())
    threshold = max(int(min_triangles), int(np.ceil(largest * float(min_ratio))))
    keep_clusters = counts >= threshold
    mesh.remove_triangles_by_mask(np.logical_not(keep_clusters[clusters]))
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "components_before": int(counts.size),
        "components_removed": int(np.count_nonzero(~keep_clusters)),
        "largest_component_triangles": largest,
        "min_component_threshold": threshold,
    }


def _color_mesh_from_pointcloud(mesh, pcd) -> None:
    if len(mesh.vertices) == 0 or len(pcd.colors) == 0:
        return
    tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = []
    for vertex in mesh.vertices:
        count, indices, _ = tree.search_knn_vector_3d(vertex, knn=1)
        vertex_colors.append(pcd.colors[indices[0]] if count else [0.5, 0.5, 0.5])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=np.float64))


def _mesh_metrics(mesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    metrics = {
        "num_vertices": int(len(vertices)),
        "num_triangles": int(len(triangles)),
        "is_watertight": bool(mesh.is_watertight()) if hasattr(mesh, "is_watertight") else False,
        "is_edge_manifold": bool(mesh.is_edge_manifold()) if hasattr(mesh, "is_edge_manifold") else False,
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()) if hasattr(mesh, "is_vertex_manifold") else False,
        "is_self_intersecting": bool(mesh.is_self_intersecting()) if hasattr(mesh, "is_self_intersecting") else False,
    }
    if len(vertices) and len(triangles):
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        metrics.update(
            {
                "surface_area": float(mesh.get_surface_area()) if hasattr(mesh, "get_surface_area") else 0.0,
                "bbox_extent_x": float(extent[0]),
                "bbox_extent_y": float(extent[1]),
                "bbox_extent_z": float(extent[2]),
            }
        )
        try:
            _, cluster_counts, _ = mesh.cluster_connected_triangles()
            cluster_counts = np.asarray(cluster_counts)
            metrics["num_connected_components"] = int(cluster_counts.size)
            metrics["largest_component_ratio"] = float(cluster_counts.max() / max(len(triangles), 1)) if cluster_counts.size else 0.0
        except Exception:
            metrics["num_connected_components"] = 0
            metrics["largest_component_ratio"] = 0.0
    return metrics


def build_poisson_mesh(
    points: np.ndarray,
    colors: np.ndarray | None,
    output_dir: Path,
    config: dict[str, Any],
    scale_to_meters: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(config.get("mesh_enabled", True)):
        return {"enabled": False, "output_dir": str(output_dir)}
    if points.size == 0:
        return {"enabled": False, "output_dir": str(output_dir), "reason": "empty_pointcloud"}
    if not O3D_AVAILABLE:
        raise RuntimeError("Open3D is required for mesh reconstruction.")

    points = np.asarray(points, dtype=np.float32)
    pcd = _make_pointcloud(points, colors)
    source_points = int(len(pcd.points))

    mesh_voxel_size = float(config.get("mesh_voxel_size", 0.006))
    if mesh_voxel_size > 0 and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size=mesh_voxel_size)
    points_after_mesh_voxel = int(len(pcd.points))

    mesh_max_points = int(config.get("mesh_max_points", 1_000_000))
    pcd, cap_applied = _cap_pointcloud(
        pcd,
        max_points=mesh_max_points,
        seed=int(config.get("mesh_random_seed", 42)),
    )
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        return {
            "enabled": False,
            "output_dir": str(output_dir),
            "reason": "empty_pointcloud_after_mesh_downsample",
            "input_points": source_points,
            "mesh_voxel_size": mesh_voxel_size,
            "mesh_max_points": mesh_max_points,
        }

    diagonal = _bbox_diagonal(points)
    normal_radius = float(config.get("mesh_normal_radius", 0.0))
    if normal_radius <= 0:
        normal_radius = max(diagonal * 0.01, 0.01)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=int(config.get("mesh_normal_max_nn", 64)),
        )
    )
    if bool(config.get("orient_normals", True)) and len(pcd.points) >= 30:
        pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(config.get("poisson_depth", 10)),
        scale=float(config.get("poisson_scale", 1.1)),
        linear_fit=bool(config.get("poisson_linear_fit", False)),
    )

    trim_quantile = float(config.get("poisson_trim_quantile", 0.005))
    if trim_quantile > 0 and len(mesh.vertices) > 0:
        density_arr = np.asarray(densities)
        threshold = float(np.quantile(density_arr, trim_quantile))
        mesh.remove_vertices_by_mask(density_arr < threshold)

    crop_margin = float(config.get("crop_bbox_margin", 0.03))
    if crop_margin > 0 and len(mesh.vertices) > 0:
        bbox = pcd.get_axis_aligned_bounding_box()
        margin = np.asarray(bbox.get_extent()) * crop_margin
        crop_bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox.get_min_bound() - margin,
            max_bound=bbox.get_max_bound() + margin,
        )
        mesh = mesh.crop(crop_bbox)

    mesh, component_report = _remove_small_components(
        mesh,
        min_triangles=int(config.get("min_component_triangles", 1000)),
        min_ratio=float(config.get("min_component_ratio", 0.01)),
    )

    laplacian_iterations = int(config.get("laplacian_iterations", 2))
    if laplacian_iterations > 0 and len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=laplacian_iterations,
            lambda_filter=float(config.get("laplacian_lambda", 0.25)),
        )

    triangles_before_decimation = int(len(mesh.triangles))
    target_triangles = int(config.get("target_triangles", 250000))
    if target_triangles > 0 and triangles_before_decimation > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    if bool(config.get("color_vertices", True)):
        _color_mesh_from_pointcloud(mesh, pcd)

    mesh_ply = output_dir / "mesh_scaled_poisson.ply"
    mesh_obj = output_dir / "mesh_scaled_poisson.obj"
    o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
    o3d.io.write_triangle_mesh(str(mesh_obj), mesh)

    pointcloud_ply = output_dir / "scaled_points_for_mesh.ply"
    o3d.io.write_point_cloud(str(pointcloud_ply), pcd)

    report = {
        "enabled": True,
        "output_dir": str(output_dir),
        "mesh_ply": str(mesh_ply),
        "mesh_obj": str(mesh_obj),
        "pointcloud_ply": str(pointcloud_ply),
        "input_points": source_points,
        "points_after_mesh_voxel": points_after_mesh_voxel,
        "points_used_for_mesh": int(len(pcd.points)),
        "mesh_voxel_size": mesh_voxel_size,
        "mesh_max_points": mesh_max_points,
        "mesh_point_cap_applied": bool(cap_applied),
        "poisson_depth": int(config.get("poisson_depth", 10)),
        "poisson_scale": float(config.get("poisson_scale", 1.1)),
        "poisson_trim_quantile": trim_quantile,
        "crop_bbox_margin": crop_margin,
        "components": component_report,
        "laplacian_iterations": laplacian_iterations,
        "laplacian_lambda": float(config.get("laplacian_lambda", 0.25)),
        "target_triangles": target_triangles,
        "triangles_before_decimation": triangles_before_decimation,
        "scale_to_meters_already_applied": float(scale_to_meters),
        "mesh_metrics": _mesh_metrics(mesh),
    }
    (output_dir / "mesh_branch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
