"""Poisson mesh reconstruction branch for scaled beta outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import compute_mesh_metrics
from .pointcloud_ops import O3D_AVAILABLE, bbox_diagonal, coerce_colors, make_pointcloud, o3d
from .reconstruction_data import ReconstructionOutput


@dataclass
class MeshBranchConfig:
    """Configuration-driven Poisson surface reconstruction and cleanup."""

    enabled: bool = True
    output_dir_name: str = "mesh"
    poisson_depth: int = 10
    poisson_scale: float = 1.1
    poisson_linear_fit: bool = False
    poisson_trim_quantile: float = 0.005
    crop_bbox_margin: float = 0.03
    orient_normals: bool = True
    normal_radius: float = 0.0
    normal_max_nn: int = 64
    min_component_triangles: int = 1000
    min_component_ratio: float = 0.01
    laplacian_iterations: int = 2
    laplacian_lambda: float = 0.25
    target_triangles: int = 250000
    color_vertices: bool = True
    save_filtered_pointcloud: bool = True


def _color_mesh_from_pointcloud(mesh, pcd) -> None:
    if len(mesh.vertices) == 0 or len(pcd.colors) == 0:
        return

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = []
    for vertex in mesh.vertices:
        k, idx, _ = pcd_tree.search_knn_vector_3d(vertex, knn=1)
        if k >= 1:
            vertex_colors.append(pcd.colors[idx[0]])
        else:
            vertex_colors.append([0.5, 0.5, 0.5])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=np.float64))


def _remove_small_components(mesh, min_triangles: int, min_ratio: float) -> tuple[object, dict]:
    if len(mesh.triangles) == 0:
        return mesh, {"components_before": 0, "components_removed": 0}

    triangle_clusters, cluster_triangle_counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(triangle_clusters)
    counts = np.asarray(cluster_triangle_counts)
    if counts.size == 0:
        return mesh, {"components_before": 0, "components_removed": 0}

    largest = int(counts.max())
    keep_cluster = counts >= max(int(min_triangles), int(np.ceil(largest * float(min_ratio))))
    remove_triangles = np.logical_not(keep_cluster[clusters])
    removed_components = int(np.count_nonzero(~keep_cluster))
    mesh.remove_triangles_by_mask(remove_triangles)
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "components_before": int(counts.size),
        "components_removed": removed_components,
        "largest_component_triangles": largest,
    }


def mesh_branch(
    reconstruction: ReconstructionOutput,
    output_root: Path,
    config: Optional[MeshBranchConfig] = None,
) -> dict:
    """Run Poisson surface reconstruction and ordered cleanup on scaled points."""

    config = config or MeshBranchConfig()
    output_dir = Path(output_root) / config.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.enabled:
        return {"enabled": False, "output_dir": str(output_dir)}
    if not O3D_AVAILABLE:
        raise RuntimeError("Open3D is required for mesh reconstruction.")

    points = np.asarray(reconstruction.points, dtype=np.float32)
    colors = coerce_colors(reconstruction.colors, points.shape[0])
    pcd = make_pointcloud(points, colors)

    diag = bbox_diagonal(points)
    normal_radius = (
        float(config.normal_radius)
        if config.normal_radius > 0
        else max(diag * 0.01, 0.01)
    )
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(config.normal_max_nn),
        )
    )
    if config.orient_normals and len(pcd.points) >= 30:
        pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(config.poisson_depth),
        scale=float(config.poisson_scale),
        linear_fit=bool(config.poisson_linear_fit),
    )

    if config.poisson_trim_quantile > 0 and len(mesh.vertices) > 0:
        density_arr = np.asarray(densities)
        threshold = float(np.quantile(density_arr, float(config.poisson_trim_quantile)))
        mesh.remove_vertices_by_mask(density_arr < threshold)

    if config.crop_bbox_margin > 0 and len(mesh.vertices) > 0:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent())
        margin = extent * float(config.crop_bbox_margin)
        crop_bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox.get_min_bound() - margin,
            max_bound=bbox.get_max_bound() + margin,
        )
        mesh = mesh.crop(crop_bbox)

    mesh, component_report = _remove_small_components(
        mesh,
        min_triangles=int(config.min_component_triangles),
        min_ratio=float(config.min_component_ratio),
    )

    if config.laplacian_iterations > 0 and len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=int(config.laplacian_iterations),
            lambda_filter=float(config.laplacian_lambda),
        )

    triangles_before_decimation = int(len(mesh.triangles))
    if config.target_triangles > 0 and triangles_before_decimation > int(config.target_triangles):
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(config.target_triangles))

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    if config.color_vertices:
        _color_mesh_from_pointcloud(mesh, pcd)

    mesh_ply = output_dir / "mesh_scaled_poisson.ply"
    mesh_obj = output_dir / "mesh_scaled_poisson.obj"
    o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
    o3d.io.write_triangle_mesh(str(mesh_obj), mesh)

    pcd_path = ""
    if config.save_filtered_pointcloud:
        pcd_path_obj = output_dir / "scaled_points_for_mesh.ply"
        o3d.io.write_point_cloud(str(pcd_path_obj), pcd)
        pcd_path = str(pcd_path_obj)

    metrics = compute_mesh_metrics(mesh)
    report = {
        "enabled": True,
        "output_dir": str(output_dir),
        "mesh_ply": str(mesh_ply),
        "mesh_obj": str(mesh_obj),
        "pointcloud_ply": pcd_path,
        "input_points": int(points.shape[0]),
        "poisson_depth": int(config.poisson_depth),
        "poisson_scale": float(config.poisson_scale),
        "poisson_linear_fit": bool(config.poisson_linear_fit),
        "poisson_trim_quantile": float(config.poisson_trim_quantile),
        "crop_bbox_margin": float(config.crop_bbox_margin),
        "normal_radius_effective": float(normal_radius),
        "normal_max_nn": int(config.normal_max_nn),
        "components": component_report,
        "laplacian_iterations": int(config.laplacian_iterations),
        "laplacian_lambda": float(config.laplacian_lambda),
        "target_triangles": int(config.target_triangles),
        "triangles_before_decimation": triangles_before_decimation,
        "scale_to_meters_already_applied": float(reconstruction.scale_applied),
        "mesh_metrics": metrics,
    }
    (output_dir / "mesh_branch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
