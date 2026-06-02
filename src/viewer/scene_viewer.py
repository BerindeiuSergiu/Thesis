from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

try:
    import open3d as o3d

    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


def _gaussian_pointcloud_from_scene(scene_result) -> Path | None:
    metadata = getattr(scene_result, "metadata", {}) or {}
    gaussian_report = metadata.get("gaussian_splat", {}) if isinstance(metadata, dict) else {}
    candidates = [
        gaussian_report.get("pointcloud_ply") if isinstance(gaussian_report, dict) else None,
        Path(scene_result.output_dir) / "gaussian_splat" / "scaled_points_for_gaussian.ply",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _find_free_port(start_port: int = 8080, end_port: int = 8999) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free viewer port found in range {start_port}-{end_port}.")


def launch_scene_viewer(scene_result, preferred_viewer: str = "viser", geometry_mode: str = "auto") -> subprocess.Popen[str]:
    pointcloud = str(scene_result.pointcloud_path) if scene_result.pointcloud_path else ""
    mesh = str(scene_result.mesh_path) if scene_result.mesh_path else ""
    gaussian_pointcloud = _gaussian_pointcloud_from_scene(scene_result)
    if geometry_mode == "gaussian" and gaussian_pointcloud is not None:
        pointcloud = str(gaussian_pointcloud)
    port = _find_free_port()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preferred-viewer",
        preferred_viewer,
        "--geometry-mode",
        geometry_mode,
        "--title",
        scene_result.scene_id,
        "--port",
        str(port),
    ]
    if pointcloud:
        command.extend(["--pointcloud", pointcloud])
    if mesh:
        command.extend(["--mesh", mesh])
    return subprocess.Popen(command)


def _resolve_geometry_mode(pointcloud_path: Path | None, mesh_path: Path | None, geometry_mode: str) -> tuple[bool, bool]:
    has_pointcloud = pointcloud_path is not None and pointcloud_path.exists()
    has_mesh = mesh_path is not None and mesh_path.exists()
    if geometry_mode in {"pointcloud", "gaussian"}:
        return has_pointcloud, False
    if geometry_mode == "mesh":
        return False, has_mesh
    if geometry_mode == "both":
        return has_pointcloud, has_mesh
    return (False, has_mesh) if has_mesh else (has_pointcloud, False)


def _run_open3d_viewer(title: str, pointcloud_path: Path | None, mesh_path: Path | None, geometry_mode: str) -> None:
    if not O3D_AVAILABLE:
        raise RuntimeError("open3d is not installed.")

    geometries = []
    show_pointcloud, show_mesh = _resolve_geometry_mode(pointcloud_path, mesh_path, geometry_mode)
    if show_pointcloud and pointcloud_path is not None:
        geometries.append(o3d.io.read_point_cloud(str(pointcloud_path)))
    if show_mesh and mesh_path is not None:
        geometries.append(o3d.io.read_triangle_mesh(str(mesh_path)))
    if not geometries:
        raise RuntimeError("No geometry was available to display.")

    o3d.visualization.draw_geometries(geometries, window_name=title)


def _run_viser_viewer(
    title: str,
    pointcloud_path: Path | None,
    mesh_path: Path | None,
    geometry_mode: str,
    port: int,
) -> None:
    import numpy as np
    import viser

    if not O3D_AVAILABLE:
        raise RuntimeError("viser mode requires open3d for geometry loading.")

    server = viser.ViserServer(port=int(port))
    show_pointcloud, show_mesh = _resolve_geometry_mode(pointcloud_path, mesh_path, geometry_mode)

    if show_pointcloud and pointcloud_path is not None:
        pcd = o3d.io.read_point_cloud(str(pointcloud_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if len(pcd.colors) else None
        server.scene.add_point_cloud(
            name="/scene/pointcloud",
            points=points,
            colors=colors,
            point_size=0.003,
        )

    if show_mesh and mesh_path is not None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        server.scene.add_mesh_simple(
            name="/scene/mesh",
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            vertex_colors=np.asarray(mesh.vertex_colors) if len(mesh.vertex_colors) else None,
        )

    url = f"http://127.0.0.1:{int(port)}"

    webbrowser.open(url)
    print(f"{title} viewer running at: {url}", flush=True)
    while True:
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a local scene viewer.")
    parser.add_argument("--preferred-viewer", default="viser", choices=["viser", "open3d"])
    parser.add_argument("--geometry-mode", default="auto", choices=["auto", "mesh", "pointcloud", "gaussian", "both"])
    parser.add_argument("--pointcloud", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Scene Viewer")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    try:
        if args.preferred_viewer == "viser":
            try:
                _run_viser_viewer(args.title, args.pointcloud, args.mesh, args.geometry_mode, args.port)
            except Exception:
                _run_open3d_viewer(args.title, args.pointcloud, args.mesh, args.geometry_mode)
        else:
            _run_open3d_viewer(args.title, args.pointcloud, args.mesh, args.geometry_mode)
    except Exception as exc:
        print(f"Failed to launch scene viewer: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
