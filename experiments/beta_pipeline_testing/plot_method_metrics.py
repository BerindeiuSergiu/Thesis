"""Plot CSV metrics emitted by the beta reconstruction pipelines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BETA_ROOT = Path(__file__).resolve().parent


def coerce_value(value: str):
    """Convert CSV strings into ints/floats/bools when possible."""
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if not isinstance(value, str):
        return value
    if value in ("", None):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    if "_" in value:
        return value
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def load_rows(csv_path: Path, run_id: str | None = None) -> list[dict]:
    """Load and optionally filter CSV rows."""
    if not csv_path.exists():
        return []

    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        rows = [
            {key: coerce_value(value) for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]

    if run_id is not None:
        rows = [row for row in rows if row.get("run_id") == run_id]
    return rows


def latest_run_id(data_root: Path) -> str:
    """Return the latest run ID recorded in run_summary.csv."""
    summary_path = data_root / "run_overview" / "run_summary.csv"
    rows = load_rows(summary_path)
    if not rows:
        raise FileNotFoundError(f"No runs found in {summary_path}")
    return rows[-1]["run_id"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, output_path: Path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stage_timings(rows: list[dict], output_dir: Path):
    if not rows:
        return
    stages = [row["stage_name"] for row in rows]
    durations = [float(row["elapsed_seconds"]) for row in rows]
    peaks = [float(row["gpu_peak_memory_gb"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(stages, durations, color="#2f6b7c", alpha=0.85, label="Elapsed seconds")
    ax1.set_ylabel("Elapsed seconds")
    ax1.set_title("Stage Timing and GPU Memory")
    ax1.tick_params(axis="x", rotation=25)

    ax2 = ax1.twinx()
    ax2.plot(stages, peaks, color="#d95f02", marker="o", linewidth=2, label="Peak GPU memory (GB)")
    ax2.set_ylabel("Peak GPU memory (GB)")

    save_figure(fig, output_dir / "stage_timings.png")


def plot_frame_quality(frame_rows: list[dict], pair_rows: list[dict], output_dir: Path):
    if frame_rows:
        x = [int(row["frame_local_index"]) for row in frame_rows]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes = axes.ravel()
        frame_columns = [
            ("laplacian_variance", "Sharpness (Laplacian var)", "#1b9e77"),
            ("entropy_bits", "Entropy (bits)", "#d95f02"),
            ("edge_density", "Edge density", "#7570b3"),
            ("orb_keypoints", "ORB keypoints", "#e7298a"),
        ]
        for ax, (column, title, color) in zip(axes, frame_columns):
            y = [float(row[column]) for row in frame_rows]
            ax.plot(x, y, color=color, marker="o", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Frame index")
        save_figure(fig, output_dir / "frame_quality.png")

    if pair_rows:
        x = [int(row["frame_local_index_a"]) for row in pair_rows]
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(x, [float(row["normalized_abs_diff"]) for row in pair_rows], color="#386cb0", marker="o")
        axes[0].set_ylabel("Normalized abs diff")
        axes[0].set_title("Consecutive-Frame Change")
        axes[1].plot(x, [float(row["orb_matches"]) for row in pair_rows], color="#f0027f", marker="o")
        axes[1].set_ylabel("ORB matches")
        axes[1].set_xlabel("Pair start frame")
        save_figure(fig, output_dir / "frame_overlap.png")


def plot_pair_graph(pair_rows: list[dict], output_dir: Path):
    if not pair_rows:
        return
    frame_gaps = np.array([float(row["frame_gap"]) for row in pair_rows], dtype=np.float64)
    frame_ids = sorted(
        set(int(row["frame_idx_a"]) for row in pair_rows) | set(int(row["frame_idx_b"]) for row in pair_rows)
    )
    degree_map = {frame_id: 0 for frame_id in frame_ids}
    for row in pair_rows:
        degree_map[int(row["frame_idx_a"])] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(frame_gaps, bins=min(20, max(5, len(np.unique(frame_gaps)))), color="#66a61e", alpha=0.85)
    axes[0].set_title("Pair-Graph Frame Gap Histogram")
    axes[0].set_xlabel("Frame gap")
    axes[0].set_ylabel("Count")
    axes[1].bar(list(degree_map.keys()), list(degree_map.values()), color="#e6ab02")
    axes[1].set_title("Pair-Graph Out Degree by Frame")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Outgoing edges")
    save_figure(fig, output_dir / "pair_graph.png")


def plot_depth_confidence(rows: list[dict], output_dir: Path):
    if not rows:
        return
    x = [int(row["frame_local_index"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()
    specs = [
        ("confidence_mean", "Mean confidence", "#1f78b4"),
        ("confidence_ratio_above_threshold", "Confidence ratio > threshold", "#33a02c"),
        ("depth_median", "Median positive depth", "#e31a1c"),
        ("points_above_conf_threshold", "Points above threshold", "#ff7f00"),
    ]
    for ax, (column, title, color) in zip(axes, specs):
        ax.plot(x, [float(row[column]) for row in rows], color=color, marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Frame index")
    save_figure(fig, output_dir / "depth_confidence.png")


def plot_pose_metrics(trajectory_rows: list[dict], step_rows: list[dict], output_dir: Path):
    if not trajectory_rows:
        return
    x = [int(row["frame_local_index"]) for row in trajectory_rows]
    tx = [float(row["tx"]) for row in trajectory_rows]
    ty = [float(row["ty"]) for row in trajectory_rows]
    tz = [float(row["tz"]) for row in trajectory_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(x, tx, label="tx", color="#1b9e77")
    axes[0, 0].plot(x, ty, label="ty", color="#d95f02")
    axes[0, 0].plot(x, tz, label="tz", color="#7570b3")
    axes[0, 0].set_title("Pose Translation Components")
    axes[0, 0].set_xlabel("Frame index")
    axes[0, 0].legend()

    axes[0, 1].plot(tx, tz, color="#66a61e", marker="o")
    axes[0, 1].set_title("Top-Down Trajectory (X/Z)")
    axes[0, 1].set_xlabel("tx")
    axes[0, 1].set_ylabel("tz")

    if step_rows:
        step_x = [int(row["frame_local_index_b"]) for row in step_rows]
        axes[1, 0].plot(step_x, [float(row["translation_step"]) for row in step_rows], color="#e7298a", marker="o")
        axes[1, 0].set_title("Translation Step Size")
        axes[1, 0].set_xlabel("Frame index")

        axes[1, 1].plot(step_x, [float(row["rotation_step_deg"]) for row in step_rows], color="#a6761d", marker="o")
        axes[1, 1].set_title("Rotation Step (deg)")
        axes[1, 1].set_xlabel("Frame index")

    save_figure(fig, output_dir / "pose_metrics.png")


def plot_pointcloud_metrics(stage_rows: list[dict], geometry_rows: list[dict], output_dir: Path):
    if stage_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        stage_names = [row["stage_name"] for row in stage_rows]
        counts = [float(row["num_points"]) for row in stage_rows]
        retentions = [float(row["retention_ratio"]) for row in stage_rows]
        axes[0].bar(stage_names, counts, color="#2b8cbe")
        axes[0].set_title("Point Count by Processing Stage")
        axes[0].tick_params(axis="x", rotation=20)
        axes[1].bar(stage_names, retentions, color="#7bccc4")
        axes[1].set_title("Retention Ratio by Stage")
        axes[1].tick_params(axis="x", rotation=20)
        save_figure(fig, output_dir / "pointcloud_stages.png")

    if geometry_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        stage_names = [row["stage_name"] for row in geometry_rows]
        extents_x = [float(row["bbox_extent_x"]) for row in geometry_rows]
        extents_y = [float(row["bbox_extent_y"]) for row in geometry_rows]
        extents_z = [float(row["bbox_extent_z"]) for row in geometry_rows]
        nn_medians = [float(row.get("nn_median", 0.0)) for row in geometry_rows]
        axes[0].plot(stage_names, extents_x, marker="o", label="extent_x")
        axes[0].plot(stage_names, extents_y, marker="o", label="extent_y")
        axes[0].plot(stage_names, extents_z, marker="o", label="extent_z")
        axes[0].set_title("Point Cloud Bounding Box Extents")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].legend()
        axes[1].bar(stage_names, nn_medians, color="#dd1c77")
        axes[1].set_title("Median Nearest-Neighbor Distance")
        axes[1].tick_params(axis="x", rotation=20)
        save_figure(fig, output_dir / "pointcloud_geometry.png")


def plot_mesh_metrics(rows: list[dict], output_dir: Path):
    if not rows:
        return
    row = rows[-1]
    labels = ["vertices", "triangles", "components"]
    values = [
        float(row.get("num_vertices", 0.0)),
        float(row.get("num_triangles", 0.0)),
        float(row.get("num_connected_components", 0.0)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, values, color=["#3182bd", "#9ecae1", "#31a354"])
    axes[0].set_title("Mesh Topology Summary")

    mesh_quality = {
        "surface_area": float(row.get("surface_area", 0.0)),
        "largest_component_ratio": float(row.get("largest_component_ratio", 0.0)),
        "edge_length_median": float(row.get("edge_length_median", 0.0)),
        "triangle_area_median": float(row.get("triangle_area_median", 0.0)),
        "aspect_ratio_p95": float(row.get("triangle_aspect_ratio_p95", 0.0)),
    }
    axes[1].bar(mesh_quality.keys(), mesh_quality.values(), color="#756bb1")
    axes[1].set_title("Mesh Quality Proxies")
    axes[1].tick_params(axis="x", rotation=30)
    save_figure(fig, output_dir / "mesh_metrics.png")


def plot_run_overview(summary_rows: list[dict], output_dir: Path):
    if not summary_rows:
        return
    row = summary_rows[-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    lines = [
        f"Run ID: {row.get('run_id', 'unknown')}",
        f"Status: {row.get('status', 'unknown')}",
        f"Total runtime (s): {row.get('total_runtime_seconds', 0.0)}",
        f"Frames extracted: {row.get('num_frames_extracted', 0)}",
        f"Raw points: {row.get('raw_points', 0)}",
        f"Mesh vertices: {row.get('mesh_vertices', 0)}",
        f"Mesh triangles: {row.get('mesh_triangles', 0)}",
        f"Pair count: {row.get('pair_count', 0)}",
        f"Final alignment loss: {row.get('final_alignment_loss', 0.0)}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
    ax.set_title("Run Overview")
    save_figure(fig, output_dir / "run_overview.png")


def main():
    parser = argparse.ArgumentParser(description="Plot beta pipeline CSV metrics")
    parser.add_argument("--method", default="dust3r", help="Method name, e.g. dust3r")
    parser.add_argument("--run-id", default="latest", help="Specific run ID or 'latest'")
    args = parser.parse_args()

    data_root = BETA_ROOT / f"{args.method}_data"
    run_id = latest_run_id(data_root) if args.run_id == "latest" else args.run_id
    output_dir = ensure_dir(data_root / "plots" / run_id)

    summary_rows = load_rows(data_root / "run_overview" / "run_summary.csv", run_id=run_id)
    stage_rows = load_rows(data_root / "system" / "stage_timings.csv", run_id=run_id)
    frame_rows = load_rows(data_root / "frames" / "frame_metrics.csv", run_id=run_id)
    frame_pair_rows = load_rows(data_root / "frames" / "frame_pair_metrics.csv", run_id=run_id)
    pair_graph_rows = load_rows(data_root / "reconstruction" / "pair_graph.csv", run_id=run_id)
    depth_rows = load_rows(data_root / "depth" / "depth_confidence_metrics.csv", run_id=run_id)
    pose_rows = load_rows(data_root / "poses" / "pose_trajectory.csv", run_id=run_id)
    pose_step_rows = load_rows(data_root / "poses" / "pose_step_metrics.csv", run_id=run_id)
    pointcloud_stage_rows = load_rows(data_root / "pointcloud" / "pointcloud_stage_metrics.csv", run_id=run_id)
    pointcloud_geometry_rows = load_rows(data_root / "pointcloud" / "pointcloud_geometry_metrics.csv", run_id=run_id)
    mesh_rows = load_rows(data_root / "mesh" / "mesh_metrics.csv", run_id=run_id)

    plot_run_overview(summary_rows, output_dir)
    plot_stage_timings(stage_rows, output_dir)
    plot_frame_quality(frame_rows, frame_pair_rows, output_dir)
    plot_pair_graph(pair_graph_rows, output_dir)
    plot_depth_confidence(depth_rows, output_dir)
    plot_pose_metrics(pose_rows, pose_step_rows, output_dir)
    plot_pointcloud_metrics(pointcloud_stage_rows, pointcloud_geometry_rows, output_dir)
    plot_mesh_metrics(mesh_rows, output_dir)

    print(f"Wrote plots for run {run_id} to {output_dir}")


if __name__ == "__main__":
    main()
