r"""Compare latest archived runs across models.

By default this compares the latest `dust3r` and `fast3r` runs from:
    D:\GitRepos\runs_pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


DEFAULT_RUNS_ROOT = Path(r"D:\GitRepos\runs_pipeline")
DEFAULT_MODELS = ("dust3r", "fast3r")


@dataclass
class RunComparisonRow:
    model: str
    run_folder: str
    run_id: str
    status: str
    total_runtime_seconds: Optional[float]
    reconstruction_seconds: Optional[float]
    mesh_seconds: Optional[float]
    peak_gpu_memory_gb: Optional[float]
    frames: Optional[int]
    raw_points: Optional[int]
    final_points: Optional[int]
    mesh_vertices: Optional[int]
    mesh_triangles: Optional[int]
    device: str
    torch_version: str


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def last_row(path: Path) -> dict:
    rows = read_csv_rows(path)
    return rows[-1] if rows else {}


def as_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def find_latest_run_for_model(runs_root: Path, model: str) -> Optional[Path]:
    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir() and path.name.startswith(tuple(str(i) for i in range(10))) and f"_{model}" in path.name
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def pick_stage_seconds(stage_rows: list[dict], stage_name: str) -> Optional[float]:
    for row in stage_rows:
        if row.get("stage_name") == stage_name:
            return as_float(row.get("elapsed_seconds"))
    return None


def pick_final_points(pointcloud_stage_rows: list[dict]) -> Optional[int]:
    for row in pointcloud_stage_rows:
        if row.get("stage_name") == "after_voxel_downsample":
            return as_int(row.get("num_points"))
    return None


def summarize_run(model: str, run_dir: Path) -> RunComparisonRow:
    data_snapshot = run_dir / "data_snapshot"

    run_summary = last_row(data_snapshot / "run_overview" / "run_summary.csv")
    system_info = last_row(data_snapshot / "run_overview" / "system_info.csv")
    mesh_metrics = last_row(data_snapshot / "mesh" / "mesh_metrics.csv")
    stage_rows = read_csv_rows(data_snapshot / "system" / "stage_timings.csv")
    pointcloud_stage_rows = read_csv_rows(data_snapshot / "pointcloud" / "pointcloud_stage_metrics.csv")

    if model == "dust3r":
        reconstruction_stage = "dust3r_reconstruction"
    elif model == "fast3r":
        reconstruction_stage = "fast3r_reconstruction"
    else:
        reconstruction_stage = f"{model}_reconstruction"

    final_points = as_int(run_summary.get("processed_points"))
    if final_points is None:
        final_points = pick_final_points(pointcloud_stage_rows)

    mesh_vertices = as_int(run_summary.get("mesh_vertices"))
    if mesh_vertices is None:
        mesh_vertices = as_int(mesh_metrics.get("num_vertices"))

    mesh_triangles = as_int(run_summary.get("mesh_triangles"))
    if mesh_triangles is None:
        mesh_triangles = as_int(mesh_metrics.get("num_triangles"))

    return RunComparisonRow(
        model=model,
        run_folder=run_dir.name,
        run_id=str(run_summary.get("run_id", "")),
        status=str(run_summary.get("status", "")),
        total_runtime_seconds=as_float(run_summary.get("total_runtime_seconds")),
        reconstruction_seconds=pick_stage_seconds(stage_rows, reconstruction_stage),
        mesh_seconds=pick_stage_seconds(stage_rows, "mesh_reconstruction"),
        peak_gpu_memory_gb=max((as_float(row.get("gpu_peak_memory_gb")) or 0.0) for row in stage_rows)
        if stage_rows
        else None,
        frames=as_int(run_summary.get("num_frames_extracted") or run_summary.get("num_frames_requested")),
        raw_points=as_int(run_summary.get("raw_points") or run_summary.get("num_points_raw")),
        final_points=final_points,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        device=str(system_info.get("cuda_device_name", "")),
        torch_version=str(system_info.get("torch_version", "")),
    )


def fmt_num(value, decimals: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def print_table(rows: list[RunComparisonRow]) -> None:
    headers = [
        "Model",
        "Run Folder",
        "Status",
        "Total(s)",
        "Recon(s)",
        "Mesh(s)",
        "PeakGPU(GB)",
        "Frames",
        "RawPts",
        "FinalPts",
        "Triangles",
    ]
    table = []
    for row in rows:
        table.append(
            [
                row.model,
                row.run_folder,
                row.status,
                fmt_num(row.total_runtime_seconds),
                fmt_num(row.reconstruction_seconds),
                fmt_num(row.mesh_seconds),
                fmt_num(row.peak_gpu_memory_gb),
                fmt_num(row.frames, 0),
                fmt_num(row.raw_points, 0),
                fmt_num(row.final_points, 0),
                fmt_num(row.mesh_triangles, 0),
            ]
        )

    widths = [len(h) for h in headers]
    for line in table:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], len(cell))

    def render(values: list[str]) -> str:
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    print(render(headers))
    print("-+-".join("-" * w for w in widths))
    for line in table:
        print(render(line))


def print_speedup(rows: list[RunComparisonRow]) -> None:
    indexed = {row.model: row for row in rows}
    if "dust3r" in indexed and "fast3r" in indexed:
        d = indexed["dust3r"]
        f = indexed["fast3r"]
        if d.total_runtime_seconds and f.total_runtime_seconds and f.total_runtime_seconds > 0:
            print(f"\nSpeedup total (dust3r/fast3r): {d.total_runtime_seconds / f.total_runtime_seconds:.2f}x")
        if d.reconstruction_seconds and f.reconstruction_seconds and f.reconstruction_seconds > 0:
            print(f"Speedup reconstruction (dust3r/fast3r): {d.reconstruction_seconds / f.reconstruction_seconds:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare latest archived runs for selected models.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=r"Archive root folder (default: D:\GitRepos\runs_pipeline).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names to compare (default: dust3r fast3r).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output JSON path for machine-readable comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    rows: list[RunComparisonRow] = []
    missing: list[str] = []
    for model in args.models:
        latest = find_latest_run_for_model(runs_root, model)
        if latest is None:
            missing.append(model)
            continue
        rows.append(summarize_run(model, latest))

    if not rows:
        raise FileNotFoundError(f"No matching archived runs found in: {runs_root}")

    print_table(rows)
    print_speedup(rows)

    if missing:
        print("\nMissing models:", ", ".join(missing))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(row) for row in rows]
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
