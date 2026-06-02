"""Archive and visualize model metric data folders.

This module scans one or more ``*_data`` folders, creates a timestamped archive
under ``D:\\GitRepos\\runs_pipeline``, generates generic CSV plots, and can
clean source data after a successful archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


BETA_ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_ROOT = Path(r"D:\GitRepos\runs_pipeline")
PREFERRED_X_COLUMNS = (
    "frame_local_index",
    "frame_local_index_a",
    "frame_local_index_b",
    "source_frame_index",
    "source_frame_index_a",
    "source_frame_index_b",
    "timestamp_seconds",
    "pair_index",
)
PREFERRED_CATEGORY_COLUMNS = (
    "stage_name",
    "scene_graph",
    "mesh_method",
    "sampling_strategy",
)
IGNORED_VALUE_COLUMNS = {
    "run_id",
    "recorded_at_utc",
    "video_path",
    "output_dir",
    "error_message",
    "direction",
    "cwd",
    "executable",
    "platform",
    "processor",
}


@dataclass
class PlotManifestRow:
    csv_file: str
    plot_files: list[str]
    row_count: int
    run_id: str | None
    x_column: str | None
    category_column: str | None
    numeric_columns: list[str]


def coerce_value(value):
    """Convert CSV strings to python scalars where possible."""
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if "_" in text:
        return text
    try:
        if "." not in text and "e" not in lower:
            return int(text)
        return float(text)
    except ValueError:
        return text


def as_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        return num if math.isfinite(num) else None
    return None


def load_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: coerce_value(val) for key, val in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]


def resolve_latest_run_id(data_root: Path) -> str | None:
    summary_path = data_root / "run_overview" / "run_summary.csv"
    if not summary_path.exists():
        return None
    rows = load_csv_rows(summary_path)
    if not rows:
        return None
    run_id = rows[-1].get("run_id")
    return str(run_id) if run_id is not None else None


def infer_model_name(data_root: Path) -> str:
    name = data_root.name
    if name.endswith("_data") and len(name) > 5:
        return name[:-5]
    return name


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_unique_archive_root(parent: Path, model_name: str) -> Path:
    label = f"{timestamp_label()}_{model_name}"
    candidate = parent / label
    counter = 1
    while candidate.exists():
        candidate = parent / f"{label}_{counter:02d}"
        counter += 1
    return candidate


def discover_csv_files(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.rglob("*.csv") if path.is_file())


def filter_rows_for_run(rows: list[dict], run_id: str | None) -> list[dict]:
    if not rows or run_id is None:
        return rows
    if "run_id" not in rows[0]:
        return rows
    return [row for row in rows if str(row.get("run_id")) == run_id]


def choose_numeric_columns(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    columns = list(rows[0].keys())
    numeric_columns: list[str] = []
    for column in columns:
        if column in IGNORED_VALUE_COLUMNS:
            continue
        converted = [as_float(row.get(column)) for row in rows]
        numeric_count = sum(val is not None for val in converted)
        if numeric_count >= 2 and numeric_count / max(1, len(rows)) >= 0.5:
            numeric_columns.append(column)
    return numeric_columns


def choose_x_column(rows: list[dict], numeric_columns: list[str]) -> str | None:
    if not rows:
        return None
    for preferred in PREFERRED_X_COLUMNS:
        if preferred in numeric_columns:
            return preferred
    if "frame_idx_a" in numeric_columns:
        return "frame_idx_a"
    if "frame_idx_b" in numeric_columns:
        return "frame_idx_b"
    return None


def choose_category_column(rows: list[dict]) -> str | None:
    if not rows:
        return None
    columns = list(rows[0].keys())
    for preferred in PREFERRED_CATEGORY_COLUMNS:
        if preferred in columns:
            values = [row.get(preferred) for row in rows]
            unique = {v for v in values if v is not None}
            if 1 < len(unique) <= 40:
                return preferred
    for column in columns:
        if column in IGNORED_VALUE_COLUMNS:
            continue
        values = [row.get(column) for row in rows]
        if any(isinstance(v, str) for v in values):
            unique = {v for v in values if isinstance(v, str)}
            if 1 < len(unique) <= 30:
                return column
    return None


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(path_or_name: str) -> str:
    return path_or_name.replace("\\", "__").replace("/", "__").replace(" ", "_")


def write_rows_csv(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        return
    ensure_dir(output_csv.parent)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_line_plots(
    rows: list[dict],
    numeric_columns: list[str],
    x_column: str,
    base_name: str,
    output_dir: Path,
) -> list[Path]:
    y_columns = [column for column in numeric_columns if column != x_column]
    if not y_columns:
        return []

    plots: list[Path] = []
    x_all = [as_float(row.get(x_column)) for row in rows]
    for chunk_index, columns_chunk in enumerate(chunked(y_columns, 4), start=1):
        fig, axes = plt.subplots(len(columns_chunk), 1, figsize=(11, 2.8 * len(columns_chunk)), sharex=False)
        if len(columns_chunk) == 1:
            axes = [axes]
        for axis, column in zip(axes, columns_chunk):
            y_all = [as_float(row.get(column)) for row in rows]
            points = [(x, y) for x, y in zip(x_all, y_all) if x is not None and y is not None]
            if not points:
                axis.text(0.5, 0.5, "No numeric points", ha="center", va="center", transform=axis.transAxes)
                axis.set_title(column)
                continue
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            axis.plot(x_vals, y_vals, marker="o", linewidth=1.3, markersize=3.4)
            axis.set_title(f"{column} vs {x_column}")
            axis.set_xlabel(x_column)
            axis.set_ylabel(column)
            axis.grid(alpha=0.25)
        output_path = output_dir / f"{base_name}__line_{chunk_index:02d}.png"
        save_figure(fig, output_path)
        plots.append(output_path)
    return plots


def create_category_bar_plots(
    rows: list[dict],
    numeric_columns: list[str],
    category_column: str,
    base_name: str,
    output_dir: Path,
) -> list[Path]:
    plots: list[Path] = []
    categories = [str(row.get(category_column)) for row in rows if row.get(category_column) is not None]
    if not categories:
        return plots

    unique_categories = list(dict.fromkeys(categories))
    for chunk_index, columns_chunk in enumerate(chunked(numeric_columns, 4), start=1):
        fig, axes = plt.subplots(len(columns_chunk), 1, figsize=(12, 2.9 * len(columns_chunk)))
        if len(columns_chunk) == 1:
            axes = [axes]
        for axis, column in zip(axes, columns_chunk):
            grouped: dict[str, list[float]] = {cat: [] for cat in unique_categories}
            for row in rows:
                category = row.get(category_column)
                value = as_float(row.get(column))
                if category is None or value is None:
                    continue
                grouped[str(category)].append(value)

            x_labels: list[str] = []
            values: list[float] = []
            for category in unique_categories:
                vals = grouped.get(category, [])
                if not vals:
                    continue
                x_labels.append(category)
                values.append(float(np.mean(vals)))

            if not values:
                axis.text(0.5, 0.5, "No grouped values", ha="center", va="center", transform=axis.transAxes)
                axis.set_title(column)
                continue

            axis.bar(x_labels, values)
            axis.set_title(f"{column} by {category_column} (mean)")
            axis.set_ylabel(column)
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)

        output_path = output_dir / f"{base_name}__bar_{chunk_index:02d}.png"
        save_figure(fig, output_path)
        plots.append(output_path)
    return plots


def create_hist_plots(
    rows: list[dict],
    numeric_columns: list[str],
    base_name: str,
    output_dir: Path,
) -> list[Path]:
    plots: list[Path] = []
    for chunk_index, columns_chunk in enumerate(chunked(numeric_columns, 4), start=1):
        fig, axes = plt.subplots(len(columns_chunk), 1, figsize=(10, 2.9 * len(columns_chunk)))
        if len(columns_chunk) == 1:
            axes = [axes]
        for axis, column in zip(axes, columns_chunk):
            values = [as_float(row.get(column)) for row in rows]
            values = [val for val in values if val is not None]
            if len(values) < 2:
                axis.text(0.5, 0.5, "Insufficient values", ha="center", va="center", transform=axis.transAxes)
                axis.set_title(column)
                continue
            bins = min(40, max(10, int(len(values) ** 0.5)))
            axis.hist(values, bins=bins, alpha=0.85)
            axis.set_title(f"{column} distribution")
            axis.set_xlabel(column)
            axis.set_ylabel("count")
            axis.grid(alpha=0.2)
        output_path = output_dir / f"{base_name}__hist_{chunk_index:02d}.png"
        save_figure(fig, output_path)
        plots.append(output_path)
    return plots


def plot_single_csv(rows: list[dict], relative_csv_path: Path, output_dir: Path) -> tuple[list[Path], str | None, str | None, list[str]]:
    numeric_columns = choose_numeric_columns(rows)
    if not numeric_columns:
        return [], None, None, []

    base_name = sanitize_name(str(relative_csv_path.with_suffix("")))
    x_column = choose_x_column(rows, numeric_columns)
    category_column = choose_category_column(rows)

    plots: list[Path] = []
    if x_column is not None:
        plots.extend(create_line_plots(rows, numeric_columns, x_column, base_name, output_dir))
    elif category_column is not None:
        plots.extend(create_category_bar_plots(rows, numeric_columns, category_column, base_name, output_dir))
    else:
        plots.extend(create_hist_plots(rows, numeric_columns, base_name, output_dir))

    if not plots:
        plots.extend(create_hist_plots(rows, numeric_columns, base_name, output_dir))

    return plots, x_column, category_column, numeric_columns


def clean_data_folder(data_root: Path) -> None:
    resolved = data_root.resolve()
    if resolved.name.endswith("_data") is False:
        raise ValueError(f"Refusing to clean non '*_data' folder: {resolved}")
    for child in resolved.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def snapshot_data_folder(
    data_root: Path,
    snapshot_root: Path,
    run_id: str | None,
    isolate_run: bool,
) -> None:
    ensure_dir(snapshot_root)
    for source_path in sorted(data_root.rglob("*")):
        if source_path.is_dir():
            continue

        relative = source_path.relative_to(data_root)
        destination = snapshot_root / relative
        ensure_dir(destination.parent)

        if source_path.suffix.lower() != ".csv":
            shutil.copy2(source_path, destination)
            continue

        rows = load_csv_rows(source_path)
        rows_to_write = filter_rows_for_run(rows, run_id) if isolate_run else rows
        if not rows_to_write:
            continue
        write_rows_csv(rows_to_write, destination)


def process_data_folder(
    data_root: Path,
    output_root: Path | None,
    clean: bool,
    explicit_run_id: str | None,
    isolate_run: bool,
) -> Path:
    data_root = data_root.resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data folder does not exist: {data_root}")

    model_name = infer_model_name(data_root)
    archive_parent = ensure_dir(output_root.resolve()) if output_root else ensure_dir(DEFAULT_ARCHIVE_ROOT)
    archive_root = create_unique_archive_root(archive_parent, model_name)
    plots_dir = ensure_dir(archive_root / "plots")
    data_snapshot_dir = archive_root / "data_snapshot"
    plot_data_dir = archive_root / "plot_data"
    metadata_dir = ensure_dir(archive_root / "metadata")

    run_id = explicit_run_id if explicit_run_id else resolve_latest_run_id(data_root)
    snapshot_data_folder(
        data_root=data_root,
        snapshot_root=data_snapshot_dir,
        run_id=run_id,
        isolate_run=isolate_run,
    )
    csv_files = discover_csv_files(data_snapshot_dir)

    manifest: list[PlotManifestRow] = []
    for csv_path in csv_files:
        relative = csv_path.relative_to(data_snapshot_dir)
        rows = load_csv_rows(csv_path)
        rows_for_plot = filter_rows_for_run(rows, run_id)
        if not rows_for_plot:
            continue

        write_rows_csv(rows_for_plot, plot_data_dir / relative)
        plots, x_column, category_column, numeric_columns = plot_single_csv(rows_for_plot, relative, plots_dir)
        manifest.append(
            PlotManifestRow(
                csv_file=str(relative).replace("\\", "/"),
                plot_files=[str(path.relative_to(archive_root)).replace("\\", "/") for path in plots],
                row_count=len(rows_for_plot),
                run_id=run_id,
                x_column=x_column,
                category_column=category_column,
                numeric_columns=numeric_columns,
            )
        )

    manifest_csv = metadata_dir / "plot_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["csv_file", "row_count", "run_id", "x_column", "category_column", "numeric_columns", "plot_files"])
        for item in manifest:
            writer.writerow(
                [
                    item.csv_file,
                    item.row_count,
                    item.run_id or "",
                    item.x_column or "",
                    item.category_column or "",
                    "|".join(item.numeric_columns),
                    "|".join(item.plot_files),
                ]
            )

    run_info = {
        "source_data_root": str(data_root),
        "archive_root": str(archive_root),
        "model_name": model_name,
        "timestamp": archive_root.name.split("_", 2)[0] + "_" + archive_root.name.split("_", 2)[1],
        "run_id_used_for_plots": run_id,
        "snapshot_scope": "run_id_only" if isolate_run else "full_data_folder",
        "csv_files_discovered": len(csv_files),
        "csv_files_plotted": len([item for item in manifest if item.plot_files]),
        "plots_generated": sum(len(item.plot_files) for item in manifest),
        "cleaned_source_data": clean,
    }
    with (metadata_dir / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2)

    if clean:
        clean_data_folder(data_root)

    return archive_root


def find_data_roots(base_dir: Path) -> list[Path]:
    return sorted(path for path in base_dir.iterdir() if path.is_dir() and path.name.endswith("_data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive and plot model metric folders (e.g., dust3r_data, fast3r_data)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=BETA_ROOT / "dust3r_data",
        help="Path to one model data folder (default: beta_pipeline_testing/dust3r_data).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Process all '*_data' folders from --base-dir instead of a single --data-root folder.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BETA_ROOT,
        help="Directory scanned when --all-models is used.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Root where timestamped archives are created (default: D:\\GitRepos\\runs_pipeline).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id to plot. By default uses latest run_id from run_summary.csv when available.",
    )
    parser.add_argument(
        "--full-snapshot",
        action="store_true",
        help="Copy full CSV history into archive instead of isolating to selected/latest run_id.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean source data folder contents after successful archiving and plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_models:
        targets = find_data_roots(args.base_dir.resolve())
    else:
        targets = [args.data_root.resolve()]

    if not targets:
        raise FileNotFoundError("No '*_data' folders found to process.")

    created_archives: list[Path] = []
    for target in targets:
        archive_path = process_data_folder(
            data_root=target,
            output_root=args.output_root,
            clean=args.clean,
            explicit_run_id=args.run_id,
            isolate_run=not args.full_snapshot,
        )
        created_archives.append(archive_path)

    for archive in created_archives:
        print(f"Created archive: {archive}")


if __name__ == "__main__":
    main()
