from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import read_csv_rows, save_figure, to_float, write_markdown


def main() -> None:
    timing_rows = read_csv_rows(
        "experiments/beta_pipeline_testing/dual_output_fast3r_data/system/stage_timings.csv"
    )
    summary_rows = read_csv_rows(
        "experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv"
    )
    run_ids = ["20260422_064701", "20260422_070036", "20260422_090421", "20260422_105125"]
    scene_labels = ["debara", "camera mica", "living room", "camera medie"]
    stages = ["frame_selection", "inference", "scale_normalization", "shared_pointcloud_processing", "gaussian_splat_branch"]
    stage_labels = {
        "frame_selection": "selectare cadre",
        "inference": "inferenta Fast3R",
        "scale_normalization": "normalizare scara",
        "shared_pointcloud_processing": "procesare puncte",
        "gaussian_splat_branch": "generare Gaussian",
    }
    by_run = {run_id: {stage: 0.0 for stage in stages} for run_id in run_ids}
    for row in timing_rows:
        if row["run_id"] in by_run and row["stage_name"] in stages:
            by_run[row["run_id"]][row["stage_name"]] += to_float(row["elapsed_seconds"])

    summary_by_run = {row["run_id"]: row for row in summary_rows if row["run_id"] in run_ids}
    x_labels = []
    for run_id, label in zip(run_ids, scene_labels):
        summary = summary_by_run.get(run_id, {})
        frames = int(to_float(summary.get("frames", 0)))
        points_m = to_float(summary.get("processed_scaled_points", 0)) / 1_000_000
        x_labels.append(f"{label}\n{frames} cadre\n{points_m:.1f}M puncte")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bottoms = [0.0] * len(run_ids)
    for stage in stages:
        values = [by_run[run_id][stage] for run_id in run_ids]
        ax.bar(x_labels, values, bottom=bottoms, label=stage_labels[stage])
        bottoms = [a + b for a, b in zip(bottoms, values)]

    for index, total in enumerate(bottoms):
        ax.text(index, total + 8, f"{total:.0f}s", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Elapsed seconds")
    ax.set_title("FIGURA 6.2 - Timpii pe etape pentru rulari reprezentative")
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    path = save_figure(fig, "figura_6_2_timpii_pe_etape_pentru_rulari_reprezentative.png")
    plt.close(fig)

    md = write_markdown(
        "figura_6_2_timpii_pe_etape_pentru_rulari_reprezentative.md",
        "# FIGURA 6.2 - Timpii pe etape pentru rulari reprezentative\n\n"
        f"- Run ids: `{', '.join(run_ids)}`\n"
        f"- Scene labels: `{', '.join(scene_labels)}`\n"
        f"- Stages: `{', '.join(stage_labels[stage] for stage in stages)}`\n"
        "- Mesh branch is excluded because mesh generation was disabled in these representative runs.\n"
        "- X-axis labels include frame count and final processed point count, because runtime depends on both.\n"
        "- Plot: stacked bar chart showing where time is spent in the application pipeline.\n"
        f"- Output: `{path}`\n",
    )
    print(path)
    print(md)


if __name__ == "__main__":
    main()
