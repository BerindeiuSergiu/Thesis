from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import read_csv_rows, save_figure, to_float, write_markdown


def main() -> None:
    rows = read_csv_rows(
        "experiments/beta_pipeline_testing/dual_output_fast3r_data/system/stage_timings.csv"
    )
    run_ids = ["20260422_064701", "20260422_070036", "20260422_090421", "20260422_105125"]
    stages = [
        "frame_selection",
        "inference",
        "scale_normalization",
        "shared_pointcloud_processing",
        "gaussian_splat_branch",
        "mesh_branch",
    ]
    by_run = {run_id: {stage: 0.0 for stage in stages} for run_id in run_ids}
    for row in rows:
        if row["run_id"] in by_run and row["stage_name"] in stages:
            by_run[row["run_id"]][row["stage_name"]] += to_float(row["elapsed_seconds"])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bottoms = [0.0] * len(run_ids)
    for stage in stages:
        values = [by_run[run_id][stage] for run_id in run_ids]
        ax.bar(run_ids, values, bottom=bottoms, label=stage)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_ylabel("Elapsed seconds")
    ax.set_title("FIGURA 6.2 - Timpii pe etape pentru rulari reprezentative")
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    path = save_figure(fig, "figura_6_2_timpii_pe_etape_pentru_rulari_reprezentative.png")
    plt.close(fig)

    md = write_markdown(
        "figura_6_2_timpii_pe_etape_pentru_rulari_reprezentative.md",
        "# FIGURA 6.2 - Timpii pe etape pentru rulari reprezentative\n\n"
        f"- Run ids: `{', '.join(run_ids)}`\n"
        f"- Stages: `{', '.join(stages)}`\n"
        "- Plot: stacked bar chart showing where time is spent in the application pipeline.\n"
        f"- Output: `{path}`\n",
    )
    print(path)
    print(md)


if __name__ == "__main__":
    main()
