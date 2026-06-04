from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import read_json, save_figure, write_markdown


def main() -> None:
    metadata = read_json("src/outputs/scene_20260422_171717/scene_metadata.json")
    report = read_json("src/outputs/scene_20260422_171717/fast3r_point_retention_report.json")
    retention = report["point_retention"]
    final_points = metadata.get("points", {}).get("final_points", 0)

    labels = [
        "predicted dense",
        "after min conf",
        "after quantile",
        "after view pruning",
        "final processed",
    ]
    values = [
        retention["predicted_dense_points"],
        retention["points_after_min_conf"],
        retention["points_after_quantile"],
        retention["points_after_view_pruning"],
        final_points,
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2"])
    ax.set_ylabel("Point count")
    ax.set_title("FIGURA 6.3 - Retentia punctelor pentru scena reprezentativa")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    path = save_figure(fig, "figura_6_3_retentia_punctelor_pentru_scena_reprezentativa.png")
    plt.close(fig)

    md = write_markdown(
        "figura_6_3_retentia_punctelor_pentru_scena_reprezentativa.md",
        "# FIGURA 6.3 - Retentia punctelor pentru scena reprezentativa\n\n"
        "- Source scene: `scene_20260422_171717`\n"
        f"- Predicted dense points: `{retention['predicted_dense_points']:,}`\n"
        f"- Final processed points: `{final_points:,}`\n"
        "- Plot: point-count reduction through confidence filtering, view pruning, and final processing.\n"
        f"- Output: `{path}`\n",
    )
    print(path)
    print(md)


if __name__ == "__main__":
    main()
