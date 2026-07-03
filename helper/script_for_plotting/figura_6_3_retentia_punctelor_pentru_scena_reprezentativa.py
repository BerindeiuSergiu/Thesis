from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import save_figure, write_markdown


def main() -> None:
    labels = [
        "Predictie densa",
        "Puncte brute utile",
        "Valide numeric",
        "Dupa filtrare",
        "Puncte finale",
    ]
    values = [
        35_389_440,
        26_264_029,
        26_264_029,
        26_185_236,
        8_146_338,
    ]
    initial = values[0]
    raw_points = values[1]
    percentages_from_initial = [value / initial * 100.0 for value in values]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#4c78a8", "#72b7b2", "#72b7b2", "#54a24b", "#b279a2"]
    y_positions = range(len(labels))
    bars = ax.barh(y_positions, [value / 1_000_000 for value in values], color=colors, alpha=0.86)
    ax.set_yticks(list(y_positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Numar de puncte (milioane)")
    ax.set_title("FIGURA 6.3 - Retentia punctelor pentru scena reprezentativa")
    ax.grid(axis="x", alpha=0.25)

    for bar, value, percentage in zip(bars, values, percentages_from_initial):
        ax.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,} ({percentage:.2f}%)",
            va="center",
            fontsize=8,
        )

    final_from_raw = values[-1] / raw_points * 100.0
    ax.text(
        0.02,
        -0.14,
        f"Punctele finale reprezinta {final_from_raw:.2f}% din punctele brute utile.",
        transform=ax.transAxes,
        fontsize=9,
    )

    path = save_figure(fig, "figura_6_3_retentia_punctelor_pentru_scena_reprezentativa.png")
    plt.close(fig)

    md = write_markdown(
        "figura_6_3_retentia_punctelor_pentru_scena_reprezentativa.md",
        "# FIGURA 6.3 - Retentia punctelor pentru scena reprezentativa\n\n"
        "- Values used in the chapter text:\n"
        f"  - Predicted dense points: `{values[0]:,}`\n"
        f"  - Useful raw points: `{values[1]:,}`\n"
        f"  - Numerically valid points: `{values[2]:,}`\n"
        f"  - Points after filtering: `{values[3]:,}`\n"
        f"  - Final points: `{values[4]:,}`\n"
        f"- Final/raw retention: `{final_from_raw:.2f}%`\n"
        f"- Final/initial retention: `{percentages_from_initial[-1]:.2f}%`\n"
        "- Plot: horizontal bar chart showing point-count reduction through the processing stages.\n"
        f"- Output: `{path}`\n",
    )
    print(path)
    print(md)


if __name__ == "__main__":
    main()
