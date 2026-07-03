from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import read_csv_rows, save_figure, to_float, to_int, write_markdown


def main() -> None:
    rows = read_csv_rows(
        "experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv"
    )
    selected_runs = [
        ("20260422_064701", 80),
        ("20260422_070036", 120),
        ("20260422_090421", 300),
    ]
    rows_by_id = {row["run_id"]: row for row in rows}
    plot_rows = [(rows_by_id[run_id], frame_label) for run_id, frame_label in selected_runs]
    frame_labels = [str(frame_label) for _, frame_label in plot_rows]
    runtimes = [to_float(row["total_runtime_seconds"]) for row, _ in plot_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(frame_labels, runtimes, color="#1f77b4", alpha=0.82)
    for bar, runtime in zip(bars, runtimes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{runtime:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Numar de cadre selectate")
    ax.set_ylabel("Runtime total (secunde)")
    ax.set_title("FIGURA 6.1 - Runtime in functie de numarul de cadre")
    ax.grid(alpha=0.25)
    path = save_figure(fig, "figura_6_1_runtime_in_functie_de_numarul_de_cadre.png")
    plt.close(fig)

    md = write_markdown(
        "figura_6_1_runtime_in_functie_de_numarul_de_cadre.md",
        "# FIGURA 6.1 - Runtime in functie de numarul de cadre\n\n"
        f"- Runs plotted: `{', '.join(run_id for run_id, _ in selected_runs)}`\n"
        "- The first run selected 79 frames in practice and is presented as approximately 80 frames.\n"
        "- Bars show total runtime for the selected representative successful runs.\n"
        f"- Output: `{path}`\n",
    )
    print(path)
    print(md)


if __name__ == "__main__":
    main()
