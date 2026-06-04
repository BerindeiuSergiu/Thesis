from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import read_csv_rows, save_figure, to_float, to_int, write_markdown


def main() -> None:
    rows = read_csv_rows(
        "experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv"
    )
    frames = [to_int(r["frames"]) for r in rows]
    runtimes = [to_float(r["total_runtime_seconds"]) for r in rows]
    raw_points = [to_int(r["raw_points"]) for r in rows]
    colors = ["#2ca02c" if r["status"] == "success" else "#d62728" for r in rows]
    sizes = [max(25, min(250, p / 100000)) for p in raw_points]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(frames, runtimes, s=sizes, c=colors, alpha=0.75, edgecolor="black", linewidth=0.35)
    ax.set_xlabel("Selected frames")
    ax.set_ylabel("Total runtime (seconds)")
    ax.set_title("FIGURA 6.1 - Runtime in functie de numarul de cadre")
    ax.grid(alpha=0.25)
    for marker_id in ["20260422_064701", "20260422_070036", "20260422_090421"]:
        match = next((r for r in rows if r["run_id"] == marker_id), None)
        if match:
            ax.annotate(
                marker_id[-4:],
                (to_int(match["frames"]), to_float(match["total_runtime_seconds"])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    path = save_figure(fig, "figura_6_1_runtime_in_functie_de_numarul_de_cadre.png")
    plt.close(fig)

    success_rows = [r for r in rows if r["status"] == "success"]
    md = write_markdown(
        "figura_6_1_runtime_in_functie_de_numarul_de_cadre.md",
        "# FIGURA 6.1 - Runtime in functie de numarul de cadre\n\n"
        f"- Runs plotted: `{len(rows)}`\n"
        f"- Successful runs: `{len(success_rows)}`\n"
        "- Marker size approximates raw point count.\n"
        "- Green = success; red = failed run.\n"
        f"- Output: `{path}`\n",
    )
    print(path)
    print(md)


if __name__ == "__main__":
    main()
