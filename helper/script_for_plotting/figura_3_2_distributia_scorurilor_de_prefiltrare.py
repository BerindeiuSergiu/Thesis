from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import read_csv_rows, save_figure, to_float, write_markdown


def main() -> None:
    candidates = read_csv_rows(
        "experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/candidate_frame_scores.csv"
    )
    selected = read_csv_rows(
        "experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/selected_frame_manifest.csv"
    )
    candidate_scores = [to_float(r["prefilter_score"]) for r in candidates]
    selected_scores = [to_float(r["prefilter_score"]) for r in selected]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(candidate_scores, bins=30, alpha=0.65, label="Candidate frames")
    ax.hist(selected_scores, bins=20, alpha=0.85, label="Selected frames")
    ax.axvline(sum(selected_scores) / len(selected_scores), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prefilter score")
    ax.set_ylabel("Frame count")
    ax.set_title("FIGURA 3.2 - Distributia scorurilor de prefiltrare")
    ax.legend()
    path = save_figure(fig, "figura_3_2_distributia_scorurilor_de_prefiltrare.png")
    plt.close(fig)

    mean_selected = sum(selected_scores) / len(selected_scores)
    md = write_markdown(
        "figura_3_2_distributia_scorurilor_de_prefiltrare.md",
        "# FIGURA 3.2 - Distributia scorurilor de prefiltrare\n\n"
        f"- Candidate frames: `{len(candidate_scores)}`\n"
        f"- Selected frames: `{len(selected_scores)}`\n"
        f"- Mean selected score: `{mean_selected:.12f}`\n"
        f"- Plot: histogram of all candidate prefilter scores with selected frames overlaid.\n"
        f"- Output: `{path}`\n",
    )
    print(path)
    print(md)


if __name__ == "__main__":
    main()
