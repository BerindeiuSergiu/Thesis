from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import read_csv_rows, save_figure, to_float, to_int, write_markdown


METRICS = "experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/metrics.csv"


def validation_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(METRICS)
    return [r for r in rows if r.get("val/loss")]


def main() -> None:
    rows = validation_rows()
    steps = [to_int(r["step"]) for r in rows]
    losses = [to_float(r["val/loss"]) for r in rows]
    best_index = min(range(len(losses)), key=losses.__getitem__)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(steps, losses, marker="o", linewidth=1.5)
    ax.scatter([steps[0], steps[-1], steps[best_index]], [losses[0], losses[-1], losses[best_index]], color=["gray", "black", "red"], zorder=3)
    ax.annotate("first", (steps[0], losses[0]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.annotate("final", (steps[-1], losses[-1]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.annotate("best", (steps[best_index], losses[best_index]), textcoords="offset points", xytext=(5, -12), fontsize=8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title("FIGURA 6.4 - Curba de validare pentru fine-tuning GA-head")
    ax.grid(alpha=0.25)
    path = save_figure(fig, "figura_6_4_curba_de_validare_pentru_fine_tuning_ga_head.png")
    plt.close(fig)

    first_to_final = (losses[0] - losses[-1]) / losses[0] * 100
    first_to_best = (losses[0] - losses[best_index]) / losses[0] * 100
    md = write_markdown(
        "figura_6_4_curba_de_validare_pentru_fine_tuning_ga_head.md",
        "# FIGURA 6.4 - Curba de validare pentru fine-tuning GA-head\n\n"
        f"- Validation points: `{len(rows)}`\n"
        f"- First validation loss: `{losses[0]:.10f}`\n"
        f"- Final validation loss: `{losses[-1]:.10f}`\n"
        f"- Best validation loss: `{losses[best_index]:.10f}` at step `{steps[best_index]}`\n"
        f"- First-to-final reduction: `{first_to_final:.2f}%`\n"
        f"- First-to-best reduction: `{first_to_best:.2f}%`\n"
        "- Caveat: this is a single fine-tuning run curve, not a paired pretrained-vs-finetuned benchmark.\n"
        f"- Output: `{path}`\n",
    )
    print(path)
    print(md)


if __name__ == "__main__":
    main()
