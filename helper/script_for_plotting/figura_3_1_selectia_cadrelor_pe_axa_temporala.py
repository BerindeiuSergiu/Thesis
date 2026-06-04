from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_common import read_json, save_figure, write_markdown


def main() -> None:
    metadata = read_json("src/outputs/scene_20260422_171717/scene_metadata.json")
    selection = metadata.get("frame_selection", {})
    selected = selection.get("selected_frame_indices", [])
    total = int(selection.get("total_source_frames") or max(selected))

    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.scatter(selected, [1] * len(selected), s=18, color="#1f77b4")
    ax.set_xlim(0, total)
    ax.set_ylim(0.8, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Video frame index")
    ax.set_title("FIGURA 3.1 - Selectia cadrelor pe axa temporala")
    ax.grid(axis="x", alpha=0.25)
    path = save_figure(fig, "figura_3_1_selectia_cadrelor_pe_axa_temporala.png")
    plt.close(fig)

    summary = (
        "# FIGURA 3.1 - Selectia cadrelor pe axa temporala\n\n"
        f"- Source scene: `scene_20260422_171717`\n"
        f"- Total source frames: `{total}`\n"
        f"- Selected frames: `{len(selected)}`\n"
        f"- Plot: selected frame index positions across the original video timeline.\n"
        f"- Output: `{path}`\n"
    )
    md = write_markdown("figura_3_1_selectia_cadrelor_pe_axa_temporala.md", summary)
    print(path)
    print(md)


if __name__ == "__main__":
    main()
