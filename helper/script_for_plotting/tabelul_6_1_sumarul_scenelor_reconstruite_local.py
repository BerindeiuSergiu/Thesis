from __future__ import annotations

from pathlib import Path

from plotting_common import markdown_table, read_json, repo_root, write_markdown


def main() -> None:
    scenes_root = repo_root() / "src/outputs"
    rows = []
    for metadata_path in sorted(scenes_root.glob("scene_*/scene_metadata.json")):
        rel = metadata_path.relative_to(repo_root()).as_posix()
        metadata = read_json(rel)
        frame_selection = metadata.get("frame_selection", {})
        scale = metadata.get("scale_normalization", {})
        gaussians = metadata.get("gaussian_splat", {})
        mesh = metadata.get("mesh_branch", {})
        mesh_metrics = mesh.get("mesh_metrics", {})
        raw_points = metadata.get("num_points_raw", "")
        final_points = metadata.get("num_points_final", "")
        gaussian_count = gaussians.get("output_gaussians", "")
        mesh_triangles = mesh_metrics.get("num_triangles", "")
        rows.append(
            [
                metadata_path.parent.name,
                metadata.get("num_frames", frame_selection.get("selected_frame_count", "")),
                f"{raw_points:,}" if isinstance(raw_points, int) else raw_points,
                f"{final_points:,}" if isinstance(final_points, int) else final_points,
                scale.get("scale_to_meters", ""),
                f"{gaussian_count:,}" if isinstance(gaussian_count, int) else gaussian_count,
                f"{mesh_triangles:,}" if isinstance(mesh_triangles, int) else mesh_triangles,
            ]
        )

    content = "# TABELUL 6.1 - Sumarul scenelor reconstruite local\n\n"
    content += markdown_table(
        ["Scena", "Cadre", "Puncte brute", "Puncte finale", "Scala", "Gaussians", "Triunghiuri mesh"],
        rows,
    )
    content += f"\n\nSursa: `{Path('src/outputs')}`"
    path = write_markdown("tabelul_6_1_sumarul_scenelor_reconstruite_local.md", content)
    print(path)


if __name__ == "__main__":
    main()
