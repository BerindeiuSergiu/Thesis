from __future__ import annotations

from plotting_common import markdown_table, read_json, write_markdown


def main() -> None:
    report = read_json("src/outputs/scene_20260422_171717/mesh/mesh_branch_report.json")
    metrics = report["mesh_metrics"]
    rows = [
        ["Input points", f'{report["input_points"]:,}', "Punctele finale intrate in ramura mesh"],
        ["Mesh voxel points", f'{report["points_after_mesh_voxel"]:,}', "Puncte dupa reducere spatiala pentru mesh"],
        ["Points used for mesh", f'{report["points_used_for_mesh"]:,}', "Cap aplicat pentru Poisson reconstruction"],
        ["Triangles before decimation", f'{report["triangles_before_decimation"]:,}', "Triunghiuri inainte de decimare"],
        ["Final vertices", f'{metrics["num_vertices"]:,}', "Varfuri in mesh-ul final"],
        ["Final triangles", f'{metrics["num_triangles"]:,}', "Triunghiuri in mesh-ul final"],
        ["Watertight", metrics["is_watertight"], "Indicator de inchidere topologica"],
        ["Edge manifold", metrics["is_edge_manifold"], "Calitate topologica pe muchii"],
        ["Vertex manifold", metrics["is_vertex_manifold"], "Calitate topologica pe varfuri"],
        ["Self intersecting", metrics["is_self_intersecting"], "Prezenta auto-intersectiilor"],
        ["Largest component ratio", f'{metrics["largest_component_ratio"]:.6f}', "Dominanta componentei principale"],
    ]
    content = "# TABELUL 6.2 - Calitatea mesh-ului pentru scena reprezentativa\n\n"
    content += markdown_table(["Metric", "Valoare", "Interpretare"], rows)
    path = write_markdown("tabelul_6_2_calitatea_meshului_pentru_scena_reprezentativa.md", content)
    print(path)


if __name__ == "__main__":
    main()
