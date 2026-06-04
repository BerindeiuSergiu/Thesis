from __future__ import annotations

from plotting_common import markdown_table, read_csv_rows, write_markdown


def main() -> None:
    dust3r_rows = read_csv_rows(
        "experiments/beta_pipeline_testing/dust3r_data/run_overview/run_summary.csv"
    )
    fast3r_rows = read_csv_rows(
        "experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv"
    )

    dust3r = max(dust3r_rows, key=lambda r: float(r.get("total_runtime_seconds") or 0))
    fast3r = next(r for r in fast3r_rows if r["run_id"] == "20260422_064701")

    rows = [
        [
            "COLMAP SfM",
            "100",
            "aprox. 480",
            "1,230 sparse points; 24 camera poses",
            "Reconstructie clasica si interpretabila",
            "Lenta pentru feedback rapid si produce sparse geometry in rularea arhivata",
        ],
        [
            "MiDaS-style monocular depth",
            "100",
            "aprox. 35",
            "664,102 points before filtering",
            "Rapid ca estimare de adancime",
            "Geometrie zgomotoasa, fara coerenta multiview suficienta",
        ],
        [
            "DUSt3R standalone",
            dust3r.get("frames", "40"),
            f'{float(dust3r["total_runtime_seconds"]):.2f}',
            f'{int(float(dust3r.get("raw_points") or 0)):,} raw points; {dust3r.get("mesh_triangles", "196536")} mesh triangles',
            "Reconstructie invatata cu output dens",
            "Foarte lent in testul local si mai greu de extins la multe cadre",
        ],
        [
            "Fast3R application pipeline",
            fast3r["frames"],
            f'{float(fast3r["total_runtime_seconds"]):.2f}',
            f'{int(fast3r["raw_points"]):,} raw points; Gaussian output generated',
            "Forward-pass multiview, output rapid inspectabil",
            "Necesita filtrare si management de memorie",
        ],
    ]

    content = "# TABELUL 2.1 - Comparatia principalelor solutii analizate\n\n"
    content += markdown_table(
        ["Solutie", "Cadre", "Runtime local (s)", "Output local", "Avantaj", "Limitare"],
        rows,
    )
    content += (
        "\n\nNota: tabelul compara rulari interne de explorare, nu un benchmark "
        "controlat cu obiective identice pentru toate metodele."
    )
    path = write_markdown("tabelul_2_1_comparatia_principalelor_solutii_analizate.md", content)
    print(path)


if __name__ == "__main__":
    main()
