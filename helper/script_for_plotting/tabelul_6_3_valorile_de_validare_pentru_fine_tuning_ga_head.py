from __future__ import annotations

from plotting_common import markdown_table, read_csv_rows, to_float, write_markdown


METRICS = "experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/metrics.csv"


def main() -> None:
    rows = [r for r in read_csv_rows(METRICS) if r.get("val/loss")]
    table_rows = []
    first = to_float(rows[0]["val/loss"])
    for row in rows:
        loss = to_float(row["val/loss"])
        reduction = (first - loss) / first * 100
        table_rows.append(
            [
                row["epoch"],
                row["step"],
                f"{loss:.10f}",
                f'{to_float(row.get("val/arkitscenes_Regr3DMultiviewV3_pts3d_loss_global")):.10f}',
                f'{to_float(row.get("val/arkitscenes_Regr3DMultiviewV3_pts3d_loss_local")):.10f}',
                f"{reduction:.2f}%",
            ]
        )

    content = "# TABELUL 6.3 - Valorile de validare pentru fine-tuning GA-head\n\n"
    content += markdown_table(
        [
            "Epoch",
            "Step",
            "Val loss",
            "Pts3D global",
            "Pts3D local",
            "Reducere fata de primul val",
        ],
        table_rows,
    )
    content += (
        "\n\nNota: tabelul descrie dinamica unei rulari head-only GA pe ARKitScenes. "
        "Nu trebuie prezentat ca benchmark complet pretrained-vs-fine-tuned."
    )
    path = write_markdown("tabelul_6_3_valorile_de_validare_pentru_fine_tuning_ga_head.md", content)
    print(path)


if __name__ == "__main__":
    main()
