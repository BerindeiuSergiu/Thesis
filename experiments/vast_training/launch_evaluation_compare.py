from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import launch_evaluation


def _find_metrics_csv(root: Path) -> Path | None:
    candidates = sorted(root.rglob("metrics.csv"))
    return candidates[0] if candidates else None


def _load_metrics_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _collect_final_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    final_metrics: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            parsed = _numeric_or_none(value)
            if parsed is not None and key not in {"epoch", "step"}:
                final_metrics[key] = parsed
    return final_metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_comparison_csv(path: Path, pretrained: dict[str, float], finetuned: dict[str, float]) -> None:
    fieldnames = ["metric", "pretrained", "finetuned", "delta_finetuned_minus_pretrained"]
    metric_names = sorted(set(pretrained) | set(finetuned))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in metric_names:
            base_value = pretrained.get(metric_name)
            tuned_value = finetuned.get(metric_name)
            delta = None
            if base_value is not None and tuned_value is not None:
                delta = tuned_value - base_value
            writer.writerow(
                {
                    "metric": metric_name,
                    "pretrained": "" if base_value is None else base_value,
                    "finetuned": "" if tuned_value is None else tuned_value,
                    "delta_finetuned_minus_pretrained": "" if delta is None else delta,
                }
            )


def _copy_metrics_if_present(src_root: Path, dst_path: Path) -> str:
    metrics_path = _find_metrics_csv(src_root)
    if metrics_path is None:
        return ""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, dst_path)
    return str(metrics_path)


def _build_eval_args(
    common_args: argparse.Namespace,
    checkpoint_path: str,
    run_name: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        config_name=common_args.config_name,
        dataset_name=common_args.dataset_name,
        dataset_root=common_args.dataset_root,
        arkitscenes_root=common_args.arkitscenes_root,
        scannet_root=common_args.scannet_root,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        output_dir=common_args.output_dir,
        image_size=common_args.image_size,
        num_views=common_args.num_views,
        batch_size_val=common_args.batch_size_val,
        num_workers_val=common_args.num_workers_val,
        print_only=False,
    )


def _run_eval(label: str, command: list[str], working_dir: Path, env: dict[str, str]) -> int:
    print(f"{label} working directory:", working_dir)
    print(f"{label} command:")
    print(" ".join(command))
    result = subprocess.run(command, cwd=working_dir, env=env)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained and fine-tuned Fast3R evaluations on the same validation split and summarize the results."
    )
    parser.add_argument("--config-name", default="arkitscenes_10hour")
    parser.add_argument("--dataset-name", default="arkitscenes")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--arkitscenes-root", default="")
    parser.add_argument("--scannet-root", default="")
    parser.add_argument("--pretrained-checkpoint-path", required=True)
    parser.add_argument("--finetuned-checkpoint-path", required=True)
    parser.add_argument("--run-name-prefix", default="arkitscenes_eval_compare")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=0)
    parser.add_argument("--batch-size-val", type=int, default=1)
    parser.add_argument("--num-workers-val", type=int, default=2)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = launch_evaluation._load_profile(args.config_name)
    output_dir = args.output_dir or "/workspace/logs"
    results_dir = Path(args.results_dir) if args.results_dir else Path(output_dir) / "eval_comparisons" / args.run_name_prefix

    pretrained_run_name = f"{args.run_name_prefix}_pretrained"
    finetuned_run_name = f"{args.run_name_prefix}_finetuned"

    pretrained_args = _build_eval_args(args, args.pretrained_checkpoint_path, pretrained_run_name)
    finetuned_args = _build_eval_args(args, args.finetuned_checkpoint_path, finetuned_run_name)

    pretrained_command, fast3r_root, pretrained_env = launch_evaluation.build_command(profile, pretrained_args)
    finetuned_command, _, finetuned_env = launch_evaluation.build_command(profile, finetuned_args)

    comparison_plan = {
        "config_name": args.config_name,
        "dataset_name": args.dataset_name,
        "dataset_root": args.dataset_root or profile.get("dataset", {}).get("root", ""),
        "pretrained_checkpoint_path": args.pretrained_checkpoint_path,
        "finetuned_checkpoint_path": args.finetuned_checkpoint_path,
        "pretrained_run_name": pretrained_run_name,
        "finetuned_run_name": finetuned_run_name,
        "pretrained_command": pretrained_command,
        "finetuned_command": finetuned_command,
        "working_directory": str(fast3r_root),
        "results_dir": str(results_dir),
    }
    if args.print_only:
        print("Pretrained evaluation command:")
        print(" ".join(pretrained_command))
        print("Fine-tuned evaluation command:")
        print(" ".join(finetuned_command))
        print("Comparison results directory:", results_dir)
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "comparison_plan.json", comparison_plan)

    pretrained_rc = _run_eval("Pretrained", pretrained_command, fast3r_root, pretrained_env)
    if pretrained_rc != 0:
        return pretrained_rc

    finetuned_rc = _run_eval("Fine-tuned", finetuned_command, fast3r_root, finetuned_env)
    if finetuned_rc != 0:
        return finetuned_rc

    pretrained_run_dir = Path(output_dir) / "eval_runs" / pretrained_run_name
    finetuned_run_dir = Path(output_dir) / "eval_runs" / finetuned_run_name

    pretrained_metrics_source = _copy_metrics_if_present(pretrained_run_dir, results_dir / "pretrained_metrics.csv")
    finetuned_metrics_source = _copy_metrics_if_present(finetuned_run_dir, results_dir / "finetuned_metrics.csv")

    pretrained_rows = _load_metrics_rows(results_dir / "pretrained_metrics.csv") if (results_dir / "pretrained_metrics.csv").exists() else []
    finetuned_rows = _load_metrics_rows(results_dir / "finetuned_metrics.csv") if (results_dir / "finetuned_metrics.csv").exists() else []
    pretrained_final = _collect_final_metrics(pretrained_rows)
    finetuned_final = _collect_final_metrics(finetuned_rows)

    _write_comparison_csv(results_dir / "comparison_metrics.csv", pretrained_final, finetuned_final)

    deltas: dict[str, float] = {}
    for metric_name in sorted(set(pretrained_final) & set(finetuned_final)):
        deltas[metric_name] = finetuned_final[metric_name] - pretrained_final[metric_name]

    summary = {
        "config_name": args.config_name,
        "dataset_name": args.dataset_name,
        "dataset_root": args.dataset_root or profile.get("dataset", {}).get("root", ""),
        "pretrained": {
            "checkpoint_path": args.pretrained_checkpoint_path,
            "run_name": pretrained_run_name,
            "run_dir": str(pretrained_run_dir),
            "metrics_csv_source": pretrained_metrics_source,
            "final_metrics": pretrained_final,
        },
        "finetuned": {
            "checkpoint_path": args.finetuned_checkpoint_path,
            "run_name": finetuned_run_name,
            "run_dir": str(finetuned_run_dir),
            "metrics_csv_source": finetuned_metrics_source,
            "final_metrics": finetuned_final,
        },
        "deltas_finetuned_minus_pretrained": deltas,
    }
    _write_json(results_dir / "comparison_summary.json", summary)
    print("Comparison summary written to:", results_dir / "comparison_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
