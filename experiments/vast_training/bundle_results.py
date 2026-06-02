from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    except Exception as exc:  # pragma: no cover - best effort system info
        return f"unavailable ({exc})"


def _parse_prefixed_count(dataset_expr: str) -> int | None:
    match = re.match(r"^\s*([0-9_]+)\s*@\s*", dataset_expr)
    if not match:
        return None
    return int(match.group(1).replace("_", ""))


def _parse_split(dataset_expr: str) -> str | None:
    match = re.search(r"split='([^']+)'", dataset_expr)
    return match.group(1) if match else None


def _load_metrics_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _find_metrics_csv(root: Path) -> Path | None:
    candidates = sorted(root.rglob("metrics.csv"))
    return candidates[0] if candidates else None


def _row_has_metric(row: dict[str, str], needle: str) -> bool:
    needle = needle.lower()
    for key, value in row.items():
        if not value:
            continue
        if needle in key.lower():
            return True
    return False


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _find_checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("*.ckpt"))


def _best_checkpoint_from_dir(checkpoint_dir: Path) -> Path | None:
    yaml_path = checkpoint_dir / "best_k_models.yaml"
    if yaml_path.exists():
        text = yaml_path.read_text(encoding="utf-8")
        matches = re.findall(r"^(.+\.ckpt):\s*([0-9eE+\-.]+)\s*$", text, flags=re.MULTILINE)
        if matches:
            best_path = Path(matches[0][0].strip())
            if best_path.exists():
                return best_path
            joined = checkpoint_dir / best_path.name
            if joined.exists():
                return joined
    ckpts = [path for path in _find_checkpoint_files(checkpoint_dir) if path.name != "last.ckpt"]
    return ckpts[0] if ckpts else None


def _checkpoint_summary(checkpoint_dir: Path) -> dict[str, Any]:
    last_ckpt = checkpoint_dir / "last.ckpt"
    best_ckpt = _best_checkpoint_from_dir(checkpoint_dir)
    all_ckpts = _find_checkpoint_files(checkpoint_dir)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else "",
        "best_checkpoint": str(best_ckpt) if best_ckpt else "",
        "all_checkpoints": [str(path) for path in all_ckpts],
    }


def _state_dict_counts(checkpoint_path: Path, trainable_tokens: list[str]) -> dict[str, int | str]:
    if not checkpoint_path.exists() or checkpoint_path.is_dir():
        return {
            "total_parameter_count": "unavailable",
            "trainable_parameter_count": "unavailable",
            "frozen_parameter_count": "unavailable",
        }
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    total = 0
    trainable = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        numel = int(tensor.numel())
        total += numel
        if any(token in name for token in trainable_tokens):
            trainable += numel
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": total - trainable,
    }


def _extract_yaml_target(text: str, section: str) -> str:
    pattern = rf"{section}:\s*(?:\n(?:[ \t].*\n?)*)"
    match = re.search(pattern, text)
    if not match:
        return "unavailable"
    block = match.group(0)
    target_match = re.search(r"_target_:\s*([^\n]+)", block)
    return target_match.group(1).strip() if target_match else "unavailable"


def _read_upload_source_info(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    info: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def _read_lora_report(train_log_path: Path) -> dict[str, int | str]:
    if not train_log_path.exists():
        return {}
    text = train_log_path.read_text(encoding="utf-8", errors="replace")
    fields = {
        "target_scope": r"^\s*target_scope=(.+?)\s*$",
        "target_module_count": r"^\s*target_modules=([0-9,]+)\s*$",
        "trainable_parameter_count": r"^\s*trainable_params=([0-9,]+)\s*$",
        "total_parameter_count": r"^\s*total_params=([0-9,]+)\s*$",
    }
    report: dict[str, int | str] = {}
    for field, pattern in fields.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            continue
        value = match.group(1).replace(",", "")
        report[field] = int(value) if value.isdigit() else value
    return report


def _copy_if_exists(src: Path | None, dst: Path) -> None:
    if src is None or not src.exists():
        return
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_checkpoint_artifact(src: Path, dst: Path) -> None:
    if src.is_dir():
        _copy_tree_if_exists(src, dst)
    else:
        _copy_if_exists(src, dst)


def _system_info_text(run_spec: dict[str, Any], upload_info: dict[str, str], training_seconds: float) -> str:
    lines = [
        f"generated_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={socket.gethostname()}",
        f"platform={platform.platform()}",
        f"python_version={sys.version.splitlines()[0]}",
        f"torch_version={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"gpu_info={_safe_run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])}",
        f"git_fast3r_commit={upload_info.get('FAST3R_GIT_COMMIT', 'unavailable')}",
        f"git_repo_commit={upload_info.get('THESIS_GIT_COMMIT', 'unavailable')}",
        f"training_time_seconds={training_seconds:.3f}",
        f"working_directory={run_spec.get('working_directory', '')}",
        f"run_directory={run_spec.get('run_dir', '')}",
    ]
    return "\n".join(lines) + "\n"


def _write_checkpoint_summary(path: Path, summary: dict[str, Any], exported_checkpoint_dir: str) -> None:
    lines = [
        f"checkpoint_dir={summary['checkpoint_dir']}",
        f"last_checkpoint={summary['last_checkpoint']}",
        f"best_checkpoint={summary['best_checkpoint']}",
        f"exported_checkpoint_dir={exported_checkpoint_dir}",
        "all_checkpoints:",
    ]
    for ckpt in summary["all_checkpoints"]:
        lines.append(f"  - {ckpt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _results_readme(
    run_spec: dict[str, Any],
    checkpoint_summary: dict[str, Any],
    train_metrics_path: Path,
    val_metrics_path: Path,
    eval_summary_path: Path,
    system_info_path: Path,
    command_path: Path,
    config_path: Path,
) -> str:
    cfg = run_spec["run_cfg"]
    dataset_name = cfg["dataset_name"]
    stage = cfg["stage"]
    lora_cfg = run_spec.get("lora", {})
    lora_lines = ""
    if lora_cfg:
        lora_lines = (
            f"- LoRA target scope: `{lora_cfg.get('target_scope', '')}`\n"
            f"- LoRA rank / alpha / dropout: `{lora_cfg.get('rank', '')}` / "
            f"`{lora_cfg.get('alpha', '')}` / `{lora_cfg.get('dropout', '')}`\n"
        )
    return f"""# Experiment Results Bundle

## What experiment was run

- Model family: Fast3R
- Fine-tuning stage: `{stage}`
- Fine-tuned components: {", ".join(run_spec.get("trainable_name_substrings", []))}
{lora_lines}- Dataset: `{dataset_name}`
- Run name: `{cfg['run_name']}`
- Task name: `{cfg['task_name']}`
- Base checkpoint: `{cfg['pretrained_ckpt']}`

## Which scripts were used

- Local upload script: `experiments/vast_training/vast_upload_repo.ksh`
- Remote training script: `experiments/vast_training/vast_run_training.ksh`
- Local download script: `experiments/vast_training/vast_download_results.ksh`

## Where each metric is stored

- Resolved experiment config: `{config_path.name}`
- Training metrics rows: `{train_metrics_path.name}`
- Validation metrics rows: `{val_metrics_path.name}`
- Evaluation summary: `{eval_summary_path.name}`
- System / hardware info: `{system_info_path.name}`
- Exact training command: `{command_path.name}`
- Full console log: `train_log.txt`
- Checkpoint summary: `checkpoint_summary.txt`

## Which checkpoint is the best

- Best checkpoint: `{checkpoint_summary.get('best_checkpoint', '') or 'not detected automatically'}`
- Last checkpoint: `{checkpoint_summary.get('last_checkpoint', '') or 'not detected automatically'}`

## How to interpret the logged files

- `training_metrics.csv` contains step/epoch rows with train-side values, including learning-rate logger rows when available.
- `validation_metrics.csv` contains validation-side rows extracted from Lightning CSV logs.
- `evaluation_summary.json` contains the final numeric values found in the post-training evaluation run.
- `system_info.txt` contains GPU, Torch, platform, timing, and source-commit information.
- `checkpoint_summary.txt` records the checkpoint directory and the paths that should be referenced in the thesis.

## What values should be copied into the thesis experiments chapter

- training and validation loss curves / final values
- learning rate and optimizer choice
- batch size, precision, image size, view count, max steps
- dataset name and split configuration
- frozen vs trainable components
- parameter counts
- GPU model and VRAM usage
- total training time and average epoch time
- best checkpoint path
- qualitative and evaluation metrics from `evaluation_summary.json`
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Bundle Fast3R fine-tuning artifacts into a thesis-ready results folder.")
    parser.add_argument("--run-spec-json", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--training-seconds", type=float, required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-dir", default="")
    parser.add_argument("--exported-checkpoint-dir", default="")
    parser.add_argument("--upload-source-info", default="")
    parser.add_argument("--fast3r-root", default="")
    args = parser.parse_args()

    run_spec_path = Path(args.run_spec_json).resolve()
    results_dir = Path(args.results_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    eval_dir = Path(args.eval_dir).resolve() if args.eval_dir else None
    upload_source_info_path = Path(args.upload_source_info).resolve() if args.upload_source_info else None
    fast3r_root = Path(args.fast3r_root).resolve() if args.fast3r_root else None

    results_dir.mkdir(parents=True, exist_ok=True)
    run_spec = _read_json(run_spec_path)
    upload_info = _read_upload_source_info(upload_source_info_path)

    config_out = results_dir / "experiment_config.json"
    _write_json(config_out, run_spec)

    command_out = results_dir / "run_command.txt"
    command_out.write_text(" ".join(run_spec.get("command", [])) + "\n", encoding="utf-8")

    _copy_if_exists(Path(args.train_log).resolve(), results_dir / "train_log.txt")
    _copy_tree_if_exists(run_dir / ".hydra", results_dir / "hydra_train")
    if eval_dir is not None:
        _copy_tree_if_exists(eval_dir / ".hydra", results_dir / "hydra_eval")

    metrics_path = _find_metrics_csv(run_dir)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    all_train_rows: list[dict[str, str]] = []
    final_train_metrics: dict[str, float] = {}
    final_val_metrics: dict[str, float] = {}
    unique_epochs: set[int] = set()
    max_step = 0
    max_vram = 0.0
    if metrics_path is not None:
        rows = _load_metrics_rows(metrics_path)
        for row in rows:
            epoch_value = _numeric_or_none(row.get("epoch"))
            step_value = _numeric_or_none(row.get("step"))
            if epoch_value is not None:
                unique_epochs.add(int(epoch_value))
            if step_value is not None:
                max_step = max(max_step, int(step_value))
            for key in ("vram/peak_allocated_gb", "vram/peak_reserved_gb"):
                metric_value = _numeric_or_none(row.get(key))
                if metric_value is not None:
                    max_vram = max(max_vram, metric_value)
            if _row_has_metric(row, "val/"):
                val_rows.append(row)
            elif _row_has_metric(row, "train/") or _row_has_metric(row, "loss") or _row_has_metric(row, "lr"):
                all_train_rows.append(row)
        train_rows = all_train_rows
        final_train_metrics = _collect_final_metrics(train_rows)
        final_val_metrics = _collect_final_metrics(val_rows)
        _copy_if_exists(metrics_path, results_dir / "all_metrics.csv")

    training_metrics_out = results_dir / "training_metrics.csv"
    validation_metrics_out = results_dir / "validation_metrics.csv"
    _write_rows(training_metrics_out, train_rows)
    _write_rows(validation_metrics_out, val_rows)

    eval_summary: dict[str, Any] = {
        "evaluation_run_dir": str(eval_dir) if eval_dir else "",
        "final_metrics": {},
    }
    if eval_dir is not None:
        eval_metrics_path = _find_metrics_csv(eval_dir)
        if eval_metrics_path is not None:
            eval_rows = _load_metrics_rows(eval_metrics_path)
            eval_summary["metrics_csv"] = str(eval_metrics_path)
            eval_summary["final_metrics"] = _collect_final_metrics(eval_rows)
            _copy_if_exists(eval_metrics_path, results_dir / "evaluation_metrics.csv")

    eval_summary_out = results_dir / "evaluation_summary.json"
    _write_json(eval_summary_out, eval_summary)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_summary = _checkpoint_summary(checkpoint_dir)
    primary_checkpoint_path = checkpoint_summary["best_checkpoint"] or checkpoint_summary["last_checkpoint"]
    primary_checkpoint = Path(primary_checkpoint_path) if primary_checkpoint_path else None
    param_counts = _state_dict_counts(primary_checkpoint, run_spec.get("trainable_name_substrings", [])) if primary_checkpoint is not None else {
        "total_parameter_count": "unavailable",
        "trainable_parameter_count": "unavailable",
        "frozen_parameter_count": "unavailable",
    }
    checkpoint_summary.update(param_counts)

    checkpoint_summary_out = results_dir / "checkpoint_summary.txt"
    _write_checkpoint_summary(checkpoint_summary_out, checkpoint_summary, args.exported_checkpoint_dir)

    copied_checkpoints_dir = results_dir / "checkpoints"
    copied_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for key in ("best_checkpoint", "last_checkpoint"):
        ckpt_path = checkpoint_summary.get(key, "")
        if ckpt_path:
            _copy_checkpoint_artifact(Path(ckpt_path), copied_checkpoints_dir / Path(ckpt_path).name)
    if args.exported_checkpoint_dir:
        export_dir = Path(args.exported_checkpoint_dir).resolve()
        if export_dir.exists():
            _copy_tree_if_exists(export_dir, results_dir / "exported_hf_checkpoint")

    fast3r_model_yaml = ""
    experiment_yaml = ""
    optimizer_name = "unavailable"
    scheduler_name = "unavailable"
    seed_value = run_spec.get("seed", "unavailable")
    if fast3r_root is not None:
        model_cfg = fast3r_root / "configs" / "model" / "fast3r.yaml"
        exp_cfg = fast3r_root / "configs" / "experiment" / "super_long_training" / "super_long_training.yaml"
        if model_cfg.exists():
            fast3r_model_yaml = model_cfg.read_text(encoding="utf-8")
            optimizer_name = _extract_yaml_target(fast3r_model_yaml, "optimizer")
            scheduler_name = _extract_yaml_target(fast3r_model_yaml, "scheduler")
        if exp_cfg.exists():
            experiment_yaml = exp_cfg.read_text(encoding="utf-8")
            match = re.search(r"^seed:\s*([0-9]+)\s*$", experiment_yaml, flags=re.MULTILINE)
            if match:
                seed_value = int(match.group(1))

    run_cfg = run_spec["run_cfg"]
    lora_report = _read_lora_report(Path(args.train_log).resolve()) if run_spec.get("lora") else {}
    train_exprs = run_spec.get("train_datasets", [])
    val_exprs = run_spec.get("validation_datasets", [])
    sample_summary = {
        "training_samples": sum(count for expr in train_exprs if (count := _parse_prefixed_count(expr)) is not None),
        "validation_samples": sum(count for expr in val_exprs if (count := _parse_prefixed_count(expr)) is not None),
        "training_splits": [_parse_split(expr) for expr in train_exprs],
        "validation_splits": [_parse_split(expr) for expr in val_exprs],
    }

    epochs_completed = len(unique_epochs)
    average_epoch_time = args.training_seconds / epochs_completed if epochs_completed > 0 else None

    summary_payload = {
        "run_name": run_cfg["run_name"],
        "task_name": run_cfg["task_name"],
        "dataset_name": run_cfg["dataset_name"],
        "dataset_root": run_cfg["dataset_root"],
        "stage": run_cfg["stage"],
        "trainable_name_substrings": run_spec.get("trainable_name_substrings", []),
        "lora": run_spec.get("lora", {}),
        "lora_report": lora_report,
        "frozen_components_summary": (
            "base weights were frozen by PEFT; only LoRA adapter matrices were trainable"
            if run_spec.get("lora")
            else "all parameters outside the selected trainable substrings were frozen by callback"
        ),
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "random_seed": seed_value,
        "batch_size": run_cfg["training"]["batch_size"],
        "batch_size_val": run_cfg["training"].get("batch_size_val", 1),
        "num_views": run_cfg["training"]["num_views"],
        "image_size": run_cfg["training"]["image_size"],
        "precision": run_cfg["training"].get("precision", ""),
        "learning_rate": run_cfg["training"]["lr"],
        "max_steps": run_cfg["training"]["max_steps"],
        "max_epochs": run_cfg["training"].get("max_epochs", ""),
        "training_time_seconds": args.training_seconds,
        "average_epoch_time_seconds": average_epoch_time,
        "epochs_completed": epochs_completed,
        "max_step_logged": max_step,
        "max_vram_gb_logged": max_vram,
        "base_checkpoint": run_cfg["pretrained_ckpt"],
        "resume_checkpoint": run_cfg.get("resume_ckpt", ""),
        "command": " ".join(run_spec.get("command", [])),
        "run_dir": str(run_dir),
        "final_train_metrics": final_train_metrics,
        "final_validation_metrics": final_val_metrics,
        "evaluation_summary": eval_summary,
        "checkpoint_summary": checkpoint_summary,
        "parameter_counts": param_counts,
        "sample_summary": sample_summary,
        "source_info": upload_info,
    }
    _write_json(results_dir / "summary_for_thesis.json", summary_payload)

    system_info_out = results_dir / "system_info.txt"
    system_info_out.write_text(_system_info_text(run_spec, upload_info, args.training_seconds), encoding="utf-8")

    readme_out = results_dir / "README_RESULTS.md"
    readme_out.write_text(
        _results_readme(
            run_spec,
            checkpoint_summary,
            training_metrics_out,
            validation_metrics_out,
            eval_summary_out,
            system_info_out,
            command_out,
            config_out,
        ),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
