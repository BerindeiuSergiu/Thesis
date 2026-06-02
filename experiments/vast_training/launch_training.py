from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ENV = {
    "DATASET_NAME": "arkitscenes",
    "SCANNET_ROOT": "/workspace/datasets/scannet",
    "ARKITSCENES_ROOT": "/workspace/datasets/arkitscenes_processed",
    "OUTPUT_DIR": "/workspace/logs",
    "CHECKPOINT_DIR": "/workspace/checkpoints",
    "PRETRAINED_FAST3R_CKPT": "/workspace/checkpoints/fast3r_vit_large_hf_as_lightning.ckpt",
    "NUM_VIEWS": "6",
    "IMAGE_SIZE": "512",
    "BATCH_SIZE": "1",
    "LR": "1.5e-5",
    "MAX_STEPS": "12000",
    "RESUME_CKPT": "",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_layout() -> tuple[Path, Path, str, str]:
    script_path = Path(__file__).resolve()
    for anchor in [script_path.parent, *script_path.parents]:
        fast3r_root = anchor / "fast3r"
        if (fast3r_root / "fast3r" / "train.py").exists():
            experiments_dir = anchor / "experiments" / "vast_training"
            standalone_dir = anchor / "vast_training"
            if experiments_dir.exists():
                return (
                    anchor,
                    fast3r_root,
                    "experiments.vast_training.head_finetune_callback.HeadOnlyFinetuneCallback",
                    "experiments.vast_training.vram_monitor_callback.VRAMMonitorCallback",
                )
            if standalone_dir.exists():
                return (
                    anchor,
                    fast3r_root,
                    "vast_training.head_finetune_callback.HeadOnlyFinetuneCallback",
                    "vast_training.vram_monitor_callback.VRAMMonitorCallback",
                )
    raise FileNotFoundError("Could not locate fast3r/ next to vast_training/.")


def _profile_dir() -> Path:
    return Path(__file__).resolve().parent / "configs"


def _load_profile(config_name: str) -> dict[str, Any]:
    config_path = _profile_dir() / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Unknown config profile: {config_name} ({config_path})")
    return _read_json(config_path)


def _env_or_default(name: str) -> str:
    return os.environ.get(name, DEFAULT_ENV[name])


def _dataset_default_root(dataset_name: str) -> str:
    if dataset_name == "scannet":
        return _env_or_default("SCANNET_ROOT")
    if dataset_name == "arkitscenes":
        return _env_or_default("ARKITSCENES_ROOT")
    raise KeyError(f"Unsupported dataset: {dataset_name}")


def _default_task_name(dataset_name: str) -> str:
    return {
        "scannet": "vast_finetune_scannet",
        "arkitscenes": "vast_finetune_arkitscenes",
    }[dataset_name]


def _image_resolutions(image_size: int) -> list[tuple[int, int]]:
    ratios = [0.75, 0.6666667, 0.5625, 0.5, 0.3333333]
    resolutions: list[tuple[int, int]] = []
    for ratio in ratios:
        height = int(round((image_size * ratio) / 16.0) * 16)
        height = max(128, min(height, image_size))
        pair = (image_size, height)
        if pair not in resolutions:
            resolutions.append(pair)
    return resolutions


def _hydra_list(items: list[str]) -> str:
    return json.dumps(items)


def _trainer_val_check_interval(train_cfg: dict[str, Any]) -> int | float:
    interval = int(train_cfg.get("val_check_interval_steps", 250))
    limit_train_batches = train_cfg.get("limit_train_batches", 1.0)
    if isinstance(limit_train_batches, int):
        return max(1, min(interval, limit_train_batches))
    return interval


def _trainable_tokens(stage: str) -> list[str]:
    mapping = {
        "ga_head": ["downstream_head"],
        "ga_head_plus_local": ["downstream_head", "downstream_head_local"],
        "decoder_plus_ga_head": ["decoder", "downstream_head"],
        "local_head": ["downstream_head_local"],
        "lora_decoder_attention": ["lora_A", "lora_B"],
    }
    if stage not in mapping:
        raise KeyError(f"Unsupported training stage: {stage}")
    return mapping[stage]


def _is_lora_stage(stage: str) -> bool:
    return stage.startswith("lora_")


def _build_scannet_train_expr(profile: dict[str, Any], dataset_root: str, num_views: int, image_size: int) -> list[str]:
    train = profile["dataset"]["train"]
    resolution = _image_resolutions(image_size)
    expr = (
        f"{int(train['num_seq'])} @ Scannet("
        "split='train', "
        f"ROOT='{dataset_root}', "
        f"num_frames={num_views}, "
        f"num_seq={int(train['num_seq'])}, "
        f"min_thresh={int(train['min_thresh'])}, "
        f"max_thresh={int(train['max_thresh'])}, "
        f"kf_every={int(train.get('kf_every', 1))}, "
        f"frame_stride={int(train.get('frame_stride', 1))}, "
        f"blur_filter={bool(train.get('blur_filter', True))}, "
        f"blur_threshold={float(train.get('blur_threshold', 40.0))}, "
        f"max_scenes={int(train['max_scenes']) if train.get('max_scenes') is not None else 'None'}, "
        f"resolution={resolution})"
    )
    return [expr]


def _build_scannet_val_expr(profile: dict[str, Any], dataset_root: str, num_views: int, image_size: int) -> list[str]:
    val = profile["dataset"]["val"]
    primary_resolution = _image_resolutions(image_size)[0]
    datasets = [
        (
            f"{int(val['num_seq'])} @ Scannet("
            "split='val', "
            f"ROOT='{dataset_root}', "
            f"num_frames={num_views}, "
            f"num_seq={int(val['num_seq'])}, "
            f"min_thresh={int(val['min_thresh'])}, "
            f"max_thresh={int(val['max_thresh'])}, "
            f"kf_every={int(val.get('kf_every', 1))}, "
            f"frame_stride={int(val.get('frame_stride', 1))}, "
            f"blur_filter={bool(val.get('blur_filter', False))}, "
            f"blur_threshold={float(val.get('blur_threshold', 40.0))}, "
            f"max_scenes={int(val['max_scenes']) if val.get('max_scenes') is not None else 'None'}, "
            f"resolution={primary_resolution})"
        )
    ]
    return datasets


def _arkit_sample_count(cfg: dict[str, Any], fallback: int) -> int:
    return int(cfg.get("sample_count", cfg.get("num_seq", fallback)))


def _build_arkitscenes_train_expr(profile: dict[str, Any], dataset_root: str, num_views: int, image_size: int) -> list[str]:
    train = profile["dataset"]["train"]
    resolution = _image_resolutions(image_size)
    sample_count = _arkit_sample_count(train, 80_000)
    window_size = int(train.get("window_size", num_views * int(train.get("window_size_factor", 5))))
    expr = (
        f"{sample_count} @ ARKitScenes_Multiview("
        "split='train', "
        f"data_scaling={float(train.get('data_scaling', 1.0))}, "
        f"num_views={num_views}, "
        f"window_size={window_size}, "
        f"num_samples_per_window={int(train.get('num_samples_per_window', 5))}, "
        f"ordered={bool(train.get('ordered', False))}, "
        f"ROOT='{dataset_root}', "
        f"aug_crop={int(train.get('aug_crop', 256))}, "
        f"resolution={resolution}, "
        "transform=ColorJitter)"
    )
    return [expr]


def _build_arkitscenes_val_expr(profile: dict[str, Any], dataset_root: str, num_views: int, image_size: int) -> list[str]:
    val = profile["dataset"]["val"]
    primary_resolution = _image_resolutions(image_size)[0]
    sample_count = _arkit_sample_count(val, 100)
    window_size = int(val.get("window_size", num_views * int(val.get("window_size_factor", 5))))
    datasets = [
        (
            f"{sample_count} @ ARKitScenes_Multiview("
            "split='test', "
            f"data_scaling={float(val.get('data_scaling', 1.0))}, "
            f"num_views={num_views}, "
            f"window_size={window_size}, "
            f"num_samples_per_window={int(val.get('num_samples_per_window', 1))}, "
            f"ordered={bool(val.get('ordered', True))}, "
            f"ROOT='{dataset_root}', "
            f"resolution={primary_resolution}, "
            "seed=777)"
        )
    ]
    return datasets


def _build_train_expr(profile: dict[str, Any], dataset_name: str, dataset_root: str, num_views: int, image_size: int) -> list[str]:
    if dataset_name == "scannet":
        return _build_scannet_train_expr(profile, dataset_root, num_views, image_size)
    if dataset_name == "arkitscenes":
        return _build_arkitscenes_train_expr(profile, dataset_root, num_views, image_size)
    raise KeyError(f"Unsupported dataset: {dataset_name}")


def _build_val_expr(profile: dict[str, Any], dataset_name: str, dataset_root: str, num_views: int, image_size: int) -> list[str]:
    if dataset_name == "scannet":
        datasets = _build_scannet_val_expr(profile, dataset_root, num_views, image_size)
    elif dataset_name == "arkitscenes":
        datasets = _build_arkitscenes_val_expr(profile, dataset_root, num_views, image_size)
    else:
        raise KeyError(f"Unsupported dataset: {dataset_name}")

    if profile["dataset"].get("include_indoor_eval_sets", False):
        datasets.extend(
            [
                f"SevenScenes(split='test', ROOT='{profile['dataset']['seven_scenes_root']}', resolution={image_size}, num_seq=1, full_video=True, kf_every=20)",
                f"NRGBD(split='test', ROOT='{profile['dataset']['nrgbd_root']}', resolution={image_size}, num_seq=1, full_video=True, kf_every=40)",
            ]
        )
    return datasets


def _merge_profile_with_cli(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = json.loads(json.dumps(profile))
    dataset_name = (args.dataset_name or merged.get("dataset", {}).get("name") or _env_or_default("DATASET_NAME")).lower()
    if dataset_name not in {"scannet", "arkitscenes"}:
        raise KeyError(f"Unsupported dataset_name: {dataset_name}")

    merged["dataset_name"] = dataset_name
    merged["run_name"] = args.run_name or merged.get("run_name", args.config_name)
    merged["task_name"] = args.task_name or merged.get("task_name", _default_task_name(dataset_name))
    merged["stage"] = args.stage or merged.get("stage", "ga_head")
    merged["dataset_root"] = (
        args.dataset_root
        or args.arkitscenes_root
        or args.scannet_root
        or merged.get("dataset", {}).get("root")
        or _dataset_default_root(dataset_name)
    )
    merged["output_dir"] = args.output_dir or _env_or_default("OUTPUT_DIR")
    merged["checkpoint_dir"] = args.checkpoint_dir or _env_or_default("CHECKPOINT_DIR")
    merged["pretrained_ckpt"] = args.pretrained or _env_or_default("PRETRAINED_FAST3R_CKPT")
    merged["resume_ckpt"] = args.resume_ckpt or _env_or_default("RESUME_CKPT")

    train_cfg = merged.setdefault("training", {})
    train_cfg["num_views"] = args.num_views or int(os.environ.get("NUM_VIEWS", train_cfg.get("num_views", 8)))
    train_cfg["image_size"] = args.image_size or int(os.environ.get("IMAGE_SIZE", train_cfg.get("image_size", 512)))
    train_cfg["batch_size"] = args.batch_size or int(os.environ.get("BATCH_SIZE", train_cfg.get("batch_size", 1)))
    train_cfg["lr"] = args.lr or float(os.environ.get("LR", train_cfg.get("lr", 2e-5)))
    train_cfg["max_steps"] = args.max_steps or int(os.environ.get("MAX_STEPS", train_cfg.get("max_steps", 3000)))
    if args.batch_size_val is not None:
        train_cfg["batch_size_val"] = args.batch_size_val
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.num_workers_val is not None:
        train_cfg["num_workers_val"] = args.num_workers_val
    if args.precision:
        train_cfg["precision"] = args.precision

    lora_cfg = merged.setdefault("lora", {})
    lora_cfg["target_scope"] = args.lora_target_scope or lora_cfg.get("target_scope", "decoder_attention")
    lora_cfg["rank"] = args.lora_rank or int(os.environ.get("LORA_RANK", lora_cfg.get("rank", 8)))
    lora_cfg["alpha"] = args.lora_alpha or int(os.environ.get("LORA_ALPHA", lora_cfg.get("alpha", 16)))
    lora_cfg["base_source_checkpoint"] = args.lora_base_source or lora_cfg.get("base_source_checkpoint", "")
    if args.lora_dropout is not None:
        lora_cfg["dropout"] = args.lora_dropout
    else:
        lora_cfg["dropout"] = float(os.environ.get("LORA_DROPOUT", lora_cfg.get("dropout", 0.05)))
    return merged


def build_command(run_cfg: dict[str, Any]) -> tuple[list[str], Path, dict[str, str]]:
    package_root, fast3r_root, freeze_callback_target, vram_callback_target = _resolve_layout()
    train_script = fast3r_root / "fast3r" / "train.py"

    train_cfg = run_cfg["training"]
    num_views = int(train_cfg["num_views"])
    image_size = int(train_cfg["image_size"])
    dataset_root = run_cfg["dataset_root"]
    train_datasets = _build_train_expr(run_cfg, run_cfg["dataset_name"], dataset_root, num_views, image_size)
    val_datasets = _build_val_expr(run_cfg, run_cfg["dataset_name"], dataset_root, int(run_cfg["validation"]["num_views"]), image_size)
    trainable_tokens = _trainable_tokens(run_cfg["stage"])
    is_lora_stage = _is_lora_stage(run_cfg["stage"])
    callback_package = freeze_callback_target.rsplit(".", 2)[0]
    lora_module_target = f"{callback_package}.lora_finetune_module.LoRAFast3RFinetuneModule"
    hydra_run_dir = f"{run_cfg['output_dir']}/{run_cfg['task_name']}/runs/{run_cfg['run_name']}"

    command = [
        sys.executable,
        str(train_script),
        "experiment=super_long_training/super_long_training",
        "logger=csv",
        "test=False",
        f"task_name={run_cfg['task_name']}",
        f"paths.run_folder_name={run_cfg['run_name']}",
        f"model.pretrained={run_cfg['pretrained_ckpt']}",
        f"model.resume_from_checkpoint={run_cfg['resume_ckpt'] or 'null'}",
        f"model.optimizer.lr={train_cfg['lr']}",
        f"model.net.freeze={'none' if is_lora_stage else 'encoder'}",
        f"model.net.encoder_args.img_size={image_size}",
        "model.net.head_args.with_local_head=True",
        f"data.num_views={num_views}",
        f"data.num_views_val={int(run_cfg['validation']['num_views'])}",
        f"data.data_module.batch_size_per_device={int(train_cfg['batch_size'])}",
        f"data.data_module.batch_size_per_device_val={int(train_cfg.get('batch_size_val', 1))}",
        f"data.data_module.num_workers={int(train_cfg.get('num_workers', 6))}",
        f"data.data_module.num_workers_val={int(train_cfg.get('num_workers_val', 2))}",
        f"trainer.devices={run_cfg['trainer'].get('devices', 1)}",
        "trainer.plugins=[]",
        f"trainer.precision={train_cfg.get('precision', '16-mixed')}",
        f"trainer.accumulate_grad_batches={int(train_cfg.get('accumulate_grad_batches', 1))}",
        f"trainer.max_steps={int(train_cfg['max_steps'])}",
        f"trainer.max_epochs={int(train_cfg.get('max_epochs', 1000))}",
        f"trainer.limit_train_batches={float(train_cfg.get('limit_train_batches', 1.0))}",
        f"trainer.limit_val_batches={float(train_cfg.get('limit_val_batches', 1.0))}",
        f"trainer.num_sanity_val_steps={int(train_cfg.get('num_sanity_val_steps', 1))}",
        f"trainer.log_every_n_steps={int(train_cfg.get('log_every_n_steps', 10))}",
        f"trainer.val_check_interval={_trainer_val_check_interval(train_cfg)}",
        f"callbacks.model_checkpoint.every_n_epochs={int(train_cfg.get('checkpoint_every_n_epochs', 1))}",
        "callbacks.model_checkpoint.save_last=True",
        f"callbacks.early_stopping.patience={int(train_cfg.get('early_stopping_patience', 6))}",
        f"+callbacks.vram_monitor._target_={vram_callback_target}",
        "+callbacks.vram_monitor.log_on_epoch_end=True",
        f"data.data_module.train_datasets={_hydra_list(train_datasets)}",
        f"data.data_module.validation_datasets={_hydra_list(val_datasets)}",
        f"data.data_module.pin_memory={str(bool(train_cfg.get('pin_memory', True)))}",
        f"hydra.run.dir={hydra_run_dir}",
        "hydra.job.chdir=False",
    ]

    if is_lora_stage:
        lora_cfg = run_cfg["lora"]
        command.extend(
            [
                f"model._target_={lora_module_target}",
                f"+model.lora_target_scope={lora_cfg['target_scope']}",
                f"+model.lora_rank={int(lora_cfg['rank'])}",
                f"+model.lora_alpha={int(lora_cfg['alpha'])}",
                f"+model.lora_dropout={float(lora_cfg['dropout'])}",
            ]
        )
    else:
        command.extend(
            [
                f"+callbacks.freeze_selected_params._target_={freeze_callback_target}",
                f"+callbacks.freeze_selected_params.trainable_name_substrings={_hydra_list(trainable_tokens)}",
                "+callbacks.freeze_selected_params.verbose=True",
            ]
        )

    if run_cfg["resume_ckpt"]:
        command.append(f"ckpt_path={run_cfg['resume_ckpt']}")
    else:
        command.append("ckpt_path=null")

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(package_root) if not current_pythonpath else f"{package_root}{os.pathsep}{current_pythonpath}"
    return command, fast3r_root, env


def build_run_spec(run_cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = run_cfg["training"]
    num_views = int(train_cfg["num_views"])
    image_size = int(train_cfg["image_size"])
    val_num_views = int(run_cfg["validation"]["num_views"])
    dataset_root = run_cfg["dataset_root"]
    train_datasets = _build_train_expr(run_cfg, run_cfg["dataset_name"], dataset_root, num_views, image_size)
    val_datasets = _build_val_expr(run_cfg, run_cfg["dataset_name"], dataset_root, val_num_views, image_size)
    command, fast3r_root, env = build_command(run_cfg)
    hydra_run_dir = f"{run_cfg['output_dir']}/{run_cfg['task_name']}/runs/{run_cfg['run_name']}"
    return {
        "run_cfg": run_cfg,
        "run_dir": hydra_run_dir,
        "working_directory": str(fast3r_root),
        "command": command,
        "env": {
            key: env[key]
            for key in ["PYTHONPATH"]
            if key in env
        },
        "train_datasets": train_datasets,
        "validation_datasets": val_datasets,
        "trainable_name_substrings": _trainable_tokens(run_cfg["stage"]),
        "lora": run_cfg.get("lora", {}) if _is_lora_stage(run_cfg["stage"]) else {},
        "seed": 42,
        "optimizer_config_source": "fast3r/configs/model/fast3r.yaml",
        "experiment_config_source": "fast3r/configs/experiment/super_long_training/super_long_training.yaml",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Fast3R GA-head fine-tuning on ScanNet or ARKitScenes.")
    parser.add_argument("--config-name", default="vast_config")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--arkitscenes-root", default="")
    parser.add_argument("--scannet-root", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument("--num-views", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batch-size-val", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-workers-val", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--precision", default="")
    parser.add_argument("--lora-target-scope", default="")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-base-source", default="")
    parser.add_argument("--dump-run-spec", default="")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = _load_profile(args.config_name)
    run_cfg = _merge_profile_with_cli(profile, args)
    run_spec = build_run_spec(run_cfg)
    if args.dump_run_spec:
        dump_path = Path(args.dump_run_spec).resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(run_spec, indent=2), encoding="utf-8")
    command, fast3r_root, env = build_command(run_cfg)
    print("Working directory:", fast3r_root)
    print("Run directory:", run_spec["run_dir"])
    print("Command:")
    print(" ".join(command))
    if args.print_only:
        return 0
    result = subprocess.run(command, cwd=fast3r_root, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
