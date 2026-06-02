from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from omegaconf import OmegaConf


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_layout() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    for anchor in [script_path.parent, *script_path.parents]:
        fast3r_root = anchor / "fast3r"
        if (fast3r_root / "fast3r" / "eval.py").exists():
            return anchor, fast3r_root
    raise FileNotFoundError("Could not locate fast3r/ next to vast_training/.")


def _profile_dir() -> Path:
    return Path(__file__).resolve().parent / "configs"


def _load_profile(config_name: str) -> dict[str, Any]:
    return _read_json(_profile_dir() / f"{config_name}.json")


def _hydra_list(items: list[str]) -> str:
    return json.dumps(items)


def _image_resolution(image_size: int) -> tuple[int, int]:
    height = int(round((image_size * 0.75) / 16.0) * 16)
    return image_size, max(128, height)


def _default_dataset_root(dataset_name: str) -> str:
    if dataset_name == "scannet":
        return os.environ.get("SCANNET_ROOT", "/workspace/datasets/scannet")
    if dataset_name == "arkitscenes":
        return os.environ.get("ARKITSCENES_ROOT", "/workspace/datasets/arkitscenes_processed")
    raise KeyError(f"Unsupported dataset_name: {dataset_name}")


def _checkpoint_sidecar_path(checkpoint_path: Path) -> Path:
    return (checkpoint_path.parent.parent / ".hydra" / "config.yaml").resolve()


def _checkpoint_is_weights_only_safe(checkpoint_path: Path) -> bool:
    try:
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return True
    except Exception:
        return False


def _write_lightning_compatible_checkpoint(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict in {src}, got {type(checkpoint)!r}")

    state_dict = checkpoint.get("state_dict", {})
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state_dict in {src} to be a dict, got {type(state_dict)!r}")
    state_dict.setdefault("epoch_fraction", torch.tensor(0.0, dtype=torch.float32))
    state_dict.setdefault("train_total_samples", torch.tensor(0, dtype=torch.long))
    state_dict.setdefault("train_total_images", torch.tensor(0, dtype=torch.long))

    sanitized_checkpoint = {
        "state_dict": state_dict,
        "pytorch-lightning_version": str(checkpoint.get("pytorch-lightning_version", "2.5.5")),
        "epoch": int(checkpoint.get("epoch", 0) or 0),
        "global_step": int(checkpoint.get("global_step", 0) or 0),
    }
    torch.save(sanitized_checkpoint, dst)


def _resolve_checkpoint_payload(src: Path, bundle_root: Path) -> Path:
    if src.is_dir():
        aggregated_src = bundle_root / "checkpoints" / "source_last_aggregated.ckpt"
        aggregated_src.parent.mkdir(parents=True, exist_ok=True)
        if not aggregated_src.exists():
            convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_dir=str(src),
                output_file=str(aggregated_src),
                tag=None,
            )
        return aggregated_src
    return src


def _copy_checkpoint_sidecar_if_present(src_checkpoint_path: Path, dst_cfg_path: Path) -> bool:
    src_cfg_path = _checkpoint_sidecar_path(src_checkpoint_path)
    if not src_cfg_path.exists():
        return False
    dst_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_cfg_path, dst_cfg_path)
    return True


def _materialize_checkpoint_bundle(
    fast3r_root: Path,
    run_name: str,
    original_checkpoint_path: Path,
    image_size: int,
    output_dir: str,
    *,
    force_standalone_model_cfg: bool,
) -> Path:
    bundle_root = Path(output_dir) / "eval_checkpoint_bundles" / run_name
    bundle_ckpt_path = bundle_root / "checkpoints" / "last.ckpt"
    bundle_cfg_path = bundle_root / ".hydra" / "config.yaml"

    payload_path = _resolve_checkpoint_payload(original_checkpoint_path, bundle_root)
    _write_lightning_compatible_checkpoint(payload_path, bundle_ckpt_path)

    if not force_standalone_model_cfg and _copy_checkpoint_sidecar_if_present(original_checkpoint_path, bundle_cfg_path):
        return bundle_ckpt_path

    model_cfg = OmegaConf.load(fast3r_root / "configs" / "model" / "fast3r.yaml")
    OmegaConf.update(model_cfg, "pretrained", str(original_checkpoint_path), force_add=True)
    OmegaConf.update(model_cfg, "resume_from_checkpoint", str(bundle_ckpt_path), force_add=True)
    OmegaConf.update(model_cfg, "net.encoder_args.img_size", image_size, force_add=True)
    OmegaConf.update(model_cfg, "net.head_args.with_local_head", True, force_add=True)
    OmegaConf.update(model_cfg, "net.decoder_args.embed_dim", 1024, force_add=True)
    OmegaConf.update(model_cfg, "net.decoder_args.num_heads", 16, force_add=True)
    OmegaConf.update(model_cfg, "net.decoder_args.depth", 24, force_add=True)

    bundle_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create({"model": model_cfg}), f=str(bundle_cfg_path))
    return bundle_ckpt_path


def _scannet_val_datasets(profile: dict[str, Any], dataset_root: str, num_views: int, image_size: int) -> list[str]:
    width, height = _image_resolution(image_size)
    val = profile["dataset"]["val"]
    return [
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
            "blur_filter=False, "
            f"blur_threshold={float(val.get('blur_threshold', 40.0))}, "
            f"max_scenes={int(val['max_scenes']) if val.get('max_scenes') is not None else 'None'}, "
            f"resolution=({width}, {height}))"
        )
    ]


def _arkitscenes_val_datasets(profile: dict[str, Any], dataset_root: str, num_views: int, image_size: int) -> list[str]:
    width, height = _image_resolution(image_size)
    val = profile["dataset"]["val"]
    sample_count = int(val.get("sample_count", val.get("num_seq", 100)))
    window_size = int(val.get("window_size", num_views * int(val.get("window_size_factor", 5))))
    return [
        (
            f"{sample_count} @ ARKitScenes_Multiview("
            "split='test', "
            f"data_scaling={float(val.get('data_scaling', 1.0))}, "
            f"num_views={num_views}, "
            f"window_size={window_size}, "
            f"num_samples_per_window={int(val.get('num_samples_per_window', 1))}, "
            f"ordered={bool(val.get('ordered', True))}, "
            f"ROOT='{dataset_root}', "
            f"resolution=({width}, {height}), "
            "seed=777)"
        )
    ]


def build_command(profile: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], Path, dict[str, str]]:
    package_root, fast3r_root = _resolve_layout()
    eval_script = fast3r_root / "fast3r" / "eval.py"

    dataset_name = (args.dataset_name or profile.get("dataset", {}).get("name") or os.environ.get("DATASET_NAME", "arkitscenes")).lower()
    image_size = int(args.image_size or profile["training"]["image_size"])
    num_views = int(args.num_views or profile["validation"]["num_views"])
    dataset_root = args.dataset_root or args.arkitscenes_root or args.scannet_root or profile.get("dataset", {}).get("root") or _default_dataset_root(dataset_name)
    output_dir = args.output_dir or os.environ.get("OUTPUT_DIR", "/workspace/logs")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path_for_eval = checkpoint_path
    extra_model_overrides: list[str] = []

    checkpoint_has_sidecar = _checkpoint_sidecar_path(checkpoint_path).exists() if checkpoint_path.exists() else False
    checkpoint_needs_bundle = checkpoint_path.is_dir() or (
        checkpoint_path.is_file() and not _checkpoint_is_weights_only_safe(checkpoint_path)
    )

    if checkpoint_needs_bundle:
        checkpoint_path_for_eval = _materialize_checkpoint_bundle(
            fast3r_root=fast3r_root,
            run_name=args.run_name,
            original_checkpoint_path=checkpoint_path,
            image_size=image_size,
            output_dir=output_dir,
            force_standalone_model_cfg=not checkpoint_has_sidecar,
        )
    elif checkpoint_path.is_file() and not checkpoint_has_sidecar:
        checkpoint_path_for_eval = _materialize_checkpoint_bundle(
            fast3r_root=fast3r_root,
            run_name=args.run_name,
            original_checkpoint_path=checkpoint_path,
            image_size=image_size,
            output_dir=output_dir,
            force_standalone_model_cfg=True,
        )
        extra_model_overrides = [
            f"model.pretrained={checkpoint_path}",
            "model.resume_from_checkpoint=null",
            "+model.net.head_args.with_local_head=True",
            f"model.net.encoder_args.img_size={image_size}",
            "model.net.decoder_args.embed_dim=1024",
            "model.net.decoder_args.num_heads=16",
            "model.net.decoder_args.depth=24",
        ]

    if dataset_name == "scannet":
        validation_datasets = _scannet_val_datasets(profile, dataset_root, num_views, image_size)
    elif dataset_name == "arkitscenes":
        validation_datasets = _arkitscenes_val_datasets(profile, dataset_root, num_views, image_size)
    else:
        raise KeyError(f"Unsupported dataset_name: {dataset_name}")

    if profile["dataset"].get("include_indoor_eval_sets", False):
        validation_datasets.extend(
            [
                f"SevenScenes(split='test', ROOT='{profile['dataset']['seven_scenes_root']}', resolution={image_size}, num_seq=1, full_video=True, kf_every=20)",
                f"NRGBD(split='test', ROOT='{profile['dataset']['nrgbd_root']}', resolution={image_size}, num_seq=1, full_video=True, kf_every=40)",
            ]
        )

    hydra_run_dir = output_dir + f"/eval_runs/{args.run_name}"
    command = [
        sys.executable,
        str(eval_script),
        "eval=eval_cam_pose/default",
        "logger=csv",
        f"ckpt_path={checkpoint_path_for_eval}",
        f"data.num_views_val={num_views}",
        "data.data_module.train_datasets=[]",
        f"data.data_module.validation_datasets={_hydra_list(validation_datasets)}",
        f"data.data_module.batch_size_per_device_val={int(args.batch_size_val or 1)}",
        f"+data.data_module.num_workers_val={int(args.num_workers_val or 2)}",
        "trainer.plugins=[]",
        f"hydra.run.dir={hydra_run_dir}",
        "hydra.job.chdir=False",
    ]
    command.extend(extra_model_overrides)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(package_root) if not current_pythonpath else f"{package_root}{os.pathsep}{current_pythonpath}"
    return command, fast3r_root, env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Fast3R fine-tune run on ScanNet or ARKitScenes.")
    parser.add_argument("--config-name", default="vast_config")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--arkitscenes-root", default="")
    parser.add_argument("--scannet-root", default="")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--run-name", default="eval_run")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=0)
    parser.add_argument("--batch-size-val", type=int, default=1)
    parser.add_argument("--num-workers-val", type=int, default=2)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = _load_profile(args.config_name)
    command, fast3r_root, env = build_command(profile, args)
    print("Working directory:", fast3r_root)
    print("Command:")
    print(" ".join(command))
    if args.print_only:
        return 0
    result = subprocess.run(command, cwd=fast3r_root, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
