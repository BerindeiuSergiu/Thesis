from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

try:
    from vast_training.materialize_checkpoint import materialize_checkpoint
except ImportError:  # Local repository layout.
    from experiments.vast_training.materialize_checkpoint import materialize_checkpoint


def _select_checkpoint(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint
    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    return candidates[0]


def _model_config(model: torch.nn.Module) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for attr in dir(model):
        if attr.endswith("_args"):
            config[attr] = getattr(model, attr)
    return config


def export_merged_lora_checkpoint(run_dir: Path, output_path: Path) -> None:
    """Restore a LoRA Lightning run, merge its adapters, and export plain Fast3R weights."""
    run_dir = run_dir.resolve()
    output_path = output_path.resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    lit_module = hydra.utils.instantiate(
        cfg.model,
        train_criterion=None,
        validation_criterion=None,
    )

    selected_checkpoint = _select_checkpoint(run_dir)
    materialized_checkpoint = run_dir / "checkpoints" / "lora_export_aggregated.ckpt"
    materialize_checkpoint(selected_checkpoint, materialized_checkpoint)
    checkpoint = torch.load(materialized_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_result = lit_module.load_state_dict(state_dict, strict=False)
    missing_lora = [name for name in load_result.missing_keys if "lora_" in name]
    if missing_lora:
        raise RuntimeError(f"LoRA checkpoint did not restore adapter weights: {missing_lora}")

    model = lit_module.merge_lora_weights()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, config=_model_config(model))
    print(f"Merged LoRA model saved to: {output_path}")
