from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule


_TARGET_PATTERNS = {
    "decoder_attention": r"decoder\.dec_blocks\.\d+\.attn\.(qkv|proj)",
    "decoder_attention_mlp": r"decoder\.dec_blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))",
    "decoder_and_late_encoder_attention": (
        r"(decoder\.dec_blocks\.\d+|encoder\.enc_blocks\.(18|19|20|21|22|23))"
        r"\.attn\.(qkv|proj)"
    ),
}


@dataclass(frozen=True)
class LoRAReport:
    target_scope: str
    target_modules: tuple[str, ...]
    trainable_parameter_count: int
    total_parameter_count: int


def resolve_lora_target_modules(net: nn.Module, target_scope: str) -> list[str]:
    """Return the exact linear-layer names adapted by a supported LoRA scope."""
    try:
        pattern = re.compile(_TARGET_PATTERNS[target_scope])
    except KeyError as exc:
        supported = ", ".join(sorted(_TARGET_PATTERNS))
        raise KeyError(f"Unsupported LoRA target scope: {target_scope}. Choose one of: {supported}") from exc

    target_names = [
        name
        for name, module in net.named_modules()
        if isinstance(module, nn.Linear) and pattern.fullmatch(name)
    ]
    if not target_names:
        raise RuntimeError(f"LoRA target scope {target_scope!r} did not match any nn.Linear modules.")
    return target_names


def _parameter_count(net: nn.Module, *, trainable_only: bool) -> int:
    return sum(
        param.numel()
        for param in net.parameters()
        if not trainable_only or param.requires_grad
    )


class LoRAFast3RFinetuneModule(MultiViewDUSt3RLitModule):
    """Fast3R Lightning module that adapts selected transformer layers with LoRA."""

    def __init__(
        self,
        *args: Any,
        lora_target_scope: str = "decoder_attention",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.pretrained:
            raise ValueError("LoRA fine-tuning requires a materialized GA-head checkpoint in model.pretrained.")

        self.lora_target_scope = str(lora_target_scope)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)

        # Loading must happen before adapter injection because the GA-head checkpoint
        # contains the ordinary Fast3R state-dict keys.
        self._load_pretrained_weights()
        self._inject_lora_adapters()

    def _inject_lora_adapters(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "LoRA fine-tuning requires Hugging Face PEFT. "
                "Install experiments/vast_training/requirements-lora.txt."
            ) from exc

        target_modules = resolve_lora_target_modules(self.net, self.lora_target_scope)
        config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        self.net = get_peft_model(self.net, config)
        self.lora_report = LoRAReport(
            target_scope=self.lora_target_scope,
            target_modules=tuple(target_modules),
            trainable_parameter_count=_parameter_count(self.net, trainable_only=True),
            total_parameter_count=_parameter_count(self.net, trainable_only=False),
        )
        print("[LoRAFast3RFinetuneModule] Enabled LoRA adapters:")
        print(f"  target_scope={self.lora_report.target_scope}")
        print(f"  target_modules={len(self.lora_report.target_modules)}")
        print(f"  trainable_params={self.lora_report.trainable_parameter_count:,}")
        print(f"  total_params={self.lora_report.total_parameter_count:,}")

    def setup(self, stage: str) -> None:
        # The base checkpoint is loaded and adapters are inserted in __init__ so
        # Lightning can also restore LoRA checkpoints before setup is called.
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def merge_lora_weights(self) -> nn.Module:
        """Fold adapter weights into a standalone Fast3R model for application use."""
        merge_and_unload = getattr(self.net, "merge_and_unload", None)
        if not callable(merge_and_unload):
            raise RuntimeError("Expected a PEFT LoRA model with merge_and_unload().")
        self.net = merge_and_unload(safe_merge=True)
        return self.net

