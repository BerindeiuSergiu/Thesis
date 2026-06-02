from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from lightning import Callback, LightningModule, Trainer


@dataclass
class _FreezeReport:
    trainable_names: list[str] = field(default_factory=list)
    frozen_count: int = 0
    trainable_count: int = 0


class HeadOnlyFinetuneCallback(Callback):
    """Freeze all Fast3R params except the requested parameter groups."""

    def __init__(
        self,
        trainable_name_substrings: Iterable[str] | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.trainable_name_substrings = list(trainable_name_substrings or ["downstream_head"])
        self.verbose = bool(verbose)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        net = getattr(pl_module, "net", None)
        if net is None:
            raise RuntimeError("Expected LightningModule to expose the Fast3R network as `net`.")

        report = _FreezeReport()
        for name, param in net.named_parameters():
            should_train = any(token in name for token in self.trainable_name_substrings)
            param.requires_grad = should_train
            if should_train:
                report.trainable_names.append(name)
                report.trainable_count += param.numel()
            else:
                report.frozen_count += param.numel()

        if report.trainable_count == 0:
            raise RuntimeError(
                "HeadOnlyFinetuneCallback left zero trainable parameters. "
                f"Check the name filters: {self.trainable_name_substrings}"
            )

        if self.verbose and trainer.is_global_zero:
            print("[HeadOnlyFinetuneCallback] Enabled parameter groups:")
            for name in report.trainable_names:
                print(f"  - {name}")
            print(
                "[HeadOnlyFinetuneCallback] trainable params="
                f"{report.trainable_count:,} | frozen params={report.frozen_count:,}"
            )
