from __future__ import annotations

import torch
from lightning import Callback, LightningModule, Trainer


class VRAMMonitorCallback(Callback):
    """Log lightweight GPU memory metrics at epoch boundaries."""

    def __init__(self, log_on_epoch_end: bool = True) -> None:
        super().__init__()
        self.log_on_epoch_end = bool(log_on_epoch_end)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.log_on_epoch_end or not torch.cuda.is_available():
            return
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        if trainer.is_global_zero:
            print(f"[VRAM] peak_allocated_gb={allocated:.3f} peak_reserved_gb={reserved:.3f}")
        pl_module.log("vram/peak_allocated_gb", allocated, prog_bar=False, on_epoch=True, sync_dist=False)
        pl_module.log("vram/peak_reserved_gb", reserved, prog_bar=False, on_epoch=True, sync_dist=False)
        torch.cuda.reset_peak_memory_stats()
