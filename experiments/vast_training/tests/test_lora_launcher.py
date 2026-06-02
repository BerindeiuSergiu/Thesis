from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.vast_training.export_final_checkpoint import _is_lora_run
from experiments.vast_training.materialize_checkpoint import materialize_checkpoint


class LoRALauncherTest(unittest.TestCase):
    def test_lora_profile_emits_adapter_module_without_head_callback(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        launcher = repo_root / "experiments" / "vast_training" / "launch_training.py"
        result = subprocess.run(
            [
                sys.executable,
                str(launcher),
                "--config-name",
                "arkitscenes_lora_decoder_small",
                "--pretrained",
                "/workspace/checkpoints/ga_head_aggregated.ckpt",
                "--print-only",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

        output = result.stdout
        self.assertIn("LoRAFast3RFinetuneModule", output)
        self.assertIn("+model.lora_target_scope=decoder_attention", output)
        self.assertIn("+model.lora_rank=8", output)
        self.assertIn("+model.lora_alpha=16", output)
        self.assertIn("+model.lora_dropout=0.05", output)
        self.assertIn("trainer.max_steps=1000", output)
        self.assertNotIn("HeadOnlyFinetuneCallback", output)

    def test_regular_checkpoint_materialization_copies_file(self) -> None:
        source = Path("ga_head.ckpt")
        output = Path("materialized") / "ga_head_aggregated.ckpt"
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_dir", return_value=False),
            patch.object(Path, "mkdir"),
            patch("experiments.vast_training.materialize_checkpoint.shutil.copy2") as copy_file,
        ):
            result = materialize_checkpoint(source, output)

        self.assertEqual(result, output.resolve())
        copy_file.assert_called_once_with(source.resolve(), output.resolve())

    def test_lora_run_detection_reads_hydra_model_target(self) -> None:
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(
                Path,
                "read_text",
                return_value="_target_: vast_training.lora_finetune_module.LoRAFast3RFinetuneModule\n",
            ),
        ):
            self.assertTrue(_is_lora_run(Path("run")))


if __name__ == "__main__":
    unittest.main()
