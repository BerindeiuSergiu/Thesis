from __future__ import annotations

import argparse
from pathlib import Path


def _is_lora_run(run_dir: Path) -> bool:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return False
    return "LoRAFast3RFinetuneModule" in config_path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Lightning Fast3R run directory to Hugging Face checkpoint format.")
    parser.add_argument("--run-dir", required=True, help="Hydra run directory containing .hydra/ and checkpoints/.")
    parser.add_argument("--output-path", required=True, help="Directory where the HF checkpoint will be written.")
    args = parser.parse_args()

    from fast3r.utils.checkpoint_utils import convert_checkpoint_to_hf_checkpoint

    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_lora_run(run_dir):
        try:
            from vast_training.export_merged_lora_checkpoint import export_merged_lora_checkpoint
        except ImportError:  # Local repository layout.
            from experiments.vast_training.export_merged_lora_checkpoint import export_merged_lora_checkpoint

        export_merged_lora_checkpoint(run_dir, output_path)
    else:
        convert_checkpoint_to_hf_checkpoint(str(run_dir), str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
