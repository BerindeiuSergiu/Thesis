from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def materialize_checkpoint(source: Path, output_path: Path) -> Path:
    """Convert a DeepSpeed checkpoint directory into a regular Lightning file."""
    source = source.resolve()
    output_path = output_path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

        convert_zero_checkpoint_to_fp32_state_dict(
            checkpoint_dir=str(source),
            output_file=str(output_path),
            tag=None,
        )
    elif source != output_path:
        shutil.copy2(source, output_path)

    print(f"Materialized checkpoint: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a Fast3R Lightning or DeepSpeed checkpoint.")
    parser.add_argument("--input", required=True, help="Checkpoint file or DeepSpeed checkpoint directory.")
    parser.add_argument("--output", required=True, help="Destination Lightning checkpoint file.")
    args = parser.parse_args()
    materialize_checkpoint(Path(args.input), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

