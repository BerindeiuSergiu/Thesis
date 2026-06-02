from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert_checkpoint(model_name_or_path: str, output_path: Path) -> Path:
    from fast3r.models.fast3r import Fast3R

    print(f"Loading Fast3R Hugging Face weights from: {model_name_or_path}")
    model = Fast3R.from_pretrained(model_name_or_path)
    state_dict = {f"net.{name}": tensor.detach().cpu() for name, tensor in model.state_dict().items()}

    checkpoint = {
        "state_dict": state_dict,
        "source_format": "huggingface_fast3r_model",
        "source_model": model_name_or_path,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Wrote Lightning-compatible checkpoint to: {output_path}")
    print(f"Saved {len(state_dict):,} tensors.")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Fast3R Hugging Face weights into a Lightning-style checkpoint.")
    parser.add_argument(
        "--model",
        default="jedyang97/Fast3R_ViT_Large_512",
        help="HF repo id or a local directory accepted by Fast3R.from_pretrained().",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the .ckpt file that will be written.",
    )
    args = parser.parse_args()
    convert_checkpoint(args.model, Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
