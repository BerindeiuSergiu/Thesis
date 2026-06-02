from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# This is the exact synthetic holdout we used on 2026-05-26 when the machine
# only had ARKitScenes raw Training scenes and not the official Validation split.
# We keep the list here on purpose so a future machine can recreate the same
# processed dataset layout without having to recover old notebook cells or logs.
DEFAULT_HELD_OUT_SCENES = [
    "41069099",
    "41069100",
    "41069111",
    "41069115",
    "41069117",
    "41069125",
    "41069127",
    "41069128",
    "41069130",
    "41069132",
    "41069134",
    "41069135",
    "41069140",
    "41069142",
    "41069144",
    "41069146",
    "41069157",
    "41069161",
    "41069162",
    "41069164",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the synthetic ARKitScenes Training/Validation subset used for the "
            "Fast3R subset experiment when only raw Training scenes are available."
        )
    )
    parser.add_argument(
        "--raw-training-root",
        default="/workspace/datasets/arkitscenes_raw/raw/Training",
        help="Directory containing the downloaded raw ARKitScenes Training scene folders.",
    )
    parser.add_argument(
        "--subset-root",
        default="/workspace/datasets/arkitscenes_raw_trainval_subset",
        help="Where to create the synthetic Training/Validation split.",
    )
    parser.add_argument(
        "--processed-root",
        default="/workspace/datasets/arkitscenes_processed",
        help="Where to rebuild the Fast3R-compatible processed dataset.",
    )
    parser.add_argument(
        "--prepare-script",
        default="/workspace/vast_training/prepare_arkitscenes_processed.py",
        help="Path to the preparation script that converts the synthetic split into the processed layout.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to use when invoking the preparation script.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How prepare_arkitscenes_processed.py should materialize scene folders.",
    )
    parser.add_argument(
        "--skip-processed-rebuild",
        action="store_true",
        help="Only recreate the synthetic raw subset tree and skip rebuilding arkitscenes_processed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing subset_root and processed_root before recreating them.",
    )
    return parser.parse_args()


def remove_tree_if_needed(path: Path, *, force: bool, label: str) -> None:
    if not path.exists():
        return
    if not force:
        raise SystemExit(
            f"{label} already exists: {path}\n"
            "Re-run with --force if you really want to replace it."
        )
    shutil.rmtree(path)


def create_subset(raw_training_root: Path, subset_root: Path, held_out_scenes: list[str]) -> tuple[list[str], list[str]]:
    scene_names = sorted(path.name for path in raw_training_root.iterdir() if path.is_dir())
    scene_set = set(scene_names)
    missing = [scene for scene in held_out_scenes if scene not in scene_set]
    if missing:
        raise SystemExit(
            "The synthetic holdout cannot be recreated because some expected raw Training "
            f"scenes are missing: {missing}"
        )

    held_out_set = set(held_out_scenes)
    train_scenes = [scene for scene in scene_names if scene not in held_out_set]

    (subset_root / "Training").mkdir(parents=True, exist_ok=True)
    (subset_root / "Validation").mkdir(parents=True, exist_ok=True)

    # We intentionally use directory symlinks here because the subset is just a
    # lightweight view over the real raw Training download. This keeps rebuilds cheap.
    for scene in train_scenes:
        (subset_root / "Training" / scene).symlink_to(raw_training_root / scene, target_is_directory=True)
    for scene in held_out_scenes:
        (subset_root / "Validation" / scene).symlink_to(raw_training_root / scene, target_is_directory=True)

    return train_scenes, held_out_scenes


def run_prepare_script(args: argparse.Namespace, subset_root: Path, processed_root: Path) -> None:
    command = [
        args.python_bin,
        str(Path(args.prepare_script)),
        "--raw-root",
        str(subset_root),
        "--output-root",
        str(processed_root),
        "--link-mode",
        args.link_mode,
    ]
    print("Running prepare script:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def print_processed_summary(processed_root: Path) -> None:
    for split in ["Training", "Test"]:
        metadata_path = processed_root / split / "all_metadata.npz"
        if not metadata_path.exists():
            print(f"{split}: metadata missing at {metadata_path}")
            continue
        data = np.load(metadata_path, allow_pickle=True)
        print()
        print(f"=== {split} ===")
        print("metadata:", metadata_path)
        print("scenes:", len(data["scenes"]))
        print("frames:", len(data["images"]))
        print("pairs:", data["pairs"].shape)
        print("first scenes:", data["scenes"][:10].tolist())


def main() -> int:
    args = parse_args()

    raw_training_root = Path(args.raw_training_root).resolve()
    subset_root = Path(args.subset_root).resolve()
    processed_root = Path(args.processed_root).resolve()

    if not raw_training_root.exists():
        raise SystemExit(f"Missing raw Training root: {raw_training_root}")

    remove_tree_if_needed(subset_root, force=args.force, label="subset_root")
    if not args.skip_processed_rebuild:
        remove_tree_if_needed(processed_root, force=args.force, label="processed_root")

    train_scenes, held_out_scenes = create_subset(
        raw_training_root=raw_training_root,
        subset_root=subset_root,
        held_out_scenes=DEFAULT_HELD_OUT_SCENES,
    )

    print("Synthetic subset created.")
    print("subset_root =", subset_root)
    print("train scenes =", len(train_scenes))
    print("val scenes =", len(held_out_scenes))
    print("val scene ids =", held_out_scenes)

    if args.skip_processed_rebuild:
        print()
        print("Skipping processed rebuild because --skip-processed-rebuild was set.")
        return 0

    run_prepare_script(args, subset_root, processed_root)
    print_processed_summary(processed_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
