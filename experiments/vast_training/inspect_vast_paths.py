from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the relevant Vast.ai / ARKitScenes paths and a quick summary."
    )
    parser.add_argument("--remote-root", default=os.environ.get("REMOTE_ROOT", "/workspace"))
    parser.add_argument(
        "--raw-root",
        default=os.environ.get("ARKITSCENES_RAW_ROOT", "/workspace/datasets/arkitscenes_raw"),
    )
    parser.add_argument(
        "--processed-root",
        default=os.environ.get("DATASET_ROOT", "/workspace/datasets/arkitscenes_processed"),
    )
    parser.add_argument(
        "--subset-root",
        default="/workspace/datasets/arkitscenes_raw_trainval_subset",
    )
    parser.add_argument(
        "--video-id-csv",
        default=os.environ.get(
            "ARKITSCENES_VIDEO_ID_CSV",
            "/workspace/datasets/arkitscenes_raw_subset_approx_100gb.csv",
        ),
    )
    parser.add_argument("--sample-limit", type=int, default=10)
    return parser.parse_args()


def fmt_path(path: Path) -> str:
    suffix = []
    if path.exists():
        if path.is_symlink():
            try:
                suffix.append(f"symlink->{path.resolve()}")
            except OSError:
                suffix.append("symlink-><broken>")
        elif path.is_dir():
            suffix.append("dir")
        elif path.is_file():
            suffix.append("file")
    else:
        suffix.append("missing")
    return f"{path} [{' ,'.join(suffix) if suffix else ''}]"


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_path_block(label: str, path: Path, sample_limit: int) -> None:
    print(f"{label}: {fmt_path(path)}")
    if not path.exists():
        return

    if path.is_dir():
        children = sorted(path.iterdir(), key=lambda item: item.name)[:sample_limit]
        if children:
            print("  sample children:")
            for child in children:
                print(f"    - {child.name}")
        else:
            print("  directory is empty")


def count_scene_dirs(path: Path) -> int | None:
    if not path.exists() or not path.is_dir():
        return None
    return sum(1 for item in path.iterdir() if item.is_dir())


def print_split_summary(label: str, split_root: Path) -> None:
    print(f"{label}: {fmt_path(split_root)}")
    scene_count = count_scene_dirs(split_root)
    if scene_count is not None:
        print(f"  scene_count={scene_count}")


def main() -> int:
    args = parse_args()

    remote_root = Path(args.remote_root)
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    subset_root = Path(args.subset_root)
    video_id_csv = Path(args.video_id_csv)
    sample_limit = args.sample_limit

    print_header("Environment")
    for key in [
        "PYTHON_BIN",
        "DATASET_NAME",
        "DATASET_ROOT",
        "ARKITSCENES_RAW_ROOT",
        "ARKITSCENES_VIDEO_ID_CSV",
        "DO_SETUP",
        "DO_DOWNLOAD_DATASET",
        "DO_PREPARE_DATASET",
        "DO_TRAIN",
        "DO_EVALUATE",
        "DO_EXPORT_CHECKPOINT",
    ]:
        print(f"{key}={os.environ.get(key, '')}")

    print_header("Primary Paths")
    print_path_block("remote_root", remote_root, sample_limit)
    print_path_block("fast3r_dir", remote_root / "fast3r", sample_limit)
    print_path_block("vast_training_dir", remote_root / "vast_training", sample_limit)
    print_path_block("datasets_dir", remote_root / "datasets", sample_limit)
    print_path_block("raw_root", raw_root, sample_limit)
    print_path_block("processed_root", processed_root, sample_limit)
    print_path_block("subset_root", subset_root, sample_limit)
    print_path_block("video_id_csv", video_id_csv, sample_limit)

    print_header("Raw Split Candidates")
    raw_candidates = [
        raw_root / "Training",
        raw_root / "Validation",
        raw_root / "raw" / "Training",
        raw_root / "raw" / "Validation",
        subset_root / "Training",
        subset_root / "Validation",
    ]
    for candidate in raw_candidates:
        print_split_summary(candidate.name if candidate.parent != subset_root else f"{candidate.parent.name}/{candidate.name}", candidate)

    print_header("Processed Split Candidates")
    processed_candidates = [
        processed_root / "Training",
        processed_root / "Test",
        processed_root / "Training" / "all_metadata.npz",
        processed_root / "Test" / "all_metadata.npz",
    ]
    for candidate in processed_candidates:
        print_path_block(candidate.name if candidate.parent != processed_root else candidate.name, candidate, sample_limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
