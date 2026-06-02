from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an incremental ARKitScenes raw-download CSV containing only scenes "
            "that are not already present in the current raw Training download."
        )
    )
    parser.add_argument(
        "--source-csv",
        default="/workspace/external/ARKitScenes/raw/raw_train_val_splits.csv",
        help="Official ARKitScenes split CSV used by Apple's downloader.",
    )
    parser.add_argument(
        "--existing-raw-training-root",
        default="/workspace/datasets/arkitscenes_raw/raw/Training",
        help="Current raw Training scene root on the remote machine.",
    )
    parser.add_argument(
        "--existing-subset-csv",
        default="",
        help="Optional previously used subset CSV. Its listed scene ids are also excluded.",
    )
    parser.add_argument(
        "--output-csv",
        default="/workspace/datasets/arkitscenes_raw_subset_incremental.csv",
        help="Path to write the incremental CSV for the next download batch.",
    )
    parser.add_argument(
        "--additional-scene-count",
        type=int,
        default=48,
        help="How many unseen Training scenes to include in the next batch.",
    )
    return parser.parse_args()


def _load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return fieldnames, rows


def _find_split_key(fieldnames: list[str]) -> str | None:
    for key in fieldnames:
        if key.lower() in {"split", "fold"}:
            return key
    return None


def _find_scene_key(fieldnames: list[str], split_key: str | None) -> str:
    preferred = [
        "video_id",
        "videoid",
        "scene_id",
        "sceneid",
        "video",
        "id",
    ]
    lowered = {key.lower(): key for key in fieldnames}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    for key in fieldnames:
        if key != split_key:
            return key
    raise ValueError(f"Could not determine scene-id column from fields: {fieldnames}")


def _existing_scene_ids(raw_training_root: Path, existing_subset_csv: Path | None, scene_key: str) -> set[str]:
    existing: set[str] = set()
    if raw_training_root.exists():
        existing.update(path.name for path in raw_training_root.iterdir() if path.is_dir())
    if existing_subset_csv and existing_subset_csv.exists():
        _, rows = _load_csv_rows(existing_subset_csv)
        for row in rows:
            value = str(row.get(scene_key, "")).strip()
            if value:
                existing.add(value)
    return existing


def main() -> int:
    args = parse_args()
    source_csv = Path(args.source_csv).resolve()
    existing_raw_training_root = Path(args.existing_raw_training_root).resolve()
    existing_subset_csv = Path(args.existing_subset_csv).resolve() if args.existing_subset_csv else None
    output_csv = Path(args.output_csv).resolve()

    fieldnames, rows = _load_csv_rows(source_csv)
    split_key = _find_split_key(fieldnames)
    scene_key = _find_scene_key(fieldnames, split_key)

    if split_key is not None:
        training_rows = [row for row in rows if str(row.get(split_key, "")).strip().lower() == "training"]
        candidate_rows = training_rows if training_rows else rows
    else:
        candidate_rows = rows

    existing_ids = _existing_scene_ids(existing_raw_training_root, existing_subset_csv, scene_key)
    selected_rows: list[dict[str, str]] = []
    for row in candidate_rows:
        scene_id = str(row.get(scene_key, "")).strip()
        if not scene_id or scene_id in existing_ids:
            continue
        selected_rows.append(row)
        if len(selected_rows) >= args.additional_scene_count:
            break

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print("source_csv =", source_csv)
    print("existing_raw_training_root =", existing_raw_training_root)
    print("existing_subset_csv =", existing_subset_csv or "")
    print("scene_key =", scene_key)
    print("existing_scene_count =", len(existing_ids))
    print("requested_additional_scene_count =", args.additional_scene_count)
    print("selected_scene_count =", len(selected_rows))
    print("output_csv =", output_csv)
    print("selected_scene_ids =", [str(row.get(scene_key, "")).strip() for row in selected_rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
