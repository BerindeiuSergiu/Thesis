from __future__ import annotations

import csv
import sys
import time
from pathlib import Path


def pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    for key, value in row.items():
        if value != "" and any(name in key for name in names):
            return value
    return "-"


def main() -> int:
    path = Path(sys.argv[1])
    while True:
        if not path.exists() or path.stat().st_size == 0:
            print(f"waiting for {path}")
            time.sleep(10)
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print(f"waiting for rows in {path}")
            time.sleep(10)
            continue
        row = rows[-1]
        print(
            "step={} epoch={} train_loss={} val_loss={} lr={} samples={} images={}".format(
                pick(row, "step"),
                pick(row, "epoch", "trainer/epoch"),
                pick(row, "train/loss"),
                pick(row, "val/loss"),
                pick(row, "trainer/lr", "lr"),
                pick(row, "trainer/total_samples", "total_samples"),
                pick(row, "trainer/total_images", "total_images"),
            )
        )
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
