from __future__ import annotations

import csv
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def generated_dir() -> Path:
    out = Path(__file__).resolve().parent / "generated"
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv_rows(relative_path: str) -> list[dict[str, str]]:
    path = repo_root() / relative_path
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(relative_path: str):
    path = repo_root() / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown(file_name: str, content: str) -> Path:
    path = generated_dir() / file_name
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def save_figure(fig, file_name: str) -> Path:
    path = generated_dir() / file_name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def to_float(value, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def to_int(value, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)
