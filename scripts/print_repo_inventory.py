from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


EXCLUDED_NAMES = {
    ".git",
    ".pip_tmp",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}


def build_inventory(repo_root: Path) -> str:
    lines: list[str] = []

    lines.append("# Repository inventory")
    lines.append(f"# Root: {repo_root}")
    lines.append(f"# Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}")
    lines.append(f"# Excluded: {', '.join(sorted(EXCLUDED_NAMES))}")
    lines.append("")
    lines.append("[DIR] .")

    for current_root, dir_names, file_names in __import__("os").walk(repo_root):
        dir_names[:] = sorted(name for name in dir_names if name not in EXCLUDED_NAMES)
        file_names[:] = sorted(name for name in file_names if name not in EXCLUDED_NAMES)

        current_path = Path(current_root)
        if current_path != repo_root:
            rel_dir = current_path.relative_to(repo_root).as_posix().replace("/", "\\")
            lines.append(f"[DIR] {rel_dir}")

        for file_name in file_names:
            rel_file = (current_path / file_name).relative_to(repo_root).as_posix().replace("/", "\\")
            lines.append(f"[FILE] {rel_file}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print or save a repository inventory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional file path to save the inventory.")
    parser.add_argument(
        "--append-to-knowledge-base",
        type=Path,
        default=None,
        help="Optional markdown file that receives the inventory appended at the end.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    inventory = build_inventory(repo_root)

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(inventory, encoding="utf-8")

    if args.append_to_knowledge_base is not None:
        kb_path = args.append_to_knowledge_base.resolve()
        original = kb_path.read_text(encoding="utf-8")
        marker = "\n## Repository Inventory\n"
        section = (
            "\n## Repository Inventory\n\n"
            "Generated project map for future agents. Environment and cache folders are excluded to keep the listing useful.\n\n"
            "```text\n"
            f"{inventory}"
            "```\n"
        )
        if marker in original:
            original = original.split(marker, 1)[0].rstrip() + "\n"
        kb_path.write_text(original + section, encoding="utf-8")

    if args.output is None and args.append_to_knowledge_base is None:
        print(inventory, end="")


if __name__ == "__main__":
    main()
