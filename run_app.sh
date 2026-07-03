#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x ".venv/Scripts/python.exe" ]]; then
  VENV_PYTHON=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  VENV_PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  VENV_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  VENV_PYTHON="$(command -v python)"
else
  echo "No Python interpreter found. Create a venv first: python -m venv .venv" >&2
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Expected Python interpreter not found at: $VENV_PYTHON" >&2
  exit 1
fi

exec "$VENV_PYTHON" -m src.main "$@"
