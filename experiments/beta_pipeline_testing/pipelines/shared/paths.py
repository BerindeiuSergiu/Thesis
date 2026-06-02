"""Shared path helpers for beta pipeline experiments."""

from pathlib import Path


BETA_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = BETA_ROOT.parent
REPO_ROOT = EXPERIMENTS_ROOT.parent

OUTPUTS_ROOT = BETA_ROOT / "outputs"
DOCS_ROOT = BETA_ROOT / "docs"
LOGS_ROOT = BETA_ROOT / "logs"
NOTEBOOKS_ROOT = BETA_ROOT / "notebooks"
TESTS_ROOT = BETA_ROOT / "tests"

DATA_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_VIDEO_PATH = DATA_ROOT / "irl_room_video_2.mp4"

OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
