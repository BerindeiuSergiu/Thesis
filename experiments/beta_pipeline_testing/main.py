"""Thin entrypoint for the legacy depth-first beta pipeline."""

from pipelines.depth_legacy.main import main


if __name__ == "__main__":
    raise SystemExit(main())
