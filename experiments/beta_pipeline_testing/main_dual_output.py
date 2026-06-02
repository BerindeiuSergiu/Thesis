"""Thin entrypoint for the primary scaled dual-output beta pipeline."""

from pipelines.dual_output.main import main


if __name__ == "__main__":
    raise SystemExit(main())

