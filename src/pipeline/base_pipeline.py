# NOTE: Do NOT import directly from experiments/.
# All pipeline code must be copied into src/pipeline/fast3r/ before modification.
# Best methods from experiments/ have been audited and integrated — see src/EXPERIMENTS_AUDIT.md

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    @abstractmethod
    def load(self, config: dict) -> None:
        """Initialize/load the pipeline with given config."""

    @abstractmethod
    def run(self, video_path: str):
        """Run the pipeline on a video file. Returns a SceneResult."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources."""
