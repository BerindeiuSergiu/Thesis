# NOTE: Do NOT import directly from experiments/.
# All pipeline code must be copied into src/pipeline/fast3r/ before modification.
# Best methods from experiments/ have been audited and integrated — see src/EXPERIMENTS_AUDIT.md

from __future__ import annotations

from src.pipeline.base_pipeline import BasePipeline
from src.pipeline.fast3r_pipeline import Fast3rPipeline


PIPELINE_REGISTRY: dict[str, type[BasePipeline]] = {
    "default": Fast3rPipeline,
}


def get_pipeline(name: str = "default") -> BasePipeline:
    if name not in PIPELINE_REGISTRY:
        raise KeyError(f"Unknown pipeline: {name}")
    return PIPELINE_REGISTRY[name]()


def list_registered_pipelines() -> list[str]:
    return list(PIPELINE_REGISTRY.keys())
