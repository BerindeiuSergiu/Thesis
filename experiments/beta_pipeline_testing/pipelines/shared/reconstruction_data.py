"""Shared data containers for model reconstruction outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ReconstructionOutput:
    """Point-based reconstruction plus camera/model metadata.

    Points and poses are kept in the same coordinate system emitted by the
    inference model. Scale normalization is responsible for converting them to
    real-world units before any output branch consumes them.
    """

    points: np.ndarray
    colors: Optional[np.ndarray] = None
    poses: Optional[list[np.ndarray]] = None
    intrinsics: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    scale_applied: float = 1.0

    def copy(self) -> "ReconstructionOutput":
        return ReconstructionOutput(
            points=np.array(self.points, copy=True),
            colors=None if self.colors is None else np.array(self.colors, copy=True),
            poses=None if self.poses is None else [np.array(pose, copy=True) for pose in self.poses],
            intrinsics=None if self.intrinsics is None else np.array(self.intrinsics, copy=True),
            metadata=dict(self.metadata),
            scale_applied=float(self.scale_applied),
        )

