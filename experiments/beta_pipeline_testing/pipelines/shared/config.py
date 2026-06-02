"""Shared configuration objects for active beta pipelines."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FrameExtractionConfig:
    """Configuration for video frame extraction."""

    num_frames: int = 300
    target_size: Tuple[int, int] = (1024, 768)
    uniform_sampling: bool = False
    skip_frames: int = 20
    selection_mode: str = "skip"
    scan_stride: int = 15
    max_frame_gap: int = 120
    min_laplacian_variance: float = 40.0
    max_clipped_ratio: float = 0.12
    min_orb_matches: int = 150
    min_thumbnail_corr: float = 0.15
    overlap_window_size: int = 3
    force_accept_after_gap: int = 240
    force_accept_after_rejections: int = 20


@dataclass
class MeshReconstructionConfig:
    """Reusable mesh reconstruction settings."""

    method: str = "bpa"
    poisson_depth: int = 8
    compute_normals: bool = True

    outlier_removal: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    bpa_radii: list = None
    remove_isolated_vertices: bool = True
    color_vertices: bool = True
    downsample_voxel_size: float = 0.01
    max_points_for_mesh: int = 5000000

    def __post_init__(self):
        if self.bpa_radii is None:
            self.bpa_radii = [0.005, 0.01, 0.02]
