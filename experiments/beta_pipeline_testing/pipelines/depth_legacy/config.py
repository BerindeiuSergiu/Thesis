"""Updated & working config for 2025 indoor 3D reconstruction."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from ..shared.config import FrameExtractionConfig, MeshReconstructionConfig
from ..shared.paths import DEFAULT_VIDEO_PATH, OUTPUTS_ROOT


@dataclass
class DepthEstimationConfig:
    """Configuration for Marigold depth estimation (metric depth)."""
    model_type: str = "marigold"             # MUST be "marigold" (MiDAS inadequate for rooms)
    marigold_model: str = "prs-eth/marigold-depth-v1-0"  # Official working repo
    input_size: Tuple[int, int] = (1024, 768)
    device: str = "cuda"
    half_precision: bool = True
    # Marigold-specific parameters
    denoising_steps: int = 10                # Reduced from 20 for speed (still good quality)
    ensemble_size: int = 4                   # Reduced from 8 for speed (less noise filtering needed)
    batch_size: int = 8                      # Increased from 4 to process more frames at once


@dataclass
class PointCloudConfig:
    """Configuration for point cloud generation with improved intrinsics."""
    # Better intrinsics for 1024x768 resolution
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 512.0                        # 1024 / 2
    cy: float = 384.0   
    depth_scale: float = 1.0                 # 768 / 2
    min_depth: float = 0.3
    max_depth: float = 10.0
    voxel_size: float = 0.02                 # Tighter voxels = cleaner geometry


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    video_path: Path = DEFAULT_VIDEO_PATH
    output_dir: Path = OUTPUTS_ROOT / "checkpointed_depth"
    
    frame_config: FrameExtractionConfig = None
    depth_config: DepthEstimationConfig = None
    point_cloud_config: PointCloudConfig = None
    mesh_config: MeshReconstructionConfig = None
    
    verbose: bool = True
    save_intermediate: bool = True
    
    def __post_init__(self):
        """Initialize default sub-configs if not provided."""
        if self.frame_config is None:
            self.frame_config = FrameExtractionConfig()
        if self.depth_config is None:
            self.depth_config = DepthEstimationConfig()
        if self.point_cloud_config is None:
            self.point_cloud_config = PointCloudConfig()
        if self.mesh_config is None:
            self.mesh_config = MeshReconstructionConfig()
        
        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# THIS IS THE ONE YOU SHOULD USE RIGHT NOW
DEFAULT_CONFIG = PipelineConfig(
    depth_config=DepthEstimationConfig(
        model_type="marigold",
        marigold_model="prs-eth/marigold-depth-v1-0",
        denoising_steps=10,
        ensemble_size=4,
        batch_size=8,
    ),
    frame_config=FrameExtractionConfig(
        num_frames=300,
        target_size=(1024, 768),
        uniform_sampling=False,    # MOST IMPORTANT CHANGE
        skip_frames=20,             # Every 20th frame at 60fps = ~0.33 seconds between frames
    ),
    mesh_config=MeshReconstructionConfig(
        method="bpa",
        bpa_radii=[0.005, 0.01, 0.02]
    )
)
