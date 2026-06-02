"""Main 3D reconstruction pipeline."""
from pathlib import Path
import numpy as np
import cv2
import json
import torch
import gc
from datetime import datetime

from ..shared.paths import DEFAULT_VIDEO_PATH, OUTPUTS_ROOT
from ..shared.video_processor import FrameExtractor
from .config import PipelineConfig, DEFAULT_CONFIG
from .depth_estimator import create_depth_estimator
from .reconstruction_3d import PointCloudGenerator, MeshReconstructor

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


class ReconstructionPipeline:
    """Complete 3D reconstruction pipeline."""
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize pipeline.
        
        Args:
            config: PipelineConfig instance
        """
        self.config = config or DEFAULT_CONFIG
        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames = None
        self.frame_metadata = None
        self.depth_maps = None
        self.pointclouds = None
        self.merged_points = None
        self.merged_colors = None
        self.mesh = None
        
        self._log("Pipeline initialized")
    
    def _log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.config.verbose:
            print(f"[{timestamp}] {message}")
    
    def _get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        checkpoint_dir = self.config.output_dir / ".checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def _save_checkpoint(self, name: str, data: dict):
        """Save checkpoint data.
        
        Args:
            name: Checkpoint name (e.g., 'frames', 'depth_maps', 'point_clouds')
            data: Dictionary to save
        """
        checkpoint_dir = self._get_checkpoint_dir()
        checkpoint_path = checkpoint_dir / f"{name}.npz"
        
        try:
            np.savez_compressed(checkpoint_path, **data)
            self._log(f"Saved checkpoint: {name}")
        except Exception as e:
            self._log(f"Failed to save checkpoint {name}: {e}")
    
    def _load_checkpoint(self, name: str) -> dict:
        """Load checkpoint data.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Dictionary loaded from checkpoint, or None if not found
        """
        checkpoint_dir = self._get_checkpoint_dir()
        checkpoint_path = checkpoint_dir / f"{name}.npz"
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with np.load(checkpoint_path, allow_pickle=True) as data:
                result = {key: data[key] for key in data.files}
            self._log(f"Loaded checkpoint: {name}")
            return result
        except Exception as e:
            self._log(f"Failed to load checkpoint {name}: {e}")
            # Try to remove corrupted checkpoint
            try:
                checkpoint_path.unlink()
                self._log(f"Removed corrupted checkpoint: {name}")
            except:
                pass
            return None
    
    def _checkpoint_exists(self, name: str) -> bool:
        """Check if checkpoint exists."""
        checkpoint_dir = self._get_checkpoint_dir()
        return (checkpoint_dir / f"{name}.npz").exists()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.config.verbose:
            print(f"[{timestamp}] {message}")
    
    def run(self) -> bool:
        """Run the complete pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._log("=" * 70)
            self._log("3D RECONSTRUCTION PIPELINE (BETA)")
            self._log("=" * 70)
            
            # Stage 1: Extract frames
            self._log("\nSTAGE 1: Frame Extraction")
            self._log("-" * 70)
            if not self.extract_frames():
                return False
            
            # Stage 2: Estimate depth
            depth_model = self.config.depth_config.model_type.upper()
            self._log(f"\nSTAGE 2: Depth Estimation ({depth_model})")
            self._log("-" * 70)
            if not self.estimate_depth():
                return False
            
            # Stage 3: Generate point clouds
            self._log("\nSTAGE 3: Point Cloud Generation")
            self._log("-" * 70)
            if not self.generate_pointclouds():
                return False
            
            # Stage 4: Reconstruct mesh
            self._log("\nSTAGE 4: 3D Mesh Reconstruction")
            self._log("-" * 70)
            if not self.reconstruct_mesh():
                return False
            
            # Stage 5: Save results
            self._log("\nSTAGE 5: Saving Results")
            self._log("-" * 70)
            if not self.save_results():
                return False
            
            self._log("\n" + "=" * 70)
            self._log("PIPELINE COMPLETED SUCCESSFULLY")
            self._log("=" * 70)
            
            self.print_summary()
            
            return True
            
        except Exception as e:
            self._log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_frames(self) -> bool:
        """Extract frames from video with checkpoint support."""
        try:
            # Check if frames checkpoint exists and is from the same video
            if self._checkpoint_exists("frames"):
                checkpoint = self._load_checkpoint("frames")
                if checkpoint is not None:
                    # Verify checkpoint is from the same video
                    stored_video = str(checkpoint.get("video_path", ""))
                    current_video = str(self.config.video_path.resolve())
                    
                    if stored_video == current_video:
                        self.frames = [checkpoint[f"frame_{i}"] for i in range(len(checkpoint) - 2)]  # -2 for video_path and metadata
                        self.frame_metadata = checkpoint["metadata"]
                        self._log(f"Loaded {len(self.frames)} frames from checkpoint (video: {self.config.video_path.name})")
                        return True
                    else:
                        self._log(f"Checkpoint is from different video, recalculating frames")
                        # Delete old checkpoint
                        checkpoint_dir = self._get_checkpoint_dir()
                        (checkpoint_dir / "frames.npz").unlink(missing_ok=True)
            
            extractor = FrameExtractor(self.config.frame_config)
            self.frames, self.frame_metadata = extractor.extract_frames(
                self.config.video_path
            )
            
            self._log(f"Extracted {len(self.frames)} frames from {self.config.video_path.name}")
            
            # Save checkpoint with video path for validation
            if self.config.save_intermediate:
                checkpoint_data = {
                    f"frame_{i}": frame for i, frame in enumerate(self.frames)
                }
                checkpoint_data["metadata"] = np.array([self.frame_metadata], dtype=object)[0]
                checkpoint_data["video_path"] = np.array([str(self.config.video_path.resolve())], dtype=object)[0]
                self._save_checkpoint("frames", checkpoint_data)
            
            # Save frames to disk for COLMAP and preview
            if self.config.save_intermediate:
                frames_dir = self.config.output_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                
                # Save ALL frames (for COLMAP)
                for i, frame in enumerate(self.frames):
                    cv2.imwrite(
                        str(frames_dir / f"frame_{i:06d}.jpg"),
                        frame
                    )
                self._log(f"Saved all {len(self.frames)} frames to {frames_dir}")
            
            return True
            
        except Exception as e:
            self._log(f"Frame extraction failed: {e}")
            return False
    
    def estimate_depth(self) -> bool:
        """Estimate depth for all frames using configured model with checkpoint support."""
        try:
            # Check if depth maps checkpoint exists
            if self._checkpoint_exists("depth_maps"):
                checkpoint = self._load_checkpoint("depth_maps")
                if checkpoint is not None:
                    self.depth_maps = [checkpoint[f"depth_{i}"] for i in range(len(checkpoint))]
                    self._log(f"Loaded {len(self.depth_maps)} depth maps from checkpoint")
                    return True
            
            estimator = create_depth_estimator(self.config.depth_config)
            self._log(f"Estimating depth for {len(self.frames)} frames...")
            
            self.depth_maps = estimator.estimate_batch(self.frames)
            
            self._log(f"Generated {len(self.depth_maps)} depth maps")
            
            # Save checkpoint
            if self.config.save_intermediate:
                checkpoint_data = {
                    f"depth_{i}": depth for i, depth in enumerate(self.depth_maps)
                }
                self._save_checkpoint("depth_maps", checkpoint_data)
            
            # Save sample depth maps visualization
            if self.config.save_intermediate:
                depth_dir = self.config.output_dir / "depth_maps"
                depth_dir.mkdir(exist_ok=True)
                
                for i, depth in enumerate(self.depth_maps[::3]):
                    depth_normalized = (depth * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                    cv2.imwrite(
                        str(depth_dir / f"depth_{i:03d}.jpg"),
                        depth_colored
                    )
                self._log(f"Saved sample depth maps to {depth_dir}")
            
            return True
            
        except Exception as e:
            self._log(f"Depth estimation failed: {e}")
            return False
    
    def generate_pointclouds(self) -> bool:
        """Generate point clouds from frames and depth maps with checkpoint support."""
        try:
            # Check if merged point cloud checkpoint exists
            if self._checkpoint_exists("point_cloud"):
                checkpoint = self._load_checkpoint("point_cloud")
                if checkpoint is not None:
                    self.merged_points = checkpoint["points"]
                    self.merged_colors = checkpoint["colors"]
                    self._log(f"Loaded merged point cloud from checkpoint: {len(self.merged_points)} points")
                    
                    # Clear frames and depth maps from memory
                    self.frames = None
                    self.depth_maps = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    
                    return True
            
            generator = PointCloudGenerator(self.config.point_cloud_config)
            
            self._log("Converting depth to point clouds...")
            pointclouds = []
            
            for i, (frame, depth) in enumerate(zip(self.frames, self.depth_maps)):
                points, colors = generator.depth_to_pointcloud(depth, frame)
                points, colors = generator.filter_pointcloud(points, colors)
                pointclouds.append((points, colors))
                
                # Cleanup memory every 20 frames
                if (i + 1) % 20 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    self._log(f"  Generated point cloud {i + 1}/{len(self.frames)}")
            
            self.pointclouds = pointclouds
            
            # Clear frames and depth maps from memory (no longer needed)
            self.frames = None
            self.depth_maps = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # Merge all point clouds
            self._log("Merging point clouds...")
            self.merged_points, self.merged_colors = generator.merge_pointclouds(pointclouds)
            
            self._log(f"Merged point cloud: {len(self.merged_points)} points")
            
            # Save checkpoint
            if self.config.save_intermediate:
                self._save_checkpoint("point_cloud", {
                    "points": self.merged_points,
                    "colors": self.merged_colors
                })
            
            # Clear individual point clouds
            self.pointclouds = None
            gc.collect()
            
            return True
            
        except Exception as e:
            self._log(f"Point cloud generation failed: {e}")
            return False
    
    def reconstruct_mesh(self) -> bool:
        """Reconstruct 3D mesh from point cloud."""
        try:
            reconstructor = MeshReconstructor(self.config.mesh_config)
            
            self.mesh = reconstructor.reconstruct(
                self.merged_points,
                self.merged_colors
            )
            
            if self.mesh is None:
                self._log("Mesh reconstruction failed, continuing with point cloud only")
                return True
            
            self._log("Mesh reconstruction successful")
            
            return True
            
        except Exception as e:
            self._log(f"Mesh reconstruction failed: {e}")
            return False
    
    def save_results(self) -> bool:
        """Save all results to disk."""
        try:
            # Save metadata (frames count may be None after cleanup)
            frames_count = self.config.frame_config.num_frames if self.frames is None else len(self.frames)
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'num_frames': self.config.frame_config.num_frames,
                    'frame_resolution': self.config.frame_config.target_size,
                    'depth_model': self.config.depth_config.model_type,
                    'mesh_method': self.config.mesh_config.method,
                },
                'results': {
                    'frames_extracted': frames_count,
                    'points_in_cloud': len(self.merged_points),
                }
            }
            
            if self.mesh is not None:
                if hasattr(self.mesh, 'vertices'):
                    metadata['results']['mesh_vertices'] = len(self.mesh.vertices)
                    metadata['results']['mesh_triangles'] = len(self.mesh.triangles)
                elif isinstance(self.mesh, dict):
                    metadata['results']['mesh_vertices'] = len(self.mesh['vertices'])
                    metadata['results']['mesh_triangles'] = len(self.mesh['triangles'])
            
            # Save metadata JSON
            metadata_path = self.config.output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self._log(f"Saved metadata to {metadata_path}")
            
            # Save point cloud
            if O3D_AVAILABLE:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.merged_points)
                pcd.colors = o3d.utility.Vector3dVector(self.merged_colors)
                
                pc_path = self.config.output_dir / "pointcloud.ply"
                o3d.io.write_point_cloud(str(pc_path), pcd)
                self._log(f"Saved point cloud to {pc_path}")
                
                # Cleanup point cloud object
                del pcd
                gc.collect()
            
            # Save mesh or point cloud (if using Gaussian Splatting)
            if self.mesh is not None and O3D_AVAILABLE:
                # Check if mesh is a point cloud (from Gaussian Splatting)
                if hasattr(self.mesh, 'has_normals'):
                    # It's a point cloud from Gaussian Splatting
                    pcd_path = self.config.output_dir / "gaussian_splatting.ply"
                    o3d.io.write_point_cloud(str(pcd_path), self.mesh)
                    self._log(f"Saved Gaussian Splatting point cloud to {pcd_path}")
                # Mesh types
                elif hasattr(self.mesh, 'vertices'):
                    mesh = self.mesh
                    # PLY format
                    mesh_ply = self.config.output_dir / "mesh.ply"
                    o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
                    self._log(f"Saved mesh (PLY) to {mesh_ply}")
                    
                    # OBJ format
                    mesh_obj = self.config.output_dir / "mesh.obj"
                    o3d.io.write_triangle_mesh(str(mesh_obj), mesh)
                    self._log(f"Saved mesh (OBJ) to {mesh_obj}")
                    
                    # Cleanup mesh object
                    del mesh
                    gc.collect()
                elif isinstance(self.mesh, dict):
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(self.mesh['vertices'])
                    mesh.triangles = o3d.utility.Vector3iVector(self.mesh['triangles'])
                    
                    # PLY format
                    mesh_ply = self.config.output_dir / "mesh.ply"
                    o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
                    self._log(f"Saved mesh (PLY) to {mesh_ply}")
                    
                    # OBJ format
                    mesh_obj = self.config.output_dir / "mesh.obj"
                    o3d.io.write_triangle_mesh(str(mesh_obj), mesh)
                    self._log(f"Saved mesh (OBJ) to {mesh_obj}")
                    
                    # Cleanup mesh object
                    del mesh
                    gc.collect()
            
            # Final GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            return True
            
        except Exception as e:
            self._log(f"Save results failed: {e}")
            return False
    
    def print_summary(self):
        """Print pipeline execution summary."""
        frames_count = self.config.frame_config.num_frames if self.frames is None else len(self.frames)
        
        # Get depth model details
        if self.config.depth_config.model_type.lower() == 'midas':
            depth_variant = self.config.depth_config.depth_model
        else:
            depth_variant = self.config.depth_config.marigold_model
        
        summary = f"""
{'=' * 70}
PIPELINE SUMMARY
{'=' * 70}

INPUT:
  Video: {self.config.video_path}
  Frames extracted: {frames_count}
  Resolution: {self.config.frame_config.target_size}

PROCESSING:
  Depth estimation model: {self.config.depth_config.model_type.upper()}
  Depth model variant: {depth_variant}
  Mesh reconstruction: {self.config.mesh_config.method.upper()}

RESULTS:
  Point cloud size: {len(self.merged_points):,} points
  Point cloud size (MB): {(len(self.merged_points) * 4 * 3) / (1024 * 1024):.2f}
  
"""
        
        if self.mesh is not None:
            if hasattr(self.mesh, 'vertices'):
                summary += f"""  Mesh vertices: {len(self.mesh.vertices):,}
  Mesh triangles: {len(self.mesh.triangles):,}
  Mesh size (MB): {(len(self.mesh.vertices) * 4 * 3 + len(self.mesh.triangles) * 4 * 3) / (1024 * 1024):.2f}
"""
            elif isinstance(self.mesh, dict):
                summary += f"""  Mesh vertices: {len(self.mesh['vertices']):,}
  Mesh triangles: {len(self.mesh['triangles']):,}
"""
        
        summary += f"""
OUTPUT DIRECTORY:
  {self.config.output_dir.resolve()}

FILES GENERATED:
  - pointcloud.ply (point cloud with colors)
  - mesh.ply (3D mesh)
  - mesh.obj (3D mesh for editing)
  - metadata.json (configuration and stats)
  - frames/ (all extracted frames for COLMAP)
  - depth_maps/ (sample depth visualizations)

NEXT STEPS:
  1. For SfM refinement: python colmap_pipeline.py
  2. View in CloudCompare: https://www.cloudcompare.org/
  3. Edit in Blender: https://www.blender.org/
  4. For dense reconstruction: COLMAP MVS pipeline

{'=' * 70}
"""
        
        print(summary)
        
        # Save summary to file
        summary_path = self.config.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)


def main():
    """Main entry point."""
    # Use DEFAULT_CONFIG but override paths for this script location
    config = PipelineConfig(
        video_path=DEFAULT_VIDEO_PATH,
        output_dir=OUTPUTS_ROOT / "legacy_depth",
        depth_config=DEFAULT_CONFIG.depth_config  # Use default depth model
    )
    
    # Run pipeline
    pipeline = ReconstructionPipeline(config)
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
