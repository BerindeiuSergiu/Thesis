"""
Fast3R-based 3D Room Reconstruction Pipeline.

This pipeline uses Fast3R for joint depth + pose estimation.
Fast3R is ~10x faster than DUSt3R because it processes all views
in a single forward pass instead of pairwise inference.

Pipeline:
1. Extract frames from video
2. Run Fast3R to get aligned point cloud + camera poses
3. Filter and clean point cloud
4. Reconstruct mesh using BPA

Usage:
    python main_fast3r.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import json
import torch
import gc
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple

from ..shared.config import FrameExtractionConfig
from ..shared.metrics import (
    MethodMetricsLogger,
    collect_system_info,
    compute_frame_metrics,
    compute_mesh_metrics,
    compute_pointcloud_geometry_metrics,
    compute_pointcloud_nearest_neighbor_metrics,
    get_peak_gpu_memory_gb,
    reset_peak_gpu_memory_stats,
)
from ..shared.paths import DEFAULT_VIDEO_PATH, OUTPUTS_ROOT, REPO_ROOT
from ..shared.video_processor import FrameExtractor

# Add fast3r to path
FAST3R_PATH = REPO_ROOT / "fast3r"
if FAST3R_PATH.exists() and str(FAST3R_PATH) not in sys.path:
    sys.path.insert(0, str(FAST3R_PATH))

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


@dataclass
class Fast3RPipelineConfig:
    """Configuration for Fast3R-based pipeline."""
    # Input
    video_path: Path = DEFAULT_VIDEO_PATH
    output_dir: Path = OUTPUTS_ROOT / "fast3r"
    
    # Frame extraction
    # Fast3R can handle MORE frames than DUSt3R (lower VRAM usage)
    # Video: 10428 frames at 60fps = 173 seconds
    num_frames: int = 80  # Can handle 80+ frames on 8GB VRAM!
    skip_frames: int = 130  # ~2.2s between frames
    target_size: Tuple[int, int] = (1024, 768)
    
    # Fast3R settings
    fast3r_image_size: int = 512
    min_conf_thr: float = 1.0  # Lower = more points
    niter_pnp: int = 100  # PnP iterations
    dtype: str = "float32"  # or "bfloat16" for speed
    
    # Point cloud processing
    voxel_size: float = 0.002  # 2mm voxels
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    
    # Mesh reconstruction
    mesh_method: str = "bpa"
    bpa_radii: list = None
    
    verbose: bool = True
    save_intermediate: bool = True
    save_metrics: bool = True
    save_raw_pointcloud: bool = False
    
    def __post_init__(self):
        if self.bpa_radii is None:
            self.bpa_radii = [0.004, 0.008, 0.016, 0.032]
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class Fast3RPipeline:
    """Complete 3D reconstruction pipeline using Fast3R."""
    
    def __init__(self, config: Fast3RPipelineConfig = None):
        self.config = config or Fast3RPipelineConfig()
        self.metrics_logger = MethodMetricsLogger("fast3r") if self.config.save_metrics else None
        
        self.frames = None
        self.points = None
        self.colors = None
        self.poses = None
        self.pcd = None
        self.mesh = None
        self.reconstruction_stats = {}
        
        self._log("=" * 70)
        self._log("Fast3R 3D RECONSTRUCTION PIPELINE")
        self._log("=" * 70)
        self._log("Fast3R: 3D Reconstruction in One Forward Pass (CVPR 2025)")
        self._log("=" * 70)
        if self.metrics_logger is not None:
            self._log(f"Metrics CSVs will be written to: {self.metrics_logger.data_root}")
    
    def _log(self, message: str):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.config.verbose:
            print(f"[{timestamp}] {message}")

    def _log_stage_timing(self, stage_name: str, elapsed_seconds: float, success: bool):
        """Persist stage timing and GPU memory usage."""
        if self.metrics_logger is None:
            return
        self.metrics_logger.append_row(
            "system",
            "stage_timings.csv",
            {
                "stage_name": stage_name,
                "elapsed_seconds": float(elapsed_seconds),
                "gpu_peak_memory_gb": get_peak_gpu_memory_gb(),
                "success": bool(success),
            },
        )

    def _run_stage(self, stage_name: str, fn) -> bool:
        """Run one stage and persist timing/memory stats."""
        reset_peak_gpu_memory_stats()
        started = time.perf_counter()
        success = fn()
        elapsed = time.perf_counter() - started
        self._log_stage_timing(stage_name, elapsed, success)
        return success

    def _record_system_info(self):
        """Record environment and config metadata for reproducibility."""
        if self.metrics_logger is None:
            return
        system_info = collect_system_info()
        system_info.update(
            {
                "video_path": self.config.video_path,
                "output_dir": self.config.output_dir,
                "num_frames": self.config.num_frames,
                "skip_frames": self.config.skip_frames,
                "target_width": self.config.target_size[0],
                "target_height": self.config.target_size[1],
                "fast3r_image_size": self.config.fast3r_image_size,
                "confidence_threshold": self.config.min_conf_thr,
                "niter_pnp": self.config.niter_pnp,
                "voxel_size": self.config.voxel_size,
                "mesh_method": self.config.mesh_method,
                "dtype": self.config.dtype,
                "save_raw_pointcloud": bool(self.config.save_raw_pointcloud),
            }
        )
        self.metrics_logger.append_row("run_overview", "system_info.csv", system_info)

    def _record_run_summary(self, status: str, total_runtime_seconds: float, error_message: str = ""):
        """Write one-row success/failure run summary."""
        if self.metrics_logger is None:
            return
        row = {
            "status": status,
            "total_runtime_seconds": float(total_runtime_seconds),
            "video_path": self.config.video_path,
            "output_dir": self.config.output_dir,
            "num_frames_requested": self.config.num_frames,
            "num_frames_extracted": len(self.frames) if self.frames is not None else 0,
            "raw_points": len(self.points) if self.points is not None else 0,
            "processed_points": len(self.pcd.points) if self.pcd is not None else 0,
            "mesh_vertices": len(self.mesh.vertices) if self.mesh is not None else 0,
            "mesh_triangles": len(self.mesh.triangles) if self.mesh is not None else 0,
            "num_poses": len(self.poses) if self.poses is not None else 0,
            "error_message": error_message,
        }
        row.update(self.reconstruction_stats)
        self.metrics_logger.append_row("run_overview", "run_summary.csv", row)
    
    def run(self) -> bool:
        """Run the complete pipeline."""
        total_started = time.perf_counter()
        self._record_system_info()
        try:
            # Stage 1: Extract frames
            self._log("\n" + "=" * 70)
            self._log("STAGE 1: FRAME EXTRACTION")
            self._log("=" * 70)
            if not self._run_stage("frame_extraction", self._extract_frames):
                self._record_run_summary("failed", time.perf_counter() - total_started, "frame_extraction_failed")
                return False
            
            # Stage 2: Fast3R reconstruction
            self._log("\n" + "=" * 70)
            self._log("STAGE 2: Fast3R RECONSTRUCTION")
            self._log("=" * 70)
            if not self._run_stage("fast3r_reconstruction", self._run_fast3r):
                self._record_run_summary("failed", time.perf_counter() - total_started, "fast3r_reconstruction_failed")
                return False
            
            # Stage 3: Point cloud processing
            self._log("\n" + "=" * 70)
            self._log("STAGE 3: POINT CLOUD PROCESSING")
            self._log("=" * 70)
            if not self._run_stage("pointcloud_processing", self._process_pointcloud):
                self._record_run_summary("failed", time.perf_counter() - total_started, "pointcloud_processing_failed")
                return False
            
            # Stage 4: Mesh reconstruction
            self._log("\n" + "=" * 70)
            self._log("STAGE 4: MESH RECONSTRUCTION")
            self._log("=" * 70)
            if not self._run_stage("mesh_reconstruction", self._reconstruct_mesh):
                self._record_run_summary("failed", time.perf_counter() - total_started, "mesh_reconstruction_failed")
                return False
            
            # Stage 5: Save results
            self._log("\n" + "=" * 70)
            self._log("STAGE 5: SAVING RESULTS")
            self._log("=" * 70)
            if not self._run_stage("save_results", self._save_results):
                self._record_run_summary("failed", time.perf_counter() - total_started, "save_results_failed")
                return False
            
            total_seconds = time.perf_counter() - total_started
            self._record_run_summary("success", total_seconds)
            self._print_summary()
            
            return True
            
        except Exception as e:
            self._log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            self._record_run_summary("failed", time.perf_counter() - total_started, str(e))
            return False
    
    def _extract_frames(self) -> bool:
        """Extract frames from video."""
        try:
            frame_config = FrameExtractionConfig(
                num_frames=self.config.num_frames,
                skip_frames=self.config.skip_frames,
                target_size=self.config.target_size,
                uniform_sampling=False
            )
            
            extractor = FrameExtractor(frame_config)
            self.frames, metadata = extractor.extract_frames(self.config.video_path)
            if self.metrics_logger is not None:
                frame_rows, pair_rows = compute_frame_metrics(self.frames, metadata)
                self.metrics_logger.append_rows("frames", "frame_metrics.csv", frame_rows)
                self.metrics_logger.append_rows("frames", "frame_pair_metrics.csv", pair_rows)
                self.metrics_logger.append_row(
                    "frames",
                    "frame_extraction_summary.csv",
                    {
                        "fps": float(metadata["fps"]),
                        "total_video_frames": int(metadata["total_frames"]),
                        "frames_extracted": int(metadata["extracted_frames"]),
                        "target_width": int(metadata["target_resolution"][0]),
                        "target_height": int(metadata["target_resolution"][1]),
                        "sampling_strategy": "uniform" if frame_config.uniform_sampling else "skip",
                        "skip_frames": int(frame_config.skip_frames),
                        "requested_frames": int(frame_config.num_frames),
                    },
                )
            
            self._log(f"Extracted {len(self.frames)} frames")
            self._log(f"  Video FPS: {metadata['fps']}")
            self._log(f"  Frame spacing: {self.config.skip_frames} frames (~{self.config.skip_frames/metadata['fps']:.2f}s)")
            
            # Save frames for inspection
            if self.config.save_intermediate:
                frames_dir = self.config.output_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                for i, frame in enumerate(self.frames):
                    cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), frame)
                self._log(f"Saved frames to {frames_dir}")
            
            return True
            
        except Exception as e:
            self._log(f"Frame extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_fast3r(self) -> bool:
        """Run Fast3R for joint depth + pose estimation."""
        try:
            from .reconstruction import Fast3RReconstructor, Fast3RConfig
            
            fast3r_config = Fast3RConfig(
                device="cuda" if torch.cuda.is_available() else "cpu",
                image_size=self.config.fast3r_image_size,
                min_conf_thr=self.config.min_conf_thr,
                niter_pnp=self.config.niter_pnp,
                dtype=self.config.dtype,
                verbose=self.config.verbose
            )
            
            reconstructor = Fast3RReconstructor(fast3r_config)
            
            self.points, self.colors, self.poses = reconstructor.reconstruct_from_frames(
                self.frames,
                output_dir=self.config.output_dir
            )
            
            self._log(f"Fast3R output:")
            self._log(f"  Points: {len(self.points):,}")
            self._log(f"  Camera poses: {len(self.poses)}")
            self.reconstruction_stats = {
                "num_points_raw": int(len(self.points)),
                "num_poses": int(len(self.poses)),
                "fast3r_image_size": int(self.config.fast3r_image_size),
                "min_conf_thr": float(self.config.min_conf_thr),
                "niter_pnp": int(self.config.niter_pnp),
                "dtype": self.config.dtype,
            }
            if self.metrics_logger is not None:
                self.metrics_logger.append_row(
                    "reconstruction",
                    "reconstruction_summary.csv",
                    dict(self.reconstruction_stats),
                )
            if self.config.save_raw_pointcloud:
                self._save_raw_pointcloud()
            
            # Save intermediate
            if self.config.save_intermediate:
                np.save(self.config.output_dir / "raw_points.npy", self.points)
                np.save(self.config.output_dir / "raw_colors.npy", self.colors)
                np.save(self.config.output_dir / "poses.npy", np.array(self.poses))
            
            return True
            
        except ImportError as e:
            self._log(f"Fast3R not installed! Error: {e}")
            self._log("Install with:")
            self._log("  git clone https://github.com/facebookresearch/fast3r.git")
            self._log("  cd fast3r && pip install -r requirements.txt && pip install -e .")
            return False
        except Exception as e:
            self._log(f"Fast3R reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_raw_pointcloud(self) -> None:
        """Persist raw model output point cloud before filtering."""
        if self.points is None or self.colors is None:
            self._log("Skipping raw point cloud save: no raw points available")
            return

        raw_npz = self.config.output_dir / "raw_pointcloud.npz"
        np.savez_compressed(raw_npz, points=self.points, colors=self.colors)
        self._log(f"Saved raw point cloud arrays: {raw_npz}")

        if not O3D_AVAILABLE:
            self._log("Open3D not available, raw point cloud exported as NPZ only")
            return

        raw_pcd = o3d.geometry.PointCloud()
        raw_pcd.points = o3d.utility.Vector3dVector(self.points)
        raw_pcd.colors = o3d.utility.Vector3dVector(self.colors)
        raw_ply = self.config.output_dir / "raw_pointcloud.ply"
        o3d.io.write_point_cloud(str(raw_ply), raw_pcd)
        self._log(f"Saved raw point cloud PLY: {raw_ply}")
    
    def _process_pointcloud(self) -> bool:
        """Process and clean the point cloud."""
        try:
            if not O3D_AVAILABLE:
                self._log("Open3D not available, skipping processing")
                return True
            
            self._log(f"Input points: {len(self.points):,}")
            if self.metrics_logger is not None:
                self.metrics_logger.append_row(
                    "pointcloud",
                    "pointcloud_stage_metrics.csv",
                    {
                        "stage_name": "raw_fast3r_output",
                        "num_points": int(len(self.points)),
                        "retention_ratio": 1.0,
                    },
                )
                raw_metrics = compute_pointcloud_geometry_metrics(self.points, self.colors)
                raw_metrics.update(compute_pointcloud_nearest_neighbor_metrics(self.points))
                raw_metrics["stage_name"] = "raw_fast3r_output"
                self.metrics_logger.append_row("pointcloud", "pointcloud_geometry_metrics.csv", raw_metrics)
            
            # Create Open3D point cloud
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
            self.pcd.colors = o3d.utility.Vector3dVector(self.colors)
            
            # Statistical outlier removal
            self._log("Removing statistical outliers...")
            self.pcd, inliers = self.pcd.remove_statistical_outlier(
                nb_neighbors=self.config.outlier_nb_neighbors,
                std_ratio=self.config.outlier_std_ratio
            )
            self._log(f"  After outlier removal: {len(self.pcd.points):,} points")
            if self.metrics_logger is not None:
                filtered_points = np.asarray(self.pcd.points)
                filtered_colors = np.asarray(self.pcd.colors)
                self.metrics_logger.append_row(
                    "pointcloud",
                    "pointcloud_stage_metrics.csv",
                    {
                        "stage_name": "after_outlier_removal",
                        "num_points": int(len(filtered_points)),
                        "retention_ratio": float(len(filtered_points) / max(len(self.points), 1)),
                    },
                )
                filtered_metrics = compute_pointcloud_geometry_metrics(filtered_points, filtered_colors)
                filtered_metrics.update(compute_pointcloud_nearest_neighbor_metrics(filtered_points))
                filtered_metrics["stage_name"] = "after_outlier_removal"
                self.metrics_logger.append_row("pointcloud", "pointcloud_geometry_metrics.csv", filtered_metrics)
            
            # Voxel downsampling
            self._log(f"Voxel downsampling (size={self.config.voxel_size}m)...")
            self.pcd = self.pcd.voxel_down_sample(voxel_size=self.config.voxel_size)
            self._log(f"  After downsampling: {len(self.pcd.points):,} points")
            if self.metrics_logger is not None:
                processed_points = np.asarray(self.pcd.points)
                processed_colors = np.asarray(self.pcd.colors)
                self.metrics_logger.append_row(
                    "pointcloud",
                    "pointcloud_stage_metrics.csv",
                    {
                        "stage_name": "after_voxel_downsample",
                        "num_points": int(len(processed_points)),
                        "retention_ratio": float(len(processed_points) / max(len(self.points), 1)),
                    },
                )
                processed_metrics = compute_pointcloud_geometry_metrics(processed_points, processed_colors)
                processed_metrics.update(compute_pointcloud_nearest_neighbor_metrics(processed_points))
                processed_metrics["stage_name"] = "after_voxel_downsample"
                self.metrics_logger.append_row("pointcloud", "pointcloud_geometry_metrics.csv", processed_metrics)
            
            # Save processed point cloud
            if self.config.save_intermediate:
                pc_path = self.config.output_dir / "pointcloud.ply"
                o3d.io.write_point_cloud(str(pc_path), self.pcd)
                self._log(f"Saved point cloud to {pc_path}")
            
            return True
            
        except Exception as e:
            self._log(f"Point cloud processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _reconstruct_mesh(self) -> bool:
        """Reconstruct mesh from point cloud."""
        try:
            if not O3D_AVAILABLE:
                self._log("Open3D not available, skipping mesh reconstruction")
                return True
            
            if self.pcd is None:
                self._log("No point cloud available")
                return False
            
            self._log(f"Mesh reconstruction using {self.config.mesh_method.upper()}...")
            
            # Estimate normals
            self._log("Estimating normals...")
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            if self.config.mesh_method == "bpa":
                # Ball Pivoting Algorithm
                self._log(f"Running BPA with radii: {self.config.bpa_radii}")
                radii = o3d.utility.DoubleVector(self.config.bpa_radii)
                self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    self.pcd, radii
                )
            elif self.config.mesh_method == "poisson":
                # Poisson reconstruction
                self._log("Running Poisson reconstruction...")
                self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    self.pcd, depth=9
                )
            else:
                self._log(f"Unknown mesh method: {self.config.mesh_method}")
                return False
            
            self._log(f"Mesh created:")
            self._log(f"  Vertices: {len(self.mesh.vertices):,}")
            self._log(f"  Triangles: {len(self.mesh.triangles):,}")
            if self.metrics_logger is not None:
                mesh_metrics = compute_mesh_metrics(self.mesh)
                mesh_metrics.update(
                    {
                        "mesh_method": self.config.mesh_method,
                        "bpa_radii": "|".join(str(v) for v in self.config.bpa_radii),
                    }
                )
                self.metrics_logger.append_row("mesh", "mesh_metrics.csv", mesh_metrics)
            
            # Color mesh vertices from point cloud
            self._log("Coloring mesh vertices...")
            self._color_mesh_from_pointcloud()
            
            # Compute vertex normals
            self.mesh.compute_vertex_normals()
            
            return True
            
        except Exception as e:
            self._log(f"Mesh reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _color_mesh_from_pointcloud(self):
        """Transfer colors from point cloud to mesh vertices."""
        if self.pcd is None or self.mesh is None:
            return
        
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        vertex_colors = []
        
        for vertex in self.mesh.vertices:
            k, idx, _ = pcd_tree.search_knn_vector_3d(vertex, knn=1)
            if k >= 1:
                vertex_colors.append(self.pcd.colors[idx[0]])
            else:
                vertex_colors.append([0.5, 0.5, 0.5])
        
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
    
    def _save_results(self) -> bool:
        """Save final results."""
        try:
            if O3D_AVAILABLE and self.mesh is not None:
                # Save mesh in multiple formats
                mesh_ply = self.config.output_dir / "mesh.ply"
                mesh_obj = self.config.output_dir / "mesh.obj"
                
                o3d.io.write_triangle_mesh(str(mesh_ply), self.mesh)
                o3d.io.write_triangle_mesh(str(mesh_obj), self.mesh)
                
                self._log(f"Saved mesh to:")
                self._log(f"  - {mesh_ply}")
                self._log(f"  - {mesh_obj}")
            
            # Save summary
            summary = {
                "num_frames": len(self.frames) if self.frames else 0,
                "num_points_raw": len(self.points) if self.points is not None else 0,
                "num_points_processed": len(self.pcd.points) if self.pcd else 0,
                "num_vertices": len(self.mesh.vertices) if self.mesh else 0,
                "num_triangles": len(self.mesh.triangles) if self.mesh else 0,
                "num_poses": len(self.poses) if self.poses else 0,
                "config": {
                    "num_frames": self.config.num_frames,
                    "skip_frames": self.config.skip_frames,
                    "fast3r_image_size": self.config.fast3r_image_size,
                    "min_conf_thr": self.config.min_conf_thr,
                    "voxel_size": self.config.voxel_size,
                    "mesh_method": self.config.mesh_method,
                }
            }
            
            with open(self.config.output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            return True
            
        except Exception as e:
            self._log(f"Failed to save results: {e}")
            return False
    
    def _print_summary(self):
        """Print final summary."""
        self._log("\n" + "=" * 70)
        self._log("RECONSTRUCTION COMPLETE!")
        self._log("=" * 70)
        self._log(f"\n📊 Results:")
        self._log(f"   Frames processed: {len(self.frames) if self.frames else 0}")
        self._log(f"   Raw points: {len(self.points):,}" if self.points is not None else "   Raw points: N/A")
        self._log(f"   Processed points: {len(self.pcd.points):,}" if self.pcd else "   Processed points: N/A")
        if self.mesh:
            self._log(f"   Mesh vertices: {len(self.mesh.vertices):,}")
            self._log(f"   Mesh triangles: {len(self.mesh.triangles):,}")
        self._log(f"   Camera poses: {len(self.poses)}" if self.poses else "   Camera poses: N/A")
        self._log(f"\n📁 Output saved to: {self.config.output_dir}")
        self._log("=" * 70)


def main():
    """Main entry point."""
    import argparse

    def parse_bool(value):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
    
    parser = argparse.ArgumentParser(description="Fast3R 3D Room Reconstruction")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO_PATH),
                        help="Path to input video")
    parser.add_argument("--output", type=str, default=str(OUTPUTS_ROOT / "fast3r"),
                        help="Output directory")
    parser.add_argument("--num-frames", type=int, default=80,
                        help="Number of frames to extract")
    parser.add_argument("--skip-frames", type=int, default=130,
                        help="Frames to skip between extractions")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16"],
                        help="Model dtype (bfloat16 is faster but may be less accurate)")
    parser.add_argument(
        "--save-pointcloud",
        "--save_pointcloud",
        type=parse_bool,
        default=False,
        help="Save raw (pre-filter) Fast3R output point cloud as NPZ and PLY.",
    )
    args = parser.parse_args()
    
    config = Fast3RPipelineConfig(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        num_frames=args.num_frames,
        skip_frames=args.skip_frames,
        dtype=args.dtype,
        save_raw_pointcloud=args.save_pointcloud,
    )
    
    pipeline = Fast3RPipeline(config)
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
