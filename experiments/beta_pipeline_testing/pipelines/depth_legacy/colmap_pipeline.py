"""COLMAP-based 3D reconstruction pipeline."""
import subprocess
import json
from pathlib import Path
import numpy as np
import shutil
import sys

from ..shared.paths import OUTPUTS_ROOT

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


class COLMAPReconstructor:
    """3D reconstruction using COLMAP Structure from Motion."""
    
    def __init__(self, project_dir: Path):
        """Initialize COLMAP project.
        
        Args:
            project_dir: Directory for COLMAP project
        """
        self.project_dir = Path(project_dir)
        self.image_dir = self.project_dir / "images"
        self.database_path = self.project_dir / "database.db"
        self.sparse_dir = self.project_dir / "sparse"
        self.dense_dir = self.project_dir / "dense"
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create required directories."""
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_frames(self, frames_dir: Path):
        """Copy extracted frames to COLMAP image directory.
        
        Args:
            frames_dir: Directory containing extracted frames
        """
        print(f"Copying frames from {frames_dir}...")
        
        frame_files = sorted(frames_dir.glob("*.jpg"))
        for i, frame_file in enumerate(frame_files):
            dest = self.image_dir / f"{i:06d}.jpg"
            shutil.copy2(frame_file, dest)
            
            if (i + 1) % 20 == 0:
                print(f"  Copied {i + 1}/{len(frame_files)} frames")
        
        print(f"Total frames: {len(frame_files)}")
    
    def feature_extraction(self):
        """Extract SIFT features."""
        print("\n" + "="*70)
        print("STAGE 1: Feature Extraction")
        print("="*70)
        
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("[OK] Feature extraction complete")
            return True
        except FileNotFoundError:
            print("[ERROR] COLMAP not found. Install: https://colmap.github.io/")
            return False
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return False
    
    def feature_matching(self):
        """Match features between images."""
        print("\n" + "="*70)
        print("STAGE 2: Feature Matching")
        print("="*70)
        
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(self.database_path),
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            print("[OK] Feature matching complete")
            return True
        except subprocess.TimeoutExpired:
            print("[WARN] Feature matching timed out - continuing anyway")
            return True
        except Exception as e:
            print(f"[ERROR] Feature matching failed: {e}")
            print("Trying sequential matcher as fallback...")
            
            # Try sequential matcher as fallback
            cmd_seq = [
                "colmap", "sequential_matcher",
                "--database_path", str(self.database_path),
            ]
            
            try:
                subprocess.run(cmd_seq, check=True, capture_output=True, text=True, timeout=600)
                print(" Sequential matching complete")
                return True
            except Exception as e2:
                print(f" Sequential matching also failed: {e2}")
                return False
    
    def structure_from_motion(self):
        """Run Structure from Motion reconstruction."""
        print("\n" + "="*70)
        print("STAGE 3: Structure from Motion")
        print("="*70)
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--output_path", str(self.sparse_dir),
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("[OK] SfM reconstruction complete")
            return True
        except Exception as e:
            print(f"[ERROR] SfM failed: {e}")
            return False
    
    def dense_reconstruction(self):
        """Dense reconstruction using MVS."""
        print("\n" + "="*70)
        print("STAGE 4: Dense Reconstruction (MVS)")
        print("="*70)
        
        # Image undistortion
        print("Undistorting images...")
        cmd_undistort = [
            "colmap", "image_undistorter",
            "--image_path", str(self.image_dir),
            "--input_path", str(self.sparse_dir / "0"),
            "--output_path", str(self.dense_dir),
            "--output_type", "COLMAP",
        ]
        
        try:
            subprocess.run(cmd_undistort, check=True, capture_output=True, text=True)
            print("✅ Image undistortion complete")
        except Exception as e:
            print(f"⚠️  Undistortion skipped: {e}")
        
        # Dense matching
        print("Running dense stereo matching...")
        cmd_stereo = [
            "colmap", "stereo_matcher",
            "--workspace_path", str(self.dense_dir),
            "--workspace_format", "COLMAP",
            "--StereoMatcher.geom_consistency", "true",
        ]
        
        try:
            subprocess.run(cmd_stereo, check=True, capture_output=True, text=True)
            print(" Stereo matching complete")
        except Exception as e:
            print(f"  Stereo matching skipped: {e}")
        
        # Fusion
        print("Fusing depth maps...")
        cmd_fusion = [
            "colmap", "point_cloud_merger",
            "--workspace_path", str(self.dense_dir),
            "--workspace_format", "COLMAP",
        ]
        
        try:
            subprocess.run(cmd_fusion, check=True, capture_output=True, text=True)
            print(" Point cloud fusion complete")
        except Exception as e:
            print(f"  Fusion skipped: {e}")
    
    def export_results(self, output_dir: Path):
        """Export results as PLY point cloud.
        
        Args:
            output_dir: Directory to save results
        """
        print("\n" + "="*70)
        print("STAGE 5: Exporting Results")
        print("="*70)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find sparse reconstruction
        sparse_0 = self.sparse_dir / "0"
        if not sparse_0.exists():
            print(" No sparse reconstruction found")
            return False
        
        if not O3D_AVAILABLE:
            print("  Open3D not available. Cannot export PLY.")
            return False
        
        print("Converting to PLY format...")
        
        try:
            import pycolmap
            
            # Find the reconstruction with the most 3D points
            reconstructions = sorted(self.sparse_dir.iterdir(), key=lambda x: x.name)
            best_recon = None
            best_count = 0
            best_path = None
            
            for recon_dir in reconstructions:
                if recon_dir.is_dir():
                    try:
                        recon = pycolmap.Reconstruction(str(recon_dir))
                        pt_count = len(recon.points3D)
                        print(f"  Reconstruction {recon_dir.name}: {pt_count} points")
                        if pt_count > best_count:
                            best_count = pt_count
                            best_recon = recon
                            best_path = recon_dir
                    except:
                        pass
            
            if best_recon is None or best_count == 0:
                print(" No valid reconstruction found")
                return False
            
            print(f"  Using reconstruction with {best_count} points")
            reconstruction = best_recon
            
            # Extract 3D points
            xyz_list = []
            rgb_list = []
            
            for point3d_id, point3d in reconstruction.points3D.items():
                xyz_list.append(point3d.xyz)
                rgb_list.append(point3d.color)
            
            if not xyz_list:
                print(" No 3D points found in reconstruction")
                return False
            
            xyz = np.array(xyz_list)
            rgb = np.array(rgb_list)
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
            
            # Save
            output_path = output_dir / "colmap_pointcloud.ply"
            o3d.io.write_point_cloud(str(output_path), pcd)
            
            print(f"[OK] Saved {len(xyz):,} points to {output_path}")
            
            # Statistics
            print(f"\nReconstruction Statistics:")
            print(f"  3D Points: {len(xyz):,}")
            print(f"  Cameras: {len(reconstruction.cameras)}")
            print(f"  Images: {len(reconstruction.images)}")
            
            # Camera info
            for img_id, image in list(reconstruction.images.items())[:3]:
                print(f"    Image {img_id}: camera_id={image.camera_id}")
            
            return True
            
        except ImportError:
            print("  pycolmap not available")
            print("Install: pip install pycolmap")
            return False
        except Exception as e:
            print(f" Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_full_pipeline(self, frames_dir: Path, output_dir: Path):
        """Run complete COLMAP pipeline.
        
        Args:
            frames_dir: Directory with extracted frames
            output_dir: Directory for results
            
        Returns:
            bool: Success status
        """
        print("\n" + "#"*70)
        print("# COLMAP 3D RECONSTRUCTION PIPELINE")
        print("#"*70)
        
        # Copy frames
        self.copy_frames(frames_dir)
        
        # Run pipeline stages
        if not self.feature_extraction():
            return False
        
        if not self.feature_matching():
            return False
        
        if not self.structure_from_motion():
            return False
        
        # Optional: dense reconstruction (slow)
        # self.dense_reconstruction()
        
        # Export results
        if not self.export_results(output_dir):
            return False
        
        print("\n" + "#"*70)
        print("# PIPELINE COMPLETE")
        print("#"*70)
        
        return True


def main():
    """Main entry point."""
    output_dir = OUTPUTS_ROOT / "legacy_depth"
    frames_dir = output_dir / "frames"
    colmap_dir = output_dir / "colmap"
    
    # Check frames
    if not frames_dir.exists():
        print(f" Frames not found: {frames_dir}")
        print("Run main.py first to extract frames")
        return 1
    
    # Run COLMAP
    reconstructor = COLMAPReconstructor(colmap_dir)
    success = reconstructor.run_full_pipeline(frames_dir, output_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

