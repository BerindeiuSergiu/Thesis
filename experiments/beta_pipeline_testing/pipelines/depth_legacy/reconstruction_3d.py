"""3D reconstruction from depth maps."""
import numpy as np
import cv2
import time
from typing import Tuple, List, Dict, Optional

from .config import PointCloudConfig, MeshReconstructionConfig

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


class PointCloudGenerator:
    """Convert depth maps to point clouds."""
    
    def __init__(self, config: PointCloudConfig):
        """Initialize point cloud generator.
        
        Args:
            config: PointCloudConfig with camera parameters
        """
        self.config = config
    
    def depth_to_pointcloud(self, depth: np.ndarray, 
                           frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert depth map to point cloud with colors.
        
        Args:
            depth: Depth map (normalized 0-1 from MiDaS)
            frame: RGB frame for colors
            
        Returns:
            Tuple of (points, colors) as numpy arrays
        """
        height, width = depth.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Scale normalized depth to metric depth (assume 0-1 maps to 0-8 meters)
        depth_metric = depth * 8.0 / self.config.depth_scale
        
        # Unproject to 3D using camera intrinsics
        x = (u - self.config.cx) * depth_metric / self.config.fx
        y = (v - self.config.cy) * depth_metric / self.config.fy
        z = depth_metric
        
        # Stack into point cloud
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Get colors from frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        colors = frame_rgb.reshape(-1, 3) / 255.0
        
        # Filter valid points based on depth range
        valid_mask = ((depth.reshape(-1) >= self.config.min_depth) & 
                     (depth.reshape(-1) <= self.config.max_depth))
        
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        return points, colors
    
    def merge_pointclouds(self, 
                         pointclouds: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Merge multiple point clouds.
        
        Args:
            pointclouds: List of (points, colors) tuples
            
        Returns:
            Merged (points, colors)
        """
        all_points = []
        all_colors = []
        
        for points, colors in pointclouds:
            all_points.append(points)
            all_colors.append(colors)
        
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        return merged_points, merged_colors
    
    def filter_pointcloud(self, points: np.ndarray, 
                         colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter point cloud with voxel downsampling.
        
        Args:
            points: Point cloud points
            colors: Point colors
            
        Returns:
            Filtered (points, colors)
        """
        if len(points) == 0:
            return points, colors
        
        # Voxel grid downsampling
        voxel_indices = np.floor(points / self.config.voxel_size).astype(int)
        unique_voxels, indices = np.unique(voxel_indices, axis=0, return_index=True)
        
        return points[indices], colors[indices]


class MeshReconstructor:
    """Reconstruct 3D mesh from point cloud."""
    
    def __init__(self, config: MeshReconstructionConfig):
        """Initialize mesh reconstructor.
        
        Args:
            config: MeshReconstructionConfig
        """
        self.config = config
        
        if not O3D_AVAILABLE:
            print("Warning: Open3D not available. Mesh reconstruction will be limited.")
    
    def reconstruct(self, points: np.ndarray, 
                   colors: np.ndarray) -> Optional[Dict]:
        """Reconstruct 3D mesh from point cloud.
        
        Args:
            points: Point cloud points
            colors: Point colors
            
        Returns:
            Mesh object or dict with vertices/triangles
        """
        if not O3D_AVAILABLE:
            return self._convex_hull_fallback(points)
        
        print(f"Reconstructing mesh with {len(points)} points...")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove statistical outliers
        if self.config.outlier_removal:
            print("Removing statistical outliers...")
            pcd, inliers = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.outlier_nb_neighbors,
                std_ratio=self.config.outlier_std_ratio
            )
            print(f"Points after outlier removal: {len(pcd.points)}")
        
        # Downsample if too many points for BPA
        if len(pcd.points) > self.config.max_points_for_mesh:
            original_count = len(pcd.points)
            target_count = self.config.max_points_for_mesh
            print(f"Point cloud too large ({original_count} points), aggressive downsampling to {target_count}...")
            
            # Calculate voxel size needed to reach target point count
            # Rough estimate: points_per_voxel ≈ (voxel_size)^3 / voxel_volume
            import math
            reduction_ratio = original_count / target_count
            voxel_multiplier = math.pow(reduction_ratio, 1/3)  # Cube root for 3D
            target_voxel_size = self.config.downsample_voxel_size * voxel_multiplier
            
            print(f"  - Original voxel size: {self.config.downsample_voxel_size:.4f}m")
            print(f"  - Reduction ratio needed: {reduction_ratio:.1f}x")
            print(f"  - Target voxel size: {target_voxel_size:.4f}m")
            
            start_time = time.time()
            pcd = pcd.voxel_down_sample(voxel_size=target_voxel_size)
            elapsed = time.time() - start_time
            
            actual_count = len(pcd.points)
            actual_ratio = original_count / actual_count
            print(f"  - Downsampling completed in {elapsed:.1f} seconds")
            print(f"  - Actual reduction: {actual_count} points ({actual_ratio:.1f}x reduction)")
            
            # If still too large, do another pass with even larger voxels
            if actual_count > self.config.max_points_for_mesh:
                print(f"  - Still too large, doing second pass...")
                remaining_ratio = actual_count / target_count
                second_pass_voxel = target_voxel_size * math.pow(remaining_ratio, 1/3)
                print(f"  - Second pass voxel size: {second_pass_voxel:.4f}m")
                
                pcd = pcd.voxel_down_sample(voxel_size=second_pass_voxel)
                actual_count = len(pcd.points)
                print(f"  - After second pass: {actual_count} points")
        
        # Use method based on config
        if self.config.method == "tsdf_fusion":
            return self._tsdf_fusion_reconstruction(points, colors, pcd)
        elif self.config.method == "gaussian_splatting":
            return self._gaussian_splatting_reconstruction(points, colors, pcd)
        elif self.config.method == "poisson":
            return self._poisson_reconstruction(pcd)
        elif self.config.method == "bpa":
            return self._ball_pivot_reconstruction(pcd)
        else:
            raise ValueError(f"Unknown reconstruction method: {self.config.method}")
    
    def _tsdf_fusion_reconstruction(self, points: np.ndarray, 
                                    colors: np.ndarray, 
                                    pcd) -> Optional[object]:
        """TSDF Fusion fallback - not available in Open3D 0.19.0.
        
        Falls back to BPA which is superior for noisy depth maps anyway.
        """
        print("ScalableTSDFVolume not available in this Open3D version")
        print("Falling back to Ball Pivoting Algorithm (BPA)...")
        return self._ball_pivot_reconstruction(pcd)
    
    def _gaussian_splatting_reconstruction(self, points: np.ndarray, 
                                          colors: np.ndarray, 
                                          pcd) -> Optional[Dict]:
        """Gaussian Splatting reconstruction (saves as PLY for rendering).
        
        Creates a dense point cloud suitable for Gaussian Splatting rendering.
        The output PLY file can be used with Gaussian Splatting viewers.
        """
        try:
            print(f"Using Gaussian Splatting representation with {len(points)} points")
            
            # For Gaussian Splatting, we output the point cloud with normals
            # This can be rendered with GS viewers
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            
            # Return as point cloud (will be saved as-is)
            # In practice, this would be fed to a GS training pipeline
            print(f"Gaussian Splatting point cloud ready: {len(pcd.points)} points with normals")
            
            # Return point cloud object instead of mesh
            # The main.py will detect this and save as point cloud
            return pcd
            
        except Exception as e:
            print(f"Gaussian Splatting preparation failed: {e}")
            print("Falling back to Poisson reconstruction...")
            return self._poisson_reconstruction(pcd)
    
    def _poisson_reconstruction(self, pcd) -> Optional[object]:
        """Poisson surface reconstruction."""
        try:
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            
            # Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.config.poisson_depth
            )
            
            # Color mesh vertices from point cloud
            if len(pcd.colors) > 0 and self.config.color_vertices:
                try:
                    print("Coloring Poisson mesh vertices from point cloud...")
                    
                    # For large meshes, sample colors for speed
                    num_vertices = len(mesh.vertices)
                    if num_vertices > 100000:
                        sample_rate = max(1, num_vertices // 50000)
                        print(f"Large mesh, sampling every {sample_rate}th vertex for speed")
                        
                        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                        vertex_colors = np.full((num_vertices, 3), [0.5, 0.5, 0.5], dtype=np.float64)
                        
                        for i in range(0, num_vertices, sample_rate):
                            vertex = mesh.vertices[i]
                            k, idx, _ = pcd_tree.search_knn_vector_3d(vertex, knn=1)
                            if k >= 1:
                                vertex_colors[i] = pcd.colors[idx[0]]
                        
                        # Interpolate colors
                        for i in range(num_vertices):
                            if i % sample_rate != 0:
                                nearest_sample = (i // sample_rate) * sample_rate
                                if nearest_sample < num_vertices:
                                    vertex_colors[i] = vertex_colors[nearest_sample]
                        
                        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                    else:
                        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                        vertex_colors = []
                        for vertex in mesh.vertices:
                            k, idx, _ = pcd_tree.search_knn_vector_3d(vertex, knn=1)
                            if k >= 1:
                                vertex_colors.append(pcd.colors[idx[0]])
                            else:
                                vertex_colors.append([0.5, 0.5, 0.5])
                        
                        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
                    
                    print("Colored Poisson mesh vertices from point cloud")
                except Exception as color_e:
                    print(f"Warning: Could not color Poisson mesh: {color_e}")
            
            # Optionally remove low density vertices
            if self.config.remove_isolated_vertices:
                vertices_to_remove = densities < np.quantile(densities, 0.1)
                mesh.remove_vertices_by_mask(vertices_to_remove)
            
            if self.config.compute_normals:
                mesh.compute_vertex_normals()
            
            print(f"Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            
            return mesh
            
        except Exception as e:
            print(f"Poisson reconstruction failed: {e}")
            return None
    
    def _ball_pivot_reconstruction(self, pcd) -> Optional[object]:
        """Ball Pivoting Algorithm reconstruction - best for depth maps."""
        try:
            print(f"\n{'='*70}")
            print(f"BALL PIVOTING ALGORITHM RECONSTRUCTION")
            print(f"{'='*70}")
            print(f"Input point cloud: {len(pcd.points)} points")
            print(f"Using radii: {self.config.bpa_radii}")
            
            # CRITICAL: Estimate normals with FAST parameters for large clouds
            print(f"\n[1/3] ESTIMATING NORMALS (this is the slow part)...")
            num_points = len(pcd.points)
            
            # Adaptive normal estimation parameters
            if num_points > 5000000:
                print(f"  ⚠️  VERY LARGE point cloud ({num_points} points)")
                print(f"  Using FAST normal estimation:")
                search_radius = 0.2      # Larger radius = faster
                max_neighbors = 10       # Fewer neighbors = faster
                print(f"    - Search radius: {search_radius}m")
                print(f"    - Max neighbors: {max_neighbors}")
            elif num_points > 2000000:
                print(f"  📊 Large point cloud ({num_points} points)")
                print(f"  Using BALANCED normal estimation:")
                search_radius = 0.15
                max_neighbors = 15
                print(f"    - Search radius: {search_radius}m")
                print(f"    - Max neighbors: {max_neighbors}")
            else:
                print(f"  ✓ Medium point cloud ({num_points} points)")
                print(f"  Using QUALITY normal estimation:")
                search_radius = 0.1
                max_neighbors = 30
                print(f"    - Search radius: {search_radius}m")
                print(f"    - Max neighbors: {max_neighbors}")
            
            print(f"  Starting normal estimation...")
            start_time = time.time()
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=search_radius, max_nn=max_neighbors
                )
            )
            
            elapsed = time.time() - start_time
            speed = num_points / elapsed if elapsed > 0 else 0
            print(f"  ✓ Normal estimation completed!")
            print(f"    - Time: {elapsed:.1f} seconds")
            print(f"    - Speed: {speed/1e6:.1f}M points/second")
            
            # BPA reconstruction
            print(f"\n[2/3] RUNNING BALL PIVOTING ALGORITHM...")
            start_time = time.time()
            
            radii = o3d.utility.DoubleVector(self.config.bpa_radii)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, radii
            )
            
            elapsed = time.time() - start_time
            print(f"  ✓ BPA completed!")
            print(f"    - Time: {elapsed:.1f} seconds")
            print(f"    - Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            
            # Color mesh vertices
            print(f"\n[3/3] COLORING MESH VERTICES...")
            if len(pcd.colors) > 0 and self.config.color_vertices:
                try:
                    num_vertices = len(mesh.vertices)
                    print(f"  Coloring {num_vertices} vertices...")
                    
                    # ALWAYS sample for speed - don't color all vertices
                    max_vertices_to_color = 50000  # Only color 50k vertices, interpolate rest
                    if num_vertices > max_vertices_to_color:
                        sample_rate = max(1, num_vertices // max_vertices_to_color)
                        num_to_color = num_vertices // sample_rate
                        print(f"  ⚡ FAST coloring: sampling {num_to_color} vertices (every {sample_rate}th)")
                        
                        start_time = time.time()
                        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                        vertex_colors = np.full((num_vertices, 3), [0.5, 0.5, 0.5], dtype=np.float64)
                        
                        for i in range(0, num_vertices, sample_rate):
                            vertex = mesh.vertices[i]
                            k, idx, _ = pcd_tree.search_knn_vector_3d(vertex, knn=1)
                            if k >= 1:
                                vertex_colors[i] = pcd.colors[idx[0]]
                            
                            if (i // sample_rate + 1) % 5000 == 0:
                                elapsed_iter = time.time() - start_time
                                progress = (i + 1) / num_vertices * 100
                                print(f"    - {progress:.1f}% ({i + 1}/{num_vertices}) in {elapsed_iter:.1f}s")
                        
                        # Interpolate colors for unsampled vertices
                        print(f"  Interpolating colors for remaining vertices...")
                        for i in range(num_vertices):
                            if i % sample_rate != 0:
                                nearest_sample = (i // sample_rate) * sample_rate
                                if nearest_sample < num_vertices:
                                    vertex_colors[i] = vertex_colors[nearest_sample]
                        
                        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                        elapsed = time.time() - start_time
                        print(f"  ✓ Coloring completed in {elapsed:.1f} seconds")
                    else:
                        print(f"  Coloring all {num_vertices} vertices (small mesh)")
                        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                        vertex_colors = []
                        
                        for idx, vertex in enumerate(mesh.vertices):
                            k, idx_arr, _ = pcd_tree.search_knn_vector_3d(vertex, knn=1)
                            if k >= 1:
                                vertex_colors.append(pcd.colors[idx_arr[0]])
                            else:
                                vertex_colors.append([0.5, 0.5, 0.5])
                            
                            if (idx + 1) % 10000 == 0:
                                print(f"    - Colored {idx + 1}/{num_vertices}")
                        
                        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
                        print(f"  ✓ Coloring completed")
                        
                except Exception as color_e:
                    print(f"  ⚠️  Could not color mesh: {color_e}")
            
            if self.config.compute_normals:
                print(f"Computing mesh vertex normals...")
                mesh.compute_vertex_normals()
                print(f"✓ Mesh vertex normals computed")
            
            print(f"\n{'='*70}")
            print(f"RECONSTRUCTION COMPLETE")
            print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            print(f"{'='*70}\n")
            
            return mesh
            
        except Exception as e:
            print(f"\n❌ BPA reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _convex_hull_fallback(points: np.ndarray) -> Dict:
        """Create convex hull as fallback."""
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(points)
            return {
                'vertices': hull.points,
                'triangles': hull.simplices
            }
        except Exception as e:
            print(f"Convex hull failed: {e}")
            return None
