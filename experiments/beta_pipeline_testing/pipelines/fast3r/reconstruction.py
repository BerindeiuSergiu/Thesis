"""
Fast3R-based 3D Reconstruction Pipeline.

Fast3R (CVPR 2025) performs 3D reconstruction of 1000+ images in one forward pass.
It jointly estimates:
- Dense depth maps
- Camera poses (intrinsics + extrinsics)
- Dense point cloud with proper alignment

This is significantly faster than DUSt3R for multi-view reconstruction.

Installation:
    git clone https://github.com/facebookresearch/fast3r.git
    cd fast3r
    pip install -r requirements.txt
    pip install -e .
    
Then set PYTHONPATH or add to sys.path before importing.
"""

import sys
from pathlib import Path

from ..shared.paths import REPO_ROOT

# Add fast3r to path if not already installed
FAST3R_PATH = REPO_ROOT / "fast3r"
if FAST3R_PATH.exists() and str(FAST3R_PATH) not in sys.path:
    sys.path.insert(0, str(FAST3R_PATH))

import numpy as np
import cv2
import torch
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import gc
import json
from concurrent.futures import ThreadPoolExecutor

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


def fast3r_precision(dtype_name: str) -> str:
    """Map local config dtype names to Fast3R inference precision strings."""

    normalized = str(dtype_name).strip().lower()
    if normalized in {"float32", "fp32", "32"}:
        return "32"
    if normalized in {"float16", "fp16", "16", "16-mixed"}:
        return "16-mixed"
    if normalized in {"bfloat16", "bf16", "bf16-mixed"}:
        return "bf16-mixed"
    raise ValueError(f"Unsupported Fast3R dtype: {dtype_name}")


def _estimate_focal(pts3d_i: torch.Tensor, conf_i: torch.Tensor, min_conf_thr_percentile: int = 10) -> float:
    """Estimate focal length from 3D points and confidence."""
    from fast3r.dust3r.post_process import estimate_focal_knowing_depth_and_confidence_mask

    bsz, h, w, _ = pts3d_i.shape
    assert bsz == 1
    pp = torch.tensor((w / 2, h / 2), device=pts3d_i.device).view(1, 2)

    conf_flat = conf_i.reshape(-1)
    percentile = min_conf_thr_percentile / 100.0
    conf_threshold = torch.quantile(conf_flat, percentile)
    conf_mask = (conf_i >= conf_threshold).view(bsz, h, w)

    focal = estimate_focal_knowing_depth_and_confidence_mask(
        pts3d_i,
        pp.unsqueeze(0),
        conf_mask,
        focal_mode="weiszfeld",
    ).ravel()
    return float(focal)


def _estimate_cam_pose_one_sample(sample_preds: list[dict], niter_pnp: int = 10) -> tuple[list[np.ndarray], list[float]]:
    """Estimate camera poses for a single sample using Fast3R's fast_pnp."""
    from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp

    poses_c2w: list[np.ndarray] = []
    estimated_focals: list[float] = []

    def process_view(view_idx: int) -> tuple[np.ndarray, float]:
        pts3d = sample_preds[view_idx]["pts3d_in_other_view"].cpu().numpy().squeeze()
        valid_mask = sample_preds[view_idx]["conf"].cpu().numpy().squeeze() > 1.0
        focal_length = float(sample_preds[view_idx]["focal_length"]) if "focal_length" in sample_preds[view_idx] else None

        focal_length, pose_c2w = fast_pnp(
            torch.tensor(pts3d),
            focal_length,
            torch.tensor(valid_mask, dtype=torch.bool),
            "cpu",
            pp=None,
            niter_PnP=niter_pnp,
        )
        if pose_c2w is None or focal_length is None:
            return np.eye(4, dtype=np.float32), 0.0
        return pose_c2w.cpu().numpy(), float(focal_length)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_view, range(len(sample_preds))))

    for pose_c2w_result, focal_length_result in results:
        poses_c2w.append(pose_c2w_result)
        estimated_focals.append(focal_length_result)

    return poses_c2w, estimated_focals


def estimate_camera_poses_simple(
    preds: list[dict],
    niter_pnp: int = 10,
    focal_length_estimation_method: str = "individual",
) -> tuple[list[list[np.ndarray]], list[list[float]]]:
    """Fast3R pose estimation without importing Lightning module."""
    batch_size = len(preds[0]["pts3d_in_other_view"])
    data_for_processing: list[list[dict]] = []

    for i in range(batch_size):
        sample_preds = [{key: value[i].cpu() for key, value in view.items()} for view in preds]
        data_for_processing.append(sample_preds)

    def estimate_focal_for_sample(sample_preds: list[dict]) -> list[dict]:
        if focal_length_estimation_method == "first_view_from_global_head":
            pts3d_i = sample_preds[0]["pts3d_in_other_view"].unsqueeze(0)
            conf_i = sample_preds[0]["conf"].unsqueeze(0)
        elif focal_length_estimation_method == "first_view_from_local_head":
            pts3d_i = sample_preds[0]["pts3d_local_aligned_to_global"].unsqueeze(0)
            conf_i = sample_preds[0]["conf_local"].unsqueeze(0)
        elif focal_length_estimation_method == "individual":
            return sample_preds
        else:
            raise ValueError(f"Unknown focal_length_estimation_method: {focal_length_estimation_method}")

        estimated_focal = _estimate_focal(pts3d_i, conf_i, min_conf_thr_percentile=10)
        for view_pred in sample_preds:
            view_pred["focal_length"] = estimated_focal
        return sample_preds

    with ThreadPoolExecutor() as executor:
        data_for_processing = list(executor.map(estimate_focal_for_sample, data_for_processing))

    poses_c2w_all: list[list[np.ndarray]] = []
    estimated_focals_all: list[list[float]] = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda sample: _estimate_cam_pose_one_sample(sample, niter_pnp=niter_pnp), data_for_processing))

    for poses_c2w_sample, estimated_focals_sample in results:
        poses_c2w_all.append(poses_c2w_sample)
        estimated_focals_all.append(estimated_focals_sample)

    return poses_c2w_all, estimated_focals_all


@dataclass
class Fast3RConfig:
    """Configuration for Fast3R reconstruction."""
    model_name: str = "jedyang97/Fast3R_ViT_Large_512"
    device: str = "cuda"
    image_size: int = 512  # Fast3R native resolution
    dtype: str = "float32"  # float32 or bfloat16
    
    # Camera pose estimation
    niter_pnp: int = 100  # PnP iterations for pose estimation
    focal_method: str = "first_view_from_global_head"
    
    # Point cloud filtering
    min_conf_thr: float = 1.0  # Confidence threshold (lower = more points)
    confidence_keep_ratio: float = 1.0  # 1.0 keeps all points above min_conf_thr; 0.4 keeps top 40% per view.
    view_conf_p50_min: float = 0.0  # Drop whole views below this median confidence.
    view_conf_p90_min: float = 0.0  # Drop whole views below this 90th percentile confidence.
    weak_texture_retention: bool = True  # Keep plausible low-texture wall/plane points.
    weak_texture_percentile: float = 35.0
    weak_texture_min_conf_thr: float = 0.55
    weak_texture_keep_ratio: float = 0.35
    
    # Output settings
    save_intermediate: bool = True
    save_depth_maps: bool = True
    verbose: bool = True


class Fast3RReconstructor:
    """
    Fast3R-based 3D reconstruction.
    
    This class handles:
    1. Loading the Fast3R model
    2. Running single-pass multi-view inference
    3. Estimating camera poses via PnP
    4. Extracting aligned point cloud with colors
    
    Fast3R is much faster than DUSt3R because it processes all views
    in a single forward pass instead of pairwise inference + global alignment.
    """
    
    def __init__(self, config: Fast3RConfig = None):
        self.config = config or Fast3RConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self._fast3r_error = ""
        self.last_metadata: Dict[str, object] = {}
        self.last_confidences: np.ndarray | None = None
        self.last_source_view_ids: np.ndarray | None = None
        self.last_source_pixel_xy: np.ndarray | None = None
        self.last_confidence_maps: np.ndarray | None = None
        self._load_model()
    
    def _log(self, message: str):
        """Log message if verbose."""
        if self.config.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [Fast3R] {message}")
    
    def _load_model(self):
        """Load Fast3R model."""
        self._log(f"Loading Fast3R model: {self.config.model_name}")
        
        try:
            from fast3r.models.fast3r import Fast3R
            
            # Load pretrained model from HuggingFace
            self.model = Fast3R.from_pretrained(self.config.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._log(f"Model loaded successfully on {self.device}")
            if torch.cuda.is_available():
                self._log(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            self._fast3r_available = True
            self._fast3r_error = ""
            
        except ImportError as e:
            self._log(f"ERROR: Fast3R not installed. Install with:")
            self._log(f"  git clone https://github.com/facebookresearch/fast3r.git")
            self._log(f"  cd fast3r && pip install -r requirements.txt && pip install -e .")
            self._log(f"Import error: {e}")
            self._fast3r_available = False
            self._fast3r_error = str(e)
        except Exception as e:
            self._log(f"ERROR loading model: {e}")
            self._fast3r_available = False
            self._fast3r_error = str(e)
    
    def reconstruct_from_frames(
        self, 
        frames: List[np.ndarray], 
        output_dir: Path = None
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Reconstruct 3D from a list of frames.
        
        Args:
            frames: List of BGR frames (numpy arrays)
            output_dir: Optional directory to save intermediate results
            
        Returns:
            Tuple of (points, colors, camera_poses)
            - points: (N, 3) array of 3D points
            - colors: (N, 3) array of RGB colors (0-1 range)
            - camera_poses: List of 4x4 camera-to-world matrices
        """
        if not self._fast3r_available:
            detail = f" Reason: {self._fast3r_error}" if self._fast3r_error else ""
            raise RuntimeError(f"Fast3R not available.{detail}")
        
        from fast3r.dust3r.utils.image import load_images
        from fast3r.dust3r.inference_multiview import inference
        
        # CUDA memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self._log(f"Processing {len(frames)} frames...")
        start_time = time.time()
        
        # Save frames temporarily for Fast3R loading
        temp_dir = Path(output_dir or ".") / ".fast3r_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for i, frame in enumerate(frames):
            path = temp_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
            image_paths.append(str(path))
        
        self._log(f"Saved {len(image_paths)} temporary images")
        
        # Load images for Fast3R
        self._log("Loading images into Fast3R format...")
        images = load_images(image_paths, size=self.config.image_size, verbose=self.config.verbose)
        self._log(f"Loaded {len(images)} images")
        
        # Run inference (single forward pass for all views!)
        self._log("Running Fast3R inference (single forward pass)...")
        inference_start = time.time()
        
        precision = fast3r_precision(self.config.dtype)
        
        output_dict, profiling_info = inference(
            images,
            self.model,
            self.device,
            dtype=precision,
            verbose=self.config.verbose,
            profiling=True,
        )
        
        inference_time = time.time() - inference_start
        self._log(f"Inference completed in {inference_time:.1f}s")
        if torch.cuda.is_available():
            self._log(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Estimate camera poses using PnP
        self._log("Estimating camera poses...")
        pose_start = time.time()
        
        # Move predictions to GPU
        preds_gpu = []
        for pred in output_dict['preds']:
            pred_gpu = {}
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    pred_gpu[k] = v.to(self.device)
                else:
                    pred_gpu[k] = v
            preds_gpu.append(pred_gpu)
        
        poses_c2w_batch, estimated_focals = estimate_camera_poses_simple(
            preds_gpu,
            niter_pnp=self.config.niter_pnp,
            focal_length_estimation_method=self.config.focal_method
        )
        
        camera_poses = poses_c2w_batch[0]  # First (and only) batch
        
        pose_time = time.time() - pose_start
        self._log(f"Camera pose estimation completed in {pose_time:.1f}s")
        self._log(f"Got {len(camera_poses)} camera poses")
        
        # Extract point cloud with colors
        self._log("Extracting point cloud...")
        
        preds = output_dict['preds']
        all_points = []
        all_colors = []
        all_confidences = []
        all_source_view_ids = []
        all_source_pixel_xy = []
        confidence_maps = []
        per_view_confidence_stats = []
        total_predicted_points = 0
        total_finite_points = 0
        total_after_min_conf = 0
        total_after_quantile = 0
        total_weak_texture_added = 0
        total_after_view_pruning = 0
        
        # Create depth visualization directory
        if self.config.save_depth_maps and output_dir:
            depth_dir = Path(output_dir) / "frames_depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
        else:
            depth_dir = None
        
        for view_idx, pred in enumerate(preds):
            # Get point cloud (shape: 1, H, W, 3)
            pts3d = pred['pts3d_in_other_view']
            if isinstance(pts3d, torch.Tensor):
                pts3d = pts3d.cpu().numpy()
            pts3d = pts3d[0]  # Remove batch dim -> (H, W, 3)
            
            # Get confidence if available
            if 'conf' in pred:
                conf = pred['conf']
                if isinstance(conf, torch.Tensor):
                    conf = conf.cpu().numpy()
                conf = conf[0] if conf.ndim > 2 else conf
            else:
                conf = np.ones(pts3d.shape[:2])
            conf = np.asarray(conf, dtype=np.float32)
            confidence_maps.append(conf)
            
            out_h, out_w = pts3d.shape[:2]
            
            # Save depth visualization
            if depth_dir is not None:
                depth = pts3d[:, :, 2]
                valid = np.isfinite(depth)
                if valid.any():
                    d_min, d_max = np.percentile(depth[valid], [2.0, 98.0])
                    if d_max > d_min:
                        depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
                    else:
                        depth_norm = np.zeros_like(depth)
                else:
                    depth_norm = np.zeros_like(depth)
                
                depth_colored = cv2.applyColorMap(
                    (depth_norm * 255).astype(np.uint8), 
                    cv2.COLORMAP_VIRIDIS
                )
                frame_resized = cv2.resize(frames[view_idx], (out_w, out_h))
                combined = np.hstack([frame_resized, depth_colored])
                
                cv2.imwrite(str(depth_dir / f"depth_{view_idx:04d}.jpg"), depth_colored)
                cv2.imwrite(str(depth_dir / f"combined_{view_idx:04d}.jpg"), combined)
            
            # Extract colors from frame
            frame_resized = cv2.resize(frames[view_idx], (out_w, out_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            colors_flat = frame_rgb.reshape(-1, 3) / 255.0
            
            # Flatten points and apply confidence mask
            pts_flat = pts3d.reshape(-1, 3)
            conf_flat = conf.reshape(-1)
            finite_mask = np.isfinite(pts_flat).all(axis=1)
            total_predicted_points += int(pts_flat.shape[0])
            total_finite_points += int(np.count_nonzero(finite_mask))

            conf_p10 = float(np.percentile(conf_flat, 10)) if conf_flat.size else 0.0
            conf_p50 = float(np.percentile(conf_flat, 50)) if conf_flat.size else 0.0
            conf_p90 = float(np.percentile(conf_flat, 90)) if conf_flat.size else 0.0
            conf_min = float(np.min(conf_flat)) if conf_flat.size else 0.0
            conf_max = float(np.max(conf_flat)) if conf_flat.size else 0.0

            view_reject_reasons = []
            if float(self.config.view_conf_p50_min) > 0 and conf_p50 < float(self.config.view_conf_p50_min):
                view_reject_reasons.append("view_conf_p50_min")
            if float(self.config.view_conf_p90_min) > 0 and conf_p90 < float(self.config.view_conf_p90_min):
                view_reject_reasons.append("view_conf_p90_min")
            view_rejected = bool(view_reject_reasons)

            min_conf_mask = finite_mask & (conf_flat > self.config.min_conf_thr)
            mask = min_conf_mask.copy()
            points_after_min_conf = int(np.count_nonzero(min_conf_mask))
            total_after_min_conf += points_after_min_conf

            keep_ratio = float(self.config.confidence_keep_ratio)
            confidence_quantile_threshold = None
            if 0.0 < keep_ratio < 1.0:
                confidence_quantile_threshold = float(np.quantile(conf_flat, 1.0 - keep_ratio))
                mask &= conf_flat >= confidence_quantile_threshold
            points_after_quantile = int(np.count_nonzero(mask))
            total_after_quantile += points_after_quantile

            weak_texture_added = 0
            if self.config.weak_texture_retention and not view_rejected:
                gray_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                texture = np.abs(cv2.Laplacian(gray_resized, cv2.CV_32F)).reshape(-1)
                texture_threshold = float(
                    np.percentile(texture[np.isfinite(texture)], float(self.config.weak_texture_percentile))
                ) if texture.size else 0.0
                weak_texture = texture <= texture_threshold
                weak_conf = finite_mask & weak_texture & (conf_flat >= float(self.config.weak_texture_min_conf_thr))
                weak_ratio = float(self.config.weak_texture_keep_ratio)
                if 0.0 < weak_ratio < 1.0 and np.any(weak_conf):
                    weak_threshold = float(np.quantile(conf_flat[weak_conf], 1.0 - weak_ratio))
                    weak_conf &= conf_flat >= weak_threshold
                before_weak = int(np.count_nonzero(mask))
                mask |= weak_conf
                weak_texture_added = int(np.count_nonzero(mask) - before_weak)
                total_weak_texture_added += weak_texture_added

            if view_rejected:
                mask = np.zeros_like(mask, dtype=bool)
            points_after_view_pruning = int(np.count_nonzero(mask))
            total_after_view_pruning += points_after_view_pruning

            all_points.append(pts_flat[mask])
            all_colors.append(colors_flat[mask])
            all_confidences.append(conf_flat[mask].astype(np.float32))
            all_source_view_ids.append(np.full(int(np.count_nonzero(mask)), view_idx, dtype=np.int32))
            yy, xx = np.indices((out_h, out_w), dtype=np.int32)
            pixel_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            all_source_pixel_xy.append(pixel_xy[mask])
            per_view_confidence_stats.append(
                {
                    "view_idx": int(view_idx),
                    "points_before_filter": int(conf_flat.shape[0]),
                    "points_after_filter": int(np.count_nonzero(mask)),
                    "min_conf_thr": float(self.config.min_conf_thr),
                    "confidence_keep_ratio": float(keep_ratio),
                    "view_conf_p50_min": float(self.config.view_conf_p50_min),
                    "view_conf_p90_min": float(self.config.view_conf_p90_min),
                    "view_rejected": bool(view_rejected),
                    "view_reject_reasons": "|".join(view_reject_reasons),
                    "confidence_quantile_threshold": confidence_quantile_threshold,
                    "finite_points": int(np.count_nonzero(finite_mask)),
                    "points_after_min_conf": points_after_min_conf,
                    "points_after_quantile": points_after_quantile,
                    "weak_texture_retention": bool(self.config.weak_texture_retention),
                    "weak_texture_added_points": weak_texture_added,
                    "points_after_view_pruning": points_after_view_pruning,
                    "confidence_min": conf_min,
                    "confidence_p10": conf_p10,
                    "confidence_p50": conf_p50,
                    "confidence_p90": conf_p90,
                    "confidence_max": conf_max,
                }
            )
            
            if (view_idx + 1) % 10 == 0:
                self._log(f"  Processed {view_idx + 1}/{len(frames)} views")
        
        # Merge all points
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        confidences = np.concatenate(all_confidences).astype(np.float32)
        source_view_ids = np.concatenate(all_source_view_ids).astype(np.int32)
        source_pixel_xy = np.vstack(all_source_pixel_xy).astype(np.int32)
        confidence_maps_arr = np.stack(confidence_maps).astype(np.float32)
        self.last_confidences = confidences
        self.last_source_view_ids = source_view_ids
        self.last_source_pixel_xy = source_pixel_xy
        self.last_confidence_maps = confidence_maps_arr
        
        if depth_dir is not None:
            self._log(f"Saved {len(preds)} depth maps to {depth_dir}")
        
        total_time = time.time() - start_time
        self._log(f"Reconstruction complete!")
        self._log(f"  Total time: {total_time:.1f}s")
        self._log(f"  Points: {len(points):,}")
        self._log(f"  Camera poses: {len(camera_poses)}")
        self.last_metadata = {
            "model_name": self.config.model_name,
            "model_family": "fast3r",
            "poses_convention": "camera_to_world",
            "camera_poses": np.asarray(camera_poses),
            "estimated_focals": np.asarray(estimated_focals[0] if estimated_focals else []),
            "image_size": int(self.config.image_size),
            "confidence_threshold": float(self.config.min_conf_thr),
            "confidence_keep_ratio": float(self.config.confidence_keep_ratio),
            "view_conf_p50_min": float(self.config.view_conf_p50_min),
            "view_conf_p90_min": float(self.config.view_conf_p90_min),
            "weak_texture_retention": bool(self.config.weak_texture_retention),
            "weak_texture_percentile": float(self.config.weak_texture_percentile),
            "weak_texture_min_conf_thr": float(self.config.weak_texture_min_conf_thr),
            "weak_texture_keep_ratio": float(self.config.weak_texture_keep_ratio),
            "focal_method": self.config.focal_method,
            "precision": precision,
            "point_retention": {
                "predicted_dense_points": int(total_predicted_points),
                "finite_points": int(total_finite_points),
                "points_after_min_conf": int(total_after_min_conf),
                "points_after_quantile": int(total_after_quantile),
                "weak_texture_added_points": int(total_weak_texture_added),
                "points_after_view_pruning": int(total_after_view_pruning),
                "final_extracted_points": int(points.shape[0]),
                "retention_after_min_conf_ratio": float(total_after_min_conf / total_predicted_points) if total_predicted_points else 0.0,
                "retention_after_quantile_ratio": float(total_after_quantile / total_predicted_points) if total_predicted_points else 0.0,
                "final_retention_ratio": float(points.shape[0] / total_predicted_points) if total_predicted_points else 0.0,
                "points_removed_by_min_conf": int(total_finite_points - total_after_min_conf),
                "points_removed_by_quantile": int(total_after_min_conf - total_after_quantile),
                "points_removed_by_view_pruning": int(total_after_quantile + total_weak_texture_added - total_after_view_pruning),
                "views_total": int(len(preds)),
                "views_rejected_by_confidence": int(
                    sum(1 for row in per_view_confidence_stats if row.get("view_rejected"))
                ),
            },
            "raw_confidence_stats": {
                "points": int(confidences.shape[0]),
                "min": float(np.min(confidences)) if confidences.size else 0.0,
                "p10": float(np.percentile(confidences, 10)) if confidences.size else 0.0,
                "p50": float(np.percentile(confidences, 50)) if confidences.size else 0.0,
                "p90": float(np.percentile(confidences, 90)) if confidences.size else 0.0,
                "max": float(np.max(confidences)) if confidences.size else 0.0,
            },
            "per_view_confidence_stats": per_view_confidence_stats,
            "absolute_scale_available": False,
            "absolute_scale_note": (
                "Fast3R provides camera pose/focal estimates in its reconstruction frame, "
                "but this run did not expose an explicit meters-per-unit scale."
            ),
        }
        if output_dir is not None:
            output_root = Path(output_dir)
            np.save(output_root / "raw_confidences.npy", confidences)
            np.save(output_root / "source_view_ids.npy", source_view_ids)
            np.save(output_root / "source_pixel_xy.npy", source_pixel_xy)
            np.save(output_root / "confidence_maps.npy", confidence_maps_arr)
            stats_path = output_root / "confidence_export_report.json"
            stats_path.write_text(
                json.dumps(
                    {
                        "confidence_threshold": float(self.config.min_conf_thr),
                        "confidence_keep_ratio": float(self.config.confidence_keep_ratio),
                        "view_conf_p50_min": float(self.config.view_conf_p50_min),
                        "view_conf_p90_min": float(self.config.view_conf_p90_min),
                        "weak_texture_retention": bool(self.config.weak_texture_retention),
                        "weak_texture_percentile": float(self.config.weak_texture_percentile),
                        "weak_texture_min_conf_thr": float(self.config.weak_texture_min_conf_thr),
                        "weak_texture_keep_ratio": float(self.config.weak_texture_keep_ratio),
                        "views_rejected_by_confidence": int(
                            sum(1 for row in per_view_confidence_stats if row.get("view_rejected"))
                        ),
                        "point_retention": self.last_metadata["point_retention"],
                        "raw_confidence_stats": self.last_metadata["raw_confidence_stats"],
                        "per_view_confidence_stats": per_view_confidence_stats,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.last_metadata.update(
                {
                    "raw_confidences_path": str(output_root / "raw_confidences.npy"),
                    "source_view_ids_path": str(output_root / "source_view_ids.npy"),
                    "source_pixel_xy_path": str(output_root / "source_pixel_xy.npy"),
                    "confidence_maps_path": str(output_root / "confidence_maps.npy"),
                    "confidence_export_report": str(stats_path),
                }
            )
        
        # Cleanup temp files
        for path in image_paths:
            try:
                Path(path).unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return points, colors, list(camera_poses)
    
    def reconstruct_from_directory(
        self, 
        image_dir: Path, 
        output_dir: Path = None
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Reconstruct 3D from images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Optional output directory
            
        Returns:
            Tuple of (points, colors, camera_poses)
        """
        image_dir = Path(image_dir)
        
        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(sorted(image_dir.glob(ext)))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        self._log(f"Found {len(image_paths)} images in {image_dir}")
        
        # Load frames
        frames = []
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)
        
        return self.reconstruct_from_frames(frames, output_dir)
    
    def save_results(
        self, 
        points: np.ndarray, 
        colors: np.ndarray, 
        poses: List[np.ndarray], 
        output_dir: Path
    ):
        """Save reconstruction results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save point cloud
        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            pc_path = output_dir / "fast3r_pointcloud.ply"
            o3d.io.write_point_cloud(str(pc_path), pcd)
            self._log(f"Saved point cloud to {pc_path}")
        
        # Save as numpy arrays
        np.save(output_dir / "points.npy", points)
        np.save(output_dir / "colors.npy", colors)
        np.save(output_dir / "poses.npy", np.array(poses))
        
        # Save poses as text (for compatibility)
        poses_txt = output_dir / "camera_poses.txt"
        with open(poses_txt, 'w') as f:
            for i, pose in enumerate(poses):
                f.write(f"# Camera {i}\n")
                for row in pose:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
                f.write("\n")
        
        self._log(f"Saved camera poses to {poses_txt}")


def install_fast3r():
    """Helper function to install Fast3R."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    Fast3R INSTALLATION GUIDE                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Clone and install:                                                  ║
║  ──────────────────                                                  ║
║  git clone https://github.com/facebookresearch/fast3r.git           ║
║  cd fast3r                                                           ║
║  pip install -r requirements.txt                                     ║
║  pip install -e .                                                    ║
║                                                                      ║
║  Note: Do NOT install cuROPE - it breaks Fast3R predictions!         ║
║                                                                      ║
║  Requirements:                                                       ║
║  - PyTorch 2.0+ with CUDA                                            ║
║  - ~4GB VRAM for inference (much less than DUSt3R!)                  ║
║  - FlashAttention recommended (but optional)                         ║
║                                                                      ║
║  Model weights are downloaded automatically from HuggingFace:        ║
║  https://huggingface.co/jedyang97/Fast3R_ViT_Large_512               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# Simple test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_fast3r()
    else:
        print("Fast3R Reconstruction Module")
        print("=" * 50)
        print()
        print("Fast3R is ~10x faster than DUSt3R for multi-view reconstruction!")
        print("It processes all views in a single forward pass.")
        print()
        print("Usage:")
        print("  python fast3r_reconstruction.py --install    # Show installation guide")
        print()
        print("Or use in your pipeline:")
        print("  from pipelines.fast3r.reconstruction import Fast3RReconstructor, Fast3RConfig")
        print("  config = Fast3RConfig()")
        print("  reconstructor = Fast3RReconstructor(config)")
        print("  points, colors, poses = reconstructor.reconstruct_from_directory('frames/')")
