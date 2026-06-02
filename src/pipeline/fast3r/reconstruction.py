# NOTE: Do NOT import directly from experiments/.
# All pipeline code must be copied into src/pipeline/fast3r/ before modification.
# Best methods from experiments/ have been audited and integrated — see src/EXPERIMENTS_AUDIT.md

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch


ProgressCallback = Callable[[int, str], None] | None
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_FAST3R_REPO = REPO_ROOT / "fast3r"
if LOCAL_FAST3R_REPO.exists() and str(LOCAL_FAST3R_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_FAST3R_REPO))


def _emit(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback is not None:
        callback(percent, message)


def fast3r_precision(dtype_name: str) -> str:
    dtype_name = str(dtype_name).lower()
    if dtype_name == "float32":
        return "32"
    if dtype_name == "float16":
        return "16-mixed"
    if dtype_name in {"bfloat16", "bf16"}:
        return "bf16-mixed"
    raise ValueError(f"Unsupported Fast3R dtype: {dtype_name}")


def _estimate_focal(pts3d_i: torch.Tensor, conf_i: torch.Tensor, min_conf_thr_percentile: int = 10) -> float:
    from fast3r.dust3r.post_process import estimate_focal_knowing_depth_and_confidence_mask

    bsz, h, w, _ = pts3d_i.shape
    pp = torch.tensor((w / 2, h / 2), device=pts3d_i.device).view(1, 2)
    conf_threshold = torch.quantile(conf_i.reshape(-1), min_conf_thr_percentile / 100.0)
    conf_mask = (conf_i >= conf_threshold).view(bsz, h, w)
    focal = estimate_focal_knowing_depth_and_confidence_mask(
        pts3d_i,
        pp.unsqueeze(0),
        conf_mask,
        focal_mode="weiszfeld",
    ).ravel()
    return float(focal)


def _estimate_cam_pose_one_sample(sample_preds: list[dict], niter_pnp: int = 10) -> tuple[list[np.ndarray], list[float]]:
    from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp

    poses_c2w: list[np.ndarray] = []
    estimated_focals: list[float] = []
    for view_idx in range(len(sample_preds)):
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
            poses_c2w.append(np.eye(4, dtype=np.float32))
            estimated_focals.append(0.0)
        else:
            poses_c2w.append(pose_c2w.cpu().numpy())
            estimated_focals.append(float(focal_length))
    return poses_c2w, estimated_focals


def estimate_camera_poses_simple(
    preds: list[dict],
    niter_pnp: int = 10,
    focal_length_estimation_method: str = "individual",
) -> tuple[list[list[np.ndarray]], list[list[float]]]:
    batch_size = len(preds[0]["pts3d_in_other_view"])
    data_for_processing: list[list[dict]] = []

    for batch_index in range(batch_size):
        sample_preds = [{key: value[batch_index].cpu() for key, value in view.items()} for view in preds]
        if focal_length_estimation_method == "first_view_from_global_head":
            pts3d_i = sample_preds[0]["pts3d_in_other_view"].unsqueeze(0)
            conf_i = sample_preds[0]["conf"].unsqueeze(0)
            estimated_focal = _estimate_focal(pts3d_i, conf_i, min_conf_thr_percentile=10)
            for view_pred in sample_preds:
                view_pred["focal_length"] = estimated_focal
        data_for_processing.append(sample_preds)

    poses_c2w_all: list[list[np.ndarray]] = []
    estimated_focals_all: list[list[float]] = []
    for sample_preds in data_for_processing:
        poses_c2w_sample, focals_sample = _estimate_cam_pose_one_sample(sample_preds, niter_pnp=niter_pnp)
        poses_c2w_all.append(poses_c2w_sample)
        estimated_focals_all.append(focals_sample)
    return poses_c2w_all, estimated_focals_all


@dataclass
class Fast3RReconstructionConfig:
    model_name: str = "jedyang97/Fast3R_ViT_Large_512"
    device: str = "cuda"
    image_size: int = 512
    dtype: str = "float32"
    niter_pnp: int = 100
    focal_method: str = "first_view_from_global_head"
    min_conf_thr: float = 1.0
    confidence_keep_ratio: float = 1.0
    view_conf_p50_min: float = 0.0
    view_conf_p90_min: float = 0.0
    weak_texture_retention: bool = True
    weak_texture_percentile: float = 35.0
    weak_texture_min_conf_thr: float = 0.55
    weak_texture_keep_ratio: float = 0.35
    save_depth_maps: bool = True


class Fast3RReconstructor:
    def __init__(self, config: Fast3RReconstructionConfig):
        self.config = config
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self._fast3r_error = ""
        self.last_metadata: dict[str, object] = {}
        self.last_confidences: np.ndarray | None = None
        self.last_source_view_ids: np.ndarray | None = None
        self.last_source_pixel_xy: np.ndarray | None = None
        self.last_confidence_maps: np.ndarray | None = None

    def load(self) -> None:
        from fast3r.models.fast3r import Fast3R

        self.model = Fast3R.from_pretrained(self.config.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._fast3r_error = ""

    def reconstruct_from_frames(
        self,
        frames: list[np.ndarray],
        output_dir: Path | None = None,
        progress_callback: ProgressCallback = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        if self.model is None:
            self.load()

        from fast3r.dust3r.inference_multiview import inference
        from fast3r.dust3r.utils.image import load_images

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        temp_dir = Path(output_dir or ".") / ".fast3r_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        image_paths: list[str] = []

        _emit(progress_callback, 55, "Preparing frames for Fast3R")
        for index, frame in enumerate(frames):
            image_path = temp_dir / f"frame_{index:04d}.jpg"
            cv2.imwrite(str(image_path), frame)
            image_paths.append(str(image_path))

        images = load_images(image_paths, size=self.config.image_size, verbose=False)
        precision = fast3r_precision(self.config.dtype)

        _emit(progress_callback, 65, "Running Fast3R inference")
        output_dict, _ = inference(
            images,
            self.model,
            self.device,
            dtype=precision,
            verbose=False,
            profiling=True,
        )

        _emit(progress_callback, 75, "Estimating camera poses")
        preds_gpu = []
        for pred in output_dict["preds"]:
            pred_gpu = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in pred.items()}
            preds_gpu.append(pred_gpu)

        poses_c2w_batch, _ = estimate_camera_poses_simple(
            preds_gpu,
            niter_pnp=self.config.niter_pnp,
            focal_length_estimation_method=self.config.focal_method,
        )
        camera_poses = poses_c2w_batch[0]

        _emit(progress_callback, 82, "Collecting point maps")
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        all_confidences: list[np.ndarray] = []
        all_source_view_ids: list[np.ndarray] = []
        all_source_pixel_xy: list[np.ndarray] = []
        confidence_maps: list[np.ndarray] = []
        per_view_confidence_stats: list[dict[str, object]] = []
        total_predicted_points = 0
        total_finite_points = 0
        total_after_min_conf = 0
        total_after_quantile = 0
        total_weak_texture_added = 0
        total_after_view_pruning = 0

        depth_dir = None
        if self.config.save_depth_maps and output_dir is not None:
            depth_dir = Path(output_dir) / "frames_depth"
            depth_dir.mkdir(parents=True, exist_ok=True)

        for view_idx, pred in enumerate(output_dict["preds"]):
            pts3d = pred["pts3d_in_other_view"]
            if isinstance(pts3d, torch.Tensor):
                pts3d = pts3d.cpu().numpy()
            pts3d = pts3d[0]

            conf = pred["conf"]
            if isinstance(conf, torch.Tensor):
                conf = conf.cpu().numpy()
            conf = conf[0] if getattr(conf, "ndim", 0) > 2 else conf
            conf = np.asarray(conf, dtype=np.float32)
            confidence_maps.append(conf)

            out_h, out_w = pts3d.shape[:2]
            frame_resized = cv2.resize(frames[view_idx], (out_w, out_h))
            if depth_dir is not None:
                depth = pts3d[:, :, 2]
                valid_depth = np.isfinite(depth)
                if valid_depth.any():
                    d_min, d_max = np.percentile(depth[valid_depth], [2.0, 98.0])
                    if d_max > d_min:
                        depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
                    else:
                        depth_norm = np.zeros_like(depth)
                else:
                    depth_norm = np.zeros_like(depth)
                depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(str(depth_dir / f"depth_{view_idx:04d}.jpg"), depth_colored)
                cv2.imwrite(str(depth_dir / f"combined_{view_idx:04d}.jpg"), np.hstack([frame_resized, depth_colored]))

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            colors_flat = frame_rgb.reshape(-1, 3) / 255.0
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

            view_reject_reasons: list[str] = []
            if self.config.view_conf_p50_min > 0 and conf_p50 < self.config.view_conf_p50_min:
                view_reject_reasons.append("view_conf_p50_min")
            if self.config.view_conf_p90_min > 0 and conf_p90 < self.config.view_conf_p90_min:
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
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F)).reshape(-1)
                finite_texture = texture[np.isfinite(texture)]
                texture_threshold = float(np.percentile(finite_texture, self.config.weak_texture_percentile)) if finite_texture.size else 0.0
                weak_texture = texture <= texture_threshold
                weak_conf = finite_mask & weak_texture & (conf_flat >= self.config.weak_texture_min_conf_thr)
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
            all_source_view_ids.append(np.full(points_after_view_pruning, view_idx, dtype=np.int32))
            yy, xx = np.indices((out_h, out_w), dtype=np.int32)
            pixel_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            all_source_pixel_xy.append(pixel_xy[mask])
            per_view_confidence_stats.append(
                {
                    "view_idx": int(view_idx),
                    "points_before_filter": int(conf_flat.shape[0]),
                    "points_after_filter": points_after_view_pruning,
                    "finite_points": int(np.count_nonzero(finite_mask)),
                    "points_after_min_conf": points_after_min_conf,
                    "points_after_quantile": points_after_quantile,
                    "weak_texture_added_points": weak_texture_added,
                    "points_after_view_pruning": points_after_view_pruning,
                    "min_conf_thr": float(self.config.min_conf_thr),
                    "confidence_keep_ratio": float(keep_ratio),
                    "confidence_quantile_threshold": confidence_quantile_threshold,
                    "view_rejected": view_rejected,
                    "view_reject_reasons": "|".join(view_reject_reasons),
                    "confidence_min": conf_min,
                    "confidence_p10": conf_p10,
                    "confidence_p50": conf_p50,
                    "confidence_p90": conf_p90,
                    "confidence_max": conf_max,
                }
            )

        points = np.vstack(all_points).astype(np.float32)
        colors = np.vstack(all_colors).astype(np.float32)
        confidences = np.concatenate(all_confidences).astype(np.float32) if all_confidences else np.empty(0, dtype=np.float32)
        source_view_ids = np.concatenate(all_source_view_ids).astype(np.int32) if all_source_view_ids else np.empty(0, dtype=np.int32)
        source_pixel_xy = np.vstack(all_source_pixel_xy).astype(np.int32) if all_source_pixel_xy else np.empty((0, 2), dtype=np.int32)
        confidence_maps_arr = np.stack(confidence_maps).astype(np.float32) if confidence_maps else np.empty((0,), dtype=np.float32)

        self.last_confidences = confidences
        self.last_source_view_ids = source_view_ids
        self.last_source_pixel_xy = source_pixel_xy
        self.last_confidence_maps = confidence_maps_arr
        self.last_metadata = {
            "model_name": self.config.model_name,
            "model_family": "fast3r",
            "poses_convention": "camera_to_world",
            "image_size": int(self.config.image_size),
            "precision": precision,
            "confidence_threshold": float(self.config.min_conf_thr),
            "confidence_keep_ratio": float(self.config.confidence_keep_ratio),
            "view_conf_p50_min": float(self.config.view_conf_p50_min),
            "view_conf_p90_min": float(self.config.view_conf_p90_min),
            "weak_texture_retention": bool(self.config.weak_texture_retention),
            "weak_texture_percentile": float(self.config.weak_texture_percentile),
            "weak_texture_min_conf_thr": float(self.config.weak_texture_min_conf_thr),
            "weak_texture_keep_ratio": float(self.config.weak_texture_keep_ratio),
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
                "views_total": int(len(output_dict["preds"])),
                "views_rejected_by_confidence": int(sum(1 for row in per_view_confidence_stats if row.get("view_rejected"))),
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
        }

        if output_dir is not None:
            output_root = Path(output_dir)
            np.save(output_root / "raw_confidences.npy", confidences)
            np.save(output_root / "source_view_ids.npy", source_view_ids)
            np.save(output_root / "source_pixel_xy.npy", source_pixel_xy)
            np.save(output_root / "confidence_maps.npy", confidence_maps_arr)
            report_path = output_root / "fast3r_point_retention_report.json"
            report_path.write_text(json.dumps(self.last_metadata, indent=2, default=str), encoding="utf-8")
            self.last_metadata["point_retention_report"] = str(report_path)

        for image_path in image_paths:
            try:
                Path(image_path).unlink()
            except OSError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        _emit(progress_callback, 88, "Fast3R reconstruction complete")
        return points, colors, list(camera_poses)
