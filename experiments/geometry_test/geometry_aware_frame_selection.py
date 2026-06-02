from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore[import-not-found]
except ModuleNotFoundError:
    cv2 = None

try:
    import torch  # type: ignore[import-not-found]
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[2]
FAST3R_REPO = REPO_ROOT / "fast3r"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if FAST3R_REPO.exists() and str(FAST3R_REPO) not in sys.path:
    sys.path.insert(0, str(FAST3R_REPO))


@dataclass
class VisualCandidate:
    source_frame_index: int
    timestamp_seconds: float
    laplacian_variance: float
    entropy_bits: float
    brightness_mean: float
    clipped_ratio: float
    orb_keypoints: int
    thumbnail_vector: np.ndarray
    visual_score: float = 0.0
    shortlist_rank: int = -1
    probe_rank: int = -1


@dataclass
class ProbeViewMetric:
    probe_rank: int
    source_frame_index: int
    timestamp_seconds: float
    pose_valid: bool
    estimated_focal: float
    confidence_mean: float
    confidence_median: float
    confidence_ratio_above_threshold: float
    valid_depth_ratio: float
    point_extent_x: float
    point_extent_y: float
    point_extent_z: float
    point_spread: float
    camera_center_x: float
    camera_center_y: float
    camera_center_z: float
    forward_x: float
    forward_y: float
    forward_z: float
    visual_score: float
    geometry_quality_score: float = 0.0


@dataclass
class PairMetric:
    source_frame_index_a: int
    source_frame_index_b: int
    probe_rank_a: int
    probe_rank_b: int
    camera_distance: float
    view_angle_deg: float
    confidence_iou: float
    baseline_diversity_score: float = 0.0
    overlap_balance_score: float = 0.0


@dataclass
class SelectedFrameRecord:
    selection_order: int
    probe_rank: int
    source_frame_index: int
    timestamp_seconds: float
    visual_score: float
    geometry_quality_score: float
    final_selection_score: float
    pose_valid: bool
    confidence_ratio_above_threshold: float
    point_spread: float


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _require_cv2() -> Any:
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV (cv2) is required to run this experiment.")
    return cv2


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run this experiment.")
    return torch


def _grayscale_entropy(gray_frame: np.ndarray) -> float:
    cv2_mod = _require_cv2()
    hist = cv2_mod.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(float(hist.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _robust_normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.float32)
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        normalized = np.full(values.shape, 0.5, dtype=np.float64)
    else:
        normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32)


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = _normalize_vector(vector_a)
    b = _normalize_vector(vector_b)
    if not np.any(a) or not np.any(b):
        return 0.0
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _confidence_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return _safe_div(intersection, union)


def _candidate_metrics(frame_bgr: np.ndarray, source_frame_index: int, fps: float, orb: Any) -> VisualCandidate:
    cv2_mod = _require_cv2()
    gray = cv2_mod.cvtColor(frame_bgr, cv2_mod.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2_mod.Laplacian(gray, cv2_mod.CV_64F).var())
    entropy_bits = _grayscale_entropy(gray)
    brightness_mean = float(gray.mean())
    under_ratio = float(np.mean(gray <= 10))
    over_ratio = float(np.mean(gray >= 245))
    clipped_ratio = float(under_ratio + over_ratio)
    orb_keypoints = len(orb.detect(gray, None))

    thumb = cv2_mod.resize(gray, (24, 24), interpolation=cv2_mod.INTER_AREA).astype(np.float32).reshape(-1)
    thumb -= float(thumb.mean())
    norm = float(np.linalg.norm(thumb))
    if norm > 0:
        thumb /= norm

    return VisualCandidate(
        source_frame_index=source_frame_index,
        timestamp_seconds=_safe_div(source_frame_index, fps),
        laplacian_variance=laplacian_variance,
        entropy_bits=entropy_bits,
        brightness_mean=brightness_mean,
        clipped_ratio=clipped_ratio,
        orb_keypoints=orb_keypoints,
        thumbnail_vector=thumb,
    )


def score_visual_candidates(candidates: list[VisualCandidate]) -> None:
    if not candidates:
        return

    sharpness = np.log1p(np.array([c.laplacian_variance for c in candidates], dtype=np.float64))
    texture = np.log1p(np.array([c.orb_keypoints for c in candidates], dtype=np.float64))
    entropy = np.array([c.entropy_bits for c in candidates], dtype=np.float64)
    clipped = np.array([c.clipped_ratio for c in candidates], dtype=np.float64)
    brightness_dev = np.abs(np.array([c.brightness_mean for c in candidates], dtype=np.float64) - 127.5)

    sharpness_score = _robust_normalize(sharpness)
    texture_score = _robust_normalize(texture)
    entropy_score = _robust_normalize(entropy)
    exposure_score = _robust_normalize(clipped, invert=True)
    brightness_score = _robust_normalize(brightness_dev, invert=True)

    for idx, candidate in enumerate(candidates):
        candidate.visual_score = float(
            0.40 * sharpness_score[idx]
            + 0.25 * texture_score[idx]
            + 0.15 * entropy_score[idx]
            + 0.10 * exposure_score[idx]
            + 0.10 * brightness_score[idx]
        )


def shortlist_visual_candidates(
    candidates: list[VisualCandidate],
    target_frames: int,
    shortlist_multiplier: int,
    dedupe_similarity: float,
    min_frame_gap: int,
) -> list[VisualCandidate]:
    sorted_candidates = sorted(candidates, key=lambda item: item.source_frame_index)
    num_bins = min(len(sorted_candidates), max(1, target_frames))
    max_frame_index = max(candidate.source_frame_index for candidate in sorted_candidates)
    bin_edges = np.linspace(0, max(max_frame_index + 1, 1), num_bins + 1, dtype=np.int64)
    bins: list[list[VisualCandidate]] = [[] for _ in range(num_bins)]

    for candidate in sorted_candidates:
        bin_index = int(np.searchsorted(bin_edges, candidate.source_frame_index, side="right") - 1)
        bin_index = max(0, min(bin_index, num_bins - 1))
        bins[bin_index].append(candidate)

    shortlist: list[VisualCandidate] = []
    per_bin_keep = max(1, int(shortlist_multiplier))
    for bin_index, bucket in enumerate(bins):
        if not bucket:
            continue
        bin_left = float(bin_edges[bin_index])
        bin_right = float(bin_edges[bin_index + 1])
        bin_center = 0.5 * (bin_left + bin_right)
        bin_half_width = max((bin_right - bin_left) * 0.5, 1.0)
        bucket.sort(
            key=lambda item: (
                item.visual_score - 0.12 * min(abs(item.source_frame_index - bin_center) / bin_half_width, 1.0),
                item.visual_score,
            ),
            reverse=True,
        )
        shortlist.extend(bucket[:per_bin_keep])

    shortlist = sorted(shortlist, key=lambda item: item.visual_score, reverse=True)
    deduped: list[VisualCandidate] = []
    for candidate in shortlist:
        valid = True
        for existing in deduped:
            if abs(candidate.source_frame_index - existing.source_frame_index) < min_frame_gap:
                valid = False
                break
            if _cosine_similarity(candidate.thumbnail_vector, existing.thumbnail_vector) >= dedupe_similarity:
                valid = False
                break
        if valid:
            candidate.shortlist_rank = len(deduped)
            deduped.append(candidate)

    return deduped


def cap_probe_pool(
    shortlist: list[VisualCandidate],
    probe_max_frames: int | None,
) -> list[VisualCandidate]:
    if probe_max_frames is None or len(shortlist) <= probe_max_frames:
        for idx, candidate in enumerate(shortlist):
            candidate.probe_rank = idx
        return shortlist

    target_count = max(1, probe_max_frames)
    num_bins = min(len(shortlist), target_count)
    sorted_candidates = sorted(shortlist, key=lambda item: item.source_frame_index)
    max_frame_index = max(candidate.source_frame_index for candidate in sorted_candidates)
    bin_edges = np.linspace(0, max(max_frame_index + 1, 1), num_bins + 1, dtype=np.int64)
    bins: list[list[VisualCandidate]] = [[] for _ in range(num_bins)]
    for candidate in sorted_candidates:
        bin_index = int(np.searchsorted(bin_edges, candidate.source_frame_index, side="right") - 1)
        bin_index = max(0, min(bin_index, num_bins - 1))
        bins[bin_index].append(candidate)

    capped: list[VisualCandidate] = []
    for bucket in bins:
        if not bucket:
            continue
        bucket.sort(key=lambda item: item.visual_score, reverse=True)
        capped.append(bucket[0])

    if len(capped) < target_count:
        used = {candidate.source_frame_index for candidate in capped}
        for candidate in sorted(shortlist, key=lambda item: item.visual_score, reverse=True):
            if candidate.source_frame_index in used:
                continue
            capped.append(candidate)
            used.add(candidate.source_frame_index)
            if len(capped) >= target_count:
                break

    capped = sorted(capped[:target_count], key=lambda item: item.source_frame_index)
    for idx, candidate in enumerate(capped):
        candidate.probe_rank = idx
    return capped


def scan_video_candidates(
    video_path: Path,
    scan_stride: int,
    analysis_size: tuple[int, int],
    orb_features: int,
) -> tuple[list[VisualCandidate], dict[str, Any]]:
    cv2_mod = _require_cv2()
    cap = cv2_mod.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2_mod.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2_mod.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2_mod.CAP_PROP_FRAME_HEIGHT) or 0)
    orb = cv2_mod.ORB_create(nfeatures=orb_features)

    candidates: list[VisualCandidate] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, scan_stride) == 0:
            analysis_frame = cv2_mod.resize(frame, analysis_size, interpolation=cv2_mod.INTER_AREA)
            candidates.append(_candidate_metrics(analysis_frame, frame_idx, fps, orb))
        frame_idx += 1
    cap.release()

    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "original_resolution": [width, height],
        "analysis_resolution": list(analysis_size),
        "scan_stride": int(scan_stride),
        "candidate_count": len(candidates),
    }
    return candidates, metadata


def load_video_frames_by_index(
    video_path: Path,
    frame_indices: list[int],
    output_size: tuple[int, int],
) -> dict[int, np.ndarray]:
    cv2_mod = _require_cv2()
    cap = cv2_mod.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    frames: dict[int, np.ndarray] = {}
    for frame_index in frame_indices:
        cap.set(cv2_mod.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        frames[int(frame_index)] = cv2_mod.resize(frame, output_size, interpolation=cv2_mod.INTER_AREA)
    cap.release()
    return frames


def load_fast3r_model(model_name: str, device: Any, max_parallel_views_for_head: int | None) -> Any:
    from fast3r.models.fast3r import Fast3R

    model = Fast3R.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    if max_parallel_views_for_head is not None and hasattr(model, "set_max_parallel_views_for_head"):
        model.set_max_parallel_views_for_head(int(max_parallel_views_for_head))
    return model


def run_fast3r_probe(
    model: Any,
    frames_by_index: dict[int, np.ndarray],
    ordered_candidates: list[VisualCandidate],
    output_dir: Path,
    image_size: int,
    dtype_name: str,
    pose_iterations: int,
    focal_method: str,
    confidence_threshold: float,
    device: Any,
) -> tuple[list[ProbeViewMetric], list[PairMetric]]:
    cv2_mod = _require_cv2()
    torch_mod = _require_torch()
    from src.pipeline.fast3r.reconstruction import estimate_camera_poses_simple
    from fast3r.dust3r.inference_multiview import inference
    from fast3r.dust3r.utils.image import load_images

    temp_dir = Path(tempfile.mkdtemp(prefix="geometry_probe_fast3r_"))

    image_paths: list[str] = []
    ordered_frames: list[np.ndarray] = []
    for candidate in ordered_candidates:
        frame = frames_by_index[candidate.source_frame_index]
        ordered_frames.append(frame)
        frame_path = temp_dir / f"probe_{candidate.probe_rank:03d}_src_{candidate.source_frame_index:06d}.jpg"
        cv2_mod.imwrite(str(frame_path), frame)
        image_paths.append(str(frame_path))

    images = load_images(image_paths, size=image_size, verbose=False)
    dtype = torch_mod.float32 if dtype_name == "float32" else torch_mod.bfloat16
    output_dict, _ = inference(images, model, device, dtype=dtype, verbose=False, profiling=True)

    preds_gpu: list[dict[str, Any]] = []
    for pred in output_dict["preds"]:
        pred_gpu: dict[str, Any] = {}
        for key, value in pred.items():
            pred_gpu[key] = value.to(device) if isinstance(value, torch_mod.Tensor) else value
        preds_gpu.append(pred_gpu)

    poses_c2w_batch, estimated_focals_batch = estimate_camera_poses_simple(
        preds_gpu,
        niter_pnp=pose_iterations,
        focal_length_estimation_method=focal_method,
    )
    poses = poses_c2w_batch[0]
    focals = estimated_focals_batch[0]

    probe_metrics: list[ProbeViewMetric] = []
    conf_masks: list[np.ndarray] = []
    for probe_rank, candidate in enumerate(ordered_candidates):
        pred = output_dict["preds"][probe_rank]
        pts3d = pred["pts3d_in_other_view"]
        if isinstance(pts3d, torch_mod.Tensor):
            pts3d = pts3d.detach().cpu().numpy()
        pts3d = pts3d[0]

        conf = pred.get("conf")
        if isinstance(conf, torch_mod.Tensor):
            conf = conf.detach().cpu().numpy()
        if conf is None:
            conf = np.ones(pts3d.shape[:2], dtype=np.float32)
        elif getattr(conf, "ndim", 0) > 2:
            conf = conf[0]

        conf_mask = conf > confidence_threshold
        conf_masks.append(conf_mask)

        valid_points = pts3d[conf_mask]
        if valid_points.size == 0:
            point_extent = np.zeros(3, dtype=np.float32)
            point_spread = 0.0
            valid_depth_ratio = 0.0
        else:
            bbox_min = np.percentile(valid_points, 5, axis=0)
            bbox_max = np.percentile(valid_points, 95, axis=0)
            point_extent = (bbox_max - bbox_min).astype(np.float32)
            point_spread = float(np.linalg.norm(point_extent))
            valid_depth_ratio = float(np.mean(pts3d[:, :, 2] > 0))

        pose = np.asarray(poses[probe_rank], dtype=np.float32)
        focal = float(focals[probe_rank]) if probe_rank < len(focals) else 0.0
        pose_valid = bool(focal > 0.0 and not np.allclose(pose, np.eye(4, dtype=np.float32), atol=1e-4))
        camera_center = pose[:3, 3] if pose.shape == (4, 4) else np.zeros(3, dtype=np.float32)
        forward = pose[:3, 2] if pose.shape == (4, 4) else np.zeros(3, dtype=np.float32)
        forward = _normalize_vector(forward)

        probe_metrics.append(
            ProbeViewMetric(
                probe_rank=probe_rank,
                source_frame_index=candidate.source_frame_index,
                timestamp_seconds=candidate.timestamp_seconds,
                pose_valid=pose_valid,
                estimated_focal=focal,
                confidence_mean=float(conf.mean()),
                confidence_median=float(np.median(conf)),
                confidence_ratio_above_threshold=float(np.mean(conf_mask)),
                valid_depth_ratio=valid_depth_ratio,
                point_extent_x=float(point_extent[0]),
                point_extent_y=float(point_extent[1]),
                point_extent_z=float(point_extent[2]),
                point_spread=point_spread,
                camera_center_x=float(camera_center[0]),
                camera_center_y=float(camera_center[1]),
                camera_center_z=float(camera_center[2]),
                forward_x=float(forward[0]),
                forward_y=float(forward[1]),
                forward_z=float(forward[2]),
                visual_score=float(candidate.visual_score),
            )
        )

    shutil.rmtree(temp_dir, ignore_errors=True)

    assign_geometry_quality_scores(probe_metrics)
    pair_metrics = build_pair_metrics(probe_metrics, conf_masks)
    return probe_metrics, pair_metrics


def assign_geometry_quality_scores(probe_metrics: list[ProbeViewMetric]) -> None:
    if not probe_metrics:
        return

    confidence_ratio = np.array([item.confidence_ratio_above_threshold for item in probe_metrics], dtype=np.float64)
    confidence_mean = np.array([item.confidence_mean for item in probe_metrics], dtype=np.float64)
    point_spread = np.array([item.point_spread for item in probe_metrics], dtype=np.float64)
    valid_depth_ratio = np.array([item.valid_depth_ratio for item in probe_metrics], dtype=np.float64)
    pose_valid = np.array([1.0 if item.pose_valid else 0.0 for item in probe_metrics], dtype=np.float64)

    conf_ratio_score = _robust_normalize(confidence_ratio)
    conf_mean_score = _robust_normalize(confidence_mean)
    spread_score = _robust_normalize(np.log1p(point_spread))
    depth_score = _robust_normalize(valid_depth_ratio)

    for idx, item in enumerate(probe_metrics):
        item.geometry_quality_score = float(
            0.35 * conf_ratio_score[idx]
            + 0.20 * conf_mean_score[idx]
            + 0.20 * spread_score[idx]
            + 0.10 * depth_score[idx]
            + 0.15 * pose_valid[idx]
        )


def build_pair_metrics(probe_metrics: list[ProbeViewMetric], conf_masks: list[np.ndarray]) -> list[PairMetric]:
    pair_metrics: list[PairMetric] = []
    if len(probe_metrics) < 2:
        return pair_metrics

    centers = [
        np.array([item.camera_center_x, item.camera_center_y, item.camera_center_z], dtype=np.float32)
        for item in probe_metrics
    ]
    forwards = [
        np.array([item.forward_x, item.forward_y, item.forward_z], dtype=np.float32)
        for item in probe_metrics
    ]

    raw_pairs: list[PairMetric] = []
    for idx_a in range(len(probe_metrics)):
        for idx_b in range(idx_a + 1, len(probe_metrics)):
            metric_a = probe_metrics[idx_a]
            metric_b = probe_metrics[idx_b]
            distance = float(np.linalg.norm(centers[idx_a] - centers[idx_b]))
            angle = _angle_degrees(forwards[idx_a], forwards[idx_b])
            iou = _confidence_iou(conf_masks[idx_a], conf_masks[idx_b])
            raw_pairs.append(
                PairMetric(
                    source_frame_index_a=metric_a.source_frame_index,
                    source_frame_index_b=metric_b.source_frame_index,
                    probe_rank_a=metric_a.probe_rank,
                    probe_rank_b=metric_b.probe_rank,
                    camera_distance=distance,
                    view_angle_deg=angle,
                    confidence_iou=iou,
                )
            )

    distance_scores = _robust_normalize(np.array([item.camera_distance for item in raw_pairs], dtype=np.float64))
    angle_scores = _robust_normalize(np.array([item.view_angle_deg for item in raw_pairs], dtype=np.float64))
    overlap_target = 0.35
    overlap_balance = []
    for item in raw_pairs:
        overlap_balance.append(max(0.0, 1.0 - abs(item.confidence_iou - overlap_target) / overlap_target))
    overlap_scores = np.array(overlap_balance, dtype=np.float64)

    for idx, item in enumerate(raw_pairs):
        item.baseline_diversity_score = float(0.60 * distance_scores[idx] + 0.40 * angle_scores[idx])
        item.overlap_balance_score = float(overlap_scores[idx])
        pair_metrics.append(item)

    return pair_metrics


def pair_metric_lookup(pair_metrics: list[PairMetric]) -> dict[tuple[int, int], PairMetric]:
    lookup: dict[tuple[int, int], PairMetric] = {}
    for item in pair_metrics:
        lookup[(item.probe_rank_a, item.probe_rank_b)] = item
        lookup[(item.probe_rank_b, item.probe_rank_a)] = item
    return lookup


def select_geometry_aware_subset(
    probe_metrics: list[ProbeViewMetric],
    pair_metrics: list[PairMetric],
    target_frames: int,
) -> list[SelectedFrameRecord]:
    if not probe_metrics:
        return []

    lookup = pair_metric_lookup(pair_metrics)
    target_count = min(max(1, target_frames), len(probe_metrics))

    selected: list[ProbeViewMetric] = []
    selected_records: list[SelectedFrameRecord] = []
    selected_ranks: set[int] = set()

    first_candidate = max(
        probe_metrics,
        key=lambda item: (item.geometry_quality_score, item.visual_score, -abs(item.timestamp_seconds)),
    )
    selected.append(first_candidate)
    selected_ranks.add(first_candidate.probe_rank)
    selected_records.append(
        SelectedFrameRecord(
            selection_order=0,
            probe_rank=first_candidate.probe_rank,
            source_frame_index=first_candidate.source_frame_index,
            timestamp_seconds=first_candidate.timestamp_seconds,
            visual_score=first_candidate.visual_score,
            geometry_quality_score=first_candidate.geometry_quality_score,
            final_selection_score=first_candidate.geometry_quality_score,
            pose_valid=first_candidate.pose_valid,
            confidence_ratio_above_threshold=first_candidate.confidence_ratio_above_threshold,
            point_spread=first_candidate.point_spread,
        )
    )

    while len(selected) < target_count:
        remaining = [item for item in probe_metrics if item.probe_rank not in selected_ranks]
        if not remaining:
            break

        best_item: ProbeViewMetric | None = None
        best_score = -float("inf")
        for candidate in remaining:
            pair_candidates = [lookup.get((candidate.probe_rank, existing.probe_rank)) for existing in selected]
            pair_candidates = [pair for pair in pair_candidates if pair is not None]

            if pair_candidates:
                diversity = float(max(pair.baseline_diversity_score for pair in pair_candidates))
                overlap = float(max(pair.overlap_balance_score for pair in pair_candidates))
            else:
                diversity = 0.0
                overlap = 0.0

            temporal_gap = min(
                abs(candidate.source_frame_index - existing.source_frame_index)
                for existing in selected
            )
            temporal_score = float(np.tanh(temporal_gap / 180.0))

            final_score = float(
                0.40 * candidate.geometry_quality_score
                + 0.20 * candidate.visual_score
                + 0.20 * diversity
                + 0.10 * overlap
                + 0.10 * temporal_score
            )
            if final_score > best_score:
                best_score = final_score
                best_item = candidate

        if best_item is None:
            break

        selected.append(best_item)
        selected_ranks.add(best_item.probe_rank)
        selected_records.append(
            SelectedFrameRecord(
                selection_order=len(selected_records),
                probe_rank=best_item.probe_rank,
                source_frame_index=best_item.source_frame_index,
                timestamp_seconds=best_item.timestamp_seconds,
                visual_score=best_item.visual_score,
                geometry_quality_score=best_item.geometry_quality_score,
                final_selection_score=best_score,
                pose_valid=best_item.pose_valid,
                confidence_ratio_above_threshold=best_item.confidence_ratio_above_threshold,
                point_spread=best_item.point_spread,
            )
        )

    return selected_records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def visual_candidates_to_rows(candidates: list[VisualCandidate]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "source_frame_index": candidate.source_frame_index,
                "timestamp_seconds": candidate.timestamp_seconds,
                "laplacian_variance": candidate.laplacian_variance,
                "entropy_bits": candidate.entropy_bits,
                "brightness_mean": candidate.brightness_mean,
                "clipped_ratio": candidate.clipped_ratio,
                "orb_keypoints": candidate.orb_keypoints,
                "visual_score": candidate.visual_score,
                "shortlist_rank": candidate.shortlist_rank,
                "probe_rank": candidate.probe_rank,
            }
        )
    return rows


def dataclass_rows(items: list[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]


def build_summary(
    config: argparse.Namespace,
    scan_metadata: dict[str, Any],
    shortlist: list[VisualCandidate],
    probe_metrics: list[ProbeViewMetric],
    selected_records: list[SelectedFrameRecord],
) -> dict[str, Any]:
    selected_indices = [item.source_frame_index for item in selected_records]
    probe_pose_valid = int(sum(1 for item in probe_metrics if item.pose_valid))
    return {
        "video_path": str(Path(config.video).resolve()),
        "target_frames_requested": int(config.target_frames),
        "visual_shortlist_target_requested": int(config.visual_shortlist_target),
        "probe_max_frames_requested": int(config.probe_max_frames),
        "selected_frame_count": int(len(selected_records)),
        "selected_frame_indices": selected_indices,
        "scan_metadata": scan_metadata,
        "visual_shortlist_count": int(len(shortlist)),
        "probe_pool_count": int(len(probe_metrics)),
        "probe_pose_valid_count": probe_pose_valid,
        "probe_pose_valid_ratio": _safe_div(probe_pose_valid, len(probe_metrics)),
        "probe_image_size": int(config.probe_image_size),
        "probe_dtype": config.probe_dtype,
        "probe_confidence_threshold": float(config.probe_confidence_threshold),
        "focal_method": config.focal_method,
    }


def write_report_txt(
    path: Path,
    summary: dict[str, Any],
    selected_records: list[SelectedFrameRecord],
) -> None:
    lines = [
        "Geometry-Aware Frame Selection Report",
        "",
        f"Video: {summary['video_path']}",
        f"Requested final frames: {summary['target_frames_requested']}",
        f"Requested visual shortlist target: {summary['visual_shortlist_target_requested']}",
        f"Requested Fast3R probe cap: {summary['probe_max_frames_requested']}",
        f"Final selected frames: {summary['selected_frame_count']}",
        f"Visual shortlist count: {summary['visual_shortlist_count']}",
        f"Fast3R probe pool count: {summary['probe_pool_count']}",
        f"Pose-valid probe views: {summary['probe_pose_valid_count']} / {summary['probe_pool_count']}",
        "",
        "Selection design:",
        "- Stage 1 performs cheap visual prefiltering using sharpness, texture, entropy, exposure, and deduplication.",
        "- Stage 2 runs a low-resolution Fast3R probe pass on the shortlist.",
        "- Final greedy selection favors geometry quality, viewpoint diversity, moderate overlap, and temporal coverage.",
        "",
        "Selected frames:",
    ]
    for item in selected_records:
        lines.append(
            f"- order={item.selection_order} src={item.source_frame_index} "
            f"time={item.timestamp_seconds:.2f}s visual={item.visual_score:.3f} "
            f"geometry={item.geometry_quality_score:.3f} final={item.final_selection_score:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage geometry-aware Fast3R frame selector experiment.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument(
        "--target-frames",
        type=int,
        default=64,
        help="Final number of frames kept for the full-resolution reconstruction stage.",
    )
    parser.add_argument(
        "--visual-shortlist-target",
        type=int,
        default=96,
        help="How many visually filtered candidates to keep before the Fast3R probe stage.",
    )
    parser.add_argument("--scan-stride", type=int, default=15)
    parser.add_argument("--shortlist-multiplier", type=int, default=3)
    parser.add_argument("--dedupe-similarity", type=float, default=0.985)
    parser.add_argument("--min-frame-gap", type=int, default=120)
    parser.add_argument("--analysis-width", type=int, default=384)
    parser.add_argument("--analysis-height", type=int, default=288)
    parser.add_argument("--output-width", type=int, default=1024)
    parser.add_argument("--output-height", type=int, default=768)
    parser.add_argument("--orb-features", type=int, default=1000)
    parser.add_argument(
        "--probe-max-frames",
        type=int,
        default=80,
        help="Maximum number of shortlisted frames passed to the low-resolution Fast3R probe.",
    )
    parser.add_argument("--probe-image-size", type=int, default=256)
    parser.add_argument("--probe-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--probe-confidence-threshold", type=float, default=1.0)
    parser.add_argument("--probe-pnp-iters", type=int, default=50)
    parser.add_argument(
        "--focal-method",
        choices=["first_view_from_global_head", "individual"],
        default="first_view_from_global_head",
    )
    parser.add_argument("--model-name", type=str, default="jedyang97/Fast3R_ViT_Large_512")
    parser.add_argument("--probe-max-parallel-views", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_cv2()
    torch_mod = _require_torch()
    if args.target_frames < 1:
        raise ValueError("--target-frames must be at least 1.")
    if args.visual_shortlist_target < args.target_frames:
        raise ValueError("--visual-shortlist-target must be greater than or equal to --target-frames.")
    if args.probe_max_frames < args.target_frames:
        raise ValueError("--probe-max-frames must be greater than or equal to --target-frames.")

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates, scan_metadata = scan_video_candidates(
        video_path=video_path,
        scan_stride=args.scan_stride,
        analysis_size=(int(args.analysis_width), int(args.analysis_height)),
        orb_features=args.orb_features,
    )
    if not candidates:
        raise RuntimeError("No candidates were extracted from the source video.")

    score_visual_candidates(candidates)
    shortlist = shortlist_visual_candidates(
        candidates=candidates,
        target_frames=args.visual_shortlist_target,
        shortlist_multiplier=args.shortlist_multiplier,
        dedupe_similarity=args.dedupe_similarity,
        min_frame_gap=args.min_frame_gap,
    )
    if not shortlist:
        raise RuntimeError("Visual prefilter produced an empty shortlist.")

    probe_pool = cap_probe_pool(shortlist, args.probe_max_frames)
    frame_indices = [candidate.source_frame_index for candidate in probe_pool]
    frames_by_index = load_video_frames_by_index(
        video_path=video_path,
        frame_indices=frame_indices,
        output_size=(int(args.output_width), int(args.output_height)),
    )
    probe_pool = [candidate for candidate in probe_pool if candidate.source_frame_index in frames_by_index]
    if not probe_pool:
        raise RuntimeError("Could not load any shortlisted frames for the probe pass.")

    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    model = load_fast3r_model(
        model_name=args.model_name,
        device=device,
        max_parallel_views_for_head=args.probe_max_parallel_views,
    )
    probe_metrics, pair_metrics = run_fast3r_probe(
        model=model,
        frames_by_index=frames_by_index,
        ordered_candidates=probe_pool,
        output_dir=output_dir,
        image_size=args.probe_image_size,
        dtype_name=args.probe_dtype,
        pose_iterations=args.probe_pnp_iters,
        focal_method=args.focal_method,
        confidence_threshold=args.probe_confidence_threshold,
        device=device,
    )
    selected_records = select_geometry_aware_subset(
        probe_metrics=probe_metrics,
        pair_metrics=pair_metrics,
        target_frames=args.target_frames,
    )

    for item in selected_records:
        frame = frames_by_index[item.source_frame_index]
        output_path = output_dir / (
            f"selected_frame_{item.selection_order:03d}_src_{item.source_frame_index:06d}_"
            f"score_{item.final_selection_score:.3f}.jpg"
        )
        cv2_mod = _require_cv2()
        cv2_mod.imwrite(str(output_path), frame)

    visual_rows = visual_candidates_to_rows(sorted(candidates, key=lambda item: item.source_frame_index))
    write_csv(
        output_dir / "visual_candidates.csv",
        visual_rows,
        fieldnames=list(visual_rows[0].keys()),
    )
    if probe_metrics:
        probe_rows = dataclass_rows(probe_metrics)
        write_csv(
            output_dir / "probe_view_metrics.csv",
            probe_rows,
            fieldnames=list(probe_rows[0].keys()),
        )
    if pair_metrics:
        pair_rows = dataclass_rows(pair_metrics)
        write_csv(
            output_dir / "probe_pair_metrics.csv",
            pair_rows,
            fieldnames=list(pair_rows[0].keys()),
        )
    if selected_records:
        selected_rows = dataclass_rows(selected_records)
        write_csv(
            output_dir / "selected_frame_manifest.csv",
            selected_rows,
            fieldnames=list(selected_rows[0].keys()),
        )

    summary = build_summary(args, scan_metadata, shortlist, probe_metrics, selected_records)
    (output_dir / "geometry_selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report_txt(output_dir / "geometry_selection_report.txt", summary, selected_records)

    print(f"Wrote geometry-aware experiment outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
