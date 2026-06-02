# NOTE: Do NOT import directly from experiments/.
# All pipeline code must be copied into src/pipeline/fast3r/ before modification.
# Best methods from experiments/ have been audited and integrated — see src/EXPERIMENTS_AUDIT.md

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch


ProgressCallback = Callable[[int, str], None] | None
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_FAST3R_REPO = REPO_ROOT / "fast3r"
if LOCAL_FAST3R_REPO.exists() and str(LOCAL_FAST3R_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_FAST3R_REPO))


@dataclass
class CandidateFrame:
    source_frame_index: int
    timestamp_seconds: float
    laplacian_variance: float
    entropy_bits: float
    brightness_mean: float
    clipped_ratio: float
    orb_keypoints: int
    thumbnail_vector: np.ndarray
    bin_index: int = -1
    score: float = 0.0
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


def _emit(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback is not None:
        callback(percent, message)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _effective_min_frame_gap(total_frames: int, target_frames: int, requested_gap: int, scan_stride: int) -> tuple[int, int]:
    _ = scan_stride
    if requested_gap <= 0 or target_frames <= 1 or total_frames <= 0:
        return max(0, int(requested_gap)), 0

    max_gap_for_target = max(0, (int(total_frames) - 1) // max(1, int(target_frames) - 1))
    if max_gap_for_target <= 0:
        return 0, max_gap_for_target

    effective_gap = min(int(requested_gap), max_gap_for_target)
    return effective_gap, max_gap_for_target


def _grayscale_entropy(gray_frame: np.ndarray) -> float:
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
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


def _candidate_metrics(frame_bgr: np.ndarray, source_frame_index: int, fps: float, orb: Any) -> CandidateFrame:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy_bits = _grayscale_entropy(gray)
    brightness_mean = float(gray.mean())
    under_ratio = float(np.mean(gray <= 10))
    over_ratio = float(np.mean(gray >= 245))
    clipped_ratio = float(under_ratio + over_ratio)
    orb_keypoints = len(orb.detect(gray, None))

    thumb = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    thumb -= float(thumb.mean())
    norm = float(np.linalg.norm(thumb))
    if norm > 0:
        thumb /= norm

    return CandidateFrame(
        source_frame_index=source_frame_index,
        timestamp_seconds=_safe_div(source_frame_index, fps),
        laplacian_variance=laplacian_variance,
        entropy_bits=entropy_bits,
        brightness_mean=brightness_mean,
        clipped_ratio=clipped_ratio,
        orb_keypoints=orb_keypoints,
        thumbnail_vector=thumb,
    )


def _score_candidates(candidates: list[CandidateFrame]) -> None:
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
        candidate.score = float(
            0.40 * sharpness_score[idx]
            + 0.25 * texture_score[idx]
            + 0.15 * entropy_score[idx]
            + 0.10 * exposure_score[idx]
            + 0.10 * brightness_score[idx]
        )


def _shortlist_candidates(
    candidates: list[CandidateFrame],
    shortlist_target: int,
    shortlist_multiplier: int,
    dedupe_similarity: float,
    min_frame_gap: int,
) -> list[CandidateFrame]:
    sorted_candidates = sorted(candidates, key=lambda item: item.source_frame_index)
    num_bins = min(len(sorted_candidates), max(1, shortlist_target))
    max_frame_index = max(candidate.source_frame_index for candidate in sorted_candidates)
    bin_edges = np.linspace(0, max(max_frame_index + 1, 1), num_bins + 1, dtype=np.int64)
    bins: list[list[CandidateFrame]] = [[] for _ in range(num_bins)]

    for candidate in sorted_candidates:
        bin_index = int(np.searchsorted(bin_edges, candidate.source_frame_index, side="right") - 1)
        bin_index = max(0, min(bin_index, num_bins - 1))
        candidate.bin_index = bin_index
        bins[bin_index].append(candidate)

    shortlist: list[CandidateFrame] = []
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
                item.score - 0.12 * min(abs(item.source_frame_index - bin_center) / bin_half_width, 1.0),
                item.score,
            ),
            reverse=True,
        )
        shortlist.extend(bucket[:per_bin_keep])

    shortlist = sorted(shortlist, key=lambda item: item.score, reverse=True)
    deduped: list[CandidateFrame] = []
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


def _is_valid_candidate(
    candidate: CandidateFrame,
    selected: list[CandidateFrame],
    dedupe_similarity: float,
    min_frame_gap: int,
    enforce_gap: bool = True,
    enforce_similarity: bool = True,
) -> bool:
    for existing in selected:
        if enforce_gap and abs(candidate.source_frame_index - existing.source_frame_index) < min_frame_gap:
            return False
        if enforce_similarity and _cosine_similarity(candidate.thumbnail_vector, existing.thumbnail_vector) >= dedupe_similarity:
            return False
    return True


def _select_visual_prefilter_frames(
    shortlist: list[CandidateFrame],
    target_frames: int,
    dedupe_similarity: float,
    min_frame_gap: int,
) -> list[CandidateFrame]:
    selected: list[CandidateFrame] = []
    selected_indices: set[int] = set()
    grouped_by_bin: dict[int, list[CandidateFrame]] = {}

    for candidate in shortlist:
        grouped_by_bin.setdefault(candidate.bin_index, []).append(candidate)

    for bucket in grouped_by_bin.values():
        bucket.sort(key=lambda item: item.score, reverse=True)

    for bin_index in sorted(grouped_by_bin.keys()):
        if len(selected) >= target_frames:
            break
        bucket = grouped_by_bin[bin_index]
        chosen = None
        for candidate in bucket:
            if candidate.source_frame_index in selected_indices:
                continue
            if _is_valid_candidate(candidate, selected, dedupe_similarity, min_frame_gap, True, True):
                chosen = candidate
                break
        if chosen is None:
            for candidate in bucket:
                if candidate.source_frame_index in selected_indices:
                    continue
                if _is_valid_candidate(candidate, selected, dedupe_similarity, min_frame_gap, False, True):
                    chosen = candidate
                    break
        if chosen is None and bucket:
            chosen = bucket[0]
        if chosen is not None:
            selected.append(chosen)
            selected_indices.add(chosen.source_frame_index)

    for candidate in shortlist:
        if len(selected) >= target_frames:
            break
        if candidate.source_frame_index in selected_indices:
            continue
        if _is_valid_candidate(candidate, selected, dedupe_similarity, min_frame_gap, True, True):
            selected.append(candidate)
            selected_indices.add(candidate.source_frame_index)

    for candidate in shortlist:
        if len(selected) >= target_frames:
            break
        if candidate.source_frame_index in selected_indices:
            continue
        if _is_valid_candidate(candidate, selected, dedupe_similarity, min_frame_gap, False, True):
            selected.append(candidate)
            selected_indices.add(candidate.source_frame_index)

    for candidate in shortlist:
        if len(selected) >= target_frames:
            break
        if candidate.source_frame_index in selected_indices:
            continue
        selected.append(candidate)
        selected_indices.add(candidate.source_frame_index)

    return sorted(selected[:target_frames], key=lambda item: item.source_frame_index)


def _cap_probe_pool(shortlist: list[CandidateFrame], probe_max_frames: int | None) -> list[CandidateFrame]:
    if probe_max_frames is None or len(shortlist) <= probe_max_frames:
        for idx, candidate in enumerate(shortlist):
            candidate.probe_rank = idx
        return shortlist

    target_count = max(1, probe_max_frames)
    num_bins = min(len(shortlist), target_count)
    sorted_candidates = sorted(shortlist, key=lambda item: item.source_frame_index)
    max_frame_index = max(candidate.source_frame_index for candidate in sorted_candidates)
    bin_edges = np.linspace(0, max(max_frame_index + 1, 1), num_bins + 1, dtype=np.int64)
    bins: list[list[CandidateFrame]] = [[] for _ in range(num_bins)]
    for candidate in sorted_candidates:
        bin_index = int(np.searchsorted(bin_edges, candidate.source_frame_index, side="right") - 1)
        bin_index = max(0, min(bin_index, num_bins - 1))
        bins[bin_index].append(candidate)

    capped: list[CandidateFrame] = []
    for bucket in bins:
        if not bucket:
            continue
        bucket.sort(key=lambda item: item.score, reverse=True)
        capped.append(bucket[0])

    if len(capped) < target_count:
        used = {candidate.source_frame_index for candidate in capped}
        for candidate in sorted(shortlist, key=lambda item: item.score, reverse=True):
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


def _scan_video_candidates(
    video_path: Path,
    scan_stride: int,
    analysis_size: tuple[int, int],
    orb_features: int,
    progress_callback: ProgressCallback = None,
) -> tuple[list[CandidateFrame], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    orb = cv2.ORB_create(nfeatures=orb_features)
    candidates: list[CandidateFrame] = []
    frame_idx = 0
    total_candidate_slots = max(1, total_frames // max(1, scan_stride))
    _emit(progress_callback, 10, "Scanning video for candidate frames")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, scan_stride) == 0:
            analysis_frame = cv2.resize(frame, analysis_size, interpolation=cv2.INTER_AREA)
            candidates.append(_candidate_metrics(analysis_frame, frame_idx, fps, orb))
            if len(candidates) % 25 == 0:
                percent = 10 + int(15 * min(len(candidates) / total_candidate_slots, 1.0))
                _emit(progress_callback, percent, "Scanning video for candidate frames")
        frame_idx += 1
    cap.release()

    metadata = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "original_resolution": (width, height),
        "analysis_resolution": analysis_size,
        "scan_stride": int(scan_stride),
        "candidate_frames": len(candidates),
    }
    return candidates, metadata


def _load_video_frames_by_index(
    video_path: Path,
    frame_indices: list[int],
    output_size: tuple[int, int],
    progress_callback: ProgressCallback = None,
    percent_base: int = 45,
    percent_span: int = 20,
    message: str = "Loading selected frames",
) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    frames: dict[int, np.ndarray] = {}
    for idx, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if ok:
            frames[int(frame_index)] = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        percent = percent_base + int(percent_span * ((idx + 1) / max(len(frame_indices), 1)))
        _emit(progress_callback, percent, message)
    cap.release()
    return frames


def _load_fast3r_probe_model(model_name: str, device: torch.device, max_parallel_views_for_head: int | None) -> Any:
    from fast3r.models.fast3r import Fast3R

    model = Fast3R.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    if max_parallel_views_for_head is not None and hasattr(model, "set_max_parallel_views_for_head"):
        model.set_max_parallel_views_for_head(int(max_parallel_views_for_head))
    return model


def _assign_geometry_quality_scores(probe_metrics: list[ProbeViewMetric]) -> None:
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


def _build_pair_metrics(probe_metrics: list[ProbeViewMetric], conf_masks: list[np.ndarray]) -> list[PairMetric]:
    if len(probe_metrics) < 2:
        return []

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
            raw_pairs.append(
                PairMetric(
                    source_frame_index_a=metric_a.source_frame_index,
                    source_frame_index_b=metric_b.source_frame_index,
                    probe_rank_a=metric_a.probe_rank,
                    probe_rank_b=metric_b.probe_rank,
                    camera_distance=float(np.linalg.norm(centers[idx_a] - centers[idx_b])),
                    view_angle_deg=_angle_degrees(forwards[idx_a], forwards[idx_b]),
                    confidence_iou=_confidence_iou(conf_masks[idx_a], conf_masks[idx_b]),
                )
            )

    distance_scores = _robust_normalize(np.array([item.camera_distance for item in raw_pairs], dtype=np.float64))
    angle_scores = _robust_normalize(np.array([item.view_angle_deg for item in raw_pairs], dtype=np.float64))
    overlap_target = 0.35
    overlap_scores = np.array(
        [max(0.0, 1.0 - abs(item.confidence_iou - overlap_target) / overlap_target) for item in raw_pairs],
        dtype=np.float64,
    )

    for idx, item in enumerate(raw_pairs):
        item.baseline_diversity_score = float(0.60 * distance_scores[idx] + 0.40 * angle_scores[idx])
        item.overlap_balance_score = float(overlap_scores[idx])

    return raw_pairs


def _pair_metric_lookup(pair_metrics: list[PairMetric]) -> dict[tuple[int, int], PairMetric]:
    lookup: dict[tuple[int, int], PairMetric] = {}
    for item in pair_metrics:
        lookup[(item.probe_rank_a, item.probe_rank_b)] = item
        lookup[(item.probe_rank_b, item.probe_rank_a)] = item
    return lookup


def _select_geometry_aware_subset(
    probe_metrics: list[ProbeViewMetric],
    pair_metrics: list[PairMetric],
    target_frames: int,
) -> list[SelectedFrameRecord]:
    if not probe_metrics:
        return []

    lookup = _pair_metric_lookup(pair_metrics)
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

            diversity = float(max((pair.baseline_diversity_score for pair in pair_candidates), default=0.0))
            overlap = float(max((pair.overlap_balance_score for pair in pair_candidates), default=0.0))
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


def _run_fast3r_probe(
    ordered_candidates: list[CandidateFrame],
    frames_by_index: dict[int, np.ndarray],
    model_name: str,
    image_size: int,
    dtype_name: str,
    niter_pnp: int,
    focal_method: str,
    confidence_threshold: float,
    max_parallel_views_for_head: int | None,
    progress_callback: ProgressCallback = None,
) -> tuple[list[ProbeViewMetric], list[PairMetric]]:
    from fast3r.dust3r.inference_multiview import inference
    from fast3r.dust3r.utils.image import load_images
    from src.pipeline.fast3r.reconstruction import estimate_camera_poses_simple, fast3r_precision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _emit(progress_callback, 40, "Loading Fast3R probe model")
    model = _load_fast3r_probe_model(model_name, device, max_parallel_views_for_head)

    temp_dir = Path(tempfile.mkdtemp(prefix="fast3r_geometry_probe_"))
    image_paths: list[str] = []
    for candidate in ordered_candidates:
        frame = frames_by_index[candidate.source_frame_index]
        frame_path = temp_dir / f"probe_{candidate.probe_rank:03d}_src_{candidate.source_frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        image_paths.append(str(frame_path))

    images = load_images(image_paths, size=image_size, verbose=False)
    precision = fast3r_precision(dtype_name)

    _emit(progress_callback, 48, "Running low-resolution Fast3R probe")
    output_dict, _ = inference(images, model, device, dtype=precision, verbose=False, profiling=True)

    _emit(progress_callback, 55, "Estimating probe camera poses")
    preds_gpu: list[dict[str, Any]] = []
    for pred in output_dict["preds"]:
        pred_gpu = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in pred.items()}
        preds_gpu.append(pred_gpu)

    poses_c2w_batch, estimated_focals_batch = estimate_camera_poses_simple(
        preds_gpu,
        niter_pnp=niter_pnp,
        focal_length_estimation_method=focal_method,
    )
    poses = poses_c2w_batch[0]
    focals = estimated_focals_batch[0]

    probe_metrics: list[ProbeViewMetric] = []
    conf_masks: list[np.ndarray] = []
    for probe_rank, candidate in enumerate(ordered_candidates):
        pred = output_dict["preds"][probe_rank]
        pts3d = pred["pts3d_in_other_view"]
        if isinstance(pts3d, torch.Tensor):
            pts3d = pts3d.detach().cpu().numpy()
        pts3d = pts3d[0]

        conf = pred.get("conf")
        if isinstance(conf, torch.Tensor):
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
        forward = _normalize_vector(pose[:3, 2] if pose.shape == (4, 4) else np.zeros(3, dtype=np.float32))

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
                visual_score=float(candidate.score),
            )
        )

    _assign_geometry_quality_scores(probe_metrics)
    pair_metrics = _build_pair_metrics(probe_metrics, conf_masks)

    shutil.rmtree(temp_dir, ignore_errors=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return probe_metrics, pair_metrics


def select_frames_from_video(
    video_path: Path,
    target_frames: int,
    scan_stride: int,
    shortlist_multiplier: int,
    dedupe_similarity: float,
    min_frame_gap: int,
    analysis_size: tuple[int, int],
    output_size: tuple[int, int],
    orb_features: int,
    progress_callback: ProgressCallback = None,
    selection_mode: str = "prefilter",
    visual_shortlist_target: int | None = None,
    probe_max_frames: int | None = None,
    probe_image_size: int = 256,
    probe_dtype: str = "float32",
    probe_confidence_threshold: float = 1.0,
    probe_pnp_iters: int = 50,
    probe_focal_method: str = "first_view_from_global_head",
    probe_model_name: str = "jedyang97/Fast3R_ViT_Large_512",
    probe_max_parallel_views: int | None = 16,
) -> tuple[list[np.ndarray], dict]:
    video_path = Path(video_path)
    if selection_mode not in {"prefilter", "geometry_aware"}:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    requested_min_frame_gap = int(min_frame_gap)
    shortlist_target = max(int(visual_shortlist_target or target_frames), int(target_frames))
    probe_cap = probe_max_frames if probe_max_frames is None else max(int(probe_max_frames), int(target_frames))

    candidates, metadata = _scan_video_candidates(
        video_path=video_path,
        scan_stride=scan_stride,
        analysis_size=analysis_size,
        orb_features=orb_features,
        progress_callback=progress_callback,
    )
    if not candidates:
        raise RuntimeError("No candidate frames were extracted from the video.")

    effective_gap, max_gap_for_target = _effective_min_frame_gap(
        total_frames=int(metadata.get("total_frames", 0)),
        target_frames=int(target_frames),
        requested_gap=requested_min_frame_gap,
        scan_stride=int(scan_stride),
    )

    _emit(progress_callback, 28, "Scoring candidate frames")
    _score_candidates(candidates)
    shortlist = _shortlist_candidates(
        candidates=candidates,
        shortlist_target=shortlist_target,
        shortlist_multiplier=shortlist_multiplier,
        dedupe_similarity=dedupe_similarity,
        min_frame_gap=effective_gap,
    )
    if not shortlist:
        raise RuntimeError("Visual prefilter produced an empty shortlist.")

    warnings: list[str] = []
    if effective_gap < requested_min_frame_gap:
        warnings.append(
            "Requested minimum frame gap was too large for the target frame count; "
            f"using a {effective_gap}-frame gap instead of {requested_min_frame_gap} for this run."
        )

    if selection_mode == "prefilter":
        _emit(progress_callback, 35, "Selecting diverse frames across the video")
        selected = _select_visual_prefilter_frames(shortlist, target_frames, dedupe_similarity, effective_gap)
        selected_indices = [candidate.source_frame_index for candidate in selected]
        frames_by_index = _load_video_frames_by_index(
            video_path=video_path,
            frame_indices=selected_indices,
            output_size=output_size,
            progress_callback=progress_callback,
            percent_base=45,
            percent_span=20,
            message="Loading selected frames",
        )
        ordered_indices = [frame_index for frame_index in selected_indices if frame_index in frames_by_index]
        frames = [frames_by_index[frame_index] for frame_index in ordered_indices]
        selected_scores = [candidate.score for candidate in selected if candidate.source_frame_index in frames_by_index]
        metadata.update(
            {
                "target_resolution": output_size,
                "shortlist_size": len(shortlist),
                "target_frame_count": int(target_frames),
                "requested_min_frame_gap": requested_min_frame_gap,
                "effective_min_frame_gap": effective_gap,
                "max_gap_for_target_frames": max_gap_for_target,
                "selected_frame_count": len(frames),
                "selected_frame_indices": ordered_indices,
                "selected_frame_scores": selected_scores,
                "selection_mode": "prefilter",
            }
        )
    else:
        _emit(progress_callback, 35, "Building geometry-aware probe pool")
        probe_pool = _cap_probe_pool(shortlist, probe_cap)
        probe_frame_indices = [candidate.source_frame_index for candidate in probe_pool]
        probe_frames_by_index = _load_video_frames_by_index(
            video_path=video_path,
            frame_indices=probe_frame_indices,
            output_size=output_size,
            progress_callback=progress_callback,
            percent_base=36,
            percent_span=8,
            message="Loading probe frames",
        )
        probe_pool = [candidate for candidate in probe_pool if candidate.source_frame_index in probe_frames_by_index]
        if not probe_pool:
            raise RuntimeError("Could not load any shortlisted frames for the geometry-aware probe.")

        probe_metrics, pair_metrics = _run_fast3r_probe(
            ordered_candidates=probe_pool,
            frames_by_index=probe_frames_by_index,
            model_name=probe_model_name,
            image_size=int(probe_image_size),
            dtype_name=str(probe_dtype),
            niter_pnp=int(probe_pnp_iters),
            focal_method=str(probe_focal_method),
            confidence_threshold=float(probe_confidence_threshold),
            max_parallel_views_for_head=probe_max_parallel_views,
            progress_callback=progress_callback,
        )

        _emit(progress_callback, 62, "Pruning frames with geometry-aware scoring")
        selected_records = _select_geometry_aware_subset(probe_metrics, pair_metrics, target_frames)
        greedy_indices = [item.source_frame_index for item in selected_records]
        selected_records = sorted(selected_records, key=lambda item: item.source_frame_index)
        selected_indices = [item.source_frame_index for item in selected_records]
        frames = [probe_frames_by_index[frame_index] for frame_index in selected_indices if frame_index in probe_frames_by_index]

        probe_pose_valid_count = int(sum(1 for item in probe_metrics if item.pose_valid))
        if len(probe_pool) <= target_frames:
            warnings.append("Geometry-aware stage reordered the visual shortlist but had little room to prune it further.")
        if probe_pose_valid_count < max(1, len(probe_metrics) // 2):
            warnings.append("Many probe poses were invalid; geometry-aware scoring may be unstable on this video.")
        if selected_records and min(item.confidence_ratio_above_threshold for item in selected_records) < 0.70:
            warnings.append("Some geometry-selected frames have weak confidence coverage; inspect the final frame set.")

        metadata.update(
            {
                "target_resolution": output_size,
                "shortlist_size": len(shortlist),
                "target_frame_count": int(target_frames),
                "requested_min_frame_gap": requested_min_frame_gap,
                "effective_min_frame_gap": effective_gap,
                "max_gap_for_target_frames": max_gap_for_target,
                "probe_pool_count": len(probe_metrics),
                "probe_pose_valid_count": probe_pose_valid_count,
                "probe_pose_valid_ratio": _safe_div(probe_pose_valid_count, len(probe_metrics)),
                "selected_frame_count": len(frames),
                "selected_frame_indices": selected_indices,
                "geometry_greedy_selection_indices": greedy_indices,
                "selected_frame_scores": [item.final_selection_score for item in selected_records],
                "selected_frame_visual_scores": [item.visual_score for item in selected_records],
                "selected_frame_geometry_scores": [item.geometry_quality_score for item in selected_records],
                "selection_mode": "geometry_aware",
                "visual_shortlist_target": shortlist_target,
                "probe_max_frames": probe_cap,
                "probe_image_size": int(probe_image_size),
                "probe_dtype": str(probe_dtype),
                "probe_confidence_threshold": float(probe_confidence_threshold),
                "probe_pnp_iters": int(probe_pnp_iters),
                "probe_focal_method": str(probe_focal_method),
                "probe_model_name": str(probe_model_name),
            }
        )

    if len(metadata.get("selected_frame_indices", [])) < int(target_frames):
        warnings.append(
            f"Selected {len(metadata.get('selected_frame_indices', []))} frames out of requested {int(target_frames)}; "
            "consider lowering the frame gap, lowering dedupe strictness, or increasing the visual shortlist."
        )

    chronological_indices = sorted(metadata["selected_frame_indices"])
    gaps = [
        chronological_indices[idx + 1] - chronological_indices[idx]
        for idx in range(len(chronological_indices) - 1)
    ]
    if metadata.get("selected_frame_scores") and float(np.median(metadata["selected_frame_scores"])) < 0.55:
        warnings.append("Selected frame quality is weak; the source video may be blurry or poorly lit.")
    if gaps and effective_gap > 0 and min(gaps) < max(15, effective_gap // 2):
        warnings.append("Some selected views are still close together; consider increasing min_frame_gap.")

    metadata["min_selected_gap_frames"] = min(gaps) if gaps else 0
    metadata["median_selected_gap_frames"] = int(np.median(gaps)) if gaps else 0
    metadata["warnings"] = warnings
    return frames, metadata
