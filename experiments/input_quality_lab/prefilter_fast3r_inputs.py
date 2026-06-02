"""Order-agnostic frame prefilter experiment for Fast3R inputs.

This experiment scans a long video with a coarse stride, scores candidate frames
for sharpness/texture/exposure, keeps the strongest candidates across the full
video span, and exports a curated subset of frames plus a manifest.

The selection logic is intentionally order-agnostic at the final stage: Fast3R
consumes a set of views in one forward pass, so we experiment with informative
frame sets instead of relying on strict temporal adjacency.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = THIS_DIR / "results"


@dataclass
class CandidateFrame:
    source_frame_index: int
    timestamp_seconds: float
    laplacian_variance: float
    entropy_bits: float
    brightness_mean: float
    brightness_std: float
    edge_density: float
    underexposed_ratio: float
    overexposed_ratio: float
    clipped_ratio: float
    orb_keypoints: int
    thumbnail_vector: np.ndarray
    bin_index: int = -1
    sharpness_score: float = 0.0
    texture_score: float = 0.0
    entropy_score: float = 0.0
    exposure_score: float = 0.0
    brightness_score: float = 0.0
    prefilter_score: float = 0.0
    selection_rank: int = 0
    selection_reason: str = ""


def log(message: str) -> None:
    print(message, flush=True)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def grayscale_entropy(gray_frame: np.ndarray) -> float:
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(hist.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def robust_normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.float32)

    values = values.astype(np.float64, copy=False)
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        normalized = np.full(values.shape, 0.5, dtype=np.float64)
    else:
        normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    if invert:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32)


def compute_candidate_metrics(
    frame_bgr: np.ndarray,
    source_frame_index: int,
    fps: float,
    orb: cv2.ORB,
) -> CandidateFrame:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy_bits = grayscale_entropy(gray)
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))
    underexposed_ratio = float(np.mean(gray <= 10))
    overexposed_ratio = float(np.mean(gray >= 245))
    clipped_ratio = float(underexposed_ratio + overexposed_ratio)

    keypoints = orb.detect(gray, None)
    orb_keypoints = len(keypoints)

    thumb = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    thumb = thumb.reshape(-1)
    thumb -= thumb.mean()
    thumb_norm = float(np.linalg.norm(thumb))
    if thumb_norm > 0:
        thumb /= thumb_norm

    return CandidateFrame(
        source_frame_index=int(source_frame_index),
        timestamp_seconds=safe_div(source_frame_index, fps),
        laplacian_variance=laplacian_variance,
        entropy_bits=entropy_bits,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        edge_density=edge_density,
        underexposed_ratio=underexposed_ratio,
        overexposed_ratio=overexposed_ratio,
        clipped_ratio=clipped_ratio,
        orb_keypoints=int(orb_keypoints),
        thumbnail_vector=thumb.astype(np.float32),
    )


def score_candidates(candidates: list[CandidateFrame]) -> None:
    if not candidates:
        return

    sharpness = np.log1p(np.array([c.laplacian_variance for c in candidates], dtype=np.float64))
    texture = np.log1p(np.array([c.orb_keypoints for c in candidates], dtype=np.float64))
    entropy = np.array([c.entropy_bits for c in candidates], dtype=np.float64)
    clipped = np.array([c.clipped_ratio for c in candidates], dtype=np.float64)
    brightness_deviation = np.abs(
        np.array([c.brightness_mean for c in candidates], dtype=np.float64) - 127.5
    )

    sharpness_score = robust_normalize(sharpness)
    texture_score = robust_normalize(texture)
    entropy_score = robust_normalize(entropy)
    exposure_score = robust_normalize(clipped, invert=True)
    brightness_score = robust_normalize(brightness_deviation, invert=True)

    for idx, candidate in enumerate(candidates):
        candidate.sharpness_score = float(sharpness_score[idx])
        candidate.texture_score = float(texture_score[idx])
        candidate.entropy_score = float(entropy_score[idx])
        candidate.exposure_score = float(exposure_score[idx])
        candidate.brightness_score = float(brightness_score[idx])
        candidate.prefilter_score = float(
            0.40 * candidate.sharpness_score
            + 0.25 * candidate.texture_score
            + 0.15 * candidate.entropy_score
            + 0.10 * candidate.exposure_score
            + 0.10 * candidate.brightness_score
        )


def shortlist_candidates(
    candidates: list[CandidateFrame],
    target_frames: int,
    shortlist_multiplier: int,
) -> list[CandidateFrame]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda item: item.source_frame_index)
    num_bins = min(len(sorted_candidates), max(1, target_frames))
    if num_bins <= 0:
        return []

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
        bucket.sort(key=lambda item: item.prefilter_score, reverse=True)
        bucket.sort(
            key=lambda item: (
                item.prefilter_score
                - 0.12 * min(abs(item.source_frame_index - bin_center) / bin_half_width, 1.0),
                item.prefilter_score,
            ),
            reverse=True,
        )
        shortlist.extend(bucket[:per_bin_keep])

    shortlist.sort(key=lambda item: item.prefilter_score, reverse=True)
    return shortlist


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def candidate_is_valid(
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
        if enforce_similarity:
            similarity = cosine_similarity(candidate.thumbnail_vector, existing.thumbnail_vector)
            if similarity >= dedupe_similarity:
                return False
    return True


def select_diverse_frames(
    shortlist: list[CandidateFrame],
    target_frames: int,
    dedupe_similarity: float,
    min_frame_gap: int,
) -> list[CandidateFrame]:
    if not shortlist:
        return []

    selected: list[CandidateFrame] = []
    selected_indices: set[int] = set()
    grouped_by_bin: dict[int, list[CandidateFrame]] = {}

    for candidate in shortlist:
        grouped_by_bin.setdefault(candidate.bin_index, []).append(candidate)

    for bucket in grouped_by_bin.values():
        bucket.sort(key=lambda item: item.prefilter_score, reverse=True)

    ordered_bins = sorted(grouped_by_bin.keys())

    # Pass 1: choose one strong representative per coverage bin.
    for bin_index in ordered_bins:
        if len(selected) >= target_frames:
            break
        bucket = grouped_by_bin[bin_index]

        chosen = None
        reason = ""

        for candidate in bucket:
            if candidate.source_frame_index in selected_indices:
                continue
            if candidate_is_valid(candidate, selected, dedupe_similarity, min_frame_gap, True, True):
                chosen = candidate
                reason = "coverage_bin_strict"
                break

        if chosen is None:
            for candidate in bucket:
                if candidate.source_frame_index in selected_indices:
                    continue
                if candidate_is_valid(candidate, selected, dedupe_similarity, min_frame_gap, False, True):
                    chosen = candidate
                    reason = "coverage_bin_similarity_only"
                    break

        if chosen is None:
            for candidate in bucket:
                if candidate.source_frame_index in selected_indices:
                    continue
                chosen = candidate
                reason = "coverage_bin_fallback"
                break

        if chosen is None:
            continue

        chosen.selection_reason = reason
        selected.append(chosen)
        selected_indices.add(chosen.source_frame_index)

    # Pass 2: fill remaining slots with global best-scoring candidates.
    for candidate in shortlist:
        if len(selected) >= target_frames:
            break
        if candidate.source_frame_index in selected_indices:
            continue

        if candidate_is_valid(candidate, selected, dedupe_similarity, min_frame_gap, True, True):
            candidate.selection_reason = "score_plus_diversity"
            selected.append(candidate)
            selected_indices.add(candidate.source_frame_index)

    if len(selected) < target_frames:
        for candidate in shortlist:
            if len(selected) >= target_frames:
                break
            if candidate.source_frame_index in selected_indices:
                continue
            if candidate_is_valid(candidate, selected, dedupe_similarity, min_frame_gap, False, True):
                candidate.selection_reason = "score_fill_after_gap_relax"
                selected.append(candidate)
                selected_indices.add(candidate.source_frame_index)

    if len(selected) < target_frames:
        for candidate in shortlist:
            if len(selected) >= target_frames:
                break
            if candidate.source_frame_index in selected_indices:
                continue
            candidate.selection_reason = "score_fill_after_dedupe"
            selected.append(candidate)
            selected_indices.add(candidate.source_frame_index)

    for rank, candidate in enumerate(selected, start=1):
        candidate.selection_rank = rank
        if not candidate.selection_reason:
            candidate.selection_reason = "score_only"

    return selected[:target_frames]


def reorder_selected_frames(
    selected: list[CandidateFrame],
    output_order: str,
    seed: int,
) -> list[CandidateFrame]:
    if output_order == "source":
        return sorted(selected, key=lambda item: item.source_frame_index)
    if output_order == "shuffle":
        rng = np.random.default_rng(seed)
        shuffled = list(selected)
        rng.shuffle(shuffled)
        return shuffled
    return sorted(selected, key=lambda item: item.prefilter_score, reverse=True)


def scan_video_candidates(
    video_path: Path,
    scan_stride: int,
    max_candidates: int,
    analysis_size: tuple[int, int],
    orb_features: int,
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

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, scan_stride) == 0:
            analysis_frame = cv2.resize(frame, analysis_size, interpolation=cv2.INTER_AREA)
            candidates.append(
                compute_candidate_metrics(
                    analysis_frame,
                    source_frame_index=frame_idx,
                    fps=fps,
                    orb=orb,
                )
            )
            if max_candidates > 0 and len(candidates) >= max_candidates:
                break
        frame_idx += 1

    cap.release()

    video_info = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "analysis_width": int(analysis_size[0]),
        "analysis_height": int(analysis_size[1]),
        "scan_stride": int(scan_stride),
        "candidate_frames": int(len(candidates)),
    }
    return candidates, video_info


def fetch_and_write_selected_frames(
    video_path: Path,
    selected: list[CandidateFrame],
    output_dir: Path,
    target_size: tuple[int, int],
) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video for export: {video_path}")

    frames_dir = output_dir / "selected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    exported_rows: list[dict] = []

    for output_position, candidate in enumerate(selected, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.source_frame_index)
        ok, frame = cap.read()
        if not ok:
            continue

        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        filename = (
            f"frame_{output_position:04d}_src_{candidate.source_frame_index:06d}"
            f"_score_{candidate.prefilter_score:.3f}.jpg"
        )
        frame_path = frames_dir / filename
        cv2.imwrite(str(frame_path), resized)

        exported_rows.append(
            {
                "selection_rank": int(output_position),
                "source_frame_index": int(candidate.source_frame_index),
                "timestamp_seconds": float(candidate.timestamp_seconds),
                "bin_index": int(candidate.bin_index),
                "prefilter_score": float(candidate.prefilter_score),
                "sharpness_score": float(candidate.sharpness_score),
                "texture_score": float(candidate.texture_score),
                "entropy_score": float(candidate.entropy_score),
                "exposure_score": float(candidate.exposure_score),
                "brightness_score": float(candidate.brightness_score),
                "laplacian_variance": float(candidate.laplacian_variance),
                "orb_keypoints": int(candidate.orb_keypoints),
                "clipped_ratio": float(candidate.clipped_ratio),
                "selection_reason": candidate.selection_reason,
                "output_frame_path": str(frame_path),
            }
        )

    cap.release()
    return exported_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(path: Path, report: dict) -> None:
    lines: list[str] = []
    lines.append("Fast3R Input Prefilter Experiment")
    lines.append("=" * 80)
    lines.append(f"timestamp: {report['timestamp']}")
    lines.append(f"video_path: {report['video']['video_path']}")
    lines.append(f"fps: {report['video']['fps']}")
    lines.append(f"total_frames: {report['video']['total_frames']}")
    lines.append(f"candidate_frames_scanned: {report['video']['candidate_frames']}")
    lines.append(f"target_frames: {report['selection']['target_frames']}")
    lines.append(f"frames_exported: {report['selection']['frames_exported']}")
    lines.append(f"output_order: {report['selection']['output_order']}")
    lines.append("")
    lines.append("Selection Settings")
    lines.append("-" * 80)
    for key, value in report["settings"].items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Selection Stats")
    lines.append("-" * 80)
    for key, value in report["selection"].items():
        if key == "top_selected_source_indices":
            continue
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Top Selected Source Indices")
    lines.append("-" * 80)
    lines.append(", ".join(str(v) for v in report["selection"]["top_selected_source_indices"]))
    lines.append("")
    lines.append("Artifacts")
    lines.append("-" * 80)
    for key, value in report["artifacts"].items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment with order-agnostic Fast3R frame prefiltering.")
    parser.add_argument("--video", type=Path, required=True, help="Source video to scan.")
    parser.add_argument("--target-frames", type=int, default=80, help="How many frames to export.")
    parser.add_argument(
        "--scan-stride",
        type=int,
        default=15,
        help="Analyze every N-th video frame as a candidate.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional hard cap on candidate frames after scan-stride sampling.",
    )
    parser.add_argument(
        "--shortlist-multiplier",
        type=int,
        default=3,
        help="How many coverage bins to use relative to target-frames before final selection.",
    )
    parser.add_argument(
        "--dedupe-similarity",
        type=float,
        default=0.985,
        help="Cosine-similarity threshold for thumbnail-based near-duplicate rejection.",
    )
    parser.add_argument(
        "--min-frame-gap",
        type=int,
        default=120,
        help="Preferred minimum source-frame distance between selected frames.",
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=384,
        help="Width used for scoring candidate frames.",
    )
    parser.add_argument(
        "--analysis-height",
        type=int,
        default=288,
        help="Height used for scoring candidate frames.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=1024,
        help="Export width for selected frames.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=768,
        help="Export height for selected frames.",
    )
    parser.add_argument("--orb-features", type=int, default=1000, help="ORB features for texture scoring.")
    parser.add_argument(
        "--output-order",
        choices=["score", "source", "shuffle"],
        default="score",
        help="How to order exported frames after selection.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used when output-order=shuffle.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"{stamp}_fast3r_prefilter"
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_size = (int(args.analysis_width), int(args.analysis_height))
    output_size = (int(args.output_width), int(args.output_height))

    log("[1/5] Scanning candidate frames")
    candidates, video_info = scan_video_candidates(
        video_path=args.video,
        scan_stride=args.scan_stride,
        max_candidates=args.max_candidates,
        analysis_size=analysis_size,
        orb_features=args.orb_features,
    )
    if not candidates:
        raise RuntimeError("No candidate frames were extracted from the video.")
    log(f"  candidates scanned: {len(candidates)}")

    log("[2/5] Scoring candidates")
    score_candidates(candidates)

    log("[3/5] Building shortlist and selecting diverse frames")
    shortlist = shortlist_candidates(
        candidates=candidates,
        target_frames=args.target_frames,
        shortlist_multiplier=args.shortlist_multiplier,
    )
    selected = select_diverse_frames(
        shortlist=shortlist,
        target_frames=args.target_frames,
        dedupe_similarity=args.dedupe_similarity,
        min_frame_gap=args.min_frame_gap,
    )
    selected = reorder_selected_frames(selected, args.output_order, args.seed)
    log(f"  shortlist size: {len(shortlist)}")
    log(f"  selected frames: {len(selected)}")

    log("[4/5] Exporting selected frames and manifests")
    manifest_rows = fetch_and_write_selected_frames(
        video_path=args.video,
        selected=selected,
        output_dir=out_dir,
        target_size=output_size,
    )
    if not manifest_rows:
        raise RuntimeError("No frames were exported from the selected candidate set.")

    candidate_rows = [
        {
            "source_frame_index": int(candidate.source_frame_index),
            "timestamp_seconds": float(candidate.timestamp_seconds),
            "bin_index": int(candidate.bin_index),
            "prefilter_score": float(candidate.prefilter_score),
            "sharpness_score": float(candidate.sharpness_score),
            "texture_score": float(candidate.texture_score),
            "entropy_score": float(candidate.entropy_score),
            "exposure_score": float(candidate.exposure_score),
            "brightness_score": float(candidate.brightness_score),
            "laplacian_variance": float(candidate.laplacian_variance),
            "entropy_bits": float(candidate.entropy_bits),
            "brightness_mean": float(candidate.brightness_mean),
            "brightness_std": float(candidate.brightness_std),
            "edge_density": float(candidate.edge_density),
            "underexposed_ratio": float(candidate.underexposed_ratio),
            "overexposed_ratio": float(candidate.overexposed_ratio),
            "clipped_ratio": float(candidate.clipped_ratio),
            "orb_keypoints": int(candidate.orb_keypoints),
        }
        for candidate in sorted(candidates, key=lambda item: item.prefilter_score, reverse=True)
    ]

    candidate_csv = out_dir / "candidate_frame_scores.csv"
    manifest_csv = out_dir / "selected_frame_manifest.csv"
    write_csv(candidate_csv, candidate_rows)
    write_csv(manifest_csv, manifest_rows)

    selected_source_indices = [int(row["source_frame_index"]) for row in manifest_rows]
    selected_bins = [int(row["bin_index"]) for row in manifest_rows]
    selected_gaps = [
        selected_source_indices[idx + 1] - selected_source_indices[idx]
        for idx in range(len(selected_source_indices) - 1)
    ]

    report = {
        "timestamp": stamp,
        "video": video_info,
        "settings": {
            "target_frames": int(args.target_frames),
            "scan_stride": int(args.scan_stride),
            "max_candidates": int(args.max_candidates),
            "shortlist_multiplier": int(args.shortlist_multiplier),
            "dedupe_similarity": float(args.dedupe_similarity),
            "min_frame_gap": int(args.min_frame_gap),
            "analysis_width": int(args.analysis_width),
            "analysis_height": int(args.analysis_height),
            "output_width": int(args.output_width),
            "output_height": int(args.output_height),
            "orb_features": int(args.orb_features),
            "output_order": args.output_order,
            "seed": int(args.seed),
        },
        "selection": {
            "target_frames": int(args.target_frames),
            "frames_exported": int(len(manifest_rows)),
            "shortlist_size": int(len(shortlist)),
            "output_order": args.output_order,
            "mean_selected_score": float(np.mean([row["prefilter_score"] for row in manifest_rows])),
            "median_selected_score": float(np.median([row["prefilter_score"] for row in manifest_rows])),
            "unique_bins_selected": int(len(set(selected_bins))),
            "min_selected_gap_frames": int(min(selected_gaps)) if selected_gaps else 0,
            "median_selected_gap_frames": int(np.median(selected_gaps)) if selected_gaps else 0,
            "max_selected_gap_frames": int(max(selected_gaps)) if selected_gaps else 0,
            "top_selected_source_indices": [int(row["source_frame_index"]) for row in manifest_rows[:20]],
        },
        "artifacts": {
            "candidate_frame_scores_csv": str(candidate_csv),
            "selected_frame_manifest_csv": str(manifest_csv),
            "selected_frames_dir": str(out_dir / "selected_frames"),
            "summary_json": str(out_dir / "prefilter_summary.json"),
            "summary_txt": str(out_dir / "prefilter_summary.txt"),
        },
    }

    summary_json = out_dir / "prefilter_summary.json"
    summary_txt = out_dir / "prefilter_summary.txt"
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_text_summary(summary_txt, report)

    log("[5/5] Done")
    log(f"  selected frame manifest: {manifest_csv}")
    log(f"  selected frames dir: {out_dir / 'selected_frames'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
