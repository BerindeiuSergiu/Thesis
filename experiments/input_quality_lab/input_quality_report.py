"""Input quality diagnostics for 3D reconstruction experiments.

This module helps inspect what is fed into Fast3R/DUSt3R before reconstruction:
- blur / sharpness
- exposure quality
- temporal overlap / motion continuity
- optional camera-trajectory coverage (when poses.npy is available)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = THIS_DIR.parent
BETA_ROOT = EXPERIMENTS_ROOT / "beta_pipeline_testing"
OUTPUTS_ROOT = BETA_ROOT / "outputs"
DEFAULT_RESULTS_ROOT = THIS_DIR / "results"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class InputBundle:
    frames: list[np.ndarray]
    source_indices: list[int]
    fps: float
    source_type: str
    source_path: str


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def default_frames_dir_for_model(model: str) -> Path:
    return OUTPUTS_ROOT / model / "frames"


def default_poses_path_for_model(model: str) -> Path:
    return OUTPUTS_ROOT / model / "poses.npy"


def extract_index_from_name(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    return fallback


def list_frame_paths(frames_dir: Path) -> list[Path]:
    paths = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths, key=lambda p: p.name)


def load_frames_from_directory(frames_dir: Path, max_frames: int, stride: int) -> InputBundle:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    all_paths = list_frame_paths(frames_dir)
    if not all_paths:
        raise FileNotFoundError(f"No image frames found in: {frames_dir}")

    selected_paths = all_paths[:: max(1, stride)]
    if max_frames > 0:
        selected_paths = selected_paths[:max_frames]

    frames: list[np.ndarray] = []
    source_indices: list[int] = []
    for idx, path in enumerate(selected_paths):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frames.append(frame)
        source_indices.append(extract_index_from_name(path, idx))

    if not frames:
        raise RuntimeError("Could not load any frames from the selected directory.")

    return InputBundle(
        frames=frames,
        source_indices=source_indices,
        fps=0.0,
        source_type="frames_dir",
        source_path=str(frames_dir),
    )


def load_frames_from_video(video_path: Path, max_frames: int, stride: int) -> InputBundle:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    source_indices: list[int] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, stride) == 0:
            frames.append(frame)
            source_indices.append(frame_idx)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if not frames:
        raise RuntimeError("Could not extract frames from video.")

    return InputBundle(
        frames=frames,
        source_indices=source_indices,
        fps=fps,
        source_type="video",
        source_path=str(video_path),
    )


def grayscale_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probs = hist / max(hist.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def compute_frame_and_pair_metrics(
    frames: list[np.ndarray],
    source_indices: list[int],
    fps: float,
    orb_features: int,
    underexposed_threshold: int,
    overexposed_threshold: int,
) -> tuple[list[dict], list[dict]]:
    orb = cv2.ORB_create(nfeatures=orb_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    frame_rows: list[dict] = []
    pair_rows: list[dict] = []

    grays: list[np.ndarray] = []
    descriptors: list[Optional[np.ndarray]] = []
    keypoint_counts: list[int] = []

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(gray)

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_mean = float(gray.mean())
        brightness_std = float(gray.std())
        entropy = grayscale_entropy(gray)
        p05 = float(np.percentile(gray, 5))
        p95 = float(np.percentile(gray, 95))
        dynamic_range = float(p95 - p05)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        under_ratio = float(np.mean(gray <= underexposed_threshold))
        over_ratio = float(np.mean(gray >= overexposed_threshold))
        clipped_ratio = float(under_ratio + over_ratio)

        kps, desc = orb.detectAndCompute(gray, None)
        descriptors.append(desc)
        keypoints = int(len(kps))
        keypoint_counts.append(keypoints)

        src_idx = source_indices[i] if i < len(source_indices) else i
        frame_rows.append(
            {
                "frame_local_index": i,
                "source_frame_index": int(src_idx),
                "timestamp_seconds": safe_div(src_idx, fps),
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
                "brightness_mean": brightness_mean,
                "brightness_std": brightness_std,
                "entropy_bits": entropy,
                "dynamic_range_p95_p05": dynamic_range,
                "laplacian_variance": lap_var,
                "edge_density": edge_density,
                "underexposed_ratio": under_ratio,
                "overexposed_ratio": over_ratio,
                "clipped_ratio": clipped_ratio,
                "orb_keypoints": keypoints,
            }
        )

    for i in range(max(0, len(frames) - 1)):
        gray_a = grays[i]
        gray_b = grays[i + 1]
        diff = cv2.absdiff(gray_a, gray_b)
        mean_abs_diff = float(diff.mean())
        normalized_abs_diff = safe_div(mean_abs_diff, 255.0)

        desc_a = descriptors[i]
        desc_b = descriptors[i + 1]
        matches = []
        if desc_a is not None and desc_b is not None and len(desc_a) > 0 and len(desc_b) > 0:
            matches = matcher.match(desc_a, desc_b)

        dist = np.array([m.distance for m in matches], dtype=np.float64)
        src_a = source_indices[i] if i < len(source_indices) else i
        src_b = source_indices[i + 1] if (i + 1) < len(source_indices) else i + 1
        pair_rows.append(
            {
                "frame_local_index_a": i,
                "frame_local_index_b": i + 1,
                "source_frame_index_a": int(src_a),
                "source_frame_index_b": int(src_b),
                "time_gap_seconds": safe_div(abs(src_b - src_a), fps),
                "mean_abs_diff": mean_abs_diff,
                "normalized_abs_diff": normalized_abs_diff,
                "orb_matches": int(len(matches)),
                "orb_match_ratio_to_keypoints": safe_div(
                    len(matches),
                    max(1, min(keypoint_counts[i], keypoint_counts[i + 1])),
                ),
                "match_distance_mean": float(dist.mean()) if dist.size else 0.0,
                "match_distance_median": float(np.median(dist)) if dist.size else 0.0,
                "match_distance_p95": float(np.percentile(dist, 95)) if dist.size else 0.0,
            }
        )

    return frame_rows, pair_rows


def load_poses(poses_path: Path) -> np.ndarray:
    poses = np.load(poses_path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shape (N,4,4), got {poses.shape}")
    return poses.astype(np.float32, copy=False)


def extract_camera_centers_auto(poses: np.ndarray, ref_points: np.ndarray) -> tuple[np.ndarray, str]:
    centers_t = poses[:, :3, 3]
    r = poses[:, :3, :3]
    t = poses[:, :3, 3]
    centers_inv = -np.einsum("nij,nj->ni", np.transpose(r, (0, 2, 1)), t)

    ref_center = np.median(ref_points, axis=0) if ref_points.size else np.zeros(3, dtype=np.float32)
    med_t = float(np.median(np.linalg.norm(centers_t - ref_center, axis=1)))
    med_inv = float(np.median(np.linalg.norm(centers_inv - ref_center, axis=1)))
    if med_t <= med_inv:
        return centers_t.astype(np.float32), "translation_column"
    return centers_inv.astype(np.float32), "inverse_extrinsic"


def select_pose_indices(total_poses: int, source_indices: list[int], n_frames: int) -> tuple[np.ndarray, str]:
    if n_frames <= 0:
        return np.array([], dtype=np.int64), "none"
    if len(source_indices) == n_frames and source_indices and max(source_indices) < total_poses:
        return np.array(source_indices, dtype=np.int64), "source_frame_index"
    if total_poses == n_frames:
        return np.arange(n_frames, dtype=np.int64), "direct_match"
    if total_poses < n_frames:
        return np.arange(total_poses, dtype=np.int64), "pose_count_lt_frame_count"
    idx = np.linspace(0, max(0, total_poses - 1), n_frames, dtype=np.int64)
    return idx, "linspace_resample"


def pose_coverage_metrics(
    poses: np.ndarray,
    source_indices: list[int],
    n_frames: int,
) -> dict:
    centers_all, convention = extract_camera_centers_auto(poses, poses[:, :3, 3])
    pose_idx, mapping_mode = select_pose_indices(len(centers_all), source_indices, n_frames)
    centers = centers_all[pose_idx] if pose_idx.size else centers_all

    if centers.shape[0] < 2:
        return {
            "poses_loaded": True,
            "num_poses_total": int(len(centers_all)),
            "num_poses_used": int(len(centers)),
            "pose_convention_selected": convention,
            "pose_mapping_mode": mapping_mode,
            "pose_path_length": 0.0,
            "pose_bbox_extent_x": 0.0,
            "pose_bbox_extent_y": 0.0,
            "pose_bbox_extent_z": 0.0,
            "pose_bbox_diagonal": 0.0,
            "pose_step_median": 0.0,
            "pose_step_p95": 0.0,
            "pose_step_median_norm_diag": 0.0,
            "camera_centers": centers,
        }

    diffs = centers[1:] - centers[:-1]
    step = np.linalg.norm(diffs, axis=1)

    cmin = centers.min(axis=0)
    cmax = centers.max(axis=0)
    ext = cmax - cmin
    diag = float(np.linalg.norm(ext))
    step_med = float(np.median(step))
    step_p95 = float(np.percentile(step, 95))

    return {
        "poses_loaded": True,
        "num_poses_total": int(len(centers_all)),
        "num_poses_used": int(len(centers)),
        "pose_convention_selected": convention,
        "pose_mapping_mode": mapping_mode,
        "pose_path_length": float(step.sum()),
        "pose_bbox_extent_x": float(ext[0]),
        "pose_bbox_extent_y": float(ext[1]),
        "pose_bbox_extent_z": float(ext[2]),
        "pose_bbox_diagonal": diag,
        "pose_step_median": step_med,
        "pose_step_p95": step_p95,
        "pose_step_median_norm_diag": safe_div(step_med, max(diag, 1e-9)),
        "camera_centers": centers,
    }


def baseline_norm_score(baseline_norm: float) -> float:
    if baseline_norm <= 0:
        return 0.0
    if baseline_norm < 0.01:
        return clamp(baseline_norm / 0.01)
    if baseline_norm <= 0.15:
        return 1.0
    if baseline_norm < 0.35:
        return clamp((0.35 - baseline_norm) / 0.20)
    return 0.0


def compute_quality_summary(
    frame_rows: list[dict],
    pair_rows: list[dict],
    pose_stats: Optional[dict],
    blur_threshold: float,
    low_overlap_threshold: float,
    jump_threshold: float,
    static_threshold: float,
) -> dict:
    lap = np.array([to_float(r["laplacian_variance"]) for r in frame_rows], dtype=np.float64)
    clipped = np.array([to_float(r["clipped_ratio"]) for r in frame_rows], dtype=np.float64)
    under = np.array([to_float(r["underexposed_ratio"]) for r in frame_rows], dtype=np.float64)
    over = np.array([to_float(r["overexposed_ratio"]) for r in frame_rows], dtype=np.float64)

    match_ratio = np.array([to_float(r["orb_match_ratio_to_keypoints"]) for r in pair_rows], dtype=np.float64)
    norm_diff = np.array([to_float(r["normalized_abs_diff"]) for r in pair_rows], dtype=np.float64)

    blurred_ratio = float(np.mean(lap < blur_threshold)) if lap.size else 1.0
    blur_detail_factor = clamp(float(np.median(lap) / max(blur_threshold * 2.0, 1.0))) if lap.size else 0.0
    blur_score = 100.0 * (0.6 * (1.0 - blurred_ratio) + 0.4 * blur_detail_factor)

    bad_exposure_ratio = float(np.mean((under > 0.20) | (over > 0.20))) if clipped.size else 1.0
    avg_clipped = float(clipped.mean()) if clipped.size else 1.0
    exposure_penalty = clamp(0.7 * bad_exposure_ratio + 1.8 * avg_clipped)
    exposure_score = 100.0 * (1.0 - exposure_penalty)

    low_overlap_ratio = float(np.mean(match_ratio < low_overlap_threshold)) if match_ratio.size else 1.0
    jump_ratio = float(np.mean(norm_diff > jump_threshold)) if norm_diff.size else 1.0
    static_ratio = float(np.mean(norm_diff < static_threshold)) if norm_diff.size else 1.0
    coverage_penalty = clamp(0.55 * low_overlap_ratio + 0.25 * jump_ratio + 0.20 * static_ratio)
    coverage_raw_score = 100.0 * (1.0 - coverage_penalty)

    pose_score = None
    coverage_score = coverage_raw_score
    if pose_stats is not None and pose_stats.get("poses_loaded", False):
        baseline_norm = to_float(pose_stats.get("pose_step_median_norm_diag", 0.0))
        baseline_score = baseline_norm_score(baseline_norm)
        span_factor = clamp(to_float(pose_stats.get("pose_bbox_diagonal", 0.0)) / 0.25)
        pose_score = 100.0 * (0.65 * baseline_score + 0.35 * span_factor)
        coverage_score = 0.70 * coverage_raw_score + 0.30 * pose_score

    overall_score = 0.40 * blur_score + 0.25 * exposure_score + 0.35 * coverage_score

    recommendations: list[str] = []
    if blur_score < 70:
        recommendations.append("Reduce camera motion speed and increase shutter speed / lighting to lower blur.")
    if exposure_score < 70:
        recommendations.append("Lock exposure and avoid clipped highlights/shadows (manual ISO/exposure if possible).")
    if low_overlap_ratio > 0.30:
        recommendations.append("Increase frame overlap: slower camera motion and denser sampling around objects.")
    if jump_ratio > 0.20:
        recommendations.append("Avoid abrupt viewpoint jumps; use smoother camera path transitions.")
    if static_ratio > 0.35:
        recommendations.append("Current sequence may be too static; add angular diversity around the scene.")
    if pose_score is not None and pose_score < 65:
        recommendations.append("Camera trajectory coverage is weak; widen path and include multi-height viewpoints.")
    if not recommendations:
        recommendations.append("Input quality looks healthy for reconstruction. Proceed to parameter sweeps.")

    return {
        "scores": {
            "blur_score": float(blur_score),
            "exposure_score": float(exposure_score),
            "coverage_score": float(coverage_score),
            "coverage_raw_score": float(coverage_raw_score),
            "pose_score": float(pose_score) if pose_score is not None else None,
            "overall_score": float(overall_score),
        },
        "ratios": {
            "blurred_frame_ratio": float(blurred_ratio),
            "bad_exposure_ratio": float(bad_exposure_ratio),
            "avg_clipped_ratio": float(avg_clipped),
            "low_overlap_ratio": float(low_overlap_ratio),
            "jump_ratio": float(jump_ratio),
            "static_ratio": float(static_ratio),
        },
        "thresholds": {
            "blur_threshold": float(blur_threshold),
            "low_overlap_threshold": float(low_overlap_threshold),
            "jump_threshold": float(jump_threshold),
            "static_threshold": float(static_threshold),
        },
        "recommendations": recommendations,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_flags(frame_rows: list[dict], blur_threshold: float) -> list[dict]:
    flags: list[dict] = []
    for row in frame_rows:
        blurry = to_float(row["laplacian_variance"]) < blur_threshold
        clipped = to_float(row["clipped_ratio"]) > 0.25
        low_texture = to_float(row["orb_keypoints"]) < 100
        keep = not (blurry or clipped)
        flags.append(
            {
                "frame_local_index": row["frame_local_index"],
                "source_frame_index": row["source_frame_index"],
                "is_blurry": bool(blurry),
                "is_clipped": bool(clipped),
                "is_low_texture": bool(low_texture),
                "recommended_keep": bool(keep),
            }
        )
    return flags


def plot_reports(
    frame_rows: list[dict],
    pair_rows: list[dict],
    pose_stats: Optional[dict],
    scores: dict,
    output_dir: Path,
) -> list[str]:
    if not MATPLOTLIB_AVAILABLE:
        return []

    written: list[str] = []

    x = [int(r["frame_local_index"]) for r in frame_rows]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs = axs.ravel()
    axs[0].plot(x, [to_float(r["laplacian_variance"]) for r in frame_rows], color="#1b9e77")
    axs[0].set_title("Sharpness (Laplacian variance)")
    axs[1].plot(x, [to_float(r["brightness_mean"]) for r in frame_rows], color="#7570b3", label="mean")
    axs[1].plot(x, [to_float(r["brightness_std"]) for r in frame_rows], color="#e7298a", label="std")
    axs[1].set_title("Brightness")
    axs[1].legend()
    axs[2].plot(x, [to_float(r["orb_keypoints"]) for r in frame_rows], color="#d95f02")
    axs[2].set_title("ORB keypoints")
    axs[3].plot(x, [to_float(r["underexposed_ratio"]) for r in frame_rows], color="#386cb0", label="under")
    axs[3].plot(x, [to_float(r["overexposed_ratio"]) for r in frame_rows], color="#f0027f", label="over")
    axs[3].set_title("Exposure clipping ratios")
    axs[3].legend()
    for ax in axs:
        ax.set_xlabel("Frame index")
    fig.tight_layout()
    p1 = output_dir / "frame_quality.png"
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p1))

    if pair_rows:
        xp = [int(r["frame_local_index_a"]) for r in pair_rows]
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axs[0].plot(xp, [to_float(r["normalized_abs_diff"]) for r in pair_rows], color="#1f78b4")
        axs[0].set_title("Consecutive-frame normalized difference")
        axs[1].plot(xp, [to_float(r["orb_match_ratio_to_keypoints"]) for r in pair_rows], color="#33a02c")
        axs[1].set_title("Consecutive-frame ORB match ratio")
        axs[1].set_xlabel("Pair start frame")
        fig.tight_layout()
        p2 = output_dir / "temporal_overlap.png"
        fig.savefig(p2, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p2))

    if pose_stats is not None and pose_stats.get("poses_loaded", False):
        centers = pose_stats.get("camera_centers")
        if isinstance(centers, np.ndarray) and centers.size > 0:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.plot(centers[:, 0], centers[:, 2], marker="o", linewidth=1.3, color="#6a3d9a")
            ax.set_title("Camera trajectory (X/Z)")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.axis("equal")
            fig.tight_layout()
            p3 = output_dir / "camera_trajectory_xz.png"
            fig.savefig(p3, dpi=180, bbox_inches="tight")
            plt.close(fig)
            written.append(str(p3))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["blur", "exposure", "coverage", "overall"]
    vals = [
        to_float(scores["blur_score"]),
        to_float(scores["exposure_score"]),
        to_float(scores["coverage_score"]),
        to_float(scores["overall_score"]),
    ]
    ax.bar(labels, vals, color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"])
    ax.set_ylim(0, 100)
    ax.set_title("Input quality scores")
    ax.set_ylabel("Score (0-100)")
    fig.tight_layout()
    p4 = output_dir / "quality_scores.png"
    fig.savefig(p4, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p4))

    return written


def write_text_summary(path: Path, report: dict) -> None:
    lines: list[str] = []
    lines.append("Input Quality Report")
    lines.append("=" * 80)
    lines.append(f"timestamp: {report['timestamp']}")
    lines.append(f"source_type: {report['source']['type']}")
    lines.append(f"source_path: {report['source']['path']}")
    lines.append(f"num_frames: {report['source']['num_frames']}")
    lines.append(f"fps: {report['source']['fps']}")
    lines.append(f"poses_loaded: {report['poses']['poses_loaded']}")
    lines.append("")
    lines.append("Scores")
    lines.append("-" * 80)
    for k, v in report["quality"]["scores"].items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Ratios")
    lines.append("-" * 80)
    for k, v in report["quality"]["ratios"].items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Pose coverage")
    lines.append("-" * 80)
    for k, v in report["poses"].items():
        if k == "camera_centers":
            continue
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Recommendations")
    lines.append("-" * 80)
    for rec in report["quality"]["recommendations"]:
        lines.append(f"- {rec}")
    lines.append("")
    lines.append("Artifacts")
    lines.append("-" * 80)
    for k, v in report["artifacts"].items():
        lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate blur/exposure/coverage report for input frames.")
    parser.add_argument("--model", choices=["fast3r", "dust3r"], default=None, help="Optional model shortcut for default paths.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Directory with input frames.")
    parser.add_argument("--video", type=Path, default=None, help="Input video path (alternative to --frames-dir).")
    parser.add_argument("--poses", type=Path, default=None, help="Optional poses.npy (Nx4x4) for camera-coverage metrics.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to analyze (0 means all selected).")
    parser.add_argument("--stride", type=int, default=1, help="Take every N-th frame.")
    parser.add_argument("--orb-features", type=int, default=1000)
    parser.add_argument("--underexposed-threshold", type=int, default=10)
    parser.add_argument("--overexposed-threshold", type=int, default=245)
    parser.add_argument("--blur-threshold", type=float, default=140.0)
    parser.add_argument("--low-overlap-threshold", type=float, default=0.08)
    parser.add_argument("--jump-threshold", type=float, default=0.35)
    parser.add_argument("--static-threshold", type=float, default=0.02)
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    frames_dir = args.frames_dir
    poses_path = args.poses
    if args.model is not None:
        if frames_dir is None:
            frames_dir = default_frames_dir_for_model(args.model)
        if poses_path is None:
            poses_path = default_poses_path_for_model(args.model)

    if args.video is None and frames_dir is None:
        raise ValueError("Provide one input source: --frames-dir or --video (or --model shortcut).")
    if args.video is not None and frames_dir is not None:
        raise ValueError("Use either --video or --frames-dir, not both.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_tag = args.model if args.model is not None else ("video" if args.video is not None else "frames")
    out_dir = args.out_root / f"{stamp}_{source_tag}_input_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("[1/6] Loading input frames")
    if args.video is not None:
        bundle = load_frames_from_video(args.video, args.max_frames, args.stride)
    else:
        bundle = load_frames_from_directory(frames_dir, args.max_frames, args.stride)  # type: ignore[arg-type]
    log(f"  loaded frames: {len(bundle.frames)} from {bundle.source_path}")

    log("[2/6] Computing frame and pair metrics")
    frame_rows, pair_rows = compute_frame_and_pair_metrics(
        bundle.frames,
        bundle.source_indices,
        bundle.fps,
        args.orb_features,
        args.underexposed_threshold,
        args.overexposed_threshold,
    )

    log("[3/6] Computing optional pose coverage")
    pose_stats: Optional[dict] = None
    camera_centers_artifact = ""
    if poses_path is not None and poses_path.exists():
        try:
            poses = load_poses(poses_path)
            pose_stats = pose_coverage_metrics(poses, bundle.source_indices, len(bundle.frames))
            centers = pose_stats.get("camera_centers")
            if isinstance(centers, np.ndarray):
                centers_path = out_dir / "camera_centers.npy"
                np.save(centers_path, centers)
                camera_centers_artifact = str(centers_path)
            log(
                f"  poses loaded: used={pose_stats.get('num_poses_used')} "
                f"convention={pose_stats.get('pose_convention_selected')}"
            )
        except Exception as exc:
            log(f"  warning: failed to load poses ({exc}), continuing without pose coverage")
            pose_stats = {"poses_loaded": False, "error": str(exc)}
    else:
        pose_stats = {"poses_loaded": False}

    log("[4/6] Computing quality scores and frame flags")
    quality = compute_quality_summary(
        frame_rows=frame_rows,
        pair_rows=pair_rows,
        pose_stats=pose_stats if pose_stats.get("poses_loaded", False) else None,
        blur_threshold=args.blur_threshold,
        low_overlap_threshold=args.low_overlap_threshold,
        jump_threshold=args.jump_threshold,
        static_threshold=args.static_threshold,
    )
    flag_rows = make_flags(frame_rows, args.blur_threshold)

    log("[5/6] Writing CSV/JSON/TXT artifacts")
    frame_csv = out_dir / "frame_metrics.csv"
    pair_csv = out_dir / "frame_pair_metrics.csv"
    flags_csv = out_dir / "frame_quality_flags.csv"
    write_csv(frame_csv, frame_rows)
    write_csv(pair_csv, pair_rows)
    write_csv(flags_csv, flag_rows)

    pose_report = dict(pose_stats) if pose_stats is not None else {"poses_loaded": False}
    if "camera_centers" in pose_report:
        pose_report.pop("camera_centers")

    report = {
        "timestamp": stamp,
        "source": {
            "type": bundle.source_type,
            "path": bundle.source_path,
            "num_frames": len(bundle.frames),
            "fps": bundle.fps,
            "stride": int(args.stride),
            "max_frames": int(args.max_frames),
            "model_hint": args.model,
            "poses_path": str(poses_path) if poses_path is not None else "",
        },
        "thresholds": {
            "underexposed_threshold": int(args.underexposed_threshold),
            "overexposed_threshold": int(args.overexposed_threshold),
            "blur_threshold": float(args.blur_threshold),
            "low_overlap_threshold": float(args.low_overlap_threshold),
            "jump_threshold": float(args.jump_threshold),
            "static_threshold": float(args.static_threshold),
        },
        "quality": quality,
        "poses": pose_report,
        "artifacts": {
            "frame_metrics_csv": str(frame_csv),
            "frame_pair_metrics_csv": str(pair_csv),
            "frame_quality_flags_csv": str(flags_csv),
            "summary_json": str(out_dir / "input_quality_report.json"),
            "summary_txt": str(out_dir / "input_quality_report.txt"),
            "camera_centers_npy": camera_centers_artifact,
        },
    }
    json_path = out_dir / "input_quality_report.json"
    txt_path = out_dir / "input_quality_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_text_summary(txt_path, report)

    log("[6/6] Plotting")
    plot_files: list[str] = []
    if args.disable_plots:
        log("  plots disabled by flag")
    elif not MATPLOTLIB_AVAILABLE:
        log("  matplotlib not installed, skipping plots")
    else:
        plot_files = plot_reports(
            frame_rows=frame_rows,
            pair_rows=pair_rows,
            pose_stats=pose_stats if pose_stats is not None else {"poses_loaded": False},
            scores=quality["scores"],
            output_dir=out_dir,
        )
        log(f"  plots written: {len(plot_files)}")

    if plot_files:
        report["artifacts"]["plots"] = plot_files
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_text_summary(txt_path, report)

    overall = quality["scores"]["overall_score"]
    log(f"Done. Overall input quality score: {overall:.1f}/100")
    log(f"Results folder: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
