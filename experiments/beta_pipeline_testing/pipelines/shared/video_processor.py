"""Video frame extraction module."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from .config import FrameExtractionConfig


class FrameExtractor:
    """Extract frames from video with various sampling strategies."""
    
    def __init__(self, config: FrameExtractionConfig):
        """Initialize frame extractor.
        
        Args:
            config: FrameExtractionConfig with extraction settings
        """
        self.config = config
    
    def extract_frames(self, video_path: Path) -> Tuple[List[np.ndarray], dict]:
        """Extract frames from video.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Tuple of (frames list, metadata dict)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"Loading video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties:")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Extract frames
        frames = []
        frame_indices = []

        selection_mode = str(self.config.selection_mode or "").lower()
        if self.config.uniform_sampling:
            selection_mode = "uniform"
        if selection_mode == "coverage_aware":
            frames, frame_indices, selection_report = self._extract_coverage_aware(
                cap,
                total_frames=total_frames,
            )
        elif selection_mode == "overlap_aware":
            frames, frame_indices, selection_report = self._extract_overlap_aware(
                cap,
                total_frames=total_frames,
            )
        elif selection_mode == "uniform":
            # Uniform sampling: evenly distribute frames across entire video
            interval = max(1, total_frames // self.config.num_frames)
            frame_count = 0
            selection_report = {"mode": "uniform", "interval": int(interval)}
            
            while len(frames) < self.config.num_frames and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    # Resize if needed
                    frame_resized = cv2.resize(frame, self.config.target_size)
                    frames.append(frame_resized)
                    frame_indices.append(frame_count)
                
                frame_count += 1
        else:
            # Skip frames mode: extract every N frames for heavy temporal overlap
            # Best for COLMAP and pose consistency with panning video
            frame_count = 0
            selection_report = {
                "mode": "skip",
                "skip_frames": int(self.config.skip_frames),
            }
            
            while len(frames) < self.config.num_frames and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.config.skip_frames == 0:
                    # Resize if needed
                    frame_resized = cv2.resize(frame, self.config.target_size)
                    frames.append(frame_resized)
                    frame_indices.append(frame_count)
                
                frame_count += 1
        
        cap.release()
        
        metadata = {
            'fps': fps,
            'total_frames': total_frames,
            'extracted_frames': len(frames),
            'frame_indices': frame_indices,
            'original_resolution': (frame_width, frame_height),
            'target_resolution': self.config.target_size,
            'selection_report': selection_report,
        }
        
        print(f"Extracted {len(frames)} frames at {self.config.target_size} resolution")
        
        return frames, metadata

    def _frame_quality(self, frame: np.ndarray, orb) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        clipped_ratio = float(np.mean(gray <= 10) + np.mean(gray >= 245))
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        thumb = thumb.reshape(-1)
        thumb -= float(thumb.mean())
        norm = float(np.linalg.norm(thumb))
        if norm > 0:
            thumb /= norm
        return {
            "gray": gray,
            "laplacian_variance": laplacian_variance,
            "clipped_ratio": clipped_ratio,
            "orb_keypoints": int(len(keypoints)),
            "descriptors": descriptors,
            "thumbnail": thumb,
        }

    def _pair_overlap(self, prev: dict, curr: dict) -> dict:
        corr = float(np.dot(prev["thumbnail"], curr["thumbnail"]))
        matches = 0
        median_match_distance = None
        if prev["descriptors"] is not None and curr["descriptors"] is not None:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            raw_matches = matcher.match(prev["descriptors"], curr["descriptors"])
            matches = int(len(raw_matches))
            if raw_matches:
                median_match_distance = float(np.median([m.distance for m in raw_matches]))
        return {
            "orb_matches": matches,
            "thumbnail_corr": corr,
            "median_match_distance": median_match_distance,
        }

    def _extract_overlap_aware(self, cap, total_frames: int) -> Tuple[List[np.ndarray], list[int], dict]:
        """Select sharp frames while preserving local visual overlap."""

        scan_stride = max(1, int(self.config.scan_stride))
        min_gap = max(1, int(self.config.skip_frames))
        max_gap = max(min_gap, int(self.config.max_frame_gap))
        target = max(1, int(self.config.num_frames))
        overlap_window_size = max(1, int(self.config.overlap_window_size))
        force_accept_after_gap = max(max_gap + 1, int(self.config.force_accept_after_gap))
        force_accept_after_rejections = max(1, int(self.config.force_accept_after_rejections))
        orb = cv2.ORB_create(nfeatures=2500)

        candidates: list[dict] = []
        frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % scan_stride == 0:
                frame_resized = cv2.resize(frame, self.config.target_size)
                quality = self._frame_quality(frame_resized, orb)
                quality_ok = (
                    quality["laplacian_variance"] >= float(self.config.min_laplacian_variance)
                    and quality["clipped_ratio"] <= float(self.config.max_clipped_ratio)
                )
                candidates.append(
                    {
                        "frame": frame_resized,
                        "index": int(frame_count),
                        "quality": quality,
                        "quality_ok": bool(quality_ok),
                    }
                )
            frame_count += 1

        selected: list[dict] = []
        rejected_quality = 0
        rejected_gap = 0
        rejected_overlap = 0
        forced_accept_gap = 0
        forced_accept_recovery = 0
        consecutive_overlap_rejections = 0
        pair_rows: list[dict] = []

        for candidate in candidates:
            if len(selected) >= target:
                break
            if not candidate["quality_ok"]:
                rejected_quality += 1
                continue
            if not selected:
                selected.append(candidate)
                continue

            gap = int(candidate["index"] - selected[-1]["index"])
            if gap < min_gap:
                rejected_gap += 1
                continue

            references = selected[-overlap_window_size:]
            best_overlap = None
            best_reference = None
            for reference in references:
                overlap = self._pair_overlap(reference["quality"], candidate["quality"])
                score = (
                    int(overlap["orb_matches"]),
                    float(overlap["thumbnail_corr"]),
                )
                if best_overlap is None or score > (
                    int(best_overlap["orb_matches"]),
                    float(best_overlap["thumbnail_corr"]),
                ):
                    best_overlap = overlap
                    best_reference = reference

            overlap = best_overlap or {
                "orb_matches": 0,
                "thumbnail_corr": 0.0,
                "median_match_distance": None,
            }
            reference_index = int(best_reference["index"]) if best_reference is not None else int(selected[-1]["index"])
            reference_gap = int(candidate["index"] - reference_index)
            overlap_ok = (
                overlap["orb_matches"] >= int(self.config.min_orb_matches)
                or overlap["thumbnail_corr"] >= float(self.config.min_thumbnail_corr)
            )
            force_reason = ""
            if gap >= force_accept_after_gap:
                force_reason = "force_accept_after_gap"
            elif consecutive_overlap_rejections >= force_accept_after_rejections:
                force_reason = "force_accept_after_rejections"

            pair_row = {
                "prev_frame_index": int(reference_index),
                "curr_frame_index": int(candidate["index"]),
                "frame_gap": reference_gap,
                "gap_from_last_selected": gap,
                "overlap_window_size": overlap_window_size,
                "accepted": False,
                "accept_reason": "",
                **overlap,
            }

            if gap > max_gap and not force_reason:
                rejected_gap += 1
                pair_row["accept_reason"] = "rejected_max_gap"
                pair_rows.append(pair_row)
                continue

            if not overlap_ok and not force_reason:
                rejected_overlap += 1
                consecutive_overlap_rejections += 1
                pair_row["accept_reason"] = "rejected_overlap"
                pair_rows.append(pair_row)
                continue

            if force_reason == "force_accept_after_gap":
                forced_accept_gap += 1
            elif force_reason == "force_accept_after_rejections":
                forced_accept_recovery += 1
            consecutive_overlap_rejections = 0
            pair_row["accepted"] = True
            pair_row["accept_reason"] = force_reason or "overlap"
            pair_rows.append(pair_row)
            selected.append(candidate)

        frames = [item["frame"] for item in selected]
        frame_indices = [int(item["index"]) for item in selected]
        report = {
            "mode": "overlap_aware",
            "target_frames": target,
            "selected_frames": int(len(frames)),
            "scan_stride": scan_stride,
            "candidate_frames": int(len(candidates)),
            "min_frame_gap": min_gap,
            "max_frame_gap": max_gap,
            "min_laplacian_variance": float(self.config.min_laplacian_variance),
            "max_clipped_ratio": float(self.config.max_clipped_ratio),
            "min_orb_matches": int(self.config.min_orb_matches),
            "min_thumbnail_corr": float(self.config.min_thumbnail_corr),
            "overlap_window_size": int(overlap_window_size),
            "force_accept_after_gap": int(force_accept_after_gap),
            "force_accept_after_rejections": int(force_accept_after_rejections),
            "rejected_quality": int(rejected_quality),
            "rejected_gap": int(rejected_gap),
            "rejected_overlap": int(rejected_overlap),
            "forced_accept_after_gap": int(forced_accept_gap),
            "forced_accept_after_rejections": int(forced_accept_recovery),
            "pair_overlap_checks": int(len(pair_rows)),
            "selected_frame_indices": frame_indices,
            "pair_overlap_rows": pair_rows,
        }
        return frames, frame_indices, report

    def _quality_score(self, quality: dict) -> float:
        lap = float(quality["laplacian_variance"])
        keypoints = float(quality["orb_keypoints"])
        clipped = float(quality["clipped_ratio"])
        return lap + keypoints * 0.05 - clipped * 500.0

    def _extract_coverage_aware(self, cap, total_frames: int) -> Tuple[List[np.ndarray], list[int], dict]:
        """Select quality frames from temporal bins spanning the whole video."""

        scan_stride = max(1, int(self.config.scan_stride))
        target = max(1, int(self.config.num_frames))
        orb = cv2.ORB_create(nfeatures=2500)

        candidates: list[dict] = []
        frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % scan_stride == 0:
                frame_resized = cv2.resize(frame, self.config.target_size)
                quality = self._frame_quality(frame_resized, orb)
                quality_ok = (
                    quality["laplacian_variance"] >= float(self.config.min_laplacian_variance)
                    and quality["clipped_ratio"] <= float(self.config.max_clipped_ratio)
                )
                candidates.append(
                    {
                        "frame": frame_resized,
                        "index": int(frame_count),
                        "quality": quality,
                        "quality_ok": bool(quality_ok),
                        "score": self._quality_score(quality),
                    }
                )
            frame_count += 1

        selected: list[dict] = []
        bin_rows: list[dict] = []
        fallback_bins = 0
        empty_bins = 0
        for bin_idx in range(target):
            start = int(round(bin_idx * total_frames / target))
            end = int(round((bin_idx + 1) * total_frames / target))
            bin_candidates = [c for c in candidates if start <= c["index"] < end]
            if not bin_candidates:
                empty_bins += 1
                continue
            quality_candidates = [c for c in bin_candidates if c["quality_ok"]]
            pool = quality_candidates or bin_candidates
            if not quality_candidates:
                fallback_bins += 1
            chosen = max(pool, key=lambda c: c["score"])
            selected.append(chosen)
            bin_rows.append(
                {
                    "bin_index": int(bin_idx),
                    "bin_start": int(start),
                    "bin_end": int(end),
                    "candidate_count": int(len(bin_candidates)),
                    "quality_candidate_count": int(len(quality_candidates)),
                    "selected_frame_index": int(chosen["index"]),
                    "selected_score": float(chosen["score"]),
                    "laplacian_variance": float(chosen["quality"]["laplacian_variance"]),
                    "clipped_ratio": float(chosen["quality"]["clipped_ratio"]),
                    "orb_keypoints": int(chosen["quality"]["orb_keypoints"]),
                    "fallback_used": bool(not quality_candidates),
                }
            )

        selected = sorted(selected, key=lambda c: c["index"])
        frames = [item["frame"] for item in selected]
        frame_indices = [int(item["index"]) for item in selected]
        report = {
            "mode": "coverage_aware",
            "target_frames": target,
            "selected_frames": int(len(frames)),
            "scan_stride": int(scan_stride),
            "candidate_frames": int(len(candidates)),
            "total_video_frames": int(total_frames),
            "min_laplacian_variance": float(self.config.min_laplacian_variance),
            "max_clipped_ratio": float(self.config.max_clipped_ratio),
            "empty_bins": int(empty_bins),
            "fallback_bins": int(fallback_bins),
            "selected_frame_indices": frame_indices,
            "bin_rows": bin_rows,
        }
        return frames, frame_indices, report
    
    def get_video_info(self, video_path: Path) -> dict:
        """Get video information without extracting frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        cap = cv2.VideoCapture(str(video_path))
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        cap.release()
        
        return info
