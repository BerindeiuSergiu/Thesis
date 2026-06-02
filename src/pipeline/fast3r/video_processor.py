# NOTE: Do NOT import directly from experiments/.
# All pipeline code must be copied into src/pipeline/fast3r/ before modification.
# Best methods from experiments/ have been audited and integrated — see src/EXPERIMENTS_AUDIT.md

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_stride_frames(
    video_path: Path,
    num_frames: int,
    stride: int,
    target_size: tuple[int, int],
) -> tuple[list[np.ndarray], dict]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, stride) == 0:
            frames.append(cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA))
            frame_indices.append(frame_idx)
            if len(frames) >= num_frames:
                break
        frame_idx += 1

    cap.release()

    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "selected_frame_indices": frame_indices,
        "original_resolution": (width, height),
        "target_resolution": target_size,
        "selected_frame_count": len(frames),
        "selection_mode": "stride",
    }
    return frames, metadata
