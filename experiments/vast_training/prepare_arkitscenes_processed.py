from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def traj_string_to_matrix(traj_string: str) -> tuple[str, np.ndarray]:
    tokens = traj_string.split()
    if len(tokens) != 7:
        raise ValueError(f"Unexpected trajectory format: {traj_string}")
    ts = tokens[0]
    angle_axis = np.asarray([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=np.float64)
    r_w_to_p, _ = cv2.Rodrigues(angle_axis)
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])], dtype=np.float64)
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    return ts, np.linalg.inv(extrinsics).astype(np.float32)


def load_pose_map(traj_path: Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ts, pose = traj_string_to_matrix(line)
        poses[f"{round(float(ts), 3):.3f}"] = pose
    return poses


def resolve_pose(frame_id: str, pose_map: dict[str, np.ndarray]) -> np.ndarray | None:
    if frame_id in pose_map:
        return pose_map[frame_id]
    for key, pose in pose_map.items():
        if abs(float(frame_id) - float(key)) < 0.1:
            return pose
    return None


def finalize_pose(pose: np.ndarray) -> np.ndarray:
    pose = pose.copy()
    pose[0:3, 1:3] *= -1
    pose = pose[np.array([1, 0, 2, 3]), :]
    pose[2, :] *= -1
    pose[:, 1:3] *= -1.0
    return pose.astype(np.float32)


def resolve_intrinsics(intrinsics_dir: Path, scene_id: str, frame_id: str) -> np.ndarray | None:
    candidates = [
        intrinsics_dir / f"{scene_id}_{frame_id}.pincam",
        intrinsics_dir / f"{scene_id}_{float(frame_id) - 0.001:.3f}.pincam",
        intrinsics_dir / f"{scene_id}_{float(frame_id) + 0.001:.3f}.pincam",
    ]
    for candidate in candidates:
        if candidate.exists():
            width, height, fx, fy, cx, cy = np.loadtxt(candidate, dtype=np.float32)
            return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return None


def resolve_rgb_file(rgb_dir: Path, basename: str) -> Path | None:
    stem = basename.replace(".png", "")
    candidates = [
        rgb_dir / f"{stem}.jpg",
        rgb_dir / f"{stem}.jpeg",
        rgb_dir / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_scene_asset_dirs(scene_dir: Path) -> tuple[Path, Path]:
    rgb_candidates = [
        scene_dir / "lowres_wide",
        scene_dir / "vga_wide",
    ]
    intr_candidates = [
        scene_dir / "lowres_wide_intrinsics",
        scene_dir / "vga_wide_intrinsics",
    ]
    rgb_dir = next((path for path in rgb_candidates if path.exists()), None)
    intr_dir = next((path for path in intr_candidates if path.exists()), None)
    if rgb_dir is None or intr_dir is None:
        raise FileNotFoundError(f"Missing RGB/intrinsics assets in {scene_dir}")
    return rgb_dir, intr_dir


def ensure_scene_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
        return
    if mode == "hardlink":
        shutil.copytree(src, dst, copy_function=os.link)
        return
    if mode == "copy":
        shutil.copytree(src, dst)
        return
    raise ValueError(f"Unsupported link mode: {mode}")


def build_split_metadata(raw_split_dir: Path, processed_split_dir: Path) -> None:
    scene_names = sorted([path.name for path in raw_split_dir.iterdir() if path.is_dir()])
    scenes = np.asarray(scene_names)
    sceneids: list[int] = []
    images: list[str] = []
    intrinsics: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    pairs: list[tuple[int, int]] = []

    running_index = 0
    for scene_idx, scene_name in enumerate(scene_names):
        scene_dir = raw_split_dir / scene_name
        depth_dir = scene_dir / "lowres_depth"
        traj_path = scene_dir / "lowres_wide.traj"
        try:
            rgb_dir, intr_dir = resolve_scene_asset_dirs(scene_dir)
        except FileNotFoundError:
            continue
        if not (depth_dir.exists() and traj_path.exists()):
            continue

        pose_map = load_pose_map(traj_path)
        local_indices: list[int] = []
        for depth_file in sorted(depth_dir.glob("*.png")):
            basename = depth_file.name
            frame_id = basename.replace(".png", "").split("_", 1)[1]
            rgb_file = resolve_rgb_file(rgb_dir, basename)
            if rgb_file is None:
                continue
            pose = resolve_pose(frame_id, pose_map)
            if pose is None:
                continue
            pose = finalize_pose(pose)
            intrinsic = resolve_intrinsics(intr_dir, scene_name, frame_id)
            if intrinsic is None:
                continue

            sceneids.append(scene_idx)
            images.append(rgb_file.name)
            intrinsics.append(intrinsic)
            trajectories.append(pose)
            local_indices.append(running_index)
            running_index += 1

        for idx_a, idx_b in zip(local_indices[:-1], local_indices[1:]):
            pairs.append((idx_a, idx_b))

    if not images:
        raise RuntimeError(f"No valid ARKitScenes frames found in {raw_split_dir}")

    np.savez_compressed(
        processed_split_dir / "all_metadata.npz",
        scenes=scenes,
        sceneids=np.asarray(sceneids, dtype=np.int32),
        images=np.asarray(images),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        trajectories=np.stack(trajectories).astype(np.float32),
        pairs=np.asarray(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), dtype=np.int32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ARKitScenes raw download into the Fast3R arkitscenes_processed layout.")
    parser.add_argument("--raw-root", required=True, help="Path to the official raw ARKitScenes download root.")
    parser.add_argument("--output-root", required=True, help="Path to write the Fast3R-compatible arkitscenes_processed root.")
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    return parser.parse_args()


def resolve_raw_root(raw_root: Path) -> Path:
    direct_training = raw_root / "Training"
    nested_training = raw_root / "raw" / "Training"
    if direct_training.exists():
        return raw_root
    if nested_training.exists():
        return raw_root / "raw"
    return raw_root


def main() -> int:
    args = parse_args()
    raw_root = resolve_raw_root(Path(args.raw_root).resolve())
    output_root = Path(args.output_root).resolve()

    split_mapping = {
        "Training": "Training",
        "Validation": "Test",
    }

    for raw_split_name, processed_split_name in split_mapping.items():
        raw_split_dir = raw_root / raw_split_name
        if not raw_split_dir.exists():
            raise FileNotFoundError(f"Missing ARKitScenes split directory: {raw_split_dir}")
        processed_split_dir = output_root / processed_split_name
        processed_split_dir.mkdir(parents=True, exist_ok=True)

        for scene_dir in sorted([path for path in raw_split_dir.iterdir() if path.is_dir()]):
            ensure_scene_link(scene_dir, processed_split_dir / scene_dir.name, args.link_mode)

        build_split_metadata(raw_split_dir, processed_split_dir)
        print(f"Prepared {processed_split_name}: {processed_split_dir}")

    print(f"ARKitScenes processed root ready at: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
