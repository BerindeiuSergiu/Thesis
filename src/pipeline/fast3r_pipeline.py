# NOTE: Do NOT import directly from experiments/.
# All pipeline code must be copied into src/pipeline/fast3r/ before modification.
# Best methods from experiments/ have been audited and integrated — see src/EXPERIMENTS_AUDIT.md

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from src.models.scene_result import SceneResult
from src.pipeline.base_pipeline import BasePipeline
from src.pipeline.fast3r.frame_selection import select_frames_from_video
from src.pipeline.fast3r.gaussian_splats import build_gaussian_splats
from src.pipeline.fast3r.mesh_reconstruction import build_poisson_mesh
from src.pipeline.fast3r.postprocess import (
    create_dummy_pointcloud,
    numeric_sanity_filter,
    radius_percentile_filter,
    save_pointcloud_ply,
    voxel_downsample,
)
from src.pipeline.fast3r.reconstruction import Fast3RReconstructionConfig, Fast3RReconstructor
from src.pipeline.fast3r.scale_normalization import scale_normalization
from src.pipeline.fast3r.video_processor import extract_stride_frames


class Fast3rPipeline(BasePipeline):
    def __init__(self) -> None:
        self.config: dict = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.progress_callback: Callable[[int, str], None] | None = None

    def set_progress_callback(self, callback: Callable[[int, str], None] | None) -> None:
        self.progress_callback = callback

    def _emit(self, percent: int, message: str) -> None:
        self.logger.info("%s (%s%%)", message, percent)
        if self.progress_callback is not None:
            self.progress_callback(percent, message)

    def load(self, config: dict) -> None:
        self.config = config
        self.logger.info("Fast3rPipeline loaded with config.")

    def _apply_fast3r_preset(self, fast3r_config: dict) -> dict:
        """Merge the active UI preset into the runtime config when requested."""

        merged = dict(fast3r_config)
        if not bool(merged.get("auto_preset_enabled", False)):
            return merged
        presets = merged.get("presets", {})
        if not isinstance(presets, dict):
            return merged
        preset_name = str(merged.get("active_preset") or merged.get("preset_name") or "")
        preset = presets.get(preset_name)
        if not isinstance(preset, dict):
            return merged
        merged.update(preset)
        merged["auto_preset_enabled"] = True
        merged["active_preset"] = preset_name
        return merged

    def run(self, video_path: str) -> SceneResult:
        if not self.config:
            raise RuntimeError("Pipeline config was not loaded.")

        src_root = Path(__file__).resolve().parents[1]
        app_config = self.config.get("app", {})
        pipeline_config = self.config.get("pipeline", {})
        fast3r_config = self._apply_fast3r_preset(dict(pipeline_config.get("fast3r", {})))

        outputs_root = Path(app_config.get("outputs_root", "outputs"))
        if not outputs_root.is_absolute():
            outputs_root = src_root / outputs_root
        outputs_root.mkdir(parents=True, exist_ok=True)

        scene_id = datetime.now().strftime("scene_%Y%m%d_%H%M%S")
        scene_dir = outputs_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        source_video = Path(video_path).resolve()
        self._emit(5, "Preparing scene workspace")

        if bool(pipeline_config.get("stub_mode", True)):
            return self._run_stub(scene_id, source_video, scene_dir)
        return self._run_real(scene_id, source_video, scene_dir, fast3r_config)

    def _run_stub(self, scene_id: str, source_video: Path, scene_dir: Path) -> SceneResult:
        self._emit(15, "Stub mode enabled")
        pointcloud_path = create_dummy_pointcloud(scene_dir / "dummy_pointcloud.ply")
        metadata = {
            "stub_mode": True,
            "message": "Stub scene generated for UI and viewer testing.",
        }
        (scene_dir / "scene_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._emit(100, "Stub scene ready")
        return SceneResult(
            scene_id=scene_id,
            source_video=source_video,
            output_dir=scene_dir,
            pointcloud_path=pointcloud_path,
            mesh_path=None,
            camera_poses=[],
            metadata=metadata,
        )

    def _run_real(self, scene_id: str, source_video: Path, scene_dir: Path, fast3r_config: dict) -> SceneResult:
        target_size = (int(fast3r_config.get("target_width", 1024)), int(fast3r_config.get("target_height", 768)))
        analysis_size = (
            int(fast3r_config.get("analysis_width", 384)),
            int(fast3r_config.get("analysis_height", 288)),
        )

        self._emit(10, "Selecting input frames")
        if bool(fast3r_config.get("prefilter_enabled", True)):
            frames, frame_metadata = select_frames_from_video(
                video_path=source_video,
                target_frames=int(fast3r_config.get("target_frames", 80)),
                scan_stride=int(fast3r_config.get("scan_stride", 15)),
                shortlist_multiplier=int(fast3r_config.get("shortlist_multiplier", 3)),
                dedupe_similarity=float(fast3r_config.get("dedupe_similarity", 0.985)),
                min_frame_gap=int(fast3r_config.get("min_frame_gap", 120)),
                analysis_size=analysis_size,
                output_size=target_size,
                orb_features=int(fast3r_config.get("orb_features", 1000)),
                progress_callback=self.progress_callback,
                selection_mode=str(fast3r_config.get("selection_mode", "prefilter")),
                visual_shortlist_target=int(fast3r_config.get("visual_shortlist_target", 120)),
                probe_max_frames=int(fast3r_config.get("probe_max_frames", 96)),
                probe_image_size=int(fast3r_config.get("probe_image_size", 256)),
                probe_dtype=str(fast3r_config.get("probe_dtype", "float32")),
                probe_confidence_threshold=float(fast3r_config.get("probe_confidence_threshold", 1.0)),
                probe_pnp_iters=int(fast3r_config.get("probe_pnp_iters", 50)),
                probe_focal_method=str(fast3r_config.get("probe_focal_method", "first_view_from_global_head")),
                probe_model_name=str(fast3r_config.get("probe_model_name", "jedyang97/Fast3R_ViT_Large_512")),
                probe_max_parallel_views=int(fast3r_config.get("probe_max_parallel_views", 16)),
            )
        else:
            frames, frame_metadata = extract_stride_frames(
                video_path=source_video,
                num_frames=int(fast3r_config.get("target_frames", 80)),
                stride=int(fast3r_config.get("scan_stride", 15)),
                target_size=target_size,
            )

        frames_dir = scene_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            import cv2

            cv2.imwrite(str(frames_dir / f"frame_{index:04d}.jpg"), frame)

        reconstructor = Fast3RReconstructor(
            Fast3RReconstructionConfig(
                image_size=int(fast3r_config.get("fast3r_image_size", 512)),
                min_conf_thr=float(fast3r_config.get("min_confidence_threshold", 1.0)),
                confidence_keep_ratio=float(fast3r_config.get("confidence_keep_ratio", 1.0)),
                view_conf_p50_min=float(fast3r_config.get("view_conf_p50_min", 0.0)),
                view_conf_p90_min=float(fast3r_config.get("view_conf_p90_min", 0.0)),
                weak_texture_retention=bool(fast3r_config.get("weak_texture_retention", True)),
                weak_texture_percentile=float(fast3r_config.get("weak_texture_percentile", 35.0)),
                weak_texture_min_conf_thr=float(fast3r_config.get("weak_texture_min_conf_thr", 0.55)),
                weak_texture_keep_ratio=float(fast3r_config.get("weak_texture_keep_ratio", 0.35)),
                save_depth_maps=bool(fast3r_config.get("save_depth_maps", True)),
                niter_pnp=int(fast3r_config.get("niter_pnp", 100)),
                dtype=str(fast3r_config.get("dtype", "float32")),
            )
        )

        self._emit(60, "Running Fast3R reconstruction")
        points, colors, poses = reconstructor.reconstruct_from_frames(
            frames=frames,
            output_dir=scene_dir,
            progress_callback=self.progress_callback,
        )

        np.save(scene_dir / "raw_points.npy", points)
        np.save(scene_dir / "raw_colors.npy", colors)
        np.save(scene_dir / "poses.npy", np.array(poses))
        reconstruction_metadata = dict(getattr(reconstructor, "last_metadata", {}) or {})

        self._emit(76, "Filtering invalid reconstruction coordinates")
        sane_points, sane_colors, sanity_stats = numeric_sanity_filter(
            points,
            colors,
            max_abs_coordinate=float(fast3r_config.get("max_abs_coordinate", 100.0)),
        )
        if len(sane_points) == 0:
            raise RuntimeError(
                "Fast3R produced no finite, numerically sane points. "
                "Try fewer target frames, a higher confidence threshold, or a more stable input clip."
            )
        np.save(scene_dir / "sane_points.npy", sane_points)
        np.save(scene_dir / "sane_colors.npy", sane_colors)

        self._emit(78, "Normalizing reconstruction scale")
        scale_metadata = {
            "camera_poses": [np.asarray(pose).tolist() for pose in poses],
            "model_family": "fast3r",
        }
        scaled_points, scaled_poses, scale_report = scale_normalization(
            points=sane_points,
            poses=[np.asarray(pose) for pose in poses],
            metadata=scale_metadata,
            config=fast3r_config,
            output_dir=scene_dir,
        )
        np.save(scene_dir / "scaled_points.npy", scaled_points)
        np.save(scene_dir / "poses_scaled.npy", np.array(scaled_poses))

        self._emit(86, "Preparing scaled point cloud")
        filter_stats = {}
        outlier_method = str(fast3r_config.get("outlier_method", "radius_percentile"))
        filtered_points = scaled_points
        filtered_colors = sane_colors
        high_detail_mode = bool(fast3r_config.get("high_detail_mode", False))
        if outlier_method == "radius_percentile":
            filtered_points, filtered_colors, filter_stats = radius_percentile_filter(
                scaled_points,
                sane_colors,
                percentile=float(fast3r_config.get("radius_percentile", 99.0 if not high_detail_mode else 99.9)),
            )

        voxel_size = float(fast3r_config.get("voxel_size", 0.002))
        if high_detail_mode:
            voxel_size = 0.0
        down_points, down_colors = voxel_downsample(
            filtered_points,
            filtered_colors,
            voxel_size=voxel_size,
        )
        if len(down_points) == 0:
            raise RuntimeError(
                "Point cloud cleanup removed every point. "
                "Inspect raw_points.npy, lower the confidence threshold, or reduce target frames."
            )

        pointcloud_path = save_pointcloud_ply(scene_dir / "pointcloud.ply", down_points, down_colors)
        scale_to_meters = float(scale_report.get("scale_to_meters", 1.0))
        pointcloud_report = {
            "input_points": int(len(points)),
            "sane_points": int(len(sane_points)),
            "scaled_points": int(len(scaled_points)),
            "filtered_points": int(len(filtered_points)),
            "final_points": int(len(down_points)),
            "outlier_method": outlier_method,
            "filter_stats": filter_stats,
            "voxel_size_requested": float(fast3r_config.get("voxel_size", 0.002)),
            "voxel_size_effective": float(voxel_size),
            "high_detail_mode": bool(high_detail_mode),
        }
        (scene_dir / "pointcloud_processing_report.json").write_text(
            json.dumps(pointcloud_report, indent=2),
            encoding="utf-8",
        )

        self._emit(92, "Building Gaussian splat output")
        gaussian_report = build_gaussian_splats(
            down_points,
            down_colors,
            output_dir=scene_dir / "gaussian_splat",
            config=fast3r_config,
            scale_to_meters=scale_to_meters,
        )

        mesh_report = {"enabled": False, "reason": "disabled_by_config", "output_dir": str(scene_dir / "mesh")}
        if bool(fast3r_config.get("mesh_enabled", True)):
            self._emit(96, "Building Poisson mesh output")
            mesh_report = build_poisson_mesh(
                down_points,
                down_colors,
                output_dir=scene_dir / "mesh",
                config=fast3r_config,
                scale_to_meters=scale_to_meters,
            )
        mesh_path = Path(mesh_report["mesh_ply"]) if mesh_report.get("mesh_ply") else None

        metadata = {
            "stub_mode": False,
            "frame_selection": frame_metadata,
            "num_frames": len(frames),
            "num_points_raw": int(len(points)),
            "num_points_sane": int(len(sane_points)),
            "num_points_scaled": int(len(scaled_points)),
            "num_points_filtered": int(len(filtered_points)),
            "num_points_final": int(len(down_points)),
            "numeric_sanity_filter": sanity_stats,
            "fast3r_reconstruction": reconstruction_metadata,
            "pointcloud_processing": pointcloud_report,
            "filter_method": outlier_method,
            "filter_stats": filter_stats,
            "scale_normalization": scale_report,
            "gaussian_splat": gaussian_report,
            "mesh_branch": mesh_report,
            "outputs": {
                "pointcloud_ply": str(pointcloud_path),
                "gaussian_ply": gaussian_report.get("gaussian_ply", ""),
                "gaussian_npz": gaussian_report.get("gaussian_npz", ""),
                "mesh_ply": mesh_report.get("mesh_ply", ""),
                "mesh_obj": mesh_report.get("mesh_obj", ""),
            },
            "settings_used": fast3r_config,
            "warnings": list(frame_metadata.get("warnings", [])),
        }
        (scene_dir / "scene_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._emit(100, "Scene reconstruction complete")

        pose_records = [{"index": idx, "matrix": np.asarray(pose).tolist()} for idx, pose in enumerate(scaled_poses)]
        return SceneResult(
            scene_id=scene_id,
            source_video=source_video,
            output_dir=scene_dir,
            pointcloud_path=pointcloud_path,
            mesh_path=mesh_path,
            camera_poses=pose_records,
            metadata=metadata,
        )

    def cleanup(self) -> None:
        self.logger.info("Fast3rPipeline cleanup complete.")
