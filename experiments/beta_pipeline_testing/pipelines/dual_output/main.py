"""Primary beta pipeline: scaled Gaussian splats plus Poisson mesh output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ..shared.config import FrameExtractionConfig
from ..shared.gaussian_splats import GaussianSplatConfig, gaussian_splat_branch
from ..shared.mesh_reconstruction import MeshBranchConfig, mesh_branch
from ..shared.metrics import (
    MethodMetricsLogger,
    collect_system_info,
    compute_frame_metrics,
    compute_pointcloud_geometry_metrics,
    compute_pointcloud_nearest_neighbor_metrics,
    get_peak_gpu_memory_gb,
    reset_peak_gpu_memory_stats,
)
from ..shared.paths import DEFAULT_VIDEO_PATH, OUTPUTS_ROOT
from ..shared.pointcloud_ops import PointCloudProcessingConfig, make_pointcloud, process_pointcloud
from ..shared.reconstruction_data import ReconstructionOutput
from ..shared.scale_normalization import (
    ReferenceMeasurement,
    ScaleNormalizationConfig,
    scale_normalization,
)
from ..shared.video_processor import FrameExtractor

try:
    import open3d as o3d

    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


@dataclass
class DualOutputPipelineConfig:
    """Configuration for the primary dual-output experimental pipeline."""

    model: str = "fast3r"
    video_path: Path = DEFAULT_VIDEO_PATH
    output_dir: Path = OUTPUTS_ROOT / "dual_output"
    selected_frames_dir: Optional[Path] = None

    frame_selection: FrameExtractionConfig = field(
        default_factory=lambda: FrameExtractionConfig(
            num_frames=100,
            skip_frames=30,
            target_size=(1024, 768),
            uniform_sampling=False,
            selection_mode="overlap_aware",
            scan_stride=15,
            max_frame_gap=120,
        )
    )

    dust3r_image_size: int = 512
    dust3r_global_alignment_iters: int = 200
    fast3r_image_size: int = 512
    fast3r_niter_pnp: int = 100
    fast3r_dtype: str = "float32"
    fast3r_confidence_keep_ratio: float = 1.0
    fast3r_view_conf_p50_min: float = 0.0
    fast3r_view_conf_p90_min: float = 0.0
    fast3r_weak_texture_retention: bool = True
    fast3r_weak_texture_percentile: float = 35.0
    fast3r_weak_texture_min_conf_thr: float = 0.55
    fast3r_weak_texture_keep_ratio: float = 0.35
    min_conf_thr: float = 1.0

    scale: ScaleNormalizationConfig = field(default_factory=ScaleNormalizationConfig)
    pointcloud: PointCloudProcessingConfig = field(
        default_factory=lambda: PointCloudProcessingConfig(
            statistical_outlier_removal=False,
            voxel_size=0.0,
        )
    )
    gaussian: GaussianSplatConfig = field(default_factory=GaussianSplatConfig)
    mesh: MeshBranchConfig = field(default_factory=lambda: MeshBranchConfig(enabled=False))

    save_intermediate: bool = True
    save_metrics: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        self.output_dir = Path(self.output_dir)
        if self.selected_frames_dir is not None:
            self.selected_frames_dir = Path(self.selected_frames_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class DualOutputPipeline:
    """Frame selection -> inference -> scale normalization -> dual outputs."""

    def __init__(self, config: Optional[DualOutputPipelineConfig] = None):
        self.config = config or DualOutputPipelineConfig()
        method_name = f"dual_output_{self.config.model}"
        self.metrics_logger = MethodMetricsLogger(method_name) if self.config.save_metrics else None

        self.frames: list[np.ndarray] = []
        self.frame_metadata: dict = {}
        self.raw_reconstruction: Optional[ReconstructionOutput] = None
        self.scaled_reconstruction: Optional[ReconstructionOutput] = None
        self.processed_reconstruction: Optional[ReconstructionOutput] = None
        self.scale_report: dict = {}
        self.pointcloud_report: dict = {}
        self.gaussian_report: dict = {}
        self.mesh_report: dict = {}

        self._log("=" * 70)
        self._log("DUAL-OUTPUT RECONSTRUCTION PIPELINE")
        self._log("=" * 70)
        self._log(f"Model: {self.config.model}")
        self._log(f"Output: {self.config.output_dir}")

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def _run_stage(self, stage_name: str, fn) -> bool:
        reset_peak_gpu_memory_stats()
        started = time.perf_counter()
        success = False
        try:
            success = bool(fn())
            return success
        finally:
            elapsed = time.perf_counter() - started
            if self.metrics_logger is not None:
                self.metrics_logger.append_row(
                    "system",
                    "stage_timings.csv",
                    {
                        "stage_name": stage_name,
                        "elapsed_seconds": float(elapsed),
                        "gpu_peak_memory_gb": get_peak_gpu_memory_gb(),
                        "success": bool(success),
                    },
                )

    def _record_system_info(self) -> None:
        if self.metrics_logger is None:
            return
        row = collect_system_info()
        row.update(
            {
                "model": self.config.model,
                "video_path": self.config.video_path,
                "output_dir": self.config.output_dir,
                "selected_frames_dir": self.config.selected_frames_dir or "",
                "num_frames": int(self.config.frame_selection.num_frames),
                "skip_frames": int(self.config.frame_selection.skip_frames),
                "target_width": int(self.config.frame_selection.target_size[0]),
                "target_height": int(self.config.frame_selection.target_size[1]),
                "min_conf_thr": float(self.config.min_conf_thr),
                "fast3r_confidence_keep_ratio": float(self.config.fast3r_confidence_keep_ratio),
                "fast3r_view_conf_p50_min": float(self.config.fast3r_view_conf_p50_min),
                "fast3r_view_conf_p90_min": float(self.config.fast3r_view_conf_p90_min),
                "fast3r_weak_texture_retention": bool(self.config.fast3r_weak_texture_retention),
                "fast3r_weak_texture_min_conf_thr": float(self.config.fast3r_weak_texture_min_conf_thr),
                "fast3r_weak_texture_keep_ratio": float(self.config.fast3r_weak_texture_keep_ratio),
                "frame_selection_mode": self.config.frame_selection.selection_mode,
            }
        )
        self.metrics_logger.append_row("run_overview", "system_info.csv", row)

    def run(self) -> bool:
        started = time.perf_counter()
        self._record_system_info()
        try:
            stages = [
                ("frame_selection", self._frame_selection),
                ("inference", self._inference),
                ("scale_normalization", self._scale_normalization),
                ("shared_pointcloud_processing", self._shared_pointcloud_processing),
                ("gaussian_splat_branch", self._gaussian_splat_branch),
                ("mesh_branch", self._mesh_branch),
                ("save_run_summary", lambda: self._save_run_summary("success", time.perf_counter() - started)),
            ]

            for stage_name, stage_fn in stages:
                self._log("\n" + "=" * 70)
                self._log(stage_name.upper())
                self._log("=" * 70)
                if not self._run_stage(stage_name, stage_fn):
                    self._save_run_summary("failed", time.perf_counter() - started, f"{stage_name}_failed")
                    return False

            self._print_summary()
            return True
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            import traceback

            traceback.print_exc()
            self._save_run_summary("failed", time.perf_counter() - started, str(exc))
            return False

    def _frame_selection(self) -> bool:
        """Preserve the existing selected-frame input stage."""

        if self.config.selected_frames_dir is not None:
            self.frames, self.frame_metadata = self._load_selected_frames(self.config.selected_frames_dir)
        else:
            extractor = FrameExtractor(self.config.frame_selection)
            self.frames, self.frame_metadata = extractor.extract_frames(self.config.video_path)

        if not self.frames:
            raise RuntimeError("No frames were selected for inference.")

        if self.metrics_logger is not None:
            frame_rows, pair_rows = compute_frame_metrics(self.frames, self.frame_metadata)
            self.metrics_logger.append_rows("frames", "frame_metrics.csv", frame_rows)
            self.metrics_logger.append_rows("frames", "frame_pair_metrics.csv", pair_rows)
            self.metrics_logger.append_row(
                "frames",
                "frame_selection_summary.csv",
                {
                    "source": "selected_frames_dir" if self.config.selected_frames_dir else "video_extractor",
                    "selected_frames_dir": self.config.selected_frames_dir or "",
                    "frames_selected": int(len(self.frames)),
                    "video_path": self.config.video_path,
                    "frame_indices": "|".join(str(v) for v in self.frame_metadata.get("frame_indices", [])),
                    "selection_report": json.dumps(
                        self.frame_metadata.get("selection_report", {}),
                        default=str,
                    ),
                },
            )

        if self.config.save_intermediate:
            frames_dir = self.config.output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for idx, frame in enumerate(self.frames):
                cv2.imwrite(str(frames_dir / f"frame_{idx:04d}.jpg"), frame)
            (self.config.output_dir / "frame_selection_report.json").write_text(
                json.dumps(self.frame_metadata, indent=2, default=str),
                encoding="utf-8",
            )
            (self.config.output_dir / "selected_frame_indices.txt").write_text(
                "\n".join(str(v) for v in self.frame_metadata.get("frame_indices", [])),
                encoding="utf-8",
            )

        self._log(f"Frames ready for inference: {len(self.frames)}")
        return True

    def _load_selected_frames(self, frames_dir: Path) -> tuple[list[np.ndarray], dict]:
        if not frames_dir.exists():
            raise FileNotFoundError(f"Selected frames directory not found: {frames_dir}")
        image_paths = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            image_paths.extend(frames_dir.glob(pattern))
        image_paths = sorted(image_paths)
        if not image_paths:
            raise RuntimeError(f"No image frames found in: {frames_dir}")

        frames = []
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)

        h, w = frames[0].shape[:2] if frames else (0, 0)
        metadata = {
            "fps": 0.0,
            "total_frames": len(frames),
            "extracted_frames": len(frames),
            "frame_indices": list(range(len(frames))),
            "original_resolution": (w, h),
            "target_resolution": (w, h),
            "source": str(frames_dir),
        }
        return frames, metadata

    def _inference(self) -> bool:
        if self.config.model == "dust3r":
            points, colors, poses, metadata = self._run_dust3r()
        elif self.config.model == "fast3r":
            points, colors, poses, metadata = self._run_fast3r()
        else:
            raise ValueError(f"Unsupported model: {self.config.model}")

        self.raw_reconstruction = ReconstructionOutput(
            points=points,
            colors=colors,
            poses=poses,
            metadata=metadata,
        )
        if self.config.save_intermediate:
            np.save(self.config.output_dir / "raw_points.npy", points)
            np.save(self.config.output_dir / "raw_colors.npy", colors)
            np.save(self.config.output_dir / "poses_raw.npy", np.asarray(poses))
            retention = metadata.get("point_retention", {}) if isinstance(metadata, dict) else {}
            (self.config.output_dir / "fast3r_point_retention_report.json").write_text(
                json.dumps(retention, indent=2, default=str),
                encoding="utf-8",
            )

        self._log(f"Inference points: {len(points):,}")
        self._log(f"Camera poses: {len(poses) if poses is not None else 0}")
        return True

    def _run_dust3r(self):
        from ..dust3r.reconstruction import DUSt3RConfig, DUSt3RReconstructor

        reconstructor = DUSt3RReconstructor(
            DUSt3RConfig(
                device="cuda" if torch.cuda.is_available() else "cpu",
                image_size=int(self.config.dust3r_image_size),
                min_conf_thr=float(self.config.min_conf_thr),
                niter=int(self.config.dust3r_global_alignment_iters),
                verbose=bool(self.config.verbose),
            ),
            metrics_logger=self.metrics_logger,
        )
        points, colors, poses = reconstructor.reconstruct_from_frames(self.frames, output_dir=self.config.output_dir)
        metadata = dict(getattr(reconstructor, "last_metadata", {}))
        metadata.update({"frame_metadata": self.frame_metadata})
        return points, colors, poses, metadata

    def _run_fast3r(self):
        from ..fast3r.reconstruction import Fast3RConfig, Fast3RReconstructor

        reconstructor = Fast3RReconstructor(
            Fast3RConfig(
                device="cuda" if torch.cuda.is_available() else "cpu",
                image_size=int(self.config.fast3r_image_size),
                min_conf_thr=float(self.config.min_conf_thr),
                niter_pnp=int(self.config.fast3r_niter_pnp),
                dtype=self.config.fast3r_dtype,
                confidence_keep_ratio=float(self.config.fast3r_confidence_keep_ratio),
                view_conf_p50_min=float(self.config.fast3r_view_conf_p50_min),
                view_conf_p90_min=float(self.config.fast3r_view_conf_p90_min),
                weak_texture_retention=bool(self.config.fast3r_weak_texture_retention),
                weak_texture_percentile=float(self.config.fast3r_weak_texture_percentile),
                weak_texture_min_conf_thr=float(self.config.fast3r_weak_texture_min_conf_thr),
                weak_texture_keep_ratio=float(self.config.fast3r_weak_texture_keep_ratio),
                verbose=bool(self.config.verbose),
            )
        )
        points, colors, poses = reconstructor.reconstruct_from_frames(self.frames, output_dir=self.config.output_dir)
        metadata = dict(getattr(reconstructor, "last_metadata", {}))
        metadata.update({"frame_metadata": self.frame_metadata})
        return points, colors, poses, metadata

    def _scale_normalization(self) -> bool:
        if self.raw_reconstruction is None:
            raise RuntimeError("Inference must run before scale normalization.")
        self.scaled_reconstruction, self.scale_report = scale_normalization(
            self.raw_reconstruction,
            self.config.scale,
            output_dir=self.config.output_dir,
        )
        if self.config.save_intermediate:
            np.save(self.config.output_dir / "scaled_points.npy", self.scaled_reconstruction.points)
            np.save(self.config.output_dir / "poses_scaled.npy", np.asarray(self.scaled_reconstruction.poses or []))
            if O3D_AVAILABLE:
                pcd = make_pointcloud(self.scaled_reconstruction.points, self.scaled_reconstruction.colors)
                o3d.io.write_point_cloud(str(self.config.output_dir / "scaled_pointcloud_raw.ply"), pcd)

        if self.metrics_logger is not None:
            self.metrics_logger.append_row("scale", "scale_normalization.csv", dict(self.scale_report))
        self._log(f"Applied scale_to_meters: {self.scale_report.get('scale_to_meters', 1.0):.6f}")
        self._log(f"Scale source: {self.scale_report.get('source')}")
        return True

    def _shared_pointcloud_processing(self) -> bool:
        if self.scaled_reconstruction is None:
            raise RuntimeError("Scale normalization must run before point cloud processing.")
        points, colors, report = process_pointcloud(
            self.scaled_reconstruction.points,
            self.scaled_reconstruction.colors,
            self.config.pointcloud,
        )
        self.pointcloud_report = report
        self.processed_reconstruction = ReconstructionOutput(
            points=points,
            colors=colors,
            poses=self.scaled_reconstruction.poses,
            intrinsics=self.scaled_reconstruction.intrinsics,
            metadata=dict(self.scaled_reconstruction.metadata),
            scale_applied=float(self.scaled_reconstruction.scale_applied),
        )

        if self.config.save_intermediate:
            np.save(self.config.output_dir / "processed_scaled_points.npy", points)
            np.save(self.config.output_dir / "processed_scaled_colors.npy", colors)
            (self.config.output_dir / "pointcloud_processing_report.json").write_text(
                json.dumps(report, indent=2, default=str),
                encoding="utf-8",
            )
            if O3D_AVAILABLE:
                pcd = make_pointcloud(points, colors)
                o3d.io.write_point_cloud(str(self.config.output_dir / "processed_scaled_pointcloud.ply"), pcd)

        if self.metrics_logger is not None:
            geom = compute_pointcloud_geometry_metrics(points, colors)
            geom.update(compute_pointcloud_nearest_neighbor_metrics(points))
            geom["stage_name"] = "processed_scaled_pointcloud"
            self.metrics_logger.append_row("pointcloud", "pointcloud_geometry_metrics.csv", geom)
            self.metrics_logger.append_row("pointcloud", "pointcloud_stage_metrics.csv", dict(report))

        self._log(f"Processed scaled points: {len(points):,}")
        return True

    def _gaussian_splat_branch(self) -> bool:
        if self.processed_reconstruction is None:
            raise RuntimeError("Point cloud processing must run before output branches.")
        self.gaussian_report = gaussian_splat_branch(
            self.processed_reconstruction,
            output_root=self.config.output_dir,
            config=self.config.gaussian,
        )
        if self.metrics_logger is not None:
            self.metrics_logger.append_row("gaussian_splat", "gaussian_splat_metrics.csv", dict(self.gaussian_report))
        return True

    def _mesh_branch(self) -> bool:
        if self.processed_reconstruction is None:
            raise RuntimeError("Point cloud processing must run before output branches.")
        self.mesh_report = mesh_branch(
            self.processed_reconstruction,
            output_root=self.config.output_dir,
            config=self.config.mesh,
        )
        if self.metrics_logger is not None:
            mesh_row = dict(self.mesh_report)
            mesh_row.pop("mesh_metrics", None)
            mesh_row.update(self.mesh_report.get("mesh_metrics", {}))
            self.metrics_logger.append_row("mesh", "mesh_branch_metrics.csv", mesh_row)
        return True

    def _save_run_summary(self, status: str, total_runtime_seconds: float, error_message: str = "") -> bool:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "error_message": error_message,
            "total_runtime_seconds": float(total_runtime_seconds),
            "model": self.config.model,
            "video_path": str(self.config.video_path),
            "selected_frames_dir": str(self.config.selected_frames_dir) if self.config.selected_frames_dir else "",
            "frames": len(self.frames),
            "raw_points": int(len(self.raw_reconstruction.points)) if self.raw_reconstruction else 0,
            "scaled_points": int(len(self.scaled_reconstruction.points)) if self.scaled_reconstruction else 0,
            "processed_scaled_points": int(len(self.processed_reconstruction.points)) if self.processed_reconstruction else 0,
            "fast3r_point_retention": (
                self.raw_reconstruction.metadata.get("point_retention", {})
                if self.raw_reconstruction and isinstance(self.raw_reconstruction.metadata, dict)
                else {}
            ),
            "scale": self.scale_report,
            "pointcloud": self.pointcloud_report,
            "gaussian": self.gaussian_report,
            "mesh": self.mesh_report,
            "config": self._json_config(),
        }
        path = self.config.output_dir / "dual_output_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        geometry_report = {
            "frames": len(self.frames),
            "raw_points_after_fast3r_filtering": summary["raw_points"],
            "scaled_points_before_cleanup": summary["scaled_points"],
            "processed_scaled_points_after_cleanup": summary["processed_scaled_points"],
            "fast3r_point_retention": summary["fast3r_point_retention"],
            "pointcloud_processing": self.pointcloud_report,
            "notes": [
                "Compare predicted_dense_points to raw_points_after_fast3r_filtering to see confidence/view-filter loss.",
                "Compare scaled_points_before_cleanup to processed_scaled_points_after_cleanup to see cleanup loss.",
            ],
        }
        (self.config.output_dir / "geometry_retention_report.json").write_text(
            json.dumps(geometry_report, indent=2, default=str),
            encoding="utf-8",
        )
        if self.metrics_logger is not None:
            self.metrics_logger.append_row(
                "run_overview",
                "run_summary.csv",
                {
                    "status": status,
                    "error_message": error_message,
                    "total_runtime_seconds": float(total_runtime_seconds),
                    "model": self.config.model,
                    "frames": len(self.frames),
                    "raw_points": int(len(self.raw_reconstruction.points)) if self.raw_reconstruction else 0,
                    "processed_scaled_points": int(len(self.processed_reconstruction.points)) if self.processed_reconstruction else 0,
                    "scale_to_meters": float(self.scale_report.get("scale_to_meters", 1.0)),
                    "gaussian_ply": self.gaussian_report.get("gaussian_ply", ""),
                    "mesh_ply": self.mesh_report.get("mesh_ply", ""),
                },
            )
        return True

    def _json_config(self) -> dict:
        data = asdict(self.config)
        for key in ("video_path", "output_dir", "selected_frames_dir"):
            data[key] = str(data[key]) if data[key] is not None else None
        return data

    def _print_summary(self) -> None:
        self._log("\n" + "=" * 70)
        self._log("DUAL-OUTPUT RECONSTRUCTION COMPLETE")
        self._log("=" * 70)
        self._log(f"Scale source: {self.scale_report.get('source')}")
        self._log(f"Scale to meters: {self.scale_report.get('scale_to_meters', 1.0)}")
        self._log(f"Gaussian PLY: {self.gaussian_report.get('gaussian_ply', 'disabled')}")
        self._log(f"Mesh PLY: {self.mesh_report.get('mesh_ply', 'disabled')}")
        self._log(f"Summary: {self.config.output_dir / 'dual_output_summary.json'}")


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaled dual-output reconstruction pipeline.")
    parser.add_argument("--model", choices=["dust3r", "fast3r"], default="fast3r")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUTS_ROOT / "dual_output")
    parser.add_argument("--selected-frames-dir", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument(
        "--selection-mode",
        choices=["skip", "uniform", "overlap_aware", "coverage_aware"],
        default="coverage_aware",
        help="Frame selection strategy. coverage_aware spreads quality frames across the whole video.",
    )
    parser.add_argument("--scan-stride", type=int, default=15)
    parser.add_argument("--max-frame-gap", type=int, default=120)
    parser.add_argument("--min-laplacian", type=float, default=40.0)
    parser.add_argument("--max-clipped-ratio", type=float, default=0.12)
    parser.add_argument("--min-orb-matches", type=int, default=150)
    parser.add_argument("--min-thumbnail-corr", type=float, default=0.15)
    parser.add_argument("--overlap-window-size", type=int, default=3)
    parser.add_argument("--force-accept-after-gap", type=int, default=240)
    parser.add_argument("--force-accept-after-rejections", type=int, default=20)
    parser.add_argument("--output-width", type=int, default=1024)
    parser.add_argument("--output-height", type=int, default=768)
    parser.add_argument("--min-conf-thr", type=float, default=1.0)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.002,
        help="Shared voxel downsample after scaling. Default mirrors src Fast3R cleanup.",
    )
    parser.add_argument(
        "--high-detail-mode",
        type=_parse_bool,
        default=False,
        help="Preserve fine geometry by disabling voxel/subsample cleanup and relaxing radius trimming.",
    )
    parser.add_argument("--max-abs-coordinate", type=float, default=100.0)
    parser.add_argument("--radius-percentile", type=float, default=99.0)
    parser.add_argument("--scale-reference-name", type=str, default="room_height")
    parser.add_argument("--scale-reference-real", type=float, default=2.5)
    parser.add_argument("--scale-reference-axis", choices=["x", "y", "z", "height"], default="z")
    parser.add_argument("--scale-reference-measured", type=float, default=None)
    parser.add_argument("--fail-without-scale", type=_parse_bool, default=False)
    parser.add_argument("--no-gaussian", action="store_true")
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument("--mesh", action="store_true", help="Enable the Poisson mesh branch. Disabled by default for fast experiments.")
    parser.add_argument("--poisson-depth", type=int, default=10)
    parser.add_argument("--target-triangles", type=int, default=250000)
    parser.add_argument(
        "--fast3r-image-size",
        type=int,
        default=512,
        help="Fast3R inference image size. Try 768/1024 for sharper detail, with fewer frames to fit VRAM.",
    )
    parser.add_argument("--fast3r-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--confidence-keep-ratio",
        type=float,
        default=1.0,
        help="Keep top ratio of Fast3R confidence values per view after min-conf filtering. Example: 0.4 keeps top 40%%.",
    )
    parser.add_argument(
        "--view-conf-p50-min",
        type=float,
        default=0.0,
        help="Drop whole Fast3R views whose median confidence is below this. 0 disables view pruning.",
    )
    parser.add_argument(
        "--view-conf-p90-min",
        type=float,
        default=0.0,
        help="Drop whole Fast3R views whose p90 confidence is below this. 0 disables view pruning.",
    )
    parser.add_argument("--weak-texture-retention", type=_parse_bool, default=True)
    parser.add_argument("--weak-texture-percentile", type=float, default=35.0)
    parser.add_argument("--weak-texture-min-conf-thr", type=float, default=0.55)
    parser.add_argument("--weak-texture-keep-ratio", type=float, default=0.35)
    parser.add_argument("--gaussian-nn-scale-mult", type=float, default=0.7)
    parser.add_argument("--gaussian-min-scale", type=float, default=0.00025)
    parser.add_argument("--gaussian-max-scale", type=float, default=0.02)
    parser.add_argument("--gaussian-alpha", type=float, default=0.85)
    parser.add_argument("--gaussian-density-opacity", type=_parse_bool, default=True)
    parser.add_argument("--gaussian-min-alpha", type=float, default=0.35)
    parser.add_argument("--gaussian-max-alpha", type=float, default=0.85)
    parser.add_argument("--gaussian-scale-percentile-low", type=float, default=5.0)
    parser.add_argument("--gaussian-scale-percentile-high", type=float, default=95.0)
    parser.add_argument("--gaussian-surface-aligned", type=_parse_bool, default=True)
    parser.add_argument("--gaussian-tangent-scale-mult", type=float, default=1.15)
    parser.add_argument("--gaussian-normal-scale-mult", type=float, default=0.25)
    parser.add_argument("--gaussian-max-surface-aligned-points", type=int, default=8000000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_config = FrameExtractionConfig(
        num_frames=int(args.num_frames),
        skip_frames=int(args.skip_frames),
        target_size=(int(args.output_width), int(args.output_height)),
        uniform_sampling=args.selection_mode == "uniform",
        selection_mode=args.selection_mode,
        scan_stride=int(args.scan_stride),
        max_frame_gap=int(args.max_frame_gap),
        min_laplacian_variance=float(args.min_laplacian),
        max_clipped_ratio=float(args.max_clipped_ratio),
        min_orb_matches=int(args.min_orb_matches),
        min_thumbnail_corr=float(args.min_thumbnail_corr),
        overlap_window_size=int(args.overlap_window_size),
        force_accept_after_gap=int(args.force_accept_after_gap),
        force_accept_after_rejections=int(args.force_accept_after_rejections),
    )
    scale_config = ScaleNormalizationConfig(
        reference=ReferenceMeasurement(
            name=args.scale_reference_name,
            real_world_dimension=float(args.scale_reference_real),
            axis=args.scale_reference_axis,
            measured_dimension=args.scale_reference_measured,
        ),
        fail_without_scale=bool(args.fail_without_scale),
    )
    config = DualOutputPipelineConfig(
        model=args.model,
        video_path=args.video,
        output_dir=args.output,
        selected_frames_dir=args.selected_frames_dir,
        frame_selection=frame_config,
        min_conf_thr=float(args.min_conf_thr),
        fast3r_image_size=int(args.fast3r_image_size),
        fast3r_dtype=args.fast3r_dtype,
        fast3r_confidence_keep_ratio=float(args.confidence_keep_ratio),
        fast3r_view_conf_p50_min=float(args.view_conf_p50_min),
        fast3r_view_conf_p90_min=float(args.view_conf_p90_min),
        fast3r_weak_texture_retention=bool(args.weak_texture_retention),
        fast3r_weak_texture_percentile=float(args.weak_texture_percentile),
        fast3r_weak_texture_min_conf_thr=float(args.weak_texture_min_conf_thr),
        fast3r_weak_texture_keep_ratio=float(args.weak_texture_keep_ratio),
        scale=scale_config,
        pointcloud=PointCloudProcessingConfig(
            high_detail_mode=bool(args.high_detail_mode),
            max_abs_coordinate=float(args.max_abs_coordinate),
            radius_percentile=float(args.radius_percentile),
            statistical_outlier_removal=False,
            voxel_size=float(args.voxel_size),
        ),
        gaussian=GaussianSplatConfig(
            enabled=not args.no_gaussian,
            nn_scale_mult=float(args.gaussian_nn_scale_mult),
            min_scale=float(args.gaussian_min_scale),
            max_scale=float(args.gaussian_max_scale),
            alpha=float(args.gaussian_alpha),
            density_opacity=bool(args.gaussian_density_opacity),
            min_alpha=float(args.gaussian_min_alpha),
            max_alpha=float(args.gaussian_max_alpha),
            scale_percentile_low=float(args.gaussian_scale_percentile_low),
            scale_percentile_high=float(args.gaussian_scale_percentile_high),
            surface_aligned=bool(args.gaussian_surface_aligned),
            tangent_scale_mult=float(args.gaussian_tangent_scale_mult),
            normal_scale_mult=float(args.gaussian_normal_scale_mult),
            max_surface_aligned_points=int(args.gaussian_max_surface_aligned_points),
        ),
        mesh=MeshBranchConfig(
            enabled=bool(args.mesh and not args.no_mesh),
            poisson_depth=int(args.poisson_depth),
            target_triangles=int(args.target_triangles),
        ),
    )
    return 0 if DualOutputPipeline(config).run() else 1


if __name__ == "__main__":
    sys.exit(main())
