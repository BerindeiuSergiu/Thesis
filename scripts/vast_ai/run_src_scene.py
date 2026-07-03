from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def _apply(config: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        config[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the production Fast3R pipeline without the PyQt UI.")
    parser.add_argument("--video", type=Path, default=Path("data/raw/irl_room_video_2.mp4"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/fast3r_arkitscenes_ga_head_hf"))
    parser.add_argument("--model-profile", choices=["default", "thesis_ga_head"], default="thesis_ga_head")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs_vast"))
    parser.add_argument("--target-frames", type=int, default=180)
    parser.add_argument("--probe-max-frames", type=int, default=180)
    parser.add_argument("--visual-shortlist-target", type=int, default=220)
    parser.add_argument("--fast3r-image-size", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--mesh", action="store_true")

    parser.add_argument("--prefilter-enabled", type=_str_to_bool, default=None)
    parser.add_argument("--selection-mode", choices=["geometry_aware", "prefilter"], default=None)
    parser.add_argument("--scan-stride", type=int, default=None)
    parser.add_argument("--min-frame-gap", type=int, default=None)
    parser.add_argument("--shortlist-multiplier", type=int, default=None)
    parser.add_argument("--dedupe-similarity", type=float, default=None)
    parser.add_argument("--probe-image-size", type=int, default=None)
    parser.add_argument("--probe-confidence-threshold", type=float, default=None)
    parser.add_argument("--probe-pnp-iters", type=int, default=None)
    parser.add_argument("--probe-max-parallel-views", type=int, default=None)

    parser.add_argument("--min-conf-thr", "--min-confidence-threshold", dest="min_confidence_threshold", type=float, default=None)
    parser.add_argument("--confidence-keep-ratio", type=float, default=None)
    parser.add_argument("--view-conf-p50-min", type=float, default=None)
    parser.add_argument("--view-conf-p90-min", type=float, default=None)
    parser.add_argument("--weak-texture-retention", type=_str_to_bool, default=None)
    parser.add_argument("--weak-texture-percentile", type=float, default=None)
    parser.add_argument("--weak-texture-min-conf-thr", type=float, default=None)
    parser.add_argument("--weak-texture-keep-ratio", type=float, default=None)

    parser.add_argument("--max-abs-coordinate", type=float, default=None)
    parser.add_argument("--outlier-method", default=None)
    parser.add_argument("--radius-percentile", type=float, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--high-detail-mode", type=_str_to_bool, default=None)

    parser.add_argument("--scale-enabled", type=_str_to_bool, default=None)
    parser.add_argument("--scale-reference-real", type=float, default=None)
    parser.add_argument("--scale-reference-measured", type=float, default=None)
    parser.add_argument("--scale-reference-axis", choices=["x", "y", "z"], default=None)
    parser.add_argument("--fail-without-scale", type=_str_to_bool, default=None)

    parser.add_argument("--gaussian-enabled", type=_str_to_bool, default=None)
    parser.add_argument("--gaussian-voxel-size", type=float, default=None)
    parser.add_argument("--gaussian-normal-radius", type=float, default=None)
    parser.add_argument("--gaussian-normal-max-nn", type=int, default=None)
    parser.add_argument("--gaussian-nn-scale-mult", type=float, default=None)
    parser.add_argument("--gaussian-min-scale", type=float, default=None)
    parser.add_argument("--gaussian-max-scale", type=float, default=None)
    parser.add_argument("--gaussian-alpha", type=float, default=None)
    parser.add_argument("--gaussian-density-opacity", type=_str_to_bool, default=None)
    parser.add_argument("--gaussian-min-alpha", type=float, default=None)
    parser.add_argument("--gaussian-max-alpha", type=float, default=None)
    parser.add_argument("--gaussian-scale-percentile-low", type=float, default=None)
    parser.add_argument("--gaussian-scale-percentile-high", type=float, default=None)
    parser.add_argument("--gaussian-surface-aligned", type=_str_to_bool, default=None)
    parser.add_argument("--gaussian-max-surface-aligned-points", type=int, default=None)

    parser.add_argument("--dump-runtime-config", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "src/config/settings.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    app_config = dict(config.get("app", {}))
    app_config["outputs_root"] = str(args.outputs_root)
    config["app"] = app_config

    pipeline_config = dict(config.get("pipeline", {}))
    pipeline_config["stub_mode"] = False
    fast3r = dict(pipeline_config.get("fast3r", {}))
    fast3r["auto_preset_enabled"] = False
    model_dir = args.model_dir
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    fast3r["active_model_profile"] = args.model_profile
    if args.model_profile == "thesis_ga_head":
        fast3r["model_name"] = str(model_dir)
        fast3r["probe_model_name"] = str(model_dir)
    fast3r["target_frames"] = int(args.target_frames)
    fast3r["probe_max_frames"] = int(args.probe_max_frames)
    fast3r["visual_shortlist_target"] = int(args.visual_shortlist_target)
    fast3r["fast3r_image_size"] = int(args.fast3r_image_size)
    fast3r["dtype"] = args.dtype
    fast3r["probe_dtype"] = args.dtype
    fast3r["mesh_enabled"] = bool(args.mesh)
    profiles = dict(fast3r.get("model_profiles", {}))
    if args.model_profile == "thesis_ga_head":
        profiles["thesis_ga_head"] = {
            "label": "Fine-Tuned Fast3R",
            "model_name": str(model_dir),
            "probe_model_name": str(model_dir),
            "source_type": "local",
        }
        fast3r["model_profiles"] = profiles

    _apply(fast3r, "prefilter_enabled", args.prefilter_enabled)
    _apply(fast3r, "selection_mode", args.selection_mode)
    _apply(fast3r, "scan_stride", args.scan_stride)
    _apply(fast3r, "min_frame_gap", args.min_frame_gap)
    _apply(fast3r, "shortlist_multiplier", args.shortlist_multiplier)
    _apply(fast3r, "dedupe_similarity", args.dedupe_similarity)
    _apply(fast3r, "probe_image_size", args.probe_image_size)
    _apply(fast3r, "probe_confidence_threshold", args.probe_confidence_threshold)
    _apply(fast3r, "probe_pnp_iters", args.probe_pnp_iters)
    _apply(fast3r, "probe_max_parallel_views", args.probe_max_parallel_views)

    _apply(fast3r, "min_confidence_threshold", args.min_confidence_threshold)
    _apply(fast3r, "confidence_keep_ratio", args.confidence_keep_ratio)
    _apply(fast3r, "view_conf_p50_min", args.view_conf_p50_min)
    _apply(fast3r, "view_conf_p90_min", args.view_conf_p90_min)
    _apply(fast3r, "weak_texture_retention", args.weak_texture_retention)
    _apply(fast3r, "weak_texture_percentile", args.weak_texture_percentile)
    _apply(fast3r, "weak_texture_min_conf_thr", args.weak_texture_min_conf_thr)
    _apply(fast3r, "weak_texture_keep_ratio", args.weak_texture_keep_ratio)

    _apply(fast3r, "max_abs_coordinate", args.max_abs_coordinate)
    _apply(fast3r, "outlier_method", args.outlier_method)
    _apply(fast3r, "radius_percentile", args.radius_percentile)
    _apply(fast3r, "voxel_size", args.voxel_size)
    _apply(fast3r, "high_detail_mode", args.high_detail_mode)

    _apply(fast3r, "scale_enabled", args.scale_enabled)
    _apply(fast3r, "scale_reference_real", args.scale_reference_real)
    _apply(fast3r, "scale_reference_measured", args.scale_reference_measured)
    _apply(fast3r, "scale_reference_axis", args.scale_reference_axis)
    _apply(fast3r, "fail_without_scale", args.fail_without_scale)

    _apply(fast3r, "gaussian_enabled", args.gaussian_enabled)
    _apply(fast3r, "gaussian_voxel_size", args.gaussian_voxel_size)
    _apply(fast3r, "gaussian_normal_radius", args.gaussian_normal_radius)
    _apply(fast3r, "gaussian_normal_max_nn", args.gaussian_normal_max_nn)
    _apply(fast3r, "gaussian_nn_scale_mult", args.gaussian_nn_scale_mult)
    _apply(fast3r, "gaussian_min_scale", args.gaussian_min_scale)
    _apply(fast3r, "gaussian_max_scale", args.gaussian_max_scale)
    _apply(fast3r, "gaussian_alpha", args.gaussian_alpha)
    _apply(fast3r, "gaussian_density_opacity", args.gaussian_density_opacity)
    _apply(fast3r, "gaussian_min_alpha", args.gaussian_min_alpha)
    _apply(fast3r, "gaussian_max_alpha", args.gaussian_max_alpha)
    _apply(fast3r, "gaussian_scale_percentile_low", args.gaussian_scale_percentile_low)
    _apply(fast3r, "gaussian_scale_percentile_high", args.gaussian_scale_percentile_high)
    _apply(fast3r, "gaussian_surface_aligned", args.gaussian_surface_aligned)
    _apply(fast3r, "gaussian_max_surface_aligned_points", args.gaussian_max_surface_aligned_points)

    pipeline_config["fast3r"] = fast3r
    config["pipeline"] = pipeline_config

    if args.dry_run_config:
        print(json.dumps({"pipeline": {"fast3r": fast3r}}, indent=2, default=str))
        return 0

    video = args.video
    if not video.is_absolute():
        video = repo_root / video
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if args.model_profile == "thesis_ga_head" and not model_dir.exists():
        raise FileNotFoundError(f"Converted model folder not found: {model_dir}")

    from src.pipeline.fast3r_pipeline import Fast3rPipeline

    pipeline = Fast3rPipeline()
    if args.dump_runtime_config:
        resolved = pipeline._apply_fast3r_model_profile(dict(fast3r))
        print(json.dumps({"pipeline": {"fast3r": resolved}}, indent=2, default=str))

    last_progress: dict[str, Any] = {"pct": None, "msg": None}

    def print_progress(pct: int, msg: str) -> None:
        if pct == last_progress["pct"] and msg == last_progress["msg"]:
            return
        last_progress["pct"] = pct
        last_progress["msg"] = msg
        print(f"[{pct:03d}%] {msg}", flush=True)

    pipeline.set_progress_callback(print_progress)
    pipeline.load(config)
    result = pipeline.run(str(video))

    print(json.dumps(result.to_record(), indent=2, default=str))
    print(f"OUTPUT_DIR={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
