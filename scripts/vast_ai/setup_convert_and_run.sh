#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV="${VENV:-.venv_vast}"
VIDEO="${VIDEO:-data/raw/irl_room_video_2.mp4}"
RUN_DIR="${RUN_DIR:-experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head}"
BEST_CKPT="${BEST_CKPT:-epoch_000.ckpt}"
MODEL_OUT="${MODEL_OUT:-data/models/fast3r_arkitscenes_ga_head_hf}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs_vast}"
TARGET_FRAMES="${TARGET_FRAMES:-180}"
PROBE_MAX_FRAMES="${PROBE_MAX_FRAMES:-180}"
VISUAL_SHORTLIST_TARGET="${VISUAL_SHORTLIST_TARGET:-220}"
FAST3R_IMAGE_SIZE="${FAST3R_IMAGE_SIZE:-512}"
FAST3R_DTYPE="${FAST3R_DTYPE:-bfloat16}"
ENABLE_MESH="${ENABLE_MESH:-0}"
MODEL_PROFILE="${MODEL_PROFILE:-thesis_ga_head}"

# Match the local high_quality_detail application profile explicitly. This keeps
# Vast runs stable even if settings.yaml was edited on the VM.
PREFILTER_ENABLED="${PREFILTER_ENABLED:-true}"
SELECTION_MODE="${SELECTION_MODE:-geometry_aware}"
SCAN_STRIDE="${SCAN_STRIDE:-10}"
MIN_FRAME_GAP="${MIN_FRAME_GAP:-10}"
MIN_CONFIDENCE_THRESHOLD="${MIN_CONFIDENCE_THRESHOLD:-1.05}"
CONFIDENCE_KEEP_RATIO="${CONFIDENCE_KEEP_RATIO:-0.85}"
VIEW_CONF_P50_MIN="${VIEW_CONF_P50_MIN:-1.2}"
VIEW_CONF_P90_MIN="${VIEW_CONF_P90_MIN:-1.45}"
WEAK_TEXTURE_RETENTION="${WEAK_TEXTURE_RETENTION:-true}"
WEAK_TEXTURE_PERCENTILE="${WEAK_TEXTURE_PERCENTILE:-35.0}"
WEAK_TEXTURE_MIN_CONF_THR="${WEAK_TEXTURE_MIN_CONF_THR:-0.55}"
WEAK_TEXTURE_KEEP_RATIO="${WEAK_TEXTURE_KEEP_RATIO:-0.35}"
OUTLIER_METHOD="${OUTLIER_METHOD:-radius_percentile}"
RADIUS_PERCENTILE="${RADIUS_PERCENTILE:-99.7}"
VOXEL_SIZE="${VOXEL_SIZE:-0.001}"
HIGH_DETAIL_MODE="${HIGH_DETAIL_MODE:-false}"
SCALE_ENABLED="${SCALE_ENABLED:-true}"
SCALE_REFERENCE_REAL="${SCALE_REFERENCE_REAL:-2.5}"
SCALE_REFERENCE_AXIS="${SCALE_REFERENCE_AXIS:-z}"
FAIL_WITHOUT_SCALE="${FAIL_WITHOUT_SCALE:-false}"
GAUSSIAN_ENABLED="${GAUSSIAN_ENABLED:-true}"
GAUSSIAN_VOXEL_SIZE="${GAUSSIAN_VOXEL_SIZE:-0.0}"
GAUSSIAN_NORMAL_RADIUS="${GAUSSIAN_NORMAL_RADIUS:-0.0}"
GAUSSIAN_NORMAL_MAX_NN="${GAUSSIAN_NORMAL_MAX_NN:-64}"
GAUSSIAN_NN_SCALE_MULT="${GAUSSIAN_NN_SCALE_MULT:-0.35}"
GAUSSIAN_MIN_SCALE="${GAUSSIAN_MIN_SCALE:-0.00025}"
GAUSSIAN_MAX_SCALE="${GAUSSIAN_MAX_SCALE:-0.008}"
GAUSSIAN_ALPHA="${GAUSSIAN_ALPHA:-0.85}"
GAUSSIAN_DENSITY_OPACITY="${GAUSSIAN_DENSITY_OPACITY:-true}"
GAUSSIAN_MIN_ALPHA="${GAUSSIAN_MIN_ALPHA:-0.35}"
GAUSSIAN_MAX_ALPHA="${GAUSSIAN_MAX_ALPHA:-0.82}"
GAUSSIAN_SCALE_PERCENTILE_LOW="${GAUSSIAN_SCALE_PERCENTILE_LOW:-5.0}"
GAUSSIAN_SCALE_PERCENTILE_HIGH="${GAUSSIAN_SCALE_PERCENTILE_HIGH:-85.0}"
GAUSSIAN_SURFACE_ALIGNED="${GAUSSIAN_SURFACE_ALIGNED:-false}"
GAUSSIAN_MAX_SURFACE_ALIGNED_POINTS="${GAUSSIAN_MAX_SURFACE_ALIGNED_POINTS:-8000000}"

echo "[vast] Repo: $(pwd)"
echo "[vast] Video: $VIDEO"
echo "[vast] Run dir: $RUN_DIR"
echo "[vast] Best checkpoint: $BEST_CKPT"
echo "[vast] Model output: $MODEL_OUT"

if [[ ! -f "$VIDEO" ]]; then
  echo "Missing video: $VIDEO" >&2
  exit 1
fi

if [[ ! -d "$RUN_DIR/checkpoints" ]]; then
  echo "Missing checkpoints under: $RUN_DIR/checkpoints" >&2
  exit 1
fi

if [[ ! -d "$RUN_DIR/checkpoints/$BEST_CKPT" ]]; then
  echo "Missing best checkpoint: $RUN_DIR/checkpoints/$BEST_CKPT" >&2
  exit 1
fi

if [[ ! -f "$RUN_DIR/.hydra/config.yaml" ]]; then
  echo "[vast] Creating $RUN_DIR/.hydra/config.yaml from csv/version_0/hparams.yaml"
  mkdir -p "$RUN_DIR/.hydra"
  cp "$RUN_DIR/csv/version_0/hparams.yaml" "$RUN_DIR/.hydra/config.yaml"
fi

if [[ "$BEST_CKPT" != "last.ckpt" ]]; then
  echo "[vast] Pointing checkpoints/last.ckpt to $BEST_CKPT for the Fast3R export helper"
  rm -rf "$RUN_DIR/checkpoints/last.ckpt"
  ln -s "$BEST_CKPT" "$RUN_DIR/checkpoints/last.ckpt"
fi

if [[ ! -d "$VENV" ]]; then
  echo "[vast] Creating venv: $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel packaging "setuptools<81"

if python - <<'PY'
import sys
try:
    import torch
    print(f"[vast] torch already installed: {torch.__version__}, cuda={torch.cuda.is_available()}")
except Exception:
    sys.exit(42)
PY
then
  :
else
  echo "[vast] Installing PyTorch CUDA 12.1 wheels"
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

echo "[vast] Installing runtime dependencies"
python -m pip install \
  numpy==1.26.4 scipy scikit-learn scikit-image opencv-python pillow matplotlib tqdm \
  einops roma rich rootutils torchinfo hydra-core hydra-colorlog hydra-optuna-sweeper \
  omegaconf lightning lightning-bolts lightning-utilities torchmetrics transformers timm \
  huggingface-hub safetensors pyyaml open3d trimesh plotly viser

echo "[vast] Installing DeepSpeed on Linux, without custom ops"
DS_BUILD_OPS=0 DS_BUILD_AIO=0 DS_BUILD_FUSED_ADAM=0 DS_BUILD_CPU_ADAM=0 \
  python -m pip install --no-build-isolation deepspeed

echo "[vast] Installing local Fast3R package"
python -m pip install -e ./fast3r

echo "[vast] Environment check"
python - <<'PY'
import pkg_resources
import torch
import deepspeed
import fast3r
print("[vast] torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("[vast] deepspeed:", deepspeed.__version__)
print("[vast] fast3r:", fast3r.__file__)
print("[vast] pkg_resources:", pkg_resources.__file__)
PY

export PYTHONPATH="./fast3r:."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[vast] Converting DeepSpeed checkpoint to Hugging Face format"
python ./experiments/vast_training/export_final_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --output-path "$MODEL_OUT"

echo "[vast] Converted model contents:"
find "$MODEL_OUT" -maxdepth 2 -type f -printf "%p\t%s bytes\n" | sort

RUN_ARGS=()
if [[ "$ENABLE_MESH" == "1" || "$ENABLE_MESH" == "true" ]]; then
  RUN_ARGS+=(--mesh)
fi

INFERENCE_LOG_ROOT="$OUTPUTS_ROOT/_vast_run_logs"
mkdir -p "$INFERENCE_LOG_ROOT"
INFERENCE_LOG="$INFERENCE_LOG_ROOT/inference_stdout.log"
NVIDIA_SMI_LOG="$INFERENCE_LOG_ROOT/nvidia_smi_inference.csv"
HARDWARE_SUMMARY="$INFERENCE_LOG_ROOT/hardware_inference_summary.txt"

echo "[vast] Capturing hardware snapshot"
{
  echo "timestamp_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "hostname=$(hostname)"
  echo "python=$(python -c 'import sys; print(sys.executable)')"
  python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"gpu_{idx}_name={props.name}")
        print(f"gpu_{idx}_total_vram_gb={props.total_memory / 1024**3:.3f}")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,cuda_version --format=csv,noheader,nounits
  else
    echo "nvidia_smi=not_found"
  fi
} | tee "$HARDWARE_SUMMARY"

MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "timestamp,index,name,utilization.gpu [%],memory.used [MiB],memory.total [MiB],temperature.gpu [C],power.draw [W]" > "$NVIDIA_SMI_LOG"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits >> "$NVIDIA_SMI_LOG" || true
      sleep 1
    done
  ) &
  MONITOR_PID="$!"
fi

echo "[vast] Running reconstruction with converted Fine-Tuned Fast3R"
RUN_STATUS=0
RUN_START_SECONDS="$(date +%s)"
python ./scripts/vast_ai/run_src_scene.py \
    --video "$VIDEO" \
    --model-dir "$MODEL_OUT" \
    --model-profile "$MODEL_PROFILE" \
    --outputs-root "$OUTPUTS_ROOT" \
    --target-frames "$TARGET_FRAMES" \
    --probe-max-frames "$PROBE_MAX_FRAMES" \
    --visual-shortlist-target "$VISUAL_SHORTLIST_TARGET" \
    --fast3r-image-size "$FAST3R_IMAGE_SIZE" \
    --dtype "$FAST3R_DTYPE" \
    --prefilter-enabled "$PREFILTER_ENABLED" \
    --selection-mode "$SELECTION_MODE" \
    --scan-stride "$SCAN_STRIDE" \
    --min-frame-gap "$MIN_FRAME_GAP" \
    --min-confidence-threshold "$MIN_CONFIDENCE_THRESHOLD" \
    --confidence-keep-ratio "$CONFIDENCE_KEEP_RATIO" \
    --view-conf-p50-min "$VIEW_CONF_P50_MIN" \
    --view-conf-p90-min "$VIEW_CONF_P90_MIN" \
    --weak-texture-retention "$WEAK_TEXTURE_RETENTION" \
    --weak-texture-percentile "$WEAK_TEXTURE_PERCENTILE" \
    --weak-texture-min-conf-thr "$WEAK_TEXTURE_MIN_CONF_THR" \
    --weak-texture-keep-ratio "$WEAK_TEXTURE_KEEP_RATIO" \
    --outlier-method "$OUTLIER_METHOD" \
    --radius-percentile "$RADIUS_PERCENTILE" \
    --voxel-size "$VOXEL_SIZE" \
    --high-detail-mode "$HIGH_DETAIL_MODE" \
    --scale-enabled "$SCALE_ENABLED" \
    --scale-reference-real "$SCALE_REFERENCE_REAL" \
    --scale-reference-axis "$SCALE_REFERENCE_AXIS" \
    --fail-without-scale "$FAIL_WITHOUT_SCALE" \
    --gaussian-enabled "$GAUSSIAN_ENABLED" \
    --gaussian-voxel-size "$GAUSSIAN_VOXEL_SIZE" \
    --gaussian-normal-radius "$GAUSSIAN_NORMAL_RADIUS" \
    --gaussian-normal-max-nn "$GAUSSIAN_NORMAL_MAX_NN" \
    --gaussian-nn-scale-mult "$GAUSSIAN_NN_SCALE_MULT" \
    --gaussian-min-scale "$GAUSSIAN_MIN_SCALE" \
    --gaussian-max-scale "$GAUSSIAN_MAX_SCALE" \
    --gaussian-alpha "$GAUSSIAN_ALPHA" \
    --gaussian-density-opacity "$GAUSSIAN_DENSITY_OPACITY" \
    --gaussian-min-alpha "$GAUSSIAN_MIN_ALPHA" \
    --gaussian-max-alpha "$GAUSSIAN_MAX_ALPHA" \
    --gaussian-scale-percentile-low "$GAUSSIAN_SCALE_PERCENTILE_LOW" \
    --gaussian-scale-percentile-high "$GAUSSIAN_SCALE_PERCENTILE_HIGH" \
    --gaussian-surface-aligned "$GAUSSIAN_SURFACE_ALIGNED" \
    --gaussian-max-surface-aligned-points "$GAUSSIAN_MAX_SURFACE_ALIGNED_POINTS" \
    "${RUN_ARGS[@]}" 2>&1 | tee "$INFERENCE_LOG" || RUN_STATUS="${PIPESTATUS[0]}"
RUN_END_SECONDS="$(date +%s)"

if [[ -n "$MONITOR_PID" ]]; then
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
  wait "$MONITOR_PID" >/dev/null 2>&1 || true
fi

{
  echo "inference_status=$RUN_STATUS"
  echo "inference_runtime_seconds=$((RUN_END_SECONDS - RUN_START_SECONDS))"
  if [[ -f "$NVIDIA_SMI_LOG" ]]; then
    python - "$NVIDIA_SMI_LOG" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path))
if rows:
    def num(row, key):
        try:
            return float(row[key])
        except Exception:
            return 0.0
    peak = max(rows, key=lambda r: num(r, "memory.used [MiB]"))
    peak_util = max(num(r, "utilization.gpu [%]") for r in rows)
    peak_power = max(num(r, "power.draw [W]") for r in rows)
    print(f"nvidia_smi_samples={len(rows)}")
    print(f"peak_gpu_memory_used_mib={num(peak, 'memory.used [MiB]'):.0f}")
    print(f"peak_gpu_memory_used_gb={num(peak, 'memory.used [MiB]') / 1024:.3f}")
    print(f"gpu_memory_total_mib={num(peak, 'memory.total [MiB]'):.0f}")
    print(f"gpu_memory_total_gb={num(peak, 'memory.total [MiB]') / 1024:.3f}")
    print(f"peak_gpu_utilization_percent={peak_util:.0f}")
    print(f"peak_power_draw_w={peak_power:.2f}")
PY
  fi
} | tee -a "$HARDWARE_SUMMARY"

if [[ "$RUN_STATUS" != "0" ]]; then
  echo "[vast] Reconstruction failed with status $RUN_STATUS. See $INFERENCE_LOG" >&2
  exit "$RUN_STATUS"
fi

echo
echo "[vast] Done."
echo "[vast] Model folder: $MODEL_OUT"
echo "[vast] Outputs root: $OUTPUTS_ROOT"
echo "[vast] Hardware/inference summary: $HARDWARE_SUMMARY"
