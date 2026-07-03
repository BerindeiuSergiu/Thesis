#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_ROOT="${1:-/workspace}"
FAST3R_DIR="$REMOTE_ROOT/fast3r"
VAST_TRAINING_DIR="$REMOTE_ROOT/vast_training"
RESULTS_ROOT="$REMOTE_ROOT/TO_BE_DOWNLOADED_TO_LOCAL_MACHINE"

: "${PYTHON_BIN:=}"
: "${PIP_NO_CACHE_DIR:=1}"
export PIP_NO_CACHE_DIR
if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Could not find a Python interpreter. Set PYTHON_BIN explicitly."
    exit 1
  fi
fi
: "${DATASET_NAME:=arkitscenes}"
: "${DATASET_ROOT:=/workspace/datasets/arkitscenes_processed}"
: "${ARKITSCENES_RAW_ROOT:=/workspace/datasets/arkitscenes_raw}"
: "${ARKITSCENES_ROOT:=/workspace/datasets/arkitscenes_processed}"
: "${SCANNET_ROOT:=/workspace/datasets/scannet}"
: "${OUTPUT_DIR:=/workspace/logs}"
: "${CHECKPOINT_DIR:=/workspace/checkpoints}"
: "${PRETRAINED_FAST3R_CKPT:=$CHECKPOINT_DIR/fast3r_vit_large_hf_as_lightning.ckpt}"
: "${CONFIG_NAME:=vast_config}"
: "${RUN_NAME:=vast_ga_head_arkitscenes}"
: "${TASK_NAME:=vast_finetune_arkitscenes}"
: "${TRAIN_STAGE:=ga_head}"
: "${NUM_VIEWS:=6}"
: "${IMAGE_SIZE:=512}"
: "${BATCH_SIZE:=1}"
: "${LR:=1.5e-5}"
: "${MAX_STEPS:=12000}"
: "${PRECISION:=bf16-mixed}"
: "${RESUME_CKPT:=}"
: "${LORA_ENABLED:=0}"
: "${LORA_BASE_CKPT:=}"
: "${LORA_TARGET_SCOPE:=decoder_attention}"
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=16}"
: "${LORA_DROPOUT:=0.05}"
: "${NUM_WORKERS:=8}"
: "${NUM_WORKERS_VAL:=2}"
: "${BATCH_SIZE_VAL:=1}"
: "${DO_SETUP:=1}"
: "${DO_DOWNLOAD_DATASET:=0}"
: "${DO_PREPARE_DATASET:=1}"
: "${DO_TRAIN:=1}"
: "${DO_EVALUATE:=1}"
: "${DO_EXPORT_CHECKPOINT:=1}"
: "${ENABLE_OOM_FALLBACK:=1}"
: "${ARKITSCENES_LINK_MODE:=symlink}"
: "${ARKITSCENES_SPLIT:=Training}"
: "${ARKITSCENES_VIDEO_IDS:=}"
: "${ARKITSCENES_VIDEO_ID_CSV:=}"
: "${ARKITSCENES_RAW_ASSETS:=lowres_depth lowres_wide lowres_wide_intrinsics lowres_wide.traj}"

if [ "$LORA_ENABLED" = "1" ] && [ "$TRAIN_STAGE" = "ga_head" ]; then
  TRAIN_STAGE="lora_decoder_attention"
fi

RESULTS_DIR="$RESULTS_ROOT/$RUN_NAME"
TRAIN_LOG="$RESULTS_DIR/train_log.txt"
SCRIPT_LOG="$RESULTS_DIR/run_script_log.txt"
RUN_SPEC_JSON="$RESULTS_DIR/experiment_config.json"
UPLOAD_INFO_FILE="$VAST_TRAINING_DIR/UPLOAD_SOURCE_INFO.txt"
EVAL_RUN_NAME="${RUN_NAME}_evaluation"
EXPORT_DIR=""

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$RESULTS_ROOT"
exec > >(tee -a "$SCRIPT_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

run_python() {
  log "Running: $*"
  "$PYTHON_BIN" "$@"
}

python_version_ok() {
  "$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if ((major == 3 and 9 <= minor <= 12)) else 1)
PY
}

ensure_python_modules() {
  missing_modules=""
  for module_name in "$@"; do
    if ! "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
      package_name="$module_name"
      case "$module_name" in
        cv2)
          package_name="opencv-python-headless"
          ;;
      esac
      missing_modules="${missing_modules} ${package_name}"
    fi
  done
  if [ -n "${missing_modules# }" ]; then
    log "Installing missing Python modules:${missing_modules}"
    "$PYTHON_BIN" -m pip install --no-cache-dir ${missing_modules}
  fi
}

write_run_spec() {
  run_python "$VAST_TRAINING_DIR/launch_training.py" \
    --config-name "$CONFIG_NAME" \
    --dataset-name "$DATASET_NAME" \
    --dataset-root "$DATASET_ROOT" \
    --stage "$TRAIN_STAGE" \
    --task-name "$TASK_NAME" \
    --run-name "$1" \
    --pretrained "$PRETRAINED_FAST3R_CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --resume-ckpt "$RESUME_CKPT" \
    --num-views "$2" \
    --image-size "$3" \
    --batch-size "$4" \
    --batch-size-val "$BATCH_SIZE_VAL" \
    --num-workers "$NUM_WORKERS" \
    --num-workers-val "$NUM_WORKERS_VAL" \
    --lr "$LR" \
    --max-steps "$MAX_STEPS" \
    --precision "$PRECISION" \
    --lora-target-scope "$LORA_TARGET_SCOPE" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --lora-base-source "$LORA_BASE_CKPT" \
    --dump-run-spec "$5" \
    --print-only
}

read_json_field() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
field = sys.argv[2].split(".")
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload
for key in field:
    value = value[key]
print(value)
PY
}

run_training_command() {
  local spec_json="$1"
  local run_log_path="$2"
  "$PYTHON_BIN" - "$spec_json" <<'PY' | tee "$run_log_path"
import json, os, subprocess, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
command = spec["command"]
cwd = spec["working_directory"]
env = os.environ.copy()
env.update(spec.get("env", {}))
print("Working directory:", cwd)
print("Command:")
print(" ".join(command))
result = subprocess.run(command, cwd=cwd, env=env)
raise SystemExit(result.returncode)
PY
}

select_checkpoint() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
from pathlib import Path
checkpoint_dir = Path(sys.argv[1])
last_ckpt = checkpoint_dir / "last.ckpt"
if last_ckpt.exists():
    print(last_ckpt)
    raise SystemExit(0)
others = sorted(path for path in checkpoint_dir.glob("*.ckpt") if path.name != "last.ckpt")
print(others[0] if others else "")
PY
}

if [ "$DO_SETUP" = "1" ]; then
  if ! python_version_ok; then
    "$PYTHON_BIN" - <<'PY'
import sys
print(
    "Fast3R training dependencies require Python 3.9-3.12. "
    f"Current interpreter is {sys.version.split()[0]}."
)
PY
    echo "Set PYTHON_BIN to a compatible interpreter before running setup."
    exit 1
  fi
  log "Setting up environment and base checkpoint."
  mkdir -p "$SCANNET_ROOT" "$ARKITSCENES_RAW_ROOT" "$ARKITSCENES_ROOT" "$OUTPUT_DIR" "$CHECKPOINT_DIR"
  cd "$FAST3R_DIR"
  "$PYTHON_BIN" -m pip install --no-cache-dir -r requirements.txt
  "$PYTHON_BIN" -m pip install --no-cache-dir -e .
  if [ "$LORA_ENABLED" = "1" ]; then
    "$PYTHON_BIN" -m pip install --no-cache-dir -r "$VAST_TRAINING_DIR/requirements-lora.txt"
  fi
  if [ ! -f "$PRETRAINED_FAST3R_CKPT" ]; then
    cd "$REMOTE_ROOT"
    run_python "$VAST_TRAINING_DIR/convert_hf_fast3r_to_lightning_ckpt.py" \
      --model jedyang97/Fast3R_ViT_Large_512 \
      --output "$PRETRAINED_FAST3R_CKPT"
  else
    log "Base checkpoint already exists: $PRETRAINED_FAST3R_CKPT"
  fi
fi

if [ "$DO_DOWNLOAD_DATASET" = "1" ] && [ "$DATASET_NAME" = "arkitscenes" ]; then
  log "Downloading raw ARKitScenes subset/data."
  ensure_python_modules pandas
  mkdir -p "$REMOTE_ROOT/external" "$ARKITSCENES_RAW_ROOT"
  if [ ! -d "$REMOTE_ROOT/external/ARKitScenes/.git" ]; then
    git clone https://github.com/apple/ARKitScenes.git "$REMOTE_ROOT/external/ARKitScenes"
  fi
  cd "$REMOTE_ROOT/external/ARKitScenes"
  assets=($ARKITSCENES_RAW_ASSETS)
  if [ -n "$ARKITSCENES_VIDEO_IDS" ]; then
    "$PYTHON_BIN" download_data.py raw \
      --split "$ARKITSCENES_SPLIT" \
      --video_id $ARKITSCENES_VIDEO_IDS \
      --download_dir "$ARKITSCENES_RAW_ROOT" \
      --raw_dataset_assets "${assets[@]}"
  else
    csv_path="$ARKITSCENES_VIDEO_ID_CSV"
    if [ -z "$csv_path" ]; then
      csv_path="$REMOTE_ROOT/external/ARKitScenes/raw/raw_train_val_splits.csv"
    fi
    "$PYTHON_BIN" download_data.py raw \
      --video_id_csv "$csv_path" \
      --download_dir "$ARKITSCENES_RAW_ROOT" \
      --raw_dataset_assets "${assets[@]}"
  fi
fi

if [ "$DO_PREPARE_DATASET" = "1" ] && [ "$DATASET_NAME" = "arkitscenes" ]; then
  if [ ! -f "$ARKITSCENES_ROOT/Training/all_metadata.npz" ] || [ ! -f "$ARKITSCENES_ROOT/Test/all_metadata.npz" ]; then
    log "Preparing ARKitScenes processed dataset."
    ensure_python_modules cv2 numpy
    run_python "$VAST_TRAINING_DIR/prepare_arkitscenes_processed.py" \
      --raw-root "$ARKITSCENES_RAW_ROOT" \
      --output-root "$ARKITSCENES_ROOT" \
      --link-mode "$ARKITSCENES_LINK_MODE"
  else
    log "Processed ARKitScenes metadata already present."
  fi
fi

if [ "$LORA_ENABLED" = "1" ]; then
  if [ -z "$LORA_BASE_CKPT" ]; then
    log "LORA_BASE_CKPT must point to the completed GA-head checkpoint file or DeepSpeed directory."
    exit 1
  fi
  MATERIALIZED_LORA_BASE="$CHECKPOINT_DIR/${RUN_NAME}_ga_head_base_aggregated.ckpt"
  log "Materializing GA-head base checkpoint for LoRA adaptation."
  run_python "$VAST_TRAINING_DIR/materialize_checkpoint.py" \
    --input "$LORA_BASE_CKPT" \
    --output "$MATERIALIZED_LORA_BASE"
  PRETRAINED_FAST3R_CKPT="$MATERIALIZED_LORA_BASE"
fi

ACTIVE_RUN_NAME="$RUN_NAME"
ACTIVE_SPEC_JSON="$RUN_SPEC_JSON"
ACTIVE_TRAIN_LOG="$TRAIN_LOG"
ACTIVE_RUN_DIR=""
ACTIVE_CHECKPOINT_DIR=""
SELECTED_CHECKPOINT=""

if [ "$DO_TRAIN" = "1" ]; then
  log "Preparing resolved training spec."
  write_run_spec "$RUN_NAME" "$NUM_VIEWS" "$IMAGE_SIZE" "$BATCH_SIZE" "$RUN_SPEC_JSON"

  TRAINING_START_TS="$(date +%s)"
  set +e
  run_training_command "$RUN_SPEC_JSON" "$TRAIN_LOG"
  TRAIN_EXIT_CODE=$?
  set -e
  TRAINING_END_TS="$(date +%s)"
  TRAINING_SECONDS=$((TRAINING_END_TS - TRAINING_START_TS))

  if [ "$TRAIN_EXIT_CODE" -ne 0 ] && [ "$ENABLE_OOM_FALLBACK" = "1" ]; then
    log "Primary run failed with exit code $TRAIN_EXIT_CODE. Attempting OOM fallback."
    FALLBACK_RUN_NAME="${RUN_NAME}_oom_fallback"
    FALLBACK_SPEC_JSON="$RESULTS_DIR/experiment_config_oom_fallback.json"
    FALLBACK_LOG="$RESULTS_DIR/train_log_oom_fallback.txt"
    write_run_spec "$FALLBACK_RUN_NAME" 4 384 1 "$FALLBACK_SPEC_JSON"
    TRAINING_START_TS="$(date +%s)"
    set +e
    run_training_command "$FALLBACK_SPEC_JSON" "$FALLBACK_LOG"
    TRAIN_EXIT_CODE=$?
    set -e
    TRAINING_END_TS="$(date +%s)"
    TRAINING_SECONDS=$((TRAINING_END_TS - TRAINING_START_TS))
    ACTIVE_RUN_NAME="$FALLBACK_RUN_NAME"
    ACTIVE_SPEC_JSON="$FALLBACK_SPEC_JSON"
    ACTIVE_TRAIN_LOG="$FALLBACK_LOG"
  fi

  if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
    log "Training failed after fallback handling."
    exit "$TRAIN_EXIT_CODE"
  fi
else
  TRAINING_SECONDS=0
fi

if [ "$DO_TRAIN" = "1" ]; then
  ACTIVE_RUN_DIR="$(read_json_field "$ACTIVE_SPEC_JSON" run_dir)"
  ACTIVE_CHECKPOINT_DIR="$ACTIVE_RUN_DIR/checkpoints"
  SELECTED_CHECKPOINT="$(select_checkpoint "$ACTIVE_CHECKPOINT_DIR")"
  if [ -z "$SELECTED_CHECKPOINT" ]; then
    log "No checkpoint detected in $ACTIVE_CHECKPOINT_DIR"
  else
    log "Selected checkpoint for evaluation: $SELECTED_CHECKPOINT"
  fi
fi

ACTIVE_EVAL_DIR=""
if [ "$DO_EVALUATE" = "1" ] && [ -n "$SELECTED_CHECKPOINT" ]; then
  log "Running post-training evaluation."
  "$PYTHON_BIN" "$VAST_TRAINING_DIR/launch_evaluation.py" \
    --config-name "$CONFIG_NAME" \
    --dataset-name "$DATASET_NAME" \
    --dataset-root "$DATASET_ROOT" \
    --checkpoint-path "$SELECTED_CHECKPOINT" \
    --run-name "$EVAL_RUN_NAME" \
    --output-dir "$OUTPUT_DIR"
  ACTIVE_EVAL_DIR="$OUTPUT_DIR/eval_runs/$EVAL_RUN_NAME"
fi

if [ "$DO_EXPORT_CHECKPOINT" = "1" ] && [ -n "$ACTIVE_RUN_DIR" ] && [ -d "$ACTIVE_RUN_DIR" ]; then
  log "Exporting final checkpoint to Hugging Face format."
  EXPORT_DIR="$CHECKPOINT_DIR/${ACTIVE_RUN_NAME}_hf_export"
  "$PYTHON_BIN" "$VAST_TRAINING_DIR/export_final_checkpoint.py" \
    --run-dir "$ACTIVE_RUN_DIR" \
    --output-path "$EXPORT_DIR"
fi

if [ "$DO_TRAIN" = "1" ]; then
  log "Bundling results for download."
  run_python "$VAST_TRAINING_DIR/bundle_results.py" \
    --run-spec-json "$ACTIVE_SPEC_JSON" \
    --results-dir "$RESULTS_DIR" \
    --run-dir "$ACTIVE_RUN_DIR" \
    --training-seconds "$TRAINING_SECONDS" \
    --train-log "$ACTIVE_TRAIN_LOG" \
    --eval-dir "$ACTIVE_EVAL_DIR" \
    --exported-checkpoint-dir "$EXPORT_DIR" \
    --upload-source-info "$UPLOAD_INFO_FILE" \
    --fast3r-root "$FAST3R_DIR"

  log "Results bundle ready at: $RESULTS_DIR"
  log "Download this folder back locally after the run."
else
  log "Preparation-only run completed."
fi
