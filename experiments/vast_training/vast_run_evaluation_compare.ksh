#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_ROOT="${1:-/workspace}"
VAST_TRAINING_DIR="$REMOTE_ROOT/vast_training"
RESULTS_ROOT="$REMOTE_ROOT/TO_BE_DOWNLOADED_TO_LOCAL_MACHINE"

: "${PYTHON_BIN:=}"
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

: "${PYTHONPATH:=$REMOTE_ROOT}"
export PYTHONPATH

: "${CONFIG_NAME:=arkitscenes_10hour}"
: "${DATASET_NAME:=arkitscenes}"
: "${DATASET_ROOT:=/workspace/datasets/arkitscenes_processed}"
: "${OUTPUT_DIR:=/workspace/logs}"
: "${COMPARE_RUN_NAME:=arkitscenes_10hour_eval_compare}"
: "${PRETRAINED_FAST3R_CKPT:=/workspace/checkpoints/fast3r_vit_large_hf_as_lightning.ckpt}"
: "${FINETUNED_FAST3R_CKPT:=/workspace/logs/vast_finetune_arkitscenes/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt}"
: "${IMAGE_SIZE:=0}"
: "${NUM_VIEWS:=0}"
: "${BATCH_SIZE_VAL:=1}"
: "${NUM_WORKERS_VAL:=2}"

RESULTS_DIR="$RESULTS_ROOT/$COMPARE_RUN_NAME"
SCRIPT_LOG="$RESULTS_DIR/eval_compare_script_log.txt"

mkdir -p "$RESULTS_DIR" "$RESULTS_ROOT"
exec > >(tee -a "$SCRIPT_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

log "Running paired evaluation comparison."
log "Results directory: $RESULTS_DIR"

"$PYTHON_BIN" "$VAST_TRAINING_DIR/launch_evaluation_compare.py" \
  --config-name "$CONFIG_NAME" \
  --dataset-name "$DATASET_NAME" \
  --dataset-root "$DATASET_ROOT" \
  --pretrained-checkpoint-path "$PRETRAINED_FAST3R_CKPT" \
  --finetuned-checkpoint-path "$FINETUNED_FAST3R_CKPT" \
  --run-name-prefix "$COMPARE_RUN_NAME" \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --image-size "$IMAGE_SIZE" \
  --num-views "$NUM_VIEWS" \
  --batch-size-val "$BATCH_SIZE_VAL" \
  --num-workers-val "$NUM_WORKERS_VAL"

log "Evaluation comparison complete. Results ready at: $RESULTS_DIR"
