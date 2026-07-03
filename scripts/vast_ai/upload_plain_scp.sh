#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <ssh-host> <ssh-port> <remote-dir>"
  echo "Example: $0 194.14.47.19 22071 /workspace/Thesis"
  exit 2
fi

HOST="$1"
PORT="$2"
REMOTE_DIR="$3"
TARGET="root@$HOST"

echo "[local] Creating remote folders"
ssh -p "$PORT" "$TARGET" "mkdir -p '$REMOTE_DIR/src' '$REMOTE_DIR/data/raw' '$REMOTE_DIR/experiments' '$REMOTE_DIR/scripts'"

echo "[local] Uploading application source only, excluding src/outputs and generated local data"
scp -P "$PORT" -r \
  src/app \
  src/application \
  src/config \
  src/core \
  src/models \
  src/pipeline \
  src/utils \
  src/viewer \
  src/EXPERIMENTS_AUDIT.md \
  src/README.md \
  src/main.py \
  src/requirements.txt \
  src/__init__.py \
  "$TARGET:$REMOTE_DIR/src/"

echo "[local] Uploading vendored Fast3R, scripts, and root launch files"
scp -P "$PORT" -r fast3r scripts/vast_ai requirements.txt quick_run.sh quick_run_highres.sh "$TARGET:$REMOTE_DIR/"

echo "[local] Uploading only the specific good fine-tuning run/checkpoint"
ssh -p "$PORT" "$TARGET" "\
  rm -rf '$REMOTE_DIR/experiments/vast_training' && \
  mkdir -p \
    '$REMOTE_DIR/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/checkpoints' \
    '$REMOTE_DIR/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0' \
    '$REMOTE_DIR/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/.hydra'"
scp -P "$PORT" \
  experiments/vast_training/export_final_checkpoint.py \
  experiments/vast_training/export_merged_lora_checkpoint.py \
  "$TARGET:$REMOTE_DIR/experiments/vast_training/"
scp -P "$PORT" \
  experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/hparams.yaml \
  "$TARGET:$REMOTE_DIR/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/"
scp -P "$PORT" \
  experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/.hydra/config.yaml \
  "$TARGET:$REMOTE_DIR/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/.hydra/"
scp -P "$PORT" -r \
  experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/checkpoints/epoch_000.ckpt \
  "$TARGET:$REMOTE_DIR/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/checkpoints/"

echo "[local] Uploading video"
scp -P "$PORT" data/raw/irl_room_video_2.mp4 "$TARGET:$REMOTE_DIR/data/raw/"

echo
echo "Upload complete."
echo "Next:"
echo "  ssh -p $PORT $TARGET"
echo "  cd $REMOTE_DIR"
echo "  bash scripts/vast_ai/setup_convert_and_run.sh"
