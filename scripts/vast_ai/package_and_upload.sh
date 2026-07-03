#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ssh-target> [remote-dir]"
  echo "Example: $0 root@123.45.67.89 /workspace/Thesis"
  exit 2
fi

SSH_TARGET="$1"
REMOTE_DIR="${2:-/workspace/Thesis}"
ARCHIVE_DIR="transfer"
ARCHIVE="$ARCHIVE_DIR/thesis_vast_payload.tgz"

mkdir -p "$ARCHIVE_DIR"

echo "[local] Creating payload archive: $ARCHIVE"
tar \
  --exclude-vcs \
  --exclude='.venv' \
  --exclude='.venv_export' \
  --exclude='.venv_export_linux' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data/models/fast3r_arkitscenes_ga_head_hf' \
  --exclude='src/outputs' \
  --exclude='experiments/beta_pipeline_testing/outputs' \
  -czf "$ARCHIVE" \
  src \
  fast3r \
  experiments/vast_training \
  experiments/beta_pipeline_testing \
  data/raw/irl_room_video_2.mp4 \
  scripts/vast_ai \
  requirements.txt \
  quick_run.sh \
  quick_run_highres.sh

echo "[local] Uploading to $SSH_TARGET:$REMOTE_DIR"
ssh "$SSH_TARGET" "mkdir -p '$REMOTE_DIR'"
scp "$ARCHIVE" "$SSH_TARGET:$REMOTE_DIR/thesis_vast_payload.tgz"
ssh "$SSH_TARGET" "cd '$REMOTE_DIR' && tar -xzf thesis_vast_payload.tgz"

echo
echo "Upload complete."
echo "Next on the VM:"
echo "  cd $REMOTE_DIR"
echo "  bash scripts/vast_ai/setup_convert_and_run.sh"
