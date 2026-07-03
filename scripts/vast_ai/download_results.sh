#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ssh-target> [remote-dir] [local-dir]"
  echo "Example: $0 root@123.45.67.89 /workspace/Thesis ./transfer/vast_results"
  exit 2
fi

SSH_TARGET="$1"
REMOTE_DIR="${2:-/workspace/Thesis}"
LOCAL_DIR="${3:-transfer/vast_results}"

mkdir -p "$LOCAL_DIR"

echo "[local] Downloading converted model"
mkdir -p "$LOCAL_DIR/data/models"
scp -r "$SSH_TARGET:$REMOTE_DIR/data/models/fast3r_arkitscenes_ga_head_hf" "$LOCAL_DIR/data/models/"

echo "[local] Downloading Vast outputs"
mkdir -p "$LOCAL_DIR/outputs_vast"
scp -r "$SSH_TARGET:$REMOTE_DIR/outputs_vast/." "$LOCAL_DIR/outputs_vast/"

echo
echo "Downloaded to: $LOCAL_DIR"
echo
echo "To install the converted model into your local app path:"
echo "  mkdir -p data/models"
echo "  cp -r $LOCAL_DIR/data/models/fast3r_arkitscenes_ga_head_hf data/models/"
