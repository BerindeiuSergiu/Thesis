#!/usr/bin/env bash
set -Eeuo pipefail

# Run this on your LOCAL machine.
# Example:
#   bash experiments/vast_training/vast_upload_repo.ksh root@1.2.3.4 /workspace

if [ "$#" -lt 2 ]; then
  echo "Usage: bash experiments/vast_training/vast_upload_repo.ksh <user@host> <remote_bundle_root>"
  exit 1
fi

REMOTE_HOST="$1"
REMOTE_BUNDLE_ROOT="$2"
SSH_PORT="${VAST_SSH_PORT:-${SSH_PORT:-22}}"
SSH_CMD=(ssh -p "$SSH_PORT")
SCP_CMD=(scp -P "$SSH_PORT")
RSYNC_RSH="ssh -p $SSH_PORT"
LOCAL_REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_FAST3R_DIR="$LOCAL_REPO_DIR/fast3r"
LOCAL_VAST_TRAINING_DIR="$LOCAL_REPO_DIR/experiments/vast_training"

echo "Local repo : $LOCAL_REPO_DIR"
echo "Remote host: $REMOTE_HOST"
echo "Remote root: $REMOTE_BUNDLE_ROOT"
echo "SSH port   : $SSH_PORT"
echo "Uploading only:"
echo "  - $LOCAL_FAST3R_DIR"
echo "  - $LOCAL_VAST_TRAINING_DIR"

"${SSH_CMD[@]}" "$REMOTE_HOST" "mkdir -p \"$REMOTE_BUNDLE_ROOT/fast3r\" \"$REMOTE_BUNDLE_ROOT/vast_training\""

THESIS_GIT_COMMIT="$(git -C "$LOCAL_REPO_DIR" rev-parse HEAD 2>/dev/null || printf 'unavailable')"
FAST3R_GIT_COMMIT="$(git -C "$LOCAL_FAST3R_DIR" rev-parse HEAD 2>/dev/null || printf 'unavailable')"
UPLOAD_INFO_FILE="$(mktemp)"
trap 'rm -f "$UPLOAD_INFO_FILE"' EXIT
{
  echo "THESIS_GIT_COMMIT=$THESIS_GIT_COMMIT"
  echo "FAST3R_GIT_COMMIT=$FAST3R_GIT_COMMIT"
  echo "UPLOAD_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
} > "$UPLOAD_INFO_FILE"

if command -v rsync >/dev/null 2>&1; then
  rsync -av --progress \
    -e "$RSYNC_RSH" \
    --exclude ".git/" \
    --exclude "archive/" \
    --exclude "runs/" \
    --exclude "runs.zip" \
    --exclude "my_manifest.txt" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    "$LOCAL_FAST3R_DIR"/ "$REMOTE_HOST":"$REMOTE_BUNDLE_ROOT/fast3r"/
  rsync -av --progress \
    -e "$RSYNC_RSH" \
    --exclude "archive/" \
    --exclude "runs/" \
    --exclude "runs.zip" \
    --exclude "my_manifest.txt" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    "$LOCAL_VAST_TRAINING_DIR"/ "$REMOTE_HOST":"$REMOTE_BUNDLE_ROOT/vast_training"/
else
  echo "rsync not found, falling back to filtered tar streams over ssh"
  tar -C "$LOCAL_REPO_DIR" \
    --exclude ".git" \
    --exclude "archive" \
    --exclude "runs" \
    --exclude "runs.zip" \
    --exclude "my_manifest.txt" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    -czf - fast3r |
    "${SSH_CMD[@]}" "$REMOTE_HOST" "tar -xzf - -C \"$REMOTE_BUNDLE_ROOT\""
  tar -C "$LOCAL_REPO_DIR/experiments" \
    --exclude "archive" \
    --exclude "runs" \
    --exclude "runs.zip" \
    --exclude "my_manifest.txt" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    -czf - vast_training |
    "${SSH_CMD[@]}" "$REMOTE_HOST" "tar -xzf - -C \"$REMOTE_BUNDLE_ROOT\""
fi

"${SCP_CMD[@]}" "$UPLOAD_INFO_FILE" "$REMOTE_HOST":"$REMOTE_BUNDLE_ROOT/vast_training/UPLOAD_SOURCE_INFO.txt"

echo "Upload complete."
