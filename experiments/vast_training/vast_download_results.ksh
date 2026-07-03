#!/usr/bin/env bash
set -Eeuo pipefail

# Run this on your LOCAL machine.
# Example:
#   bash experiments/vast_training/vast_download_results.ksh root@1.2.3.4 vast_ga_head_arkitscenes D:/GitRepos/Thesis/vast_runs

if [ "$#" -lt 3 ]; then
  echo "Usage: bash experiments/vast_training/vast_download_results.ksh <user@host> <run_name> <local_output_dir>"
  exit 1
fi

REMOTE_HOST="$1"
RUN_NAME="$2"
LOCAL_OUTPUT_DIR="$3"
SSH_PORT="${VAST_SSH_PORT:-${SSH_PORT:-22}}"
SCP_CMD=(scp -P "$SSH_PORT")
RSYNC_RSH="ssh -p $SSH_PORT"
: "${REMOTE_RESULTS_ROOT:=/workspace/TO_BE_DOWNLOADED_TO_LOCAL_MACHINE}"
REMOTE_RESULTS_DIR="$REMOTE_RESULTS_ROOT/$RUN_NAME"

mkdir -p "$LOCAL_OUTPUT_DIR"
echo "Remote host: $REMOTE_HOST"
echo "SSH port   : $SSH_PORT"
echo "Remote dir : $REMOTE_RESULTS_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -av --progress \
    -e "$RSYNC_RSH" \
    "$REMOTE_HOST":"$REMOTE_RESULTS_DIR"/ "$LOCAL_OUTPUT_DIR"/"$RUN_NAME"/
else
  echo "rsync not found, falling back to scp -r"
  "${SCP_CMD[@]}" -r "$REMOTE_HOST":"$REMOTE_RESULTS_DIR" "$LOCAL_OUTPUT_DIR"/
fi

echo "Download complete: $LOCAL_OUTPUT_DIR/$RUN_NAME"
