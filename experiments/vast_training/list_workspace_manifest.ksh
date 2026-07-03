#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${1:-/workspace}"
OUTPUT_PATH="${2:-$ROOT_DIR/TO_BE_DOWNLOADED_TO_LOCAL_MACHINE/workspace_file_manifest.txt}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

{
  printf '# workspace_file_manifest\n'
  printf '# generated_utc=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf '# root_dir=%s\n' "$ROOT_DIR"
  printf '# columns=type<TAB>size_bytes<TAB>modified_utc<TAB>path\n'
  find "$ROOT_DIR" \( -type f -o -type l \) -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TSZ\t%p\n' | LC_ALL=C sort
} > "$OUTPUT_PATH"

FILE_COUNT="$(grep -vc '^#' "$OUTPUT_PATH" || true)"
printf 'Wrote %s entries to %s\n' "$FILE_COUNT" "$OUTPUT_PATH"
