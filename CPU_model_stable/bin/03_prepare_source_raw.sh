#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=standalone_env.sh
source "$SCRIPT_DIR/standalone_env.sh"

DEFAULT_LOG_PARENT="$CPU_MODEL_STABLE_ROOT/data/case_dataset_scheduler_logs"
if [[ -z "${LOGS_ROOT:-}" ]]; then
  LOGS_ROOT=$(
    find "$DEFAULT_LOG_PARENT" -mindepth 2 -maxdepth 2 -type f -name scheduler_summary.csv -printf '%T@ %h\n' 2>/dev/null \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
  )
fi

if [[ -z "${LOGS_ROOT:-}" || ! -d "$LOGS_ROOT" ]]; then
  echo "ERROR: cannot find scheduler logs. Set LOGS_ROOT=/path/to/scheduler_log_dir." >&2
  exit 1
fi

SOURCE_OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT:-$CPU_MODEL_STABLE_ROOT/data/source_kunpeng920_dataset3_scheduler_20260419_raw}"
SOURCE_HW_CONFIG="${SOURCE_HW_CONFIG:-$CPU_MODEL_STABLE_ROOT/calibrator/hw_config/kunpeng920_single_numa.yaml}"
MAX_FILES_PER_CASE="${MAX_FILES_PER_CASE:-0}"

exec "$PYTHON_BIN" "$CPU_MODEL_STABLE_ROOT/calibrator/prepare_source_scheduler_20260419.py" \
  --output-root "$SOURCE_OUTPUT_ROOT" \
  --logs-root "$LOGS_ROOT" \
  --source-hw-config "$SOURCE_HW_CONFIG" \
  --max-files-per-case "$MAX_FILES_PER_CASE" \
  "$@"
