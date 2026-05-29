#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=standalone_env.sh
source "$SCRIPT_DIR/standalone_env.sh"

JOBS_FILE="${JOBS_FILE:-$CPU_MODEL_STABLE_ROOT/ort_src/jobs_scheduler_20260419.csv}"
DATA_ROOT="${DATA_ROOT:-$CPU_MODEL_STABLE_ROOT/data}"
RUN_CONFIG_ROOT="${RUN_CONFIG_ROOT:-$DATA_ROOT/run_configs}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/case_dataset_scheduler_logs}"
AVAILABLE_NUMAS="${AVAILABLE_NUMAS:-auto}"
MACHINE_CONFIG_JSON="${MACHINE_CONFIG_JSON:-$RUN_CONFIG_ROOT/machine_config.json}"
export ONNX_ROOT="${ONNX_ROOT:-$CPU_MODEL_STABLE_ROOT/ort_src/dlrm_onnx_dyn}"

mkdir -p "$RUN_CONFIG_ROOT"

exec "$PYTHON_BIN" "$CPU_MODEL_STABLE_ROOT/ort_src/run_case_dataset_jobs.py" \
  --no-discover-case-scripts \
  --jobs-file "$JOBS_FILE" \
  --log-root "$LOG_ROOT" \
  --available-numas "$AVAILABLE_NUMAS" \
  --machine-config-json "$MACHINE_CONFIG_JSON" \
  "$@"
