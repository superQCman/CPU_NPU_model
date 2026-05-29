#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=standalone_env.sh
source "$SCRIPT_DIR/standalone_env.sh"

export DLRM_SCRIPT="${DLRM_SCRIPT:-$CPU_MODEL_STABLE_ROOT/dlrm_src/dlrm_msprof.py}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$CPU_MODEL_STABLE_ROOT/ort_src/dlrm_onnx_dyn}"

exec bash "$CPU_MODEL_STABLE_ROOT/ort_src/generate_dlrm_dyn_onnx.sh" "$@"

