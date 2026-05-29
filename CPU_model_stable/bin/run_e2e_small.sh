#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=standalone_env.sh
source "$SCRIPT_DIR/standalone_env.sh"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CROSS_HW_RF_N_JOBS="${CROSS_HW_RF_N_JOBS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DATA_ROOT="${DATA_ROOT:-$CPU_MODEL_STABLE_ROOT/data_e2e_small_1_16}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$CPU_MODEL_STABLE_ROOT/outputs/e2e_small_validation}"
ROUTE="${ROUTE:-whitebox_family_prior_plus_gather_slice_rf_v5}"

cd "$CPU_MODEL_STABLE_ROOT"
"$PYTHON_BIN" "$CPU_MODEL_STABLE_ROOT/calibrator/run.py" \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root "$DATA_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --scales 100 \
  --passes 1 \
  --routes "$ROUTE" \
  "$@"

"$PYTHON_BIN" "$CPU_MODEL_STABLE_ROOT/scripts/write_validation_report.py" \
  --artifact-root "$ARTIFACT_ROOT" \
  --route "$ROUTE"
