#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=standalone_env.sh
source "$SCRIPT_DIR/standalone_env.sh"

SOURCE_ROOT="${SOURCE_ROOT:-$CPU_MODEL_STABLE_ROOT/data/source_kunpeng920_dataset3_scheduler_20260419_raw}"
SPLIT_OUTPUT_ROOT="${SPLIT_OUTPUT_ROOT:-$CPU_MODEL_STABLE_ROOT/data/data_e2e_small_1_16}"
TARGET_TEST_RATIO="${TARGET_TEST_RATIO:-0.30}"
E2E_WORKERS="${E2E_WORKERS:-0}"
E2E_SMALL_DIVISOR="${E2E_SMALL_DIVISOR:-16}"
E2E_SMALL_SEED="${E2E_SMALL_SEED:-20260425}"

exec "$PYTHON_BIN" "$CPU_MODEL_STABLE_ROOT/calibrator/build_splits.py" \
  --policy joint_6d_tuple \
  --dataset-scope source \
  --source-root "$SOURCE_ROOT" \
  --output-root "$SPLIT_OUTPUT_ROOT" \
  --target-test-ratio "$TARGET_TEST_RATIO" \
  --e2e-workers "$E2E_WORKERS" \
  --e2e-small-divisor "$E2E_SMALL_DIVISOR" \
  --e2e-small-seed "$E2E_SMALL_SEED" \
  "$@"
