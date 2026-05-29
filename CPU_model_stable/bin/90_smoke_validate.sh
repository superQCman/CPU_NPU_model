#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=standalone_env.sh
source "$SCRIPT_DIR/standalone_env.sh"

FIXTURE_ROOT="${FIXTURE_ROOT:-$CPU_MODEL_STABLE_ROOT/fixtures/joint_6d_tuple_small_sample}"
EXPECTED_MANIFEST="${EXPECTED_MANIFEST:-$FIXTURE_ROOT/expected_manifest.json}"

"$PYTHON_BIN" "$CPU_MODEL_STABLE_ROOT/scripts/validate_fixture.py" \
  --fixture-root "$FIXTURE_ROOT" \
  --expected-manifest "$EXPECTED_MANIFEST"

