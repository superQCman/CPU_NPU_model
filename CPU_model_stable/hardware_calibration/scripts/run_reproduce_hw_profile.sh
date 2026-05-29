#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <profile_name> <output_dir> [numa_node] [repeats]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PROFILE_NAME=$1
OUTPUT_DIR=$2
NUMA_NODE=${3:-0}
REPEATS=${4:-1}

base_args=(
  --profile "${PROFILE_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --numa-node "${NUMA_NODE}"
)

if python3 - <<'PY' >/dev/null 2>&1
import yaml
PY
then
  base_args+=(--expected-yaml "${ROOT_DIR}/../calibrator/hw_config/${PROFILE_NAME}.yaml")
fi

if [ "${REPEATS}" -gt 1 ]; then
  base_args+=(--repeats "${REPEATS}")
  python3 "${SCRIPT_DIR}/repeat_reproduce_hw_profile.py" "${base_args[@]}"
else
  python3 "${SCRIPT_DIR}/reproduce_hw_profile.py" "${base_args[@]}"
fi
