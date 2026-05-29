#!/usr/bin/env bash
set -euo pipefail

# Training data collection command template (single-node 8 NPU, HCCL-test).
# This file prints commands only and does NOT auto-run data collection.
# Coverage target: Group1 + Group2 + Group3 across whole message-size range.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/alltoall_test"
OUT_DIR="${PROJECT_ROOT}/data/raw"
OUT_FILE="${OUT_DIR}/a2a_train_full_types.txt"
SIZE_FILE="${OUT_DIR}/a2a_train_sizes.txt"

mkdir -p "${OUT_DIR}"

mapfile -t SIZES < <(python3 "${PROJECT_ROOT}/scripts/generate_size_manifest.py" --points 96 --phase 0.0)
printf "%s\n" "${SIZES[@]}" > "${SIZE_FILE}"

cat <<EOF
[HCCL A2A TRAIN Collection Command Template]
- NPU count: 8
- HCCL binary: ${BIN}
- Output file: ${OUT_FILE}
- Size manifest: ${SIZE_FILE}
- Size count: ${#SIZES[@]} (covers Group1/2/3)

# 1) initialize output
: > "${OUT_FILE}"

# 2) run one-size-per-command collection
for sz in \
$(paste -sd' ' "${SIZE_FILE}"); do
  mpirun -n 8 "${BIN}" -b "\${sz}" -e "\${sz}" -f 2 -p 8 -n 100 -w 50 >> "${OUT_FILE}" 2>&1
done

echo "[done] train data: ${OUT_FILE}"
EOF
