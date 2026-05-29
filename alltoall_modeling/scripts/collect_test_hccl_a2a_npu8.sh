#!/usr/bin/env bash
set -euo pipefail

# External TEST data collection for Method-B evaluation.
# This script performs actual collection and writes an independent test dataset.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/alltoall_test"
OUT_DIR="${PROJECT_ROOT}/data/raw"
OUT_FILE="${OUT_DIR}/a2a_test_external.txt"
SIZE_FILE="${OUT_DIR}/a2a_test_sizes.txt"

mkdir -p "${OUT_DIR}"

mapfile -t SIZES < <(python3 "${PROJECT_ROOT}/scripts/generate_size_manifest.py" --points 84 --phase 0.5)
printf "%s\n" "${SIZES[@]}" > "${SIZE_FILE}"

echo "[info] external test collection starts"
echo "[info] size count=${#SIZES[@]} (Group1/2/3 coverage), size file=${SIZE_FILE}"
echo "[info] output=${OUT_FILE}"

: > "${OUT_FILE}"
for sz in "${SIZES[@]}"; do
  mpirun -n 8 "${BIN}" -b "${sz}" -e "${sz}" -f 2 -p 8 -n 100 -w 50 >> "${OUT_FILE}" 2>&1
done

echo "[done] external test data: ${OUT_FILE}"
