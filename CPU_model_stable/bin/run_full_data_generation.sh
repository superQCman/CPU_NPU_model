#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=standalone_env.sh
source "$SCRIPT_DIR/standalone_env.sh"

DATA_ROOT="${DATA_ROOT:-$CPU_MODEL_STABLE_ROOT/data}"
RUN_CONFIG_ROOT="${RUN_CONFIG_ROOT:-$DATA_ROOT/run_configs}"
JOBS_FILE_IN="${JOBS_FILE:-$CPU_MODEL_STABLE_ROOT/ort_src/jobs_scheduler_20260419_bs1024_2048_step512_nip1000_2000_step200.csv}"
JOBS_FILE_LOCAL="${JOBS_FILE_LOCAL:-$RUN_CONFIG_ROOT/$(basename "${JOBS_FILE_IN%.csv}")_data_local.csv}"
MACHINE_CONFIG_JSON="${MACHINE_CONFIG_JSON:-$RUN_CONFIG_ROOT/machine_config.json}"
AVAILABLE_NUMAS="${AVAILABLE_NUMAS:-auto}"

ONNX_ROOT="${ONNX_ROOT:-$DATA_ROOT/dlrm_onnx_dyn}"
SCHED_LOG_ROOT="${SCHED_LOG_ROOT:-$DATA_ROOT/case_dataset_scheduler_logs}"
SOURCE_OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT:-$DATA_ROOT/source_kunpeng920_dataset3_scheduler_20260419_raw}"
SPLIT_OUTPUT_ROOT="${SPLIT_OUTPUT_ROOT:-$DATA_ROOT/data_e2e_small_1_16}"
SOURCE_HW_CONFIG="${SOURCE_HW_CONFIG:-$CPU_MODEL_STABLE_ROOT/calibrator/hw_config/kunpeng920_single_numa.yaml}"

RUN_ONNX="${RUN_ONNX:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"
RUN_RAW="${RUN_RAW:-1}"
RUN_SPLIT="${RUN_SPLIT:-1}"

E2E_WORKERS="${E2E_WORKERS:-0}"
E2E_SMALL_DIVISOR="${E2E_SMALL_DIVISOR:-16}"
E2E_SMALL_SEED="${E2E_SMALL_SEED:-20260425}"
TARGET_TEST_RATIO="${TARGET_TEST_RATIO:-0.30}"
MAX_FILES_PER_CASE="${MAX_FILES_PER_CASE:-0}"

mkdir -p "$DATA_ROOT" "$RUN_CONFIG_ROOT"

"$PYTHON_BIN" -m calibrator.machine_topology --write-json "$MACHINE_CONFIG_JSON" >/dev/null

if [[ ! -f "$JOBS_FILE_IN" ]]; then
  echo "ERROR: JOBS_FILE does not exist: $JOBS_FILE_IN" >&2
  exit 1
fi

"$PYTHON_BIN" - "$JOBS_FILE_IN" "$JOBS_FILE_LOCAL" "$DATA_ROOT" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
data_root = Path(sys.argv[3]).resolve()

rows = list(csv.DictReader(src.open("r", encoding="utf-8", newline="")))
if not rows:
    raise SystemExit(f"empty jobs file: {src}")

extra = ["out_root", "feature_dataset_root", "feature_subset_root", "generated_onnx_tmp_root"]
fieldnames = list(rows[0].keys())
for name in extra:
    if name not in fieldnames:
        fieldnames.append(name)

for row in rows:
    job_name = str(row.get("job_name", "")).strip()
    if not job_name:
        raise SystemExit(f"missing job_name in {src}")
    row["out_root"] = str(data_root / f"sweep_runs_extensible_{job_name}")
    row["feature_dataset_root"] = str(data_root / f"features_extensible_{job_name}")
    row["feature_subset_root"] = str(data_root / f"features_extensible_{job_name}_selected")
    row["generated_onnx_tmp_root"] = str(data_root / "tmp_generated_onnx")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(dst)
PY

echo "[cpu-model-stable-full] data_root=$DATA_ROOT"
echo "[cpu-model-stable-full] jobs_file=$JOBS_FILE_LOCAL"
echo "[cpu-model-stable-full] onnx_root=$ONNX_ROOT"
echo "[cpu-model-stable-full] scheduler_log_root=$SCHED_LOG_ROOT"
echo "[cpu-model-stable-full] source_output_root=$SOURCE_OUTPUT_ROOT"
echo "[cpu-model-stable-full] split_output_root=$SPLIT_OUTPUT_ROOT"
echo "[cpu-model-stable-full] machine_config_json=$MACHINE_CONFIG_JSON"
echo "[cpu-model-stable-full] available_numas=$AVAILABLE_NUMAS"

if [[ "$RUN_ONNX" == "1" ]]; then
  echo "[cpu-model-stable-full] step 1/4: generate ONNX"
  OUTPUT_ROOT="$ONNX_ROOT" ONNX_ROOT="$ONNX_ROOT" "$CPU_MODEL_STABLE_ROOT/bin/01_generate_onnx.sh"
else
  echo "[cpu-model-stable-full] skip ONNX generation"
fi

if [[ "$RUN_PROFILE" == "1" ]]; then
  echo "[cpu-model-stable-full] step 2/4: run no-trace ORT profiling jobs"
  JOBS_FILE="$JOBS_FILE_LOCAL" \
  LOG_ROOT="$SCHED_LOG_ROOT" \
  ONNX_ROOT="$ONNX_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  AVAILABLE_NUMAS="$AVAILABLE_NUMAS" \
  MACHINE_CONFIG_JSON="$MACHINE_CONFIG_JSON" \
  "$CPU_MODEL_STABLE_ROOT/bin/02_run_case_jobs.sh"
else
  echo "[cpu-model-stable-full] skip profiling jobs"
fi

if [[ "$RUN_RAW" == "1" ]]; then
  echo "[cpu-model-stable-full] step 3/4: build scheduler raw source dataset"
  latest_logs=$(
    find "$SCHED_LOG_ROOT" -mindepth 2 -maxdepth 2 -type f -name scheduler_summary.csv -printf '%T@ %h\n' 2>/dev/null \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
  )
  if [[ -z "$latest_logs" || ! -d "$latest_logs" ]]; then
    echo "ERROR: no scheduler log directory with scheduler_summary.csv found under $SCHED_LOG_ROOT" >&2
    exit 1
  fi
  echo "[cpu-model-stable-full] selected scheduler logs: $latest_logs"
  LOGS_ROOT="$latest_logs" \
  SOURCE_OUTPUT_ROOT="$SOURCE_OUTPUT_ROOT" \
  SOURCE_HW_CONFIG="$SOURCE_HW_CONFIG" \
  MAX_FILES_PER_CASE="$MAX_FILES_PER_CASE" \
  "$CPU_MODEL_STABLE_ROOT/bin/03_prepare_source_raw.sh"
else
  echo "[cpu-model-stable-full] skip raw dataset build"
fi

if [[ "$RUN_SPLIT" == "1" ]]; then
  echo "[cpu-model-stable-full] step 4/4: build joint_6d_tuple small split"
  SOURCE_ROOT="$SOURCE_OUTPUT_ROOT" \
  SPLIT_OUTPUT_ROOT="$SPLIT_OUTPUT_ROOT" \
  TARGET_TEST_RATIO="$TARGET_TEST_RATIO" \
  E2E_WORKERS="$E2E_WORKERS" \
  E2E_SMALL_DIVISOR="$E2E_SMALL_DIVISOR" \
  E2E_SMALL_SEED="$E2E_SMALL_SEED" \
  "$CPU_MODEL_STABLE_ROOT/bin/04_build_joint_small_split.sh"
else
  echo "[cpu-model-stable-full] skip split build"
fi

echo "[cpu-model-stable-full] done"
echo "[cpu-model-stable-full] final split root: $SPLIT_OUTPUT_ROOT/source_kunpeng920_dataset3_unseen_joint_6d_tuple"
