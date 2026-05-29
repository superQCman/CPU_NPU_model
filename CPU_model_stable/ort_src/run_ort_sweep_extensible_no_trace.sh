#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ANALYSIS_DIR="$SCRIPT_DIR/onnx_operator_analysis"

ASCEND_ENV_SH=${ASCEND_ENV_SH:-}
ONNX_ROOT=${ONNX_ROOT:-$SCRIPT_DIR/dlrm_onnx_dyn}
ONNX_VARIANT=${ONNX_VARIANT:-}
ONNX_FILENAME=${ONNX_FILENAME:-dlrm_s_pytorch.onnx}
ONNX_MANIFEST_CSV=${ONNX_MANIFEST_CSV:-$ONNX_ROOT/manifest.csv}
DEFAULT_ONNX_PATH="$SCRIPT_DIR/../dlrm_onnx/dlrm_s_pytorch.onnx"
if [[ -z "${ONNX_PATH:-}" ]]; then
  if [[ -n "$ONNX_VARIANT" ]]; then
    ONNX_PATH="$ONNX_ROOT/$ONNX_VARIANT/$ONNX_FILENAME"
  elif [[ -f "$ONNX_ROOT/$ONNX_FILENAME" ]]; then
    ONNX_PATH="$ONNX_ROOT/$ONNX_FILENAME"
  else
    ONNX_PATH="$DEFAULT_ONNX_PATH"
  fi
fi
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}

BATCH_START=${BATCH_START:-32}
BATCH_END=${BATCH_END:-2048}
BATCH_STEP=${BATCH_STEP:-32}

NUM_INDICES_START=${NUM_INDICES_START:-100}
NUM_INDICES_END=${NUM_INDICES_END:-1000}
NUM_INDICES_STEP=${NUM_INDICES_STEP:-50}

NUM_BATCHES=${NUM_BATCHES:-3}
WARMUP_BATCHES=${WARMUP_BATCHES:-2}
INTRA_THREADS=${INTRA_THREADS:-4}
INTER_THREADS=${INTER_THREADS:-1}
DEVICE_ID=${DEVICE_ID:-0}
USE_CANN=${USE_CANN:-0}
NO_REPLACE_LOOP=${NO_REPLACE_LOOP:-0}
PROFILE_WARMUP=${PROFILE_WARMUP:-0}
DISABLE_GRAPH_OPTIMIZATIONS=${DISABLE_GRAPH_OPTIMIZATIONS:-1}

RUNNER_MODE=${RUNNER_MODE:-branch_parallel}
RUNNER_SCRIPT=${RUNNER_SCRIPT:-}
PARALLEL_BRANCHES=${PARALLEL_BRANCHES:-0}
TAIL_INTRA_THREADS=${TAIL_INTRA_THREADS:-0}
VERIFY_FULL_OUTPUT=${VERIFY_FULL_OUTPUT:-0}
BRANCH_SUBMODEL_ROOT=${BRANCH_SUBMODEL_ROOT:-}
FORCE_CPU_OPS=${FORCE_CPU_OPS:-}
STOP_AFTER_INFER=${STOP_AFTER_INFER:-0}

USE_NUMACTL=${USE_NUMACTL:-1}
NUMA_NODE=${NUMA_NODE:-0}

DATA_ROOT=${DATA_ROOT:-}
DEFAULT_ARTIFACT_ROOT=${DATA_ROOT:-$SCRIPT_DIR}

OUT_ROOT=${OUT_ROOT:-$DEFAULT_ARTIFACT_ROOT/sweep_runs_extensible_no_trace}
OP_SHAPES_DIR=${OP_SHAPES_DIR:-$OUT_ROOT/op_shapes}
PROFILE_ROOT=${PROFILE_ROOT:-$OUT_ROOT/onnx_profiles}
LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/logs}
FEATURE_DATASET_ROOT=${FEATURE_DATASET_ROOT:-$DEFAULT_ARTIFACT_ROOT/features_extensible_no_trace}
FEATURE_DATASET_MERGED_CSV=${FEATURE_DATASET_MERGED_CSV:-$FEATURE_DATASET_ROOT/all_features.csv}
FEATURE_SUBSET_ROOT=${FEATURE_SUBSET_ROOT:-$DEFAULT_ARTIFACT_ROOT/features_extensible_no_trace_selected}
FEATURE_SUBSET_MERGED_CSV=${FEATURE_SUBSET_MERGED_CSV:-$FEATURE_SUBSET_ROOT/all_features.csv}
SELECT_FEATURE_SUBSET=${SELECT_FEATURE_SUBSET:-1}
SELECT_FEATURE_SUBSET_DUR_SOURCE=${SELECT_FEATURE_SUBSET_DUR_SOURCE:-avg}
GENERATED_ONNX_ROOT=${GENERATED_ONNX_ROOT:-$OUT_ROOT/generated_onnx}
PRESERVE_GENERATED_ONNX=${PRESERVE_GENERATED_ONNX:-1}
GENERATED_ONNX_TMP_ROOT=${GENERATED_ONNX_TMP_ROOT:-/tmp/ort_generated_onnx}
SUMMARY_CSV=${SUMMARY_CSV:-$OUT_ROOT/sweep_summary.csv}

MAX_COMBOS=${MAX_COMBOS:-0}
RESUME=${RESUME:-1}

if [[ "$PRESERVE_GENERATED_ONNX" == "1" ]]; then
  GENERATED_ONNX_WORK_ROOT="$GENERATED_ONNX_ROOT"
else
  GENERATED_ONNX_WORK_ROOT="$GENERATED_ONNX_TMP_ROOT/$(basename "$OUT_ROOT")"
fi

mkdir -p "$OUT_ROOT" "$OP_SHAPES_DIR" "$PROFILE_ROOT" "$LOG_ROOT" "$FEATURE_DATASET_ROOT" "$FEATURE_SUBSET_ROOT"
if [[ -n "$GENERATED_ONNX_WORK_ROOT" ]]; then
  mkdir -p "$GENERATED_ONNX_WORK_ROOT"
fi

if [[ -n "$ASCEND_ENV_SH" && -f "$ASCEND_ENV_SH" ]]; then
  # shellcheck disable=SC1090
  source "$ASCEND_ENV_SH"
fi

if [[ "$BATCH_STEP" -le 0 || "$NUM_INDICES_STEP" -le 0 ]]; then
  echo "ERROR: BATCH_STEP and NUM_INDICES_STEP must be > 0"
  exit 1
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ERROR: ONNX model not found: $ONNX_PATH"
  exit 1
fi

echo "[CONFIG] ONNX_PATH=$ONNX_PATH"
echo "[CONFIG] ONNX_ROOT=$ONNX_ROOT"
echo "[CONFIG] ONNX_VARIANT=${ONNX_VARIANT:-<unset>}"
echo "[CONFIG] ONNX_MANIFEST_CSV=$ONNX_MANIFEST_CSV"
echo "[CONFIG] PRESERVE_GENERATED_ONNX=$PRESERVE_GENERATED_ONNX"
echo "[CONFIG] GENERATED_ONNX_WORK_ROOT=$GENERATED_ONNX_WORK_ROOT"

MODEL_ARCH_EMBEDDING_SIZE=""
MODEL_ARCH_MLP_BOT=""
MODEL_ARCH_MLP_TOP=""

resolve_realpath() {
  local path="$1"
  if command -v readlink > /dev/null 2>&1; then
    readlink -f "$path" 2>/dev/null || printf '%s\n' "$path"
  else
    printf '%s\n' "$path"
  fi
}

load_onnx_arch_metadata() {
  local manifest_csv="$1"
  local target_path="$2"
  local target_real=""
  local variant=""
  local arch_embedding_size=""
  local arch_mlp_bot=""
  local arch_mlp_top=""
  local manifest_onnx_path=""
  local manifest_real=""

  if [[ ! -f "$manifest_csv" ]]; then
    echo "[WARN] ONNX manifest not found, skipping arch metadata: $manifest_csv"
    return 0
  fi

  target_real=$(resolve_realpath "$target_path")
  while IFS=',' read -r variant arch_embedding_size arch_mlp_bot arch_mlp_top manifest_onnx_path; do
    [[ "$variant" == "variant" ]] && continue
    manifest_real=$(resolve_realpath "$manifest_onnx_path")
    if [[ -n "$ONNX_VARIANT" && "$variant" == "$ONNX_VARIANT" ]] || [[ "$manifest_onnx_path" == "$target_path" ]] || [[ "$manifest_real" == "$target_real" ]]; then
      MODEL_ARCH_EMBEDDING_SIZE="$arch_embedding_size"
      MODEL_ARCH_MLP_BOT="$arch_mlp_bot"
      MODEL_ARCH_MLP_TOP="$arch_mlp_top"
      echo "[CONFIG] arch_embedding_size=$MODEL_ARCH_EMBEDDING_SIZE"
      echo "[CONFIG] arch_mlp_bot=$MODEL_ARCH_MLP_BOT"
      echo "[CONFIG] arch_mlp_top=$MODEL_ARCH_MLP_TOP"
      return 0
    fi
  done < "$manifest_csv"

  echo "[WARN] no manifest row matched ONNX_PATH=$target_path"
}

load_onnx_arch_metadata "$ONNX_MANIFEST_CSV" "$ONNX_PATH"

case "$RUNNER_MODE" in
  standard)
    : "${RUNNER_SCRIPT:=$SCRIPT_DIR/run_ort_dlrm.py}"
    ;;
  branch_parallel)
    : "${RUNNER_SCRIPT:=$SCRIPT_DIR/run_ort_dlrm_branch_parallel.py}"
    ;;
  *)
    echo "ERROR: unsupported RUNNER_MODE=$RUNNER_MODE (expected standard or branch_parallel)"
    exit 1
    ;;
esac

if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "ERROR: runner script not found: $RUNNER_SCRIPT"
  exit 1
fi

echo "timestamp,batch_size,num_indices_per_lookup,arch_embedding_size,arch_mlp_bot,arch_mlp_top,runner_mode,status,failed_stage,shape_csv,profile_dir,profile_json,effective_onnx_path,cpu_thread_detail_csv,training_feature_csv,log_dir" > "$SUMMARY_CSV"

has_profile_json() {
  local dir="$1"
  compgen -G "$dir/ort_cann_profile*.json" > /dev/null
}

csv_has_header() {
  local path="$1"
  local first_line=""
  [[ -s "$path" ]] || return 1
  IFS= read -r first_line < "$path" || true
  [[ -n "${first_line//[[:space:]]/}" ]]
}

csv_has_columns() {
  local path="$1"
  shift
  local first_line=""
  local required_col=""
  [[ -s "$path" ]] || return 1
  IFS= read -r first_line < "$path" || true
  [[ -n "$first_line" ]] || return 1
  for required_col in "$@"; do
    [[ ",$first_line," == *",$required_col,"* ]] || return 1
  done
}

find_latest_profile_json() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name 'ort_cann_profile*.json' | sort | tail -n 1
}

cpu_parse_completed() {
  local cpu_detail_csv="$1"
  local cpu_agg_csv="$2"
  csv_has_header "$cpu_detail_csv" && csv_has_header "$cpu_agg_csv"
}

dataset_completed() {
  local final_feature_csv="$1"
  local aligned_cpu_detail_csv="$2"
  local required_columns=()

  if [[ -n "$MODEL_ARCH_EMBEDDING_SIZE" ]]; then
    required_columns+=(arch_embedding_size)
  fi
  if [[ -n "$MODEL_ARCH_MLP_BOT" ]]; then
    required_columns+=(arch_mlp_bot)
  fi
  if [[ -n "$MODEL_ARCH_MLP_TOP" ]]; then
    required_columns+=(arch_mlp_top)
  fi

  csv_has_header "$final_feature_csv" || return 1
  if (( ${#required_columns[@]} > 0 )); then
    csv_has_columns "$final_feature_csv" "${required_columns[@]}" || return 1
  fi
  csv_has_header "$aligned_cpu_detail_csv"
}

append_summary() {
  local batch_size="$1"
  local num_indices="$2"
  local status="$3"
  local failed_stage="$4"
  local shape_csv="$5"
  local profile_dir="$6"
  local profile_json="$7"
  local effective_onnx_path="$8"
  local cpu_detail_csv="$9"
  local training_feature_csv="${10}"
  local log_dir="${11}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date '+%F %T')" \
    "$batch_size" \
    "$num_indices" \
    "$MODEL_ARCH_EMBEDDING_SIZE" \
    "$MODEL_ARCH_MLP_BOT" \
    "$MODEL_ARCH_MLP_TOP" \
    "$RUNNER_MODE" \
    "$status" \
    "$failed_stage" \
    "$shape_csv" \
    "$profile_dir" \
    "$profile_json" \
    "$effective_onnx_path" \
    "$cpu_detail_csv" \
    "$training_feature_csv" \
    "$log_dir" >> "$SUMMARY_CSV"
}

prepare_combo_onnx() {
  local combo_source_onnx="$1"
  mkdir -p "$(dirname "$combo_source_onnx")"
  cp -f "$ONNX_PATH" "$combo_source_onnx"
}

resolve_effective_onnx_path() {
  local combo_source_onnx="$1"
  local infer_log="$2"
  local parsed=""

  if [[ -f "$infer_log" ]]; then
    parsed=$(awk -F': ' '/^\[ORT\] 加载模型:/ {path=$2} END{print path}' "$infer_log")
    if [[ -n "$parsed" ]]; then
      if [[ ! -f "$parsed" ]]; then
        echo "[WARN] resolved ONNX path no longer exists on disk, keeping logged path: $parsed" >&2
      fi
      printf '%s\n' "$parsed"
      return 0
    fi
  fi

  for candidate in \
    "$combo_source_onnx.cann_patched.onnx.loop_to_gather.onnx.cpu_ops.onnx" \
    "$combo_source_onnx.cann_patched.onnx.loop_to_gather.onnx" \
    "$combo_source_onnx.loop_to_gather.onnx.cpu_ops.onnx" \
    "$combo_source_onnx.loop_to_gather.onnx" \
    "$combo_source_onnx.cann_patched.onnx" \
    "$combo_source_onnx"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -n "$combo_source_onnx" ]]; then
    printf '%s\n' "$combo_source_onnx"
    return 0
  fi

  return 1
}

cleanup_combo_onnx_dir() {
  local combo_onnx_dir="$1"
  if [[ "$PRESERVE_GENERATED_ONNX" != "1" && -n "$combo_onnx_dir" && -d "$combo_onnx_dir" ]]; then
    rm -rf "$combo_onnx_dir"
    rmdir "$GENERATED_ONNX_WORK_ROOT" 2>/dev/null || true
    rmdir "$GENERATED_ONNX_TMP_ROOT" 2>/dev/null || true
  fi
}

build_inference_command() {
  local onnx_path="$1"
  local batch_size="$2"
  local num_indices="$3"
  local shape_csv="$4"
  local profile_dir="$5"
  local submodel_dir="$6"

  local cmd=()
  if [[ "$USE_NUMACTL" == "1" ]] && command -v numactl > /dev/null 2>&1; then
    cmd+=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
  fi

  cmd+=(
    "$PYTHON_BIN" "$RUNNER_SCRIPT"
    --onnx-path "$onnx_path"
    --batch-size "$batch_size"
    --num-batches "$NUM_BATCHES"
    --warmup-batches "$WARMUP_BATCHES"
    --shape-csv "$shape_csv"
    --enable-profiling
    --profile-dir "$profile_dir"
    --intra-threads "$INTRA_THREADS"
    --inter-threads "$INTER_THREADS"
    --num-indices-per-lookup "$num_indices"
  )

  if [[ "$USE_CANN" == "1" ]]; then
    cmd+=(--use-cann --device-id "$DEVICE_ID")
  fi
  if [[ "$NO_REPLACE_LOOP" == "1" ]]; then
    cmd+=(--no-replace-loop)
  fi
  if [[ "$PROFILE_WARMUP" == "1" ]]; then
    cmd+=(--profile-warmup)
  fi
  if [[ "$DISABLE_GRAPH_OPTIMIZATIONS" == "1" ]]; then
    cmd+=(--disable-graph-optimizations)
  fi
  if [[ -n "$FORCE_CPU_OPS" ]]; then
    cmd+=(--force-cpu-ops "$FORCE_CPU_OPS")
  fi

  if [[ "$RUNNER_MODE" == "branch_parallel" ]]; then
    if [[ "$PARALLEL_BRANCHES" -gt 0 ]]; then
      cmd+=(--parallel-branches "$PARALLEL_BRANCHES")
    fi
    if [[ "$TAIL_INTRA_THREADS" -gt 0 ]]; then
      cmd+=(--tail-intra-threads "$TAIL_INTRA_THREADS")
    fi
    if [[ "$VERIFY_FULL_OUTPUT" == "1" ]]; then
      cmd+=(--verify-full-output)
    fi
    if [[ -n "$submodel_dir" ]]; then
      cmd+=(--submodel-dir "$submodel_dir")
    fi
    cmd+=(--out-dir "$profile_dir")
  fi

  printf '%q ' "${cmd[@]}"
  printf '\n'
}

run_inference_stage() {
  local onnx_path="$1"
  local batch_size="$2"
  local num_indices="$3"
  local shape_csv="$4"
  local profile_dir="$5"
  local submodel_dir="$6"
  local log_path="$7"

  local cmd_text
  cmd_text=$(build_inference_command "$onnx_path" "$batch_size" "$num_indices" "$shape_csv" "$profile_dir" "$submodel_dir")
  # shellcheck disable=SC2086
  eval "$cmd_text" > "$log_path" 2>&1
}

run_cpu_parse_stage() {
  local profile_json="$1"
  local profile_dir="$2"
  local log_path="$3"

  "$PYTHON_BIN" -u "$ANALYSIS_DIR/extract_cpu_thread_usage.py" \
    "$profile_json" \
    --out-dir "$profile_dir" > "$log_path" 2>&1
}

run_dataset_stage() {
  local batch_size="$1"
  local num_indices="$2"
  local shape_csv="$3"
  local cpu_detail_csv="$4"
  local aligned_cpu_detail_csv="$5"
  local cpu_node_agg_csv="$6"
  local cpu_unmatched_csv="$7"
  local final_feature_csv="$8"
  local log_path="$9"

  local cmd=(
    "$PYTHON_BIN" -u "$ANALYSIS_DIR/build_training_features_no_trace.py"
    --op-shapes "$shape_csv"
    --cpu-detail "$cpu_detail_csv"
    --aligned-cpu-detail-out "$aligned_cpu_detail_csv"
    --cpu-agg-out "$cpu_node_agg_csv"
    --unmatched-out "$cpu_unmatched_csv"
    --out "$final_feature_csv"
    --batch-size "$batch_size"
    --num-indices-per-lookup "$num_indices"
  )

  if [[ -n "$MODEL_ARCH_EMBEDDING_SIZE" ]]; then
    cmd+=(--arch-embedding-size "$MODEL_ARCH_EMBEDDING_SIZE")
  fi
  if [[ -n "$MODEL_ARCH_MLP_BOT" ]]; then
    cmd+=(--arch-mlp-bot "$MODEL_ARCH_MLP_BOT")
  fi
  if [[ -n "$MODEL_ARCH_MLP_TOP" ]]; then
    cmd+=(--arch-mlp-top "$MODEL_ARCH_MLP_TOP")
  fi

  "${cmd[@]}" > "$log_path" 2>&1
}

merge_feature_datasets() {
  local feature_root="$1"
  local merged_csv="$2"

  "$PYTHON_BIN" - "$feature_root" "$merged_csv" <<'PY'
import csv
import sys
from pathlib import Path

feature_root = Path(sys.argv[1])
merged_csv = Path(sys.argv[2])

csv_paths = sorted(
    path for path in feature_root.glob("*.csv")
    if path.name != merged_csv.name and path.is_file()
)

if not csv_paths:
    print(f"[WARN] no feature CSVs found under {feature_root}")
    sys.exit(0)

header = []
rows_written = 0

for path in csv_paths:
    with path.open("r", encoding="utf-8", newline="") as in_f:
        reader = csv.DictReader(in_f)
        if not reader.fieldnames:
            continue
        for field in reader.fieldnames:
            if field not in header:
                header.append(field)

if not header:
    print(f"[WARN] feature CSVs under {feature_root} are empty")
    sys.exit(0)

merged_csv.parent.mkdir(parents=True, exist_ok=True)
with merged_csv.open("w", encoding="utf-8", newline="") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=header)
    writer.writeheader()
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as in_f:
            reader = csv.DictReader(in_f)
            if not reader.fieldnames:
                continue
            for row in reader:
                writer.writerow({field: row.get(field, "") for field in header})
                rows_written += 1

print(f"[ OK ] merged {len(csv_paths)} feature CSVs into {merged_csv} ({rows_written} rows)")
PY
}

select_feature_subset_stage() {
  local input_root="$1"
  local output_root="$2"

  SELECT_FEATURE_SUBSET_PROFILE_ROOT="$PROFILE_ROOT" \
  "$PYTHON_BIN" -u "$ANALYSIS_DIR/select_feature_subset_no_trace.py" \
    --input "$input_root" \
    --output "$output_root" \
    --dur-source "$SELECT_FEATURE_SUBSET_DUR_SOURCE"
}

completed=0
failed=0
combo_count=0

for ((batch_size = BATCH_START; batch_size <= BATCH_END; batch_size += BATCH_STEP)); do
  for ((num_indices = NUM_INDICES_START; num_indices <= NUM_INDICES_END; num_indices += NUM_INDICES_STEP)); do
    ((combo_count += 1))
    if (( MAX_COMBOS > 0 && combo_count > MAX_COMBOS )); then
      break 2
    fi

    combo_tag="bs${batch_size}_nip${num_indices}"
    shape_csv="$OP_SHAPES_DIR/op_shapes_${batch_size}_${num_indices}.csv"
    profile_dir="$PROFILE_ROOT/$combo_tag"
    final_feature_csv="$FEATURE_DATASET_ROOT/${combo_tag}.csv"
    combo_onnx_dir="$GENERATED_ONNX_WORK_ROOT/$combo_tag"
    combo_source_onnx="$combo_onnx_dir/$(basename "$ONNX_PATH")"
    log_dir="$LOG_ROOT/$combo_tag"
    submodel_dir=""
    if [[ "$RUNNER_MODE" == "branch_parallel" ]]; then
      if [[ -n "$BRANCH_SUBMODEL_ROOT" ]]; then
        submodel_dir="$BRANCH_SUBMODEL_ROOT/$combo_tag"
      else
        submodel_dir="$OUT_ROOT/branch_parallel_submodels/$combo_tag"
      fi
    fi

    infer_log="$log_dir/run_ort.log"
    cpu_parse_log="$log_dir/extract_cpu_threads.log"
    dataset_log="$log_dir/build_training_features.log"

    mkdir -p "$profile_dir" "$combo_onnx_dir" "$log_dir"
    if [[ -n "$submodel_dir" ]]; then
      mkdir -p "$submodel_dir"
    fi
    prepare_combo_onnx "$combo_source_onnx"

    echo "=================================================================="
    echo "[COMBO] $combo_tag"
    echo "  runner    : $RUNNER_MODE"
    echo "  shape_csv : $shape_csv"
    echo "  profile   : $profile_dir"
    echo "  dataset   : $final_feature_csv"

    infer_ran=0
    if [[ "$RESUME" != "1" ]] || [[ ! -f "$shape_csv" ]] || ! has_profile_json "$profile_dir"; then
      echo "[RUN ] inference + profiling"
      if ! run_inference_stage "$combo_source_onnx" "$batch_size" "$num_indices" "$shape_csv" "$profile_dir" "$submodel_dir" "$infer_log"; then
        echo "[FAIL] inference stage: $combo_tag"
        append_summary "$batch_size" "$num_indices" "FAILED" "inference" "$shape_csv" "$profile_dir" "" "" "" "$final_feature_csv" "$log_dir"
        cleanup_combo_onnx_dir "$combo_onnx_dir"
        ((failed += 1))
        continue
      fi
      infer_ran=1
    else
      echo "[SKIP] inference already completed"
    fi

    profile_json=$(find_latest_profile_json "$profile_dir")
    if [[ -z "$profile_json" || ! -f "$profile_json" ]]; then
      echo "[FAIL] cannot resolve profile JSON: $combo_tag"
      append_summary "$batch_size" "$num_indices" "FAILED" "profile_json" "$shape_csv" "$profile_dir" "" "" "" "$final_feature_csv" "$log_dir"
      cleanup_combo_onnx_dir "$combo_onnx_dir"
      ((failed += 1))
      continue
    fi

    profile_stem=$(basename "${profile_json%.json}")
    cpu_detail_csv="$profile_dir/${profile_stem}_cpu_thread_detail.csv"
    cpu_agg_csv="$profile_dir/${profile_stem}_cpu_thread_aggregated.csv"
    aligned_cpu_detail_csv="$profile_dir/${profile_stem}_cpu_thread_detail_aligned.csv"
    cpu_node_agg_csv="$profile_dir/${profile_stem}_cpu_thread_node_aggregated.csv"
    cpu_unmatched_csv="$profile_dir/${profile_stem}_cpu_thread_unmatched.csv"

    cpu_parse_ran=0
    if [[ "$RESUME" != "1" ]] || [[ "$infer_ran" == "1" ]] || ! cpu_parse_completed "$cpu_detail_csv" "$cpu_agg_csv"; then
      echo "[RUN ] extract CPU thread usage"
      if ! run_cpu_parse_stage "$profile_json" "$profile_dir" "$cpu_parse_log"; then
        echo "[FAIL] cpu profile parse stage: $combo_tag"
        append_summary "$batch_size" "$num_indices" "FAILED" "cpu_profile_parse" "$shape_csv" "$profile_dir" "$profile_json" "" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
        cleanup_combo_onnx_dir "$combo_onnx_dir"
        ((failed += 1))
        continue
      fi
      cpu_parse_ran=1
    else
      echo "[SKIP] CPU thread extraction already completed"
    fi

    if ! effective_onnx_path=$(resolve_effective_onnx_path "$combo_source_onnx" "$infer_log"); then
      echo "[FAIL] cannot resolve rewritten ONNX path: $combo_tag"
      append_summary "$batch_size" "$num_indices" "FAILED" "resolve_onnx" "$shape_csv" "$profile_dir" "$profile_json" "" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
      cleanup_combo_onnx_dir "$combo_onnx_dir"
      ((failed += 1))
      continue
    fi

    if [[ "$STOP_AFTER_INFER" == "1" ]]; then
      echo "[SMOKE] stop after inference requested; skipping dataset merge"
      append_summary "$batch_size" "$num_indices" "OK" "smoke_only" "$shape_csv" "$profile_dir" "$profile_json" "$effective_onnx_path" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
      cleanup_combo_onnx_dir "$combo_onnx_dir"
      ((completed += 1))
      continue
    fi

    if [[ "$RESUME" != "1" ]] || [[ "$cpu_parse_ran" == "1" ]] || ! dataset_completed "$final_feature_csv" "$aligned_cpu_detail_csv"; then
      echo "[RUN ] build no-trace training features"
      if ! run_dataset_stage "$batch_size" "$num_indices" "$shape_csv" "$cpu_detail_csv" "$aligned_cpu_detail_csv" "$cpu_node_agg_csv" "$cpu_unmatched_csv" "$final_feature_csv" "$dataset_log"; then
        echo "[FAIL] training features stage: $combo_tag"
        append_summary "$batch_size" "$num_indices" "FAILED" "training_features" "$shape_csv" "$profile_dir" "$profile_json" "$effective_onnx_path" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
        cleanup_combo_onnx_dir "$combo_onnx_dir"
        ((failed += 1))
        continue
      fi
    else
      echo "[SKIP] training features already completed"
    fi

    append_summary "$batch_size" "$num_indices" "OK" "" "$shape_csv" "$profile_dir" "$profile_json" "$effective_onnx_path" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
    cleanup_combo_onnx_dir "$combo_onnx_dir"
    ((completed += 1))
  done
done

echo "=================================================================="
echo "[DONE] completed=$completed failed=$failed"
echo "[SUMMARY] $SUMMARY_CSV"

if (( completed > 0 )); then
  echo "[MERGE] full dataset"
  if ! merge_feature_datasets "$FEATURE_DATASET_ROOT" "$FEATURE_DATASET_MERGED_CSV"; then
    echo "[WARN] failed to merge per-combo feature CSVs"
  fi
  echo "  merged    : $FEATURE_DATASET_MERGED_CSV"

  if [[ "$SELECT_FEATURE_SUBSET" == "1" ]]; then
    if ! select_feature_subset_stage "$FEATURE_DATASET_ROOT" "$FEATURE_SUBSET_ROOT"; then
      echo "[WARN] failed to build selected no-trace feature subset"
    fi
    if ! merge_feature_datasets "$FEATURE_SUBSET_ROOT" "$FEATURE_SUBSET_MERGED_CSV"; then
      echo "[WARN] failed to merge selected no-trace feature subset CSVs"
    fi
    echo "  selected  : $FEATURE_SUBSET_ROOT"
    echo "  sel_merge : $FEATURE_SUBSET_MERGED_CSV"
  fi
fi
