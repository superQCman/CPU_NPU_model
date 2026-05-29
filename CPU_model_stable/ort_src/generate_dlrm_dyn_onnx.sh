#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

ASCEND_ENV_SH=${ASCEND_ENV_SH:-}
DEFAULT_PYTHON_BIN=${DEFAULT_PYTHON_BIN:-}
DEFAULT_PYTHON_SITE_PACKAGES=${DEFAULT_PYTHON_SITE_PACKAGES:-}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "$DEFAULT_PYTHON_BIN" && -x "$DEFAULT_PYTHON_BIN" ]]; then
    PYTHON_BIN="$DEFAULT_PYTHON_BIN"
  else
    PYTHON_BIN=python3
  fi
fi
PYTHON_SITE_PACKAGES=${PYTHON_SITE_PACKAGES:-$DEFAULT_PYTHON_SITE_PACKAGES}
USE_ASCEND_ENV=${USE_ASCEND_ENV:-0}

if [[ "$USE_ASCEND_ENV" == "1" && -n "$ASCEND_ENV_SH" && -f "$ASCEND_ENV_SH" ]]; then
  # shellcheck disable=SC1090
  source "$ASCEND_ENV_SH"
fi

DLRM_SCRIPT=${DLRM_SCRIPT:-"$REPO_ROOT/dlrm_msprof.py"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$SCRIPT_DIR/dlrm_onnx_dyn"}
ONNX_NAME=${ONNX_NAME:-dlrm_s_pytorch.onnx}
VARIANT_FILTER=${VARIANT_FILTER:-}

MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
NUM_BATCHES=${NUM_BATCHES:-1}
NUM_INDICES_PER_LOOKUP=${NUM_INDICES_PER_LOOKUP:-100}
NUM_WORKERS=${NUM_WORKERS:-1}
DATA_GENERATION=${DATA_GENERATION:-random}
DATA_SIZE=${DATA_SIZE:-1}
SAVE_CSV=${SAVE_CSV:-0}

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$DLRM_SCRIPT" ]]; then
  echo "ERROR: dlrm_msprof.py not found: $DLRM_SCRIPT" >&2
  exit 1
fi

MANIFEST_CSV="$OUTPUT_ROOT/manifest.csv"
echo "variant,arch_embedding_size,arch_mlp_bot,arch_mlp_top,onnx_path" > "$MANIFEST_CSV"

# Portable default: reproduce the scheduler_20260419 source dataset.
CONFIGS=(
  "case_7|25600-25600-25600-25600-25600-25600-25600-25600|400-300-200|1600-800-1"
  "case_8|25600-25600-25600-25600-25600-25600-25600-25600|800-800-400|1600-800-1"
  "case_9|51200-51200-51200-51200-51200-51200-51200-51200|400-300-200|1600-800-1"
  "case_10|51200-51200-51200-51200-51200-51200-51200-51200|800-800-400|1600-800-1"
)

variant_enabled() {
  local variant="$1"
  if [[ -z "$VARIANT_FILTER" ]]; then
    return 0
  fi

  local needle
  IFS=',' read -ra needles <<< "$VARIANT_FILTER"
  for needle in "${needles[@]}"; do
    needle="${needle//[[:space:]]/}"
    [[ -n "$needle" && "$needle" == "$variant" ]] && return 0
  done
  return 1
}

echo "[GEN] output_root=$OUTPUT_ROOT"
echo "[GEN] mini_batch_size=$MINI_BATCH_SIZE num_batches=$NUM_BATCHES num_indices_per_lookup=$NUM_INDICES_PER_LOOKUP num_workers=$NUM_WORKERS"
echo "[GEN] data_generation=$DATA_GENERATION data_size=$DATA_SIZE"
echo "[GEN] python_bin=$PYTHON_BIN"
echo "[GEN] python_site_packages=$PYTHON_SITE_PACKAGES"
echo "[GEN] ascend_env=${USE_ASCEND_ENV}"
echo "[GEN] variant_filter=${VARIANT_FILTER:-<unset>}"

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r variant emb bot top <<< "$spec"
  if ! variant_enabled "$variant"; then
    echo "[GEN] skip variant=$variant (filtered out)"
    continue
  fi
  variant_dir="$OUTPUT_ROOT/$variant"
  onnx_path="$variant_dir/$ONNX_NAME"
  mkdir -p "$variant_dir"

  echo "=================================================================="
  echo "[GEN] variant=$variant"
  echo "  emb=$emb"
  echo "  bot=$bot"
  echo "  top=$top"
  echo "  onnx=$onnx_path"

  sparse_feature_size="${bot##*-}"
  echo "  sparse_feature_size=$sparse_feature_size"

  (
    cd "$REPO_ROOT"
    # Keep optional Ascend/Torch-NPU toggles controlled by the caller's environment.
    PYTHONPATH="$REPO_ROOT${PYTHON_SITE_PACKAGES:+:$PYTHON_SITE_PACKAGES}${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -S "$DLRM_SCRIPT" \
      --use-gpu \
      --num-threads=8 \
      --arch-sparse-feature-size "$sparse_feature_size" \
      --arch-embedding-size "$emb" \
      --arch-mlp-bot "$bot" \
      --arch-mlp-top "$top" \
      --mini-batch-size "$MINI_BATCH_SIZE" \
      --num-batches "$NUM_BATCHES" \
      --num-indices-per-lookup "$NUM_INDICES_PER_LOOKUP" \
      --num-workers "$NUM_WORKERS" \
      --data-size "$DATA_SIZE" \
      --data-generation "$DATA_GENERATION" \
      --save-onnx \
      --num-indices-per-lookup-fixed=True \
      --onnx-output-root "$OUTPUT_ROOT" \
      --onnx-output-subdir "$variant" \
      --onnx-output-name "$ONNX_NAME"
  )

  if [[ ! -f "$onnx_path" ]]; then
    echo "ERROR: expected ONNX not found: $onnx_path" >&2
    exit 1
  fi

  printf '%s,%s,%s,%s,%s\n' \
    "$variant" "$emb" "$bot" "$top" "$onnx_path" >> "$MANIFEST_CSV"
  echo "[GEN] saved: $onnx_path"
done

echo "[GEN] manifest: $MANIFEST_CSV"
