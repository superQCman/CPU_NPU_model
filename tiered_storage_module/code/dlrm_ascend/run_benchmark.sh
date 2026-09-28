#!/bin/bash
set -euo pipefail

# ============================================================
# 记录整体耗时
# ============================================================
OVERALL_START=$(date +%s.%N)
echo "=============================================="
echo "  Benchmark started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# 结果汇总文件
TIMING_LOG="timing_results.txt"
echo "Timing Results - $(date '+%Y-%m-%d %H:%M:%S')" > "$TIMING_LOG"
echo "==============================================" >> "$TIMING_LOG"

# ============================================================
# Step 1: reuse_distance.py
# ============================================================
for N in 1000 5000 10000 50000 100000 500000 1000000; do
  echo ""
  echo "=== [Step 1] reuse_distance.py N=$N ==="

  STEP_START=$(date +%s.%N)

  python reuse_distance.py \
    --num-embeddings $N \
    --trace-length 50000 \
    --th-hbm 1000 --th-dram 10000 \
    --output-json reuse_n${N}.json \
    --seed 42 \
    --fast-hit-rates \
    > reuse_n${N}.log 2>&1

  STEP_END=$(date +%s.%N)
  STEP_ELAPSED=$(echo "$STEP_END - $STEP_START" | bc)
  echo "  [TIMING] reuse_distance N=$N: ${STEP_ELAPSED}s"
  echo "reuse_distance N=$N: ${STEP_ELAPSED}s" >> "$TIMING_LOG"
done

# ============================================================
# Step 2: memory_model.py
# ============================================================
for N in 1000 5000 10000 50000 100000 500000 1000000; do
  echo ""
  echo "=== [Step 2] memory_model.py N=$N ==="

  STEP_START=$(date +%s.%N)

  python memory_model.py \
    --from-reuse reuse_n${N}.json \
    --total-accesses 25600 \
    --dim 64000

  STEP_END=$(date +%s.%N)
  STEP_ELAPSED=$(echo "$STEP_END - $STEP_START" | bc)
  echo "  [TIMING] memory_model N=$N: ${STEP_ELAPSED}s"
  echo "memory_model N=$N: ${STEP_ELAPSED}s" >> "$TIMING_LOG"
done

# ============================================================
# 汇总
# ============================================================
OVERALL_END=$(date +%s.%N)
OVERALL_ELAPSED=$(echo "$OVERALL_END - $OVERALL_START" | bc)

echo ""
echo "=============================================="
echo "  Benchmark finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total elapsed: ${OVERALL_ELAPSED}s"
echo "=============================================="

echo "" >> "$TIMING_LOG"
echo "==============================================" >> "$TIMING_LOG"
echo "Total elapsed: ${OVERALL_ELAPSED}s" >> "$TIMING_LOG"

echo ""
echo "Timing log saved to: $TIMING_LOG"
cat "$TIMING_LOG"
