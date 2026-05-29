#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
OUT_DIR=${1:-"${ROOT_DIR}/raw/local"}
NUMA_NODE=${2:-0}
CPU=${3:-0}
THREADS=${4:-1}
BUILD_DIR=${ROOT_DIR}/build

mkdir -p "${OUT_DIR}"
"${SCRIPT_DIR}/build_benchmarks.sh" "${BUILD_DIR}" >/dev/null

lscpu > "${OUT_DIR}/lscpu.txt"
numactl -H > "${OUT_DIR}/numactl_H.txt"
lscpu -B -C > "${OUT_DIR}/lscpu_cache.txt"
perf --version > "${OUT_DIR}/perf_version.txt"
for f in /sys/devices/system/edac/mc/mc0/csrow*/mem_type; do
  printf '%s:%s\n' "$f" "$(cat "$f")"
done | sort -u > "${OUT_DIR}/edac_mem_type.txt"
for f in /sys/devices/system/edac/mc/mc0/csrow*/size_mb; do
  printf '%s:%s\n' "$f" "$(cat "$f")"
done | sort -u > "${OUT_DIR}/edac_size_mb.txt"

: > "${OUT_DIR}/ptrchase_sweep.txt"
for size in 32768 49152 65536 131072 262144 393216 524288 786432 1048576 2097152 8388608 16777216 25165824; do
  nodes=$((size / 64))
  rounds=$((600000000 / nodes))
  if [ "${rounds}" -lt 200 ]; then
    rounds=200
  fi
  numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
    "${BUILD_DIR}/aarch64_cpu_calib" ptrchase "${CPU}" "${size}" "${rounds}" >> "${OUT_DIR}/ptrchase_sweep.txt"
done
numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/aarch64_cpu_calib" ptrchase "${CPU}" 67108864 95 > "${OUT_DIR}/ptrchase_mem_64m.txt"
numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/aarch64_cpu_calib" ptrchase "${CPU}" 268435456 24 > "${OUT_DIR}/ptrchase_mem_256m.txt"

perf stat -x, -e cycles,task-clock,instructions \
  "${BUILD_DIR}/aarch64_cpu_calib" issue "${CPU}" 200000000 \
  > "${OUT_DIR}/issue_width.stdout.txt" 2> "${OUT_DIR}/issue_width.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_cpu_calib" scalar_chain "${CPU}" fadd64 50000000 \
  > "${OUT_DIR}/scalar_fadd64.stdout.txt" 2> "${OUT_DIR}/scalar_fadd64.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_cpu_calib" scalar_chain "${CPU}" fmul64 50000000 \
  > "${OUT_DIR}/scalar_fmul64.stdout.txt" 2> "${OUT_DIR}/scalar_fmul64.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_cpu_calib" scalar_chain "${CPU}" fma64 50000000 \
  > "${OUT_DIR}/scalar_fma64.stdout.txt" 2> "${OUT_DIR}/scalar_fma64.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_cpu_calib" neon_chain "${CPU}" fadd4s 50000000 \
  > "${OUT_DIR}/neon_fadd4s.stdout.txt" 2> "${OUT_DIR}/neon_fadd4s.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_cpu_calib" neon_chain "${CPU}" fmul4s 50000000 \
  > "${OUT_DIR}/neon_fmul4s.stdout.txt" 2> "${OUT_DIR}/neon_fmul4s.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_cpu_calib" neon_chain "${CPU}" fma4s 50000000 \
  > "${OUT_DIR}/neon_fma4s.stdout.txt" 2> "${OUT_DIR}/neon_fma4s.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_neon_fp_throughput" "${CPU}" fadd4s 50000000 \
  > "${OUT_DIR}/neon_throughput_fadd4s.stdout.txt" 2> "${OUT_DIR}/neon_throughput_fadd4s.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_neon_fp_throughput" "${CPU}" fmul4s 50000000 \
  > "${OUT_DIR}/neon_throughput_fmul4s.stdout.txt" 2> "${OUT_DIR}/neon_throughput_fmul4s.perf.csv"
perf stat -x, -e cycles,instructions \
  "${BUILD_DIR}/aarch64_neon_fp_throughput" "${CPU}" fma4s 50000000 \
  > "${OUT_DIR}/neon_throughput_fma4s.stdout.txt" 2> "${OUT_DIR}/neon_throughput_fma4s.perf.csv"

env OMP_NUM_THREADS="${THREADS}" numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/numa_stream_omp" 50000000 5 3.0 > "${OUT_DIR}/stream_triad.txt"

env OMP_NUM_THREADS=1 numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/copy_bw_omp" 50000000 5 > "${OUT_DIR}/copy_bw_1t.txt"
env OMP_NUM_THREADS="${THREADS}" numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/copy_bw_omp" 50000000 5 > "${OUT_DIR}/copy_bw_saturated.txt"

env OMP_NUM_THREADS=1 numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/random_gather_bw_omp" 25000000 5 > "${OUT_DIR}/random_gather_bw_1t.txt"
env OMP_NUM_THREADS="${THREADS}" numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/random_gather_bw_omp" 25000000 5 > "${OUT_DIR}/random_gather_bw_saturated.txt"

env OMP_NUM_THREADS="${THREADS}" numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/thread_launch_overhead_omp" parallel_for_empty 10000 > "${OUT_DIR}/thread_launch_overhead_intra.txt"
env OMP_NUM_THREADS="${THREADS}" numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${BUILD_DIR}/thread_launch_overhead_omp" task_spawn_empty 5000 > "${OUT_DIR}/thread_launch_overhead_inter.txt"

printf 'captured local calibration into %s\n' "${OUT_DIR}"
