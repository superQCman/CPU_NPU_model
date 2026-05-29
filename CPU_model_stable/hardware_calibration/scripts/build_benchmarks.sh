#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
OUT_DIR=${1:-"${ROOT_DIR}/build"}

mkdir -p "${OUT_DIR}"

gcc -O3 -march=armv8.2-a \
  -o "${OUT_DIR}/aarch64_cpu_calib" \
  "${ROOT_DIR}/tools/aarch64_cpu_calib.c"

gcc -O3 -march=armv8.2-a \
  -o "${OUT_DIR}/aarch64_neon_fp_throughput" \
  "${ROOT_DIR}/tools/aarch64_neon_fp_throughput.c"

if gcc -O3 -march=armv8.2-a+sve \
  -o "${OUT_DIR}/aarch64_sve_fp_throughput" \
  "${ROOT_DIR}/tools/aarch64_sve_fp_throughput.c" >/dev/null 2>&1; then
  :
fi

gcc -Ofast -ffast-math -funroll-loops -fopenmp -march=armv8.2-a \
  -o "${OUT_DIR}/numa_stream_omp" \
  "${ROOT_DIR}/tools/numa_stream_omp.c"

gcc -Ofast -ffast-math -funroll-loops -fopenmp -march=armv8.2-a \
  -o "${OUT_DIR}/copy_bw_omp" \
  "${ROOT_DIR}/tools/copy_bw_omp.c"

gcc -Ofast -ffast-math -funroll-loops -fopenmp -march=armv8.2-a \
  -o "${OUT_DIR}/random_gather_bw_omp" \
  "${ROOT_DIR}/tools/random_gather_bw_omp.c"

gcc -Ofast -ffast-math -funroll-loops -fopenmp -march=armv8.2-a \
  -o "${OUT_DIR}/thread_launch_overhead_omp" \
  "${ROOT_DIR}/tools/thread_launch_overhead_omp.c"

printf 'built %s\n' "${OUT_DIR}"
