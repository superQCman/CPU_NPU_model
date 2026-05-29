#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <ssh_target> <identity_file> <output_dir> [remote_tmp_dir]" >&2
  exit 2
fi

SSH_TARGET=$1
IDENTITY_FILE=$2
OUT_DIR=$3
REMOTE_TMP_DIR=${4:-/tmp/codex_hw_calibration}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

mkdir -p "${OUT_DIR}"

SSH_BASE=(ssh -F none -i "${IDENTITY_FILE}" -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new "${SSH_TARGET}")
SCP_BASE=(scp -F none -i "${IDENTITY_FILE}" -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)

"${SSH_BASE[@]}" "mkdir -p '${REMOTE_TMP_DIR}'"
"${SCP_BASE[@]}" \
  "${ROOT_DIR}/tools/aarch64_cpu_calib.c" \
  "${ROOT_DIR}/tools/aarch64_neon_fp_throughput.c" \
  "${ROOT_DIR}/tools/aarch64_sve_fp_throughput.c" \
  "${ROOT_DIR}/tools/numa_stream_omp.c" \
  "${SSH_TARGET}:${REMOTE_TMP_DIR}/"
"${SSH_BASE[@]}" "gcc -O3 -march=armv8.2-a -o '${REMOTE_TMP_DIR}/aarch64_cpu_calib' '${REMOTE_TMP_DIR}/aarch64_cpu_calib.c'"
"${SSH_BASE[@]}" "gcc -O3 -march=armv8.2-a -o '${REMOTE_TMP_DIR}/aarch64_neon_fp_throughput' '${REMOTE_TMP_DIR}/aarch64_neon_fp_throughput.c'"
"${SSH_BASE[@]}" "gcc -O3 -march=armv8.2-a+sve -o '${REMOTE_TMP_DIR}/aarch64_sve_fp_throughput' '${REMOTE_TMP_DIR}/aarch64_sve_fp_throughput.c'"
"${SSH_BASE[@]}" "gcc -Ofast -ffast-math -funroll-loops -fopenmp -march=armv8.2-a -o '${REMOTE_TMP_DIR}/numa_stream_omp' '${REMOTE_TMP_DIR}/numa_stream_omp.c'"

"${SSH_BASE[@]}" "lscpu" > "${OUT_DIR}/lscpu.txt"
"${SSH_BASE[@]}" "numactl -H" > "${OUT_DIR}/numactl_H.txt"
"${SSH_BASE[@]}" "lscpu -B -C" > "${OUT_DIR}/lscpu_cache.txt"
"${SSH_BASE[@]}" "perf --version" > "${OUT_DIR}/perf_version.txt"
"${SSH_BASE[@]}" "cat /proc/sys/abi/sve_default_vector_length" > "${OUT_DIR}/sve_default_vector_length.txt"
"${SSH_BASE[@]}" "cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq" > "${OUT_DIR}/cpuinfo_max_freq.txt"
"${SSH_BASE[@]}" "for f in /sys/devices/system/edac/mc/mc*/csrow*/mem_type; do printf '%s:%s\n' \"\$f\" \"\$(cat \"\$f\")\"; done | sort -u" > "${OUT_DIR}/edac_mem_type.txt"
"${SSH_BASE[@]}" "for f in /sys/devices/system/edac/mc/mc*/csrow*/size_mb; do printf '%s:%s\n' \"\$f\" \"\$(cat \"\$f\")\"; done | sort -u" > "${OUT_DIR}/edac_size_mb.txt"

: > "${OUT_DIR}/ptrchase_sweep.txt"
for size in 32768 49152 65536 131072 262144 524288 1048576 1310720 2097152 8388608 33554432 67108864 73400320 268435456; do
  nodes=$((size / 64))
  rounds=$((300000000 / nodes))
  if [ "${rounds}" -lt 120 ]; then
    rounds=120
  fi
  "${SSH_BASE[@]}" "numactl --cpunodebind=0 --membind=0 '${REMOTE_TMP_DIR}/aarch64_cpu_calib' ptrchase 0 ${size} ${rounds}" >> "${OUT_DIR}/ptrchase_sweep.txt"
done

"${SSH_BASE[@]}" "perf stat -x, -e cycles,task-clock,instructions '${REMOTE_TMP_DIR}/aarch64_cpu_calib' issue 0 200000000" > "${OUT_DIR}/issue_width.stdout.txt" 2> "${OUT_DIR}/issue_width.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_cpu_calib' scalar_chain 0 fadd64 50000000" > "${OUT_DIR}/scalar_fadd64.stdout.txt" 2> "${OUT_DIR}/scalar_fadd64.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_cpu_calib' scalar_chain 0 fmul64 50000000" > "${OUT_DIR}/scalar_fmul64.stdout.txt" 2> "${OUT_DIR}/scalar_fmul64.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_cpu_calib' scalar_chain 0 fma64 50000000" > "${OUT_DIR}/scalar_fma64.stdout.txt" 2> "${OUT_DIR}/scalar_fma64.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_cpu_calib' neon_chain 0 fadd4s 50000000" > "${OUT_DIR}/neon_fadd4s.stdout.txt" 2> "${OUT_DIR}/neon_fadd4s.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_cpu_calib' neon_chain 0 fmul4s 50000000" > "${OUT_DIR}/neon_fmul4s.stdout.txt" 2> "${OUT_DIR}/neon_fmul4s.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_cpu_calib' neon_chain 0 fma4s 50000000" > "${OUT_DIR}/neon_fma4s.stdout.txt" 2> "${OUT_DIR}/neon_fma4s.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_sve_fp_throughput' 0 fadds 50000000" > "${OUT_DIR}/sve_throughput_fadds.stdout.txt" 2> "${OUT_DIR}/sve_throughput_fadds.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_sve_fp_throughput' 0 fmuls 50000000" > "${OUT_DIR}/sve_throughput_fmuls.stdout.txt" 2> "${OUT_DIR}/sve_throughput_fmuls.perf.csv"
"${SSH_BASE[@]}" "perf stat -x, -e cycles,instructions '${REMOTE_TMP_DIR}/aarch64_sve_fp_throughput' 0 fmas 50000000" > "${OUT_DIR}/sve_throughput_fmas.stdout.txt" 2> "${OUT_DIR}/sve_throughput_fmas.perf.csv"
"${SSH_BASE[@]}" "env OMP_NUM_THREADS=40 numactl --cpunodebind=0 --membind=0 '${REMOTE_TMP_DIR}/numa_stream_omp' 50000000 5 3.0" > "${OUT_DIR}/stream_triad.txt"

printf 'captured remote calibration into %s\n' "${OUT_DIR}"
