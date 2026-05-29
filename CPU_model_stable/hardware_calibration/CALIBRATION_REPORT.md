# Cross-Host Calibration Report

Date:

- 2026-04-19
- 2026-04-21
- 2026-04-27

Scope:

- local source host: `Kunpeng-920`
- remote target host: `S920X20`
- profile output paths:
  - `hw_config/kunpeng920_single_numa.yaml`
  - `hw_config/s920x20_single_numa.yaml`

Primary evidence:

- local raw bundle:
  - `hardware_calibration/raw/local/`
- local convergence bundle:
  - `hardware_calibration/raw/local_convergence_5run/`
- remote raw bundle:
  - `hardware_calibration/raw/remote/`
- remote whitebox add-on bundle:
  - `hardware_calibration/raw/remote_addon_20260427/`
- remote 3-run recheck bundle:
  - `hardware_calibration/raw/remote_recheck_20260427_3run/`
- benchmark sources:
  - `hardware_calibration/tools/aarch64_cpu_calib.c`
  - `hardware_calibration/tools/numa_stream_omp.c`
  - `hardware_calibration/tools/copy_bw_omp.c`
  - `hardware_calibration/tools/random_gather_bw_omp.c`
  - `hardware_calibration/tools/thread_launch_overhead_omp.c`

## What Was Calibrated

- single-NUMA core count
- cache sizes and associativity
- effective random-access latency proxies for L1/L2/L3/DRAM
- CPU frequency
- effective pipeline width proxy
- vector-width capability
- scalar/vector FP latency proxies
- vector FP throughput proxies
- sustained NUMA-local memory bandwidth
- whitebox-only copy bandwidth proxies at `1` thread and saturated single-NUMA threads
- whitebox-only random-gather bandwidth proxies at `1` thread and saturated single-NUMA threads
- whitebox-only OpenMP launch overhead proxies for intra-op and inter-op scheduling

## Selection Rules

- L1 proxy:
  use the `48 KiB` pointer-chase point because it stays below the `64 KiB` L1 size but avoids trivial warm-cache artifacts.
- L2 proxy:
  use a point that is comfortably above L1 and below the nominal L2 capacity:
  - local: `256 KiB`
  - remote: `1 MiB`
- L3 proxy:
  use a point that is clearly beyond L2 but still inside the LLC regime before the DRAM-sized working set:
  - local: `16 MiB`
  - remote: `70 MiB`
- DRAM proxy:
  use `256 MiB` pointer-chase.
- pipeline width:
  use `perf stat instructions / cycles` from the independent integer-add loop.
- vector throughput:
  use independent vector loops and compute `logical_vector_insts / cycles`.
- bandwidth:
  use the best `STREAM triad` result at the full single-NUMA thread count.
- CPU frequency:
  use `perf stat cycles / task-clock` from the same pinned issue-loop microbench used for the issue-width proxy.
- reproducibility check:
  the wrapper script treats topology and byte-exact cache sizes as exact matches, but allows bounded relative drift on microbenchmark-derived proxies because `perf` and memory traffic vary slightly between runs.
- local convergence rule:
  use the `5-run` median after reducing host background load, and prefer the cache working-set point with the better repeatability when two candidate points both fit the intended cache level.

## Calibrated Values

| Parameter | Local Kunpeng-920 | Remote S920X20 | Evidence |
| --- | ---: | ---: | --- |
| single NUMA cores | `24` | `40` | `lscpu`, `numactl -H` |
| L1D / L1I size | `64 KiB / 64 KiB` | `64 KiB / 64 KiB` | `lscpu -B -C` |
| L2 size | `512 KiB` | `1280 KiB` | `lscpu -B -C` |
| L3 size | `24 MiB` | `70 MiB` | `lscpu -B -C` |
| L1 random-access proxy | `4.4992 ns` | `4.5227 ns` | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/l1_focus_ptrchase_10run.txt` |
| L2 random-access proxy | `8.2301 ns @ 256 KiB` | `10.9299 ns @ 1 MiB` | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |
| L3 random-access proxy | `46.9976 ns @ 16 MiB` | `43.4575 ns @ 70 MiB` | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |
| DRAM random-access proxy | `91.0687 ns @ 256 MiB` | `87.7740 ns @ 256 MiB` | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |
| CPU frequency | `2.592166 GHz` | `2.895478 GHz` | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |
| effective issue-width proxy | `2.938467` | `3.891631` | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |
| hardware-max vector width | `128-bit NEON` | `256-bit SVE` | `lscpu`, `sve_default_vector_length.txt` |
| scalar FP latency proxy | `add 4.000206`, `mul 5.000218`, `fma 7.000221` cycles | `add 2.000105`, `mul 3.000098`, `fma 4.000091` cycles | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |
| vector FP latency proxy | `add 5.000226`, `mul 5.000210`, `fma 4.000215` cycles | `add 2.000090`, `mul 3.000092`, `fma 2.000087` cycles | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |
| vector FP throughput proxy | `add 1.499752`, `mul 1.499756`, `fma 1.749673` inst/cycle | `add 1.999797`, `mul 1.999816`, `fma 1.999818` inst/cycle | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |
| STREAM triad bandwidth | `41.0935 GB/s @ 24 threads` | `136.8997 GB/s @ 40 threads` | local `local_convergence_5run`, remote `remote_recheck_20260427_3run/convergence_summary.json` |

## 2026-04-21 Local Whitebox Add-On Measurements

These measurements were added to close the hard-policy whitebox prior loop on the local source host. They come from the dedicated OpenMP microbenches under `hardware_calibration/tools/` after rebuilding them into `/tmp/cross_hw_whitebox_build/` on `2026-04-21`.

| Parameter | Local Kunpeng-920 | Evidence |
| --- | ---: | --- |
| copy bandwidth, 1 thread | `5.048952 GB/s` | `copy_bw_omp 50000000 5`, `OMP_NUM_THREADS=1`, NUMA-local pinning |
| copy bandwidth, 24 threads | `30.820562 GB/s` | `copy_bw_omp 50000000 5`, `OMP_NUM_THREADS=24`, NUMA-local pinning |
| random gather bandwidth, 1 thread | `0.840198 GB/s` | `random_gather_bw_omp 25000000 5`, `OMP_NUM_THREADS=1`, NUMA-local pinning |
| random gather bandwidth, 24 threads | `4.160243 GB/s` | `random_gather_bw_omp 25000000 5`, `OMP_NUM_THREADS=24`, NUMA-local pinning |
| OpenMP intra-op launch overhead | `843.905855 us` | `thread_launch_overhead_omp parallel_for_empty 10000`, `OMP_NUM_THREADS=24`, NUMA-local pinning |
| OpenMP inter-op launch overhead | `832.877529 us` | `thread_launch_overhead_omp task_spawn_empty 5000`, `OMP_NUM_THREADS=24`, NUMA-local pinning |

These values are written directly into `hw_config/kunpeng920_single_numa.yaml` as whitebox-required fields:

- `memory.copy_bw_gb_s_1t`
- `memory.copy_bw_gb_s_saturated`
- `memory.random_gather_bw_gb_s_1t`
- `memory.random_gather_bw_gb_s_saturated`
- `runtime.parallel_launch_overhead_us_intra`
- `runtime.parallel_launch_overhead_us_inter`

## 2026-04-27 Remote Whitebox Add-On Measurements

These measurements were added to bring the remote target profile to the same whitebox hardware contract as the local source profile. The three OpenMP microbenches were first compiled in `/tmp/codex_s920x20_hw_addon/` on the S920X20 host and run with NUMA node 0 CPU and memory binding. They were rechecked later on 2026-04-27 with the full host-local reproducibility wrapper under `hardware_calibration/raw/remote_recheck_20260427_3run/`; the values below use that 3-run median recheck.

| Parameter | Remote S920X20 | Evidence |
| --- | ---: | --- |
| copy bandwidth, 1 thread | `36.235973 GB/s` | `copy_bw_omp 50000000 5`, `OMP_NUM_THREADS=1`, NUMA-local pinning |
| copy bandwidth, 40 threads | `142.035542 GB/s` | `copy_bw_omp 50000000 5`, `OMP_NUM_THREADS=40`, NUMA-local pinning |
| random gather bandwidth, 1 thread | `2.525676 GB/s` | `random_gather_bw_omp 25000000 5`, `OMP_NUM_THREADS=1`, NUMA-local pinning |
| random gather bandwidth, 40 threads | `20.180037 GB/s` | `random_gather_bw_omp 25000000 5`, `OMP_NUM_THREADS=40`, NUMA-local pinning |
| OpenMP intra-op launch overhead | `10.522971 us` | `thread_launch_overhead_omp parallel_for_empty 10000`, `OMP_NUM_THREADS=40`, NUMA-local pinning |
| OpenMP inter-op launch overhead | `19.426464 us` | `thread_launch_overhead_omp task_spawn_empty 5000`, `OMP_NUM_THREADS=40`, NUMA-local pinning |

These values are written directly into `hw_config/s920x20_single_numa.yaml` as whitebox-required fields:

- `memory.copy_bw_gb_s_1t`
- `memory.copy_bw_gb_s_saturated`
- `memory.random_gather_bw_gb_s_1t`
- `memory.random_gather_bw_gb_s_saturated`
- `runtime.parallel_launch_overhead_us_intra`
- `runtime.parallel_launch_overhead_us_inter`

## Remote 3-Run Recheck Snapshot

The remote host was remeasured across `3` sequential full calibration runs on 2026-04-27. The aggregated convergence summary is stored under `hardware_calibration/raw/remote_recheck_20260427_3run/`.

Key repeatability results:

- `cpu_clock_ghz`: median `2.895478`, `cv 0.001%`
- `dram_ns`: median `87.7740`, `cv 0.10%`
- `bandwidth_gb_s_total`: median `136.8997`, `cv 0.25%`
- `issue_width`: median `3.891631`, `cv ~0%`
- `l2_ns @ 1 MiB`: median `10.9299`, `cv 1.23%`
- `l3_ns @ 70 MiB`: median `43.4575`, `cv 0.64%`
- whitebox add-on medians:
  - copy bandwidth `36.235973 / 142.035542 GB/s` for `1 / 40` threads
  - random gather bandwidth `2.525676 / 20.180037 GB/s` for `1 / 40` threads
  - OpenMP launch overhead `10.522971 / 19.426464 us` for intra/inter probes

The full-wrapper L1 selected point varied across the 3 sequential runs (`4.4804`, `5.1042`, `5.0413 ns`). A focused 10-run 48KiB pointer-chase check on the same host produced median `4.5227 ns` with `cv 2.00%`, so the YAML uses that focused L1 median instead of the noisier 3-run full-wrapper median.

## Local 5-Run Convergence Snapshot

The local host was remeasured across `5` sequential runs after background load was reduced. The aggregated convergence summary is stored under `hardware_calibration/raw/local_convergence_5run/`.

Key repeatability results:

- `cpu_clock_ghz`: median `2.592166`, `cv 0.01%`
- `dram_ns`: median `91.0687`, `cv 1.40%`
- `bandwidth_gb_s_total`: median `41.0935`, `cv 0.83%`
- `issue_width`: median `2.938467`, `cv ~0%`
- `l2_ns @ 256 KiB`: median `8.2301`, `cv 4.36%`
- `l3_ns @ 16 MiB`: median `46.9976`, `cv 0.84%`

Why the local cache proxy points changed:

- the earlier `384 KiB` local L2 point had `cv 16.91%` across the same 5 runs, so it was too sensitive to reuse and background traffic
- the `16 MiB` local LLC point had lower drift than the earlier `8 MiB` point while still staying inside the `24 MiB` L3 capacity

## Additional Mismatches Beyond The Requested Fields

- host-scale topology differs:
  - local: `4 sockets / 8 NUMA / 192 cores`
  - remote: `2 sockets / 4 NUMA / 160 cores`
- per-NUMA memory capacity differs:
  - local: about `192.7 GiB`
  - remote: about `128.8 GiB`
- LLC associativity differs:
  - local: `15-way`
  - remote: `28-way`
- L2 associativity differs:
  - local: `8-way`
  - remote: `10-way`
- NUMA distance matrices differ substantially; the remote host has a sharper local-vs-remote split.
- observed DIMM population differs:
  - local EDAC rows are `64 GiB`
  - remote EDAC rows are `32 GiB`
- EDAC memory-type exposure differs:
  - local: `Registered-DDR4`
  - remote: `Registered-SDR`
  The remote EDAC label is too generic to support a strict DDR generation or speed claim, so those fields were omitted from the calibrated YAML.

## What Was Intentionally Not Claimed As Measured Hardware Truth

- remote DDR generation / MT/s
  The remote host did not expose a precise enough user-space memory-type string to write a defensible `dram_speed_mt_s` field.
- datasheet tag/data cache latencies
  The included pointer-chase only supports effective random-access latency proxies.

## Why The YAML Looks Different From The Older Profiles

- stale placeholders for `dram_speed_mt_s`, `channels_*`, `controllers_*`, and `remote_mem_delay_ns` were removed when they were not directly supportable
- `response_latency_cycles` now comes from calibrated ns proxies mapped through the selected CPU frequency
- all pipeline stage widths now store the same effective issue-width proxy because that is what the model actually consumes
