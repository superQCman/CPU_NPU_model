# Hardware Calibration

This directory holds the reproducible evidence for the calibrated single-NUMA
hardware profiles used by `cross_hw_unseen_axis_calibrator`.

Scope:

- local host: `Kunpeng-920`
- remote host: `S920X20`

The contract is strict:

- values written back into `hw_config/*.yaml` must be either directly observed,
  measured by the included tools, or explicitly documented as a derived value
  with its source called out
- placeholder values should not be reintroduced silently

Layout:

- `tools/`
  - `aarch64_cpu_calib.c`: pointer-chase and dependent instruction microbench
  - `aarch64_neon_fp_throughput.c`: independent NEON vector throughput microbench
  - `aarch64_sve_fp_throughput.c`: independent SVE vector throughput microbench
  - `numa_stream_omp.c`: NUMA-local STREAM-style bandwidth probe
  - `copy_bw_omp.c`: OpenMP copy-bandwidth probe for whitebox copy-like priors
  - `random_gather_bw_omp.c`: OpenMP random-gather bandwidth probe for whitebox Gather priors
  - `thread_launch_overhead_omp.c`: OpenMP launch/scheduling overhead probe for whitebox thread-overhead priors
- `scripts/`
  - `build_benchmarks.sh`: builds the local benchmark binaries
  - `auto_calibrate_hw_config.py`: one-command automatic calibration, YAML candidate generation, quality gates, and optional promotion into `calibrator/hw_config`
  - `capture_local.sh`: captures one full local evidence bundle
  - `capture_remote.sh`: captures the remote evidence bundle through SSH with a user-provided identity file
  - `run_reproduce_hw_profile.sh`: one-command host-local repro script that can run on either machine, with an optional repeat count
  - `reproduce_hw_profile.py`: capture + summarize + optional YAML-check driver behind the shell wrapper
  - `repeat_reproduce_hw_profile.py`: repeated-run aggregation driver that writes median/stability summaries across multiple captures
- `raw/`
  - raw command outputs and `perf stat` CSV logs used for the current configs
  - `raw/local_convergence_5run/`: local 5-run convergence summary used for the current `kunpeng920_single_numa.yaml`
- `CALIBRATION_REPORT.md`
  - the selection rationale and the exact values written into the current YAMLs

Measurement conventions:

- cache latency:
  single-thread, CPU-pinned, NUMA-local pointer-chase `ns/load`
- instruction latency:
  dependent FP chains, converted from `perf stat cycles / logical_ops`
- instruction throughput:
  independent vector loops, converted from `logical_vector_insts / cycles`
- pipeline width:
  effective steady-state scalar issue width proxy from
  `perf stat instructions / cycles` on the independent integer-add loop
- CPU frequency:
  effective GHz from `perf stat cycles / task-clock` on the same pinned
  issue-loop microbench
- memory bandwidth:
  NUMA-local OpenMP STREAM triad best-of-N result, in GB/s
- whitebox add-on memory/runtime proxies:
  optional host-local microbench outputs for copy bandwidth, random gather bandwidth,
  and thread launch overhead; if those files are absent the current source-only
  whitebox routes fall back to deterministic derivations from the existing YAML
  instead of writing guessed values back into `hw_config/*.yaml`

Notes:

- The mainline hardware YAML directory is `calibrator/hw_config`. The older
  `CPU_model_stable/hw_config` location is not used by the current model.
- For production use, prefer the automatic entrypoint:
  ```bash
  /data/qc/miniconda3/envs/ort/bin/python hardware_calibration/scripts/auto_calibrate_hw_config.py \
    --profile-name <profile> \
    --numa-node 0 \
    --repeats 5 \
    --output-dir hardware_calibration/raw/<profile>_auto_<date> \
    --promote-if-pass
  ```
- `auto_calibrate_hw_config.py` first writes `candidate_hw_config.yaml`, then
  promotes it to `calibrator/hw_config/<profile>.yaml` only if all gates pass.
  The previous YAML is backed up as `<profile>.yaml.bak.<timestamp>`.
- Automatic gates include required field coverage, microbenchmark stability,
  cache-probe legality, and `_load_hw_params(..., require_measured_whitebox=True)`
  readability. Failed gates keep the candidate and reports but do not update the
  trusted YAML.
- The main automatic reports are:
  - `calibration_gate_report.json/md`
  - `selected_probe_report.csv`
  - `yaml_diff_report.md`
  - `promotion_record.json`
  - `convergence_summary.json`
  - `convergence_stats.json`
- `response_latency_cycles` in the hardware YAML is the model-facing effective
  latency proxy. The current code path only consumes `response_latency_cycles`,
  so `tag_latency_cycles` and `data_latency_cycles` are mirrored to the same
  calibrated value to avoid leaving stale placeholders behind.
- Stage-by-stage pipeline widths are not directly observable from user space.
  The calibrated YAML therefore stores the same effective width proxy under all
  width fields because the model consumes `min(fetch, decode, rename, dispatch,
  issue, commit)`.
- `capture_remote.sh` assumes the SSH key is already usable in the current
  shell context. It does not embed passphrases.
- `run_reproduce_hw_profile.sh` can run directly on a host with just
  `bash + python3 + gcc + perf + numactl`; YAML comparison is skipped
  automatically if `PyYAML` is not installed.
- `run_reproduce_hw_profile.sh <profile> <output_dir> <numa_node> <repeats>`
  switches to the repeated-run aggregator automatically when `repeats > 1`.
- The YAML-check path uses exact matches for topology and byte-exact cache
  sizes, then bounded relative tolerances for microbenchmark-derived proxies to
  account for normal run-to-run noise.
