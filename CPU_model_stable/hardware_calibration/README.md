# 硬件校准

本目录保存 `cross_hw_unseen_axis_calibrator` 使用的单 NUMA 硬件 profile 校准证据，目标是让硬件 YAML 的每个关键字段都能追溯到真实观测、microbenchmark 测量或明确记录的派生规则。

## 范围

- 本机：`Kunpeng-920`
- 远端：`S920X20`

## 严格约束

- 写回 `hw_config/*.yaml` 的值必须来自直接观测、随附工具测量，或带来源说明的派生值。
- 不应静默重新引入占位值。

## 目录结构

| 路径 | 说明 |
|---|---|
| `tools/aarch64_cpu_calib.c` | pointer-chase 和依赖指令 microbenchmark。 |
| `tools/aarch64_neon_fp_throughput.c` | 独立 NEON 向量吞吐 microbenchmark。 |
| `tools/aarch64_sve_fp_throughput.c` | 独立 SVE 向量吞吐 microbenchmark。 |
| `tools/numa_stream_omp.c` | NUMA-local STREAM 风格带宽探针。 |
| `tools/copy_bw_omp.c` | OpenMP copy bandwidth 探针，用于 whitebox copy-like prior。 |
| `tools/random_gather_bw_omp.c` | OpenMP random gather bandwidth 探针，用于 whitebox Gather prior。 |
| `tools/thread_launch_overhead_omp.c` | OpenMP 线程启动/调度开销探针。 |
| `scripts/build_benchmarks.sh` | 构建本地 benchmark 二进制。 |
| `scripts/auto_calibrate_hw_config.py` | 一键自动校准、生成 YAML candidate、执行质量门禁，并可选择提升为正式配置。 |
| `scripts/capture_local.sh` | 采集本机完整证据包。 |
| `scripts/capture_remote.sh` | 通过 SSH 和用户提供的 identity file 采集远端证据包。 |
| `scripts/run_reproduce_hw_profile.sh` | 主机本地一键复现实验脚本，可在任一机器运行，并支持 repeat count。 |
| `scripts/reproduce_hw_profile.py` | shell wrapper 背后的采集、汇总和 YAML 检查 driver。 |
| `scripts/repeat_reproduce_hw_profile.py` | 多次运行聚合 driver，输出 median 和 stability summary。 |
| `raw/` | 当前配置使用的原始命令输出和 `perf stat` CSV 日志。 |
| `raw/local_convergence_5run/` | 当前 `kunpeng920_single_numa.yaml` 使用的本机 5-run 收敛汇总。 |
| `CALIBRATION_REPORT.md` | 当前 YAML 取值的选择依据和精确写入值。 |

## 测量约定

- cache latency：单线程、CPU 绑定、NUMA-local pointer-chase，单位 `ns/load`。
- instruction latency：依赖 FP chain，通过 `perf stat cycles / logical_ops` 换算。
- instruction throughput：独立向量循环，通过 `logical_vector_insts / cycles` 换算。
- pipeline width：独立 integer-add loop 上 `perf stat instructions / cycles` 得到的稳态标量发射宽度代理。
- CPU frequency：同一 pinned issue-loop microbench 上由 `perf stat cycles / task-clock` 得到的有效 GHz。
- memory bandwidth：NUMA-local OpenMP STREAM triad 的 best-of-N 结果，单位 GB/s。
- whitebox 附加代理：copy bandwidth、random gather bandwidth 和 thread launch overhead 可由本机 microbenchmark 输出；如果这些文件不存在，当前 source-only whitebox route 会从已有 YAML 做确定性派生，而不会把猜测值写回 `hw_config/*.yaml`。

## 使用说明

当前主线硬件 YAML 目录是：

```text
calibrator/hw_config
```

旧的 `CPU_model_stable/hw_config` 位置不再作为当前模型主入口使用。

生产使用建议走自动入口：

```bash
/data/qc/miniconda3/envs/ort/bin/python hardware_calibration/scripts/auto_calibrate_hw_config.py \
  --profile-name <profile> \
  --numa-node 0 \
  --repeats 5 \
  --output-dir hardware_calibration/raw/<profile>_auto_<date> \
  --promote-if-pass
```

`auto_calibrate_hw_config.py` 会先写出 `candidate_hw_config.yaml`，只有所有门禁通过后才提升为 `calibrator/hw_config/<profile>.yaml`。旧 YAML 会备份为 `<profile>.yaml.bak.<timestamp>`。

自动门禁包括必需字段覆盖、microbenchmark 稳定性、YAML 字段一致性和硬件 profile 可复现性。
