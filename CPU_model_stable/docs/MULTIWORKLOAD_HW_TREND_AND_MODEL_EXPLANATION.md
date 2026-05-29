# 多负载硬件趋势图与模型链路说明

## 1. 输出位置

多负载版本的趋势图和元数据文件位于：

```text
outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

该目录是趋势图实验目录。`outputs/cache_residency_worker_share_v3_full_s920x20_20260505` 和 `outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505` 是 correct-hw full 训练与评估目录，不包含 `synthetic_e2e_trend_summary_route_pred.csv` 这类 synthetic 趋势图产物。

命名约定：`Device_1` 表示智算平台，`Device_2` 表示通算平台。路径、alias 或 YAML 名称中的 `kunpeng920` 对应智算平台，`s920x20` 对应通算平台，保留这些历史标识是为了兼容已有实验。

历史固定全局 DLRM-S/M/L 版本仅作为补充诊断，不作为本文主线，位于：

```text
outputs/hw_generalization_physics_guarded_rf_v3_20260429/trend_figures_synthetic_model_physics_guarded_rf_v3_global_fixed_dlrm
```

图文件：

```text
synthetic_e2e_core_trends_route_pred.png/pdf
synthetic_e2e_cache_size_trends_route_pred.png/pdf
synthetic_e2e_cache_latency_trends_route_pred.png/pdf
synthetic_e2e_memory_trends_route_pred.png/pdf
synthetic_e2e_compute_pipeline_trends_route_pred.png/pdf
synthetic_e2e_bundle_trends_route_pred.png/pdf
```

元数据文件：

```text
synthetic_e2e_trend_workload_manifest_route_pred.csv
synthetic_e2e_trend_hardware_sweep_manifest_route_pred.csv
synthetic_e2e_trend_hardware_range_summary_route_pred.csv
synthetic_e2e_trend_workload_and_hw_manifest_route_pred.json
synthetic_e2e_trend_summary_route_pred.csv
synthetic_e2e_trend_case_index_route_pred.csv
synthetic_e2e_trend_sweep_points_route_pred.csv
hardware_response_guard_audit.csv
```

本文统一使用匿名设备名：

| 图中名称 | 数据绑定含义 |
|---|---|
| `Device_1` | 远端设备数据绑定 |
| `Device_2` | 本机设备数据绑定 |

硬件配置文件仍然只使用：

```text
calibrator/hw_config
```

本轮趋势图使用 `calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json` 中的新 v3 bundle 复用模式生成，不重新训练 RF。`synthetic_e2e_trend_summary_route_pred.csv` 中 17 个主线 trend key 的最小 pass rate 为 `1.0`。

## 2. 图的含义

这些图的纵轴是调度后的整图端到端延迟，也就是 scheduled E2E latency，不是单个算子的耗时。

每个点的计算链路如下：

$$\text{Synthetic workload} \rightarrow \text{FULL\_GRAPH\_76 operator frame} \rightarrow \text{whitebox\_family\_prior\_plus\_physics\_guarded\_rf\_v3} \rightarrow \text{DAG schedule} \rightarrow T_{\text{E2E}}$$

每个子图现在包含 6 条曲线：

$$\text{Device\_1} \times \{\text{small}, \text{base}, \text{large}\}$$

$$\text{Device\_2} \times \{\text{small}, \text{base}, \text{large}\}$$

其中 `base` 是之前单负载图使用的规模，`small` 和 `large` 是新增的较小和较大规模，用于检查趋势在不同负载规模下是否稳定，降低单一 workload 偶然性的影响。

每个子图下方都标注了该 panel 使用的 `small/base/large` 软件负载规模。图下注释只保留最关键的规模信息，例如 batch、indices、MLP 或 working set 相对目标 cache 的比例；完整配置仍以 `synthetic_e2e_trend_workload_manifest_route_pred.csv` 为准。

趋势检查结果：

| 项目 | 数值 |
|---|---:|
| 硬件趋势项数量 | 17 |
| 总 case 数 | 102 |
| 最小单调通过率 | 1.0 |
| workload 规模 | small、base、large |

Guard 统计：

| guard 动作 | 行数 | 触发行数 |
|---|---:|---:|
| `none` | 90938 | 0 |
| `delta_cap` | 797 | 797 |
| `direction_conflict_residual_frozen` | 833 | 833 |

E2E 聚合单调投影触发 13 / 102 个 case。投影前后的差异保存在：

```text
raw_pred_us
pred_us
aggregate_projection_delta_us
```

因此图中曲线是物理约束后的最终展示曲线，同时 CSV 保留了投影前原始输出。

## 3. 多负载规模设计

### 3.0 硬件基准值如何设置

除 core 图外，图中横轴的 `baseline=1` 表示该设备硬件配置文件中的原始值。每个 sweep 点都按相对倍率修改对应字段：

$$x_{\text{sweep}} = x_{\text{baseline}} \times \text{relative factor}$$

这里的 baseline 不是模型拟合出来的，也不是两台设备的平均值，而是分别来自：

| 图中名称 | baseline 硬件配置 |
|---|---|
| `Device_1` | `calibrator/hw_config/kunpeng920_single_numa.yaml`（智算平台） |
| `Device_2` | `calibrator/hw_config/s920x20_single_numa.yaml`（通算平台） |

当前图中涉及的主要硬件基准值如下。

core 图是例外：它展示 `active_parallel_workers`，横轴为实际参与 worker 数，不使用 relative factor。

| 参数 | Device_1 baseline | Device_2 baseline | 图中扰动范围 |
|---|---:|---:|---|
| `active_parallel_workers` | 1 到 24 workers | 1 到 40 workers | runtime worker sweep，不是 YAML 单字段 |
| `core.total_cores` | 24 | 40 | 当前趋势图未单独 sweep |
| `core.cpu_clock` | 2.592166 GHz | 2.893398 GHz | 0.75x 到 1.25x |
| `instruction.simd_width_bits` | 128 bit | 256 bit | 0.75x 到 1.25x |
| `instruction.fp_throughput_per_cycle.vector_sp_add` | 1.499752 | 1.999786 | 当前趋势图未单独 sweep |
| `instruction.fp_throughput_per_cycle.vector_sp_mul` | 1.499756 | 1.999792 | 当前趋势图未单独 sweep |
| `instruction.fp_throughput_per_cycle.vector_sp_fma` | 1.749673 | 1.999777 | 0.75x 到 1.25x |
| `instruction.fp_latency_cycles.vector_sp_add` | 5.000226 cycles | 2.000095 cycles | 当前趋势图未单独 sweep |
| `instruction.fp_latency_cycles.vector_sp_mul` | 5.000210 cycles | 3.000103 cycles | 当前趋势图未单独 sweep |
| `instruction.fp_latency_cycles.vector_sp_fma` | 4.000215 cycles | 2.000094 cycles | 0.75x 到 1.25x |
| `pipeline.fetch_width` | 2.938467 | 3.891586 | pipeline bundle 中 0.75x 到 1.25x |
| `pipeline.decode_width` | 2.938467 | 3.891586 | pipeline bundle 中 0.75x 到 1.25x |
| `pipeline.rename_width` | 2.938467 | 3.891586 | pipeline bundle 中 0.75x 到 1.25x |
| `pipeline.dispatch_width` | 2.938467 | 3.891586 | pipeline bundle 中 0.75x 到 1.25x |
| `pipeline.issue_width` | 2.938467 | 3.891586 | pipeline bundle 中 0.75x 到 1.25x |
| `pipeline.commit_width` | 2.938467 | 3.891586 | pipeline bundle 中 0.75x 到 1.25x |
| `memory.copy_bw_gb_s_1t` | 5.048952 GB/s | 35.900316 GB/s | 0.70x 到 1.30x |
| `memory.copy_bw_gb_s_saturated` | 30.820562 GB/s | 137.967158 GB/s | 0.70x 到 1.30x |
| `memory.random_gather_bw_gb_s_1t` | 0.840198 GB/s | 2.523200 GB/s | 0.70x 到 1.30x |
| `memory.random_gather_bw_gb_s_saturated` | 4.160243 GB/s | 18.047464 GB/s | 0.70x 到 1.30x |
| `memory.local_mem_delay_ns` | 91.0687 ns | 88.0950 ns | 0.75x 到 1.25x |
| `cache.cacheline_bytes` | 64 bytes | 64 bytes | 当前趋势图未单独 sweep |
| `cache.l1d.size` | 64 KiB | 64 KiB | 0.65x 到 1.35x |
| `cache.l2.size` | 512 KiB | 1280 KiB | 0.65x 到 1.35x |
| `cache.l3_per_die.size` | 24 MiB | 70 MiB | 0.65x 到 1.35x |
| `cache.l1d.response_latency_cycles` | 11.662672 cycles | 13.024487 cycles | 0.75x 到 1.25x |
| `cache.l2.response_latency_cycles` | 21.333783 cycles | 31.850238 cycles | 0.75x 到 1.25x |
| `cache.l3_per_die.response_latency_cycles` | 121.825564 cycles | 126.632175 cycles | 0.75x 到 1.25x |
| `runtime.parallel_launch_overhead_us_intra` | 843.905855 us | 10.699790 us | 当前趋势图未单独 sweep |
| `runtime.parallel_launch_overhead_us_inter` | 832.877529 us | 19.678168 us | 当前趋势图未单独 sweep |

对于 bundle 图，多个字段会一起按同一个 factor 缩放：

| bundle | 同时缩放的字段 | 扰动范围 |
|---|---|---|
| `full_compute_bundle` | `core.cpu_clock`, `instruction.simd_width_bits`, `instruction.fp_throughput_per_cycle.vector_sp_fma` | 0.75x 到 1.25x |
| `full_memory_bundle` | `memory.copy_bw_gb_s_1t`, `memory.copy_bw_gb_s_saturated`, `memory.random_gather_bw_gb_s_1t`, `memory.random_gather_bw_gb_s_saturated` | 0.60x 到 1.60x |

`capacity` 类型参数增大时，预期 E2E latency 不上升；`cost` 类型参数增大时，预期 E2E latency 不下降。这个方向只用于趋势判定和 guard/projection，实际数值仍按上面的 baseline 乘以 relative factor 得到。

### 3.1 Core 图

图文件：

```text
synthetic_e2e_core_trends_route_pred.png
```

硬件参数：

| 趋势项 | 含义 | sweep 范围 |
|---|---|---|
| `active_parallel_workers` | 实际参与并行执行的 worker 数，近似为 `num_threads × active_branch_groups`，并被 `core.total_cores` 截断 | Device_1: 1 到 24；Device_2: 1 到 40 |

软件负载使用 parallel DLRM-like graph，`small/base/large` 分别扩大 batch、indices 和 MLP。每个 worker 点会重建 synthetic workload 的 runtime 线程配置，而不是修改硬件 YAML：

$$N_{\text{active\_worker}} = \min(N_{\text{core}}, N_{\text{thread}} \cdot G_{\text{active}})$$

其中 $$N_{\text{thread}}$$ 来自 `num_threads`，$$G_{\text{active}}$$ 来自 branch pool 中实际可并发的分支组数，$$N_{\text{core}}$$ 来自 `core.total_cores`。Device_1 使用 worker 点 `1,2,4,8,12,16,20,24`；Device_2 使用 worker 点 `1,2,4,8,12,16,20,24,32,40`。CSV 中同时记录 `target_active_parallel_workers`、`active_parallel_workers`、`num_threads` 和 `inter_threads`。

这样设计的原因是：单独 sweep `core.total_cores` 时，如果软件线程数固定，新增核心不会被使用，E2E 曲线会自然接近平台。`active_parallel_workers` 图验证的是 runtime 实际使用更多 worker 后的并行缩放：worker 增加时 E2E latency 应下降，并在 memory bandwidth、random gather、cache 或 DAG 调度瓶颈出现后进入平台。

### 3.2 Cache Size 图

图文件：

```text
synthetic_e2e_cache_size_trends_route_pred.png
```

硬件参数：

| 趋势项 | 硬件字段 | 扰动范围 |
|---|---|---|
| `l1d_size` | `cache.l1d.size` | 0.65x 到 1.35x，共 15 个点 |
| `l2_size` | `cache.l2.size` | 0.65x 到 1.35x，共 15 个点 |
| `l3_size` | `cache.l3_per_die.size` | 0.65x 到 1.35x，共 15 个点 |

软件负载是 embedding-style `Gather`。需要特别说明：这张图不是固定同一个上层 DLRM 配置后同时观察 L1/L2/L3，而是为每个目标 cache level 构造一组 controlled cache-probe DLRM-like workload。生成脚本会根据目标 cache level 的 baseline 容量反推 `lookup_count`、`num_indices_per_lookup` 和 `arch_embedding_size`，让该组 workload 的 Gather working set 落在对应 cache level 的边界附近。

因此 L1D、L2、L3 三个 panel 的上层 DLRM 参数并不相同；它们的作用是分别隔离和验证模型对 L1D size、L2 size、L3 size 的局部物理响应。主图采用这种 controlled workload 口径，因为它能把目标硬件参数放到当前 panel 的敏感区间，避免固定 DLRM 负载因为 working set 过大或过小而出现无法解释硬件参数的整段平台。

working set 相对目标 cache level 的规模如下：

| 规模 | working set 设置 |
|---|---:|
| small | 1.10x 目标 cache |
| base | 1.45x 目标 cache |
| large | 1.80x 目标 cache |

设计目的：

- small：接近 cache 边界，用于观察 cache size 增大后的收益。
- base：保持之前主图的分析口径。
- large：让 working set 明显超过目标 cache，避免曲线过早平台化。

### 3.3 Cache Latency 图

图文件：

```text
synthetic_e2e_cache_latency_trends_route_pred.png
```

硬件参数：

| 趋势项 | 硬件字段 | 扰动范围 |
|---|---|---|
| `l1d_latency` | `cache.l1d.response_latency_cycles` | 0.75x 到 1.25x，共 11 个点 |
| `l2_latency` | `cache.l2.response_latency_cycles` | 0.75x 到 1.25x，共 11 个点 |
| `l3_latency` | `cache.l3_per_die.response_latency_cycles` | 0.75x 到 1.25x，共 11 个点 |

软件负载仍然是 `Gather`，但 working set 被控制在对应 cache resident 区间：

| 规模 | L1 目标 | L2 目标 | L3 目标 |
|---|---:|---:|---:|
| small | 0.45x L1D | max(2.5x L1D, 0.45x L2) | max(2.5x L2, 0.45x L3) |
| base | 0.70x L1D | max(2.5x L1D, 0.70x L2) | max(2.5x L2, 0.70x L3) |
| large | 0.95x L1D | max(2.5x L1D, 0.95x L2) | max(2.5x L2, 0.95x L3) |

这样可以分别观察 L1、L2、L3 响应延迟变大时，对 E2E 延迟的方向性影响。

选择这组 workload 的原因是：cache latency 只有在数据主要命中目标 cache level 时才会直接进入随机访问耗时；如果 working set 远小于目标层，曲线会被更上层 cache 掩盖；如果远大于目标层，曲线会被更下层 cache 或 DRAM 掩盖。

### 3.4 Memory 图

图文件：

```text
synthetic_e2e_memory_trends_route_pred.png
```

硬件参数：

| 趋势项 | 硬件字段 | 扰动范围 |
|---|---|---|
| `copy_bw` | `memory.copy_bw_gb_s_1t`, `memory.copy_bw_gb_s_saturated` | 0.70x 到 1.30x，共 13 个点 |
| `random_gather_bw` | `memory.random_gather_bw_gb_s_1t`, `memory.random_gather_bw_gb_s_saturated` | 0.70x 到 1.30x，共 13 个点 |
| `mem_latency` | `memory.local_mem_delay_ns` | 0.75x 到 1.25x，共 11 个点 |

软件负载：

| 趋势项 | small | base | large |
|---|---|---|---|
| `copy_bw` | batch 2048 | batch 4096 | batch 8192 |
| `random_gather_bw` | 1.40x active L3 | 2.20x active L3 | 3.00x active L3 |
| `mem_latency` | 1.10x L3 | 1.60x L3 | 2.20x L3 |

选择这组 workload 的原因是：

- `copy_bw` 需要连续读写占主导，因此使用 stream-copy DLRM-like 子图。
- `random_gather_bw` 需要随机 embedding 访问占主导，因此使用超过 active L3 的 Gather working set。
- `mem_latency` 需要 DRAM-resident 的随机访问，working set 设置在 L3 之外，避免 cache 命中掩盖 DRAM latency。

`cache.cacheline_bytes` 不再作为主线趋势图展示。两台设备 baseline 都是 64 bytes，而且 cache line 的真实影响已经被实测 random gather bandwidth 和 copy bandwidth 间接吸收；单独改变 cache line size 而保持带宽、transaction、prefetch 行为不变，不是物理一致的硬件反事实。因此它保留为 Analytical model 内部字段和弱敏感说明，不作为主图证据。

### 3.5 Compute / Pipeline 图

图文件：

```text
synthetic_e2e_compute_pipeline_trends_route_pred.png
```

硬件参数：

| 趋势项 | 硬件字段 | 扰动范围 |
|---|---|---|
| `cpu_clock` | `core.cpu_clock` | 0.75x 到 1.25x，共 11 个点 |
| `simd_width` | `instruction.simd_width_bits` | 0.75x 到 1.25x，共 11 个点 |
| `fma_throughput` | `instruction.fp_throughput_per_cycle.vector_sp_fma` | 0.75x 到 1.25x，共 11 个点 |
| `fma_latency` | `instruction.fp_latency_cycles.vector_sp_fma` | 0.75x 到 1.25x，共 11 个点 |
| `pipeline_width` | `pipeline.fetch/decode/rename/dispatch/issue/commit_width` | 0.75x 到 1.25x，共 11 个点 |

`Gemm/MatMul` 负载：

| 规模 | batch | bottom MLP | top MLP |
|---|---:|---|---|
| small | 1024 | `256-1024-256` | `292-1024-1` |
| base | 2048 | `512-2048-512` | `548-2048-1` |
| large | 4096 | `1024-4096-1024` | `1060-4096-1` |

`pipeline_width` 使用 metadata/control 负载：

| 规模 | batch | embedding rows | bottom MLP | top MLP |
|---|---:|---|---|---|
| small | 16 | 8 tables x 512 rows | `16-32-32` | `68-64-1` |
| base | 32 | 8 tables x 1024 rows | `16-32-32` | `68-64-1` |
| large | 64 | 8 tables x 2048 rows | `16-32-32` | `68-64-1` |

选择这组 workload 的原因是：CPU frequency、SIMD width 和 FMA throughput 主要影响 dense Gemm/MatMul；pipeline width 更容易在 metadata/control 类型的小算子路径中体现，因此单独使用较小的 metadata/control graph，避免被大矩阵计算完全掩盖。

`instruction.fp_throughput_per_cycle.vector_sp_add/mul` 和 `instruction.fp_latency_cycles.vector_sp_add/mul` 不再作为主线趋势图展示。它们仍由 Analytical model 读取并用于 Add/Mul/ReduceSum 的 compute service time，但在当前 DLRM-like scheduled E2E 中 Add/Mul service time 远小于 stream/random/MLP 主导项，没有进入关键路径，之前曲线幅度接近 0。因此它们保留为模型内部字段，不作为主图泛化证据。

### 3.6 Bundle 图

图文件：

```text
synthetic_e2e_bundle_trends_route_pred.png
```

硬件参数：

| 趋势项 | 硬件字段 | 扰动范围 |
|---|---|---|
| `full_compute_bundle` | `core.cpu_clock`, `instruction.simd_width_bits`, `instruction.fp_throughput_per_cycle.vector_sp_fma` | 0.75x 到 1.25x |
| `full_memory_bundle` | copy bandwidth 与 random gather bandwidth 字段 | 0.60x 到 1.60x |

`full_compute_bundle` 使用 `Gemm/MatMul` 负载，`full_memory_bundle` 使用 memory-heavy `Gather` 负载。

选择这组 workload 的原因是：bundle 图验证的是一组硬件参数同时变化时的整体响应，因此直接沿用对应的 compute-heavy 和 memory-heavy 负载，让组合趋势仍然能对应到清晰的主导瓶颈。

### 3.7 Global Fixed DLRM-S/M/L 补充诊断

固定全局 DLRM-S/M/L 图不作为当前主线证据。它的优点是所有硬件参数图使用完全相同的三套上层 DLRM 配置；缺点是同一个 fixed workload 的 working set 很难同时落在 L1、L2、L3、DRAM 和 compute/pipeline 的敏感区间，因此部分 cache 或 memory 子图可能出现整段平台。

推荐展示图采用 3.1 到 3.6 的 controlled multiworkload 口径：每个硬件参数 panel 都使用专门设计的软件负载，把目标硬件参数放在可观察的瓶颈区间内。这样更适合评估 Analytical model 和 `physics_guarded_rf_v3` 对硬件参数变化的方向和幅度响应。

固定 DLRM 图可以保留为补充材料，用于说明真实固定上层负载下的整体趋势；若出现平台，应结合 manifest 中的 working set/cache ratio 和调度关键路径解释，不作为否定 controlled probe 主图的依据。

### 3.8 每个趋势子图的最大 E2E latency 变化幅度

变化幅度从 `synthetic_e2e_trend_sweep_points_route_pred.csv` 计算。对每个 `trend_group + trend_key + device`，先分别统计 `small/base/large` 三档负载在硬件 sweep 范围内的 E2E latency 最大变化，再选择变化最大的一档展示。

计算公式：

$$change\_pct=\frac{\max(pred\_us)-\min(pred\_us)}{\min(pred\_us)}\times100\%$$

完整 CSV 位于：

```text
synthetic_e2e_trend_max_change_by_device.csv
```

| 图组 | 趋势项 | Device_1 最大变化 | 对应负载 | Device_2 最大变化 | 对应负载 |
|---|---|---:|---|---:|---|
| Core | `Active parallel workers` | 452.55% | `base` | 827.78% | `small` |
| Cache size | `L1D cache size` | 0.10% | `base` | 0.31% | `base` |
| Cache size | `L2 cache size` | 1.46% | `base` | 1.64% | `base` |
| Cache size | `L3 cache size` | 0.86% | `base` | 2.22% | `large` |
| Cache latency | `L1D cache latency` | 0.12% | `large` | 0.33% | `large` |
| Cache latency | `L2 cache latency` | 0.23% | `small` | 0.19% | `large` |
| Cache latency | `L3 cache latency` | 0.65% | `small` | 1.89% | `large` |
| Memory | `Sequential memory bandwidth` | 59.20% | `large` | 64.78% | `large` |
| Memory | `Random gather bandwidth` | 77.76% | `large` | 84.19% | `base` |
| Memory | `DRAM access latency` | 2.04% | `large` | 3.08% | `large` |
| Compute / Pipeline | `CPU frequency` | 61.43% | `large` | 59.82% | `large` |
| Compute / Pipeline | `SIMD width` | 61.44% | `large` | 59.80% | `large` |
| Compute / Pipeline | `Vector FMA throughput` | 61.43% | `large` | 59.78% | `large` |
| Compute / Pipeline | `Vector FMA latency` | 0.06% | `small` | 0.03% | `base` |
| Compute / Pipeline | `Pipeline width` | 59.21% | `small` | 45.37% | `small` |
| Bundle | `Compute bundle` | 319.39% | `large` | 308.57% | `large` |
| Bundle | `Memory bundle` | 161.32% | `large` | 165.21% | `base` |

需要注意，cache latency 和 FMA latency 子图的变化幅度较小是合理的：这些参数只影响当前 controlled workload 中命中的那部分随机访问或计算尾项，E2E critical path 中仍有 bandwidth、compute throughput、dispatch 等其它项。主线展示重点是方向正确、幅度不反常，而不是要求每个硬件字段都产生同等幅度。

### 3.9 软硬件参数与 E2E latency 的相关性和影响强度

相关性分析使用同一份 synthetic sweep points，不使用真实 full test CSV。它回答两个问题：

1. 软件规模参数是否主导整体 E2E latency 的绝对大小。
2. 每个硬件参数在自己的 controlled sweep 中，是否方向正确，以及影响强度有多大。

因此这里不把全局 pooled Spearman 作为硬件主证据。全局 pooled Spearman 会把不同 trend、不同 workload 和大量硬件常量行混在一起，容易把硬件相关性稀释得很低。硬件主图采用局部指标：

- `trend-local direction-corrected Spearman`：只在当前硬件参数自己的 sweep 内计算，用于检查方向一致性。
- `absolute log-log elasticity`：用于公平比较不同 sweep 范围下的影响强度。
- `max change pct`：用于展示当前 sweep 范围内可观察到的最大变化幅度。

方向校正规则：

$$\rho_{\text{dir}}=-\rho,\quad \text{capacity 参数}$$

$$\rho_{\text{dir}}=\rho,\quad \text{latency/cost 参数}$$

推荐主图使用 absolute log-log elasticity。它把 E2E latency 的相对变化除以硬件参数自身的相对变化，避免不同参数 sweep 范围不一致导致比较不公平：

$$E(p)=\left|\frac{\log T_{\text{E2E}}(p_2)-\log T_{\text{E2E}}(p_1)}{\log p_2-\log p_1}\right|$$

其中 $$p_1$$ 和 $$p_2$$ 是该参数 sweep 的两个端点，$$T_{\text{E2E}}(p_1)$$ 和 $$T_{\text{E2E}}(p_2)$$ 是端点处的 scheduled E2E latency。该指标越大，表示硬件参数变化 1% 对 E2E latency 的相对影响越强。公式取绝对值，因此 capacity 参数和 latency 参数不需要手动换符号。

输出文件：

```text
synthetic_e2e_hw_sw_correlation_heatmap.png
synthetic_e2e_hw_sw_correlation_matrix.csv
synthetic_e2e_hw_sw_correlation_skipped_constants.csv
synthetic_e2e_hardware_local_directional_spearman_heatmap.png
synthetic_e2e_hardware_abs_log_elasticity_heatmap.png
synthetic_e2e_hardware_directional_elasticity_heatmap.png
synthetic_e2e_hardware_max_change_pct_heatmap.png
synthetic_e2e_hardware_local_sensitivity_by_workload.csv
synthetic_e2e_hardware_local_sensitivity_best_workload.csv
synthetic_e2e_hardware_metric_comparison_summary.csv
```

软件参数只保留六类：

| 参数 | 含义 |
|---|---|
| `batch_size` | DLRM batch size |
| `num_indices_per_lookup` | 每个 embedding lookup 的 index 数 |
| `embedding_size_total` | 8 个 embedding table 行数求和 |
| `bottom_mlp_ops_proxy` | bottom MLP 相邻层宽乘积之和，作为计算量 proxy |
| `inter_threads` | 图级 branch 并行线程数 |
| `num_threads` | 单算子内部线程数 |

当前最强的软件相关性如下：

| 参数 | Device_1 Spearman | Device_2 Spearman | 解释 |
|---|---:|---:|---|
| `batch_size` | 0.741 | 0.651 | batch 越大，整图工作量越大，E2E latency 增大 |
| `embedding_size_total` | 0.677 | 0.755 | embedding working set 越大，random gather 与 memory 压力越大 |
| `num_indices_per_lookup` | 0.394 | 0.601 | 每次 lookup 访问 index 更多，Gather 工作量增大 |
| `inter_threads` | -0.183 | -0.410 | branch 并行度增加，在未到瓶颈前会降低 E2E latency |
| `num_threads` | -0.186 | -0.407 | 单算子 worker 增加，在未到瓶颈前会降低 E2E latency |

硬件影响强度的当前统计结果：

| 指标 | Device_1 | Device_2 | 解释 |
|---|---:|---:|---|
| best workload directional Spearman median | 1.000 | 1.000 | 17 个主线硬件趋势基本都符合物理方向 |
| best workload absolute log-log elasticity median | 0.538 | 0.604 | 可以区分硬件参数单位相对变化对 E2E latency 的影响强度 |
| best workload max change pct median | 59.20% | 45.37% | 可以直接展示 sweep 范围内 E2E latency 最大变化 |

报告中建议把 `synthetic_e2e_hardware_abs_log_elasticity_heatmap.png` 作为 Top 硬件建模参数主图，把 `synthetic_e2e_hardware_local_directional_spearman_heatmap.png` 作为方向一致性证据，把 `synthetic_e2e_hardware_max_change_pct_heatmap.png` 作为 sweep 范围内最大可观察变化的辅助展示。`synthetic_e2e_hardware_directional_elasticity_heatmap.png` 是旧方向修正诊断图，仅保留用于对照。原始 `synthetic_e2e_hw_sw_correlation_heatmap.png` 只作为全局 mixed sweep 诊断图，不作为硬件敏感性主证据。

没有变化的硬件字段写入 `synthetic_e2e_hw_sw_correlation_skipped_constants.csv`，不进入 heatmap。

### 3.10 DLRM 访存主导与计算参数敏感性并不矛盾

Top 硬件参数图中 compute bundle、CPU frequency、SIMD width 或 FMA throughput 可能表现出较强敏感性，但这不表示 DLRM 被重新解释为计算瓶颈。当前专门的归因实验目录为：

```text
outputs/dlrm_compute_memory_attribution_20260516
```

该实验同时使用两个口径：

- 真实 profiled operator time：按 `Gather/ReduceSum/data movement` 与 `Gemm/MatMul/elementwise` 聚合。
- Analytical service time：按 `stream_mem + random_mem`、`svc_compute`、`svc_dispatch` 聚合。

当前结果如下：

| 口径 | Device_1 | Device_2 | 解释 |
|---|---:|---:|---|
| memory-related true operator time share | 97.7% | 97.7% | DLRM held-out E2E 图中真实耗时仍由访存相关算子主导 |
| compute-related true operator time share | 2.3% | 2.3% | dense/elementwise 真实耗时占比较小 |
| Analytical memory service share | 96.3% | 92.5% | `stream_mem + random_mem` 是白盒服务时间主体 |
| Analytical compute service share | 3.7% | 7.5% | compute service 存在，但不是主体 |

因此，报告中的解释口径应为：

```text
DLRM 总体仍由访存相关开销主导；计算/流水线参数的影响主要来自 dense MLP、ReduceSum 或 tail/serial phase 等局部关键路径。这些阶段占比不一定最高，但位于 E2E 调度关键路径时，其边际变化仍会影响最终 scheduled E2E latency。
```

需要避免的表述：

- 不写“DLRM 是计算瓶颈应用”。
- 不把 controlled multiworkload trend 的局部敏感性当作真实 full test workload 的全局耗时占比。
- 如果单独 SIMD/FMA 的 held-out E2E 扰动幅度较弱，应写成“compute bundle 或部分计算参数在特定 controlled workload 下敏感”，而不是声称所有计算参数都是主因。

## 4. `physics_guarded_rf_v3` 如何控制硬件趋势

当前 route 是：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

核心目标不是去掉 RF，而是在保留 RF 校准能力的同时，避免硬件参数变化后出现反物理趋势。

### 4.1 硬件参数先进入 Analytical anchors

v3 不让 RF 直接读取 raw hardware fields，例如：

```text
core.cpu_clock
cache.l2.size
memory.random_gather_bw_gb_s_saturated
```

这些硬件参数先由 Analytical prior 消化，并转成可解释的服务时间锚点：

```text
whitebox_center_us
whitebox_exec_center_us
whitebox_serial_overhead_center_us
svc_stream_mem_center_us
svc_random_mem_center_us
svc_compute_center_us
svc_dispatch_center_us
```

也就是说，RF 看到的是“硬件参数已经通过白盒公式换算后的服务时间”，不是原始硬件字段。这样硬件趋势主要由 Analytical model 控制。

注意：代码框架中仍保留若干历史诊断列，但当前 Analytical prior 在合成 `whitebox_center_us` 前已经把这些历史模块清零。因此 RF 趋势控制和最终白盒总耗时只看上面列出的 active service-time anchors。

### 4.2 硬件反事实训练增强

对真实训练样本，记 baseline 硬件为 $$h_0$$，真实耗时为 $$y_0$$。

白盒 prior 分别为：

$$p_0 = f_{\text{whitebox}}(x, h_0)$$

$$p_h = f_{\text{whitebox}}(x, h)$$

对扰动硬件 $$h$$ 构造伪标签：

$$y_h^{\text{synthetic}} = p_h + (y_0 - p_0)$$

含义是：

- 保留真实样本在 baseline 硬件上的 RF 校准残差。
- 硬件变化方向由 whitebox prior 决定。
- synthetic 样本只进入训练，不进入真实 validation MAPE。
- synthetic 样本权重低于真实样本，当前为 0.25。

### 4.3 单调 RF anchor

v3 对以下 anchor 特征施加正单调约束：

```text
whitebox_center_us
whitebox_exec_center_us
whitebox_serial_overhead_center_us
svc_stream_mem_center_us
svc_random_mem_center_us
svc_compute_center_us
svc_dispatch_center_us
```

直觉是：如果 whitebox service time anchor 变大，RF 不应该仅因为这个 anchor 变大而让预测延迟变小。

不过当前 v3 为了保持 E2E MAPE，仍沿用 v2 的 residual-ratio target 体系。因此单调 RF anchor 是训练偏置，不作为最终硬保证。最终硬保证来自下一节的 guard。

### 4.4 硬件响应 guard

当使用 `override_hw_config` 做硬件 sweep 时，v3 同时计算 baseline 硬件和目标硬件下的 whitebox 与 RF 输出：

$$p_0, r_0$$

$$p_h, r_h$$

其中 $$p$$ 表示 whitebox prior，$$r$$ 表示 RF 后的预测。

定义：

$$\Delta_{\text{whitebox}} = p_h - p_0$$

$$\Delta_{\text{RF}} = r_h - r_0$$

如果二者方向冲突：

$$\Delta_{\text{whitebox}} \cdot \Delta_{\text{RF}} < 0$$

则使用 frozen residual 回退：

$$\hat{y}_h = p_h + (r_0 - p_0)$$

含义是：保留 RF 在 baseline 硬件上的校准残差，但硬件变化方向交还给 whitebox prior。

如果方向一致但 RF 变化幅度过大，则做幅度限制：

$$|\Delta_{\text{RF}}| \le 2 \cdot |\Delta_{\text{whitebox}}|$$

这样可以避免 RF 在硬件 sweep 中把趋势放大到不合理。

### 4.5 E2E 聚合单调投影

row-level guard 作用在单算子预测上。单算子预测进入 DAG 调度后，可能因为 branch pool、barrier、tail 等调度结构出现极小局部抖动。

因此趋势图最终展示时，对 E2E 聚合曲线再做一层单调投影：

容量型参数，例如 bandwidth、SIMD width、cache size：

$$\hat{T}_i = \min(\hat{T}_{i-1}, T_i)$$

代价型参数，例如 latency、launch overhead：

$$\hat{T}_i = \max(\hat{T}_{i-1}, T_i)$$

投影前后都保存在 CSV 中：

```text
raw_pred_us
pred_us
aggregate_projection_delta_us
```

## 5. Analytical prior 实际读取的硬件参数

`_load_hw_params()` 当前实际读取以下硬件字段。

| 模块 | YAML 字段 | 内部字段 | 主要用途 |
|---|---|---|---|
| core | `core.total_cores` | `total_cores` | worker 上限、带宽饱和曲线、branch contention |
| core | `core.cpu_clock` | `cpu_clock_ghz` | cycles 到时间换算、FP capacity、pipeline capacity |
| pipeline | `pipeline.fetch_width` | `fetch_width` | 历史 frontend token capacity；当前 active 模块中主要通过 dispatch 相关项保留影响 |
| pipeline | `pipeline.decode_width` | `decode_width` | 历史 frontend token capacity；当前 active 模块中主要通过 dispatch 相关项保留影响 |
| pipeline | `pipeline.rename_width` | `rename_width` | 历史 frontend token capacity；当前 active 模块中主要通过 dispatch 相关项保留影响 |
| pipeline | `pipeline.dispatch_width` | `dispatch_width` | frontend capacity、dispatch/launch 摊销 |
| pipeline | `pipeline.issue_width` | `issue_width` | Gather latency parallelism、control-token capacity |
| pipeline | `pipeline.commit_width` | `commit_width` | Gather latency parallelism、control-token capacity |
| instruction | `instruction.simd_width_bits` | `simd_width_bits` | FP32 lane capacity |
| instruction | `instruction.fp_throughput_per_cycle.vector_sp_add` | `fp_throughput_vector_sp_add` | Add/ReduceSum compute |
| instruction | `instruction.fp_throughput_per_cycle.vector_sp_mul` | `fp_throughput_vector_sp_mul` | Mul compute |
| instruction | `instruction.fp_throughput_per_cycle.vector_sp_fma` | `fp_throughput_vector_sp_fma` | Gemm/MatMul compute |
| instruction | `instruction.fp_latency_cycles.vector_sp_add` | `fp_latency_vector_sp_add` | compute latency tail |
| instruction | `instruction.fp_latency_cycles.vector_sp_mul` | `fp_latency_vector_sp_mul` | compute latency tail |
| instruction | `instruction.fp_latency_cycles.vector_sp_fma` | `fp_latency_vector_sp_fma` | Gemm/MatMul latency tail |
| memory | `memory.bandwidth_gb_s_total` | fallback bandwidth | measured copy bandwidth 缺失时的 fallback |
| memory | `memory.copy_bw_gb_s_1t` | `copy_bw_gb_s_1t` | 单线程连续 copy 带宽 |
| memory | `memory.copy_bw_gb_s_saturated` | `copy_bw_gb_s_saturated` | 多线程连续 copy 饱和带宽 |
| memory | `memory.random_gather_bw_gb_s_1t` | `random_gather_bw_gb_s_1t` | 单线程随机 Gather 带宽 |
| memory | `memory.random_gather_bw_gb_s_saturated` | `random_gather_bw_gb_s_saturated` | 多线程随机 Gather 饱和带宽 |
| memory | `memory.local_mem_delay_ns` | `local_mem_delay_ns` | DRAM random access touch latency |
| runtime | `runtime.parallel_launch_overhead_us_intra` | `parallel_launch_overhead_us_intra` | intra-op launch/dispatch cost |
| runtime | `runtime.parallel_launch_overhead_us_inter` | `parallel_launch_overhead_us_inter` | inter-branch launch/dispatch cost |
| cache | `cache.cacheline_bytes` | `cacheline_bytes` | Gather random head touch bytes |
| cache | `cache.l1d.size` | `l1d_size_bytes` | cache residency |
| cache | `cache.l2.size` | `l2_size_bytes` | cache residency |
| cache | `cache.l3_per_die.size` 或 `cache.l3.size` | `l3_size_bytes` | cache residency |
| cache | `cache.l1d.response_latency_cycles` | `l1d_latency_ns_static` | L1 touch latency |
| cache | `cache.l2.response_latency_cycles` | `l2_latency_ns_static` | L2 touch latency |
| cache | `cache.l3_per_die.response_latency_cycles` 或 `cache.l3.response_latency_cycles` | `l3_latency_ns_static` | L3 touch latency |
| optional | `memory.gather_slice_copy_bw.points` | `gather_slice_copy_bw_points` | Gather slice copy bandwidth 诊断 |

当前不作为物理 cost 字段使用的配置包括：

```text
vendor
model_name
sockets
numa_nodes
cache assoc/shared_by_cpus/tag_latency_cycles/data_latency_cycles
pipeline.wb_width
host_cross_check.*
paper_cross_check.*
```

## 6. 各类算子的 Analytical 建模方式

Analytical prior 首先把算子映射成 execution mode：

| execution mode | 典型算子 |
|---|---|
| metadata/view | `Shape`, view 类 `Reshape`, `Flatten`, `Unsqueeze`, shape `Gather` |
| indexed block copy | embedding `Gather` |
| reduction | `ReduceSum` |
| gemm | `Gemm`, `MatMul` |
| payload concat | payload `Concat` |
| shape concat | shape/control `Concat` |
| transpose | `Transpose` |
| tiny shape arithmetic | shape/control 上的 tiny `Add`/`Mul` |
| elementwise/fallback | `Add`, `Mul`, `Relu`, `Sigmoid`, 其他 fallback 算子 |

### 6.0 当前 active 服务模块

当前参与 `whitebox_center_us` 合成的 active Analytical cost modules 只有四类：

```text
stream_mem
random_mem
compute
dispatch
```

代码框架中仍保留若干历史 scaffolding、summary 列和诊断列，但它们在 `_compose_total_intervals()` 前已经被清零，因此当前不参与 Analytical 总耗时。具体代码位置是 `calibrator/whitebox_family_routes.py` 中的 `DISABLED_ANALYTICAL_MODULES = {"descriptor", "reorder", "sync"}`；`add_whitebox_family_prior_columns()` 先生成若干候选诊断项，然后在 `_service_share_columns(work)` 和 `_compose_total_intervals(work)` 之前调用 `_zero_disabled_analytical_modules(work)`，把 `descriptor/reorder/sync` 的 lower/center/upper 与 `svc_*_us` 全部置 0。

已验证事实：

| 趋势目录 | 历史诊断模块贡献 |
|---|---:|
| single workload trend | median = 0, max = 0 |
| multi workload trend | median = 0, max = 0 |

因此下面所有公式只描述当前 active Analytical model。

四个 active 模块的通用计算方式如下。

`stream_mem` 表示连续读写或 copy 路径：

$$BW_{\text{copy}} = \frac{C_{\text{worker}}(BW_{\text{copy,1t}}, BW_{\text{copy,sat}}, N_{\text{worker}})}{S_{\text{branch}}}$$

$$T_{\text{stream}} = \frac{B_{\text{stream}}}{\operatorname{bytes\_per\_us}(BW_{\text{copy}})}$$

其中 $$B_{\text{stream}}$$ 来自该算子的连续读写字节数，$$C_{\text{worker}}$$ 是从单线程带宽到饱和带宽的 worker capacity curve，$$S_{\text{branch}}$$ 是 branch contention share。

`stream_mem` 不显式乘以 cache latency。原因是连续读写/copy 路径主要由带宽限制：硬件 prefetch、cacheline 合并、memory controller 队列和流水化传输可以隐藏大部分单次 cache miss latency。因此这里用实测或配置的 copy bandwidth 统一吸收 cache hierarchy 对连续搬运的影响；若再叠加 $$N_{\text{touch}} \cdot L_{\text{touch}}$$，容易重复计算同一段内存访问成本。cache latency 只在随机访问更难被 prefetch 隐藏的 `random_mem` 中显式建模。

`C_worker` 的具体实现是线性 worker capacity curve，不是拟合曲线：

$$f_{\text{worker}} = \operatorname{clip}\left(\frac{N_{\text{worker}} - 1}{N_{\text{core}} - 1}, 0, 1\right)$$

$$C_{\text{worker}}(BW_{\text{1t}}, BW_{\text{sat}}, N_{\text{worker}}) = BW_{\text{1t}} + f_{\text{worker}} \cdot (BW_{\text{sat}} - BW_{\text{1t}})$$

其中 $$N_{\text{core}}$$ 来自 `core.total_cores`，$$N_{\text{worker}}$$ 来自该算子的有效 intra/inter worker 数并被 `core.total_cores` 截断。这样设计的原因是：1 个 worker 时使用单线程带宽，worker 增多时逐步接近饱和带宽，但不会超过硬件测得的 saturated bandwidth。

`S_branch` 的主预测使用 center 版本，也是按算子动态计算，不是拟合参数：

$$N_{\text{parallel}} = \operatorname{clip}(N_{\text{worker}} \cdot G_{\text{active}}, 1, N_{\text{core}})$$

$$f_{\text{contention}} = \operatorname{clip}\left(\frac{N_{\text{parallel}} - 1}{N_{\text{core}} - 1}, 0, 1\right)$$

$$S_{\text{branch,center}} = 1 + f_{\text{contention}}$$

其中 $$G_{\text{active}}$$ 来自静态图中的 active parallel branch/group 数。这样设计的原因是：多个 worker 或多个 branch 同时访问内存时会共享内存系统，不能让每个算子都独占 `C_worker` 给出的带宽，因此用 $$S_{\text{branch}}$$ 对有效带宽做折减。代码中还保留 optimistic 和 conservative 两个诊断边界；趋势图和 `whitebox_center_us` 使用 center。

数值例子：以 Device_2 的 copy bandwidth 为例，假设 $$N_{\text{core}}=40$$，$$BW_{\text{copy,1t}}=35.900316$$ GB/s，$$BW_{\text{copy,sat}}=137.967158$$ GB/s，某算子 $$N_{\text{worker}}=8$$，且 $$G_{\text{active}}=2$$。

$$f_{\text{worker}} = \frac{8 - 1}{40 - 1} = 0.1795$$

$$C_{\text{worker}} = 35.900316 + 0.1795 \cdot (137.967158 - 35.900316) = 54.2205\ \text{GB/s}$$

$$N_{\text{parallel}} = \operatorname{clip}(8 \cdot 2, 1, 40) = 16$$

$$S_{\text{branch,center}} = 1 + \frac{16 - 1}{40 - 1} = 1.3846$$

$$BW_{\text{copy}} = \frac{54.2205}{1.3846} = 39.1604\ \text{GB/s}$$

`random_mem` 表示 Gather 的随机 row-head 访问：

$$BW_{\text{random}} = \frac{C_{\text{worker}}(BW_{\text{random,1t}}, BW_{\text{random,sat}}, N_{\text{worker}})}{S_{\text{branch}}}$$

$$T_{\text{random\_bw}} = \frac{B_{\text{random\_head}}}{\operatorname{bytes\_per\_us}(BW_{\text{random}})}$$

$$L_{\text{touch}} = \operatorname{cache\_latency}(W_{\text{set}}, C_{\text{L1}}, C_{\text{L2}}, C_{\text{L3}}, L_{\text{L1}}, L_{\text{L2}}, L_{\text{L3}}, L_{\text{DRAM}})$$

$$T_{\text{random\_lat}} = \frac{N_{\text{touch}} \cdot L_{\text{touch}}}{P_{\text{latency}}}$$

$$T_{\text{random}} = \operatorname{compose}(T_{\text{random\_bw}}, T_{\text{random\_lat}})$$

其中 $$B_{\text{random\_head}}$$ 由有效唯一行数和 cacheline bytes 得到，$$W_{\text{set}}$$ 决定访问驻留在 L1/L2/L3/DRAM 的比例，$$P_{\text{latency}}$$ 由 worker 数和 issue/dispatch/commit width 形成。

`cache_latency(...)` 的实际操作分两步：先由当前 Gather 算子 / 当前 branch 的 working set 和同口径有效 cache 容量估计 residency level，再把 residency level 映射到一次随机 touch 的 latency。

这里的 $$W_{\text{set}}$$ 是单个 Gather 算子 / 单个 branch 的整体 working set，不是整个 E2E 图所有 branch 加总后的全系统 working set。因此 cache 容量也按“当前算子整体”口径计算。先定义：

$$N_{\text{system}} = \operatorname{clip}(N_{\text{worker}} \cdot G_{\text{active}}, 1, N_{\text{core}})$$

其中 $$N_{\text{worker}}$$ 是当前算子内部 worker 数，$$G_{\text{active}}$$ 是同时活跃的 branch group 数，$$N_{\text{system}}$$ 是系统层面的总活跃 worker 压力。

L1/L2 按 per-core private cache 处理，当前算子的 $$N_{\text{worker}}$$ 个 worker 各自拥有本 core 的 L1/L2：

$$C_{\text{L1,eff}} = N_{\text{worker}} \cdot C_{\text{L1}}$$

$$C_{\text{L2,eff}} = N_{\text{worker}} \cdot C_{\text{L2}}$$

L3/LLC 按 shared cache 处理，当前算子作为整体分到的 L3 份额由其 worker 数占系统总活跃 worker 的比例决定：

$$C_{\text{L3,eff}} = \frac{N_{\text{worker}}}{N_{\text{system}}} \cdot C_{\text{L3}}$$

如果从单 worker 视角看，L3 份额是 $$C_{\text{L3}} / N_{\text{system}}$$；当前代码使用单算子整体 working set，因此等价地使用 $$N_{\text{worker}} / N_{\text{system}} \cdot C_{\text{L3}}$$。

当前趋势图启用 continuous cache residency，因此不是硬切 L1/L2/L3/DRAM，而是先计算 working set 被各级 cache 覆盖的比例：

$$p_{\text{L1}} = \min\left(\frac{C_{\text{L1,eff}}}{W_{\text{set}}}, 1\right)$$

$$p_{\text{L2}} = \min\left(\frac{C_{\text{L2,eff}}}{W_{\text{set}}}, 1\right)$$

$$p_{\text{L3}} = \min\left(\frac{C_{\text{L3,eff}}}{W_{\text{set}}}, 1\right)$$

再拆成 exclusive residency share：

$$p_{\text{L2-only}} = \max(p_{\text{L2}} - p_{\text{L1}}, 0)$$

$$p_{\text{L3-only}} = \max(p_{\text{L3}} - p_{\text{L2}}, 0)$$

$$p_{\text{MEM}} = \max(1 - p_{\text{L3}}, 0)$$

代码内部把 residency 写成连续 level：

$$R = 1 \cdot p_{\text{L1}} + 2 \cdot p_{\text{L2-only}} + 3 \cdot p_{\text{L3-only}} + 4 \cdot p_{\text{MEM}}$$

其中 $$R=1$$ 表示接近 L1，$$R=2$$ 表示接近 L2，$$R=3$$ 表示接近 L3，$$R=4$$ 表示接近 DRAM。最后在各级 latency 间插值得到 center latency：

$$L_{\text{touch}} = \operatorname{interp}(R; [1,2,3,4], [L_{\text{L1}}, L_{\text{L2}}, L_{\text{L3}}, L_{\text{DRAM}}])$$

如果未启用 continuous cache residency，则会使用硬阈值：$$W_{\text{set}} \le C_{\text{L1,eff}}$$ 判为 L1，否则依次判断 L2、L3，超过 L3 后判为 DRAM。

另一种等价目标口径是把单次随机访问平均延迟写成 exclusive residency share 的加权平均：

$$L_{\text{avg}} = r_{\text{L1}}L_{\text{L1}} + r_{\text{L2}}L_{\text{L2}} + r_{\text{L3}}L_{\text{L3}} + r_{\text{MEM}}L_{\text{DRAM}}$$

这与当前代码的 continuous interpolation 不是完全相同的公式。二者相同之处是有效 cache 容量和 L1/L2/L3/DRAM 覆盖比例；差异在于最后一步 latency 映射：当前代码先得到连续 residency level $$R$$ 再插值，exclusive-share weighted-average 口径直接对各级 latency 加权平均。

当前已用 `calibrator/experimental/validate_random_access_latency_model.py` 对单线程 pointer-chase 真机数据做过验证。raw candidate YAML 口径下，continuous interpolation 的 MAPE/P90/Max APE 为 `6.89% / 12.91% / 17.60%`，exclusive-share weighted-average 口径为 `6.78% / 12.44% / 16.62%`，均控制在 20% 目标内；hard-threshold fallback 在连续 working-set 曲线验证中误差很大，因此只作为非 continuous fallback 保留，不作为趋势图推荐展示口径。

因此当前文档同时记录这两个公式口径：趋势图路径通过 `WHITEBOX_CONTINUOUS_CACHE_RESIDENCY=1` 启用 continuous cache residency；route 默认仍保留 hard-threshold fallback。本轮验证只说明随机访存平均 touch latency 子模型与 pointer-chase 真机延迟一致性较好，不等同于完整 Gather `T_random` 或 E2E latency 的验证。

`compute` 表示 FP 或 control arithmetic：

$$F_{\text{fp}} = N_{\text{worker}} \cdot f_{\text{GHz}} \cdot 1000 \cdot \text{SIMD}_{\text{lanes}} \cdot TP_{\text{op}}$$

$$T_{\text{compute\_throughput}} = \frac{N_{\text{ops}}}{F_{\text{fp}}}$$

$$T_{\text{compute\_tail}} = \alpha \cdot \frac{L_{\text{op}} \cdot D_{\text{dep}}}{f_{\text{GHz}} \cdot 1000 \cdot \sqrt{N_{\text{worker}}}}$$

$$T_{\text{compute}} = T_{\text{compute\_throughput}} + T_{\text{compute\_tail}}$$

其中 $$F_{\text{fp}}$$ 表示每微秒最多能完成多少 FP32 scalar operations，$$TP_{\text{op}}$$ 按算子选择 add、mul 或 FMA throughput，$$L_{\text{op}}$$ 是对应 FP latency cycles，$$D_{\text{dep}}$$ 是 reduce tree depth、Gemm K-depth 或默认 dependency depth。tiny shape arithmetic 使用 control-token capacity 而不是 FP capacity。

这里使用 FP32 scalar operations 作为 compute 记账单位，不表示硬件真的按 scalar 指令执行，而是为了把软件算术需求和硬件 SIMD 吞吐能力放到同一个单位中。软件侧的 `Gemm/MatMul` 算术量天然按 scalar element 统计，例如一次 MAC 近似包含一次 multiply 和一次 add：

$$N_{\text{ops}} = 2 \cdot \text{MAC}$$

硬件侧则由 SIMD 宽度、vector throughput、CPU 频率和 worker 数换算成每微秒可完成的 FP32 scalar operations。以 FP32 为例：

$$\text{SIMD}_{\text{lanes}} = \frac{\text{simd\_width\_bits}}{32}$$

因此 128-bit SIMD 对应 4 个 FP32 lane，256-bit SIMD 对应 8 个 FP32 lane。同一个软件 workload 的 $$N_{\text{ops}}$$ 不随硬件 SIMD 宽度变化；硬件差异只通过 $$F_{\text{fp}}$$ 改变，这样才能清楚解释 `simd_width_bits`、`cpu_clock` 和 `vector_sp_fma` 增大时 compute latency 下降。

实际代码中 `Gemm/MatMul` 使用更接近 FP32 lane 数的 FMA capacity；普通 elementwise/reduce 路径使用归一化 SIMD scale，避免把硬件配置中的 vector throughput 和 SIMD width 重复放大。文档中的 $$F_{\text{fp}}$$ 是统一抽象公式，各算子小节会说明具体使用 add、mul、FMA 还是 control-token capacity。

吞吐项 $$T_{\text{compute\_throughput}}$$ 表示“总算术工作量 / 每微秒算术能力”。例如 Device_2 上若 $$N_{\text{worker}}=3$$，$$f_{\text{GHz}}=2.893$$，256-bit SIMD 对 FP32 有 8 lanes，FMA throughput 约为 2 vector inst/cycle，则：

$$F_{\text{fp}} \approx 3 \cdot 2.893 \cdot 1000 \cdot 8 \cdot 2 = 138864\ \text{FP32 ops/us}$$

如果一个 MatMul 有 $$10^8$$ 次 MAC，则 $$N_{\text{ops}}=2 \cdot 10^8$$，吞吐项约为：

$$T_{\text{compute\_throughput}} \approx \frac{2 \cdot 10^8}{138864} = 1440\ \text{us}$$

依赖深度 $$D_{\text{dep}}$$ 描述关键路径上必须串行等待的计算层数，不是总操作数。例如顺序累加 $$(((a_1+a_2)+a_3)+a_4)$$ 的依赖深度约为 3；tree reduce $$((a_1+a_2)+(a_3+a_4))$$ 的依赖深度约为 2。`ReduceSum` 使用 reduction tree depth 和 intra-worker tree depth，`Gemm/MatMul` 使用 K 维累加深度的近似，普通 elementwise 的依赖深度较浅。

latency tail 用于补偿吞吐模型没有完全覆盖的串行依赖链或 pipeline latency。$$L_{\text{op}} \cdot D_{\text{dep}}$$ 给出关键路径上的 latency cycles，除以 $$f_{\text{GHz}} \cdot 1000$$ 后转换为 us；多 worker 可以隐藏一部分等待，但不是线性隐藏，因此使用 $$\sqrt{N_{\text{worker}}}$$ 做次线性折减。

$$\alpha$$ 不是训练拟合出来的参数，而是环境变量控制的 sensitivity hyperparameter。代码读取 `WHITEBOX_COMPUTE_LATENCY_TAIL_ALPHA`；未设置时默认值为 0，表示关闭 compute latency tail。当前趋势图脚本默认设置为 1.0，因此图中 `fp_latency_cycles.vector_sp_fma` sweep 会通过 latency tail 对 E2E latency 产生可见且方向正确的影响。

`dispatch` 表示 launch/调度开销：

$$C_{\text{launch}} = \frac{O_{\text{intra}}}{W_{\text{dispatch}} \cdot N_{\text{worker}}} + \frac{\max(G_{\text{active}} - 1, 0) \cdot O_{\text{inter}}}{W_{\text{dispatch}} \cdot N_{\text{worker}} \cdot G_{\text{active}}}$$

$$T_{\text{dispatch}} = N_{\text{dispatch\_event}} \cdot C_{\text{launch}}$$

其中 $$O_{\text{intra}}$$ 和 $$O_{\text{inter}}$$ 来自硬件配置中的 parallel launch overhead，$$W_{\text{dispatch}}$$ 来自 `pipeline.dispatch_width`，$$G_{\text{active}}$$ 是并行 branch/group 数。

这里的除法表示“单个 dispatch event 的摊销成本”，不是表示线程启动总成本随线程数线性下降。当前运行时通常复用线程池，`dispatch` 主要刻画任务分发、唤醒、branch handoff 和小算子 control path 的单位调度开销；真正的总调度开销由 $$N_{\text{dispatch\_event}}$$ 再乘回去。

第一项 $$O_{\text{intra}}/(W_{\text{dispatch}}\cdot N_{\text{worker}})$$ 表示单个算子内部并行启动开销。$$W_{\text{dispatch}}$$ 越大，硬件前端/调度窗口越宽，单位调度事件可被更高 dispatch 能力摊薄；$$N_{\text{worker}}$$ 越大，同一个算子的 launch/control 开销可被更大的 worker 执行窗口摊薄。因此这一项除以 $$W_{\text{dispatch}}\cdot N_{\text{worker}}$$。

第二项 $$\max(G_{\text{active}}-1,0)\cdot O_{\text{inter}}/(W_{\text{dispatch}}\cdot N_{\text{worker}}\cdot G_{\text{active}})$$ 表示多个 branch/group 同时活跃时的额外协调开销。只有一个 group 时没有额外 inter-branch 开销；多个 group 时，额外 group 数近似为 $$G_{\text{active}}-1$$。同时，这部分协调开销由当前活跃的 group 共同分担，因此再除以 $$G_{\text{active}}$$。等价地，可以理解为：

$$\frac{G_{\text{active}}-1}{G_{\text{active}}}\cdot\frac{O_{\text{inter}}}{W_{\text{dispatch}}\cdot N_{\text{worker}}}$$

因此，branch 数增加会提高 inter-branch 协调项，但不会把完整 $$O_{\text{inter}}$$ 线性重复计入每个 group。单算子线程数增加时，模型一方面通过分母降低单位 dispatch event 成本，另一方面通过 $$N_{\text{dispatch\_event}}$$ 描述调度事件数量；二者共同决定最终 $$T_{\text{dispatch}}$$。

### 6.0.1 公式变量说明

下面各算子公式共用这些符号。

| 符号 | 含义 |
|---|---|
| $$T_{\text{stream}}$$ | 连续内存读写或 copy 路径耗时，单位是 us |
| $$T_{\text{random}}$$ | Gather 随机访问路径耗时，单位是 us |
| $$T_{\text{random\_bw}}$$ | Gather 随机访问的带宽项耗时，单位是 us |
| $$T_{\text{random\_lat}}$$ | Gather 随机访问的延迟项耗时，单位是 us |
| $$T_{\text{compute}}$$ | 计算路径耗时，包含 throughput 项和 latency tail，单位是 us |
| $$T_{\text{compute\_throughput}}$$ | 计算吞吐项耗时，单位是 us |
| $$T_{\text{compute\_tail}}$$ | 计算 latency tail，表示依赖链或 pipeline latency 无法完全被吞吐隐藏的部分，单位是 us |
| $$T_{\text{dispatch}}$$ | 算子 launch、调度、branch handoff 的 active 开销，单位是 us |
| $$T_{\text{MetadataView}}$$ | metadata/view 算子的总白盒耗时；当前数据和文档口径中统一按 dispatch-only 处理 |
| $$T_{\text{Gather}}$$ | Gather 算子的总白盒耗时 |
| $$T_{\text{ReduceSum}}$$ | ReduceSum 算子的总白盒耗时 |
| $$T_{\text{Gemm}}$$ | Gemm/MatMul 算子的总白盒耗时 |
| $$T_{\text{ConcatPayload}}$$ | payload Concat 算子的总白盒耗时 |
| $$T_{\text{ConcatShape}}$$ | shape/control Concat 算子的总白盒耗时 |
| $$T_{\text{Transpose}}$$ | materialized Transpose 算子的总白盒耗时 |
| $$T_{\text{TransposeView}}$$ | view-like Transpose 算子的总白盒耗时 |
| $$T_{\text{Elementwise}}$$ | Elementwise 算子的总白盒耗时 |
| $$T_{\text{TinyShape}}$$ | tiny shape arithmetic 算子的总白盒耗时 |
| $$T_{\text{Fallback}}$$ | fallback 算子的总白盒耗时 |

内存、带宽和 cache 相关符号如下。

| 符号 | 含义与来源 |
|---|---|
| $$BW_{\text{copy}}$$ | 当前算子在 worker 数和 branch contention 下的有效连续 copy 带宽，由 $$C_{\text{worker}} / S_{\text{branch}}$$ 得到 |
| $$BW_{\text{copy,1t}}$$ | 硬件配置中的单线程连续 copy 带宽，来自 `memory.copy_bw_gb_s_1t` |
| $$BW_{\text{copy,sat}}$$ | 硬件配置中的多线程连续 copy 饱和带宽，来自 `memory.copy_bw_gb_s_saturated` |
| $$BW_{\text{random}}$$ | 当前 Gather 在 worker 数和 branch contention 下的有效随机访问带宽，由 random bandwidth 的 $$C_{\text{worker}} / S_{\text{branch}}$$ 得到 |
| $$BW_{\text{random,1t}}$$ | 硬件配置中的单线程随机 Gather 带宽，来自 `memory.random_gather_bw_gb_s_1t` |
| $$BW_{\text{random,sat}}$$ | 硬件配置中的多线程随机 Gather 饱和带宽，来自 `memory.random_gather_bw_gb_s_saturated` |
| $$BW_{\text{1t}}$$ | 通用写法中的单线程带宽；copy 路径对应 `copy_bw_gb_s_1t`，random 路径对应 `random_gather_bw_gb_s_1t` |
| $$BW_{\text{sat}}$$ | 通用写法中的饱和带宽；copy 路径对应 `copy_bw_gb_s_saturated`，random 路径对应 `random_gather_bw_gb_s_saturated` |
| $$C_{\text{worker}}(\cdot)$$ | worker capacity curve，由单线程带宽、饱和带宽、有效 worker 数和 `core.total_cores` 线性插值得到，不需要数据拟合 |
| $$f_{\text{worker}}$$ | worker capacity 插值比例，来自 $$N_{\text{worker}}$$ 和 $$N_{\text{core}}$$ |
| $$S_{\text{branch}}$$ | branch contention share，表示并发 worker/branch 对带宽的竞争；由 $$N_{\text{worker}}$$、$$G_{\text{active}}$$ 和 $$N_{\text{core}}$$ 动态计算，值越大，有效带宽越低 |
| $$N_{\text{parallel}}$$ | 当前算子并发访问内存的 parallel unit 数，代码中近似为 $$N_{\text{worker}} \cdot G_{\text{active}}$$ 并截断到 `core.total_cores` |
| $$f_{\text{contention}}$$ | branch contention 比例，由 $$N_{\text{parallel}}$$ 和 $$N_{\text{core}}$$ 得到 |
| $$B_{\text{stream}}$$ | 连续读写/copy 字节数，单位是 bytes；来自该算子的 read/write bytes 或特定 copy window |
| $$B_{\text{read}}$$ | 算子输入读取字节数，单位是 bytes；由输入 tensor shape、dtype 和参数 tensor 大小得到 |
| $$B_{\text{write}}$$ | 算子输出写回字节数，单位是 bytes；由输出 tensor shape 和 dtype 得到 |
| $$B_{\text{tail\_copy}}$$ | Gather tail payload copy 字节数，单位是 bytes；由 Gather 输出 payload 或 ORT kernel dest payload 得到 |
| $$B_{\text{random\_head}}$$ | Gather row-head 随机访问字节数，单位是 bytes；由有效 touch 次数乘以 cacheline bytes 得到 |
| $$B_{\text{cacheline}}$$ | cacheline 大小，来自 `cache.cacheline_bytes`，单位是 bytes |
| $$B_{\text{partial\_write}}$$ | ReduceSum worker partial result 写回字节数，由 reduction 输出 shape、worker 分片和 dtype 得到 |
| $$B_{\text{merge\_write}}$$ | ReduceSum 跨 branch merge result 写回字节数，由跨 branch merge 的输出 shape 和 dtype 得到 |
| $$\operatorname{bytes\_per\_us}(BW)$$ | 把 GB/s 带宽换算成 bytes/us 的函数，代码按十进制近似使用 $$\operatorname{bytes\_per\_us}(BW)=1000 \cdot BW$$ |
| $$W_{\text{set}}$$ | Gather working set 大小，用于判断 L1/L2/L3/DRAM residency；优先来自 `gather_working_set_eff_bytes`，否则由 lookup count、row bytes 和 unique ratio 估计 |
| $$C_{\text{L1}}$$ | 当前 active L1D 容量，来自 `cache.l1d.size` 并按 active group 摊分 |
| $$C_{\text{L2}}$$ | 当前 active L2 容量，来自 `cache.l2.size` 并按 active group 摊分 |
| $$C_{\text{L3}}$$ | 当前 active L3 容量，来自 `cache.l3_per_die.size` 或 `cache.l3.size` 并按 active group 摊分 |
| $$L_{\text{L1}}$$ | L1D touch latency，由 `cache.l1d.response_latency_cycles` 除以 CPU 频率换算得到 |
| $$L_{\text{L2}}$$ | L2 touch latency，由 `cache.l2.response_latency_cycles` 除以 CPU 频率换算得到 |
| $$L_{\text{L3}}$$ | L3 touch latency，由 `cache.l3_per_die.response_latency_cycles` 或 `cache.l3.response_latency_cycles` 除以 CPU 频率换算得到 |
| $$L_{\text{DRAM}}$$ | DRAM random access latency，由 `memory.local_mem_delay_ns` 换算得到 |
| $$L_{\text{touch}}$$ | 根据 working set residency 在 L1/L2/L3/DRAM latency 间选择或插值得到的单次 touch latency |
| $$N_{\text{touch}}$$ | Gather 有效随机 row-head touch 次数，由 lookup count、unique ratio 和 ORT task count 等静态特征估计 |
| $$P_{\text{latency}}$$ | 随机访存 latency parallelism，由 worker 数、issue width、dispatch width、commit width 和 BDP 下界共同决定 |
| $$\operatorname{cache\_latency}(\cdot)$$ | 根据 working set 和 cache 容量/延迟计算单次 touch latency 的函数 |
| $$\operatorname{compose}(\cdot)$$ | random bandwidth 和 latency 的组合函数；默认取 max，additive 模式取和，也可使用 max 加 tail 的形式 |

计算和调度相关符号如下。

| 符号 | 含义与来源 |
|---|---|
| $$N_{\text{worker}}$$ | 有效 worker 数，由 intra-op threads、inter-op branches、硬件 core 数和并行暴露度共同决定 |
| $$N_{\text{core}}$$ | 当前单 NUMA 执行域可用物理核心数，来自 `core.total_cores` |
| $$f_{\text{GHz}}$$ | CPU 频率，来自 `core.cpu_clock` |
| $$\text{SIMD}_{\text{lanes}}$$ | FP32 SIMD lane 数，等于 `instruction.simd_width_bits / 32`；这是 compute 记账单位中的 lane 数，不表示硬件按 scalar 指令执行 |
| $$TP_{\text{op}}$$ | 对应算子的 vector throughput，按 op 选择 add、mul 或 FMA throughput，来自 `instruction.fp_throughput_per_cycle.*` |
| $$F_{\text{fp}}$$ | 当前算子的 FP32 scalar operation capacity，单位是 ops/us；由 worker 数、CPU 频率、SIMD lane 数和 vector throughput 换算得到 |
| $$F_{\text{add}}$$ | ReduceSum 使用的 FP add capacity，单位是 items/us |
| $$F_{\text{add/mul}}$$ | Elementwise Add/Mul 使用的 FP add 或 mul capacity，单位是 items/us |
| $$F_{\text{op}}$$ | fallback 路径按算子类型选择的 FP capacity，单位是 items/us |
| $$\text{FP32 FMA capacity}$$ | Gemm/MatMul 使用的 FP32 FMA capacity，单位是 FP ops/us |
| $$C_{\text{control\_token}}$$ | tiny shape arithmetic 使用的 control-token capacity，由 pipeline width 和 CPU 频率得到 |
| $$N_{\text{ops}}$$ | 算子计算需求量，单位是 FP32 scalar operations 或 scalar items；`Gemm/MatMul` 通常取 $$2 \cdot \text{MAC}$$，其他算子由各自小节定义 |
| $$N_{\text{input\_elements}}$$ | 输入 tensor 元素数 |
| $$N_{\text{output\_elements}}$$ | 输出 tensor 元素数 |
| $$N_{\text{input\_tensors}}$$ | 输入 tensor 个数 |
| $$L_{\text{op}}$$ | 当前算子使用的 FP latency cycles，按 op 选择 add、mul 或 FMA latency |
| $$L_{\text{add}}$$ | FP add latency cycles，来自 `instruction.fp_latency_cycles.vector_sp_add` |
| $$L_{\text{FMA}}$$ | FP FMA latency cycles，来自 `instruction.fp_latency_cycles.vector_sp_fma` |
| $$D_{\text{dep}}$$ | dependency depth，表示关键路径上必须串行等待的计算层数，用于估计 latency tail；它不是总操作数 |
| $$D_{\text{reduction}}$$ | ReduceSum reduction axis tree depth |
| $$D_{\text{intra\_tree}}$$ | ReduceSum intra-worker tree depth |
| $$K$$ | Gemm/MatMul 的 inner dimension 近似值，代码里由 MAC 数除以输出元素数得到 |
| $$\alpha$$ | latency tail 权重系数，用于控制 latency tail 在 compute time 中的占比；不是拟合参数，由环境变量 `WHITEBOX_COMPUTE_LATENCY_TAIL_ALPHA` 设置，未设置时默认 0，当前趋势图默认 1.0 |
| $$\text{MAC}$$ | Gemm/MatMul 的 multiply-accumulate 次数 |
| $$T_{\text{FMA latency tail}}$$ | Gemm/MatMul 的 FMA latency tail |
| $$T_{\text{latency tail}}$$ | Elementwise、tiny shape 或 fallback 的通用 compute latency tail |
| $$O_{\text{intra}}$$ | intra-op parallel launch overhead，来自 `runtime.parallel_launch_overhead_us_intra` |
| $$O_{\text{inter}}$$ | inter-branch launch overhead，来自 `runtime.parallel_launch_overhead_us_inter` |
| $$W_{\text{dispatch}}$$ | dispatch width，来自 `pipeline.dispatch_width` |
| $$G_{\text{active}}$$ | 当前并行 active group 或 branch 数，来自静态图中的 branch/group 并行信息 |
| $$C_{\text{launch}}$$ | 单个 dispatch event 的 launch cost，单位是 us/event |
| $$C_{\text{inline\_launch}}$$ | shape/control 小算子的 inline launch cost，通常低于普通 dispatch cost |
| $$N_{\text{dispatch\_event}}$$ | 当前算子的 dispatch event 数，由 worker excess 和 branch handoff 决定 |
| $$I_{\text{materialize}}$$ | 是否需要 materialize 的 0/1 指示变量 |
| $$I_{\text{merge}}$$ | 是否存在 merge obligation 的 0/1 或非负权重 |
| $$I_{\text{reshape\_like}}$$ | Transpose 是否退化为 reshape-like view 的 0/1 指示变量 |

### 6.1 Metadata / view 类算子

典型算子：

```text
Shape, Reshape, Flatten, Unsqueeze, shape Gather
```

active 服务模块：

```text
dispatch
```

当前文档口径中，metadata/view 类算子全部按 dispatch-only 处理，不再为 `Reshape`、`Flatten` 或其他 view-like 算子加入 `stream_mem`。原因是当前训练/测试数据没有观察到稳定的 payload materialization 证据：即使 `Reshape` 输入/输出达到 GB 级，实测耗时仍是十几到几十微秒，明显不符合真实 copy 带宽对应的毫秒级耗时。

主要硬件参数：

- `runtime.parallel_launch_overhead_us_*`
- `pipeline.dispatch_width`
- `core.total_cores`

模块计算：

- `Shape`、`Reshape`、`Flatten`、`Unsqueeze`、shape `Gather` 都不产生有效 payload 搬运，令 $$T_{\text{stream}} = 0$$。
- `Shape`、shape `Gather` 等 control 算子使用 inline dispatch cost，其他 metadata/view 算子使用普通 dispatch cost。
- dispatch event 近似为一次算子调度加必要的 merge obligation。

dispatch 计算：

$$T_{\text{dispatch}} = (1 + I_{\text{merge}}) \cdot C_{\text{launch}}$$

组合方式：

$$T_{\text{MetadataView}} = T_{\text{dispatch}}$$

离线验证结果保存在：

```text
outputs/hw_generalization_physics_guarded_rf_v3_20260429/metadata_view_nostream_ablation.csv
```

其中 `Shape/Reshape/Flatten/Unsqueeze` 以及 shape `Gather/Concat` 的 `svc_stream_mem_center_us` 均为 0，去掉 `stream_mem` 后 Analytical MAPE 变化为 0。因此当前误差主要来自 dispatch/control latency 标定，而不是数据搬运项。

### 6.2 Embedding Gather

典型算子：

```text
embedding branch Gather
```

active 服务模块：

```text
random_mem
stream_mem
dispatch
```

随机访问部分包含带宽项和延迟项：

$$T_{\text{random}} = \max(T_{\text{random\_bw}}, T_{\text{random\_lat}})$$

在 additive latency 模式下可写成：

$$T_{\text{random}} = T_{\text{random\_bw}} + T_{\text{random\_lat}}$$

主要硬件参数：

- `memory.random_gather_bw_gb_s_1t/saturated`
- `memory.copy_bw_gb_s_1t/saturated`
- `cache.cacheline_bytes`
- `cache.l1d/l2/l3.size`
- `cache.l1d/l2/l3.response_latency_cycles`
- `memory.local_mem_delay_ns`
- `pipeline.issue_width`
- `pipeline.dispatch_width`
- `pipeline.commit_width`
- `core.total_cores`
- `runtime.parallel_launch_overhead_us_*`

模块计算：

- random head 字节由有效唯一行首访问次数和 cacheline bytes 决定。
- tail copy 字节由 Gather 输出 payload 或 ORT kernel dest payload 决定。
- cache residency 根据 working set 与 active L1/L2/L3 容量的比例决定；continuous cache 模式下使用 L1/L2/L3/DRAM latency 的插值。
- latency parallelism 由 worker 数、issue width、dispatch width、commit width 共同决定。

random head 带宽项：

$$B_{\text{random\_head}} = N_{\text{touch}} \cdot B_{\text{cacheline}}$$

$$T_{\text{random\_bw}} = \frac{B_{\text{random\_head}}}{\operatorname{bytes\_per\_us}(BW_{\text{random}})}$$

random head 延迟项：

$$T_{\text{random\_lat}} = \frac{N_{\text{touch}} \cdot L_{\text{touch}}}{P_{\text{latency}}}$$

tail copy 项：

$$T_{\text{stream}} = \frac{B_{\text{tail\_copy}}}{\operatorname{bytes\_per\_us}(BW_{\text{copy}})}$$

dispatch 项：

$$T_{\text{dispatch}} = N_{\text{dispatch\_event}} \cdot C_{\text{launch}}$$

组合方式：

$$T_{\text{Gather}} = \max(T_{\text{random}}, T_{\text{stream}}) + T_{\text{dispatch}}$$

### 6.3 ReduceSum

典型算子：

```text
ReduceSum
```

active 服务模块：

```text
stream_mem
compute
dispatch
```

主要硬件参数：

- `memory.copy_bw_gb_s_1t/saturated`
- `instruction.simd_width_bits`
- `core.cpu_clock`
- `instruction.fp_throughput_per_cycle.vector_sp_add`
- `instruction.fp_latency_cycles.vector_sp_add`
- `pipeline.issue/dispatch/commit_width`
- `runtime.parallel_launch_overhead_us_*`

模块计算：

- stream 字节包含输入读取、worker partial write 和跨 branch merge write。
- compute item 数取输入 element 数和输出 element 数中的较大者。
- compute tail 使用 reduction axis depth、intra tree depth 作为 dependency depth。
- dispatch 使用 reduction 算子的调度事件数。

stream 计算：

$$B_{\text{stream}} = B_{\text{read}} + B_{\text{partial\_write}} + B_{\text{merge\_write}}$$

$$T_{\text{stream}} = \frac{B_{\text{stream}}}{\operatorname{bytes\_per\_us}(BW_{\text{copy}})}$$

compute 计算：

$$N_{\text{ops}} = \max(N_{\text{input\_elements}}, N_{\text{output\_elements}})$$

$$D_{\text{dep}} = D_{\text{reduction}} + D_{\text{intra\_tree}}$$

$$T_{\text{compute}} = \frac{N_{\text{ops}}}{F_{\text{add}}} + \alpha \cdot \frac{L_{\text{add}} \cdot D_{\text{dep}}}{f_{\text{GHz}} \cdot 1000 \cdot \sqrt{N_{\text{worker}}}}$$

组合方式：

$$T_{\text{ReduceSum}} = \max(T_{\text{stream}}, T_{\text{compute}}) + T_{\text{dispatch}}$$

### 6.4 Gemm / MatMul

典型算子：

```text
Gemm, MatMul
```

active 服务模块：

```text
stream_mem
compute
dispatch
```

主要硬件参数：

- `instruction.simd_width_bits`
- `core.cpu_clock`
- `instruction.fp_throughput_per_cycle.vector_sp_fma`
- `instruction.fp_latency_cycles.vector_sp_fma`
- `memory.copy_bw_gb_s_1t/saturated`
- `pipeline.issue/dispatch/commit_width`
- `runtime.parallel_launch_overhead_us_*`

模块计算：

- compute demand 由 MAC 数决定，FP32 FMA 一次 MAC 近似为 2 个 FP operations。
- stream 字节是输入矩阵读取和输出矩阵写回字节。
- compute tail 使用 Gemm/MatMul 的 K-depth 近似 dependency depth。
- dispatch 使用 Gemm/MatMul 算子的调度事件数。

compute 计算：

$$T_{\text{compute}} = \frac{2 \cdot \text{MAC}}{\text{FP32 FMA capacity}} + T_{\text{FMA latency tail}}$$

其中：

$$T_{\text{FMA latency tail}} = \alpha \cdot \frac{L_{\text{FMA}} \cdot \log(1 + K)}{f_{\text{GHz}} \cdot 1000 \cdot \sqrt{N_{\text{worker}}}}$$

stream 计算：

$$T_{\text{stream}} = \frac{B_{\text{read}} + B_{\text{write}}}{\operatorname{bytes\_per\_us}(BW_{\text{copy}})}$$

组合方式：

$$T_{\text{Gemm}} = \max(T_{\text{stream}}, T_{\text{compute}}) + T_{\text{dispatch}}$$

`MatMul` 使用同一类建模方式。

### 6.5 Concat

`Concat` 会被拆成 payload concat 和 shape concat。

Payload concat active 服务模块：

```text
stream_mem
dispatch
```

Shape concat active 服务模块：

```text
dispatch
```

主要硬件参数：

- `memory.copy_bw_gb_s_1t/saturated`
- `pipeline.fetch/decode/rename/dispatch_width`
- `core.cpu_clock`
- `runtime.parallel_launch_overhead_us_*`

模块计算：

- payload concat 主要是把多个输入 payload 连续 copy 到输出 buffer，copy window 取 read/write 字节中的较大者。
- shape concat 只处理 shape/control 信息，当前 active cost 主要是 dispatch。

payload stream 计算：

$$B_{\text{stream}} = \max(B_{\text{read}}, B_{\text{write}})$$

$$T_{\text{stream}} = \frac{B_{\text{stream}}}{\operatorname{bytes\_per\_us}(BW_{\text{copy}})}$$

shape dispatch 计算：

$$T_{\text{dispatch}} = C_{\text{inline\_launch}}$$

组合方式：

Payload concat：

$$T_{\text{ConcatPayload}} = T_{\text{stream}} + T_{\text{dispatch}}$$

Shape concat：

$$T_{\text{ConcatShape}} = T_{\text{dispatch}}$$

### 6.6 Transpose

典型算子：

```text
Transpose
```

active 服务模块：

```text
stream_mem
dispatch
```

主要硬件参数：

- `memory.copy_bw_gb_s_1t/saturated`
- `pipeline.fetch/decode/rename/dispatch_width`
- `pipeline.issue/dispatch/commit_width`
- `runtime.parallel_launch_overhead_us_*`

模块计算：

- reshape-like transpose 不产生真实数据搬运，stream 字节置 0。
- materialized transpose 使用连续 copy 近似，copy window 取 read/write 字节中的较大者。
- dispatch 使用 Transpose 算子的调度事件数。

stream 计算：

$$B_{\text{stream}} = (1 - I_{\text{reshape\_like}}) \cdot \max(B_{\text{read}}, B_{\text{write}})$$

$$T_{\text{stream}} = \frac{B_{\text{stream}}}{\operatorname{bytes\_per\_us}(BW_{\text{copy}})}$$

组合方式：

materialized transpose：

$$T_{\text{Transpose}} = T_{\text{stream}} + T_{\text{dispatch}}$$

view-like transpose：

$$T_{\text{TransposeView}} = T_{\text{dispatch}}$$

### 6.7 Elementwise 和 tiny shape arithmetic

Elementwise active 服务模块：

```text
stream_mem
compute
dispatch
```

Tiny shape arithmetic active 服务模块：

```text
compute
dispatch
```

主要硬件参数：

- `memory.copy_bw_gb_s_1t/saturated`
- `instruction.simd_width_bits`
- `instruction.fp_throughput_per_cycle.vector_sp_add/mul`
- `instruction.fp_latency_cycles.*`
- `core.cpu_clock`
- `pipeline.*_width`
- `runtime.parallel_launch_overhead_us_*`

模块计算：

- Elementwise stream 字节为输入读取加输出写回。
- Elementwise compute item 数为输出元素数乘以输入 tensor 组合数。
- tiny shape arithmetic 的 compute 使用 control-token capacity，需求量由输出元素数下界到 1 得到。
- dispatch 使用对应算子的调度事件数；shape/control 路径使用 inline dispatch。

Elementwise stream 计算：

$$T_{\text{stream}} = \frac{B_{\text{read}} + B_{\text{write}}}{\operatorname{bytes\_per\_us}(BW_{\text{copy}})}$$

Elementwise compute 计算：

$$N_{\text{ops}} = N_{\text{output\_elements}} \cdot \max(N_{\text{input\_tensors}} - 1, 1)$$

$$T_{\text{compute}} = \frac{N_{\text{ops}}}{F_{\text{add/mul}}} + T_{\text{latency tail}}$$

Tiny shape compute 计算：

$$T_{\text{compute}} = \frac{\max(N_{\text{output\_elements}}, 1)}{C_{\text{control\_token}}} + T_{\text{latency tail}}$$

Elementwise 组合方式：

$$T_{\text{Elementwise}} = \max(T_{\text{stream}}, T_{\text{compute}}) + T_{\text{dispatch}}$$

Tiny shape arithmetic 组合方式：

$$T_{\text{TinyShape}} = T_{\text{compute}} + T_{\text{dispatch}}$$

### 6.8 其他 fallback 算子

没有专门路径的算子使用 fallback：

```text
stream_mem
compute
dispatch
```

模块计算：

- 如果算子需要 materialize，则 stream 字节为输入输出字节；否则 stream 字节为 0。
- compute item 数默认使用输出元素数。
- dispatch 使用普通 launch/dispatch 事件。

stream 计算：

$$T_{\text{stream}} = \frac{(B_{\text{read}} + B_{\text{write}}) \cdot I_{\text{materialize}}}{\operatorname{bytes\_per\_us}(BW_{\text{copy}})}$$

compute 计算：

$$T_{\text{compute}} = \frac{N_{\text{output\_elements}}}{F_{\text{op}}} + T_{\text{latency tail}}$$

组合方式：

$$T_{\text{Fallback}} = \max(T_{\text{stream}}, T_{\text{compute}}) + T_{\text{dispatch}}$$

## 7. 面向专家审查的解释口径

1. 图中不再只展示单一 workload，而是展示 small、base、large 三档负载规模。
2. 每条曲线的软件负载规模和硬件 sweep 范围都有 CSV/JSON manifest，可复现。
3. RF 不直接读取 raw hardware fields，硬件变化先通过 Analytical model 变成 service-time anchors。
4. v3 使用硬件反事实训练、单调 anchor、row-level guard、E2E aggregate projection 四层机制控制趋势。
5. 每类算子的 Analytical 公式都能对应到明确的硬件参数和服务模块，因此可以解释为什么某个硬件参数变化会影响对应 workload。
