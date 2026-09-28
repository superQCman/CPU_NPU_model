# CPU 性能模型内部原理

本文件说明当前主线模型的内部机制、关键边界和趋势控制方式。使用命令见 `USER_GUIDE.md`。

## 1. 主线链路

当前主线 route：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

完整数据流：

```text
原始单算子样本 + 硬件 YAML
  -> Analytical prior
  -> op-thread scale
  -> physics-guarded RF
  -> 单算子预测
  -> E2E pipeline
  -> 整图预测和误差指标
```

训练后静态配置推理使用另一条入口：

```text
DLRM 上层配置 + 已导出模型 bundle
  -> static full_graph_76
  -> static model_input_51
  -> Analytical prior + op-thread scale + physics-guarded RF
  -> static E2E pipeline
  -> predicted_e2e_us
```

核心约束：

- RF 不直接读取硬件型号、硬件 ID 或 raw hardware fields。
- 硬件影响先由 Analytical prior 转成 service-time anchors。
- RF 只在白盒估计基础上修正残差。
- E2E 结果由 pipeline 调度得到，不使用简单算子求和作为最终口径。

## 2. 输入样本和硬件参数

一行训练样本通常对应一个算子节点，包含：

| 字段类型 | 例子 | 用途 |
|---|---|---|
| 图和样本定位 | `row_uid`, `case_id`, `combo`, `node_name` | 调试、E2E 分组、artifact 对齐 |
| workload 规模 | `batch_size`, `num_indices_per_lookup`, `arch_embedding_size`, `arch_mlp_bot`, `arch_mlp_top` | 推导 bytes、FLOPs、Gather working set |
| 线程配置 | `num_threads`, `inter_threads` | worker 数、op-thread scale、E2E branch pool |
| 算子语义 | `op_type`, shape/static flags, graph position fields | 选择 Analytical 规则和 RF group |
| 监督标签 | `actual_us` | 训练 label 和误差计算 |

硬件 YAML 的可信来源固定为：

```text
calibrator/hw_config
```

主线读取的关键硬件字段包括：

| 模块 | 字段示例 | 影响 |
|---|---|---|
| core | `core.total_cores`, `core.cpu_clock` | worker 上限、cycles 到时间换算、吞吐能力 |
| pipeline | `fetch_width`, `decode_width`, `dispatch_width`, `issue_width`, `commit_width` | dispatch、control path、latency parallelism |
| instruction | `simd_width_bits`, FP throughput, FP latency | compute service time |
| memory | copy/random bandwidth, `local_mem_delay_ns` | stream/random memory service time |
| cache | L1/L2/L3 size 和 response latency | random memory residency 和 touch latency |
| runtime | parallel launch overhead | dispatch service time |

`vendor`、`model_name`、`sockets`、`numa_nodes` 等字段可用于审计或记录，不作为 RF 硬件泛化输入。

## 3. Analytical prior

Analytical prior 是白盒估计层。它根据硬件参数和 workload 规模估计每个算子的服务时间。

当前参与 `whitebox_center_us` 合成的 active modules 只有四类：

| 模块 | 输出字段 | 含义 |
|---|---|---|
| `stream_mem` | `svc_stream_mem_us` | 连续读写、copy、payload 搬运 |
| `random_mem` | `svc_random_mem_us` | Gather 随机访问 |
| `compute` | `svc_compute_us` | Gemm、MatMul、Reduce、Elementwise 计算 |
| `dispatch` | `svc_dispatch_us` | 线程启动、任务调度、小算子 floor |

历史模块 `descriptor`、`reorder`、`sync` 仍可能出现在 CSV schema、bundle schema 或诊断列中。当前代码在合成总耗时前将这些模块清零：

```text
DISABLED_ANALYTICAL_MODULES = {"descriptor", "reorder", "sync"}
```

因此主线 `whitebox_center_us` 只由 active modules 贡献。

关键输出：

| 字段 | 含义 |
|---|---|
| `whitebox_lower_us` | 白盒偏乐观估计 |
| `whitebox_center_us` | 白盒中心估计，后续 scale 和 RF 的主要 anchor |
| `whitebox_upper_us` | 白盒偏保守估计 |
| `whitebox_exec_mode` | 算子执行模式 |
| `whitebox_rf_group` | RF group 分类 |
| `svc_*_center_us` | 分模块 service-time anchor |

## 4. 算子组合规则

各算子按执行模式组合 active service modules。

| 算子类 | 组合口径 |
|---|---|
| Metadata / view | `T_dispatch` |
| Embedding Gather | `max(T_random, T_stream) + T_dispatch` |
| ReduceSum | `max(T_stream, T_compute) + T_dispatch` |
| Gemm / MatMul | `max(T_stream, T_compute) + T_dispatch` |
| Payload Concat | `T_stream + T_dispatch` |
| Shape Concat / Shape / Unsqueeze / Flatten | `T_dispatch` |
| Transpose | materialized 时近似 `T_stream + T_dispatch` |
| Elementwise / fallback | `max(T_stream, T_compute) + T_dispatch` |

Metadata/view 类当前统一按 dispatch-only 处理。已有训练/测试数据未显示稳定 payload materialization 证据，避免把 view 类误计为大规模 copy。

## 5. Random memory 和 cache residency

`random_mem` 先估计 Gather working set，再判断随机访问更可能命中 L1、L2、L3 还是 DRAM。

working set 口径是“当前 Gather 算子 / 当前 branch 的整体 working set”，不是整张 E2E 图的所有 branch 加总。

有效 cache 容量口径：

```text
N_worker = min(num_threads, total_cores)
N_system = clip(N_worker * active_groups, 1, total_cores)
C_L1_eff = N_worker * C_L1
C_L2_eff = N_worker * C_L2
C_L3_eff = (N_worker / N_system) * C_L3
```

L1/L2 是 per-core private cache，随当前算子 worker 数扩展。L3/LLC 是共享 cache，按当前算子 worker 在系统活跃 worker 中的占比分配。

当前 route 代码默认 `WHITEBOX_CONTINUOUS_CACHE_RESIDENCY=0`，保留 hard-residency fallback。趋势图脚本默认设置：

```text
WHITEBOX_CONTINUOUS_CACHE_RESIDENCY=1
```

趋势图路径下，continuous cache residency 先计算各级累计覆盖比例，再转成 continuous residency level，并在 L1/L2/L3/DRAM latency 间插值。pointer-chase 验证中，continuous interpolation 和 exclusive-share weighted-average 两种口径都能达到 20% 内误差目标；hard-threshold fallback 在连续 working-set 曲线验证中误差较大，主要作为非 continuous fallback 保留。

## 6. op-thread scale

op-thread scale 位于 Analytical prior 之后、RF 之前。它不是 RF，也不是新的硬件模型。

拟合输入来自 fit split：

```text
op_type
num_threads
actual_us
whitebox_center_us
```

每个 `(op_type, num_threads)` group 拟合一个比例系数：

```text
s_op,t = weighted_median(actual_us / whitebox_center_us)
```

应用时：

```text
whitebox_scaled_us = whitebox_unscaled_us * s_op,t
```

输出会保留：

| 字段 | 含义 |
|---|---|
| `whitebox_op_thread_scale` | 当前行使用的 scale |
| `whitebox_op_thread_scale_source` | scale 来源 |
| `*_unscaled_us` | 缩放前备份列 |

当前主线训练命令必须启用：

```text
--op-thread-scale-mode fit
```

## 7. physics-guarded RF

RF 学习 whitebox 估计之外的残差。当前 v3 链路使用三类保护机制：

- 硬件反事实训练增强。
- 单调 anchor。
- 推理阶段 hardware guard。

### 7.1 RF 输入边界

当前 `physics_guarded_rf_v3` 的 `allow_workload_features=False`。RF 默认不直接读取大多数原始 workload size 列，例如：

```text
output_size
activation_size
feat_lookup_count
feat_gemm_mac_count
exact_read_bytes
exact_write_bytes
```

这些规模信息先进入 Analytical prior，转成：

```text
whitebox_center_us
whitebox_exec_center_us
svc_stream_mem_center_us
svc_random_mem_center_us
svc_compute_center_us
svc_dispatch_center_us
```

RF 使用的是白盒 service-time anchor、静态语义字段、图位置字段和 RF group 特征，而不是原始硬件字段。

### 7.2 硬件反事实训练增强

真实训练数据只覆盖少量真实机器。为了让 RF 在训练阶段见到合理硬件变化方向，v3 会构造 synthetic 样本。

对真实样本：

```text
y0 = baseline 硬件实测耗时
p0 = baseline 硬件 whitebox prior
ph = 扰动硬件 whitebox prior
```

synthetic label：

```text
yh_synthetic = ph + (y0 - p0)
```

含义：

- 硬件变化方向和幅度由 whitebox prior 决定。
- baseline 上 RF 需要补的残差保留。
- synthetic 样本只进入训练，不进入真实 validation MAPE。
- 当前真实样本权重为 `1.0`，synthetic 样本权重为 `0.25`。

### 7.3 单调 anchor

v3 对关键 service-time anchors 施加正单调偏置：

```text
whitebox_center_us
whitebox_exec_center_us
whitebox_serial_overhead_center_us
svc_stream_mem_center_us
svc_random_mem_center_us
svc_compute_center_us
svc_dispatch_center_us
```

当 service-time anchor 增大时，RF 不应仅因为该 anchor 增大而预测更快。当前 v3 仍沿用 residual-ratio target 体系，因此该单调项是训练偏置，不是最终硬保证。

### 7.4 hardware guard

硬件 sweep 或 `override_hw_config` 推理时，v3 同时计算 baseline 硬件和目标硬件下的 whitebox/RF 输出：

```text
delta_whitebox = p_h - p_0
delta_rf = r_h - r_0
```

如果方向冲突：

```text
delta_whitebox * delta_rf < 0
```

使用 frozen residual 回退：

```text
y_h = p_h + (r_0 - p_0)
```

如果方向一致但 RF 变化幅度过大，则限制幅度：

```text
|delta_rf| <= 2 * |delta_whitebox|
```

correct-hw 精度评估通常不依赖 guard。frozen-swap、硬件参数扰动和 synthetic trend sweep 更容易触发 guard。

## 8. E2E pipeline

单算子预测不能直接求和作为最终整图延迟。DLRM-like 图中存在：

- 并行分支。
- barrier。
- branch pool。
- merge。
- tail 和 critical path。

E2E pipeline 会把单算子预测放回图结构中排流水线，输出完整图预测延迟。

训练评估阶段需要两个 artifact：

| artifact | 用途 |
|---|---|
| `op_shapes_*.csv` | 节点 shape 和静态规模 |
| `branch_parallel_op_timeline.csv` | branch 并行时间线 |

远端数据缺少 portable artifact 时，E2E 会退化为 partial combo，不能作为精度结果。

静态配置推理阶段没有真实 profile artifact。系统会从 DLRM 上层配置生成：

| 文件 | 含义 |
|---|---|
| `static_full_graph_76.csv` | 完整静态图，76 个节点 |
| `static_model_input_51.csv` | 主线可观测输入，51 个节点 |
| `static_e2e_summary.json` | 主线静态 E2E 预测结果 |

`predicted_e2e_us` 使用 `model_input_51_observable_profiled_nodes` 口径。`full_graph_76` 用于审计完整静态图，不替代当前主线 E2E latency 口径。

上层参数到算子规模的映射由 `calibrator/static_feature_mapping/core.py` 统一生成：

| 上层参数 | 算子规模影响 |
|---|---|
| `arch_embedding_size` | 映射到 8 个 embedding table 的行数；单值输入会扩展为 8 个表 |
| `batch_size * num_indices_per_lookup` | 形成每个 embedding branch 的 flattened indices 数量和 `Gather` lookup count |
| `arch_mlp_bot` | 生成 bottom MLP 的 `Gemm` 输入/输出维度和 M/N/K |
| interaction 固定结构 | 1 个 dense branch + 8 个 embedding branch 形成 9 路输入、`9x9` interaction matrix 和 36 个 pair features |
| `arch_mlp_top` | 生成 top MLP 的 `Gemm` 输入/输出维度和 M/N/K |
| `num_threads`, `inter_threads` | 写入静态特征行，用于线程尺度和 E2E 调度口径 |

独立 demo `standalone_demos/dlrm_operator_scale_mapping/dlrm_operator_scale_mapping_demo.py` 可以只导出映射，不加载模型 bundle，不做 latency 预测。它显式定义完整 76 节点结构，只依赖 Python 标准库，输出完整 76 节点 shape 表、完整 76 节点特征表、51 节点模型输入表，以及面向审阅的算子规模 summary。静态映射行中的 `has_cpu_profile=0`、`shape_source=static_config` 表示该阶段不读取 profiling 数据、shape CSV 或 timeline CSV。

## 9. 趋势图机制

multiworkload 硬件趋势图使用 synthetic controlled workload，不要求每张图固定同一个 DLRM 配置。该设计用于让不同硬件参数落在各自敏感区间。

趋势图路径：

```text
Synthetic workload
  -> full_graph_76 operator frame
  -> mainline v3 route
  -> DAG schedule
  -> scheduled E2E latency
```

每个趋势图包含：

```text
Device_1 x {small, base, large}
Device_2 x {small, base, large}
```

cache size 图会为 L1/L2/L3 构造不同 working set 规模，使 workload 落在目标 cache level 边界附近。固定全局 DLRM-S/M/L 图仅作为补充诊断，不作为当前主线趋势证据。

趋势图最终会对 E2E 聚合曲线做单调投影：

| 参数类型 | 预期方向 |
|---|---|
| bandwidth、SIMD width、cache size 等 capacity 参数增大 | E2E latency 不上升 |
| cache latency、DRAM latency、launch overhead 等 cost 参数增大 | E2E latency 不下降 |

CSV 中保留投影前后结果：

```text
raw_pred_us
pred_us
aggregate_projection_delta_us
```

## 10. 硬件敏感性解释

趋势后处理使用 synthetic sweep points，推荐三个指标：

| 指标 | 用途 |
|---|---|
| local directional Spearman | 检查方向是否符合物理预期 |
| absolute log-log elasticity | 比较单位相对变化对 E2E latency 的影响强度 |
| max change pct | 展示当前 sweep 范围内可观察变化幅度 |

absolute log-log elasticity：

```text
E(p) = |(log T_E2E(p2) - log T_E2E(p1)) / (log p2 - log p1)|
```

该指标无量纲。数值越大，表示硬件参数变化 1% 对 E2E latency 的相对影响越强。

max change pct：

```text
change_pct = (max(T_E2E) - min(T_E2E)) / min(T_E2E) * 100%
```

这两个指标都是 controlled sweep 下的 local sensitivity，不是真实生产 workload 的全局因果归因。

## 11. DLRM 访存主导与计算参数敏感性

当前 held-out DLRM 负载中，真实算子耗时和 Analytical service share 都以 memory-related 路径为主：

| 口径 | Device_1 | Device_2 |
|---|---:|---:|
| memory-related true operator time share | 97.7% | 97.7% |
| compute-related true operator time share | 2.3% | 2.3% |
| Analytical memory service share | 96.3% | 92.5% |
| Analytical compute service share | 3.7% | 7.5% |

计算/流水线参数仍可能影响 E2E latency，原因是局部 critical path 中的 dense MLP、ReduceSum、dispatch 或 latency tail 会影响最终调度结果。该结论不表示 DLRM 变成计算瓶颈。

## 12. 成本优化边界

成本优化不重新训练 RF，不重新拟合 op-thread scale。它加载两个已导出的 `.pkl` bundle，并对候选硬件做 `override_hw_config` 推理。

`reference_v2` 成本函数来自 `docs/参数约束v2.docx` 的 x1-x25 两点参考成本，配置固化在：

```text
calibrator/configs/hardware_cost_constraints_v2.json
```

成本优化结果用于候选排序和可解释分析，不替代真实 BOM 或采购报价。没有 RF bundle 时，prior-only fallback 只能用于流程检查。

## 13. 结论边界

- CPU 主线指标只来自 correct-hw full test。
- small split 结果只作历史快速验证参考。
- few-shot 泛化固定 source RF，只拟合 target 少量校准层，不覆盖 full 主线指标。
- CPU+NPU experimental 是误差替换和趋势验证，不代表真实 NPU 加速收益。
- static dense trend 没有真实标签，不能替代 MAPE/P90 精度验证。
