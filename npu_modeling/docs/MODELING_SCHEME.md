# 建模方案摘要

## 算子分类

特征层保留 `7` 个 physical families，用于描述算子的底层执行行为：

| Physical family | 覆盖算子 | 主要性能依据 |
|---|---|---|
| `DenseGemm` | `MatMulV2`, `BatchMatMulV2`, `Linear`, `GemmBias` | cube/GEMM compute、tile alignment、arithmetic intensity。 |
| `VectorStreaming` | elementwise 和 normalization 类算子 | vector compute 和 streaming memory。 |
| `AxisCompute` | `ReduceMean`, `ReduceSum`, `Softmax`, `TopK`, `Sort` | axis length、outer size、reduction/selection 行为。 |
| `DataMovement` | `Transpose`, `ConcatD`, `Tile`, `PadV3`, `MemSet` | 数据搬移字节数、layout/stride penalty、alignment。 |
| `SparseIndex` | `Gather`, `Index`, `ScatterAdd`, `EmbeddingBag`, `AdvancedIndexing` | index density、payload、不规则访存。 |
| `SpatialWindow` | `MaxPool2D` | window reuse 和 spatial scan pattern。 |
| `DataDependent` | `NonZero` | 输出规模依赖实际数据分布。 |

最终训练策略使用 `4` 个 compact model groups，

| Model group |
|---|
| `DataDependent` |
| `MovementSelectionWindow` |
| `StreamingReduction` |
| `StructuredComputeMemory` |

模型组映射如下：

| Model group | 覆盖算子 |
|---|---|
| `DataDependent` | `NonZero` |
| `MovementSelectionWindow` | `Transpose`, `ConcatD`, `Tile`, `PadV3`, `MemSet`, `TopK`, `Sort`, `MaxPool2D` |
| `StreamingReduction` | elementwise、normalization、`Softmax`, `ReduceMean`, `ReduceSum` |
| `StructuredComputeMemory` | GEMM 类算子和 sparse/index 类算子 |

## 分层访存特征

训练脚本根据 `models/hardware_descriptor.json` 添加 L0/L1/L2/HBM 相关特征：

| 特征组 | 示例 |
|---|---|
| 容量压力 | `l0_pressure`, `l1_pressure`, `l2_pressure`, `fits_l0`, `fits_l1`, `fits_l2` |
| 带宽下界 | `t_l1_us`, `t_l2_us`, `t_hbm_us`, `t_mem_hier_us` |
| 物理 roofline | `t_compute_physical_us`, `t_roofline_hier_us` |
| 访问模式修正 | `effective_memory_bytes`, `hier_arithmetic_intensity`, `bytes_per_output_elem` |

L0 按容量压力和复用压力建模，不作为主要带宽瓶颈，因为当前假设中 L0 带宽很高。L1 和 L2 使用显式带宽下界项建模。

## 验收口径

最终 strict compact 结果使用 stable-label aggregation 和 train-only supplement data：

| 验收项 | 结果 |
|---|---:|
| Overall `MAPE@1us-floor <= 10%` | 8.1909% |
| Overall `p90@1us-floor <= 20%` | 17.2121% |



## DLRM 验证

端到端 DLRM 验证使用交付目录内置的轻量级多配置 profiles：

`input_data/dlrm_profiles/results_20260426_135144`

覆盖范围如下：

| 维度 | 数值 |
|---|---:|
| Profiled DLRM cases | 1407 |
| Architecture cases | 4 |
| Batch sizes | 33 |
| `num_indices_per_lookup` values | 11 |

NPU-side 结果如下：

| 指标 | Mean | P50 | P90 |
|---|---:|---:|---:|
| NPU compute replacement APE | 6.7569% | 6.2503% | 13.4940% |

该验证口径保持 CPU、通信、同步、transfer 和未建模 NPU 时间为 measured，仅替换已建模的 NPU compute operator invocations。
