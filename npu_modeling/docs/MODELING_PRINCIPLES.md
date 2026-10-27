# NPU模型原理说明

## 1. 文档目的

本文档说明 NPU内部建模原理，包括问题定义、算子分类、分层访存特征、模型分组、训练验证流程、验收指标以及 DLRM 端到端验证方法。

## 2. 建模目标

该模型用于预测语义算子在 NPU 上的 compute latency。

预测目标字段为：

`target_kernel_compute_us`

该字段表示经过语义算子数据构建流程规范化后的 NPU kernel compute duration，单位为微秒。

模型主要服务于 operator-level NPU compute latency 估计和下游应用级 what-if 分析。它不直接建模 CPU overhead、host scheduling、通信、同步、数据传输或完整框架 runtime overhead。

## 3. 数据流概览

训练数据流如下：

```text
semantic_operator_dataset.csv
  -> taxonomy assignment
  -> hierarchical memory feature construction
  -> stable-label aggregation
  -> train/test split on stable rows
  -> append train-only supplements to train split
  -> train one model per final model group
  -> evaluate on original stable holdout
  -> export metrics, predictions, report, and joblib artifact
```

DLRM 验证数据流如下：

```text
DLRM profiler operator_details.csv
  -> map supported ATen ops to semantic operators
  -> reconstruct semantic params from input shapes
  -> build semantic feature records
  -> apply same taxonomy and memory features
  -> predict modeled NPU compute latency
  -> replace measured modeled NPU compute time
  -> report case-level and overall error
```

## 4. 算子分类体系

### 4.1 Physical Family

第一层分类是 physical family，即根据算子的主要执行行为进行物理类别归纳。

| Physical Family | 覆盖算子 | 建模依据 |
|---|---|---|
| `DenseGemm` | `MatMulV2`, `BatchMatMulV2`, `Linear`, `GemmBias` | 稠密矩阵/张量计算，受 cube compute、tile shape 和访存量共同影响。 |
| `VectorStreaming` | `Add`, `Mul`, `Relu`, `Sigmoid`, `Gelu`, `Tanh`, `Cast`, `Where`, `LayerNorm`, `BatchNorm1D` | elementwise 或 normalization 类 streaming/vector 算子。 |
| `AxisCompute` | `ReduceMean`, `ReduceSum`, `Softmax`, `TopK`, `Sort` | 与 axis size、outer size、selection/reduction pattern 强相关。 |
| `DataMovement` | `Transpose`, `ConcatD`, `Tile`, `PadV3`, `MemSet` | 主要由数据搬移量、layout 变化和写入模式决定。 |
| `SparseIndex` | `Gather`, `Index`, `ScatterAdd`, `EmbeddingBag`, `AdvancedIndexing` | 具有索引驱动或非连续访问特征，局部性弱于规则 streaming。 |
| `SpatialWindow` | `MaxPool2D` | 窗口访问模式，受 kernel、stride、channel layout 和复用影响。 |
| `DataDependent` | `NonZero` | 输出规模和 latency 与实际数据值相关，仅靠 shape 难以完全确定。 |

### 4.2 最终 Compact Model Groups

交付模型没有采用每个 physical family 一个模型，也没有采用每个算子一个模型，而是采用 `4` 个 compact model groups。该设计在物理相似性、样本量和验收精度之间取得平衡。

| 最终模型组 | 包含内容 | 设计原因 |
|---|---|---|
| `StructuredComputeMemory` | Dense GEMM 类算子和 sparse/index 类算子 | 两类算子都需要考虑结构化 shape、访存 footprint、计算强度和 access pattern。通过物理特征区分 dense 与 sparse 行为，同时共享样本提升泛化。 |
| `StreamingReduction` | vector streaming 和 reduction 类算子 | 共享 streaming/bandwidth 特征，reduction 相关差异由 axis features 进一步区分。 |
| `MovementSelectionWindow` | data movement、selection 和 spatial window 类算子 | 这些算子更多由数据搬移模式、选择模式、窗口复用和 layout 变化决定，而不是纯计算吞吐决定。 |
| `DataDependent` | `NonZero` | 数据相关性强，单独建模可以避免污染其他稳定 shape-driven 算子。 |

该映射在 `train_physical_memory_models.py` 中由 `STRICT_COMPACT_GROUP_BY_OP` 定义。

## 5. 分层访存建模

### 5.1 建模动机

原始语义特征已经包含 shape、估算 FLOPs、估算 bytes 和部分参数信息。但对于 NPU latency，仅有 FLOPs 或总 bytes 不足以解释性能差异。

两个算子即使 FLOPs 或 bytes 接近，也可能因为 working set 是否进入 L0/L1/L2、数据是否规则访问、是否存在 transpose/sparse penalty、是否有 axis tail 等因素产生明显 latency 差异。因此模型引入显式的 hierarchical memory features。

### 5.2 硬件层级处理方式

| 层级 | 建模处理方式 |
|---|---|
| L0 | 建模为容量压力和复用压力。由于 L0 带宽很高，不把 L0 作为主要带宽瓶颈项。 |
| L1 | 建模容量压力和带宽下界时间。默认容量 `1.5MB/core`，带宽 `2TB/s`。 |
| L2 | 建模容量压力和带宽下界时间。默认容量 `32MB`，带宽 `1TB/s`。 |
| HBM | 建模 effective HBM bytes 和 HBM 带宽下界时间。默认带宽 `1TB/s`。 |

### 5.3 核心访存特征

基于 `bytes_rw_estimate`、`working_set_bytes`、`flops_estimate`、`dtype_bytes` 和 `output_elems_estimate` 等语义估计量，模型派生如下特征：

| 特征 | 含义 |
|---|---|
| `l0_pressure` | `working_set_bytes / l0_capacity_bytes` |
| `l1_pressure` | `working_set_bytes / l1_capacity_bytes` |
| `l2_pressure` | `working_set_bytes / l2_capacity_bytes` |
| `fits_l0` | working set 是否可放入 L0。 |
| `fits_l1` | working set 是否可放入 L1。 |
| `fits_l2` | working set 是否可放入 L2。 |
| `effective_memory_bytes` | 经过 access-pattern penalty 修正后的有效访存字节数。 |
| `estimated_l1_bytes` | 估算的 L1 层级服务字节数。 |
| `estimated_l2_bytes` | 估算的 L2 层级服务字节数。 |
| `estimated_hbm_bytes` | 估算的 HBM 服务字节数。 |
| `t_l1_us` | 基于 L1 bandwidth 的 latency 下界。 |
| `t_l2_us` | 基于 L2 bandwidth 的 latency 下界。 |
| `t_hbm_us` | 基于 HBM bandwidth 的 latency 下界。 |
| `t_mem_hier_us` | L1/L2/HBM 三个下界中的最大值。 |
| `t_compute_physical_us` | 基于 cube/vector peak flops 的计算下界。 |
| `t_roofline_hier_us` | 计算下界和访存下界的最大值。 |
| `hier_arithmetic_intensity` | `flops_estimate / effective_memory_bytes`。 |

对高动态范围特征，脚本还会添加 `log1p_*` 变换形式，例如 `log1p_t_mem_hier_us`、`log1p_effective_memory_bytes` 和 `log1p_hier_arithmetic_intensity`。

### 5.4 Effective Bytes 修正

模型不直接把 raw bytes 当作最终访存量，而是根据访问模式修正为 `effective_memory_bytes`。

| 修正项 | 触发条件 | 作用 |
|---|---|---|
| sparse penalty | `SparseIndex` family | 增加有效访存量，反映不规则访问和较弱局部性。 |
| transpose penalty | `Transpose` op | 根据 permutation complexity 增加有效访存量。 |
| axis penalty | `AxisCompute` family | 对小 axis 或较难并行的 axis pattern 施加惩罚。 |
| movement penalty | `DataMovement` family | 表征简单 bytes 之外的搬移和 layout overhead。 |

这种设计保留了可解释性：模型不仅看到原始语义特征，也看到具有物理意义的访存下界和压力指标。

## 6. 算子专项物理特征

除通用访存层级特征外，模型还针对常见物理模式构造专项特征。

### 6.1 Axis/Reduction/Selection 特征

对 `Softmax`、`ReduceMean`、`ReduceSum`、`Sort`、`TopK` 等算子，模型派生：

| 特征 | 含义 |
|---|---|
| `axis_total_elems` | `d0 * d1 * d2` 的总元素数。 |
| `axis_size_physical` | reduction/selection 轴长度。 |
| `axis_outer_physical` | axis 之外的元素规模。 |
| `axis_bytes` | axis 对应字节数。 |
| `axis_outer_bytes` | outer 对应字节数。 |
| `axis_is_inner/middle/outer` | axis 位置指示器，用于表征 locality 差异。 |
| `axis_size_mod_8/16/32` | alignment/tail 指示器。 |
| `axis_sort_work` | 近似排序工作量 `outer * axis * log2(axis)`。 |
| `axis_softmax_work` | 近似 softmax 多 pass 工作量。 |

这些特征用于区分总元素数相近但 axis locality 和 reduction complexity 不同的情况。

### 6.2 Spatial Window 特征

对 `MaxPool2D`，模型派生：

| 特征 | 含义 |
|---|---|
| `pool_out_h`, `pool_out_w` | 估算输出空间尺寸。 |
| `pool_output_elems_physical` | 估算输出元素数。 |
| `pool_input_bytes` | 输入 tensor bytes。 |
| `pool_output_bytes` | 输出 tensor bytes。 |
| `pool_overlap_ratio` | kernel/stride 近似重叠比例。 |
| `pool_reuse_work` | 包含 kernel area 的窗口访问工作量。 |
| `pool_channels_mod_32` | channel alignment 指示器。 |

这些特征用于描述窗口复用、stride overlap 和 channel/tile 对齐影响。

### 6.3 GEMM Tile/Tail 特征

对 dense matrix 类算子，模型基于 16 元素 tile 粒度构造：

| 特征 | 含义 |
|---|---|
| `gemm_m_tiles_16` | M 维 tile 数量。 |
| `gemm_k_tiles_16` | K 维 tile 数量。 |
| `gemm_n_tiles_16` | N 维 tile 数量。 |
| `gemm_tile_volume_16` | M/K/N tile 数量乘积。 |
| `gemm_m_tail_16` | M 维 tail。 |
| `gemm_k_tail_16` | K 维 tail。 |
| `gemm_n_tail_16` | N 维 tail。 |

这些特征能够提升模型对小矩阵和非对齐 tail regime 的敏感度，因为这类场景通常无法达到理想峰值吞吐。



## 7. 模型训练

### 7.1 特征矩阵构造

模型输入包含：

| 特征类型 | 示例 |
|---|---|
| 原始语义数值特征 | FLOPs、bytes、output elements、shape-derived params。 |
| 分层访存特征 | L0/L1/L2 pressure、estimated L1/L2/HBM bytes、bandwidth lower-bound times。 |
| 算子专项物理特征 | axis、pooling、GEMM tile/tail features。 |
| 类别 one-hot 特征 | `op_type`, `dtype`, `physical_family`, `residual_group`, `model_group`。 |

类别特征通过 one-hot 编码处理。推理阶段必须按照每个模型组 artifact 中保存的 `feature_columns` 对齐列顺序，以保证训练和推理输入一致。

### 7.2 候选模型

每个模型组会评估多个候选 estimator：

| 候选模型 | 作用 |
|---|---|
| `ExtraTreesRegressor` | 对非线性 tabular 特征表现稳定，调参负担较低。 |
| `ExtraTreesRegressor` smooth variant | 使用更大的 leaf size，降低过拟合风险。 |
| `RandomForestRegressor` | 稳健的 bagging ensemble baseline。 |
| `HistGradientBoostingRegressor` on `log1p(target)` | 适合正 latency target 的平滑 boosted model。 |

最终每个模型组选择的 estimator 如下：

| 模型组 | 最终 estimator |
|---|---|
| `DataDependent` | `extra_trees_smooth` |
| `MovementSelectionWindow` | `random_forest` |
| `StreamingReduction` | `extra_trees` |
| `StructuredComputeMemory` | `extra_trees` |

### 7.3 模型选择目标

每个候选模型在该模型组对应的 holdout subset 上计算指标，选择分数为：

`MAPE@1us-floor + p90_weight * p90@1us-floor`

最终 compact group 训练中 `p90_weight=0.25`。

该目标既关注平均相对误差，也显式惩罚尾部误差。

### 7.4  预测约束

latency 不可能为负，因此所有预测值会裁剪为非负：

`pred_compute_us = max(pred_compute_us, 0)`

## 8. 评估指标

### 8.1 APE@1us-floor

绝对百分比误差定义为：

```text
APE@1us-floor = abs(pred - target) / max(abs(target), 1.0) * 100
```

使用 1us denominator floor 是为了避免极小 kernel 因分母过小而在百分比指标中产生不合理放大。

### 8.2 报告指标

| 指标 | 含义 |
|---|---|
| `MAE` | 平均绝对误差，单位微秒。 |
| `MAPE@1us-floor` | 使用 1us denominator floor 的平均 APE。 |
| `WMAPE` | 绝对误差和除以 target 绝对值和。 |
| `p50@1us-floor` | APE 中位数。 |
| `p90@1us-floor` | APE 90 分位数。 |



## 9. 模型结果

### 9.1 整体结果

| 指标 | 数值 |
|---|---:|
| Stable validation rows | `8666` |
| Holdout rows | `1734` |
| Model count | `4` |
| Overall MAPE@1us-floor | `8.1909%` |
| Overall p90@1us-floor | `17.2121%` |

### 9.2 每个模型组结果

| 模型组 | MAPE@1us-floor |
|---|---:|
| `DataDependent` | `3.4359%` |
| `MovementSelectionWindow` | `7.1184%` |
| `StreamingReduction` | `7.9468%` |
| `StructuredComputeMemory` | `9.1546%` |

## 10. DLRM 端到端验证

operator-level holdout 指标验证模型在语义算子配置上的精度。应用级验证用于确认模型在真实 DLRM profile trace 中仍具备可用性，因为真实应用包含重复 invocation、应用特定 shape、已建模和未建模 runtime 组件的混合。

### 10.1 Replacement 公式

对每个 DLRM case，验证脚本计算：

```text
modeled_real = sum(measured device time for modeled operator invocations)
modeled_pred = sum(model predicted compute time for modeled operator invocations)

npu_compute_pred_replace = max(npu_compute_real - modeled_real, 0) + modeled_pred
stage_pred_replace       = max(stage_real - modeled_real, 0) + modeled_pred
```

即仅替换已建模 NPU compute operator 部分，其余 measured 时间保持不变。

### 10.2 DLRM 指标

| 指标 | 含义 |
|---|---|
| `npu_compute_replace_modeled_ape_percent` | 在 total NPU compute time 中替换 modeled operator time 后的误差，是主要 NPU-side 应用指标。 |
| `stage_replace_modeled_ape_percent` | 在 full stage time 中替换 modeled operator time 后的误差。由于CPU侧、通信延时占主导，该指标通常非常小。 |

### 10.3 DLRM 最终结果

最终 DLRM 验证覆盖 `1407` 个 case，包含 `4` 个 architecture case、`33` 个 batch size 和 `11` 个 `num_indices_per_lookup`。

| 指标 | Mean | P50 | P90 |
|---|---:|---:|---:|
| NPU compute replacement APE | `6.7569%` | `6.2503%` | `13.4940%` |
| Full stage replacement APE | `0.000442%` | `0.000367%` | `0.000886%` |

