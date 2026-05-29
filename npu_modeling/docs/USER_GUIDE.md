# NPU 模型使用说明

## 1. 文档目的

本文档说明 NPU 物理访存模型交付件的使用方法、复现实验流程、输入输出文件、模型产物消费方式，该目录仅包含的 NPU 建模代码、模型、结果和文档。

## 2. 目录结构

交付目录中的代码、文档和结果文件均使用相对路径描述输入输出。

| 路径 | 内容说明 |
|---|---|
| `code/` | 可执行脚本，包括模型训练、targeted case 生成、DLRM 端到端验证。 |
| `input_data/` | 交付内置输入数据，包括训练数据集和轻量级 DLRM profile 输入。 |
| `models/` | 本次建模使用的 NPU 硬件描述文件。 |
| `results/` | 最终模型产物、预测明细、指标、报告、图表和 DLRM 验证结果。 |
| `results/train_supplements/` | 仅用于训练侧补充的数据，用于改善尾部误差，不进入原始 stable holdout。 |
| `results/figures/` | 模型组误差箱线图，包含 PNG/PDF/SVG 和统计摘要 CSV。 |
| `results/dlrm_end_to_end/` | 多配置 DLRM 端到端 NPU compute replacement 验证结果。 |
| `docs/` | 使用说明、建模方案和内部原理文档。 |

## 3. 运行环境

所有命令建议从交付目录根路径执行，本文档中的命令均按交付目录内的相对路径书写。

Python 依赖如下：

| 依赖包 | 用途 |
|---|---|
| `numpy` | 数值特征构造和指标计算。 |
| `pandas` | CSV 读写、聚合、统计和报告生成。 |
| `scikit-learn` | 树模型和梯度提升模型训练。 |
| `joblib` | 模型 artifact 序列化和加载。 |
| `matplotlib` | 误差箱线图生成。 |

## 4. 主要代码文件

| 文件 | 作用 |
|---|---|
| `code/train_physical_memory_models.py` | 主训练脚本，负责特征构造、模型训练、验证、指标输出、报告生成和模型 artifact 导出。 |
| `code/build_semantic_operator_dataset.py` | 语义特征构建器，用于把 supplemental core labels 和 DLRM profile invocation 转换为统一语义特征。 |
| `code/generate_strict_targeted_cases.py` | 生成面向尾部误差优化的 targeted NPU operator cases。 |
| `code/make_error_boxplot.py` | 根据 holdout 预测结果生成按模型组划分的 APE 箱线图。 |
| `code/validate_dlrm_end_to_end.py` | 使用训练好的模型，对多配置 DLRM profile 执行 NPU compute replacement 验证。 |

## 5. 硬件描述文件

硬件参数文件路径：

`models/hardware_descriptor.json`

当前参数如下：

| 字段 | 当前值 | 含义 |
|---|---:|---|
| `peak_vector_flops` | `1.8e13` | 向量/streaming 类算子的峰值计算吞吐。 |
| `peak_cube_flops` | `2.95e14` | GEMM/矩阵类算子的 cube 峰值计算吞吐。 |
| `memory_capacity_bytes` | `68719476736` | HBM 容量。 |
| `l0_capacity_bytes` | `65536` | L0 容量。本模型中 L0 主要建模容量压力和复用压力，不将其作为主要带宽瓶颈。 |
| `l1_capacity_bytes` | `1572864` | 单核 L1 容量，默认约 `1.5MB/core`。 |
| `l2_capacity_bytes` | `33554432` | L2 容量，默认 `32MB`。 |
| `l1_bandwidth_bytes_per_s` | `2.0e12` | L1 带宽下界项。 |
| `l2_bandwidth_bytes_per_s` | `1.0e12` | L2 带宽下界项。 |
| `memory_bandwidth_bytes_per_s` | `1.0e12` | HBM 带宽下界项。 |

如果目标硬件发生变化，应先更新该 JSON 文件，再重新训练模型。不建议在硬件参数已变更的情况下直接复用已有的 `physical_memory_models.joblib`，除非明确是在做消融实验或敏感性分析。

## 6. 模型训练脚本

### 6.1 脚本目标

`train_physical_memory_models.py` 用于训练 compact model groups，在 stable-label holdout 上评估模型，导出指标和训练后的模型 artifact。

交付模型使用 `4` 个模型组：

| 模型组 | 覆盖算子 |
|---|---|
| `StructuredComputeMemory` | `MatMulV2`, `BatchMatMulV2`, `Linear`, `GemmBias`, `Gather`, `Index`, `ScatterAdd`, `EmbeddingBag`, `AdvancedIndexing` |
| `StreamingReduction` | `Add`, `Mul`, `Relu`, `Sigmoid`, `Gelu`, `Tanh`, `Cast`, `Where`, `LayerNorm`, `BatchNorm1D`, `Softmax`, `ReduceMean`, `ReduceSum` |
| `MovementSelectionWindow` | `Transpose`, `ConcatD`, `Tile`, `PadV3`, `MemSet`, `TopK`, `Sort`, `MaxPool2D` |
| `DataDependent` | `NonZero` |

### 6.2 训练复现命令

从交付目录根路径执行：

```bash
python code/train_physical_memory_models.py \
  --strategy physical_family \
  --grouping strict_compact \
  --expert-ops "" \
  --seed 20260505
```

预期输出应包含 `OK`，指标应接近：

```text
[OK] strategy=physical_family models=4 mape=8.1909% p90=17.2121% ...
```

### 6.3 关键参数说明

| 参数 | 是否必需 | 默认值 | 说明 |
|---|---|---|---|
| `--dataset` | 否 | `input_data/semantic_operator_dataset.csv` | 主语义算子数据集，包含 NPU compute target 和语义特征。 |
| `--hardware` | 否 | `models/hardware_descriptor.json` | L0/L1/L2/HBM 和计算峰值参数。 |
| `--output-dir` | 否 | `results/` | 输出模型、预测、指标和报告的目录。 |
| `--seed` | 否 | `20260522` | train/test split 和候选模型随机种子。交付复现使用 `20260505`。 |
| `--mape-target` | 否 | `10.0` | MAPE@1us-floor 验收阈值。 |
| `--tail-target` | 否 | `20.0` | p90 APE@1us-floor 验收阈值。 |
| `--strategy` | 否 | `global_residual` | 交付复现使用 `physical_family`。 |
| `--grouping` | 否 | `physical_family` | 交付复现使用 `strict_compact`。 |
| `--extra-core-labels-train-only` | 否 | `results/train_supplements/*.csv` | stable split 后仅追加到训练侧的补充数据。默认已指向交付内置 supplement 文件。 |

### 6.4 训练输出文件

| 输出文件 | 说明 |
|---|---|
| `physical_memory_dataset_stable.csv` | stable-label 聚合后的数据集。 |
| `physical_memory_predictions.csv` | holdout 预测明细，包含 target、prediction、model group 和 APE。 |
| `physical_memory_overall_metrics.csv` | 整体 holdout 指标。 |
| `physical_memory_metrics_by_family.csv` | 最终模型组指标和验收标记。 |
| `physical_memory_metrics_by_op.csv` | 按算子类型统计的工程诊断指标。该文件用于内部分析，不放入交付验收报告。 |
| `physical_memory_residual_experts.csv` | residual/expert 使用情况。交付模型不使用 residual expert。 |
| `physical_memory_models.joblib` | 训练后的模型 artifact。 |
| `run_summary.json` | 机器可读的运行摘要，包括模型数量和验收状态。 |
| `PHYSICAL_MEMORY_MODEL_REPORT.md` | 自动生成的实验报告。 |

## 7. DLRM 端到端验证脚本

### 7.1 脚本目标

`validate_dlrm_end_to_end.py` 使用训练好的模型，在未参与训练的多配置 DLRM profiles 上进行端到端验证。

验证时只替换已建模的 NPU compute operator invocations。CPU 时间、通信、同步、transfer 和未建模 NPU 时间均保持 measured。该口径可以隔离 NPU compute 模型自身对端到端结果的影响。

### 7.2 命令

```bash
python code/validate_dlrm_end_to_end.py
```

预期输出：

```text
[OK] DLRM validation written to <delivery>/results/dlrm_end_to_end; cases=1407
```

### 7.3 支持的 ATen 到语义算子映射

| DLRM profile 名称 | 语义算子 |
|---|---|
| `aten::linear` | `MatMulV2` |
| `aten::bmm` | `BatchMatMulV2` |
| `aten::relu` | `Relu` |
| `aten::sigmoid` | `Sigmoid` |
| `aten::cat` | `ConcatD` |
| `aten::index` | `AdvancedIndexing` |

### 7.4 DLRM 输出文件

| 输出文件 | 说明 |
|---|---|
| `dlrm_case_summary.csv` | 每个 DLRM case 的 measured/predicted replacement 摘要。 |
| `dlrm_invocation_predictions.csv` | 每个已建模 operator invocation 的预测明细。 |
| `dlrm_overall_metrics.csv` | replacement metrics 的 mean、p50、p90。 |
| `dlrm_by_op_summary.csv` | 按语义算子聚合的 real/predicted sum。 |
| `DLRM_END_TO_END_REPORT.md` | DLRM 验证报告。 |

### 10.5 DLRM 最终验证结果

当前 DLRM 验证覆盖 `1407` 个 case，包含 `4` 个 architecture case、`33` 个 batch size 和 `11` 个 `num_indices_per_lookup`。

| 指标 | Mean | P50 | P90 |
|---|---:|---:|---:|
| NPU compute replacement APE | `6.7569%` | `6.2503%` | `13.4940%` |
| Full stage replacement APE | `0.000442%` | `0.000367%` | `0.000886%` |

主要引用指标为 `NPU compute replacement APE`。

## 11. 模型 artifact 使用方式

训练后的模型 artifact 路径为：

`results/physical_memory_models.joblib`

artifact 包含：

| 字段 | 含义 |
|---|---|
| `strategy` | 最终 grouping 策略，当前为 `strict_compact`。 |
| `models` | 每个最终模型组对应的模型包。 |
| `physical_family_by_op` | operator type 到 physical family 的映射。 |
| `strict_compact_group_by_op` | operator type 到最终 compact model group 的映射。 |
| `hardware` | 训练时使用的硬件描述。 |

每个模型包包含：

| 字段 | 含义 |
|---|---|
| `model` | 训练好的 scikit-learn estimator。 |
| `feature_columns` | 推理时需要对齐的 one-hot/numeric 特征列顺序。 |
| `model_choice` | 该模型组最终选择的 estimator 类型。 |

集成该 artifact 时，调用方必须构造相同的语义特征 schema，应用相同的 taxonomy，调用相同的 hierarchical memory feature 逻辑，执行 categorical one-hot encoding，并按 artifact 中保存的 `feature_columns` 对齐。

参考实现位于 `validate_dlrm_end_to_end.py` 中的 `predict_rows()` 和 `feature_frame()`。

## 12. 复现顺序

1. 确认 `models/hardware_descriptor.json` 与目标 NPU 假设一致。
2. 使用最终命令运行 `train_physical_memory_models.py`。
3. 检查 `run_summary.json` 和 `physical_memory_metrics_by_family.csv`，确认验收标记为通过。
5. 运行 `validate_dlrm_end_to_end.py`，刷新 DLRM 验证结果。
6. 审阅 `PHYSICAL_MEMORY_MODEL_REPORT.md` 和 `results/dlrm_end_to_end/DLRM_END_TO_END_REPORT.md`。

