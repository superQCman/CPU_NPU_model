# NPU模型

本目录包含最终交付的 NPU模型。

训练输入、语义特征构建器、轻量级 DLRM 验证 profiles 均已内置在本目录中。脚本默认路径会根据交付目录位置自动解析，不依赖本机绝对路径或仓库外工程目录。

## 交付范围

- 建模特征包括Cube/Vector峰值算力，各级访存容量以及访存带宽
- 使用 L0/L1/L2/HBM 层次访存特征描述容量压力、复用压力和带宽下界。
- 使用 4 个最终 compact model groups

## 最终指标

| 指标 | 数值 |
|---|---:|
| Stable validation rows | 8666 |
| Holdout rows | 1734 |
| Train rows including train-only supplements | 10171 |
| Overall MAPE@1us-floor | 8.1909% |
| Overall p90@1us-floor | 17.2121% |

最终模型组为 `DataDependent`、`MovementSelectionWindow`、`StreamingReduction` 和 `StructuredComputeMemory`。

## DLRM 端到端验证

多配置 DLRM held-out 验证覆盖 `1407` 个 profiled cases，包含 `4` 个 architecture cases、`33` 个 batch sizes 和 `11` 个 `num_indices_per_lookup` 取值。

| 指标 | Mean | P50 | P90 |
|---|---:|---:|---:|
| NPU compute replacement APE | 6.7569% | 6.2503% | 13.4940% |
| Full stage replacement APE | 0.000442% | 0.000367% | 0.000886% |



## 命令

从本目录根路径执行：

```bash
python code/train_physical_memory_models.py \
  --strategy physical_family \
  --grouping strict_compact \
  --expert-ops "" \
  --seed 20260505
```



## 关键文件

| 路径 | 说明 |
|---|---|
| `code/train_physical_memory_models.py` | 模型训练、特征构造、验证、指标输出和 artifact 导出脚本。 |
| `code/build_semantic_operator_dataset.py` | 交付内置语义特征构建器，用于 supplemental core labels 和 DLRM 验证。 |
| `input_data/semantic_operator_dataset.csv` | 交付内置训练数据集。 |
| `input_data/dlrm_profiles/` | DLRM 验证使用的轻量级 profile 输入。 |
| `models/hardware_descriptor.json` | L0/L1/L2/HBM 参数和计算峰值的硬件描述文件。 |
| `results/physical_memory_models.joblib` | 最终模型 artifact。 |
| `results/physical_memory_overall_metrics.csv` | 整体验收指标。 |
| `results/physical_memory_metrics_by_family.csv` | 最终模型组指标。 |
| `results/physical_memory_metrics_by_op.csv` | 按算子统计的工程诊断指标，不属于交付验收报告。 |
| `results/dlrm_end_to_end/` | 多配置 DLRM NPU compute replacement 验证结果。 |
| `results/train_supplements/` | 仅用于训练侧补充的数据和 targeted cases。 |
| `results/PHYSICAL_MEMORY_MODEL_REPORT.md` | 交付验收报告。 |
| `docs/MODELING_SCHEME.md` | 模型分组和访存建模方案说明。 |
| `docs/USER_GUIDE.md` | 使用说明、复现流程、输入输出 |
| `docs/MODELING_PRINCIPLES.md` | 建模原理、特征体系和验证方法说明 |
