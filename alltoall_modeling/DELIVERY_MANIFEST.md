# All-to-All 建模交付清单

## 1. 交付范围

本目录为 All-to-All 通信建模最终交付包，包含：

- 代码文件：单算子带宽/时延建模、异步掩盖建模、DLRM profiler 验证脚本。
- 使用文档：如何采集、训练、验证、复现和调用模型。
- 原理文档：All-to-All 单算子建模原理与异步通信掩盖建模原理。
- 模型参数：JSON 格式保存的带宽、时延、异步 overlap 参数。
- 验证结果：CSV、Markdown 报告和 PNG 图表。

## 2. 交付入口

| 文件 | 说明 |
|---|---|
| `README.md` | 总入口，快速说明目录和核心命令 |
| `docs/USER_GUIDE.md` | 面向使用者的完整操作说明 |
| `docs/MODELING_PRINCIPLES.md` | 建模原理说明 |
| `docs/CODE_INVENTORY.md` | 代码、脚本、模型、结果 |
| `docs/modeling_scheme.md` | 方案级说明，包含单算子和异步验证结果摘要 |

## 3. 核心代码文件

| 文件 | 功能 |
|---|---|
| `src/bw_model.py` | All-to-All 带宽模型，按 128KB/32KB/非对齐三类建模 |
| `src/latency_model.py` | All-to-All 时延模型，三段模型加残差样条修正 |
| `src/async_overlap_model.py` | 异步掩盖模型，提供预测和 eta 反推函数 |
| `src/memory_bw_utils.py` | 公共数学函数与通信带宽换算 |
| `scripts/run_model_pipeline.py` | 单算子模型训练、参数保存和验证总流程 |
| `scripts/run_async_overlap_validation.py` | 基于单算子验证结果的异步 replay 验证 |
| `scripts/run_dlrm_async_overlap_validation.py` | DLRM 单 trace 实测异步验证 |
| `scripts/run_dlrm_async_overlap_multiconfig_validation.py` | DLRM case sweep 实测异步验证 |
| `scripts/collect_train_hccl_a2a_npu8.sh` | 训练数据采集命令生成 |
| `scripts/collect_test_hccl_a2a_npu8.sh` | 外部测试数据采集命令生成 |

## 4. 数据与模型参数

| 文件 | 说明 |
|---|---|
| `data/raw/a2a_32kb_aligned.txt` | 当前基线 HCCL-test All-to-All 训练数据 |
| `data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT/step_trace_time.csv` | DLRM 单 trace 异步验证 step trace |
| `data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT/operator_details.csv` | DLRM 单 trace marker 验证数据 |
| `data/profiler/dlrm_sweep/sweep_app_summary.csv` | DLRM 多配置 sweep 元信息 |
| `data/profiler/dlrm_sweep/case_*/profiling/rank0/*/ASCEND_PROFILER_OUTPUT/step_trace_time.csv` | DLRM 多配置 step trace |
| `artifacts/models/bw_model_params.json` | 带宽模型参数 |
| `artifacts/models/latency_model_params.json` | 时延模型参数 |
| `artifacts/models/async_overlap_model_params.json` | replay 异步掩盖参数 |
| `artifacts/models/dlrm_async_overlap_model_params.json` | DLRM 单 trace overlap 校准参数 |
| `artifacts/models/dlrm_async_overlap_multiconfig_model_params.json` | DLRM 多配置 overlap 校准参数 |

## 5. 验证结果

| 文件 | 说明 |
|---|---|
| `artifacts/reports/validation_summary.md` | All-to-All 单算子 size-holdout 验证摘要 |
| `artifacts/reports/validation_metrics.csv` | 单算子逐点预测明细 |
| `artifacts/reports/coverage_summary.csv` | train/test 覆盖与泄漏检查 |
| `artifacts/reports/async_overlap_summary.md` | 异步 replay 验证摘要 |
| `artifacts/reports/dlrm_async_overlap_summary.md` | DLRM 单 trace 实测异步验证摘要 |
| `artifacts/reports/dlrm_async_overlap_multiconfig_steps.csv` | DLRM 多配置逐 step bounded-overlap 验证明细 |
| `artifacts/reports/dlrm_async_overlap_multiconfig_cases.csv` | DLRM 多配置按 case bounded-overlap 汇总 |
| `artifacts/reports/dlrm_async_overlap_multiconfig_by_variant.csv` | DLRM 多配置按模型配置汇总 |
| `artifacts/reports/dlrm_async_overlap_multiconfig_summary.md` | DLRM 多配置实测异步验证摘要 |
| `artifacts/plots/bw_model_fit.png` | 带宽拟合图 |
| `artifacts/plots/latency_model_fit.png` | 时延拟合图 |
| `artifacts/plots/latency_compare.png` | 单算子预测对比图 |
| `artifacts/plots/async_overlap_validation.png` | replay 异步验证图 |
| `artifacts/plots/dlrm_async_overlap_multiconfig_validation.png` | DLRM 多配置实测异步验证图 |

## 6. 当前验收指标

### All-to-All 单算子时延模型

| 指标 | 当前值 |
|---|---:|
| 验证方式 | size-holdout |
| Overall MAPE | 0.712% |
| Overall RMSE | 30.990 us |
| Relative error P50/P90/P95 | 0.345% / 1.596% / 3.078% |

### 异步 replay 验证

| 指标 | 当前值 |
|---|---:|
| Samples | 1440 |
| Async total MAPE | 0.466% |
| Async total RMSE | 23.078 us |
| Relative error P50/P90/P95 | 0.111% / 0.952% / 2.253% |
| Mean hidden communication fraction | 0.531 |

### DLRM 多配置实测异步验证

| 指标 | 当前值 |
|---|---:|
| Cases used | 1244 |
| Mean communication | 1804397.306 us |
| Mean compute proxy | 255.304 us |
| Observed eta mean/median | 0.6347 / 1.0000 |
| Mean bounded hidden fraction | 0.002451 |
| Bounded async-region MAPE | 0.027% |

## 7. 复现命令

单算子训练与验证：

```bash
cd a2a_final
python3 scripts/run_model_pipeline.py \
  --raw-data data/raw/a2a_32kb_aligned.txt \
  --split-mode size_holdout \
  --test-ratio 0.2
```

异步 replay 验证：

```bash
python3 scripts/run_async_overlap_validation.py
```

DLRM 多配置实测异步验证：

```bash
python3 scripts/run_dlrm_async_overlap_multiconfig_validation.py
```

语法检查：

```bash
python3 -m py_compile src/*.py scripts/*.py
```

## 8. 交付边界与限制

- 当前模型只对单机 8 NPU HCCL All-to-All 验证充分。
- 异步掩盖效果由应用计算窗口决定；当前 DLRM 配置通信远大于可重叠计算窗口，因此真实 hidden fraction 很低。
- 若迁移到新机器、新 HCCL 版本或跨机通信，必须重新采集 HCCL-test 数据，并用目标 workload profiler 重新校准 `eta`。

当前单算子模型在 size-holdout 验证下 MAPE 为 0.712%，异步 replay 组合误差为 0.466%，DLRMcase 实测验证证明异步路径存在但当前应用可隐藏通信比例受计算窗口限制。
