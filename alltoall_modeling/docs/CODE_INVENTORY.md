# 代码与产物清单

## 1. 核心代码

| 路径 | 类型 | 作用 | 交付状态 |
|---|---|---|---|
| `src/bw_model.py` | Python module | 解析 HCCL-test 输出，按 Group1/2/3 拟合 All-to-All bus bandwidth 曲线 | 必交付 |
| `src/latency_model.py` | Python module | 基于带宽模型构建三段时延模型，并用残差样条修正 | 必交付 |
| `src/async_overlap_model.py` | Python module | 实现异步掩盖公式、`AsyncOverlapParams`、`predict_async_region`、`derive_eta` | 必交付 |
| `src/memory_bw_utils.py` | Python module | sigmoid 与通信带宽换算工具 | 必交付 |
| `src/__init__.py` | Python package marker | 允许 `src.*` import | 必交付 |

## 2. 执行脚本

| 路径 | 作用 | 是否会自动采集 |
|---|---|---|
| `scripts/run_model_pipeline.py` | 训练带宽/时延模型，保存 JSON 参数，生成 size-holdout 或 external-test 验证报告 | 否 |
| `scripts/generate_size_manifest.py` | 生成覆盖 Group1/2/3 的消息大小清单 | 否 |
| `scripts/collect_train_hccl_a2a_npu8.sh` | 打印训练采集命令，覆盖三类对齐消息大小 | 否 |
| `scripts/collect_test_hccl_a2a_npu8.sh` | 打印外部测试采集命令 | 否 |
| `scripts/run_async_overlap_validation.py` | 基于单算子验证 CSV 做异步 replay 验证 | 否 |
| `scripts/run_dlrm_async_overlap_validation.py` | 读取单个 DLRM Ascend profiler trace，校准/验证异步 overlap | 否 |
| `scripts/run_dlrm_async_overlap_multiconfig_validation.py` | 读取 DLRM 1452-case sweep profiler，输出多配置 overlap 验证 | 否 |

## 3. 数据文件

| 路径 | 说明 | 是否用于训练 |
|---|---|---|
| `data/raw/a2a_32kb_aligned.txt` | 当前基线 HCCL-test All-to-All 原始数据 | 是 |
| `data/raw/a2a_train_sizes.txt` | 训练 size manifest | 辅助 |
| `data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT/step_trace_time.csv` | DLRM 单 trace 异步验证 step trace | 验证 |
| `data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT/operator_details.csv` | DLRM 单 trace all-to-all start/wait marker 验证 | 验证 |
| `data/profiler/dlrm_sweep/sweep_app_summary.csv` | DLRM 多配置 sweep case 元信息 | 验证 |
| `data/profiler/dlrm_sweep/case_*/profiling/rank0/*/ASCEND_PROFILER_OUTPUT/step_trace_time.csv` | DLRM 多配置 sweep step trace，共 1452 份 | 验证 |
| `artifacts/reports/split_data/train_data_size_holdout.txt` | size-holdout 训练拆分结果 | 是，pipeline 生成 |
| `artifacts/reports/split_data/test_data_size_holdout.txt` | size-holdout 测试拆分结果 | 是，pipeline 生成 |

## 4. 模型参数

| 路径 | 模型 | 内容 |
|---|---|---|
| `artifacts/models/bw_model_params.json` | All-to-All 带宽模型 | sigmoid、dip、饱和带宽、Group3 drop size 等参数 |
| `artifacts/models/latency_model_params.json` | All-to-All 时延模型 | `t_s/m1/m2/bw_max` 与残差样条节点 |
| `artifacts/models/async_overlap_model_params.json` | Replay 异步掩盖模型 | `eta/launch/sync/window_scale` 与 replay 配置 |
| `artifacts/models/dlrm_async_overlap_model_params.json` | DLRM 单 trace overlap 校准 | 单 trace median eta 与观测指标 |
| `artifacts/models/dlrm_async_overlap_multiconfig_model_params.json` | DLRM 多配置 overlap 校准 | 1452-case sweep median eta 与观测指标 |

## 5. 指标报告

| 路径 | 内容 |
|---|---|
| `artifacts/reports/fit_metrics.csv` | 每个 alignment group 的训练/拟合误差 |
| `artifacts/reports/validation_metrics.csv` | 每个测试点的真实时延、预测时延、绝对误差、相对误差 |
| `artifacts/reports/coverage_summary.csv` | train/test 中 Group1/2/3 的样本数、unique size 和 size 范围 |
| `artifacts/reports/validation_summary.md` | 单算子 size-holdout 汇总报告 |
| `artifacts/reports/async_overlap_validation.csv` | replay 异步验证逐点结果 |
| `artifacts/reports/async_overlap_metrics_by_group.csv` | replay 异步验证按 group 汇总 |
| `artifacts/reports/async_overlap_metrics_by_compute_ratio.csv` | replay 异步验证按计算窗口比例汇总 |
| `artifacts/reports/async_overlap_summary.md` | replay 异步验证 Markdown 报告 |
| `artifacts/reports/dlrm_async_overlap_validation.csv` | DLRM 单 trace 逐 step overlap 验证 |
| `artifacts/reports/dlrm_async_overlap_summary.md` | DLRM 单 trace Markdown 报告 |
| `artifacts/reports/dlrm_async_overlap_multiconfig_steps.csv` | DLRM 多配置逐 step bounded-overlap 验证 |
| `artifacts/reports/dlrm_async_overlap_multiconfig_cases.csv` | DLRM 多配置按 case bounded-overlap 汇总 |
| `artifacts/reports/dlrm_async_overlap_multiconfig_by_variant.csv` | DLRM 多配置按 case_7..case_10 汇总 |
| `artifacts/reports/dlrm_async_overlap_multiconfig_summary.md` | DLRM 多配置 Markdown 报告 |

## 6. 图表

| 路径 | 内容 |
|---|---|
| `artifacts/plots/bw_model_fit.png` | Group1/2/3 带宽拟合曲线 |
| `artifacts/plots/latency_model_fit.png` | 三段时延模型与残差样条修正 |
| `artifacts/plots/latency_compare.png` | 测试集预测 vs 真实时延对比 |
| `artifacts/plots/async_overlap_validation.png` | replay 异步验证预测对比与隐藏比例趋势 |
| `artifacts/plots/dlrm_async_overlap_validation.png` | DLRM 单 trace overlap 验证图 |
| `artifacts/plots/dlrm_async_overlap_multiconfig_validation.png` | DLRM 多配置 overlap 验证图 |

## 7. 文档

| 路径 | 内容 |
|---|---|
| `DELIVERY_MANIFEST.md` | 交付文件总清单、验收口径、结果摘要 |
| `docs/USER_GUIDE.md` | 使用说明、复现流程、输入输出、常见问题 |
| `docs/MODELING_PRINCIPLES.md` | All-to-All 单算子与异步掩盖建模原理 |
| `docs/CODE_INVENTORY.md` | 代码、脚本、数据、模型、报告和图表清单 |
| `docs/modeling_scheme.md` | 方案说明 |


