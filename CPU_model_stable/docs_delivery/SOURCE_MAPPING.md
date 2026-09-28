# 交付文档来源映射

本目录是正式交付文档入口。原始 `docs/` 保留为过程文档和历史细节；本文件说明交付文档与原始文档、代码和产物之间的对应关系。

## 1. 文档清单

| 文件 | 角色 | 目标读者 |
|---|---|---|
| `USER_GUIDE.md` | 使用说明 | 需要运行、复现、验收或接入模型的人 |
| `INTERNAL_PRINCIPLES.md` | 内部原理 | 需要理解模型链路、物理约束和解释边界的人 |
| `CPU_OPERATOR_PERFORMANCE_MODELING_DESIGN.md` | 第一阶段 CPU 算子性能建模方案 | 需要审核 CPU 建模特征、技术路线、精度耗时和模型代码的人 |
| `SOURCE_MAPPING.md` | 来源映射 | 审核交付文档与原始文档对应关系的人 |

## 2. USER_GUIDE.md 来源

| 交付章节 | 主要来源 |
|---|---|
| 交付范围、主线 route、当前正式结果 | `docs/README.md`, `docs/MODEL_CHAINS_AND_MAINLINE.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md` |
| 环境配置和验收工作区 | 旧版根目录 `README.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md` |
| 硬件 YAML、数据绑定、portable 数据要求 | `docs/README.md`, `docs/MAINLINE_IO_CONTRACT.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md` |
| 新机器完整链路、数据生成和主线训练 | 旧版根目录 `README.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md` |
| 静态配置推理和上层参数到算子规模映射 | `calibrator/static_feature_mapping/core.py`, `standalone_demos/dlrm_operator_scale_mapping/dlrm_operator_scale_mapping_demo.py`, `docs/MAINLINE_IO_CONTRACT.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md` |
| 批量静态配置推理耗时 | 旧版根目录 `README.md`, `docs/README.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md` |
| CDF、消融、算子类误差、图聚合任务 CDF、误差切片、软件负载泛化和趋势图 | `calibrator/calculate_graph_task_mape.py`, `calibrator/plot_task_class_relative_error_cdf.py`, 旧版根目录 `README.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md`, `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md` |
| 白盒连续/随机访存模块 CPU 计算开销 | `calibrator/benchmark_memory_model_overhead.py`, `calibrator/whitebox_family_routes.py` |
| few-shot、随机访存验证和 NSGA-II | 旧版根目录 `README.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md`, `docs/HARDWARE_COST_OPTIMIZATION.md` |
| 成本约束硬件参数优化 | `docs/HARDWARE_COST_OPTIMIZATION.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md`, `docs/参数约束v2.docx` |
| 常见失败排查和使用边界 | `docs/README.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md` |

## 3. INTERNAL_PRINCIPLES.md 来源

| 交付章节 | 主要来源 |
|---|---|
| 主线链路 | `docs/MODEL_CHAINS_AND_MAINLINE.md`, `docs/MAINLINE_IO_CONTRACT.md` |
| 输入样本和硬件参数 | `docs/MAINLINE_IO_CONTRACT.md`, `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md` |
| Analytical prior | `docs/MAINLINE_IO_CONTRACT.md`, `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md` |
| 算子组合规则 | `docs/MAINLINE_IO_CONTRACT.md`, `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md` |
| random memory 和 cache residency | `docs/MAINLINE_IO_CONTRACT.md`, `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md` |
| op-thread scale | `docs/MODEL_CHAINS_AND_MAINLINE.md`, `docs/MAINLINE_IO_CONTRACT.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md` |
| physics-guarded RF | `docs/MODEL_CHAINS_AND_MAINLINE.md`, `docs/MAINLINE_IO_CONTRACT.md`, `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md` |
| E2E pipeline 和 static graph 口径 | `docs/MAINLINE_IO_CONTRACT.md`, `docs/MODEL_CHAINS_AND_MAINLINE.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md` |
| 趋势图机制和硬件敏感性解释 | `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md` |
| DLRM 访存主导与计算参数敏感性 | `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md`, `docs/MAINLINE_EVALUATION_EXPERIMENTS.md`, `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md` |
| 成本优化边界 | `docs/HARDWARE_COST_OPTIMIZATION.md` |

## 4. 代码和产物事实源

| 事实 | 校对来源 |
|---|---|
| 主线 route 名称 | `calibrator/ml_semantic_routes.py`, `calibrator/plot_operator_class_mape.py`, `calibrator/plot_e2e_relative_error_cdf.py` |
| active Analytical modules 与禁用模块 | `calibrator/whitebox_family_routes.py` |
| CPU 指令位宽、指令延时、CPU 频率和流水线宽度字段 | `calibrator/hw_config/s920x20_single_numa.yaml`, `calibrator/hw_config/kunpeng920_single_numa.yaml`, `hardware_calibration/README.md` |
| CPU Analytical prior、op-thread scale、physics-guarded RF 和 guard | `calibrator/whitebox_family_routes.py`, `calibrator/whitebox_perf_feature_rf.py` |
| route 默认 continuous cache residency 关闭 | `calibrator/whitebox_family_routes.py` |
| 趋势图默认启用 continuous cache residency | `calibrator/plot_synthetic_hw_trend_predictions.py` |
| static `model_input_51` / `full_graph_76` | `calibrator/static_feature_mapping/core.py`, `calibrator/static_feature_mapping/predict_from_config.py` |
| E2E MAPE/P90/combo 数 | `outputs/cache_residency_worker_share_v3_full_s920x20_20260505/best_e2e_pipeline_by_policy.csv`, `outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/best_e2e_pipeline_by_policy.csv` |
| 单算子 MAPE/P90/eval rows | `outputs/cache_residency_worker_share_v3_full_s920x20_20260505/combined_summary.csv`, `outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/combined_summary.csv` |
| static inference 批量耗时 | `outputs/static_inference_bulk_latency_benchmark_20260507/static_inference_bulk_latency_summary.csv` |
| 图聚合任务 MAPE、任务级 APE 与 CDF 阈值覆盖率 | `outputs/dlrm_task_class_mape_cache_residency_worker_share_20260722/dlrm_task_class_prediction_rows.csv`, `calibrator/calculate_graph_task_mape.py`, `calibrator/plot_task_class_relative_error_cdf.py` |
| 白盒连续/随机访存模块 CPU 计算开销 | `outputs/memory_model_overhead_20260803/memory_model_overhead_summary.csv`, `outputs/memory_model_overhead_20260803/memory_model_overhead_metadata.json`, `calibrator/benchmark_memory_model_overhead.py` |
| multiworkload trend 通过率和趋势 key | `outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload/synthetic_e2e_trend_summary_route_pred.csv` |

## 5. CPU_OPERATOR_PERFORMANCE_MODELING_DESIGN.md 来源

| 交付章节 | 主要来源 |
|---|---|
| 项目背景与验收要求 | 第一阶段验收要求、`docs_delivery/USER_GUIDE.md`, `docs_delivery/INTERNAL_PRINCIPLES.md` |
| CPU 算子性能建模背景与挑战 | `docs/MODEL_CHAINS_AND_MAINLINE.md`, `docs/MAINLINE_IO_CONTRACT.md`, `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md` |
| CPU 建模输入特征 | `hardware_calibration/README.md`, `calibrator/hw_config/s920x20_single_numa.yaml`, `calibrator/hw_config/kunpeng920_single_numa.yaml`, `docs/MAINLINE_IO_CONTRACT.md` |
| CPU Analytical Model 设计 | `calibrator/whitebox_family_routes.py`, `docs_delivery/INTERNAL_PRINCIPLES.md`, `docs/MULTIWORKLOAD_HW_TREND_AND_MODEL_EXPLANATION.md` |
| CPU 机器学习残差模型与校准 | `calibrator/whitebox_family_routes.py`, `calibrator/whitebox_perf_feature_rf.py`, `docs/MODEL_CHAINS_AND_MAINLINE.md` |
| 精度、耗时和验收口径 | `outputs/cache_residency_worker_share_v3_full_s920x20_20260505/*`, `outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/*`, `outputs/dlrm_task_class_mape_cache_residency_worker_share_20260722/*`, `outputs/memory_model_overhead_20260803/*`, `outputs/static_inference_bulk_latency_benchmark_20260507/static_inference_bulk_latency_summary.csv` |
| 模型代码说明 | `hardware_calibration/`, `calibrator/`, `cpu_perf_model/`, `standalone_demos/dlrm_operator_scale_mapping/` |

## 6. 整理规则

- 原始 `docs/*.md` 保留为过程文档和历史细节。
- `docs/参数约束v2.docx` 只作为成本约束来源引用，未编辑。
- 交付文档删除了模板化开头，合并了重复命令。
- 保留关键路径、命令、指标、判断标准和失败排查。
- 对实验分支统一标明边界，避免把诊断结果写成 CPU 主线结论。

## 7. 阅读建议

建议按下面顺序审核：

1. 先读 `USER_GUIDE.md`，确认使用路径。
2. 再读 `INTERNAL_PRINCIPLES.md`，理解内部原理和边界。
3. 验收 CPU 建模方案时读 `CPU_OPERATOR_PERFORMANCE_MODELING_DESIGN.md`。
4. 对照本文件检查每个章节的原始来源。
