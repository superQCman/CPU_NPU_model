# 主线模型评估实验

## 本文回答什么问题

本文说明当前主线模型如何评估、每个实验为什么要做、怎么运行、看哪个输出文件，以及如何判断结果是否合理。

## 读完你应该知道什么

你应该能独立复现这些评估：

- correct-hw：数据集绑定正确硬件配置。
- frozen-swap：固定 RF 权重，仅替换硬件参数，验证硬件响应来自 whitebox anchor。RF 是 Random Forest，即随机森林模型。
- parameter perturbation：逐个扰动硬件参数，检查方向和幅度。
- multiworkload trend：画硬件参数变化下的 E2E 趋势图。E2E 是 end-to-end，即整图端到端耗时。

文中 MAPE 表示平均绝对百分比误差，P90 APE 表示第 90 百分位绝对百分比误差。YAML 是一种配置文件格式，本文中的硬件 YAML 指 `calibrator/hw_config` 下的硬件参数配置。

命名约定：`Device_1` 表示智算平台，`Device_2` 表示通算平台。路径、alias 或 YAML 名称中的 `kunpeng920` 对应智算平台，`s920x20` 对应通算平台，保留这些历史标识是为了兼容已有实验。

## 实验覆盖索引

| 实验 | 入口脚本 | 文档覆盖状态 |
|---|---|---|
| 单算子算子类 MAPE | `calibrator/plot_operator_class_mape.py` | 本文“单算子算子类建模精度”和复现手册均包含目的、命令、输出和解释边界 |
| E2E batch/并行度/算子规模误差切片 | `calibrator/experimental/plot_error_explainability_slices.py` | 本文和复现手册均包含复现命令、输出文件和泛化性解读 |
| 软件负载泛化性验证 | `calibrator/experimental/plot_software_workload_generalization.py` | 本文和复现手册说明 held-out、optional unseen 和 dense static trend 三层证据边界 |
| 随机访存延迟子模型真机验证 | `calibrator/experimental/validate_random_access_latency_model.py` | 本文和复现手册包含原理、命令、输出和适用边界 |
| 主线 E2E 相对误差 CDF | `calibrator/plot_e2e_relative_error_cdf.py` | 本文和复现手册均包含输入、命令、输出和阈值解释 |
| multiworkload trend 与 elasticity | `calibrator/plot_synthetic_hw_trend_predictions.py`, `calibrator/plot_synthetic_e2e_correlation.py` | 本文、趋势解释文档和复现手册均包含 bundle 复用、后处理和主图口径 |
| 跨平台 few-shot 泛化 | `calibrator/experimental/cross_platform_fewshot_generalization.py` | 本文、复现手册和主线链路文档均说明独立实验边界和正式判定口径 |
| reference_v2 成本优化 | `calibrator/hardware_cost_optimizer.py --cost-model reference_v2` | 成本优化文档和复现手册均说明 v2 成本函数、NSGA-II 和输出检查 |
| CPU+NPU experimental | `calibrator/experimental/plot_cpu_npu_*` | 本文、复现手册和主线链路文档均说明这是独立实验分支，不更新 CPU 主线指标 |
| DLRM compute/memory attribution | `calibrator/experimental/analyze_dlrm_compute_memory_attribution.py` | 复现手册包含目的、命令、输出和解释边界 |
| E2E ablation CDF | `calibrator/experimental/plot_e2e_ablation_cdf.py` | 复现手册包含 stage 定义、命令、输出和诊断边界 |

## 1. 当前主线结果

当前主线 route：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

当前 full split 主线结果目录：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505
```

| 数据/硬件绑定 | E2E MAPE | E2E P90 APE | E2E median APE | test 完整图组合数 |
|---|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 5.96% | 12.35% | 5.01% | 151 |
| Device_1 / 智算平台 full portable 数据 | 5.47% | 11.19% | 4.38% | 454 |

| 数据/硬件绑定 | 单算子 MAPE | 单算子 P90 APE | 单算子 median APE | eval rows |
|---|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 20.61% | 36.96% | 12.76% | 7701 |
| Device_1 / 智算平台 full portable 数据 | 15.71% | 35.44% | 8.82% | 23154 |

当前 full split 训练/测试数据规模：

| 数据/硬件绑定 | 总完整图组合数 | train 完整图组合数 | test 完整图组合数 | train 单算子行数 | test 单算子行数 | 总单算子行数 |
|---|---:|---:|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 504 | 353 | 151 | 18003 | 7701 | 25704 |
| Device_1 / 智算平台 full portable 数据 | 1512 | 1058 | 454 | 53958 | 23154 | 77112 |

完整图组合数表示一个完整 DLRM-like 配置；单算子行数表示主线 `model_input_51` 口径下的算子样本数，每个完整图组合对应 51 行单算子样本。

full split 的上层负载取值范围：

| 数据/硬件绑定 | batch size | number of indices per lookup | inter threads | op threads |
|---|---|---|---|---|
| Device_2 / 通算平台 full 数据 | `1024,1536,2048` | `1000,1200,1400,1600,1800,2000` | `1,3,5` | `1,2,3,4,5` |
| Device_1 / 智算平台 full portable 数据 | `1024,1152,1280,1408,1536,1664,1792,1920,2048` | `1000,1200,1400,1600,1800,2000` | `1,3,5` | `1,2,3,4,5` |

结果来源：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505/best_e2e_pipeline_by_policy.csv
outputs/cache_residency_worker_share_v3_full_s920x20_20260505/combined_summary.csv
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/best_e2e_pipeline_by_policy.csv
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/combined_summary.csv
```

20260429 small split 结果只有 28 个 test 完整图组合，只保留为历史快速验证参考。
当前 full 数据主线结论和趋势图展示以上面两个 20260505 结果目录为准。

### 单算子算子类建模精度

除了 `combined_summary.csv` 中的全体单算子 MAPE，还需要按算子类查看误差来源。该实验只读取两个 full 训练目录下已有的 `prediction_error_rows.csv`，不重新训练、不改 RF 权重。

复现命令：

```bash
# 在仓库根目录执行

python calibrator/plot_operator_class_mape.py \
  --device1-prediction-rows outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/prediction_error_rows.csv \
  --device2-prediction-rows outputs/cache_residency_worker_share_v3_full_s920x20_20260505/prediction_error_rows.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/operator_class_mape_cache_residency_worker_share_20260505 \
  --split-gather
```

输出文件：

| 文件 | 含义 |
|---|---|
| `operator_class_mape_bar.png/pdf` | 不拆分 Gather 的算子类 MAPE |
| `operator_class_weighted_mape_bar.png/pdf` | 不拆分 Gather 的算子类 weighted MAPE |
| `operator_class_mape_summary.csv` | 不拆分 Gather 的汇总表 |
| `operator_class_mape_gather_split_bar.png/pdf` | 将 Gather 拆成数据搬运和调度两类后的 MAPE |
| `operator_class_weighted_mape_gather_split_bar.png/pdf` | Gather 拆分后的 weighted MAPE |
| `operator_class_mape_gather_split_summary.csv` | Gather 拆分后的汇总表 |
| `OPERATOR_CLASS_MAPE_GATHER_SPLIT_REPORT.md` | 可读报告 |

`Gather-data` 表示 `whitebox_rf_group = indexed_block_copy_rf` 或 `exec_region_static = embedding_private` 的 embedding 数据搬运路径；`Gather-dispatch` 表示剩余 Gather，通常是小 shape/control 或 metadata 路径，耗时很小但相对误差容易偏高。`weighted_mape_pct` 按真实耗时加权，可以避免微秒级小算子在解释端到端误差时被过度放大。

当前 Gather 拆分口径下的 MAPE：

| 算子类 | Device_1 / 智算平台 MAPE | Device_2 / 通算平台 MAPE | 解释重点 |
|---|---:|---:|---|
| Metadata/view | 24.34% | 34.64% | 单行耗时约十微秒，百分比误差高但 E2E 权重低 |
| Gather-data | 8.22% | 12.49% | embedding 数据搬运主路径，是 E2E 关键贡献项 |
| Gather-dispatch | 27.58% | 24.60% | 小 Gather 调度路径，平均耗时低于 11 微秒 |
| ReduceSum | 7.72% | 9.41% | embedding branch reduction 主路径 |
| Gemm/MatMul | 10.76% | 15.07% | dense MLP 计算路径 |
| Concat | 9.22% | 17.12% | payload/shape concat 混合路径 |
| Transpose | 13.07% | 14.27% | 转置或 reshape-like view 路径 |
| Elementwise | 19.04% | 11.66% | 单行耗时低，E2E 贡献通常较小 |

这张图用于解释“单算子平均误差”和“E2E 误差”之间的差异：单算子 MAPE 是逐行不加权平均，metadata/view 和 dispatch-dominated 小算子数量多、耗时短，会抬高单算子 MAPE；E2E pipeline 按完整图任务图、分支并行和关键路径汇总，长耗时的 `Gather-data`、`ReduceSum`、`Gemm/MatMul` 对整体更关键，因此 E2E MAPE 可以低于全体单算子的无权平均 MAPE。

### 误差随 batch、并行度和算子规模变化分析

该实验回答审核时常见的三个问题：

- 整图误差是否只在某些 batch size 上表现好。
- 整图误差是否随图级并行度或算子线程数明显恶化。
- 代表性算子的误差是否随主导规模特征增大而系统性漂移。

实验只读取 held-out full test 的已有输出，不重新训练、不重新预测、不重新执行 E2E pipeline：

```bash
# 在仓库根目录执行

python calibrator/experimental/plot_error_explainability_slices.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --device1-test-csv data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv \
  --device2-test-csv data/data_e2e_full_s920x20_20260501/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/error_explainability_slices_cache_residency_worker_share_20260516
```

输出文件：

| 文件 | 含义 |
|---|---|
| `e2e_error_by_batch_distribution.png/pdf` | 不同 batch size 下的 E2E APE 分布 |
| `e2e_error_by_parallelism_distribution.png/pdf` | `inter_threads` 和 `num_threads` 下的 E2E APE 分布 |
| `e2e_error_by_parallelism_bubble.png/pdf` | `inter_threads × num_threads` 实际观测组合下的 E2E MAPE 气泡图 |
| `operator_error_vs_scale_facets.png/pdf` | 代表性算子 APE 随主导规模特征变化 |
| `operator_pred_actual_vs_scale_facets.png/pdf` | 代表性算子的真实耗时和预测耗时随规模特征变化 |
| `operator_scale_correlation.csv` | 主导规模特征与 APE、signed error、真实耗时、预测耗时的 Spearman 相关 |
| `ERROR_EXPLAINABILITY_SLICES_REPORT.md` | 可读报告 |

当前结果中，batch slice 的 E2E MAPE 在 Device_1 / 智算平台各 batch 下约 `4.5% - 6.9%`，在 Device_2 / 通算平台各 batch 下约 `5.7% - 6.5%`。并行组合气泡图只画真实存在的 `inter_threads × num_threads` 组合：颜色表示 E2E MAPE，点大小表示该组合样本数。Device_1 / 智算平台各组合 MAPE 约 `3.1% - 7.8%`，Device_2 / 通算平台各组合 MAPE 约 `4.4% - 7.4%`，没有出现单一并行度下的失控误差。

代表性算子规模分析中，`operator_scale_correlation.csv` 的 `spearman_actual_vs_scale` 与 `spearman_pred_vs_scale` 用于检查真实耗时和预测耗时是否都随主导规模特征同向增长；`spearman_ape_vs_scale` 用于检查误差是否随规模特征系统性恶化。当前 `Gather-data`、`ReduceSum`、`Gemm/MatMul`、`Concat` 等关键类的真实耗时和预测耗时相关性大多在 `0.7` 以上，说明模型对主导规模变化的方向响应是可解释的。

### 软件负载泛化性验证

该实验回答“CPU 性能模型是否只适用于固定 DLRM 配置”的审核问题。证据分三层：

1. 已有 held-out full test：使用真实标签，按 batch size、NIP、`inter_threads`、`num_threads` 和 `case_id` 检查 E2E APE 是否稳定。
2. 可选新增 unseen software workload：如果后续单独采集未进入 train/test split 的 DLRM-like workload，可把 unseen root/test CSV 传入同一脚本，和 full held-out test 并列输出 CDF、MAPE、P90。
3. 训练后 dense static-config trend：只加载已导出的 v3 bundle，对 batch、NIP、MLP 宽度和并行度做密集推理。该部分没有真实标签，只用于说明模型对软件规模变化的趋势响应，不作为精度指标。

复现命令：

```bash
# 在仓库根目录执行

python calibrator/experimental/plot_software_workload_generalization.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --device1-test-csv data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv \
  --device2-test-csv data/data_e2e_full_s920x20_20260501/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/software_workload_generalization_20260518
```

只有一个 Device 时，只传该 Device 的 `--device*-root`、`--device*-test-csv` 和 `--device*-bundle` 即可。若有新增 unseen workload，在命令中增加：

```bash
  --device1-unseen-root <device1_unseen_eval_output_root> \
  --device1-unseen-test-csv <device1_unseen_test_csv> \
  --device2-unseen-root <device2_unseen_eval_output_root> \
  --device2-unseen-test-csv <device2_unseen_test_csv>
```

输出文件：

| 文件 | 含义 |
|---|---|
| `software_generalization_e2e_cdf.png/pdf` | full held-out test 与 optional unseen workload 的 E2E relative error CDF |
| `software_generalization_mape_by_axis.png/pdf` | batch、NIP、图级并行和算子线程数下的 MAPE/P90 |
| `software_generalization_pred_actual_trends.png/pdf` | 有真实标签的数据上，预测 E2E 与真实 E2E 随软件轴变化的归一化趋势 |
| `operator_scale_generalization_facets.png/pdf` | 代表性算子误差随主导规模特征变化 |
| `operator_pred_actual_scale_generalization_facets.png/pdf` | 代表性算子的预测/真实耗时随主导规模特征变化 |
| `static_config_dense_workload_trends.png/pdf` | 训练后 bundle 对密集软件配置的预测趋势，仅作无标签趋势说明 |
| `SOFTWARE_WORKLOAD_GENERALIZATION_REPORT.md` | 可读报告，明确哪些结果有真实标签、哪些只是趋势 probe |

正式软件负载泛化结论只使用带真实测量值的 held-out 或 unseen workload。dense static-config 趋势图不能替代精度验证；它的作用是检查模型是否对 batch、NIP、MLP 宽度和并行度变化给出方向合理的响应。

### 随机访存延迟子模型真机验证

该实验验证 `random_mem` 子模块中的单次随机访问平均延迟 `L_touch/L_avg`，不是完整 Gather 的 `T_random`，也不是 E2E latency。完整 Gather 还包含随机带宽项、tail copy、latency parallelism 和 dispatch，因此不能用本实验单独替代 Gather/E2E 精度评估。

真机基准来自 pointer-chase microbenchmark。pointer-chase 构造依赖链式随机访问，下一次 load 地址依赖上一次 load 的结果，可有效抑制硬件预取器影响。脚本读取 `run_*/ptrchase_sweep.txt` 中的 `ns_per_load`，按 working set size 聚合多轮测量的 median/mean/std/CV，并只在主图中展示 `CV <= 15%` 且重复次数不少于 3 的稳定点。

复现命令：

```bash
# 在仓库根目录执行

python calibrator/experimental/validate_random_access_latency_model.py \
  --hw-config hardware_calibration/raw/test_40_auto_20260502/candidate_hw_config.yaml \
  --ptrchase-root hardware_calibration/raw/test_40_auto_20260502 \
  --num-threads 1 \
  --active-groups 1 \
  --modes current_interp,ppt_weighted,hard_threshold \
  --stable-cv-threshold 15 \
  --target-mape-threshold 20 \
  --out-dir outputs/random_access_latency_validation_20260517
```

三种口径：

| 口径 | 含义 |
|---|---|
| `continuous_residency` | 当前 continuous cache residency 代码口径：先计算连续 residency level，再在 L1/L2/L3/DRAM latency 间插值 |
| `weighted_average` | exclusive-share 加权平均口径：按 L1/L2/L3/DRAM exclusive residency share 直接加权 |
| `hard_threshold` | 非 continuous fallback 口径：按 working set 所属 cache level 做硬阈值判断 |

当前 raw candidate YAML 验证结果：

| 口径 | MAPE | P90 APE | Max APE | 结论 |
|---|---:|---:|---:|---|
| `continuous_residency` | 6.89% | 12.91% | 17.60% | 通过 20% 目标 |
| `weighted_average` | 6.78% | 12.44% | 16.62% | 通过 20% 目标，略优 |
| `hard_threshold` | 124.17% | 335.25% | 421.68% | 不适合连续曲线验证 |

推荐展示图：

```text
outputs/random_access_latency_validation_20260517/random_access_latency_model_vs_measured_continuous_residency.png
outputs/random_access_latency_validation_20260517/random_access_latency_model_vs_measured_weighted_average.png
```

脚本仍会把 `hard_threshold` 的结果保留在 CSV 和报告中，但主图只展示通过阈值的口径，避免失败模式的极端误差拉乱纵轴。本轮结论是：`continuous_residency` 和 `weighted_average` 都能在 pointer-chase 真机基准上达到 20% 内误差；`weighted_average` 略优，但优势不足以强制修改主线实现。

### 主线 E2E 相对误差 CDF

除了 MAPE、P90 APE 这类汇总指标，还需要看完整图误差的整体分布。CDF 是 cumulative distribution function 的缩写，中文是累积分布函数。这里的 CDF 表示“相对误差不超过某个阈值的完整图比例”。

本实验只画当前主线 route，不比较 `prior_only`、`simple_rf_v2` 或其它历史 route：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

相对误差定义为：

$$APE=\frac{|predicted\_e2e\_us-actual\_e2e\_us|}{actual\_e2e\_us}$$

其中 `predicted_e2e_us` 是模型预测的完整图端到端耗时，单位微秒；`actual_e2e_us` 是真实 profile 得到的完整图端到端耗时，单位微秒。

运行命令：

```bash
# 在仓库根目录执行

python calibrator/plot_e2e_relative_error_cdf.py \
  --device1-combo-metrics outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/e2e_pipeline_combo_metrics.csv \
  --device2-combo-metrics outputs/cache_residency_worker_share_v3_full_s920x20_20260505/e2e_pipeline_combo_metrics.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --threshold 0.15 \
  --out-dir outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505
```

输入文件必须是 combo-level E2E 结果，即一行对应一个完整图组合。脚本会过滤 `coverage_ratio = 1.0` 的完整图组合，并在 `relative_error = 0.15` 处画竖线。

输出文件：

| 文件 | 含义 |
|---|---|
| `e2e_relative_error_cdf_mainline.png` | PNG 版本 CDF 图 |
| `e2e_relative_error_cdf_mainline.pdf` | PDF 版本 CDF 图 |
| `e2e_relative_error_cdf_points.csv` | 两条 CDF 曲线的所有绘图点 |
| `e2e_relative_error_cdf_summary.csv` | 每个 device 的 MAPE、P90、P95、max 和阈值覆盖率 |
| `E2E_RELATIVE_ERROR_CDF_REPORT.md` | 可读报告 |

当前实测结果：

| Device | combo 数 | MAPE | median APE | P90 APE | P95 APE | max APE | `APE <= 15%` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Device_1 | 454 | 5.47% | 4.38% | 11.19% | 13.70% | 26.48% | 95.8% |
| Device_2 | 151 | 5.96% | 5.01% | 12.35% | 14.88% | 27.65% | 95.4% |

图路径：

```text
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/e2e_relative_error_cdf_mainline.png
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/e2e_relative_error_cdf_mainline.pdf
```

`E2E_RELATIVE_ERROR_CDF_REPORT.md` 末尾包含单独复现该图的完整命令。该命令使用报告目录下的匿名输入别名：

```text
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/input_aliases/device1_e2e_pipeline_combo_metrics.csv
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/input_aliases/device2_e2e_pipeline_combo_metrics.csv
```

### 跨平台少样本泛化实验

该实验回答另一个泛化性问题：在一个平台完成 full 训练后，迁移到另一个平台时，是否只需要约一个数量级更少的目标平台数据做校准，就能达到接近目标平台 full 训练的 E2E 精度。

实验双向覆盖：

- `Device_1 / 智算平台 -> Device_2 / 通算平台`
- `Device_2 / 通算平台 -> Device_1 / 智算平台`

少样本预算按 target 平台的 train 完整图组合数计算，而不是按单算子行数计算。当前 full split 中，Device_2 / 通算平台 train 有 353 个完整图组合，10% 表示约 35 个完整图组合；Device_1 / 智算平台 train 有 1058 个完整图组合，10% 表示约 105 个完整图组合。

实验组：

| 组别 | 含义 |
|---|---|
| `zero_shot` | source bundle 直接迁移，不使用 target 数据 |
| `fewshot_op_rescale` | 固定 source RF，只用少量 target 数据拟合 `(op_type, num_threads)` 残差缩放 |
| `fewshot_op_rescale_plus_e2e_scale` | 在 op rescale 基础上增加一个 E2E 全局缩放 |
| `fewshot_op_rescale_plus_e2e_surface` | 在 op rescale 基础上，用少量完整图拟合低容量 E2E log-space residual surface |
| `target_only_no_calibration_anchored` | 只用 target 少量数据从头训练 v3，不拟合 op-thread scale 或 E2E 校准；再在算子 `pred_us` 层按 target 数据比例向 `target_full` 算子预测做 log-space 锚定，并重新执行 E2E pipeline 调度，作为弱基准曲线 |
| `target_only_fewshot` | 只用 target 少量数据从头训练 v3，并拟合 target op-thread scale，作为诊断对照 |
| `target_full` | 目标平台 100% full 训练上界 |

运行命令：

```bash
# 在仓库根目录执行

python calibrator/experimental/cross_platform_fewshot_generalization.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device1-data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --device2-data-root data/data_e2e_full_s920x20_20260501 \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --sample-fractions 0,1,2,5,10,20,100 \
  --repeats 5 \
  --seed 20260511 \
  --primary-method fewshot_op_rescale_plus_e2e_surface \
  --workers 16 \
  --target-only-baseline-mode no_calibration \
  --anchor-target-only-no-calibration-to-target-full \
  --target-only-anchor-power 0.5 \
  --out-dir outputs/cross_platform_fewshot_generalization_20260512
```

上面的命令会同时生成主迁移校准曲线和 `target_only_no_calibration_anchored` 诊断曲线。该诊断曲线不使用 source bundle，也不做 op-thread/E2E 缩放校准，用于观察“目标平台从零少样本建模”的收敛速度；正式通过标准只看 `fewshot_op_rescale_plus_e2e_surface` 的 5 seed mean / std / worst。若只想快速复现主评估，可加 `--skip-target-only-fewshot` 跳过 target-only 诊断曲线。

如果主迁移校准结果已经存在，只需要复现或更新当前展示用的弱基准曲线和简化 MAPE 图，可以使用增量命令。该命令读取已有 `fewshot_generalization_summary.csv`，只补 `target_only_no_calibration_anchored`，再重新生成图和报告：

```bash
# 在仓库根目录执行

python calibrator/experimental/cross_platform_fewshot_generalization.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device1-data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --device2-data-root data/data_e2e_full_s920x20_20260501 \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --sample-fractions 1,2,5,10,20,100 \
  --repeats 5 \
  --seed 20260511 \
  --primary-method fewshot_op_rescale_plus_e2e_surface \
  --workers 16 \
  --target-only-baseline-mode no_calibration \
  --only-target-only-baseline \
  --merge-existing-summary outputs/cross_platform_fewshot_generalization_20260512/fewshot_generalization_summary.csv \
  --anchor-target-only-no-calibration-to-target-full \
  --target-only-anchor-power 0.5 \
  --out-dir outputs/cross_platform_fewshot_generalization_20260512
```

校准规模与测试规模：

| 迁移方向 | target train full combos | fraction | 校准完整图数 | 用于 op rescale 的算子行数 | E2E surface 拟合完整图数 | target test full combos |
|---|---:|---:|---:|---:|---:|---:|
| Device_1 / 智算平台 -> Device_2 / 通算平台 | 353 | 1% | 4 | 204 | 4 | 151 |
| Device_1 / 智算平台 -> Device_2 / 通算平台 | 353 | 5% | 18 | 916 | 18 | 151 |
| Device_1 / 智算平台 -> Device_2 / 通算平台 | 353 | 10% | 36 | 1831 | 36 | 151 |
| Device_2 / 通算平台 -> Device_1 / 智算平台 | 1058 | 1% | 11 | 561 | 11 | 454 |
| Device_2 / 通算平台 -> Device_1 / 智算平台 | 1058 | 5% | 53 | 2703 | 53 | 454 |
| Device_2 / 通算平台 -> Device_1 / 智算平台 | 1058 | 10% | 106 | 5406 | 106 | 454 |

校准参数不是 RF 参数。RF 权重固定，少样本阶段只拟合轻量校准层：先按 `(op_type, num_threads)` 拟合 `actual_us / pred_us` 的 median ratio，样本不足时回退到 `op_type` median 和全局 median；再在完整图层面拟合 E2E residual surface。E2E residual surface 的目标为 `r = log(actual_e2e_us) - log(predicted_e2e_us)`，模型为低容量 Ridge，默认 `alpha = 5.0`，残差裁剪到 `[-0.7, 0.7]`，输入特征为 `log(predicted_e2e_us)`、`log(batch_size)`、`log(num_indices_per_lookup)`、`inter_threads` 和 `case_id` 类别编码。target test actual 不参与任何校准。

输出文件：

| 文件 | 含义 |
|---|---|
| `fewshot_generalization_summary.csv` | 每个方向、实验组、target 数据比例和 seed 的 single-op/E2E 指标 |
| `fewshot_e2e_combo_metrics.csv` | combo-level E2E 指标 |
| `fewshot_single_op_metrics.csv` | single-op 指标 |
| `fewshot_sample_manifest.csv` | 每个少样本预算实际抽到的 target train combo |
| `fewshot_e2e_mape_vs_target_data.png/pdf` | E2E MAPE 随 target 数据比例变化 |
| `fewshot_e2e_mape_vs_target_data_simplified_from_1pct.png/pdf` | 从 1% 开始、只保留 `Base-model calibration` 与 `Train from scratch` 的展示用 MAPE 图 |
| `fewshot_e2e_p90_vs_target_data.png/pdf` | E2E P90 APE 随 target 数据比例变化 |
| `fewshot_10pct_vs_full_bar.png/pdf` | 10% few-shot 与 100% target full 对比 |
| `fewshot_data_efficiency_summary.png/pdf` | 达到目标精度所需 target 数据比例 |
| `fewshot_seed_stability_boxplot.png/pdf` | 10% target data 下不同 seed 的稳定性 |
| `fewshot_zero_to_calibrated_gain.png/pdf` | zero-shot 到少样本校准后的误差下降幅度 |
| `CROSS_PLATFORM_FEWSHOT_GENERALIZATION_REPORT.md` | 可读报告 |

通过标准主要看 `fewshot_op_rescale_plus_e2e_surface` 在 10% target data 下 5 个 seed 的均值表现：

- E2E MAPE mean 不超过 `1.25x target_full`，或绝对 E2E MAPE mean 不超过 8%。
- E2E P90 APE mean 不超过 `1.35x target_full`。
- `combo_count_partial = 0`。
- `coverage_ratio_mean = 1.0`。

当前 20260512 正式结果：

| 迁移方向 | target train full combos | 10% calibration combos | target full E2E MAPE | 10% few-shot E2E MAPE mean/std | target full E2E P90 APE | 10% few-shot E2E P90 APE mean/std | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| Device_1 / 智算平台 -> Device_2 / 通算平台 | 353 | 36 | 5.96% | 6.64% / 0.40% | 12.35% | 13.85% / 0.41% | PASS |
| Device_2 / 通算平台 -> Device_1 / 智算平台 | 1058 | 106 | 5.47% | 5.38% / 0.13% | 11.19% | 10.74% / 0.41% | PASS |

这组结果来自 `outputs/cross_platform_fewshot_generalization_20260512`，正式运行跳过了 `target_only_fewshot` 诊断重训；主结论只依赖固定 source RF 的迁移校准方法。

图中会在 `10%` 位置标出 `10x less target data`，并输出 seed boxplot。若某个方向未通过，报告只给出未通过结论、zero-shot 退化幅度、few-shot 校准收益、失败指标，以及达到标准所需的最小 target 数据比例；正式结果不使用单次随机抽样、删样本或替换 test split。

## 2. 统一运行前提

所有主线实验都使用：

| 项 | 当前值 |
|---|---|
| policy | `joint_6d_tuple` |
| scale | `100` |
| source dataset name | `source` |
| 主线 route | `whitebox_family_prior_plus_physics_guarded_rf_v3` |
| scale 模式 | `--op-thread-scale-mode fit` |
| 绑定文件 | `calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json` |

op-thread scale 是 Analytical prior 之后、RF 之前的校准层。它按 `(op_type, num_threads)` 拟合 `actual_us / whitebox_center_us`，因此所有主线评估命令都必须显式启用：

```text
--op-thread-scale-mode fit
```

scale 产物会写到：

```text
op_thread_scale_config_joint_6d_tuple_source_100.json
op_thread_scale_config_joint_6d_tuple_source_100.csv
```

## 3. correct-hw 实验

### 实验目的

验证每个数据集在真实硬件配置下的单算子和 E2E 预测精度。correct-hw 是主线指标来源。

### 怎么运行

Device_2 / 通算平台：

```bash
python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root data/data_e2e_full_s920x20_20260501 \
  --artifact-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --source-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --target-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --routes whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode fit \
  --export-model-bundles
```

Device_1 / 智算平台：

```bash
CROSS_HW_RF_N_JOBS=128 python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --artifact-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --source-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --target-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --routes whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode fit \
  --export-model-bundles
```

### 看哪个文件

| 文件 | 看什么 |
|---|---|
| `combined_summary.csv` | 单算子 MAPE、P90 APE |
| `best_e2e_pipeline_by_policy.csv` | E2E MAPE、P90 APE |
| `e2e_pipeline_summary.csv` | E2E 汇总、完整 combo 数、partial combo 数和 coverage |
| `op_thread_scale_config_*.csv` | 每个 `(op_type, num_threads)` 的 scale |
| `model_bundles/*.pkl` | 训练后静态配置推理使用的模型 bundle |
| `model_bundle_manifest.csv` | 已导出 bundle 的索引 |
| `family_mlp_enablement.csv` | RF group 是否启用 |
| `pipeline_artifact_index_audit.csv` | E2E 需要的 shape/timeline artifact 是否存在 |

### 通过标准

主线应同时满足：

- `ml_raw_hw_params_used = False`，说明 RF 没有直接吃 raw hardware 字段。
- op-thread scale JSON/CSV 存在，且命令使用 `--op-thread-scale-mode fit`。
- full split E2E 中 `combo_count_partial = 0`，`coverage_ratio_mean = 1.0`。
- Device_2 full split 应有 `combo_count_full = 151`；Device_1 full portable split 应有 `combo_count_full = 454`。

如果 `combo_count_full = 0` 且 `combo_count_partial > 0`，不要把这个 E2E MAPE 当成模型结果。先检查 `e2e_pipeline_partial_combo_metrics.csv` 和 `pipeline_artifact_index_audit.csv`，常见原因是远端数据没有 portable 打包，导致 `shape_csv_missing` 或 `timeline_csv_missing`。

### 只补 E2E、不重训 RF 的场景

如果单算子训练和 `prediction_error_rows.csv` 已经完成，但后来才拿到正确的 portable artifact，可以只重跑 E2E pipeline。这个流程不会重新训练 RF，也不会改动单算子指标。

历史上 Device_1 full 数据曾经使用过这种补齐方式：先在非 portable 目录完成单算子预测，再用 portable 目录补齐 shape/timeline artifact 只重算 E2E。当前 `outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505` 已经直接基于 portable full 数据重新训练并导出 bundle，不再依赖该补齐流程。

如果使用该补齐流程，重算后必须检查：

| 文件 | 通过条件 |
|---|---|
| `e2e_pipeline_summary.csv` | `combo_count_full = 454`、`combo_count_partial = 0`、`coverage_ratio_mean = 1.0` |
| `pipeline_artifact_index_audit.csv` | `shape_csv_exists = 1512 / 1512`、`timeline_csv_exists = 1512 / 1512` |
| `REUSED_SINGLEOP_PREDICTIONS.txt` | 说明本次只复用单算子预测并重算 E2E |

`effective_onnx_path` 对当前 E2E 指标不是必需项；`op_shapes_*.csv` 和 `branch_parallel_op_timeline.csv` 是必需项。

### 训练后静态配置推理 smoke test

correct-hw 训练命令加 `--export-model-bundles` 后，可以用导出的 bundle 做一次纯配置推理。这个测试只验证推理链路可运行，不计算 MAPE，因为没有真实 `actual_e2e_us`。

```bash
python calibrator/static_feature_mapping/predict_from_config.py \
  --model-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --batch-size 1024 \
  --num-indices-per-lookup 1000 \
  --arch-embedding-size 25600 \
  --arch-mlp-bot 400-300-200 \
  --inter-threads 1 \
  --num-threads 3 \
  --out-dir outputs/static_inference_smoke
```

通过标准：

- `static_model_input_51.csv` 有 51 行。
- `static_full_graph_76.csv` 有 76 行。
- `static_e2e_summary.json` 中 `predicted_e2e_us > 0`。
- 运行命令不需要 `--train-root`、`--hw-config`、profile root、shape CSV 或 timeline CSV。

### 训练后静态配置推理耗时实验

需求方要求训练完成后的推理耗时必须小于 `1 min`。这里的测试规模用 `sample_count` 表示，含义是“一次提交给性能模型的 DLRM 静态配置样本数”，不是 DLRM 内部的 batch size。

该测试覆盖 Device_1 和 Device_2 两个已训练 v3 bundle，并覆盖两类正式批量推理口径：

- `cold_bulk_cli`：严格冷启动口径。每次启动新的 Python 进程，加载一个 Device bundle，然后批量预测 `sample_count` 个静态配置。
- `warm_bulk_inprocess`：部署口径。bundle 已经加载在内存中，然后批量预测 `sample_count` 个静态配置。

脚本仍保留旧的 `cold_cli` 和 `warm_inprocess` 单样本兼容模式；正式批量实验主要看 `cold_bulk_cli` 和 `warm_bulk_inprocess`。

运行命令：

```bash
# 在仓库根目录执行

python calibrator/static_feature_mapping/benchmark_static_inference_latency.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --out-dir outputs/static_inference_bulk_latency_benchmark_20260507 \
  --threshold-seconds 60 \
  --sample-counts 1,2,4,8,16,32,48,64 \
  --bulk-repeats 10 \
  --warmup 2 \
  --workload-set dlrm_s_m_l
```

如果只需要快速生成全 `sample_count` 覆盖的 CSV 和图，可以把 `--bulk-repeats 10` 临时改成 `--bulk-repeats 1`。当前 `outputs/static_inference_bulk_latency_benchmark_20260507` 目录就是该快速全覆盖口径，用于检查两个 Device、两种批量模式和 8 个 sample count 的路径完整性；其中 `64` 样本点已经接近 `60s` 阈值，图面上半部分不会空置。

批量样本由脚本按确定性规则生成：循环使用 DLRM-S/M/L 三档模板，并对 batch size、NIP、MLP 宽度和线程配置做轻微变化。这样横轴只表示一次提交的样本数，而不是某个单一 DLRM workload 规模。

基础模板如下：

| template | batch size | NIP | embedding tables | bottom MLP | top MLP | inter threads | op threads |
|---|---:|---:|---|---|---|---:|---:|
| `DLRM-S` | 1024 | 16 | `65536 x 8` | `256-1024-256` | `292-1024-1` | 8 | 3 |
| `DLRM-M` | 2048 | 16 | `65536 x 8` | `512-2048-512` | `548-2048-1` | 8 | 3 |
| `DLRM-L` | 4096 | 16 | `65536 x 8` | `1024-4096-1024` | `1060-4096-1` | 8 | 3 |

输出文件：

| 文件 | 含义 |
|---|---|
| `static_inference_bulk_latency_runs.csv` | 每次批量重复测试的原始耗时 |
| `static_inference_bulk_latency_summary.csv` | 按 device、mode、sample_count 汇总 mean、P50、P90、P95、max |
| `static_inference_bulk_sample_predictions.csv` | 每个 Device、sample_count 下最后一次重复的逐样本预测 |
| `static_inference_bulk_latency_manifest.json` | bundle 路径、model hash、sample count、重复次数和 Python 环境 |
| `STATIC_INFERENCE_BULK_LATENCY_REPORT.md` | 可读报告，包含整体 PASS/FAIL |
| `static_inference_bulk_latency_vs_samples.png/pdf` | 整批推理耗时随 `sample_count` 的变化 |
| `static_inference_bulk_per_sample_latency.png/pdf` | 均摊单样本耗时随 `sample_count` 的变化 |
| `static_inference_bulk_throughput.png/pdf` | 吞吐 samples/s 随 `sample_count` 的变化 |

通过标准：

- 所有 `cold_bulk_cli` 分组的 `max_s <= 60`。
- 所有 `warm_bulk_inprocess` 分组的 `max_s <= 60`。
- 每次批量推理中 `static_model_input_rows = 51 * sample_count`。
- 每次批量推理中 `static_full_graph_rows = 76 * sample_count`。
- 每个样本的 `predicted_e2e_us > 0`。

解释方式：

- `static_inference_bulk_latency_vs_samples.png/pdf` 用于说明一次提交更多待预测样本时，整批耗时是否仍低于 `60s`。
- `static_inference_bulk_per_sample_latency.png/pdf` 用于说明批量提交后均摊到单个样本的耗时是否下降。
- `static_inference_bulk_throughput.png/pdf` 用于说明模型服务化时的预测吞吐能力。

当前快速全覆盖结果：

| 指标 | 数值 |
|---|---:|
| `cold_bulk_cli` 最大整批耗时 | 57.780s |
| `warm_bulk_inprocess` 最大整批耗时 | 56.315s |
| 最大均摊单样本耗时 | 3.874s/sample |
| 最高吞吐 | 1.140 samples/s |

结论：在 `sample_count=1,2,4,8,16,32,48,64` 的快速全覆盖实验中，两个 Device 的 cold/warm bulk 整批推理耗时都低于 `60s`；当前最接近阈值的是 `sample_count=64`。

## 4. frozen-swap 实验

### 实验目的

验证硬件变化时，RF 权重不重新训练，仅替换硬件参数，预测仍沿 Analytical prior 的方向变化。

### 看哪个文件

| 文件 | 看什么 |
|---|---|
| `frozen_swap_audit.csv` | RF 权重固定时的单算子变化 |
| `frozen_swap_e2e_summary.csv` | RF 权重固定时的 E2E 变化 |
| `frozen_swap_e2e_combo_metrics.csv` | 每个 combo 的 E2E 变化 |

### 通过标准

- RF 权重 hash 不变。
- 硬件变化方向与 whitebox prior 同号比例足够高。
- 若 RF 输出和白盒方向冲突，v3 guard 应投影回物理合法方向。

## 5. 硬件参数扰动实验

### 实验目的

逐个修改硬件参数，检查 Analytical prior 和主线预测是否符合物理预期。

### 怎么运行

使用统一审计入口即可覆盖参数扰动：

```bash
python calibrator/hw_generalization_audit.py \
  --binding-config calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json \
  --artifact-root outputs/hw_generalization_audit_cache_residency_worker_share_20260505 \
  --routes whitebox_family_prior_only_v2,whitebox_family_prior_plus_simple_rf_v2,whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --perturbation-routes whitebox_family_prior_only_v2,whitebox_family_prior_plus_simple_rf_v2,whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode fit
```

### 看哪个文件

| 文件 | 看什么 |
|---|---|
| `analytical_hw_perturbation_audit.csv` | Analytical prior 对硬件扰动的响应 |
| `parameter_perturbation_audit.csv` | 每个参数扰动方向是否通过 |
| `module_sensitivity_audit.csv` | 哪个服务模块对参数最敏感 |
| `generalization_claim_matrix.csv` | 面向报告的结论矩阵 |

### 判断方向

| 参数类型 | 预期方向 |
|---|---|
| 带宽、吞吐、频率、pipeline width 增大 | 延迟不应上升 |
| cache size 增大 | 延迟不应上升 |
| cache latency、DRAM latency、launch overhead 增大 | 延迟不应下降 |

## 6. multiworkload 硬件趋势图实验

### 实验目的

用 synthetic controlled workload 展示硬件参数变化时的 E2E 延迟趋势。这里的 synthetic 表示人工构造的软件负载，不要求每张图都使用同一个固定 DLRM 配置；它的目的，是让每类硬件参数在合适负载下显现趋势。

### 怎么运行：重新 fit 复现

该方式会读取 full/small split 数据，重新拟合 op-thread scale 和 v3 RF，然后出趋势图。

```bash
python calibrator/plot_synthetic_hw_trend_predictions.py \
  --binding-config calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json \
  --bundle-source fit \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --plot-source route_pred \
  --prediction-scope e2e_pipeline \
  --workload-variants small,base,large \
  --datasets s920x20_local,kunpeng920_remote \
  --cache-size-working-set-scale 1.45 \
  --op-thread-scale-mode fit \
  --output-root outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

### 怎么运行：复用已训练 bundle 快速出图

该方式固定训练好的 v3 bundle，不重新训练 RF，只改变 synthetic workload 和硬件 sweep。适合在模型已经训练完成后快速复现实验图，或调整趋势图输出目录。

```bash
python calibrator/plot_synthetic_hw_trend_predictions.py \
  --binding-config calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json \
  --bundle-source binding \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --plot-source route_pred \
  --prediction-scope e2e_pipeline \
  --workload-variants small,base,large \
  --datasets s920x20_local,kunpeng920_remote \
  --cache-size-working-set-scale 1.45 \
  --output-root outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

bundle 复用模式读取 binding config 中每个 dataset 的 `model_bundle`。该 `.pkl` 必须由训练命令的 `--export-model-bundles` 生成；已有 CSV summary 或 prediction rows 不能恢复 RF 权重。

### 看哪个文件

注意：下面这些文件由趋势图脚本写入 `--output-root` 指定的趋势图目录，不会出现在两个 correct-hw full 训练目录中。当前目录是：

```text
outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

两个 full 训练目录只保存 correct-hw 训练、单算子评估、E2E 评估和 model bundle：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505
```

| 文件 | 看什么 |
|---|---|
| `synthetic_e2e_trend_summary_route_pred.csv` | 每个 trend key 的通过率和变化幅度 |
| `synthetic_e2e_trend_workload_manifest_route_pred.csv` | 每张图使用的软件负载规模 |
| `synthetic_e2e_trend_hardware_range_summary_route_pred.csv` | 每个硬件参数的 sweep 范围 |
| `hardware_response_guard_audit.csv` | guard 触发情况 |
| `synthetic_e2e_*_trends_route_pred.png/pdf` | 论文风格趋势图 |
| `synthetic_e2e_trend_env_route_pred.json` | `bundle_source`、`model_bundle_path`、`model_sha256` 等复现信息 |
| `synthetic_e2e_trend_max_change_by_device.csv` | 每个趋势子图中 Device_1/Device_2 的最大 E2E latency 变化幅度 |
| `synthetic_e2e_hw_sw_correlation_heatmap.png/pdf` | 全局 mixed sweep 相关性诊断图，不作为硬件敏感性主证据 |
| `synthetic_e2e_hw_sw_correlation_matrix.csv` | 全局相关系数矩阵 |
| `synthetic_e2e_hw_sw_correlation_skipped_constants.csv` | 当前 sweep 中没有变化、因此不参与相关性计算的硬件字段 |
| `synthetic_e2e_hardware_local_directional_spearman_heatmap.png/pdf` | 硬件参数局部方向一致性热力图 |
| `synthetic_e2e_hardware_abs_log_elasticity_heatmap.png/pdf` | 硬件参数绝对 log-log elasticity 主图 |
| `synthetic_e2e_hardware_directional_elasticity_heatmap.png/pdf` | 旧方向修正 elasticity 诊断图 |
| `synthetic_e2e_hardware_max_change_pct_heatmap.png/pdf` | 每个硬件参数在 sweep 范围内造成的最大 E2E latency 变化 |
| `synthetic_e2e_hardware_local_sensitivity_best_workload.csv` | 每个硬件趋势选择变化最大 workload 后的方向和强度指标 |
| `synthetic_e2e_hardware_metric_comparison_summary.csv` | 全局相关性、局部方向一致性、elasticity 和变化幅度的汇总对比 |

### 当前主线趋势 key

当前 multiworkload 主线包含 17 个趋势 key：

```text
active_parallel_workers
copy_bw
cpu_clock
fma_latency
fma_throughput
full_compute_bundle
full_memory_bundle
l1d_latency
l1d_size
l2_latency
l2_size
l3_latency
l3_size
mem_latency
pipeline_width
random_gather_bw
simd_width
```

当前结果中 `synthetic_e2e_trend_summary_route_pred.csv` 的最小 pass rate 为 1.0。

### 趋势变化幅度和相关性图后处理

趋势图生成完成后，可以单独运行后处理脚本，计算每个子图的最大变化幅度，并生成软硬件参数与最终 E2E latency 的相关性热力图：

```bash
# 在仓库根目录执行

python calibrator/plot_synthetic_e2e_correlation.py \
  --points-csv outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload/synthetic_e2e_trend_sweep_points_route_pred.csv \
  --binding-config calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json \
  --out-dir outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload \
  --target-column pred_us
```

相关性和影响强度分析使用 synthetic sweep points，不使用真实 full test CSV。它用于解释当前趋势实验中哪些输入随 `pred_us` 同向或反向变化，以及哪些硬件参数对 E2E latency 的影响更强，不等同于真实生产 workload 的因果结论。

软件参数只包含：

```text
batch_size
num_indices_per_lookup
embedding_size_total
bottom_mlp_ops_proxy
inter_threads
num_threads
```

硬件参数来自 Analytical model 实际读取的字段；如果某个硬件字段在当前 sweep 中是常量，会写入 `synthetic_e2e_hw_sw_correlation_skipped_constants.csv`，不会强行画进 heatmap。

硬件参数不使用全局 pooled correlation 作为主图，因为当前趋势实验是 one-factor-at-a-time 设计：每个硬件参数只在自己的 panel 中变化。主线报告应优先使用：

- `synthetic_e2e_hardware_local_directional_spearman_heatmap.png`：检查方向是否符合物理预期。
- `synthetic_e2e_hardware_abs_log_elasticity_heatmap.png`：区分各硬件参数单位相对变化对 E2E latency 的影响强度。
- `synthetic_e2e_hardware_max_change_pct_heatmap.png`：展示 sweep 范围内最大 E2E latency 变化。

其中 `synthetic_e2e_hardware_max_change_pct_heatmap.png` 的含义是：在当前 synthetic multiworkload 趋势实验中，每个硬件参数在设定 sweep 范围内，最多能让最终 scheduled E2E latency 变化多少百分比。它不是相关系数，而是可观察性能变化幅度。

计算公式：

$$change\_pct=\frac{\max(T_{\text{E2E}})-\min(T_{\text{E2E}})}{\min(T_{\text{E2E}})}\times100\%$$

对每个硬件参数和每个 Device，脚本会分别评估 `small/base/large` 三档 controlled workload，并选择 E2E latency 变化幅度最大的一档作为 heatmap 中该格子的数值。数值越大，表示在当前实验范围内该硬件参数对最终 E2E latency 越敏感。

解释这张图时要注意两点：

- 它受 sweep 范围影响。例如 memory bundle 的范围是 `0.60x` 到 `1.60x`，CPU frequency 的范围是 `0.75x` 到 `1.25x`，因此不同参数之间不能直接当作严格的全局重要性排序。
- 它是 controlled sweep 下的 local sensitivity，不是真实生产 workload 的全局因果归因。它的作用是展示模型在硬件参数变化时，最终 E2E latency 的响应幅度是否可观、方向是否合理。

报告中推荐的展示顺序是：

1. 先使用 `synthetic_e2e_hardware_local_directional_spearman_heatmap.png` 说明趋势方向基本符合物理预期。
2. 再使用 `synthetic_e2e_hardware_abs_log_elasticity_heatmap.png` 说明哪些硬件参数单位相对变化影响更强。
3. 最后用 `synthetic_e2e_hardware_max_change_pct_heatmap.png` 辅助展示当前 sweep 范围内的最大可观察变化。

absolute log-log elasticity 的公式是：

$$E(p)=\left|\frac{\log T_{E2E}(p_2)-\log T_{E2E}(p_1)}{\log p_2-\log p_1}\right|$$

其中 `p1/p2` 是当前硬件参数 sweep 的两个端点，`T_E2E` 是对应的 scheduled E2E latency。该指标无量纲，数值越大表示硬件参数变化 1% 对 E2E latency 的相对影响越强。

## 7. DLRM 访存主导与计算参数敏感性归因

该实验用于回应 Top 硬件参数分析中的一个常见问题：为什么 DLRM 总体以访存为主，但 CPU frequency、SIMD width、FMA throughput、pipeline width 等计算/流水线参数仍可能影响 E2E latency。

实验只做后处理和 bundle 推理，不重新训练 RF，不改变 CPU 主线指标。输出目录：

```text
outputs/dlrm_compute_memory_attribution_20260516
```

复现命令：

```bash
python calibrator/experimental/analyze_dlrm_compute_memory_attribution.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device1-data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --device2-data-root data/data_e2e_full_s920x20_20260501 \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/dlrm_compute_memory_attribution_20260516
```

主要结果：

| 口径 | Device_1 | Device_2 | 解释 |
|---|---:|---:|---|
| memory-related true operator time share | 97.7% | 97.7% | `Gather/ReduceSum/data movement` 在真实 profile 中占主导 |
| compute-related true operator time share | 2.3% | 2.3% | `Gemm/MatMul/elementwise` 真实耗时占比较小 |
| Analytical memory service share | 96.3% | 92.5% | `stream_mem + random_mem` 是白盒服务时间主体 |
| Analytical compute service share | 3.7% | 7.5% | compute service 存在但不是主体 |

输出文件：

| 文件 | 作用 |
|---|---|
| `memory_vs_compute_actual_time_share.png/pdf` | 真实算子耗时中 memory-related 与 compute-related 占比 |
| `memory_vs_compute_service_share.png/pdf` | Analytical service 中 memory/compute/dispatch 占比 |
| `operator_actual_time_share_stacked_bar.png/pdf` | `Gather/ReduceSum/Gemm/MatMul` 等 op 的真实耗时占比 |
| `compute_parameter_sensitivity_top_ops.png/pdf` | compute/pipeline 参数扰动时，主要受影响的 op type |
| `compute_parameter_sensitivity_waterfall.png/pdf` | compute/pipeline 参数扰动对 scheduled E2E 的边际影响 |
| `DLRM_COMPUTE_MEMORY_ATTRIBUTION_REPORT.md` | 归因报告和复现命令 |

结论边界：

- DLRM 总体仍是访存相关开销主导，不能把该结果解释成“DLRM 是计算瓶颈”。
- 计算/流水线参数的影响来自局部关键路径：dense MLP、ReduceSum 或 tail/serial phase 即使总体占比不高，也可能影响最终 scheduled E2E latency。
- 当前 held-out DLRM 上，单个计算/流水线参数 `+25%` 的 E2E 边际影响是小幅的；此前 Top 硬件参数图展示的是 controlled workload 下更宽 sweep 范围的局部敏感性。

## 8. E2E ablation CDF 诊断

该实验用于拆解端到端精度来自哪些模块：raw Analytical prior、RF/scale 校正、direct sum 与 E2E pipeline 调度。它是只读诊断后处理，不重新训练模型，不覆盖当前主线 CDF。

复现前需要先准备 raw prior no-scale 输出。当前 20260516 结果使用下面两个 raw root：

```text
outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device1
outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device2
```

如果需要从头生成 raw prior root，可以先运行：

```bash
# Device_1 / 智算平台 raw prior
python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --artifact-root outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device1 \
  --source-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --target-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --routes whitebox_family_prior_only_v2 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode off \
  --force

# Device_2 / 通算平台 raw prior
python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root data/data_e2e_full_s920x20_20260501 \
  --artifact-root outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device2 \
  --source-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --target-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --routes whitebox_family_prior_only_v2 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode off \
  --force
```

然后运行 ablation CDF：

```bash
python calibrator/experimental/plot_e2e_ablation_cdf.py \
  --device1-raw-root outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device1 \
  --device2-raw-root outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device2 \
  --device1-full-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-full-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --rf-route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --prior-route whitebox_family_prior_only_v2 \
  --threshold 0.10,0.15 \
  --module-improvement-threshold 0.1 \
  --out-dir outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516
```

输出文件：

| 文件 | 含义 |
|---|---|
| `e2e_ablation_cdf_device1.png/pdf` | Device_1 / 智算平台 ablation CDF |
| `e2e_ablation_cdf_device2.png/pdf` | Device_2 / 通算平台 ablation CDF |
| `e2e_ablation_cdf_combined.png/pdf` | 双平台合并展示图 |
| `ablation_e2e_relative_error_cdf_summary.csv` | 每个 ablation stage 的 MAPE、P90、CDF@10%/15% |
| `ablation_module_effect_summary.csv` | 模块收益诊断表 |
| `E2E_ABLATION_CDF_REPORT.md` | 可读报告 |

ablation stage 的含义：

| Stage | 含义 |
|---|---|
| `raw_direct_sum` | raw Analytical prior，不做 scale/RF，直接把算子耗时求和 |
| `raw_pipeline` | raw Analytical prior，加 E2E pipeline 调度 |
| `rf_scale_direct_sum` | 使用主线 RF/scale 单算子预测，但直接求和 |
| `rf_scale_pipeline` | 使用主线 RF/scale 单算子预测，并执行 E2E pipeline 调度；这是当前主线口径 |

当前诊断结论是：`rf_scale_pipeline` 在两个平台上分别达到 `5.47%` 和 `5.96%` E2E MAPE，明显优于 raw prior 与 direct-sum 口径。报告中的 `review_status` 可能标记为 `NEEDS_REVIEW`，原因是 Device_2 / 通算平台的 `rf_scale_direct_sum` 不一定优于 `raw_direct_sum`；这不影响主线结论，因为主线从来不使用 direct sum 作为最终 E2E 预测，而是使用 pipeline 调度后的 `rf_scale_pipeline`。

## 9. CPU+NPU experimental 分支

这些实验只作为独立 CPU+NPU hybrid 后处理或静态推理验证，不更新 CPU 主线 MAPE/P90 表，也不代表真实 NPU 加速收益。`Device_1` 和 `Device_2` 在这里表示两个 CPU 校准模型和数据绑定，不表示两台机器都有 NPU；特别是 Device_2 是通算服务器，没有 NPU。

### 9.1 CPU+NPU hybrid CDF

该实验读取已有 CPU full E2E 输出，把指定 offload 算子的 CPU 误差替换为 NPU 算子类别平均误差，生成独立 CDF 图。

```bash
python calibrator/experimental/plot_cpu_npu_hybrid_e2e_error_cdf.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --offload-profile compute_island \
  --out-dir outputs/experimental_cpu_npu_hybrid_cdf_20260510
```

输出包括：

```text
hybrid_e2e_error_cdf.png/pdf
hybrid_e2e_error_cdf_cpu_npu_only.png/pdf
hybrid_e2e_error_cdf_summary.csv
HYBRID_CPU_NPU_E2E_CDF_REPORT.md
```

### 9.2 CPU+NPU batch/NIP sensitivity

该实验使用已有 full E2E 结果，按 batch size 和 number of indices per lookup 聚合 CPU+NPU hybrid prediction，只展示归一化趋势和最大变化百分比。

```bash
python calibrator/experimental/plot_cpu_npu_hybrid_batch_nip_sensitivity.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --offload-profile compute_island \
  --style-reference-dir outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload \
  --out-dir outputs/experimental_cpu_npu_hybrid_batch_nip_sensitivity_20260510
```

### 9.3 Dense static-config model-inference sensitivity

该实验不局限于训练/测试 combo，直接加载两个已训练 CPU v3 bundle，生成更密集的 batch/NIP 采样点，验证训练后模型对上层负载变化的趋势是否符合预期。

```bash
python calibrator/experimental/plot_cpu_npu_hybrid_batch_nip_model_inference.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --offload-profile compute_island \
  --out-dir outputs/experimental_cpu_npu_hybrid_batch_nip_model_inference_20260510
```

通过标准：

- 所有输出只写入 `outputs/experimental_cpu_npu_*`。
- 图中使用 `Device_1`、`Device_2` 匿名模型名。
- 报告明确说明该实验不是 NPU 加速收益结论。
- batch size 和 NIP 增大时，归一化 E2E latency 的趋势应整体非下降。

## 10. 专家审查时的解释口径

- 主线不是黑盒 RF。硬件参数先进入 Analytical prior，RF 只能基于白盒 anchor 做残差修正。
- op-thread scale 是 RF 前的尺度校准，解决不同算子和线程数组合的整体偏差。
- multiworkload 趋势图不是固定同一个软件负载，而是 controlled workload，用来隔离验证各类硬件参数。
- Device_1 和 Device_2 的真实硬件绑定来自 `calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json`，不是数据包内部可能过时的 metadata。
