# CPU_model_stable 文档入口

## 本文回答什么问题

本文是项目文档入口，说明当前 CPU 性能模型的主线链路、可信数据绑定、评估证据和推荐阅读顺序。`CPU_model_stable` 的目标是：基于 CPU 硬件参数和 DLRM-like 软件负载信息，预测单算子延迟与 E2E 延迟。E2E 是 end-to-end 的缩写，表示一个完整计算图从开始到结束的总耗时。

## 读完你应该知道什么

你应该知道当前主线模型是哪条链路、两个设备和数据集如何绑定、每篇文档负责解释哪一部分，以及从哪里复现实验结果。

## 当前主线结论

当前主线 route 是：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

route 可以理解为“一套从输入特征到预测结果的完整建模链路”。RF 是 Random Forest 的缩写，即随机森林模型。当前主线不是单纯的黑盒模型，而是以硬件参数驱动的白盒估计为基础，再用受约束的 RF 修正残差：

1. Analytical prior：先用白盒公式根据硬件和软件负载估计延迟。
2. op-thread scale：在 Analytical 之后、RF 之前拟合 `(op_type, num_threads)` 缩放系数，修正不同算子和线程数下的整体尺度。
3. RF residual：用随机森林学习白盒公式没有解释完的残差。
4. E2E pipeline：把单算子预测放回计算图调度中，得到整图延迟。

这套分层设计的优势是：硬件参数首先进入 Analytical prior，RF 主要修正白盒模型的数值偏差；E2E pipeline 再把单算子预测放回图调度结构中，避免用简单求和替代端到端调度。

训练完成后，也可以走“静态配置推理”入口：只输入 DLRM 上层负载参数和导出的模型 bundle，自动生成算子规模、静态图和主线 v3 E2E 延迟。该入口不读取 profile 生成的静态算子特征、shape CSV 或 timeline CSV，适合在固定模型 bundle 后快速评估新静态配置。

训练后静态配置推理已扩展为批量耗时实验，覆盖 Device_1/Device_2 两个模型 bundle 和 `sample_count=1,2,4,8,16,32,48,64` 的一次提交样本数变化。当前快速全覆盖结果中，cold bulk 最大整批耗时 `57.780s`，warm bulk 最大整批耗时 `56.315s`，均满足 `< 1 min`；复现方法见 [MAINLINE_EVALUATION_EXPERIMENTS.md](MAINLINE_EVALUATION_EXPERIMENTS.md)。

MAPE 是 mean absolute percentage error，中文是平均绝对百分比误差，越低越好。P90 APE 是第 90 百分位的绝对百分比误差，表示 90% 样本的误差不超过该值，也越低越好。

| 数据/硬件绑定 | route | E2E MAPE | E2E P90 APE | test 完整图组合数 |
|---|---|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | `whitebox_family_prior_plus_physics_guarded_rf_v3` | 5.96% | 12.35% | 151 |
| Device_1 / 智算平台 full portable 数据 | `whitebox_family_prior_plus_physics_guarded_rf_v3` | 5.47% | 11.19% | 454 |

## 设备和数据绑定

YAML 是一种人类可读的配置文件格式，当前硬件配置都使用 YAML 文件保存。当前唯一可信的硬件配置目录是：

```text
calibrator/hw_config
```

当前主线结果对应的实验绑定文件是：

```text
calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json
```

命名约定：文档展示中 `Device_1` 表示智算平台，`Device_2` 表示通算平台。历史文件名、目录名、dataset alias 和 YAML 名称中的 `kunpeng920` 对应智算平台，`s920x20` 对应通算平台；这些标识仅为兼容已有实验和脚本，不作为展示名称。

| 设备别名 | 实际硬件 | 数据目录 | 硬件 YAML |
|---|---|---|---|
| Device_1 | 智算平台远端机器 | `data/data_e2e_full_kunpeng920_20260501_portable`（智算平台） | `calibrator/hw_config/kunpeng920_single_numa.yaml`（智算平台） |
| Device_2 | 通算平台本机 | `data/data_e2e_full_s920x20_20260501`（通算平台） | `calibrator/hw_config/s920x20_single_numa.yaml`（通算平台） |

注意：某些数据包内部 metadata 可能仍写着旧硬件名。当前判断硬件真值时，只看绑定文件和 `calibrator/hw_config` 下的 YAML。

远端数据必须使用 portable 包。portable 包会把 E2E 评估需要的 `op_shapes_*.csv` 和 `branch_parallel_op_timeline.csv` 一起带到当前机器；如果只拷贝 split CSV，E2E 会出现 `combo_count_full=0`、`shape_csv_missing` 或 `timeline_csv_missing`。

## 推荐阅读顺序

1. [MODEL_CHAINS_AND_MAINLINE.md](MODEL_CHAINS_AND_MAINLINE.md)
   - 回答“当前有哪些模型链路，主线是哪条，为什么这样分层”。
   - 如果只想快速理解项目，先读这篇。

2. [MAINLINE_IO_CONTRACT.md](MAINLINE_IO_CONTRACT.md)
   - 回答“主线每个模块吃什么输入、吐什么输出、字段是什么意思”。
   - 如果要改代码、查字段或写实验报告，读这篇。

3. [MAINLINE_EVALUATION_EXPERIMENTS.md](MAINLINE_EVALUATION_EXPERIMENTS.md)
   - 回答“当前主线如何评估，实验怎么跑，看哪些文件，怎么解释结果”。
   - 如果要复查 MAPE、P90 APE 或硬件泛化趋势，读这篇。

4. [REPRODUCIBLE_END_TO_END_PIPELINE.md](REPRODUCIBLE_END_TO_END_PIPELINE.md)
   - 回答“从硬件参数采集、训练数据采集到训练预测，完整流程怎么复现”。
   - 如果要在新机器或新数据上重跑，读这篇。

5. [HARDWARE_COST_OPTIMIZATION.md](HARDWARE_COST_OPTIMIZATION.md)
   - 回答“在没有正式成本函数时，如何先用代理成本函数做硬件参数优化”。
   - 如果要把性能模型用于成本约束硬件配置搜索，读这篇。

CPU+NPU hybrid 相关脚本是独立 experimental 分支，只用于误差替换和趋势验证，不改变 CPU 主线训练、预测或指标结论。复现命令见 [REPRODUCIBLE_END_TO_END_PIPELINE.md](REPRODUCIBLE_END_TO_END_PIPELINE.md)，实验解释见 [MAINLINE_EVALUATION_EXPERIMENTS.md](MAINLINE_EVALUATION_EXPERIMENTS.md)。

跨平台 few-shot 泛化实验也是独立评估分支。它固定 source 平台 full 训练得到的 RF bundle，只用少量 target 平台完整图组合拟合轻量校准层；正式结论以 5 个 seed 的均值、标准差和阈值判定为准，不改变上表 CPU full 主线指标。

软件负载泛化性验证用于回答“模型是否只适用于固定 DLRM 配置”。该实验把 held-out full test、可选新增 unseen workload 和训练后 dense static-config 趋势放在同一报告中；只有带真实测量值的 held-out / unseen 数据用于精度结论，dense static trend 只用于说明模型对 batch、NIP、MLP 宽度和并行度变化的响应方向。复现命令见 [MAINLINE_EVALUATION_EXPERIMENTS.md](MAINLINE_EVALUATION_EXPERIMENTS.md) 和 [REPRODUCIBLE_END_TO_END_PIPELINE.md](REPRODUCIBLE_END_TO_END_PIPELINE.md)。

## 当前结果来源

主线结果目录：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505
```

关键结果文件：

| 文件 | 说明 |
|---|---|
| `best_e2e_pipeline_by_policy.csv` | 正确硬件绑定下的 E2E 指标 |
| `e2e_pipeline_summary.csv` | E2E 汇总指标和完整/部分 combo 计数 |
| `combined_summary.csv` | 单算子指标 |
| `op_thread_scale_config_joint_6d_tuple_source_100.json` | op-thread scale 系数 |
| `model_bundles/*.pkl` | 训练后静态配置推理使用的模型 bundle；只有运行时显式加 `--export-model-bundles` 才会生成 |
| `pipeline_artifact_index_audit.csv` | E2E 所需 shape/timeline artifact 是否完整 |

当前 multiworkload 硬件趋势图来源是：

```text
outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

20260429 目录只作为历史 small split 快速验证和旧趋势图参考。
当前 full split 主线精度和趋势展示以上面 20260505 结果目录为准。

注意：当前已有 full 结果目录中的 CSV summary、prediction rows 和 op-thread scale config 不是完整 RF 模型权重，不能替代 `model_bundles/*.pkl`。成本优化和训练后静态配置推理如果要使用 RF v3，必须提供已经导出的 pickle bundle；否则只能显式使用 prior-only fallback。

## 文档维护原则

- 文档默认面向不了解项目的新读者。
- 第一次出现的缩写必须解释。
- 公式必须单行书写，例如 `$$T = A + B$$`。
- 不把 baseline 或诊断 route 写成当前主线。
- 不在文档中使用过时硬件配置目录作为可信来源。
- 展示图和诊断图必须区分：主线结论只使用 correct-hw full test、显式导出的 bundle 和完整 E2E combo；辅助图用于解释趋势、误差来源或敏感性，不替代主线 MAPE/P90。
