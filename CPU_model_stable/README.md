# CPU_model_stable

`CPU_model_stable` 是 CPU 性能建模与硬件泛化验证项目的稳定交付目录。项目目标是基于 CPU 硬件参数和 DLRM-like 软件负载信息，预测单算子延迟与完整计算图的 E2E latency。

当前主线 route：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

主线框架：

```text
硬件 YAML + workload 数据
  -> Analytical prior
  -> op-thread scale
  -> physics-guarded RF
  -> 单算子预测
  -> E2E pipeline
  -> 整图延迟、评估指标、趋势图和优化结果
```

## 交付文档

正式交付文档在 [docs_delivery](docs_delivery)：

| 文档 | 内容 |
|---|---|
| [USER_GUIDE.md](docs_delivery/USER_GUIDE.md) | 怎么使用：环境、数据绑定、训练评估、静态推理、趋势图、成本优化和排查 |
| [INTERNAL_PRINCIPLES.md](docs_delivery/INTERNAL_PRINCIPLES.md) | 内部原理：Analytical prior、op-thread scale、physics-guarded RF、E2E pipeline 和趋势控制 |
| [SOURCE_MAPPING.md](docs_delivery/SOURCE_MAPPING.md) | 交付文档来源映射，便于从原始文档、代码和产物追溯事实 |

`docs/` 保留为过程文档和历史细节，不作为第一入口。

## 项目框架

| 路径 | 作用 |
|---|---|
| `hardware_calibration/` | 硬件参数自动校准、microbenchmark、硬件 YAML 生成 |
| `calibrator/` | 主线训练、评估、趋势图、成本优化和静态配置推理入口 |
| `cpu_perf_model/` | E2E 图调度、pipeline evaluation 和图结构相关逻辑 |
| `bin/` | 数据生成、ONNX 生成、case job、split 构建和 smoke validation 脚本 |
| `data/` | 当前实验数据、portable 数据和 split 数据 |
| `outputs/` | 训练结果、评估指标、bundle、趋势图和后处理产物 |
| `env/` | Python 依赖和环境检查 |
| `docs_delivery/` | 正式交付文档 |
| `docs/` | 原始过程文档、实验说明和历史记录 |

## 核心输入

主线只信任 `calibrator/hw_config` 下的硬件 YAML。当前正式绑定：

| 设备 | 数据目录 | 硬件 YAML |
|---|---|---|
| Device_1 / 智算平台 | `data/data_e2e_full_kunpeng920_20260501_portable` | `calibrator/hw_config/kunpeng920_single_numa.yaml` |
| Device_2 / 通算平台 | `data/data_e2e_full_s920x20_20260501` | `calibrator/hw_config/s920x20_single_numa.yaml` |

绑定配置：

```text
calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json
```

路径中的 `kunpeng920` 和 `s920x20` 是历史兼容标识；交付展示统一使用 Device_1 / Device_2。

## 核心输出

当前正式结果目录：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505
```

关键产物：

| 文件 | 作用 |
|---|---|
| `combined_summary.csv` | 单算子指标 |
| `best_e2e_pipeline_by_policy.csv` | 主线 E2E 指标 |
| `e2e_pipeline_combo_metrics.csv` | combo-level E2E 明细 |
| `op_thread_scale_config_*.json/.csv` | op-thread scale 系数 |
| `model_bundles/*.pkl` | 训练后静态推理、趋势图和成本优化使用的模型 bundle |
| `pipeline_artifact_index_audit.csv` | shape/timeline artifact 完整性审计 |

训练后静态配置推理、批量推理耗时、趋势图复现和成本优化入口见 [USER_GUIDE.md](docs_delivery/USER_GUIDE.md)。

## 结果边界

- CPU 主线指标只来自 correct-hw full test、完整 E2E combo 和显式导出的 bundle。
- 远端数据必须使用 portable root，不能只复制 split CSV。
- few-shot、CPU+NPU、趋势图和成本优化是独立评估或应用分支，不覆盖 CPU 主线 MAPE/P90。
- `outputs/20260429` 相关 small split 结果只作历史快速验证参考，不作为当前 full 数据主线结论。
