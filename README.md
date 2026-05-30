# CPU/NPU 端到端性能建模交付

本仓库把 CPU、NPU、All-to-All 通信、CPU/NPU 融合端到端模型，以及分级存储模块放在同一个顶层工程中。各子模块保持相对独立，最终端到端预测入口位于 `cpu_npu_e2e/`。

## 目录说明

| 目录 | 说明 |
|---|---|
| `CPU_model_stable/` | CPU 侧 DLRM 图、静态特征、CPU 性能模型训练/推理代码和交付文档。 |
| `npu_modeling/` | NPU 语义算子物理内存模型、硬件描述、训练/推理工具和模型结果。 |
| `alltoall_modeling/` | 单机 8 NPU HCCL All-to-All 通信时延模型与异步 overlap 模型。 |
| `cpu_npu_e2e/` | 独立 CPU/NPU/All-to-All 融合层，按算子类型路由、插入通信区域、调度 DAG 并输出端到端性能预测。 |
| `tiered_storage_module/` | DLRM embedding 分级存储实验模块，包含 HBM/DRAM/SSD 三级存储、DRAM/SSD 两级存储和 reuse-distance 离线分析工具。 |

## 最终端到端模型

融合模型在 `cpu_npu_e2e/` 中运行，主要消费以下输入：

- `CPU_model_stable/` 生成的 DLRM `op_shapes.csv` 图和 shape 信息。
- `CPU_model_stable` 正式 model bundle 或 CPU prediction CSV。
- `npu_modeling/results/physical_memory_models.joblib.gz` 中的 NPU 算子模型。
- `alltoall_modeling/artifacts/models/*.json` 中的 All-to-All 通信与异步 overlap 参数。

推荐的基础验收命令从仓库根目录进入 `cpu_npu_e2e/` 后执行，所有输入都来自本仓库内的相对路径：

```bash
cd cpu_npu_e2e
python -m cpu_npu_e2e.run_fused_e2e \
  --cases-config configs/e2e_test_cases.yaml \
  --output-dir outputs/e2e_cases
```

严格 CPU bundle 模式需要先按 `CPU_model_stable/docs_delivery/USER_GUIDE.md` 生成或放置项目内 bundle，再传入相对路径：

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --cases-config configs/e2e_test_cases.yaml \
  --cpu-model-bundle ../CPU_model_stable/outputs/<cpu_run>/model_bundles/<cpu_bundle>.pkl \
  --no-cpu-static-fallback \
  --output-dir outputs/e2e_cases_strict_cpu
```

严格模式验收口径：

- `summary.json` 中 `cpu_static_fallback_used=false`。
- `fused_node_predictions.csv` 中 CPU 路由行使用正式 CPU bundle 预测来源。
- `routing_coverage.csv`、`alltoall_region_predictions.csv`、`fused_e2e_combo_metrics.csv` 均正常生成。


## 主要文档

| 文档 | 内容 |
|---|---|
| `cpu_npu_e2e/README.md` | 端到端融合模型快速入口。 |
| `cpu_npu_e2e/docs_delivery/USER_GUIDE.md` | 端到端模型使用说明、输入输出和验收方式。 |
| `cpu_npu_e2e/docs_delivery/INTERNAL_PRINCIPLES.md` | CPU/NPU 路由、NPU 特征适配、All-to-All overlap 和 DAG 调度原理。 |
| `CPU_model_stable/docs_delivery/USER_GUIDE.md` | CPU 模型训练、评估、静态推理和 bundle 生成说明。 |
| `npu_modeling/docs/USER_GUIDE.md` | NPU 模型 artifact、训练复现和验证说明。 |
| `tiered_storage_module/README.md` | 分级存储模块的中文入口和融合边界说明。 |
| `alltoall_modeling/README.md` | All-to-All 建模模块的中文入口。 |
