# CPU/NPU/All-to-All E2E 融合模型使用指南

本指南说明独立融合交付目录 `cpu_npu_e2e` 的目录结构、运行方式、输入输出和验收方法。

## 1. 交付范围

融合模型用于把三类已交付模型组合成端到端性能预测：

- `CPU_model_stable`：CPU 单算子和图结构 artifact。
- `npu_modeling`：NPU semantic operator 物理访存模型。
- `alltoall_modeling`：8 NPU HCCL All-to-All latency 和 async overlap 模型。

目录关系：

```text
CPU_NPU/
  CPU_model_stable/
  npu_modeling/
  alltoall_modeling/
  cpu_npu_e2e/
```

`cpu_npu_e2e` 不修改 `CPU_model_stable`，只读取其 `pipeline_artifacts/**/op_shapes.csv` 和可选 CPU prediction CSV。

## 2. 环境

在交付目录执行：

```bash
cd cpu_npu_e2e
python -m pip install -e .
```

依赖包括 `numpy`、`pandas`、`PyYAML`、`joblib`、`scikit-learn`。当前环境已包含这些包时可以直接运行 CLI。

## 3. 输入检查

确认关键 artifact 存在：

```bash
test -f ../npu_modeling/results/physical_memory_models.joblib -o -f ../npu_modeling/results/physical_memory_models.joblib.gz
test -f ../npu_modeling/models/hardware_descriptor.json
test -f ../alltoall_modeling/artifacts/models/latency_model_params.json
test -f ../alltoall_modeling/artifacts/models/dlrm_async_overlap_multiconfig_model_params.json
find ../CPU_model_stable/pipeline_artifacts -name op_shapes.csv | head
```

如果 `npu_modeling/` 或 `alltoall_modeling/` 不存在，先在仓库根目录下创建英文目录，再把本地 RAR 包内容解压进去。解压工具只要求在 `PATH` 中可用，不依赖固定安装位置：

```bash
mkdir -p npu_modeling alltoall_modeling
bsdtar -xf npu建模.rar -C npu_modeling --strip-components 1
bsdtar -xf alltoall建模.rar -C alltoall_modeling --strip-components 1
```

## 4. 路由配置

默认路由文件：

```text
configs/cpu_npu_op_routing.yaml
```

默认显式映射：

| 后端 | op_type |
|---|---|
| NPU | `Gather`, `ReduceSum`, `Gemm`, `MatMul`, `Relu`, `Sigmoid`, `Add`, `Mul`, `Concat`, `Transpose` |
| CPU | `Reshape`, `Shape`, `Flatten`, `Unsqueeze` |
| Ignore | `Constant` |
| Synthetic | `AllToAll` |

未声明的 `op_type` 会直接报错。新增算子时必须先更新该 YAML。

## 5. 运行

默认 smoke / shape-only 运行：

```bash
cd cpu_npu_e2e
python -m cpu_npu_e2e.run_fused_e2e \
  --cpu-model-root ../CPU_model_stable \
  --npu-root ../npu_modeling \
  --alltoall-root ../alltoall_modeling \
  --routing configs/cpu_npu_op_routing.yaml \
  --output-dir outputs/default
```

默认会自动选择第一个 `CPU_model_stable/pipeline_artifacts/**/op_shapes.csv`。指定某个 combo：

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --shape-csv ../CPU_model_stable/pipeline_artifacts/<source>/<case>/<combo>/op_shapes.csv \
  --output-dir outputs/one_combo
```

正式 CPU 口径建议传入 CPU 单算子预测 CSV：

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --cpu-predictions <combined_predictions_test.csv> \
  --no-cpu-static-fallback \
  --output-dir outputs/with_cpu_predictions
```

当既没有 `--cpu-predictions` 也没有 `--cpu-model-bundle` 时，CPU-routed nodes 使用静态 fallback，仅用于 smoke、接口和调度验证；`summary.json` 中会标记 `cpu_static_fallback_used=true`。

推荐的严格 CPU 接入口是 `CPU_model_stable` 训练后导出的 `model_bundles/*.pkl`。融合代码会从当前 `op_shapes.csv` 反推 DLRM 静态配置，调用 CPU 主线的 `full_graph_76` 静态推理，并严格校验 `node_idx/node_name/op_type` 与 E2E 图对齐。bundle 的生成、命名和放置方式见 `../CPU_model_stable/docs_delivery/USER_GUIDE.md`，NPU artifact 的口径见 `../npu_modeling/docs/USER_GUIDE.md`。

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --shape-csv ../CPU_model_stable/pipeline_artifacts/source_kunpeng920_dataset3_unseen_arch_embedding_size/case_10_run_1_1/bs1056_nip2000/op_shapes.csv \
  --cpu-model-bundle ../CPU_model_stable/outputs/<cpu_run>/model_bundles/<cpu_bundle>.pkl \
  --no-cpu-static-fallback \
  --output-dir outputs/strict_cpu_single
```

批量严格运行：

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --cases-config configs/e2e_test_cases.yaml \
  --cpu-model-bundle ../CPU_model_stable/outputs/<cpu_run>/model_bundles/<cpu_bundle>.pkl \
  --no-cpu-static-fallback \
  --output-dir outputs/e2e_cases_strict_cpu
```

严格 CPU 口径通过时：

- `summary.json::cpu_static_fallback_used=false`
- `summary.json::cpu_prediction_source=cpu_model_bundle_full_graph_76`
- `fused_node_predictions.csv` 中所有 `route_backend=CPU` 的 `prediction_source` 都是 `cpu_model_bundle_full_graph_76`
- 输出目录包含 `cpu_operator_predictions.csv` 和 `cpu_model_summary.json`

批量端到端 case 运行：

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --cases-config configs/e2e_test_cases.yaml \
  --output-dir outputs/e2e_cases
```

`configs/e2e_test_cases.yaml` 当前配置了 3 个来自 `CPU_model_stable/pipeline_artifacts` 的端到端 shape case。批量模式会在 `outputs/e2e_cases/cases/<case_name>/` 下保留每个 case 的完整输出，并在 `outputs/e2e_cases/` 顶层生成带 `e2e_case_name` 字段的聚合 CSV。

## 6. 输出

输出目录包含：

| 文件 | 内容 |
|---|---|
| `fused_node_predictions.csv` | 每个图节点的路由、目标 op、预测时延、调度起止时间 |
| `alltoall_region_predictions.csv` | All-to-All message size、通信时延、隐藏通信和可见通信 |
| `fused_e2e_combo_metrics.csv` | combo 级 E2E latency 汇总 |
| `routing_coverage.csv` | 按 op_type/backend 汇总的覆盖和耗时 |
| `cpu_operator_predictions.csv` | 使用 `--cpu-model-bundle` 时输出的 CPU full_graph_76 正式推理明细 |
| `cpu_model_summary.json` | 使用 `--cpu-model-bundle` 时输出的 CPU 模型、配置和覆盖摘要 |
| `summary.json` | 机器可读运行摘要和 artifact 路径 |

核心结果字段：

```text
fused_e2e_combo_metrics.csv::predicted_e2e_us
```

## 7. 验收

运行测试：

```bash
cd cpu_npu_e2e
python -m pytest
```

测试覆盖：

- 默认路由覆盖当前 CPU 图实际算子。
- 未知算子必须报错。
- CPU、NPU、All-to-All adapter 返回非负 latency。
- 单 case E2E smoke 生成全部交付输出文件。
- CPU bundle 严格模式覆盖所有 CPU-routed 节点且不使用 fallback。
- 多 case E2E 配置生成逐 case 输出和聚合 CSV。

## 8. 常见问题

`routing table is missing op_type`：输入图出现了新算子，先更新 `configs/cpu_npu_op_routing.yaml`。

`missing CPU prediction`：使用了 `--no-cpu-static-fallback`，但 CPU prediction CSV 中没有该 node 的预测。补齐 CSV 或取消该参数做 smoke。

`CPU model bundle full_graph_76 does not align`：`--cpu-model-bundle` 生成的静态图与输入 `op_shapes.csv` 的 `node_idx/node_name/op_type` 不一致。检查该 shape 是否属于当前 DLRM 76 节点静态图口径，或显式指定正确的 `--cpu-model-num-threads` / `--cpu-model-inter-threads`。

NPU import 失败：确认 `../npu_modeling/code` 存在，且当前 Python 环境包含 `scikit-learn`、`joblib`。

NPU 模型 artifact 缺失：确认 `../npu_modeling/results` 下存在 `physical_memory_models.joblib` 或 `physical_memory_models.joblib.gz`。GitHub 交付默认跟踪压缩版 `.joblib.gz`，E2E 适配器会自动加载。

All-to-All 参数缺失：确认 `../alltoall_modeling/artifacts/models` 下有 `bw_model_params.json`、`latency_model_params.json` 和 `dlrm_async_overlap_multiconfig_model_params.json`。
