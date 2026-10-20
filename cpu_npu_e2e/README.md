# CPU/NPU/All-to-All E2E 融合模型

本目录是独立交付目录，和 `CPU_model_stable`、`npu_modeling`、`alltoall_modeling` 并列。融合模型不修改 CPU 主线目录，只消费三个已交付模型的 artifact。

正式文档：

| 文档 | 内容 |
|---|---|
| [docs_delivery/USER_GUIDE.md](docs_delivery/USER_GUIDE.md) | 环境、目录、运行命令、输入输出和验收方式 |
| [docs_delivery/INTERNAL_PRINCIPLES.md](docs_delivery/INTERNAL_PRINCIPLES.md) | 路由、NPU 特征适配、All-to-All overlap 和 DAG 调度原理 |

快速运行：

```bash
cd cpu_npu_e2e
python -m cpu_npu_e2e.run_fused_e2e \
  --cpu-model-root ../CPU_model_stable \
  --npu-root ../npu_modeling \
  --alltoall-root ../alltoall_modeling \
  --routing configs/cpu_npu_op_routing.yaml \
  --output-dir outputs/default
```

批量端到端测试：

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --cases-config configs/e2e_test_cases.yaml \
  --output-dir outputs/e2e_cases
```

若有正式 CPU 单算子预测 CSV，建议显式传入：

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --cpu-predictions <combined_predictions_test.csv> \
  --no-cpu-static-fallback \
  --output-dir outputs/with_cpu_predictions
```

仓库在 `CPU_model_stable/models/` 下包含两个平台的正式 CPU bundle（无损压缩为 `.pkl.xz`，推理时直接读取）：

```bash
python -m cpu_npu_e2e.run_fused_e2e \
  --cases-config configs/e2e_test_cases.yaml \
  --cpu-model-bundle ../CPU_model_stable/models/s920x20/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl.xz \
  --no-cpu-static-fallback \
  --output-dir outputs/e2e_cases_strict_cpu
```

CPU bundle 清单、SHA256 和两个平台的选择方式见 `../CPU_model_stable/models/README.md`；重新训练说明见 CPU 使用指南。NPU artifact 口径见 `../npu_modeling/docs/USER_GUIDE.md`。原始 `.pkl` 仍可通过同一参数使用。
