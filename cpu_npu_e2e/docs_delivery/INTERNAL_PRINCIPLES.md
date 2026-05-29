# CPU/NPU/All-to-All E2E 融合模型内部原理

本文件说明 `cpu_npu_e2e` 的内部机制、数据流和模型边界。使用命令见 `USER_GUIDE.md`。

## 1. 总体链路

融合链路：

```text
op_shapes.csv
  -> 构建 DAG
  -> 显式 op_type 路由
  -> CPU/NPU 单算子预测
  -> embedding terminal tensor 推导 All-to-All message bytes
  -> All-to-All latency + async overlap
  -> DAG 调度
  -> predicted_e2e_us
```

融合模型只做 artifact 级推理，不重新训练 CPU、NPU 或 All-to-All 模型。

## 2. 路由原则

路由完全由 `configs/cpu_npu_op_routing.yaml` 决定。

| 路由 | 处理 |
|---|---|
| `NPU` | 将 CPU/ONNX `op_type` 映射到 NPU semantic op，再调用 NPU 模型 |
| `CPU` | 从 CPU prediction CSV 读取预测；没有 CSV 时可用静态 fallback 做 smoke |
| `IGNORE` | 对 `Constant` 等无运行时节点给 0 latency |
| `AllToAll` | 图中不存在的 synthetic communication region |

未声明算子是硬错误。这保证实验结果可审计，不会因为隐式 fallback 改变 CPU/NPU 分配。

## 3. NPU 特征适配

NPU 模型训练在 semantic operator schema 上。融合模块从 `op_shapes.csv` 提取输入/输出 shape，并转换成 NPU 交付代码 `build_record()` 需要的参数：

| CPU/ONNX op | NPU semantic op | 关键参数 |
|---|---|---|
| `Gemm` | `GemmBias` | `m`, `k`, `n` |
| `MatMul` | `MatMulV2` | `m`, `k`, `n` |
| `Gather` | `Gather` | table `rows/cols`, `index_count` |
| `ReduceSum` | `ReduceSum` | `d0/d1/d2`, `axis` |
| `Relu/Sigmoid/Add/Mul` | 同名语义算子 | `rows`, `cols` |
| `Concat` | `ConcatD` | `batch`, input feature sizes |
| `Transpose` | `Transpose` | `d0/d1/d2`, `perm` |

随后复用 NPU 包中的：

```text
build_semantic_operator_dataset.build_record
train_physical_memory_models.add_taxonomy
train_physical_memory_models.add_hierarchical_memory_features
results/physical_memory_models.joblib
```

GitHub 交付中同时支持压缩 artifact：

```text
results/physical_memory_models.joblib.gz
```

适配器优先加载 `.joblib`，不存在时自动加载 `.joblib.gz`。

推理时按 artifact 中保存的 `feature_columns` 对齐 one-hot 和数值特征。

## 4. CPU 侧输入

CPU 侧支持两种正式输入，推荐使用模型 bundle。

### 4.1 CPU bundle 推理

`--cpu-model-bundle` 接收 `CPU_model_stable` 训练后导出的 `model_bundles/*.pkl`。融合层不会重新训练 CPU 模型，而是复用 CPU 主线静态推理模块：

```text
calibrator.static_feature_mapping.core.build_static_feature_frame
calibrator.static_feature_mapping.predict_from_config._load_model_bundle
calibrator.static_feature_mapping.predict_from_config._predict_with_bundle
```

接入流程：

1. 从当前 `op_shapes.csv` 的唯一节点图反推 DLRM 静态配置：
   - `batch_size` / `num_indices_per_lookup` 来自 embedding `Reshape` 输出。
   - `arch_embedding_size` 来自 8 个 `Gather` table shape。
   - `arch_mlp_bot` 来自 bottom MLP 两个 `Gemm` 的输入/输出维度。
   - `arch_mlp_top` 来自 top MLP 三个 `Gemm` 的输出维度。
   - `num_threads` / `inter_threads` 默认从路径里的 `case_<id>_run_<num_threads>_<inter_threads>` 解析，也可由 CLI 显式覆盖。
2. 调用 CPU 主线 `full_graph_76` 静态推理，得到 76 个节点的 `pred_us`。
3. 严格校验 `op_idx/node_name/op_type` 与 E2E `op_shapes.csv` 构建出的 76 节点图一致。
4. 只把路由到 CPU 的节点 latency 写入融合调度，预测来源标记为 `cpu_model_bundle_full_graph_76`。

使用 `--no-cpu-static-fallback` 时，任何 CPU-routed 节点缺失都会报错。

### 4.2 CPU prediction CSV

CSV 路径读取 prediction rows，要求包含：

- `op_idx` 或可解析 `opidx:<id>` 的 `row_uid`。
- `pred_us`、`prediction_us`、`predicted_us`、`predicted_latency_us`、`prediction` 或 `y_pred` 之一。

### 4.3 CPU fallback

如果既未提供 CPU bundle，也未提供 CSV，静态 fallback 只根据 shape、bytes、粗略 FLOPs 和 dispatch floor 估计 CPU-routed nodes。该 fallback 用于集成 smoke，不作为正式性能结论。

## 5. All-to-All 建模

All-to-All communication region 是 synthetic node，不来自 `op_shapes.csv`。

message size 默认取 embedding 分支 terminal nodes 的输出 tensor bytes 总和。terminal node 指 embedding partition 中没有 embedding successor 的节点，通常是每个 embedding branch 的 `ReduceSum` 输出。

单算子通信 latency 读取：

```text
alltoall_modeling/artifacts/models/bw_model_params.json
alltoall_modeling/artifacts/models/latency_model_params.json
```

融合代码复现交付模型的 alignment group、sigmoid bandwidth、三段 latency 和 residual correction 参数。async overlap 读取：

```text
alltoall_modeling/artifacts/models/dlrm_async_overlap_multiconfig_model_params.json
```

核心公式：

```text
overlap_bound = min(T_comm, T_compute_window)
hidden_comm = eta * overlap_bound
visible_comm = T_comm - hidden_comm
```

## 6. DAG 调度

图结构来自 `op_shapes.csv` 的 `producer_node` / `consumer_nodes` 字段。当前分区：

| partition | 判定 |
|---|---|
| `bottom` | node name 以 `/bot_l/` 开头 |
| `embedding` | node name 匹配 `/emb_l<idx>/` 且 op 是 `Gather/Reshape/ReduceSum` |
| `tail` | 其他非 Constant 节点 |
| `constant` | `op_type=Constant` |

调度分两段：

1. 调度 bottom 和 embedding partition，得到 `bottom_end_us` 和 `embedding_end_us`。
2. 在 embedding 完成后插入 All-to-All，可见通信与 bottom compute window 重叠。
3. tail partition 等待原始依赖和 All-to-All barrier 后继续调度。

默认 `--cpu-slots=1`、`--npu-slots=1`。这表示每个后端按单执行流保守调度；需要模拟更多并发时可显式提高 slots。

## 7. 多 case 聚合

批量入口读取 `configs/e2e_test_cases.yaml`。每个 case 指向一个 `op_shapes.csv`，路径可以是绝对路径，也可以是相对 `CPU_model_stable` 的路径。

调度和预测仍逐 case 独立执行，避免跨 combo 共享状态。批量层只负责：

- 在 `cases/<case_name>/` 下保留单 case 的完整输出。
- 给每个单 case CSV 增加 `e2e_case_name` 字段。
- 在顶层拼接生成聚合 `fused_node_predictions.csv`、`alltoall_region_predictions.csv`、`fused_e2e_combo_metrics.csv` 和 `routing_coverage.csv`。
- 在顶层 `summary.json` 记录 `case_count`、每个 case 的 combo、预测时延和输出目录。

## 8. 模型边界

- 不修改 `CPU_model_stable`。
- 不重新训练任何模型。
- 默认 NPU 和 All-to-All 硬件口径来自交付包中的 8 NPU 参数。
- 当前只支持 `op_shapes.csv` 中可推导形状的算子；新增算子需要同时更新路由和 NPU 参数抽取逻辑。
- 没有 CPU prediction CSV 时的 CPU fallback 不是正式性能模型，只用于 smoke 和交付接口自检。
