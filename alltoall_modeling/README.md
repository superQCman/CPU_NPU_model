# All-to-All 建模模块

本目录提供单机 8 NPU HCCL All-to-All 通信建模和异步 overlap 建模，用于 `cpu_npu_e2e/` 中 embedding 分支后的通信区域预测。

## 主要内容

| 路径 | 说明 |
|---|---|
| `src/bw_model.py` | 根据 message size 和对齐类别预测 bus bandwidth。 |
| `src/latency_model.py` | 根据 message size、带宽模型和残差修正预测通信 latency。 |
| `src/async_overlap_model.py` | 根据通信时延和可重叠计算窗口预测可见通信时间。 |
| `artifacts/models/` | 已交付的 bandwidth、latency 和 async overlap 参数。 |
| `docs/USER_GUIDE.md` | 运行、训练、验证和输出说明。 |
| `docs/MODELING_PRINCIPLES.md` | All-to-All 单算子建模和异步掩盖建模原理。 |

## 在端到端模型中的作用

`cpu_npu_e2e` 会根据 DLRM embedding 输出规模估算 All-to-All message size，并调用本目录的参数文件计算：

- 单次 All-to-All 通信 latency。
- 与 bottom/embedding 分支并行时被掩盖的通信时间。
- 端到端调度中需要暴露出来的 visible communication latency。

当前模型口径是单机 8 NPU。如果 NPU 数量、HCCL 版本、跨机拓扑或通信库变化，需要重新采集通信数据并重新生成 `artifacts/models/*.json`。

## 常用命令

训练和 size-holdout 验证：

```bash
cd alltoall_modeling
python3 scripts/run_model_pipeline.py \
  --raw-data data/raw/a2a_32kb_aligned.txt \
  --split-mode size_holdout \
  --test-ratio 0.2
```

异步 overlap 验证：

```bash
cd alltoall_modeling
python3 scripts/run_async_overlap_validation.py
```

更多采集与验证流程见 `docs/USER_GUIDE.md`。
