# DLRM 分级存储模块 v2


## 目录结构

| 路径 | 说明 |
|---|---|
| `code/dlrm_ascend/` | DLRM Ascend/NPU 代码和分级存储实现。 |
| `code/dlrm_ascend/dlrm_msprof_mtrain.py` | HBM -> DRAM -> SSD 三级存储核心实现。 |
| `code/dlrm_ascend/two_cache.py` | DRAM -> SSD 两级存储核心实现。 |
| `code/dlrm_ascend/tier_prediction/reuse_distance.py` | Embedding 访问序列 reuse-distance 离线分析工具。 |
| `解释文档/使用方法.md` | 运行方式、参数和日志说明。 |
| `解释文档/原理介绍.md` | 分级存储和 reuse-distance 分析原理。 |

v2 新增 `query/` 查询耗时模型、`migration/` 迁移带宽测量、`update/` 写回模型，以及 `ssd_model.py`。详细流程见 [modeling_pipeline.md](code/dlrm_ascend/modeling_pipeline.md)。

## 主要能力

- 三级存储：把热点 embedding 放在 HBM，温数据缓存在 DRAM，冷数据放在 SSD 模拟层，并记录 HBM/DRAM/SSD 各路径耗时。
- 两级存储：去掉 HBM，只保留 DRAM cache 和 SSD，适合做功能验证或三级存储对照组。
- Reuse-distance 分析：根据 embedding 访问序列估计 HBM、DRAM、SSD 的层级分布和命中比例。

## 快速入口

三级存储示例：

```bash
cd tiered_storage_module/code/dlrm_ascend
bash dist_dlrm.sh
```

两级存储示例：

```bash
cd tiered_storage_module/code/dlrm_ascend
python dlrm_s_pytorch.py \
  --inference-only \
  --use-two-level-embedding \
  --dram-cache-ratio=0.3
```

Reuse-distance 分析示例：

```bash
cd tiered_storage_module/code/dlrm_ascend
python tier_prediction/reuse_distance.py \
  --num-embeddings 100000 \
  --trace-length 500000 \
  --zipf-alpha 1.0 \
  --th-hbm 100 \
  --th-dram 1000
```

完整参数和输出日志格式见 `解释文档/使用方法.md`。
