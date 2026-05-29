# DLRM 分层存储 — 运行指南

## 文件组织

```
dlrm_ascend/
├── dlrm_msprof_mtrain.py        # 三级存储核心模块 (HBM → DRAM → SSD)
├── dlrm_msprof.py               # DLRM NPU 分布式推理
├── two_cache.py                 # 两级存储核心模块 (DRAM → SSD)
├── dlrm_s_pytorch.py            # DLRM 原始 PyTorch 实现
├── reuse_distance.py            # 重用距离离线预测工具
├── dlrm_data_pytorch.py         # PyTorch 数据加载与预处理
├── data_loader_terabyte.py      # Criteo Terabyte 数据集加载
├── data_utils.py                # 数据工具函数
├── extend_distributed.py        # 分布式训练扩展
├── extract_json.py              # JSON 提取工具
├── mlperf_logger.py             # MLPerf 日志工具
├── dist_dlrm.sh                 # 三级存储多卡 NPU 启动脚本
├── requirements.txt             # Python 依赖
├── Dockerfile                   # Docker 构建文件
├── .gitignore
├── README.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── kaggle_dac_loss_accuracy_plots.png
├── terabyte_0875_loss_accuracy_plots.png
├── bench/                        # 基准测试脚本
├── cython/                       # Cython 编译与数据处理
├── input/                        # 输入日志文件
├── optim/                        # 优化器
├── test/                         # 测试脚本
├── tools/                        # 可视化工具
├── torchrec_dlrm/                # TorchRec DLRM 实现
└── tricks/                       # Embedding 优化技巧
```

---

## 一、三级存储 (dlrm_msprof_mtrain.py)

### 1.1 运行方式

通过 `dist_dlrm.sh` 启动 8 卡 NPU 分布式推理：

```bash
bash dist_dlrm.sh
```

等价于：

```bash
torchrun --nproc_per_node=8 dlrm_msprof.py \
    --use-mtrain \
    --use-gpu \
    --inference-only \
    --mini-batch-size=32 \
    --arch-embedding-size="100000-100000-100000-100000-100000-100000-100000-100000" \
    --arch-sparse-feature-size=64000 \
    --arch-mlp-bot="600-500-400-300-64000" \
    --arch-mlp-top="640-600-500-320-1" \
    --data-size=320 \
    --dist-backend="hccl" \
    --num-indices-per-lookup-fixed=True \
    --num-indices-per-lookup=100 \
    --num-workers=8 \
    --profiler-path="./profiling_test_17"
```

`dlrm_msprof_mtrain.py` 在 import 时将 `DLRM_Net.create_emb` 替换为 `MTrainSEmbeddingBag`，`--use-mtrain` 触发三级存储逻辑，其余参数透传给 `dlrm_msprof.py`。

### 1.2 可选参数

**MTrainS 专用：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--use-mtrain` | 关闭 | 启用三级存储 |
| `--hbm-ratio` | 0.01 | HBM 容量占 Embedding 总行数的比例 |
| `--dram-cache-ratio` | 0.05 | DRAM 缓存容量占 Embedding 总行数的比例 |

**继承自 dlrm_msprof.py 的关键参数：**

| 参数 | 含义 |
|------|------|
| `--mini-batch-size` | 每卡 batch 大小 |
| `--inference-only` | 推理模式（不训练） |
| `--arch-embedding-size` | 各 Embedding Table 行数，`-` 分隔 |
| `--arch-sparse-feature-size` | Embedding 向量维度 |
| `--arch-mlp-bot` | 底部 MLP 各层宽度，`-` 分隔 |
| `--arch-mlp-top` | 顶部 MLP 各层宽度，`-` 分隔 |
| `--data-size` | 数据集样本数 |
| `--use-gpu` | 使用 NPU/GPU |
| `--dist-backend` | 分布式后端（NPU 用 `hccl`） |
| `--num-workers` | DataLoader 进程数 |
| `--profiler-path` | profiling 输出目录 |

### 1.3 示例输出

输出文件：`memory_access_timings.log`

格式如下：

```
# Timestamp: 2026-05-29 18:02:40
[Metrics] step=10 HBM_hit=1.0% DRAM_hit=4.7% SSD_hit=94.3%
  HBM -> NPU cache : 14.707 ms/step, 78.1 MB total, 0.5 GB/s
  DRAM -> CPU cache: 30.782 ms/step, 365.3 MB total, 1.2 GB/s
  SSD -> DRAM      : 2044.258 ms/step, 7353.3 MB total, 0.4 GB/s
  DRAM -> HBM      : 4.248 ms/step, 20.0 MB total, 0.5 GB/s
  TOTAL wall time: 2115.063 ms/step
```

程序将输出各段访存路径的时间，以及总时间。

---

## 二、两级存储 (two_cache.py)

### 2.1 运行方式

```bash
python dlrm_s_pytorch.py \
    --inference-only \
    --arch-embedding-size="1000-1000-1000-1000-1000-1000-1000-1000" \
    --arch-sparse-feature-size=64000 \
    --mini-batch-size=32 \
    --data-size=320 \
    --arch-mlp-bot="600-500-400-300-64000" \
    --arch-mlp-top="640-600-500-32" \
    --use-two-level-embedding \
    --dram-cache-ratio=0.3
```

`dlrm_s_pytorch.py` 在 `create_emb` 中检测 `--use-two-level-embedding` 标志，启用时用 `TwoLevelEmbedding` 替换原始的 `nn.EmbeddingBag`。

### 2.2 可选参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--use-two-level-embedding` | 关闭 | 启用两级存储 |
| `--dram-cache-ratio` | 0.05 | DRAM 缓存容量占 Embedding 总行数的比例 |

其余参数（`--arch-embedding-size`、`--mini-batch-size` 等）与 `dlrm_s_pytorch.py` 原生参数一致。

### 2.3 示例输出

输出文件：`memory_access_timings.log`

格式如下：

```
=== Memory Access Timings ===
SSD_to_DRAM: 0.223752 seconds, Data size: 44288000 bytes
DRAM_to_CPU: 0.000035 seconds, Data size: 768000 bytes
```

---

## 三、重用距离预测 (reuse_distance.py)

### 3.1 运行方式

**合成 Trace（Zipfian 分布）：**

```bash
python reuse_distance.py \
    --num-embeddings 100000 \
    --trace-length 500000 \
    --zipf-alpha 1.0 \
    --th-hbm 100 \
    --th-dram 1000
```

**如果使用真实 Trace 文件：**

```bash
python reuse_distance.py \
    --trace-file ./data/my_trace.txt \
    --th-hbm 50 \
    --th-dram 500
```

`trace.txt` 每行一个整数索引，记录 Embedding 的访问顺序。

### 3.2 可选参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--num-embeddings` | 100000 | 唯一 Embedding 数量（合成模式） |
| `--trace-length` | 500000 | 访问序列长度（合成模式） |
| `--zipf-alpha` | 1.0 | Zipf 分布偏度，越大热点越集中 |
| `--trace-file` | — | 真实 trace 文件路径 |
| `--th-hbm` | 100 | HBM 阈值：平均重用距离 ≤ N → HBM |
| `--th-dram` | 1000 | DRAM 阈值：平均重用距离 ≤ N → DRAM |
| `--seed` | 42 | 随机种子（合成模式） |

### 3.3 示例输出

输出格式如下：

```
=================================================================
  Reuse Distance Storage Tier Prediction
=================================================================
  Total accesses:              500,000
  Unique embeddings:            61,965
  Thresholds:             HBM ≤ 100,  DRAM ≤ 1000
=================================================================

  Tier Distribution:
    HBM  :         17 embeddings (  0.0%)  avg RD=57.3
    DRAM :        233 embeddings (  0.4%)  avg RD=584.4
    SSD  :     61,715 embeddings ( 99.6%)  avg RD=23609.3

  Hit Rates (simulated):
    HBM  :  25.07%  ██████████
    DRAM :  21.59%  ████████
    SSD  :  53.34%  █████████████████████
=================================================================
```
输出预测的各级存储中embedding数量，以及预测的hit率
