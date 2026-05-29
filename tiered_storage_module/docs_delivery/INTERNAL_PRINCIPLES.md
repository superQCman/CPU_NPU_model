# DLRM 分层存储 — 基本原理

## 一、三级存储 (dlrm_msprof_mtrain.py)

### 1.1 核心思路

推荐系统中 Embedding Table 往往极大（百万～十亿行），远超单卡 NPU HBM 容量。MTrainS 的核心思想是按访问热度将 Embedding 分层存储：热点放 HBM（NPU 显存）、温数据缓存在 DRAM（CPU 内存）、冷数据存在 SSD，按需逐级加载。

### 1.2 存储层级

| 层级 | 介质 | 容量 | 实现 |
|------|------|------|------|
| Level 0: HBM | NPU 显存 | `num_embeddings × hbm_ratio`（默认 1%） | `nn.Parameter`，参与梯度更新 |
| Level 1: DRAM | CPU 内存 | `num_embeddings × dram_cache_ratio`（默认 5%） | `LRUCache`，OrderedDict 实现 LRU 淘汰 |
| Level 2: SSD | 模拟磁盘 | TB级 | `SSDStorage`，dict + 随机初始化 |

为便于测试，各层级容量占比采用手动配置的方式，通过设置命令行参数调整比例。

### 1.3 查找流程

```
forward(indices)
  │
  ├─ torch.isin(indices, hbm_indices)  → HBM 命中？
  │   └─ Yes: hbm_weight[position] 向量化 gather（NPU 片上读）
  │
  ├─ LRUCache.get(idx)  → DRAM 命中？
  │   └─ Yes: torch.stack(vecs).to(device) 批量 CPU→NPU
  │
  └─ SSDStorage.deferred_init(idx)  → SSD 冷加载
      └─ 随机初始化 + .to(device) + 回写 DRAM 缓存
```

关键设计：HBM 通过 `hbm_indices` 动态维护当前 HBM 中存的是哪些逻辑索引，用 `torch.isin` + `torch.searchsorted` 实现向量化查找。

### 1.4 冷热迁移

每次forward完成时触发一次：

1. 按 `access_freq` 降序排列所有被访问过的索引
2. 取 top-`hbm_rows` 个作为新 HBM 集合
3. `evicted = 旧HBM - 新HBM` → `hbm_weight[pos].cpu()` 写入 DRAM 缓存
4. `promoted = 新HBM - 旧HBM` → 从 DRAM/SSD 取数据，`vec.to(device)` 写入 HBM
5. 频率衰减 `// 2`，淘汰零频项

### 1.5 访存计时

forward 内对 4 段数据通路分别计时，自动打印：

| 计时段 | 测量内容 |
|--------|---------|
| HBM → NPU | hit发生在HBM上的数据传输时间 |
| DRAM → CPU | hit发生在DRAM上的数据传输时间 |
| SSD → DRAM | 数据从SSD迁移到DRAM耗时 |
| DRAM → HBM | 数据从DRAM迁移到HBM耗时 |

---

## 二、两级存储 (two_cache.py)

### 2.1 与三级存储的区别

`two_cache.py` 没有 NPU HBM 层，只有 **DRAM 缓存 → SSD** 两级，运行时回退调用 `dlrm_s_pytorch.py`（纯 CPU/GPU）。可以看作三级存储去掉 HBM 后的简化版本。

### 2.2 存储层级

| 层级 | 介质 | 容量 | 实现 |
|------|------|------|------|
| Level 1: DRAM | CPU 内存 | `num_embeddings × dram_cache_ratio`（默认 5%） | `LRUCache` |
| Level 2: SSD | 模拟磁盘 | TB级 | `SSDStorage` |

### 2.3 查找流程

```
forward(indices)
  │
  ├─ LRUCache.contains(idx)  → DRAM 命中？
  │   └─ Yes: 从缓存取，.to(device)
  │
  └─ SSDStorage.deferred_init(idx)  → SSD 冷加载
      └─ 随机初始化 + .to(device) + 回写 DRAM
```

与三级存储的关键差异：

- **无 HBM 层**：没有 `nn.Parameter`、没有梯度直接在 NPU 上更新
- **无动态迁移**：DRAM 缓存纯靠 LRU 自然淘汰，不做基于频率的主动晋升/驱逐
- **支持 offsets 聚合**：`forward` 可接收 `offsets` 参数做 EmbeddingBag 式的分段求和

### 2.4 适用场景

适合纯 CPU 时的功能验证，或作为三级存储的对照组，运行依赖 `dlrm_s_pytorch.py`。

### 2.5 访存计时

forward 内对 2 段数据通路分别计时，自动打印：

| 计时段 | 测量内容 |
|--------|---------|
| DRAM → CPU | hit发生在DRAM上的数据传输时间 |
| SSD → DRAM | 数据从SSD迁移到DRAM耗时 |

---

## 三、基于重用距离的存储层级预测 (reuse_distance.py)

### 3.1 解决的问题

已知一个 Embedding 的访问序列（训练或推理时的索引访问记录），如何**预测每个 Embedding 应该放在哪一级存储**，进而**预测各级访存路径的命中比例**？

这是对存储层级行为的离线建模——不依赖实际部署，仅凭访问模式就能估算出 HBM/DRAM/SSD 各自的命中率，用于架构评估和性能建模。

### 3.2 核心概念：重用距离

**重用距离（Reuse Distance）**：同一个 Embedding 两次相邻访问之间，夹了多少个**不同**的 Embedding。

```
LRU 栈: [栈底 ... 栈顶]  （栈顶 = 最近访问）

访问序列: [A, B, C, A, D, B, A]

访问 A（第 4 步）时:
  栈 = [A, B, C]  （栈顶为 C）
  A 在栈中位置 = 从栈顶往下数第 3 个 → 重用距离 = 3
  将 A 移至栈顶 → 栈 = [B, C, A]

访问 B（第 6 步）时:
  栈 = [B, C, A, D]
  B 在栈中位置 = 从栈顶往下数第 4 个 → 重用距离 = 4
```

- 重用距离**小** → 频繁被访问 → 热数据 → 预测放 HBM
- 重用距离**中** → 偶尔被访问 → 温数据 → 预测放 DRAM
- 重用距离**大 / 只访问一次** → 冷数据 → 预测放 SSD

### 3.3 算法流程

```
输入: 访问序列 S = [e1, e2, ..., eN], 阈值 THBM, TDRAM

Step 1: 遍历 S，维护 LRU 栈
  for each ei in S:
      if ei 不在栈中:
          入栈（首次访问，无重用距离）
      else:
          di = ei 在栈中位置（栈顶 = 1）
          记录 rd_map[ei].append(di)
          将 ei 移至栈顶

Step 2: 对每个 Embedding 分配层级
  for each 唯一 Embedding e:
      d̄(e) = mean(rd_map[e])     # 平均重用距离
      if   d̄(e) ≤ THBM   → L(e) = HBM
      elif d̄(e) ≤ TDRAM  → L(e) = DRAM
      else                → L(e) = SSD

Step 3: 统计各级命中比例
  回放 S，按 L(e) 计数:
      h_HBM  = Σ [L(ei) == HBM]   / N
      h_DRAM = Σ [L(ei) == DRAM]  / N
      h_SSD  = Σ [L(ei) == SSD]   / N
```
