# DLRM 分级存储推理延迟建模

## 1. 推荐业务与 Embedding 存储访问特征

推荐模型的 Embedding 表总规模通常远超 NPU/GPU 片上 HBM 容量，因此依赖分级存储架构——将热数据置于片上 HBM，温数据缓存于主机 DRAM，冷数据存放于 NVMe SSD，按需逐级加载。每次前向传播中，模型需对多张 Embedding 表执行批量索引查询，但单次查询仅触及 Embedding 总行数的极小比例。

Embedding 访问呈现两个根本性特征——长尾偏斜与时间局部性。其一，Embedding 索引的访问频率严格遵循 Zipf 定律 $p(i) \propto 1 / i^{\alpha}$：少量头部 Embedding 承载了绝大部分总访问量，而大量长尾 Embedding 在整个训练周期内仅被访问数次。$\alpha$ 越大，访问越集中，小容量 HBM 即可捕获高命中率；$\alpha$ 越小，访问趋于均匀，各级缓存的命中率梯度平缓。其二，Embedding 访问具有显著的时空复用模式——同一 batch 内热门索引被多个样本重复引用，相邻 batch 间也存在跨 batch 复用。该特征可由重用距离（Reuse Distance）量化，定义为同一索引相邻两次访问之间所经历的不同索引数量：热数据重用距离小，冷数据重用距离大。基于 LRU 栈对访问序列建模，可按平均重用距离将 Embedding 分配至不同存储层级。尽管 SSD 命中次数占比在各层级中最低，但因其带宽瓶颈与高排队延迟，SSD 阶段的耗时占比往往成为端到端推理延迟的主导因素。

## 2. 分级存储架构与数据流转流程

### 分级存储架构

针对 Embedding 表总规模远超片上存储容量的矛盾，系统采用三级存储架构组织数据。第一级为片上 HBM（High Bandwidth Memory），直接面向 NPU 计算单元，存放访问频率最高的热 Embedding 向量，提供最低延迟和最高带宽的数据通路。第二级为主机 DRAM 缓存，用作 HBM 的后备缓冲，以 LRU 策略管理，存放次热温数据，经 PCIe 总线与设备侧交换数据。第三级为 NVMe SSD 持久化存储，作为全量 Embedding 的最终驻留地，冷数据按需加载至 DRAM 后逐级向上迁移。各级容量由配置参数决定：HBM 占比控制片上热数据规模，DRAM 缓存占比控制主机侧缓冲容量，剩余长尾 Embedding 全部驻留 SSD。初始化阶段，系统将排名靠前的 Embedding 行预加载入 HBM，次靠前的行预取至 DRAM 缓存，以消除首次前向传播时的 SSD 冷启动开销。

### 数据流转流程

单次前向传播中，Embedding 索引沿四级通路自下而上完成数据搬移。首先，查询 HBM 逻辑索引集合，命中则通过向量化批量 gather 操作将对应的 Embedding 向量直接从片上 HBM 传入 NPU 计算单元（HBM → NPU）。未命中 HBM 的索引落入 DRAM 查找路径：若命中 DRAM 缓存，经由 PCIe 总线将向量从主机内存搬运至设备侧（DRAM → CPU）；若 DRAM 也未命中，则触发 SSD 读取——冷数据从 NVMe 固态盘按需加载，传至设备后同步回填至 DRAM 缓存（SSD → DRAM），期间可能触发 LRU 淘汰。此外，系统以固定步数为周期执行冷热迁移：统计近期访问频率（经衰减加权以捕捉时效性），将访问频率最高的索引集合保留或晋升至 HBM，驱逐低频数据至 DRAM，该迁移过程产生 DRAM → HBM 通路的数据搬移开销。各段数据搬移的延迟均可由统一模型 $T = \alpha \cdot n + \text{DataSize} / \text{BW}$ 刻画，其中固定开销 $\alpha$ 涵盖设备唤醒、队列调度等延迟，带宽项反映各通路的物理传输能力差异。四级通路的带宽与延迟特征存在数量级差距，SSD → DRAM 段因其低带宽和高固定开销成为端到端访存延迟的主要贡献者。

## 3. 建模对象、输入与输出

### 建模对象

本建模工作针对 DLRM 三级存储架构下的内存访问延迟进行预测，建模对象为单次推理中 Embedding 索引查询所触发的全链路数据搬移耗时。模型将四级数据通路——HBM → NPU、DRAM → CPU、SSD → DRAM 以及冷热迁移产生的 DRAM → HBM——统一纳入考量，覆盖从片上高速缓存到 NVMe 冷存储的完整访存路径。模型核心为 $\alpha + \text{DataSize} / \text{BW}$ 形式的线性延迟方程，其中 $\alpha$ 表征设备唤醒与队列调度的固定开销，带宽项刻画各通路物理传输能力决定的吞吐上限。

核心设计原则是**全链路零 Profiling**：各级命中率不依赖 Trace 模拟或 LRU 栈回放，而是由 Embedding 访问的 Zipf 分布参数 $\alpha$ 经累积概率公式直接解析给出。$\alpha$ 来源于线上请求日志拟合、训练数据统计或业务先验，均无需启动 DLRM 模型。

令 $i \in \{1, 2, 3, 4\}$ 分别对应四段通路：

- $i = 1$：HBM → NPU
- $i = 2$：DRAM → CPU
- $i = 3$：SSD → DRAM
- $i = 4$：DRAM → HBM

每段通路的访存时间统一表示为：

$$T_i = \alpha_i \cdot n_i + \frac{n_i \cdot m \cdot b}{\text{BW}_i}$$

总访存延迟为四段之和：

$$T_{\text{total}} = \sum_{i=1}^{4} T_i$$

### 输入参数

| 参数 | 说明 |
|------|------|
| $\alpha_{\text{zipf}}$ | Embedding 访问的 Zipf 偏斜参数，反映线上流量的访问集中度。典型值 0.8–1.2，默认取 1.0。用户可通过请求日志拟合或业务先验提供 |
| $C_{\text{HBM}}$ | HBM 物理容量（GB） |
| $C_{\text{DRAM}}$ | DRAM 缓存容量（GB） |
| $\text{BW}_i$ | 四段链路 $i = 1,2,3,4$ 的有效带宽（GB/s） |
| $\alpha_i$ | 四段链路 $i = 1,2,3,4$ 的每操作固定延迟（μs/op），涵盖设备唤醒、队列调度等开销 |
| $K$ | 单次推理的 Embedding 总查询量（$B \times k \times T$） |
| $m$ | Embedding 向量维度 |
| $b$ | 单元素字节数，由精度类型决定（float32 = 4，bf16/fp16 = 2） |

### 输出

| 输出 | 说明 |
|------|------|
| $r_{\text{HBM}}$ | HBM 命中率 |
| $r_{\text{DRAM}}$ | DRAM 缓存命中率 |
| $r_{\text{SSD}}$ | SSD 命中率 |
| $T_1$ | HBM → NPU 通路访存耗时 |
| $T_2$ | DRAM → CPU 通路访存耗时 |
| $T_3$ | SSD → DRAM 通路访存耗时 |
| $T_4$ | DRAM → HBM 迁移通路访存耗时 |
| $T_{\text{total}}$ | 四级通路总访存延迟 |

## 4. Zipf 参数 $\alpha$ 的获取

$\alpha$ 是**流量属性**而非**模型属性**，获取过程不需要运行 DLRM 模型。按用户条件分三级获取：

| 等级 | 条件 | 方式 | 精度 |
|------|------|------|:---:|
| A | 有线上推理日志 | 提取请求中的类别特征值，统计频次分布，幂律拟合 | 最高 |
| B | 无日志但有训练数据 | 读取训练数据中稀疏特征列（$X_{\text{cat}}$）的频次分布作为线上流量的近似 | 较高 |
| C | 仅有模型配置 | 默认值 $\alpha = 1.0$，辅以灵敏度分析覆盖 $[0.8, 1.2]$ 区间 | 参考级 |

**A 级：请求日志拟合**

取一小段线上推理请求日志，提取每个请求携带的类别特征值，统计各索引的出现频次，按频次降序排列后做幂律拟合：

```python
from collections import Counter
import numpy as np

indices = [...]  # 从请求日志提取的索引序列
freq = Counter(indices)
sorted_freq = sorted(freq.values(), reverse=True)

ranks = np.arange(1, len(sorted_freq) + 1)
alpha = -np.polyfit(np.log(ranks), np.log(sorted_freq), 1)[0]
```

**B 级：训练数据统计**

训练数据和推理流量反映同一物品/用户 ID 池，频次分布高度相关。直接读取训练数据 .npz 文件中的 $X_{\text{cat}}$ 字段，按 A 级相同方式拟合，可作为上线后流量的合理近似。

**C 级：默认值 + 灵敏度分析**

推荐业务中 $\alpha$ 集中于 0.8–1.2，默认取 1.0。模型同时提供灵敏度分析：遍历 $\alpha \in [0.8, 1.2]$，输出 $T_{\text{total}}$ 的变化区间。

## 5. Zipf 解析 → 存储层级预测

### 第一步：计算访问概率分布

给定 Zipf 参数 $\alpha$ 和 Embedding 总行数 $N$，索引 $i$（按访问频次降序排列，$i = 1$ 为最热）的访问概率为：

$$p_i = \frac{1 / i^{\alpha}}{\sum_{j=1}^{N} 1 / j^{\alpha}} = \frac{1 / i^{\alpha}}{H_N(\alpha)}$$

其中 $H_N(\alpha) = \sum_{j=1}^{N} 1 / j^{\alpha}$ 为广义调和数。此步仅需 $\alpha$ 和 $N$，不涉及任何 Trace 或 Profiling。

### 第二步：容量转行数

由硬件容量和模型参数换算出各级存储可容纳的 Embedding 行数：

$$N_{\text{HBM}} = \frac{C_{\text{HBM}}}{m \cdot b}, \quad N_{\text{DRAM}} = \frac{C_{\text{DRAM}}}{m \cdot b}$$

### 第三步：按容量切割累积概率

各级命中率即为对应区间内访问概率之和：

$$r_{\text{HBM}} = \sum_{i=1}^{N_{\text{HBM}}} p_i = \frac{H_{N_{\text{HBM}}}(\alpha)}{H_N(\alpha)}$$

$$r_{\text{DRAM}} = \sum_{i=N_{\text{HBM}}+1}^{N_{\text{HBM}}+N_{\text{DRAM}}} p_i = \frac{H_{N_{\text{HBM}}+N_{\text{DRAM}}}(\alpha) - H_{N_{\text{HBM}}}(\alpha)}{H_N(\alpha)}$$

$$r_{\text{SSD}} = \sum_{i=N_{\text{HBM}}+N_{\text{DRAM}}+1}^{N} p_i = 1 - r_{\text{HBM}} - r_{\text{DRAM}}$$

### 第四步：输出存储层级分布

| 输出 | 含义 |
|------|------|
| $r_{\text{HBM}}$ | HBM 命中率 |
| $r_{\text{DRAM}}$ | DRAM 缓存命中率 |
| $r_{\text{SSD}}$ | SSD 命中率 |
| $N_{\text{HBM}}$, $N_{\text{DRAM}}$, $N - N_{\text{HBM}} - N_{\text{DRAM}}$ | 各级分配的 Embedding 行数 |

流程输入输出关系：

$$\underbrace{(\alpha, \; N, \; C_{\text{HBM}}, \; C_{\text{DRAM}}, \; m, \; b)}_{\text{输入}} \;\longrightarrow\; \underbrace{(r_{\text{HBM}}, \; r_{\text{DRAM}}, \; r_{\text{SSD}})}_{\text{存储层级预测}}$$

### 累积分布切割示意

将 $N_{\text{HBM}}$ 和 $N_{\text{HBM}} + N_{\text{DRAM}}$ 作为累积分布曲线上的两个切割点，直接读取纵轴值即得命中率：

```
累积概率 P(i)
  1.0 ┤                                    ╭──
      ┤                              ╭─────╯
      ┤                        ╭─────╯
r_HBM+r_DRAM ┤ - - - - - - - - ╋ - - - ─ ─ ─ ─   ← 第二切割点 N_HBM+N_DRAM
      ┤                   ╭──╯  │
 r_HBM ┤ - - - - - - - - ╋ - - -│- - ─ ─ ─ ─ ─   ← 第一切割点 N_HBM
      ┤              ╭──╯│      │
      ┤         ╭───╯   │      │
      ┤    ╭───╯        │      │
    0 ┼────╯────────────┼──────┼──────────────────→ 索引 i (按频次降序)
      0               N_HBM  N_HBM+N_DRAM          N

      ├── r_HBM ──┤├ r_DRAM ┤├──── r_SSD ────┤
```

曲线形状由 $\alpha$ 控制：$\alpha$ 越大曲线越陡峭，头部越集中；$\alpha$ 越小曲线越平坦，尾部越重。第一条竖线（$N_{\text{HBM}}$）能捕获的面积极大程度上决定了总延迟的优劣。

## 6. 四级通路延迟模型

### 各级访问次数

单次推理的 Embedding 总查询量：

$$K = B \times k \times T$$

各级命中对应的访问次数：

$$n_1 = n_{\text{HBM}} = r_{\text{HBM}} \cdot K$$
$$n_2 = n_{\text{DRAM}} = r_{\text{DRAM}} \cdot K$$
$$n_3 = n_{\text{SSD}} = r_{\text{SSD}} \cdot K$$

$n_4$ 为冷热迁移行数，可取 $N_{\text{HBM}} \cdot \eta$（$\eta$ 为迁移比例），或在解析模式下按保守估计。

### 各段延迟与总延迟

$$T_i = \alpha_i \cdot n_i + \frac{n_i \cdot m \cdot b}{\text{BW}_i}, \quad i = 1, 2, 3, 4$$

$$T_{\text{total}} = \sum_{i=1}^{4} T_i$$

## 7. 双模式对比

| 维度 | Trace 模式 | Zipf 解析模式 |
|------|------|------|
| 输入 | 访问序列 $S$ | Zipf 参数 $\alpha$ |
| 命中率来源 | LRU 栈模拟重用距离 | 累积概率公式解析 |
| 计算代价 | $O(\text{trace\_length})$ | $O(1)$ |
| 需要 Profiling | 是（需 Trace） | 否 |
| 需要启动 DLRM | 否（Trace 可从请求日志获取） | 否 |
| 捕获时间局部性 | 是 | 否（IRM 假设） |
| 适用场景 | 有真实 Trace 可回放 | 选型评估、快速估算、容量规划 |

两种模式输出格式一致，下游延迟模型无需修改。Zipf 解析模式因忽略时间局部性，倾向于低估命中率、高估延迟，在容量规划场景下给出保守上界。

## 8. 整体建模框架

```
模型配置 (N, m, T, dtype)  +  硬件规格 (C_HBM, C_DRAM, BW_i, α_link,i)
              │                              │
              ▼                              ▼
    Zipf α ← 请求日志 / 训练数据 / 默认值     N_HBM = C_HBM / (m·b)
    (外部获取，不跑模型)                      N_DRAM = C_DRAM / (m·b)
              │                              │
              ▼                              ▼
         r_HBM = H_{N_HBM}(α) / H_N(α)      n_i = r_i × K
         r_DRAM = H_{N_HBM+N_DRAM}(α)-...    n_4 ≈ N_HBM × η
         r_SSD = 1 - r_HBM - r_DRAM
              │                              │
              └──────────┬───────────────────┘
                         ▼
                T_i = α_link,i × n_i + n_i × m × b / BW_i
                         │
                         ▼
                T_total = Σ T_i
```
