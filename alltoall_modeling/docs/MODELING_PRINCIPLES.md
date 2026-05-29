# All-to-All 单算子建模与异步掩盖建模原理

## 1. 建模对象与边界

本交付针对单机 8 NPU 上的 HCCL All-to-All 通信建立两层模型：

- 单算子模型：只描述一次 All-to-All 通信本身的带宽和时延。
- 异步掩盖模型：描述 All-to-All 与计算窗口并行时，通信中有多少被计算掩盖，以及异步区域总耗时。

两者分层建模的原因是：

- 单算子通信性能主要由消息大小、对齐方式、通信库实现和硬件拓扑决定。
- 异步掩盖效果还取决于应用中通信和计算的调度关系，不能只靠单算子时延解释。
- 将 `T_comm` 与 overlap 公式拆开，可以复用同一单算子模型，并按不同 workload 校准 `eta`。

## 2. 数据口径

训练数据来自 HCCL-test `alltoall_test` 输出：

```text
data_size(Bytes) | aveg_time(us) | alg_bandwidth(GB/s) | check_result
```

字段含义：

- `size`：每次 All-to-All 的消息大小，单位 byte。
- `latency`：HCCL-test 报告的平均通信时延，单位 microsecond。
- `alg_bw`：算法带宽，单位 GB/s。
- `bus_bw`：模型使用的总线带宽，由 `alg_bw * MUL_FACTOR_FUNCS['all_to_all'](num_npus)` 得到。

当前 `num_npus=8`。如果 NPU 数量变化，All-to-All bus bandwidth factor 和通信路径都会变化，不能直接复用当前参数。

## 3. 对齐分组

All-to-All 在不同消息对齐方式下表现出明显不同的带宽平台和跌落区间，因此先按 size 划分三类：

| Group | 判定条件 | 性能特征 |
|---|---|---|
| Group1 | `size % 128KB == 0` | 高带宽平台，当前约 115 GB/s |
| Group2 | `size % 128KB != 0` 且 `size % 32KB == 0` | 中带宽平台，当前约 85 GB/s |
| Group3 | `size % 32KB != 0` | 小消息段接近 Group1，大消息后跌落到低平台 |

这种分组不是人为分类展示，而是建模主特征。若忽略对齐类别，单一曲线会同时拟合高平台、中平台和非对齐跌落，导致系统性误差。

## 4. 带宽模型

### 4.1 核心形式

带宽模型在 `src/bw_model.py` 中实现。主趋势使用 log-size 空间的 sigmoid：

```text
x = log2(size_bytes)
log10(BW) = sigmoid(x; L, x0, k, b)
```

其中：

- `L` 控制上升幅度。
- `x0` 控制上升拐点。
- `k` 控制上升斜率。
- `b` 控制小消息低带宽基线。

通信带宽随 size 增大通常经历：小消息低带宽、快速上升、平台饱和。sigmoid 可以较好表达这种 S 型趋势。

### 4.2 凹陷修正

大消息区间存在非单调带宽凹陷，因此在 sigmoid 趋势上叠加 Gaussian dip：

```text
BW_log(size) = sigmoid(log2(size)) - A * exp(-0.5 * ((log2(size) - mu) / sigma)^2)
```

其中：

- `A`：凹陷深度。
- `mu`：凹陷中心。
- `sigma`：凹陷宽度。

代码流程：

- 先拟合鲁棒 sigmoid 趋势。
- 通过趋势残差自动检测 dip range。
- 在 dip range 内拟合 sigmoid + Gaussian dip。
- 非 dip 饱和段 clip 到检测出的平台带宽。

### 4.3 Group3 非对齐建模

Group3 的特点是前段与 Group1 上升趋势接近，但大消息后跌落到低平台。代码中采用数据驱动检测：

- 从大消息段开始做滑动中位数平滑。
- 找峰值位置。
- 在峰值后寻找持续下降并稳定到低平台的位置。
- `drop_size` 之后直接使用 `sat_bw_low3`。

因此 Group3 模型是分段的：

```text
size < drop_size: 复用 Group1 final/trend
size >= drop_size: 使用 Group3 低平台 sat_bw_low3
```

### 4.4 输出参数

带宽参数保存到：

```text
artifacts/models/bw_model_params.json
```

关键字段：

- `popt_high_trend`、`popt_mid_trend`：Group1/Group2 sigmoid 趋势参数。
- `popt_high_final`、`popt_mid_final`：可能包含 Gaussian dip 的最终参数。
- `sat_bw_high`、`sat_bw_mid`、`sat_bw_low3`：平台带宽。
- `split_size`：Group1/Group2 趋势分叉点。
- `dip_high`、`dip_mid`、`dip_low3`：凹陷区间。
- `low3_drop_size`：Group3 跌落稳定起点。

## 5. 时延模型

### 5.1 三段基础模型

时延模型在 `src/latency_model.py` 中实现。基础形式为三段模型：

```text
m <= m1:       T_raw = t_s
m1 < m < m2:   T_raw = size / BW(size)
m >= m2:       T_raw = t_s + size / BW_max
```

变量解释：

- `m`：消息大小。
- `t_s`：小消息启动开销，由小消息延时均值估计。
- `m1`：小消息边界，当前固定为 `2 MiB`。
- `m2`：进入饱和线性段的边界，由饱和带宽反推。
- `BW(size)`：带宽模型输出。
- `BW_max`：对应 group 的平台带宽。

三段模型的物理含义：

- 小消息段主要由固定启动开销主导。
- 中间段由带宽曲线主导。
- 大消息段接近固定启动开销加上线性传输时间。

### 5.2 残差样条修正

基础三段模型有系统性偏差，尤其在中大消息段。因此引入 log-size 空间残差样条：

```text
r_i = T_raw(size_i) - T_true(size_i)
correction(size) = spline(log2(size))
T_final(size) = max(0, T_raw(size) - correction(size))
```

样条修正策略：

- 小消息段 `size <= m1` 不修正，保持 `t_s`。
- 中大消息段按 `log2(size)` 聚合残差中位数，降低噪声影响。
- 在 `m1` 插入残差为 0 的锚点，保证边界平滑。
- 右端界外使用端点斜率线性外推，防止样条外推爆炸。

这一修正不改变带宽模型，只修正 `T_raw` 到 `T_true` 的系统偏差。

### 5.3 输出参数

时延参数保存到：

```text
artifacts/models/latency_model_params.json
```

关键字段：

- `t_s`：小消息启动开销。
- `m1`、`m2`：三段边界。
- `bw_max`：平台带宽。
- `spline_x_log2`：样条节点横坐标。
- `spline_residual_us`：样条拟合残差。
- `mape`、`rmse`：训练/拟合误差。

## 6. 单算子验证协议

默认验证采用 size-holdout：

- 按 Group1/Group2/Group3 分组。
- 在每个 group 内按消息大小抽取测试 size。
- 同一个 unique size 不同时出现在训练集和测试集。
- 输出 coverage report，检查 group 覆盖、size 覆盖和 train/test 泄漏。

指标定义：

```text
abs_err_us = |T_pred - T_true|
rel_err_percent = abs_err_us / max(T_true, eps) * 100
MAPE = mean(rel_err_percent)
RMSE = sqrt(mean(abs_err_us^2))
P50/P90/P95 = rel_err_percent 分位数
```

当前结果：

| 指标 | 数值 |
|---|---:|
| Overall MAPE | 0.712% |
| Overall RMSE | 30.990 us |
| P50/P90/P95 | 0.345% / 1.596% / 3.078% |
| Unique-size overlap | 0 |

该结果说明：在当前硬件、HCCL-test 和消息大小覆盖范围内，单算子时延模型已经达到低误差可交付状态。

## 7. 异步掩盖模型

### 7.1 建模目标

异步 All-to-All 的应用结构通常是：

```text
pre work -> async alltoall launch -> overlap compute window -> wait/sync -> post work
```

异步掩盖模型预测：

- `hidden_comm_us`：被计算窗口掩盖的通信时间。
- `visible_comm_us`：仍暴露在关键路径上的通信时间。
- `async_total_us`：异步区域总耗时。
- `speedup`：相对同步串行执行的加速比。
- `hidden_fraction_of_comm`：通信被隐藏的比例。

### 7.2 公式

定义：

```text
T_comm = All-to-All 单算子时延
T_comp = async launch 到 wait 之间的可重叠计算窗口
eta = 掩盖效率，范围 [0, 1]
T_launch = async launch 附加开销
T_sync = wait/sync 附加开销
scale = overlap_window_scale
```

有效计算窗口：

```text
T_window = scale * T_comp
```

物理可掩盖上界：

```text
T_bound = min(T_comm, T_window)
```

实际隐藏通信：

```text
T_hidden = eta * T_bound
```

可见通信：

```text
T_visible = max(0, T_comm - T_hidden)
```

同步基线：

```text
T_sync_baseline = T_pre + T_comm + T_comp + T_post
```

异步区域预测：

```text
T_async = T_pre + T_comp + T_visible + T_launch + T_sync + T_post
```

### 7.3 eta 的含义

`eta` 表示可重叠上界中真正被隐藏的比例。它不是单算子通信参数，而是 workload/runtime 调度参数。

影响 `eta` 的因素包括：

- 通信启动是否足够早。
- 计算是否真的与通信并行执行。
- stream/event/wait 的调度开销。
- HCCL 通信与 NPU compute 对资源的竞争。
- profiler 聚合口径。

有真实 profiler 后可反推：

```text
T_visible = T_async_measured - T_pre - T_comp - T_launch - T_sync - T_post
eta = (T_comm - T_visible) / min(T_comm, scale * T_comp)
```

代码入口：

```python
from src.async_overlap_model import derive_eta
```

## 8. 异步验证协议

### 8.1 Replay 验证

Replay 验证的目的不是证明真实应用已重叠，而是验证异步组合公式，并量化单算子误差传递到异步区域后的误差。

流程：

- 使用 `validation_metrics.csv` 中实测单算子时延作为 `T_comm_true`。
- 使用单算子模型预测值作为 `T_comm_pred`。
- 构造 `T_comp/T_comm = 0, 0.25, 0.5, 1, 2, 4` 的计算窗口。
- 分别计算 replay observed async total 和 predicted async total。
- 统计 async total MAPE/RMSE/P50/P90/P95。

当前结果：

| 指标 | 数值 |
|---|---:|
| Samples | 1440 |
| Async total MAPE | 0.466% |
| Async total RMSE | 23.078 us |
| P50/P90/P95 | 0.111% / 0.952% / 2.253% |
| Mean speedup | 1.316x |
| Mean hidden communication fraction | 0.531 |

### 8.2 DLRM 单 trace 实测验证

DLRM 中存在如下异步结构：

```text
DLRM alltoall communication start
bottom MLP forward
DLRM alltoall communication wait
```

脚本读取 Ascend profiler `step_trace_time.csv` 中的：

- `Computing`
- `Communication`
- `Overlapped`

实测指标：

```text
eta_observed = Overlapped / min(Communication, Computing)
hidden_fraction_of_comm = Overlapped / Communication
async_region_observed = Computing + Communication - Overlapped
```

单 trace 结果显示：通信约几十毫秒，bottom MLP 计算窗口约百微秒，因此虽然存在异步启动和 wait，能掩盖的通信比例很低。

### 8.3 DLRM 1452 配置 sweep 验证

多配置验证复用 1452 组 DLRM profiler trace。当前结果：

| 指标 | 数值 |
|---|---:|
| Traces found | 1452 |
| Cases used | 1244 |
| Mean communication | 1804397.306 us |
| Mean compute proxy | 255.304 us |
| Observed eta mean/median | 0.6347 / 1.0000 |
| Mean bounded hidden fraction | 0.002451 |
| Mean bounded speedup | 1.002450x |
| Bounded async-region MAPE | 0.027% |

重要解释：

- `Communication` 远大于 `Computing`，因此 `min(T_comm, T_comp)` 很小。
- 即使 `eta` 接近 1，也只能隐藏 compute window 大小以内的通信。
- 最终交付报告采用 bounded overlap，确保验证口径与 `min(Communication, Computing)` 的模型公式一致。

## 9. 为什么不直接用 HCCL kernel duration 替代通信模型

HCCL kernel duration 可能包含等待、同步、runtime 调度和跨 stream 依赖，不一定等价于纯通信传输时延。单算子模型使用 HCCL-test 稳定测量通信本身，异步模型再通过应用 profiler 校准 overlap 效率，可以避免把通信建模和应用调度混在一个黑盒指标里。

## 10. 交付结论

本交付形成了完整的 All-to-All 建模链路：

- 对齐感知的带宽模型解释 128KB、32KB 和非对齐消息的差异。
- 三段时延模型加残差样条将单算子 size-holdout MAPE 降至 0.712%。
- 异步掩盖模型明确区分通信本身、可重叠计算窗口和 workload 级 `eta`。
- Replay 验证显示单算子误差传递到异步区域后仍保持低误差。
- DLRM 实测验证说明当前配置下通信远大于计算窗口，实际可隐藏通信比例很低，这是应用结构限制，不是单算子模型失效。
