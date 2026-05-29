# All-to-All 单算子与异步掩盖建模方案

## 1. 交付范围

本目录交付两部分模型：

- 单算子模型：预测单机 8 NPU HCCL All-to-All 在不同消息大小和对齐方式下的带宽与时延。
- 异步掩盖模型：预测 All-to-All 与后续计算窗口异步并行时的可见通信开销、异步区域总时延和通信隐藏比例。

当前模型适用范围为 910B、单机 8 NPU、HCCL-test All-to-All。若迁移到其他 NPU 数量、HCCL 版本或跨机网络，需要重新采集单算子数据并校准异步参数。

## 2. 单算子建模

### 2.1 输入与输出

输入：

- `size_bytes`：单次 All-to-All 消息大小。
- `num_npus`：当前基线为 8。
- `alignment_class`：由消息大小自动判定。

对齐类别：

- `Group1 (128KB)`：`size % 128KB == 0`。
- `Group2 (32KB)`：非 128KB 对齐，但 `size % 32KB == 0`。
- `Group3 (unaligned)`：非 32KB 对齐。

输出：

- `bus_bw(size, alignment)`：总线带宽，单位 GB/s。
- `latency(size, alignment)`：单算子通信时延，单位 us。

### 2.2 带宽模型

带宽模型位于 `src/bw_model.py`，核心是消息大小到带宽的分段拟合：

```text
BW(size) = sigmoid(log2(size)) - gaussian_dip(log2(size))
```

三类对齐数据分别拟合或继承趋势：

- Group1：128KB 对齐，高带宽平台。
- Group2：32KB 对齐，中带宽平台。
- Group3：非对齐，小消息段跟随 Group1，上升后跌落到 Group2 附近平台。

模型参数保存在：

```text
artifacts/models/bw_model_params.json
```

### 2.3 时延模型

时延模型位于 `src/latency_model.py`，采用三段模型加残差样条修正：

```text
m <= m1:       T_raw = t_s
m1 < m < m2:   T_raw = size / BW(size)
m >= m2:       T_raw = t_s + size / BW_max

T_final = T_raw - spline_residual(log2(size))
```

其中：

- `t_s`：小消息启动/固定开销。
- `m1`：小消息边界，当前固定为 2 MiB。
- `m2`：从带宽平台反推得到的饱和边界。
- `BW_max`：各 group 的平台带宽。
- `spline_residual`：训练数据上对 `T_raw - T_true` 的残差样条修正。

模型参数保存在：

```text
artifacts/models/latency_model_params.json
```

### 2.4 单算子验证结果

当前 size-holdout 验证结果见：

```text
artifacts/reports/validation_summary.md
```

关键结果：

- Overall MAPE: `0.712%`
- Overall RMSE: `30.990 us`
- Relative Error P50/P90/P95: `0.345% / 1.596% / 3.078%`

## 3. 异步掩盖建模

### 3.1 建模目标

异步掩盖模型不重新建通信算子本身，而是组合单算子通信时延与可重叠计算窗口，预测异步区域总时延。

典型执行结构：

```text
pre work -> async alltoall launch -> overlap compute window -> wait/sync -> post work
```

模型输出：

- `visible_comm_us`：未被计算掩盖、最终仍可见的通信时间。
- `hidden_comm_us`：被计算窗口掩盖的通信时间。
- `async_total_us`：异步区域总时延。
- `speedup`：相对于同步串行执行的加速比。
- `hidden_fraction_of_comm`：通信时延中被掩盖的比例。

### 3.2 异步模型公式

定义：

```text
T_comm = All-to-All 单算子时延模型输出
T_comp = async launch 到 wait 之间可与通信重叠的计算窗口
eta    = 掩盖效率，范围 [0, 1]
T_launch = 异步 launch 附加开销
T_sync   = wait/sync 附加开销
```

有效计算窗口：

```text
T_window = overlap_window_scale * T_comp
```

可掩盖上界：

```text
T_bound = min(T_comm, T_window)
```

被掩盖通信：

```text
T_hidden = eta * T_bound
```

可见通信：

```text
T_visible_comm = max(0, T_comm - T_hidden)
```

同步基线：

```text
T_sync_baseline = T_pre + T_comm + T_comp + T_post
```

异步区域：

```text
T_async = T_pre + T_comp + T_visible_comm + T_launch + T_sync + T_post
```

该模型位于：

```text
src/async_overlap_model.py
```

### 3.3 参数说明

当前参数文件：

```text
artifacts/models/async_overlap_model_params.json
```

默认参数：

- `eta = 0.85`：假设可重叠窗口中 85% 可有效掩盖通信。
- `launch_overhead_us = 3.0`：异步 launch 开销。
- `sync_overhead_us = 8.0`：wait/sync 开销。
- `overlap_window_scale = 1.0`：计算窗口全部可用于重叠。

`eta` 是场景参数。没有真实异步 trace 时，它用于 what-if 评估；有真实 profiler 数据后，应通过以下反推公式校准：

```text
eta = (T_comm - T_visible_comm) / min(T_comm, T_window)
T_visible_comm = T_async_measured - T_pre - T_comp - T_launch - T_sync - T_post
```

代码入口为 `derive_eta`：

```python
from src.async_overlap_model import derive_eta
```

## 4. 异步验证方案与结果

### 4.1 replay 验证方式

- 使用 `artifacts/reports/validation_metrics.csv` 中的实测 All-to-All 时延作为通信真实值。
- 使用单算子时延模型输出作为通信预测值。
- 构造多组计算窗口：`T_comp / T_comm = 0, 0.25, 0.5, 1, 2, 4`。
- 分别计算 replay observed async total 与 predicted async total。
- 误差反映“单算子预测误差进入异步组合模型后的传播误差”。

验证脚本：

```bash
python3 scripts/run_async_overlap_validation.py
```

输出文件：

```text
artifacts/models/async_overlap_model_params.json
artifacts/reports/async_overlap_validation.csv
artifacts/reports/async_overlap_metrics_by_group.csv
artifacts/reports/async_overlap_metrics_by_compute_ratio.csv
artifacts/reports/async_overlap_summary.md
artifacts/plots/async_overlap_validation.png
```

### 4.2 replay验证结果

当前回放验证结果见：

```text
artifacts/reports/async_overlap_summary.md
```

关键结果：

- Samples: `1440`
- Async total MAPE: `0.466%`
- Async total RMSE: `23.078 us`
- Relative Error P50/P90/P95: `0.111% / 0.952% / 2.253%`
- Mean observed speedup vs sync baseline: `1.316x`
- Mean hidden communication fraction: `0.531`

按 group 的结果：

- Group1 MAPE: `0.480%`
- Group2 MAPE: `0.376%`
- Group3 MAPE: `0.542%`

按计算窗口比例的趋势：

- `T_comp/T_comm = 0` 时没有掩盖，且异步 launch/sync 使速度略低于同步基线。
- `T_comp/T_comm = 1` 时通信隐藏比例达到 `0.85`，平均 speedup 为 `1.728x`。
- `T_comp/T_comm >= 2` 时通信已接近最大掩盖，额外计算会稀释整体 speedup。

### 4.3 真实 profiler 接入要求

若要把 replay 验证升级为真实异步验证，需要采集以下字段：

- `size_bytes`：All-to-All 消息大小。
- `alignment_class`：对齐类别。
- `T_comm_measured`：单独 All-to-All 测得时延。
- `T_comp_window_measured`：async launch 到 wait 之间可重叠计算窗口。
- `T_async_region_measured`：包含 launch、overlap compute、wait 的完整区域耗时。
- `T_launch`、`T_sync`：如 profiler 能拆分则直接使用，否则用小规模实验估计。

然后对每条 trace 使用 `derive_eta` 反推 `eta`，并按 group、消息大小区间、计算窗口比例统计 eta 分布。

### 4.4 DLRM 实测异步验证

DLRM 的分布式前向中存在真实异步结构：

```text
record_function("DLRM alltoall communication start")
ext_dist.alltoall(...)
record_function("DLRM bottom nlp forward")
record_function("DLRM alltoall communication wait")
a2a_req.wait()
```

因此可以直接用 DLRM profiler 的 `step_trace_time.csv` 做实测验证。当前脚本为：

```bash
python3 scripts/run_dlrm_async_overlap_validation.py
```

该脚本读取 Ascend profiler 输出中的：

- `Computing`：DLRM step 中 NPU 计算时间。
- `Communication`：通信时间。
- `Overlapped`：profiler 识别出的通信与计算重叠时间。

实测校准公式：

```text
eta_observed = Overlapped / min(Communication, Computing)
hidden_fraction_of_comm = Overlapped / Communication
async_region_observed = Computing + Communication - Overlapped
```

DLRM profiler 示例数据位置：

```text
data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT
```

生成文件：

```text
artifacts/models/dlrm_async_overlap_model_params.json
artifacts/reports/dlrm_async_overlap_validation.csv
artifacts/reports/dlrm_async_overlap_summary.md
artifacts/plots/dlrm_async_overlap_validation.png
```

当前 DLRM 实测结果：

- All-to-All start markers: `80`
- All-to-All wait markers: `80`
- Steps used: `70`
- Mean communication: `48817.287 us`
- Mean overlappable compute: `171.177 us`
- Mean overlapped time: `23.617 us`
- Observed eta mean/median: `0.1380 / 0.1347`
- Mean hidden fraction of communication: `0.000488`
- Mean speedup over sync comm+compute region: `1.000487x`

结论：DLRM 确实形成了 All-to-All 与 bottom MLP 的异步重叠，但在当前配置下通信约为几十毫秒，bottom MLP 计算窗口只有约 170us，因此可掩盖窗口远小于通信时间，实际隐藏的通信比例很低。

复用 DLRM sweep profiler 数据：

```text
data/profiler/dlrm_sweep
```

验证脚本为：

```bash
python3 scripts/run_dlrm_async_overlap_multiconfig_validation.py
```

生成文件：

```text
artifacts/models/dlrm_async_overlap_multiconfig_model_params.json
artifacts/reports/dlrm_async_overlap_multiconfig_summary.md
artifacts/reports/dlrm_async_overlap_multiconfig_steps.csv
artifacts/reports/dlrm_async_overlap_multiconfig_cases.csv
artifacts/reports/dlrm_async_overlap_multiconfig_by_variant.csv
artifacts/plots/dlrm_async_overlap_multiconfig_validation.png
```

关键结果：

- Traces found: `1452`
- Cases used after `Communication >= 1000 us`: `1244`
- Mean communication: `1804397.306 us`
- Mean overlappable compute proxy: `255.304 us`
- Mean bounded overlapped time: `160.083 us`
- Observed eta mean/median: `0.6347 / 1.0000`
- Mean bounded hidden fraction of communication: `0.002451`
- Mean bounded speedup over sync comm+compute region: `1.002450x`
- Bounded async-region MAPE: `0.027%`

多配置报告采用 bounded overlap 作为最终交付口径，使验证结果与异步掩盖公式保持一致。

## 5. 交付结论

本交付已覆盖：

- All-to-All 单算子带宽模型。
- All-to-All 单算子时延模型。
- 参数持久化与 size-holdout 验证报告。
- 异步掩盖建模公式、参数文件、代码入口。
- 异步回放验证 CSV、汇总报告。
- DLRM 实测 profiler trace 异步验证报告。

当前已经包含 replay 验证和 DLRM 实测 trace 验证。若迁移到新的 DLRM 配置或其他 workload，需要重新采集 profiler trace，并按 `Computing/Communication/Overlapped` 或更细粒度时间线重新校准 `eta`。
