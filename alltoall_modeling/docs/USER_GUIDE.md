# All-to-All 建模代码使用说明

## 1. 交付目标

本目录提供单机 8 NPU HCCL All-to-All 通信建模与异步掩盖建模的完整可复现流程，覆盖：

- All-to-All 单算子带宽模型：`message size -> bus bandwidth`。
- All-to-All 单算子时延模型：`message size + alignment class -> latency`。
- 异步掩盖模型：`communication latency + overlap compute window -> visible communication / async region latency`。
- DLRM 实测 profiler trace 上的异步掩盖校准和验证。
- 模型参数、验证指标的输出路径。

当前交付基线适用于 910B 单机 8 NPU、HCCL-test `alltoall_test`。迁移到其他 NPU 数量、跨机拓扑、HCCL 版本或其他通信库时，需要重新采集训练数据并重新生成参数。

## 2. 目录结构

```text
a2a_final/
  DELIVERY_MANIFEST.md
  bin/
    alltoall_test
  src/
    bw_model.py
    latency_model.py
    async_overlap_model.py
    memory_bw_utils.py
  scripts/
    collect_train_hccl_a2a_npu8.sh
    collect_test_hccl_a2a_npu8.sh
    run_model_pipeline.py
    run_async_overlap_validation.py
    run_dlrm_async_overlap_validation.py
    run_dlrm_async_overlap_multiconfig_validation.py
    generate_size_manifest.py
  data/raw/
    a2a_32kb_aligned.txt
    a2a_train_sizes.txt
  artifacts/models/
    bw_model_params.json
    latency_model_params.json
    async_overlap_model_params.json
    dlrm_async_overlap_model_params.json
    dlrm_async_overlap_multiconfig_model_params.json
  artifacts/reports/
    validation_summary.md
    validation_metrics.csv
    fit_metrics.csv
    coverage_summary.csv
    async_overlap_summary.md
    dlrm_async_overlap_summary.md
    dlrm_async_overlap_multiconfig_summary.md
  artifacts/plots/
    bw_model_fit.png
    latency_model_fit.png
    latency_compare.png
    async_overlap_validation.png
    dlrm_async_overlap_validation.png
    dlrm_async_overlap_multiconfig_validation.png
  docs/
    USER_GUIDE.md
    MODELING_PRINCIPLES.md
    CODE_INVENTORY.md
    modeling_scheme.md
```

## 3. 环境要求

Python 侧依赖：

```bash
python3 -m pip install numpy pandas scipy matplotlib
```

采集侧依赖：

- Ascend/NPU 运行环境已初始化。
- HCCL-test 可执行文件可运行。
- `mpirun` 可用。
- 当前脚本假设单机 8 NPU，即 `mpirun -n 8` 与 HCCL-test 参数 `-p 8`。

说明：本交付默认不自动启动 HCCL-test 采集。采集脚本用于打印或组织命令，避免误触发长时间硬件任务。

## 4. 训练数据

默认训练数据：

```text
data/raw/a2a_32kb_aligned.txt
```

该文件是 HCCL-test 输出文本，核心字段为：

```text
data_size(Bytes) | aveg_time(us) | alg_bandwidth(GB/s) | check_result
```

解析逻辑位于 `src/bw_model.py::parse_hccl_output`。模型 pipeline 会将 `alg_bandwidth` 根据 All-to-All bus bandwidth factor 转换为 `bus_bw`。

## 5. 生成训练采集命令

生成覆盖 Group1/Group2/Group3 的训练采集命令：

```bash
cd a2a_final
./scripts/collect_train_hccl_a2a_npu8.sh
```

输出内容包括：

- 128KB 对齐消息大小，即 Group1。
- 32KB 对齐但非 128KB 对齐消息大小，即 Group2。
- 非 32KB 对齐消息大小，即 Group3。

该脚本不会自动执行采集；如需真实采集，应先确认输出命令、输出路径、NPU 空闲状态和 HCCL 环境。

## 6. 训练与 size-holdout 验证

推荐使用 Method-A，即按消息大小做训练/测试拆分，保证测试 size 不进入训练集：

```bash
cd a2a_final
python3 scripts/run_model_pipeline.py \
  --raw-data data/raw/a2a_32kb_aligned.txt \
  --split-mode size_holdout \
  --test-ratio 0.2
```

主要输出：

```text
artifacts/models/bw_model_params.json
artifacts/models/latency_model_params.json
artifacts/reports/fit_metrics.csv
artifacts/reports/validation_metrics.csv
artifacts/reports/coverage_summary.csv
artifacts/reports/validation_summary.md
artifacts/plots/bw_model_fit.png
artifacts/plots/latency_model_fit.png
artifacts/plots/latency_compare.png
```

当前 size-holdout 验证结果：

| 指标 | 数值 |
|---|---:|
| Overall MAPE | 0.712% |
| Overall RMSE | 30.990 us |
| P50 relative error | 0.345% |
| P90 relative error | 1.596% |
| P95 relative error | 3.078% |
| Train/test unique-size overlap | 0 |

## 7. 外部测试集验证

如需完全独立的测试文件，先生成外部测试采集命令：

```bash
cd a2a_final
./scripts/collect_test_hccl_a2a_npu8.sh
```

采集完成后，将结果保存为：

```text
data/raw/a2a_test_external.txt
```

然后运行：

```bash
python3 scripts/run_model_pipeline.py \
  --raw-data data/raw/a2a_32kb_aligned.txt \
  --split-mode external_test \
  --validation-file data/raw/a2a_test_external.txt
```

## 8. 异步掩盖 replay 验证

基于单算子验证结果做异步组合 replay：

```bash
cd a2a_final
python3 scripts/run_async_overlap_validation.py
```

输入：

```text
artifacts/reports/validation_metrics.csv
```

输出：

```text
artifacts/models/async_overlap_model_params.json
artifacts/reports/async_overlap_validation.csv
artifacts/reports/async_overlap_metrics_by_group.csv
artifacts/reports/async_overlap_metrics_by_compute_ratio.csv
artifacts/reports/async_overlap_summary.md
artifacts/plots/async_overlap_validation.png
```

当前 replay 验证结果：

| 指标 | 数值 |
|---|---:|
| Samples | 1440 |
| Async total MAPE | 0.466% |
| Async total RMSE | 23.078 us |
| P50/P90/P95 relative error | 0.111% / 0.952% / 2.253% |
| Mean observed speedup | 1.316x |
| Mean hidden communication fraction | 0.531 |

## 9. DLRM 实测异步验证

单个 DLRM profiler trace 验证：

```bash
cd a2a_final
python3 scripts/run_dlrm_async_overlap_validation.py
```

默认读取交付目录下的 profiler 示例位置：

```text
data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT
```

如需验证新的 profiler 输出目录，可通过参数覆盖默认输入：

```bash
python3 scripts/run_dlrm_async_overlap_validation.py \
  --profiler-dir data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT \
  --step-trace step_trace_time.csv \
  --operator-details operator_details.csv
```

DLRM 配置 sweep 验证：

```bash
python3 scripts/run_dlrm_async_overlap_multiconfig_validation.py
```

默认读取：

```text
data/profiler/dlrm_sweep
```

当前多配置验证结果：

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

## 10. 模型参数如何使用

单算子在线预测建议直接复用 Python 函数：

```python
from src.bw_model import parse_hccl_output, run_two_stage_piecewise
from src.latency_model import run_latency_model, eval_latency
from src.memory_bw_utils import MUL_FACTOR_FUNCS

raw_df = parse_hccl_output("data/raw/a2a_32kb_aligned.txt")
raw_df["bus_bw"] = raw_df["alg_bw"] * MUL_FACTOR_FUNCS["all_to_all"](8)

bw_results = run_two_stage_piecewise("data/raw/a2a_32kb_aligned.txt", num_gpus=8)
lat_results = run_latency_model(bw_results, raw_df)

lat_us = eval_latency(128 * 1024 * 1024, lat_results)
```

异步区域预测：

```python
from src.async_overlap_model import AsyncOverlapParams, predict_async_region

params = AsyncOverlapParams(eta=0.85, launch_overhead_us=3.0, sync_overhead_us=8.0)
out = predict_async_region(comm_us=50000.0, compute_us=10000.0, params=params)
print(out["async_total_us"], out["hidden_fraction_of_comm"])
```

