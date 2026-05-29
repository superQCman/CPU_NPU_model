# 成本约束硬件参数优化

## 本文回答什么问题

本文说明如何使用需求方提供的 `参数约束v2.docx` 两点参考成本数据，把“硬件参数配置选择”转换成可复现的成本约束优化问题。

## 读完你应该知道什么

你应该知道三件事：

- 当前推荐成本函数来自 `docs/参数约束v2.docx` 中的 x1-x25 两点参考成本，而不是旧的手写 surrogate cost。
- 优化阶段只使用已有训练好的模型 bundle，不重新训练 RF，也不重新拟合 op-thread scale。
- 优化算法是在合法硬件参数组组合上做穷举搜索，再输出预算内最优点和 Pareto 前沿。

命名约定：`Device_1` 表示智算平台，`Device_2` 表示通算平台。路径、alias 或 YAML 名称中的 `kunpeng920` 对应智算平台，`s920x20` 对应通算平台，保留这些历史标识是为了兼容已有实验。

## 1. 为什么不能直接优化二十多个独立参数

`analytical_base/hardware_cost_function_request.md` 中列出了二十多个硬件参数，例如 core 数、频率、pipeline width、SIMD 宽度、内存带宽、cache size 和 cache latency。

这些参数不会被忽略，但也不应该完全自由组合。原因是很多参数来自同一个硬件子系统：

- `pipeline.fetch_width/decode_width/rename_width/dispatch_width/issue_width/commit_width` 属于同一个微架构宽度能力。
- `instruction.simd_width_bits`、FP throughput 和 FP latency 属于同一个向量执行能力。
- copy bandwidth、random gather bandwidth 和 DRAM latency 属于同一个内存子系统。
- L1/L2/L3 cache size 和 response latency 属于同一个 cache hierarchy。

因此优化器采用硬件参数组口径：先把二十多个参数聚合进五个硬件参数组，再在硬件参数组档位上搜索。

## 2. 参数约束 v2 成本函数

需求方在 `docs/参数约束v2.docx` 中为 x1-x25 每个硬件变量提供了两个参考成本点。Pair 对的语义是“其他参数不变，只改变该参数时的整机参考成本”。因此这些成本点不能直接相加，否则会把整机成本重复计价 25 次。

当前实现使用 `calibrator/configs/hardware_cost_constraints_v2.json` 固化这些参考点。已核对：只有 `x13 memory.bandwidth_gb_s_total` 的低点是 `{40, 6w}`，其余已见参数低点大多为 `8w`。该特殊点按原文保留，不默认当作笔误。

### 2.1 单参数成本曲线

对每个参数 \(p\)，设需求方给出的两个点为：

$$
(x_{p,1},K_{p,1}),\quad (x_{p,2},K_{p,2})
$$

用两点幂律函数拟合：

$$
k_p(x)=K_{p,1}\left(\frac{q_p(x)}{q_p(x_{p,1})}\right)^{\gamma_p}
$$

$$
\gamma_p=\frac{\log(K_{p,2}/K_{p,1})}{\log(q_p(x_{p,2})/q_p(x_{p,1}))}
$$

其中：

$$
q_p(x)=
\begin{cases}
x, & \text{核心数、频率、带宽、cache size、SIMD width、throughput 等越大越贵的参数}\\
1/x, & \text{cache latency、DRAM latency 等越小越贵的参数}
\end{cases}
$$

这个函数严格经过两个参考点，且成本始终为正。使用幂律而不是简单线性插值，是因为硬件规格通常按倍率变化，例如带宽、cache 容量和 SIMD 宽度。

### 2.2 单参数增量成本

Pair 对的语义是“只改变该参数时的整机参考成本”。因此 v2 成本不能再把单参数影响用组权重摊薄，而是先计算该参数相对基准的正向成本增量：

$$
A_p=\max(k_p(x_{p,\text{ref}}), C_0)
$$

$$
\delta_p(x)=\max(k_p(x)-A_p,0)
$$

其中 \(C_0=8w\)，\(x_{p,\text{ref}}\) 是 Device_1 / 智算平台基准规格或文档 baseline 规格。多数参数的低点是 `8w`，因此如果只把一个参数从低点改到第二参考点，整机成本会严格到达该 Pair 对成本。例如 `x9 SIMD width {128,8w;256,9w}` 单独变化时，\(\delta_{x9}=1w\)，整机成本为 `9w`，不会被参数组权重摊薄。

`x13 memory.bandwidth_gb_s_total` 的 `{40,6w;100,8.5w}` 是唯一确认的非 `8w` 低点。由于整机基准仍定义为 `8w`，实现使用 \(A_p=\max(k_p(x_{p,\text{ref}}),C_0)\) 防止单个低成本参考点把整机成本拉低到 `8w` 以下；当 x13 从 40 提到 100 时，整机成本增加到 `8.5w`。

### 2.3 参数组成本

按需求方绑定关系分组：

| 参数组 | 参数 |
|---|---|
| Core/Frequency | x1-x2 |
| Pipeline width | x3-x8 |
| SIMD/FP execution | x9-x12 |
| Memory subsystem | x13-x18 |
| Cache hierarchy | x19-x25 |

组内使用正向增量的 L2 范数：

$$
\Delta_g=\sqrt{\sum_{p\in P_g}\delta_p(x_p)^2}
$$

理由是同组参数在真实硬件中通常绑定变化，不能把每个“单参数整机成本”简单相加；L2 范数能保留单参数 Pair 对约束，同时比直接求和更保守地合并同组多个参数的成本增量。

### 2.4 整机成本

Device_1 / 智算平台基准成本定义为：

$$
C_0=8w
$$

万元级参考成本为：

$$
C_{\text{wan}}(h)=C_0+\sum_{g\in G}\Delta_g
$$

整机相对成本为：

$$
C_{\text{rel}}(h)=\frac{C_{\text{wan}}(h)}{C_0}
$$

五个硬件参数组仍用于组织参数和约束候选空间，但 v2 reference cost 不再使用旧的组权重摊薄单参数 Pair 对。旧权重只保留给 `--cost-model surrogate` 兼容模式。

| 硬件参数组 | v2 聚合方式 |
|---|---|
| Core/Frequency | 组内正向增量 L2 范数 |
| Pipeline width | 组内正向增量 L2 范数 |
| SIMD/FP execution | 组内正向增量 L2 范数 |
| Memory subsystem | 组内正向增量 L2 范数 |
| Cache hierarchy | 组内正向增量 L2 范数 |

输出中的 `cost_wan` 是参考成本，单位为万元；`relative_cost` 是相对 Device_1 / 智算平台的无量纲成本。该成本函数用于优化排序和可解释分析，不替代真实 BOM 或采购报价。

所有候选硬件参数都会经过 `hardware_cost_constraints_v2.json` 中的 `range_min/range_max` 硬约束过滤。输出目录中的
`hardware_cost_candidate_range_audit.csv` 记录每个 x 参数在候选集合中的最小值、最大值和越界计数；当前
`outputs/hardware_cost_optimization_nsga2_only_20260501` 结果中所有参与优化的参数均满足范围约束。

## 3. 优化问题

主目标是预算内最快：

$$h^*(B)=\arg\min_{h\in\mathcal{H}_{legal},C(h)\le B}L(h)$$

其中 `B` 是预算上限，`\mathcal{H}_{legal}` 是合法硬件参数组候选集合，`L(h)` 是 DLRM-S/M/L 三档 workload 的加权 E2E latency。

默认目标函数是：

$$L(h)=0.25L_S(h)+0.50L_M(h)+0.25L_L(h)$$

当前 v2 参考成本实验的预算档以万元为单位：

```text
8, 9, 10, 12, 14, 16, Device_2_cost
```

## 4. 优化算法

优化器支持两类搜索算法：

- `grid`：合法硬件参数组网格穷举搜索，是当前小规模候选空间下的精确基线。
- `nsga2`：离散 NSGA-II 启发式搜索，用于候选空间变大时减少需要评估的候选数量。
- `both`：同时运行 grid 和 NSGA-II，用 grid 精确结果检查 NSGA-II 近似结果。

grid 流程如下：

1. 为 core、pipeline、vector、memory、cache 定义有限档位。
2. 遍历所有硬件参数组档位组合，生成候选硬件配置。
3. 对每个候选配置计算 reference v2 成本。
4. 用已有 Device_1 和 Device_2 v3 模型 bundle 固定权重预测 E2E latency。
5. 对每个预算档选择 `cost <= budget` 中 latency 最低的候选。
6. 输出 Pareto 前沿，删除“更贵且更慢”的候选。

这里的 v3 指 `whitebox_family_prior_plus_physics_guarded_rf_v3`。

NSGA-II 的基因是 5 维离散整数：

$$gene=[idx_{core},idx_{pipeline},idx_{vector},idx_{memory},idx_{cache}]$$

每个整数表示该硬件参数组选择哪个候选档位。例如 `[2,1,3,4,0]` 表示 core 选第 2 档、pipeline 选第 1 档、vector 选第 3 档、memory 选第 4 档、cache 选第 0 档。NSGA-II 的两个目标都是最小化：

$$minimize\ (C(h),L(h))$$

实现上使用非支配排序、crowding distance、锦标赛选择、均匀 crossover 和整数 mutation。当前环境没有 `pymoo` 或 `deap`，因此优化器内置了离散 NSGA-II，不引入新依赖。

如果候选空间较大，可以用 `--prediction-workers` 对 candidate evaluation 并行。并行只改变执行速度，不改变候选排序；NSGA-II 使用固定 seed 保持可复现。

## 5. 模型输入要求

优化器必须加载两个已经存在的 pickle bundle：

```text
--device1-bundle <path/to/kunpeng920_v3_bundle.pkl>
--device2-bundle <path/to/s920x20_v3_bundle.pkl>
```

优化阶段不会调用 `calibrator/run.py`，不会读取 train/test CSV 来训练模型，也不会调用任何 `fit_whitebox_family_*` 函数。

如果没有 bundle，优化器会失败并提示：

```text
model bundle not found; existing CSV summaries cannot restore RF weights.
Run is aborted because --no-refit is enforced.
```

注意：`combined_summary.csv`、`prediction_error_rows.csv`、`e2e_pipeline_summary.csv` 和 `op_thread_scale_config_*.json` 都不是完整模型权重，不能替代 pickle bundle。

## 6. 可选 prior-only fallback

如果必须在没有 RF bundle 的情况下先做低置信度实验，可以显式启用：

```text
--allow-prior-only-fallback
```

该模式只使用 Analytical prior，以及用户可选提供的 op-thread scale config。它不使用 RF，因此报告会标注为 `LOWER-CONFIDENCE PRIOR-ONLY FALLBACK`。这种结果只能用于检查成本优化流程，不应用作 RF v3 成本优化结论。

## 7. 运行命令

有现成 bundle 时运行当前 v2 成本实验。下面命令复现
`outputs/hardware_cost_optimization_nsga2_only_20260501` 中的当前产物。当前命令使用 64 个种群和 5 代
NSGA-II，在 Pair 区间内评估数百个候选点，用于生成更密的 Pareto 点云；如果只做流程 smoke，可将
`--nsga2-population/--nsga2-generations` 降到 `32/3`。

```bash
# 在仓库根目录执行

python calibrator/hardware_cost_optimizer.py \
  --device1-bundle <existing_kunpeng920_v3_bundle.pkl> \
  --device2-bundle <existing_s920x20_v3_bundle.pkl> \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --workload-set dlrm_s_m_l \
  --search-algorithm nsga2 \
  --cost-model reference_v2 \
  --cost-constraints calibrator/configs/hardware_cost_constraints_v2.json \
  --pair-extrapolation-mode forbid \
  --budget-unit wan \
  --budgets 8,9,10,12,14,16,device2 \
  --grid-profile full \
  --nsga2-population 64 \
  --nsga2-generations 5 \
  --prediction-workers 16 \
  --no-refit \
  --out-dir outputs/hardware_cost_optimization_nsga2_only_20260501
```

正式报告建议使用 `--pair-extrapolation-mode forbid`。该模式要求所有候选参数都落在需求方两个参考成本点覆盖的区间内，避免使用外推出来的低成本高收益配置。`allow` 只适合探索，不建议作为验收图使用。

没有 bundle 时不要把 CSV 结果当作模型输入。如果只是验证流程，可以运行 prior-only fallback：

```bash
python calibrator/hardware_cost_optimizer.py \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --device1-op-thread-scale-config outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/op_thread_scale_config_joint_6d_tuple_source_100.json \
  --device2-op-thread-scale-config outputs/cache_residency_worker_share_v3_full_s920x20_20260505/op_thread_scale_config_joint_6d_tuple_source_100.json \
  --budgets 1.0,1.1,1.2,1.3,1.5,1.75,2.0,device2 \
  --workload-set dlrm_s_m_l \
  --search-algorithm grid \
  --no-refit \
  --allow-prior-only-fallback \
  --out-dir outputs/hardware_cost_optimization_surrogate_prior_only_20260505
```

## 8. 输出文件

| 文件 | 说明 |
|---|---|
| `candidate_hardware_grid.csv` | 所有候选硬件参数和代理成本 |
| `candidate_parameter_costs.csv` | 每个候选、每个 x1-x25 参数的取值和相对成本 |
| `hardware_cost_candidate_range_audit.csv` | 每个参数的候选取值范围审计，确认是否越过 v2 约束 |
| `hardware_cost_constraints_v2_resolved.csv` | 解析后的参数约束配置 |
| `candidate_predictions.csv` | 每个候选、每个 workload、每个模型的 E2E latency |
| `candidate_objectives.csv` | 每个候选的加权目标函数值 |
| `pareto_frontier.csv` | 成本-性能 Pareto 前沿 |
| `best_by_budget.csv` | 每个预算档的最优候选；v2 实验中预算单位为万元 |
| `nsga2_candidate_objectives.csv` | NSGA-II 实际评估过的候选及目标值 |
| `nsga2_pareto_frontier.csv` | NSGA-II 评估集合上的 Pareto 前沿 |
| `nsga2_best_by_budget.csv` | NSGA-II 评估集合中每个预算档的最优候选 |
| `nsga2_generation_summary.csv` | 每代 population、frontier 和已评估候选统计 |
| `optimization_sanity_audit.csv` | Pareto、预算单调性和候选物理合理性检查 |
| `pareto_frontier_cost_latency.png/pdf` | 成本-性能 Pareto 前沿可视化 |
| `best_candidate_hw_config.yaml` | 默认推荐候选硬件配置 |
| `optimization_manifest.json` | 运行配置、模型 hash、候选数量 |
| `HARDWARE_COST_OPTIMIZATION_REPORT.md` | 可读报告 |

## 9. 结果检查

基本检查项：

- `optimization_manifest.json` 中 `refit=false`。
- Device_1 baseline cost 等于 `1.0`。
- `best_by_budget.csv` 中每一行都满足 `cost_wan <= budget_wan`。
- 预算升高时，最优 `objective_e2e_us` 不应上升。
- `pareto_frontier.csv` 中不存在更贵且更慢的候选。
- `optimization_sanity_audit.csv` 不应出现 `SANITY_FAIL`。
- 如果有 `SANITY_WARNING`，报告必须解释原因；有 warning 的候选不会优先写成默认推荐配置。
- `pareto_frontier_cost_latency.png/pdf` 应存在且能展示候选点、Pareto 前沿、预算线和预算最优点。

如果使用 prior-only fallback，报告标题和 manifest 中必须明确标注，不能把它当作 RF v3 优化结果。
