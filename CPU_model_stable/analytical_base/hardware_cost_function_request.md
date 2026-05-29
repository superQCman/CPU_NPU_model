# Hardware Cost Function Request

提供一个整体成本转换函数：

```text
cost = f(x1, x2, ..., x25)
```

其中 `x1...x25` 是下表中的硬件参数。函数可以是显式公式、分段函数、SKU lookup table、插值函数，或可执行代码接口，但输出应是同一计价口径下的总成本。



| 变量 | 模块 | 参数 | 单位 | 含义 | 绑定关系/关联参数 | `kunpeng920_single_numa` | `s920x20_single_numa` | 需求方补充 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `x1` | Core | `core.total_cores` | cores | 当前单 NUMA 执行域可用物理核心数 | B1 CPU SKU；关联 B4 饱和带宽、B5 active cache 容量 | 24 | 40 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x2` | Core | `core.cpu_clock` | GHz | 有效 CPU 频率，用于吞吐和 cycles-to-time 转换 | B1 CPU SKU；也影响 B2/B3/B5 latency 换算 | 2.592166 | 2.893398 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x3` | Pipeline | `pipeline.fetch_width` | inst/cycle | 前端取指有效宽度 | B2 pipeline/uarch width bundle；与 x4-x8 绑定 | 2.938467 | 3.891586 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x4` | Pipeline | `pipeline.decode_width` | inst/cycle | 指令解码有效宽度 | B2 pipeline/uarch width bundle；与 x3-x8 绑定 | 2.938467 | 3.891586 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x5` | Pipeline | `pipeline.rename_width` | inst/cycle | rename 有效宽度 | B2 pipeline/uarch width bundle；与 x3-x8 绑定 | 2.938467 | 3.891586 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x6` | Pipeline | `pipeline.dispatch_width` | inst/cycle | dispatch 有效宽度 | B2 pipeline/uarch width bundle；与 x3-x8 绑定 | 2.938467 | 3.891586 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x7` | Pipeline | `pipeline.issue_width` | inst/cycle | issue 有效宽度 | B2 pipeline/uarch width bundle；与 x3-x8 绑定 | 2.938467 | 3.891586 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x8` | Pipeline | `pipeline.commit_width` | inst/cycle | commit/retire 有效宽度 | B2 pipeline/uarch width bundle；与 x3-x8 绑定 | 2.938467 | 3.891586 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x9` | Instruction | `instruction.simd_width_bits` | bits | SIMD 向量宽度 | B3 SIMD/FP execution bundle；与 x10-x12 绑定 | 128 | 256 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x10` | Instruction | `instruction.fp_throughput_per_cycle.vector_sp_add` | vector inst/cycle | FP32 vector add 吞吐 | B3 SIMD/FP execution bundle；与 x9、x11、x12 绑定 | 1.499752 | 1.999786 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x11` | Instruction | `instruction.fp_throughput_per_cycle.vector_sp_mul` | vector inst/cycle | FP32 vector multiply 吞吐 | B3 SIMD/FP execution bundle；与 x9、x10、x12 绑定 | 1.499756 | 1.999792 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x12` | Instruction | `instruction.fp_throughput_per_cycle.vector_sp_fma` | vector inst/cycle | FP32 vector FMA 吞吐 | B3 SIMD/FP execution bundle；与 x9-x11 绑定 | 1.749673 | 1.999777 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x13` | Memory | `memory.bandwidth_gb_s_total` | GB/s | 当前执行域总顺序内存带宽 | B4 memory subsystem bundle；与 x14-x18 绑定 | 41.0935 | 138.2280 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x14` | Memory | `memory.copy_bw_gb_s_1t` | GB/s | 单线程连续 copy 带宽 | B4 memory subsystem bundle；copy 1T 与 x15 强绑定 | 5.048952 | 35.900316 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x15` | Memory | `memory.copy_bw_gb_s_saturated` | GB/s | 多线程连续 copy 饱和带宽 | B4 memory subsystem bundle；与 x1、x13、x14 关联 | 30.820562 | 137.967158 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x16` | Memory | `memory.random_gather_bw_gb_s_1t` | GB/s | 单线程随机 gather 带宽 | B4 memory subsystem bundle；random 1T 与 x17 强绑定 | 0.840198 | 2.523200 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x17` | Memory | `memory.random_gather_bw_gb_s_saturated` | GB/s | 多线程随机 gather 饱和带宽 | B4 memory subsystem bundle；与 x1、x16、x18 关联 | 4.160243 | 18.047464 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x18` | Memory | `memory.local_mem_delay_ns` | ns | 本地 NUMA DRAM 延迟，越低通常越好 | B4 memory subsystem bundle；与 x13-x17 同源 | 91.0687 | 88.0950 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x19` | Cache | `cache.cacheline_bytes` | bytes | cache line 大小 | B5 cache/uarch hierarchy bundle；通常与架构固定 | 64 | 64 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x20` | Cache | `cache.l1d.size` | KiB | 每核心 L1D 容量 | B5 cache/uarch hierarchy bundle；与 x23 latency 关联 | 64 | 64 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x21` | Cache | `cache.l2.size` | KiB | 每核心 L2 容量 | B5 cache/uarch hierarchy bundle；与 x24 latency 关联 | 512 | 1280 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x22` | Cache | `cache.l3_per_die.size` | MiB | 每 die/NUMA L3 容量 | B5 cache/uarch hierarchy bundle；与 x1、x25 关联 | 24 | 70 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x23` | Cache | `cache.l1d.response_latency_cycles` | cycles | L1D 有效响应延迟，越低通常越好 | B5 cache/uarch hierarchy bundle；与 x20 容量关联 | 11.662672 | 13.024487 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x24` | Cache | `cache.l2.response_latency_cycles` | cycles | L2 有效响应延迟，越低通常越好 | B5 cache/uarch hierarchy bundle；与 x21 容量关联 | 21.333783 | 31.850238 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |
| `x25` | Cache | `cache.l3_per_die.response_latency_cycles` | cycles | L3 有效响应延迟，越低通常越好 | B5 cache/uarch hierarchy bundle；与 x22 容量关联 | 121.825564 | 126.632175 | 取值范围、合法 SKU 组合、整体成本函数中的处理方式 |

绑定组说明：

- B1 CPU SKU/Core-Frequency：`x1-x2` 通常应作为合法 CPU/SKU tuple 变化，不建议随意组合任意核心数和任意频率。
- B2 Pipeline/Uarch Width：`x3-x8` 是同一微架构流水线能力的不同观测维度，当前两个 YAML 中也都使用同一个 effective issue-width proxy；除非有明确微架构模型，否则应绑定变化。
- B3 SIMD/FP Execution：`x9-x12` 由 SIMD 宽度和 FP/vector 执行单元共同决定，应作为向量执行能力 bundle。
- B4 Memory Subsystem：`x13-x18` 应来自同一内存通道、NUMA、内存代际和实测带宽/延迟配置；其中 `x14-x15`、`x16-x17` 分别是 1T 与饱和能力对。
- B5 Cache/Uarch Hierarchy：`x19-x25` 应来自同一 cache hierarchy；容量和 latency 不应被当成完全独立旋钮。

需求方需要返回的内容建议至少包含：

1. `cost=f(x1,...,x25)` 的定义形式，或可以调用的表/接口。
2. 每个参数的合法取值范围，以及哪些参数必须作为同一个 CPU/SKU bundle 同时变化。
3. 单位口径，例如成本是整机价格、单 socket 价格、单 NUMA 域折算价格，还是相对成本指数。
4. 单调性约束说明：吞吐/容量类参数升高时成本通常不应下降；延迟类参数降低时成本通常不应下降。

## 当前临时 surrogate cost 方案

在需求方提供正式成本函数前，项目内先使用一个临时代理成本函数，用于把硬件参数搜索流程跑通。该函数不是报价模型，也不代表真实采购成本。

代理成本使用两层结构：

$$C(h)=\sum_{g\in G}W_gC_g(h_g)$$

其中 `G={core,pipeline,vector,memory,cache}`，`W_g` 是一级 bundle 权重，`C_g(h_g)` 是组内多个硬件参数的子成本。Device_1 / `kunpeng920_single_numa` baseline 成本定义为 `1.0`。

一级权重如下：

| Bundle | 权重 | 覆盖参数 |
|---|---:|---|
| Core/Frequency | 0.25 | `x1-x2` |
| Pipeline width | 0.10 | `x3-x8` |
| SIMD/FP execution | 0.15 | `x9-x12`，以及当前 YAML 中的 FP latency 字段 |
| Memory subsystem | 0.30 | `x13-x18` |
| Cache hierarchy | 0.20 | `x20-x25` |

`x19 cache.cacheline_bytes` 当前固定为 64，不进入优化成本项。

容量和吞吐类参数越高，代理成本越高；latency 类参数越低，代理成本越高。该 surrogate cost 只用于 `calibrator/hardware_cost_optimizer.py` 的临时设计空间分析。后续拿到真实 `cost=f(x1,...,x25)` 后，只替换 `C(h)`，性能模型、候选硬件枚举和 Pareto 筛选框架保持不变。

优化阶段必须使用已有训练好的模型 bundle，不重新训练 RF，也不重新拟合 op-thread scale。已有 `combined_summary.csv`、`prediction_error_rows.csv`、`e2e_pipeline_summary.csv` 和 `op_thread_scale_config_*.json` 不能恢复 RF 权重，不能替代 `model_bundles/*.pkl`。
