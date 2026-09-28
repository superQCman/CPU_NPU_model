# CPU 算子计算性能建模方案设计文档

本文件用于支撑第一阶段验收中的 CPU 算子计算性能建模部分，说明模型为什么这样设计、输入特征如何进入模型、Analytical Model 与机器学习残差模型如何配合，以及当前交付代码和结果如何对应验收要求。NPU 建模中关于 Cube/Vector 算力的部分不在本文件展开，后续可在同一套文档结构下补充 NPU 特征、算子映射和校准方案。

## 1. 项目背景与验收要求

当前项目面向 DLRM-like 计算图的 CPU 性能预测。模型首先预测图中各类 CPU 算子的执行时间，再把单算子预测结果放回图结构中，通过 E2E pipeline 计算完整计算图的端到端 latency。这样做的原因是 DLRM-like 图中存在 embedding branch、dense MLP、feature interaction、shape/control 小算子和多个并行分支。仅对算子耗时求和无法反映 branch 并行、barrier、merge 和 critical path 对最终端到端耗时的影响。

当前 CPU 主线 route 为：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

文中使用的几个缩写和指标口径如下。YAML 指硬件配置文件，用于保存 CPU 频率、指令位宽、指令延时、流水线宽度、cache、内存带宽和 runtime 开销等参数。RF 指 Random Forest，即随机森林残差模型。E2E 指 end-to-end，表示完整计算图从输入到输出的端到端耗时。MAPE 是平均绝对百分比误差，计算方式为 `mean(abs(pred - actual) / actual)`；P90 APE 是绝对百分比误差的第 90 百分位，表示 90% 的样本误差不超过该值。文中所有 latency 默认使用微秒 `us`，除非显式标注为 `ns` 或 `s`。

整体链路从硬件 YAML 和 workload 特征开始，先经过 Analytical prior 得到白盒服务时间估计，再通过 `(op_type, num_threads)` 级别的 op-thread scale 修正线程尺度偏差，然后由 physics-guarded RF 学习白盒估计之外的残差，最后进入 E2E pipeline 得到完整图预测。这个链路的设计重点不是让随机森林直接记住某台机器，而是先把 CPU 频率、SIMD 位宽、指令延时、流水线宽度、cache 和内存等硬件影响转成可解释的服务时间基准量，再让机器学习模型只承担残差校准。

第一阶段验收对 CPU 部分的要求可以归纳为三类。第一类是文档要求，需要给出 CPU 算子计算性能建模方案，并说明技术路线。第二类是模型代码要求，需要交付可运行的模型代码和代码说明。第三类是指标要求，CPU 模型必须覆盖指令位宽和延时、CPU 频率、流水线宽度，建模精度高于 85%，推理耗时小于 1min。当前交付中，CPU 指令位宽对应 `instruction.simd_width_bits`，指令延时对应 `instruction.fp_latency_cycles.*`，CPU 频率对应 `core.cpu_clock`，流水线宽度对应 `pipeline.fetch_width/decode_width/dispatch_width/issue_width/commit_width` 等字段。这些字段均来自 `calibrator/hw_config/*.yaml`，并在 `calibrator/whitebox_family_routes.py` 中进入 Analytical Model。

精度口径使用：

```text
accuracy = 1 - MAPE
```

当前建议将完整图 E2E latency 作为第一阶段 CPU 主验收口径。原因是模型最终交付目标是完整计算图性能预测，而 E2E latency 同时覆盖了单算子预测、图结构并行和调度路径。单算子 MAPE 仍保留为诊断指标，用于定位不同算子类别的误差来源。

## 2. CPU 算子性能建模背景与挑战

DLRM-like 负载不是单一瓶颈的计算图。Embedding `Gather` 主要受随机访存和 cache residency 影响，dense MLP 中的 `Gemm/MatMul` 主要受 MAC 数、SIMD 宽度、FP throughput 和内存读写共同影响，`ReduceSum` 同时包含连续访存和规约计算，`Concat` 可能退化为 payload copy，而 `Shape`、`Unsqueeze`、`Flatten` 等小算子更多体现 runtime dispatch 和 metadata touch。不同算子的规模差异也很大：一部分算子真实耗时在微秒级，相对误差容易被放大；另一部分算子虽然数量不多，却可能落在 E2E critical path 上，对完整图 latency 影响更直接。

CPU 硬件侧同样不是一个单独参数能描述的系统。计算吞吐受 SIMD 位宽、每周期 FP throughput、CPU 频率和可用 worker 数影响；短依赖链或规约尾部会受到 FP 指令延时和流水线并行能力影响；连续访存取决于 copy/STREAM 类带宽；embedding lookup 则更依赖随机访存带宽、cache 容量、cache/DRAM 延时和 working set。线程数变化时，算子内部并行度、branch 并行组、调度开销以及共享 cache 分配都会同步变化。

如果把这些硬件字段直接输入 RF，模型在训练机器数量有限时容易学习到硬件 ID 或历史平台差异，而不是可外推的硬件趋势。因此当前方案采用白盒先验和残差学习组合的方式：Analytical Model 先负责把硬件和算子规模转换成物理含义明确的服务时间估计，RF 再学习 runtime、library 实现、小算子 floor 和不可完全建模的系统误差。

## 3. CPU 算子性能建模整体框架

该框架以 DLRM 负载特征、DLRM 计算图和 CPU 硬件参数为输入，输出每个算子的预测耗时。DLRM 负载特征包括 batch size、每次 lookup 的索引数、embedding size、bottom MLP size、inter-op 线程数和每算子线程数；计算图给出 `Gather`、`ReduceSum`、`MatMul` 等算子的连接关系与执行位置。两者共同经过静态特征转换，形成每个节点的算子规模与 shape 特征。硬件参数则描述 CPU 的可用计算、访存和调度能力，包括频率、SIMD 指令位宽、指令延时与吞吐、流水线宽度、cache、内存带宽和 runtime 开销。

整体处理链路如下。训练时以单算子实测耗时完成模型校准；部署时从 DLRM 配置直接构造静态特征，不需要重新采集 profiling 数据。

```text
DLRM 负载特征 + DLRM 计算图
batch / lookup / embedding / MLP / intra-op、inter-op 线程
                              |
                              v
                    算子静态特征转换
                              |
                              v
       算子规模与 shape 特征              硬件参数特征
 I/O bytes / lookup / Gemm M/N/K / MAC     频率 / SIMD / cache / 带宽 / 流水线
                   \                         /
                    \                       /
                     v                     v
                     Analytical Model
    连续访存模块 | 随机访存模块 | 计算模块 | 运行时调度开销模块
                              |
                              v
     算子耗时上下界 / 算子基准耗时 / 算子分组 /
     各模块耗时上下界 / 各模块占比与关系
                              |
     非规模静态语义与算子上下文特征 ---------+
                              |
                              v
                        随机森林模型
                              |
                              v
                        单算子预测耗时
```

图中的第一条输入线由 DLRM 负载特征和计算图共同构成。负载特征回答的是某一轮执行的规模：例如，`batch_size * num_indices_per_lookup` 给出 embedding lookup 次数，MLP 层宽可以展开为 `Gemm/MatMul` 的 M、N、K 和 MAC 数。计算图回答的是这些规模属于哪个算子、该算子具有何种输入输出 shape，以及它处在 embedding、dense 或 interaction 等何种上下文中。静态特征转换将这两类信息落实为一行一算子的特征表，使上层配置可以被转换成模型可处理的算子工作量。

硬件参数不直接输入随机森林，而是先进入 Analytical Model。该模型按照图中的四个模块估计单算子的服务时间：连续访存模块处理激活和参数的顺序读写；随机访存模块处理 `Gather` 中由 cache 命中、working set 和 DRAM 访问决定的 lookup；计算模块处理向量化浮点运算；运行时调度开销模块处理线程启动、任务调度和 metadata 小算子的控制路径。对同一模块，模型分别给出乐观、中心和保守三种估计，以形成模块耗时的下界、基准值和上界。

四类模块依据算子行为组合为白盒先验。`Gather` 主要由随机访存与连续访存竞争决定，`Gemm/MatMul` 主要在计算和连续访存之间取瓶颈，`ReduceSum` 同时包含规约计算与数据搬运，metadata/view 类算子则以调度开销为主。组合后输出的 `whitebox_lower_us`、`whitebox_center_us` 和 `whitebox_upper_us` 分别对应图中的算子耗时下界、基准耗时和上界。模型还会为每个算子标记供残差模型分别学习的类别，例如索引访存、矩阵计算、规约、数据搬运和元数据处理；同时给出连续访存、随机访存、计算和运行时调度等组成部分的耗时下界、基准值和上界，并计算各组成部分在总服务时间中的占比、主导瓶颈，以及访存、计算和调度之间的相对关系。这些可解释的中间结果既作为随机森林的输入，也用于分析预测误差来自哪个环节。

随机森林还接收非规模静态语义和算子上下文特征，例如数据类型、broadcast 类型、transpose/reduce 布局、shape metadata 类型、算子输入输出个数、线程配置和图中分支属性。它学习的是白盒基准耗时之外的残差，用于修正 runtime 实现、库函数选择、小算子 floor 和未被完全表达的 cache 离散效应。主线 v3 在白盒基准耗时上额外应用按 `(op_type, num_threads)` 拟合的 scale，以校正同类算子在线程数变化时的系统性偏差；这一校准后的白盒量与上述特征共同输入随机森林。最终输出为单算子的预测耗时，而不是由随机森林直接根据硬件型号或原始硬件字段作出预测。

## 4. CPU 建模输入特征

CPU 模型的输入由硬件参数、算子规模特征和图结构特征共同组成。硬件参数的可信来源固定为 `calibrator/hw_config/*.yaml`。这些 YAML 不是手写占位配置，而是由 `hardware_calibration/` 下的 microbenchmark 和校准脚本生成或复核。当前方案采用的测量约定是：cache latency 来自 pointer-chase `ns/load`，指令延时来自 dependent FP chains，指令吞吐来自 independent vector loops，流水线宽度使用 independent integer-add loop 上 `perf instructions/cycles` 的有效宽度代理，CPU 频率使用同一 pinned issue-loop microbench 上的 `perf cycles/task-clock`。

在当前 CPU route 中，`core.cpu_clock` 不作为厂商标称峰值频率使用，而是作为实测有效 GHz 参与 cycles 到时间的换算。`instruction.simd_width_bits` 表示向量指令位宽，用于把 vector throughput 转成 FP32 lane 级吞吐能力。`instruction.fp_latency_cycles.vector_sp_*` 描述 add、mul、fma 等操作的依赖链延时，用于估计 compute latency tail。`pipeline.fetch_width/decode_width/rename_width/dispatch_width/issue_width/commit_width` 则提供流水线吞吐和调度能力的代理值，进入 front-end/back-end capacity、dispatch service time 和 latency parallelism 的估计。

当前正式两台设备的关键 CPU 字段如下。

| 字段 | Device_1 / 智算平台 | Device_2 / 通算平台 |
|---|---:|---:|
| `core.cpu_clock` | 2.592166 GHz | 2.893398 GHz |
| `core.total_cores` | 24 | 40 |
| `instruction.simd_width_bits` | 128 | 256 |
| `instruction.fp_latency_cycles.vector_sp_fma` | 4.000215 cycles | 2.000094 cycles |
| `instruction.fp_throughput_per_cycle.vector_sp_fma` | 1.749673 | 1.999777 |
| `pipeline.fetch/decode/dispatch/issue/commit_width` | 2.938467 | 3.891586 |
| `memory.local_mem_delay_ns` | 91.0687 ns | 88.0950 ns |

需要说明的是，用户态 microbenchmark 无法稳定拆分真实硬件中每个流水线阶段的内部宽度。因此当前 YAML 把同一个有效 issue-width 代理写入多个流水线宽度字段，模型再按使用场景取指定字段或统计量。这比保留不可验证的占位值更稳妥，也方便后续在有更细粒度硬件资料时替换。

算子规模特征来自训练数据或静态配置展开结果。训练阶段，一行样本通常对应图中的一个算子节点，包含 `op_type`、shape、输入输出 tensor 信息、`batch_size`、`num_indices_per_lookup`、`arch_embedding_size`、`arch_mlp_bot`、`arch_mlp_top`、`num_threads`、`inter_threads`、图位置字段和监督标签 `actual_us`。这些字段不会全部直接交给 RF。它们首先用于推导 `output_size`、`activation_size`、I/O bytes、`feat_lookup_count`、`feat_gemm_m/n/k`、`feat_gemm_mac_count`、`feat_reduction_work_items` 等规模字段，再进入 Analytical Model。

训练完成后的静态配置推理使用另一条入口。用户只需给出 DLRM 上层配置，例如 `batch_size`、`num_indices_per_lookup`、`arch_embedding_size`、`arch_mlp_bot`、`arch_mlp_top`、`inter_threads` 和 `num_threads`，`calibrator/static_feature_mapping/core.py` 会展开出完整 `full_graph_76` 和主线可观测的 `model_input_51`。如果只想查看上层参数如何映射到各个算子规模，而不加载模型 bundle、不做 latency 预测，可以运行 `standalone_demos/dlrm_operator_scale_mapping/dlrm_operator_scale_mapping_demo.py`。该 demo 会输出每个算子的 shape、Gemm M/N/K、MAC 数、lookup count、reduction work items 和 I/O bytes。

profiling 数据只在训练和评估阶段作为监督信号使用。训练时需要 `actual_us` 作为标签，也需要 shape CSV 和 branch timeline CSV 来评估 E2E pipeline。模型训练完成后，`predict_from_config.py` 不再读取 profiling 数据、profile 生成的 shape CSV 或 timeline CSV；它只读取模型 bundle 和 DLRM 上层配置。静态推理输出中的 `has_cpu_profile=0`、`shape_source=static_config` 就表示该阶段不依赖 profiling 产物。

## 5. CPU Analytical Model 设计

Analytical Model 是当前方案的物理估计层。它把硬件参数和 workload 规模转换为四类 active service-time modules：连续访存 `stream_mem`、随机访存 `random_mem`、计算 `compute` 和调度 `dispatch`。历史 schema 中仍保留 `descriptor`、`reorder`、`sync` 等列，但当前主线在合成 `whitebox_center_us` 前会把这些历史模块清零，避免未验证项影响结果。

compute prior 分为吞吐项和延时尾项。吞吐项根据算子的 work items、SIMD 位宽、每周期 FP throughput、CPU 频率和有效 worker 数估计：

```text
simd_lane_items = simd_width_bits / 32
capacity_items_per_us
  = fp_throughput_per_cycle
    * simd_lane_items
    * effective_workers
    * cpu_clock_ghz
    * 1000

T_compute_throughput = work_items / capacity_items_per_us
```

这里的 `simd_width_bits` 对应验收要求中的指令位宽，`cpu_clock_ghz` 对应 CPU 频率，`fp_throughput_per_cycle` 来自硬件校准。乘以 `1000` 是因为 GHz 表示每纳秒 cycle 数，需要换算到每微秒处理能力。延时尾项用于描述依赖链或规约尾部难以完全被吞吐隐藏的部分：

```text
T_compute_latency_tail
  ~= alpha * fp_latency_cycles * latency_depth
      / (cpu_clock_ghz * 1000 * parallelism)
```

其中 `fp_latency_cycles` 对应验收要求中的指令延时，`parallelism` 由 worker 数和 `issue/dispatch/commit_width` 等流水线宽度代理共同决定。代码中 `Gemm/MatMul` 默认使用 FMA latency，`Add/ReduceSum` 使用 add latency，`Mul` 使用 mul latency。

stream memory prior 处理连续读写和 payload 搬运，基本形式是用 I/O bytes 除以带宽。带宽来自 `memory.copy_bandwidth_*`、`memory.bandwidth` 等校准字段。对于 shape 或 view 类算子，模型不会简单按 tensor 规模强行加入大规模 copy，因为这类算子在实际运行中通常只修改 metadata，不一定发生 payload materialization。

random memory prior 主要用于 embedding `Gather`。模型先根据 `batch_size * num_indices_per_lookup` 估计 lookup count，再结合 embedding table 规模估计 working set。随后根据当前 worker 数、活跃 branch group 数和 cache 容量，判断访问更接近 L1、L2、L3 还是 DRAM。L1/L2 按 per-core private cache 随 worker 数扩展，L3/LLC 作为共享 cache，会按当前算子 worker 在系统活跃 worker 中的占比分配。这个设计使得相同 lookup count 在不同线程数和不同并行组下可以得到不同的随机访存估计。

dispatch prior 则用于补充线程启动、任务调度和小算子控制开销。它会读取 `runtime.parallel_launch_overhead_us_*`，并通过 `pipeline.dispatch_width`、worker span 和 group span 做分摊。当图中存在多个并行 branch group 时，模型还会加入 inter-group overhead。metadata/view 类小算子最终多按 dispatch-only 处理，这是因为它们的真实 payload work 很少，主要时间来自 runtime control path。

不同算子的最终白盒估计由这些 service-time modules 组合得到。Embedding `Gather` 使用 `max(T_random, T_stream) + T_dispatch`，`Gemm/MatMul` 和 `ReduceSum` 使用 `max(T_stream, T_compute) + T_dispatch`，payload `Concat` 使用 `T_stream + T_dispatch`，metadata/view 类使用 `T_dispatch`。组合后的中心估计写入 `whitebox_center_us`，同时保留 `whitebox_lower_us`、`whitebox_upper_us` 和 `svc_*_center_us` 等列，供后续校准和诊断使用。

## 6. CPU 机器学习残差模型与校准

Analytical Model 能描述主要物理趋势，但真实系统中仍存在 runtime 实现细节、cache 行为离散化、小算子 floor、library 选择和测量噪声等误差。因此当前主线在白盒估计之后增加两层校准：先做 op-thread scale，再做 physics-guarded RF。

op-thread scale 是一个简单但有效的比例校准层。它按 `(op_type, num_threads)` 分组，在 fit split 上拟合：

```text
s_op,t = weighted_median(actual_us / whitebox_center_us)
whitebox_scaled_us = whitebox_unscaled_us * s_op,t
```

这一步不是新的黑盒模型，而是把不同算子类型在线程数变化下的系统性偏差显式记录下来。比例只在训练集上拟合，训练完成后作为模型的一部分固定下来，使同一算子在线程数变化时先获得稳定的尺度修正，再进入残差学习。

机器学习层的目标不是从零开始预测 `actual_us`，而是补足 Analytical Model 尚未表达的部分。对于 `Gather`、`Concat`、`ReduceSum`、`Transpose`、`Gemm/MatMul` 等具有明确数据搬运或计算规模的算子，模型学习白盒基准耗时的对数比例残差：

```text
r     = log(actual_us / whitebox_center_us)
r_hat = RF(features)
pred_us = whitebox_center_us * exp(r_hat)
```

其中，`actual_us` 是训练样本的实测算子耗时，`whitebox_center_us` 是 Analytical Model 给出的中心估计，`r` 是由实测值计算出的真实对数比例残差，也是随机森林的训练标签。`features` 表示输入给随机森林的白盒性能基准特征、算子静态语义和图上下文特征；`RF(features)` 输出对该残差的预测，记为 `r_hat`。最后用 `exp(r_hat)` 将对数残差还原为比例系数，并乘回白盒基准，得到最终预测 `pred_us`。因此 `r_hat = 0` 时，预测值等于白盒基准；`r_hat = log(2)` 时，预测值为白盒基准的两倍；`r_hat = log(0.5)` 时，预测值为白盒基准的一半。

这种形式让随机森林回答“白盒估计应放大还是缩小多少”，而不是重新学习完整的硬件规律。对数比例能使两倍高估和两倍低估在训练空间中具有对称量级，也使延时跨越微秒到毫秒时的目标尺度更稳定。`whitebox_center_us` 已包含频率、SIMD 位宽、指令延时、流水线宽度、cache 和内存带宽的影响，因此残差模型主要修正 runtime 实现、并行分块、库函数路径、cache 离散跳变和未建模的调度开销。

metadata/view 与 tiny shape arithmetic 两类小算子采用另一种目标。它们的白盒基准可能接近零，而绝对耗时常由 runtime floor 决定，直接计算 `actual_us / whitebox_center_us` 会产生不稳定的大比例。因此这两类算子直接学习对数耗时：

```text
z = log(1 + actual_us)
pred_us = exp(RF(features)) - 1
```

这不是放弃白盒信息。白盒基准、白盒不确定区间和调度模块耗时仍作为输入特征；只是训练目标从“比例校正”改为“稳定地拟合微秒级绝对耗时”。最终预测仍会受白盒上下界扩展出的合理区间限制，避免小算子出现不合物理量级的输出。

当前实现采用 sklearn 的 `RandomForestRegressor`，而不是一个覆盖全部算子的统一回归器。模型按执行行为拆成七个子模型：metadata/view、tiny shape arithmetic、indexed block copy、payload concat、reduction layout、transpose generic 和 gemm-like。每个子模型只面对相近的瓶颈机制，例如 indexed block copy 主要处理 `Gather` 的随机访存，gemm-like 主要处理矩阵计算，reduction layout 主要处理规约与布局。这样可以避免让同一棵树同时比较“一个 metadata 查询”和“数十亿次 MAC”，也使每类模型的特征和学习目标更符合该类算子的误差来源。训练样本不足的类别不会勉强拟合，而是回退到预设的相近类别模型或白盒基准。

每个子模型使用 512 棵决策树，单棵树最大深度为 16，叶节点至少包含 4 个训练样本，每次分裂只从候选特征的平方根数量中选择。随机森林通过对训练样本和特征进行随机抽样并平均多棵树的输出，降低单棵树对特定 workload 或测量噪声的依赖。每个子模型最多使用 30 个特征，并在训练前进行按配置组划分的验证；只有验证结果不劣于白盒基准时才启用该子模型。这些限制的目的不是压缩工程规模，而是控制模型容量，使残差学习保持在“校正白盒”而不是“记忆样本”的范围内。

随机森林的输入可以分为三层。第一层是白盒性能基准特征，包括算子基准耗时、上下界、连续访存/随机访存/计算/调度模块耗时及其占比；它们告诉模型当前算子的主要物理瓶颈。第二层是算子静态语义，例如数据类型、broadcast 类型、规约轴、transpose 布局、shape/view 类型和输入输出张量关系；它们用于区分相同规模但执行路径不同的算子。第三层是图上下文，例如前驱和后继算子类型、fan-in/fan-out、距 merge 或输出节点的距离、分支数和线程配置；它们用于描述同一个算子在不同图位置下可能面对的同步和调度差异。数值跨度很大的白盒耗时和服务时间特征会做 `log1p` 变换，其余类别编码和比例特征保持原值。

physics-guarded RF 的关键约束是 RF 不直接读取硬件型号、硬件 ID 或原始硬件字段。硬件影响先通过 Analytical Model 转化为 `whitebox_center_us`、`whitebox_exec_center_us`、连续访存、随机访存、计算和调度等白盒性能基准特征；RF 再依据这些基准特征、静态语义和图上下文学习残差。这样，残差模型不能绕过物理估计层直接把某台机器当作一个类别记住。

为了让 RF 在训练阶段看到合理的硬件变化方向，v3 使用硬件参数扰动训练。本文将由真实样本派生、但没有重新实测的一行数据称为“硬件参数扰动构造样本”，其耗时标签称为“构造耗时标签”。对一条真实样本，原始硬件下的实测耗时记为 `y0`，白盒估计记为 `p0`。训练时依次将 cache 容量和延时、内存带宽和延时、CPU 频率、SIMD 位宽、向量指令吞吐/延时及流水线宽度等一个硬件字段调整为原值的 80% 或 120%，并在该硬件条件下重新计算白盒基准和白盒性能基准特征，得到 `ph` 和 `features_h`。对应的构造耗时标签为：

```text
y_h = p_h + (y0 - p0)
```

这里保留的是原始硬件下以微秒计的白盒误差 `y0 - p0`，而硬件变快或变慢的方向及幅度由新的白盒估计 `ph` 决定。例如原始实测为 `100 us`、原始白盒估计为 `90 us`，则白盒误差为 `+10 us`；若调高内存带宽后白盒估计变为 `70 us`，构造标签就是 `80 us`。反过来，若调高内存延时后白盒估计变为 `125 us`，构造标签就是 `135 us`。这不是把构造标签视为新的实测值，而是在缺少该硬件组合实测数据时，为 RF 提供受物理规律约束的训练方向。

`y_h` 只作为构造样本的训练标签，不是 RF 的输入特征。对于采用对数比例残差的算子组，RF 看到的是扰动硬件下重新计算出的 `features_h`，学习目标相应变为：

```text
r_h = log(y_h / p_h)
r_hat_h = RF(features_h)
pred_h = p_h * exp(r_hat_h)
```

也就是说，RF 学习“在这组白盒基准特征下还应补多少比例误差”。对于直接预测对数耗时的小算子组，训练目标则为 `z_h = log(1 + y_h)`。真实样本的训练权重为 `1.0`，每条硬件参数扰动构造样本的权重为 `0.25`，因此构造数据用于补充硬件变化信息，但不会取代真实测量。交叉验证时，验证折只包含真实样本；属于该验证折同一来源样本的所有构造版本也会从训练折排除，所以真实 validation MAPE 不受构造标签影响。

推理阶段还有两类保护。第一类是单调约束，它对关键服务时间基准特征施加正向约束，避免基准特征变大时 RF 反而预测更快。第二类是 hardware guard。当进行硬件参数 sweep 或 `override_hw_config` 推理时，系统会比较 baseline 硬件和目标硬件下的 whitebox/RF 变化方向。如果 RF 变化方向与 whitebox 方向冲突，则回退到 frozen residual；如果方向一致但 RF 变化幅度过大，则限制 RF 变化幅度。这样可以减少残差模型破坏硬件物理趋势的风险。

## 7. 精度、耗时和验收口径

当前正式结果来自 full split 目录：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505
```

E2E 主验收指标如下。

| 数据/硬件绑定 | E2E MAPE | E2E accuracy = 1-MAPE | E2E P90 APE | 完整图组合数 |
|---|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 5.96% | 94.04% | 12.35% | 151 |
| Device_1 / 智算平台 full portable 数据 | 5.47% | 94.53% | 11.19% | 454 |

两台设备的 E2E accuracy 均高于 85%，满足当前定义下的第一阶段主验收精度要求。该结果不是单算子求和误差，而是经过 E2E pipeline 后的完整图端到端误差，覆盖了图结构并行和 critical path。

单算子精度的最终统计采用计算图任务 MAPE。统计单元不是孤立的一行算子记录，而是图中一个数据处理任务及其相邻的控制类节点。`Shape`、`Unsqueeze`、`Flatten`、`Reshape`、`Add` 和 `Mul` 先按静态图依赖关系归入相邻的数据处理算子：优先选择直接前驱；若没有可用的直接前驱，则选择直接后继。归属规则只依赖静态计算图和算子类型，不读取 `actual_us`、预测误差或测试标签。对每个任务，实测耗时与预测耗时分别求和，再计算该任务的绝对百分比误差；最终 MAPE 为全部任务绝对百分比误差的平均值：

```text
actual_task = sum(actual_us of task members)
pred_task   = sum(pred_us of task members)
APE_task    = abs(pred_task - actual_task) / actual_task
MAPE        = mean(APE_task)
```

当前 `whitebox_family_prior_plus_physics_guarded_rf_v3` 的计算图任务 MAPE 如下。训练集仅用于拟合 Analytical Model 的校准项和 RF 残差模型；表中的计算图任务数、MAPE 和 accuracy 均在未参与训练的测试集上统计。

| 数据/硬件绑定 | 训练集算子行数 | 测试集算子行数 | 测试集计算图任务数 | MAPE | accuracy = 1-MAPE |
|---|---:|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 18,003 | 7,701 | 4,983 | 12.86% | 87.14% |
| Device_1 / 智算平台 full portable 数据 | 53,958 | 23,154 | 14,982 | 10.49% | 89.51% |

两台设备在测试集上的计算图任务 accuracy 均超过 85%。原始逐行单算子 MAPE 仍作为诊断指标保留：Device_2 为 20.61%，Device_1 为 15.71%。逐行统计中包含大量微秒级 metadata/view 和 elementwise 节点，其相对误差容易被放大；计算图任务 MAPE 则以完整的数据处理任务作为评价对象。两个指标的样本单元不同，不能混用或互相替代。

为分离残差模型带来的影响，补充使用 `whitebox_family_prior_only_v2` 路线进行对照。该路线保留 Analytical Model 和基于训练集拟合的 `(op_type, num_threads)` 缩放系数，但不启用 RF 残差模型、硬件参数扰动训练或 hardware guard。对照使用与主线相同的测试集和原始预测行，但统计单元调整为 DLRM 的四个连续任务类别：embedding lookup/reduction、bottom MLP、feature interaction 和 top MLP。对每个配置，属于同一类别的节点实测耗时和预测耗时分别求和后计算 APE，再在全部配置上取平均。

| 计算图任务类别 | Device_1 Analytical Model + 缩放校准 MAPE | Device_2 Analytical Model + 缩放校准 MAPE |
|---|---:|---:|
| Embedding lookup + reduction | 5.72% | 9.28% |
| Bottom MLP | 9.56% | 8.50% |
| Feature interaction | 6.21% | 9.75% |
| Top MLP | 9.58% | 3.56% |

该对照下八个“任务类别 × 平台”结果均低于 15%。与主线 RF 结果相比，RF 在 Device_2 的 embedding lookup/reduction 和 feature interaction 上分别将 MAPE 从 9.28% 降至 6.06%、从 9.75% 降至 7.98%；其余类别的差异较小。这表明 Analytical Model 和缩放校准已提供主要的尺度与硬件约束，RF 的作用是针对部分运行时实现差异继续进行残差修正。该类别 MAPE 用于比较各连续执行阶段，不等同于前表按相邻节点构成的逐个计算图任务 MAPE，二者不能直接替代。

计算图任务 MAPE 由 `calibrator/calculate_graph_task_mape.py` 生成。该脚本只读取已有的预测明细和 op-shape CSV，不训练、不重拟合，也不改变主模型预测。复现命令如下：

```bash
conda run -n ort python calibrator/calculate_graph_task_mape.py \
  --graph-shape-csv data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/portable_artifacts/case_10_run_1_1/bs1024_nip1800/op_shapes/op_shapes_1024_1800.csv \
  --device1-prediction-rows outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/prediction_error_rows.csv \
  --device2-prediction-rows outputs/cache_residency_worker_share_v3_full_s920x20_20260505/prediction_error_rows.csv \
  --out-dir outputs/graph_task_mape_cache_residency_worker_share_20260720
```

Analytical Model + 缩放校准对照使用同一命令，并将 route 切换为 `whitebox_family_prior_only_v2`：

```bash
conda run -n ort python calibrator/calculate_graph_task_mape.py \
  --graph-shape-csv data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/portable_artifacts/case_10_run_1_1/bs1024_nip1800/op_shapes/op_shapes_1024_1800.csv \
  --device1-prediction-rows outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/prediction_error_rows.csv \
  --device2-prediction-rows outputs/cache_residency_worker_share_v3_full_s920x20_20260505/prediction_error_rows.csv \
  --route whitebox_family_prior_only_v2 \
  --task-class-plot-title "Analytical Model + op-thread Scale: DLRM Task-Class MAPE" \
  --out-dir outputs/dlrm_task_class_mape_analytical_scale_20260801
```

两条路线的任务类别 MAPE 可合并绘制到同一张柱状图中。图中浅色斜线柱为 Analytical Model + 缩放校准，实色柱为完整模型；橙色和蓝色分别表示 Device_1 与 Device_2：

```bash
conda run -n ort python calibrator/plot_task_class_mape_comparison.py \
  --analytical-summary outputs/dlrm_task_class_mape_analytical_scale_20260801/dlrm_task_class_mape_summary.csv \
  --complete-model-summary outputs/dlrm_task_class_mape_cache_residency_worker_share_20260722/dlrm_task_class_mape_summary.csv \
  --out-dir outputs/dlrm_task_class_mape_comparison_20260801
```

输出目录中的 `graph_task_mape_summary.csv` 保存逐个计算图任务的最终 MAPE，`graph_task_assignment_rows.csv` 保存每个原始算子行归属的计算图任务，`graph_task_prediction_rows.csv` 保存任务级实测值、预测值和 APE，可用于逐项审计结果。连续任务类别实验还会生成 `dlrm_task_class_mape_summary.csv`、`dlrm_task_class_prediction_rows.csv` 和 `dlrm_task_class_mape_bar.png/pdf`，分别保存四类连续任务的 MAPE、每个配置的聚合结果和柱状图。

为查看四类连续任务在全部测试 workload combination 上的误差分布，可基于 `dlrm_task_class_prediction_rows.csv` 绘制相对误差 CDF。该图保留 10% 和 15% 阈值，两个设备分别绘制四条任务类别曲线：

```bash
conda run -n cpu-model-acceptance python calibrator/plot_task_class_relative_error_cdf.py \
  --task-prediction-rows outputs/dlrm_task_class_mape_cache_residency_worker_share_20260722/dlrm_task_class_prediction_rows.csv \
  --threshold 0.10,0.15 \
  --out-dir outputs/dlrm_task_class_relative_error_cdf_20260803
```

每个 Device_1 任务类别包含 454 个测试配置，Device_2 包含 151 个。15% 阈值内，Device_1 四类任务的覆盖率依次为 95.8%、79.7%、92.5% 和 84.6%，Device_2 分别为 94.7%、81.5%、91.4% 和 89.4%。因此，任务类别 MAPE 低于 15% 并不表示每一个 workload configuration 都低于 15%；CDF 用于保留尾部误差分布，作为 MAPE 的补充诊断，而不替代主验收的 E2E 指标。

训练后静态配置批量推理耗时来自：

```text
outputs/static_inference_bulk_latency_benchmark_20260507/static_inference_bulk_latency_summary.csv
```

在 heterogeneous static configs、`sample_count<=64`、阈值 60s 的口径下，`cold_bulk_cli` 最大整批耗时为 57.780s，`warm_bulk_inprocess` 最大整批耗时为 56.315s，均低于 1min。该耗时口径对应模型 bundle 已训练完成后的静态配置推理，不包含重新生成训练数据或重新训练 RF 的时间。

白盒模型还单独测量了连续访存与随机访存模块的计算开销，目的是确认可解释模块本身不会成为评估流程中的隐藏瓶颈。该基准只统计静态特征准备完成后的数值方程执行时间，不包含数据读取、静态特征展开、模型 bundle 加载或 RF 推理，也不应被理解为真实硬件的 Cache/DRAM 访问延迟。连续访存模块包含带宽曲线、争用修正和服务时间计算；随机访存模块包含 `Gather` 的随机带宽、缓存层级延迟、未命中并行度与 ORT Gather 内核重标定。

```bash
conda run -n cpu-model-acceptance python calibrator/benchmark_memory_model_overhead.py \
  --dataset data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/e2e_test_raw.csv \
  --hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --sample-sizes 1,76,1024,4096 \
  --repeats 200 \
  --warmups 20 \
  --out-dir outputs/memory_model_overhead_20260803
```

脚本仅支持 `ort_kernel_v3`，连续访存包含 ReduceSum 的内核带宽修正，并将仅改变视图的 Transpose 搬运量置零。一致性校验逐行比较上下界和中心值，避免汇总求和掩盖偏差；默认校验通过后才写出结果。`--skip-equivalence-check` 会显式记录跳过状态，仅供诊断。历史 20260803 耗时来自修正前脚本，不能代表当前实现；当前开销应以上述命令重新测量。

验收复核时，E2E 精度读取 `best_e2e_pipeline_by_policy.csv`，筛选 route `whitebox_family_prior_plus_physics_guarded_rf_v3` 后检查 `e2e_mape`、`e2e_p90`、`combo_count_full`、`combo_count_partial` 和 `coverage_ratio_mean`。通过条件是 `combo_count_partial=0`、`coverage_ratio_mean=1.0`，并且 `1 - e2e_mape > 85%`。计算图任务 MAPE 读取 `graph_task_mape_summary.csv` 中的 `mape_pct` 和 `accuracy_pct`，并以 `graph_task_assignment_rows.csv`、`graph_task_prediction_rows.csv` 复核任务归属和 APE。原始逐行单算子诊断指标读取 `combined_summary.csv` 中同一 route 的 `mape`、`p90`、`median` 和 `eval_rows`。静态推理耗时读取 `static_inference_bulk_latency_summary.csv`，检查 `max_s` 和 `pass_threshold`；当前验收阈值为 `max_s < 60` 且 `pass_threshold=True`。

## 8. 模型代码说明

CPU 模型代码按职责分为硬件校准、训练评估、静态推理和解释性工具四部分。硬件校准位于 `hardware_calibration/`，其中 `tools/` 存放 cache latency、FP latency、FP throughput、pipeline width、STREAM/copy/random-gather bandwidth 和 thread launch overhead 等 microbenchmark，`scripts/auto_calibrate_hw_config.py` 负责自动运行校准、生成 candidate YAML，并在 gate 通过后提升为 `calibrator/hw_config/<profile>.yaml`。

训练和评估入口集中在 `calibrator/`。`calibrator/run.py` 是主线训练、评估和 bundle 导出的入口；`calibrator/whitebox_family_routes.py` 实现 Analytical prior、op-thread scale、physics-guarded RF、硬件反事实增强、单调约束和 hardware guard；`calibrator/whitebox_perf_feature_rf.py` 提供 RF 残差训练所需的特征与工具函数；`calibrator/pipeline_eval.py` 负责 E2E pipeline 评估、完整图 combo 指标和 coverage 审计；`calibrator/calculate_graph_task_mape.py` 根据固定的静态图归属规则计算计算图任务 MAPE 并导出审计行；`calibrator/plot_task_class_mape_comparison.py` 和 `calibrator/plot_task_class_relative_error_cdf.py` 分别绘制任务类别 MAPE 对照和 CDF；`calibrator/benchmark_memory_model_overhead.py` 测量白盒连续/随机访存模块的 CPU 计算开销；`cpu_perf_model/` 承担图结构和 pipeline scheduling 相关逻辑。

训练后静态推理位于 `calibrator/static_feature_mapping/`。其中 `core.py` 负责从 DLRM 上层配置生成 `full_graph_76` 和 `model_input_51`，`predict_from_config.py` 负责加载模型 bundle 并输出 `predicted_e2e_us`，`benchmark_static_inference_latency.py` 用于验证批量静态配置推理耗时。独立 demo `standalone_demos/dlrm_operator_scale_mapping/` 只用于解释上层参数到算子规模特征的映射，不接入主模型推理链路。

主线训练示例：

```bash
python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root data/data_e2e_full_s920x20_20260501 \
  --artifact-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --source-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --target-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --routes whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode fit \
  --export-model-bundles
```

训练后静态推理示例：

```bash
python calibrator/static_feature_mapping/predict_from_config.py \
  --model-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --batch-size 1024 \
  --num-indices-per-lookup 1000 \
  --arch-embedding-size 25600 \
  --arch-mlp-bot 400-300-200 \
  --inter-threads 1 \
  --num-threads 3 \
  --out-dir outputs/static_inference_smoke
```

该推理命令不需要 `--train-root`、`--hw-config`、profile CSV、shape CSV 或 timeline CSV。硬件参数、白盒配置、op-thread scale 和 RF 参数已经固化在模型 bundle 中，输入侧只需要 DLRM 上层配置。

## 9. CPU 与 NPU 文档边界

本文件只完成 CPU 建模方案。后续如果合并为完整的“CPU/NPU 算子计算性能建模方案设计文档”，NPU 部分需要补充 Cube/Vector 算力、Cube/Vector op latency、NPU 频率、片上存储层级、NPU memory bandwidth/latency、NPU 算子映射规则和 NPU 残差校准方案。CPU 部分与 NPU 部分可以按同一逻辑组织：先定义硬件特征，再定义算子规模特征，再说明白盒估计、残差校准、精度和耗时。

当前 CPU 部分的结论是：主线 E2E 建模 accuracy 超过 85%，计算图任务 MAPE 对应的 accuracy 也超过 85%，训练后静态配置批量推理耗时小于 1min；模型特征已经显式覆盖 CPU 指令位宽、指令延时、CPU 频率和流水线宽度；模型代码、运行命令、结果目录和复核字段已经在本文中说明。
