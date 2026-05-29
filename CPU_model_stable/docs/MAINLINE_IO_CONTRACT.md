# 主线链路输入输出合同

## 本文回答什么问题

本文逐层说明当前主线 `whitebox_family_prior_plus_physics_guarded_rf_v3` 的输入、输出、字段含义和单位。它的目的不是介绍实验结论，而是让读者知道每个模块到底吃什么、算什么、吐什么。

## 读完你应该知道什么

你应该能回答这些问题：

- 一条训练样本需要哪些软件负载字段？
- 硬件 YAML 里哪些字段会进入主线？YAML 是一种配置文件格式。
- Analytical prior 输出哪些白盒服务时间？
- op-thread scale 在 Analytical 和 RF 之间如何工作？RF 是 Random Forest，也就是随机森林模型。
- 随机森林在主线中读取什么特征？
- E2E pipeline 如何把单算子预测变成整图预测？E2E 是 end-to-end，表示端到端整图耗时。

命名约定：`Device_1` 表示智算平台，`Device_2` 表示通算平台。路径、alias 或 YAML 名称中的 `kunpeng920` 对应智算平台，`s920x20` 对应通算平台，保留这些历史标识是为了兼容已有实验。

## 1. 总体数据流

主线数据流如下：

```text
原始单算子样本 + 硬件 YAML
  -> Analytical prior
  -> op-thread scale
  -> physics-guarded RF
  -> 单算子预测
  -> E2E pipeline
  -> 整图预测和误差指标
```

其中最重要的约束是：RF 不直接读取原始硬件型号或硬件 ID。硬件影响先由 Analytical prior 变成白盒 service-time anchor，再进入 RF。

训练后的静态配置推理使用另一条输入路径：

```text
DLRM 上层配置 + 已导出模型 bundle
  -> 静态算子规模和 full_graph_76
  -> model_input_51 主线节点
  -> Analytical prior + op-thread scale + physics-guarded RF
  -> 静态 E2E pipeline
  -> predicted_e2e_us
```

这条推理路径不读取 profile 生成的静态算子特征、shape CSV 或 timeline CSV。硬件 YAML、op-thread scale 和 RF 权重都从模型 bundle 中读取。

## 2. 原始样本输入

原始样本通常是一行算子记录。它描述“这个算子在某个软件负载和线程配置下，真实跑了多久”。

| 字段 | 单位 | 含义 | 下游用途 |
|---|---:|---|---|
| `row_uid` | 无 | 样本唯一 ID | 误差追踪和 debug |
| `case_id` | 无 | DLRM-like case ID | E2E 聚合时定位图 |
| `source_name` | 无 | 数据来源名称 | 数据审计 |
| `combo` | 无 | workload 组合标识 | E2E pipeline 分组 |
| `op_type` | 无 | 算子类型，例如 `Gather`、`Gemm` | 选择 Analytical 组合规则和 op-thread scale |
| `node_name` | 无 | 图中节点名 | E2E 调度和 debug |
| `actual_us` | 微秒 | 真实测量耗时 | 训练 label 和误差计算 |
| `batch_size` | samples | batch 大小 | 推导读写字节数、Gemm 规模、Gather 次数 |
| `num_indices_per_lookup` | indices | 每次 embedding lookup 的 index 数 | 推导 Gather 随机访问规模 |
| `arch_embedding_size` | 行数和维度 | embedding 表结构 | 推导 working set 和 row bytes |
| `arch_mlp_bot` | hidden units | bottom MLP 形状 | 推导 Gemm/MatMul FLOPs |
| `arch_mlp_top` | hidden units | top MLP 形状 | 推导 Gemm/MatMul FLOPs |
| `num_threads` | threads | 单算子内部线程数 | 影响有效 worker、op-thread scale 和吞吐 |
| `inter_threads` | threads | 图级并行线程数 | 影响 E2E branch pool 并行度 |

`actual_us` 是监督训练的目标。MAPE 是平均绝对百分比误差，P90 APE 是第 90 百分位绝对百分比误差；两者都基于 `actual_us` 与 `pred_us` 的差异计算。

## 3. 硬件 YAML 输入

硬件 YAML 是主线唯一可信的硬件参数来源。当前只使用：

```text
calibrator/hw_config/kunpeng920_single_numa.yaml  # 智算平台
calibrator/hw_config/s920x20_single_numa.yaml     # 通算平台
```

核心硬件字段如下：

| 硬件字段 | 单位 | 含义 | 主要影响 |
|---|---:|---|---|
| `core.total_cores` | cores | 当前 single NUMA 域可用物理核心数 | 限制有效 worker 上限 |
| `core.cpu_clock` | GHz | 有效 CPU 频率 | cycles 到时间的换算、计算吞吐 |
| `pipeline.fetch_width` 等 pipeline width | inst/cycle | 前端和后端流水线宽度 | 小算子 dispatch 和控制开销 |
| `instruction.simd_width_bits` | bit | SIMD 向量宽度 | 向量计算吞吐 |
| `instruction.fp_throughput_per_cycle.vector_sp_fma` | vector inst/cycle | FP32 FMA 吞吐 | Gemm/MatMul compute 时间 |
| `instruction.fp_latency_cycles.vector_sp_fma` | cycles | FP32 FMA latency | compute latency 趋势审计 |
| `memory.copy_bw_gb_s_1t` | GB/s | 单线程连续 copy 带宽 | `stream_mem` |
| `memory.copy_bw_gb_s_saturated` | GB/s | 多线程连续 copy 饱和带宽 | `stream_mem` |
| `memory.random_gather_bw_gb_s_1t` | GB/s | 单线程随机 gather 带宽 | `random_mem` |
| `memory.random_gather_bw_gb_s_saturated` | GB/s | 多线程随机 gather 饱和带宽 | `random_mem` |
| `memory.local_mem_delay_ns` | ns | 本地 NUMA DRAM 延迟 | DRAM resident 随机访问 |
| `cache.l1d.size`、`cache.l2.size`、`cache.l3_per_die.size` | KiB 或 MiB | cache 容量 | 判断 working set 驻留层级 |
| `cache.l1d.response_latency_cycles` 等 cache latency | cycles | cache 响应延迟 | cache resident 随机访问 |
| `runtime.parallel_launch_overhead_us_intra` | 微秒 | NUMA 内并行启动开销 | dispatch floor |
| `runtime.parallel_launch_overhead_us_inter` | 微秒 | 跨域并行启动开销 | dispatch floor |

## 4. Analytical prior 输出

Analytical prior 是白盒模型。它不通过训练直接记忆样本，而是用硬件参数和软件负载规模估计服务时间。

### 4.1 active 服务模块

当前参与总耗时合成的 active 模块只有四个：

| 模块 | 输出字段 | 单位 | 直观含义 |
|---|---|---:|---|
| `stream_mem` | `svc_stream_mem_us` | 微秒 | 连续读写 payload 的耗时 |
| `random_mem` | `svc_random_mem_us` | 微秒 | Gather 类随机访问耗时 |
| `compute` | `svc_compute_us` | 微秒 | Gemm、MatMul、Reduce、Elementwise 的计算耗时 |
| `dispatch` | `svc_dispatch_us` | 微秒 | 线程启动、算子调度和小算子 floor |

### 4.2 关键输出字段

| 字段 | 单位 | 含义 |
|---|---:|---|
| `whitebox_lower_us` | 微秒 | 白盒下界估计 |
| `whitebox_center_us` | 微秒 | 白盒中心估计，后续 scale 和 RF 的主要 anchor |
| `whitebox_upper_us` | 微秒 | 白盒上界估计 |
| `whitebox_band_width_us` | 微秒 | 上界减下界，表示白盒不确定性宽度 |
| `whitebox_exec_mode` | 无 | 算子执行模式，例如 gather、gemm、metadata |
| `whitebox_prior_family` | 无 | 白盒 family 分类 |
| `whitebox_rf_group` | 无 | RF group 分类 |

`whitebox_center_us` 的基础思想是：先计算各服务模块耗时，再按算子类型选择相加或瓶颈取最大。

已禁用历史诊断列：`descriptor/reorder/sync`。它们可能仍在部分 CSV schema 中出现，但当前值应为 0，不参与 `whitebox_center_us`。代码位置是 `calibrator/whitebox_family_routes.py` 中的 `DISABLED_ANALYTICAL_MODULES = {"descriptor", "reorder", "sync"}`；在 `add_whitebox_family_prior_columns()` 内部，算子规则可能先写入这些诊断中间列，但 `_zero_disabled_analytical_modules(work)` 会在 `_service_share_columns(work)` 和 `_compose_total_intervals(work)` 之前把它们的 lower/center/upper 和 `svc_*_us` 全部清零。因此后续 share、bottleneck 和 `whitebox_center_us` 合成都不会使用这三个禁用模块。

常见形式如下：

$$T_{whitebox}=T_{exec}+T_{dispatch}$$

其中 `T_exec` 是 payload 执行部分，`T_dispatch` 是调度启动部分。

## 5. 各类算子的 Analytical 组合规则

这里的公式是帮助理解的主线口径。每个变量含义如下：

| 变量 | 单位 | 含义 |
|---|---:|---|
| `T_stream` | 微秒 | `stream_mem` 模块耗时 |
| `T_random` | 微秒 | `random_mem` 模块耗时 |
| `T_compute` | 微秒 | `compute` 模块耗时 |
| `T_dispatch` | 微秒 | `dispatch` 模块耗时 |
| `T_op` | 微秒 | 单算子白盒预测耗时 |

### 5.0 Random memory 的 cache residency 口径

`random_mem` 会先估计当前 Gather 算子的随机访问 working set，再判断这些访问更可能命中 L1、L2、L3 还是 DRAM。这里的 `W_set` 是“当前 Gather 算子 / 当前 branch 的整体 working set”，不是整张 E2E 图所有 branch 的总 working set。

并行 worker 的两个数量定义如下：

| 变量 | 单位 | 含义 |
|---|---:|---|
| `N_worker` | workers | 当前单个算子内部实际参与执行的 worker 数，等于 `min(num_threads, N_core)` |
| `G_active` | groups | 当前阶段同时活跃的 branch/group 数 |
| `N_system` | workers | 当前系统里竞争共享资源的总活跃 worker 数 |
| `C_L1`、`C_L2` | bytes | 每个 core 私有的 L1D/L2 容量 |
| `C_L3` | bytes | 当前 NUMA/die 共享的 L3/LLC 容量 |

对应公式为：

$$N_{worker}=\min(num\_threads,N_{core})$$

$$N_{system}=clip(N_{worker}\cdot G_{active},1,N_{core})$$

L1/L2 是 per-core private cache，所以当前算子整体能使用的 L1/L2 容量随本算子 worker 数增加：

$$C_{L1,eff}=N_{worker}\cdot C_{L1}$$

$$C_{L2,eff}=N_{worker}\cdot C_{L2}$$

L3/LLC 是系统共享 cache，所以当前算子整体只拿到总活跃 worker 中属于自己的份额：

$$C_{L3,eff}=\frac{N_{worker}}{N_{system}}\cdot C_{L3}$$

continuous cache residency 模式下，先计算累计覆盖比例：

$$p_{L1}=\min(C_{L1,eff}/W_{set},1)$$

$$p_{L2}=\min(C_{L2,eff}/W_{set},1)$$

$$p_{L3}=\min(C_{L3,eff}/W_{set},1)$$

然后转成 exclusive share：L1 share 为 `p_L1`，L2-only share 为 `max(p_L2-p_L1,0)`，L3-only share 为 `max(p_L3-p_L2,0)`，剩余部分为 DRAM share。最后用各级 latency 的加权平均得到随机访问 latency 侧的估计。

兼容说明：历史 CSV 中可能仍有 `hw_ratio_working_set_to_l1d_active_bytes`、`hw_ratio_working_set_to_l2_active_bytes`、`hw_ratio_working_set_to_l3_active_bytes` 等诊断列，但当前主线 Analytical prior 不再直接用这些旧 ratio 决定 residency，而是按上述公式重新计算有效 cache 容量和驻留比例。

### 5.1 Metadata / view 类

Metadata / view 类算子主要处理 shape、view 或 metadata，不搬运大 payload。非 materialized view 主要落到 dispatch floor。

$$T_{op}=T_{dispatch}$$

如果 view 需要 materialize payload，则加入连续内存读写：

$$T_{op}=T_{stream}+T_{dispatch}$$

### 5.2 Gather 类

Gather 同时可能有随机读取和连续写出。随机访问和连续搬运可以部分重叠，所以主瓶颈取两者最大，再加调度。

$$T_{op}=\max(T_{random},T_{stream})+T_{dispatch}$$

### 5.3 ReduceSum 类

ReduceSum 同时读写数据并做加法规约。内存读写和计算取较慢的一项。

$$T_{op}=\max(T_{stream},T_{compute})+T_{dispatch}$$

### 5.4 Gemm / MatMul 类

Gemm 和 MatMul 的主要成本是矩阵乘法计算，同时也有输入输出读写。主线按 compute 与 stream 的瓶颈处理。

$$T_{op}=\max(T_{stream},T_{compute})+T_{dispatch}$$

### 5.5 Payload Concat

Payload concat 主要是把多个 payload 拷贝到输出 buffer。

$$T_{op}=T_{stream}+T_{dispatch}$$

### 5.6 Shape Concat / Shape / Unsqueeze / Flatten view

这些算子大多只改 metadata 或 shape 表达，不搬运大 payload。

$$T_{op}=T_{dispatch}$$

### 5.7 Transpose

Transpose 如果是 view-like 情况，主要是 dispatch；如果需要 materialize，则近似为连续或准连续内存搬运。

$$T_{op}=T_{stream}+T_{dispatch}$$

### 5.8 Elementwise 和 fallback

Elementwise 和 fallback 算子一般同时有内存搬运和简单计算，取更慢的一项。

$$T_{op}=\max(T_{stream},T_{compute})+T_{dispatch}$$

## 6. op-thread scale 输入输出

op-thread scale 位于 Analytical prior 之后、RF 之前。它的输入来自 fit split，不来自 test split。

### 6.1 输入

| 字段 | 单位 | 含义 |
|---|---:|---|
| `op_type` | 无 | 算子类型 |
| `num_threads` | threads | 单算子内部线程数 |
| `actual_us` | 微秒 | 真实耗时 |
| `whitebox_center_us` | 微秒 | 未缩放白盒中心估计 |

### 6.2 拟合方式

对每个 `(op_type, num_threads)` group，计算真实耗时与白盒估计的比例，并取加权中位数：

$$s_{op,t}=\mathrm{weighted\_median}\left(\frac{actual\_us}{whitebox\_center\_us}\right)$$

其中 `s_{op,t}` 是算子类型 `op` 和线程数 `t` 对应的缩放系数。权重使用 `actual_us`，这样耗时更大的样本对尺度校准影响更大。

### 6.3 输出

| 字段或文件 | 单位 | 含义 |
|---|---:|---|
| `whitebox_op_thread_scale` | 倍数 | 当前行使用的缩放系数 |
| `whitebox_op_thread_scale_source` | 无 | scale 来源，关闭时为 `off` |
| `*_unscaled_us` | 微秒 | 缩放前备份值 |
| `op_thread_scale_config_*.json` | 无 | scale 配置，供复现和加载 |
| `op_thread_scale_config_*.csv` | 无 | 每个 `(op_type, num_threads)` 的可读系数表 |

应用方式：

$$whitebox\_scaled\_us=whitebox\_unscaled\_us \times s_{op,t}$$

## 7. physics-guarded RF 输入输出

### 7.1 输入

physics-guarded RF 的输入不是原始 DLRM 参数，也不是硬件 YAML 原字段。它读取的是前面模块已经整理好的“单算子语义表”。每一行仍然对应一个算子节点，RF 用这些列判断：这个算子属于什么执行形态、在图中处于什么位置、白盒模型认为瓶颈在哪里，以及是否需要对白盒估计做校准。

RF 输入分四类：

| 类型 | 示例字段 | 含义 |
|---|---|---|
| 白盒 anchor | `whitebox_center_us`、`whitebox_band_width_us`、`svc_stream_mem_center_us`、`svc_random_mem_center_us`、`svc_compute_center_us`、`svc_dispatch_center_us` | 白盒服务时间和不确定性 |
| 模块占比和瓶颈关系 | `share_stream_mem_center`、`share_random_mem_center`、`share_compute_center`、`bottleneck_margin_center`、`random_vs_stream_ratio_center`、`mem_vs_compute_ratio_center` | 白盒模块谁占主导、两个候选瓶颈差多少 |
| 静态语义 | `feat_*`、`*_flag`、`*_code`、`merge_sync_obligation_flag`、`submodel_role_static_code`、`exec_region_static_code` | 算子的 dtype、layout、shape、merge/sync 和执行区域语义 |
| 图位置 | `pred_count`、`succ_count`、`topo_pos_norm`、`distance_to_output`、`distance_to_nearest_merge`、`static_parallel_branch_count` | 算子在 DAG 中的位置、邻居类型和 branch/merge 结构 |

RF 不直接读取 raw hardware ID、vendor 或 model name。硬件差异通过白盒 anchor 表达。

### 7.2 静态语义字段是什么

静态语义字段是从算子类型、shape、局部图结构和 DLRM 静态图推导出来的数值列。它们不是 profile 出来的真实耗时，也不是硬件参数。

| 字段族 | 单位 | 含义 | 例子 |
|---|---:|---|---|
| `feat_*_class_code` | 类别编码 | 把离散语义转成数值类别 | `feat_primary_dtype_class_code` 表示主要 dtype 类别；`feat_reduction_layout_class_code` 表示 Reduce 的 layout 形态 |
| `*_flag` | 0/1 | 某个布尔语义是否成立 | `feat_tiny_shape_arith_flag` 表示是否是 tiny shape 算术；`feat_transpose_generic_flag` 表示是否是 generic transpose |
| `merge_sync_obligation_flag` | 0/1 | 当前节点是否承担 merge 或同步语义 | 用于解释小算子和 branch merge 附近的调度偏差 |
| `submodel_role_static_code` | 类别编码 | 节点属于 embedding、bottom MLP、top MLP、metadata 等哪类子结构 | 帮助 RF 区分相同 op type 在不同子图中的行为 |
| `exec_region_static_code` | 类别编码 | 节点处在 bottom、branch pool、tail 等执行区域 | 和 E2E branch-pool/barrier/tail 调度口径一致 |
| `branch_lifetime_class_code` | 类别编码 | branch 生命周期类别 | 描述节点是否在 branch 内、branch 末端或 merge 附近 |
| `merge_fanin_static` | 个数 | 静态图中 merge 节点的输入分支数 | fan-in 越大，merge 附近调度语义越强 |
| `phase_barrier_count_static` | 个数 | 静态图中阶段性 barrier 数量 | 表示图级同步结构 |
| `worker_shard_count_static` | 个数 | 当前算子可切成的静态 worker shard 数 | 与并行暴露度相关 |

这些字段的共同作用是补充白盒模型没有完全表达的“结构上下文”。例如两个 `Concat` 的读写字节数可能接近，但一个在 embedding branch merge 处，另一个只是 shape concat；RF 需要通过静态语义区分它们。

兼容说明：`materialize_required_flag`、`producer_materialization_state_code`、`consumer_materialization_state_code` 仍可能出现在静态特征 CSV、Analytical 中间列或历史 bundle schema 中，但当前新训练的 v3 RF 不再把它们作为候选特征或输入特征。保留这些列是为了兼容已有数据格式和 exact I/O bytes 推导，不代表 RF 会继续学习 materialize 上下文。

### 7.3 图位置字段是什么

图位置字段描述当前节点在静态 DAG 中的位置和邻域。DAG 是 directed acyclic graph，即有向无环图；这里可以简单理解为算子依赖图。

| 字段族 | 单位 | 含义 |
|---|---:|---|
| `pred_op_type_primary_code`、`succ_op_type_primary_code` | 类别编码 | 主要前驱/后继的算子类型 |
| `pred_exec_mode_primary_code`、`succ_exec_mode_primary_code` | 类别编码 | 主要前驱/后继的执行模式 |
| `pred_count`、`succ_count` | 个数 | 前驱和后继数量 |
| `cross_scope_pred_count`、`cross_scope_succ_count` | 个数 | 跨执行区域的前驱/后继数量 |
| `topo_pos_norm`、`reverse_topo_pos_norm` | 0 到 1 | 从输入到输出、从输出反向看的拓扑位置 |
| `distance_to_output` | 跳数 | 当前节点到图输出的距离 |
| `distance_to_nearest_merge`、`distance_to_nearest_concat`、`distance_to_nearest_reduce`、`distance_to_nearest_gather` | 跳数 | 到最近关键结构节点的距离 |
| `nearest_merge_fanin` | 个数 | 最近 merge 节点的 fan-in |
| `static_is_embedding_branch_flag` | 0/1 | 是否处在 embedding branch |
| `static_parallel_branch_count` | 个数 | 当前区域的并行 branch 数 |
| `static_distance_to_merge` | 跳数 | 在静态 branch 内到 merge 的距离 |
| `graph_1hop_*_count`、`graph_2hop_*_count` | 个数 | 一跳或两跳邻域内某类 RF group 的数量 |

这些字段主要帮助 RF 学到调度上下文，而不是替代 E2E pipeline。单算子预测阶段只用它们修正节点耗时；最终整图并行、barrier 和 tail 仍由 E2E pipeline 处理。

### 7.4 RF group 和特征选择

主线不是训练一个全局 RF，而是按算子家族训练多个 RF group。每个 group 有自己的候选特征列表、最多选择 30 个特征。训练时会过滤掉缺失严重或完全常数的列。

| RF group | 主要覆盖的算子/语义 | 典型静态语义输入 |
|---|---|---|
| `metadata_view_rf` | Shape、Reshape、Flatten、Unsqueeze 等 metadata/view 类 | shape metadata class、merge/sync、submodel role、exec region、图位置、branch/merge 信息 |
| `tiny_shape_arith_rf` | 小 shape 算术和 tiny elementwise | dtype、broadcast class、tiny shape flag、worker shard、图位置 |
| `indexed_block_copy_rf` | Gather 和 indexed copy | branch/merge 语义、前后继类型、距离 merge/reduce、embedding branch 标记 |
| `payload_concat_rf` | payload concat | concat shape flag、merge/sync、merge fan-in、距离 concat/merge |
| `reduction_layout_rf` | ReduceSum 和 reduction layout | reduction layout class、reduction axis、keepdims、partial shape、worker shard |
| `transpose_generic_rf` | Transpose 和 layout reorder | transpose mode、generic transpose flag、submodel role、exec region、前后继和距离 concat |
| `gemm_like_rf` | Gemm、MatMul 和 MLP 计算节点 | dtype、submodel role、exec region、邻域 metadata/reduction 信息 |

当前主线 `physics_guarded_rf_v3` 的 `allow_workload_features=False`。这表示 RF 默认不直接读取大多数原始 workload size 列，例如 `output_size`、`activation_size`、`feat_lookup_count`、`feat_gemm_mac_count`、`exact_read_bytes`、`exact_write_bytes`。这些规模信息先进入 Analytical prior，转成 `whitebox_center_us` 和各个 `svc_*_center_us` anchor，再被 RF 使用。这样做的目的，是让硬件变化主要通过白盒物理模型表达，避免 RF 直接记忆 workload size 与某台机器的经验关系。

### 7.5 单调约束输入

v3 会对一组白盒 service-time anchor 加正向单调约束。单调约束的意思是：如果某个白盒 service time 变大，RF 预测不允许仅因为这个 anchor 变大而反向变小。

| 单调 anchor | 单位 | 含义 |
|---|---:|---|
| `whitebox_center_us` | 微秒 | 缩放后的白盒中心耗时 |
| `whitebox_exec_center_us` | 微秒 | payload 执行部分中心耗时 |
| `whitebox_serial_overhead_center_us` | 微秒 | 串行/调度相关开销 |
| `svc_stream_mem_center_us` | 微秒 | 连续内存服务时间 |
| `svc_random_mem_center_us` | 微秒 | 随机访问服务时间 |
| `svc_compute_center_us` | 微秒 | 计算服务时间 |
| `svc_dispatch_center_us` | 微秒 | dispatch 服务时间 |

`svc_descriptor_center_us`、`svc_reorder_center_us`、`svc_sync_center_us` 仍可能存在于特征 schema、历史 CSV 或诊断输出中，但当前 Analytical active module 已禁用 `descriptor/reorder/sync`，这些列会在 `_compose_total_intervals()` 前被清零，不贡献主线白盒总耗时。

### 7.6 输出

| 字段 | 单位 | 含义 |
|---|---:|---|
| `pred_us` | 微秒 | 单算子最终预测耗时 |
| `route` | 无 | 当前 route 名 |
| `ape` | 比例 | 单行绝对百分比误差 |
| `family_mlp_enabled_count` | 个 | 启用的 RF group 数量 |
| `ml_raw_hw_params_used` | 布尔 | 是否直接使用 raw hardware 字段，主线应为 `False` |

RF 校准的直观形式可以理解为：

$$pred\_us=whitebox\_scaled\_us+residual_{RF}$$

其中 `residual_RF` 是 RF 学到的残差修正。实现上不同 group 的训练目标可能是 `direct_log` 或 `analytical_residual`，但最终输出统一为微秒级 `pred_us`。

## 8. hardware guard 输出

hardware guard 只在硬件 override 或硬件 sweep 场景下特别重要。它比较白盒方向和 RF 方向。如果 RF 方向违反物理预期，就使用白盒方向进行投影。

直观规则是：

$$pred_h=prior_h+(pred_0-prior_0)$$

其中 `prior_h` 是目标硬件下的白盒预测，`pred_0` 是基准硬件下的 RF 输出，`prior_0` 是基准硬件下的白盒预测。这个公式表示：保留 RF 在基准硬件上的校准残差，但硬件变化方向由白盒 prior 决定。

主要审计文件：

```text
outputs/hw_generalization_physics_guarded_rf_v3_20260429/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload/hardware_response_guard_audit.csv
```

## 9. E2E pipeline 输入输出

E2E pipeline 的输入不是单行，而是一组属于同一个图和 workload combo 的算子预测。

### 9.1 输入

| 字段 | 单位 | 含义 |
|---|---:|---|
| `case_id` | 无 | 图或 case 标识 |
| `combo` | 无 | workload 组合 |
| `node_name` | 无 | 图节点名 |
| `pred_us` | 微秒 | 单算子预测耗时 |
| `actual_us` | 微秒 | 单算子真实耗时 |
| `inter_threads` | threads | 图级并行线程数 |
| pipeline artifact index | 无 | 节点、shape、时间线和图结构索引 |

E2E pipeline 必须能找到每个 combo 的两个 artifact：

| artifact | 是否必需 | 作用 |
|---|---|---|
| `op_shapes_*.csv` | 必需 | 提供每个节点的 shape 和静态规模，用于建立图结构和节点覆盖 |
| `branch_parallel_op_timeline.csv` | 必需 | 提供 branch 并行时间线，用于排流水线 |
| `effective_onnx_path` | 非必需 | 用于审计或复查图来源，当前 E2E 指标不依赖它 |

远端数据必须使用 portable 包。portable 包会把 shape 和 timeline 文件复制到数据目录下的 `portable_artifacts/`，并把索引改成当前机器可解析的路径。如果只传 split CSV，不传 artifact，E2E 会退化成 partial combo，不能作为模型精度结果。

### 9.2 输出

| 文件 | 关键字段 | 含义 |
|---|---|---|
| `best_e2e_pipeline_by_policy.csv` | `e2e_mape`、`e2e_p90`、`e2e_median` | 每条 route 的整图误差 |
| `e2e_pipeline_summary.csv` | route、policy、E2E 指标、完整/部分 combo 数 | E2E 汇总 |
| `pipeline_prediction_error_rows.csv` | combo-level error 和 row-level span | 每个完整图组合和节点的 pipeline 明细 |
| `e2e_pipeline_partial_combo_metrics.csv` | partial reason | 无法完整评估的 combo 及原因 |
| `pipeline_artifact_index_audit.csv` | `shape_csv_exists`、`timeline_csv_exists` | shape/timeline artifact 完整性审计 |

当 `e2e_pipeline_summary.csv` 中 `combo_count_full = 0` 或 `combo_count_partial > 0` 时，应先看 `e2e_pipeline_partial_combo_metrics.csv`。如果原因是 `shape_csv_missing` 或 `timeline_csv_missing`，问题在数据打包或传输，不是 RF 或 Analytical model 本身。

E2E MAPE 的计算方式是：

$$MAPE=\mathrm{mean}\left(\left|\frac{pred\_e2e\_us-actual\_e2e\_us}{actual\_e2e\_us}\right|\right)$$

其中 `pred_e2e_us` 是整图预测耗时，`actual_e2e_us` 是整图真实耗时。

## 10. 训练后静态配置推理 I/O

静态配置推理用于回答：“模型已经训练好后，只输入 DLRM 上层参数，能否得到 E2E latency？”答案是可以，但它输出的是预测值，不会计算 MAPE，因为没有真实 `actual_e2e_us`。

### 10.1 输入

| 输入 | 单位 | 含义 |
|---|---:|---|
| `--model-bundle` | 文件路径 | `calibrator/run.py --export-model-bundles` 导出的 v3 pickle bundle |
| `--batch-size` | samples | DLRM batch 大小 |
| `--num-indices-per-lookup` | indices | 每个 embedding lookup 的 index 数 |
| `--arch-embedding-size` | rows | 8 个 embedding table 行数；单值会自动扩展为 8 个 table |
| `--arch-mlp-bot` | hidden units | bottom MLP 三层结构，例如 `400-300-200` |
| `--arch-mlp-top` | hidden units | top MLP 三层结构，默认 `1600-800-1` |
| `--inter-threads` | threads | 图级并行线程数 |
| `--num-threads` | threads | 单算子内部线程数 |

### 10.2 自动生成的中间结果

| 文件 | 含义 |
|---|---|
| `static_op_shapes_full_graph_76.csv` | 从 DLRM 上层配置推导出的完整 shape 表 |
| `static_full_graph_76.csv` | 完整图 76 个节点的静态算子特征 |
| `static_model_input_51.csv` | 进入主线 v3 模型的 51 个节点 |
| `static_operator_predictions.csv` | 51 个主线节点的单算子预测 |
| `static_e2e_schedule_spans.csv` | 静态 E2E 调度后的 task/node 时间段 |

### 10.3 输出

主结果在 `static_e2e_summary.json` 中：

| 字段 | 单位 | 含义 |
|---|---:|---|
| `predicted_e2e_us` | 微秒 | 主线口径 E2E 预测延迟 |
| `predicted_full_graph_76_us` | 微秒 | 完整合成图 76 节点预测延迟，仅作审计 |
| `e2e_graph_scope` | 无 | 固定为 `model_input_51_observable_profiled_nodes` |
| `model_sha256` | 无 | 使用的模型 bundle hash |
| `op_thread_scale_enabled` | 布尔 | bundle 中是否包含 op-thread scale |
| `bottom_end_us`、`branch_pool_end_us`、`tail_barrier_us` | 微秒 | E2E 调度分解 |
