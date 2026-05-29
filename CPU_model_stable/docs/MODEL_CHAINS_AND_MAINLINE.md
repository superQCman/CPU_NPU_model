# 模型链路与当前主线

## 本文回答什么问题

本文解释当前项目里有哪些模型链路、哪一条是主线、主线每一步在做什么，以及当前主线的核心指标是多少。

## 读完你应该知道什么

你应该能够用一句话说明主线链路：先用 Analytical prior 做物理估计，再用 op-thread scale 校准尺度，再用 physics-guarded RF 学习残差，最后进入 E2E pipeline 计算整图延迟。RF 是 Random Forest 的缩写，中文是随机森林模型；E2E 是 end-to-end 的缩写，表示完整计算图从开始到结束的端到端耗时。

命名约定：`Device_1` 表示智算平台，`Device_2` 表示通算平台。路径、alias 或 YAML 名称中的 `kunpeng920` 对应智算平台，`s920x20` 对应通算平台，保留这些历史标识是为了兼容已有实验。

## 1. 什么是模型链路

模型链路指从一条输入样本到最终预测结果的完整路径。这里的输入样本可以是一条单算子记录，也可以是一个完整 DLRM-like 计算图中的多个算子。DLRM-like 表示负载结构接近 DLRM 推荐模型，包含 embedding gather、MLP、reduce、concat 等典型算子。

当前项目同时保留多条 route。route 是代码里的链路名字，不同 route 的区别主要在于是否使用白盒模型、是否使用 RF、是否加入专项诊断特征。

## 2. 当前主线 route

当前主线 route 是：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

这条链路的目标不是让 RF 自由学习所有规律，而是让硬件变化先通过 Analytical prior 表达，再让 RF 学残差，并用物理方向约束避免出现“硬件变强但预测变慢”这类反直觉趋势。

这里 RF 不是直接读取原始硬件字段，而是读取白盒模型生成的 service-time anchor 和静态语义特征。

## 3. 主线 6 步流程

### 第 1 步：输入样本

输入样本来自 split 后的数据集。每行通常表示一个算子，并带有：

- 算子类型，例如 `Gather`、`Gemm`、`ReduceSum`。
- 软件负载规模，例如 batch size、embedding 行数、MLP 形状。
- 线程配置，例如 `num_threads` 和 `inter_threads`。
- 实测耗时 `actual_us`。`actual_us` 的单位是微秒，表示真实运行得到的算子耗时。

### 第 2 步：Analytical prior

Analytical prior 是白盒估计层。它用硬件 YAML 和软件负载字段，估计每个算子的服务时间。YAML 是一种配置文件格式，当前可信硬件 YAML 都在 `calibrator/hw_config` 下。

Analytical prior 输出三类核心值：

- `whitebox_lower_us`：偏乐观估计，单位微秒。
- `whitebox_center_us`：中心估计，单位微秒，是后续 scale 和 RF 的主要输入。
- `whitebox_upper_us`：偏保守估计，单位微秒。

当前 active 服务模块只有四个：

- `stream_mem`：连续内存读写。
- `random_mem`：随机 gather 访问。
- `compute`：计算吞吐。
- `dispatch`：调度和启动开销。

已禁用历史诊断列：`descriptor/reorder/sync`。这些列可能仍出现在历史 schema 或 CSV 中，但当前不参与 `whitebox_center_us` 的贡献。代码位置是 `calibrator/whitebox_family_routes.py` 中的 `DISABLED_ANALYTICAL_MODULES = {"descriptor", "reorder", "sync"}`；`add_whitebox_family_prior_columns()` 会先生成若干诊断中间列，然后在 `_service_share_columns(work)` 和 `_compose_total_intervals(work)` 之前调用 `_zero_disabled_analytical_modules(work)` 清零这些模块。因此当前 active Analytical modules 只有 `stream_mem`、`random_mem`、`compute` 和 `dispatch`。

### 第 3 步：op-thread scale

op-thread scale 是 Analytical 之后、RF 之前的尺度校准层。它不是 RF，也不是新的硬件模型。

它解决的问题是：Analytical prior 的趋势方向通常可靠，但不同算子、不同线程数下的绝对尺度可能偏大或偏小。因此在 fit split 上，为每个 `(op_type, num_threads)` 拟合一个比例系数。

核心公式是：

$$s_{op,t}=\mathrm{weighted\_median}\left(\frac{actual\_us}{whitebox\_center\_us}\right)$$

其中 `op` 是算子类型，`t` 是 `num_threads`，`actual_us` 是真实耗时，`whitebox_center_us` 是未缩放的 Analytical 中心估计。

应用公式是：

$$whitebox\_scaled\_us=whitebox\_unscaled\_us \times s_{op,t}$$

这个 scale 会同时作用到白盒总耗时列和相关诊断预测列。输出中会保留：

- `whitebox_op_thread_scale`：当前行使用的缩放系数。
- `whitebox_op_thread_scale_source`：缩放来源，启用时为 op-thread scale 类型，关闭时为 `off`。
- `*_unscaled_us`：缩放前备份列。

系数产物位于：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505/op_thread_scale_config_joint_6d_tuple_source_100.json
outputs/cache_residency_worker_share_v3_full_s920x20_20260505/op_thread_scale_config_joint_6d_tuple_source_100.csv
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/op_thread_scale_config_joint_6d_tuple_source_100.json
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/op_thread_scale_config_joint_6d_tuple_source_100.csv
```

### 第 4 步：physics-guarded RF

RF 学的是 whitebox 估计之外的残差。当前 v3 链路加入三种保护：

- 反事实硬件增强：训练时生成硬件扰动样本，让 RF 看到符合白盒方向的伪标签。
- 单调 anchor：让最终预测对关键白盒 service-time anchor 保持合理单调关系。
- hardware guard：硬件 sweep 或 override 时，如果 RF 输出方向和白盒物理方向冲突，就投影回合法方向。

直观理解：RF 可以修正白盒模型的数值偏差，但不能推翻硬件参数变化的基本物理方向。

#### 4.1 反事实硬件增强到底在做什么

真实训练数据通常只来自少数真实机器。例如当前主线分别有 Device_1 和 Device_2 的真实采集数据，但不会覆盖所有可能的 cache、memory、frequency、pipeline width 组合。如果只用真实样本训练 RF，RF 很容易只学到当前数据分布中的相关性；当我们在硬件泛化图里修改某个硬件参数时，RF 可能给出和物理事实相反的变化方向。

反事实硬件增强的目标，是在训练阶段额外构造一些“同一个软件负载、假设硬件参数变化后会怎样”的 synthetic 样本。synthetic 表示人工构造，不是真实机器重新 profile 得到的数据。

对每条真实训练样本，先计算三件事：

- `actual_0`：真实硬件上的实测耗时，单位微秒。
- `prior_0`：真实硬件上的 whitebox prior，即 `whitebox_center_us`。
- `residual_0`：真实耗时与 whitebox prior 的差值。

残差定义是：

$$residual_0=actual_0-prior_0$$

然后对硬件 YAML 中的部分字段做扰动，例如 cache size、cache latency、memory bandwidth、DRAM latency、CPU frequency、SIMD width、FMA throughput、FMA latency、pipeline width 等。每次只改一个硬件字段，例如把 memory bandwidth 变成 `0.8x` 或 `1.2x`。

扰动后，不重新运行真实 workload，而是让 Analytical prior 在这个“假设硬件”上重新计算一次：

- `prior_h`：目标硬件参数下的 whitebox prior。

再构造 synthetic 训练标签：

$$synthetic\_actual_h=prior_h+residual_0$$

这个公式的含义是：硬件变化带来的方向和幅度由 whitebox prior 决定；RF 在真实硬件上学到的校准残差继续保留。这样 RF 会看到“硬件参数变强时，标签也应该随 whitebox prior 变快；硬件参数变弱时，标签也应该随 whitebox prior 变慢”。

举一个简化例子：

| 项 | 数值 | 含义 |
|---|---:|---|
| `actual_0` | 120 us | 真实硬件实测耗时 |
| `prior_0` | 100 us | 真实硬件 whitebox 估计 |
| `residual_0` | 20 us | RF 需要补的残差 |
| `prior_h` | 80 us | 假设带宽提高后 whitebox 估计 |
| `synthetic_actual_h` | 100 us | synthetic 标签 |

因此，这条 synthetic 样本会告诉 RF：如果同一个软件负载在更高带宽硬件上运行，预测不应该继续停在 120 us，也不应该反向变慢，而应该跟随 whitebox prior 下降到 100 us 附近。

synthetic 样本不会完全压过真实样本。当前真实样本权重是 `1.0`，反事实 synthetic 样本权重是 `0.25`。也就是说，它主要作为趋势约束和泛化提示，而不是替代真实 profile 数据。

#### 4.2 它和 hardware guard 不是一回事

反事实硬件增强发生在训练阶段。它的作用是让 RF 在训练时见过合理的硬件变化方向。

hardware guard 发生在推理阶段，尤其是硬件 sweep 或 `override_hw_config` 时。如果 RF 已经输出了和 whitebox prior 一致的方向，就直接保留 RF 结果；只有 RF 方向冲突或变化幅度过大时，才会投影回物理合法方向。

guard 的兜底公式是：

$$pred_h=prior_h+(pred_0-prior_0)$$

其中 `pred_0` 是基准硬件下 RF 的最终预测，`prior_0` 是基准硬件下 whitebox prior，`prior_h` 是目标硬件下 whitebox prior。这个公式可以理解为：保留 RF 在基准硬件上的残差，但硬件变化方向交给 whitebox prior。

因此两者分工不同：

| 机制 | 阶段 | 作用 |
|---|---|---|
| 反事实硬件增强 | 训练阶段 | 让 RF 学到硬件变化的合理方向 |
| hardware guard | 推理阶段 | 当 RF 仍然违反物理方向时做兜底修正 |

当前 correct-hw 精度评估通常没有 `override_hw_config`，因此不会主要依赖 guard；硬件趋势图和 frozen-swap 这类实验才会更明显触发 guard。

### 第 5 步：E2E pipeline

E2E pipeline 表示整图从开始到结束的总耗时预测。单算子预测不能简单相加，因为实际计算图里存在并行分支、barrier、branch pool 和 tail 等调度结构。

E2E pipeline 会把单算子预测放回图结构中排流水线，输出完整图的预测延迟。

训练评估阶段的 E2E pipeline 用数据集中 profile 产物计算真实 `actual_e2e_us`，因此可以计算 MAPE 和 P90 APE。训练后的静态配置推理阶段没有真实运行时间，它只用 DLRM 上层配置生成 `model_input_51` 静态算子和 `full_graph_76` 静态图，再输出主线口径的 `predicted_e2e_us`。

静态推理的主输出采用 `model_input_51 / observable_profiled_nodes` 口径。这与当前已报告的 E2E MAPE、P90 APE 口径一致。`full_graph_76` 会同时导出，用于审计完整静态图，但不是默认主 E2E latency 口径。

静态推理入口是：

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

### 第 6 步：输出指标

MAPE 是 mean absolute percentage error，平均绝对百分比误差，越低越好。APE 是 absolute percentage error，单个样本的绝对百分比误差。P90 APE 是第 90 百分位 APE，表示 90% 样本误差不超过该值。

当前主线结果使用 20260501 full split。full split 表示 `small_dataset_divisor=1`，不会像 quick validation 那样抽稀 batch size 和 `num_indices_per_lookup` 组合。

| 数据/硬件绑定 | E2E MAPE | E2E P90 APE | E2E median APE | 完整图组合数 |
|---|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 5.96% | 12.35% | 5.01% | 151 |
| Device_1 / 智算平台 full portable 数据 | 5.47% | 11.19% | 4.38% | 454 |

单算子补充结果：

| 数据/硬件绑定 | 单算子 MAPE | 单算子 P90 APE | 单算子 median APE | eval rows |
|---|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 20.61% | 36.96% | 12.76% | 7701 |
| Device_1 / 智算平台 full portable 数据 | 15.71% | 35.44% | 8.82% | 23154 |

full split 训练/测试数据规模如下。完整图组合数表示一个完整 DLRM-like 配置；单算子行数表示进入主线 `model_input_51` 口径的算子样本数，因此每个完整图组合对应 51 行单算子样本。

| 数据/硬件绑定 | 总完整图组合数 | train 完整图组合数 | test 完整图组合数 | train 单算子行数 | test 单算子行数 | 总单算子行数 |
|---|---:|---:|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 504 | 353 | 151 | 18003 | 7701 | 25704 |
| Device_1 / 智算平台 full portable 数据 | 1512 | 1058 | 454 | 53958 | 23154 | 77112 |

full split 的上层负载取值范围如下：

| 数据/硬件绑定 | batch size | number of indices per lookup | inter threads | op threads |
|---|---|---|---|---|
| Device_2 / 通算平台 full 数据 | `1024,1536,2048` | `1000,1200,1400,1600,1800,2000` | `1,3,5` | `1,2,3,4,5` |
| Device_1 / 智算平台 full portable 数据 | `1024,1152,1280,1408,1536,1664,1792,1920,2048` | `1000,1200,1400,1600,1800,2000` | `1,3,5` | `1,2,3,4,5` |

早期 20260429 small split 结果只有 28 个 test 完整图组合，适合快速验证链路和画趋势图，不作为当前 full 数据主线精度结论。

结果来源：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505/best_e2e_pipeline_by_policy.csv
outputs/cache_residency_worker_share_v3_full_s920x20_20260505/combined_summary.csv
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/best_e2e_pipeline_by_policy.csv
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/combined_summary.csv
```

主线 E2E 相对误差 CDF 图补充展示完整图误差分布。CDF 是累积分布函数，表示相对误差不超过某个阈值的完整图比例。当前 `APE <= 15%` 时，Device_1 / 智算平台 覆盖 `95.8%` 的 test 完整图，Device_2 / 通算平台 覆盖 `95.4%` 的 test 完整图。

```text
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/e2e_relative_error_cdf_mainline.png
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/e2e_relative_error_cdf_mainline.pdf
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/e2e_relative_error_cdf_summary.csv
```

Device_1 的 full E2E 指标使用 `data/data_e2e_full_kunpeng920_20260501_portable`，本轮重新训练并导出了新的 v3 bundle，不再复用旧 `_njobs128` 目录中的单算子预测。

### 跨平台少样本泛化实验

主线模型还可以做一个独立的跨平台 few-shot 泛化实验。这个实验不改变上面的 CPU full 主线指标，而是回答：如果一个平台已经完成 full 训练，迁移到另一个平台时，是否只需要少量目标平台数据做校准。

实验双向覆盖 `Device_1 / 智算平台 -> Device_2 / 通算平台` 和 `Device_2 / 通算平台 -> Device_1 / 智算平台`。source 平台的 RF 权重固定不变，target 平台只提供少量完整图组合，用来拟合轻量校准层：

- `fewshot_op_rescale`：按 `(op_type, num_threads)` 校准 source prediction 的残差尺度。
- `fewshot_op_rescale_plus_e2e_scale`：在单算子尺度校准后，再拟合一个 E2E 全局缩放。
- `fewshot_op_rescale_plus_e2e_surface`：在单算子尺度校准后，用少量完整图拟合低容量 E2E log-space residual surface，主判断使用该方法。
- `target_only_no_calibration_anchored`：只用 target 少量数据从头训练 v3，不使用 source RF，也不拟合 op-thread/E2E 缩放校准；该曲线仅作为“目标平台从零少样本建模”的收敛速度诊断，不作为正式精度判定依据。

该实验的重点是 10% target data。这里的 10% 按 target 平台 train 完整图组合数计算：Device_2 / 通算平台约 35 个完整图组合，Device_1 / 智算平台约 105 个完整图组合。正式结论看 5 个 seed 的均值，并同时报告 std 和 seed 间波动范围。若 10% few-shot 未达到 target full upper bound 的阈值要求，报告只给出未通过结论和达到阈值所需的最小 target 数据比例；正式结论不使用单次随机抽样、删样本或替换 test split。输出图包括 E2E MAPE/P90 随 target 数据比例变化、10% few-shot 与 100% target full 上界的对比、seed 稳定性、zero-shot 到校准后的收益，以及达到目标精度所需的数据比例。

当前 20260512 正式结果使用 `fewshot_op_rescale_plus_e2e_surface` 作为主判断方法，5 个 seed 均值如下：

| 迁移方向 | target full E2E MAPE | 10% few-shot E2E MAPE mean/std | target full E2E P90 APE | 10% few-shot E2E P90 APE mean/std | 判定 |
|---|---:|---:|---:|---:|---|
| Device_1 / 智算平台 -> Device_2 / 通算平台 | 5.96% | 6.64% / 0.40% | 12.35% | 13.85% / 0.41% | PASS |
| Device_2 / 通算平台 -> Device_1 / 智算平台 | 5.47% | 5.38% / 0.13% | 11.19% | 10.74% / 0.41% | PASS |

该结果说明，在这两个 full split 上，10% target 完整图校准已经达到接近 target full upper bound 的 E2E 精度；该结论只适用于当前数据绑定、当前主线 route 和当前校准方法。

## 4. CPU+NPU experimental 分支

CPU+NPU hybrid 脚本是独立 experimental 分支，不进入上面的 CPU 主线 MAPE/P90 表。它们读取已有 CPU full E2E 输出或已导出的 CPU v3 bundle，用外部 NPU 算子类别误差 profile 做误差替换或趋势验证。

这里的 `Device_1` 和 `Device_2` 表示两个 CPU 校准模型和对应数据绑定，不表示两台机器都装有 NPU。尤其 Device_2 是通算服务器；CPU+NPU 图只是用 Device_2 的 CPU 校准模型作为一个对照模型，不能解释为 Device_2 上真实存在 NPU。

这些脚本只输出到 `outputs/experimental_cpu_npu_*`，不修改 CPU 主线训练、预测、bundle 或 CDF 目录。它们也不代表真实 NPU 加速收益；正式异构性能预测还需要 placement、device transfer cost 和异构 E2E scheduler。

## 5. 其他保留 route 怎么理解

| 分组 | route | 用途 |
|---|---|---|
| 主线 | `whitebox_family_prior_plus_physics_guarded_rf_v3` | 当前用于硬件泛化图、主线结果和专家审查 |
| baseline | `whitebox_family_prior_plus_simple_rf_v2` | 对比没有 physics guard 时的 RF 行为 |
| Analytical 诊断 | `whitebox_family_prior_only_v2` | 只看白盒先验，不看 RF 残差 |
| 专项诊断 | `whitebox_family_prior_plus_ort_kernel_rf_v3` | 检查 ORT kernel 相关 residual 特征 |
| 专项诊断 | `whitebox_family_prior_plus_embedding_mem_rf_v4` | 检查 embedding/memory-heavy 行为 |
| 专项诊断 | `whitebox_family_prior_plus_gather_slice_rf_v5` | 检查 gather/slice 专项修正 |
| 专项诊断 | `whitebox_family_prior_plus_gather_slice_rf_v5_compact_probe` | gather/slice compact probe 版本 |
| benchmark | `whitebox_family_prior_plus_workload_rf_benchmark_v1[...]` | workload benchmark 变体，用于对比不同校正口径 |
| 纯 ML | `ml_semantic_thread_rf`、`ml_semantic_thread_family_rf` | 不以白盒硬件响应为核心，主要作为历史或对照 |

这些 route 都可能在代码里存在，但不要把它们混成一条链路。当前文档和主线结论只围绕 v3。
