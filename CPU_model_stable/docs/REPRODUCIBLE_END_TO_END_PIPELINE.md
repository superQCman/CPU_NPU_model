# 从硬件采集到模型评估的完整复现流程

## 本文回答什么问题

本文说明如何从一台机器开始，采集硬件参数、生成训练数据、构建 split、训练主线模型、运行预测和评估，最终复现当前主线结果。

## 读完你应该知道什么

你应该知道每一步要运行什么命令、输入目录是什么、输出目录是什么，以及失败时应该优先检查什么。

## 1. 总体流程

YAML 是一种配置文件格式。当前建模只信任 `calibrator/hw_config` 下的硬件 YAML。E2E 是 end-to-end，表示完整计算图的端到端耗时。RF 是 Random Forest，即随机森林模型。MAPE 是平均绝对百分比误差，P90 APE 是第 90 百分位绝对百分比误差。

完整流程按下面顺序执行：

```text
硬件参数采集
  -> 硬件 YAML 固化
  -> 训练数据采集
  -> split 构建
  -> 主线模型训练和预测
  -> 静态配置推理
  -> 推理耗时 benchmark
  -> 审计、CDF、趋势图和成本优化
  -> CPU+NPU experimental
```

每一步都应记录输入目录和输出目录，避免后续把旧数据或旧硬件配置混入结果。

所有命令默认在仓库根目录执行，并使用当前 shell 的 `python`。如果新机器上使用 conda，先激活 `ort` 环境；如果没有 conda 环境，就使用默认 Python。

```bash
conda activate ort  # 如果使用 conda 环境
python --version
python -m pip install -r env/requirements.txt
```

`env/requirements.txt` 是可迁移依赖清单，不包含本机 wheel 路径，也不包含真实 NPU 运行时的厂商专用包。CPU+NPU experimental 分支只做后处理误差替换，不要求安装真实 NPU runtime。

命名约定：`Device_1` 表示智算平台，`Device_2` 表示通算平台。路径、alias 或 YAML 名称中的 `kunpeng920` 对应智算平台，`s920x20` 对应通算平台，保留这些历史标识是为了兼容已有实验。

## 2. 硬件参数自动校准

### 目的

硬件参数自动校准要回答：这台 CPU 的核心数、频率、cache、内存带宽、随机 gather 带宽、延迟和线程启动开销是多少，并把这些结果自动固化为主线模型可读取的可信 YAML。

主线不再要求人工从多轮 microbench 结果中挑选数值，也不再要求人工回填 YAML。推荐入口是：

```bash
python hardware_calibration/scripts/auto_calibrate_hw_config.py \
  --profile-name <profile> \
  --probe-profile auto \
  --numa-node 0 \
  --repeats 5 \
  --output-dir hardware_calibration/raw/<profile>_auto_<date> \
  --promote-if-pass
```

其中 `<profile>` 是输出硬件配置的名字，可以是已有正式名字，也可以是新机器自定义名字，例如：

```text
s920x20_single_numa
kunpeng920_single_numa
test_40
```

`--probe-profile auto` 是当前推荐模式。它会先从当前机器自动发现 NUMA CPU list、cache size、cacheline、L3/DRAM 规模、ISA flags 和可用内存，再自动选择 pointer-chase probe 点、stream 元素数和 NEON/SVE throughput microbenchmark。也就是说，主线模型实际读取的硬件字段不再依赖手写 profile 先验。

如果需要复现实验历史结果，可以把 probe 设计固定为已有设备：

```bash
python hardware_calibration/scripts/auto_calibrate_hw_config.py \
  --profile-name s920x20_single_numa \
  --probe-profile s920x20_single_numa \
  --numa-node 0 \
  --repeats 5 \
  --output-dir hardware_calibration/raw/s920x20_single_numa_compat_<date> \
  --promote-if-pass
```

该入口会自动完成：

```text
硬件拓扑检测
  -> 自动生成 auto_probe_plan.json/md
  -> microbench 多轮采集
  -> 稳定性评估
  -> 自动选择 cache/DRAM pointer-chase probe 点
  -> 生成 candidate_hw_config.yaml
  -> 生成 field_source_report.csv
  -> 质量门禁
  -> 通过后备份并覆盖 calibrator/hw_config/<profile>.yaml
```

microbench 是微基准测试，意思是用很小的专用程序测量某一类硬件能力，例如连续 copy 带宽、随机 gather 带宽、随机访问延迟或线程启动开销。

### 远端机器

远端机器也应在目标机器本地运行同一个自动入口。运行完成后，把整个输出目录和生成的 `calibrator/hw_config/<profile>.yaml` 传回当前 workspace。不要只传单个 summary CSV，因为后续审计需要 raw evidence、candidate YAML 和 gate report。

### 输出和门禁

典型输出包括：

```text
candidate_hw_config.yaml
auto_probe_plan.json
auto_probe_plan.md
field_source_report.csv
calibration_gate_report.json
calibration_gate_report.md
selected_probe_report.csv
yaml_diff_report.md
promotion_record.json
convergence_summary.json
convergence_stats.json
run_*/
```

自动覆盖 `calibrator/hw_config/<profile>.yaml` 的前提是所有门禁通过：

- NUMA core count、cache size、SIMD width、NUMA CPU list 解析成功。
- Analytical model 必需字段全部非空。
- `field_source_report.csv` 中主线必需字段不能来自 `manual_template`。
- 多轮稳定性达标。
- cache probe 选择符合层级约束。
- 新 YAML 可被 Analytical model 的 `_load_hw_params()` 成功读取。

如果任一门禁失败，脚本只保留 candidate 和报告，不覆盖可信 YAML。

失败时优先检查：

- microbench 是否编译成功。
- 当前用户是否能访问性能计数器或 NUMA 信息。
- 机器是否有其他重负载进程干扰。
- `calibration_gate_report.md` 中具体失败的字段。

### 随机访存延迟子模型真机验证

硬件自动校准完成后，可以用同一批 pointer-chase raw 数据验证 `random_mem` 中的随机访问平均延迟模型。该实验只验证 `L_touch/L_avg`，不验证完整 Gather `T_random` 或 E2E latency。

推荐先用同批 raw candidate YAML 复现。这样硬件 latency anchor 和 pointer-chase 真机测量来自同一次采集：

```bash
# 在仓库根目录执行

python calibrator/experimental/validate_random_access_latency_model.py \
  --hw-config hardware_calibration/raw/test_40_auto_20260502/candidate_hw_config.yaml \
  --ptrchase-root hardware_calibration/raw/test_40_auto_20260502 \
  --num-threads 1 \
  --active-groups 1 \
  --modes current_interp,ppt_weighted,hard_threshold \
  --stable-cv-threshold 15 \
  --target-mape-threshold 20 \
  --out-dir outputs/random_access_latency_validation_20260517
```

如果需要验证当前正式 YAML，而不是 raw candidate YAML，可以运行：

```bash
python calibrator/experimental/validate_random_access_latency_model.py \
  --hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --ptrchase-root hardware_calibration/raw/test_40_auto_20260502 \
  --num-threads 1 \
  --active-groups 1 \
  --modes current_interp,ppt_weighted,hard_threshold \
  --stable-cv-threshold 15 \
  --target-mape-threshold 20 \
  --out-dir outputs/random_access_latency_validation_20260517_formal_yaml
```

输出检查：

| 文件 | 说明 |
|---|---|
| `random_access_latency_validation_summary.csv` | 每种口径的 stable/all MAPE、P90、Max APE |
| `RANDOM_ACCESS_LATENCY_VALIDATION_REPORT.md` | continuous residency 与 exclusive-share weighted-average 口径差异、结果表和 warning |
| `random_access_latency_model_vs_measured_continuous_residency.png/pdf` | 当前 continuous cache residency 口径的实测-预测对比 |
| `random_access_latency_model_vs_measured_weighted_average.png/pdf` | exclusive-share weighted-average 口径的实测-预测对比 |
| `random_access_latency_ape_by_size.png/pdf` | 各口径 APE 随 working set size 变化 |

通过检查：

- `random_access_latency_validation_summary.csv` 中通过口径的 stable MAPE 应小于或等于 `20%`，P90 APE 建议小于或等于 `30%`。
- `RANDOM_ACCESS_LATENCY_VALIDATION_REPORT.md` 应明确说明 `continuous_residency` 和 `weighted_average` 的区别。
- `hard_threshold` 若出现极端误差，必须保留在 CSV/report 中；主图可以不展示失败模式，避免异常纵轴影响结论图。

## 3. 硬件 YAML 固化

### 目的

把自动校准通过的硬件参数固化为主线模型可读取的 YAML。

当前建模只信任：

```text
calibrator/hw_config
```

自动入口会先写：

```text
candidate_hw_config.yaml
```

质量门禁通过后才会备份并覆盖：

```text
calibrator/hw_config/<profile>.yaml
```

### 生成基础 YAML 的旧入口

```bash
python calibrator/generate_machine_hw_config.py \
  --output calibrator/hw_config/$(hostname)_single_numa.yaml
```

这个命令只保留为 fallback/debug scaffold。它只根据当前机器拓扑生成基础 YAML，不会自动完成 microbench 多轮采集、稳定性门禁、cache probe 选择和性能字段固化。主线复现不应把它作为可信硬件 YAML 生成步骤。

### 当前可信 YAML

```text
calibrator/hw_config/kunpeng920_single_numa.yaml
calibrator/hw_config/s920x20_single_numa.yaml
```

不要把数据包内部 metadata 当成硬件真值。当前绑定关系只看：

```text
calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json
```

## 4. 训练数据采集

### 目的

训练数据采集要得到单算子和 E2E 所需的 workload、图结构、时间线和真实耗时。

### 一键生成当前机器数据

```bash
bash bin/run_full_data_generation.sh
```

这个脚本串联：

1. 环境检查。
2. ONNX 生成。
3. case jobs 运行。
4. raw source 数据准备。
5. joint split 构建。
6. smoke validation。

### 分步运行

如果需要定位问题，可以分步执行：

```bash
bash bin/00_check_env.sh
bash bin/01_generate_onnx.sh
bash bin/02_run_case_jobs.sh
bash bin/03_prepare_source_raw.sh
bash bin/04_build_joint_small_split.sh
bash bin/90_smoke_validate.sh
```

注意：`04_build_joint_small_split.sh` 名字里带 `small`，适合快速验证。如果目标是复现当前 full split 指标，需要确认实际构建参数为 `small_dataset_divisor=1`，或使用已经生成好的 full 数据目录。

### 输出

本机 Device_2 当前 full 数据目录：

```text
data/data_e2e_full_s920x20_20260501
```

远端 Device_1 当前 full portable 数据目录：

```text
data/data_e2e_full_kunpeng920_20260501_portable
```

full 数据表示 `small_dataset_divisor=1`，即不抽稀 batch size 和 `num_indices_per_lookup` 组合。small split 只适合快速验证，不作为当前主线精度结论。

失败时优先检查：

- ORT 运行环境是否正确。
- ONNX 是否生成。
- profiling CSV 是否存在。
- `split_manifest.json` 是否和当前数据目录匹配。

## 5. split 构建

### 目的

split 构建把 raw 数据划分成 train/test，并固定 unseen tuple 评估范围。

当前主线 policy 是：

```text
joint_6d_tuple
```

### 运行

```bash
python calibrator/build_splits.py \
  --policy joint_6d_tuple \
  --dataset-scope source
```

如果使用一键脚本，`bin/run_full_data_generation.sh` 会自动触发 split 构建。

如果需要 full split，确认构建脚本或命令使用 `small_dataset_divisor=1`。如果只是快速 smoke test，可以使用抽稀 small split，但结果不能和当前 full 指标直接比较。

### 输出

典型输出：

```text
split_manifest.json
source_kunpeng920_dataset3_unseen_joint_6d_tuple/
```

当前 full split 规模：

| 数据/硬件绑定 | 总完整图组合数 | train 完整图组合数 | test 完整图组合数 | train 单算子行数 | test 单算子行数 | 总单算子行数 |
|---|---:|---:|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 504 | 353 | 151 | 18003 | 7701 | 25704 |
| Device_1 / 智算平台 full portable 数据 | 1512 | 1058 | 454 | 53958 | 23154 | 77112 |

每个完整图组合对应主线 `model_input_51` 的 51 行单算子样本，因此 `train/test 单算子行数 = train/test 完整图组合数 * 51`。

full split 的上层负载取值范围：

| 数据/硬件绑定 | batch size | number of indices per lookup | inter threads | op threads |
|---|---|---|---|---|
| Device_2 / 通算平台 full 数据 | `1024,1536,2048` | `1000,1200,1400,1600,1800,2000` | `1,3,5` | `1,2,3,4,5` |
| Device_1 / 智算平台 full portable 数据 | `1024,1152,1280,1408,1536,1664,1792,1920,2048` | `1000,1200,1400,1600,1800,2000` | `1,3,5` | `1,2,3,4,5` |

失败时优先检查：

- raw 数据目录是否完整。
- split manifest 中的数据来源是否过时。
- 是否误用了旧数据目录。

## 6. portable 数据打包和校验

### 目的

E2E 评估不仅需要 split CSV，还需要每个 workload combo 对应的静态 shape 和 timeline artifact。shape 是算子规模表，timeline 是分支并行时间线。远端机器上的原始数据通常包含绝对路径，如果直接拷贝到本机，这些路径会失效。

portable 包的作用是把 E2E 必需 artifact 一起打到数据目录中，并把索引改成当前机器可解析的相对路径。

### 什么时候必须打 portable 包

- 数据从远端机器传到本机评估。
- 希望数据目录自包含，不依赖采集机器上的外部路径。
- `e2e_pipeline_summary.csv` 出现 `combo_count_full=0`、`combo_count_partial>0`。
- `e2e_pipeline_partial_combo_metrics.csv` 中出现 `shape_csv_missing` 或 `timeline_csv_missing`。

### 打包命令

远端 Device_1 在源机器上打包：

```bash
python calibrator/package_e2e_dataset.py \
  --input-root data/data_e2e_full_kunpeng920_20260501 \
  --output-root data/data_e2e_full_kunpeng920_20260501_portable
```

如果希望 Device_2 本机数据也完全自包含，也可以打包：

```bash
python calibrator/package_e2e_dataset.py \
  --input-root data/data_e2e_full_s920x20_20260501 \
  --output-root data/data_e2e_full_s920x20_20260501_portable
```

### 传输规则

传输远端数据时，应传整个 portable root，而不是只传 CSV：

```text
data/data_e2e_full_kunpeng920_20260501_portable/
```

portable root 下面应包含：

```text
source_kunpeng920_dataset3_unseen_joint_6d_tuple/
source_kunpeng920_dataset3_unseen_joint_6d_tuple/portable_artifacts/
source_kunpeng920_dataset3_unseen_joint_6d_tuple/split_manifest.json
```

### 校验规则

训练或补跑 E2E 后检查：

| 文件 | 通过条件 |
|---|---|
| `pipeline_artifact_index_audit.csv` | `shape_csv_exists` 全为 true，`timeline_csv_exists` 全为 true |
| `e2e_pipeline_summary.csv` | `combo_count_partial = 0`，`coverage_ratio_mean = 1.0` |
| `e2e_pipeline_partial_combo_metrics.csv` | 应为空或没有 missing artifact 原因 |

`effective_onnx_path` 不是当前 E2E 指标的必需项。缺 ONNX 不会直接导致 E2E MAPE 无效；缺 shape 或 timeline 会导致 E2E 无法排流水线。

当前 Device_1 已确认可用于 E2E 的目录是：

```text
data/data_e2e_full_kunpeng920_20260501_portable
```

不要用只包含远端绝对路径的非 portable 目录直接做 E2E 结论；那会把数据缺失问题误判成模型问题。

## 7. 主线模型训练和预测

### 目的

训练主线 route，输出单算子预测、E2E 预测和所有审计 CSV。

### Device_2 / 通算平台

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

### Device_1 / 智算平台

```bash
CROSS_HW_RF_N_JOBS=128 python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --artifact-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --source-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --target-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --routes whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode fit \
  --export-model-bundles
```

`CROSS_HW_RF_N_JOBS=128` 用于加速 full split 随机森林训练。机器资源不足时可以调小，例如 16 或 32；这只影响训练速度，不应改变模型口径。

### op-thread scale 必须保留

op-thread scale 是 Analytical prior 之后、RF 之前的校准层。它按 `(op_type, num_threads)` 拟合比例：

$$s_{op,t}=\mathrm{weighted\_median}\left(\frac{actual\_us}{whitebox\_center\_us}\right)$$

之后按下面方式缩放白盒估计：

$$whitebox\_scaled\_us=whitebox\_unscaled\_us \times s_{op,t}$$

如果少了 `--op-thread-scale-mode fit`，当前结果不可直接和主线指标对比。

### 输出

| 文件 | 说明 |
|---|---|
| `combined_summary.csv` | 单算子 MAPE、P90 APE |
| `best_e2e_pipeline_by_policy.csv` | E2E MAPE、P90 APE |
| `prediction_error_rows.csv` | 单算子误差明细 |
| `pipeline_prediction_error_rows.csv` | pipeline 误差明细 |
| `family_mlp_enablement.csv` | RF group 训练状态 |
| `op_thread_scale_config_*.json/.csv` | scale 系数 |
| `model_bundles/*.pkl` | 训练后静态配置推理使用的模型 bundle |
| `model_bundle_manifest.csv` | 已导出 bundle 的索引 |
| `manifest.json` | 本次运行的配置和产物索引 |

### 单算子算子类建模精度图复现

训练完成后，可以从 `prediction_error_rows.csv` 直接复现“各类算子的建模精度”柱状图。这个步骤是只读后处理，不重新训练 RF，也不重新排 E2E pipeline。

```bash
# 在仓库根目录执行

python calibrator/plot_operator_class_mape.py \
  --device1-prediction-rows outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/prediction_error_rows.csv \
  --device2-prediction-rows outputs/cache_residency_worker_share_v3_full_s920x20_20260505/prediction_error_rows.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/operator_class_mape_cache_residency_worker_share_20260505 \
  --split-gather
```

输出检查：

| 文件 | 说明 |
|---|---|
| `operator_class_mape_gather_split_bar.png/pdf` | 推荐展示图，Gather 拆分为 `Gather-data` 和 `Gather-dispatch` |
| `operator_class_weighted_mape_gather_split_bar.png/pdf` | 按真实耗时加权的算子类误差图 |
| `operator_class_mape_gather_split_summary.csv` | 每个 device、每个算子类的 rows、MAPE、weighted MAPE、median 和 P90 |
| `OPERATOR_CLASS_MAPE_GATHER_SPLIT_REPORT.md` | 可读报告和 Gather 拆分定义 |

`Gather-data` 表示 embedding 数据搬运路径，`Gather-dispatch` 表示主要受调度或 metadata 影响的小 Gather。复现时重点检查：

- `operator_class_mape_gather_split_summary.csv` 中 Device_1 / 智算平台和 Device_2 / 通算平台都有结果。
- `Gather-data`、`ReduceSum`、`Gemm/MatMul` 是解释 E2E 耗时的关键类别。
- `Metadata/view` 和 `Gather-dispatch` 的 MAPE 可能较高，但平均真实耗时只有微秒级；解释 E2E 精度时应同时参考 `weighted_mape_pct` 和 E2E CDF。

### E2E 误差切片和算子规模泛化图复现

如果需要进一步检查模型误差是否集中在某个 batch、并行度或算子规模区间，可以运行下面的只读后处理实验。该实验读取当前 full test 的 E2E combo 结果、单算子预测行和 held-out `test.csv`，不重新训练、不重新预测。

```bash
# 在仓库根目录执行

python calibrator/experimental/plot_error_explainability_slices.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --device1-test-csv data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv \
  --device2-test-csv data/data_e2e_full_s920x20_20260501/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/error_explainability_slices_cache_residency_worker_share_20260516
```

输出检查：

| 文件 | 说明 |
|---|---|
| `e2e_error_by_batch_distribution.png/pdf` | batch size 维度的 E2E APE 分布 |
| `e2e_error_by_parallelism_distribution.png/pdf` | `inter_threads` 与 `num_threads` 维度的 E2E APE 分布 |
| `e2e_error_by_parallelism_bubble.png/pdf` | 真实观测并行组合维度的 E2E MAPE 气泡图 |
| `operator_error_vs_scale_facets.png/pdf` | 代表性算子误差随主导规模特征变化 |
| `operator_pred_actual_vs_scale_facets.png/pdf` | 代表性算子的真实/预测耗时随规模特征变化 |
| `operator_scale_error_summary.csv` | 算子规模分箱后的 MAPE、P90、signed error |
| `operator_scale_correlation.csv` | 主导规模特征与误差、真实耗时、预测耗时的 Spearman 相关 |
| `ERROR_EXPLAINABILITY_SLICES_REPORT.md` | 可读报告 |

通过检查：

- `error_explainability_manifest.json` 中 `full_combo_rows_by_device` 应为 `Device_1 = 454`、`Device_2 = 151`。
- `operator_scale_correlation.csv` 中默认代表性算子类的 `status` 应为 `ok`；若出现 `insufficient_rows`，报告只作提示，不强行得出泛化结论。
- `spearman_actual_vs_scale` 和 `spearman_pred_vs_scale` 同时为正且较高，表示真实耗时和预测耗时都随主导规模特征同向增长。
- `spearman_ape_vs_scale` 不应在关键算子类上持续接近 `1.0`；否则说明误差可能随规模变大而系统性恶化，需要回看 RF 特征或 Analytical prior。

### 软件负载泛化性验证复现

该实验把已有 held-out full test、可选新增 unseen software workload 和训练后 dense static-config 趋势放在同一个报告中。带真实标签的 held-out / unseen 数据用于精度结论；dense static-config sweep 只用于趋势说明，不作为 MAPE/P90 证据。

```bash
# 在仓库根目录执行

python calibrator/experimental/plot_software_workload_generalization.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --device1-test-csv data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv \
  --device2-test-csv data/data_e2e_full_s920x20_20260501/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/software_workload_generalization_20260518
```

只有一个 Device 时，只传该 Device 的 root/test CSV/bundle。若新增了完全未参与 train/test split 的软件负载实测结果，在同一命令中增加：

```bash
  --device1-unseen-root <device1_unseen_eval_output_root> \
  --device1-unseen-test-csv <device1_unseen_test_csv> \
  --device2-unseen-root <device2_unseen_eval_output_root> \
  --device2-unseen-test-csv <device2_unseen_test_csv>
```

输出检查：

| 文件 | 说明 |
|---|---|
| `software_generalization_e2e_cdf.png/pdf` | held-out 与 optional unseen 的 E2E relative error CDF |
| `software_generalization_mape_by_axis.png/pdf` | batch、NIP、`inter_threads`、`num_threads` 下的 MAPE/P90 |
| `software_generalization_pred_actual_trends.png/pdf` | 有真实标签数据上的预测/真实 E2E 趋势 |
| `operator_scale_generalization_facets.png/pdf` | 代表性算子误差随主导规模特征变化 |
| `static_config_dense_workload_trends.png/pdf` | 训练后 bundle 的密集软件配置趋势推理 |
| `software_generalization_e2e_summary.csv` | held-out / optional unseen 的 MAPE、P90、coverage |
| `static_config_dense_workload_trend_summary.csv` | 无标签 dense static sweep 的变化幅度和单调性 |
| `SOFTWARE_WORKLOAD_GENERALIZATION_REPORT.md` | 明确哪些结果是精度证据，哪些只是趋势 probe |

通过检查：

- `software_generalization_e2e_summary.csv` 中 held-out full test 的 `coverage_ratio_mean = 1.0`，`combo_count_partial = 0`。
- 如果提供 unseen workload，unseen 的 E2E MAPE 建议不超过 full held-out MAPE 的 `1.25x` 或绝对值不超过 `8%`，P90 APE 建议不超过 full held-out P90 的 `1.35x`。
- `static_config_dense_workload_trend_summary.csv` 只用于检查 batch/NIP/MLP/并行度变化时预测趋势是否方向合理，不能替代真实测量精度。

### DLRM 访存主导与计算参数敏感性归因复现

该实验用于解释“DLRM 以访存为主”与“计算/流水线参数在趋势图中仍有响应”这两个结论如何同时成立。脚本只读取当前 full 输出、bundle、数据 root 和硬件 YAML，做后处理与 `override_hw_config` 推理，不重新训练 RF，也不改变 CPU 主线指标。

```bash
# 在仓库根目录执行

python calibrator/experimental/analyze_dlrm_compute_memory_attribution.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device1-data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --device2-data-root data/data_e2e_full_s920x20_20260501 \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/dlrm_compute_memory_attribution_20260516
```

输出检查：

| 文件 | 说明 |
|---|---|
| `memory_vs_compute_actual_time_share.png/pdf` | 真实 profile 中 memory-related 与 compute-related 算子耗时占比 |
| `memory_vs_compute_service_share.png/pdf` | Analytical service 中 memory/compute/dispatch 占比 |
| `operator_actual_time_share_stacked_bar.png/pdf` | 主要 op type 的真实耗时占比 |
| `compute_parameter_sensitivity_top_ops.png/pdf` | 计算/流水线参数扰动时主要受影响的 op type |
| `compute_parameter_sensitivity_waterfall.png/pdf` | 计算/流水线参数扰动对 scheduled E2E 的边际影响 |
| `DLRM_COMPUTE_MEMORY_ATTRIBUTION_REPORT.md` | 可读报告 |

解读边界：

- memory-related true operator time share 和 Analytical memory service share 用于说明 DLRM 当前负载仍以访存相关路径为主。
- compute/pipeline 参数敏感性只说明局部关键路径可能受 CPU frequency、SIMD/FMA 或 pipeline width 影响，不能解释为“DLRM 是计算瓶颈”。
- 该实验不更新 E2E MAPE/P90 表，也不作为性能优化结论，只作为模型解释性证据。

### E2E ablation CDF 诊断复现

该实验拆解端到端精度来自 raw Analytical prior、RF/scale 和 E2E pipeline 调度中的哪一部分。它是诊断后处理，不覆盖主线 CDF。

如果 raw prior no-scale 输出不存在，先生成两个 raw root：

```bash
# Device_1 / 智算平台 raw prior
python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --artifact-root outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device1 \
  --source-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --target-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --routes whitebox_family_prior_only_v2 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode off \
  --force

# Device_2 / 通算平台 raw prior
python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root data/data_e2e_full_s920x20_20260501 \
  --artifact-root outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device2 \
  --source-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --target-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --routes whitebox_family_prior_only_v2 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode off \
  --force
```

再生成 ablation CDF：

```bash
python calibrator/experimental/plot_e2e_ablation_cdf.py \
  --device1-raw-root outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device1 \
  --device2-raw-root outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516/raw_prior_device2 \
  --device1-full-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-full-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --rf-route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --prior-route whitebox_family_prior_only_v2 \
  --threshold 0.10,0.15 \
  --module-improvement-threshold 0.1 \
  --out-dir outputs/e2e_ablation_cdf_cache_residency_worker_share_20260516
```

输出检查：

| 文件 | 说明 |
|---|---|
| `e2e_ablation_cdf_device1.png/pdf` | Device_1 / 智算平台 ablation CDF |
| `e2e_ablation_cdf_device2.png/pdf` | Device_2 / 通算平台 ablation CDF |
| `e2e_ablation_cdf_combined.png/pdf` | 双平台合并 CDF |
| `ablation_e2e_relative_error_cdf_summary.csv` | 每个 ablation stage 的 MAPE、P90、CDF@10%/15% |
| `ablation_module_effect_summary.csv` | 模块收益诊断表 |
| `E2E_ABLATION_CDF_REPORT.md` | 可读报告 |

四个 ablation stage 的含义：

| Stage | 含义 |
|---|---|
| `raw_direct_sum` | raw Analytical prior，不做 scale/RF，直接求和 |
| `raw_pipeline` | raw Analytical prior，加 E2E pipeline 调度 |
| `rf_scale_direct_sum` | 使用主线 RF/scale 单算子预测，但直接求和 |
| `rf_scale_pipeline` | 使用主线 RF/scale 单算子预测，并执行 E2E pipeline 调度；这是当前主线口径 |

如果 `E2E_ABLATION_CDF_REPORT.md` 的 `review_status` 是 `NEEDS_REVIEW`，需要查看 `ablation_module_effect_summary.csv` 的具体失败项。当前已知边界是：Device_2 / 通算平台的 `rf_scale_direct_sum` 未必优于 `raw_direct_sum`，但主线最终使用的是 `rf_scale_pipeline`，因此不能把 direct-sum 诊断失败误读成主线 E2E 失败。

### 跨平台少样本泛化实验复现

该实验不覆盖当前 full 主线结果。它固定一个平台 full 训练得到的 source v3 bundle，只用少量 target 平台数据拟合轻量校准层，并在 target held-out test 上评估 E2E 精度。目标是验证 target 数据量减少约一个数量级时，是否仍能接近 target full 训练精度。

运行命令：

```bash
# 在仓库根目录执行

python calibrator/experimental/cross_platform_fewshot_generalization.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device1-data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --device2-data-root data/data_e2e_full_s920x20_20260501 \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --sample-fractions 0,1,2,5,10,20,100 \
  --repeats 5 \
  --seed 20260511 \
  --primary-method fewshot_op_rescale_plus_e2e_surface \
  --workers 16 \
  --target-only-baseline-mode no_calibration \
  --anchor-target-only-no-calibration-to-target-full \
  --target-only-anchor-power 0.5 \
  --out-dir outputs/cross_platform_fewshot_generalization_20260512
```

`--workers` 控制并行执行数量。当前脚本会尽量按迁移方向并行运行，推荐先用 `--workers 4` 做 smoke，再用 `--workers 16` 跑正式实验。并行只改变运行时间，不改变随机种子和输出排序。

正式复现命令默认生成 `target_only_no_calibration_anchored` 诊断曲线：该曲线只用 target 少量数据从头训练 v3，不拟合 op-thread scale，也不拟合 E2E scale/surface，用于对比迁移校准与“目标平台从零少样本建模”的收敛速度。正式判定仍只依赖固定 source RF 的 `fewshot_op_rescale_plus_e2e_surface`；若只想快速复现主评估，可额外加 `--skip-target-only-fewshot` 跳过 target-only 诊断曲线。

如果已经有主迁移校准结果，只需要复现当前 `outputs/cross_platform_fewshot_generalization_20260512` 中的弱基准曲线和简化 MAPE 图，可以用下面的增量命令。它会合并已有 summary，只重算 `target_only_no_calibration_anchored` 并重新出图：

```bash
python calibrator/experimental/cross_platform_fewshot_generalization.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device1-data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --device2-data-root data/data_e2e_full_s920x20_20260501 \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --sample-fractions 1,2,5,10,20,100 \
  --repeats 5 \
  --seed 20260511 \
  --primary-method fewshot_op_rescale_plus_e2e_surface \
  --workers 16 \
  --target-only-baseline-mode no_calibration \
  --only-target-only-baseline \
  --merge-existing-summary outputs/cross_platform_fewshot_generalization_20260512/fewshot_generalization_summary.csv \
  --anchor-target-only-no-calibration-to-target-full \
  --target-only-anchor-power 0.5 \
  --out-dir outputs/cross_platform_fewshot_generalization_20260512
```

快速 smoke 可以先跑：

```bash
python calibrator/experimental/cross_platform_fewshot_generalization.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device1-data-root data/data_e2e_full_kunpeng920_20260501_portable \
  --device2-data-root data/data_e2e_full_s920x20_20260501 \
  --device1-hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --device2-hw-config calibrator/hw_config/s920x20_single_numa.yaml \
  --sample-fractions 0,10 \
  --repeats 1 \
  --primary-method fewshot_op_rescale_plus_e2e_surface \
  --workers 4 \
  --target-only-baseline-mode no_calibration \
  --anchor-target-only-no-calibration-to-target-full \
  --target-only-anchor-power 0.5 \
  --out-dir outputs/cross_platform_fewshot_generalization_smoke
```

检查项：

- `fewshot_e2e_mape_vs_target_data.png/pdf` 和 `fewshot_e2e_p90_vs_target_data.png/pdf` 存在。
- `fewshot_e2e_mape_vs_target_data_simplified_from_1pct.png/pdf` 存在；该图从 1% 开始，只保留 `Base-model calibration` 与 `Train from scratch` 两条曲线。
- `fewshot_10pct_vs_full_bar.png/pdf` 能直接展示 10% few-shot 与 100% target full 的差距。
- `fewshot_seed_stability_boxplot.png/pdf` 能展示 10% target data 下 5 个 seed 的稳定性。
- `fewshot_zero_to_calibrated_gain.png/pdf` 能展示 zero-shot 到 few-shot 校准后的误差下降幅度。
- `fewshot_generalization_summary.csv` 中两个迁移方向都有结果。
- Device_2 / 通算平台 target test `combo_count_full = 151`。
- Device_1 / 智算平台 target test `combo_count_full = 454`。
- `10%` target data 的 `data_reduction_factor >= 10`。
- 通过标准主要看 `fewshot_op_rescale_plus_e2e_surface` 的 5 seed mean：E2E MAPE mean 不超过 `1.25x target_full` 或绝对值不超过 8%，E2E P90 APE mean 不超过 `1.35x target_full`。
- 如果 `CROSS_PLATFORM_FEWSHOT_GENERALIZATION_REPORT.md` 中某个方向标记为 `FAIL`，文档应只报告未通过结论、失败指标和达到阈值所需的最小 target fraction。

当前 20260512 正式输出 `outputs/cross_platform_fewshot_generalization_20260512` 的 10% few-shot 主方法结果为：

| 迁移方向 | 10% combos | E2E MAPE mean/std | E2E P90 APE mean/std | 判定 |
|---|---:|---:|---:|---|
| Device_1 / 智算平台 -> Device_2 / 通算平台 | 36 | 6.64% / 0.40% | 13.85% / 0.41% | PASS |
| Device_2 / 通算平台 -> Device_1 / 智算平台 | 106 | 5.38% / 0.13% | 10.74% / 0.41% | PASS |

## 8. 训练后只用 DLRM 上层配置推理

### 目的

训练结束后，只输入 DLRM 上层负载参数和模型 bundle，自动生成算子规模、静态图和主线 v3 E2E latency。这个步骤不读取 profile 生成的静态算子特征、shape CSV 或 timeline CSV。

### 运行

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

`arch_embedding_size` 可以写成单个值，例如 `25600`，程序会扩展成 8 个 embedding table；也可以写成 8 个 dash-separated 值，例如 `25600-25600-25600-25600-25600-25600-25600-25600`。

### 输出

| 文件 | 说明 |
|---|---|
| `static_config.json` | 输入配置、bundle 路径和模型 hash |
| `static_op_shapes_full_graph_76.csv` | 推导出的完整静态 shape 表 |
| `static_full_graph_76.csv` | 完整图 76 个节点的静态特征 |
| `static_model_input_51.csv` | 主线 51 个输入节点 |
| `static_operator_predictions.csv` | 单算子预测 |
| `static_e2e_schedule_spans.csv` | E2E 调度时间段 |
| `static_e2e_summary.json` | `predicted_e2e_us` 等主结果 |

### 注意

静态配置推理没有真实运行时间，因此不会输出 MAPE 或 P90 APE。它的用途是训练后部署式预测；MAPE 和 P90 APE 仍由第 7 节的训练评估流程计算。

### 训练后批量静态配置推理耗时复现

如果需要验证需求方提出的“训练完成后推理耗时小于 `1 min`”要求，运行专门的 benchmark。这里的 `sample_count` 表示一次提交给性能模型的 DLRM 静态配置样本数，不是 DLRM 内部 batch size。

```bash
# 在仓库根目录执行

python calibrator/static_feature_mapping/benchmark_static_inference_latency.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --out-dir outputs/static_inference_bulk_latency_benchmark_20260507 \
  --threshold-seconds 60 \
  --sample-counts 1,2,4,8,16,32,48,64 \
  --bulk-repeats 10 \
  --warmup 2 \
  --workload-set dlrm_s_m_l
```

正式统计建议保留 `--bulk-repeats 10`。如果只需要快速确认 8 个 `sample_count` 都能跑通并生成图，可以临时改成 `--bulk-repeats 1`。当前 `sample_count=64` 已经接近 `60s` 阈值，适合检查图面和阈值关系。

这个 benchmark 不重新训练模型，只加载已经导出的两个 v3 bundle。它会测试：

- `cold_bulk_cli`：每次启动新的 Python 进程、加载一个 Device bundle，并批量预测 `sample_count` 个静态配置。
- `warm_bulk_inprocess`：bundle 常驻内存后，批量预测 `sample_count` 个静态配置。

通过标准是所有分组的 `max_s <= 60`，且每次批量推理都满足：

- `static_model_input_rows = 51 * sample_count`
- `static_full_graph_rows = 76 * sample_count`
- 每个样本的 `predicted_e2e_us > 0`

重点查看：

```text
outputs/static_inference_bulk_latency_benchmark_20260507/static_inference_bulk_latency_summary.csv
outputs/static_inference_bulk_latency_benchmark_20260507/static_inference_bulk_latency_vs_samples.png
outputs/static_inference_bulk_latency_benchmark_20260507/static_inference_bulk_per_sample_latency.png
outputs/static_inference_bulk_latency_benchmark_20260507/static_inference_bulk_throughput.png
outputs/static_inference_bulk_latency_benchmark_20260507/STATIC_INFERENCE_BULK_LATENCY_REPORT.md
```

## 9. 硬件泛化审计

### 目的

一次性跑硬件绑定清单、frozen-swap 和参数扰动。

### 运行

```bash
python calibrator/hw_generalization_audit.py \
  --binding-config calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json \
  --artifact-root outputs/hw_generalization_audit_cache_residency_worker_share_20260505 \
  --routes whitebox_family_prior_only_v2,whitebox_family_prior_plus_simple_rf_v2,whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --perturbation-routes whitebox_family_prior_only_v2,whitebox_family_prior_plus_simple_rf_v2,whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode fit
```

### 输出

| 文件 | 说明 |
|---|---|
| `hardware_dataset_binding.csv` | 数据和硬件绑定 |
| `hardware_parameter_inventory.csv` | 两个设备的硬件参数 |
| `frozen_swap_audit.csv` | 固定 RF 权重的硬件替换 |
| `parameter_perturbation_audit.csv` | 参数扰动方向检查 |

## 10. 趋势图复现

### 目的

生成论文风格硬件趋势图，展示 E2E 延迟随硬件参数变化的方向和幅度。

### 运行方式 A：重新 fit 后出图

这个命令会读取 binding config 里的 `data_root` 和 `hw_config`，重新拟合 op-thread scale 和 v3 RF，然后再做硬件参数 sweep。适合完整复现实验，但耗时较长。

```bash
python calibrator/plot_synthetic_hw_trend_predictions.py \
  --binding-config calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json \
  --bundle-source fit \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --plot-source route_pred \
  --prediction-scope e2e_pipeline \
  --workload-variants small,base,large \
  --datasets s920x20_local,kunpeng920_remote \
  --cache-size-working-set-scale 1.45 \
  --op-thread-scale-mode fit \
  --output-root outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

### 运行方式 B：复用已训练 bundle 后出图

这个命令不会重新训练 RF，也不会从 `train.csv` 重新拟合模型。它从 binding config 的 `model_bundle` 字段读取 `calibrator/run.py --export-model-bundles` 导出的 `.pkl`，然后直接做 synthetic workload 的硬件 sweep 和 E2E pipeline 调度。

```bash
python calibrator/plot_synthetic_hw_trend_predictions.py \
  --binding-config calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json \
  --bundle-source binding \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --plot-source route_pred \
  --prediction-scope e2e_pipeline \
  --workload-variants small,base,large \
  --datasets s920x20_local,kunpeng920_remote \
  --cache-size-working-set-scale 1.45 \
  --output-root outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

bundle 复用模式下，`--op-thread-scale-mode` 不再生效；op-thread scale 已经保存在 bundle 内。CSV summary、prediction rows、scale config 不能替代 `.pkl`，因为它们不能恢复 RF 权重。换新机器时，复制 `calibrator/configs/hw_generalization_bindings_model_bundle_template.json`，修改对应 dataset alias 的 `hw_config` 和 `model_bundle` 即可。

### 输出

```text
outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

注意：趋势图 CSV 和 PNG/PDF 只写入上面的 `--output-root` 目录，不会写入 correct-hw full 训练目录。两个 full 训练目录只保存 `combined_summary.csv`、`e2e_pipeline_summary.csv`、`best_e2e_pipeline_by_policy.csv`、`model_bundles/*.pkl` 等训练和评估产物。

关键文件：

| 文件 | 说明 |
|---|---|
| `synthetic_e2e_trend_summary_route_pred.csv` | 每个硬件趋势是否通过 |
| `synthetic_e2e_trend_workload_manifest_route_pred.csv` | 每张图的软件负载规模 |
| `synthetic_e2e_trend_hardware_range_summary_route_pred.csv` | 硬件参数 sweep 范围 |
| `hardware_response_guard_audit.csv` | guard 触发详情 |
| `*.png`、`*.pdf` | 趋势图 |

## 11. 趋势后处理和 absolute log-log elasticity

### 目的

趋势图生成后，单独计算每个硬件参数的方向一致性、最大变化幅度和无量纲影响强度。推荐用于 Top 硬件建模参数分析的主指标是 absolute log-log elasticity：

$$E(p)=\left|\frac{\log T_{E2E}(p_2)-\log T_{E2E}(p_1)}{\log p_2-\log p_1}\right|$$

这个指标表示硬件参数变化 1% 时，E2E latency 相对变化的强度；它会除以硬件参数本身的变化比例，因此比 `max_change_pct` 更适合比较不同 sweep 范围的硬件参数。

### 运行

```bash
python calibrator/plot_synthetic_e2e_correlation.py \
  --points-csv outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload/synthetic_e2e_trend_sweep_points_route_pred.csv \
  --binding-config calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json \
  --target-column pred_us \
  --out-dir outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

### 输出

| 文件 | 说明 |
|---|---|
| `synthetic_e2e_hardware_abs_log_elasticity_heatmap.png/pdf` | Top 硬件参数影响强度主图 |
| `synthetic_e2e_hardware_local_directional_spearman_heatmap.png/pdf` | 方向一致性图 |
| `synthetic_e2e_hardware_max_change_pct_heatmap.png/pdf` | 当前 sweep 范围内最大变化幅度 |
| `synthetic_e2e_hardware_local_sensitivity_by_workload.csv` | 每条 workload 曲线的局部敏感度 |
| `synthetic_e2e_hardware_local_sensitivity_best_workload.csv` | 每个硬件参数选择影响最大 workload 后的汇总 |
| `synthetic_e2e_hardware_metric_comparison_summary.csv` | 各敏感度指标汇总 |

如果 `synthetic_e2e_trend_sweep_points_route_pred.csv` 不存在，说明还没有成功运行第 10 节趋势图命令，不能直接运行后处理。

## 12. CDF 图复现

### 目的

CDF 图展示完整图相对误差分布。它比单个 MAPE 更容易看出大多数完整图是否落在可接受误差范围内。

### 运行

```bash
python calibrator/plot_e2e_relative_error_cdf.py \
  --device1-combo-metrics outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/e2e_pipeline_combo_metrics.csv \
  --device2-combo-metrics outputs/cache_residency_worker_share_v3_full_s920x20_20260505/e2e_pipeline_combo_metrics.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --threshold 0.15 \
  --out-dir outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505
```

### 输出

```text
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/e2e_relative_error_cdf_mainline.png
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/e2e_relative_error_cdf_summary.csv
outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505/E2E_RELATIVE_ERROR_CDF_REPORT.md
```

通过标准：

- Device_1 和 Device_2 都只显示匿名 label。
- 图中标出 MAPE、`relative_error=10%` 和 `relative_error=15%` 对应的 CDF 值。
- `E2E_RELATIVE_ERROR_CDF_REPORT.md` 中包含单独复现命令。

## 13. 成本约束硬件参数优化

### 目的

使用已经训练好的两个 v3 bundle，在需求方 `参数约束v2.docx` 给出的 x1-x25 参考成本约束下搜索候选硬件参数配置。该步骤不重新训练 RF，只对候选硬件做 `override_hw_config` 推理。
下面命令复现当前 `outputs/hardware_cost_optimization_nsga2_only_20260501` 产物；更长搜索可把 `--nsga2-population/--nsga2-generations` 提高到 `96/50`。

### 运行

```bash
python calibrator/hardware_cost_optimizer.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
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

建议验收复现使用 `--pair-extrapolation-mode forbid`。该模式不允许候选硬件参数超出需求方给出的两个参考成本点区间，避免把成本函数外推结果作为正式结论。

### 输出

| 文件 | 说明 |
|---|---|
| `candidate_predictions.csv` | 已评估候选及其预测 latency |
| `candidate_parameter_costs.csv` | 每个候选、每个 x1-x25 参数的取值和相对成本 |
| `hardware_cost_candidate_range_audit.csv` | 每个参数的候选取值范围审计，确认是否越过 v2 约束 |
| `hardware_cost_constraints_v2_resolved.csv` | 解析后的参数约束配置 |
| `pareto_frontier.csv` | 非支配 Pareto 前沿 |
| `best_by_budget.csv` | 每个万元级预算约束下的最优候选 |
| `optimization_sanity_audit.csv` | 合理性审计 |
| `pareto_frontier_cost_latency.png/pdf` | 成本-性能 Pareto 图 |
| `HARDWARE_COST_OPTIMIZATION_REPORT.md` | 可读报告 |

`reference_v2` 成本使用两点幂律拟合每个参数的参考成本曲线，再计算相对 Device_1 / 智算平台 `8w` 基准的正向成本增量。组内用 L2 范数聚合增量，整机成本为 `8w + 各组增量`，因此单参数变化会严格经过需求方 Pair 对参考成本点，不再被旧的组权重摊薄。图中 `cost_wan` 是万元级参考成本。没有 bundle 时不能用 CSV summary 恢复 RF 权重。

## 14. CPU+NPU experimental 复现

### 目的

CPU+NPU experimental 是独立实验分支。它不修改 CPU 主线模型、不重新训练、不覆盖 CPU CDF 或趋势图结果。`Device_1` 和 `Device_2` 在这些图里指两个 CPU 校准模型，不表示两台机器都有 NPU；Device_2 是通算服务器，没有 NPU。

当前实验只验证误差替换口径和上层负载变化趋势，不代表真实 NPU 加速收益。正式异构预测还需要 CPU/NPU placement、device transfer cost 和异构 E2E scheduler。

### 14.1 CDF 误差替换实验

```bash
python calibrator/experimental/plot_cpu_npu_hybrid_e2e_error_cdf.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --offload-profile compute_island \
  --out-dir outputs/experimental_cpu_npu_hybrid_cdf_20260510
```

重点输出：

```text
hybrid_e2e_error_cdf.png/pdf
hybrid_e2e_error_cdf_cpu_npu_only.png/pdf
hybrid_e2e_error_cdf_summary.csv
HYBRID_CPU_NPU_E2E_CDF_REPORT.md
```

### 14.2 batch/NIP sensitivity 实验

```bash
python calibrator/experimental/plot_cpu_npu_hybrid_batch_nip_sensitivity.py \
  --device1-root outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505 \
  --device2-root outputs/cache_residency_worker_share_v3_full_s920x20_20260505 \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --offload-profile compute_island \
  --style-reference-dir outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload \
  --out-dir outputs/experimental_cpu_npu_hybrid_batch_nip_sensitivity_20260510
```

该实验使用已有 full E2E combo，按 batch size 和 `num_indices_per_lookup` 聚合，输出归一化趋势和最大变化百分比。

### 14.3 dense static-config model-inference sensitivity 实验

```bash
python calibrator/experimental/plot_cpu_npu_hybrid_batch_nip_model_inference.py \
  --device1-bundle outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --device2-bundle outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --offload-profile compute_island \
  --out-dir outputs/experimental_cpu_npu_hybrid_batch_nip_model_inference_20260510
```

该实验不局限于训练/测试数据集中的 combo，而是用训练好的模型直接对更密集的 batch size 和 NIP 采样点做推理。通过标准是 batch size 和 NIP 增大时，归一化 E2E latency 整体非下降。

## 15. 当前复现结果

当前主线结果应接近：

| 数据/硬件绑定 | E2E MAPE | E2E P90 APE | 单算子 MAPE | 单算子 P90 APE |
|---|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 5.96% | 12.35% | 20.61% | 36.96% |
| Device_1 / 智算平台 full portable 数据 | 5.47% | 11.19% | 15.71% | 35.44% |

如果结果偏差很大，优先检查：

- 是否使用 `calibrator/hw_config` 下的 YAML。
- 是否使用正确数据目录。
- 远端数据是否使用 portable root，而不是只复制 split CSV。
- `e2e_pipeline_summary.csv` 是否满足 `combo_count_partial = 0` 和 `coverage_ratio_mean = 1.0`。
- 是否启用 `--op-thread-scale-mode fit`。
- route 是否包含 `whitebox_family_prior_plus_physics_guarded_rf_v3`。
- split 是否仍是 `joint_6d_tuple`。

## 16. 常见失败排查

| 现象 | 优先检查 |
|---|---|
| `model_bundles/*.pkl` 不存在 | 训练命令是否加了 `--export-model-bundles` |
| 静态配置推理要求 `--train-root` 或 `--hw-config` | 是否使用了旧入口或没有传 `--model-bundle` |
| `combo_count_full=0` | portable artifact 是否完整，尤其是 shape/timeline |
| `shape_csv_missing` 或 `timeline_csv_missing` | 是否只复制了 split CSV，没有复制 portable root |
| 趋势后处理找不到 points CSV | 是否先运行了趋势图脚本，且 `--output-root` 是否一致 |
| CDF 数值与文档不一致 | 输入是否是 combo-level `e2e_pipeline_combo_metrics.csv`，并过滤了 `coverage_ratio=1.0` |
| 成本优化器报 bundle missing | CSV summary 不能恢复 RF 权重，必须提供 `.pkl` bundle |
| CPU+NPU 图被误读为真实加速 | 报告中应明确它是 independent synthetic/误差替换实验 |
