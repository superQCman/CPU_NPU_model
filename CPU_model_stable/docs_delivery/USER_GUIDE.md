# CPU 性能模型使用指南

本指南面向交付使用，覆盖环境准备、数据绑定、完整链路复现、训练评估、训练后静态配置推理、图表复现、few-shot 泛化和成本约束硬件搜索。

仓库已在 `models/s920x20/`（Device_2）和 `models/kunpeng920/`（Device_1）交付正式训练 bundle。文件为原始 `.pkl` 的无损 XZ 压缩版，`--model-bundle` 可直接读取 `.pkl.xz`，无需重新训练；清单和 SHA256 见 [模型说明](../models/README.md)。本指南中的 `outputs/.../model_bundles/*.pkl` 路径用于重新训练的输出。

从 `CPU_model_stable/` 直接运行已训练模型：

```bash
python calibrator/static_feature_mapping/predict_from_config.py \
  --model-bundle models/s920x20/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl.xz \
  --batch-size 1024 --num-indices-per-lookup 1000 \
  --arch-embedding-size 25600 --arch-mlp-bot 400-300-200 \
  --inter-threads 1 --num-threads 3 --out-dir outputs/pretrained_static_inference
```

Device_1 使用 `models/kunpeng920/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl.xz.parts.json`。该文件是分片清单，加载器自动读取同目录下的压缩分片，详见模型说明。

## 1. 交付范围

当前项目用于基于 CPU 硬件参数和 DLRM-like 软件负载信息预测：

- 单算子延迟。
- E2E latency，即完整计算图端到端延迟。
- 训练后静态配置推理结果。
- 硬件参数趋势图、误差解释图和成本约束硬件配置搜索结果。

当前主线 route 为：

```text
whitebox_family_prior_plus_physics_guarded_rf_v3
```

主线链路：

```text
硬件 YAML + workload 特征
  -> Analytical prior
  -> op-thread scale
  -> physics-guarded RF
  -> 单算子预测
  -> E2E pipeline
  -> 整图预测
```

RF 是 Random Forest，即随机森林。YAML 是硬件配置文件格式。MAPE 是平均绝对百分比误差，P90 APE 是第 90 百分位绝对百分比误差。

## 2. 当前正式结果

当前正式结果来自 20260505 full split 目录：

```text
outputs/cache_residency_worker_share_v3_full_s920x20_20260505
outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505
```

主线 E2E 指标：

| 数据/硬件绑定 | E2E MAPE | E2E P90 APE | E2E median APE | test 完整图组合数 |
|---|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 5.96% | 12.35% | 5.01% | 151 |
| Device_1 / 智算平台 full portable 数据 | 5.47% | 11.19% | 4.38% | 454 |

单算子指标：

| 数据/硬件绑定 | 单算子 MAPE | 单算子 P90 APE | 单算子 median APE | eval rows |
|---|---:|---:|---:|---:|
| Device_2 / 通算平台 full 数据 | 20.61% | 36.96% | 12.76% | 7701 |
| Device_1 / 智算平台 full portable 数据 | 15.71% | 35.44% | 8.82% | 23154 |

设备命名：

| 交付名称 | 平台 | 历史路径标识 |
|---|---|---|
| Device_1 | 智算平台远端机器 | `kunpeng920` |
| Device_2 | 通算平台本机 | `s920x20` |

历史路径名、dataset alias 和 YAML 名称中的 `kunpeng920/s920x20` 仅为兼容已有数据和脚本；展示、报告和交付说明统一使用 Device_1 / Device_2。

## 3. 环境配置

先把仓库放到独立验收目录，然后进入仓库根目录。任选一种方式：

```bash
export ACCEPTANCE_WORKSPACE="<acceptance_workspace_root>"
mkdir -p "$ACCEPTANCE_WORKSPACE"

# 方式 A：从 Git 仓库获取
git clone "<repo_url>" "$ACCEPTANCE_WORKSPACE/CPU_model_stable"

# 方式 B：从已有工作树复制，适合内网验收或离线包
rsync -a --exclude .git --exclude outputs --exclude data --exclude hardware_calibration/raw \
  "<source_repo_root>/" "$ACCEPTANCE_WORKSPACE/CPU_model_stable/"

cd "$ACCEPTANCE_WORKSPACE/CPU_model_stable"
```

后续命令默认在仓库根目录执行，并使用当前 shell 的 `python`：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n cpu-model-acceptance python=3.10 -y
conda activate cpu-model-acceptance
python --version
python -m pip install -r env/requirements.txt
python env/check_env.py
```

`env/requirements.txt` 面向 CPU 主线与后处理实验。PyTorch 不放在默认 requirements 中，因为不同 CPU 架构和加速器环境需要不同 wheel。运行 DLRM ONNX 或数据生成前必须安装本平台匹配的 PyTorch，并通过数据生成依赖检查：

```bash
# Linux aarch64 CPU / Ascend 主机常用；若厂商环境已内置 PyTorch，可只安装 tensorboard 后做检查
python -m pip install "torch==2.9.0" tensorboard

# x86_64 CPU-only 主机建议使用 CPU wheel index，避免拉取 CUDA runtime
# python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.9.0"
# python -m pip install tensorboard

python env/check_env.py --include-data-generation
```

真实 NPU runtime 依赖不包含在默认 requirements 中。CPU+NPU experimental 分支只做后处理误差替换，不要求安装真实 NPU runtime。

## 4. 平台和目录约定

在新机器上复现时，在仓库根目录执行下面变量初始化。`REPO_ROOT="$(pwd)"` 会绑定到当前验收目录；如果数据放在外部磁盘，只需要覆盖对应变量。

```bash
export REPO_ROOT="$(pwd)"
export DEVICE1_DATA_ROOT="$REPO_ROOT/data/device1_full_portable"
export DEVICE2_DATA_ROOT="$REPO_ROOT/data/device2_full"
export DEVICE1_RAW_DATA_ROOT="$REPO_ROOT/data/device1_full_raw"
export DEVICE2_DATA_WORK_ROOT="$REPO_ROOT/data/device2_work"
export DEVICE2_SOURCE_RAW_ROOT="$REPO_ROOT/data/source_kunpeng920_dataset3_scheduler_20260419_raw"
export DEVICE1_TEST_CSV="$DEVICE1_DATA_ROOT/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv"
export DEVICE2_TEST_CSV="$DEVICE2_DATA_ROOT/source_kunpeng920_dataset3_unseen_joint_6d_tuple/test.csv"
export DEVICE1_HW_CONFIG="$REPO_ROOT/calibrator/hw_config/device1_auto.yaml"
export DEVICE2_HW_CONFIG="$REPO_ROOT/calibrator/hw_config/device2_auto.yaml"
export DEVICE1_TRAIN_OUT="$REPO_ROOT/outputs/device1_train"
export DEVICE2_TRAIN_OUT="$REPO_ROOT/outputs/device2_train"
export DEVICE1_BUNDLE="$DEVICE1_TRAIN_OUT/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl"
export DEVICE2_BUNDLE="$DEVICE2_TRAIN_OUT/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl"
export PROFILE_NAME="device2_auto"
export HW_CALIB_RAW_OUT="$REPO_ROOT/hardware_calibration/raw/${PROFILE_NAME}_auto"
export PTRCHASE_ROOT="$HW_CALIB_RAW_OUT"
export PTRCHASE_HW_CONFIG="$HW_CALIB_RAW_OUT/candidate_hw_config.yaml"
export FEWSHOT_SEED="20260511"
export HW_COST_OPT_OUT="$REPO_ROOT/outputs/hardware_cost_optimization_nsga2"
```

当前正式交付绑定使用已归档 full 数据：

| 设备 | 数据目录 | 硬件 YAML |
|---|---|---|
| Device_1 / 智算平台 | `data/data_e2e_full_kunpeng920_20260501_portable` | `calibrator/hw_config/kunpeng920_single_numa.yaml` |
| Device_2 / 通算平台 | `data/data_e2e_full_s920x20_20260501` | `calibrator/hw_config/s920x20_single_numa.yaml` |

正式绑定配置：

```text
calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json
```

主线只信任 `calibrator/hw_config` 下的硬件 YAML。数据包内部 metadata 可能包含旧硬件名，不作为硬件真值。

远端数据必须使用 portable root。portable root 会把 E2E pipeline 所需的 shape 和 timeline artifact 一起带到当前机器。只复制 split CSV 会导致：

```text
combo_count_full=0
shape_csv_missing
timeline_csv_missing
```

## 5. 新机器完整链路快速复现

### 5.1 硬件参数提取

```bash
python hardware_calibration/scripts/auto_calibrate_hw_config.py \
  --profile-name "$PROFILE_NAME" \
  --probe-profile auto \
  --numa-node 0 \
  --repeats 8 \
  --output-dir "$HW_CALIB_RAW_OUT" \
  --promote-if-pass
```

通过后会写入 `calibrator/hw_config/<profile_name>.yaml`。随后把对应平台的 `DEVICE*_HW_CONFIG` 变量设置为该 YAML 路径，例如：

```bash
export DEVICE2_HW_CONFIG="$REPO_ROOT/calibrator/hw_config/$PROFILE_NAME.yaml"
```

失败时先查看：

```text
hardware_calibration/raw/<profile_name>_auto/calibration_gate_report.md
hardware_calibration/raw/<profile_name>_auto/field_source_report.csv
```

### 5.2 数据集生成与收集

在每个平台本机完成数据生成。复现 full 指标时必须设置 `E2E_SMALL_DIVISOR=1`；默认 small split 只用于 smoke。

硬件校准运行时间较长时，可以并行启动不依赖硬件 YAML 的 ONNX 生成阶段：

```bash
DATA_ROOT="$DEVICE2_DATA_WORK_ROOT" \
RUN_ONNX=1 RUN_PROFILE=0 RUN_RAW=0 RUN_SPLIT=0 \
ONNX_ROOT="$DEVICE2_DATA_WORK_ROOT/dlrm_onnx_dyn" \
bash bin/run_full_data_generation.sh
```

该步骤只做 ONNX/run-config 准备，不采集性能数据。profiling 会占用 CPU/内存并影响 microbenchmark，raw dataset 和 split 又需要本轮硬件 YAML；因此 profiling、raw dataset 和 split 构建必须等 `DEVICE*_HW_CONFIG` 指向本轮硬件提取生成的 YAML 后再运行。

Device_2 / 通算平台本机：

```bash
DATA_ROOT="$DEVICE2_DATA_WORK_ROOT" \
SOURCE_HW_CONFIG="$DEVICE2_HW_CONFIG" \
SOURCE_OUTPUT_ROOT="$DEVICE2_SOURCE_RAW_ROOT" \
SPLIT_OUTPUT_ROOT="$DEVICE2_DATA_ROOT" \
E2E_SMALL_DIVISOR=1 \
E2E_WORKERS=0 \
bash bin/run_full_data_generation.sh
```

分步调试入口：

```bash
bash bin/00_check_env.sh
bash bin/01_generate_onnx.sh
bash bin/02_run_case_jobs.sh
bash bin/03_prepare_source_raw.sh
bash bin/04_build_joint_small_split.sh
bash bin/90_smoke_validate.sh
```

Device_1 / 智算平台也应在该平台本机运行同一命令，并把 `SOURCE_HW_CONFIG` 指向 `DEVICE1_HW_CONFIG`。`SOURCE_OUTPUT_ROOT` 使用该平台仓库内的固定 source raw root，`SPLIT_OUTPUT_ROOT` 指向 `DEVICE1_RAW_DATA_ROOT`。随后打成 portable root 后再复制到当前验收目录。

如果验收目标是“在当前新机器上重建 Device_2 / 通算平台结果，并复用已归档的 Device_1 / 智算平台数据做后续双平台实验”，可以跳过 Device_1 数据复采：直接把既有 Device_1 portable root 复制到 `$DEVICE1_DATA_ROOT`，并把 `$DEVICE1_HW_CONFIG` 指向随数据归档的 Device_1 YAML，例如 `calibrator/hw_config/kunpeng920_single_numa.yaml`。后续主线训练、CDF、few-shot 和误差解释性实验均可继续执行。

### 5.3 portable 数据打包

远端平台数据必须打成 portable root 后再复制到当前机器：

```bash
python calibrator/package_e2e_dataset.py \
  --input-root "$DEVICE1_RAW_DATA_ROOT" \
  --output-root "$DEVICE1_DATA_ROOT"
```

portable root 必须包含 split CSV、`portable_artifacts/`、shape CSV 和 branch timeline CSV。

当前归档 Device_1 数据如需重新打包，可使用：

```bash
python calibrator/package_e2e_dataset.py \
  --input-root data/data_e2e_full_kunpeng920_20260501 \
  --output-root data/data_e2e_full_kunpeng920_20260501_portable
```

### 5.4 主线训练

Device_2 / 通算平台：

```bash
python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root "$DEVICE2_DATA_ROOT" \
  --artifact-root "$DEVICE2_TRAIN_OUT" \
  --source-hw-config "$DEVICE2_HW_CONFIG" \
  --target-hw-config "$DEVICE1_HW_CONFIG" \
  --routes whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode fit \
  --export-model-bundles
```

Device_1 / 智算平台：

```bash
CROSS_HW_RF_N_JOBS=128 python calibrator/run.py \
  --policy joint_6d_tuple \
  --datasets source \
  --data-root "$DEVICE1_DATA_ROOT" \
  --artifact-root "$DEVICE1_TRAIN_OUT" \
  --source-hw-config "$DEVICE1_HW_CONFIG" \
  --target-hw-config "$DEVICE2_HW_CONFIG" \
  --routes whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --scales 100 \
  --passes 1 \
  --op-thread-scale-mode fit \
  --export-model-bundles
```

必须保留 `--op-thread-scale-mode fit`。当前主线指标使用 Analytical prior 后、RF 前的 `(op_type, num_threads)` 尺度校准。

训练后检查：

| 文件 | 用途 |
|---|---|
| `combined_summary.csv` | 单算子 MAPE、P90 APE |
| `best_e2e_pipeline_by_policy.csv` | 主线 E2E MAPE、P90 APE |
| `e2e_pipeline_summary.csv` | E2E coverage、完整/partial combo 数 |
| `prediction_error_rows.csv` | 单算子预测明细 |
| `e2e_pipeline_combo_metrics.csv` | combo-level E2E 明细 |
| `op_thread_scale_config_*.json/.csv` | op-thread scale 系数 |
| `model_bundles/*.pkl` | 训练后静态配置推理、趋势图和成本优化使用的模型 bundle |
| `pipeline_artifact_index_audit.csv` | shape/timeline artifact 完整性审计 |

通过标准：

- `ml_raw_hw_params_used = False`。
- `combo_count_partial = 0`。
- `coverage_ratio_mean = 1.0`。
- Device_2 full test `combo_count_full = 151`。
- Device_1 full portable test `combo_count_full = 454`。
- `model_bundles/*.pkl` 存在。

### 5.5 使用正式归档路径复现主线指标

如果直接使用当前仓库中的正式归档数据和输出目录，可以把上面的变量命令替换为固定路径。

Device_2 / 通算平台：

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

Device_1 / 智算平台：

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

## 6. 训练后静态配置推理

训练完成后，可以只输入 DLRM 上层配置和模型 bundle，直接得到主线 v3 E2E latency。该入口不读取 profile 生成的静态算子特征、shape CSV 或 timeline CSV。

如果只需要查看 DLRM 上层参数如何映射到各个算子规模特征，不加载模型 bundle、不做 latency 预测，可以运行独立映射脚本：

```bash
python standalone_demos/dlrm_operator_scale_mapping/dlrm_operator_scale_mapping_demo.py \
  --batch-size 1024 \
  --num-indices-per-lookup 1000 \
  --arch-embedding-size 25600 \
  --arch-mlp-bot 400-300-200 \
  --arch-mlp-top 1600-800-1 \
  --inter-threads 1 \
  --num-threads 3 \
  --out-dir outputs/static_mapping_explain_smoke
```

该脚本输出 `dlrm_static_op_shapes_full_graph_76.csv`、`dlrm_static_features_full_graph_76.csv`、`dlrm_static_model_input_51.csv` 和 `dlrm_operator_scale_mapping_summary.csv/json`。其中 summary 会列出每个算子的 shape、Gemm M/N/K、MAC 数、lookup count、reduction work items 和 I/O bytes。该 demo 只使用 Python 标准库，不依赖项目内其他脚本，不接入主模型推理链路。

训练后性能模型推理命令如下：

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

主要输出：

| 文件 | 用途 |
|---|---|
| `static_config.json` | 输入配置、bundle 路径和模型 hash |
| `static_op_shapes_full_graph_76.csv` | 自动生成的完整静态 shape 表 |
| `static_full_graph_76.csv` | 完整静态图，76 个节点 |
| `static_model_input_51.csv` | 主线可观测输入，51 个节点 |
| `static_operator_predictions.csv` | 单算子预测 |
| `static_e2e_schedule_spans.csv` | E2E 调度时间段 |
| `static_e2e_summary.json` | `predicted_e2e_us` 等主结果 |

通过标准：

- `static_model_input_51.csv` 有 51 行。
- `static_full_graph_76.csv` 有 76 行。
- `static_e2e_summary.json` 中 `predicted_e2e_us > 0`。
- 运行命令不需要 `--train-root`、`--hw-config`、shape CSV 或 timeline CSV。

## 7. 相关结果快速复现

E2E CDF、算子类 MAPE、batch/并行度/算子规模误差切片、软件负载泛化、训练后静态配置批量推理耗时和随机访存延迟验证均支持单 Device 复现：只有一个平台的数据时，只传该平台对应参数。跨平台 few-shot、NSGA-II 双平台设计空间探索和双平台 ablation 对比需要两个平台的 bundle/data/root；只有一个 Device 时跳过。

### 7.1 主线 E2E CDF

双 Device：

```bash
python calibrator/plot_e2e_relative_error_cdf.py \
  --device1-combo-metrics "$DEVICE1_TRAIN_OUT/e2e_pipeline_combo_metrics.csv" \
  --device2-combo-metrics "$DEVICE2_TRAIN_OUT/e2e_pipeline_combo_metrics.csv" \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --threshold 0.15 \
  --out-dir "<e2e_cdf_output_root>"
```

只有 Device_2 / 通算平台时：

```bash
python calibrator/plot_e2e_relative_error_cdf.py \
  --device2-combo-metrics "$DEVICE2_TRAIN_OUT/e2e_pipeline_combo_metrics.csv" \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --threshold 0.15 \
  --out-dir "<e2e_cdf_output_root>"
```

只有 Device_1 / 智算平台时，把上面参数改为：

```bash
--device1-combo-metrics "$DEVICE1_TRAIN_OUT/e2e_pipeline_combo_metrics.csv"
```

主要输出：

```text
e2e_relative_error_cdf_mainline.png
e2e_relative_error_cdf_mainline.pdf
```

当前正式结果可直接使用：

```bash
python calibrator/plot_e2e_relative_error_cdf.py \
  --device1-combo-metrics outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/e2e_pipeline_combo_metrics.csv \
  --device2-combo-metrics outputs/cache_residency_worker_share_v3_full_s920x20_20260505/e2e_pipeline_combo_metrics.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --threshold 0.15 \
  --out-dir outputs/e2e_relative_error_cdf_cache_residency_worker_share_20260505
```

### 7.2 E2E 消融 CDF

该实验用于比较两个平台上的消融曲线；只有一个 Device 时跳过，或改用 `e2e_pipeline_summary.csv` 和 `e2e_pipeline_combo_metrics.csv` 查看该 Device 的主线指标。

如 raw prior 输出已存在，直接运行：

```bash
python calibrator/experimental/plot_e2e_ablation_cdf.py \
  --device1-raw-root "<device1_raw_prior_output_root>" \
  --device2-raw-root "<device2_raw_prior_output_root>" \
  --device1-full-root "$DEVICE1_TRAIN_OUT" \
  --device2-full-root "$DEVICE2_TRAIN_OUT" \
  --rf-route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --prior-route whitebox_family_prior_only_v2 \
  --threshold 0.10,0.15 \
  --module-improvement-threshold 0.1 \
  --out-dir "<e2e_ablation_output_root>"
```

主要输出：

```text
e2e_ablation_cdf_combined.png
e2e_ablation_cdf_combined.pdf
```

raw prior 生成命令见 `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md`。

### 7.3 算子类 MAPE

双 Device：

```bash
python calibrator/plot_operator_class_mape.py \
  --device1-prediction-rows "$DEVICE1_TRAIN_OUT/prediction_error_rows.csv" \
  --device2-prediction-rows "$DEVICE2_TRAIN_OUT/prediction_error_rows.csv" \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir "<operator_class_mape_output_root>" \
  --split-gather
```

只有一个 Device 时只传对应参数，例如 Device_2 / 通算平台：

```bash
python calibrator/plot_operator_class_mape.py \
  --device2-prediction-rows "$DEVICE2_TRAIN_OUT/prediction_error_rows.csv" \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir "<operator_class_mape_output_root>" \
  --split-gather
```

主要输出：

```text
operator_class_mape_gather_split_bar.png
operator_class_mape_gather_split_bar.pdf
```

当前正式结果可直接使用：

```bash
python calibrator/plot_operator_class_mape.py \
  --device1-prediction-rows outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/prediction_error_rows.csv \
  --device2-prediction-rows outputs/cache_residency_worker_share_v3_full_s920x20_20260505/prediction_error_rows.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir outputs/operator_class_mape_cache_residency_worker_share_20260505 \
  --split-gather
```

### 7.3.1 图聚合任务精度与相对误差 CDF

该实验输出两种统计口径：逐个计算图任务按直接依赖关系将 `Shape`、`Unsqueeze`、`Flatten`、`Reshape`、`Add` 和 `Mul` 归入相邻的数据处理节点；四类执行阶段则按原节点名称前缀划分 embedding、bottom MLP、interaction 和 top MLP。两者均先对组内实测耗时和预测耗时分别求和，再计算 APE；本节 CDF 使用四类执行阶段口径。该过程只读取已有预测结果，不训练、不重拟合，也不改变主线预测。

```bash
TASK_CLASS_OUT=outputs/dlrm_task_class_mape_repro

python calibrator/calculate_graph_task_mape.py \
  --graph-shape-csv data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/portable_artifacts/case_10_run_1_1/bs1024_nip1800/op_shapes/op_shapes_1024_1800.csv \
  --device1-prediction-rows outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/prediction_error_rows.csv \
  --device2-prediction-rows outputs/cache_residency_worker_share_v3_full_s920x20_20260505/prediction_error_rows.csv \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir "$TASK_CLASS_OUT"

python calibrator/plot_task_class_relative_error_cdf.py \
  --task-prediction-rows "$TASK_CLASS_OUT/dlrm_task_class_prediction_rows.csv" \
  --threshold 0.10,0.15 \
  --out-dir "$TASK_CLASS_OUT/task_class_cdf"
```

主要输出：

```text
dlrm_task_class_mape_summary.csv
dlrm_task_class_prediction_rows.csv
dlrm_task_class_relative_error_cdf.png
dlrm_task_class_relative_error_cdf.pdf
dlrm_task_class_relative_error_cdf_summary.csv
DLRM_TASK_CLASS_RELATIVE_ERROR_CDF_REPORT.md
```

脚本会保留 MAPE 达到或超过 15% 的结果，并打印告警，便于诊断。

正式结果中，四类任务在 Device_1 的 MAPE 为 5.42% 至 9.52%，在 Device_2 为 3.56% 至 8.49%。在 15% 相对误差阈值内，Device_1 各任务类别的样本覆盖率为 79.7% 至 95.8%，Device_2 为 81.5% 至 94.7%。`dlrm_task_class_relative_error_cdf_summary.csv` 保存每类任务的 MAPE、P90/P95 APE 和阈值覆盖率。

### 7.4 batch、并行度和算子规模误差切片

双 Device：

```bash
python calibrator/experimental/plot_error_explainability_slices.py \
  --device1-root "$DEVICE1_TRAIN_OUT" \
  --device2-root "$DEVICE2_TRAIN_OUT" \
  --device1-test-csv "$DEVICE1_TEST_CSV" \
  --device2-test-csv "$DEVICE2_TEST_CSV" \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir "<error_explainability_output_root>"
```

只有一个 Device 时只传对应 root/test CSV，例如 Device_2 / 通算平台：

```bash
python calibrator/experimental/plot_error_explainability_slices.py \
  --device2-root "$DEVICE2_TRAIN_OUT" \
  --device2-test-csv "$DEVICE2_TEST_CSV" \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir "<error_explainability_output_root>"
```

主要输出：

```text
e2e_error_by_batch_distribution.png
e2e_error_by_batch_distribution.pdf
e2e_error_by_parallelism_distribution.png
e2e_error_by_parallelism_distribution.pdf
operator_error_vs_scale_facets.png
operator_error_vs_scale_facets.pdf
operator_pred_actual_vs_scale_facets.png
operator_pred_actual_vs_scale_facets.pdf
```

### 7.5 软件负载泛化性验证

该实验把真实 held-out full test 的误差切片、可选新增 unseen workload 评估，以及训练后 dense software workload 趋势推理整理到同一输出目录。只有一个 Device 时只传该 Device 的 root/test CSV/bundle；如果暂时没有新增 unseen 实测数据，可以不传 `--device*-unseen-*` 参数，脚本会在报告中标明只把 held-out full test 作为精度证据。

双 Device：

```bash
python calibrator/experimental/plot_software_workload_generalization.py \
  --device1-root "$DEVICE1_TRAIN_OUT" \
  --device1-test-csv "$DEVICE1_TEST_CSV" \
  --device1-bundle "$DEVICE1_BUNDLE" \
  --device2-root "$DEVICE2_TRAIN_OUT" \
  --device2-test-csv "$DEVICE2_TEST_CSV" \
  --device2-bundle "$DEVICE2_BUNDLE" \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir "<software_workload_generalization_output_root>"
```

只有 Device_2 / 通算平台时：

```bash
python calibrator/experimental/plot_software_workload_generalization.py \
  --device2-root "$DEVICE2_TRAIN_OUT" \
  --device2-test-csv "$DEVICE2_TEST_CSV" \
  --device2-bundle "$DEVICE2_BUNDLE" \
  --route whitebox_family_prior_plus_physics_guarded_rf_v3 \
  --out-dir "<software_workload_generalization_output_root>"
```

如已采集新增 unseen workload，把对应 root/test CSV 加到同一命令：

```bash
  --device2-unseen-root "<device2_unseen_eval_output_root>" \
  --device2-unseen-test-csv "<device2_unseen_test_csv>"
```

主要输出：

```text
software_generalization_e2e_cdf.png
software_generalization_e2e_cdf.pdf
software_generalization_mape_by_axis.png
software_generalization_mape_by_axis.pdf
software_generalization_pred_actual_trends.png
software_generalization_pred_actual_trends.pdf
operator_scale_generalization_facets.png
operator_scale_generalization_facets.pdf
static_config_dense_workload_trends.png
static_config_dense_workload_trends.pdf
SOFTWARE_WORKLOAD_GENERALIZATION_REPORT.md
```

### 7.6 训练后静态配置批量推理耗时

该实验评估模型 bundle 已加载后，一次提交不同数量静态配置样本时的整批推理耗时。`sample_count` 表示一次提交给性能模型的静态配置样本数，不是 DLRM 内部 batch size。只有一个 Device 时只传该 Device 的 bundle。

双 Device：

```bash
python calibrator/static_feature_mapping/benchmark_static_inference_latency.py \
  --device1-bundle "$DEVICE1_BUNDLE" \
  --device2-bundle "$DEVICE2_BUNDLE" \
  --out-dir "<static_inference_bulk_latency_output_root>" \
  --threshold-seconds 60 \
  --sample-counts 1,2,4,8,16,32,48,64 \
  --bulk-repeats 10 \
  --warmup 2 \
  --workload-set dlrm_s_m_l
```

只有 Device_2 / 通算平台时：

```bash
python calibrator/static_feature_mapping/benchmark_static_inference_latency.py \
  --device2-bundle "$DEVICE2_BUNDLE" \
  --out-dir "<static_inference_bulk_latency_output_root>" \
  --threshold-seconds 60 \
  --sample-counts 1,2,4,8,16,32,48,64 \
  --bulk-repeats 10 \
  --warmup 2 \
  --workload-set dlrm_s_m_l
```

当前快速全覆盖结果使用正式 bundle：

```bash
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

当前结果：

| 指标 | 数值 |
|---|---:|
| `cold_bulk_cli` 最大整批耗时 | 57.780s |
| `warm_bulk_inprocess` 最大整批耗时 | 56.315s |
| 最大均摊单样本耗时 | 3.874s/sample |
| 最高吞吐 | 1.140 samples/s |

主要输出：

```text
static_inference_warm_bulk_latency_vs_samples.png
static_inference_warm_bulk_latency_vs_samples.pdf
static_inference_bulk_latency_vs_samples.png
static_inference_bulk_latency_vs_samples.pdf
static_inference_bulk_latency_summary.csv
STATIC_INFERENCE_BULK_LATENCY_REPORT.md
```

当前归档输出目录：

```text
outputs/static_inference_bulk_latency_benchmark_20260507
```

### 7.6.1 白盒访存模块 CPU 计算开销

该实验测量已完成静态特征准备后，白盒连续访存和随机访存方程在 CPU 上的计算开销。它不是硬件 Cache/DRAM 实测访问延迟，也不包含 CSV 读取、静态特征展开、模型 bundle 加载或随机森林推理。连续访存部分覆盖带宽曲线、并发争用和数据搬运服务时间；随机访存部分覆盖 `Gather` 的随机带宽、缓存层级延迟、未命中并行度和 ORT Gather 内核带宽重标定。

```bash
python calibrator/benchmark_memory_model_overhead.py \
  --dataset data/data_e2e_full_kunpeng920_20260501_portable/source_kunpeng920_dataset3_unseen_joint_6d_tuple/e2e_test_raw.csv \
  --hw-config calibrator/hw_config/kunpeng920_single_numa.yaml \
  --sample-sizes 1,76,1024,4096 \
  --repeats 200 \
  --warmups 20 \
  --out-dir outputs/memory_model_overhead
```

脚本仅支持 `ort_kernel_v3`，连续访存包含 ReduceSum 的内核带宽修正，并将仅改变视图的 Transpose 搬运量置零。一致性校验逐行比较上下界和中心值，避免汇总求和掩盖偏差；默认校验通过后才写出结果。`--skip-equivalence-check` 会显式记录跳过状态，仅供诊断。历史 20260803 耗时来自修正前脚本，不能代表当前实现；当前开销应以上述命令重新测量。

主要输出：

```text
memory_model_overhead_summary.csv
memory_model_overhead_samples.csv
memory_model_overhead_metadata.json
memory_model_overhead_report.md
```

### 7.7 随机访存延迟验证

使用硬件自动校准生成的同批 raw root 和 candidate YAML：

```bash
python calibrator/experimental/validate_random_access_latency_model.py \
  --hw-config "$PTRCHASE_HW_CONFIG" \
  --ptrchase-root "$PTRCHASE_ROOT" \
  --num-threads 1 \
  --active-groups 1 \
  --modes current_interp,ppt_weighted,hard_threshold \
  --stable-cv-threshold 15 \
  --target-mape-threshold 20 \
  --out-dir "<random_access_latency_validation_output_root>"
```

主要输出：

```text
random_access_latency_model_vs_measured_continuous_residency.png
random_access_latency_model_vs_measured_continuous_residency.pdf
random_access_latency_model_vs_measured_weighted_average.png
random_access_latency_model_vs_measured_weighted_average.pdf
random_access_latency_validation_summary.csv
```

### 7.8 跨平台 few-shot 泛化

该实验必须同时具备两个平台的 data root、hardware YAML 和 model bundle；只有一个 Device 时跳过。

```bash
python calibrator/experimental/cross_platform_fewshot_generalization.py \
  --device1-bundle "$DEVICE1_BUNDLE" \
  --device2-bundle "$DEVICE2_BUNDLE" \
  --device1-data-root "$DEVICE1_DATA_ROOT" \
  --device2-data-root "$DEVICE2_DATA_ROOT" \
  --device1-hw-config "$DEVICE1_HW_CONFIG" \
  --device2-hw-config "$DEVICE2_HW_CONFIG" \
  --sample-fractions 0,1,2,5,10,20,100 \
  --repeats 5 \
  --seed "$FEWSHOT_SEED" \
  --primary-method fewshot_op_rescale_plus_e2e_surface \
  --workers 16 \
  --target-only-baseline-mode no_calibration \
  --anchor-target-only-no-calibration-to-target-full \
  --target-only-anchor-power 0.5 \
  --out-dir "<fewshot_generalization_output_root>"
```

主要输出：

```text
fewshot_10pct_vs_full_bar.png
fewshot_10pct_vs_full_bar.pdf
fewshot_e2e_mape_vs_target_data.png
fewshot_e2e_mape_vs_target_data.pdf
fewshot_e2e_mape_vs_target_data_simplified_from_1pct.png
fewshot_e2e_mape_vs_target_data_simplified_from_1pct.pdf
fewshot_e2e_p90_vs_target_data.png
fewshot_e2e_p90_vs_target_data.pdf
fewshot_generalization_summary.csv
fewshot_generalization_plot_summary.csv
fewshot_data_efficiency_summary.csv
CROSS_PLATFORM_FEWSHOT_GENERALIZATION_REPORT.md
```

### 7.9 NSGA-II 设计空间探索

该实验比较 Device_1 baseline 与 Device_2 reference design，需要两个平台 bundle 和硬件 YAML；只有一个 Device 时跳过。它会批量评估候选硬件配置，耗时取决于 `--nsga2-population`、`--nsga2-generations` 和 `--prediction-workers`。快速连通性验证可把 population/generations 降到 `16/3`；报告图建议使用下列正式参数。

```bash
python calibrator/hardware_cost_optimizer.py \
  --device1-bundle "$DEVICE1_BUNDLE" \
  --device2-bundle "$DEVICE2_BUNDLE" \
  --device1-hw-config "$DEVICE1_HW_CONFIG" \
  --device2-hw-config "$DEVICE2_HW_CONFIG" \
  --workload-set dlrm_s_m_l \
  --search-algorithm nsga2 \
  --cost-model reference_v2 \
  --cost-constraints calibrator/configs/hardware_cost_constraints_v2.json \
  --pair-extrapolation-mode forbid \
  --budget-unit wan \
  --budgets 8,8.5,9,10,12,14,16,device2 \
  --grid-profile full \
  --nsga2-population 96 \
  --nsga2-generations 50 \
  --prediction-workers 16 \
  --no-refit \
  --out-dir "$HW_COST_OPT_OUT"
```

主要输出：

```text
pareto_frontier_cost_latency.png
pareto_frontier_cost_latency.pdf
nsga2_candidate_objectives.csv
nsga2_best_by_budget.csv
candidate_parameter_costs.csv
HARDWARE_COST_OPTIMIZATION_REPORT.md
```

不要用 CSV summary、prediction rows 或 op-thread scale config 代替 `.pkl` bundle。缺 bundle 时只能显式启用 prior-only fallback，该结果只适合流程检查，不作为 RF v3 优化结论。

### 7.10 multiworkload 硬件趋势图

推荐使用已训练 bundle 复用模式：

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

趋势图输出目录：

```text
outputs/cache_residency_worker_share_v3_trends_20260505/trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload
```

### 7.11 其他实验入口

| 实验 | 复现说明 |
|---|---|
| multiworkload 硬件趋势图和 elasticity | `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md` |
| 训练后静态配置推理细节 | `docs/MAINLINE_EVALUATION_EXPERIMENTS.md` |
| CPU+NPU experimental | `docs/REPRODUCIBLE_END_TO_END_PIPELINE.md` |

## 8. 成本约束来源

成本优化使用已有 RF v3 bundle，不重新训练模型。`docs/参数约束v2.docx` 的 x1-x25 两点参考成本已固化到：

```text
calibrator/configs/hardware_cost_constraints_v2.json
```

正式 NSGA-II 命令见 7.9。单次快速连通性验证可降低 `--nsga2-population` 和 `--nsga2-generations`，但交付报告不使用降规模参数作为正式结果。

## 9. 生成产物和 git 约定

以下目录是本地生成产物，不进入 git：

```text
outputs/
data/
hardware_calibration/raw/
```

模型 bundle 只有运行训练时显式加 `--export-model-bundles` 才会生成。`combined_summary.csv`、`prediction_error_rows.csv`、`e2e_pipeline_summary.csv` 和 `op_thread_scale_config_*.json` 不能替代 `model_bundles/*.pkl`。

## 10. 常见失败排查

| 现象 | 优先检查 |
|---|---|
| `model_bundles/*.pkl` 不存在 | 训练命令是否带 `--export-model-bundles` |
| 静态配置推理要求 `--train-root` 或 `--hw-config` | 是否使用了旧入口，或没有传 `--model-bundle` |
| `combo_count_full=0` | portable artifact 是否完整 |
| `shape_csv_missing` 或 `timeline_csv_missing` | 是否只复制 split CSV，没有复制 portable root |
| E2E 指标与交付表不一致 | route、data root、硬件 YAML、`--op-thread-scale-mode fit` 是否一致 |
| CDF 数值不一致 | 输入是否为 combo-level `e2e_pipeline_combo_metrics.csv`，并过滤 `coverage_ratio=1.0` |
| 趋势后处理找不到 points CSV | 是否先运行趋势图脚本，且 `--output-root` 是否一致 |
| 成本优化器报 bundle missing | 必须提供 `.pkl` bundle，CSV 不能恢复 RF 权重 |
| CPU+NPU 图被误读为真实加速 | CPU+NPU 当前只是 independent experimental 后处理，不覆盖 CPU 主线 |

## 11. 使用边界

- 当前 CPU 主线结论只使用 correct-hw full test、完整 E2E combo 和显式导出的 bundle。
- 远端数据必须使用 portable root，不能只复制 split CSV。
- few-shot、CPU+NPU、趋势图、误差解释和成本优化都是独立评估或应用分支，不修改 CPU 主线 MAPE/P90 表。
- static dense trend 没有真实标签，只能说明模型对上层配置变化的响应方向，不能替代精度验证。
- `outputs/20260429` 相关 small split 结果只保留为历史快速验证参考，不作为当前 full 数据主线结论。
