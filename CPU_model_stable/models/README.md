# 已训练 CPU 模型

本目录直接交付两个平台的正式 CPU 模型，主线 route 为 `whitebox_family_prior_plus_physics_guarded_rf_v3`。模型来自 20260505 full split 训练结果，本次仅对原始 pickle 文件进行无损 XZ 压缩，没有重新训练或修改模型参数。

| 平台 | 模型路径 | 原训练结果目录 |
|---|---|---|
| Device_1 / 智算平台 | `kunpeng920/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl.xz.parts.json` | `cache_residency_worker_share_v3_full_kunpeng920_20260505` |
| Device_2 / 通算平台 | `s920x20/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl.xz` | `cache_residency_worker_share_v3_full_s920x20_20260505` |

`kunpeng920` / `s920x20` 沿用项目的历史平台标识，与交付文档中的 Device_1 / Device_2 对应。

## 使用

从 `CPU_model_stable/` 执行，不需要手动解压：

```bash
python calibrator/static_feature_mapping/predict_from_config.py \
  --model-bundle models/s920x20/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl.xz \
  --batch-size 1024 --num-indices-per-lookup 1000 \
  --arch-embedding-size 25600 --arch-mlp-bot 400-300-200 \
  --inter-threads 1 --num-threads 3 \
  --out-dir outputs/pretrained_static_inference
```

Device_1 使用 `models/kunpeng920/` 下的 `.pkl.xz.parts.json` 清单。该模型压缩后仍超过单文件限制，因此按 80 MiB 分片提交；清单和分片必须放在同一目录，加载器自动拼接压缩流到临时文件并读取，退出时删除临时文件。模型 bundle 已含硬件参数、op-thread scale 和随机森林模型。CPU/NPU 融合入口的 `--cpu-model-bundle` 也支持压缩文件和分片清单，严格模式使用方式见 [融合模型说明](../../cpu_npu_e2e/README.md)。

原始 `.pkl` 继续支持。独立使用只接受原始 pickle 的历史分析工具时，可先解压到被忽略的本机输出目录：

```bash
mkdir -p outputs/pretrained_models
xz -dc models/s920x20/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl.xz \
  > outputs/pretrained_models/s920x20.pkl
cat models/kunpeng920/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl.xz.part?? \
  | xz -dc > outputs/pretrained_models/kunpeng920.pkl
```

## 完整性与结果一致性

[manifest.json](manifest.json) 保存压缩文件和原始 pickle 的字节数与 SHA256。压缩版解压后的字节与原始训练文件完全一致。

[verification_20261020.json](verification_20261020.json) 保存原始 `.pkl` 与交付 `.pkl.xz` 的实际推理对比：每个平台覆盖三种静态配置、51/76 节点两种口径及三组严格 CPU/NPU 融合用例，比较逐算子预测、静态调度和融合输出表，要求模型字节完全一致、预测和调度数值在绝对容差 `1e-9` 微秒和相对容差 `1e-12` 内一致，离散字段保持相同。验收范围是打包前后的行为一致，不是重新训练或复测硬件准确率。

在仓库根目录验证文件校验和及原始、压缩和分片格式的加载兼容性：

```bash
python -m pytest -q CPU_model_stable/tests/test_model_bundles.py
```

在 `cpu_npu_e2e/` 运行 `python -m pytest -q`，正式 CPU 模型测试只使用本仓库中的文件，覆盖两个平台，不依赖项目外的模型目录。

本次对比使用的 Python、NumPy、pandas 和 scikit-learn 版本记录在验证 JSON 中。已有序列化模型来自 scikit-learn 1.6.1；在本次环境 scikit-learn 1.8.0 中验证了原始与压缩模型结果一致，其他环境仍应按项目环境说明部署。
