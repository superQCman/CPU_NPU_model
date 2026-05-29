# 环境清单

`requirements.txt` 是当前 CPU 建模和分析流程迁移到其他机器时使用的可移植 pip 依赖输入，覆盖主线 RF route、硬件趋势/成本实验以及报告后处理。

`environment.yml` 和 `conda_list.txt` 保留为本地环境审计参考。如果目标机器已经有合适的 Python 环境，它们不是必需输入。`pip_freeze.txt` 特意不包含旧机器上的本地 wheel 路径。

本目录不包含二进制 conda 环境。迁移到目标机器时，可以先激活需要使用的环境，再用当前 shell 的 `python` 安装可移植依赖：

```bash
conda activate ort  # 可选
python --version
python -m pip install -r env/requirements.txt
python env/check_env.py
```

PyTorch 没有放进默认可移植依赖，因为正确 wheel 取决于 CPU 架构和加速器运行时。在运行 DLRM ONNX/数据生成或历史 MLP baseline 之前，需要先安装与平台匹配的 PyTorch，然后执行：

```bash
python env/check_env.py --include-data-generation
```

厂商 NPU 运行时包，例如 `torch-npu` 或 `onnxruntime-cann`，也不属于默认可移植依赖。只有需要真实加速器执行时，才需要单独安装这些包。
