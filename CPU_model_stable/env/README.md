# Environment Manifests

`requirements.txt` is the portable pip input for moving the current CPU modeling
and analysis workflow to another machine. It covers the mainline RF route,
hardware trend/cost experiments, and report post-processing.

`environment.yml` and `conda_list.txt` are retained as local environment audit
references. They are not required if the target machine already has a suitable
Python environment. `pip_freeze.txt` intentionally does not contain the old
machine-local local wheel paths.

This directory does not contain the binary conda environment. On the target
machine, activate the desired environment if needed and install the portable
requirements with the current shell's `python`:

```bash
conda activate ort  # optional
python --version
python -m pip install -r env/requirements.txt
python env/check_env.py
```

PyTorch is intentionally not part of the default portable requirements because
the correct wheel depends on CPU architecture and accelerator runtime. Install a
platform-compatible PyTorch before running DLRM ONNX/data generation or legacy
MLP baselines, then verify it with:

```bash
python env/check_env.py --include-data-generation
```

Vendor NPU runtime packages such as `torch-npu` or `onnxruntime-cann` are not
part of the default portable requirements. Install them separately only when
real accelerator execution is required.
