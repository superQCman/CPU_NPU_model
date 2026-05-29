#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CPU_MODEL_STABLE_ROOT:-}" ]]; then
  CPU_MODEL_STABLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
else
  CPU_MODEL_STABLE_ROOT=$(cd "$CPU_MODEL_STABLE_ROOT" && pwd)
fi
export CPU_MODEL_STABLE_ROOT

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CPU_MODEL_STABLE_PYTHON_BIN:-}" && -x "$CPU_MODEL_STABLE_PYTHON_BIN" ]]; then
    PYTHON_BIN="$CPU_MODEL_STABLE_PYTHON_BIN"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  else
    PYTHON_BIN=$(command -v python3 || command -v python)
  fi
fi
export PYTHON_BIN

export CPU_MODEL_STABLE_ORT_SRC_ROOT="${CPU_MODEL_STABLE_ORT_SRC_ROOT:-$CPU_MODEL_STABLE_ROOT/ort_src}"

_stable_py_path="$CPU_MODEL_STABLE_ROOT:$CPU_MODEL_STABLE_ROOT/dlrm_src:$CPU_MODEL_STABLE_ROOT/ort_src:$CPU_MODEL_STABLE_ROOT/calibrator"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$_stable_py_path:$PYTHONPATH"
else
  export PYTHONPATH="$_stable_py_path"
fi

if [[ -z "${PYTHON_SITE_PACKAGES:-}" ]]; then
  PYTHON_SITE_PACKAGES=$("$PYTHON_BIN" -c 'import site; paths=site.getsitepackages(); print(paths[0] if paths else "")' 2>/dev/null || true)
fi
export PYTHON_SITE_PACKAGES

if [[ -z "${ASCEND_ENV_SH:-}" && -n "${ASCEND_HOME:-}" ]]; then
  export ASCEND_ENV_SH="$ASCEND_HOME/ascend-toolkit/set_env.sh"
else
  export ASCEND_ENV_SH="${ASCEND_ENV_SH:-}"
fi

# CPU data-generation jobs should not auto-load optional device backends.
# If torch_npu is installed without the Ascend runtime on LD_LIBRARY_PATH,
# PyTorch can fail during a plain "import torch".
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
