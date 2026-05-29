#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import platform
import sys


MODULES = [
    "numpy",
    "pandas",
    "sklearn",
    "onnx",
    "onnxruntime",
]

DATA_GENERATION_MODULES = [
    "torch",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the portable CPU_model_stable Python environment.")
    parser.add_argument(
        "--include-data-generation",
        action="store_true",
        help="Also require optional PyTorch dependency used by DLRM ONNX/data generation.",
    )
    args = parser.parse_args()

    print(f"python={sys.executable}")
    print(f"version={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    ok = True
    modules = list(MODULES)
    if args.include_data_generation:
        modules.extend(DATA_GENERATION_MODULES)
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - environment diagnostic
            ok = False
            print(f"[FAIL] import {name}: {exc}")
            continue
        version = getattr(module, "__version__", "<unknown>")
        print(f"[ OK ] import {name}: {version}")
    try:
        import onnxruntime as ort

        print(f"onnxruntime_providers={ort.get_available_providers()}")
    except Exception as exc:  # pragma: no cover - environment diagnostic
        ok = False
        print(f"[FAIL] onnxruntime provider query: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
