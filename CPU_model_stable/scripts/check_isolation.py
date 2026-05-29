from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "/data/qc/dlrm/ORT/" + "CPU_Perf_Model",
    "ORT." + "CPU_Perf_Model",
)
SCAN_DIRS = [
    "bin",
    "calibrator",
    "analytical_base",
    "generalization_benchmarks",
    "cpu_perf_model",
    "ort_src",
    "dlrm_src",
    "scripts",
]
MODULES = [
    "calibrator.run",
    "calibrator.pipeline_eval",
    "calibrator.whitebox_family_routes",
    "analytical_base.run",
    "cpu_perf_model.dataset_builder",
    "generalization_benchmarks.dataset3_context_proxy_study",
]


def _assert_local_module(name: str) -> None:
    module = importlib.import_module(name)
    module_file = Path(getattr(module, "__file__", "")).resolve()
    if ROOT not in module_file.parents and module_file != ROOT:
        raise SystemExit(f"module {name} resolved outside CPU_model_stable: {module_file}")


def _scan_for_forbidden_paths() -> None:
    hits: list[str] = []
    for rel in SCAN_DIRS:
        base = ROOT / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".sh", ".json", ".yaml", ".yml", ".md", ".csv"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for needle in FORBIDDEN:
                if needle in text:
                    hits.append(f"{path}: contains {needle}")
    if hits:
        raise SystemExit("\n".join(hits))


def main() -> None:
    for module_name in MODULES:
        _assert_local_module(module_name)
    _scan_for_forbidden_paths()
    print(f"isolation_ok root={ROOT}")


if __name__ == "__main__":
    main()
