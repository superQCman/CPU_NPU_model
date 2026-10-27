"""Regression checks for the updated NPU delivery reports."""

from pathlib import Path
import sys

import pandas as pd
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_ROOT))

import benchmark_prediction_time as benchmark
from train_physical_memory_models import write_report


def test_operator_report_supports_partial_operator_datasets(tmp_path):
    write_report(
        tmp_path,
        summary={"stable_rows": 4, "test_rows": 1, "models_total": 1},
        overall={"sample_count": 1, "mape_1us_floor_percent": 5.0},
        family_df=pd.DataFrame([{"model_group": "StructuredComputeMemory", "sample_count": 1}]),
        op_df=pd.DataFrame([{"op_type": "Relu", "mape_1us_floor_percent": 5.0}]),
        residual_df=pd.DataFrame(),
    )
    report = (tmp_path / "PHYSICAL_MEMORY_MODEL_REPORT.md").read_text()
    assert "Relu" in report
    assert "BatchMatMulV2" not in report
    assert "Average Error By Operator" in report


def test_timing_benchmark_rejects_unpredicted_holdout_rows(monkeypatch):
    monkeypatch.setattr(benchmark.joblib, "load", lambda _: {"hardware": {}, "models": {}})
    monkeypatch.setattr(benchmark, "add_taxonomy", lambda frame: frame.copy())
    monkeypatch.setattr(benchmark, "add_hierarchical_memory_features", lambda frame, _: frame)
    holdout = pd.DataFrame({"case_id": ["missing"], "model_group": ["unsupported"]})
    with pytest.raises(ValueError, match="finite predictions for every holdout row"):
        benchmark.run_once(holdout, Path("unused.joblib.gz"))
