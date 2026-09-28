"""Regression checks for the standalone mapping and delivery reports."""

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibrator import benchmark_memory_model_overhead as memory
from calibrator import calculate_graph_task_mape as tasks
from calibrator.static_feature_mapping.core import (
    DlrmStaticConfig, FULL_GRAPH_76, build_static_feature_frame, build_static_op_shape_table,
)
from standalone_demos.dlrm_operator_scale_mapping import dlrm_operator_scale_mapping_demo as demo


@pytest.mark.parametrize("batch,nip,threads", [(1,1,1), (9,9,3), (16,10,4)])
def test_standalone_mapping_matches_mainline(batch, nip, threads):
    config = DlrmStaticConfig(batch, nip, "1000-2000-3000-4000-5000-6000-7000-8000", "64-32-16", 2, threads)
    mapping = demo.map_dlrm_config_to_operator_scale(demo.StaticConfig(**asdict(config)))
    demo.run_self_test(mapping)
    expected = build_static_feature_frame(config, FULL_GRAPH_76).set_index("node_idx")
    actual = pd.DataFrame(mapping["feature_rows"]).set_index("node_idx")
    columns = [column for column in actual if column.startswith("feat_") and column in expected]
    columns += ["output_size", "activation_size", "parameter_size"]
    np.testing.assert_allclose(actual[columns], expected.loc[actual.index, columns], rtol=1e-12)
    assert {row["shape_source"] for row in mapping["summary_rows"]} == {"static_config"}
    pd.testing.assert_frame_equal(
        pd.DataFrame(mapping["shape_rows"]), build_static_op_shape_table(config), check_dtype=False,
    )


@pytest.mark.parametrize("field", ["batch_size", "num_indices_per_lookup", "num_threads", "inter_threads"])
def test_standalone_rejects_invalid_sizes(field):
    config = asdict(DlrmStaticConfig(16, 10, "1000", "64-32-16", 1, 2))
    config[field] = 0
    with pytest.raises(ValueError, match=field):
        demo.map_dlrm_config_to_operator_scale(demo.StaticConfig(**config))


@pytest.mark.parametrize("batch,inter_threads,threads", [(1,1,1), (16,2,4)])
@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_memory_intervals_match_every_formal_row(batch, inter_threads, threads):
    raw = build_static_feature_frame(
        DlrmStaticConfig(batch, 10, "1000", "64-32-16", inter_threads, threads), FULL_GRAPH_76,
    )
    hw = memory._load_hw_params(ROOT / "calibrator/hw_config/kunpeng920_single_numa.yaml")
    prepared = memory._prepare_base_columns(raw, hw, prior_variant="ort_kernel_v3").copy()
    random = prepared.loc[prepared.whitebox_exec_mode.eq(memory.EXEC_MODE_INDEXED_BLOCK_COPY)]
    continuous = prepared.loc[prepared.whitebox_exec_mode.isin(memory.CONTINUOUS_EXEC_MODES)]
    assert set(continuous.whitebox_exec_mode) == memory.CONTINUOUS_EXEC_MODES
    result = memory._verify_formal_equivalence(
        raw, prepared, random, continuous, hw, prior_variant="ort_kernel_v3", rows_per_module=100,
    )
    assert result["status"] == "passed"


def test_reports_export_above_threshold_and_conserve_latency(tmp_path, monkeypatch):
    config = DlrmStaticConfig(16, 10, "1000", "64-32-16", 1, 2)
    shape_path = tmp_path / "shapes.csv"
    build_static_op_shape_table(config).to_csv(shape_path, index=False)
    raw = build_static_feature_frame(config)
    raw["route"] = tasks.DEFAULT_ROUTE
    raw["actual_us"] = np.arange(1, len(raw) + 1, dtype=float)
    raw["pred_us"] = raw.actual_us * 2.0
    input_path = tmp_path / "predictions.csv"
    raw.to_csv(input_path, index=False)
    out_dir = tmp_path / "reports"
    monkeypatch.setattr(sys, "argv", [
        "calculate_graph_task_mape.py", "--graph-shape-csv", str(shape_path),
        "--device1-prediction-rows", str(input_path), "--device2-prediction-rows", str(input_path),
        "--out-dir", str(out_dir),
    ])
    tasks.main()
    for filename in ("graph_task_prediction_rows.csv", "dlrm_task_class_prediction_rows.csv"):
        report = pd.read_csv(out_dir / filename)
        for _, group in report.groupby("device"):
            assert group.actual_us.sum() == raw.actual_us.sum()
            assert group.pred_us.sum() == raw.pred_us.sum()
            np.testing.assert_allclose(group.ape, 1.0)
    summary = pd.read_csv(out_dir / "dlrm_task_class_mape_summary.csv")
    assert not summary.mape_below_15pct.any()


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_reports_reject_invalid_actual_latency(tmp_path, invalid):
    frame = pd.DataFrame([dict(
        row_uid="sample::opidx:0", case_id="case", source_name="source", combo="combo",
        node_name="/emb_l0/Gather", op_type="Gather", route=tasks.DEFAULT_ROUTE,
        actual_us=invalid, pred_us=1.0,
    )])
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid actual_us"):
        tasks.read_prediction_rows(path, "Device_1", tasks.DEFAULT_ROUTE)
