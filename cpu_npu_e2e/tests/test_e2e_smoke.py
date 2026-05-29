from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cpu_npu_e2e.fusion import run_fusion, run_fusion_cases


ROOT = Path(__file__).resolve().parents[1]
DELIVERY_ROOT = ROOT.parent
CPU_MODEL_BUNDLE = Path(
    "/data/qc/dlrm/CPU_model_stable/outputs_2/outputs/cache_residency_worker_share_v3_full_s920x20_20260505/"
    "model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl"
)


def test_e2e_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    shape_csv = next((DELIVERY_ROOT / "CPU_model_stable" / "pipeline_artifacts").glob("**/op_shapes.csv"))
    summary = run_fusion(
        cpu_model_root=DELIVERY_ROOT / "CPU_model_stable",
        npu_root=DELIVERY_ROOT / "npu_modeling",
        alltoall_root=DELIVERY_ROOT / "alltoall_modeling",
        routing_path=ROOT / "configs/cpu_npu_op_routing.yaml",
        output_dir=tmp_path,
        shape_csv=shape_csv,
    )
    assert summary["predicted_e2e_us"] >= 0
    for name in [
        "fused_node_predictions.csv",
        "alltoall_region_predictions.csv",
        "fused_e2e_combo_metrics.csv",
        "routing_coverage.csv",
        "summary.json",
    ]:
        assert (tmp_path / name).exists()
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["cpu_static_fallback_used"] is True


def test_e2e_with_cpu_model_bundle_disables_static_fallback(tmp_path: Path) -> None:
    shape_csv = (
        DELIVERY_ROOT
        / "CPU_model_stable"
        / "pipeline_artifacts/source_kunpeng920_dataset3_unseen_arch_embedding_size/case_10_run_1_1/bs1056_nip2000/op_shapes.csv"
    )
    summary = run_fusion(
        cpu_model_root=DELIVERY_ROOT / "CPU_model_stable",
        npu_root=DELIVERY_ROOT / "npu_modeling",
        alltoall_root=DELIVERY_ROOT / "alltoall_modeling",
        routing_path=ROOT / "configs/cpu_npu_op_routing.yaml",
        output_dir=tmp_path,
        shape_csv=shape_csv,
        cpu_model_bundle=CPU_MODEL_BUNDLE,
        allow_cpu_static_fallback=False,
    )
    assert summary["cpu_static_fallback_used"] is False
    assert summary["cpu_prediction_source"] == "cpu_model_bundle_full_graph_76"
    assert (tmp_path / "cpu_operator_predictions.csv").exists()
    node_df = pd.read_csv(tmp_path / "fused_node_predictions.csv")
    cpu_sources = set(node_df[node_df["route_backend"].eq("CPU")]["prediction_source"])
    assert cpu_sources == {"cpu_model_bundle_full_graph_76"}


def test_e2e_cases_config_writes_aggregate_outputs(tmp_path: Path) -> None:
    summary = run_fusion_cases(
        cpu_model_root=DELIVERY_ROOT / "CPU_model_stable",
        npu_root=DELIVERY_ROOT / "npu_modeling",
        alltoall_root=DELIVERY_ROOT / "alltoall_modeling",
        routing_path=ROOT / "configs/cpu_npu_op_routing.yaml",
        output_dir=tmp_path,
        cases_config=ROOT / "configs/e2e_test_cases.yaml",
    )
    assert summary["case_count"] == 3
    metrics = pd.read_csv(tmp_path / "fused_e2e_combo_metrics.csv")
    assert len(metrics) == 3
    assert set(metrics["e2e_case_name"]) == {case["name"] for case in summary["cases"]}
    for case in summary["cases"]:
        assert (tmp_path / "cases" / case["name"] / "summary.json").exists()
