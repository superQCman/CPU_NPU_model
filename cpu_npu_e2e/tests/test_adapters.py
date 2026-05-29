from __future__ import annotations

from pathlib import Path

from cpu_npu_e2e.alltoall_adapter import AllToAllAdapter
from cpu_npu_e2e.cpu_adapter import CpuLatencyAdapter, run_cpu_static_model_inference
from cpu_npu_e2e.npu_adapter import NpuLatencyAdapter
from cpu_npu_e2e.routing import RoutingTable
from cpu_npu_e2e.shapes import build_graph, load_op_shapes


ROOT = Path(__file__).resolve().parents[1]
DELIVERY_ROOT = ROOT.parent
CPU_MODEL_BUNDLE = Path(
    "/data/qc/dlrm/CPU_model_stable/outputs_2/outputs/cache_residency_worker_share_v3_full_s920x20_20260505/"
    "model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl"
)


def _sample_shape_csv() -> Path:
    return (
        DELIVERY_ROOT
        / "CPU_model_stable"
        / "pipeline_artifacts/source_kunpeng920_dataset3_unseen_arch_embedding_size/case_10_run_1_1/bs1056_nip2000/op_shapes.csv"
    )


def _sample_graph():
    shape_csv = _sample_shape_csv()
    return build_graph(load_op_shapes(shape_csv))


def test_alltoall_adapter_returns_nonnegative_latency() -> None:
    adapter = AllToAllAdapter(DELIVERY_ROOT / "alltoall_modeling")
    out = adapter.predict_region(128 * 1024 * 1024, compute_window_us=500.0)
    assert out.comm_us >= 0
    assert out.visible_comm_us >= 0
    assert 0 <= out.hidden_fraction_of_comm <= 1


def test_cpu_static_fallback_returns_nonnegative_latency() -> None:
    graph = _sample_graph()
    node = next(node for node in graph.values() if node.op_type == "Reshape")
    pred, source = CpuLatencyAdapter().predict_us(node)
    assert pred >= 0
    assert source == "cpu_static_fallback"


def test_cpu_model_bundle_covers_cpu_routed_nodes_without_fallback() -> None:
    graph = _sample_graph()
    routing = RoutingTable.from_yaml(ROOT / "configs/cpu_npu_op_routing.yaml")
    required = {idx for idx, node in graph.items() if routing.route_for(node.op_type).backend == "CPU"}
    result = run_cpu_static_model_inference(
        cpu_model_root=DELIVERY_ROOT / "CPU_model_stable",
        model_bundle_path=CPU_MODEL_BUNDLE,
        graph=graph,
        shape_csv=_sample_shape_csv(),
        required_node_ids=required,
    )
    pred_by_idx = set(result.predictions["op_idx"].astype(int))
    assert required.issubset(pred_by_idx)
    assert result.summary["required_cpu_node_count"] == len(required)
    assert result.summary["cpu_prediction_source"] == "cpu_model_bundle_full_graph_76"


def test_npu_adapter_predicts_one_gather() -> None:
    graph = _sample_graph()
    node = next(node for node in graph.values() if node.op_type == "Gather")
    adapter = NpuLatencyAdapter(DELIVERY_ROOT / "npu_modeling")
    pred, source = adapter.predict_us(node, "Gather")
    assert pred >= 0
    assert source


def test_npu_adapter_can_load_compressed_model_artifact(tmp_path: Path) -> None:
    npu_root = tmp_path / "npu_modeling"
    npu_root.mkdir()
    (npu_root / "code").symlink_to(DELIVERY_ROOT / "npu_modeling" / "code", target_is_directory=True)
    (npu_root / "models").symlink_to(DELIVERY_ROOT / "npu_modeling" / "models", target_is_directory=True)
    (npu_root / "results").mkdir()
    (npu_root / "results" / "physical_memory_models.joblib.gz").symlink_to(
        DELIVERY_ROOT / "npu_modeling" / "results" / "physical_memory_models.joblib.gz"
    )

    graph = _sample_graph()
    node = next(node for node in graph.values() if node.op_type == "Gather")
    adapter = NpuLatencyAdapter(npu_root)
    pred, source = adapter.predict_us(node, "Gather")
    assert adapter.model_artifact_path.name == "physical_memory_models.joblib.gz"
    assert pred >= 0
    assert source
