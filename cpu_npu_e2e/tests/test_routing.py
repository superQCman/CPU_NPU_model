from __future__ import annotations

from pathlib import Path

import pytest

from cpu_npu_e2e.routing import RoutingTable


ROOT = Path(__file__).resolve().parents[1]


def test_default_routing_covers_delivered_cpu_graph_ops() -> None:
    routing = RoutingTable.from_yaml(ROOT / "configs/cpu_npu_op_routing.yaml")
    routing.validate_all(
        [
            "Add",
            "Concat",
            "Constant",
            "Flatten",
            "Gather",
            "Gemm",
            "MatMul",
            "Mul",
            "ReduceSum",
            "Relu",
            "Reshape",
            "Shape",
            "Sigmoid",
            "Transpose",
            "Unsqueeze",
        ]
    )


def test_unknown_op_type_is_hard_error() -> None:
    routing = RoutingTable.from_yaml(ROOT / "configs/cpu_npu_op_routing.yaml")
    with pytest.raises(KeyError):
        routing.validate_all(["UnknownOp"])

