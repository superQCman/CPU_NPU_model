from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "uint16": 2,
    "int16": 2,
    "float": 4,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "double": 8,
    "float64": 8,
    "uint64": 8,
    "int64": 8,
}

BRANCH_NAME_RE = re.compile(r"^/emb_l(?P<branch_idx>\d+)/")


@dataclass(frozen=True)
class TensorSpec:
    tensor_name: str
    dtype: str
    dims: tuple[int, ...]
    tensor_role: str
    port_idx: int

    @property
    def element_count(self) -> int:
        product = 1
        if not self.dims:
            return 0
        for dim in self.dims:
            product *= max(int(dim), 1)
        return product

    @property
    def byte_count(self) -> int:
        return int(self.element_count * DTYPE_BYTES.get(self.dtype.lower(), 4))


@dataclass(frozen=True)
class GraphNode:
    node_idx: int
    node_name: str
    op_type: str
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]
    partition: str
    branch_idx: int | None
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


def parse_dims(raw: Any) -> tuple[int, ...]:
    text = "" if raw is None else str(raw).strip()
    if not text or text.lower() == "nan":
        return ()
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ()
    if not isinstance(value, list):
        return ()
    dims: list[int] = []
    for item in value:
        if isinstance(item, int) and item > 0:
            dims.append(int(item))
        elif isinstance(item, float) and item.is_integer() and item > 0:
            dims.append(int(item))
    return tuple(dims)


def parse_branch_idx(node_name: str) -> int | None:
    match = BRANCH_NAME_RE.match(str(node_name))
    return int(match.group("branch_idx")) if match else None


def partition_for_node(node_name: str, op_type: str) -> str:
    if op_type == "Constant":
        return "constant"
    if str(node_name).startswith("/bot_l/"):
        return "bottom"
    if parse_branch_idx(str(node_name)) is not None and op_type in {"Gather", "Reshape", "ReduceSum"}:
        return "embedding"
    return "tail"


def parse_producer_idx(producer_node: Any) -> int | None:
    if producer_node is None or pd.isna(producer_node):
        return None
    text = str(producer_node)
    if text in {"initializer", "graph_input"}:
        return None
    head = text.split(":", 1)[0]
    return int(head) if head.isdigit() else None


def _specs_from_rows(rows: pd.DataFrame) -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for row in rows.sort_values(["tensor_role", "port_idx"]).itertuples(index=False):
        specs.append(
            TensorSpec(
                tensor_name=str(row.tensor_name),
                dtype=str(row.dtype),
                dims=parse_dims(row.shape),
                tensor_role=str(row.tensor_role),
                port_idx=int(row.port_idx),
            )
        )
    return tuple(specs)


def load_op_shapes(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    required = {"node_idx", "node_name", "op_type", "tensor_role", "port_idx", "tensor_name", "dtype", "shape"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return frame


def build_graph(op_shapes_df: pd.DataFrame) -> dict[int, GraphNode]:
    unique_nodes = (
        op_shapes_df[["node_idx", "node_name", "op_type"]]
        .drop_duplicates()
        .sort_values("node_idx")
    )
    op_type_by_idx = {int(row.node_idx): str(row.op_type) for row in unique_nodes.itertuples(index=False)}

    predecessors: dict[int, set[int]] = {int(row.node_idx): set() for row in unique_nodes.itertuples(index=False)}
    successors: dict[int, set[int]] = {int(row.node_idx): set() for row in unique_nodes.itertuples(index=False)}
    input_rows = op_shapes_df[op_shapes_df["tensor_role"].astype(str).eq("input")]
    for row in input_rows.itertuples(index=False):
        pred_idx = parse_producer_idx(row.producer_node) if hasattr(row, "producer_node") else None
        if pred_idx is None or op_type_by_idx.get(pred_idx) == "Constant":
            continue
        predecessors[int(row.node_idx)].add(pred_idx)
        successors.setdefault(pred_idx, set()).add(int(row.node_idx))

    graph: dict[int, GraphNode] = {}
    grouped = {
        int(node_idx): group
        for node_idx, group in op_shapes_df.groupby("node_idx", sort=False)
    }
    for row in unique_nodes.itertuples(index=False):
        node_idx = int(row.node_idx)
        group = grouped[node_idx]
        inputs = _specs_from_rows(group[group["tensor_role"].astype(str).eq("input")])
        outputs = _specs_from_rows(group[group["tensor_role"].astype(str).eq("output")])
        branch_idx = parse_branch_idx(str(row.node_name))
        graph[node_idx] = GraphNode(
            node_idx=node_idx,
            node_name=str(row.node_name),
            op_type=str(row.op_type),
            predecessors=tuple(sorted(predecessors.get(node_idx, set()))),
            successors=tuple(sorted(successors.get(node_idx, set()))),
            partition=partition_for_node(str(row.node_name), str(row.op_type)),
            branch_idx=branch_idx,
            inputs=inputs,
            outputs=outputs,
        )
    return graph


def tensor_bytes(specs: tuple[TensorSpec, ...]) -> int:
    return int(sum(spec.byte_count for spec in specs))


def tensor_elements(specs: tuple[TensorSpec, ...]) -> int:
    return int(sum(spec.element_count for spec in specs))


def product(dims: tuple[int, ...] | list[int]) -> int:
    if not dims:
        return 0
    out = 1
    for dim in dims:
        out *= max(int(dim), 1)
    return out


def infer_gemm_mnk(node: GraphNode) -> tuple[float, float, float]:
    if not node.outputs:
        return 0.0, 0.0, 0.0
    out_dims = node.outputs[0].dims
    if not out_dims:
        return 0.0, 0.0, 0.0
    m = float(product(out_dims[:-1]) if len(out_dims) > 1 else 1)
    n = float(out_dims[-1])
    k = 0.0
    if node.inputs:
        a_dims = node.inputs[0].dims
        if a_dims:
            k = float(a_dims[-1])
    if len(node.inputs) > 1:
        b_dims = node.inputs[1].dims
        if len(b_dims) >= 2:
            if float(b_dims[0]) == n:
                k = float(b_dims[1])
            elif float(b_dims[-1]) == n:
                k = float(b_dims[-2])
    return m, k, n


def infer_reduction_axis(node: GraphNode) -> int:
    if not node.inputs or not node.outputs:
        return 0
    inp = node.inputs[0].dims
    out = node.outputs[0].dims
    if not inp:
        return 0
    if not out or len(out) < len(inp):
        return len(inp) - 1
    for idx, dim in enumerate(inp):
        if idx >= len(out) or int(out[idx]) != int(dim):
            return idx
    return len(inp) - 1

