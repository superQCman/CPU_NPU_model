from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from ..graph_static_features import graph_feature_table_from_op_shapes_frame
    from ..static_feature_contract import augment_static_feature_frame
except ImportError:
    from graph_static_features import graph_feature_table_from_op_shapes_frame  # type: ignore
    from static_feature_contract import augment_static_feature_frame  # type: ignore


MODEL_INPUT_51 = "model_input_51"
FULL_GRAPH_76 = "full_graph_76"

MODEL_INPUT_NODE_IDXS = {
    *[idx for branch in range(8) for idx in (branch * 4, branch * 4 + 2, branch * 4 + 3)],
    32,
    34,
    38,
    39,
    41,
    42,
    43,
    45,
    46,
    47,
    49,
    50,
    52,
    54,
    55,
    56,
    58,
    60,
    61,
    64,
    65,
    66,
    68,
    69,
    70,
    72,
    74,
}

DTYPE_BYTES = {
    "float16": 2.0,
    "float32": 4.0,
    "float64": 8.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "uint8": 1.0,
    "int16": 2.0,
    "uint16": 2.0,
    "int32": 4.0,
    "uint32": 4.0,
    "int64": 8.0,
    "uint64": 8.0,
    "bool": 1.0,
}


@dataclass(frozen=True)
class DlrmStaticConfig:
    batch_size: int
    num_indices_per_lookup: int
    arch_embedding_size: str
    arch_mlp_bot: str
    inter_threads: int
    num_threads: int
    arch_mlp_top: str = "1600-800-1"

    @property
    def combo(self) -> str:
        return f"bs{int(self.batch_size)}_nip{int(self.num_indices_per_lookup)}"

    @property
    def source_name(self) -> str:
        return f"static_t{int(self.num_threads)}_i{int(self.inter_threads)}"

    @property
    def case_id(self) -> str:
        return f"static_{int(self.num_threads)}_{int(self.inter_threads)}"


@dataclass(frozen=True)
class _TensorSpec:
    tensor_name: str
    dtype: str
    dims: tuple[int, ...]
    source: str = ""


@dataclass(frozen=True)
class _NodeSpec:
    node_idx: int
    node_name: str
    op_type: str
    inputs: tuple[_TensorSpec, ...]
    outputs: tuple[_TensorSpec, ...]


def _parse_dash_ints(value: str | int, *, expected: int | None = None, repeat_to: int | None = None) -> list[int]:
    text = str(value).strip()
    if not text:
        raise ValueError("empty numeric architecture string")
    parts = [part.strip() for part in text.split("-") if part.strip()]
    values = [int(part) for part in parts]
    if repeat_to is not None and len(values) == 1:
        values = values * int(repeat_to)
    if expected is not None and len(values) != int(expected):
        raise ValueError(f"expected {expected} values in {value!r}, got {len(values)}")
    if any(item <= 0 for item in values):
        raise ValueError(f"architecture values must be positive: {value!r}")
    return values


def _normalized_arch_embedding_size(value: str | int) -> str:
    return "-".join(str(item) for item in _parse_dash_ints(value, expected=8, repeat_to=8))


def _normalized_arch(value: str | int, *, expected: int) -> str:
    return "-".join(str(item) for item in _parse_dash_ints(value, expected=expected))


def _element_count(dims: tuple[int, ...] | list[int]) -> int:
    if not dims:
        return 1
    result = 1
    for dim in dims:
        result *= max(int(dim), 1)
    return int(result)


def _tensor_bytes(spec: _TensorSpec) -> float:
    return float(_element_count(spec.dims)) * float(DTYPE_BYTES.get(spec.dtype.lower(), 4.0))


def _shape_cell(dims: tuple[int, ...]) -> str:
    return "scalar" if not dims else str([int(dim) for dim in dims])


def _typed_shape_text(specs: list[_TensorSpec]) -> str:
    payload = [{spec.dtype: [int(dim) for dim in spec.dims]} for spec in specs]
    return repr(payload)


def _node_ref(node: _NodeSpec) -> str:
    return f"{int(node.node_idx)}:{node.node_name}:{node.op_type}"


def _scope_from_node_name(node_name: str) -> str:
    stripped = str(node_name or "").strip("/")
    if not stripped:
        return "root"
    return stripped.split("/", 1)[0]


def _make_node(
    node_idx: int,
    node_name: str,
    op_type: str,
    inputs: list[_TensorSpec] | None = None,
    outputs: list[_TensorSpec] | None = None,
) -> _NodeSpec:
    return _NodeSpec(
        node_idx=int(node_idx),
        node_name=str(node_name),
        op_type=str(op_type),
        inputs=tuple(inputs or []),
        outputs=tuple(outputs or []),
    )


def _build_node_specs(config: DlrmStaticConfig) -> list[_NodeSpec]:
    batch = int(config.batch_size)
    num_indices = int(config.num_indices_per_lookup)
    embedding_sizes = _parse_dash_ints(config.arch_embedding_size, expected=8, repeat_to=8)
    bot = _parse_dash_ints(config.arch_mlp_bot, expected=3)
    top = _parse_dash_ints(config.arch_mlp_top, expected=3)
    dense_dim, bot_hidden, bot_out = bot
    interaction_width = 9 * bot_out
    interaction_pair_count = 36
    nodes: list[_NodeSpec] = []

    for branch in range(8):
        base = branch * 4
        gather_out = f"__emb_l{branch}_gather_out"
        reshape_shape = f"__emb_l{branch}_reshape_shape"
        reshape_out = f"__emb_l{branch}_reshape_out"
        reduce_out = f"/emb_l.{branch}/Loop_output_0"
        lookup_count = batch * num_indices
        nodes.append(
            _make_node(
                base,
                f"/emb_l{branch}/Gather",
                "Gather",
                [
                    _TensorSpec(f"emb_l.{branch}.weight", "float32", (embedding_sizes[branch], bot_out), "initializer"),
                    _TensorSpec(f"indices_{branch}", "int64", (lookup_count,), "graph_input"),
                ],
                [_TensorSpec(gather_out, "float32", (lookup_count, bot_out))],
            )
        )
        nodes.append(
            _make_node(
                base + 1,
                f"/emb_l{branch}/ReshapeShape",
                "Constant",
                outputs=[_TensorSpec(reshape_shape, "int64", (3,))],
            )
        )
        nodes.append(
            _make_node(
                base + 2,
                f"/emb_l{branch}/Reshape",
                "Reshape",
                [
                    _TensorSpec(gather_out, "float32", (lookup_count, bot_out)),
                    _TensorSpec(reshape_shape, "int64", (3,)),
                ],
                [_TensorSpec(reshape_out, "float32", (batch, num_indices, bot_out))],
            )
        )
        nodes.append(
            _make_node(
                base + 3,
                f"/emb_l{branch}/ReduceSum",
                "ReduceSum",
                [_TensorSpec(reshape_out, "float32", (batch, num_indices, bot_out))],
                [_TensorSpec(reduce_out, "float32", (batch, bot_out))],
            )
        )

    nodes.extend(
        [
            _make_node(
                32,
                "/bot_l/bot_l.0/Gemm",
                "Gemm",
                [
                    _TensorSpec("dense_x", "float32", (batch, dense_dim), "graph_input"),
                    _TensorSpec("bot_l.0.weight", "float32", (bot_hidden, dense_dim), "initializer"),
                    _TensorSpec("bot_l.0.bias", "float32", (bot_hidden,), "initializer"),
                ],
                [_TensorSpec("/bot_l/bot_l.0/Gemm_output_0", "float32", (batch, bot_hidden))],
            ),
            _make_node(
                33,
                "/bot_l/bot_l.1/Relu",
                "Relu",
                [_TensorSpec("/bot_l/bot_l.0/Gemm_output_0", "float32", (batch, bot_hidden))],
                [_TensorSpec("/bot_l/bot_l.1/Relu_output_0", "float32", (batch, bot_hidden))],
            ),
            _make_node(
                34,
                "/bot_l/bot_l.2/Gemm",
                "Gemm",
                [
                    _TensorSpec("/bot_l/bot_l.1/Relu_output_0", "float32", (batch, bot_hidden)),
                    _TensorSpec("bot_l.2.weight", "float32", (bot_out, bot_hidden), "initializer"),
                    _TensorSpec("bot_l.2.bias", "float32", (bot_out,), "initializer"),
                ],
                [_TensorSpec("/bot_l/bot_l.2/Gemm_output_0", "float32", (batch, bot_out))],
            ),
            _make_node(
                35,
                "/bot_l/bot_l.3/Relu",
                "Relu",
                [_TensorSpec("/bot_l/bot_l.2/Gemm_output_0", "float32", (batch, bot_out))],
                [_TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out))],
            ),
            _make_node(
                36,
                "/Shape",
                "Shape",
                [_TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out))],
                [_TensorSpec("/Shape_output_0", "int64", (2,))],
            ),
            _make_node(37, "/Constant_8", "Constant", outputs=[_TensorSpec("/Constant_8_output_0", "int64", ())]),
            _make_node(
                38,
                "/Gather_8",
                "Gather",
                [
                    _TensorSpec("/Shape_output_0", "int64", (2,)),
                    _TensorSpec("/Constant_8_output_0", "int64", ()),
                ],
                [_TensorSpec("/Gather_8_output_0", "int64", ())],
            ),
            _make_node(
                39,
                "/Shape_1",
                "Shape",
                [_TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out))],
                [_TensorSpec("/Shape_1_output_0", "int64", (2,))],
            ),
            _make_node(40, "/Constant_9", "Constant", outputs=[_TensorSpec("/Constant_9_output_0", "int64", ())]),
            _make_node(
                41,
                "/Gather_9",
                "Gather",
                [
                    _TensorSpec("/Shape_1_output_0", "int64", (2,)),
                    _TensorSpec("/Constant_9_output_0", "int64", ()),
                ],
                [_TensorSpec("/Gather_9_output_0", "int64", ())],
            ),
            _make_node(
                42,
                "/Concat",
                "Concat",
                [
                    _TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out)),
                    *[
                        _TensorSpec(f"/emb_l.{branch}/Loop_output_0", "float32", (batch, bot_out))
                        for branch in range(8)
                    ],
                ],
                [_TensorSpec("/Concat_output_0", "float32", (batch, interaction_width))],
            ),
            _make_node(
                43,
                "/Unsqueeze",
                "Unsqueeze",
                [_TensorSpec("/Gather_8_output_0", "int64", ())],
                [_TensorSpec("/Unsqueeze_output_0", "int64", (1,))],
            ),
            _make_node(44, "/Constant_10", "Constant", outputs=[_TensorSpec("/Constant_10_output_0", "int64", (1,))]),
            _make_node(
                45,
                "/Unsqueeze_1",
                "Unsqueeze",
                [_TensorSpec("/Gather_9_output_0", "int64", ())],
                [_TensorSpec("/Unsqueeze_1_output_0", "int64", (1,))],
            ),
            _make_node(
                46,
                "/Concat_1",
                "Concat",
                [
                    _TensorSpec("/Unsqueeze_output_0", "int64", (1,)),
                    _TensorSpec("/Constant_10_output_0", "int64", (1,)),
                    _TensorSpec("/Unsqueeze_1_output_0", "int64", (1,)),
                ],
                [_TensorSpec("/Concat_1_output_0", "int64", (3,))],
            ),
            _make_node(
                47,
                "/Reshape",
                "Reshape",
                [
                    _TensorSpec("/Concat_output_0", "float32", (batch, interaction_width)),
                    _TensorSpec("/Concat_1_output_0", "int64", (3,)),
                ],
                [_TensorSpec("/Reshape_output_0", "float32", (batch, 9, bot_out))],
            ),
            _make_node(
                48,
                "/Transpose",
                "Transpose",
                [_TensorSpec("/Reshape_output_0", "float32", (batch, 9, bot_out))],
                [_TensorSpec("/Transpose_output_0", "float32", (batch, bot_out, 9))],
            ),
            _make_node(
                49,
                "/MatMul",
                "MatMul",
                [
                    _TensorSpec("/Reshape_output_0", "float32", (batch, 9, bot_out)),
                    _TensorSpec("/Transpose_output_0", "float32", (batch, bot_out, 9)),
                ],
                [_TensorSpec("/MatMul_output_0", "float32", (batch, 9, 9))],
            ),
            _make_node(
                50,
                "/Shape_2",
                "Shape",
                [_TensorSpec("/MatMul_output_0", "float32", (batch, 9, 9))],
                [_TensorSpec("/Shape_2_output_0", "int64", (3,))],
            ),
            _make_node(51, "/Constant_11", "Constant", outputs=[_TensorSpec("/Constant_11_output_0", "int64", (1,))]),
            _make_node(
                52,
                "/Gather_10",
                "Gather",
                [
                    _TensorSpec("/Shape_2_output_0", "int64", (3,)),
                    _TensorSpec("/Constant_11_output_0", "int64", (1,)),
                ],
                [_TensorSpec("/Gather_10_output_0", "int64", (1,))],
            ),
            _make_node(53, "/Constant_12", "Constant", outputs=[_TensorSpec("/Constant_12_output_0", "int64", (1,))]),
            _make_node(
                54,
                "/Gather_11",
                "Gather",
                [
                    _TensorSpec("/Shape_2_output_0", "int64", (3,)),
                    _TensorSpec("/Constant_12_output_0", "int64", (1,)),
                ],
                [_TensorSpec("/Gather_11_output_0", "int64", (1,))],
            ),
            _make_node(
                55,
                "/Transpose_1",
                "Transpose",
                [_TensorSpec("/MatMul_output_0", "float32", (batch, 9, 9))],
                [_TensorSpec("/Transpose_1_output_0", "float32", (9, 9, batch))],
            ),
            _make_node(
                56,
                "/Flatten",
                "Flatten",
                [_TensorSpec("/Transpose_1_output_0", "float32", (9, 9, batch))],
                [_TensorSpec("/Flatten_output_0", "float32", (81, batch))],
            ),
            _make_node(57, "/Constant_13", "Constant", outputs=[_TensorSpec("/Constant_13_output_0", "int64", (interaction_pair_count,))]),
            _make_node(
                58,
                "/Mul",
                "Mul",
                [
                    _TensorSpec("/Constant_13_output_0", "int64", (interaction_pair_count,)),
                    _TensorSpec("/Gather_11_output_0", "int64", (1,)),
                ],
                [_TensorSpec("/Mul_output_0", "int64", (interaction_pair_count,))],
            ),
            _make_node(59, "/Constant_14", "Constant", outputs=[_TensorSpec("/Constant_14_output_0", "int64", (interaction_pair_count,))]),
            _make_node(
                60,
                "/Add",
                "Add",
                [
                    _TensorSpec("/Constant_14_output_0", "int64", (interaction_pair_count,)),
                    _TensorSpec("/Mul_output_0", "int64", (interaction_pair_count,)),
                ],
                [_TensorSpec("/Add_output_0", "int64", (interaction_pair_count,))],
            ),
            _make_node(
                61,
                "/Gather_12",
                "Gather",
                [
                    _TensorSpec("/Flatten_output_0", "float32", (81, batch)),
                    _TensorSpec("/Add_output_0", "int64", (interaction_pair_count,)),
                ],
                [_TensorSpec("/Gather_12_output_0", "float32", (interaction_pair_count, batch))],
            ),
            _make_node(
                62,
                "/Shape_3",
                "Shape",
                [_TensorSpec("/Add_output_0", "int64", (interaction_pair_count,))],
                [_TensorSpec("/Shape_3_output_0", "int64", (1,))],
            ),
            _make_node(63, "/Constant_15", "Constant", outputs=[_TensorSpec("/Constant_15_output_0", "int64", (1,))]),
            _make_node(
                64,
                "/Concat_2",
                "Concat",
                [
                    _TensorSpec("/Constant_15_output_0", "int64", (1,)),
                    _TensorSpec("/Gather_10_output_0", "int64", (1,)),
                ],
                [_TensorSpec("/Concat_2_output_0", "int64", (2,))],
            ),
            _make_node(
                65,
                "/Reshape_1",
                "Reshape",
                [
                    _TensorSpec("/Gather_12_output_0", "float32", (interaction_pair_count, batch)),
                    _TensorSpec("/Concat_2_output_0", "int64", (2,)),
                ],
                [_TensorSpec("/Reshape_1_output_0", "float32", (interaction_pair_count, batch))],
            ),
            _make_node(
                66,
                "/Transpose_2",
                "Transpose",
                [_TensorSpec("/Reshape_1_output_0", "float32", (interaction_pair_count, batch))],
                [_TensorSpec("/Transpose_2_output_0", "float32", (batch, interaction_pair_count))],
            ),
            _make_node(
                67,
                "/Concat_3",
                "Concat",
                [
                    _TensorSpec("/Gather_10_output_0", "int64", (1,)),
                    _TensorSpec("/Shape_3_output_0", "int64", (1,)),
                ],
                [_TensorSpec("/Concat_3_output_0", "int64", (2,))],
            ),
            _make_node(
                68,
                "/Reshape_2",
                "Reshape",
                [
                    _TensorSpec("/Transpose_2_output_0", "float32", (batch, interaction_pair_count)),
                    _TensorSpec("/Concat_3_output_0", "int64", (2,)),
                ],
                [_TensorSpec("/Reshape_2_output_0", "float32", (batch, interaction_pair_count))],
            ),
            _make_node(
                69,
                "/Concat_4",
                "Concat",
                [
                    _TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out)),
                    _TensorSpec("/Reshape_2_output_0", "float32", (batch, interaction_pair_count)),
                ],
                [_TensorSpec("/Concat_4_output_0", "float32", (batch, bot_out + interaction_pair_count))],
            ),
        ]
    )

    top_input = bot_out + interaction_pair_count
    top_specs = [
        (70, "/top_l/top_l.0/Gemm", "/Concat_4_output_0", top_input, top[0], "/top_l/top_l.0/Gemm_output_0"),
        (72, "/top_l/top_l.2/Gemm", "/top_l/top_l.1/Relu_output_0", top[0], top[1], "/top_l/top_l.2/Gemm_output_0"),
        (74, "/top_l/top_l.4/Gemm", "/top_l/top_l.3/Relu_output_0", top[1], top[2], "/top_l/top_l.4/Gemm_output_0"),
    ]
    for idx, name, input_tensor, input_dim, output_dim, output_tensor in top_specs:
        nodes.append(
            _make_node(
                idx,
                name,
                "Gemm",
                [
                    _TensorSpec(input_tensor, "float32", (batch, input_dim)),
                    _TensorSpec(f"top_l.{idx - 70}.weight", "float32", (output_dim, input_dim), "initializer"),
                    _TensorSpec(f"top_l.{idx - 70}.bias", "float32", (output_dim,), "initializer"),
                ],
                [_TensorSpec(output_tensor, "float32", (batch, output_dim))],
            )
        )
        if idx != 74:
            relu_idx = idx + 1
            relu_output = f"/top_l/top_l.{relu_idx - 70}/Relu_output_0"
            nodes.append(
                _make_node(
                    relu_idx,
                    f"/top_l/top_l.{relu_idx - 70}/Relu",
                    "Relu",
                    [_TensorSpec(output_tensor, "float32", (batch, output_dim))],
                    [_TensorSpec(relu_output, "float32", (batch, output_dim))],
                )
            )
    nodes.append(
        _make_node(
            75,
            "/top_l/top_l.5/Sigmoid",
            "Sigmoid",
            [_TensorSpec("/top_l/top_l.4/Gemm_output_0", "float32", (batch, top[2]))],
            [_TensorSpec("pred", "float32", (batch, top[2]))],
        )
    )
    return sorted(nodes, key=lambda item: item.node_idx)


def build_static_op_shape_table(config: DlrmStaticConfig) -> pd.DataFrame:
    config = DlrmStaticConfig(
        batch_size=int(config.batch_size),
        num_indices_per_lookup=int(config.num_indices_per_lookup),
        arch_embedding_size=_normalized_arch_embedding_size(config.arch_embedding_size),
        arch_mlp_bot=_normalized_arch(config.arch_mlp_bot, expected=3),
        inter_threads=int(config.inter_threads),
        num_threads=int(config.num_threads),
        arch_mlp_top=_normalized_arch(config.arch_mlp_top, expected=3),
    )
    nodes = _build_node_specs(config)
    producer_by_tensor: dict[str, str] = {}
    consumers_by_tensor: dict[str, list[str]] = {}
    source_by_tensor: dict[str, str] = {}
    graph_outputs = {"pred"}
    for node in nodes:
        for output in node.outputs:
            producer_by_tensor[output.tensor_name] = _node_ref(node)
        for input_spec in node.inputs:
            if input_spec.source:
                source_by_tensor.setdefault(input_spec.tensor_name, input_spec.source)
            consumers_by_tensor.setdefault(input_spec.tensor_name, []).append(_node_ref(node))

    rows: list[dict[str, Any]] = []
    for node in nodes:
        for port_idx, spec in enumerate(node.inputs):
            rows.append(
                {
                    "node_idx": int(node.node_idx),
                    "node_name": node.node_name,
                    "op_type": node.op_type,
                    "tensor_role": "input",
                    "port_idx": int(port_idx),
                    "tensor_name": spec.tensor_name,
                    "dtype": spec.dtype,
                    "shape": _shape_cell(spec.dims),
                    "is_dynamic": False,
                    "shape_source": "static_config",
                    "producer_node": producer_by_tensor.get(spec.tensor_name, source_by_tensor.get(spec.tensor_name, "graph_input")),
                    "consumer_nodes": ", ".join(consumers_by_tensor.get(spec.tensor_name, [])),
                }
            )
        for port_idx, spec in enumerate(node.outputs):
            consumers = list(consumers_by_tensor.get(spec.tensor_name, []))
            if spec.tensor_name in graph_outputs:
                consumers.append("graph_output")
            rows.append(
                {
                    "node_idx": int(node.node_idx),
                    "node_name": node.node_name,
                    "op_type": node.op_type,
                    "tensor_role": "output",
                    "port_idx": int(port_idx),
                    "tensor_name": spec.tensor_name,
                    "dtype": spec.dtype,
                    "shape": _shape_cell(spec.dims),
                    "is_dynamic": False,
                    "shape_source": "static_config",
                    "producer_node": _node_ref(node),
                    "consumer_nodes": ", ".join(consumers),
                }
            )
    return pd.DataFrame(rows)


def _is_parameter_input(row: pd.Series) -> bool:
    producer = str(row.get("producer_node", "")).strip()
    return producer == "initializer" or producer.endswith(":Constant")


def _specs_from_shape_rows(rows: pd.DataFrame) -> list[_TensorSpec]:
    specs: list[_TensorSpec] = []
    for _, row in rows.sort_values("port_idx").iterrows():
        shape_text = str(row.get("shape", "")).strip()
        if shape_text == "scalar":
            dims: tuple[int, ...] = ()
        else:
            dims = tuple(int(part.strip()) for part in shape_text.strip("[]").split(",") if part.strip())
        specs.append(_TensorSpec(str(row["tensor_name"]), str(row["dtype"]), dims))
    return specs


def _infer_reduction_dims(input_dims: tuple[int, ...], output_dims: tuple[int, ...]) -> list[int]:
    if not input_dims or input_dims == output_dims:
        return []
    out_idx = 0
    reduced: list[int] = []
    for dim in input_dims:
        if out_idx < len(output_dims) and output_dims[out_idx] == dim:
            out_idx += 1
        else:
            reduced.append(int(dim))
    if out_idx == len(output_dims):
        return reduced
    return [int(dim) for idx, dim in enumerate(input_dims) if idx >= len(output_dims) or output_dims[idx] != dim]


def _infer_gemm_mnk(input_specs: list[_TensorSpec], output_specs: list[_TensorSpec]) -> tuple[float, float, float]:
    if not input_specs or not output_specs:
        return 0.0, 0.0, 0.0
    a_dims = tuple(int(dim) for dim in input_specs[0].dims)
    b_dims = tuple(int(dim) for dim in input_specs[1].dims) if len(input_specs) > 1 else ()
    c_dims = tuple(int(dim) for dim in output_specs[0].dims)
    if not c_dims:
        return 0.0, 0.0, 0.0
    n_dim = float(c_dims[-1])
    m_dim = float(_element_count(c_dims[:-1]) if len(c_dims) > 1 else 1)
    k_dim = float(a_dims[-1]) if a_dims else 0.0
    if len(b_dims) >= 2:
        if float(b_dims[0]) == n_dim:
            k_dim = float(b_dims[1])
        elif float(b_dims[-1]) == n_dim:
            k_dim = float(b_dims[-2])
    return m_dim, n_dim, k_dim


def _shape_feature_values(specs: list[_TensorSpec], prefix: str) -> dict[str, float]:
    ranks = [float(len(spec.dims)) for spec in specs]
    elems = [float(_element_count(spec.dims)) for spec in specs]
    bytes_values = [_tensor_bytes(spec) for spec in specs]
    return {
        f"feat_{prefix}_tensor_count": float(len(specs)),
        f"feat_{prefix}_rank_sum": float(sum(ranks)),
        f"feat_{prefix}_rank_max": float(max(ranks) if ranks else 0.0),
        f"feat_{prefix}_elements_sum": float(sum(elems)),
        f"feat_{prefix}_elements_max": float(max(elems) if elems else 0.0),
        f"feat_{prefix}_bytes_sum": float(sum(bytes_values)),
        f"feat_{prefix}_dynamic_dims": 0.0,
    }


def _feature_rows_from_op_shapes(config: DlrmStaticConfig, op_shapes: pd.DataFrame) -> pd.DataFrame:
    graph_table = graph_feature_table_from_op_shapes_frame(op_shapes)
    rows: list[dict[str, Any]] = []
    for (node_idx, node_name, op_type), group in op_shapes.groupby(["node_idx", "node_name", "op_type"], sort=True):
        input_rows = group.loc[group["tensor_role"].astype(str).eq("input")].copy()
        output_rows = group.loc[group["tensor_role"].astype(str).eq("output")].copy()
        input_specs = _specs_from_shape_rows(input_rows)
        output_specs = _specs_from_shape_rows(output_rows)
        output_size = float(sum(_tensor_bytes(spec) for spec in output_specs))
        activation_size = float(
            sum(
                _tensor_bytes(spec)
                for spec, (_, input_row) in zip(input_specs, input_rows.sort_values("port_idx").iterrows(), strict=False)
                if not _is_parameter_input(input_row)
            )
        )
        parameter_size = float(
            sum(
                _tensor_bytes(spec)
                for spec, (_, input_row) in zip(input_specs, input_rows.sort_values("port_idx").iterrows(), strict=False)
                if _is_parameter_input(input_row)
            )
        )
        output_elements = float(sum(_element_count(spec.dims) for spec in output_specs))
        activation_elements = float(
            sum(
                _element_count(spec.dims)
                for spec, (_, input_row) in zip(input_specs, input_rows.sort_values("port_idx").iterrows(), strict=False)
                if not _is_parameter_input(input_row)
            )
        )
        lookup_count = 0.0
        if str(op_type) == "Gather" and len(input_specs) >= 2:
            lookup_count = float(_element_count(input_specs[1].dims))

        reduction_dims: list[int] = []
        if str(op_type) == "ReduceSum" and input_specs and output_specs:
            reduction_dims = _infer_reduction_dims(input_specs[0].dims, output_specs[0].dims)
        reduction_axes_product = float(np.prod(reduction_dims, dtype=float)) if reduction_dims else 0.0
        reduction_work_items = 0.0
        if reduction_axes_product > 0.0 and output_elements > 0.0:
            reduction_work_items = float(output_elements * max(reduction_axes_product - 1.0, 0.0))

        gemm_m, gemm_n, gemm_k = _infer_gemm_mnk(input_specs, output_specs)
        gemm_mac = float(gemm_m * gemm_n * gemm_k) if str(op_type) in {"Gemm", "MatMul"} else 0.0
        io_bytes = float(output_size + activation_size + parameter_size)
        row_uid = (
            f"{config.source_name}::{config.combo}::opidx:{int(node_idx)}::"
            f"bs{int(config.batch_size)}_nip{int(config.num_indices_per_lookup)}"
        )
        record: dict[str, Any] = {
            "row_uid": row_uid,
            "case_id": config.case_id,
            "source_name": config.source_name,
            "source_mode": "static_config",
            "hardware_id": "static_config",
            "hardware_family": "cpu",
            "hardware_profile_name": "",
            "backend_family": "cpu",
            "combo": config.combo,
            "sample_group": config.combo,
            "split": "static_inference",
            "op_idx": int(node_idx),
            "node_idx": int(node_idx),
            "node_name": str(node_name),
            "trace_op_name": "",
            "feature_dialect_observed": "static_config_v1",
            "has_cpu_profile": 0,
            "input_type_shape": _typed_shape_text(input_specs),
            "output_type_shape": _typed_shape_text(output_specs),
            "op_attrs_json": np.nan,
            "attr_axis": np.nan,
            "attr_perm_json": np.nan,
            "attr_axes_json": np.nan,
            "attr_keepdims": np.nan,
            "gather_unique_idx_count": np.nan,
            "gather_working_set_eff_bytes": np.nan,
            "op_type": str(op_type),
            "node_scope": _scope_from_node_name(str(node_name)),
            "node_name_normalized": str(node_name).strip(),
            "arch_embedding_size": _normalized_arch_embedding_size(config.arch_embedding_size),
            "arch_mlp_bot": _normalized_arch(config.arch_mlp_bot, expected=3),
            "arch_mlp_top": _normalized_arch(config.arch_mlp_top, expected=3),
            "batch_size": int(config.batch_size),
            "num_indices_per_lookup": int(config.num_indices_per_lookup),
            "num_threads": int(config.num_threads),
            "inter_threads": int(config.inter_threads),
            "output_size": output_size,
            "activation_size": activation_size,
            "parameter_size": parameter_size,
            "feat_io_bytes_sum": io_bytes,
            "feat_output_input_bytes_ratio": float(output_size / max(activation_size + parameter_size, 1.0)),
            "feat_lookup_count": lookup_count,
            "feat_output_elements_per_lookup": float(output_elements / max(lookup_count, 1.0)) if lookup_count > 0.0 else 0.0,
            "feat_output_elements_per_batch": float(output_elements / max(float(config.batch_size), 1.0)),
            "feat_activation_elements_per_batch": float(activation_elements / max(float(config.batch_size), 1.0)),
            "feat_reduction_axes_count": float(len(reduction_dims)) if str(op_type) == "ReduceSum" else 0.0,
            "feat_reduction_axes_product": reduction_axes_product if str(op_type) == "ReduceSum" else 0.0,
            "feat_reduction_input_rank": float(len(input_specs[0].dims)) if str(op_type) == "ReduceSum" and input_specs else 0.0,
            "feat_reduction_output_rank": float(len(output_specs[0].dims)) if str(op_type) == "ReduceSum" and output_specs else 0.0,
            "feat_reduction_work_items": reduction_work_items if str(op_type) == "ReduceSum" else 0.0,
            "feat_gemm_m": gemm_m if str(op_type) in {"Gemm", "MatMul"} else 0.0,
            "feat_gemm_n": gemm_n if str(op_type) in {"Gemm", "MatMul"} else 0.0,
            "feat_gemm_k": gemm_k if str(op_type) in {"Gemm", "MatMul"} else 0.0,
            "feat_gemm_mac_count": gemm_mac,
            "feat_gemm_bytes_per_mac": float(io_bytes / max(gemm_mac, 1.0)) if gemm_mac > 0.0 else 0.0,
            "feat_memory_ops": 0.0,
            "feat_working_set_bytes": io_bytes,
            "feat_threads_effective": float(max(int(config.num_threads), 1)),
            "feat_output_elements": output_elements,
            "feat_activation_elements": activation_elements,
            "feat_parameter_elements": float(parameter_size / 4.0),
            "feat_activation_per_output_element": float(activation_elements / max(output_elements, 1.0)),
            "feat_reduction_factor": float(activation_elements / max(output_elements, 1.0)) if str(op_type) == "ReduceSum" else 0.0,
        }
        record.update(_shape_feature_values(input_specs, "input"))
        record.update(_shape_feature_values(output_specs, "output"))
        rows.append(record)
    frame = pd.DataFrame(rows).sort_values("op_idx").reset_index(drop=True)
    return augment_static_feature_frame(frame, graph_feature_table=graph_table)


def build_static_feature_frame(config: DlrmStaticConfig, graph_scope: str = MODEL_INPUT_51) -> pd.DataFrame:
    normalized = DlrmStaticConfig(
        batch_size=int(config.batch_size),
        num_indices_per_lookup=int(config.num_indices_per_lookup),
        arch_embedding_size=_normalized_arch_embedding_size(config.arch_embedding_size),
        arch_mlp_bot=_normalized_arch(config.arch_mlp_bot, expected=3),
        inter_threads=int(config.inter_threads),
        num_threads=int(config.num_threads),
        arch_mlp_top=_normalized_arch(config.arch_mlp_top, expected=3),
    )
    op_shapes = build_static_op_shape_table(normalized)
    frame = _feature_rows_from_op_shapes(normalized, op_shapes)
    scope = str(graph_scope)
    if scope == MODEL_INPUT_51:
        frame = frame.loc[frame["op_idx"].astype(int).isin(MODEL_INPUT_NODE_IDXS)].copy()
    elif scope == FULL_GRAPH_76:
        frame = frame.copy()
    else:
        raise ValueError(f"Unsupported graph_scope: {graph_scope}")
    return frame.sort_values("op_idx").reset_index(drop=True)
