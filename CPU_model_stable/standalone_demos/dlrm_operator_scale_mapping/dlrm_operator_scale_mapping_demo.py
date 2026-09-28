from __future__ import annotations

"""独立 DLRM 上层参数到算子规模映射 demo。

这个脚本是一个完全独立的交付 demo：

- 不 import `calibrator/`、`cpu_perf_model/` 或项目里的其他 Python 脚本。
- 不依赖 pandas / numpy，只使用 Python 标准库。
- 不加载模型 bundle，不读取 profiling 数据，不预测 latency。
- 只演示一件事：DLRM 上层负载参数如何展开成 76 个静态算子节点，
  以及这些节点如何得到性能模型常用的规模字段。

为什么还要保留 76 个节点？

训练后推理时，系统内部会先构造完整静态图 `full_graph_76`，再从中筛选
当前主线模型实际读取的 51 个节点 `model_input_51`。因此 demo 同时输出：

- `dlrm_static_op_shapes_full_graph_76.csv`：完整 76 节点的输入/输出 shape。
- `dlrm_static_features_full_graph_76.csv`：完整 76 节点的规模特征。
- `dlrm_static_model_input_51.csv`：实际进入性能模型的 51 节点规模特征。
- `dlrm_operator_scale_mapping_summary.csv/json`：适合人工审阅的解释表。

这份 demo 的逻辑和主线静态映射口径保持一致，但代码是自包含的，便于单独交付。
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

FEATURE_COLUMNS = [
    "row_uid",
    "case_id",
    "source_name",
    "source_mode",
    "hardware_id",
    "hardware_family",
    "hardware_profile_name",
    "backend_family",
    "combo",
    "sample_group",
    "split",
    "op_idx",
    "node_idx",
    "node_name",
    "op_type",
    "stage",
    "in_model_input_51",
    "feature_dialect_observed",
    "has_cpu_profile",
    "input_type_shape",
    "output_type_shape",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
    "batch_size",
    "num_indices_per_lookup",
    "num_threads",
    "inter_threads",
    "output_size",
    "activation_size",
    "parameter_size",
    "feat_io_bytes_sum",
    "feat_output_input_bytes_ratio",
    "feat_lookup_count",
    "feat_output_elements_per_lookup",
    "feat_output_elements_per_batch",
    "feat_activation_elements_per_batch",
    "feat_reduction_axes_count",
    "feat_reduction_axes_product",
    "feat_reduction_input_rank",
    "feat_reduction_output_rank",
    "feat_reduction_work_items",
    "feat_gemm_m",
    "feat_gemm_n",
    "feat_gemm_k",
    "feat_gemm_mac_count",
    "feat_gemm_bytes_per_mac",
    "feat_memory_ops",
    "feat_working_set_bytes",
    "feat_threads_effective",
    "feat_output_elements",
    "feat_activation_elements",
    "feat_parameter_elements",
    "feat_activation_per_output_element",
    "feat_reduction_factor",
    "scale_rule",
]

SHAPE_COLUMNS = [
    "node_idx",
    "node_name",
    "op_type",
    "tensor_role",
    "port_idx",
    "tensor_name",
    "dtype",
    "shape",
    "is_dynamic",
    "shape_source",
    "producer_node",
    "consumer_nodes",
]

SUMMARY_COLUMNS = [
    "node_idx",
    "node_name",
    "op_type",
    "stage",
    "in_model_input_51",
    "shape_source",
    "has_cpu_profile",
    "input_type_shape",
    "output_type_shape",
    "batch_size",
    "num_indices_per_lookup",
    "num_threads",
    "inter_threads",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
    "output_size",
    "activation_size",
    "parameter_size",
    "feat_io_bytes_sum",
    "feat_lookup_count",
    "feat_output_elements",
    "feat_activation_elements",
    "feat_parameter_elements",
    "feat_working_set_bytes",
    "feat_threads_effective",
    "feat_gemm_m",
    "feat_gemm_n",
    "feat_gemm_k",
    "feat_gemm_mac_count",
    "feat_reduction_axes_count",
    "feat_reduction_axes_product",
    "feat_reduction_work_items",
    "scale_rule",
]

FIELD_MEANINGS = {
    "output_size": "输出 tensor 的字节数，由静态 shape 和 dtype 计算得到。",
    "activation_size": "非参数输入的字节数，包含图输入和上游算子输出。",
    "parameter_size": "参数或 Constant 输入的字节数，包含权重和 shape 描述 tensor。",
    "feat_io_bytes_sum": "总 I/O 字节数，等于 output_size + activation_size + parameter_size。",
    "feat_lookup_count": "embedding Gather 的查表次数，等于 batch_size * num_indices_per_lookup。",
    "feat_gemm_m/n/k": "Gemm/MatMul 的矩阵规模，由输入输出 shape 推导得到。",
    "feat_gemm_mac_count": "Gemm/MatMul 的乘加次数，等于 M * N * K。",
    "feat_reduction_work_items": "ReduceSum 的近似工作量，等于 output_elements * (reduction_axes_product - 1)。",
    "feat_working_set_bytes": "静态 working set 代理量，当前等于 feat_io_bytes_sum。",
    "has_cpu_profile": "本 demo 固定为 0，表示没有读取 CPU profiling 产物。",
    "shape_source": "本 demo 固定为 static_config，表示 shape 由 DLRM 上层参数静态生成。",
}


@dataclass(frozen=True)
class StaticConfig:
    """DLRM 上层配置。

    这些参数是用户在静态配置推理时能直接提供的参数，不包含 profiling 数据：

    - batch_size：一次 DLRM 推理的 batch 大小。
    - num_indices_per_lookup：每个 embedding table 每个样本平均查多少个 index。
    - arch_embedding_size：8 个 embedding table 的行数；如果只给一个数，会复制成 8 个表。
    - arch_mlp_bot：bottom MLP 的三段维度，例如 400-300-200。
    - arch_mlp_top：top MLP 的三段维度，例如 1600-800-1。
    - inter_threads / num_threads：线程配置，写入规模特征和调度口径。
    """

    batch_size: int
    num_indices_per_lookup: int
    arch_embedding_size: str
    arch_mlp_bot: str
    arch_mlp_top: str
    inter_threads: int
    num_threads: int

    @property
    def combo(self) -> str:
        return f"bs{self.batch_size}_nip{self.num_indices_per_lookup}"

    @property
    def source_name(self) -> str:
        return f"static_t{self.num_threads}_i{self.inter_threads}"

    @property
    def case_id(self) -> str:
        return f"static_{self.num_threads}_{self.inter_threads}"


@dataclass(frozen=True)
class TensorSpec:
    tensor_name: str
    dtype: str
    dims: tuple[int, ...]
    source: str = ""


@dataclass(frozen=True)
class NodeSpec:
    node_idx: int
    node_name: str
    op_type: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


def _parse_dash_ints(value: str | int, *, expected: int | None = None, repeat_to: int | None = None) -> list[int]:
    """解析形如 `400-300-200` 的维度字符串。

    `arch_embedding_size` 支持两种写法：
    - `25600`：表示 8 个 embedding table 都是 25600 行。
    - `25600-32000-...`：显式给出 8 个 table 的行数。
    """

    text = str(value).strip()
    if not text:
        raise ValueError("numeric architecture string is empty")
    values = [int(part.strip()) for part in text.split("-")]
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


def _element_count(dims: tuple[int, ...]) -> int:
    result = 1
    for dim in dims:
        result *= max(int(dim), 1)
    return result


def _tensor_bytes(spec: TensorSpec) -> float:
    return float(_element_count(spec.dims)) * float(DTYPE_BYTES.get(spec.dtype.lower(), 4.0))


def _shape_cell(dims: tuple[int, ...]) -> str:
    return "scalar" if not dims else str([int(dim) for dim in dims])


def _typed_shape_text(specs: list[TensorSpec]) -> str:
    return repr([{spec.dtype: [int(dim) for dim in spec.dims]} for spec in specs])


def _node_ref(node: NodeSpec) -> str:
    return f"{node.node_idx}:{node.node_name}:{node.op_type}"


def _make_node(
    node_idx: int,
    node_name: str,
    op_type: str,
    inputs: list[TensorSpec] | None = None,
    outputs: list[TensorSpec] | None = None,
) -> NodeSpec:
    return NodeSpec(
        node_idx=int(node_idx),
        node_name=str(node_name),
        op_type=str(op_type),
        inputs=tuple(inputs or []),
        outputs=tuple(outputs or []),
    )


def build_full_graph_76_nodes(config: StaticConfig) -> list[NodeSpec]:
    """显式构造完整 76 节点 DLRM 静态图。

    这个函数是 demo 的核心。它把 DLRM 上层参数展开成固定结构的静态图：

    1. 8 个 embedding 分支，每个分支 4 个节点：
       Gather -> Constant(reshape shape) -> Reshape -> ReduceSum。
    2. bottom MLP：Gemm/Relu/Gemm/Relu。
    3. interaction：把 1 个 dense 向量和 8 个 embedding 向量拼成 9 路输入，
       形成 9x9 interaction matrix，并保留 36 个 pair feature。
    4. top MLP：三层 Gemm，中间带 Relu，最后 Sigmoid。

    输出节点编号固定为 0..75，所以 CSV 能明确看到“所有算子都写全了”。
    """

    batch = int(config.batch_size)
    num_indices = int(config.num_indices_per_lookup)
    embedding_sizes = _parse_dash_ints(config.arch_embedding_size, expected=8, repeat_to=8)
    dense_dim, bot_hidden, bot_out = _parse_dash_ints(config.arch_mlp_bot, expected=3)
    top = _parse_dash_ints(config.arch_mlp_top, expected=3)

    interaction_width = 9 * bot_out
    interaction_pair_count = 36
    nodes: list[NodeSpec] = []

    # 8 个 embedding table 的结构完全一致，只是 table 行数可能不同。
    # Gather 的 indices 是扁平的一维向量，长度 = batch_size * num_indices_per_lookup。
    for branch in range(8):
        base = branch * 4
        lookup_count = batch * num_indices
        gather_out = f"__emb_l{branch}_gather_out"
        reshape_shape = f"__emb_l{branch}_reshape_shape"
        reshape_out = f"__emb_l{branch}_reshape_out"
        reduce_out = f"/emb_l.{branch}/Loop_output_0"
        nodes.extend(
            [
                _make_node(
                    base,
                    f"/emb_l{branch}/Gather",
                    "Gather",
                    [
                        TensorSpec(f"emb_l.{branch}.weight", "float32", (embedding_sizes[branch], bot_out), "initializer"),
                        TensorSpec(f"indices_{branch}", "int64", (lookup_count,), "graph_input"),
                    ],
                    [TensorSpec(gather_out, "float32", (lookup_count, bot_out))],
                ),
                _make_node(
                    base + 1,
                    f"/emb_l{branch}/ReshapeShape",
                    "Constant",
                    outputs=[TensorSpec(reshape_shape, "int64", (3,))],
                ),
                _make_node(
                    base + 2,
                    f"/emb_l{branch}/Reshape",
                    "Reshape",
                    [
                        TensorSpec(gather_out, "float32", (lookup_count, bot_out)),
                        TensorSpec(reshape_shape, "int64", (3,)),
                    ],
                    [TensorSpec(reshape_out, "float32", (batch, num_indices, bot_out))],
                ),
                _make_node(
                    base + 3,
                    f"/emb_l{branch}/ReduceSum",
                    "ReduceSum",
                    [TensorSpec(reshape_out, "float32", (batch, num_indices, bot_out))],
                    [TensorSpec(reduce_out, "float32", (batch, bot_out))],
                ),
            ]
        )

    # bottom MLP。arch_mlp_bot=400-300-200 表示：
    # dense input 400 -> hidden 300 -> output 200。
    nodes.extend(
        [
            _make_node(
                32,
                "/bot_l/bot_l.0/Gemm",
                "Gemm",
                [
                    TensorSpec("dense_x", "float32", (batch, dense_dim), "graph_input"),
                    TensorSpec("bot_l.0.weight", "float32", (bot_hidden, dense_dim), "initializer"),
                    TensorSpec("bot_l.0.bias", "float32", (bot_hidden,), "initializer"),
                ],
                [TensorSpec("/bot_l/bot_l.0/Gemm_output_0", "float32", (batch, bot_hidden))],
            ),
            _make_node(
                33,
                "/bot_l/bot_l.1/Relu",
                "Relu",
                [TensorSpec("/bot_l/bot_l.0/Gemm_output_0", "float32", (batch, bot_hidden))],
                [TensorSpec("/bot_l/bot_l.1/Relu_output_0", "float32", (batch, bot_hidden))],
            ),
            _make_node(
                34,
                "/bot_l/bot_l.2/Gemm",
                "Gemm",
                [
                    TensorSpec("/bot_l/bot_l.1/Relu_output_0", "float32", (batch, bot_hidden)),
                    TensorSpec("bot_l.2.weight", "float32", (bot_out, bot_hidden), "initializer"),
                    TensorSpec("bot_l.2.bias", "float32", (bot_out,), "initializer"),
                ],
                [TensorSpec("/bot_l/bot_l.2/Gemm_output_0", "float32", (batch, bot_out))],
            ),
            _make_node(
                35,
                "/bot_l/bot_l.3/Relu",
                "Relu",
                [TensorSpec("/bot_l/bot_l.2/Gemm_output_0", "float32", (batch, bot_out))],
                [TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out))],
            ),
        ]
    )

    # interaction 区域。这里的 9 来自：1 个 bottom MLP dense 输出 + 8 个 embedding 输出。
    # 9 路向量先拼接，再 reshape 成 [batch, 9, bot_out]，随后 MatMul 生成 [batch, 9, 9]。
    # 36 是 9 个向量两两交互的 pair 数：9 * (9 - 1) / 2 = 36。
    nodes.extend(
        [
            _make_node(36, "/Shape", "Shape", [TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out))], [TensorSpec("/Shape_output_0", "int64", (2,))]),
            _make_node(37, "/Constant_8", "Constant", outputs=[TensorSpec("/Constant_8_output_0", "int64", ())]),
            _make_node(38, "/Gather_8", "Gather", [TensorSpec("/Shape_output_0", "int64", (2,)), TensorSpec("/Constant_8_output_0", "int64", ())], [TensorSpec("/Gather_8_output_0", "int64", ())]),
            _make_node(39, "/Shape_1", "Shape", [TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out))], [TensorSpec("/Shape_1_output_0", "int64", (2,))]),
            _make_node(40, "/Constant_9", "Constant", outputs=[TensorSpec("/Constant_9_output_0", "int64", ())]),
            _make_node(41, "/Gather_9", "Gather", [TensorSpec("/Shape_1_output_0", "int64", (2,)), TensorSpec("/Constant_9_output_0", "int64", ())], [TensorSpec("/Gather_9_output_0", "int64", ())]),
            _make_node(
                42,
                "/Concat",
                "Concat",
                [
                    TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out)),
                    *[TensorSpec(f"/emb_l.{branch}/Loop_output_0", "float32", (batch, bot_out)) for branch in range(8)],
                ],
                [TensorSpec("/Concat_output_0", "float32", (batch, interaction_width))],
            ),
            _make_node(43, "/Unsqueeze", "Unsqueeze", [TensorSpec("/Gather_8_output_0", "int64", ())], [TensorSpec("/Unsqueeze_output_0", "int64", (1,))]),
            _make_node(44, "/Constant_10", "Constant", outputs=[TensorSpec("/Constant_10_output_0", "int64", (1,))]),
            _make_node(45, "/Unsqueeze_1", "Unsqueeze", [TensorSpec("/Gather_9_output_0", "int64", ())], [TensorSpec("/Unsqueeze_1_output_0", "int64", (1,))]),
            _make_node(46, "/Concat_1", "Concat", [TensorSpec("/Unsqueeze_output_0", "int64", (1,)), TensorSpec("/Constant_10_output_0", "int64", (1,)), TensorSpec("/Unsqueeze_1_output_0", "int64", (1,))], [TensorSpec("/Concat_1_output_0", "int64", (3,))]),
            _make_node(47, "/Reshape", "Reshape", [TensorSpec("/Concat_output_0", "float32", (batch, interaction_width)), TensorSpec("/Concat_1_output_0", "int64", (3,))], [TensorSpec("/Reshape_output_0", "float32", (batch, 9, bot_out))]),
            _make_node(48, "/Transpose", "Transpose", [TensorSpec("/Reshape_output_0", "float32", (batch, 9, bot_out))], [TensorSpec("/Transpose_output_0", "float32", (batch, bot_out, 9))]),
            _make_node(49, "/MatMul", "MatMul", [TensorSpec("/Reshape_output_0", "float32", (batch, 9, bot_out)), TensorSpec("/Transpose_output_0", "float32", (batch, bot_out, 9))], [TensorSpec("/MatMul_output_0", "float32", (batch, 9, 9))]),
            _make_node(50, "/Shape_2", "Shape", [TensorSpec("/MatMul_output_0", "float32", (batch, 9, 9))], [TensorSpec("/Shape_2_output_0", "int64", (3,))]),
            _make_node(51, "/Constant_11", "Constant", outputs=[TensorSpec("/Constant_11_output_0", "int64", (1,))]),
            _make_node(52, "/Gather_10", "Gather", [TensorSpec("/Shape_2_output_0", "int64", (3,)), TensorSpec("/Constant_11_output_0", "int64", (1,))], [TensorSpec("/Gather_10_output_0", "int64", (1,))]),
            _make_node(53, "/Constant_12", "Constant", outputs=[TensorSpec("/Constant_12_output_0", "int64", (1,))]),
            _make_node(54, "/Gather_11", "Gather", [TensorSpec("/Shape_2_output_0", "int64", (3,)), TensorSpec("/Constant_12_output_0", "int64", (1,))], [TensorSpec("/Gather_11_output_0", "int64", (1,))]),
            _make_node(55, "/Transpose_1", "Transpose", [TensorSpec("/MatMul_output_0", "float32", (batch, 9, 9))], [TensorSpec("/Transpose_1_output_0", "float32", (9, 9, batch))]),
            _make_node(56, "/Flatten", "Flatten", [TensorSpec("/Transpose_1_output_0", "float32", (9, 9, batch))], [TensorSpec("/Flatten_output_0", "float32", (81, batch))]),
            _make_node(57, "/Constant_13", "Constant", outputs=[TensorSpec("/Constant_13_output_0", "int64", (interaction_pair_count,))]),
            _make_node(58, "/Mul", "Mul", [TensorSpec("/Constant_13_output_0", "int64", (interaction_pair_count,)), TensorSpec("/Gather_11_output_0", "int64", (1,))], [TensorSpec("/Mul_output_0", "int64", (interaction_pair_count,))]),
            _make_node(59, "/Constant_14", "Constant", outputs=[TensorSpec("/Constant_14_output_0", "int64", (interaction_pair_count,))]),
            _make_node(60, "/Add", "Add", [TensorSpec("/Constant_14_output_0", "int64", (interaction_pair_count,)), TensorSpec("/Mul_output_0", "int64", (interaction_pair_count,))], [TensorSpec("/Add_output_0", "int64", (interaction_pair_count,))]),
            _make_node(61, "/Gather_12", "Gather", [TensorSpec("/Flatten_output_0", "float32", (81, batch)), TensorSpec("/Add_output_0", "int64", (interaction_pair_count,))], [TensorSpec("/Gather_12_output_0", "float32", (interaction_pair_count, batch))]),
            _make_node(62, "/Shape_3", "Shape", [TensorSpec("/Add_output_0", "int64", (interaction_pair_count,))], [TensorSpec("/Shape_3_output_0", "int64", (1,))]),
            _make_node(63, "/Constant_15", "Constant", outputs=[TensorSpec("/Constant_15_output_0", "int64", (1,))]),
            _make_node(64, "/Concat_2", "Concat", [TensorSpec("/Constant_15_output_0", "int64", (1,)), TensorSpec("/Gather_10_output_0", "int64", (1,))], [TensorSpec("/Concat_2_output_0", "int64", (2,))]),
            _make_node(65, "/Reshape_1", "Reshape", [TensorSpec("/Gather_12_output_0", "float32", (interaction_pair_count, batch)), TensorSpec("/Concat_2_output_0", "int64", (2,))], [TensorSpec("/Reshape_1_output_0", "float32", (interaction_pair_count, batch))]),
            _make_node(66, "/Transpose_2", "Transpose", [TensorSpec("/Reshape_1_output_0", "float32", (interaction_pair_count, batch))], [TensorSpec("/Transpose_2_output_0", "float32", (batch, interaction_pair_count))]),
            _make_node(67, "/Concat_3", "Concat", [TensorSpec("/Gather_10_output_0", "int64", (1,)), TensorSpec("/Shape_3_output_0", "int64", (1,))], [TensorSpec("/Concat_3_output_0", "int64", (2,))]),
            _make_node(68, "/Reshape_2", "Reshape", [TensorSpec("/Transpose_2_output_0", "float32", (batch, interaction_pair_count)), TensorSpec("/Concat_3_output_0", "int64", (2,))], [TensorSpec("/Reshape_2_output_0", "float32", (batch, interaction_pair_count))]),
            _make_node(69, "/Concat_4", "Concat", [TensorSpec("/bot_l/bot_l.3/Relu_output_0", "float32", (batch, bot_out)), TensorSpec("/Reshape_2_output_0", "float32", (batch, interaction_pair_count))], [TensorSpec("/Concat_4_output_0", "float32", (batch, bot_out + interaction_pair_count))]),
        ]
    )

    # top MLP。top_input = bot_out + 36，因为 top MLP 接收 bottom dense 输出和 36 个 pair features。
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
                    TensorSpec(input_tensor, "float32", (batch, input_dim)),
                    TensorSpec(f"top_l.{idx - 70}.weight", "float32", (output_dim, input_dim), "initializer"),
                    TensorSpec(f"top_l.{idx - 70}.bias", "float32", (output_dim,), "initializer"),
                ],
                [TensorSpec(output_tensor, "float32", (batch, output_dim))],
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
                    [TensorSpec(output_tensor, "float32", (batch, output_dim))],
                    [TensorSpec(relu_output, "float32", (batch, output_dim))],
                )
            )
    nodes.append(
        _make_node(
            75,
            "/top_l/top_l.5/Sigmoid",
            "Sigmoid",
            [TensorSpec("/top_l/top_l.4/Gemm_output_0", "float32", (batch, top[2]))],
            [TensorSpec("pred", "float32", (batch, top[2]))],
        )
    )
    return sorted(nodes, key=lambda item: item.node_idx)


def build_shape_rows(nodes: list[NodeSpec]) -> list[dict[str, Any]]:
    """生成完整 shape 表。

    每个算子的每个 input/output tensor 都是一行。这个表对应主线里的
    `static_op_shapes_full_graph_76.csv` 口径。
    """

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
                    "node_idx": node.node_idx,
                    "node_name": node.node_name,
                    "op_type": node.op_type,
                    "tensor_role": "input",
                    "port_idx": port_idx,
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
                    "node_idx": node.node_idx,
                    "node_name": node.node_name,
                    "op_type": node.op_type,
                    "tensor_role": "output",
                    "port_idx": port_idx,
                    "tensor_name": spec.tensor_name,
                    "dtype": spec.dtype,
                    "shape": _shape_cell(spec.dims),
                    "is_dynamic": False,
                    "shape_source": "static_config",
                    "producer_node": _node_ref(node),
                    "consumer_nodes": ", ".join(consumers),
                }
            )
    return rows


def _is_parameter_input(input_spec: TensorSpec, producer_by_tensor: dict[str, str]) -> bool:
    producer = producer_by_tensor.get(input_spec.tensor_name, input_spec.source)
    return producer == "initializer" or producer.endswith(":Constant")


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


def _infer_gemm_mnk(node: NodeSpec) -> tuple[float, float, float]:
    """从 Gemm/MatMul 的输入输出 shape 推导 M/N/K。

    约定：
    - M：输出矩阵除了最后一维以外的元素数。
    - N：输出矩阵最后一维。
    - K：输入 A 的最后一维；如果权重 shape 能更明确约束，也用权重修正。
    """

    if not node.inputs or not node.outputs:
        return 0.0, 0.0, 0.0
    a_dims = node.inputs[0].dims
    b_dims = node.inputs[1].dims if len(node.inputs) > 1 else ()
    c_dims = node.outputs[0].dims
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


def _operator_stage(node: NodeSpec) -> str:
    if node.node_name.startswith("/emb_l"):
        return "embedding"
    if node.node_name.startswith("/bot_l"):
        return "bottom_mlp"
    if node.node_name.startswith("/top_l"):
        return "top_mlp"
    if 36 <= node.node_idx <= 69:
        return "interaction"
    return "graph_meta"


def _scale_rule(row: dict[str, Any]) -> str:
    stage = str(row["stage"])
    op_type = str(row["op_type"])
    batch = int(row["batch_size"])
    nip = int(row["num_indices_per_lookup"])
    lookup = int(float(row["feat_lookup_count"]))
    gemm_m = int(float(row["feat_gemm_m"]))
    gemm_n = int(float(row["feat_gemm_n"]))
    gemm_k = int(float(row["feat_gemm_k"]))
    mac = int(float(row["feat_gemm_mac_count"]))
    reduction_work = int(float(row["feat_reduction_work_items"]))

    if stage == "embedding" and op_type == "Gather":
        return (
            "embedding table 行数来自 arch_embedding_size；"
            f"indices 数量 = batch_size * num_indices_per_lookup = {batch} * {nip} = {lookup}"
        )
    if stage == "embedding" and op_type == "ReduceSum":
        return (
            "ReduceSum 把每个样本查到的 num_indices_per_lookup 个 embedding 向量求和；"
            f"reduction work items = {reduction_work}"
        )
    if stage == "embedding" and op_type == "Reshape":
        return "Reshape 把扁平 Gather 输出还原为 [batch_size, num_indices_per_lookup, bot_out]"
    if op_type in {"Gemm", "MatMul"}:
        return f"{stage} 计算规模：M={gemm_m}, N={gemm_n}, K={gemm_k}, MAC={mac}"
    if stage == "interaction" and op_type in {"Concat", "Reshape", "Transpose", "Flatten", "Gather", "Add", "Mul"}:
        return (
            "interaction 规模来自 9 路输入 = 1 个 dense branch + 8 个 embedding branch；"
            "静态图构造 9x9 interaction matrix，并保留 36 个 pair features"
        )
    if op_type in {"Shape", "Unsqueeze", "Constant"}:
        return "shape/meta 类算子主要处理 shape 描述 tensor，规模来自静态 shape 信息"
    return "该算子的规模字段由静态输入/输出 tensor shape 推导得到"


def build_feature_rows(config: StaticConfig, nodes: list[NodeSpec]) -> list[dict[str, Any]]:
    """为 76 个节点计算规模字段。

    这里计算的是静态规模，不是 latency：

    - output_size / activation_size / parameter_size：字节规模。
    - feat_lookup_count：Gather 查表次数。
    - feat_gemm_m/n/k 和 feat_gemm_mac_count：矩阵乘规模。
    - feat_reduction_*：ReduceSum 规模。
    """

    producer_by_tensor: dict[str, str] = {}
    for node in nodes:
        for output in node.outputs:
            producer_by_tensor[output.tensor_name] = _node_ref(node)

    rows: list[dict[str, Any]] = []
    normalized_embedding = _normalized_arch_embedding_size(config.arch_embedding_size)
    normalized_bot = _normalized_arch(config.arch_mlp_bot, expected=3)
    normalized_top = _normalized_arch(config.arch_mlp_top, expected=3)

    for node in nodes:
        output_size = sum(_tensor_bytes(spec) for spec in node.outputs)
        output_elements = float(sum(_element_count(spec.dims) for spec in node.outputs))
        activation_size = 0.0
        activation_elements = 0.0
        parameter_size = 0.0
        for spec in node.inputs:
            if _is_parameter_input(spec, producer_by_tensor):
                parameter_size += _tensor_bytes(spec)
            else:
                activation_size += _tensor_bytes(spec)
                activation_elements += float(_element_count(spec.dims))

        lookup_count = 0.0
        if node.op_type == "Gather" and len(node.inputs) >= 2:
            lookup_count = float(_element_count(node.inputs[1].dims))

        reduction_dims: list[int] = []
        if node.op_type == "ReduceSum" and node.inputs and node.outputs:
            reduction_dims = _infer_reduction_dims(node.inputs[0].dims, node.outputs[0].dims)
        reduction_axes_product = 1.0
        for dim in reduction_dims:
            reduction_axes_product *= float(dim)
        if not reduction_dims:
            reduction_axes_product = 0.0
        reduction_work_items = 0.0
        if reduction_axes_product > 0.0 and output_elements > 0.0:
            reduction_work_items = float(output_elements * max(reduction_axes_product - 1.0, 0.0))

        gemm_m, gemm_n, gemm_k = _infer_gemm_mnk(node)
        gemm_mac = gemm_m * gemm_n * gemm_k if node.op_type in {"Gemm", "MatMul"} else 0.0
        io_bytes = float(output_size + activation_size + parameter_size)
        row = {
            "row_uid": f"{config.source_name}::{config.combo}::opidx:{node.node_idx}::bs{config.batch_size}_nip{config.num_indices_per_lookup}",
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
            "op_idx": node.node_idx,
            "node_idx": node.node_idx,
            "node_name": node.node_name,
            "op_type": node.op_type,
            "stage": _operator_stage(node),
            "in_model_input_51": node.node_idx in MODEL_INPUT_NODE_IDXS,
            "feature_dialect_observed": "static_config_v1",
            "has_cpu_profile": 0,
            "input_type_shape": _typed_shape_text(list(node.inputs)),
            "output_type_shape": _typed_shape_text(list(node.outputs)),
            "arch_embedding_size": normalized_embedding,
            "arch_mlp_bot": normalized_bot,
            "arch_mlp_top": normalized_top,
            "batch_size": config.batch_size,
            "num_indices_per_lookup": config.num_indices_per_lookup,
            "num_threads": config.num_threads,
            "inter_threads": config.inter_threads,
            "output_size": float(output_size),
            "activation_size": float(activation_size),
            "parameter_size": float(parameter_size),
            "feat_io_bytes_sum": io_bytes,
            "feat_output_input_bytes_ratio": float(output_size / max(activation_size + parameter_size, 1.0)),
            "feat_lookup_count": lookup_count,
            "feat_output_elements_per_lookup": float(output_elements / max(lookup_count, 1.0)) if lookup_count > 0.0 else 0.0,
            "feat_output_elements_per_batch": float(output_elements / max(float(config.batch_size), 1.0)),
            "feat_activation_elements_per_batch": float(activation_elements / max(float(config.batch_size), 1.0)),
            "feat_reduction_axes_count": float(len(reduction_dims)) if node.op_type == "ReduceSum" else 0.0,
            "feat_reduction_axes_product": reduction_axes_product if node.op_type == "ReduceSum" else 0.0,
            "feat_reduction_input_rank": float(len(node.inputs[0].dims)) if node.op_type == "ReduceSum" and node.inputs else 0.0,
            "feat_reduction_output_rank": float(len(node.outputs[0].dims)) if node.op_type == "ReduceSum" and node.outputs else 0.0,
            "feat_reduction_work_items": reduction_work_items if node.op_type == "ReduceSum" else 0.0,
            "feat_gemm_m": gemm_m if node.op_type in {"Gemm", "MatMul"} else 0.0,
            "feat_gemm_n": gemm_n if node.op_type in {"Gemm", "MatMul"} else 0.0,
            "feat_gemm_k": gemm_k if node.op_type in {"Gemm", "MatMul"} else 0.0,
            "feat_gemm_mac_count": gemm_mac,
            "feat_gemm_bytes_per_mac": float(io_bytes / max(gemm_mac, 1.0)) if gemm_mac > 0.0 else 0.0,
            "feat_memory_ops": 0.0,
            "feat_working_set_bytes": io_bytes,
            "feat_threads_effective": float(max(config.num_threads, 1)),
            "feat_output_elements": output_elements,
            "feat_activation_elements": activation_elements,
            "feat_parameter_elements": float(parameter_size / 4.0),
            "feat_activation_per_output_element": float(activation_elements / max(output_elements, 1.0)),
            "feat_reduction_factor": float(activation_elements / max(output_elements, 1.0)) if node.op_type == "ReduceSum" else 0.0,
        }
        row["scale_rule"] = _scale_rule(row)
        rows.append(row)
    return rows


def map_dlrm_config_to_operator_scale(config: StaticConfig) -> dict[str, Any]:
    for field in ("batch_size", "num_indices_per_lookup", "inter_threads", "num_threads"):
        if getattr(config, field) <= 0:
            raise ValueError(f"{field} must be positive")
    nodes = build_full_graph_76_nodes(config)
    shape_rows = build_shape_rows(nodes)
    feature_rows = build_feature_rows(config, nodes)
    model_input_rows = [row for row in feature_rows if row["in_model_input_51"]]
    summary_rows = [
        {**{column: row.get(column, "") for column in SUMMARY_COLUMNS}, "shape_source": "static_config"}
        for row in feature_rows
    ]
    metadata = {
        "config": {
            "batch_size": config.batch_size,
            "num_indices_per_lookup": config.num_indices_per_lookup,
            "arch_embedding_size": config.arch_embedding_size,
            "arch_mlp_bot": config.arch_mlp_bot,
            "arch_mlp_top": config.arch_mlp_top,
            "inter_threads": config.inter_threads,
            "num_threads": config.num_threads,
        },
        "full_graph_node_count": len({node.node_idx for node in nodes}),
        "model_input_row_count": len(model_input_rows),
        "shape_source": "static_config",
        "has_cpu_profile": 0,
        "profiling_required": False,
        "field_meanings": FIELD_MEANINGS,
    }
    return {
        "nodes": nodes,
        "shape_rows": shape_rows,
        "feature_rows": feature_rows,
        "model_input_rows": model_input_rows,
        "summary_rows": summary_rows,
        "metadata": metadata,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_mapping_outputs(mapping: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "dlrm_static_op_shapes_full_graph_76.csv", mapping["shape_rows"], SHAPE_COLUMNS)
    _write_csv(out_dir / "dlrm_static_features_full_graph_76.csv", mapping["feature_rows"], FEATURE_COLUMNS)
    _write_csv(out_dir / "dlrm_static_model_input_51.csv", mapping["model_input_rows"], FEATURE_COLUMNS)
    _write_csv(out_dir / "dlrm_operator_scale_mapping_summary.csv", mapping["summary_rows"], SUMMARY_COLUMNS)
    payload = {
        **mapping["metadata"],
        "operator_scale_rows": mapping["summary_rows"],
    }
    (out_dir / "dlrm_operator_scale_mapping_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_config(path: Path) -> StaticConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StaticConfig(
        batch_size=int(payload["batch_size"]),
        num_indices_per_lookup=int(payload["num_indices_per_lookup"]),
        arch_embedding_size=str(payload["arch_embedding_size"]),
        arch_mlp_bot=str(payload["arch_mlp_bot"]),
        arch_mlp_top=str(payload.get("arch_mlp_top", "1600-800-1")),
        inter_threads=int(payload["inter_threads"]),
        num_threads=int(payload["num_threads"]),
    )


def _config_from_args(args: argparse.Namespace) -> StaticConfig:
    if args.config_json:
        return load_config(Path(args.config_json))
    required = {
        "batch_size": args.batch_size,
        "num_indices_per_lookup": args.num_indices_per_lookup,
        "arch_embedding_size": args.arch_embedding_size,
        "arch_mlp_bot": args.arch_mlp_bot,
        "inter_threads": args.inter_threads,
        "num_threads": args.num_threads,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required args when --config-json is not used: {', '.join(missing)}")
    return StaticConfig(
        batch_size=int(args.batch_size),
        num_indices_per_lookup=int(args.num_indices_per_lookup),
        arch_embedding_size=str(args.arch_embedding_size),
        arch_mlp_bot=str(args.arch_mlp_bot),
        arch_mlp_top=str(args.arch_mlp_top or "1600-800-1"),
        inter_threads=int(args.inter_threads),
        num_threads=int(args.num_threads),
    )


def run_self_test(mapping: dict[str, Any]) -> None:
    """运行 demo 内置自检。

    自检不需要项目代码，只检查这个独立映射是否满足主线静态映射的关键不变量：
    - 完整图必须正好有 76 个 node index。
    - 模型输入必须正好有 51 行。
    - 8 个 embedding Gather 都必须进入模型输入。
    - embedding Gather 的 lookup count 必须等于 batch_size * num_indices_per_lookup。
    - Gemm/MatMul 必须有非零 M/N/K/MAC。
    - 所有行都必须标记为不依赖 profiling。
    """

    metadata = mapping["metadata"]
    feature_rows = mapping["feature_rows"]
    model_input_rows = mapping["model_input_rows"]
    expected_lookup = int(metadata["config"]["batch_size"]) * int(metadata["config"]["num_indices_per_lookup"])
    node_count = len({int(row["node_idx"]) for row in feature_rows})
    if node_count != 76:
        raise AssertionError(f"expected 76 full-graph nodes, got {node_count}")
    if len(model_input_rows) != 51:
        raise AssertionError(f"expected 51 model-input rows, got {len(model_input_rows)}")
    embedding_gathers = [
        row
        for row in feature_rows
        if row["stage"] == "embedding" and row["op_type"] == "Gather"
    ]
    if len(embedding_gathers) != 8:
        raise AssertionError(f"expected 8 embedding Gather rows, got {len(embedding_gathers)}")
    if any(int(row["feat_lookup_count"]) != expected_lookup for row in embedding_gathers):
        raise AssertionError("embedding Gather lookup count does not match batch_size * num_indices_per_lookup")
    matrix_rows = [row for row in feature_rows if row["op_type"] in {"Gemm", "MatMul"}]
    if not matrix_rows:
        raise AssertionError("missing Gemm/MatMul rows")
    for row in matrix_rows:
        values = [float(row["feat_gemm_m"]), float(row["feat_gemm_n"]), float(row["feat_gemm_k"]), float(row["feat_gemm_mac_count"])]
        if any(value <= 0.0 for value in values):
            raise AssertionError(f"non-positive Gemm/MatMul scale in node {row['node_idx']}: {values}")
    if any(int(row["has_cpu_profile"]) != 0 for row in feature_rows):
        raise AssertionError("standalone demo should not mark any row as CPU-profile based")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立 DLRM 上层参数到算子规模映射 demo。")
    parser.add_argument("--config-json", default="", help="示例配置 JSON；使用它时可省略单个配置参数。")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-indices-per-lookup", type=int)
    parser.add_argument("--arch-embedding-size")
    parser.add_argument("--arch-mlp-bot")
    parser.add_argument("--arch-mlp-top", default="1600-800-1")
    parser.add_argument("--inter-threads", type=int)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--self-test", action="store_true", help="写出文件后执行内置一致性检查。")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _config_from_args(args)
    mapping = map_dlrm_config_to_operator_scale(config)
    write_mapping_outputs(mapping, Path(args.out_dir))
    if args.self_test:
        run_self_test(mapping)
    print(json.dumps(mapping["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
