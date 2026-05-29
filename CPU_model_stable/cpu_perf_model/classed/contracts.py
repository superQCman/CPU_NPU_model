from __future__ import annotations

from pathlib import Path

from ..analytical.contracts import (
    ANALYTICAL_FEATURE_DESCRIPTIONS,
    BASELINE_COMPARE_DIR,
    DEFAULT_INPUT_DATASET_DIR,
    FEATURE_DESCRIPTIONS as BASE_FEATURE_DESCRIPTIONS,
    OP_CLASS_ORDER,
    OP_TYPE_CLASS_MAP,
    SHARED_CATEGORICAL_FEATURES,
    resolve_op_class,
)
from ..paths import DEFAULT_CLASSED_MODEL_ROOT


DEFAULT_OUTPUT_ROOT = DEFAULT_CLASSED_MODEL_ROOT

FEATURE_BRANCH_WITH_ANALYTICAL = "with_analytical"
FEATURE_BRANCH_NO_ANALYTICAL = "no_analytical"
SUPPORTED_FEATURE_BRANCHES = (
    FEATURE_BRANCH_WITH_ANALYTICAL,
    FEATURE_BRANCH_NO_ANALYTICAL,
)
DEFAULT_FEATURE_BRANCH = FEATURE_BRANCH_WITH_ANALYTICAL

WITH_ANALYTICAL_CATEGORICAL_FEATURES = NO_ANALYTICAL_CATEGORICAL_FEATURES = (
    "op_type",
    "node_scope",
    "node_name_normalized",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
)

EXTRA_FEATURE_DESCRIPTIONS = {
    "node_scope": "节点所属的图内作用域或模块上下文，近似反映节点位置和调用背景。",
    "node_name_normalized": "归一化后的节点名，近似反映节点身份和固定 kernel 或 dispatch 模式。",
    "batch_size": "该样本对应的 DLRM batch size。",
    "num_indices_per_lookup": "每次 embedding lookup 的索引数配置。",
    "output_size": "算子输出张量字节量。",
    "activation_size": "算子输入激活张量字节量。",
    "parameter_size": "算子参数张量字节量。",
    "inter_threads": "该 combo 的静态 inter-op 线程配置。优先从 build_ops.log 的 default_inter_threads 恢复，缺失时回退到 case 启动脚本中的 INTER_THREADS。",
    "feat_activation_elements_per_batch": "每个 batch 对应的输入激活元素规模。",
    "feat_reduction_axes_count": "被归约轴的数量。",
    "feat_reduction_input_rank": "归约前输入张量 rank。",
    "feat_reduction_output_rank": "归约后输出张量 rank。",
}

CLASSED_FEATURE_DESCRIPTIONS = {
    **BASE_FEATURE_DESCRIPTIONS,
    **EXTRA_FEATURE_DESCRIPTIONS,
    **ANALYTICAL_FEATURE_DESCRIPTIONS,
}

WITH_ANALYTICAL_MODEL_GROUP_OP_TYPES = {
    "gather": ("Gather",),
    "layout_move": ("Concat", "Transpose"),
    "view_meta": ("Reshape", "Shape", "Unsqueeze", "Flatten"),
    "mixed_balanced": ("ReduceSum", "Sigmoid", "Relu", "Add", "Mul"),
    "compute_dominant": ("Gemm", "MatMul"),
}
WITH_ANALYTICAL_MODEL_GROUP_ORDER = tuple(WITH_ANALYTICAL_MODEL_GROUP_OP_TYPES.keys())

NO_ANALYTICAL_MODEL_GROUP_OP_TYPES = {
    "gather": ("Gather",),
    "layout_move": ("Concat", "Transpose"),
    "view_meta": ("Reshape", "Shape", "Unsqueeze", "Flatten"),
    "mixed_balanced": ("ReduceSum", "Sigmoid", "Relu", "Add", "Mul"),
    "compute_dominant": ("Gemm", "MatMul"),
}
NO_ANALYTICAL_MODEL_GROUP_ORDER = tuple(NO_ANALYTICAL_MODEL_GROUP_OP_TYPES.keys())

WITH_ANALYTICAL_PER_CLASS_NUMERIC_FEATURES = {
    "gather": (
        "batch_size",
        "num_indices_per_lookup",
        "num_threads",
        "inter_threads",
        "output_size",
        "activation_size",
        "parameter_size",
        "feat_io_bytes_sum",
        "feat_lookup_count",
        "ana_calib_mem_us",
    ),
    "layout_move": (
        "batch_size",
        "num_threads",
        "output_size",
        "activation_size",
        "feat_io_bytes_sum",
        "feat_output_elements_per_batch",
        "ana_calib_total_us",
    ),
    "view_meta": (
        "batch_size",
        "num_threads",
        "output_size",
        "activation_size",
    ),
    "mixed_balanced": (
        "batch_size",
        "num_threads",
        "inter_threads",
        "output_size",
        "activation_size",
        "feat_io_bytes_sum",
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "feat_reduction_axes_count",
        "feat_reduction_work_items",
        "feat_reduction_axes_product",
        "feat_reduction_input_rank",
        "feat_reduction_output_rank",
        "ana_calib_total_us",
    ),
    "compute_dominant": (
        "batch_size",
        "num_threads",
        "output_size",
        "activation_size",
        "parameter_size",
        "feat_io_bytes_sum",
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
        "ana_calib_compute_us",
    ),
}

NO_ANALYTICAL_PER_CLASS_NUMERIC_FEATURES = {
    "gather": (
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
    ),
    "layout_move": (
        "batch_size",
        "num_threads",
        "output_size",
        "activation_size",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_output_elements_per_batch",
    ),
    "view_meta": (
        "batch_size",
        "num_threads",
        "output_size",
        "activation_size",
        "feat_output_input_bytes_ratio",
        "feat_output_elements_per_batch",
    ),
    "mixed_balanced": (
        "batch_size",
        "num_threads",
        "inter_threads",
        "output_size",
        "activation_size",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_output_elements_per_batch",
        "feat_activation_elements_per_batch",
        "feat_reduction_axes_count",
        "feat_reduction_axes_product",
        "feat_reduction_input_rank",
        "feat_reduction_output_rank",
        "feat_reduction_work_items",
    ),
    "compute_dominant": (
        "batch_size",
        "num_threads",
        "output_size",
        "activation_size",
        "parameter_size",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
    ),
}

PER_BRANCH_NUMERIC_FEATURES = {
    FEATURE_BRANCH_WITH_ANALYTICAL: WITH_ANALYTICAL_PER_CLASS_NUMERIC_FEATURES,
    FEATURE_BRANCH_NO_ANALYTICAL: NO_ANALYTICAL_PER_CLASS_NUMERIC_FEATURES,
}
PER_BRANCH_CATEGORICAL_FEATURES = {
    FEATURE_BRANCH_WITH_ANALYTICAL: WITH_ANALYTICAL_CATEGORICAL_FEATURES,
    FEATURE_BRANCH_NO_ANALYTICAL: NO_ANALYTICAL_CATEGORICAL_FEATURES,
}
PER_BRANCH_MODEL_GROUP_ORDER = {
    FEATURE_BRANCH_WITH_ANALYTICAL: WITH_ANALYTICAL_MODEL_GROUP_ORDER,
    FEATURE_BRANCH_NO_ANALYTICAL: NO_ANALYTICAL_MODEL_GROUP_ORDER,
}
PER_BRANCH_MODEL_GROUP_OP_TYPES = {
    FEATURE_BRANCH_WITH_ANALYTICAL: WITH_ANALYTICAL_MODEL_GROUP_OP_TYPES,
    FEATURE_BRANCH_NO_ANALYTICAL: NO_ANALYTICAL_MODEL_GROUP_OP_TYPES,
}


def resolve_output_dir(output_dir: str | Path | None, feature_branch: str) -> Path:
    if output_dir:
        return Path(output_dir)
    if feature_branch == FEATURE_BRANCH_WITH_ANALYTICAL:
        return DEFAULT_OUTPUT_ROOT
    return DEFAULT_OUTPUT_ROOT / feature_branch


def resolve_branch_features(feature_branch: str) -> dict[str, tuple[str, ...]]:
    if feature_branch not in PER_BRANCH_NUMERIC_FEATURES:
        raise ValueError(f"Unsupported feature branch: {feature_branch}")
    return PER_BRANCH_NUMERIC_FEATURES[feature_branch]


def resolve_branch_categorical_features(feature_branch: str) -> tuple[str, ...]:
    if feature_branch not in PER_BRANCH_CATEGORICAL_FEATURES:
        raise ValueError(f"Unsupported feature branch: {feature_branch}")
    return PER_BRANCH_CATEGORICAL_FEATURES[feature_branch]


def resolve_model_group_order(feature_branch: str) -> tuple[str, ...]:
    if feature_branch not in PER_BRANCH_MODEL_GROUP_ORDER:
        raise ValueError(f"Unsupported feature branch: {feature_branch}")
    return PER_BRANCH_MODEL_GROUP_ORDER[feature_branch]


def resolve_model_group_op_types(feature_branch: str) -> dict[str, tuple[str, ...]]:
    if feature_branch not in PER_BRANCH_MODEL_GROUP_OP_TYPES:
        raise ValueError(f"Unsupported feature branch: {feature_branch}")
    return PER_BRANCH_MODEL_GROUP_OP_TYPES[feature_branch]


def resolve_model_group(feature_branch: str, op_type: str | None) -> str:
    key = "" if op_type is None else str(op_type).strip()
    for model_group, op_types in resolve_model_group_op_types(feature_branch).items():
        if key in op_types:
            return model_group
    return resolve_op_class(key)
