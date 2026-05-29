from __future__ import annotations

from ..paths import (
    DEFAULT_ANALYTICAL_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_SINGLE_MODEL_ROOT,
)


DEFAULT_INPUT_DATASET_DIR = DEFAULT_DATASET_ROOT
DEFAULT_INPUT_CSV = DEFAULT_INPUT_DATASET_DIR / "dataset_full.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYTICAL_ROOT
BASELINE_COMPARE_DIR = DEFAULT_SINGLE_MODEL_ROOT

OP_CLASS_ORDER = (
    "memory_pure",
    "mixed_balanced",
    "compute_dominant",
)
UNKNOWN_OP_CLASS = "mixed_balanced"

OP_TYPE_CLASS_MAP = {
    "Add": "mixed_balanced",
    "Concat": "memory_pure",
    "Flatten": "memory_pure",
    "Gather": "memory_pure",
    "Gemm": "compute_dominant",
    "MatMul": "compute_dominant",
    "Mul": "mixed_balanced",
    "ReduceSum": "mixed_balanced",
    "Relu": "mixed_balanced",
    "Reshape": "memory_pure",
    "Shape": "memory_pure",
    "Sigmoid": "mixed_balanced",
    "Transpose": "memory_pure",
    "Unsqueeze": "memory_pure",
}

HEAVY_FAMILIES = (
    "Gather",
    "ReduceSum",
    "Gemm",
    "MatMul",
    "Transpose",
    "Concat",
)
OP_AWARE_LIGHT_OP_TYPES = (
    "Relu",
    "Add",
    "Mul",
    "Sigmoid",
)
CALIBRATED_FAMILIES = HEAVY_FAMILIES + OP_AWARE_LIGHT_OP_TYPES
GENERIC_MEMORY_OP_TYPES = (
    "Reshape",
    "Shape",
    "Unsqueeze",
    "Flatten",
)
GENERIC_MIXED_OP_TYPES: tuple[str, ...] = ()

SHARED_CATEGORICAL_FEATURES = (
    "op_type",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
)

PER_CLASS_NUMERIC_FEATURES = {
    "memory_pure": (
        "num_threads",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_lookup_count",
        "feat_output_elements_per_lookup",
        "ana_calib_total_us",
        "ana_calib_mem_us",
    ),
    "mixed_balanced": (
        "num_threads",
        "feat_io_bytes_sum",
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "feat_reduction_work_items",
        "feat_reduction_axes_product",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ana_calib_compute_us",
    ),
    "compute_dominant": (
        "num_threads",
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
        "ana_calib_total_us",
        "ana_calib_compute_us",
    ),
}

ANALYTICAL_FEATURE_COLUMNS = (
    "ana_calib_total_us",
    "ana_calib_mem_us",
    "ana_calib_compute_us",
    "ana_calib_overhead_us",
    "ana_calib_family",
    "ana_calib_superfamily",
    "ana_calib_regime",
    "op_class",
)

FEATURE_DESCRIPTIONS = {
    "op_type": "ONNX 算子类型，用于静态路由到三类模型。",
    "arch_embedding_size": "DLRM embedding 维度配置。",
    "arch_mlp_bot": "DLRM bottom MLP 结构配置。",
    "arch_mlp_top": "DLRM top MLP 结构配置。",
    "num_threads": "该样本的 intra-op 线程数。",
    "feat_io_bytes_sum": "总 I/O 字节量，近似单算子的访存工作集。",
    "feat_output_input_bytes_ratio": "输出字节与输入/参数字节总量的比值，表示膨胀或压缩程度。",
    "feat_lookup_count": "Gather 的真实请求元素数，优先由 indices tensor shape 推导，shape 缺失时才回退到 batch_size * num_indices_per_lookup。",
    "feat_output_elements_per_lookup": "每个真实 Gather request 对应的平均输出元素数。",
    "feat_output_elements_per_batch": "每个 batch 样本对应的平均输出元素数。",
    "feat_reduction_work_items": "Reduce 类算子的核心归约工作量，近似需要被合并的元素规模。",
    "feat_reduction_axes_product": "被归约维度尺寸的乘积，反映归约规模。",
    "feat_gemm_m": "Gemm/MatMul 的 M 维。",
    "feat_gemm_n": "Gemm/MatMul 的 N 维。",
    "feat_gemm_k": "Gemm/MatMul 的 K 维。",
    "feat_gemm_mac_count": "矩阵乘总 MAC 数，反映计算体量。",
    "feat_gemm_bytes_per_mac": "每个 MAC 对应的字节开销，反映计算/访存比。",
}

ANALYTICAL_FEATURE_DESCRIPTIONS = {
    "ana_calib_total_us": "校准 analytical model 预测的总时延。",
    "ana_calib_mem_us": "校准 analytical model 中访存主项对应的时延。",
    "ana_calib_compute_us": "校准 analytical model 中计算主项对应的时延。",
    "ana_calib_overhead_us": "校准 analytical model 中 dispatch、启动或微核等结构性开销。",
    "ana_calib_family": "当前样本使用的 analytical family 或通用 proxy 名称。",
    "ana_calib_superfamily": "兼容导出的上层 analytical 超家族标签；`Gemm/MatMul` 统一映射到 `gemm_like`。",
    "ana_calib_regime": "当前 analytical 公式所属的具体执行区间；例如 `Gemm -> large_gemm_saturation`、`MatMul -> tiny_batched_occ`。",
    "op_class": "三分类标签，取值为 memory_pure / mixed_balanced / compute_dominant。",
}


def resolve_op_class(op_type: str | None) -> str:
    key = "" if op_type is None else str(op_type).strip()
    if not key:
        return UNKNOWN_OP_CLASS
    return OP_TYPE_CLASS_MAP.get(key, UNKNOWN_OP_CLASS)
