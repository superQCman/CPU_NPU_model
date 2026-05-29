from __future__ import annotations

import ast
import math
from typing import Any

import pandas as pd


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    if isinstance(value, float) and math.isnan(value):
        return int(default)
    try:
        text = str(value).strip()
        if not text:
            return int(default)
        return int(float(text))
    except Exception:
        return int(default)


def numel(shape: list[int] | tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= _coerce_int(dim, 0)
    return total


def dtype_size(dtype: str) -> int:
    mapping = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
        "int32": 4,
        "int64": 8,
        "int16": 2,
        "int8": 1,
        "uint8": 1,
        "bool": 1,
    }
    return mapping.get(str(dtype).lower(), 4)


def safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)


def parse_jsonish_list(text: Any) -> list[int]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    if isinstance(text, list):
        return [_coerce_int(x, 0) for x in text]
    text = str(text).strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(value, list):
        return [_coerce_int(x, 0) for x in value]
    return []


def parse_intish(value: Any, default: int = 0) -> int:
    return _coerce_int(value, default)


def parse_tensor_list(text: Any) -> list[tuple[str, list[int]]]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    if isinstance(text, list):
        value = text
    else:
        text = str(text).strip()
        if not text:
            return []
        try:
            value = ast.literal_eval(text)
        except Exception:
            return []
    tensors: list[tuple[str, list[int]]] = []
    if not isinstance(value, list):
        return tensors
    for item in value:
        if isinstance(item, dict):
            for dtype, shape in item.items():
                if isinstance(shape, (list, tuple)):
                    tensors.append((str(dtype), [_coerce_int(x, 0) for x in shape]))
    return tensors


def stride_elems(shape: list[int]) -> list[int]:
    if not shape:
        return []
    strides = [1] * len(shape)
    acc = 1
    for idx in range(len(shape) - 1, -1, -1):
        strides[idx] = acc
        acc *= int(shape[idx])
    return strides


def broadcast_stats(lhs: list[int], rhs: list[int]) -> dict[str, float]:
    max_rank = max(len(lhs), len(rhs))
    lhs_pad = [1] * (max_rank - len(lhs)) + lhs
    rhs_pad = [1] * (max_rank - len(rhs)) + rhs
    out = []
    lhs_b_dims = 0
    rhs_b_dims = 0
    for ldim, rdim in zip(lhs_pad, rhs_pad):
        odim = max(ldim, rdim)
        out.append(odim)
        if ldim == 1 and odim > 1:
            lhs_b_dims += 1
        if rdim == 1 and odim > 1:
            rhs_b_dims += 1
    out_numel = numel(out)
    lhs_numel = numel(lhs) if lhs else 1
    rhs_numel = numel(rhs) if rhs else 1
    return {
        "binary_out_numel": float(out_numel),
        "binary_lhs_broadcast_dims": float(lhs_b_dims),
        "binary_rhs_broadcast_dims": float(rhs_b_dims),
        "binary_lhs_expand_factor": safe_ratio(out_numel, lhs_numel),
        "binary_rhs_expand_factor": safe_ratio(out_numel, rhs_numel),
        "binary_same_shape": float(tuple(lhs) == tuple(rhs)),
    }


def transpose_stats(shape: list[int], perm: list[int]) -> dict[str, float]:
    rank = len(shape)
    if rank == 0 or len(perm) != rank:
        return {}
    fixed = sum(int(i == p) for i, p in enumerate(perm))
    inversions = 0
    for i in range(rank):
        for j in range(i + 1, rank):
            if perm[i] > perm[j]:
                inversions += 1
    displacement = sum(abs(i - p) for i, p in enumerate(perm))
    adjacent_preserved = 0
    inverse = {p: i for i, p in enumerate(perm)}
    for i in range(rank - 1):
        if abs(inverse[i + 1] - inverse[i]) == 1:
            adjacent_preserved += 1
    strides = stride_elems(shape)
    permuted_strides = [strides[p] for p in perm]
    return {
        "transpose_rank": float(rank),
        "transpose_fixed_points": float(fixed),
        "transpose_inversion_count": float(inversions),
        "transpose_displacement_sum": float(displacement),
        "transpose_adjacent_preserved": float(adjacent_preserved),
        "transpose_batch_dim_fixed": float(perm[0] == 0),
        "transpose_last_dim_fixed": float(perm[-1] == rank - 1),
        "transpose_max_stride": float(max(permuted_strides)),
        "transpose_min_stride": float(min(permuted_strides)),
        "transpose_stride_span": float(max(permuted_strides) / max(1, min(permuted_strides))),
    }


def reduce_stats(shape: list[int], axes: list[int], keepdims: int) -> dict[str, float]:
    if not shape or not axes:
        return {}
    rank = len(shape)
    axes = sorted(int(ax) for ax in axes)
    red_extent = 1
    for ax in axes:
        if 0 <= ax < rank:
            red_extent *= int(shape[ax])
    axes_set = set(axes)
    suffix_axes = set(range(rank - len(axes), rank))
    prefix_axes = set(range(len(axes)))
    return {
        "reduce_extent": float(red_extent),
        "reduce_fraction": safe_ratio(red_extent, numel(shape)),
        "reduce_axes_span": float(axes[-1] - axes[0] + 1),
        "reduce_has_batch_axis": float(0 in axes_set),
        "reduce_has_last_axis": float((rank - 1) in axes_set),
        "reduce_is_suffix": float(axes_set == suffix_axes),
        "reduce_is_prefix": float(axes_set == prefix_axes),
        "reduce_keepdims": float(int(keepdims)),
    }


def gather_stats(inputs: list[tuple[str, list[int]]], unique_idx_count: float, working_set_eff_bytes: float) -> dict[str, float]:
    if len(inputs) < 2:
        return {}
    data_dtype, data_shape = inputs[0]
    _, indices_shape = inputs[1]
    if not data_shape:
        return {}
    table_rows = int(data_shape[0])
    row_elems = numel(data_shape[1:]) if len(data_shape) > 1 else 1
    row_bytes = row_elems * dtype_size(data_dtype)
    lookup_count = numel(indices_shape)
    unique = float(unique_idx_count) if pd.notna(unique_idx_count) else float(lookup_count)
    ws_eff = float(working_set_eff_bytes) if pd.notna(working_set_eff_bytes) else float(unique * row_bytes)
    return {
        "gather_table_rows": float(table_rows),
        "gather_row_elems": float(row_elems),
        "gather_row_bytes": float(row_bytes),
        "gather_lookup_count_exact": float(lookup_count),
        "gather_unique_ratio": safe_ratio(unique, lookup_count),
        "gather_reuse_factor": safe_ratio(lookup_count, unique),
        "gather_table_coverage": safe_ratio(unique, table_rows),
        "gather_ws_eff_rows": safe_ratio(ws_eff, row_bytes),
        "gather_ws_eff_per_lookup": safe_ratio(ws_eff, lookup_count),
        "gather_ws_eff_to_output_bytes": safe_ratio(ws_eff, lookup_count * row_bytes),
    }


def concat_stats(inputs: list[tuple[str, list[int]]], axis: int) -> dict[str, float]:
    if not inputs:
        return {}
    shapes = [shape for _, shape in inputs]
    rank = len(shapes[0]) if shapes[0] else 0
    if rank == 0:
        return {}
    axis = int(axis)
    if axis < 0:
        axis += rank
    axis_sizes = [int(shape[axis]) for shape in shapes if len(shape) == rank]
    input_numels = [numel(shape) for shape in shapes]
    return {
        "concat_input_count_exact": float(len(inputs)),
        "concat_axis": float(axis),
        "concat_axis_size_sum": float(sum(axis_sizes)),
        "concat_axis_size_max": float(max(axis_sizes)),
        "concat_axis_size_min": float(min(axis_sizes)),
        "concat_input_numel_max": float(max(input_numels)),
        "concat_input_numel_min": float(min(input_numels)),
    }


def flatten_stats(shape: list[int], axis: int) -> dict[str, float]:
    rank = len(shape)
    if rank == 0:
        return {}
    axis = int(axis)
    if axis < 0:
        axis += rank
    left = numel(shape[:axis]) if axis > 0 else 1
    right = numel(shape[axis:]) if axis < rank else 1
    return {
        "flatten_axis_exact": float(axis),
        "flatten_left_numel": float(left),
        "flatten_right_numel": float(right),
    }


def reshape_stats(in_shape: list[int], out_shape: list[int]) -> dict[str, float]:
    if not in_shape or not out_shape:
        return {}
    return {
        "reshape_rank_delta": float(len(out_shape) - len(in_shape)),
        "reshape_keeps_batch": float(in_shape[0] == out_shape[0]),
        "reshape_keeps_last_dim": float(in_shape[-1] == out_shape[-1]),
        "reshape_is_flatten2d": float(len(out_shape) == 2),
        "reshape_input_rank": float(len(in_shape)),
        "reshape_output_rank": float(len(out_shape)),
    }


def unsqueeze_stats(in_shape: list[int], out_shape: list[int], axes: list[int]) -> dict[str, float]:
    if not in_shape or not out_shape:
        return {}
    axes = sorted(int(ax) for ax in axes)
    return {
        "unsqueeze_rank_delta": float(len(out_shape) - len(in_shape)),
        "unsqueeze_axes_count_exact": float(len(axes)),
        "unsqueeze_prefix_insertions": float(sum(1 for ax in axes if ax == 0)),
        "unsqueeze_suffix_insertions": float(sum(1 for ax in axes if ax >= len(out_shape) - len(axes))),
    }


def shape_stats(in_shape: list[int]) -> dict[str, float]:
    if not in_shape:
        return {}
    return {
        "shape_input_rank_exact": float(len(in_shape)),
        "shape_input_numel_exact": float(numel(in_shape)),
        "shape_max_dim": float(max(in_shape)),
        "shape_min_dim": float(min(in_shape)),
    }


def gemm_stats(row: pd.Series) -> dict[str, float]:
    m = float(row.get("feat_gemm_m", 0.0) or 0.0)
    n = float(row.get("feat_gemm_n", 0.0) or 0.0)
    k = float(row.get("feat_gemm_k", 0.0) or 0.0)
    positive = [v for v in [m, n, k] if v > 0]
    return {
        "gemm_m_over_n": safe_ratio(m, n),
        "gemm_m_over_k": safe_ratio(m, k),
        "gemm_n_over_k": safe_ratio(n, k),
        "gemm_smallest_dim": float(min(positive) if positive else 0.0),
        "gemm_largest_dim": float(max(m, n, k)),
    }


def derive_enhanced_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        series = pd.Series(row._asdict())
        inputs = parse_tensor_list(series.get("input_type_shape"))
        outputs = parse_tensor_list(series.get("output_type_shape"))
        input_shapes = [shape for _, shape in inputs]
        output_shapes = [shape for _, shape in outputs]
        input_numels = [numel(shape) for shape in input_shapes]
        output_numels = [numel(shape) for shape in output_shapes]
        input_bytes = [numel(shape) * dtype_size(dtype) for dtype, shape in inputs]
        output_bytes = [numel(shape) * dtype_size(dtype) for dtype, shape in outputs]
        features: dict[str, float] = {
            "struct_input_count": float(len(inputs)),
            "struct_output_count": float(len(outputs)),
            "struct_total_input_numel": float(sum(input_numels)),
            "struct_total_output_numel": float(sum(output_numels)),
            "struct_total_input_bytes": float(sum(input_bytes)),
            "struct_total_output_bytes": float(sum(output_bytes)),
            "struct_input_rank0": float(len(input_shapes[0])) if input_shapes else 0.0,
            "struct_output_rank0": float(len(output_shapes[0])) if output_shapes else 0.0,
            "struct_input_numel0": float(input_numels[0]) if input_numels else 0.0,
            "struct_output_numel0": float(output_numels[0]) if output_numels else 0.0,
            "struct_rank_delta": (
                float(len(output_shapes[0]) - len(input_shapes[0]))
                if input_shapes and output_shapes
                else 0.0
            ),
            "struct_io_numel_ratio": (
                safe_ratio(output_numels[0], input_numels[0])
                if input_numels and output_numels
                else 0.0
            ),
            "struct_input_dim0": float(input_shapes[0][0]) if input_shapes and input_shapes[0] else 0.0,
            "struct_input_last_dim": float(input_shapes[0][-1]) if input_shapes and input_shapes[0] else 0.0,
            "struct_output_last_dim": float(output_shapes[0][-1]) if output_shapes and output_shapes[0] else 0.0,
        }

        op_type = str(series.get("op_type", ""))
        if op_type in {"Add", "Mul"} and len(input_shapes) >= 2:
            features.update(broadcast_stats(input_shapes[0], input_shapes[1]))
        if op_type == "Transpose" and input_shapes:
            features.update(transpose_stats(input_shapes[0], parse_jsonish_list(series.get("attr_perm_json"))))
        if op_type == "ReduceSum" and input_shapes:
            features.update(reduce_stats(input_shapes[0], parse_jsonish_list(series.get("attr_axes_json")), parse_intish(series.get("attr_keepdims"), 0)))
        if op_type == "Gather":
            features.update(
                gather_stats(
                    inputs,
                    unique_idx_count=series.get("gather_unique_idx_count"),
                    working_set_eff_bytes=series.get("gather_working_set_eff_bytes"),
                )
            )
        if op_type == "Concat":
            features.update(concat_stats(inputs, parse_intish(series.get("attr_axis"), 0)))
        if op_type == "Flatten" and input_shapes:
            features.update(flatten_stats(input_shapes[0], parse_intish(series.get("attr_axis"), 1)))
        if op_type == "Reshape" and input_shapes and output_shapes:
            features.update(reshape_stats(input_shapes[0], output_shapes[0]))
        if op_type == "Unsqueeze" and input_shapes and output_shapes:
            features.update(unsqueeze_stats(input_shapes[0], output_shapes[0], parse_jsonish_list(series.get("attr_axes_json"))))
        if op_type == "Shape" and input_shapes:
            features.update(shape_stats(input_shapes[0]))
        if op_type in {"Gemm", "MatMul"}:
            features.update(gemm_stats(series))
        rows.append(features)

    out = pd.DataFrame(rows)
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(float)
    out = out.reindex(columns=ALL_DERIVED_FEATURES, fill_value=0.0)
    return out


DERIVED_FEATURE_GROUPS = {
    "common": [  # shared across all enhanced models
        "struct_input_count",
        "struct_output_count",
        "struct_total_input_numel",
        "struct_total_output_numel",
        "struct_total_input_bytes",
        "struct_total_output_bytes",
        "struct_input_rank0",
        "struct_output_rank0",
        "struct_input_numel0",
        "struct_output_numel0",
        "struct_rank_delta",
        "struct_io_numel_ratio",
        "struct_input_dim0",
        "struct_input_last_dim",
        "struct_output_last_dim",
    ],
    "binary": [
        "binary_out_numel",
        "binary_lhs_broadcast_dims",
        "binary_rhs_broadcast_dims",
        "binary_lhs_expand_factor",
        "binary_rhs_expand_factor",
        "binary_same_shape",
    ],
    "transpose": [
        "transpose_rank",
        "transpose_fixed_points",
        "transpose_inversion_count",
        "transpose_displacement_sum",
        "transpose_adjacent_preserved",
        "transpose_batch_dim_fixed",
        "transpose_last_dim_fixed",
        "transpose_max_stride",
        "transpose_min_stride",
        "transpose_stride_span",
    ],
    "reduce": [
        "reduce_extent",
        "reduce_fraction",
        "reduce_axes_span",
        "reduce_has_batch_axis",
        "reduce_has_last_axis",
        "reduce_is_suffix",
        "reduce_is_prefix",
        "reduce_keepdims",
    ],
    "gather": [
        "gather_table_rows",
        "gather_row_elems",
        "gather_row_bytes",
        "gather_lookup_count_exact",
        "gather_unique_ratio",
        "gather_reuse_factor",
        "gather_table_coverage",
        "gather_ws_eff_rows",
        "gather_ws_eff_per_lookup",
        "gather_ws_eff_to_output_bytes",
    ],
    "concat": [
        "concat_input_count_exact",
        "concat_axis",
        "concat_axis_size_sum",
        "concat_axis_size_max",
        "concat_axis_size_min",
        "concat_input_numel_max",
        "concat_input_numel_min",
    ],
    "flatten": [
        "flatten_axis_exact",
        "flatten_left_numel",
        "flatten_right_numel",
    ],
    "reshape": [
        "reshape_rank_delta",
        "reshape_keeps_batch",
        "reshape_keeps_last_dim",
        "reshape_is_flatten2d",
        "reshape_input_rank",
        "reshape_output_rank",
    ],
    "unsqueeze": [
        "unsqueeze_rank_delta",
        "unsqueeze_axes_count_exact",
        "unsqueeze_prefix_insertions",
        "unsqueeze_suffix_insertions",
    ],
    "shape": [
        "shape_input_rank_exact",
        "shape_input_numel_exact",
        "shape_max_dim",
        "shape_min_dim",
    ],
    "gemm": [
        "gemm_m_over_n",
        "gemm_m_over_k",
        "gemm_n_over_k",
        "gemm_smallest_dim",
        "gemm_largest_dim",
    ],
}


ALL_DERIVED_FEATURES = [
    column
    for group_columns in DERIVED_FEATURE_GROUPS.values()
    for column in group_columns
]


ENHANCED_CLASSED_GROUP_FEATURES = {
    "gather": DERIVED_FEATURE_GROUPS["common"] + DERIVED_FEATURE_GROUPS["gather"],
    "layout_move": (
        DERIVED_FEATURE_GROUPS["common"]
        + DERIVED_FEATURE_GROUPS["concat"]
        + DERIVED_FEATURE_GROUPS["transpose"]
    ),
    "view_meta": (
        DERIVED_FEATURE_GROUPS["common"]
        + DERIVED_FEATURE_GROUPS["flatten"]
        + DERIVED_FEATURE_GROUPS["reshape"]
        + DERIVED_FEATURE_GROUPS["unsqueeze"]
        + DERIVED_FEATURE_GROUPS["shape"]
    ),
    "mixed_balanced": (
        DERIVED_FEATURE_GROUPS["common"]
        + DERIVED_FEATURE_GROUPS["binary"]
        + DERIVED_FEATURE_GROUPS["reduce"]
    ),
    "compute_dominant": DERIVED_FEATURE_GROUPS["common"] + DERIVED_FEATURE_GROUPS["gemm"],
}
