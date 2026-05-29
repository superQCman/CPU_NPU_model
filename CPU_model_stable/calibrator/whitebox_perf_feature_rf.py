from __future__ import annotations

import hashlib
import inspect
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


ANALYTICAL_PERF_FEATURE_SCHEMA_VERSION = 7
RF_FEATURE_LIMIT = 30
BENCHMARK_RF_FEATURE_LIMIT = 60
RF_TRAIN_N_JOBS = max(int(os.environ.get("CROSS_HW_RF_N_JOBS", "1")), 1)

FORBIDDEN_RF_FEATURES = {
    "output_size",
    "activation_size",
    "feat_io_bytes_sum",
    "feat_lookup_count",
    "gather_working_set_eff_bytes",
    "feat_reduction_work_items",
    "feat_reduction_axes_count",
    "feat_reduction_axes_product",
    "feat_gemm_mac_count",
    "exact_input_tensor_bytes_sum",
    "exact_output_tensor_bytes_sum",
    "exact_read_bytes",
    "exact_write_bytes",
    "exact_io_bytes",
}

BENCHMARK_WORKLOAD_FEATURES = {
    "output_size",
    "activation_size",
    "feat_io_bytes_sum",
    "feat_lookup_count",
    "gather_working_set_eff_bytes",
    "feat_reduction_work_items",
    "feat_reduction_axes_count",
    "feat_reduction_axes_product",
    "feat_gemm_mac_count",
    "exact_input_tensor_bytes_sum",
    "exact_output_tensor_bytes_sum",
    "exact_read_bytes",
    "exact_write_bytes",
    "gather_tail_copy_bytes",
    "gather_effective_head_touch_count",
    "gather_ort_task_count",
    "gather_ort_m_count",
    "gather_ort_block_size_bytes",
    "gather_ort_source_payload_bytes",
    "gather_ort_dest_payload_bytes",
    "gather_ort_index_scan_count",
    "gather_ort_grain_bytes",
    "gather_slice_copy_bw_gb_s",
    "gather_slice_stream_us",
    "gather_slice_vs_current_stream_ratio",
    "gather_slice_underpredict_gap_us",
    "reduce_partial_read_bytes",
    "reduce_partial_write_bytes",
    "reduce_inter_branch_merge_bytes",
    "reduce_ort_fast_kind_code",
    "reduce_ort_stream_bytes",
    "reduce_ort_input_read_bytes",
    "reduce_ort_output_write_bytes",
    "reduce_ort_grain_tasks",
    "reduce_ort_parallel_axis_extent",
    "num_threads",
    "inter_threads",
    "log_num_threads",
    "log_inter_threads",
    "total_thread_slots",
    "effective_worker_count",
    "active_parallel_groups",
    "thread_pressure_ratio",
    "parallel_exposure_factor",
    "is_single_intra",
    "is_single_inter",
}

CONTROLLED_THREADING_WORKLOAD_FEATURES = [
    "num_threads",
    "inter_threads",
    "log_num_threads",
    "log_inter_threads",
    "total_thread_slots",
    "effective_worker_count",
    "active_parallel_groups",
    "thread_pressure_ratio",
    "parallel_exposure_factor",
    "is_single_intra",
    "is_single_inter",
]

RF_GROUP_METADATA = "metadata_view_rf"
RF_GROUP_TINY_SHAPE = "tiny_shape_arith_rf"
RF_GROUP_INDEXED_COPY = "indexed_block_copy_rf"
RF_GROUP_PAYLOAD_CONCAT = "payload_concat_rf"
RF_GROUP_REDUCTION = "reduction_layout_rf"
RF_GROUP_TRANSPOSE = "transpose_generic_rf"
RF_GROUP_GEMM = "gemm_like_rf"

SAFE_LOG1P_FEATURES = {
    "whitebox_center_us",
    "whitebox_lower_us",
    "whitebox_upper_us",
    "whitebox_band_width_us",
    "whitebox_exec_lower_us",
    "whitebox_exec_center_us",
    "whitebox_exec_upper_us",
    "whitebox_serial_overhead_lower_us",
    "whitebox_serial_overhead_center_us",
    "whitebox_serial_overhead_upper_us",
    "svc_descriptor_center_us",
    "svc_stream_mem_center_us",
    "svc_random_mem_center_us",
    "svc_compute_center_us",
    "svc_reorder_center_us",
    "svc_sync_center_us",
    "svc_dispatch_center_us",
    "gather_slice_copy_bw_gb_s",
    "gather_slice_stream_us",
    "gather_slice_vs_current_stream_ratio",
    "gather_slice_underpredict_gap_us",
}

GENERIC_ANALYTICAL_FEATURES = [
    "whitebox_center_us",
    "whitebox_band_width_us",
    "whitebox_exec_center_us",
    "whitebox_serial_overhead_center_us",
    "share_descriptor_center",
    "share_stream_mem_center",
    "share_random_mem_center",
    "share_compute_center",
    "share_reorder_center",
    "share_sync_center",
    "share_dispatch_center",
    "bottleneck_margin_center",
    "launch_vs_useful_ratio_center",
    "random_vs_stream_ratio_center",
    "mem_vs_compute_ratio_center",
]

GENERIC_STATIC_SEMANTICS = [
    "feat_primary_dtype_class_code",
    "feat_broadcast_class_code",
    "feat_transpose_mode_class_code",
    "feat_reduction_layout_class_code",
    "feat_shape_metadata_class_code",
    "feat_tiny_shape_arith_flag",
    "merge_sync_obligation_flag",
    "submodel_role_static_code",
    "exec_region_static_code",
    "branch_lifetime_class_code",
    "reduce_partial_shape_static_code",
    "merge_fanin_static",
    "phase_barrier_count_static",
    "worker_shard_count_static",
]

GENERIC_GRAPH_POSITION = [
    "pred_op_type_primary_code",
    "succ_op_type_primary_code",
    "pred_exec_mode_primary_code",
    "succ_exec_mode_primary_code",
    "pred_count",
    "succ_count",
    "cross_scope_pred_count",
    "cross_scope_succ_count",
    "topo_pos_norm",
    "reverse_topo_pos_norm",
    "distance_to_output",
    "distance_to_nearest_merge",
    "distance_to_nearest_concat",
    "distance_to_nearest_reduce",
    "distance_to_nearest_gather",
    "nearest_merge_fanin",
    "static_is_embedding_branch_flag",
    "static_parallel_branch_count",
    "static_distance_to_merge",
    "graph_1hop_metadata_view_count",
    "graph_1hop_tiny_shape_arith_count",
    "graph_1hop_indexed_block_copy_count",
    "graph_1hop_payload_concat_count",
    "graph_1hop_reduction_layout_count",
    "graph_1hop_transpose_generic_count",
    "graph_1hop_gemm_like_count",
    "graph_2hop_metadata_view_count",
    "graph_2hop_tiny_shape_arith_count",
    "graph_2hop_indexed_block_copy_count",
    "graph_2hop_payload_concat_count",
    "graph_2hop_reduction_layout_count",
    "graph_2hop_transpose_generic_count",
    "graph_2hop_gemm_like_count",
]

DEFAULT_LOG1P_COLUMNS = [
    "whitebox_center_us",
    "whitebox_band_width_us",
    "whitebox_exec_center_us",
    "whitebox_serial_overhead_center_us",
]

DEFAULT_PASSTHROUGH_COLUMNS = [
    "share_descriptor_center",
    "share_stream_mem_center",
    "share_random_mem_center",
    "share_compute_center",
    "share_reorder_center",
    "share_sync_center",
    "share_dispatch_center",
    "bottleneck_margin_center",
    "launch_vs_useful_ratio_center",
    "random_vs_stream_ratio_center",
    "mem_vs_compute_ratio_center",
    "feat_primary_dtype_class_code",
    "feat_broadcast_class_code",
    "feat_transpose_mode_class_code",
    "feat_reduction_layout_class_code",
    "feat_shape_metadata_class_code",
    "feat_tiny_shape_arith_flag",
    "merge_sync_obligation_flag",
    "submodel_role_static_code",
    "exec_region_static_code",
    "branch_lifetime_class_code",
    "merge_fanin_static",
    "phase_barrier_count_static",
    "pred_exec_mode_primary_code",
    "succ_exec_mode_primary_code",
    "pred_count",
    "succ_count",
    "topo_pos_norm",
    "reverse_topo_pos_norm",
    "distance_to_output",
    "distance_to_nearest_merge",
]

DEFAULT_FEATURE_COLUMNS = [*DEFAULT_LOG1P_COLUMNS, *DEFAULT_PASSTHROUGH_COLUMNS]

MONOTONIC_POSITIVE_ANCHOR_FEATURES = {
    "whitebox_center_us",
    "whitebox_exec_center_us",
    "whitebox_serial_overhead_center_us",
    "svc_descriptor_center_us",
    "svc_stream_mem_center_us",
    "svc_random_mem_center_us",
    "svc_compute_center_us",
    "svc_reorder_center_us",
    "svc_sync_center_us",
    "svc_dispatch_center_us",
}

RF_GROUP_SPECS: dict[str, dict[str, Any]] = {
    RF_GROUP_METADATA: {
        "target_mode": "direct_log",
        "fallback_group": None,
        "mandatory": [
            "whitebox_center_us",
            "whitebox_band_width_us",
            "whitebox_exec_center_us",
            "whitebox_serial_overhead_center_us",
            "share_descriptor_center",
            "share_dispatch_center",
            "bottleneck_margin_center",
        ],
        "optional": [
            "feat_shape_metadata_class_code",
            "merge_sync_obligation_flag",
            "submodel_role_static_code",
            "exec_region_static_code",
            "branch_lifetime_class_code",
            "merge_fanin_static",
            "phase_barrier_count_static",
            "feat_shape_query_flag",
            "feat_rank_insert_flag",
            "feat_flatten_view_flag",
            "feat_reshape_view_or_copy_flag",
            "pred_op_type_primary_code",
            "succ_op_type_primary_code",
            "pred_exec_mode_primary_code",
            "succ_exec_mode_primary_code",
            "pred_count",
            "succ_count",
            "cross_scope_pred_count",
            "cross_scope_succ_count",
            "topo_pos_norm",
            "reverse_topo_pos_norm",
            "distance_to_output",
            "distance_to_nearest_merge",
            "static_is_embedding_branch_flag",
            "static_parallel_branch_count",
            "static_distance_to_merge",
        ],
        "benchmark_workload": [
            "output_size",
            "activation_size",
            "exact_input_tensor_bytes_sum",
            "exact_output_tensor_bytes_sum",
            "exact_read_bytes",
            "exact_write_bytes",
        ],
    },
    RF_GROUP_TINY_SHAPE: {
        "target_mode": "direct_log",
        "fallback_group": RF_GROUP_METADATA,
        "mandatory": [
            "whitebox_center_us",
            "whitebox_band_width_us",
            "whitebox_exec_center_us",
            "whitebox_serial_overhead_center_us",
            "share_descriptor_center",
            "share_dispatch_center",
            "bottleneck_margin_center",
            "svc_compute_center_us",
        ],
        "optional": [
            "feat_primary_dtype_class_code",
            "feat_broadcast_class_code",
            "feat_tiny_shape_arith_flag",
            "merge_sync_obligation_flag",
            "submodel_role_static_code",
            "exec_region_static_code",
            "branch_lifetime_class_code",
            "worker_shard_count_static",
            "pred_exec_mode_primary_code",
            "succ_exec_mode_primary_code",
            "pred_count",
            "succ_count",
            "topo_pos_norm",
            "distance_to_output",
            "distance_to_nearest_merge",
            "static_parallel_branch_count",
            "static_distance_to_merge",
        ],
        "benchmark_workload": [
            "output_size",
            "activation_size",
            "feat_io_bytes_sum",
            "exact_input_tensor_bytes_sum",
            "exact_output_tensor_bytes_sum",
            "exact_read_bytes",
            "exact_write_bytes",
        ],
    },
    RF_GROUP_INDEXED_COPY: {
        "target_mode": "analytical_residual",
        "fallback_group": None,
        "mandatory": [
            "whitebox_center_us",
            "whitebox_band_width_us",
            "whitebox_exec_center_us",
            "whitebox_serial_overhead_center_us",
            "share_stream_mem_center",
            "share_random_mem_center",
            "share_sync_center",
            "launch_vs_useful_ratio_center",
            "random_vs_stream_ratio_center",
        ],
        "optional": [
            "merge_sync_obligation_flag",
            "submodel_role_static_code",
            "exec_region_static_code",
            "branch_lifetime_class_code",
            "merge_fanin_static",
            "phase_barrier_count_static",
            "worker_shard_count_static",
            "pred_op_type_primary_code",
            "succ_op_type_primary_code",
            "pred_exec_mode_primary_code",
            "succ_exec_mode_primary_code",
            "pred_count",
            "succ_count",
            "cross_scope_pred_count",
            "cross_scope_succ_count",
            "distance_to_output",
            "distance_to_nearest_merge",
            "distance_to_nearest_reduce",
            "nearest_merge_fanin",
            "static_is_embedding_branch_flag",
            "static_parallel_branch_count",
            "static_distance_to_merge",
            "graph_1hop_reduction_layout_count",
            "graph_1hop_payload_concat_count",
        ],
        "benchmark_workload": [
            "feat_lookup_count",
            "gather_working_set_eff_bytes",
            "output_size",
            "exact_input_tensor_bytes_sum",
            "exact_output_tensor_bytes_sum",
            "exact_read_bytes",
            "exact_write_bytes",
            "gather_tail_copy_bytes",
            "gather_effective_head_touch_count",
            "gather_ort_task_count",
            "gather_ort_m_count",
            "gather_ort_block_size_bytes",
            "gather_ort_source_payload_bytes",
            "gather_ort_dest_payload_bytes",
            "gather_ort_index_scan_count",
            "gather_ort_grain_bytes",
            "gather_slice_copy_bw_gb_s",
            "gather_slice_stream_us",
            "gather_slice_vs_current_stream_ratio",
            "gather_slice_underpredict_gap_us",
            *CONTROLLED_THREADING_WORKLOAD_FEATURES,
        ],
    },
    RF_GROUP_PAYLOAD_CONCAT: {
        "target_mode": "analytical_residual",
        "fallback_group": None,
        "mandatory": [
            "whitebox_center_us",
            "whitebox_band_width_us",
            "whitebox_exec_center_us",
            "share_stream_mem_center",
            "share_descriptor_center",
            "share_sync_center",
            "launch_vs_useful_ratio_center",
        ],
        "optional": [
            "feat_concat_shape_flag",
            "merge_sync_obligation_flag",
            "submodel_role_static_code",
            "exec_region_static_code",
            "branch_lifetime_class_code",
            "merge_fanin_static",
            "phase_barrier_count_static",
            "pred_op_type_primary_code",
            "succ_op_type_primary_code",
            "pred_exec_mode_primary_code",
            "succ_exec_mode_primary_code",
            "pred_count",
            "succ_count",
            "cross_scope_pred_count",
            "cross_scope_succ_count",
            "distance_to_output",
            "distance_to_nearest_merge",
            "distance_to_nearest_concat",
            "nearest_merge_fanin",
            "static_parallel_branch_count",
            "static_distance_to_merge",
            "graph_1hop_reduction_layout_count",
        ],
        "benchmark_workload": [
            "output_size",
            "activation_size",
            "feat_io_bytes_sum",
            "exact_input_tensor_bytes_sum",
            "exact_output_tensor_bytes_sum",
            "exact_read_bytes",
            "exact_write_bytes",
        ],
    },
    RF_GROUP_REDUCTION: {
        "target_mode": "analytical_residual",
        "fallback_group": None,
        "mandatory": [
            "whitebox_center_us",
            "whitebox_band_width_us",
            "whitebox_exec_center_us",
            "whitebox_serial_overhead_center_us",
            "share_stream_mem_center",
            "share_compute_center",
            "share_reorder_center",
            "share_sync_center",
            "mem_vs_compute_ratio_center",
            "bottleneck_margin_center",
        ],
        "optional": [
            "feat_reduction_layout_class_code",
            "feat_reduction_middle_axis_flag",
            "feat_reduction_rank_collapse_flag",
            "feat_reduction_keepdims_flag",
            "merge_sync_obligation_flag",
            "submodel_role_static_code",
            "exec_region_static_code",
            "branch_lifetime_class_code",
            "reduce_partial_shape_static_code",
            "merge_fanin_static",
            "phase_barrier_count_static",
            "worker_shard_count_static",
            "pred_op_type_primary_code",
            "succ_op_type_primary_code",
            "pred_exec_mode_primary_code",
            "succ_exec_mode_primary_code",
            "pred_count",
            "succ_count",
            "cross_scope_pred_count",
            "cross_scope_succ_count",
            "distance_to_output",
            "distance_to_nearest_merge",
            "distance_to_nearest_reduce",
            "nearest_merge_fanin",
            "static_parallel_branch_count",
            "static_distance_to_merge",
        ],
        "benchmark_workload": [
            "output_size",
            "activation_size",
            "feat_io_bytes_sum",
            "feat_reduction_work_items",
            "feat_reduction_axes_count",
            "feat_reduction_axes_product",
            "exact_input_tensor_bytes_sum",
            "exact_output_tensor_bytes_sum",
            "exact_read_bytes",
            "exact_write_bytes",
            "reduce_partial_read_bytes",
            "reduce_partial_write_bytes",
            "reduce_inter_branch_merge_bytes",
            "reduce_ort_fast_kind_code",
            "reduce_ort_stream_bytes",
            "reduce_ort_input_read_bytes",
            "reduce_ort_output_write_bytes",
            "reduce_ort_grain_tasks",
            "reduce_ort_parallel_axis_extent",
            *CONTROLLED_THREADING_WORKLOAD_FEATURES,
        ],
    },
    RF_GROUP_TRANSPOSE: {
        "target_mode": "analytical_residual",
        "fallback_group": RF_GROUP_PAYLOAD_CONCAT,
        "mandatory": [
            "whitebox_center_us",
            "whitebox_band_width_us",
            "whitebox_exec_center_us",
            "share_stream_mem_center",
            "share_reorder_center",
            "share_descriptor_center",
            "bottleneck_margin_center",
        ],
        "optional": [
            "feat_transpose_mode_class_code",
            "feat_transpose_generic_flag",
            "submodel_role_static_code",
            "exec_region_static_code",
            "branch_lifetime_class_code",
            "pred_op_type_primary_code",
            "succ_op_type_primary_code",
            "pred_exec_mode_primary_code",
            "succ_exec_mode_primary_code",
            "pred_count",
            "succ_count",
            "cross_scope_pred_count",
            "cross_scope_succ_count",
            "distance_to_output",
            "distance_to_nearest_merge",
            "distance_to_nearest_concat",
            "static_parallel_branch_count",
            "static_distance_to_merge",
        ],
        "benchmark_workload": [
            "output_size",
            "activation_size",
            "feat_io_bytes_sum",
            "exact_input_tensor_bytes_sum",
            "exact_output_tensor_bytes_sum",
            "exact_read_bytes",
            "exact_write_bytes",
        ],
    },
    RF_GROUP_GEMM: {
        "target_mode": "analytical_residual",
        "fallback_group": None,
        "mandatory": [
            "whitebox_center_us",
            "whitebox_band_width_us",
            "whitebox_exec_center_us",
            "share_stream_mem_center",
            "share_compute_center",
            "bottleneck_margin_center",
        ],
        "optional": [
            "feat_primary_dtype_class_code",
            "submodel_role_static_code",
            "exec_region_static_code",
            "branch_lifetime_class_code",
            "pred_op_type_primary_code",
            "succ_op_type_primary_code",
            "pred_exec_mode_primary_code",
            "succ_exec_mode_primary_code",
            "pred_count",
            "succ_count",
            "cross_scope_pred_count",
            "cross_scope_succ_count",
            "distance_to_output",
            "distance_to_nearest_merge",
            "static_parallel_branch_count",
            "static_distance_to_merge",
            "graph_1hop_metadata_view_count",
            "graph_1hop_reduction_layout_count",
        ],
        "benchmark_workload": [
            "output_size",
            "activation_size",
            "feat_io_bytes_sum",
            "feat_gemm_mac_count",
            "exact_input_tensor_bytes_sum",
            "exact_output_tensor_bytes_sum",
            "exact_read_bytes",
            "exact_write_bytes",
        ],
    },
}


def _stable_feature_hash(columns: list[str]) -> str:
    return hashlib.sha256("||".join(columns).encode("utf-8")).hexdigest()


def build_preprocess_state(
    feature_columns: list[str] | None = None,
    *,
    log1p_columns: list[str] | None = None,
) -> dict[str, Any]:
    columns = list(dict.fromkeys(feature_columns or DEFAULT_FEATURE_COLUMNS))
    log1p = [column for column in (log1p_columns or DEFAULT_LOG1P_COLUMNS) if column in columns]
    passthrough = [column for column in columns if column not in log1p]
    return {
        "version": ANALYTICAL_PERF_FEATURE_SCHEMA_VERSION,
        "feature_columns": columns,
        "log1p_columns": log1p,
        "passthrough_columns": passthrough,
        "input_dim": len(columns),
        "schema_sha256": _stable_feature_hash(columns),
    }


def transform_analytical_features(frame: pd.DataFrame, preprocess_state: dict[str, Any]) -> np.ndarray:
    columns = [str(column) for column in preprocess_state.get("feature_columns", [])]
    log1p_columns = set(str(column) for column in preprocess_state.get("log1p_columns", []))
    matrix = np.zeros((len(frame), len(columns)), dtype=np.float32)
    for index, column in enumerate(columns):
        values = pd.to_numeric(frame.get(column, pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
        values = values.replace([np.inf, -np.inf], 0.0).astype(float)
        array = values.to_numpy(dtype=np.float32)
        if column in log1p_columns:
            array = np.log1p(np.clip(array, a_min=0.0, a_max=None)).astype(np.float32)
        matrix[:, index] = array
    return matrix


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    denominator = np.clip(np.asarray(actual, dtype=np.float64), a_min=1e-6, a_max=None)
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - denominator) / denominator))


def _prediction_clip_bounds(frame: pd.DataFrame, *, clip_k: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    lower = pd.to_numeric(frame.get("whitebox_lower_us", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    upper = pd.to_numeric(frame.get("whitebox_upper_us", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    band = pd.to_numeric(frame.get("whitebox_band_width_us", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    clipped_lower = np.clip(lower - clip_k * band, a_min=0.0, a_max=None)
    clipped_upper = np.maximum(upper + clip_k * band, clipped_lower + 1e-6)
    return clipped_lower.astype(np.float32), clipped_upper.astype(np.float32)


def _group_validation_masks(frame: pd.DataFrame, group_columns: list[str] | None, *, fold_count: int = 3) -> list[np.ndarray]:
    requested = [str(column) for column in (group_columns or []) if str(column) in frame.columns]
    if not requested:
        return []
    key_frame = frame[requested].copy()
    for column in requested:
        key_frame[column] = key_frame[column].fillna("__nan__").astype(str)
    group_key = key_frame.agg("::".join, axis=1)
    unique_groups = sorted(group_key.unique().tolist())
    if len(unique_groups) < 3:
        return []
    actual_fold_count = max(1, min(int(fold_count), len(unique_groups)))
    masks: list[np.ndarray] = []
    for fold_idx in range(actual_fold_count):
        selected_groups = unique_groups[fold_idx::actual_fold_count]
        mask = group_key.isin(selected_groups).to_numpy(dtype=bool)
        if mask.sum() >= 64 and (~mask).sum() >= 64:
            masks.append(mask)
    return masks


def rf_group_spec(group_name: str) -> dict[str, Any]:
    return dict(RF_GROUP_SPECS.get(str(group_name), RF_GROUP_SPECS[RF_GROUP_METADATA]))


def _feature_is_usable(frame: pd.DataFrame, column: str, *, allow_constant: bool) -> bool:
    if column not in frame.columns:
        return False
    series = pd.to_numeric(frame[column], errors="coerce")
    if series.isna().mean() > 0.95:
        return False
    if not allow_constant and series.fillna(0.0).nunique(dropna=False) <= 1:
        return False
    return True


def _feature_category(column: str) -> str:
    if column in BENCHMARK_WORKLOAD_FEATURES:
        return "workload_performance"
    if column in GENERIC_ANALYTICAL_FEATURES or column in SAFE_LOG1P_FEATURES:
        return "analytical_anchor"
    if column in GENERIC_GRAPH_POSITION:
        return "graph_position"
    return "local_static_semantics"


def rf_group_feature_schema(
    group_name: str,
    frame: pd.DataFrame,
    *,
    allow_workload_features: bool = False,
    workload_feature_groups: set[str] | None = None,
    feature_limit: int = RF_FEATURE_LIMIT,
    feature_include: list[str] | None = None,
    feature_exclude: set[str] | None = None,
    target_mode_override: str | None = None,
) -> dict[str, Any]:
    spec = rf_group_spec(group_name)
    excluded = set(str(column) for column in (feature_exclude or set()))
    mandatory = [
        column
        for column in spec.get("mandatory", [])
        if column not in excluded and _feature_is_usable(frame, column, allow_constant=True)
    ]
    optional_candidates = list(spec.get("optional", []))
    group_allows_workload = bool(allow_workload_features or str(group_name) in set(workload_feature_groups or set()))
    if group_allows_workload:
        optional_candidates = list(dict.fromkeys([*spec.get("benchmark_workload", []), *optional_candidates]))
    optional = [
        column
        for column in optional_candidates
        if column not in mandatory and column not in excluded and _feature_is_usable(frame, column, allow_constant=False)
    ]
    if feature_include is not None:
        include_order = list(dict.fromkeys(str(column) for column in feature_include if str(column) not in excluded))
        optional_by_name = {column: column for column in optional}
        selected_optional = [
            optional_by_name[column]
            for column in include_order
            if column in optional_by_name and column not in mandatory
        ]
        selected = mandatory + selected_optional
    else:
        selected = mandatory + optional[: max(int(feature_limit) - len(mandatory), 0)]
    forbidden_selected = [column for column in selected if column in FORBIDDEN_RF_FEATURES]
    if group_allows_workload:
        forbidden_selected = [column for column in forbidden_selected if column not in BENCHMARK_WORKLOAD_FEATURES]
    if forbidden_selected:
        raise ValueError(
            f"RF group {group_name} selected forbidden workload-size features: {', '.join(sorted(forbidden_selected))}"
        )
    if len(selected) > int(feature_limit):
        raise ValueError(f"RF group {group_name} exceeded feature budget: {len(selected)} > {int(feature_limit)}")
    log1p_columns = [column for column in selected if column in SAFE_LOG1P_FEATURES]
    candidate_rows: list[dict[str, Any]] = []
    for column in [*mandatory, *optional]:
        candidate_rows.append(
            {
                "rf_group": str(group_name),
                "feature_name": str(column),
                "is_selected": bool(column in selected),
                "is_mandatory": bool(column in mandatory),
                "feature_category": _feature_category(column),
            }
        )
    selected_manifest = [
        {
            "rf_group": str(group_name),
            "feature_name": str(column),
            "feature_category": _feature_category(column),
            "log1p": bool(column in log1p_columns),
        }
        for column in selected
    ]
    return {
        "feature_columns": selected,
        "log1p_columns": log1p_columns,
        "candidate_rows": candidate_rows,
        "selected_rows": selected_manifest,
        "feature_count": int(len(selected)),
        "feature_limit": int(feature_limit),
        "allow_workload_features": bool(group_allows_workload),
        "target_mode": str(target_mode_override or spec.get("target_mode", "analytical_residual")),
        "fallback_group": spec.get("fallback_group"),
        "schema_sha256": _stable_feature_hash(selected),
    }


def fit_family_feature_rf(
    family_frame: pd.DataFrame,
    *,
    family_name: str,
    feature_columns: list[str] | None = None,
    log1p_columns: list[str] | None = None,
    target_column: str,
    baseline_column: str,
    n_estimators: int = 384,
    max_depth: int | None = 10,
    min_samples_leaf: int = 12,
    max_features: str | float | int | None = "sqrt",
    min_rows: int = 96,
    min_relative_improvement: float = 0.03,
    seed: int = 0,
    validation_group_columns: list[str] | None = None,
    target_mode: str = "analytical_residual",
    prediction_clip_k: float = 1.0,
    sample_weight_column: str | None = None,
    monotonic_feature_columns: set[str] | list[str] | None = None,
    synthetic_indicator_column: str | None = None,
    synthetic_source_index_column: str | None = None,
) -> dict[str, Any]:
    if len(family_frame) < int(min_rows):
        return {
            "family_name": str(family_name),
            "enabled": False,
            "rows": int(len(family_frame)),
            "reason": "insufficient_rows",
        }

    work = family_frame.copy().reset_index(drop=True)
    synthetic_indicator = str(synthetic_indicator_column or "")
    synthetic_source_index = str(synthetic_source_index_column or "")
    if synthetic_indicator and synthetic_indicator in work.columns:
        synthetic_mask = work[synthetic_indicator].astype(bool).to_numpy(dtype=bool)
    else:
        synthetic_mask = np.zeros(len(work), dtype=bool)
    if synthetic_source_index and synthetic_source_index in work.columns:
        fallback_source_index = pd.Series(np.arange(len(work), dtype=int), index=work.index)
        source_index = (
            pd.to_numeric(work[synthetic_source_index], errors="coerce")
            .fillna(fallback_source_index)
            .astype(int)
            .to_numpy(dtype=int)
        )
    else:
        source_index = np.arange(len(work), dtype=int)
    real_work = work.loc[~synthetic_mask].reset_index(drop=True)
    real_source_index = source_index[~synthetic_mask]
    if real_work.empty:
        return {
            "family_name": str(family_name),
            "enabled": False,
            "rows": int(len(work)),
            "reason": "no_real_rows",
        }
    min_valid_rows = max(24, min(64, int(round(len(work) * 0.15))))
    min_train_rows = max(48, int(min_rows) - min_valid_rows)
    validation_masks = _group_validation_masks(real_work, validation_group_columns, fold_count=3)
    if not validation_masks:
        validation_masks = []
        row_ids = np.arange(len(real_work), dtype=int)
        for fold_idx in range(3):
            mask = row_ids % 3 == fold_idx
            if mask.sum() >= min_valid_rows and (~mask).sum() >= min_train_rows:
                validation_masks.append(mask)

    fold_metrics: list[dict[str, float | int]] = []
    for validation_mask in validation_masks:
        if validation_mask.sum() < min_valid_rows or (~validation_mask).sum() < min_train_rows:
            continue
        valid_source_ids = set(real_source_index[validation_mask].tolist())
        train_mask = np.asarray([source not in valid_source_ids for source in source_index], dtype=bool)
        train_subset = work.loc[train_mask].reset_index(drop=True)
        valid_subset = real_work.loc[validation_mask].reset_index(drop=True)
        preprocess_state = build_preprocess_state(feature_columns=feature_columns, log1p_columns=log1p_columns)
        train_features = transform_analytical_features(train_subset, preprocess_state)
        valid_features = transform_analytical_features(valid_subset, preprocess_state)
        train_actual = np.clip(
            pd.to_numeric(train_subset[target_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            a_min=0.0,
            a_max=None,
        )
        train_baseline = np.clip(
            pd.to_numeric(train_subset[baseline_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            a_min=1e-6,
            a_max=None,
        )
        if str(target_mode) == "direct_log":
            train_targets = np.log1p(np.clip(train_actual, a_min=0.0, a_max=None)).astype(np.float32)
        else:
            train_targets = np.log(np.clip(train_actual, a_min=1e-6, a_max=None) / train_baseline).astype(np.float32)
        train_weights = None
        if sample_weight_column and sample_weight_column in train_subset.columns:
            raw_weight = pd.to_numeric(train_subset[sample_weight_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            positive = raw_weight[raw_weight > 0.0]
            if positive.size:
                median_weight_source = float(np.median(positive))
                train_weights = np.clip(np.clip(raw_weight, a_min=1e-6, a_max=None) / median_weight_source, 0.5, 8.0)
        valid_actual = np.clip(
            pd.to_numeric(valid_subset[target_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            a_min=0.0,
            a_max=None,
        )
        valid_baseline = np.clip(
            pd.to_numeric(valid_subset[baseline_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            a_min=0.0,
            a_max=None,
        )
        monotonic_set = set(str(column) for column in (monotonic_feature_columns or set()))
        monotonic_cst = [1 if str(column) in monotonic_set else 0 for column in preprocess_state["feature_columns"]]
        model_kwargs: dict[str, Any] = {}
        if any(monotonic_cst) and "monotonic_cst" in inspect.signature(RandomForestRegressor).parameters:
            model_kwargs["monotonic_cst"] = monotonic_cst
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth is None else int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=int(seed),
            n_jobs=int(RF_TRAIN_N_JOBS),
            **model_kwargs,
        )
        model.fit(train_features, train_targets, sample_weight=train_weights)

        clip_lower, clip_upper = _prediction_clip_bounds(valid_subset, clip_k=float(prediction_clip_k))
        raw_pred = model.predict(valid_features).astype(np.float32)
        if str(target_mode) == "direct_log":
            direct_pred = np.expm1(raw_pred).astype(np.float32)
            valid_pred = np.clip(direct_pred, a_min=clip_lower, a_max=clip_upper)
        else:
            valid_ratio = np.exp(raw_pred).astype(np.float32)
            valid_pred = np.clip(valid_baseline * valid_ratio, a_min=clip_lower, a_max=clip_upper)
        fold_metrics.append(
            {
                "baseline_mape": float(_mape(valid_actual, valid_baseline)),
                "model_mape": float(_mape(valid_actual, valid_pred)),
                "train_rows": int(len(train_subset)),
                "valid_rows": int(len(valid_subset)),
            }
        )

    if not fold_metrics:
        return {
            "family_name": str(family_name),
            "enabled": False,
            "rows": int(len(work)),
            "reason": "insufficient_validation_rows",
        }
    baseline_mape = float(np.mean([float(item["baseline_mape"]) for item in fold_metrics]))
    final_valid_mape = float(np.mean([float(item["model_mape"]) for item in fold_metrics]))
    improvement = float(baseline_mape - final_valid_mape)
    relative_improvement = improvement / max(float(baseline_mape), 1e-9)
    pass_count = int(
        sum(
            1
            for item in fold_metrics
            if float(item["model_mape"]) <= float(item["baseline_mape"]) * (1.0 - float(min_relative_improvement))
        )
    )
    enabled = bool(
        pass_count >= max(int(np.ceil(len(fold_metrics) / 2.0)), 1)
        and final_valid_mape <= baseline_mape * (1.0 - float(min_relative_improvement))
    )
    reason = "enabled" if enabled else "insufficient_validation_improvement"

    preprocess_state = build_preprocess_state(feature_columns=feature_columns, log1p_columns=log1p_columns)
    all_features = transform_analytical_features(work, preprocess_state)
    all_actual = np.clip(
        pd.to_numeric(work[target_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        a_min=0.0,
        a_max=None,
    )
    all_baseline = np.clip(
        pd.to_numeric(work[baseline_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        a_min=1e-6,
        a_max=None,
    )
    if str(target_mode) == "direct_log":
        final_targets = np.log1p(np.clip(all_actual, a_min=0.0, a_max=None)).astype(np.float32)
    else:
        final_targets = np.log(np.clip(all_actual, a_min=1e-6, a_max=None) / all_baseline).astype(np.float32)
    final_weights = None
    if sample_weight_column and sample_weight_column in work.columns:
        raw_weight = pd.to_numeric(work[sample_weight_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        positive = raw_weight[raw_weight > 0.0]
        if positive.size:
            median_weight_source = float(np.median(positive))
            final_weights = np.clip(np.clip(raw_weight, a_min=1e-6, a_max=None) / median_weight_source, 0.5, 8.0)

    monotonic_set = set(str(column) for column in (monotonic_feature_columns or set()))
    monotonic_cst = [1 if str(column) in monotonic_set else 0 for column in preprocess_state["feature_columns"]]
    model_kwargs = {}
    if any(monotonic_cst) and "monotonic_cst" in inspect.signature(RandomForestRegressor).parameters:
        model_kwargs["monotonic_cst"] = monotonic_cst
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth is None else int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        random_state=int(seed),
        n_jobs=int(RF_TRAIN_N_JOBS),
        **model_kwargs,
    )
    model.fit(all_features, final_targets, sample_weight=final_weights)

    importances = getattr(model, "feature_importances_", None)
    top_features: list[dict[str, Any]] = []
    if importances is not None:
        pairs = sorted(
            zip(preprocess_state["feature_columns"], np.asarray(importances, dtype=float).tolist()),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        top_features = [{"feature": str(name), "importance": float(score)} for name, score in pairs]

    summary = {
        "family_name": str(family_name),
        "enabled": enabled,
        "reason": reason,
        "rows": int(len(work)),
        "train_rows": int(round(float(np.mean([int(item["train_rows"]) for item in fold_metrics])))),
        "valid_rows": int(round(float(np.mean([int(item["valid_rows"]) for item in fold_metrics])))),
        "baseline_valid_mape": float(baseline_mape),
        "model_valid_mape": float(final_valid_mape),
        "relative_improvement": float(relative_improvement),
        "validation_fold_count": int(len(fold_metrics)),
        "validation_pass_count": int(pass_count),
        "n_estimators": int(n_estimators),
        "max_depth": None if max_depth is None else int(max_depth),
        "min_samples_leaf": int(min_samples_leaf),
        "max_features": max_features,
        "feature_count": int(preprocess_state["input_dim"]),
        "top_feature_importances": top_features,
        "baseline_column": str(baseline_column),
        "target_mode": str(target_mode),
        "validation_group_columns": list(validation_group_columns or []),
        "prediction_clip_k": float(prediction_clip_k),
        "sample_weight_column": str(sample_weight_column or ""),
        "monotonic_feature_columns": sorted(monotonic_set),
        "monotonic_cst_enabled": bool(any(monotonic_cst) and "monotonic_cst" in model_kwargs),
        "synthetic_training_rows": int(synthetic_mask.sum()),
        "real_training_rows": int((~synthetic_mask).sum()),
        "schema_sha256": str(preprocess_state.get("schema_sha256", "")),
    }
    if not enabled:
        summary["model"] = None
        summary["preprocess_state"] = preprocess_state
        return summary

    summary["model"] = model
    summary["preprocess_state"] = preprocess_state
    return summary


def predict_family_feature_rf(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    model = bundle.get("model")
    preprocess_state = bundle.get("preprocess_state", {})
    if model is None:
        raise ValueError("Family RF bundle is missing a trained model")
    features = transform_analytical_features(frame, preprocess_state)
    raw_pred = np.asarray(model.predict(features), dtype=np.float32)
    target_mode = str(bundle.get("target_mode", "analytical_residual"))
    clip_lower, clip_upper = _prediction_clip_bounds(frame, clip_k=float(bundle.get("prediction_clip_k", 1.0)))
    if target_mode == "direct_log":
        direct = np.expm1(raw_pred).astype(np.float32)
        return np.clip(direct, a_min=clip_lower, a_max=clip_upper)
    baseline_column = str(bundle.get("baseline_column", "whitebox_center_us"))
    baseline = np.clip(
        pd.to_numeric(frame.get(baseline_column, pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        a_min=1e-6,
        a_max=None,
    )
    return np.clip(baseline * np.exp(raw_pred).astype(np.float32), a_min=clip_lower, a_max=clip_upper)
