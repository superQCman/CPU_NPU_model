from __future__ import annotations

TARGET_COLUMN = "label_operator_actual_dur_us"
ANALYTICAL_RESIDUAL_TARGET_COLUMN = "label_operator_residual_log"
GROUP_COLUMN = "sample_group"
PROFILE_INSTABILITY_METRICS = (
    "kept_range_ratio",
    "kept_cv",
    "last2_range_ratio",
    "last2_cv",
)

DEFAULT_FEATURE_DIALECT = "trace"
FEATURE_DIALECTS = ("trace", "no_trace")

BASELINE_CATEGORICAL_FEATURES = (
    "op_type",
    "node_scope",
    "node_name_normalized",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
)

DIRECT_ANALYTICAL_NUMERIC_FEATURES = (
    "ana_cache_fit_level",
    "ana_expected_latency_ns",
)

SHARED_BASELINE_NUMERIC_FEATURES = (
    "batch_size",
    "num_indices_per_lookup",
    "num_threads",
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
    "hw_ratio_working_set_to_l1d_active_bytes",
    "hw_ratio_working_set_to_l2_active_bytes",
    "hw_ratio_working_set_to_l3_active_bytes",
    "local_ctx_same_op_overlap_ratio_mean",
    "comp_feat_pressure_ws_to_l2_ratio",
    "comp_feat_pressure_ws_to_l3_ratio",
    *DIRECT_ANALYTICAL_NUMERIC_FEATURES,
)

TRACE_ONLY_NUMERIC_FEATURES = (
    "load_store_ratio",
    "feat_memops_per_inst",
    "feat_insts_per_thread",
    "reuse_time_mean",
    "reuse_distance_mean",
    "reuse_distance_unique_cache_lines_per_k_accesses",
    "opc_branch_ratio",
    "opc_fp_math_ratio",
    "opc_load_ratio",
    "opc_math_ratio",
    "opc_simd_ratio",
    "opc_store_ratio",
)

ANALYTICAL_RESIDUAL_BASELINE_NUMERIC_FEATURES = (
    "ana_compute_ops",
    "ana_roofline_base_us",
    "ana_base_us",
)

ANALYTICAL_ANALYSIS_ONLY_NUMERIC_FEATURES = (
    "ana_mem_bw_time_us",
    "ana_latency_proxy_us",
    "ana_ridge_gap",
)

STAGE2_CANDIDATE_NUMERIC_FEATURES = (
    "hw_ratio_threads_to_total_cores",
    "local_ctx_overlap_ratio_mean",
    "local_ctx_cross_task_overlap_ratio_mean",
    "local_ctx_mean_other_active_mean",
    "local_ctx_mean_other_tasks_mean",
    "combo_task_parallel_fraction",
    "combo_task_weighted_mean_parallel_concurrency",
    "combo_op_parallel_fraction",
    "combo_op_weighted_mean_parallel_concurrency",
    "comp_feat_pressure_threads",
)

TRACE_FEATURE_SOURCE_COLUMNS = (
    "trace_op_name",
    "total_instructions",
    "total_loads",
    "total_stores",
    "load_store_ratio",
    "reuse_time_mean",
    "reuse_distance_mean",
    "reuse_distance_unique_cache_lines_per_k_accesses",
    "opc_branch_ratio",
    "opc_fp_math_ratio",
    "opc_load_ratio",
    "opc_math_ratio",
    "opc_simd_ratio",
    "opc_store_ratio",
)


def baseline_numeric_features_for_dialect(feature_dialect: str) -> tuple[str, ...]:
    if feature_dialect == "trace":
        return SHARED_BASELINE_NUMERIC_FEATURES + TRACE_ONLY_NUMERIC_FEATURES
    if feature_dialect == "no_trace":
        return SHARED_BASELINE_NUMERIC_FEATURES
    raise ValueError(
        f"Unsupported feature dialect {feature_dialect!r}; expected one of {FEATURE_DIALECTS}"
    )


def analysis_numeric_features_for_dialect(feature_dialect: str) -> list[str]:
    return list(
        baseline_numeric_features_for_dialect(feature_dialect)
        + ANALYTICAL_RESIDUAL_BASELINE_NUMERIC_FEATURES
        + ANALYTICAL_ANALYSIS_ONLY_NUMERIC_FEATURES
        + STAGE2_CANDIDATE_NUMERIC_FEATURES
    )


def feature_columns_for_dialect(feature_dialect: str) -> list[str]:
    return list(BASELINE_CATEGORICAL_FEATURES + baseline_numeric_features_for_dialect(feature_dialect))


def dataset_numeric_columns_for_dialect(feature_dialect: str) -> list[str]:
    return analysis_numeric_features_for_dialect(feature_dialect)


BASELINE_NUMERIC_FEATURES = baseline_numeric_features_for_dialect(DEFAULT_FEATURE_DIALECT)
FEATURE_COLUMNS = feature_columns_for_dialect(DEFAULT_FEATURE_DIALECT)
ANALYSIS_NUMERIC_FEATURES = analysis_numeric_features_for_dialect(DEFAULT_FEATURE_DIALECT)
DATASET_NUMERIC_COLUMNS = dataset_numeric_columns_for_dialect(DEFAULT_FEATURE_DIALECT)

METADATA_COLUMNS = (
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
    "node_name",
    "trace_op_name",
    "feature_dialect_observed",
    "has_cpu_profile",
    "input_type_shape",
    "output_type_shape",
    "op_attrs_json",
    "attr_axis",
    "attr_perm_json",
    "attr_axes_json",
    "attr_keepdims",
    "gather_unique_idx_count",
    "gather_working_set_eff_bytes",
    "profile_batch_count_total",
    "profile_batch_count_kept",
    "profile_dropped_batch_indices",
    "profile_kept_batch_indices",
    "profile_label_all_batch_mean_us",
    "profile_label_kept_batch_mean_us",
    "profile_label_kept_batch_median_us",
    "profile_kept_abs_diff_us",
    "profile_kept_range_ratio",
    "profile_kept_cv",
    "profile_last2_abs_diff_us",
    "profile_last2_range_ratio",
    "profile_last2_cv",
)

SUPPORTED_TARGET_COLUMNS = (
    TARGET_COLUMN,
    ANALYTICAL_RESIDUAL_TARGET_COLUMN,
)

DEFAULT_SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}
