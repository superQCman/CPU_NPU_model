from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from analytical_base import run as base_run
try:
    from .static_feature_contract import augment_static_feature_frame
except ImportError:
    from static_feature_contract import augment_static_feature_frame  # type: ignore


ML_RF_ROUTE = "ml_semantic_thread_rf"
ML_RF_SHAPE_TREE_ROUTE = "ml_semantic_thread_shape_tree_plus_lookup"
ML_FAMILY_RF_ROUTE = "ml_semantic_thread_family_rf"
WHITEBOX_PRIOR_ONLY_ROUTE = "whitebox_family_prior_only_v2"
WHITEBOX_PRIOR_SIMPLE_RF_ROUTE = "whitebox_family_prior_plus_simple_rf_v2"
WHITEBOX_PHYSICS_GUARDED_RF_ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"
WHITEBOX_ORT_KERNEL_RF_ROUTE = "whitebox_family_prior_plus_ort_kernel_rf_v3"
WHITEBOX_EMBEDDING_MEM_RF_ROUTE = "whitebox_family_prior_plus_embedding_mem_rf_v4"
WHITEBOX_GATHER_SLICE_RF_ROUTE = "whitebox_family_prior_plus_gather_slice_rf_v5"
WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE = "whitebox_family_prior_plus_gather_slice_rf_v5_compact_probe"
WHITEBOX_WORKLOAD_BENCHMARK_ROUTE = "whitebox_family_prior_plus_workload_rf_benchmark_v1"

CASE_ARCH_MAP: dict[str, dict[str, str]] = {
    "case_7": {
        "arch_embedding_size": "25600-25600-25600-25600-25600-25600-25600-25600",
        "arch_mlp_bot": "400-300-200",
        "arch_mlp_top": "1600-800-1",
    },
    "case_8": {
        "arch_embedding_size": "25600-25600-25600-25600-25600-25600-25600-25600",
        "arch_mlp_bot": "800-800-400",
        "arch_mlp_top": "1600-800-1",
    },
    "case_9": {
        "arch_embedding_size": "51200-51200-51200-51200-51200-51200-51200-51200",
        "arch_mlp_bot": "400-300-200",
        "arch_mlp_top": "1600-800-1",
    },
    "case_10": {
        "arch_embedding_size": "51200-51200-51200-51200-51200-51200-51200-51200",
        "arch_mlp_bot": "800-800-400",
        "arch_mlp_top": "1600-800-1",
    },
}

ML_COMMON_CATEGORICAL = list(
    dict.fromkeys(
        list(base_run.PER_OP_COMMON_CATEGORICAL)
        + [
            "node_scope",
            "node_name_normalized",
            "arch_embedding_size",
            "arch_mlp_bot",
            "arch_mlp_top",
            "static_role",
            "thread_regime_bucket",
        ]
    )
)

ML_COMMON_NUMERIC = list(
    dict.fromkeys(
        list(base_run.PER_OP_COMMON_NUMERIC)
        + [
            "total_threads",
            "threads_per_core",
            "log_num_threads",
            "log_inter_threads",
            "log_total_threads",
            "is_single_intra",
            "is_single_inter",
            "ana_v2_total_us",
            "ana_v2_mem_us",
            "ana_v2_compute_us",
            "ana_v2_overhead_us",
            "ana_v2_mem_gain",
            "ana_v2_compute_gain",
            "ana_v2_arch_embedding_factor",
            "ana_v2_arch_mlp_bot_factor",
            "ana_v2_copy_thread_penalty",
            "ana_v2_total_over_ana_ratio",
            "ana_v2_mem_over_ana_ratio",
            "ana_v2_compute_over_ana_ratio",
        ]
    )
)

FAMILY_CAPS = json.loads(
    (Path(__file__).resolve().parent / "configs" / "defaults.json").read_text(encoding="utf-8")
)["family_caps"]

FAMILY_ROUTE_SPECS: dict[str, dict[str, Any]] = {
    "gather_like": {
        "op_types": {"Gather"},
        "target_mode": "analytical_residual",
        "categorical": ["op_type", "node_scope", "node_name_normalized", "arch_embedding_size", "arch_mlp_bot", "arch_mlp_top", "static_role"],
        "numeric": [
            "batch_size",
            "num_indices_per_lookup",
            "output_size",
            "activation_size",
            "feat_lookup_count",
            "gather_unique_idx_count",
            "gather_working_set_eff_bytes",
            "total_threads",
            "threads_per_core",
            "log_num_threads",
            "log_inter_threads",
            "log_total_threads",
            "ana_calib_total_us",
            "ana_calib_mem_us",
            "ana_calib_compute_us",
            "ana_v2_total_us",
            "ana_v2_mem_us",
            "ana_v2_compute_us",
            "ana_v2_mem_gain",
            "ana_v2_compute_gain",
            "ana_v2_arch_embedding_factor",
            "ana_v2_arch_mlp_bot_factor",
            "ana_v2_total_over_ana_ratio",
            "ana_v2_mem_over_ana_ratio",
            "ana_v2_compute_over_ana_ratio",
        ],
    },
    "reduce_like": {
        "op_types": {"ReduceSum"},
        "target_mode": "analytical_residual",
        "categorical": ["op_type", "node_scope", "node_name_normalized", "arch_embedding_size", "arch_mlp_bot", "arch_mlp_top", "static_role"],
        "numeric": [
            "output_size",
            "activation_size",
            "feat_reduction_axes_count",
            "feat_reduction_axes_product",
            "feat_reduction_work_items",
            "feat_reduction_input_rank",
            "feat_reduction_output_rank",
            "total_threads",
            "threads_per_core",
            "log_num_threads",
            "log_inter_threads",
            "log_total_threads",
            "ana_calib_total_us",
            "ana_calib_mem_us",
            "ana_calib_compute_us",
            "ana_v2_total_us",
            "ana_v2_mem_us",
            "ana_v2_compute_us",
            "ana_v2_mem_gain",
            "ana_v2_compute_gain",
            "ana_v2_arch_embedding_factor",
            "ana_v2_arch_mlp_bot_factor",
            "ana_v2_total_over_ana_ratio",
            "ana_v2_mem_over_ana_ratio",
            "ana_v2_compute_over_ana_ratio",
        ],
    },
    "copy_like": {
        "op_types": {"Concat", "Transpose"},
        "target_mode": "analytical_residual",
        "categorical": [
            "op_type",
            "node_scope",
            "node_name_normalized",
            "arch_embedding_size",
            "arch_mlp_bot",
            "arch_mlp_top",
            "static_role",
            "attr_perm_json",
        ],
        "numeric": [
            "attr_axis",
            "output_size",
            "activation_size",
            "feat_io_bytes_sum",
            "feat_output_input_bytes_ratio",
            "total_threads",
            "threads_per_core",
            "log_num_threads",
            "log_inter_threads",
            "log_total_threads",
            "ana_calib_total_us",
            "ana_calib_mem_us",
            "ana_calib_compute_us",
            "ana_v2_total_us",
            "ana_v2_mem_us",
            "ana_v2_compute_us",
            "ana_v2_mem_gain",
            "ana_v2_compute_gain",
            "ana_v2_arch_embedding_factor",
            "ana_v2_arch_mlp_bot_factor",
            "ana_v2_copy_thread_penalty",
            "ana_v2_total_over_ana_ratio",
            "ana_v2_mem_over_ana_ratio",
            "ana_v2_compute_over_ana_ratio",
        ],
    },
    "meta_like": {
        "op_types": {"Reshape", "Shape", "Unsqueeze", "Flatten"},
        "target_mode": "direct_log",
        "categorical": ["op_type", "input_type_shape", "output_type_shape", "arch_embedding_size", "arch_mlp_bot", "arch_mlp_top", "static_role"],
        "numeric": [
            "batch_size",
            "num_indices_per_lookup",
            "num_threads",
            "inter_threads",
            "output_size",
            "activation_size",
            "total_threads",
            "threads_per_core",
            "log_num_threads",
            "log_inter_threads",
            "log_total_threads",
            "ana_v2_total_us",
            "ana_v2_arch_embedding_factor",
            "ana_v2_arch_mlp_bot_factor",
            "ana_v2_total_over_ana_ratio",
        ],
    },
}


def _case_prefix(case_id: Any) -> str:
    text = str(case_id)
    parts = text.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return text


def _backfill_arch_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    case_prefix = work["case_id"].astype(str).map(_case_prefix)
    for column in ("arch_embedding_size", "arch_mlp_bot", "arch_mlp_top"):
        values = pd.Series(work[column], copy=False) if column in work.columns else pd.Series(index=work.index, dtype=object)
        values = values.astype(str).replace({"nan": "", "unknown": "", "None": ""})
        missing = values.eq("")
        if missing.any():
            values.loc[missing] = case_prefix.loc[missing].map(lambda key: CASE_ARCH_MAP.get(str(key), {}).get(column, "unknown"))
        work[column] = values.fillna("unknown").astype(str)
    return work


def _safe_log2(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return np.log2(np.clip(numeric, a_min=1.0, a_max=None))


def _static_role(frame: pd.DataFrame) -> pd.Series:
    op_type = frame["op_type"].astype(str)
    node_scope = frame.get("node_scope", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.lower()
    node_name = frame.get("node_name_normalized", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.lower()
    role = pd.Series("other", index=frame.index, dtype=object)

    meta_mask = op_type.isin(sorted(base_run.NO_ANALYTICAL_OPS))
    role.loc[meta_mask] = "shape_meta"

    lookup_mask = op_type.eq("Gather") & (node_scope.str.contains("emb") | node_name.str.contains("/emb_"))
    role.loc[lookup_mask] = "embedding_lookup"

    reduce_mask = op_type.eq("ReduceSum") & (node_scope.str.contains("emb") | node_name.str.contains("/emb_"))
    role.loc[reduce_mask] = "embedding_reduce"

    interaction_mask = node_scope.str.contains("interaction") | node_name.str.contains("interaction")
    role.loc[interaction_mask] = "interaction"

    mlp_bot_mask = node_scope.str.contains("bot") | node_name.str.contains("/bot")
    role.loc[mlp_bot_mask] = "mlp_bottom"

    mlp_top_mask = node_scope.str.contains("top") | node_name.str.contains("/top")
    role.loc[mlp_top_mask] = "mlp_top"
    return role.astype(str)


def _thread_regime_bucket(frame: pd.DataFrame) -> pd.Series:
    num_threads = pd.to_numeric(frame["num_threads"], errors="coerce").fillna(0.0).astype(int)
    inter_threads = pd.to_numeric(frame["inter_threads"], errors="coerce").fillna(0.0).astype(int)
    total_threads = num_threads * inter_threads
    bucket = pd.Series("other", index=frame.index, dtype=object)
    bucket.loc[(num_threads == 1) & (inter_threads == 1)] = "t1_i1"
    bucket.loc[(num_threads > 1) & (inter_threads == 1)] = "intra_only"
    bucket.loc[(num_threads == 1) & (inter_threads > 1)] = "inter_only"
    bucket.loc[(num_threads > 1) & (inter_threads > 1)] = "hybrid"
    bucket.loc[total_threads >= 16] = "high_total_threads"
    return bucket.astype(str)


def _static_topology_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "node_scope" not in work.columns:
        work["static_is_embedding_branch"] = "no"
        work["static_parallel_branch_count"] = 1.0
        work["static_distance_to_merge"] = 0.0
        return work

    node_scope = work["node_scope"].fillna("").astype(str).str.lower()
    is_embedding_branch = node_scope.str.match(r"^emb_l[0-9]+$")
    work["static_is_embedding_branch"] = np.where(is_embedding_branch, "yes", "no")
    work["static_parallel_branch_count"] = 1.0
    work["static_distance_to_merge"] = 0.0

    if "case_id" not in work.columns or "combo" not in work.columns or "op_idx" not in work.columns:
        return work

    op_idx = pd.to_numeric(work["op_idx"], errors="coerce")
    for (_, _), group in work.groupby(["case_id", "combo"], sort=False):
        group_idx = group.index
        group_scope = group["node_scope"].fillna("").astype(str).str.lower()
        emb_mask = group_scope.str.match(r"^emb_l[0-9]+$")
        branch_scopes = sorted({value for value in group_scope.loc[emb_mask].tolist() if value})
        branch_count = float(max(len(branch_scopes), 1))
        work.loc[group_idx, "static_parallel_branch_count"] = np.where(emb_mask.to_numpy(), branch_count, 1.0)

        if not emb_mask.any():
            continue
        group_op_idx = op_idx.loc[group_idx]
        merge_candidates = group_op_idx.loc[~emb_mask]
        if merge_candidates.empty:
            continue
        merge_op_idx = float(pd.to_numeric(merge_candidates, errors="coerce").dropna().min())
        distances = (merge_op_idx - pd.to_numeric(group_op_idx, errors="coerce")).clip(lower=0.0).fillna(0.0)
        work.loc[group_idx, "static_distance_to_merge"] = np.where(emb_mask.to_numpy(), distances.to_numpy(dtype=float), 0.0)
    return work


def family_bucket_for_op_type(op_type: Any) -> str:
    text = str(op_type)
    if text == "Gather":
        return "gather_like"
    if text == "ReduceSum":
        return "reduce_like"
    if text in {"Concat", "Transpose"}:
        return "copy_like"
    if text in {"Reshape", "Shape", "Unsqueeze", "Flatten"}:
        return "meta_like"
    if text in {"Gemm", "MatMul"}:
        return "gemm_like"
    if text in {"Add", "Mul", "Relu", "Sigmoid"}:
        return "ewise_like"
    return "other"


def add_family_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["family_bucket"] = work["op_type"].astype(str).map(family_bucket_for_op_type).astype(str)
    return work


def _add_static_ml_columns(prepared: pd.DataFrame) -> pd.DataFrame:
    work = _backfill_arch_columns(prepared)
    work["num_threads"] = pd.to_numeric(work["num_threads"], errors="coerce").fillna(0.0)
    work["inter_threads"] = pd.to_numeric(work["inter_threads"], errors="coerce").fillna(1.0)
    work["total_threads"] = work["num_threads"] * work["inter_threads"]
    if "hw_total_cores" in work.columns:
        hw_total_cores = pd.to_numeric(work["hw_total_cores"], errors="coerce").replace(0.0, np.nan)
    else:
        hw_total_cores = pd.Series(np.nan, index=work.index, dtype=float)
    work["threads_per_core"] = (work["total_threads"] / hw_total_cores).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work["log_num_threads"] = _safe_log2(work["num_threads"])
    work["log_inter_threads"] = _safe_log2(work["inter_threads"])
    work["log_total_threads"] = _safe_log2(work["total_threads"])
    work["is_single_intra"] = (pd.to_numeric(work["num_threads"], errors="coerce").fillna(0.0) <= 1.0).astype(float)
    work["is_single_inter"] = (pd.to_numeric(work["inter_threads"], errors="coerce").fillna(0.0) <= 1.0).astype(float)
    work["thread_regime_bucket"] = _thread_regime_bucket(work)
    work["static_role"] = _static_role(work)
    work["family_bucket"] = work["op_type"].astype(str).map(family_bucket_for_op_type).astype(str)
    return _static_topology_columns(work)


def prepare_semantic_static_frame(raw_frame: pd.DataFrame, hw_config: Path, params: dict[str, float]) -> pd.DataFrame:
    rebuilt = base_run.build_analytical_features.rebuild_local_features(raw_frame, hardware_profile_path=hw_config)
    heavy_prepared = base_run.build_analytical_features.prepare_heavy_prediction_frame(rebuilt)
    feature_df = base_run.build_analytical_features.add_calibrated_analytical_columns(rebuilt, heavy_prepared, params)
    feature_df[base_run.TARGET_US_COLUMN] = pd.to_numeric(feature_df[base_run.TARGET_COLUMN], errors="coerce").fillna(0.0)
    prepared = base_run._prepare_context_augmented_frame(feature_df)
    prepared = _add_static_ml_columns(prepared)
    prepared = augment_static_feature_frame(prepared)
    prepared["row_uid"] = prepared["row_uid"].astype(str)
    prepared[base_run.TARGET_US_COLUMN] = pd.to_numeric(prepared[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0)
    prepared["ana_calib_total_us"] = pd.to_numeric(prepared["ana_calib_total_us"], errors="coerce").fillna(0.0)
    prepared["ana_calib_mem_us"] = pd.to_numeric(prepared["ana_calib_mem_us"], errors="coerce").fillna(0.0)
    prepared["ana_calib_compute_us"] = pd.to_numeric(prepared["ana_calib_compute_us"], errors="coerce").fillna(0.0)
    return prepared


def _ml_feature_spec(
    frame: pd.DataFrame,
    *,
    extra_categorical: list[str] | None = None,
    extra_numeric: list[str] | None = None,
) -> tuple[dict[str, dict[str, list[str]]], list[str], list[str]]:
    per_op_feature_columns: dict[str, dict[str, list[str]]] = {}
    categorical_union: list[str] = []
    numeric_union: list[str] = []
    for op_type in sorted(pd.Series(frame["op_type"], copy=False).dropna().astype(str).unique().tolist()):
        numeric_candidates = list(ML_COMMON_NUMERIC)
        numeric_candidates.extend(base_run.PER_OP_NUMERIC_ADDITIONS.get(str(op_type), []))
        if extra_numeric:
            numeric_candidates.extend(extra_numeric)
        categorical = [column for column in ML_COMMON_CATEGORICAL if column in frame.columns]
        if extra_categorical:
            categorical.extend([column for column in extra_categorical if column in frame.columns])
            categorical = list(dict.fromkeys(categorical))
        numeric = [column for column in dict.fromkeys(numeric_candidates) if column in frame.columns]
        per_op_feature_columns[str(op_type)] = {"categorical": categorical, "numeric": numeric}
        categorical_union.extend(categorical)
        numeric_union.extend(numeric)
    return per_op_feature_columns, list(dict.fromkeys(categorical_union)), list(dict.fromkeys(numeric_union))


def fit_semantic_thread_bundle(
    fit_frame: pd.DataFrame,
    *,
    model_family: str = "rf",
    extra_categorical: list[str] | None = None,
    extra_numeric: list[str] | None = None,
) -> dict[str, Any]:
    if model_family != "rf":
        raise ValueError(f"Unsupported semantic thread model family: {model_family}")
    per_op_feature_columns, categorical, numeric = _ml_feature_spec(
        fit_frame,
        extra_categorical=extra_categorical,
        extra_numeric=extra_numeric,
    )
    category_maps = base_run._fit_category_maps(fit_frame, categorical)
    encoded = base_run._encode_features(fit_frame, categorical, numeric, category_maps)
    models: dict[str, Any] = {}
    target_modes: dict[str, str] = {}
    global_default_pred = float(np.median(pd.to_numeric(fit_frame[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)))

    for op_type, group in fit_frame.groupby("op_type", sort=True):
        feature_spec = per_op_feature_columns.get(str(op_type), {"categorical": categorical, "numeric": numeric})
        feature_columns = feature_spec["categorical"] + feature_spec["numeric"]
        actual = pd.to_numeric(group[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if str(op_type) not in base_run.NO_ANALYTICAL_OPS:
            base = np.clip(pd.to_numeric(group["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
            target = np.log(np.clip(actual, a_min=1e-6, a_max=None) / base)
            target_modes[str(op_type)] = "analytical_residual"
        else:
            target = np.log1p(np.clip(actual, a_min=0.0, a_max=None))
            target_modes[str(op_type)] = "direct_log"
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1,
        )
        model.fit(encoded.loc[group.index, feature_columns], target)
        models[str(op_type)] = model

    return {
        "mode": "analytical",
        "model_family": model_family,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "per_op_feature_columns": per_op_feature_columns,
        "category_maps": category_maps,
        "target_modes": target_modes,
        "models": models,
        "global_default_pred_us": global_default_pred,
        "extra_categorical": list(extra_categorical or []),
        "extra_numeric": list(extra_numeric or []),
    }


def predict_semantic_thread_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    work = frame.copy()
    encoded = base_run._encode_features(work, bundle["categorical_columns"], bundle["numeric_columns"], bundle["category_maps"])
    pred = np.array(
        pd.to_numeric(work["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        dtype=float,
        copy=True,
    )
    for op_type, model in bundle["models"].items():
        mask = work["op_type"].astype(str).eq(str(op_type)).to_numpy()
        if not mask.any():
            continue
        feature_spec = bundle["per_op_feature_columns"].get(
            str(op_type),
            {"categorical": bundle["categorical_columns"], "numeric": bundle["numeric_columns"]},
        )
        feature_columns = feature_spec["categorical"] + feature_spec["numeric"]
        raw_pred = model.predict(encoded.loc[mask, feature_columns])
        target_mode = bundle["target_modes"][str(op_type)]
        if target_mode == "analytical_residual":
            base = np.clip(pd.to_numeric(work.loc[mask, "ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
            pred[mask] = base * np.exp(raw_pred)
        else:
            pred[mask] = np.expm1(raw_pred)
    return np.clip(pred, a_min=0.0, a_max=None)


def _fit_family_clip_stats(
    fit_frame: pd.DataFrame,
    *,
    family_name: str,
    target_mode: str,
) -> dict[str, dict[str, float]]:
    work = fit_frame.copy()
    work["static_role"] = work["static_role"].astype(str)
    stats: dict[str, dict[str, float]] = {}
    if target_mode == "analytical_residual":
        base = np.clip(pd.to_numeric(work["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
        work["clip_target"] = pd.to_numeric(work[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float) / base
    else:
        work["clip_target"] = pd.to_numeric(work[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for role, group in work.groupby("static_role", sort=True):
        values = pd.to_numeric(group["clip_target"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(values) == 0:
            continue
        stats[str(role)] = {
            "lower": float(np.quantile(values, 0.05)),
            "upper": float(np.quantile(values, 0.95)),
        }
    if stats:
        all_values = pd.to_numeric(work["clip_target"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        stats["__family_fallback__"] = {
            "lower": float(np.quantile(all_values, 0.05)),
            "upper": float(np.quantile(all_values, 0.95)),
        }
    return stats


def _family_feature_columns(frame: pd.DataFrame, family_name: str) -> tuple[list[str], list[str]]:
    spec = FAMILY_ROUTE_SPECS[family_name]
    categorical = [column for column in spec["categorical"] if column in frame.columns]
    numeric = [column for column in spec["numeric"] if column in frame.columns]
    return categorical, numeric


def fit_semantic_thread_family_bundle(fit_frame: pd.DataFrame) -> dict[str, Any]:
    shared_bundle = fit_semantic_thread_bundle(fit_frame, model_family="rf")
    family_models: dict[str, Any] = {}
    category_maps: dict[str, dict[str, dict[str, int]]] = {}
    feature_columns: dict[str, dict[str, list[str]]] = {}
    target_modes: dict[str, str] = {}
    clip_stats: dict[str, dict[str, dict[str, float]]] = {}
    fallback_pred = float(np.median(pd.to_numeric(fit_frame[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)))

    for family_name, spec in FAMILY_ROUTE_SPECS.items():
        mask = fit_frame["family_bucket"].astype(str).eq(family_name)
        family_frame = fit_frame.loc[mask].copy()
        if family_frame.empty:
            continue
        categorical, numeric = _family_feature_columns(family_frame, family_name)
        feature_columns[family_name] = {"categorical": categorical, "numeric": numeric}
        maps = base_run._fit_category_maps(family_frame, categorical)
        category_maps[family_name] = maps
        encoded = base_run._encode_features(family_frame, categorical, numeric, maps)
        model = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1,
        )
        target_mode = str(spec["target_mode"])
        if target_mode == "analytical_residual":
            base = np.clip(pd.to_numeric(family_frame["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
            target = np.log(
                np.clip(pd.to_numeric(family_frame[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
                / base
            )
        else:
            target = np.log1p(np.clip(pd.to_numeric(family_frame[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=0.0, a_max=None))
        model.fit(encoded[categorical + numeric], target)
        family_models[family_name] = model
        target_modes[family_name] = target_mode
        clip_stats[family_name] = _fit_family_clip_stats(family_frame, family_name=family_name, target_mode=target_mode)

    return {
        "mode": "family_rf",
        "shared_bundle": shared_bundle,
        "family_models": family_models,
        "family_feature_columns": feature_columns,
        "family_category_maps": category_maps,
        "family_target_modes": target_modes,
        "family_clip_stats": clip_stats,
        "global_default_pred_us": fallback_pred,
    }


def predict_semantic_thread_family_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    pred = predict_semantic_thread_bundle(bundle["shared_bundle"], frame)
    work = frame.copy()
    for family_name, model in bundle["family_models"].items():
        mask = work["family_bucket"].astype(str).eq(family_name)
        if not mask.any():
            continue
        feature_spec = bundle["family_feature_columns"][family_name]
        maps = bundle["family_category_maps"][family_name]
        categorical = feature_spec["categorical"]
        numeric = feature_spec["numeric"]
        encoded = base_run._encode_features(work.loc[mask], categorical, numeric, maps)
        raw_pred = model.predict(encoded[categorical + numeric])
        target_mode = bundle["family_target_modes"][family_name]
        if target_mode == "analytical_residual":
            base = np.clip(pd.to_numeric(work.loc[mask, "ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
            ratio = np.exp(raw_pred)
            roles = work.loc[mask, "static_role"].fillna("other").astype(str).tolist()
            clipped_ratio: list[float] = []
            family_stats = bundle["family_clip_stats"].get(family_name, {})
            fallback = family_stats.get("__family_fallback__", {"lower": 0.0, "upper": float("inf")})
            for role, value in zip(roles, ratio, strict=False):
                stats = family_stats.get(role, fallback)
                clipped_ratio.append(float(np.clip(value, stats["lower"], stats["upper"])))
            pred[mask] = base * np.asarray(clipped_ratio, dtype=float)
        else:
            direct = np.expm1(raw_pred)
            roles = work.loc[mask, "static_role"].fillna("other").astype(str).tolist()
            clipped_direct: list[float] = []
            family_stats = bundle["family_clip_stats"].get(family_name, {})
            fallback = family_stats.get("__family_fallback__", {"lower": 0.0, "upper": float("inf")})
            for role, value in zip(roles, direct, strict=False):
                stats = family_stats.get(role, fallback)
                clipped_direct.append(float(np.clip(value, stats["lower"], stats["upper"])))
            pred[mask] = np.asarray(clipped_direct, dtype=float)
    return np.clip(pred, a_min=0.0, a_max=None)


def route_stage(route: str) -> str:
    if route in {ML_RF_ROUTE, ML_RF_SHAPE_TREE_ROUTE, ML_FAMILY_RF_ROUTE}:
        return "ml_only"
    if route in {
        WHITEBOX_PRIOR_ONLY_ROUTE,
        WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
        WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
        WHITEBOX_ORT_KERNEL_RF_ROUTE,
        WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
    } or str(route).startswith(
        f"{WHITEBOX_WORKLOAD_BENCHMARK_ROUTE}["
    ):
        return "whitebox_prior"
    return "baseline"
