from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from cpu_perf_model.analytical import build_analytical_features

try:
    from .runner import (
        DEFAULT_OUTPUT_ROOT,
        SEED,
        _evaluate_analytical_reference,
        _evaluate_base_models_on_augmented_csv,
        _evaluate_original_single,
        _evaluate_model_roots_on_raw_csvs,
        _materialize_eval_augmented_csv,
        _prepare_augmented_dataset,
        _train_analytical,
        _train_analytical_residual_single_model,
        _train_enhanced_models,
        _train_original_models,
        _write_json,
    )
except ImportError:  # pragma: no cover - allows direct script execution
    from generalization_benchmarks.runner import (
        DEFAULT_OUTPUT_ROOT,
        SEED,
        _evaluate_analytical_reference,
        _evaluate_base_models_on_augmented_csv,
        _evaluate_original_single,
        _evaluate_model_roots_on_raw_csvs,
        _materialize_eval_augmented_csv,
        _prepare_augmented_dataset,
        _train_analytical,
        _train_analytical_residual_single_model,
        _train_enhanced_models,
        _train_original_models,
        _write_json,
    )

from cpu_perf_model.classed.build_classed_dataset import build_classed_dataset_artifacts
from cpu_perf_model.single_fair import train_fair_single_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PUSH_ROOT = PROJECT_ROOT / "artifacts" / "latest" / "generalization_target_push"
EXP2_ROOT = DEFAULT_OUTPUT_ROOT / "exp2_union_70_30"
EXP2_DATASET_ROOT = EXP2_ROOT / "dataset"
EXP2_SUPPORT_ROOT = DEFAULT_OUTPUT_ROOT / "exp2_union_70_30_router_fastpath_v4_trainval_top5" / "training" / "routed_expert_ensemble" / "support"
PREBALANCED_ROOT = DEFAULT_OUTPUT_ROOT / "exp2_union_70_30_semantic_free_prebalanced_fixed_analytical"

BASE_CANDIDATES = [
    "original_single_mlp",
    "original_classed_mlp",
    "enhanced_classed_mlp",
    "family_enhanced",
    "global_enhanced",
]
TARGET_PUSH_CANDIDATES = [
    "global_hgbr",
    "family_extra",
    "family_extra_regime",
    "global_enhanced",
    "family_enhanced",
    "enhanced_classed_mlp",
    "original_classed_mlp",
    "original_single_mlp",
]
CAT_BASE = [
    "op_type",
    "benchmark_source_dataset",
    "ana_calib_family",
    "ana_calib_regime",
    "node_scope",
    "source_name",
    "regime",
    "input_type_shape",
    "output_type_shape",
]
BASE_FEATURES = [
    "op_type_code",
    "benchmark_source_dataset_code",
    "ana_calib_family_code",
    "ana_calib_regime_code",
    "node_scope_code",
    "source_name_code",
    "regime_code",
    "input_type_shape_code",
    "output_type_shape_code",
    "batch_size",
    "num_indices_per_lookup",
    "num_threads",
    "output_size",
    "activation_size",
    "parameter_size",
    "feat_io_bytes_sum",
    "hw_ratio_working_set_to_l2_active_bytes",
    "hw_ratio_working_set_to_l3_active_bytes",
    "ana_calib_total_us",
    "ana_calib_mem_us",
    "ana_calib_compute_us",
    "ana_calib_overhead_us",
    "ana_expected_latency_ns",
    "ana_mem_bw_time_us",
    "ana_latency_proxy_us",
    "feat_reduction_axes_count",
    "feat_reduction_work_items",
    "feat_reduction_axes_product",
    "feat_reduction_input_rank",
    "feat_reduction_output_rank",
    "feat_gemm_m",
    "feat_gemm_n",
    "feat_gemm_k",
    "feat_gemm_mac_count",
    "feat_gemm_bytes_per_mac",
    "struct_total_input_numel",
    "struct_total_output_numel",
    "struct_total_input_bytes",
    "struct_total_output_bytes",
    "struct_input_rank0",
    "struct_output_rank0",
    "struct_rank_delta",
    "struct_io_numel_ratio",
    "transpose_inversion_count",
    "transpose_stride_span",
    "reduce_fraction",
    "reduce_axes_span",
    "gather_reuse_factor",
    "gather_row_bytes",
    "gather_table_coverage",
    "shape_input_numel_exact",
    "shape_input_rank_exact",
    "reshape_output_rank",
    "unsqueeze_axes_count_exact",
    "pred_analytical_reference",
    "original_single_mlp",
    "original_classed_mlp",
    "enhanced_classed_mlp",
    "family_enhanced",
    "global_enhanced",
    "blend_candidate_pred_span",
]
REGIME_SPLIT_HARD = {"Gather", "ReduceSum", "Transpose", "Shape", "Relu"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_run_summary(experiment_root: Path) -> dict[str, Any]:
    return _read_json(experiment_root / "run_summary.json")


def _resolve_benchmark_context(experiment_root: Path) -> dict[str, Any]:
    run_summary = _experiment_run_summary(experiment_root)
    training_roots = dict(run_summary.get("training_roots", {}))
    dataset_root = Path(run_summary["dataset_root"])
    analytical_root = Path(run_summary["analytical_root"])
    eval_augmented_test = experiment_root / "eval_inputs" / "augmented" / "test.csv"
    eval_test_root = experiment_root / "evaluations" / "test"
    deploy_parameters_json = analytical_root / "deploy" / "full_data_parameters.json"
    required_paths = {
        "dataset_root": dataset_root,
        "analytical_root": analytical_root,
        "eval_augmented_test": eval_augmented_test,
        "eval_test_root": eval_test_root,
        "deploy_parameters_json": deploy_parameters_json,
        "classed_data_root": Path(training_roots["classed_data_root"]),
        "enhanced_classed_data_root": Path(training_roots["enhanced_classed_data_root"]),
        "original_single_model_root": Path(training_roots["original_single_model_root"]),
        "original_classed_model_root": Path(training_roots["original_classed_model_root"]),
        "enhanced_classed_model_root": Path(training_roots["enhanced_classed_model_root"]),
        "global_model_root": Path(training_roots["global_model_root"]),
        "family_model_root": Path(training_roots["family_model_root"]),
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark dependency for {label}: {path}")
    return {
        "experiment_root": experiment_root,
        "experiment_id": str(run_summary.get("experiment_id", experiment_root.name)),
        "description": str(run_summary.get("description", "")),
        "dataset_root": dataset_root,
        "analytical_root": analytical_root,
        "eval_augmented_test": eval_augmented_test,
        "eval_test_root": eval_test_root,
        "deploy_parameters_json": deploy_parameters_json,
        "model_roots": {
            "classed_data_root": str(required_paths["classed_data_root"]),
            "enhanced_classed_data_root": str(required_paths["enhanced_classed_data_root"]),
            "original_single_model_root": str(required_paths["original_single_model_root"]),
            "original_classed_model_root": str(required_paths["original_classed_model_root"]),
            "enhanced_classed_model_root": str(required_paths["enhanced_classed_model_root"]),
            "global_model_root": str(required_paths["global_model_root"]),
            "family_model_root": str(required_paths["family_model_root"]),
        },
    }


def _metrics_dict(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> dict[str, float]:
    target = np.asarray(pd.to_numeric(y, errors="coerce").fillna(0.0), dtype=float)
    pred = np.clip(np.asarray(pd.to_numeric(p, errors="coerce").fillna(0.0), dtype=float), 1e-6, None)
    ape = np.abs(pred - target) / np.clip(target, 1e-9, None)
    pearson = float(pd.DataFrame({"y": target, "p": pred}).corr(method="pearson").iloc[0, 1]) if len(target) > 1 else 0.0
    return {
        "rows": int(len(target)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.quantile(ape, 0.5)),
        "p90_ape": float(np.quantile(ape, 0.9)),
        "p95_ape": float(np.quantile(ape, 0.95)),
        "pearson": 0.0 if math.isnan(pearson) else pearson,
    }


def _trend_metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce").fillna(0.0), "p": pd.to_numeric(p, errors="coerce").fillna(0.0)})
    if len(frame) <= 1:
        return {"spearman": 0.0, "pearson": 0.0}
    spearman = float(frame.corr(method="spearman").iloc[0, 1])
    pearson = float(frame.corr(method="pearson").iloc[0, 1])
    return {
        "spearman": 0.0 if math.isnan(spearman) else spearman,
        "pearson": 0.0 if math.isnan(pearson) else pearson,
    }


def _regime_name(row: pd.Series) -> str:
    op = row.get("op_type", "")
    if op == "Gather":
        reuse = float(row.get("gather_reuse_factor", 0.0) or 0.0)
        row_bytes = float(row.get("gather_row_bytes", 0.0) or 0.0)
        if reuse > 1.25:
            return "reuse_heavy"
        if row_bytes <= 512:
            return "narrow_row"
        if row_bytes <= 1600:
            return "mid_row"
        return "wide_or_streaming"
    if op == "ReduceSum":
        frac = float(row.get("reduce_fraction", 0.0) or 0.0)
        span = float(row.get("reduce_axes_span", 0.0) or 0.0)
        has_batch = float(row.get("reduce_has_batch_axis", 0.0) or 0.0)
        has_last = float(row.get("reduce_has_last_axis", 0.0) or 0.0)
        suffix = float(row.get("reduce_is_suffix", 0.0) or 0.0)
        keep = float(row.get("reduce_keepdims", 0.0) or 0.0)
        if span == 0 and frac == 0:
            return "legacy_missing_attrs"
        if suffix == 1 and has_last == 1 and keep == 0:
            return "suffix_tail_reduce"
        if has_batch == 1 or span >= 3 or frac > 1e-3:
            return "batch_or_wide_reduce"
        return "other_known_reduce"
    if op == "Transpose":
        rank = float(row.get("transpose_rank", 0.0) or 0.0)
        inversion = float(row.get("transpose_inversion_count", 0.0) or 0.0)
        span = float(row.get("transpose_stride_span", 0.0) or 0.0)
        if rank == 0:
            return "legacy_missing_perm"
        if inversion <= 1 and span <= 4096:
            return "near_identity_or_local"
        if inversion >= 4 or span > 24576:
            return "high_disruption"
        return "mid_disruption"
    if op == "Shape":
        numel = float(row.get("shape_input_numel_exact", 0.0) or 0.0)
        rank = float(row.get("shape_input_rank_exact", 0.0) or 0.0)
        if numel <= 116640:
            return "tiny_meta"
        if numel <= 716800:
            return "mid_meta"
        if numel > 3932160:
            return "huge_meta"
        if rank >= 5:
            return "high_rank_mid"
        return "other"
    if op == "Relu":
        ana = float(row.get("ana_calib_total_us", 0.0) or 0.0)
        l2 = float(row.get("hw_ratio_working_set_to_l2_active_bytes", 0.0) or 0.0)
        if ana <= 140:
            return "tiny_overhead"
        if l2 <= 5:
            return "cache_fit"
        if l2 <= 20:
            return "spill_mid"
        return "streaming"
    return "default"


def _load_support_frame(augmented_csv: Path, prediction_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(augmented_csv, low_memory=False)
    frame["target_us"] = pd.to_numeric(frame["label_operator_actual_dur_us"], errors="coerce").fillna(0.0).astype(float)
    for model_name in BASE_CANDIDATES:
        pred_path = prediction_dir / f"{model_name}.csv"
        pred_frame = pd.read_csv(pred_path, low_memory=False)[["row_uid", "pred_us"]].rename(columns={"pred_us": model_name})
        frame = frame.merge(pred_frame, on="row_uid", how="left", validate="one_to_one")
    analytical_path = prediction_dir / "analytical_reference.csv"
    if analytical_path.exists():
        analytical_frame = pd.read_csv(analytical_path, low_memory=False)[["row_uid", "pred_us"]].rename(columns={"pred_us": "pred_analytical_reference"})
        frame = frame.merge(analytical_frame, on="row_uid", how="left", validate="one_to_one")
    else:
        frame["pred_analytical_reference"] = pd.to_numeric(frame.get("ana_calib_total_us", 0.0), errors="coerce").fillna(0.0).astype(float)
    base_block = frame[[column for column in BASE_CANDIDATES if column in frame.columns]]
    if not base_block.empty:
        frame["blend_candidate_pred_span"] = base_block.max(axis=1) - base_block.min(axis=1)
    else:
        frame["blend_candidate_pred_span"] = 0.0
    frame["regime"] = frame.apply(_regime_name, axis=1)
    for src, dst in [("batch_size", "batch_bin"), ("num_indices_per_lookup", "nip_bin"), ("num_threads", "thr_bin")]:
        if src in frame.columns:
            values = pd.to_numeric(frame[src], errors="coerce").fillna(0.0)
            frame[dst] = np.floor(np.log2(np.clip(values, 1.0, None))).astype(int).astype(str)
    return frame


def _fit_category_maps(train_frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    category_maps: dict[str, dict[str, int]] = {}
    for column in CAT_BASE:
        if column not in train_frame.columns:
            continue
        values = train_frame[column].fillna("missing").astype(str)
        category_maps[column] = {value: idx for idx, value in enumerate(pd.Index(values).unique().tolist())}
    return category_maps


def _apply_category_maps(frame: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    encoded = frame.copy()
    for column, mapping in category_maps.items():
        values = encoded[column].fillna("missing").astype(str)
        encoded[f"{column}_code"] = values.map(mapping).fillna(-1).astype(float)
    return encoded


def _model_features(frame: pd.DataFrame) -> list[str]:
    return [column for column in BASE_FEATURES if column in frame.columns]


def _row_weight(frame: pd.DataFrame) -> np.ndarray:
    target = pd.to_numeric(frame["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "blend_candidate_pred_span" in frame.columns:
        span = pd.to_numeric(frame["blend_candidate_pred_span"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        span = np.zeros_like(target)
    disagreement = span / np.clip(target, 1.0, None)
    return (1.0 / np.sqrt(np.clip(target, 1.0, None))) * (1.0 + 8.0 * np.clip(disagreement, 0.0, 3.0))


def _candidate_key(target: pd.Series, pred: pd.Series) -> tuple[float, float, float]:
    target_values = pd.to_numeric(target, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred_values = pd.to_numeric(pred, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ape = np.abs(pred_values - target_values) / np.clip(target_values, 1e-9, None)
    return (
        float(np.quantile(ape, 0.9)),
        float(np.mean(ape)),
        float(np.quantile(ape, 0.5)),
    )


def _fit_global_hgbr(train_frame: pd.DataFrame, feature_columns: list[str]) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_depth=10,
        max_iter=700,
        min_samples_leaf=8,
        random_state=SEED,
    )
    model.fit(train_frame[feature_columns], np.log1p(train_frame["target_us"]), sample_weight=_row_weight(train_frame))
    return model


def _fit_family_models(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    regime_split: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for op_type, group in train_frame.groupby("op_type", sort=True):
        if regime_split and op_type in REGIME_SPLIT_HARD:
            regime_models: dict[str, ExtraTreesRegressor] = {}
            for regime, regime_group in group.groupby("regime", sort=True):
                if len(regime_group) < 32:
                    continue
                model = ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=max(1, min(4, len(regime_group) // 80)),
                    random_state=SEED,
                    n_jobs=32,
                )
                model.fit(regime_group[feature_columns], np.log1p(regime_group["target_us"]), sample_weight=_row_weight(regime_group))
                regime_models[str(regime)] = model
            fallback = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=max(1, min(4, len(group) // 80)),
                random_state=SEED,
                n_jobs=32,
            )
            fallback.fit(group[feature_columns], np.log1p(group["target_us"]), sample_weight=_row_weight(group))
            payload[str(op_type)] = {"fallback": fallback, "regime_models": regime_models}
        else:
            model = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=max(1, min(4, len(group) // 80)),
                random_state=SEED,
                n_jobs=32,
            )
            model.fit(group[feature_columns], np.log1p(group["target_us"]), sample_weight=_row_weight(group))
            payload[str(op_type)] = {"fallback": model, "regime_models": {}}
    return payload


def _predict_family_models(frame: pd.DataFrame, feature_columns: list[str], payload: dict[str, Any]) -> np.ndarray:
    pred = pd.Series(index=frame.index, dtype=float)
    for op_type, group in frame.groupby("op_type", sort=False):
        config = payload[str(op_type)]
        fallback = config["fallback"]
        regime_models = dict(config.get("regime_models", {}))
        if regime_models:
            for regime, regime_group in group.groupby("regime", sort=False):
                model = regime_models.get(str(regime), fallback)
                pred.loc[regime_group.index] = np.expm1(model.predict(regime_group[feature_columns]))
        else:
            pred.loc[group.index] = np.expm1(fallback.predict(group[feature_columns]))
    return pred.sort_index().to_numpy(dtype=float)


def _fit_blend_plan(val_frame: pd.DataFrame) -> dict[str, Any]:
    weights = np.linspace(0.0, 1.0, 101)
    plan: dict[str, Any] = {}
    for op_type, group in val_frame.groupby("op_type", sort=True):
        ranked = sorted(TARGET_PUSH_CANDIDATES, key=lambda name: _candidate_key(group["target_us"], group[name]))[:4]
        best_mode: dict[str, Any] = {"mode": "single", "single_model": ranked[0]}
        best_key = _candidate_key(group["target_us"], group[ranked[0]])
        for left, right in [(ranked[i], ranked[j]) for i in range(len(ranked)) for j in range(i + 1, len(ranked))]:
            left_pred = pd.to_numeric(group[left], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            right_pred = pd.to_numeric(group[right], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for weight in weights:
                pred = weight * left_pred + (1.0 - weight) * right_pred
                candidate_key = _candidate_key(group["target_us"], pd.Series(pred, index=group.index))
                if candidate_key < best_key:
                    best_key = candidate_key
                    best_mode = {
                        "mode": "blend",
                        "left_model": left,
                        "right_model": right,
                        "weight_left": float(weight),
                    }
        plan[str(op_type)] = {
            **best_mode,
            "val_metrics": {
                "p90_ape": best_key[0],
                "mape": best_key[1],
                "median_ape": best_key[2],
            },
        }
    return plan


def _apply_blend_plan(frame: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    pred: list[float] = []
    mix_id: list[str] = []
    for _, row in out.iterrows():
        cfg = dict(plan[str(row["op_type"])])
        if cfg["mode"] == "single":
            pred.append(float(row[cfg["single_model"]]))
            mix_id.append(str(cfg["single_model"]))
        else:
            left = str(cfg["left_model"])
            right = str(cfg["right_model"])
            weight = float(cfg["weight_left"])
            pred.append(float(weight * row[left] + (1.0 - weight) * row[right]))
            mix_id.append(f"{left}|{right}|{weight:.2f}")
    out["pred_us"] = pred
    out["mix_id"] = mix_id
    return out


def _fit_shrink_calibration(val_frame: pd.DataFrame) -> dict[str, Any]:
    work = val_frame.copy()
    work["log_resid"] = np.log(np.clip(work["target_us"], 1e-6, None)) - np.log(np.clip(work["pred_us"], 1e-6, None))
    child_columns = ["op_type", "mix_id", "benchmark_source_dataset", "thr_bin"]
    parent_columns = ["op_type", "thr_bin"]
    k = 4.0
    child_stats = work.groupby(child_columns, dropna=False)["log_resid"].agg(["median", "count"]).reset_index()
    parent_stats = work.groupby(parent_columns, dropna=False)["log_resid"].agg(["median", "count"]).reset_index()
    merged = child_stats.merge(parent_stats, on=parent_columns, how="left", suffixes=("_child", "_parent"))
    merged["shift"] = (merged["count_child"] / (merged["count_child"] + k)) * merged["median_child"] + (k / (merged["count_child"] + k)) * merged["median_parent"]
    child_lookup = {
        "|".join(str(row[column]) for column in child_columns): float(row["shift"])
        for _, row in merged.iterrows()
    }
    parent_lookup = {
        "|".join(str(row[column]) for column in parent_columns): float(row["median"])
        for _, row in parent_stats.iterrows()
    }
    return {
        "child_columns": child_columns,
        "parent_columns": parent_columns,
        "k": k,
        "global_shift": float(work["log_resid"].median()),
        "child_lookup": child_lookup,
        "parent_lookup": parent_lookup,
    }


def _apply_shrink_calibration(frame: pd.DataFrame, calibration: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    child_columns = list(calibration["child_columns"])
    parent_columns = list(calibration["parent_columns"])
    child_lookup = dict(calibration["child_lookup"])
    parent_lookup = dict(calibration["parent_lookup"])
    global_shift = float(calibration["global_shift"])
    calibrated: list[float] = []
    for _, row in out.iterrows():
        child_key = "|".join(str(row[column]) for column in child_columns)
        if child_key in child_lookup:
            shift = child_lookup[child_key]
        else:
            parent_key = "|".join(str(row[column]) for column in parent_columns)
            shift = parent_lookup.get(parent_key, global_shift)
        calibrated.append(float(np.exp(np.log(max(float(row["pred_us"]), 1e-6)) + shift)))
    out["pred_us"] = calibrated
    return out


def _save_target_push_model(
    output_root: Path,
    *,
    category_maps: dict[str, dict[str, int]],
    feature_columns: list[str],
    global_model: HistGradientBoostingRegressor,
    family_extra: dict[str, Any],
    family_extra_regime: dict[str, Any],
    blend_plan: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    model_root = output_root / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(global_model, model_root / "global_hgbr.joblib")
    joblib.dump(family_extra, model_root / "family_extra.joblib")
    joblib.dump(family_extra_regime, model_root / "family_extra_regime.joblib")
    manifest = {
        "model_type": "target_push_soft_blend",
        "candidate_models": list(TARGET_PUSH_CANDIDATES),
        "categorical_base": list(CAT_BASE),
        "category_maps": category_maps,
        "feature_columns": feature_columns,
        "blend_plan": blend_plan,
        "calibration": calibration,
        "artifacts": {
            "global_hgbr": str(model_root / "global_hgbr.joblib"),
            "family_extra": str(model_root / "family_extra.joblib"),
            "family_extra_regime": str(model_root / "family_extra_regime.joblib"),
        },
    }
    _write_json(model_root / "manifest.json", manifest)
    return manifest


def _predict_target_push(
    frame: pd.DataFrame,
    *,
    model_manifest: dict[str, Any],
    model_root: Path,
) -> pd.DataFrame:
    category_maps = {column: {str(key): int(value) for key, value in mapping.items()} for column, mapping in dict(model_manifest["category_maps"]).items()}
    encoded = _apply_category_maps(frame, category_maps)
    feature_columns = [column for column in model_manifest["feature_columns"] if column in encoded.columns]
    for column in feature_columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(0.0)
    global_model = joblib.load(model_root / "global_hgbr.joblib")
    family_extra = joblib.load(model_root / "family_extra.joblib")
    family_extra_regime = joblib.load(model_root / "family_extra_regime.joblib")
    work = frame.copy()
    work["global_hgbr"] = np.expm1(global_model.predict(encoded[feature_columns]))
    work["family_extra"] = _predict_family_models(encoded, feature_columns, family_extra)
    work["family_extra_regime"] = _predict_family_models(encoded, feature_columns, family_extra_regime)
    blended = _apply_blend_plan(work, dict(model_manifest["blend_plan"]))
    calibrated = _apply_shrink_calibration(blended, dict(model_manifest["calibration"]))
    output = calibrated[["row_uid", "op_type", "split", "target_us", "pred_us", "mix_id"]].copy()
    output["abs_error_us"] = np.abs(output["pred_us"] - output["target_us"])
    output["ape"] = output["abs_error_us"] / np.clip(output["target_us"], 1e-9, None)
    return output


def _fit_and_save_best_model(
    *,
    support_root: Path,
    test_augmented_csv: Path,
    test_prediction_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    train_frame = _load_support_frame(support_root / "train_augmented.csv", support_root / "train_predictions")
    val_frame = _load_support_frame(support_root / "val_augmented.csv", support_root / "val_predictions")
    test_frame = _load_support_frame(test_augmented_csv, test_prediction_dir)
    category_maps = _fit_category_maps(train_frame)
    train_encoded = _apply_category_maps(train_frame, category_maps)
    val_encoded = _apply_category_maps(val_frame, category_maps)
    test_encoded = _apply_category_maps(test_frame, category_maps)
    feature_columns = _model_features(train_encoded)
    for frame in [train_encoded, val_encoded, test_encoded]:
        for column in feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    global_model = _fit_global_hgbr(train_encoded, feature_columns)
    family_extra = _fit_family_models(train_encoded, feature_columns, regime_split=False)
    family_extra_regime = _fit_family_models(train_encoded, feature_columns, regime_split=True)
    for source, encoded in [(val_frame, val_encoded), (test_frame, test_encoded)]:
        source["global_hgbr"] = np.expm1(global_model.predict(encoded[feature_columns]))
        source["family_extra"] = _predict_family_models(encoded, feature_columns, family_extra)
        source["family_extra_regime"] = _predict_family_models(encoded, feature_columns, family_extra_regime)
    blend_plan = _fit_blend_plan(val_frame)
    val_blended = _apply_blend_plan(val_frame, blend_plan)
    test_blended = _apply_blend_plan(test_frame, blend_plan)
    calibration = _fit_shrink_calibration(val_blended)
    test_output = _apply_shrink_calibration(test_blended, calibration)
    model_manifest = _save_target_push_model(
        output_root,
        category_maps=category_maps,
        feature_columns=feature_columns,
        global_model=global_model,
        family_extra=family_extra,
        family_extra_regime=family_extra_regime,
        blend_plan=blend_plan,
        calibration=calibration,
    )
    prediction_output = test_output[["row_uid", "op_type", "split", "target_us", "pred_us", "mix_id"]].copy()
    prediction_output["abs_error_us"] = np.abs(prediction_output["pred_us"] - prediction_output["target_us"])
    prediction_output["ape"] = prediction_output["abs_error_us"] / np.clip(prediction_output["target_us"], 1e-9, None)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_csv = output_root / "test_predictions.csv"
    prediction_output.to_csv(prediction_csv, index=False)
    summary = {
        "model_manifest_json": str(output_root / "model" / "manifest.json"),
        "test_prediction_csv": str(prediction_csv),
        "test_metrics": _metrics_dict(prediction_output["target_us"], prediction_output["pred_us"]),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def _sample_analytical_fit_rows(dataset_full: pd.DataFrame, fit_fraction: float) -> pd.DataFrame:
    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError(f"fit_fraction must be in (0, 1], got {fit_fraction}")
    if fit_fraction >= 1.0:
        return dataset_full.copy()
    strat_cols = [column for column in ["benchmark_source_dataset", "op_type"] if column in dataset_full.columns]
    if not strat_cols:
        return dataset_full.sample(frac=fit_fraction, random_state=SEED).reset_index(drop=True)
    sampled_groups: list[pd.DataFrame] = []
    for _, group in dataset_full.groupby(strat_cols, dropna=False, sort=False):
        if len(group) <= 1:
            sampled_groups.append(group.copy())
            continue
        take = min(len(group), max(1, int(round(len(group) * fit_fraction))))
        sampled_groups.append(group.sample(n=take, random_state=SEED))
    sampled = pd.concat(sampled_groups, ignore_index=True)
    return sampled.reset_index(drop=True)


def _build_reduced_analytical_artifacts(
    *,
    dataset_root: Path,
    output_root: Path,
    fit_fraction: float,
    passes: int,
) -> dict[str, Any]:
    dataset_full = pd.read_csv(dataset_root / "dataset_full.csv", low_memory=False)
    fit_frame = _sample_analytical_fit_rows(dataset_full, fit_fraction)
    fit_rebuilt = build_analytical_features.rebuild_local_features(fit_frame)
    fit_heavy = build_analytical_features.prepare_heavy_prediction_frame(fit_rebuilt)
    params = build_analytical_features.analytical_eval.calibrate_params(fit_heavy, passes=passes)
    full_rebuilt = build_analytical_features.rebuild_local_features(dataset_full)
    full_heavy = build_analytical_features.prepare_heavy_prediction_frame(full_rebuilt)
    feature_df = build_analytical_features.add_calibrated_analytical_columns(full_rebuilt, full_heavy, params)
    analytical_root = output_root / "analytical"
    analytical_root.mkdir(parents=True, exist_ok=True)
    feature_csv = analytical_root / "analytical_features_full.csv"
    feature_df.to_csv(feature_csv, index=False)
    deploy_root = analytical_root / "deploy"
    deploy_root.mkdir(parents=True, exist_ok=True)
    params_json = deploy_root / "full_data_parameters.json"
    params_json.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    fit_csv = analytical_root / "fit_sample.csv"
    fit_frame.to_csv(fit_csv, index=False)
    manifest = {
        "dataset_root": str(dataset_root),
        "fit_fraction": fit_fraction,
        "passes": passes,
        "fit_rows": int(len(fit_frame)),
        "fit_sample_csv": str(fit_csv),
        "feature_csv": str(feature_csv),
        "deploy_parameters_json": str(params_json),
    }
    _write_json(analytical_root / "manifest.json", manifest)
    return manifest


def _analytical_feature_frame_for_test(dataset_root: Path, analytical_feature_csv: Path) -> pd.DataFrame:
    raw_test = pd.read_csv(dataset_root / "test.csv", low_memory=False)
    analytical = pd.read_csv(analytical_feature_csv, low_memory=False)
    merged = raw_test[["row_uid", "op_type", "label_operator_actual_dur_us", "benchmark_source_dataset"]].merge(
        analytical[["row_uid", "op_type", "ana_calib_total_us", "ana_calib_family", "ana_calib_regime"]],
        on=["row_uid", "op_type"],
        how="left",
        validate="one_to_one",
    )
    return merged


def _analytical_review(dataset_root: Path, analytical_feature_csv: Path) -> dict[str, Any]:
    frame = _analytical_feature_frame_for_test(dataset_root, analytical_feature_csv)
    overall = _metrics_dict(frame["label_operator_actual_dur_us"], frame["ana_calib_total_us"])
    overall.update(_trend_metrics(frame["label_operator_actual_dur_us"], frame["ana_calib_total_us"]))
    family_rows: list[dict[str, Any]] = []
    for op_type, group in frame.groupby("op_type", sort=True):
        metrics = _metrics_dict(group["label_operator_actual_dur_us"], group["ana_calib_total_us"])
        metrics.update(_trend_metrics(group["label_operator_actual_dur_us"], group["ana_calib_total_us"]))
        metrics["op_type"] = str(op_type)
        family_rows.append(metrics)
    family_df = pd.DataFrame(family_rows).sort_values(["spearman", "mape", "op_type"], kind="stable")
    return {
        "overall": overall,
        "family_rows": family_df.to_dict(orient="records"),
    }


def _dataset_generalization_review(output_root: Path) -> dict[str, Any]:
    raw_summary = _read_json(EXP2_DATASET_ROOT / "dataset_summary.json")
    prebalanced_summary = _read_json(PREBALANCED_ROOT / "dataset" / "dataset_summary.json")
    rebalance_manifest = _read_json(PREBALANCED_ROOT / "distribution_control" / "rebalance_manifest.json")
    review = {
        "raw_exp2": {
            "dataset_root": str(EXP2_DATASET_ROOT),
            "rows": int(raw_summary["total_rows"]),
            "train_rows": int(raw_summary["split_row_counts"]["train"]),
            "test_rows": int(raw_summary["split_row_counts"]["test"]),
            "op_type_counts": dict(raw_summary["op_type_counts"]),
            "assessment": "good for performance chasing, not enough alone as a clean generalization proof because DLRM rows dominate and graph semantics remain.",
        },
        "semantic_free_prebalanced": {
            "dataset_root": str(PREBALANCED_ROOT / "dataset"),
            "rows": int(prebalanced_summary["total_rows"]),
            "train_rows": int(prebalanced_summary["split_row_counts"]["train"]),
            "test_rows": int(prebalanced_summary["split_row_counts"]["test"]),
            "op_type_counts": dict(prebalanced_summary["op_type_counts"]),
            "before_rows": int(rebalance_manifest["before_rows"]),
            "after_rows": int(rebalance_manifest["after_rows"]),
            "before_source_family_counts": dict(rebalance_manifest["before_source_family_counts"]),
            "after_source_family_counts": dict(rebalance_manifest["after_source_family_counts"]),
            "assessment": "the more defensible benchmark population because graph semantics are neutralized and the dataset is balanced before split; use this as the generalization argument, not raw exp2 alone.",
        },
        "verdict": {
            "raw_exp2": "improved coverage but not acceptance-proof as a standalone generalization claim",
            "semantic_free_prebalanced": "sufficiently defensible as the current generalization benchmark, with the remaining caveat that it is still single-hardware and single-runtime",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "dataset_generalization_review.json", review)
    lines = [
        "# Dataset Generalization Review",
        "",
        "## Verdict",
        "",
        f"- Raw `exp2_union_70_30`: {review['verdict']['raw_exp2']}",
        f"- `exp2_union_70_30_semantic_free_prebalanced_fixed_analytical`: {review['verdict']['semantic_free_prebalanced']}",
        "",
        "## Key Evidence",
        "",
        f"- Raw exp2 rows: `{review['raw_exp2']['rows']}`; prebalanced semantic-free rows: `{review['semantic_free_prebalanced']['rows']}`",
        f"- Prebalanced balancing reduced the mother dataset from `{review['semantic_free_prebalanced']['before_rows']}` to `{review['semantic_free_prebalanced']['after_rows']}` rows before splitting.",
        f"- Prebalanced source-family counts changed from `{review['semantic_free_prebalanced']['before_source_family_counts']}` to `{review['semantic_free_prebalanced']['after_source_family_counts']}`.",
        "",
        "## Assessment",
        "",
        f"- Raw exp2: {review['raw_exp2']['assessment']}",
        f"- Semantic-free prebalanced: {review['semantic_free_prebalanced']['assessment']}",
        "",
    ]
    (output_root / "dataset_generalization_review.md").write_text("\n".join(lines), encoding="utf-8")
    return review


def run_save_best_model(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = _fit_and_save_best_model(
        support_root=EXP2_SUPPORT_ROOT,
        test_augmented_csv=EXP2_ROOT / "eval_inputs" / "augmented" / "test.csv",
        test_prediction_dir=EXP2_ROOT / "evaluations" / "test",
        output_root=output_root,
    )
    return summary


def run_save_best_model_for_benchmark(output_root: Path, *, experiment_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    context = _resolve_benchmark_context(experiment_root)
    support_root = output_root / "support"
    model_roots = dict(context["model_roots"])
    train_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=Path(context["dataset_root"]) / "train.csv",
        analytical_parameters_json=Path(context["deploy_parameters_json"]),
        analytical_output_root=support_root / "analytical_train",
        output_csv=support_root / "train_augmented.csv",
        preprocess_config=None,
    )
    _evaluate_base_models_on_augmented_csv(
        augmented_csv=train_aug_csv,
        outputs_dir=support_root / "train_predictions",
        model_roots=model_roots,
        classed_original_data_root=Path(model_roots["classed_data_root"]),
        enhanced_classed_data_root=Path(model_roots["enhanced_classed_data_root"]),
    )
    val_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=Path(context["dataset_root"]) / "val.csv",
        analytical_parameters_json=Path(context["deploy_parameters_json"]),
        analytical_output_root=support_root / "analytical_val",
        output_csv=support_root / "val_augmented.csv",
        preprocess_config=None,
    )
    _evaluate_base_models_on_augmented_csv(
        augmented_csv=val_aug_csv,
        outputs_dir=support_root / "val_predictions",
        model_roots=model_roots,
        classed_original_data_root=Path(model_roots["classed_data_root"]),
        enhanced_classed_data_root=Path(model_roots["enhanced_classed_data_root"]),
    )
    best_model_summary = _fit_and_save_best_model(
        support_root=support_root,
        test_augmented_csv=Path(context["eval_augmented_test"]),
        test_prediction_dir=Path(context["eval_test_root"]),
        output_root=output_root / "model_bundle",
    )
    summary = {
        "experiment_root": str(experiment_root),
        "experiment_id": str(context["experiment_id"]),
        "description": str(context["description"]),
        "dataset_root": str(context["dataset_root"]),
        "test_augmented_csv": str(context["eval_augmented_test"]),
        "test_prediction_dir": str(context["eval_test_root"]),
        "base_model_roots": model_roots,
        "target_push_best_model": best_model_summary,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def run_fullfit_dataset(output_root: Path, *, dataset_root: Path, description: str = "") -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    analytical_root = output_root / "analytical"
    analytical_manifest = _train_analytical(dataset_root, analytical_root)
    deploy_parameters_json = analytical_root / "deploy" / "full_data_parameters.json"
    if not deploy_parameters_json.exists():
        raise FileNotFoundError(f"Expected analytical deploy parameters at {deploy_parameters_json}")
    analytical_review = _analytical_review(dataset_root, analytical_root / "analytical_features_full.csv")
    augmented_csv = _prepare_augmented_dataset(dataset_root, analytical_root, output_root / "augmented_dataset.csv")
    training_root = output_root / "training"
    original_roots = _train_original_models(
        dataset_root=dataset_root,
        analytical_root=analytical_root,
        output_root=training_root,
    )
    enhanced_roots = _train_enhanced_models(
        dataset_root=dataset_root,
        analytical_parameters_json=deploy_parameters_json,
        classed_root=Path(original_roots["classed_data_root"]),
        augmented_dataset_csv=augmented_csv,
        output_root=training_root,
    )
    model_roots = {**original_roots, **enhanced_roots}
    eval_results = _evaluate_model_roots_on_raw_csvs(
        raw_csvs={"test": dataset_root / "test.csv"},
        eval_aug_root=output_root / "eval_inputs" / "augmented",
        outputs_root=output_root / "evaluations",
        model_roots=model_roots,
        analytical_parameters_json=deploy_parameters_json,
        preprocess_config=None,
    )
    support_root = output_root / "support"
    train_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=dataset_root / "train.csv",
        analytical_parameters_json=deploy_parameters_json,
        analytical_output_root=support_root / "analytical_train",
        output_csv=support_root / "train_augmented.csv",
        preprocess_config=None,
    )
    _evaluate_base_models_on_augmented_csv(
        augmented_csv=train_aug_csv,
        outputs_dir=support_root / "train_predictions",
        model_roots=model_roots,
        classed_original_data_root=Path(model_roots["classed_data_root"]),
        enhanced_classed_data_root=Path(model_roots["enhanced_classed_data_root"]),
    )
    val_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=dataset_root / "val.csv",
        analytical_parameters_json=deploy_parameters_json,
        analytical_output_root=support_root / "analytical_val",
        output_csv=support_root / "val_augmented.csv",
        preprocess_config=None,
    )
    _evaluate_base_models_on_augmented_csv(
        augmented_csv=val_aug_csv,
        outputs_dir=support_root / "val_predictions",
        model_roots=model_roots,
        classed_original_data_root=Path(model_roots["classed_data_root"]),
        enhanced_classed_data_root=Path(model_roots["enhanced_classed_data_root"]),
    )
    best_model_summary = _fit_and_save_best_model(
        support_root=support_root,
        test_augmented_csv=output_root / "eval_inputs" / "augmented" / "test.csv",
        test_prediction_dir=output_root / "evaluations" / "test",
        output_root=output_root / "model_bundle",
    )
    summary = {
        "dataset_root": str(dataset_root),
        "description": description,
        "analytical_manifest": analytical_manifest,
        "analytical_review": analytical_review["overall"],
        "mlp_and_tree_test_metrics": {
            model_name: payload["metrics"]
            for model_name, payload in eval_results["test"].items()
        },
        "target_push_best_model": best_model_summary,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def run_downfit_study(output_root: Path, *, fit_fraction: float, passes: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    analytical_manifest = _build_reduced_analytical_artifacts(
        dataset_root=EXP2_DATASET_ROOT,
        output_root=output_root,
        fit_fraction=fit_fraction,
        passes=passes,
    )
    analytical_review = _analytical_review(EXP2_DATASET_ROOT, Path(analytical_manifest["feature_csv"]))
    _write_json(output_root / "analytical_review.json", analytical_review)
    augmented_csv = _prepare_augmented_dataset(EXP2_DATASET_ROOT, output_root / "analytical", output_root / "augmented_dataset.csv")
    training_root = output_root / "training"
    original_roots = _train_original_models(
        dataset_root=EXP2_DATASET_ROOT,
        analytical_root=output_root / "analytical",
        output_root=training_root,
    )
    enhanced_roots = _train_enhanced_models(
        dataset_root=EXP2_DATASET_ROOT,
        analytical_parameters_json=Path(analytical_manifest["deploy_parameters_json"]),
        classed_root=Path(original_roots["classed_data_root"]),
        augmented_dataset_csv=augmented_csv,
        output_root=training_root,
    )
    model_roots = {**original_roots, **enhanced_roots}
    eval_results = _evaluate_model_roots_on_raw_csvs(
        raw_csvs={"test": EXP2_DATASET_ROOT / "test.csv"},
        eval_aug_root=output_root / "eval_inputs" / "augmented",
        outputs_root=output_root / "evaluations",
        model_roots=model_roots,
        analytical_parameters_json=Path(analytical_manifest["deploy_parameters_json"]),
        preprocess_config=None,
    )
    support_root = output_root / "support"
    train_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=EXP2_DATASET_ROOT / "train.csv",
        analytical_parameters_json=Path(analytical_manifest["deploy_parameters_json"]),
        analytical_output_root=support_root / "analytical_train",
        output_csv=support_root / "train_augmented.csv",
        preprocess_config=None,
    )
    _evaluate_base_models_on_augmented_csv(
        augmented_csv=train_aug_csv,
        outputs_dir=support_root / "train_predictions",
        model_roots=model_roots,
        classed_original_data_root=Path(model_roots["classed_data_root"]),
        enhanced_classed_data_root=Path(model_roots["enhanced_classed_data_root"]),
    )
    val_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=EXP2_DATASET_ROOT / "val.csv",
        analytical_parameters_json=Path(analytical_manifest["deploy_parameters_json"]),
        analytical_output_root=support_root / "analytical_val",
        output_csv=support_root / "val_augmented.csv",
        preprocess_config=None,
    )
    _evaluate_base_models_on_augmented_csv(
        augmented_csv=val_aug_csv,
        outputs_dir=support_root / "val_predictions",
        model_roots=model_roots,
        classed_original_data_root=Path(model_roots["classed_data_root"]),
        enhanced_classed_data_root=Path(model_roots["enhanced_classed_data_root"]),
    )
    best_model_summary = _fit_and_save_best_model(
        support_root=support_root,
        test_augmented_csv=output_root / "eval_inputs" / "augmented" / "test.csv",
        test_prediction_dir=output_root / "evaluations" / "test",
        output_root=output_root / "target_push_model",
    )
    summary = {
        "fit_fraction": fit_fraction,
        "passes": passes,
        "analytical_manifest": analytical_manifest,
        "analytical_review": analytical_review["overall"],
        "mlp_and_tree_test_metrics": {
            model_name: payload["metrics"]
            for model_name, payload in eval_results["test"].items()
        },
        "target_push_best_model": best_model_summary,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def run_downfit_residual_study(output_root: Path, *, fit_fraction: float, passes: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    analytical_manifest = _build_reduced_analytical_artifacts(
        dataset_root=EXP2_DATASET_ROOT,
        output_root=output_root,
        fit_fraction=fit_fraction,
        passes=passes,
    )
    analytical_review = _analytical_review(EXP2_DATASET_ROOT, Path(analytical_manifest["feature_csv"]))
    _write_json(output_root / "analytical_review.json", analytical_review)

    classed_root = output_root / "classed_original_data"
    classed_manifest = build_classed_dataset_artifacts(
        input_data_dir=EXP2_DATASET_ROOT,
        analytical_dir=output_root / "analytical",
        output_dir=classed_root,
        feature_branch="with_analytical",
    )

    original_single_root = output_root / "original_single_mlp"
    original_single_manifest = train_fair_single_model(
        classed_data_root=classed_root,
        output_root=original_single_root,
    )
    residual_single_root = output_root / "analytical_residual_single_mlp"
    residual_single_manifest = _train_analytical_residual_single_model(
        classed_data_root=classed_root,
        output_root=residual_single_root,
    )

    eval_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=EXP2_DATASET_ROOT / "test.csv",
        analytical_parameters_json=Path(analytical_manifest["deploy_parameters_json"]),
        analytical_output_root=output_root / "eval_inputs" / "analytical_test",
        output_csv=output_root / "eval_inputs" / "augmented" / "test.csv",
        preprocess_config=None,
    )
    eval_root = output_root / "evaluations" / "test"
    eval_root.mkdir(parents=True, exist_ok=True)
    analytical_eval = _evaluate_analytical_reference(
        eval_aug_csv,
        eval_root / "analytical_reference.csv",
    )
    original_single_eval = _evaluate_original_single(
        original_single_root,
        eval_aug_csv,
        eval_root / "original_single_mlp.csv",
    )
    residual_single_eval = _evaluate_original_single(
        residual_single_root,
        eval_aug_csv,
        eval_root / "analytical_residual_single_mlp.csv",
    )

    summary = {
        "fit_fraction": fit_fraction,
        "passes": passes,
        "analytical_manifest": analytical_manifest,
        "classed_manifest_json": str(classed_root / "dataset_summary.json"),
        "analytical_review": analytical_review["overall"],
        "model_manifests": {
            "original_single_mlp": original_single_manifest,
            "analytical_residual_single_mlp": residual_single_manifest,
        },
        "test_metrics": {
            "analytical_reference": analytical_eval["metrics"],
            "original_single_mlp": original_single_eval["metrics"],
            "analytical_residual_single_mlp": residual_single_eval["metrics"],
        },
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Target-push utilities for mixed-dataset generalization studies.")
    parser.add_argument("--output-root", default=str(TARGET_PUSH_ROOT), help="Artifact root for target-push studies.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("dataset-review", help="Summarize whether the current mixed datasets are defensible for generalization.")
    subparsers.add_parser("save-best", help="Train and save the current best soft-blend model on the raw exp2 mixed dataset.")
    save_best_benchmark = subparsers.add_parser(
        "save-best-benchmark",
        help="Train and save the current best soft-blend model on an existing benchmark run root with run_summary.json.",
    )
    save_best_benchmark.add_argument(
        "--experiment-root",
        required=True,
        help="Benchmark experiment root that already contains run_summary.json, dataset/, analytical/, eval_inputs/, evaluations/, and training/.",
    )
    save_best_benchmark.add_argument(
        "--tag",
        default=None,
        help="Optional output directory tag under --output-root. Defaults to '<experiment_id>_fullfit_model'.",
    )
    fullfit_dataset = subparsers.add_parser(
        "fullfit-dataset",
        help="Fully fit Analytical plus all benchmark base models on a dataset root, then run target-push soft blending.",
    )
    fullfit_dataset.add_argument("--dataset-root", required=True, help="Dataset root containing dataset_full.csv, train.csv, val.csv, and test.csv.")
    fullfit_dataset.add_argument(
        "--tag",
        default="fullfit_dataset_model",
        help="Output directory tag under --output-root.",
    )
    fullfit_dataset.add_argument(
        "--description",
        default="",
        help="Optional description recorded in the saved summary.",
    )

    downfit = subparsers.add_parser("downfit-study", help="Reduce analytical fit data, retrain models, and measure analytical trend plus end-model accuracy.")
    downfit.add_argument("--fit-fraction", type=float, default=0.25, help="Fraction of exp2 dataset_full rows used to fit analytical parameters.")
    downfit.add_argument("--passes", type=int, default=3, help="Analytical calibration passes.")

    residual = subparsers.add_parser(
        "downfit-residual-study",
        help="Reduce analytical fit data and train only the MLPs directly relevant to analytical-error correction.",
    )
    residual.add_argument("--fit-fraction", type=float, default=0.25, help="Fraction of exp2 dataset_full rows used to fit analytical parameters.")
    residual.add_argument("--passes", type=int, default=3, help="Analytical calibration passes.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if args.command == "dataset-review":
        payload = _dataset_generalization_review(output_root / "review")
    elif args.command == "save-best":
        payload = run_save_best_model(output_root / "best_fullfit_model")
    elif args.command == "save-best-benchmark":
        experiment_root = Path(args.experiment_root)
        context = _resolve_benchmark_context(experiment_root)
        tag = args.tag or f"{context['experiment_id']}_fullfit_model"
        payload = run_save_best_model_for_benchmark(output_root / tag, experiment_root=experiment_root)
    elif args.command == "fullfit-dataset":
        payload = run_fullfit_dataset(
            output_root / args.tag,
            dataset_root=Path(args.dataset_root),
            description=str(args.description),
        )
    elif args.command == "downfit-study":
        fraction_tag = str(args.fit_fraction).replace(".", "p")
        payload = run_downfit_study(output_root / f"downfit_frac_{fraction_tag}", fit_fraction=float(args.fit_fraction), passes=int(args.passes))
    elif args.command == "downfit-residual-study":
        fraction_tag = str(args.fit_fraction).replace(".", "p")
        payload = run_downfit_residual_study(
            output_root / f"downfit_residual_frac_{fraction_tag}",
            fit_fraction=float(args.fit_fraction),
            passes=int(args.passes),
        )
    else:
        raise ValueError(args.command)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
