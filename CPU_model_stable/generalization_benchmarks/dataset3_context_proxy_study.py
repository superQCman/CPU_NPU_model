from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cpu_perf_model import train_mlp
from cpu_perf_model.infer_mlp_onnx import run_inference
from generalization_benchmarks.feature_enhancement import ALL_DERIVED_FEATURES
from generalization_benchmarks import target_push as tp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "latest"
    / "generalization_target_push"
    / "dataset3_fullfit_semantic_free_compare"
)
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "latest" / "dataset3_context_proxy_study"
SEED = 42

BASELINE_NUMERIC_FEATURES = [
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
    "feat_output_elements_per_batch",
    "ana_calib_total_us",
    "feat_output_input_bytes_ratio",
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
    "ana_calib_compute_us",
]

RAW_CONTEXT_NUMERIC_FEATURES = [
    "local_ctx_same_op_overlap_ratio_mean",
    "local_ctx_overlap_ratio_mean",
    "local_ctx_cross_task_overlap_ratio_mean",
    "local_ctx_mean_other_active_mean",
    "local_ctx_mean_other_tasks_mean",
    "combo_task_parallel_fraction",
    "combo_task_weighted_mean_parallel_concurrency",
    "combo_op_parallel_fraction",
    "combo_op_weighted_mean_parallel_concurrency",
    "comp_feat_pressure_threads",
    "comp_feat_pressure_ws_to_l2_ratio",
    "comp_feat_pressure_ws_to_l3_ratio",
    "hw_ratio_threads_to_total_cores",
]

DERIVED_CONTEXT_CATEGORICAL = [
    "op_type",
    "ctx_prev_op_type",
    "ctx_next_op_type",
]

DERIVED_CONTEXT_EXTENDED_CATEGORICAL = [
    *DERIVED_CONTEXT_CATEGORICAL,
    "ctx_prev2_op_type",
    "ctx_next2_op_type",
    "ctx_prev_output_shape",
    "ctx_next_output_shape",
    "ctx_prev2_output_shape",
    "ctx_next2_output_shape",
]

DERIVED_CONTEXT_NUMERIC = [
    "ctx_group_len",
    "ctx_pos",
    "ctx_pos_frac",
    "ctx_remaining_ops",
    "ctx_prev_output_size",
    "ctx_prev_activation_size",
    "ctx_prev_feat_io_bytes_sum",
    "ctx_prev_feat_lookup_count",
    "ctx_prev_feat_gemm_mac_count",
    "ctx_prev_num_threads",
    "ctx_prev_ana_calib_total_us",
    "ctx_next_output_size",
    "ctx_next_activation_size",
    "ctx_next_feat_io_bytes_sum",
    "ctx_next_feat_lookup_count",
    "ctx_next_feat_gemm_mac_count",
    "ctx_next_num_threads",
    "ctx_next_ana_calib_total_us",
    "ctx_prev_same_numel",
    "ctx_next_same_numel",
    "ctx_prev_is_heavy",
    "ctx_next_is_heavy",
    "ctx_prev_is_meta",
    "ctx_next_is_meta",
]

DERIVED_CONTEXT_EXTENDED_NUMERIC = [
    *DERIVED_CONTEXT_NUMERIC,
    "ctx_prev2_output_size",
    "ctx_prev2_activation_size",
    "ctx_prev2_feat_io_bytes_sum",
    "ctx_prev2_feat_lookup_count",
    "ctx_prev2_feat_gemm_mac_count",
    "ctx_prev2_num_threads",
    "ctx_prev2_ana_calib_total_us",
    "ctx_next2_output_size",
    "ctx_next2_activation_size",
    "ctx_next2_feat_io_bytes_sum",
    "ctx_next2_feat_lookup_count",
    "ctx_next2_feat_gemm_mac_count",
    "ctx_next2_num_threads",
    "ctx_next2_ana_calib_total_us",
    "ctx_nearest_prev_heavy_distance",
    "ctx_nearest_next_heavy_distance",
    "ctx_nearest_prev_meta_distance",
    "ctx_nearest_next_meta_distance",
]

VARIANT_SPECS: dict[str, dict[str, list[str]]] = {
    "baseline_semantic_free": {
        "categorical": ["op_type"],
        "numeric": BASELINE_NUMERIC_FEATURES,
    },
    "shape_params": {
        "categorical": ["op_type", "input_type_shape", "output_type_shape"],
        "numeric": list(dict.fromkeys(BASELINE_NUMERIC_FEATURES + list(ALL_DERIVED_FEATURES))),
    },
    "local_context": {
        "categorical": DERIVED_CONTEXT_CATEGORICAL,
        "numeric": list(dict.fromkeys(BASELINE_NUMERIC_FEATURES + RAW_CONTEXT_NUMERIC_FEATURES + DERIVED_CONTEXT_NUMERIC)),
    },
    "local_context_residual": {
        "categorical": DERIVED_CONTEXT_CATEGORICAL,
        "numeric": list(dict.fromkeys(BASELINE_NUMERIC_FEATURES + RAW_CONTEXT_NUMERIC_FEATURES + DERIVED_CONTEXT_NUMERIC)),
    },
    "shape_plus_context": {
        "categorical": list(dict.fromkeys(["op_type", "input_type_shape", "output_type_shape"] + DERIVED_CONTEXT_CATEGORICAL)),
        "numeric": list(
            dict.fromkeys(
                BASELINE_NUMERIC_FEATURES
                + list(ALL_DERIVED_FEATURES)
                + RAW_CONTEXT_NUMERIC_FEATURES
                + DERIVED_CONTEXT_NUMERIC
            )
        ),
    },
    "shape_plus_context_neighbor_shapes": {
        "categorical": list(
            dict.fromkeys(
                ["op_type", "input_type_shape", "output_type_shape"] + DERIVED_CONTEXT_EXTENDED_CATEGORICAL
            )
        ),
        "numeric": list(
            dict.fromkeys(
                BASELINE_NUMERIC_FEATURES
                + list(ALL_DERIVED_FEATURES)
                + RAW_CONTEXT_NUMERIC_FEATURES
                + DERIVED_CONTEXT_EXTENDED_NUMERIC
            )
        ),
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metrics_from_predictions(pred_csv: Path) -> dict[str, float]:
    frame = pd.read_csv(pred_csv, low_memory=False)
    target = pd.to_numeric(frame["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred = pd.to_numeric(frame["pred_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ape = np.abs(pred - target) / np.clip(target, 1e-9, None)
    ss_res = float(np.square(target - pred).sum())
    ss_tot = float(np.square(target - target.mean()).sum())
    pearson = float(np.corrcoef(target, pred)[0, 1]) if len(frame) > 1 else 0.0
    return {
        "rows": int(len(frame)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.quantile(ape, 0.9)),
        "p95_ape": float(np.quantile(ape, 0.95)),
        "pearson": 0.0 if np.isnan(pearson) else pearson,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def _fit_ratio_clip_spec(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    base_column: str = "ana_calib_total_us",
    group_column: str = "op_type",
    q_low: float,
    q_high: float,
    min_rows: int = 128,
) -> dict[str, Any]:
    work = frame.copy()
    base = pd.to_numeric(work[base_column], errors="coerce").fillna(0.0).clip(lower=1e-9)
    pred = pd.to_numeric(work[prediction_column], errors="coerce").fillna(0.0).clip(lower=1e-9)
    work["_ratio"] = pred / base
    spec: dict[str, Any] = {
        "prediction_column": prediction_column,
        "base_column": base_column,
        "group_column": group_column,
        "q_low": float(q_low),
        "q_high": float(q_high),
        "min_rows": int(min_rows),
        "global": {
            "low": float(work["_ratio"].quantile(q_low)),
            "high": float(work["_ratio"].quantile(q_high)),
        },
        "groups": {},
    }
    for group_value, group in work.groupby(group_column, sort=True):
        if len(group) < min_rows:
            continue
        spec["groups"][str(group_value)] = {
            "low": float(group["_ratio"].quantile(q_low)),
            "high": float(group["_ratio"].quantile(q_high)),
            "rows": int(len(group)),
        }
    return spec


def _apply_ratio_clip_spec(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    spec: dict[str, Any],
) -> pd.DataFrame:
    out = frame.copy()
    base_column = str(spec["base_column"])
    group_column = str(spec["group_column"])
    base = pd.to_numeric(out[base_column], errors="coerce").fillna(0.0).clip(lower=1e-9).to_numpy(dtype=float)
    pred = pd.to_numeric(out[prediction_column], errors="coerce").fillna(0.0).clip(lower=1e-9).to_numpy(dtype=float)
    low = np.full(len(out), float(spec["global"]["low"]), dtype=float)
    high = np.full(len(out), float(spec["global"]["high"]), dtype=float)
    group_values = out[group_column].astype(str)
    for group_value, bounds in dict(spec["groups"]).items():
        mask = (group_values == str(group_value)).to_numpy()
        low[mask] = float(bounds["low"])
        high[mask] = float(bounds["high"])
    out[prediction_column] = np.clip(pred / base, low, high) * base
    return out


def _group_key(frame: pd.DataFrame) -> pd.Series:
    if "sample_group" in frame.columns:
        return frame["sample_group"].fillna("missing").astype(str)
    cols = [column for column in ["source_name", "case_id", "combo"] if column in frame.columns]
    if not cols:
        return pd.Series("global", index=frame.index, dtype=str)
    work = frame[cols].fillna("missing").astype(str)
    return work.agg("::".join, axis=1)


def _derive_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_ctx_group"] = _group_key(work)
    work["_ctx_pos"] = pd.to_numeric(work.get("op_idx"), errors="coerce").fillna(0).astype(int)
    work = work.sort_values(["_ctx_group", "_ctx_pos", "row_uid"], kind="stable").reset_index(drop=True)
    group_size = work.groupby("_ctx_group")["row_uid"].transform("size").astype(float)
    pos = work.groupby("_ctx_group").cumcount().astype(float)

    def shifted(column: str, offset: int, fill_value: Any) -> pd.Series:
        if column not in work.columns:
            return pd.Series(fill_value, index=work.index)
        series = work.groupby("_ctx_group", sort=False)[column].shift(offset)
        if isinstance(fill_value, str):
            return series.fillna(fill_value).astype(str)
        return pd.to_numeric(series, errors="coerce").fillna(float(fill_value)).astype(float)

    prev_op = shifted("op_type", 1, "BOS")
    next_op = shifted("op_type", -1, "EOS")
    prev2_op = shifted("op_type", 2, "BOS2")
    next2_op = shifted("op_type", -2, "EOS2")
    prev_shape = shifted("output_type_shape", 1, "BOS_SHAPE")
    next_shape = shifted("output_type_shape", -1, "EOS_SHAPE")
    prev2_shape = shifted("output_type_shape", 2, "BOS2_SHAPE")
    next2_shape = shifted("output_type_shape", -2, "EOS2_SHAPE")
    prev_out = shifted("output_size", 1, 0.0)
    next_out = shifted("output_size", -1, 0.0)
    prev2_out = shifted("output_size", 2, 0.0)
    next2_out = shifted("output_size", -2, 0.0)
    prev_act = shifted("activation_size", 1, 0.0)
    next_act = shifted("activation_size", -1, 0.0)
    prev2_act = shifted("activation_size", 2, 0.0)
    next2_act = shifted("activation_size", -2, 0.0)
    prev_io = shifted("feat_io_bytes_sum", 1, 0.0)
    next_io = shifted("feat_io_bytes_sum", -1, 0.0)
    prev2_io = shifted("feat_io_bytes_sum", 2, 0.0)
    next2_io = shifted("feat_io_bytes_sum", -2, 0.0)
    prev_lookup = shifted("feat_lookup_count", 1, 0.0)
    next_lookup = shifted("feat_lookup_count", -1, 0.0)
    prev2_lookup = shifted("feat_lookup_count", 2, 0.0)
    next2_lookup = shifted("feat_lookup_count", -2, 0.0)
    prev_gemm = shifted("feat_gemm_mac_count", 1, 0.0)
    next_gemm = shifted("feat_gemm_mac_count", -1, 0.0)
    prev2_gemm = shifted("feat_gemm_mac_count", 2, 0.0)
    next2_gemm = shifted("feat_gemm_mac_count", -2, 0.0)
    prev_thr = shifted("num_threads", 1, 0.0)
    next_thr = shifted("num_threads", -1, 0.0)
    prev2_thr = shifted("num_threads", 2, 0.0)
    next2_thr = shifted("num_threads", -2, 0.0)
    prev_ana = shifted("ana_calib_total_us", 1, 0.0)
    next_ana = shifted("ana_calib_total_us", -1, 0.0)
    prev2_ana = shifted("ana_calib_total_us", 2, 0.0)
    next2_ana = shifted("ana_calib_total_us", -2, 0.0)

    meta_ops = {"Shape", "Reshape", "Unsqueeze", "Flatten"}
    heavy_ops = {"Gather", "ReduceSum", "Gemm", "MatMul", "Transpose", "Concat"}
    activation_size = pd.to_numeric(work.get("activation_size"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    output_size = pd.to_numeric(work.get("output_size"), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "row_uid": work["row_uid"].astype(str),
            "ctx_prev_op_type": prev_op,
            "ctx_next_op_type": next_op,
            "ctx_prev2_op_type": prev2_op,
            "ctx_next2_op_type": next2_op,
            "ctx_prev_output_shape": prev_shape,
            "ctx_next_output_shape": next_shape,
            "ctx_prev2_output_shape": prev2_shape,
            "ctx_next2_output_shape": next2_shape,
            "ctx_group_len": group_size,
            "ctx_pos": pos,
            "ctx_pos_frac": np.where(group_size > 1.0, pos / (group_size - 1.0), 0.0),
            "ctx_remaining_ops": np.clip(group_size - 1.0 - pos, a_min=0.0, a_max=None),
            "ctx_prev_output_size": prev_out,
            "ctx_prev_activation_size": prev_act,
            "ctx_prev2_output_size": prev2_out,
            "ctx_prev2_activation_size": prev2_act,
            "ctx_prev_feat_io_bytes_sum": prev_io,
            "ctx_prev_feat_lookup_count": prev_lookup,
            "ctx_prev_feat_gemm_mac_count": prev_gemm,
            "ctx_prev_num_threads": prev_thr,
            "ctx_prev_ana_calib_total_us": prev_ana,
            "ctx_prev2_feat_io_bytes_sum": prev2_io,
            "ctx_prev2_feat_lookup_count": prev2_lookup,
            "ctx_prev2_feat_gemm_mac_count": prev2_gemm,
            "ctx_prev2_num_threads": prev2_thr,
            "ctx_prev2_ana_calib_total_us": prev2_ana,
            "ctx_next_output_size": next_out,
            "ctx_next_activation_size": next_act,
            "ctx_next2_output_size": next2_out,
            "ctx_next2_activation_size": next2_act,
            "ctx_next_feat_io_bytes_sum": next_io,
            "ctx_next_feat_lookup_count": next_lookup,
            "ctx_next_feat_gemm_mac_count": next_gemm,
            "ctx_next_num_threads": next_thr,
            "ctx_next_ana_calib_total_us": next_ana,
            "ctx_next2_feat_io_bytes_sum": next2_io,
            "ctx_next2_feat_lookup_count": next2_lookup,
            "ctx_next2_feat_gemm_mac_count": next2_gemm,
            "ctx_next2_num_threads": next2_thr,
            "ctx_next2_ana_calib_total_us": next2_ana,
            "ctx_prev_same_numel": np.isclose(prev_out.to_numpy(dtype=float), activation_size).astype(float),
            "ctx_next_same_numel": np.isclose(output_size, next_act.to_numpy(dtype=float)).astype(float),
            "ctx_prev_is_heavy": prev_op.isin(heavy_ops).astype(float),
            "ctx_next_is_heavy": next_op.isin(heavy_ops).astype(float),
            "ctx_prev_is_meta": prev_op.isin(meta_ops).astype(float),
            "ctx_next_is_meta": next_op.isin(meta_ops).astype(float),
        }
    )
    prev_heavy = pd.Series(np.where(work["op_type"].isin(heavy_ops), pos, np.nan), index=work.index).groupby(work["_ctx_group"], sort=False).ffill()
    next_heavy = pd.Series(np.where(work["op_type"].isin(heavy_ops), pos, np.nan), index=work.index).groupby(work["_ctx_group"], sort=False).bfill()
    prev_meta = pd.Series(np.where(work["op_type"].isin(meta_ops), pos, np.nan), index=work.index).groupby(work["_ctx_group"], sort=False).ffill()
    next_meta = pd.Series(np.where(work["op_type"].isin(meta_ops), pos, np.nan), index=work.index).groupby(work["_ctx_group"], sort=False).bfill()
    out["ctx_nearest_prev_heavy_distance"] = np.where(np.isnan(prev_heavy), group_size, np.clip(pos - prev_heavy.to_numpy(dtype=float), 0.0, None))
    out["ctx_nearest_next_heavy_distance"] = np.where(np.isnan(next_heavy), group_size, np.clip(next_heavy.to_numpy(dtype=float) - pos, 0.0, None))
    out["ctx_nearest_prev_meta_distance"] = np.where(np.isnan(prev_meta), group_size, np.clip(pos - prev_meta.to_numpy(dtype=float), 0.0, None))
    out["ctx_nearest_next_meta_distance"] = np.where(np.isnan(next_meta), group_size, np.clip(next_meta.to_numpy(dtype=float) - pos, 0.0, None))
    return out


def _prepare_context_augmented_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in ["input_type_shape", "output_type_shape"]:
        work[column] = work[column].fillna("missing").astype(str)
    for column in [
        "ctx_prev_output_shape",
        "ctx_next_output_shape",
        "ctx_prev2_output_shape",
        "ctx_next2_output_shape",
    ]:
        if column in work.columns:
            work[column] = work[column].fillna("missing").astype(str)
    for column in list(ALL_DERIVED_FEATURES) + BASELINE_NUMERIC_FEATURES + RAW_CONTEXT_NUMERIC_FEATURES:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0).astype(float)
    context = _derive_context_features(work)
    out = work.merge(context, on="row_uid", how="left", validate="one_to_one")
    return out


def _feature_manifest(variant: str) -> dict[str, Any]:
    spec = VARIANT_SPECS[variant]
    payload = {
        "feature_dialect": f"dataset3_context_proxy_{variant}",
        "numeric_features": list(dict.fromkeys(spec["numeric"])),
        "categorical_features": list(dict.fromkeys(spec["categorical"])),
        "feature_pool_description": f"dataset_3 semantic-free context surrogate study: {variant}",
    }
    if variant.endswith("_residual"):
        payload["analytical_base_column"] = "ana_calib_total_us"
        payload["residual_target_column"] = "label_operator_residual_log"
    return payload


def _build_dataset(frame: pd.DataFrame, output_dir: Path, variant: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _feature_manifest(variant)
    for split_name in ["train", "val", "test"]:
        split_df = frame[frame["split"] == split_name].copy()
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)
    frame.to_csv(output_dir / "dataset_full.csv", index=False)
    (output_dir / "feature_columns.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    payload = {
        "variant": variant,
        "output_dir": str(output_dir),
        "row_count": int(len(frame)),
        "split_row_counts": {split: int((frame["split"] == split).sum()) for split in ["train", "val", "test"]},
        "feature_manifest": manifest,
    }
    _write_json(output_dir / "manifest.json", payload)
    return payload


def _train_single_variant(dataset_dir: Path, output_root: Path, variant: str, *, npu_device_id: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / "model"
    target_mode = "analytical_residual" if variant.endswith("_residual") else "direct_us"
    summary = train_mlp.train_model(
        data_dir=dataset_dir,
        output_dir=model_dir,
        hidden_layers=(128, 128, 128, 128, 128, 128),
        batch_size=1024,
        max_iter=200,
        alpha=1e-4,
        learning_rate_init=1e-3,
        seed=SEED,
        log_target=True,
        target_mode=target_mode,
        train_device="npu",
        npu_device_id=npu_device_id,
        early_stopping_patience=12,
        export_onnx=True,
        onnx_opset=17,
    )
    metrics = _metrics_from_predictions(model_dir / "predictions_test.csv")
    payload = {
        "model_dir": str(model_dir),
        "model_summary": summary,
        "test_metrics": metrics,
    }
    _write_json(output_root / "summary.json", payload)
    return payload


def _fit_blend_plan_custom(val_frame: pd.DataFrame, candidates: list[str]) -> dict[str, Any]:
    weights = np.linspace(0.0, 1.0, 101)
    plan: dict[str, Any] = {}
    for op_type, group in val_frame.groupby("op_type", sort=True):
        ranked = sorted(candidates, key=lambda name: tp._candidate_key(group["target_us"], group[name]))[:4]
        best_mode: dict[str, Any] = {"mode": "single", "single_model": ranked[0]}
        best_key = tp._candidate_key(group["target_us"], group[ranked[0]])
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                left = ranked[i]
                right = ranked[j]
                left_pred = pd.to_numeric(group[left], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                right_pred = pd.to_numeric(group[right], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                for weight in weights:
                    pred = weight * left_pred + (1.0 - weight) * right_pred
                    candidate_key = tp._candidate_key(group["target_us"], pd.Series(pred, index=group.index))
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


def _run_single(output_root: Path, variants: list[str], *, npu_device_id: int) -> dict[str, Any]:
    augmented = pd.read_csv(EXPERIMENT_ROOT / "augmented_dataset.csv", low_memory=False)
    prepared = _prepare_context_augmented_frame(augmented)
    summary: dict[str, Any] = {"experiment_root": str(EXPERIMENT_ROOT), "variants": {}}
    for variant in variants:
        variant_root = output_root / variant
        dataset_meta = _build_dataset(prepared, variant_root / "dataset", variant)
        train_meta = _train_single_variant(variant_root / "dataset", variant_root, variant, npu_device_id=npu_device_id)
        summary["variants"][variant] = {"dataset": dataset_meta, "training": train_meta}
    _write_json(output_root / "summary.json", summary)
    return summary


def _inject_saved_target_push_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    model_root = EXPERIMENT_ROOT / "model_bundle" / "model"
    model_manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
    category_maps = {
        column: {str(key): int(value) for key, value in mapping.items()}
        for column, mapping in dict(model_manifest["category_maps"]).items()
    }
    encoded = tp._apply_category_maps(frame, category_maps)
    feature_columns = [column for column in model_manifest["feature_columns"] if column in encoded.columns]
    for column in feature_columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(0.0)
    global_model = joblib.load(model_root / "global_hgbr.joblib")
    family_extra = joblib.load(model_root / "family_extra.joblib")
    family_extra_regime = joblib.load(model_root / "family_extra_regime.joblib")
    out = frame.copy()
    out["global_hgbr"] = np.expm1(global_model.predict(encoded[feature_columns]))
    out["family_extra"] = tp._predict_family_models(encoded, feature_columns, family_extra)
    out["family_extra_regime"] = tp._predict_family_models(encoded, feature_columns, family_extra_regime)
    return out


def _augment_with_context(csv_path: Path) -> Path:
    frame = pd.read_csv(csv_path, low_memory=False)
    augmented = _prepare_context_augmented_frame(frame)
    out_csv = csv_path.parent / f"{csv_path.stem}.context_augmented.csv"
    augmented.to_csv(out_csv, index=False)
    return out_csv


def _run_bestfullfit(
    variant: str,
    single_root: Path,
    output_root: Path,
    *,
    clip_context_ratio: bool = False,
    clip_q_low: float = 0.0,
    clip_q_high: float = 0.999,
    clip_min_rows: int = 128,
) -> dict[str, Any]:
    model_dir = single_root / variant / "model"
    support_root = EXPERIMENT_ROOT / "support"
    train_aug_csv = _augment_with_context(support_root / "train_augmented.csv")
    val_aug_csv = _augment_with_context(support_root / "val_augmented.csv")
    test_aug_csv = _augment_with_context(EXPERIMENT_ROOT / "eval_inputs" / "augmented" / "test.csv")

    pred_root = output_root / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    train_pred_csv = pred_root / "train_context_single_mlp.csv"
    val_pred_csv = pred_root / "val_context_single_mlp.csv"
    test_pred_csv = pred_root / "test_context_single_mlp.csv"
    for input_csv, output_csv in [
        (train_aug_csv, train_pred_csv),
        (val_aug_csv, val_pred_csv),
        (test_aug_csv, test_pred_csv),
    ]:
        run_inference(
            model_dir=model_dir,
            input_csv=input_csv,
            output_csv=output_csv,
            provider="cpu",
            batch_size=4096,
        )

    train_frame = tp._load_support_frame(support_root / "train_augmented.csv", support_root / "train_predictions")
    val_frame = tp._load_support_frame(support_root / "val_augmented.csv", support_root / "val_predictions")
    test_frame = tp._load_support_frame(EXPERIMENT_ROOT / "eval_inputs" / "augmented" / "test.csv", EXPERIMENT_ROOT / "evaluations" / "test")

    merged_frames: list[pd.DataFrame] = []
    for frame, pred_csv in [(train_frame, train_pred_csv), (val_frame, val_pred_csv), (test_frame, test_pred_csv)]:
        pred_df = pd.read_csv(pred_csv, low_memory=False)[["row_uid", "pred_us"]].rename(columns={"pred_us": "context_single_mlp"})
        merged_frames.append(frame.merge(pred_df, on="row_uid", how="left", validate="one_to_one"))
    train_frame, val_frame, test_frame = merged_frames
    raw_context_test_metrics = tp._metrics_dict(test_frame["target_us"], test_frame["context_single_mlp"])

    clip_spec: dict[str, Any] | None = None
    if clip_context_ratio:
        clip_spec = _fit_ratio_clip_spec(
            train_frame,
            prediction_column="context_single_mlp",
            q_low=clip_q_low,
            q_high=clip_q_high,
            min_rows=clip_min_rows,
        )
        train_frame = _apply_ratio_clip_spec(train_frame, prediction_column="context_single_mlp", spec=clip_spec)
        val_frame = _apply_ratio_clip_spec(val_frame, prediction_column="context_single_mlp", spec=clip_spec)
        test_frame = _apply_ratio_clip_spec(test_frame, prediction_column="context_single_mlp", spec=clip_spec)

    train_frame = _inject_saved_target_push_candidates(train_frame)
    val_frame = _inject_saved_target_push_candidates(val_frame)
    test_frame = _inject_saved_target_push_candidates(test_frame)

    candidates = list(tp.TARGET_PUSH_CANDIDATES) + ["context_single_mlp"]
    blend_plan = _fit_blend_plan_custom(val_frame, candidates)
    val_blended = tp._apply_blend_plan(val_frame, blend_plan)
    test_blended = tp._apply_blend_plan(test_frame, blend_plan)
    calibration = tp._fit_shrink_calibration(val_blended)
    test_output = tp._apply_shrink_calibration(test_blended, calibration)

    pred_csv = output_root / "test_predictions.csv"
    final = test_output[["row_uid", "op_type", "split", "target_us", "pred_us", "mix_id"]].copy()
    final["abs_error_us"] = np.abs(final["pred_us"] - final["target_us"])
    final["ape"] = final["abs_error_us"] / np.clip(final["target_us"], 1e-9, None)
    final.to_csv(pred_csv, index=False)

    baseline_summary = json.loads((EXPERIMENT_ROOT / "summary.json").read_text(encoding="utf-8"))
    payload = {
        "experiment_root": str(EXPERIMENT_ROOT),
        "variant": variant,
        "baseline_best_fullfit_test_metrics": baseline_summary["target_push_best_model"]["test_metrics"],
        "context_single_metrics_pre_clip": raw_context_test_metrics,
        "context_single_metrics": tp._metrics_dict(test_frame["target_us"], test_frame["context_single_mlp"]),
        "blend_metrics": tp._metrics_dict(final["target_us"], final["pred_us"]),
        "candidates": candidates,
        "predictions_csv": str(pred_csv),
        "blend_plan_json": str(output_root / "blend_plan.json"),
        "calibration_json": str(output_root / "calibration.json"),
        "clip_context_ratio": bool(clip_context_ratio),
    }
    _write_json(output_root / "blend_plan.json", blend_plan)
    _write_json(output_root / "calibration.json", calibration)
    if clip_spec is not None:
        payload["clip_spec_json"] = str(output_root / "clip_spec.json")
        _write_json(output_root / "clip_spec.json", clip_spec)
    _write_json(output_root / "summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dataset_3 DLRM-only study replacing graph semantics with transferable local context features.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("run-single", help="Train dataset_3 semantic-free single-MLP variants on NPU.")
    single.add_argument(
        "--variants",
        nargs="+",
        default=["baseline_semantic_free", "shape_params", "local_context", "local_context_residual", "shape_plus_context"],
        choices=sorted(VARIANT_SPECS),
    )
    single.add_argument("--output-root", default=str(ARTIFACT_ROOT / "single"))
    single.add_argument("--npu-device-id", type=int, default=0)

    best = subparsers.add_parser("run-bestfullfit", help="Validate the best dataset_3 context-aware single MLP inside the semantic-free best_fullfit stack.")
    best.add_argument("--variant", required=True, choices=sorted(VARIANT_SPECS))
    best.add_argument("--single-root", default=str(ARTIFACT_ROOT / "single"))
    best.add_argument("--output-root", default=str(ARTIFACT_ROOT / "bestfullfit"))
    best.add_argument("--clip-context-ratio", action="store_true", help="Clip context_single_mlp to train-learned pred/ana_calib_total_us ratio bands before blend.")
    best.add_argument("--clip-q-low", type=float, default=0.0)
    best.add_argument("--clip-q-high", type=float, default=0.999)
    best.add_argument("--clip-min-rows", type=int, default=128)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run-single":
        payload = _run_single(Path(args.output_root), list(args.variants), npu_device_id=int(args.npu_device_id))
    else:
        payload = _run_bestfullfit(
            args.variant,
            Path(args.single_root),
            Path(args.output_root) / args.variant,
            clip_context_ratio=bool(args.clip_context_ratio),
            clip_q_low=float(args.clip_q_low),
            clip_q_high=float(args.clip_q_high),
            clip_min_rows=int(args.clip_min_rows),
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
