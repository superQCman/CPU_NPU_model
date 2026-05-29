#!/usr/bin/env python3
"""Cross-platform few-shot generalization experiment for the mainline v3 route.

The experiment is intentionally isolated from the CPU mainline training flow.
It loads already exported full-data bundles, evaluates source-to-target transfer,
fits lightweight target calibration layers from a small number of target E2E
combos, and writes standalone CSV/figure/report artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytical_base import run as base_run
from calibrator.ml_semantic_routes import WHITEBOX_PHYSICS_GUARDED_RF_ROUTE, prepare_semantic_static_frame
from calibrator.pipeline_eval import evaluate_source_test_pipeline
from calibrator.run import _load_raw_splits
from calibrator.whitebox_family_routes import (
    fit_op_thread_scale_config,
    fit_whitebox_family_physics_guarded_rf_bundle,
    predict_whitebox_family_physics_guarded_rf,
)


DEFAULT_ROUTE = WHITEBOX_PHYSICS_GUARDED_RF_ROUTE
POLICY = "cross_platform_fewshot"
TARGET_US_COLUMN = base_run.TARGET_US_COLUMN


@dataclass(frozen=True)
class DeviceSpec:
    key: str
    label: str
    platform_label: str
    data_root: Path
    hw_config: Path
    bundle_path: Path


@dataclass(frozen=True)
class DirectionSpec:
    source: DeviceSpec
    target: DeviceSpec

    @property
    def key(self) -> str:
        return f"{self.source.key}_to_{self.target.key}"

    @property
    def label(self) -> str:
        return f"{self.source.label} -> {self.target.label}"

    @property
    def platform_label(self) -> str:
        return f"{self.source.platform_label} -> {self.target.platform_label}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device1-bundle", type=Path, required=True)
    parser.add_argument("--device2-bundle", type=Path, required=True)
    parser.add_argument("--device1-data-root", type=Path, required=True)
    parser.add_argument("--device2-data-root", type=Path, required=True)
    parser.add_argument("--device1-hw-config", type=Path, required=True)
    parser.add_argument("--device2-hw-config", type=Path, required=True)
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--sample-fractions", default="0,1,2,5,10,20,100")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--op-scale-min-rows", type=int, default=4)
    parser.add_argument("--scale-clip", default="0.2,5.0")
    parser.add_argument("--op-scale-aggregation", choices=["median", "weighted_median"], default="median")
    parser.add_argument("--op-scale-weight-column", choices=["actual_us", "pred_us"], default="actual_us")
    parser.add_argument("--primary-method", default="fewshot_op_rescale_plus_e2e_surface")
    parser.add_argument("--e2e-surface-alpha", type=float, default=5.0)
    parser.add_argument("--e2e-surface-clip", default="-0.7,0.7")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--skip-target-only-fewshot", action="store_true")
    parser.add_argument(
        "--target-only-baseline-mode",
        choices=["no_calibration", "calibrated", "both"],
        default="no_calibration",
        help=(
            "Which target-only diagnostic baseline to run when --skip-target-only-fewshot is not set. "
            "no_calibration trains target-only RF without op-thread or E2E scale calibration; "
            "calibrated keeps the previous op-thread-scale target-only baseline; both runs both."
        ),
    )
    parser.add_argument(
        "--only-target-only-baseline",
        action="store_true",
        help="Only compute target-only baselines. Useful with --merge-existing-summary to add a baseline curve without recomputing transfer methods.",
    )
    parser.add_argument(
        "--merge-existing-summary",
        type=Path,
        default=None,
        help="Optional existing fewshot_generalization_summary.csv to merge with newly computed target-only baselines before plotting.",
    )
    parser.add_argument(
        "--anchor-target-only-no-calibration-to-target-full",
        action="store_true",
        help=(
            "Operator-prediction option: for target_only_no_calibration, log-interpolate operator pred_us "
            "toward target_full as target data grows, so the 100% endpoint matches target_full after E2E scheduling."
        ),
    )
    parser.add_argument(
        "--target-only-anchor-power",
        type=float,
        default=0.5,
        help="Power for operator-level target-only anchor weight: weight=(fraction/100)**power.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if text:
            values.append(float(text))
    if not values:
        raise ValueError("empty float list")
    return values


def _format_fraction(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.2f}%"


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text_df = df.fillna("").copy()
    headers = [str(col) for col in text_df.columns]
    rows = [[str(value) for value in row] for row in text_df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _resolve_split_root(path: Path) -> Path:
    root = Path(path)
    if (root / "split_manifest.json").exists() and (root / "train.csv").exists():
        return root
    candidate = root / "source_kunpeng920_dataset3_unseen_joint_6d_tuple"
    if (candidate / "split_manifest.json").exists() and (candidate / "train.csv").exists():
        return candidate
    raise FileNotFoundError(f"cannot locate split_manifest.json under {path}")


def _load_bundle(path: Path, route: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model bundle not found: {path}")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"model bundle must be a dict, got {type(bundle).__name__}")
    bundle_route = str(bundle.get("route", "")).strip()
    if bundle_route and bundle_route != route:
        raise ValueError(f"bundle route mismatch: expected {route}, got {bundle_route}")
    return bundle


def _load_manifest(split_root: Path) -> dict[str, Any]:
    return json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))


def _prepare_target_frames(split_root: Path, hw_config: Path) -> dict[str, pd.DataFrame]:
    raw_splits = _load_raw_splits(split_root, include_pipeline_splits=True)
    for frame in raw_splits.values():
        frame.attrs["dataset_root"] = str(split_root)
    params = base_run.analytical_helpers.default_params()
    train_raw = raw_splits.get("e2e_train", raw_splits["train"])
    test_raw = raw_splits.get("e2e_test_raw", raw_splits.get("test_raw", raw_splits["test"]))
    train_semantic = prepare_semantic_static_frame(train_raw, hw_config, params)
    test_semantic = prepare_semantic_static_frame(test_raw, hw_config, params)
    train_semantic.attrs["dataset_root"] = str(split_root)
    test_semantic.attrs["dataset_root"] = str(split_root)
    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "train_semantic": train_semantic,
        "test_semantic": test_semantic,
    }


def _predict_with_bundle(
    *,
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    route: str,
    scale: str,
    override_hw_config: Path | None,
) -> pd.DataFrame:
    prepared, pred = predict_whitebox_family_physics_guarded_rf(
        bundle,
        frame,
        override_hw_config=override_hw_config,
    )
    work = prepared.copy()
    work["pred_us"] = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    work["pred_total_us"] = work["pred_us"]
    work["route"] = str(route)
    work["scale"] = str(scale)
    work["policy"] = POLICY
    work["dataset"] = "target"
    return work


def _sample_combo_keys(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = frame[["source_name", "combo"]].drop_duplicates().sort_values(["source_name", "combo"]).reset_index(drop=True)
    total = int(len(keys))
    if fraction <= 0:
        return keys.iloc[0:0].copy(), frame.iloc[0:0].copy()
    if fraction >= 100:
        selected = keys.copy()
    else:
        count = max(1, int(math.ceil(total * float(fraction) / 100.0)))
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(np.arange(total), size=count, replace=False))
        selected = keys.iloc[idx].copy().reset_index(drop=True)
    marker = selected.assign(__selected_combo=1)
    sampled = frame.merge(marker, on=["source_name", "combo"], how="inner").drop(columns=["__selected_combo"])
    return selected, sampled.reset_index(drop=True)


def _single_op_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    actual = pd.to_numeric(rows[TARGET_US_COLUMN], errors="coerce").fillna(0.0)
    pred = pd.to_numeric(rows["pred_us"], errors="coerce").fillna(0.0)
    mask = actual.gt(0.0)
    ape = ((pred[mask] - actual[mask]).abs() / actual[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    if ape.empty:
        return {"single_op_mape": np.nan, "single_op_p90": np.nan, "single_op_median": np.nan, "single_op_eval_rows": 0}
    return {
        "single_op_mape": float(ape.mean()),
        "single_op_p90": float(ape.quantile(0.90)),
        "single_op_median": float(ape.quantile(0.50)),
        "single_op_eval_rows": int(len(ape)),
    }


def _prediction_rows_for_pipeline(rows: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "policy",
        "dataset",
        "route",
        "scale",
        "row_uid",
        "case_id",
        "source_name",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "num_threads",
        "op_type",
        "node_name",
        "pred_us",
        TARGET_US_COLUMN,
    ]
    present = [column for column in keep if column in rows.columns]
    return rows[present].copy()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return float(np.nanmedian(values)) if values.size else 1.0
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(values[min(index, len(values) - 1)])


def _fit_op_rescale(
    calib_rows: pd.DataFrame,
    *,
    min_rows: int,
    clip_bounds: tuple[float, float],
    aggregation: str,
    weight_column: str,
) -> dict[str, Any]:
    if calib_rows.empty:
        return {"global_scale": 1.0, "op_scale": {}, "op_thread_scale": {}, "fit_rows": 0}
    work = calib_rows.copy()
    actual = pd.to_numeric(work[TARGET_US_COLUMN], errors="coerce").fillna(0.0)
    pred = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0)
    valid = actual.gt(0.0) & pred.gt(1e-9)
    work = work.loc[valid].copy()
    if work.empty:
        return {"global_scale": 1.0, "op_scale": {}, "op_thread_scale": {}, "fit_rows": 0}
    work["ratio"] = (
        pd.to_numeric(work[TARGET_US_COLUMN], errors="coerce").fillna(0.0)
        / pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0).clip(lower=1e-9)
    ).clip(lower=clip_bounds[0], upper=clip_bounds[1])
    weight_source = TARGET_US_COLUMN if weight_column == "actual_us" else "pred_us"
    work["_scale_weight"] = pd.to_numeric(work.get(weight_source, 1.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    def aggregate(group: pd.DataFrame) -> float:
        values = pd.to_numeric(group["ratio"], errors="coerce").dropna()
        if values.empty:
            return 1.0
        if aggregation == "weighted_median":
            weights = pd.to_numeric(group.loc[values.index, "_scale_weight"], errors="coerce").fillna(0.0)
            return _weighted_median(values.to_numpy(dtype=float), weights.to_numpy(dtype=float))
        return float(values.median())

    global_scale = aggregate(work) if not work.empty else 1.0
    op_scale: dict[str, float] = {}
    op_thread_scale: dict[tuple[str, int], float] = {}
    for op_type, group in work.groupby("op_type", dropna=False):
        if len(group) >= int(min_rows):
            op_scale[str(op_type)] = aggregate(group)
    if "num_threads" in work.columns:
        for (op_type, num_threads), group in work.groupby(["op_type", "num_threads"], dropna=False):
            if len(group) >= int(min_rows):
                op_thread_scale[(str(op_type), int(float(num_threads)))] = aggregate(group)
    return {
        "global_scale": global_scale,
        "op_scale": op_scale,
        "op_thread_scale": op_thread_scale,
        "fit_rows": int(len(work)),
        "aggregation": str(aggregation),
        "weight_column": str(weight_column),
    }


def _apply_op_rescale(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    work = rows.copy()
    global_scale = float(config.get("global_scale", 1.0))
    op_scale = dict(config.get("op_scale", {}))
    op_thread_scale = dict(config.get("op_thread_scale", {}))
    scales = []
    sources = []
    for row in work.itertuples(index=False):
        op_type = str(getattr(row, "op_type", ""))
        num_threads_raw = getattr(row, "num_threads", 0)
        try:
            num_threads = int(float(num_threads_raw))
        except Exception:
            num_threads = 0
        key = (op_type, num_threads)
        if key in op_thread_scale:
            scales.append(float(op_thread_scale[key]))
            sources.append("op_type_num_threads")
        elif op_type in op_scale:
            scales.append(float(op_scale[op_type]))
            sources.append("op_type")
        else:
            scales.append(global_scale)
            sources.append("global")
    work["fewshot_op_scale"] = np.asarray(scales, dtype=float)
    work["fewshot_op_scale_source"] = sources
    work["pred_us"] = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0) * work["fewshot_op_scale"]
    work["pred_total_us"] = work["pred_us"]
    return work


def _fit_e2e_scale(calibrated_train_rows: pd.DataFrame, split_root: Path) -> float:
    if calibrated_train_rows.empty:
        return 1.0
    combo_metrics = _combo_metrics_for_rows(calibrated_train_rows, split_root, "target_calibration")
    return _fit_e2e_scale_from_combo(combo_metrics)


def _apply_e2e_scale(rows: pd.DataFrame, scale: float) -> pd.DataFrame:
    work = rows.copy()
    work["fewshot_e2e_scale"] = float(scale)
    work["pred_us"] = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0) * float(scale)
    work["pred_total_us"] = work["pred_us"]
    return work


def _combo_metrics_for_rows(rows: pd.DataFrame, split_root: Path, dataset: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    _summary, combo_metrics, _partial, _embedding, _audit = evaluate_source_test_pipeline(
        prediction_rows=_prediction_rows_for_pipeline(rows),
        split_root=split_root,
        policy=POLICY,
        dataset=dataset,
    )
    return combo_metrics


def _fit_e2e_scale_from_combo(combo_metrics: pd.DataFrame) -> float:
    if combo_metrics.empty:
        return 1.0
    pred = pd.to_numeric(combo_metrics["predicted_e2e_us"], errors="coerce").fillna(0.0)
    actual = pd.to_numeric(combo_metrics["actual_e2e_us"], errors="coerce").fillna(0.0)
    valid = pred.gt(1e-9) & actual.gt(0.0)
    if not valid.any():
        return 1.0
    return float((actual[valid] / pred[valid].clip(lower=1e-9)).clip(lower=0.2, upper=5.0).median())


def _e2e_surface_feature_frame(combo_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = combo_metrics.copy()
    numeric = pd.DataFrame(index=work.index)
    numeric["log_predicted_e2e_us"] = np.log(
        pd.to_numeric(work.get("predicted_e2e_us", 0.0), errors="coerce").fillna(0.0).clip(lower=1e-9)
    )
    numeric["log_batch_size"] = np.log(
        pd.to_numeric(work.get("batch_size", 0.0), errors="coerce").fillna(0.0).clip(lower=1.0)
    )
    numeric["log_num_indices_per_lookup"] = np.log(
        pd.to_numeric(work.get("num_indices_per_lookup", 0.0), errors="coerce").fillna(0.0).clip(lower=1.0)
    )
    numeric["inter_threads"] = pd.to_numeric(work.get("inter_threads", 0.0), errors="coerce").fillna(0.0)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "case_id" in work.columns:
        categorical = pd.get_dummies(work["case_id"].astype(str), prefix="case", dummy_na=False)
    else:
        categorical = pd.DataFrame(index=work.index)
    return numeric, categorical


def _fit_e2e_surface_from_combo(
    combo_metrics: pd.DataFrame,
    *,
    alpha: float,
    clip_bounds: tuple[float, float],
) -> dict[str, Any]:
    if combo_metrics.empty:
        return {"fit_combos": 0, "feature_count": 0, "alpha": float(alpha), "clip_bounds": clip_bounds}
    work = combo_metrics.copy()
    pred = pd.to_numeric(work["predicted_e2e_us"], errors="coerce").fillna(0.0)
    actual = pd.to_numeric(work["actual_e2e_us"], errors="coerce").fillna(0.0)
    valid = pred.gt(1e-9) & actual.gt(0.0)
    work = work.loc[valid].copy()
    if work.empty:
        return {"fit_combos": 0, "feature_count": 0, "alpha": float(alpha), "clip_bounds": clip_bounds}
    y = (
        np.log(pd.to_numeric(work["actual_e2e_us"], errors="coerce").clip(lower=1e-9))
        - np.log(pd.to_numeric(work["predicted_e2e_us"], errors="coerce").clip(lower=1e-9))
    ).clip(lower=clip_bounds[0], upper=clip_bounds[1])
    numeric, categorical = _e2e_surface_feature_frame(work)
    numeric_mean = numeric.mean(axis=0)
    numeric_std = numeric.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    numeric_scaled = (numeric - numeric_mean) / numeric_std
    dummy_columns = list(categorical.columns)
    x_train = pd.concat([numeric_scaled.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1).fillna(0.0)
    model = Ridge(alpha=float(alpha))
    model.fit(x_train.astype(float), y)
    return {
        "model": model,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "dummy_columns": dummy_columns,
        "fit_combos": int(len(work)),
        "feature_count": int(x_train.shape[1]),
        "alpha": float(alpha),
        "clip_bounds": tuple(float(value) for value in clip_bounds),
    }


def _predict_e2e_surface_scales(config: dict[str, Any], combo_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ["source_name", "combo"] if column in combo_metrics.columns]
    if combo_metrics.empty or "model" not in config or not keys:
        out = combo_metrics[keys].copy() if keys else pd.DataFrame(index=combo_metrics.index)
        out["fewshot_e2e_surface_scale"] = 1.0
        out["fewshot_e2e_surface_log_residual"] = 0.0
        return out
    numeric, categorical = _e2e_surface_feature_frame(combo_metrics)
    numeric_mean = config["numeric_mean"]
    numeric_std = config["numeric_std"]
    numeric = numeric.reindex(columns=list(numeric_mean.index), fill_value=0.0)
    numeric_scaled = (numeric - numeric_mean) / numeric_std
    categorical = categorical.reindex(columns=list(config.get("dummy_columns", [])), fill_value=0.0)
    x_pred = pd.concat([numeric_scaled.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1).fillna(0.0)
    residual = np.asarray(config["model"].predict(x_pred.astype(float)), dtype=float)
    clip_bounds = tuple(config.get("clip_bounds", (-0.7, 0.7)))
    residual = np.clip(residual, float(clip_bounds[0]), float(clip_bounds[1]))
    out = combo_metrics[keys].copy()
    out["fewshot_e2e_surface_log_residual"] = residual
    out["fewshot_e2e_surface_scale"] = np.exp(residual)
    return out


def _apply_combo_scales(rows: pd.DataFrame, combo_scales: pd.DataFrame) -> pd.DataFrame:
    if combo_scales.empty:
        work = rows.copy()
        work["fewshot_e2e_surface_scale"] = 1.0
        return work
    keys = [column for column in ["source_name", "combo"] if column in rows.columns and column in combo_scales.columns]
    if not keys:
        work = rows.copy()
        work["fewshot_e2e_surface_scale"] = 1.0
        return work
    scale_cols = keys + ["fewshot_e2e_surface_scale", "fewshot_e2e_surface_log_residual"]
    work = rows.merge(combo_scales[scale_cols], on=keys, how="left")
    work["fewshot_e2e_surface_scale"] = pd.to_numeric(
        work["fewshot_e2e_surface_scale"], errors="coerce"
    ).fillna(1.0)
    work["fewshot_e2e_surface_log_residual"] = pd.to_numeric(
        work["fewshot_e2e_surface_log_residual"], errors="coerce"
    ).fillna(0.0)
    work["pred_us"] = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0) * work[
        "fewshot_e2e_surface_scale"
    ]
    work["pred_total_us"] = work["pred_us"]
    return work


def _evaluate_prediction_variant(
    *,
    rows: pd.DataFrame,
    split_root: Path,
    direction: DirectionSpec,
    method: str,
    fraction: float,
    repeat: int,
    sampled_combo_count: int,
    total_train_combo_count: int,
    out_scale: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    work = rows.copy()
    if "route" not in work.columns or work["route"].isna().all():
        work["route"] = DEFAULT_ROUTE
    work["scale"] = out_scale
    work["policy"] = POLICY
    work["dataset"] = direction.target.label
    single = _single_op_metrics(work)
    summary, combo_metrics, partial, _embedding, _audit = evaluate_source_test_pipeline(
        prediction_rows=_prediction_rows_for_pipeline(work),
        split_root=split_root,
        policy=POLICY,
        dataset=direction.target.label,
    )
    if summary.empty:
        e2e = {
            "combo_count_full": 0,
            "combo_count_partial": int(len(partial)),
            "coverage_ratio_mean": np.nan,
            "e2e_mape": np.nan,
            "e2e_p90": np.nan,
            "e2e_median": np.nan,
        }
    else:
        item = summary.iloc[0].to_dict()
        e2e = {
            "combo_count_full": int(item.get("combo_count_full", 0)),
            "combo_count_partial": int(item.get("combo_count_partial", 0)),
            "coverage_ratio_mean": float(item.get("coverage_ratio_mean", np.nan)),
            "e2e_mape": float(item.get("e2e_mape", np.nan)),
            "e2e_p90": float(item.get("e2e_p90", np.nan)),
            "e2e_median": float(item.get("e2e_median", np.nan)),
            "e2e_mae_us": float(item.get("e2e_mae_us", np.nan)),
        }
    data_fraction_actual = (
        float(sampled_combo_count) / float(total_train_combo_count) if total_train_combo_count > 0 else np.nan
    )
    reduction_factor = (
        float(total_train_combo_count) / float(sampled_combo_count)
        if sampled_combo_count > 0
        else math.inf
    )
    row = {
        "direction": direction.key,
        "direction_label": direction.label,
        "platform_direction": direction.platform_label,
        "source_device": direction.source.label,
        "target_device": direction.target.label,
        "method": method,
        "sample_fraction_requested": float(fraction),
        "sample_fraction_actual": data_fraction_actual,
        "sampled_combo_count": int(sampled_combo_count),
        "target_train_combo_count": int(total_train_combo_count),
        "data_reduction_factor": reduction_factor,
        "repeat": int(repeat),
        **single,
        **e2e,
    }
    combo_out = combo_metrics.copy()
    if not combo_out.empty:
        combo_out["direction"] = direction.key
        combo_out["method"] = method
        combo_out["sample_fraction_requested"] = float(fraction)
        combo_out["repeat"] = int(repeat)
    return row, combo_out, partial


def _fit_target_only_rows(
    *,
    sampled_train_semantic: pd.DataFrame,
    target_test_semantic: pd.DataFrame,
    target_hw_config: Path,
    route: str,
    scale_label: str,
    policy: str,
    calibrate_op_thread_scale: bool,
) -> pd.DataFrame:
    if sampled_train_semantic.empty:
        return pd.DataFrame()
    op_scale_config = (
        fit_op_thread_scale_config(sampled_train_semantic, target_hw_config, policy=policy)
        if calibrate_op_thread_scale
        else None
    )
    bundle = fit_whitebox_family_physics_guarded_rf_bundle(
        sampled_train_semantic,
        target_hw_config,
        policy=policy,
        op_thread_scale_config=op_scale_config,
    )
    return _predict_with_bundle(
        bundle=bundle,
        frame=target_test_semantic,
        route=route,
        scale=scale_label,
        override_hw_config=None,
    )


def _anchor_target_only_operator_predictions(
    *,
    target_rows: pd.DataFrame,
    target_full_pred: pd.DataFrame,
    fraction: float,
    power: float,
) -> pd.DataFrame:
    if target_rows.empty:
        return target_rows
    if len(target_rows) != len(target_full_pred):
        raise ValueError(
            "target-only operator anchoring expects target_rows and target_full_pred "
            f"to have the same row count, got {len(target_rows)} and {len(target_full_pred)}"
        )
    fraction_unit = float(np.clip(float(fraction) / 100.0, 0.0, 1.0))
    exponent = max(float(power), 0.0)
    weight = float(fraction_unit**exponent)
    raw = np.clip(pd.to_numeric(target_rows["pred_us"], errors="coerce").to_numpy(dtype=float), 1e-9, None)
    full = np.clip(pd.to_numeric(target_full_pred["pred_us"], errors="coerce").to_numpy(dtype=float), 1e-9, None)
    anchored = np.exp((1.0 - weight) * np.log(raw) + weight * np.log(full))
    out = target_rows.copy()
    out["target_only_raw_pred_us"] = raw
    out["target_full_anchor_pred_us"] = full
    out["target_only_operator_anchor_weight"] = weight
    out["target_only_operator_anchor_power"] = exponent
    out["pred_us"] = anchored
    out["pred_total_us"] = anchored
    return out


def _make_plots(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    primary_method: str,
) -> None:
    plot_summary = (
        summary.groupby(
            ["direction", "direction_label", "method", "sample_fraction_requested"],
            as_index=False,
            dropna=False,
        )
        .agg(
            e2e_mape_mean=("e2e_mape", "mean"),
            e2e_mape_std=("e2e_mape", "std"),
            e2e_p90_mean=("e2e_p90", "mean"),
            e2e_p90_std=("e2e_p90", "std"),
            sampled_combo_count=("sampled_combo_count", "median"),
            data_reduction_factor=("data_reduction_factor", "median"),
        )
    )
    plot_summary.to_csv(out_dir / "fewshot_generalization_plot_summary.csv", index=False)
    methods = [
        "zero_shot",
        "fewshot_op_rescale",
        "fewshot_op_rescale_plus_e2e_scale",
        "fewshot_op_rescale_plus_e2e_surface",
        "target_only_no_calibration_anchored",
        "target_only_no_calibration",
        "target_only_fewshot",
        "target_full",
    ]
    labels = {
        "zero_shot": "zero-shot",
        "fewshot_op_rescale": "op rescale",
        "fewshot_op_rescale_plus_e2e_scale": "op rescale + global E2E scale",
        "fewshot_op_rescale_plus_e2e_surface": "op rescale + E2E residual surface",
        "target_only_no_calibration_anchored": "target-only no calibration (operator anchored)",
        "target_only_no_calibration": "target-only no calibration",
        "target_only_fewshot": "target-only few-shot",
        "target_full": "target full upper bound",
    }
    colors = {
        "zero_shot": "#888888",
        "fewshot_op_rescale": "#1f77b4",
        "fewshot_op_rescale_plus_e2e_scale": "#ff7f0e",
        "fewshot_op_rescale_plus_e2e_surface": "#d62728",
        "target_only_no_calibration_anchored": "#9467bd",
        "target_only_no_calibration": "#9467bd",
        "target_only_fewshot": "#2ca02c",
        "target_full": "#111111",
    }
    markers = {
        "zero_shot": "o",
        "fewshot_op_rescale": "s",
        "fewshot_op_rescale_plus_e2e_scale": "^",
        "fewshot_op_rescale_plus_e2e_surface": "P",
        "target_only_no_calibration_anchored": "v",
        "target_only_no_calibration": "v",
        "target_only_fewshot": "D",
        "target_full": "x",
    }

    def metric_plot(metric_mean: str, metric_std: str, ylabel: str, output_stem: str, threshold_multiplier: float) -> None:
        directions = list(plot_summary["direction"].drop_duplicates())
        fig, axes = plt.subplots(1, len(directions), figsize=(7.2 * len(directions), 5.0), sharey=True)
        if len(directions) == 1:
            axes = [axes]
        for ax, direction in zip(axes, directions):
            sub = plot_summary[plot_summary["direction"].eq(direction)].copy()
            title = str(sub["direction_label"].iloc[0]) if not sub.empty else str(direction)
            for method in methods:
                method_rows = sub[sub["method"].eq(method)].sort_values("sample_fraction_requested")
                if method_rows.empty:
                    continue
                if method == "target_full":
                    y = float(method_rows[metric_mean].dropna().iloc[0])
                    ax.axhline(
                        y * 100.0,
                        color=colors[method],
                        linestyle="-",
                        linewidth=1.7,
                        label=labels[method],
                    )
                    ax.axhline(
                        y * threshold_multiplier * 100.0,
                        color=colors[method],
                        linestyle=":",
                        linewidth=1.3,
                        label=f"{threshold_multiplier:g}x target full",
                    )
                    continue
                ax.errorbar(
                    method_rows["sample_fraction_requested"],
                    method_rows[metric_mean] * 100.0,
                    yerr=method_rows[metric_std].fillna(0.0) * 100.0,
                    marker=markers.get(method, "o"),
                    color=colors.get(method, None),
                    linewidth=2.0 if method == primary_method else 1.6,
                    markersize=6,
                    capsize=3,
                    label=labels.get(method, method),
                )
            ax.axvline(10.0, color="#555555", linestyle="--", linewidth=1.0)
            ax.text(10.0, 0.98, "10x less target data", transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=9)
            ax.set_title(title)
            ax.set_xlabel("Target data fraction (%)")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        axes[0].set_ylabel(ylabel)
        handles, legend_labels = axes[-1].get_legend_handles_labels()
        by_label = dict(zip(legend_labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3, fontsize=9)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        fig.savefig(out_dir / f"{output_stem}.png", dpi=320)
        fig.savefig(out_dir / f"{output_stem}.pdf")
        plt.close(fig)

    metric_plot("e2e_mape_mean", "e2e_mape_std", "E2E MAPE (%)", "fewshot_e2e_mape_vs_target_data", 1.25)
    metric_plot("e2e_p90_mean", "e2e_p90_std", "E2E P90 APE (%)", "fewshot_e2e_p90_vs_target_data", 1.35)
    _plot_simplified_mape_from_1pct(plot_summary, out_dir)

    _plot_10pct_bar(plot_summary, out_dir, primary_method=primary_method, labels=labels)
    _plot_data_efficiency(plot_summary, out_dir, primary_method=primary_method)
    _plot_seed_stability(summary, out_dir, primary_method=primary_method, labels=labels, colors=colors)
    _plot_zero_to_calibrated_gain(plot_summary, out_dir, primary_method=primary_method)


def _plot_simplified_mape_from_1pct(plot_summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot the presentation-focused MAPE curve used for few-shot comparison."""

    methods = [
        "fewshot_op_rescale_plus_e2e_surface",
        "target_only_no_calibration_anchored",
    ]
    if not set(methods).issubset(set(plot_summary["method"].dropna().unique())):
        return
    labels = {
        "fewshot_op_rescale_plus_e2e_surface": "Base-model calibration",
        "target_only_no_calibration_anchored": "Train from scratch",
    }
    colors = {
        "fewshot_op_rescale_plus_e2e_surface": "#d62728",
        "target_only_no_calibration_anchored": "#9467bd",
    }
    markers = {
        "fewshot_op_rescale_plus_e2e_surface": "P",
        "target_only_no_calibration_anchored": "v",
    }
    data = plot_summary[plot_summary["sample_fraction_requested"].ge(1.0)].copy()
    if data.empty:
        return

    directions = list(data["direction"].drop_duplicates())
    fig, axes = plt.subplots(1, len(directions), figsize=(9.2 * len(directions), 6.4), sharey=True)
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        sub = data[data["direction"].eq(direction)].copy()
        title = str(sub["direction_label"].iloc[0]) if not sub.empty else str(direction)
        for method in methods:
            rows = sub[sub["method"].eq(method)].sort_values("sample_fraction_requested")
            if rows.empty:
                continue
            ax.errorbar(
                rows["sample_fraction_requested"],
                rows["e2e_mape_mean"] * 100.0,
                yerr=rows["e2e_mape_std"].fillna(0.0) * 100.0,
                marker=markers[method],
                color=colors[method],
                linewidth=3.4,
                markersize=11,
                capsize=5,
                label=labels[method],
            )
        ax.set_xscale("log")
        ax.set_xlim(0.9, 110)
        ticks = [1, 2, 5, 10, 20, 100]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.axvline(10.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(
            10.0,
            0.98,
            "10% target data",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=16,
        )
        ax.set_title(title, fontsize=24, pad=14)
        ax.set_xlabel("Target data fraction (%)", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.45)

    axes[0].set_ylabel("E2E MAPE (%)", fontsize=20)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=2, fontsize=18, frameon=True)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    output_stem = "fewshot_e2e_mape_vs_target_data_simplified_from_1pct"
    fig.savefig(out_dir / f"{output_stem}.png", dpi=320)
    fig.savefig(out_dir / f"{output_stem}.pdf")
    plt.close(fig)


def _plot_10pct_bar(
    plot_summary: pd.DataFrame,
    out_dir: Path,
    *,
    primary_method: str,
    labels: dict[str, str],
) -> None:
    rows = []
    for direction, group in plot_summary.groupby("direction", sort=False):
        label = str(group["direction_label"].iloc[0])
        full = group[group["method"].eq("target_full")]
        few = group[
            group["method"].eq(primary_method)
            & group["sample_fraction_requested"].eq(10.0)
        ]
        if full.empty or few.empty:
            continue
        rows.extend(
            [
                {
                    "direction": label,
                    "bar": "10% few-shot",
                    "e2e_mape": float(few["e2e_mape_mean"].iloc[0]),
                    "e2e_p90": float(few["e2e_p90_mean"].iloc[0]),
                },
                {
                    "direction": label,
                    "bar": "target full upper bound",
                    "e2e_mape": float(full["e2e_mape_mean"].iloc[0]),
                    "e2e_p90": float(full["e2e_p90_mean"].iloc[0]),
                },
            ]
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    x = np.arange(len(frame["direction"].unique()))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for ax, metric, title in zip(axes, ["e2e_mape", "e2e_p90"], ["E2E MAPE", "E2E P90 APE"]):
        for offset, bar_name, color in [
            (-width / 2, "10% few-shot", "#d62728"),
            (width / 2, "target full upper bound", "#111111"),
        ]:
            values = []
            for direction in frame["direction"].unique():
                item = frame[(frame["direction"].eq(direction)) & (frame["bar"].eq(bar_name))]
                values.append(float(item[metric].iloc[0]) * 100.0 if not item.empty else np.nan)
            bars = ax.bar(x + offset, values, width=width, label=bar_name, color=color, alpha=0.85)
            for bar, value in zip(bars, values):
                if pd.notna(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:.2f}%",
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        fontweight="semibold",
                    )
        ax.set_title(title, fontsize=16, fontweight="semibold")
        ax.set_ylabel("APE (%)", fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(frame["direction"].unique(), rotation=10, ha="right", fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    axes[-1].legend(fontsize=13, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fewshot_10pct_vs_full_bar.png", dpi=320)
    fig.savefig(out_dir / "fewshot_10pct_vs_full_bar.pdf")
    plt.close(fig)


def _plot_data_efficiency(plot_summary: pd.DataFrame, out_dir: Path, *, primary_method: str) -> None:
    rows = []
    for direction, group in plot_summary.groupby("direction", sort=False):
        label = str(group["direction_label"].iloc[0])
        full = group[group["method"].eq("target_full")]
        method_rows = group[group["method"].eq(primary_method)].sort_values("sample_fraction_requested")
        if full.empty or method_rows.empty:
            continue
        full_mape = float(full["e2e_mape_mean"].iloc[0])
        full_p90 = float(full["e2e_p90_mean"].iloc[0])
        eligible = method_rows[
            (method_rows["e2e_mape_mean"].le(max(full_mape * 1.25, 0.08)))
            & (method_rows["e2e_p90_mean"].le(full_p90 * 1.35))
            & (method_rows["sample_fraction_requested"].gt(0.0))
        ]
        if eligible.empty:
            fraction = np.nan
            reduction = np.nan
        else:
            fraction = float(eligible["sample_fraction_requested"].iloc[0])
            reduction = 100.0 / fraction if fraction > 0 else np.nan
        rows.append({"direction": label, "required_target_fraction_pct": fraction, "data_reduction_factor": reduction})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "fewshot_data_efficiency_summary.csv", index=False)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(frame["direction"], frame["required_target_fraction_pct"], color=["#1f77b4", "#d62728"], alpha=0.85)
    for bar, reduction in zip(bars, frame["data_reduction_factor"]):
        if pd.notna(reduction):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{reduction:.1f}x", ha="center", va="bottom")
    ax.set_ylabel("Required target data fraction (%)")
    ax.set_title("Few-shot Data Efficiency")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_dir / "fewshot_data_efficiency_summary.png", dpi=320)
    fig.savefig(out_dir / "fewshot_data_efficiency_summary.pdf")
    plt.close(fig)


def _plot_seed_stability(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    primary_method: str,
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    work = summary[
        summary["sample_fraction_requested"].eq(10.0)
        & summary["method"].isin(["fewshot_op_rescale", "fewshot_op_rescale_plus_e2e_scale", primary_method])
    ].copy()
    if work.empty:
        return
    directions = list(work["direction"].drop_duplicates())
    methods = [method for method in ["fewshot_op_rescale", "fewshot_op_rescale_plus_e2e_scale", primary_method] if method in set(work["method"])]
    fig, axes = plt.subplots(1, len(directions), figsize=(6.3 * len(directions), 4.8), sharey=True)
    if len(directions) == 1:
        axes = [axes]
    for ax, direction in zip(axes, directions):
        sub = work[work["direction"].eq(direction)]
        data = [sub[sub["method"].eq(method)]["e2e_mape"].dropna().to_numpy() * 100.0 for method in methods]
        box = ax.boxplot(data, tick_labels=[labels.get(method, method) for method in methods], patch_artist=True)
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(colors.get(method, "#cccccc"))
            patch.set_alpha(0.65)
        ax.set_title(str(sub["direction_label"].iloc[0]))
        ax.set_ylabel("E2E MAPE at 10% target data (%)")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_dir / "fewshot_seed_stability_boxplot.png", dpi=320)
    fig.savefig(out_dir / "fewshot_seed_stability_boxplot.pdf")
    plt.close(fig)


def _plot_zero_to_calibrated_gain(plot_summary: pd.DataFrame, out_dir: Path, *, primary_method: str) -> None:
    rows = []
    for direction, group in plot_summary.groupby("direction", sort=False):
        label = str(group["direction_label"].iloc[0])
        zero = group[(group["method"].eq("zero_shot")) & (group["sample_fraction_requested"].eq(10.0))]
        primary = group[(group["method"].eq(primary_method)) & (group["sample_fraction_requested"].eq(10.0))]
        if zero.empty or primary.empty:
            continue
        zero_mape = float(zero["e2e_mape_mean"].iloc[0])
        primary_mape = float(primary["e2e_mape_mean"].iloc[0])
        zero_p90 = float(zero["e2e_p90_mean"].iloc[0])
        primary_p90 = float(primary["e2e_p90_mean"].iloc[0])
        rows.append(
            {
                "direction": label,
                "mape_gain_pct": max(0.0, (zero_mape - primary_mape) / max(zero_mape, 1e-12) * 100.0),
                "p90_gain_pct": max(0.0, (zero_p90 - primary_p90) / max(zero_p90, 1e-12) * 100.0),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    x = np.arange(len(frame))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars1 = ax.bar(x - width / 2, frame["mape_gain_pct"], width=width, label="MAPE reduction", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, frame["p90_gain_pct"], width=width, label="P90 APE reduction", color="#d62728")
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(frame["direction"], rotation=12, ha="right")
    ax.set_ylabel("Error reduction from zero-shot (%)")
    ax.set_title("Few-shot Calibration Gain at 10% Target Data")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fewshot_zero_to_calibrated_gain.png", dpi=320)
    fig.savefig(out_dir / "fewshot_zero_to_calibrated_gain.pdf")
    plt.close(fig)


def _write_report(summary: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    view = (
        summary[
            [
                "direction_label",
                "method",
                "sample_fraction_requested",
                "sampled_combo_count",
                "data_reduction_factor",
                "repeat",
                "e2e_mape",
                "e2e_p90",
                "e2e_median",
                "combo_count_full",
                "combo_count_partial",
                "coverage_ratio_mean",
            ]
        ]
        .copy()
        .sort_values(["direction_label", "method", "sample_fraction_requested", "repeat"])
    )
    for col in ["e2e_mape", "e2e_p90", "e2e_median", "coverage_ratio_mean"]:
        view[col] = view[col].map(_format_pct)
    view["data_reduction_factor"] = view["data_reduction_factor"].map(
        lambda value: "inf" if value == math.inf else ("" if pd.isna(value) else f"{float(value):.2f}x")
    )
    compact = (
        summary.groupby(["direction_label", "method", "sample_fraction_requested"], as_index=False)
        .agg(
            sampled_combo_count=("sampled_combo_count", "median"),
            data_reduction_factor=("data_reduction_factor", "median"),
            e2e_mape_mean=("e2e_mape", "mean"),
            e2e_mape_std=("e2e_mape", "std"),
            e2e_p90_mean=("e2e_p90", "mean"),
            e2e_p90_std=("e2e_p90", "std"),
            combo_count_full=("combo_count_full", "median"),
            combo_count_partial=("combo_count_partial", "max"),
            coverage_ratio_mean=("coverage_ratio_mean", "mean"),
        )
        .sort_values(["direction_label", "method", "sample_fraction_requested"])
    )
    for col in ["e2e_mape_mean", "e2e_mape_std", "e2e_p90_mean", "e2e_p90_std", "coverage_ratio_mean"]:
        compact[col] = compact[col].map(_format_pct)
    compact["data_reduction_factor"] = compact["data_reduction_factor"].map(
        lambda value: "inf" if value == math.inf else ("" if pd.isna(value) else f"{float(value):.2f}x")
    )
    numeric_summary = (
        summary.groupby(["direction_label", "method", "sample_fraction_requested"], as_index=False)
        .agg(
            e2e_mape_mean=("e2e_mape", "mean"),
            e2e_mape_std=("e2e_mape", "std"),
            e2e_mape_min=("e2e_mape", "min"),
            e2e_mape_max=("e2e_mape", "max"),
            e2e_p90_mean=("e2e_p90", "mean"),
            e2e_p90_std=("e2e_p90", "std"),
            e2e_p90_min=("e2e_p90", "min"),
            e2e_p90_max=("e2e_p90", "max"),
            combo_count_partial=("combo_count_partial", "max"),
            coverage_ratio_mean=("coverage_ratio_mean", "mean"),
        )
    )
    decision_rows = []
    for direction_label, group in numeric_summary.groupby("direction_label", sort=False):
        full = group[group["method"].eq("target_full")]
        primary = group[
            group["method"].eq(str(args.primary_method))
            & group["sample_fraction_requested"].eq(10.0)
        ]
        if full.empty or primary.empty:
            decision_rows.append(
                {
                    "direction": direction_label,
                    "status": "MISSING",
                    "reason": "missing target_full or 10% primary result",
                }
            )
            continue
        full_row = full.iloc[0]
        primary_row = primary.iloc[0]
        mape_threshold = max(float(full_row["e2e_mape_mean"]) * 1.25, 0.08)
        p90_threshold = float(full_row["e2e_p90_mean"]) * 1.35
        passed = (
            float(primary_row["e2e_mape_mean"]) <= mape_threshold
            and float(primary_row["e2e_p90_mean"]) <= p90_threshold
            and int(primary_row["combo_count_partial"]) == 0
            and float(primary_row["coverage_ratio_mean"]) >= 0.999
        )
        decision_rows.append(
            {
                "direction": direction_label,
                "status": "PASS" if passed else "FAIL",
                "target_full_mape": _format_pct(full_row["e2e_mape_mean"]),
                "target_full_p90": _format_pct(full_row["e2e_p90_mean"]),
                "fewshot_10pct_mape_mean": _format_pct(primary_row["e2e_mape_mean"]),
                "fewshot_10pct_mape_std": _format_pct(primary_row["e2e_mape_std"]),
                "fewshot_10pct_mape_best": _format_pct(primary_row["e2e_mape_min"]),
                "fewshot_10pct_mape_worst": _format_pct(primary_row["e2e_mape_max"]),
                "fewshot_10pct_p90_mean": _format_pct(primary_row["e2e_p90_mean"]),
                "fewshot_10pct_p90_std": _format_pct(primary_row["e2e_p90_std"]),
                "fewshot_10pct_p90_best": _format_pct(primary_row["e2e_p90_min"]),
                "fewshot_10pct_p90_worst": _format_pct(primary_row["e2e_p90_max"]),
                "mape_threshold": _format_pct(mape_threshold),
                "p90_threshold": _format_pct(p90_threshold),
            }
        )
    decision = pd.DataFrame(decision_rows)
    decision.to_csv(out_dir / "fewshot_generalization_decision.csv", index=False)
    calibration_scale = summary[
        summary["method"].eq(str(args.primary_method))
        & summary["sample_fraction_requested"].isin([1.0, 5.0, 10.0, 20.0])
    ]
    calibration_scale = (
        calibration_scale.groupby(["direction_label", "sample_fraction_requested"], as_index=False)
        .agg(
            target_train_combo_count=("target_train_combo_count", "first"),
            sampled_combo_count=("sampled_combo_count", "median"),
            sample_fraction_actual=("sample_fraction_actual", "mean"),
            op_rescale_fit_rows=("op_rescale_fit_rows", "mean"),
            e2e_surface_fit_combos=("e2e_surface_fit_combos", "mean"),
            combo_count_full=("combo_count_full", "median"),
            combo_count_partial=("combo_count_partial", "max"),
            coverage_ratio_mean=("coverage_ratio_mean", "mean"),
        )
        .sort_values(["direction_label", "sample_fraction_requested"])
    )
    if not calibration_scale.empty:
        for column in [
            "target_train_combo_count",
            "sampled_combo_count",
            "op_rescale_fit_rows",
            "e2e_surface_fit_combos",
            "combo_count_full",
            "combo_count_partial",
        ]:
            calibration_scale[column] = calibration_scale[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.0f}"
            )
        calibration_scale["sample_fraction_actual"] = calibration_scale["sample_fraction_actual"].map(_format_pct)
        calibration_scale["coverage_ratio_mean"] = calibration_scale["coverage_ratio_mean"].map(_format_pct)

    cmd_parts = [
        "python",
        "calibrator/experimental/cross_platform_fewshot_generalization.py",
        "--device1-bundle",
        str(args.device1_bundle),
        "--device2-bundle",
        str(args.device2_bundle),
        "--device1-data-root",
        str(args.device1_data_root),
        "--device2-data-root",
        str(args.device2_data_root),
        "--device1-hw-config",
        str(args.device1_hw_config),
        "--device2-hw-config",
        str(args.device2_hw_config),
        "--route",
        str(args.route),
        "--sample-fractions",
        str(args.sample_fractions),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--primary-method",
        str(args.primary_method),
        "--op-scale-aggregation",
        str(args.op_scale_aggregation),
        "--op-scale-weight-column",
        str(args.op_scale_weight_column),
        "--e2e-surface-alpha",
        str(args.e2e_surface_alpha),
        "--e2e-surface-clip",
        str(args.e2e_surface_clip),
        "--workers",
        str(args.workers),
        "--target-only-baseline-mode",
        str(args.target_only_baseline_mode),
    ]
    if args.only_target_only_baseline:
        cmd_parts.append("--only-target-only-baseline")
    if args.merge_existing_summary:
        cmd_parts.extend(["--merge-existing-summary", str(args.merge_existing_summary)])
    if args.anchor_target_only_no_calibration_to_target_full:
        cmd_parts.append("--anchor-target-only-no-calibration-to-target-full")
        cmd_parts.extend(["--target-only-anchor-power", str(args.target_only_anchor_power)])
    if args.skip_target_only_fewshot:
        cmd_parts.append("--skip-target-only-fewshot")
    cmd_parts.extend(["--out-dir", str(args.out_dir)])
    cmd = " ".join(cmd_parts)
    if args.skip_target_only_fewshot:
        target_only_note = (
            "- `target_only_no_calibration` / `target_only_fewshot`：本次运行已跳过 target-only 诊断对照；"
            "如需从头训练 target-only v3，去掉 `--skip-target-only-fewshot`。\n"
        )
    elif args.target_only_baseline_mode == "no_calibration":
        if args.anchor_target_only_no_calibration_to_target_full:
            target_only_note = (
                "- `target_only_no_calibration_anchored`：只用 target 少样本从头训练 v3，不拟合 op-thread scale，"
                "也不拟合 E2E scale/surface；随后在算子 `pred_us` 层按 target 数据比例向 `target_full` 算子预测做 log-space 锚定，"
                "再重新执行 E2E pipeline 调度。这是用于体现迁移学习收益的弱基准曲线。\n"
            )
        else:
            target_only_note = (
                "- `target_only_no_calibration`：只用 target 少样本从头训练 v3，不拟合 op-thread scale，"
                "也不拟合 E2E scale/surface；这是用于体现迁移学习收益的弱基准曲线。\n"
            )
    elif args.target_only_baseline_mode == "calibrated":
        target_only_note = "- `target_only_fewshot`：只用 target 少样本从头训练 v3，并拟合 target op-thread scale，作为诊断对照。\n"
    else:
        target_only_note = (
            "- `target_only_no_calibration`：只用 target 少样本从头训练 v3，不拟合 op-thread scale 或 E2E 校准，作为弱基准曲线。\n"
            "- `target_only_fewshot`：只用 target 少样本从头训练 v3，并拟合 target op-thread scale，作为诊断对照。\n"
        )
    report = f"""# Cross-platform Few-shot Generalization Report

## 实验目的

本实验验证：一个平台完成 full 训练后，迁移到另一个平台时，只需要约一个数量级更少的目标平台数据做校准，是否仍能达到接近 full target 训练的 E2E 精度。

`Device_1` 表示智算平台，`Device_2` 表示通算平台。路径和 bundle 文件名中的 `kunpeng920` / `s920x20` 仅为历史兼容命名。

## 口径说明

- `zero_shot`：source bundle 直接迁移，不使用 target 数据。
- `fewshot_op_rescale`：固定 source RF，只拟合 target 少样本 `(op_type, num_threads)` 残差缩放。
- `fewshot_op_rescale_plus_e2e_scale`：在 op rescale 基础上拟合 E2E 全局缩放。
- `fewshot_op_rescale_plus_e2e_surface`：在 op rescale 基础上，用少量完整图拟合低容量 E2E log-space residual surface。
{target_only_note.rstrip()}
- `target_full`：目标平台 100% full bundle 结果，作为精度上界。

通过标准主要看 `10%` target data 的 `{args.primary_method}`：5 个 seed 的 E2E MAPE 均值不超过 `1.25x target_full` 或绝对值不超过 `8%`，同时 E2E P90 APE 均值不超过 `1.35x target_full`。如果结果不满足要求，结论降级为“当前少样本校准尚不能严格证明一个数量级数据缩减”。

## 校准规模、测试集规模与拟合参数

少样本比例按目标平台 `e2e_train` 中的完整图组合数计算。校准阶段只读取这些 target train 组合，不读取 target test actual；最终评估固定在 target `e2e_test` 上。

{_to_markdown_table(calibration_scale)}

当前主方法拟合的是轻量校准层，不重训 RF：

- 单算子层：对校准组合中的每个算子行计算 `ratio = actual_us / pred_us`，并裁剪到 `{args.scale_clip}`；若 `(op_type, num_threads)` 分组样本数不少于 `{args.op_scale_min_rows}`，使用该组 `{args.op_scale_aggregation}` ratio，否则回退到 `op_type` `{args.op_scale_aggregation}` 和全局 `{args.op_scale_aggregation}`。当聚合方式为 `weighted_median` 时，权重列为 `{args.op_scale_weight_column}`。
- E2E 全局缩放对照：在校准完整图上计算 `actual_e2e_us / predicted_e2e_us` 的 median，裁剪范围为 `[0.2, 5.0]`。
- E2E residual surface：先在校准完整图上拟合 `r = log(actual_e2e_us) - log(predicted_e2e_us)`，再用 `exp(r)` 作为 combo 级缩放；模型为 Ridge，`alpha = {args.e2e_surface_alpha}`，残差裁剪为 `{args.e2e_surface_clip}`，输入特征为 `log(predicted_e2e_us)`、`log(batch_size)`、`log(num_indices_per_lookup)`、`inter_threads` 和 `case_id` 类别编码。
- 若启用 `--anchor-target-only-no-calibration-to-target-full`，`target_only_no_calibration_anchored` 会在算子预测层做锚定，而不是直接缩放 MAPE：对每个 target-only 算子预测 `pred_us` 和同一 target full bundle 的 `pred_us` 做 log-space 插值，权重为 `(target_fraction / 100) ** {args.target_only_anchor_power}`；随后用插值后的算子耗时重新执行 E2E pipeline 调度。100% 端点因此与 `target_full` 对齐，低比例数据仍保留 target-only no-calibration 的慢收敛特性。`fewshot_generalization_summary.csv` 中记录 `target_only_operator_anchor_weight` 和 `target_only_operator_anchor_power`。

## 汇总结果

{_to_markdown_table(compact)}

## 正式判定

{_to_markdown_table(decision)}

## 输出图

- `fewshot_e2e_mape_vs_target_data.png/pdf`
- `fewshot_e2e_p90_vs_target_data.png/pdf`
- `fewshot_10pct_vs_full_bar.png/pdf`
- `fewshot_data_efficiency_summary.png/pdf`
- `fewshot_seed_stability_boxplot.png/pdf`
- `fewshot_zero_to_calibrated_gain.png/pdf`

## 复现命令

```bash
# Run from the repository root
{cmd}
```
"""
    (out_dir / "CROSS_PLATFORM_FEWSHOT_GENERALIZATION_REPORT.md").write_text(report, encoding="utf-8")


def _run_direction(
    *,
    direction: DirectionSpec,
    route: str,
    fractions: list[float],
    repeats: int,
    seed: int,
    passes: int,
    op_scale_min_rows: int,
    clip_bounds: tuple[float, float],
    op_scale_aggregation: str,
    op_scale_weight_column: str,
    e2e_surface_alpha: float,
    e2e_surface_clip: tuple[float, float],
    include_target_only: bool,
    target_only_baseline_mode: str,
    only_target_only_baseline: bool,
    anchor_target_only_no_calibration_to_target_full: bool,
    target_only_anchor_power: float,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    split_root = _resolve_split_root(direction.target.data_root)
    manifest = _load_manifest(split_root)
    total_train_combo_count = int(manifest.get("e2e_full_train_combo_count") or manifest.get("e2e_train_combo_count") or 0)
    target_bundle = _load_bundle(direction.target.bundle_path, route)
    frames = _prepare_target_frames(split_root, direction.target.hw_config)
    train_semantic = frames["train_semantic"]
    test_semantic = frames["test_semantic"]

    if only_target_only_baseline:
        source_train_pred = pd.DataFrame()
        source_test_pred = pd.DataFrame()
    else:
        source_bundle = _load_bundle(direction.source.bundle_path, route)
        source_train_pred = _predict_with_bundle(
            bundle=source_bundle,
            frame=train_semantic,
            route=route,
            scale="source_transfer_train",
            override_hw_config=direction.target.hw_config,
        )
        source_test_pred = _predict_with_bundle(
            bundle=source_bundle,
            frame=test_semantic,
            route=route,
            scale="source_transfer_test",
            override_hw_config=direction.target.hw_config,
        )
    target_full_pred = _predict_with_bundle(
        bundle=target_bundle,
        frame=test_semantic,
        route=route,
        scale="target_full",
        override_hw_config=None,
    )

    summary_rows: list[dict[str, Any]] = []
    combo_frames: list[pd.DataFrame] = []
    single_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []

    # Full target reference is deterministic and does not consume few-shot target data.
    full_row, full_combo, _ = _evaluate_prediction_variant(
        rows=target_full_pred,
        split_root=split_root,
        direction=direction,
        method="target_full",
        fraction=100.0,
        repeat=0,
        sampled_combo_count=total_train_combo_count,
        total_train_combo_count=total_train_combo_count,
        out_scale="target_full",
    )
    summary_rows.append(full_row)
    combo_frames.append(full_combo)

    for fraction in fractions:
        repeat_count = 1 if fraction in {0.0, 100.0} else int(repeats)
        for repeat in range(repeat_count):
            local_seed = int(seed) + repeat * 1009 + int(round(float(fraction) * 100.0))
            if only_target_only_baseline:
                if not include_target_only or fraction <= 0:
                    continue
                if fraction >= 100:
                    selected_combos, sampled_target_train = _sample_combo_keys(train_semantic, 100.0, local_seed)
                else:
                    selected_combos, sampled_target_train = _sample_combo_keys(train_semantic, fraction, local_seed)
                sample_frame = selected_combos.copy()
                sample_frame["direction"] = direction.key
                sample_frame["direction_label"] = direction.label
                sample_frame["sample_fraction_requested"] = float(fraction)
                sample_frame["repeat"] = int(repeat)
                sample_frame["seed"] = int(local_seed)
                sample_frames.append(sample_frame)
                target_only_modes = (
                    ["no_calibration", "calibrated"]
                    if target_only_baseline_mode == "both"
                    else [target_only_baseline_mode]
                )
                for baseline_mode in target_only_modes:
                    method_name = (
                        "target_only_no_calibration"
                        if baseline_mode == "no_calibration"
                        else "target_only_fewshot"
                    )
                    scale_prefix = (
                        "target_only_no_cal"
                        if baseline_mode == "no_calibration"
                        else "target_only"
                    )
                    if (
                        baseline_mode == "no_calibration"
                        and anchor_target_only_no_calibration_to_target_full
                        and fraction >= 100
                    ):
                        target_rows = target_full_pred.copy()
                    elif baseline_mode == "calibrated" and fraction >= 100:
                        target_rows = target_full_pred
                    else:
                        target_rows = _fit_target_only_rows(
                            sampled_train_semantic=sampled_target_train,
                            target_test_semantic=test_semantic,
                            target_hw_config=direction.target.hw_config,
                            route=route,
                            scale_label=f"{direction.key}:{scale_prefix}:{_format_fraction(fraction)}:rep{repeat}",
                            policy=POLICY,
                            calibrate_op_thread_scale=(baseline_mode == "calibrated"),
                        )
                    if baseline_mode == "no_calibration" and anchor_target_only_no_calibration_to_target_full:
                        target_rows = _anchor_target_only_operator_predictions(
                            target_rows=target_rows,
                            target_full_pred=target_full_pred,
                            fraction=fraction,
                            power=target_only_anchor_power,
                        )
                        method_name = "target_only_no_calibration_anchored"
                    if target_rows.empty:
                        continue
                    target_only_row, target_only_combo, _ = _evaluate_prediction_variant(
                        rows=target_rows,
                        split_root=split_root,
                        direction=direction,
                        method=method_name,
                        fraction=fraction,
                        repeat=repeat,
                        sampled_combo_count=int(len(selected_combos)),
                        total_train_combo_count=total_train_combo_count,
                        out_scale=f"{direction.key}:{scale_prefix}:{_format_fraction(fraction)}:rep{repeat}",
                    )
                    if "target_only_operator_anchor_weight" in target_rows.columns:
                        target_only_row["target_only_operator_anchor_weight"] = float(
                            pd.to_numeric(target_rows["target_only_operator_anchor_weight"], errors="coerce")
                            .dropna()
                            .iloc[0]
                        )
                        target_only_row["target_only_operator_anchor_power"] = float(target_only_anchor_power)
                    summary_rows.append(target_only_row)
                    combo_frames.append(target_only_combo)
                    single_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "direction": direction.key,
                                    "method": method_name,
                                    "sample_fraction_requested": float(fraction),
                                    "repeat": int(repeat),
                                    **_single_op_metrics(target_rows),
                                }
                            ]
                        )
                    )
                continue
            selected_combos, sampled_train = _sample_combo_keys(source_train_pred, fraction, local_seed)
            sample_frame = selected_combos.copy()
            sample_frame["direction"] = direction.key
            sample_frame["direction_label"] = direction.label
            sample_frame["sample_fraction_requested"] = float(fraction)
            sample_frame["repeat"] = int(repeat)
            sample_frame["seed"] = int(local_seed)
            sample_frames.append(sample_frame)

            base_scale = f"{direction.key}:zero_shot:{_format_fraction(fraction)}:rep{repeat}"
            zero_row, zero_combo, _ = _evaluate_prediction_variant(
                rows=source_test_pred,
                split_root=split_root,
                direction=direction,
                method="zero_shot",
                fraction=fraction,
                repeat=repeat,
                sampled_combo_count=int(len(selected_combos)),
                total_train_combo_count=total_train_combo_count,
                out_scale=base_scale,
            )
            summary_rows.append(zero_row)
            combo_frames.append(zero_combo)

            op_config = _fit_op_rescale(
                sampled_train,
                min_rows=op_scale_min_rows,
                clip_bounds=clip_bounds,
                aggregation=op_scale_aggregation,
                weight_column=op_scale_weight_column,
            )
            op_train = _apply_op_rescale(sampled_train, op_config)
            op_test = _apply_op_rescale(source_test_pred, op_config)
            op_scale = f"{direction.key}:op_rescale:{_format_fraction(fraction)}:rep{repeat}"
            op_row, op_combo, _ = _evaluate_prediction_variant(
                rows=op_test,
                split_root=split_root,
                direction=direction,
                method="fewshot_op_rescale",
                fraction=fraction,
                repeat=repeat,
                sampled_combo_count=int(len(selected_combos)),
                total_train_combo_count=total_train_combo_count,
                out_scale=op_scale,
            )
            op_row["op_rescale_fit_rows"] = int(op_config.get("fit_rows", 0))
            op_row["op_rescale_global_scale"] = float(op_config.get("global_scale", 1.0))
            summary_rows.append(op_row)
            combo_frames.append(op_combo)

            calibration_combo = _combo_metrics_for_rows(
                op_train.assign(scale=op_scale, route=route),
                split_root,
                "target_calibration",
            )
            e2e_scale = _fit_e2e_scale_from_combo(calibration_combo)
            e2e_test = _apply_e2e_scale(op_test, e2e_scale)
            e2e_scale_name = f"{direction.key}:op_rescale_e2e:{_format_fraction(fraction)}:rep{repeat}"
            e2e_row, e2e_combo, _ = _evaluate_prediction_variant(
                rows=e2e_test,
                split_root=split_root,
                direction=direction,
                method="fewshot_op_rescale_plus_e2e_scale",
                fraction=fraction,
                repeat=repeat,
                sampled_combo_count=int(len(selected_combos)),
                total_train_combo_count=total_train_combo_count,
                out_scale=e2e_scale_name,
            )
            e2e_row["op_rescale_fit_rows"] = int(op_config.get("fit_rows", 0))
            e2e_row["op_rescale_global_scale"] = float(op_config.get("global_scale", 1.0))
            e2e_row["fewshot_e2e_scale"] = float(e2e_scale)
            summary_rows.append(e2e_row)
            combo_frames.append(e2e_combo)

            surface_config = _fit_e2e_surface_from_combo(
                calibration_combo,
                alpha=float(e2e_surface_alpha),
                clip_bounds=e2e_surface_clip,
            )
            surface_scales = _predict_e2e_surface_scales(surface_config, op_combo)
            surface_test = _apply_combo_scales(op_test, surface_scales)
            surface_scale_name = f"{direction.key}:op_rescale_e2e_surface:{_format_fraction(fraction)}:rep{repeat}"
            surface_row, surface_combo, _ = _evaluate_prediction_variant(
                rows=surface_test,
                split_root=split_root,
                direction=direction,
                method="fewshot_op_rescale_plus_e2e_surface",
                fraction=fraction,
                repeat=repeat,
                sampled_combo_count=int(len(selected_combos)),
                total_train_combo_count=total_train_combo_count,
                out_scale=surface_scale_name,
            )
            surface_row["op_rescale_fit_rows"] = int(op_config.get("fit_rows", 0))
            surface_row["op_rescale_global_scale"] = float(op_config.get("global_scale", 1.0))
            surface_row["e2e_surface_fit_combos"] = int(surface_config.get("fit_combos", 0))
            surface_row["e2e_surface_feature_count"] = int(surface_config.get("feature_count", 0))
            surface_row["e2e_surface_alpha"] = float(e2e_surface_alpha)
            surface_row["e2e_surface_clip_low"] = float(e2e_surface_clip[0])
            surface_row["e2e_surface_clip_high"] = float(e2e_surface_clip[1])
            summary_rows.append(surface_row)
            combo_frames.append(surface_combo)

            single_frames.append(
                pd.DataFrame(
                    [
                        {
                            "direction": direction.key,
                            "method": "zero_shot",
                            "sample_fraction_requested": float(fraction),
                            "repeat": int(repeat),
                            **_single_op_metrics(source_test_pred),
                        },
                        {
                            "direction": direction.key,
                            "method": "fewshot_op_rescale",
                            "sample_fraction_requested": float(fraction),
                            "repeat": int(repeat),
                            **_single_op_metrics(op_test),
                        },
                        {
                            "direction": direction.key,
                            "method": "fewshot_op_rescale_plus_e2e_scale",
                            "sample_fraction_requested": float(fraction),
                            "repeat": int(repeat),
                            **_single_op_metrics(e2e_test),
                        },
                        {
                            "direction": direction.key,
                            "method": "fewshot_op_rescale_plus_e2e_surface",
                            "sample_fraction_requested": float(fraction),
                            "repeat": int(repeat),
                            **_single_op_metrics(surface_test),
                        },
                    ]
                )
            )

            if include_target_only and fraction > 0:
                if fraction >= 100:
                    sampled_target_train = train_semantic
                else:
                    _, sampled_target_train = _sample_combo_keys(train_semantic, fraction, local_seed)
                target_only_modes = (
                    ["no_calibration", "calibrated"]
                    if target_only_baseline_mode == "both"
                    else [target_only_baseline_mode]
                )
                for baseline_mode in target_only_modes:
                    method_name = (
                        "target_only_no_calibration"
                        if baseline_mode == "no_calibration"
                        else "target_only_fewshot"
                    )
                    scale_prefix = (
                        "target_only_no_cal"
                        if baseline_mode == "no_calibration"
                        else "target_only"
                    )
                    if (
                        baseline_mode == "no_calibration"
                        and anchor_target_only_no_calibration_to_target_full
                        and fraction >= 100
                    ):
                        target_rows = target_full_pred.copy()
                    elif baseline_mode == "calibrated" and fraction >= 100:
                        target_rows = target_full_pred
                    else:
                        target_rows = _fit_target_only_rows(
                            sampled_train_semantic=sampled_target_train,
                            target_test_semantic=test_semantic,
                            target_hw_config=direction.target.hw_config,
                            route=route,
                            scale_label=f"{direction.key}:{scale_prefix}:{_format_fraction(fraction)}:rep{repeat}",
                            policy=POLICY,
                            calibrate_op_thread_scale=(baseline_mode == "calibrated"),
                        )
                    if baseline_mode == "no_calibration" and anchor_target_only_no_calibration_to_target_full:
                        target_rows = _anchor_target_only_operator_predictions(
                            target_rows=target_rows,
                            target_full_pred=target_full_pred,
                            fraction=fraction,
                            power=target_only_anchor_power,
                        )
                        method_name = "target_only_no_calibration_anchored"
                    if target_rows.empty:
                        continue
                    target_only_row, target_only_combo, _ = _evaluate_prediction_variant(
                        rows=target_rows,
                        split_root=split_root,
                        direction=direction,
                        method=method_name,
                        fraction=fraction,
                        repeat=repeat,
                        sampled_combo_count=int(len(selected_combos)),
                        total_train_combo_count=total_train_combo_count,
                        out_scale=f"{direction.key}:{scale_prefix}:{_format_fraction(fraction)}:rep{repeat}",
                    )
                    if "target_only_operator_anchor_weight" in target_rows.columns:
                        target_only_row["target_only_operator_anchor_weight"] = float(
                            pd.to_numeric(target_rows["target_only_operator_anchor_weight"], errors="coerce")
                            .dropna()
                            .iloc[0]
                        )
                        target_only_row["target_only_operator_anchor_power"] = float(target_only_anchor_power)
                    summary_rows.append(target_only_row)
                    combo_frames.append(target_only_combo)
                    single_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "direction": direction.key,
                                    "method": method_name,
                                    "sample_fraction_requested": float(fraction),
                                    "repeat": int(repeat),
                                    **_single_op_metrics(target_rows),
                                }
                            ]
                        )
                    )

    return summary_rows, combo_frames, single_frames, sample_frames


def main() -> None:
    started_at = time.perf_counter()
    args = _parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = _parse_float_list(args.sample_fractions)
    clip_values = _parse_float_list(args.scale_clip)
    if len(clip_values) != 2:
        raise ValueError("--scale-clip must contain two comma-separated values")
    clip_bounds = (float(min(clip_values)), float(max(clip_values)))
    e2e_clip_values = _parse_float_list(args.e2e_surface_clip)
    if len(e2e_clip_values) != 2:
        raise ValueError("--e2e-surface-clip must contain two comma-separated values")
    e2e_surface_clip = (float(min(e2e_clip_values)), float(max(e2e_clip_values)))
    workers = max(1, int(args.workers))

    device1 = DeviceSpec(
        key="device1",
        label="Device_1",
        platform_label="智算平台",
        data_root=args.device1_data_root,
        hw_config=args.device1_hw_config,
        bundle_path=args.device1_bundle,
    )
    device2 = DeviceSpec(
        key="device2",
        label="Device_2",
        platform_label="通算平台",
        data_root=args.device2_data_root,
        hw_config=args.device2_hw_config,
        bundle_path=args.device2_bundle,
    )
    directions = [
        DirectionSpec(source=device1, target=device2),
        DirectionSpec(source=device2, target=device1),
    ]

    all_summary: list[dict[str, Any]] = []
    all_combo: list[pd.DataFrame] = []
    all_single: list[pd.DataFrame] = []
    all_samples: list[pd.DataFrame] = []

    run_kwargs = {
        "route": args.route,
        "fractions": fractions,
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "passes": int(args.passes),
        "op_scale_min_rows": int(args.op_scale_min_rows),
        "clip_bounds": clip_bounds,
        "op_scale_aggregation": str(args.op_scale_aggregation),
        "op_scale_weight_column": str(args.op_scale_weight_column),
        "e2e_surface_alpha": float(args.e2e_surface_alpha),
        "e2e_surface_clip": e2e_surface_clip,
        "include_target_only": not bool(args.skip_target_only_fewshot),
        "target_only_baseline_mode": str(args.target_only_baseline_mode),
        "only_target_only_baseline": bool(args.only_target_only_baseline),
        "anchor_target_only_no_calibration_to_target_full": bool(
            args.anchor_target_only_no_calibration_to_target_full
        ),
        "target_only_anchor_power": float(args.target_only_anchor_power),
    }
    direction_results: list[
        tuple[str, tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]]
    ] = []
    direction_workers = min(workers, len(directions))
    if direction_workers > 1:
        with ProcessPoolExecutor(max_workers=direction_workers) as executor:
            futures = {
                executor.submit(_run_direction, direction=direction, **run_kwargs): direction
                for direction in directions
            }
            for future in as_completed(futures):
                direction = futures[future]
                print(f"[fewshot] direction={direction.label} completed", flush=True)
                direction_results.append((direction.key, future.result()))
    else:
        for direction in directions:
            print(f"[fewshot] direction={direction.label}", flush=True)
            direction_results.append((direction.key, _run_direction(direction=direction, **run_kwargs)))

    direction_order = {direction.key: index for index, direction in enumerate(directions)}
    for _direction_key, result in sorted(direction_results, key=lambda item: direction_order[item[0]]):
        summary_rows, combo_frames, single_frames, sample_frames = result
        all_summary.extend(summary_rows)
        all_combo.extend(combo_frames)
        all_single.extend(single_frames)
        all_samples.extend(sample_frames)

    summary = pd.DataFrame(all_summary)
    if args.merge_existing_summary is not None:
        existing_summary_path = Path(args.merge_existing_summary)
        if not existing_summary_path.exists():
            raise FileNotFoundError(f"--merge-existing-summary not found: {existing_summary_path}")
        existing_summary = pd.read_csv(existing_summary_path)
        baseline_methods = {
            "target_only_no_calibration",
            "target_only_no_calibration_anchored",
            "target_only_fewshot",
        }
        existing_summary = existing_summary[~existing_summary["method"].isin(baseline_methods)].copy()
        summary = pd.concat(
            [existing_summary, summary[summary["method"].isin(baseline_methods)].copy()],
            ignore_index=True,
            sort=False,
        )
    summary.to_csv(out_dir / "fewshot_generalization_summary.csv", index=False)
    combo = pd.concat([frame for frame in all_combo if not frame.empty], ignore_index=True) if all_combo else pd.DataFrame()
    combo.to_csv(out_dir / "fewshot_e2e_combo_metrics.csv", index=False)
    single = pd.concat([frame for frame in all_single if not frame.empty], ignore_index=True) if all_single else pd.DataFrame()
    single.to_csv(out_dir / "fewshot_single_op_metrics.csv", index=False)
    samples = pd.concat([frame for frame in all_samples if not frame.empty], ignore_index=True) if all_samples else pd.DataFrame()
    samples.to_csv(out_dir / "fewshot_sample_manifest.csv", index=False)

    _make_plots(summary, out_dir, primary_method=str(args.primary_method))
    _write_report(summary, out_dir, args)
    elapsed_s = time.perf_counter() - started_at
    manifest = {
        "route": args.route,
        "sample_fractions": fractions,
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "primary_method": str(args.primary_method),
        "workers_requested": workers,
        "direction_workers": direction_workers,
        "elapsed_s": elapsed_s,
        "op_scale_aggregation": str(args.op_scale_aggregation),
        "op_scale_weight_column": str(args.op_scale_weight_column),
        "e2e_surface_alpha": float(args.e2e_surface_alpha),
        "e2e_surface_clip": list(e2e_surface_clip),
        "skip_target_only_fewshot": bool(args.skip_target_only_fewshot),
        "target_only_baseline_mode": str(args.target_only_baseline_mode),
        "only_target_only_baseline": bool(args.only_target_only_baseline),
        "merge_existing_summary": str(args.merge_existing_summary) if args.merge_existing_summary else "",
        "outputs": [
            "fewshot_generalization_summary.csv",
            "fewshot_generalization_decision.csv",
            "fewshot_e2e_combo_metrics.csv",
            "fewshot_single_op_metrics.csv",
            "fewshot_sample_manifest.csv",
            "fewshot_e2e_mape_vs_target_data.png",
            "fewshot_e2e_p90_vs_target_data.png",
            "fewshot_10pct_vs_full_bar.png",
            "fewshot_data_efficiency_summary.png",
            "fewshot_seed_stability_boxplot.png",
            "fewshot_zero_to_calibrated_gain.png",
            "CROSS_PLATFORM_FEWSHOT_GENERALIZATION_REPORT.md",
        ],
    }
    (out_dir / "fewshot_generalization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
