#!/usr/bin/env python3
"""Analyze DLRM memory dominance and compute/pipeline sensitivity.

This is an independent post-processing experiment. It reloads the exported v3
bundles and held-out E2E test frames, rebuilds analytical service columns, and
generates evidence for two statements:

1. DLRM remains memory dominated in profiled operator time and analytical
   service decomposition.
2. Compute/pipeline hardware parameters can still affect E2E latency through
   local key-path stages, without implying that the full workload is compute
   bottlenecked.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytical_base import run as base_run  # noqa: E402
from calibrator.ml_semantic_routes import (  # noqa: E402
    WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
    prepare_semantic_static_frame,
)
from calibrator.pipeline_eval import evaluate_source_test_pipeline  # noqa: E402
from calibrator.run import _load_raw_splits  # noqa: E402
from calibrator.whitebox_family_routes import predict_whitebox_family_physics_guarded_rf  # noqa: E402


POLICY = "joint_6d_tuple"
TARGET_US_COLUMN = base_run.TARGET_US_COLUMN

SERVICE_MODULES = [
    "descriptor",
    "stream_mem",
    "random_mem",
    "compute",
    "reorder",
    "sync",
    "dispatch",
]

MEMORY_RELATED_OPS = {
    "Gather",
    "EmbeddingBag",
    "ReduceSum",
    "ScatterAdd",
    "Concat",
    "Transpose",
    "Tile",
    "PadV3",
    "MemSet",
}
COMPUTE_RELATED_OPS = {
    "Gemm",
    "MatMul",
    "BatchMatMulV2",
    "Linear",
    "Add",
    "Mul",
    "Relu",
    "Gelu",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "TopK",
    "ReduceMean",
}
METADATA_OPS = {
    "Cast",
    "Constant",
    "Flatten",
    "NonZero",
    "Reshape",
    "Shape",
    "Sort",
    "Unsqueeze",
    "Where",
}

PARAMETER_PERTURBATIONS = [
    {
        "parameter_key": "cpu_clock_plus25",
        "parameter_label": "CPU frequency +25%",
        "dotted_paths": ["core.cpu_clock"],
        "factor": 1.25,
        "direction": "capacity_up",
    },
    {
        "parameter_key": "simd_width_plus25",
        "parameter_label": "SIMD width +25%",
        "dotted_paths": ["instruction.simd_width_bits"],
        "factor": 1.25,
        "direction": "capacity_up",
    },
    {
        "parameter_key": "fma_throughput_plus25",
        "parameter_label": "FMA throughput +25%",
        "dotted_paths": ["instruction.fp_throughput_per_cycle.vector_sp_fma"],
        "factor": 1.25,
        "direction": "capacity_up",
    },
    {
        "parameter_key": "fma_latency_minus25",
        "parameter_label": "FMA latency -25%",
        "dotted_paths": ["instruction.fp_latency_cycles.vector_sp_fma"],
        "factor": 0.75,
        "direction": "latency_down",
    },
    {
        "parameter_key": "pipeline_width_plus25",
        "parameter_label": "Pipeline width +25%",
        "dotted_paths": [
            "pipeline.fetch_width",
            "pipeline.decode_width",
            "pipeline.rename_width",
            "pipeline.dispatch_width",
            "pipeline.issue_width",
            "pipeline.wb_width",
            "pipeline.commit_width",
        ],
        "factor": 1.25,
        "direction": "capacity_up",
    },
]


@dataclass(frozen=True)
class DeviceSpec:
    key: str
    label: str
    root: Path
    bundle: Path
    data_root: Path
    hw_config: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device1-root", required=True)
    parser.add_argument("--device2-root", required=True)
    parser.add_argument("--device1-bundle", required=True)
    parser.add_argument("--device2-bundle", required=True)
    parser.add_argument("--device1-data-root", required=True)
    parser.add_argument("--device2-data-root", required=True)
    parser.add_argument("--device1-hw-config", required=True)
    parser.add_argument("--device2-hw-config", required=True)
    parser.add_argument("--route", default=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _resolve_split_root(path: Path) -> Path:
    root = Path(path)
    if (root / "split_manifest.json").exists() and (root / "train.csv").exists():
        return root
    preferred = root / "source_kunpeng920_dataset3_unseen_joint_6d_tuple"
    if (preferred / "split_manifest.json").exists() and (preferred / "train.csv").exists():
        return preferred
    candidates = sorted(
        item
        for item in root.glob("*")
        if item.is_dir() and (item / "split_manifest.json").exists() and (item / "train.csv").exists()
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"cannot locate split_manifest.json and train.csv under {path}")


def _load_bundle(path: Path, route: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model bundle not found: {path}")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"model bundle must be a dict, got {type(bundle).__name__}")
    bundle_route = str(bundle.get("route", "")).strip()
    if bundle_route and bundle_route != route:
        raise ValueError(f"bundle route mismatch for {path}: expected {route}, got {bundle_route}")
    return bundle


def _load_hw_payload(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"hardware config must be a mapping: {path}")
    return payload


def _nested_get(payload: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _nested_set(payload: dict[str, Any], dotted: str, value: Any) -> None:
    current: Any = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise TypeError(f"cannot set {dotted}: non-mapping at {part}")
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        raise TypeError(f"cannot set {dotted}: parent is not a mapping")
    current[parts[-1]] = value


def _parse_quantity(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    multipliers = {
        "GHz": 1.0,
        "MHz": 1e-3,
        "KiB": 1024.0,
        "MiB": 1024.0 * 1024.0,
        "GiB": 1024.0 * 1024.0 * 1024.0,
    }
    for suffix, mult in multipliers.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)].strip()) * mult
    return float(text)


def _format_like_original(value: Any, numeric: float) -> Any:
    if isinstance(value, str):
        text = value.strip()
        for suffix in ("GHz", "MHz", "KiB", "MiB", "GiB"):
            if text.endswith(suffix):
                if suffix == "MHz":
                    shown = numeric / 1e-3
                elif suffix == "KiB":
                    shown = numeric / 1024.0
                elif suffix == "MiB":
                    shown = numeric / (1024.0 * 1024.0)
                elif suffix == "GiB":
                    shown = numeric / (1024.0 * 1024.0 * 1024.0)
                else:
                    shown = numeric
                return f"{shown:.6g}{suffix}"
    return float(numeric)


def _mutate_hw_config(base_payload: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(base_payload)
    for dotted in spec["dotted_paths"]:
        original = _nested_get(mutated, dotted)
        numeric = _parse_quantity(original)
        if not np.isfinite(numeric):
            raise ValueError(f"cannot mutate non-numeric hardware field: {dotted}={original!r}")
        updated = numeric * float(spec["factor"])
        _nested_set(mutated, dotted, _format_like_original(original, updated))
    return mutated


def _prepare_test_frame(split_root: Path, hw_config: Path) -> pd.DataFrame:
    raw_splits = _load_raw_splits(split_root, include_pipeline_splits=True)
    for frame in raw_splits.values():
        frame.attrs["dataset_root"] = str(split_root)
    test_raw = raw_splits.get("e2e_test_raw", raw_splits.get("test_raw", raw_splits["test"]))
    params = base_run.analytical_helpers.default_params()
    semantic = prepare_semantic_static_frame(test_raw, hw_config, params)
    semantic.attrs["dataset_root"] = str(split_root)
    return semantic


def _predict_rows(
    *,
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    route: str,
    scale: str,
    dataset: str,
    override_hw_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    prepared, pred = predict_whitebox_family_physics_guarded_rf(
        bundle,
        frame,
        override_hw_config=override_hw_config,
    )
    rows = prepared.copy()
    pred_arr = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    rows["pred_us"] = pred_arr
    rows["pred_total_us"] = pred_arr
    rows["policy"] = POLICY
    rows["dataset"] = dataset
    rows["route"] = str(route)
    rows["scale"] = str(scale)
    if TARGET_US_COLUMN not in rows.columns:
        raise KeyError(f"prepared frame is missing {TARGET_US_COLUMN}")
    return rows


def _pipeline_rows(rows: pd.DataFrame) -> pd.DataFrame:
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
    return rows.loc[:, present].copy()


def _evaluate_pipeline(rows: pd.DataFrame, split_root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, combo_metrics, partial, _embedding, _audit = evaluate_source_test_pipeline(
        prediction_rows=_pipeline_rows(rows),
        split_root=split_root,
        policy=POLICY,
        dataset=dataset,
    )
    if not partial.empty:
        partial_count = int(len(partial))
    else:
        partial_count = 0
    if summary.empty:
        raise RuntimeError(f"pipeline evaluation produced no summary for {dataset}; partial_count={partial_count}")
    return summary, combo_metrics


def _op_category(op_type: Any) -> str:
    op = str(op_type)
    if op in MEMORY_RELATED_OPS:
        return "memory_related"
    if op in COMPUTE_RELATED_OPS:
        return "compute_related"
    if op in METADATA_OPS:
        return "metadata_dispatch"
    return "other"


def _phase_label(rows: pd.DataFrame) -> pd.Series:
    if "exec_region_static" in rows.columns:
        text = rows["exec_region_static"].fillna("").astype(str)
        if text.str.len().gt(0).any():
            return text.replace("", "unknown")
    if "submodel_role_static" in rows.columns:
        text = rows["submodel_role_static"].fillna("").astype(str)
        return text.replace("", "unknown")
    return pd.Series("unknown", index=rows.index)


def _safe_share(numer: float, denom: float) -> float:
    return float(numer / denom) if np.isfinite(denom) and denom > 0 else float("nan")


def _actual_time_share(rows: pd.DataFrame, device: DeviceSpec) -> pd.DataFrame:
    work = rows.copy()
    work["actual_us"] = pd.to_numeric(work[TARGET_US_COLUMN], errors="coerce").fillna(0.0)
    work["op_category"] = work["op_type"].map(_op_category)
    grouped = (
        work.groupby(["op_category", "op_type"], dropna=False, as_index=False)["actual_us"]
        .sum()
        .rename(columns={"actual_us": "actual_us_sum"})
    )
    total = float(grouped["actual_us_sum"].sum())
    grouped["actual_time_share"] = grouped["actual_us_sum"].map(lambda value: _safe_share(float(value), total))
    grouped.insert(0, "device", device.key)
    grouped.insert(1, "device_label", device.label)
    return grouped.sort_values(["device", "actual_us_sum"], ascending=[True, False]).reset_index(drop=True)


def _service_share(rows: pd.DataFrame, device: DeviceSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = rows.copy()
    for module in SERVICE_MODULES:
        column = f"svc_{module}_center_us"
        if column not in work.columns:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    work["op_category"] = work["op_type"].map(_op_category)
    work["phase"] = _phase_label(work)
    work["memory_service_us"] = work["svc_stream_mem_center_us"] + work["svc_random_mem_center_us"]
    work["compute_service_us"] = work["svc_compute_center_us"]
    work["dispatch_metadata_service_us"] = work["svc_dispatch_center_us"] + work["svc_descriptor_center_us"]
    work["disabled_other_service_us"] = work["svc_reorder_center_us"] + work["svc_sync_center_us"]

    service_columns = [
        "memory_service_us",
        "compute_service_us",
        "dispatch_metadata_service_us",
        "disabled_other_service_us",
        *[f"svc_{module}_center_us" for module in SERVICE_MODULES],
    ]

    overall = work[service_columns].sum().to_frame().T
    overall.insert(0, "device", device.key)
    overall.insert(1, "device_label", device.label)
    total = float(overall[["memory_service_us", "compute_service_us", "dispatch_metadata_service_us", "disabled_other_service_us"]].sum(axis=1).iloc[0])
    for column in ["memory_service_us", "compute_service_us", "dispatch_metadata_service_us", "disabled_other_service_us"]:
        overall[f"{column}_share"] = overall[column].map(lambda value: _safe_share(float(value), total))

    by_op = work.groupby(["op_category", "op_type"], dropna=False, as_index=False)[service_columns].sum()
    by_op.insert(0, "device", device.key)
    by_op.insert(1, "device_label", device.label)
    denom = by_op[["memory_service_us", "compute_service_us", "dispatch_metadata_service_us", "disabled_other_service_us"]].sum(axis=1)
    for column in ["memory_service_us", "compute_service_us", "dispatch_metadata_service_us", "disabled_other_service_us"]:
        by_op[f"{column}_share_within_op"] = by_op[column] / denom.replace(0.0, np.nan)

    by_phase = work.groupby(["phase", "op_category"], dropna=False, as_index=False)[service_columns].sum()
    by_phase.insert(0, "device", device.key)
    by_phase.insert(1, "device_label", device.label)
    denom_phase = by_phase[["memory_service_us", "compute_service_us", "dispatch_metadata_service_us", "disabled_other_service_us"]].sum(axis=1)
    for column in ["memory_service_us", "compute_service_us", "dispatch_metadata_service_us", "disabled_other_service_us"]:
        by_phase[f"{column}_share_within_phase_op_category"] = by_phase[column] / denom_phase.replace(0.0, np.nan)

    return overall, by_op, by_phase


def _stable_join_key(rows: pd.DataFrame) -> pd.Series:
    if "row_uid" in rows.columns:
        return rows["row_uid"].astype(str)
    return pd.Series(np.arange(len(rows)).astype(str), index=rows.index)


def _sensitivity_rows(
    *,
    baseline_rows: pd.DataFrame,
    perturbed_rows: pd.DataFrame,
    baseline_combo: pd.DataFrame,
    perturbed_combo: pd.DataFrame,
    device: DeviceSpec,
    parameter_key: str,
    parameter_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = baseline_rows.copy()
    target = perturbed_rows.copy()
    base["__key"] = _stable_join_key(base)
    target["__key"] = _stable_join_key(target)
    keep_target = target[["__key", "pred_us"]].rename(columns={"pred_us": "pred_us_perturbed"})
    merged = base.merge(keep_target, on="__key", how="inner")
    merged["pred_us_baseline"] = pd.to_numeric(merged["pred_us"], errors="coerce").fillna(0.0)
    merged["pred_us_perturbed"] = pd.to_numeric(merged["pred_us_perturbed"], errors="coerce").fillna(0.0)
    merged["delta_us"] = merged["pred_us_perturbed"] - merged["pred_us_baseline"]
    merged["improvement_us"] = merged["pred_us_baseline"] - merged["pred_us_perturbed"]
    merged["abs_delta_us"] = merged["delta_us"].abs()
    merged["phase"] = _phase_label(merged)

    by_op = (
        merged.groupby(["op_type"], dropna=False, as_index=False)
        .agg(
            rows=("op_type", "size"),
            baseline_pred_us_sum=("pred_us_baseline", "sum"),
            perturbed_pred_us_sum=("pred_us_perturbed", "sum"),
            delta_us_sum=("delta_us", "sum"),
            improvement_us_sum=("improvement_us", "sum"),
            abs_delta_us_sum=("abs_delta_us", "sum"),
        )
        .sort_values("abs_delta_us_sum", ascending=False)
    )
    by_op.insert(0, "parameter_key", parameter_key)
    by_op.insert(1, "parameter_label", parameter_label)
    by_op.insert(0, "device", device.key)
    by_op.insert(1, "device_label", device.label)

    by_phase = (
        merged.groupby(["phase", "op_type"], dropna=False, as_index=False)
        .agg(
            rows=("op_type", "size"),
            baseline_pred_us_sum=("pred_us_baseline", "sum"),
            perturbed_pred_us_sum=("pred_us_perturbed", "sum"),
            delta_us_sum=("delta_us", "sum"),
            improvement_us_sum=("improvement_us", "sum"),
            abs_delta_us_sum=("abs_delta_us", "sum"),
        )
        .sort_values("abs_delta_us_sum", ascending=False)
    )
    by_phase.insert(0, "parameter_key", parameter_key)
    by_phase.insert(1, "parameter_label", parameter_label)
    by_phase.insert(0, "device", device.key)
    by_phase.insert(1, "device_label", device.label)

    key_cols = ["case_id", "source_name", "combo"]
    combo_base = baseline_combo[key_cols + ["predicted_e2e_us"]].rename(
        columns={"predicted_e2e_us": "baseline_e2e_us"}
    )
    combo_mut = perturbed_combo[key_cols + ["predicted_e2e_us"]].rename(
        columns={"predicted_e2e_us": "perturbed_e2e_us"}
    )
    combo = combo_base.merge(combo_mut, on=key_cols, how="inner")
    base_mean = float(combo["baseline_e2e_us"].mean()) if not combo.empty else float("nan")
    pert_mean = float(combo["perturbed_e2e_us"].mean()) if not combo.empty else float("nan")
    e2e_improvement = base_mean - pert_mean
    summary = {
        "device": device.key,
        "device_label": device.label,
        "parameter_key": parameter_key,
        "parameter_label": parameter_label,
        "combo_count": int(len(combo)),
        "baseline_e2e_mean_us": base_mean,
        "perturbed_e2e_mean_us": pert_mean,
        "e2e_improvement_us": e2e_improvement,
        "e2e_improvement_pct": _safe_share(e2e_improvement, base_mean),
    }
    return by_op, by_phase, summary


def _pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value * 100.0:.1f}%"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_actual_share(actual_share: pd.DataFrame, out_dir: Path) -> None:
    category_order = ["memory_related", "compute_related", "metadata_dispatch", "other"]
    colors = {
        "memory_related": "#2b6cb0",
        "compute_related": "#dd6b20",
        "metadata_dispatch": "#718096",
        "other": "#cbd5e0",
    }
    pivot = (
        actual_share.groupby(["device", "op_category"], as_index=False)["actual_us_sum"].sum()
        .pivot(index="device", columns="op_category", values="actual_us_sum")
        .reindex(["Device_1", "Device_2"])
        .fillna(0.0)
    )
    denom = pivot.sum(axis=1).replace(0.0, np.nan)
    shares = pivot.div(denom, axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bottom = np.zeros(len(shares))
    x = np.arange(len(shares))
    for category in category_order:
        values = shares[category].to_numpy() if category in shares.columns else np.zeros(len(shares))
        ax.bar(x, values, bottom=bottom, label=category.replace("_", " "), color=colors[category], width=0.58)
        for xi, yi, bi in zip(x, values, bottom):
            if yi >= 4.0:
                ax.text(xi, bi + yi / 2.0, f"{yi:.1f}%", ha="center", va="center", color="white", fontsize=10)
        bottom += values
    ax.set_xticks(x, ["Device_1", "Device_2"])
    ax.set_ylabel("Actual operator time share (%)")
    ax.set_title("DLRM held-out operator time is memory dominated")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True)
    _save_fig(fig, out_dir, "memory_vs_compute_actual_time_share")


def _plot_service_share(service_share: pd.DataFrame, out_dir: Path) -> None:
    columns = [
        ("memory_service_us", "memory service", "#2b6cb0"),
        ("compute_service_us", "compute service", "#dd6b20"),
        ("dispatch_metadata_service_us", "dispatch/metadata", "#718096"),
        ("disabled_other_service_us", "disabled/other", "#cbd5e0"),
    ]
    data = service_share.set_index("device").reindex(["Device_1", "Device_2"])
    totals = data[[col for col, _label, _color in columns]].sum(axis=1).replace(0.0, np.nan)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    for col, label, color in columns:
        values = (data[col] / totals * 100.0).fillna(0.0).to_numpy()
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.58)
        for xi, yi, bi in zip(x, values, bottom):
            if yi >= 4.0:
                ax.text(xi, bi + yi / 2.0, f"{yi:.1f}%", ha="center", va="center", color="white", fontsize=10)
        bottom += values
    ax.set_xticks(x, ["Device_1", "Device_2"])
    ax.set_ylabel("Analytical service share (%)")
    ax.set_title("Analytical service split: memory vs compute/dispatch")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True)
    _save_fig(fig, out_dir, "memory_vs_compute_service_share")


def _plot_operator_share(actual_share: pd.DataFrame, out_dir: Path) -> None:
    top_ops = ["Gather", "ReduceSum", "Gemm", "MatMul"]
    work = actual_share.copy()
    work["op_group"] = np.where(work["op_type"].isin(top_ops), work["op_type"], "Other")
    pivot = (
        work.groupby(["device", "op_group"], as_index=False)["actual_us_sum"].sum()
        .pivot(index="device", columns="op_group", values="actual_us_sum")
        .reindex(["Device_1", "Device_2"])
        .fillna(0.0)
    )
    order = [op for op in top_ops if op in pivot.columns] + (["Other"] if "Other" in pivot.columns else [])
    colors = ["#2b6cb0", "#38a169", "#dd6b20", "#ed8936", "#a0aec0"]
    shares = pivot.div(pivot.sum(axis=1).replace(0.0, np.nan), axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(shares))
    bottom = np.zeros(len(shares))
    for idx, op in enumerate(order):
        values = shares[op].fillna(0.0).to_numpy()
        ax.bar(x, values, bottom=bottom, label=op, width=0.58, color=colors[idx % len(colors)])
        for xi, yi, bi in zip(x, values, bottom):
            if yi >= 4.0:
                ax.text(xi, bi + yi / 2.0, f"{yi:.1f}%", ha="center", va="center", color="white", fontsize=10)
        bottom += values
    ax.set_xticks(x, ["Device_1", "Device_2"])
    ax.set_ylabel("Actual operator time share (%)")
    ax.set_title("Top operator contribution in profiled time")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(order), frameon=True)
    _save_fig(fig, out_dir, "operator_actual_time_share_stacked_bar")


def _plot_sensitivity_top_ops(by_op: pd.DataFrame, out_dir: Path) -> None:
    short_param = {
        "CPU frequency +25%": "CPU freq +25%",
        "SIMD width +25%": "SIMD width +25%",
        "FMA throughput +25%": "FMA tp +25%",
        "FMA latency -25%": "FMA lat -25%",
        "Pipeline width +25%": "Pipe width +25%",
    }
    work = by_op.copy()
    work["abs_delta_share_in_parameter"] = work.groupby(["device", "parameter_key"])["abs_delta_us_sum"].transform(
        lambda s: s / max(float(s.sum()), 1e-9)
    )
    top = (
        work.sort_values("abs_delta_us_sum", ascending=False)
        .groupby(["device", "parameter_key"], as_index=False)
        .head(3)
        .copy()
    )
    top["short_parameter_label"] = top["parameter_label"].map(short_param).fillna(top["parameter_label"])
    top["label"] = top["short_parameter_label"] + "\n" + top["op_type"].astype(str)
    devices = ["Device_1", "Device_2"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), sharex=False)
    for ax, device in zip(axes, devices):
        sub = top[top["device"].eq(device)].sort_values("abs_delta_us_sum", ascending=True).tail(6)
        values = sub["abs_delta_us_sum"].to_numpy(dtype=float) / 1000.0
        y = np.arange(len(sub))
        ax.barh(y, values, color="#2b6cb0", alpha=0.86)
        ax.set_yticks(y, sub["label"].tolist(), fontsize=10)
        xmax = max(float(values.max()) if len(values) else 0.0, 1.0)
        ax.set_xlim(0.0, xmax * 1.18)
        for yi, value in zip(y, values):
            label = f"{value:.0f} ms" if value >= 100.0 else f"{value:.1f} ms"
            ax.text(value + xmax * 0.015, yi, label, va="center", ha="left", fontsize=9)
        ax.set_xlabel("Operator prediction delta (ms, absolute)", fontsize=11)
        ax.set_title(device)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Top operator-level sensitivity to compute/pipeline perturbations", y=0.97, fontsize=15)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.90], w_pad=3.0)
    _save_fig(fig, out_dir, "compute_parameter_sensitivity_top_ops")


def _plot_sensitivity_waterfall(summary: pd.DataFrame, by_op: pd.DataFrame, out_dir: Path) -> None:
    devices = ["Device_1", "Device_2"]
    short_param = {
        "CPU frequency +25%": "CPU freq\n+25%",
        "SIMD width +25%": "SIMD width\n+25%",
        "FMA throughput +25%": "FMA tp\n+25%",
        "FMA latency -25%": "FMA lat\n-25%",
        "Pipeline width +25%": "Pipe width\n+25%",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharey=True)
    for ax, device in zip(axes, devices):
        sub_summary = summary[summary["device"].eq(device)].copy()
        labels = sub_summary["parameter_label"].map(short_param).fillna(sub_summary["parameter_label"]).tolist()
        values = sub_summary["e2e_improvement_pct"].fillna(0.0).to_numpy(dtype=float) * 100.0
        colors = ["#2b6cb0" if value >= 0 else "#c53030" for value in values]
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, alpha=0.86)
        ax.axhline(0.0, color="#555555", lw=0.9)
        ax.set_xticks(x, labels, rotation=0, ha="center", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Mean scheduled E2E improvement (%)", fontsize=11)
        ax.set_title(device)
        ax.grid(axis="y", alpha=0.25)
        finite = values[np.isfinite(values)]
        ymin = min(float(finite.min()) if finite.size else 0.0, 0.0)
        ymax = max(float(finite.max()) if finite.size else 0.0, 0.0)
        padding = max((ymax - ymin) * 0.18, 0.08)
        ax.set_ylim(ymin - padding * 1.6, ymax + padding * 2.3)
        for xi, yi in zip(x, values):
            offset = padding * 0.25 if yi >= 0 else -padding * 0.25
            ax.text(
                xi,
                yi + offset,
                f"{yi:.2f}%",
                ha="center",
                va="bottom" if yi >= 0 else "top",
                fontsize=9,
                clip_on=False,
            )
    fig.suptitle("Scheduled E2E sensitivity to compute/pipeline parameters", y=0.98, fontsize=15)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.90], w_pad=3.0)
    _save_fig(fig, out_dir, "compute_parameter_sensitivity_waterfall")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    text = frame.fillna("").copy()
    headers = [str(col) for col in text.columns]
    rows = [[str(value) for value in row] for row in text.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _build_report(
    *,
    out_dir: Path,
    actual_share: pd.DataFrame,
    service_share: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    pipeline_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    actual_cat = (
        actual_share.groupby(["device", "op_category"], as_index=False)["actual_us_sum"].sum()
        .pivot(index="device", columns="op_category", values="actual_us_sum")
        .fillna(0.0)
    )
    actual_rows = []
    for device in ["Device_1", "Device_2"]:
        row = actual_cat.loc[device] if device in actual_cat.index else pd.Series(dtype=float)
        total = float(row.sum())
        memory = _safe_share(float(row.get("memory_related", 0.0)), total)
        compute = _safe_share(float(row.get("compute_related", 0.0)), total)
        actual_rows.append(
            {
                "Device": device,
                "memory-related actual share": _pct(memory),
                "compute-related actual share": _pct(compute),
            }
        )
    actual_table = pd.DataFrame(actual_rows)

    service_rows = []
    for _, row in service_share.iterrows():
        service_rows.append(
            {
                "Device": row["device"],
                "memory service share": _pct(float(row.get("memory_service_us_share", np.nan))),
                "compute service share": _pct(float(row.get("compute_service_us_share", np.nan))),
                "dispatch/metadata share": _pct(float(row.get("dispatch_metadata_service_us_share", np.nan))),
            }
        )
    service_table = pd.DataFrame(service_rows)

    top_sens = (
        sensitivity_summary.sort_values(["device", "e2e_improvement_pct"], ascending=[True, False])
        .groupby("device", as_index=False)
        .head(3)
        .copy()
    )
    top_sens_table = top_sens[
        ["device", "parameter_label", "e2e_improvement_pct", "baseline_e2e_mean_us", "perturbed_e2e_mean_us"]
    ].copy()
    top_sens_table["e2e_improvement_pct"] = top_sens_table["e2e_improvement_pct"].map(_pct)
    for column in ["baseline_e2e_mean_us", "perturbed_e2e_mean_us"]:
        top_sens_table[column] = top_sens_table[column].map(lambda value: f"{float(value):.1f}" if np.isfinite(value) else "NA")

    combo_table = pipeline_summary.copy()
    for column in ["e2e_mape", "e2e_p90", "e2e_median", "coverage_ratio_mean"]:
        if column in combo_table.columns:
            combo_table[column] = combo_table[column].map(_pct)
    combo_cols = [col for col in ["dataset", "combo_count_full", "combo_count_partial", "coverage_ratio_mean", "e2e_mape", "e2e_p90"] if col in combo_table.columns]

    command = f"""python calibrator/experimental/analyze_dlrm_compute_memory_attribution.py \\
  --device1-root {args.device1_root} \\
  --device2-root {args.device2_root} \\
  --device1-bundle {args.device1_bundle} \\
  --device2-bundle {args.device2_bundle} \\
  --device1-data-root {args.device1_data_root} \\
  --device2-data-root {args.device2_data_root} \\
  --device1-hw-config {args.device1_hw_config} \\
  --device2-hw-config {args.device2_hw_config} \\
  --route {args.route} \\
  --out-dir {args.out_dir}"""

    report = f"""# DLRM Compute/Memory Attribution Report

Generated at: {datetime.now(timezone.utc).isoformat()}

This is an independent explanatory analysis. It does not retrain the CPU model
and does not change the mainline E2E metrics.

## Held-out Pipeline Coverage

{_markdown_table(combo_table[combo_cols])}

## 1. Memory-dominant Evidence

The first criterion uses true profiled operator time on held-out E2E graphs.
`Gather`, `ReduceSum`, and data movement operators are treated as
memory-related. Dense linear algebra and elementwise arithmetic are treated as
compute-related.

{_markdown_table(actual_table)}

The second criterion uses analytical service decomposition. The memory service
is `stream_mem + random_mem`; compute service is `svc_compute`; dispatch and
metadata are shown separately.

{_markdown_table(service_table)}

These two views are consistent: the workload is still dominated by memory
access and movement. The compute-parameter analysis below should therefore be
interpreted as local key-path sensitivity, not as evidence that DLRM is a
compute-bottleneck workload.

## 2. Why Compute/Pipeline Parameters Still Matter

For each device, the script perturbs one compute or pipeline hardware parameter
at a time and reruns the exported v3 bundle with `override_hw_config`. The RF
weights are fixed; only the hardware payload changes.

Top E2E sensitivity cases:

{_markdown_table(top_sens_table)}

The sensitivity usually concentrates in dense MLP, reduction, or tail/serial
regions. These stages can sit on the E2E critical path, so a relatively small
share of operator time can still cause visible E2E movement when its hardware
service rate changes.

## Output Files

- `operator_actual_time_share.csv`
- `memory_vs_compute_service_share.csv`
- `analytical_module_share_by_op_type.csv`
- `analytical_module_share_by_phase.csv`
- `compute_parameter_sensitivity_by_op_type.csv`
- `compute_parameter_sensitivity_by_phase.csv`
- `e2e_compute_memory_attribution_summary.csv`
- `memory_vs_compute_actual_time_share.png/pdf`
- `memory_vs_compute_service_share.png/pdf`
- `operator_actual_time_share_stacked_bar.png/pdf`
- `compute_parameter_sensitivity_top_ops.png/pdf`
- `compute_parameter_sensitivity_waterfall.png/pdf`

## Reproduce

```bash
{command}
```
"""
    (out_dir / "DLRM_COMPUTE_MEMORY_ATTRIBUTION_REPORT.md").write_text(report, encoding="utf-8")


def _process_device(device: DeviceSpec, route: str) -> dict[str, Any]:
    split_root = _resolve_split_root(device.data_root)
    bundle = _load_bundle(device.bundle, route)
    semantic = _prepare_test_frame(split_root, device.hw_config)
    baseline_rows = _predict_rows(
        bundle=bundle,
        frame=semantic,
        route=route,
        scale="baseline",
        dataset=device.key,
        override_hw_config=None,
    )
    base_summary, base_combo = _evaluate_pipeline(baseline_rows, split_root, device.key)
    hw_payload = _load_hw_payload(device.hw_config)

    by_op_sensitivity: list[pd.DataFrame] = []
    by_phase_sensitivity: list[pd.DataFrame] = []
    sensitivity_summary: list[dict[str, Any]] = []
    for spec in PARAMETER_PERTURBATIONS:
        mutated = _mutate_hw_config(hw_payload, spec)
        perturbed_rows = _predict_rows(
            bundle=bundle,
            frame=semantic,
            route=route,
            scale=str(spec["parameter_key"]),
            dataset=device.key,
            override_hw_config=mutated,
        )
        _pert_summary, pert_combo = _evaluate_pipeline(perturbed_rows, split_root, device.key)
        by_op, by_phase, summary = _sensitivity_rows(
            baseline_rows=baseline_rows,
            perturbed_rows=perturbed_rows,
            baseline_combo=base_combo,
            perturbed_combo=pert_combo,
            device=device,
            parameter_key=str(spec["parameter_key"]),
            parameter_label=str(spec["parameter_label"]),
        )
        by_op_sensitivity.append(by_op)
        by_phase_sensitivity.append(by_phase)
        sensitivity_summary.append(summary)

    service_overall, service_by_op, service_by_phase = _service_share(baseline_rows, device)
    return {
        "split_root": split_root,
        "bundle_sha": str(bundle.get("model_sha256", "")),
        "baseline_rows": baseline_rows,
        "pipeline_summary": base_summary,
        "pipeline_combo": base_combo,
        "actual_share": _actual_time_share(baseline_rows, device),
        "service_overall": service_overall,
        "service_by_op": service_by_op,
        "service_by_phase": service_by_phase,
        "sensitivity_by_op": pd.concat(by_op_sensitivity, ignore_index=True) if by_op_sensitivity else pd.DataFrame(),
        "sensitivity_by_phase": pd.concat(by_phase_sensitivity, ignore_index=True) if by_phase_sensitivity else pd.DataFrame(),
        "sensitivity_summary": pd.DataFrame(sensitivity_summary),
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    devices = [
        DeviceSpec(
            key="Device_1",
            label="Device_1 / 智算平台",
            root=Path(args.device1_root),
            bundle=Path(args.device1_bundle),
            data_root=Path(args.device1_data_root),
            hw_config=Path(args.device1_hw_config),
        ),
        DeviceSpec(
            key="Device_2",
            label="Device_2 / 通算平台",
            root=Path(args.device2_root),
            bundle=Path(args.device2_bundle),
            data_root=Path(args.device2_data_root),
            hw_config=Path(args.device2_hw_config),
        ),
    ]

    results = [_process_device(device, args.route) for device in devices]

    actual_share = pd.concat([item["actual_share"] for item in results], ignore_index=True)
    service_share = pd.concat([item["service_overall"] for item in results], ignore_index=True)
    service_by_op = pd.concat([item["service_by_op"] for item in results], ignore_index=True)
    service_by_phase = pd.concat([item["service_by_phase"] for item in results], ignore_index=True)
    sensitivity_by_op = pd.concat([item["sensitivity_by_op"] for item in results], ignore_index=True)
    sensitivity_by_phase = pd.concat([item["sensitivity_by_phase"] for item in results], ignore_index=True)
    sensitivity_summary = pd.concat([item["sensitivity_summary"] for item in results], ignore_index=True)
    pipeline_summary = pd.concat([item["pipeline_summary"] for item in results], ignore_index=True)

    summary_rows = []
    actual_cat = (
        actual_share.groupby(["device", "op_category"], as_index=False)["actual_us_sum"].sum()
        .pivot(index="device", columns="op_category", values="actual_us_sum")
        .fillna(0.0)
    )
    service_indexed = service_share.set_index("device")
    for device in devices:
        actual_row = actual_cat.loc[device.key] if device.key in actual_cat.index else pd.Series(dtype=float)
        actual_total = float(actual_row.sum())
        service_row = service_indexed.loc[device.key] if device.key in service_indexed.index else pd.Series(dtype=float)
        memory_actual_share = _safe_share(float(actual_row.get("memory_related", 0.0)), actual_total)
        compute_actual_share = _safe_share(float(actual_row.get("compute_related", 0.0)), actual_total)
        memory_service_share = float(service_row.get("memory_service_us_share", np.nan))
        compute_service_share = float(service_row.get("compute_service_us_share", np.nan))
        top_sens = sensitivity_summary[sensitivity_summary["device"].eq(device.key)].sort_values(
            "e2e_improvement_pct", ascending=False
        )
        top_param = top_sens.iloc[0].to_dict() if not top_sens.empty else {}
        summary_rows.append(
            {
                "device": device.key,
                "device_label": device.label,
                "heldout_combo_count": int(
                    pipeline_summary.loc[pipeline_summary["dataset"].eq(device.key), "combo_count_full"].iloc[0]
                ),
                "memory_related_actual_time_share": memory_actual_share,
                "compute_related_actual_time_share": compute_actual_share,
                "memory_service_share": memory_service_share,
                "compute_service_share": compute_service_share,
                "memory_actual_gt_compute_actual": bool(memory_actual_share > compute_actual_share),
                "memory_service_gt_compute_service": bool(memory_service_share > compute_service_share),
                "top_compute_pipeline_parameter": top_param.get("parameter_label", ""),
                "top_compute_pipeline_e2e_improvement_pct": top_param.get("e2e_improvement_pct", np.nan),
            }
        )
    attribution_summary = pd.DataFrame(summary_rows)

    _write_csv(actual_share, out_dir / "operator_actual_time_share.csv")
    _write_csv(service_share, out_dir / "memory_vs_compute_service_share.csv")
    _write_csv(service_by_op, out_dir / "analytical_module_share_by_op_type.csv")
    _write_csv(service_by_phase, out_dir / "analytical_module_share_by_phase.csv")
    _write_csv(sensitivity_by_op, out_dir / "compute_parameter_sensitivity_by_op_type.csv")
    _write_csv(sensitivity_by_phase, out_dir / "compute_parameter_sensitivity_by_phase.csv")
    _write_csv(sensitivity_summary, out_dir / "compute_parameter_sensitivity_summary.csv")
    _write_csv(attribution_summary, out_dir / "e2e_compute_memory_attribution_summary.csv")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "route": args.route,
        "device1_root": args.device1_root,
        "device2_root": args.device2_root,
        "device1_bundle": args.device1_bundle,
        "device2_bundle": args.device2_bundle,
        "device1_bundle_sha256": results[0]["bundle_sha"],
        "device2_bundle_sha256": results[1]["bundle_sha"],
        "device1_split_root": str(results[0]["split_root"]),
        "device2_split_root": str(results[1]["split_root"]),
        "parameter_perturbations": PARAMETER_PERTURBATIONS,
    }
    (out_dir / "dlrm_compute_memory_attribution_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_actual_share(actual_share, out_dir)
    _plot_service_share(service_share, out_dir)
    _plot_operator_share(actual_share, out_dir)
    _plot_sensitivity_top_ops(sensitivity_by_op, out_dir)
    _plot_sensitivity_waterfall(sensitivity_summary, sensitivity_by_op, out_dir)
    _build_report(
        out_dir=out_dir,
        actual_share=actual_share,
        service_share=service_share,
        sensitivity_summary=sensitivity_summary,
        pipeline_summary=pipeline_summary,
        args=args,
    )

    memory_ok = bool(
        attribution_summary["memory_actual_gt_compute_actual"].all()
        and attribution_summary["memory_service_gt_compute_service"].all()
    )
    print(f"wrote attribution outputs to {out_dir}")
    print(f"memory dominance checks: {'PASS' if memory_ok else 'NEEDS_REVIEW'}")


if __name__ == "__main__":
    main()
