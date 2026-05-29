#!/usr/bin/env python3
"""Software workload generalization post-processing.

This script is intentionally read-only for the CPU mainline. It consumes
existing full-test outputs and optional unseen software workload outputs, then
generates compact evidence for workload generalization:

1. measured held-out / unseen E2E error CDF and axis summaries;
2. measured predicted-vs-actual trends by workload axes;
3. operator scale diagnostics reused from the error explainability analysis;
4. dense static-config trend inference from exported bundles.

Only the measured held-out / unseen sections are accuracy evidence. Dense
static-config trends have no measured labels and are reported as trend probes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrator.experimental.plot_error_explainability_slices import (  # noqa: E402
    OP_CLASS_ORDER,
    _bin_operator_rows,
    _load_operator_rows,
    _plot_operator_error_facets,
    _plot_operator_pred_actual_facets,
)
from calibrator.static_feature_mapping.core import (  # noqa: E402
    MODEL_INPUT_51,
    DlrmStaticConfig,
    build_static_feature_frame,
    build_static_op_shape_table,
)
from calibrator.static_feature_mapping.predict_from_config import (  # noqa: E402
    _load_model_bundle,
    _predict_with_bundle,
    _schedule_static_e2e,
)


DEFAULT_ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"
DEVICE_INFO = {
    "Device_1": {
        "display": "Device_1 / Intelligence platform",
        "short": "Device_1",
        "color": "#C46E2D",
        "marker": "o",
    },
    "Device_2": {
        "display": "Device_2 / General-purpose platform",
        "short": "Device_2",
        "color": "#3F72B8",
        "marker": "s",
    },
}
SCOPE_INFO = {
    "heldout_full_test": {"label": "held-out full test", "linestyle": "-"},
    "unseen_software_probe": {"label": "unseen software probe", "linestyle": "--"},
}
NUMERIC_AXES = ["batch_size", "num_indices_per_lookup", "inter_threads", "num_threads"]
AXIS_LABELS = {
    "batch_size": "Batch size",
    "num_indices_per_lookup": "NIP",
    "inter_threads": "Inter threads",
    "num_threads": "Num threads",
    "case_id": "Graph template",
    "mlp_scale": "MLP width scale",
}


@dataclass(frozen=True)
class MeasuredInput:
    device: str
    scope: str
    root: Path
    test_csv: Path


@dataclass(frozen=True)
class BundleInput:
    device: str
    bundle: Path


def _parse_csv_numbers(value: str, *, cast: type = int) -> list[Any]:
    items: list[Any] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        items.append(cast(text))
    if not items:
        raise ValueError(f"empty numeric list: {value!r}")
    return items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot software workload generalization evidence from existing outputs."
    )
    parser.add_argument("--device1-root", type=Path)
    parser.add_argument("--device1-test-csv", type=Path)
    parser.add_argument("--device1-bundle", type=Path)
    parser.add_argument("--device2-root", type=Path)
    parser.add_argument("--device2-test-csv", type=Path)
    parser.add_argument("--device2-bundle", type=Path)
    parser.add_argument("--device1-unseen-root", type=Path)
    parser.add_argument("--device1-unseen-test-csv", type=Path)
    parser.add_argument("--device2-unseen-root", type=Path)
    parser.add_argument("--device2-unseen-test-csv", type=Path)
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-bin-rows", type=int, default=10)
    parser.add_argument("--num-bins", type=int, default=5)

    parser.add_argument("--batch-values", default="512,768,1024,1280,1536,1792,2048,2560,3072,3584,4096")
    parser.add_argument("--nip-values", default="200,400,600,800,1000,1200,1400,1600,1800,2000,2400,2800,3200")
    parser.add_argument("--mlp-scale-values", default="0.50,0.75,1.00,1.25,1.50,2.00")
    parser.add_argument("--inter-thread-values", default="1,2,3,4,5,8")
    parser.add_argument("--num-thread-values", default="1,2,3,4,5,8")
    parser.add_argument("--fixed-batch-size", type=int, default=1024)
    parser.add_argument("--fixed-nip", type=int, default=1600)
    parser.add_argument("--fixed-inter-threads", type=int, default=1)
    parser.add_argument("--fixed-num-threads", type=int, default=1)
    parser.add_argument("--arch-embedding-size", default="51200")
    parser.add_argument("--arch-mlp-bot", default="800-800-400")
    parser.add_argument("--arch-mlp-top", default="1600-800-1")

    args = parser.parse_args()
    measured = _selected_measured_inputs(args)
    bundles = _selected_bundle_inputs(args)
    if not measured and not bundles:
        parser.error("provide at least one measured root/test pair or one model bundle")
    return args


def _selected_measured_inputs(args: argparse.Namespace) -> list[MeasuredInput]:
    inputs: list[MeasuredInput] = []
    specs = [
        ("Device_1", "heldout_full_test", args.device1_root, args.device1_test_csv),
        ("Device_2", "heldout_full_test", args.device2_root, args.device2_test_csv),
        ("Device_1", "unseen_software_probe", args.device1_unseen_root, args.device1_unseen_test_csv),
        ("Device_2", "unseen_software_probe", args.device2_unseen_root, args.device2_unseen_test_csv),
    ]
    for device, scope, root, test_csv in specs:
        if root is None and test_csv is None:
            continue
        if root is None or test_csv is None:
            raise ValueError(f"{device} {scope} requires both root and test CSV")
        inputs.append(MeasuredInput(device=device, scope=scope, root=root, test_csv=test_csv))
    return inputs


def _selected_bundle_inputs(args: argparse.Namespace) -> list[BundleInput]:
    inputs: list[BundleInput] = []
    if args.device1_bundle is not None:
        inputs.append(BundleInput("Device_1", args.device1_bundle))
    if args.device2_bundle is not None:
        inputs.append(BundleInput("Device_2", args.device2_bundle))
    return inputs


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _mode_int(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    if numeric.empty:
        return -1
    return int(numeric.mode().iloc[0])


def _load_combo_metrics(spec: MeasuredInput, route: str) -> pd.DataFrame:
    combo_path = spec.root / "e2e_pipeline_combo_metrics.csv"
    raw = _read_csv(combo_path)
    if "route" not in raw.columns:
        raise KeyError(f"{combo_path} missing route column")
    route_frame = raw[raw["route"].eq(route)].copy()
    if route_frame.empty:
        raise ValueError(f"{combo_path} has no rows for route={route}")

    full = route_frame[route_frame["coverage_ratio"].astype(float).eq(1.0)].copy()
    if full.empty:
        raise ValueError(f"{combo_path} has no full-coverage rows for route={route}")

    wanted = {"case_id", "source_name", "combo", "inter_threads", "num_threads"}
    test = _read_csv(spec.test_csv, usecols=lambda col: col in wanted)
    if "num_threads" not in test.columns:
        raise KeyError(f"{spec.test_csv} missing num_threads")
    group_cols = [col for col in ["case_id", "source_name", "combo", "inter_threads"] if col in test.columns and col in full.columns]
    thread_map = test.groupby(group_cols, as_index=False)["num_threads"].agg(_mode_int)
    full = full.merge(thread_map, on=group_cols, how="left", validate="many_to_one")
    if full["num_threads"].isna().any():
        # Fall back to a less specific join for older artifacts.
        fallback_cols = [col for col in ["case_id", "combo", "inter_threads"] if col in test.columns and col in full.columns]
        thread_map = test.groupby(fallback_cols, as_index=False)["num_threads"].agg(_mode_int)
        full = full.drop(columns=["num_threads"]).merge(thread_map, on=fallback_cols, how="left", validate="many_to_one")
    if full["num_threads"].isna().any():
        missing = int(full["num_threads"].isna().sum())
        raise ValueError(f"failed to map num_threads for {missing} rows from {spec.test_csv}")

    full["device"] = spec.device
    full["dataset_scope"] = spec.scope
    full["ape_pct"] = pd.to_numeric(full["ape"], errors="coerce") * 100.0
    full["signed_error_pct"] = (
        (pd.to_numeric(full["predicted_e2e_us"], errors="coerce") - pd.to_numeric(full["actual_e2e_us"], errors="coerce"))
        / pd.to_numeric(full["actual_e2e_us"], errors="coerce")
        * 100.0
    )
    full["num_threads"] = full["num_threads"].astype(int)
    full.attrs["partial_combo_rows"] = int(route_frame["coverage_ratio"].astype(float).lt(1.0).sum())
    return full


def _metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        ape = pd.to_numeric(sub["ape_pct"], errors="coerce").dropna()
        signed = pd.to_numeric(sub["signed_error_pct"], errors="coerce").dropna()
        row.update(
            {
                "rows": int(len(sub)),
                "mape_pct": float(ape.mean()) if len(ape) else float("nan"),
                "median_ape_pct": float(ape.median()) if len(ape) else float("nan"),
                "p90_ape_pct": float(ape.quantile(0.90)) if len(ape) else float("nan"),
                "p95_ape_pct": float(ape.quantile(0.95)) if len(ape) else float("nan"),
                "max_ape_pct": float(ape.max()) if len(ape) else float("nan"),
                "signed_error_mean_pct": float(signed.mean()) if len(signed) else float("nan"),
                "signed_error_median_pct": float(signed.median()) if len(signed) else float("nan"),
                "coverage_ratio_mean": float(pd.to_numeric(sub["coverage_ratio"], errors="coerce").mean())
                if "coverage_ratio" in sub.columns
                else float("nan"),
                "combo_count_partial": int(pd.to_numeric(sub.get("coverage_ratio", pd.Series([1.0] * len(sub))), errors="coerce").lt(1.0).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _write_measured_tables(combo: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = _metric_summary(combo, ["dataset_scope", "device"])
    summary.to_csv(out_dir / "software_generalization_e2e_summary.csv", index=False)

    axis_frames = []
    for axis in NUMERIC_AXES + ["case_id"]:
        if axis not in combo.columns:
            continue
        frame = _metric_summary(combo, ["dataset_scope", "device", axis])
        frame.insert(2, "axis", axis)
        frame = frame.rename(columns={axis: "axis_value"})
        axis_frames.append(frame)
    axis_summary = pd.concat(axis_frames, ignore_index=True) if axis_frames else pd.DataFrame()
    axis_summary.to_csv(out_dir / "software_generalization_mape_by_axis.csv", index=False)

    cdf_rows: list[dict[str, Any]] = []
    for (scope, device), sub in combo.groupby(["dataset_scope", "device"]):
        values = np.sort(pd.to_numeric(sub["ape"], errors="coerce").dropna().to_numpy(dtype=float))
        if len(values) == 0:
            continue
        cdf = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
        for x, y in zip(values, cdf):
            cdf_rows.append({"dataset_scope": scope, "device": device, "relative_error": float(x), "cdf": float(y)})
    cdf = pd.DataFrame(cdf_rows)
    cdf.to_csv(out_dir / "software_generalization_e2e_cdf_points.csv", index=False)

    trend_rows: list[dict[str, Any]] = []
    for axis in NUMERIC_AXES:
        if axis not in combo.columns:
            continue
        grouped = combo.groupby(["dataset_scope", "device", axis], dropna=False)
        for (scope, device, value), sub in grouped:
            trend_rows.append(
                {
                    "dataset_scope": scope,
                    "device": device,
                    "axis": axis,
                    "axis_value": value,
                    "rows": int(len(sub)),
                    "actual_e2e_us_mean": float(pd.to_numeric(sub["actual_e2e_us"], errors="coerce").mean()),
                    "predicted_e2e_us_mean": float(pd.to_numeric(sub["predicted_e2e_us"], errors="coerce").mean()),
                    "mape_pct": float(pd.to_numeric(sub["ape_pct"], errors="coerce").mean()),
                }
            )
    trend = pd.DataFrame(trend_rows)
    if not trend.empty:
        trend = trend.sort_values(["dataset_scope", "device", "axis", "axis_value"]).reset_index(drop=True)
        trend["actual_norm"] = np.nan
        trend["pred_norm"] = np.nan
        for key, sub_idx in trend.groupby(["dataset_scope", "device", "axis"]).groups.items():
            sub = trend.loc[list(sub_idx)].sort_values("axis_value")
            actual0 = float(sub["actual_e2e_us_mean"].iloc[0])
            pred0 = float(sub["predicted_e2e_us_mean"].iloc[0])
            if actual0 > 0:
                trend.loc[sub.index, "actual_norm"] = sub["actual_e2e_us_mean"] / actual0
            if pred0 > 0:
                trend.loc[sub.index, "pred_norm"] = sub["predicted_e2e_us_mean"] / pred0
    trend.to_csv(out_dir / "software_generalization_pred_actual_trends.csv", index=False)
    return summary, axis_summary, cdf, trend


def _plot_e2e_cdf(cdf: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    if cdf.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for (scope, device), sub in cdf.groupby(["dataset_scope", "device"]):
        info = DEVICE_INFO[device]
        scope_info = SCOPE_INFO.get(scope, {"label": scope, "linestyle": "-."})
        summ = summary[(summary["dataset_scope"].eq(scope)) & (summary["device"].eq(device))]
        mape = float(summ["mape_pct"].iloc[0]) if not summ.empty else float("nan")
        label = f"{info['short']} {scope_info['label']} (MAPE {mape:.2f}%)"
        ax.plot(
            sub["relative_error"],
            sub["cdf"] * 100.0,
            color=info["color"],
            linestyle=scope_info["linestyle"],
            linewidth=2.0,
            label=label,
        )
    ax.axvline(0.10, color="#888888", linestyle=":", linewidth=1.2)
    ax.axvline(0.15, color="#888888", linestyle=":", linewidth=1.2)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Relative error")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Software workload generalization E2E relative error CDF")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "software_generalization_e2e_cdf.png", dpi=300)
    fig.savefig(out_dir / "software_generalization_e2e_cdf.pdf")
    plt.close(fig)


def _plot_axis_summary(axis_summary: pd.DataFrame, out_dir: Path) -> None:
    if axis_summary.empty:
        return
    axes_to_plot = [axis for axis in NUMERIC_AXES if axis in set(axis_summary["axis"])]
    if not axes_to_plot:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), sharey=False)
    axes_arr = axes.ravel()
    present_devices: list[str] = []
    present_scopes: list[str] = []
    for ax, axis in zip(axes_arr, axes_to_plot):
        sub_axis = axis_summary[axis_summary["axis"].eq(axis)].copy()
        for (scope, device), sub in sub_axis.groupby(["dataset_scope", "device"]):
            if device not in present_devices:
                present_devices.append(device)
            if scope not in present_scopes:
                present_scopes.append(scope)
            sub = sub.sort_values("axis_value")
            info = DEVICE_INFO[device]
            scope_info = SCOPE_INFO.get(scope, {"label": scope, "linestyle": "-."})
            ax.plot(
                pd.to_numeric(sub["axis_value"], errors="coerce"),
                sub["mape_pct"],
                color=info["color"],
                linestyle=scope_info["linestyle"],
                marker=info["marker"],
                linewidth=1.9,
            )
            ax.plot(
                pd.to_numeric(sub["axis_value"], errors="coerce"),
                sub["p90_ape_pct"],
                color=info["color"],
                linestyle=":",
                marker="",
                linewidth=1.4,
                alpha=0.75,
            )
        ax.set_title(AXIS_LABELS.get(axis, axis))
        ax.set_xlabel(AXIS_LABELS.get(axis, axis))
        ax.set_ylabel("APE (%)")
        ax.grid(True, linestyle="--", alpha=0.25)
    for ax in axes_arr[len(axes_to_plot):]:
        ax.set_axis_off()
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    for device in present_devices:
        info = DEVICE_INFO[device]
        legend_handles.append(
            Line2D([0], [0], color=info["color"], marker=info["marker"], linewidth=2.0)
        )
        legend_labels.append(info["short"])
    legend_handles.extend(
        [
            Line2D([0], [0], color="#222222", linestyle="-", linewidth=2.0),
            Line2D([0], [0], color="#222222", linestyle=":", linewidth=2.0),
        ]
    )
    legend_labels.extend(["MAPE", "P90 APE"])
    if len(present_scopes) > 1:
        for scope in present_scopes:
            scope_info = SCOPE_INFO.get(scope, {"label": scope, "linestyle": "-."})
            legend_handles.append(
                Line2D([0], [0], color="#666666", linestyle=scope_info["linestyle"], linewidth=2.0)
            )
            legend_labels.append(scope_info["label"])
    fig.suptitle("E2E error by software workload axis", y=0.985, fontsize=15)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(6, len(legend_handles)),
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_dir / "software_generalization_mape_by_axis.png", dpi=300)
    fig.savefig(out_dir / "software_generalization_mape_by_axis.pdf")
    plt.close(fig)


def _plot_pred_actual_trends(trend: pd.DataFrame, out_dir: Path) -> None:
    if trend.empty:
        return
    axes_to_plot = [axis for axis in ["batch_size", "num_indices_per_lookup", "inter_threads", "num_threads"] if axis in set(trend["axis"])]
    if not axes_to_plot:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), sharey=False)
    axes_arr = axes.ravel()
    present_devices: list[str] = []
    present_scopes: list[str] = []
    for ax, axis in zip(axes_arr, axes_to_plot):
        sub_axis = trend[trend["axis"].eq(axis)].copy()
        for (scope, device), sub in sub_axis.groupby(["dataset_scope", "device"]):
            if device not in present_devices:
                present_devices.append(device)
            if scope not in present_scopes:
                present_scopes.append(scope)
            sub = sub.sort_values("axis_value")
            info = DEVICE_INFO[device]
            scope_info = SCOPE_INFO.get(scope, {"label": scope, "linestyle": "-."})
            x = pd.to_numeric(sub["axis_value"], errors="coerce")
            ax.plot(
                x,
                sub["actual_norm"],
                color=info["color"],
                linestyle=scope_info["linestyle"],
                marker=info["marker"],
                linewidth=2.0,
            )
            ax.plot(
                x,
                sub["pred_norm"],
                color=info["color"],
                linestyle=":",
                marker="",
                linewidth=2.0,
            )
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
        ax.set_title(AXIS_LABELS.get(axis, axis))
        ax.set_xlabel(AXIS_LABELS.get(axis, axis))
        ax.set_ylabel("Normalized E2E latency")
        ax.grid(True, linestyle="--", alpha=0.25)
    for ax in axes_arr[len(axes_to_plot):]:
        ax.set_axis_off()
    legend_handles = []
    legend_labels = []
    for device in present_devices:
        info = DEVICE_INFO[device]
        legend_handles.append(
            Line2D([0], [0], color=info["color"], marker=info["marker"], linewidth=2.0)
        )
        legend_labels.append(info["short"])
    legend_handles.extend(
        [
            Line2D([0], [0], color="#222222", linestyle="-", linewidth=2.0),
            Line2D([0], [0], color="#222222", linestyle=":", linewidth=2.0),
        ]
    )
    legend_labels.extend(["Actual", "Predicted"])
    if len(present_scopes) > 1:
        for scope in present_scopes:
            scope_info = SCOPE_INFO.get(scope, {"label": scope, "linestyle": "-."})
            legend_handles.append(
                Line2D([0], [0], color="#666666", linestyle=scope_info["linestyle"], linewidth=2.0)
            )
            legend_labels.append(scope_info["label"])
    fig.suptitle("Measured predicted-vs-actual E2E trends by software axis", y=0.985, fontsize=15)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(6, len(legend_handles)),
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_dir / "software_generalization_pred_actual_trends.png", dpi=300)
    fig.savefig(out_dir / "software_generalization_pred_actual_trends.pdf")
    plt.close(fig)


def _dash_ints(value: str, expected: int | None = None, repeat_to: int | None = None) -> list[int]:
    parts = [int(p.strip()) for p in str(value).split("-") if p.strip()]
    if repeat_to is not None and len(parts) == 1:
        parts = parts * repeat_to
    if expected is not None and len(parts) != expected:
        raise ValueError(f"expected {expected} values in {value!r}, got {len(parts)}")
    if any(v <= 0 for v in parts):
        raise ValueError(f"architecture values must be positive: {value!r}")
    return parts


def _dash(values: Iterable[int]) -> str:
    return "-".join(str(int(v)) for v in values)


def _scale_arch(value: str, scale: float, expected: int) -> str:
    base = _dash_ints(value, expected=expected)
    scaled = [max(1, int(round(v * float(scale)))) for v in base]
    return _dash(scaled)


def _base_config(args: argparse.Namespace) -> DlrmStaticConfig:
    emb = _dash(_dash_ints(args.arch_embedding_size, expected=8, repeat_to=8))
    return DlrmStaticConfig(
        batch_size=int(args.fixed_batch_size),
        num_indices_per_lookup=int(args.fixed_nip),
        arch_embedding_size=emb,
        arch_mlp_bot=_dash(_dash_ints(args.arch_mlp_bot, expected=3)),
        arch_mlp_top=_dash(_dash_ints(args.arch_mlp_top, expected=3)),
        inter_threads=int(args.fixed_inter_threads),
        num_threads=int(args.fixed_num_threads),
    )


def _config_for_axis(base: DlrmStaticConfig, axis: str, value: Any, args: argparse.Namespace) -> DlrmStaticConfig:
    if axis == "batch_size":
        return DlrmStaticConfig(
            batch_size=int(value),
            num_indices_per_lookup=base.num_indices_per_lookup,
            arch_embedding_size=base.arch_embedding_size,
            arch_mlp_bot=base.arch_mlp_bot,
            arch_mlp_top=base.arch_mlp_top,
            inter_threads=base.inter_threads,
            num_threads=base.num_threads,
        )
    if axis == "num_indices_per_lookup":
        return DlrmStaticConfig(
            batch_size=base.batch_size,
            num_indices_per_lookup=int(value),
            arch_embedding_size=base.arch_embedding_size,
            arch_mlp_bot=base.arch_mlp_bot,
            arch_mlp_top=base.arch_mlp_top,
            inter_threads=base.inter_threads,
            num_threads=base.num_threads,
        )
    if axis == "mlp_scale":
        return DlrmStaticConfig(
            batch_size=base.batch_size,
            num_indices_per_lookup=base.num_indices_per_lookup,
            arch_embedding_size=base.arch_embedding_size,
            arch_mlp_bot=_scale_arch(args.arch_mlp_bot, float(value), 3),
            arch_mlp_top=_scale_arch(args.arch_mlp_top, float(value), 3),
            inter_threads=base.inter_threads,
            num_threads=base.num_threads,
        )
    if axis == "inter_threads":
        return DlrmStaticConfig(
            batch_size=base.batch_size,
            num_indices_per_lookup=base.num_indices_per_lookup,
            arch_embedding_size=base.arch_embedding_size,
            arch_mlp_bot=base.arch_mlp_bot,
            arch_mlp_top=base.arch_mlp_top,
            inter_threads=int(value),
            num_threads=base.num_threads,
        )
    if axis == "num_threads":
        return DlrmStaticConfig(
            batch_size=base.batch_size,
            num_indices_per_lookup=base.num_indices_per_lookup,
            arch_embedding_size=base.arch_embedding_size,
            arch_mlp_bot=base.arch_mlp_bot,
            arch_mlp_top=base.arch_mlp_top,
            inter_threads=base.inter_threads,
            num_threads=int(value),
        )
    raise ValueError(axis)


def _predict_static_config(bundle: dict[str, Any], config: DlrmStaticConfig) -> dict[str, Any]:
    op_shapes = build_static_op_shape_table(config)
    model_input = build_static_feature_frame(config, graph_scope=MODEL_INPUT_51)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        prepared, pred = _predict_with_bundle(bundle, model_input)
    predictions = prepared.copy()
    predictions["pred_us"] = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    predictions["pred_total_us"] = predictions["pred_us"]
    summary, _ = _schedule_static_e2e(config=config, op_shapes=op_shapes, predictions=predictions)
    return {
        "predicted_e2e_us": float(summary["predicted_e2e_us"]),
        "predicted_full_graph_76_us": float(summary.get("predicted_full_graph_76_us", np.nan)),
        "model_input_rows": int(len(model_input)),
        "full_graph_rows": int(len(build_static_feature_frame(config, graph_scope="full_graph_76"))),
    }


def _run_static_dense_trends(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    bundles = _selected_bundle_inputs(args)
    if not bundles:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "static_config_dense_workload_trend_points.csv", index=False)
        return empty
    axis_values: dict[str, list[Any]] = {
        "batch_size": _parse_csv_numbers(args.batch_values, cast=int),
        "num_indices_per_lookup": _parse_csv_numbers(args.nip_values, cast=int),
        "mlp_scale": _parse_csv_numbers(args.mlp_scale_values, cast=float),
        "inter_threads": _parse_csv_numbers(args.inter_thread_values, cast=int),
        "num_threads": _parse_csv_numbers(args.num_thread_values, cast=int),
    }
    base = _base_config(args)
    rows: list[dict[str, Any]] = []
    for spec in bundles:
        bundle = _load_model_bundle(spec.bundle)
        route = str(bundle.get("route", ""))
        if route and route != args.route:
            raise ValueError(f"{spec.bundle} route={route!r} does not match --route={args.route!r}")
        for axis, values in axis_values.items():
            for value in values:
                config = _config_for_axis(base, axis, value, args)
                pred = _predict_static_config(bundle, config)
                rows.append(
                    {
                        "device": spec.device,
                        "axis": axis,
                        "axis_value": float(value),
                        "axis_value_label": str(value),
                        "route": route or args.route,
                        "model_bundle": str(spec.bundle),
                        "batch_size": int(config.batch_size),
                        "num_indices_per_lookup": int(config.num_indices_per_lookup),
                        "arch_embedding_size": str(config.arch_embedding_size),
                        "arch_mlp_bot": str(config.arch_mlp_bot),
                        "arch_mlp_top": str(config.arch_mlp_top),
                        "inter_threads": int(config.inter_threads),
                        "num_threads": int(config.num_threads),
                        **pred,
                    }
                )
    points = pd.DataFrame(rows)
    if not points.empty:
        points = points.sort_values(["device", "axis", "axis_value"]).reset_index(drop=True)
        points["normalized_e2e_latency"] = np.nan
        for (_device, _axis), sub_idx in points.groupby(["device", "axis"]).groups.items():
            sub = points.loc[list(sub_idx)].sort_values("axis_value")
            base_pred = float(sub["predicted_e2e_us"].iloc[0])
            if base_pred > 0:
                points.loc[sub.index, "normalized_e2e_latency"] = sub["predicted_e2e_us"] / base_pred
    points.to_csv(out_dir / "static_config_dense_workload_trend_points.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for (device, axis), sub in points.groupby(["device", "axis"]):
        pred = pd.to_numeric(sub["predicted_e2e_us"], errors="coerce")
        summary_rows.append(
            {
                "device": device,
                "axis": axis,
                "points": int(len(sub)),
                "min_predicted_e2e_us": float(pred.min()),
                "max_predicted_e2e_us": float(pred.max()),
                "change_pct": float((pred.max() - pred.min()) / max(pred.min(), 1e-12) * 100.0),
                "spearman_pred_vs_axis": float(pd.Series(sub["axis_value"]).rank().corr(pred.rank()))
                if len(sub) >= 3 and sub["axis_value"].nunique() >= 2 and pred.nunique() >= 2
                else float("nan"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "static_config_dense_workload_trend_summary.csv", index=False)
    return points


def _plot_static_dense_trends(points: pd.DataFrame, out_dir: Path) -> None:
    if points.empty:
        return
    axes_to_plot = ["batch_size", "num_indices_per_lookup", "mlp_scale", "inter_threads", "num_threads"]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.8), sharey=False)
    axes_arr = axes.ravel()
    present_devices: list[str] = []
    for ax, axis in zip(axes_arr, axes_to_plot):
        sub_axis = points[points["axis"].eq(axis)]
        for device, sub in sub_axis.groupby("device"):
            if device not in present_devices:
                present_devices.append(device)
            info = DEVICE_INFO[device]
            sub = sub.sort_values("axis_value")
            ax.plot(
                sub["axis_value"],
                sub["normalized_e2e_latency"],
                marker=info["marker"],
                linewidth=2.0,
                color=info["color"],
                label=info["short"],
            )
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
        ax.set_title(AXIS_LABELS.get(axis, axis))
        ax.set_xlabel(AXIS_LABELS.get(axis, axis))
        ax.set_ylabel("Normalized predicted E2E latency")
        ax.grid(True, linestyle="--", alpha=0.25)
    axes_arr[-1].set_axis_off()
    handles = []
    labels = []
    for device in present_devices:
        info = DEVICE_INFO[device]
        handles.append(Line2D([0], [0], color=info["color"], marker=info["marker"], linewidth=2.0))
        labels.append(info["short"])
    fig.suptitle("Dense static-config software workload trends (prediction only)", y=0.985, fontsize=15)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=len(handles),
            frameon=True,
            fontsize=9,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(out_dir / "static_config_dense_workload_trends.png", dpi=300)
    fig.savefig(out_dir / "static_config_dense_workload_trends.pdf")
    plt.close(fig)


def _write_operator_outputs(args: argparse.Namespace, measured_inputs: list[MeasuredInput], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout_inputs = [spec for spec in measured_inputs if spec.scope == "heldout_full_test"]
    if not heldout_inputs:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "operator_scale_generalization_summary.csv", index=False)
        empty.to_csv(out_dir / "operator_scale_generalization_correlation.csv", index=False)
        return empty, empty
    op_frames = []
    for spec in heldout_inputs:
        op_frames.append(_load_operator_rows(spec.root, spec.test_csv, spec.device, args.route))
    op_df = pd.concat(op_frames, ignore_index=True)
    bin_summary, corr = _bin_operator_rows(op_df, args.num_bins, args.min_bin_rows)
    bin_summary.to_csv(out_dir / "operator_scale_generalization_summary.csv", index=False)
    corr.to_csv(out_dir / "operator_scale_generalization_correlation.csv", index=False)
    _plot_operator_error_facets(op_df, bin_summary, out_dir)
    _plot_operator_pred_actual_facets(bin_summary, out_dir)
    for src, dst in [
        ("operator_error_vs_scale_facets.png", "operator_scale_generalization_facets.png"),
        ("operator_error_vs_scale_facets.pdf", "operator_scale_generalization_facets.pdf"),
        ("operator_pred_actual_vs_scale_facets.png", "operator_pred_actual_scale_generalization_facets.png"),
        ("operator_pred_actual_vs_scale_facets.pdf", "operator_pred_actual_scale_generalization_facets.pdf"),
    ]:
        src_path = out_dir / src
        if src_path.exists():
            (out_dir / dst).write_bytes(src_path.read_bytes())
    return bin_summary, corr


def _markdown_table(df: pd.DataFrame, columns: Iterable[str], max_rows: int = 40) -> str:
    cols = list(columns)
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.head(max_rows).iterrows():
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(
    args: argparse.Namespace,
    measured_inputs: list[MeasuredInput],
    bundle_inputs: list[BundleInput],
    summary: pd.DataFrame,
    axis_summary: pd.DataFrame,
    dense_points: pd.DataFrame,
    dense_summary: pd.DataFrame,
    op_corr: pd.DataFrame,
    out_dir: Path,
    elapsed_s: float,
) -> None:
    input_lines = []
    for spec in measured_inputs:
        input_lines.append(f"- {spec.device} `{spec.scope}` root: `{spec.root}`")
        input_lines.append(f"- {spec.device} `{spec.scope}` test CSV: `{spec.test_csv}`")
    for spec in bundle_inputs:
        input_lines.append(f"- {spec.device} bundle: `{spec.bundle}`")
    unseen_present = any(spec.scope == "unseen_software_probe" for spec in measured_inputs)
    if not unseen_present:
        unseen_note = (
            "No unseen software probe root was provided. The measured accuracy evidence in this run is "
            "therefore held-out full test only; dense static-config trends are prediction-only diagnostics."
        )
    else:
        unseen_note = (
            "Unseen software probe roots were provided. Their E2E CDF and axis summaries are measured "
            "accuracy evidence because they include actual E2E labels."
        )

    key_axis = axis_summary[axis_summary["axis"].isin(NUMERIC_AXES)] if not axis_summary.empty else pd.DataFrame()
    dense_cols = ["device", "axis", "points", "change_pct", "spearman_pred_vs_axis"]
    corr_cols = [
        "device",
        "operator_class",
        "scale_feature",
        "rows",
        "status",
        "spearman_actual_vs_scale",
        "spearman_pred_vs_scale",
        "spearman_ape_vs_scale",
    ]
    body = f"""# Software Workload Generalization Report

## Inputs

{chr(10).join(input_lines) if input_lines else "_No measured inputs or bundles listed._"}

- Route: `{args.route}`
- Elapsed seconds: `{elapsed_s:.2f}`

## Evidence boundary

{unseen_note}

The dense static-config sweep does not use measured labels, so it is not used as
an accuracy metric. It only checks whether the trained bundle responds
monotonically or piecewise-monotonically to software workload scale changes.

## E2E measured accuracy summary

{_markdown_table(summary, ["dataset_scope", "device", "rows", "mape_pct", "median_ape_pct", "p90_ape_pct", "coverage_ratio_mean", "combo_count_partial"])}

## E2E axis slices

{_markdown_table(key_axis, ["dataset_scope", "device", "axis", "axis_value", "rows", "mape_pct", "p90_ape_pct"], max_rows=80)}

## Dense static-config trend summary

{_markdown_table(dense_summary, dense_cols)}

## Operator scale correlation

Positive `spearman_actual_vs_scale` and `spearman_pred_vs_scale` mean measured
and predicted operator time both grow with the dominant scale feature. A small
or non-monotonic `spearman_ape_vs_scale` means relative error does not keep
worsening as scale increases.

{_markdown_table(op_corr, corr_cols)}

## Figures

- `software_generalization_e2e_cdf.png/pdf`
- `software_generalization_mape_by_axis.png/pdf`
- `software_generalization_pred_actual_trends.png/pdf`
- `operator_scale_generalization_facets.png/pdf`
- `operator_pred_actual_scale_generalization_facets.png/pdf`
- `static_config_dense_workload_trends.png/pdf`
"""
    (out_dir / "SOFTWARE_WORKLOAD_GENERALIZATION_REPORT.md").write_text(body, encoding="utf-8")


def main() -> None:
    started = time.time()
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    measured_inputs = _selected_measured_inputs(args)
    bundle_inputs = _selected_bundle_inputs(args)

    combo = pd.DataFrame()
    summary = pd.DataFrame()
    axis_summary = pd.DataFrame()
    cdf = pd.DataFrame()
    measured_trends = pd.DataFrame()
    if measured_inputs:
        combo = pd.concat([_load_combo_metrics(spec, args.route) for spec in measured_inputs], ignore_index=True)
        combo.to_csv(args.out_dir / "software_generalization_e2e_error_rows.csv", index=False)
        summary, axis_summary, cdf, measured_trends = _write_measured_tables(combo, args.out_dir)
        _plot_e2e_cdf(cdf, summary, args.out_dir)
        _plot_axis_summary(axis_summary, args.out_dir)
        _plot_pred_actual_trends(measured_trends, args.out_dir)

    bin_summary, op_corr = _write_operator_outputs(args, measured_inputs, args.out_dir)
    dense_points = _run_static_dense_trends(args, args.out_dir)
    dense_summary_path = args.out_dir / "static_config_dense_workload_trend_summary.csv"
    dense_summary = _read_csv(dense_summary_path) if dense_summary_path.exists() else pd.DataFrame()
    _plot_static_dense_trends(dense_points, args.out_dir)

    manifest = {
        "route": args.route,
        "measured_inputs": [
            {
                "device": spec.device,
                "scope": spec.scope,
                "root": str(spec.root),
                "test_csv": str(spec.test_csv),
            }
            for spec in measured_inputs
        ],
        "bundle_inputs": [{"device": spec.device, "bundle": str(spec.bundle)} for spec in bundle_inputs],
        "dense_static_config": {
            "batch_values": _parse_csv_numbers(args.batch_values, cast=int),
            "nip_values": _parse_csv_numbers(args.nip_values, cast=int),
            "mlp_scale_values": _parse_csv_numbers(args.mlp_scale_values, cast=float),
            "inter_thread_values": _parse_csv_numbers(args.inter_thread_values, cast=int),
            "num_thread_values": _parse_csv_numbers(args.num_thread_values, cast=int),
            "fixed_batch_size": int(args.fixed_batch_size),
            "fixed_nip": int(args.fixed_nip),
            "fixed_inter_threads": int(args.fixed_inter_threads),
            "fixed_num_threads": int(args.fixed_num_threads),
            "arch_embedding_size": str(args.arch_embedding_size),
            "arch_mlp_bot": str(args.arch_mlp_bot),
            "arch_mlp_top": str(args.arch_mlp_top),
        },
        "elapsed_s": time.time() - started,
        "measured_rows": int(len(combo)),
        "static_dense_rows": int(len(dense_points)),
    }
    (args.out_dir / "software_generalization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        args,
        measured_inputs,
        bundle_inputs,
        summary,
        axis_summary,
        dense_points,
        dense_summary,
        op_corr,
        args.out_dir,
        elapsed_s=time.time() - started,
    )


if __name__ == "__main__":
    main()
