#!/usr/bin/env python3
"""Build E2E ablation CDFs from existing CPU model outputs.

This is an independent post-processing experiment.  It compares raw analytical
prior, RF/scale, direct operator summation, and E2E pipeline scheduling without
changing the CPU mainline training or prediction outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RF_ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"
DEFAULT_PRIOR_ROUTE = "whitebox_family_prior_only_v2"

KEY_COLUMNS = ["case_id", "source_name", "combo"]

ABLATION_ORDER = [
    "raw_direct_sum",
    "rf_scale_direct_sum",
    "rf_scale_pipeline",
    "raw_pipeline",
]

ABLATION_LABELS = {
    "raw_direct_sum": "Raw analytical + direct sum",
    "rf_scale_direct_sum": "Analytical + RF/scale + direct sum",
    "rf_scale_pipeline": "Analytical + RF/scale + pipeline",
    "raw_pipeline": "Raw analytical + pipeline",
}

DEVICE_SPECS = {
    "Device_1": "Device_1 / 智算平台",
    "Device_2": "Device_2 / 通算平台",
}

PLOT_DEVICE_LABELS = {
    "Device_1": "Device_1",
    "Device_2": "Device_2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device1-raw-root", required=True)
    parser.add_argument("--device2-raw-root", required=True)
    parser.add_argument("--device1-full-root", required=True)
    parser.add_argument("--device2-full-root", required=True)
    parser.add_argument("--rf-route", default=DEFAULT_RF_ROUTE)
    parser.add_argument("--prior-route", default=DEFAULT_PRIOR_ROUTE)
    parser.add_argument("--threshold", default="0.10,0.15")
    parser.add_argument("--module-improvement-threshold", type=float, default=0.10)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("--threshold must contain at least one value")
    return sorted(dict.fromkeys(values))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _filter_route(frame: pd.DataFrame, *, route: str, path: Path) -> pd.DataFrame:
    if "route" not in frame.columns:
        raise KeyError(f"{path} is missing required column: route")
    result = frame[frame["route"].astype(str).eq(str(route))].copy()
    if result.empty:
        raise ValueError(f"{path} has no rows for route={route!r}")
    return result


def _pipeline_combo_metrics(root: Path, *, route: str) -> pd.DataFrame:
    path = root / "e2e_pipeline_combo_metrics.csv"
    frame = _filter_route(_read_csv(path), route=route, path=path)
    required = set(KEY_COLUMNS + ["predicted_e2e_us", "actual_e2e_us", "coverage_ratio"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    frame["predicted_e2e_us"] = pd.to_numeric(frame["predicted_e2e_us"], errors="coerce")
    frame["actual_e2e_us"] = pd.to_numeric(frame["actual_e2e_us"], errors="coerce")
    frame["coverage_ratio"] = pd.to_numeric(frame["coverage_ratio"], errors="coerce")
    frame = frame[frame["coverage_ratio"].fillna(0.0).ge(1.0)].copy()
    frame = frame[np.isfinite(frame["predicted_e2e_us"]) & np.isfinite(frame["actual_e2e_us"])].copy()
    if frame.empty:
        raise ValueError(f"{path} has no full finite E2E combo rows for route={route!r}")
    return frame


def _prediction_rows(root: Path, *, route: str) -> pd.DataFrame:
    preferred = root / "pipeline_prediction_error_rows.csv"
    path = preferred if preferred.exists() else root / "prediction_error_rows.csv"
    frame = _filter_route(_read_csv(path), route=route, path=path)
    required = set(KEY_COLUMNS + ["pred_us"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    frame["pred_us"] = pd.to_numeric(frame["pred_us"], errors="coerce").fillna(0.0)
    return frame


def _direct_sum_combo_metrics(
    rows: pd.DataFrame,
    actual_reference: pd.DataFrame,
    *,
    device_id: str,
    ablation_id: str,
) -> pd.DataFrame:
    grouped = (
        rows.groupby(KEY_COLUMNS, dropna=False)
        .agg(
            predicted_e2e_us=("pred_us", "sum"),
            operator_row_count=("pred_us", "size"),
        )
        .reset_index()
    )
    meta_cols = [
        *KEY_COLUMNS,
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "actual_e2e_us",
        "expected_count",
        "observed_count",
        "coverage_ratio",
    ]
    available_meta = [column for column in meta_cols if column in actual_reference.columns]
    ref = actual_reference[available_meta].drop_duplicates(subset=KEY_COLUMNS)
    merged = grouped.merge(ref, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    return _finalize_combo_metrics(merged, device_id=device_id, ablation_id=ablation_id, metric_scope="direct_sum")


def _pipeline_ablation_metrics(
    combo: pd.DataFrame,
    *,
    device_id: str,
    ablation_id: str,
) -> pd.DataFrame:
    keep_cols = [
        column
        for column in [
            *KEY_COLUMNS,
            "batch_size",
            "num_indices_per_lookup",
            "inter_threads",
            "expected_count",
            "observed_count",
            "coverage_ratio",
            "predicted_e2e_us",
            "actual_e2e_us",
            "bottom_end_us",
            "branch_pool_end_us",
            "tail_barrier_us",
        ]
        if column in combo.columns
    ]
    return _finalize_combo_metrics(combo[keep_cols].copy(), device_id=device_id, ablation_id=ablation_id, metric_scope="pipeline")


def _finalize_combo_metrics(
    frame: pd.DataFrame,
    *,
    device_id: str,
    ablation_id: str,
    metric_scope: str,
) -> pd.DataFrame:
    work = frame.copy()
    work["predicted_e2e_us"] = pd.to_numeric(work["predicted_e2e_us"], errors="coerce")
    work["actual_e2e_us"] = pd.to_numeric(work["actual_e2e_us"], errors="coerce")
    work = work[np.isfinite(work["predicted_e2e_us"]) & np.isfinite(work["actual_e2e_us"])].copy()
    work = work[work["actual_e2e_us"].gt(0.0)].copy()
    work["abs_error_us"] = (work["predicted_e2e_us"] - work["actual_e2e_us"]).abs()
    work["ape"] = work["abs_error_us"] / work["actual_e2e_us"]
    work["device_id"] = device_id
    work["device_label"] = DEVICE_SPECS[device_id]
    work["ablation_id"] = ablation_id
    work["ablation_label"] = ABLATION_LABELS[ablation_id]
    work["metric_scope"] = metric_scope
    work["combo_key"] = work[KEY_COLUMNS].astype(str).agg("::".join, axis=1)
    return work


def _singleop_diagnostic(rows: pd.DataFrame, *, device_id: str, stage_id: str, stage_label: str) -> pd.DataFrame:
    work = rows.copy()
    work["actual_us"] = pd.to_numeric(work["actual_us"], errors="coerce").fillna(0.0)
    work["pred_us"] = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0)
    work["ape"] = (work["pred_us"] - work["actual_us"]).abs() / work["actual_us"].clip(lower=1e-6)
    combo_counts = work.groupby(["source_name", "combo"], dropna=False).size()
    return pd.DataFrame(
        [
            {
                "device_id": device_id,
                "device_label": DEVICE_SPECS[device_id],
                "stage_id": stage_id,
                "stage_label": stage_label,
                "row_count": int(len(work)),
                "combo_count": int(len(combo_counts)),
                "min_rows_per_combo": int(combo_counts.min()) if len(combo_counts) else 0,
                "max_rows_per_combo": int(combo_counts.max()) if len(combo_counts) else 0,
                "singleop_mape": float(work["ape"].mean()),
                "singleop_p90_ape": float(work["ape"].quantile(0.90)),
                "singleop_median_ape": float(work["ape"].quantile(0.50)),
            }
        ]
    )


def _load_device_ablation(
    *,
    device_id: str,
    raw_root: Path,
    full_root: Path,
    prior_route: str,
    rf_route: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    raw_pipeline = _pipeline_combo_metrics(raw_root, route=prior_route)
    full_pipeline = _pipeline_combo_metrics(full_root, route=rf_route)
    raw_rows = _prediction_rows(raw_root, route=prior_route)
    rf_rows = _prediction_rows(full_root, route=rf_route)
    singleop_diag = pd.concat(
        [
            _singleop_diagnostic(raw_rows, device_id=device_id, stage_id="raw_analytical", stage_label="Raw analytical"),
            _singleop_diagnostic(rf_rows, device_id=device_id, stage_id="rf_scale", stage_label="RF/scale"),
        ],
        ignore_index=True,
    )

    actual_reference = full_pipeline.copy()
    raw_direct = _direct_sum_combo_metrics(raw_rows, actual_reference, device_id=device_id, ablation_id="raw_direct_sum")
    rf_direct = _direct_sum_combo_metrics(rf_rows, actual_reference, device_id=device_id, ablation_id="rf_scale_direct_sum")
    rf_pipe = _pipeline_ablation_metrics(full_pipeline, device_id=device_id, ablation_id="rf_scale_pipeline")
    raw_pipe = _pipeline_ablation_metrics(raw_pipeline, device_id=device_id, ablation_id="raw_pipeline")

    combo_sets = {
        "raw_direct_sum": set(raw_direct["combo_key"]),
        "rf_scale_direct_sum": set(rf_direct["combo_key"]),
        "rf_scale_pipeline": set(rf_pipe["combo_key"]),
        "raw_pipeline": set(raw_pipe["combo_key"]),
    }
    common_keys = set.intersection(*combo_sets.values())
    if not common_keys:
        raise ValueError(f"{device_id} has no common combo keys across all ablations")

    frames = []
    for frame in [raw_direct, rf_direct, rf_pipe, raw_pipe]:
        frames.append(frame[frame["combo_key"].isin(common_keys)].copy())
    combined = pd.concat(frames, ignore_index=True)
    audit = {
        "device_id": device_id,
        "raw_root": str(raw_root),
        "full_root": str(full_root),
        "common_combo_count": len(common_keys),
        "combo_counts_before_intersection": {key: len(value) for key, value in combo_sets.items()},
        "dropped_combo_counts": {key: len(value - common_keys) for key, value in combo_sets.items()},
    }
    return combined, audit, singleop_diag


def _cdf_points(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (device_id, ablation_id), group in frame.groupby(["device_id", "ablation_id"], sort=False):
        values = np.sort(pd.to_numeric(group["ape"], errors="coerce").dropna().to_numpy(dtype=float))
        if len(values) == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "device_id": device_id,
                    "device_label": group["device_label"].iloc[0],
                    "ablation_id": ablation_id,
                    "ablation_label": group["ablation_label"].iloc[0],
                    "relative_error": values,
                    "cdf_percent": np.arange(1, len(values) + 1, dtype=float) / float(len(values)) * 100.0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _summary(frame: pd.DataFrame, *, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (device_id, ablation_id), group in frame.groupby(["device_id", "ablation_id"], sort=False):
        values = pd.to_numeric(group["ape"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        row: dict[str, Any] = {
            "device_id": device_id,
            "device_label": group["device_label"].iloc[0],
            "ablation_id": ablation_id,
            "ablation_label": group["ablation_label"].iloc[0],
            "combo_count": int(len(values)),
            "mape": float(np.mean(values)),
            "median_ape": float(np.percentile(values, 50)),
            "p90_ape": float(np.percentile(values, 90)),
            "p95_ape": float(np.percentile(values, 95)),
            "p99_ape": float(np.percentile(values, 99)),
            "max_ape": float(np.max(values)),
        }
        for threshold in thresholds:
            row[f"cdf_at_{threshold:.2f}_percent"] = float(np.mean(values <= threshold) * 100.0)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["ablation_order"] = result["ablation_id"].map({name: i for i, name in enumerate(ABLATION_ORDER)})
        result = result.sort_values(["device_id", "ablation_order"]).drop(columns=["ablation_order"]).reset_index(drop=True)
    return result


def _effect_summary(summary: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    pairs = [
        ("rf_scale_gain_direct_sum", "raw_direct_sum", "rf_scale_direct_sum"),
        ("pipeline_gain_after_rf_scale", "rf_scale_direct_sum", "rf_scale_pipeline"),
        ("rf_scale_gain_pipeline_context_diagnostic", "raw_pipeline", "rf_scale_pipeline"),
        ("pipeline_gain_raw_diagnostic", "raw_direct_sum", "raw_pipeline"),
    ]
    rows: list[dict[str, Any]] = []
    for device_id, device_summary in summary.groupby("device_id", sort=False):
        lookup = device_summary.set_index("ablation_id")
        for effect_id, before_id, after_id in pairs:
            if before_id not in lookup.index or after_id not in lookup.index:
                continue
            before = lookup.loc[before_id]
            after = lookup.loc[after_id]
            before_mape = float(before["mape"])
            after_mape = float(after["mape"])
            before_p90 = float(before["p90_ape"])
            after_p90 = float(after["p90_ape"])
            mape_improvement = (before_mape - after_mape) / before_mape if before_mape > 0 else np.nan
            p90_improvement = (before_p90 - after_p90) / before_p90 if before_p90 > 0 else np.nan
            pass_gate = bool(
                (np.isfinite(mape_improvement) and mape_improvement >= threshold)
                or (np.isfinite(p90_improvement) and p90_improvement >= threshold)
            )
            rows.append(
                {
                    "device_id": device_id,
                    "device_label": lookup.loc[before_id]["device_label"],
                    "effect_id": effect_id,
                    "before_ablation": before_id,
                    "after_ablation": after_id,
                    "before_mape": before_mape,
                    "after_mape": after_mape,
                    "mape_relative_improvement": float(mape_improvement),
                    "before_p90_ape": before_p90,
                    "after_p90_ape": after_p90,
                    "p90_relative_improvement": float(p90_improvement),
                    "threshold": float(threshold),
                    "pass_gate": pass_gate,
                }
            )
    return pd.DataFrame(rows)


def _review_status(effect: pd.DataFrame, device_audits: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    required_effects = {"rf_scale_gain_direct_sum", "pipeline_gain_after_rf_scale"}
    failures = []
    for device_id in sorted(effect["device_id"].dropna().unique()):
        sub = effect[effect["device_id"].eq(device_id)]
        for effect_id in required_effects:
            row = sub[sub["effect_id"].eq(effect_id)]
            if row.empty:
                failures.append({"device_id": device_id, "effect_id": effect_id, "reason": "missing_effect_row"})
                continue
            if not bool(row.iloc[0]["pass_gate"]):
                failures.append(
                    {
                        "device_id": device_id,
                        "effect_id": effect_id,
                        "reason": "improvement_below_threshold",
                        "mape_relative_improvement": float(row.iloc[0]["mape_relative_improvement"]),
                        "p90_relative_improvement": float(row.iloc[0]["p90_relative_improvement"]),
                    }
                )
    for audit in device_audits:
        if any(count != 0 for count in audit["dropped_combo_counts"].values()):
            failures.append(
                {
                    "device_id": audit["device_id"],
                    "reason": "combo_intersection_dropped_rows",
                    "dropped_combo_counts": audit["dropped_combo_counts"],
                }
            )
    return {
        "status": "PASS" if not failures else "NEEDS_REVIEW",
        "module_improvement_threshold": float(threshold),
        "required_effects": sorted(required_effects),
        "failures": failures,
        "device_audits": device_audits,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _plot_device(points: pd.DataFrame, summary: pd.DataFrame, *, device_id: str, thresholds: list[float], out_dir: Path) -> None:
    colors = {
        "raw_direct_sum": "#9E9E9E",
        "rf_scale_direct_sum": "#E45756",
        "rf_scale_pipeline": "#2F6DB3",
        "raw_pipeline": "#72B7B2",
    }
    linestyles = {
        "raw_direct_sum": "--",
        "rf_scale_direct_sum": "-.",
        "rf_scale_pipeline": "-",
        "raw_pipeline": ":",
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for ablation_id in ABLATION_ORDER:
        part = points[points["device_id"].eq(device_id) & points["ablation_id"].eq(ablation_id)]
        if part.empty:
            continue
        label = ABLATION_LABELS[ablation_id]
        ax.plot(
            part["relative_error"],
            part["cdf_percent"],
            label=label,
            color=colors[ablation_id],
            linestyle=linestyles[ablation_id],
            linewidth=2.2,
        )
    for threshold in thresholds:
        ax.axvline(threshold, color="#666666", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(threshold, 2.0, f"{threshold:.2f}", rotation=90, ha="right", va="bottom", color="#555555", fontsize=9)
    device_label = PLOT_DEVICE_LABELS[device_id]
    ax.set_title(f"{device_label} E2E ablation CDF")
    ax.set_xlabel("Relative error")
    ax.set_ylabel("CDF (%)")
    p99 = summary[summary["device_id"].eq(device_id)]["p99_ape"].max()
    ax.set_xlim(0.0, max(0.30, float(p99) + 0.03, max(thresholds) + 0.05))
    ax.set_ylim(0.0, 100.5)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    stem = f"e2e_ablation_cdf_{device_id.lower().replace('_', '')}"
    fig.savefig(out_dir / f"{stem}.png", dpi=320)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _spread_label_positions(values: list[tuple[str, float]], *, min_gap: float = 4.2) -> dict[str, float]:
    """Return readable label y positions while preserving actual CDF values in text."""
    if not values:
        return {}
    ordered = sorted(values, key=lambda item: (item[1], ABLATION_ORDER.index(item[0])))
    placed: list[list[Any]] = []
    lower = 3.5
    upper = 97.0
    for ablation_id, y_value in ordered:
        label_y = min(max(float(y_value), lower), upper)
        if placed:
            label_y = max(label_y, float(placed[-1][1]) + min_gap)
        placed.append([ablation_id, label_y])
    overflow = float(placed[-1][1]) - upper
    if overflow > 0:
        for item in placed:
            item[1] = float(item[1]) - overflow
        for index in range(len(placed) - 2, -1, -1):
            placed[index][1] = min(float(placed[index][1]), float(placed[index + 1][1]) - min_gap)
    underflow = lower - float(placed[0][1])
    if underflow > 0:
        for item in placed:
            item[1] = float(item[1]) + underflow
    return {str(ablation_id): float(label_y) for ablation_id, label_y in placed}


def _plot_combined(points: pd.DataFrame, summary: pd.DataFrame, *, thresholds: list[float], out_dir: Path) -> None:
    colors = {
        "raw_direct_sum": "#9E9E9E",
        "rf_scale_direct_sum": "#E45756",
        "rf_scale_pipeline": "#2F6DB3",
        "raw_pipeline": "#72B7B2",
    }
    linestyles = {
        "raw_direct_sum": "--",
        "rf_scale_direct_sum": "-.",
        "rf_scale_pipeline": "-",
        "raw_pipeline": ":",
    }
    threshold_markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
    for ax, device_id in zip(axes, ["Device_1", "Device_2"], strict=False):
        device_points = points[points["device_id"].eq(device_id)]
        device_summary = summary[summary["device_id"].eq(device_id)].set_index("ablation_id")
        for ablation_id in ABLATION_ORDER:
            part = device_points[device_points["ablation_id"].eq(ablation_id)]
            if part.empty:
                continue
            ax.plot(
                part["relative_error"],
                part["cdf_percent"],
                linewidth=2.6,
                label=ABLATION_LABELS[ablation_id],
                color=colors[ablation_id],
                linestyle=linestyles[ablation_id],
            )
        for threshold_index, threshold in enumerate(thresholds):
            ax.axvline(
                threshold,
                color="#777777",
                linestyle="--",
                linewidth=1.1,
                alpha=0.75 if threshold_index == len(thresholds) - 1 else 0.45,
            )
            ax.text(
                threshold,
                2.5,
                f"{threshold:.2f}",
                rotation=90,
                ha="right",
                va="bottom",
                color="#666666",
                fontsize=12,
            )
            label_x = 0.010 if threshold_index == 0 else 0.060
            marker = threshold_markers[threshold_index % len(threshold_markers)]
            y_values: list[tuple[str, float]] = []
            key = f"cdf_at_{threshold:.2f}_percent"
            for ablation_id in ABLATION_ORDER:
                if ablation_id not in device_summary.index:
                    continue
                if key not in device_summary.columns:
                    continue
                y_values.append((ablation_id, float(device_summary.loc[ablation_id, key])))
            label_positions = _spread_label_positions(y_values)
            for ablation_id, y_value in y_values:
                color = colors[ablation_id]
                ax.hlines(
                    y_value,
                    xmin=0.0,
                    xmax=threshold,
                    color=color,
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.32,
                    zorder=1,
                )
                ax.scatter(
                    [threshold],
                    [y_value],
                    s=44,
                    marker=marker,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=6,
                )
                label_y = label_positions.get(ablation_id, y_value)
                if abs(label_y - y_value) > 0.05:
                    ax.plot(
                        [label_x + 0.006, label_x + 0.006],
                        [y_value, label_y],
                        color=color,
                        linestyle="-",
                        linewidth=0.7,
                        alpha=0.45,
                        zorder=5,
                    )
                ax.text(
                    label_x,
                    label_y,
                    f"{y_value:.1f}%",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=10,
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": color, "alpha": 0.86, "lw": 0.7},
                    zorder=7,
                )
        ax.set_title(PLOT_DEVICE_LABELS[device_id], fontsize=18, pad=10)
        ax.set_xlabel("Relative error", fontsize=16)
        ax.set_xlim(0.0, 0.50)
        ax.set_ylim(0.0, 100.5)
        ax.set_xticks(np.arange(0.0, 0.51, 0.10))
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.28, linestyle="-")
    axes[0].set_ylabel("CDF (%)", fontsize=16)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=13, frameon=True)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(out_dir / "e2e_ablation_cdf_combined.png", dpi=320)
    fig.savefig(out_dir / "e2e_ablation_cdf_combined.pdf")
    plt.close(fig)


def _plot_all(points: pd.DataFrame, summary: pd.DataFrame, *, thresholds: list[float], out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "legend.fontsize": 12,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    _plot_device(points, summary, device_id="Device_1", thresholds=thresholds, out_dir=out_dir)
    _plot_device(points, summary, device_id="Device_2", thresholds=thresholds, out_dir=out_dir)
    _plot_combined(points, summary, thresholds=thresholds, out_dir=out_dir)


def _write_report(
    summary: pd.DataFrame,
    effect: pd.DataFrame,
    singleop_diag: pd.DataFrame,
    status: dict[str, Any],
    *,
    thresholds: list[float],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    threshold_cols = [f"cdf_at_{threshold:.2f}_percent" for threshold in thresholds]
    lines = [
        "# E2E Ablation CDF Report",
        "",
        f"- review status: `{status['status']}`",
        f"- module improvement threshold: `{float(args.module_improvement_threshold):.1%}`",
        f"- RF route: `{args.rf_route}`",
        f"- raw prior route: `{args.prior_route}`",
        "",
        "## Ablation definitions",
        "",
        "| ID | Meaning |",
        "| --- | --- |",
    ]
    for ablation_id in ABLATION_ORDER:
        lines.append(f"| `{ablation_id}` | {ABLATION_LABELS[ablation_id]} |")
    lines.extend(
        [
            "",
            "Raw analytical means the prior-only run with `--op-thread-scale-mode off`. Direct sum means combo-level prediction is the sum of per-operator `pred_us`; pipeline means the existing E2E scheduler is used.",
            "",
            "## CDF summary",
            "",
            "| Device | Ablation | Combos | MAPE | P90 APE | P95 APE | "
            + " | ".join(f"CDF @ {threshold:.2f}" for threshold in thresholds)
            + " |",
            "| --- | --- | ---: | ---: | ---: | ---: | "
            + " | ".join("---:" for _ in thresholds)
            + " |",
        ]
    )
    for _, row in summary.iterrows():
        cdf_values = " | ".join(f"{float(row[col]):.1f}%" for col in threshold_cols)
        lines.append(
            f"| {row['device_label']} | {row['ablation_label']} | {int(row['combo_count'])} | "
            f"{float(row['mape']):.2%} | {float(row['p90_ape']):.2%} | {float(row['p95_ape']):.2%} | {cdf_values} |"
        )
    lines.extend(
        [
            "",
            "## Module effect audit",
            "",
            "| Device | Effect | Before | After | MAPE improvement | P90 improvement | Status |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for _, row in effect.iterrows():
        status_text = "PASS" if bool(row["pass_gate"]) else "NEEDS_REVIEW"
        lines.append(
            f"| {row['device_label']} | `{row['effect_id']}` | `{row['before_ablation']}` | `{row['after_ablation']}` | "
            f"{float(row['mape_relative_improvement']):.1%} | {float(row['p90_relative_improvement']):.1%} | {status_text} |"
        )
    if not singleop_diag.empty:
        lines.extend(
            [
                "",
                "## Single-op diagnostic",
                "",
                "This diagnostic checks whether RF/scale improves per-operator prediction before E2E aggregation.",
                "",
                "| Device | Stage | Rows | Combos | Rows/combo | Single-op MAPE | Single-op P90 APE |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in singleop_diag.iterrows():
            lines.append(
                f"| {row['device_label']} | {row['stage_label']} | {int(row['row_count'])} | {int(row['combo_count'])} | "
                f"{int(row['min_rows_per_combo'])}-{int(row['max_rows_per_combo'])} | "
                f"{float(row['singleop_mape']):.2%} | {float(row['singleop_p90_ape']):.2%} |"
            )
    if status["failures"]:
        lines.extend(["", "## Review notes", ""])
        lines.append(
            "One or more required effects did not meet the configured gate, or combo alignment dropped rows. Treat the figure as diagnostic until the listed items are reviewed."
        )
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(status["failures"], indent=2, ensure_ascii=False))
        lines.append("```")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `ablation_e2e_combo_metrics.csv`",
            "- `ablation_e2e_relative_error_cdf_points.csv`",
            "- `ablation_e2e_relative_error_cdf_summary.csv`",
            "- `ablation_module_effect_summary.csv`",
            "- `singleop_ablation_diagnostic.csv`",
            "- `ablation_review_status.json`",
            "- `e2e_ablation_cdf_device1.png/pdf`",
            "- `e2e_ablation_cdf_device2.png/pdf`",
            "- `e2e_ablation_cdf_combined.png/pdf`",
            "",
            "## Reproduce",
            "",
            "Run raw analytical no-scale for both devices first, then run:",
            "",
            "```bash",
            "python calibrator/experimental/plot_e2e_ablation_cdf.py \\",
            f"  --device1-raw-root {args.device1_raw_root} \\",
            f"  --device2-raw-root {args.device2_raw_root} \\",
            f"  --device1-full-root {args.device1_full_root} \\",
            f"  --device2-full-root {args.device2_full_root} \\",
            f"  --rf-route {args.rf_route} \\",
            f"  --prior-route {args.prior_route} \\",
            f"  --threshold {args.threshold} \\",
            f"  --module-improvement-threshold {args.module_improvement_threshold} \\",
            f"  --out-dir {args.out_dir}",
            "```",
            "",
        ]
    )
    (out_dir / "E2E_ABLATION_CDF_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    thresholds = _parse_thresholds(args.threshold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device1, audit1, single1 = _load_device_ablation(
        device_id="Device_1",
        raw_root=Path(args.device1_raw_root),
        full_root=Path(args.device1_full_root),
        prior_route=args.prior_route,
        rf_route=args.rf_route,
    )
    device2, audit2, single2 = _load_device_ablation(
        device_id="Device_2",
        raw_root=Path(args.device2_raw_root),
        full_root=Path(args.device2_full_root),
        prior_route=args.prior_route,
        rf_route=args.rf_route,
    )
    combo_metrics = pd.concat([device1, device2], ignore_index=True)
    combo_metrics["ablation_order"] = combo_metrics["ablation_id"].map({name: i for i, name in enumerate(ABLATION_ORDER)})
    combo_metrics = combo_metrics.sort_values(["device_id", "ablation_order", "source_name", "combo"]).drop(columns=["ablation_order"])

    points = _cdf_points(combo_metrics)
    summary = _summary(combo_metrics, thresholds=thresholds)
    effect = _effect_summary(summary, threshold=float(args.module_improvement_threshold))
    status = _review_status(effect, [audit1, audit2], threshold=float(args.module_improvement_threshold))
    singleop_diag = pd.concat([single1, single2], ignore_index=True)

    combo_metrics.to_csv(out_dir / "ablation_e2e_combo_metrics.csv", index=False)
    points.to_csv(out_dir / "ablation_e2e_relative_error_cdf_points.csv", index=False)
    summary.to_csv(out_dir / "ablation_e2e_relative_error_cdf_summary.csv", index=False)
    effect.to_csv(out_dir / "ablation_module_effect_summary.csv", index=False)
    singleop_diag.to_csv(out_dir / "singleop_ablation_diagnostic.csv", index=False)
    (out_dir / "ablation_review_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot_all(points, summary, thresholds=thresholds, out_dir=out_dir)
    _write_report(summary, effect, singleop_diag, status, thresholds=thresholds, args=args, out_dir=out_dir)
    print(f"ablation review status: {status['status']}")


if __name__ == "__main__":
    main()
