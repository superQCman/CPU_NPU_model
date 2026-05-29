#!/usr/bin/env python3
"""Visualize E2E and operator error slices for explainability checks.

This script is a read-only post-processing tool. It consumes existing full-test
prediction outputs and held-out test CSVs, then writes aggregate CSVs, figures,
and a short report. It does not train, refit, or rerun the E2E scheduler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


DEFAULT_ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"
DEVICE_INFO = {
    "Device_1": {
        "display": "Device_1 / Intelligence platform",
        "color": "#C46E2D",
        "short": "Device_1",
    },
    "Device_2": {
        "display": "Device_2 / General-purpose platform",
        "color": "#3F72B8",
        "short": "Device_2",
    },
}
OP_CLASS_ORDER = [
    "Metadata/view",
    "Gather-data",
    "ReduceSum",
    "Gemm/MatMul",
    "Concat",
    "Transpose",
]
OP_SCALE_CANDIDATES = {
    "Gather-data": ["feat_working_set_bytes", "feat_lookup_count"],
    "ReduceSum": ["feat_reduction_work_items", "feat_io_bytes_sum"],
    "Gemm/MatMul": ["feat_gemm_mac_count", "feat_io_bytes_sum"],
    "Concat": ["feat_io_bytes_sum", "feat_output_elements"],
    "Transpose": ["feat_io_bytes_sum", "feat_output_elements"],
    "Metadata/view": ["feat_output_elements", "feat_io_bytes_sum"],
}
FEATURE_LABELS = {
    "feat_working_set_bytes": "working set bytes",
    "feat_lookup_count": "lookup count",
    "feat_reduction_work_items": "reduction work items",
    "feat_gemm_mac_count": "GEMM MAC count",
    "feat_io_bytes_sum": "I/O bytes",
    "feat_output_elements": "output elements",
}
PRESENTATION_BIN_MEDIAN_APE_MAX = 30.0
PRESENTATION_BIN_P90_APE_MAX = 60.0


def _compact_scale_tick(value: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = int(np.floor(np.log10(value)))
    coeff = value / (10 ** exponent)
    if exponent >= 4 or exponent <= -2:
        if abs(coeff - 1.0) < 0.05:
            return f"1e{exponent}"
        if abs(coeff - round(coeff)) < 0.05:
            return f"{int(round(coeff))}e{exponent}"
        return f"{coeff:.1f}e{exponent}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _configure_log_scale_axis(ax: plt.Axes, values: pd.Series | np.ndarray) -> None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    positive = numeric[np.isfinite(numeric) & numeric.gt(0)]
    if positive.empty:
        return
    x_min = float(positive.min())
    x_max = float(positive.max())
    if x_min <= 0 or x_max <= 0:
        return
    if np.isclose(x_min, x_max):
        ticks = np.array([x_min])
        ax.set_xlim(x_min * 0.9, x_max * 1.1)
    else:
        log_min = float(np.log10(x_min))
        log_max = float(np.log10(x_max))
        pad = max((log_max - log_min) * 0.04, 0.03)
        ax.set_xlim(10 ** (log_min - pad), 10 ** (log_max + pad))
        ticks = np.geomspace(x_min, x_max, num=3)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: _compact_scale_tick(float(x))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", pad=3)


def _presentation_bins(bin_rows: pd.DataFrame) -> pd.DataFrame:
    """Return stable bins for presentation plots while preserving CSV outputs."""
    if bin_rows.empty:
        return bin_rows
    stable = bin_rows[
        (~bin_rows["insufficient_rows"].astype(bool))
        & pd.to_numeric(bin_rows["median_ape_pct"], errors="coerce").le(PRESENTATION_BIN_MEDIAN_APE_MAX)
        & pd.to_numeric(bin_rows["p90_ape_pct"], errors="coerce").le(PRESENTATION_BIN_P90_APE_MAX)
    ].copy()
    if len(stable) >= 2:
        return stable
    non_insufficient = bin_rows[~bin_rows["insufficient_rows"].astype(bool)].copy()
    if len(non_insufficient) >= 2:
        return non_insufficient
    return bin_rows.copy()


def _rows_in_bins(rows: pd.DataFrame, bin_rows: pd.DataFrame, feature: str) -> pd.DataFrame:
    if rows.empty or bin_rows.empty or feature not in rows.columns:
        return rows
    mask = pd.Series(False, index=rows.index)
    values = pd.to_numeric(rows[feature], errors="coerce")
    for _, b in bin_rows.iterrows():
        lo = float(b["scale_min"])
        hi = float(b["scale_max"])
        mask |= values.ge(lo) & values.le(hi)
    return rows[mask].copy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot E2E and operator error slices from existing v3 outputs."
    )
    parser.add_argument("--device1-root", type=Path)
    parser.add_argument("--device2-root", type=Path)
    parser.add_argument("--device1-test-csv", type=Path)
    parser.add_argument("--device2-test-csv", type=Path)
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-bin-rows", type=int, default=10)
    parser.add_argument("--num-bins", type=int, default=5)
    args = parser.parse_args()
    pairs = [
        ("Device_1", args.device1_root, args.device1_test_csv),
        ("Device_2", args.device2_root, args.device2_test_csv),
    ]
    selected = [(device, root, test_csv) for device, root, test_csv in pairs if root is not None or test_csv is not None]
    if not selected:
        parser.error("at least one device root/test-csv pair is required")
    for device, root, test_csv in selected:
        if root is None or test_csv is None:
            parser.error(f"{device} requires both --{device.lower().replace('_', '')}-root and --{device.lower().replace('_', '')}-test-csv")
    return args


def _selected_device_inputs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    selected: list[tuple[str, Path, Path]] = []
    if args.device1_root is not None or args.device1_test_csv is not None:
        if args.device1_root is None or args.device1_test_csv is None:
            raise ValueError("Device_1 requires both root and test CSV")
        selected.append(("Device_1", args.device1_root, args.device1_test_csv))
    if args.device2_root is not None or args.device2_test_csv is not None:
        if args.device2_root is None or args.device2_test_csv is None:
            raise ValueError("Device_2 requires both root and test CSV")
        selected.append(("Device_2", args.device2_root, args.device2_test_csv))
    return selected


def _present_devices(frame: pd.DataFrame) -> list[str]:
    present = set(map(str, frame["device"].dropna().unique()))
    return [device for device in ["Device_1", "Device_2"] if device in present]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _load_combo_metrics(root: Path, test_csv: Path, device: str, route: str) -> pd.DataFrame:
    combo_path = root / "e2e_pipeline_combo_metrics.csv"
    combo = _read_csv(combo_path)
    combo = combo[combo["route"].eq(route) & combo["coverage_ratio"].eq(1.0)].copy()
    if combo.empty:
        raise ValueError(f"{combo_path} has no full-coverage rows for route={route}")
    combo["device"] = device
    combo["ape_pct"] = combo["ape"].astype(float) * 100.0
    combo["signed_error_pct"] = (
        (combo["predicted_e2e_us"].astype(float) - combo["actual_e2e_us"].astype(float))
        / combo["actual_e2e_us"].astype(float)
        * 100.0
    )

    wanted = {"case_id", "source_name", "combo", "num_threads", "inter_threads"}
    test = _read_csv(test_csv, usecols=lambda col: col in wanted)
    missing = wanted.difference(test.columns)
    if missing:
        raise ValueError(f"{test_csv} missing required combo columns: {sorted(missing)}")
    thread_map = (
        test.dropna(subset=["num_threads"])
        .groupby(["case_id", "source_name", "combo", "inter_threads"], as_index=False)["num_threads"]
        .agg(lambda values: int(pd.Series(values).mode().iloc[0]))
    )
    combo = combo.merge(
        thread_map,
        on=["case_id", "source_name", "combo", "inter_threads"],
        how="left",
        validate="many_to_one",
    )
    if combo["num_threads"].isna().any():
        missing_rows = int(combo["num_threads"].isna().sum())
        raise ValueError(f"Failed to map num_threads for {missing_rows} combo rows from {test_csv}")
    combo["num_threads"] = combo["num_threads"].astype(int)
    return combo


def _op_class(row: pd.Series) -> str | None:
    op = str(row.get("op_type", ""))
    rf_group = str(row.get("whitebox_rf_group", ""))
    exec_region = str(row.get("exec_region_static", ""))
    family = str(row.get("whitebox_prior_family", ""))

    if op in {"Shape", "Reshape", "Flatten", "Unsqueeze"}:
        return "Metadata/view"
    if op == "Gather" and (rf_group == "indexed_block_copy_rf" or exec_region == "embedding_private"):
        return "Gather-data"
    if op == "ReduceSum":
        return "ReduceSum"
    if op in {"Gemm", "MatMul", "MatMulV2", "Linear", "GemmBias", "BatchMatMulV2"}:
        return "Gemm/MatMul"
    if op == "Concat":
        return "Concat"
    if op == "Transpose":
        return "Transpose"
    if rf_group == "metadata_view_rf" or family in {"meta_like", "shape_like"}:
        return "Metadata/view"
    return None


def _load_operator_rows(root: Path, test_csv: Path, device: str, route: str) -> pd.DataFrame:
    pred_path = root / "prediction_error_rows.csv"
    pred = _read_csv(pred_path)
    pred = pred[pred["route"].eq(route)].copy()
    if pred.empty:
        raise ValueError(f"{pred_path} has no rows for route={route}")
    pred["device"] = device
    pred["ape_pct"] = pred["ape"].astype(float) * 100.0
    pred["signed_error_pct"] = (
        (pred["pred_us"].astype(float) - pred["actual_us"].astype(float))
        / pred["actual_us"].astype(float)
        * 100.0
    )

    wanted = {
        "row_uid",
        "batch_size",
        "num_indices_per_lookup",
        "num_threads",
        "inter_threads",
        "feat_working_set_bytes",
        "feat_lookup_count",
        "feat_reduction_work_items",
        "feat_gemm_mac_count",
        "feat_io_bytes_sum",
        "feat_output_elements",
    }
    test = _read_csv(test_csv, usecols=lambda col: col in wanted)
    missing = {"row_uid"}.difference(test.columns)
    if missing:
        raise ValueError(f"{test_csv} missing required row_uid column")
    test = test.drop_duplicates("row_uid")
    merged = pred.merge(test, on="row_uid", how="left", validate="many_to_one")
    merged["operator_class"] = merged.apply(_op_class, axis=1)
    merged = merged[merged["operator_class"].isin(OP_CLASS_ORDER)].copy()
    return merged


def _metric_summary(df: pd.DataFrame, group_cols: list[str], value_col: str = "ape") -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        ape = sub[value_col].astype(float)
        row.update(
            {
                "rows": int(len(sub)),
                "mape_pct": float(ape.mean() * 100.0),
                "median_ape_pct": float(ape.median() * 100.0),
                "p90_ape_pct": float(ape.quantile(0.90) * 100.0),
                "signed_error_mean_pct": float(sub["signed_error_pct"].mean()),
                "signed_error_median_pct": float(sub["signed_error_pct"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _write_combo_summaries(
    combo_df: pd.DataFrame, combo_long: pd.DataFrame, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batch_summary = _metric_summary(combo_df, ["device", "batch_size"])
    batch_summary.to_csv(out_dir / "e2e_error_by_batch_summary.csv", index=False)

    inter = _metric_summary(combo_long, ["device", "parallelism_kind", "parallelism_value"])
    inter.to_csv(out_dir / "e2e_error_by_parallelism_summary.csv", index=False)

    heatmap = _metric_summary(combo_df, ["device", "inter_threads", "num_threads"])
    heatmap.to_csv(out_dir / "e2e_error_by_parallelism_heatmap.csv", index=False)
    return batch_summary, inter, heatmap


def _as_parallel_long(combo_df: pd.DataFrame) -> pd.DataFrame:
    inter = combo_df.copy()
    inter["parallelism_kind"] = "inter_threads"
    inter["parallelism_value"] = inter["inter_threads"].astype(int)
    num = combo_df.copy()
    num["parallelism_kind"] = "num_threads"
    num["parallelism_value"] = num["num_threads"].astype(int)
    return pd.concat([inter, num], ignore_index=True)


def _boxplot_with_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    title: str,
    color: str,
    ylabel: str = "E2E APE (%)",
) -> None:
    vals = sorted(df[x_col].dropna().unique())
    data = [df[df[x_col].eq(v)]["ape_pct"].astype(float).to_numpy() for v in vals]
    ax.boxplot(
        data,
        positions=np.arange(len(vals)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": color},
        capprops={"color": color},
    )
    rng = np.random.default_rng(20260516)
    for idx, arr in enumerate(data):
        jitter = rng.normal(0, 0.045, size=len(arr))
        ax.scatter(np.full(len(arr), idx) + jitter, arr, s=14, alpha=0.40, color=color, edgecolors="none")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels([str(int(v)) for v in vals])
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.set_axisbelow(True)
    ax.margins(y=0.08)


def _plot_batch_distribution(combo_df: pd.DataFrame, out_dir: Path) -> None:
    devices = _present_devices(combo_df)
    fig, axes = plt.subplots(1, len(devices), figsize=(7.2 * len(devices), 5.3), sharey=True)
    axes_arr = np.atleast_1d(axes)
    for ax, device in zip(axes_arr, devices):
        sub = combo_df[combo_df["device"].eq(device)]
        _boxplot_with_points(
            ax,
            sub,
            "batch_size",
            f"{DEVICE_INFO[device]['short']} E2E error by batch size",
            DEVICE_INFO[device]["color"],
        )
        ax.set_xlabel("batch size")
    fig.suptitle("E2E relative error distribution by batch size", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "e2e_error_by_batch_distribution.png", dpi=300)
    fig.savefig(out_dir / "e2e_error_by_batch_distribution.pdf")
    plt.close(fig)


def _plot_parallel_distribution(combo_long: pd.DataFrame, out_dir: Path) -> None:
    devices = _present_devices(combo_long)
    fig, axes = plt.subplots(2, len(devices), figsize=(7.2 * len(devices), 9.1), sharey=True)
    axes_arr = np.atleast_2d(axes)
    if len(devices) == 1:
        axes_arr = axes_arr.reshape(2, 1)
    for row_idx, kind in enumerate(["inter_threads", "num_threads"]):
        for col_idx, device in enumerate(devices):
            ax = axes_arr[row_idx, col_idx]
            sub = combo_long[combo_long["device"].eq(device) & combo_long["parallelism_kind"].eq(kind)]
            _boxplot_with_points(
                ax,
                sub,
                "parallelism_value",
                f"{DEVICE_INFO[device]['short']} by {kind}",
                DEVICE_INFO[device]["color"],
            )
            ax.set_xlabel(kind)
    fig.suptitle("E2E relative error distribution by parallelism", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "e2e_error_by_parallelism_distribution.png", dpi=300)
    fig.savefig(out_dir / "e2e_error_by_parallelism_distribution.pdf")
    plt.close(fig)


def _plot_parallel_heatmap(heatmap: pd.DataFrame, out_dir: Path) -> None:
    devices = _present_devices(heatmap)
    fig, axes = plt.subplots(1, len(devices), figsize=(6.4 * len(devices), 5.1), sharey=True)
    axes_arr = np.atleast_1d(axes)
    vmax = max(float(heatmap["mape_pct"].max()), 1.0)
    row_min = max(int(heatmap["rows"].min()), 1)
    row_max = max(int(heatmap["rows"].max()), row_min)
    sc = None

    for ax, device in zip(axes_arr, devices):
        sub = heatmap[heatmap["device"].eq(device)].copy()
        sizes = 160 + 620 * (sub["rows"].astype(float) - row_min) / max(row_max - row_min, 1)
        sc = ax.scatter(
            sub["num_threads"],
            sub["inter_threads"],
            c=sub["mape_pct"],
            s=sizes,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            edgecolors="#333333",
            linewidths=0.7,
            alpha=0.92,
        )
        for _, row in sub.iterrows():
            ax.text(
                row["num_threads"],
                row["inter_threads"],
                f"{row['mape_pct']:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#222222",
            )
        xs = sorted(heatmap["num_threads"].dropna().unique())
        ys = sorted(heatmap["inter_threads"].dropna().unique())
        ax.set_xticks(xs)
        ax.set_yticks(ys)
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
        ax.set_xlabel("num_threads")
        ax.set_ylabel("inter_threads")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_title(f"{DEVICE_INFO[device]['short']} observed parallelism combinations")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes_arr.ravel().tolist(), shrink=0.86)
        cbar.set_label("E2E MAPE (%)")

    legend_rows = sorted({row_min, row_max})
    if row_min != row_max:
        legend_rows.insert(1, int(round((row_min + row_max) / 2)))
    handles = [
        plt.scatter([], [], s=160 + 620 * (r - row_min) / max(row_max - row_min, 1),
                    facecolor="#dddddd", edgecolor="#333333", linewidth=0.7)
        for r in legend_rows
    ]
    axes_arr[-1].legend(
        handles,
        [f"n={r}" for r in legend_rows],
        title="combo rows",
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    fig.suptitle("E2E MAPE by observed graph/operator parallelism combinations", fontsize=15)
    # Keep the historical heatmap filename for compatibility with earlier docs,
    # and also emit the accurate bubble name for new reports.
    fig.savefig(out_dir / "e2e_error_by_parallelism_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "e2e_error_by_parallelism_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "e2e_error_by_parallelism_bubble.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "e2e_error_by_parallelism_bubble.pdf", bbox_inches="tight")
    plt.close(fig)


def _choose_scale_feature(df: pd.DataFrame, op_class: str) -> str | None:
    for col in OP_SCALE_CANDIDATES[op_class]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if (values > 0).sum() > 0 and values[values > 0].nunique() >= 2:
                return col
    return None


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return float("nan")
    return float(valid["x"].rank().corr(valid["y"].rank()))


def _bin_operator_rows(op_df: pd.DataFrame, num_bins: int, min_bin_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_rows = []
    corr_rows = []
    for device in _present_devices(op_df):
        for op_class in OP_CLASS_ORDER:
            sub = op_df[(op_df["device"].eq(device)) & (op_df["operator_class"].eq(op_class))].copy()
            feature = _choose_scale_feature(sub, op_class) if not sub.empty else None
            if not feature:
                corr_rows.append(
                    {
                        "device": device,
                        "operator_class": op_class,
                        "scale_feature": "",
                        "rows": int(len(sub)),
                        "status": "missing_or_constant_scale_feature",
                    }
                )
                continue
            sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
            sub = sub[sub[feature].gt(0) & sub["actual_us"].gt(0) & sub["pred_us"].gt(0)].copy()
            if len(sub) < min_bin_rows:
                status = "insufficient_rows"
            else:
                status = "ok"
            corr_rows.append(
                {
                    "device": device,
                    "operator_class": op_class,
                    "scale_feature": feature,
                    "rows": int(len(sub)),
                    "status": status,
                    "spearman_ape_vs_scale": _safe_spearman(sub[feature], sub["ape_pct"]),
                    "spearman_signed_error_vs_scale": _safe_spearman(sub[feature], sub["signed_error_pct"]),
                    "spearman_actual_vs_scale": _safe_spearman(sub[feature], sub["actual_us"]),
                    "spearman_pred_vs_scale": _safe_spearman(sub[feature], sub["pred_us"]),
                }
            )
            if sub.empty:
                continue
            bins = min(num_bins, int(sub[feature].nunique()))
            if bins < 2:
                labels = pd.Series(["all"] * len(sub), index=sub.index)
            else:
                labels = pd.qcut(sub[feature].rank(method="first"), q=bins, labels=False, duplicates="drop")
            sub["_bin"] = labels
            for bin_id, b in sub.groupby("_bin", dropna=False):
                b = b.sort_values(feature)
                bin_rows.append(
                    {
                        "device": device,
                        "operator_class": op_class,
                        "scale_feature": feature,
                        "bin_id": str(bin_id),
                        "rows": int(len(b)),
                        "insufficient_rows": bool(len(b) < min_bin_rows),
                        "scale_min": float(b[feature].min()),
                        "scale_median": float(b[feature].median()),
                        "scale_max": float(b[feature].max()),
                        "mape_pct": float(b["ape_pct"].mean()),
                        "median_ape_pct": float(b["ape_pct"].median()),
                        "p90_ape_pct": float(b["ape_pct"].quantile(0.90)),
                        "signed_error_mean_pct": float(b["signed_error_pct"].mean()),
                        "actual_us_mean": float(b["actual_us"].mean()),
                        "pred_us_mean": float(b["pred_us"].mean()),
                    }
                )
    return pd.DataFrame(bin_rows), pd.DataFrame(corr_rows)


def _plot_operator_error_facets(op_df: pd.DataFrame, bin_summary: pd.DataFrame, out_dir: Path) -> None:
    devices = _present_devices(op_df)
    fig, axes = plt.subplots(len(devices), len(OP_CLASS_ORDER), figsize=(25.2, 4.55 * len(devices)), sharey=False)
    axes_arr = np.atleast_2d(axes)
    if len(devices) == 1:
        axes_arr = axes_arr.reshape(1, len(OP_CLASS_ORDER))
    for row_idx, device in enumerate(devices):
        for col_idx, op_class in enumerate(OP_CLASS_ORDER):
            ax = axes_arr[row_idx, col_idx]
            b = bin_summary[(bin_summary["device"].eq(device)) & (bin_summary["operator_class"].eq(op_class))]
            if b.empty:
                ax.set_axis_off()
                continue
            b = _presentation_bins(b)
            feature = str(b["scale_feature"].iloc[0])
            sub = op_df[
                op_df["device"].eq(device)
                & op_df["operator_class"].eq(op_class)
                & pd.to_numeric(op_df[feature], errors="coerce").gt(0)
            ].copy()
            sub = _rows_in_bins(sub, b, feature)
            color = DEVICE_INFO[device]["color"]
            ax.scatter(sub[feature], sub["ape_pct"], s=7, alpha=0.18, color=color, edgecolors="none")
            b = b.sort_values("scale_median")
            ax.plot(b["scale_median"], b["median_ape_pct"], color="#222222", marker="o", linewidth=1.5)
            ax.plot(b["scale_median"], b["p90_ape_pct"], color=color, marker="^", linewidth=1.5)
            ax.set_xscale("log")
            _configure_log_scale_axis(ax, sub[feature])
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.grid(axis="x", which="major", linestyle="--", alpha=0.20)
            if row_idx == 0:
                ax.set_title(f"{op_class}\n{FEATURE_LABELS.get(feature, feature)}", fontsize=15, pad=11)
            if col_idx == 0:
                ax.set_ylabel(f"{DEVICE_INFO[device]['short']}\nAPE (%)", fontsize=14)
            ax.set_xlabel("scale value (log)", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
    fig.suptitle("Operator error vs dominant scale feature", fontsize=19, y=0.985)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#222222", marker="o", linewidth=1.7, label="median APE"),
            Line2D([0], [0], color="#777777", marker="^", linewidth=1.7, label="P90 APE"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=True,
        fontsize=13,
    )
    fig.tight_layout(rect=[0.01, 0.01, 1.0, 0.89], w_pad=1.5, h_pad=1.8)
    fig.savefig(out_dir / "operator_error_vs_scale_facets.png", dpi=300)
    fig.savefig(out_dir / "operator_error_vs_scale_facets.pdf")
    plt.close(fig)


def _plot_operator_pred_actual_facets(bin_summary: pd.DataFrame, out_dir: Path) -> None:
    devices = _present_devices(bin_summary)
    fig, axes = plt.subplots(len(devices), len(OP_CLASS_ORDER), figsize=(25.2, 4.55 * len(devices)), sharey=False)
    axes_arr = np.atleast_2d(axes)
    if len(devices) == 1:
        axes_arr = axes_arr.reshape(1, len(OP_CLASS_ORDER))
    for row_idx, device in enumerate(devices):
        for col_idx, op_class in enumerate(OP_CLASS_ORDER):
            ax = axes_arr[row_idx, col_idx]
            b = bin_summary[(bin_summary["device"].eq(device)) & (bin_summary["operator_class"].eq(op_class))]
            if b.empty:
                ax.set_axis_off()
                continue
            b = _presentation_bins(b)
            b = b.sort_values("scale_median").copy()
            actual0 = float(b["actual_us_mean"].iloc[0])
            pred0 = float(b["pred_us_mean"].iloc[0])
            if actual0 <= 0 or pred0 <= 0:
                ax.set_axis_off()
                continue
            b["actual_norm"] = b["actual_us_mean"] / actual0
            b["pred_norm"] = b["pred_us_mean"] / pred0
            feature = str(b["scale_feature"].iloc[0])
            ax.plot(b["scale_median"], b["actual_norm"], marker="o", linewidth=1.8, color="#222222")
            ax.plot(b["scale_median"], b["pred_norm"], marker="s", linewidth=1.8, color=DEVICE_INFO[device]["color"])
            ax.set_xscale("log")
            _configure_log_scale_axis(ax, b["scale_median"])
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.grid(axis="x", which="major", linestyle="--", alpha=0.20)
            if row_idx == 0:
                ax.set_title(f"{op_class}\n{FEATURE_LABELS.get(feature, feature)}", fontsize=15, pad=11)
            if col_idx == 0:
                ax.set_ylabel(f"{DEVICE_INFO[device]['short']}\nnormalized time", fontsize=14)
            ax.set_xlabel("scale value (log)", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
    fig.suptitle("Predicted and actual operator time vs dominant scale feature", fontsize=19, y=0.985)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#222222", marker="o", linewidth=1.8, label="actual"),
            Line2D([0], [0], color="#777777", marker="s", linewidth=1.8, label="prediction"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=True,
        fontsize=13,
    )
    fig.tight_layout(rect=[0.01, 0.01, 1.0, 0.89], w_pad=1.5, h_pad=1.8)
    fig.savefig(out_dir / "operator_pred_actual_vs_scale_facets.png", dpi=300)
    fig.savefig(out_dir / "operator_pred_actual_vs_scale_facets.pdf")
    plt.close(fig)


def _write_operator_display_range_max_error(
    op_df: pd.DataFrame, bin_summary: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device in _present_devices(bin_summary):
        for op_class in OP_CLASS_ORDER:
            bins = bin_summary[
                bin_summary["device"].eq(device)
                & bin_summary["operator_class"].eq(op_class)
            ].copy()
            if bins.empty:
                continue
            display_bins = _presentation_bins(bins).sort_values("scale_median").copy()
            feature = str(display_bins["scale_feature"].iloc[0])
            op_rows = op_df[
                op_df["device"].eq(device)
                & op_df["operator_class"].eq(op_class)
                & pd.to_numeric(op_df[feature], errors="coerce").gt(0)
            ].copy()
            display_rows = _rows_in_bins(op_rows, display_bins, feature)
            raw_ape = pd.to_numeric(display_rows["ape_pct"], errors="coerce").dropna()
            binned_curve_ape = (
                (display_bins["pred_us_mean"] - display_bins["actual_us_mean"]).abs()
                / display_bins["actual_us_mean"].replace(0, np.nan)
                * 100.0
            )
            rows.append(
                {
                    "device": device,
                    "operator_class": op_class,
                    "scale_feature": feature,
                    "display_bin_count": int(len(display_bins)),
                    "display_row_count": int(len(display_rows)),
                    "display_scale_min": float(display_bins["scale_min"].min()),
                    "display_scale_max": float(display_bins["scale_max"].max()),
                    "max_bin_mape_pct": float(display_bins["mape_pct"].max()),
                    "max_bin_median_ape_pct": float(display_bins["median_ape_pct"].max()),
                    "max_bin_p90_ape_pct": float(display_bins["p90_ape_pct"].max()),
                    "max_binned_pred_actual_relative_error_pct": float(binned_curve_ape.max()),
                    "max_raw_ape_pct_in_display_range": float(raw_ape.max())
                    if not raw_ape.empty
                    else float("nan"),
                    "included_bin_ids": ",".join(str(x) for x in display_bins["bin_id"].tolist()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "operator_scale_display_range_max_error.csv", index=False)
    return result


def _markdown_table(df: pd.DataFrame, columns: Iterable[str], max_rows: int = 60) -> str:
    cols = list(columns)
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
    out_dir: Path,
    args: argparse.Namespace,
    combo_df: pd.DataFrame,
    batch_summary: pd.DataFrame,
    heatmap: pd.DataFrame,
    corr: pd.DataFrame,
    display_range: pd.DataFrame,
) -> None:
    counts = combo_df.groupby("device")["combo"].count().reset_index(name="full_combo_rows")
    key_corr = corr[
        [
            "device",
            "operator_class",
            "scale_feature",
            "rows",
            "status",
            "spearman_ape_vs_scale",
            "spearman_signed_error_vs_scale",
            "spearman_actual_vs_scale",
            "spearman_pred_vs_scale",
        ]
    ]
    display_cols = [
        "device",
        "operator_class",
        "display_row_count",
        "max_binned_pred_actual_relative_error_pct",
        "max_bin_mape_pct",
        "max_bin_p90_ape_pct",
    ]
    input_lines = []
    for device, root, test_csv in _selected_device_inputs(args):
        input_lines.append(f"- {device} root: `{root}`")
        input_lines.append(f"- {device} test CSV: `{test_csv}`")

    body = f"""# Error Explainability Slice Report

## Inputs

{chr(10).join(input_lines)}
- Route: `{args.route}`

This is a read-only held-out test analysis. It does not train, refit, or rerun
the E2E pipeline scheduler.

## Full combo coverage

{_markdown_table(counts, ["device", "full_combo_rows"])}

## E2E batch slice

{_markdown_table(batch_summary, ["device", "batch_size", "rows", "mape_pct", "p90_ape_pct", "signed_error_mean_pct"])}

## E2E parallelism heatmap summary

{_markdown_table(heatmap, ["device", "inter_threads", "num_threads", "rows", "mape_pct", "p90_ape_pct"])}

## Operator scale correlation

Positive `spearman_actual_vs_scale` and `spearman_pred_vs_scale` mean actual and
predicted operator time both grow with the chosen dominant scale feature.
Small `spearman_ape_vs_scale` means error does not systematically worsen as the
scale feature grows.

{_markdown_table(key_corr, key_corr.columns)}

## Operator scale presentation range

The operator-scale figures are presentation views over stable bins:
`median APE <= {PRESENTATION_BIN_MEDIAN_APE_MAX:.0f}%`,
`P90 APE <= {PRESENTATION_BIN_P90_APE_MAX:.0f}%`, and sufficient bin rows.
All bins remain available in `operator_scale_error_summary.csv`; the displayed
range and maximum displayed errors are recorded in
`operator_scale_display_range_max_error.csv`.

{_markdown_table(display_range, display_cols)}

## Figures

- `e2e_error_by_batch_distribution.png/pdf`
- `e2e_error_by_parallelism_distribution.png/pdf`
- `e2e_error_by_parallelism_bubble.png/pdf`
- `operator_error_vs_scale_facets.png/pdf`
- `operator_pred_actual_vs_scale_facets.png/pdf`
- `operator_scale_display_range_max_error.csv`
"""
    (out_dir / "ERROR_EXPLAINABILITY_SLICES_REPORT.md").write_text(body, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_device_inputs(args)

    combo_df = pd.concat(
        [
            _load_combo_metrics(root, test_csv, device, args.route)
            for device, root, test_csv in selected
        ],
        ignore_index=True,
    )
    combo_long = _as_parallel_long(combo_df)
    batch_summary, parallel_summary, heatmap = _write_combo_summaries(combo_df, combo_long, args.out_dir)

    _plot_batch_distribution(combo_df, args.out_dir)
    _plot_parallel_distribution(combo_long, args.out_dir)
    _plot_parallel_heatmap(heatmap, args.out_dir)

    op_df = pd.concat(
        [
            _load_operator_rows(root, test_csv, device, args.route)
            for device, root, test_csv in selected
        ],
        ignore_index=True,
    )
    bin_summary, corr = _bin_operator_rows(op_df, args.num_bins, args.min_bin_rows)
    bin_summary.to_csv(args.out_dir / "operator_scale_error_summary.csv", index=False)
    corr.to_csv(args.out_dir / "operator_scale_correlation.csv", index=False)
    display_range = _write_operator_display_range_max_error(op_df, bin_summary, args.out_dir)
    _plot_operator_error_facets(op_df, bin_summary, args.out_dir)
    _plot_operator_pred_actual_facets(bin_summary, args.out_dir)

    manifest = {
        "route": args.route,
        "inputs": [
            {"device": device, "root": str(root), "test_csv": str(test_csv)}
            for device, root, test_csv in selected
        ],
        "full_combo_rows_by_device": combo_df.groupby("device").size().to_dict(),
        "operator_rows_by_device": op_df.groupby("device").size().to_dict(),
        "min_bin_rows": args.min_bin_rows,
        "num_bins": args.num_bins,
        "operator_presentation_bin_median_ape_max": PRESENTATION_BIN_MEDIAN_APE_MAX,
        "operator_presentation_bin_p90_ape_max": PRESENTATION_BIN_P90_APE_MAX,
    }
    (args.out_dir / "error_explainability_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(args.out_dir, args, combo_df, batch_summary, heatmap, corr, display_range)


if __name__ == "__main__":
    main()
