#!/usr/bin/env python3
"""Plot operator-class single-op accuracy from prediction_error_rows.csv.

This is a read-only post-processing utility. It does not train or refit any
model; it only aggregates existing single-op prediction rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"
CLASS_ORDER = [
    "Metadata/view",
    "Gather",
    "ReduceSum",
    "Gemm/MatMul",
    "Concat",
    "Transpose",
    "Elementwise",
]
GATHER_SPLIT_ORDER = [
    "Metadata/view",
    "Gather-data",
    "Gather-dispatch",
    "ReduceSum",
    "Gemm/MatMul",
    "Concat",
    "Transpose",
    "Elementwise",
]
DEVICE_COLORS = {"Device_1": "#C46E2D", "Device_2": "#3F72B8"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate operator-class MAPE from v3 prediction_error_rows.csv."
    )
    parser.add_argument("--device1-prediction-rows", type=Path)
    parser.add_argument("--device2-prediction-rows", type=Path)
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--split-gather",
        action="store_true",
        help="Split Gather into data-movement and dispatch-dominated subclasses.",
    )
    args = parser.parse_args()
    if args.device1_prediction_rows is None and args.device2_prediction_rows is None:
        parser.error("at least one of --device1-prediction-rows or --device2-prediction-rows is required")
    return args


def _operator_class(row: pd.Series) -> str | None:
    op = str(row.get("op_type", ""))
    rf_group = str(row.get("whitebox_rf_group", ""))
    family = str(row.get("whitebox_prior_family", ""))
    exec_region = str(row.get("exec_region_static", ""))

    if op in {"Shape", "Reshape", "Flatten", "Unsqueeze"}:
        return "Metadata/view"
    if op == "Gather":
        return "Gather"
    if op == "ReduceSum":
        return "ReduceSum"
    if op in {"Gemm", "MatMul", "MatMulV2", "Linear", "GemmBias", "BatchMatMulV2"}:
        return "Gemm/MatMul"
    if op == "Concat":
        return "Concat"
    if op == "Transpose":
        return "Transpose"
    if op in {"Add", "Mul", "Relu", "Sigmoid", "Tanh", "Gelu"}:
        return "Elementwise"

    if rf_group == "metadata_view_rf" or family in {"meta_like", "shape_like"}:
        return "Metadata/view"
    if exec_region == "metadata_control":
        return "Metadata/view"
    return None


def _operator_class_with_gather_split(row: pd.Series) -> str | None:
    base = _operator_class(row)
    if base != "Gather":
        return base
    rf_group = str(row.get("whitebox_rf_group", ""))
    exec_region = str(row.get("exec_region_static", ""))
    if rf_group == "indexed_block_copy_rf" or exec_region == "embedding_private":
        return "Gather-data"
    return "Gather-dispatch"


def _read_device(path: Path, device: str, route: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"route", "op_type", "actual_us", "pred_us", "ape"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df[df["route"].eq(route)].copy()
    if df.empty:
        raise ValueError(f"{path} has no rows for route={route}")
    df["device"] = device
    return df


def _weighted_mape(frame: pd.DataFrame) -> float:
    actual = frame["actual_us"].abs().to_numpy(dtype=float)
    pred = frame["pred_us"].to_numpy(dtype=float)
    denom = float(np.nansum(actual))
    if denom <= 0:
        return float("nan")
    return float(np.nansum(np.abs(pred - frame["actual_us"].to_numpy(dtype=float))) / denom)


def _summarize(df: pd.DataFrame, class_col: str, class_order: list[str]) -> pd.DataFrame:
    rows = []
    for device in ["Device_1", "Device_2"]:
        ddev = df[df["device"].eq(device)]
        for cls in class_order:
            sub = ddev[ddev[class_col].eq(cls)]
            if sub.empty:
                continue
            rows.append(
                {
                    "device": device,
                    "operator_class": cls,
                    "rows": int(len(sub)),
                    "mape_pct": float(sub["ape"].mean() * 100.0),
                    "weighted_mape_pct": _weighted_mape(sub) * 100.0,
                    "median_ape_pct": float(sub["ape"].median() * 100.0),
                    "p90_ape_pct": float(sub["ape"].quantile(0.90) * 100.0),
                    "actual_us_mean": float(sub["actual_us"].mean()),
                    "actual_us_sum": float(sub["actual_us"].sum()),
                    "op_types": ",".join(sorted(map(str, sub["op_type"].dropna().unique()))),
                    "rf_groups": ",".join(
                        sorted(map(str, sub.get("whitebox_rf_group", pd.Series(dtype=str)).dropna().unique()))
                    ),
                }
            )
    out = pd.DataFrame(rows)
    order = {name: idx for idx, name in enumerate(class_order)}
    out["class_order"] = out["operator_class"].map(order)
    return out.sort_values(["class_order", "device"]).drop(columns=["class_order"]).reset_index(drop=True)


def _plot_grouped_bars(
    summary: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_base: Path,
    class_order: list[str],
) -> None:
    pivot = (
        summary.pivot(index="operator_class", columns="device", values=metric_col)
        .reindex(class_order)
        .dropna(how="all")
    )
    labels = list(pivot.index)
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for offset, device in [(-width / 2, "Device_1"), (width / 2, "Device_2")]:
        vals = pivot.get(device, pd.Series(index=pivot.index, dtype=float)).to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=device,
            color=DEVICE_COLORS[device],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(0.6, np.nanmax(vals) * 0.015),
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )

    ax.set_title(title, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(frameon=True, fontsize=10)
    ax.set_axisbelow(True)
    ymax = float(np.nanmax(pivot.to_numpy(dtype=float))) if not pivot.empty else 1.0
    ax.set_ylim(0, ymax * 1.22)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.2f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    route: str,
    device1_path: Path | None,
    device2_path: Path | None,
    split_gather: bool,
) -> None:
    title = "Operator-class MAPE Report"
    if split_gather:
        title = "Operator-class MAPE Report with Gather Split"
    cols = [
        "device",
        "operator_class",
        "rows",
        "mape_pct",
        "weighted_mape_pct",
        "median_ape_pct",
        "p90_ape_pct",
        "actual_us_mean",
    ]
    gather_text = ""
    if split_gather:
        gather_text = """

## Gather split

`Gather-data` contains Gather rows whose `whitebox_rf_group` is
`indexed_block_copy_rf` or whose `exec_region_static` is `embedding_private`.
These rows correspond to embedding/private data movement. `Gather-dispatch`
contains the remaining Gather rows, which are typically tiny metadata/control
operations and are dominated by dispatch overhead.
"""
    input_lines = []
    if device1_path is not None:
        input_lines.append(f"- Device_1 prediction rows: `{device1_path}`")
    if device2_path is not None:
        input_lines.append(f"- Device_2 prediction rows: `{device2_path}`")

    body = f"""# {title}

## Inputs

{chr(10).join(input_lines)}
- Route: `{route}`

This report is generated from existing `prediction_error_rows.csv` files. It
does not retrain or refit RF models.

## Metrics

- `mape_pct`: unweighted row-level mean absolute percentage error.
- `weighted_mape_pct`: absolute-error sum divided by actual-time sum, so long
  operators contribute more than microsecond-level metadata operators.
- `p90_ape_pct`: 90th percentile row-level absolute percentage error.
{gather_text}

## Summary

{_markdown_table(summary[cols], cols)}
"""
    path.write_text(body, encoding="utf-8")


def _run_one(df: pd.DataFrame, out_dir: Path, route: str, device1_path: Path, device2_path: Path, split_gather: bool) -> None:
    if split_gather:
        class_col = "operator_class_gather_split"
        df[class_col] = df.apply(_operator_class_with_gather_split, axis=1)
        class_order = GATHER_SPLIT_ORDER
        stem = "operator_class_mape_gather_split"
        report_name = "OPERATOR_CLASS_MAPE_GATHER_SPLIT_REPORT.md"
        title_suffix = " with Gather split"
    else:
        class_col = "operator_class"
        df[class_col] = df.apply(_operator_class, axis=1)
        class_order = CLASS_ORDER
        stem = "operator_class_mape"
        report_name = "OPERATOR_CLASS_MAPE_REPORT.md"
        title_suffix = ""

    filtered = df[df[class_col].notna()].copy()
    summary = _summarize(filtered, class_col, class_order)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)

    _plot_grouped_bars(
        summary,
        "mape_pct",
        "MAPE (%)",
        f"Operator-class MAPE{title_suffix}",
        out_dir / f"{stem}_bar",
        class_order,
    )
    _plot_grouped_bars(
        summary,
        "weighted_mape_pct",
        "Weighted MAPE (%)",
        f"Operator-class weighted MAPE{title_suffix}",
        out_dir / f"operator_class_weighted_mape{'_gather_split' if split_gather else ''}_bar",
        class_order,
    )
    _write_report(out_dir / report_name, summary, route, device1_path, device2_path, split_gather)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.device1_prediction_rows is not None:
        frames.append(_read_device(args.device1_prediction_rows, "Device_1", args.route))
    if args.device2_prediction_rows is not None:
        frames.append(_read_device(args.device2_prediction_rows, "Device_2", args.route))
    df = pd.concat(frames, ignore_index=True)

    _run_one(
        df.copy(),
        args.out_dir,
        args.route,
        args.device1_prediction_rows,
        args.device2_prediction_rows,
        split_gather=False,
    )
    if args.split_gather:
        _run_one(
            df.copy(),
            args.out_dir,
            args.route,
            args.device1_prediction_rows,
            args.device2_prediction_rows,
            split_gather=True,
        )


if __name__ == "__main__":
    main()
