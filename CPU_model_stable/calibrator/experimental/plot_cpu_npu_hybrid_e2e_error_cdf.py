#!/usr/bin/env python3
"""Experimental CPU+NPU hybrid E2E error CDF.

This script is intentionally independent from the CPU mainline model.  It reads
existing CPU E2E outputs, replaces the error contribution of selected operator
classes with an external NPU error profile, and writes a separate report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"


NPU_ERROR_PROFILE = {
    "ControlComposite": {
        "sample_count": 111,
        "mean_ape": 0.0882,
        "weighted_ape": 0.0411,
        "median_ape": 0.0493,
        "p90_ape": 0.1454,
    },
    "ConvolutionSpatial": {
        "sample_count": 62,
        "mean_ape": 0.0772,
        "weighted_ape": 0.0757,
        "median_ape": 0.0332,
        "p90_ape": 0.1256,
    },
    "DenseLinearAlgebra": {
        "sample_count": 602,
        "mean_ape": 0.0942,
        "weighted_ape": 0.1009,
        "median_ape": 0.0654,
        "p90_ape": 0.1570,
    },
    "ElementwiseActivation": {
        "sample_count": 183,
        "mean_ape": 0.0562,
        "weighted_ape": 0.0576,
        "median_ape": 0.0389,
        "p90_ape": 0.1006,
    },
    "IndexingSparse": {
        "sample_count": 307,
        "mean_ape": 0.0697,
        "weighted_ape": 0.1330,
        "median_ape": 0.0459,
        "p90_ape": 0.1346,
    },
    "Normalization": {
        "sample_count": 68,
        "mean_ape": 0.0603,
        "weighted_ape": 0.0477,
        "median_ape": 0.0316,
        "p90_ape": 0.1335,
    },
    "ReductionSelection": {
        "sample_count": 228,
        "mean_ape": 0.0837,
        "weighted_ape": 0.1606,
        "median_ape": 0.0550,
        "p90_ape": 0.1586,
    },
    "TensorTransformMove": {
        "sample_count": 544,
        "mean_ape": 0.0834,
        "weighted_ape": 0.0962,
        "median_ape": 0.0516,
        "p90_ape": 0.1675,
    },
}


OP_TO_NPU_CATEGORY = {
    "Cast": "ControlComposite",
    "Constant": "ControlComposite",
    "NonZero": "ControlComposite",
    "Shape": "ControlComposite",
    "Sort": "ControlComposite",
    "Where": "ControlComposite",
    "MaxPool2D": "ConvolutionSpatial",
    "BatchMatMulV2": "DenseLinearAlgebra",
    "Gemm": "DenseLinearAlgebra",
    "GemmBias": "DenseLinearAlgebra",
    "Linear": "DenseLinearAlgebra",
    "MatMul": "DenseLinearAlgebra",
    "MatMulV2": "DenseLinearAlgebra",
    "Add": "ElementwiseActivation",
    "Gelu": "ElementwiseActivation",
    "Mul": "ElementwiseActivation",
    "Relu": "ElementwiseActivation",
    "Sigmoid": "ElementwiseActivation",
    "Tanh": "ElementwiseActivation",
    "EmbeddingBag": "IndexingSparse",
    "Gather": "IndexingSparse",
    "Index": "IndexingSparse",
    "ScatterAdd": "IndexingSparse",
    "LayerNorm": "Normalization",
    "ReduceMean": "ReductionSelection",
    "ReduceSum": "ReductionSelection",
    "Softmax": "ReductionSelection",
    "TopK": "ReductionSelection",
    "Concat": "TensorTransformMove",
    "ConcatD": "TensorTransformMove",
    "Flatten": "TensorTransformMove",
    "MemSet": "TensorTransformMove",
    "PadV3": "TensorTransformMove",
    "Reshape": "TensorTransformMove",
    "Tile": "TensorTransformMove",
    "Transpose": "TensorTransformMove",
    "Unsqueeze": "TensorTransformMove",
}


OFFLOAD_PROFILES = {
    "compute_island": {
        "Gemm",
        "MatMul",
        "Relu",
        "Sigmoid",
        "Add",
        "Mul",
        "Concat",
        "Transpose",
    },
    "sparse_enabled": {
        "Gemm",
        "MatMul",
        "Relu",
        "Sigmoid",
        "Add",
        "Mul",
        "Concat",
        "Transpose",
        "Gather",
        "ReduceSum",
    },
}


KEY_COLUMNS = ["policy", "dataset", "route", "scale", "case_id", "source_name", "combo"]


@dataclass(frozen=True)
class DeviceInput:
    label: str
    root: Path


def _parse_thresholds(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * value:.2f}%"


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text_df = df.copy()
    text_df = text_df.fillna("")
    headers = [str(col) for col in text_df.columns]
    rows = [[str(value) for value in row] for row in text_df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def _sign_with_fallback(offload_signed: float, combo_signed: float) -> tuple[float, str]:
    if offload_signed > 0:
        return 1.0, "offload_cpu_signed_error"
    if offload_signed < 0:
        return -1.0, "offload_cpu_signed_error"
    if combo_signed > 0:
        return 1.0, "combo_cpu_signed_error"
    if combo_signed < 0:
        return -1.0, "combo_cpu_signed_error"
    return 1.0, "positive_default"


def _load_combo_metrics(root: Path, route: str) -> pd.DataFrame:
    path = root / "e2e_pipeline_combo_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing combo metrics: {path}")
    df = pd.read_csv(path)
    if "route" in df.columns:
        df = df[df["route"].eq(route)].copy()
    if "coverage_ratio" in df.columns:
        df = df[df["coverage_ratio"].ge(1.0)].copy()
    if df.empty:
        raise ValueError(f"no full-coverage rows for route={route} in {path}")
    required = set(KEY_COLUMNS + ["actual_e2e_us", "predicted_e2e_us"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["cpu_signed_error_us"] = df["predicted_e2e_us"] - df["actual_e2e_us"]
    df["cpu_relative_error"] = (df["cpu_signed_error_us"].abs() / df["actual_e2e_us"]).replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def _load_operator_errors(root: Path, route: str, offloaded_ops: set[str]) -> pd.DataFrame:
    path = root / "e2e_operator_signed_error_by_combo.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing operator error rows: {path}")
    df = pd.read_csv(path)
    if "route" in df.columns:
        df = df[df["route"].eq(route)].copy()
    if df.empty:
        raise ValueError(f"no operator error rows for route={route} in {path}")
    required = set(KEY_COLUMNS + ["op_type", "actual_us_sum", "signed_error_us", "abs_signed_error_us"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df["npu_category"] = df["op_type"].map(OP_TO_NPU_CATEGORY).fillna("Unmapped")
    df["npu_mean_ape"] = df["npu_category"].map(
        {name: item["mean_ape"] for name, item in NPU_ERROR_PROFILE.items()}
    )
    df["offloaded_to_npu"] = df["op_type"].isin(offloaded_ops) & df["npu_mean_ape"].notna()
    df["npu_abs_error_us"] = np.where(
        df["offloaded_to_npu"],
        df["actual_us_sum"].clip(lower=0.0) * df["npu_mean_ape"],
        0.0,
    )
    df["cpu_signed_error_offloaded_us"] = np.where(df["offloaded_to_npu"], df["signed_error_us"], 0.0)
    df["cpu_abs_error_offloaded_us"] = np.where(df["offloaded_to_npu"], df["abs_signed_error_us"], 0.0)
    df["cpu_abs_error_non_offloaded_us"] = np.where(~df["offloaded_to_npu"], df["abs_signed_error_us"], 0.0)
    return df


def _summarize_operator_selection(op_rows: pd.DataFrame, device_label: str) -> pd.DataFrame:
    grouped = (
        op_rows.groupby(["op_type", "npu_category", "offloaded_to_npu"], dropna=False)
        .agg(
            row_count=("op_type", "size"),
            actual_us_sum=("actual_us_sum", "sum"),
            cpu_signed_error_us_sum=("signed_error_us", "sum"),
            cpu_abs_signed_error_us_sum=("abs_signed_error_us", "sum"),
            npu_abs_error_us_sum=("npu_abs_error_us", "sum"),
            npu_mean_ape=("npu_mean_ape", "mean"),
        )
        .reset_index()
    )
    total_actual = float(grouped["actual_us_sum"].sum())
    grouped["device_label"] = device_label
    grouped["actual_us_share"] = grouped["actual_us_sum"] / total_actual if total_actual > 0 else np.nan
    return grouped[
        [
            "device_label",
            "op_type",
            "npu_category",
            "offloaded_to_npu",
            "row_count",
            "actual_us_sum",
            "actual_us_share",
            "cpu_signed_error_us_sum",
            "cpu_abs_signed_error_us_sum",
            "npu_mean_ape",
            "npu_abs_error_us_sum",
        ]
    ].sort_values(["device_label", "actual_us_sum"], ascending=[True, False])


def _build_hybrid_rows(device: DeviceInput, route: str, offloaded_ops: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    combo = _load_combo_metrics(device.root, route)
    op_rows = _load_operator_errors(device.root, route, offloaded_ops)

    agg = (
        op_rows.groupby(KEY_COLUMNS, dropna=False)
        .agg(
            offloaded_actual_us_sum=("actual_us_sum", lambda s: s[op_rows.loc[s.index, "offloaded_to_npu"]].sum()),
            cpu_signed_error_offloaded_us=("cpu_signed_error_offloaded_us", "sum"),
            cpu_abs_error_offloaded_us=("cpu_abs_error_offloaded_us", "sum"),
            cpu_abs_error_non_offloaded_us=("cpu_abs_error_non_offloaded_us", "sum"),
            npu_abs_error_us=("npu_abs_error_us", "sum"),
        )
        .reset_index()
    )

    merged = combo.merge(agg, on=KEY_COLUMNS, how="left")
    fill_cols = [
        "offloaded_actual_us_sum",
        "cpu_signed_error_offloaded_us",
        "cpu_abs_error_offloaded_us",
        "cpu_abs_error_non_offloaded_us",
        "npu_abs_error_us",
    ]
    merged[fill_cols] = merged[fill_cols].fillna(0.0)

    signs = [
        _sign_with_fallback(offload_signed, combo_signed)
        for offload_signed, combo_signed in zip(
            merged["cpu_signed_error_offloaded_us"], merged["cpu_signed_error_us"]
        )
    ]
    merged["npu_replacement_sign"] = [item[0] for item in signs]
    merged["npu_replacement_sign_source"] = [item[1] for item in signs]
    merged["hybrid_signed_error_us"] = (
        merged["cpu_signed_error_us"]
        - merged["cpu_signed_error_offloaded_us"]
        + merged["npu_replacement_sign"] * merged["npu_abs_error_us"]
    )
    merged["hybrid_predicted_e2e_us"] = merged["actual_e2e_us"] + merged["hybrid_signed_error_us"]
    merged["hybrid_relative_error"] = (
        merged["hybrid_signed_error_us"].abs() / merged["actual_e2e_us"]
    ).replace([np.inf, -np.inf], np.nan)
    merged["hybrid_upper_bound_relative_error"] = (
        (merged["cpu_abs_error_non_offloaded_us"] + merged["npu_abs_error_us"])
        / merged["actual_e2e_us"]
    ).replace([np.inf, -np.inf], np.nan)
    merged["offloaded_actual_us_share_of_combo"] = (
        merged["offloaded_actual_us_sum"] / merged["actual_e2e_us"]
    ).replace([np.inf, -np.inf], np.nan)
    merged["device_label"] = device.label
    merged["device_root"] = str(device.root)
    selection = _summarize_operator_selection(op_rows, device.label)
    return merged, selection


def _cdf_points(values: Iterable[float]) -> pd.DataFrame:
    arr = np.asarray([x for x in values if pd.notna(x)], dtype=float)
    arr.sort()
    if len(arr) == 0:
        return pd.DataFrame({"relative_error": [], "cdf_percent": []})
    y = np.arange(1, len(arr) + 1, dtype=float) / len(arr) * 100.0
    return pd.DataFrame({"relative_error": arr, "cdf_percent": y})


def _summarize(values: pd.Series, thresholds: list[float]) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    summary: dict[str, float] = {"combo_count": float(len(values))}
    if values.empty:
        for key in ["mape", "median_ape", "p90_ape", "p95_ape", "max_ape"]:
            summary[key] = math.nan
        return summary
    summary.update(
        {
            "mape": float(values.mean()),
            "median_ape": float(values.median()),
            "p90_ape": float(values.quantile(0.90)),
            "p95_ape": float(values.quantile(0.95)),
            "max_ape": float(values.max()),
        }
    )
    for threshold in thresholds:
        summary[f"cdf_at_{threshold:g}"] = float(values.le(threshold).mean())
    return summary


def _write_mapping(out_dir: Path, offloaded_ops: set[str], offload_profile: str) -> pd.DataFrame:
    rows = []
    for op_type, category in sorted(OP_TO_NPU_CATEGORY.items()):
        profile = NPU_ERROR_PROFILE.get(category, {})
        rows.append(
            {
                "offload_profile": offload_profile,
                "op_type": op_type,
                "npu_category": category,
                "offloaded_to_npu": op_type in offloaded_ops,
                "npu_mean_ape": profile.get("mean_ape", np.nan),
                "npu_weighted_ape": profile.get("weighted_ape", np.nan),
                "npu_median_ape": profile.get("median_ape", np.nan),
                "npu_p90_ape": profile.get("p90_ape", np.nan),
                "npu_sample_count": profile.get("sample_count", np.nan),
            }
        )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(out_dir / "npu_operator_category_mapping.csv", index=False)
    return mapping


def _plot_cdf(
    points: pd.DataFrame,
    thresholds: list[float],
    out_dir: Path,
    offload_profile: str,
    *,
    scenario_filter: str | None = None,
    output_stem: str = "hybrid_e2e_error_cdf",
    title: str = "Experimental CPU+NPU Hybrid E2E Error CDF",
    annotate_metrics: bool = False,
) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plot_points = points.copy()
    if scenario_filter is not None:
        plot_points = plot_points[plot_points["scenario"].eq(scenario_filter)].copy()
    colors = {"Device_1": "#1f77b4", "Device_2": "#d62728"}
    line_styles = {
        "CPU baseline": "-",
        f"CPU+NPU {offload_profile.replace('_', '-')}": "--",
    }
    for (device_label, scenario), sub in plot_points.groupby(["device_label", "scenario"], sort=False):
        label = device_label if scenario_filter is not None else f"{device_label} {scenario}"
        ax.plot(
            sub["relative_error"] * 100.0,
            sub["cdf_percent"],
            label=label,
            color=colors.get(device_label),
            linestyle=line_styles.get(scenario, "-"),
            linewidth=2.0,
        )
        if annotate_metrics:
            color = colors.get(device_label)
            marker = "o" if device_label == "Device_1" else "s"
            for threshold_index, threshold in enumerate(thresholds):
                x = threshold * 100.0
                y = float(sub["relative_error"].le(threshold).mean() * 100.0)
                ax.scatter(
                    [x],
                    [y],
                    s=34,
                    marker=marker,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )
                offset_map = {
                    (0, "Device_1"): (30, -8),
                    (0, "Device_2"): (30, -38),
                    (1, "Device_1"): (34, -6),
                    (1, "Device_2"): (34, -42),
                }
                offset = offset_map.get((threshold_index, device_label), (20, -10))
                ax.annotate(
                    f"{device_label} @ {threshold:.0%}: {y:.1f}%",
                    xy=(x, y),
                    xytext=offset,
                    textcoords="offset points",
                    ha="left",
                    va="bottom" if offset[1] >= 0 else "top",
                    color=color,
                    bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": color, "alpha": 0.92},
                    arrowprops={
                        "arrowstyle": "-",
                        "color": color,
                        "lw": 0.9,
                        "shrinkA": 0.0,
                        "shrinkB": 3.0,
                    },
                )
    for threshold in thresholds:
        ax.axvline(threshold * 100.0, color="#777777", linestyle=":", linewidth=1.0)
        ax.text(
            threshold * 100.0,
            3.0,
            f"{threshold:.0%}",
            rotation=90,
            va="bottom",
            ha="right",
            color="#555555",
            fontsize=8,
        )
    ax.set_title(title)
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    if annotate_metrics:
        metric_lines = ["MAPE"]
        for device_label, sub in plot_points.groupby("device_label", sort=False):
            metric_lines.append(f"{device_label}: {sub['relative_error'].mean():.2%}")
        ax.text(
            0.975,
            0.34,
            "\n".join(metric_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#BBBBBB", "alpha": 0.92},
        )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{output_stem}.png", dpi=180)
    fig.savefig(out_dir / f"{output_stem}.pdf")
    plt.close(fig)


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    thresholds: list[float],
) -> None:
    summary_view = summary.copy()
    for col in ["mape", "median_ape", "p90_ape", "p95_ape", "max_ape"] + [
        f"cdf_at_{t:g}" for t in thresholds
    ]:
        if col in summary_view.columns:
            summary_view[col] = summary_view[col].map(_format_pct)
    summary_md = _to_markdown_table(summary_view)

    offload = selection[selection["offloaded_to_npu"]].copy()
    offload_summary = (
        offload.groupby("device_label", dropna=False)
        .agg(
            offloaded_op_types=("op_type", lambda s: ", ".join(sorted(s.unique()))),
            offloaded_actual_share=("actual_us_share", "sum"),
            npu_abs_error_us_sum=("npu_abs_error_us_sum", "sum"),
        )
        .reset_index()
    )
    if not offload_summary.empty:
        offload_summary["offloaded_actual_share"] = offload_summary["offloaded_actual_share"].map(_format_pct)
    offload_md = _to_markdown_table(offload_summary) if not offload_summary.empty else "No offloaded ops."

    cmd = " ".join(
        [
            "python",
            "calibrator/experimental/plot_cpu_npu_hybrid_e2e_error_cdf.py",
            "--device1-root",
            str(args.device1_root),
            "--device2-root",
            str(args.device2_root),
            "--route",
            args.route,
            "--offload-profile",
            args.offload_profile,
            "--out-dir",
            str(args.out_dir),
        ]
    )
    report = f"""# CPU+NPU Hybrid E2E Error CDF Report

## 实验口径

本报告是独立后处理实验，不修改 CPU 主线模型、训练流程、预测流程或现有 CDF 结果。实验只读取已有 CPU full E2E 结果，然后把指定 offload 算子的 CPU 误差替换为 NPU 性能模型类别平均误差。

该实验评估的是误差传播口径，不代表 NPU 加速收益。正式异构性能预测仍需要 CPU/NPU placement、device transfer cost 和异构 E2E scheduler。

## Offload 策略

- `offload_profile`: `{args.offload_profile}`
- 默认展示策略 `compute_island` 只 offload dense / activation / tensor transform 计算岛。
- sparse embedding 分支默认保留在 CPU 上，避免在没有 NPU embedding table 驻留和 sparse memory 策略时误读结果。

## 误差替换公式

对每个 E2E combo，先从 CPU 结果得到 signed error：

$$e_{{CPU}}=\\hat T_{{CPU}}-T_{{actual}}$$

对 offload 算子集合 $O$，移除原 CPU signed error，再加入 NPU 类别平均误差估计：

$$e_{{hybrid}}=e_{{CPU}}-\\sum_{{o\\in O}}e_{{CPU,o}}+s\\sum_{{o\\in O}}T_{{actual,o}}\\cdot APE_{{NPU,cat(o)}}$$

其中 $s$ 优先使用 offload 算子 CPU signed error 的符号；若为 0，则回退到 combo 级 CPU signed error 的符号。CDF 使用：

$$APE_{{hybrid}}=\\frac{{|e_{{hybrid}}|}}{{T_{{actual}}}}$$

## CDF 汇总

{summary_md}

## Offload 覆盖

{offload_md}

## 输出文件

- `hybrid_e2e_error_rows.csv`
- `hybrid_e2e_error_cdf_points.csv`
- `hybrid_e2e_error_cdf_summary.csv`
- `npu_operator_category_mapping.csv`
- `npu_offload_selection_summary.csv`
- `hybrid_e2e_error_cdf.png`
- `hybrid_e2e_error_cdf.pdf`

## 单独复现命令

```bash
# Run from the repository root

{cmd}
```
"""
    (out_dir / "HYBRID_CPU_NPU_E2E_CDF_REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device1-root", type=Path, required=True)
    parser.add_argument("--device2-root", type=Path, required=True)
    parser.add_argument("--device1-label", default="Device_1")
    parser.add_argument("--device2-label", default="Device_2")
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--offload-profile", choices=sorted(OFFLOAD_PROFILES), default="compute_island")
    parser.add_argument("--thresholds", default="0.10,0.15")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_thresholds(args.thresholds)
    offloaded_ops = OFFLOAD_PROFILES[args.offload_profile]
    _write_mapping(out_dir, offloaded_ops, args.offload_profile)

    devices = [
        DeviceInput(args.device1_label, args.device1_root),
        DeviceInput(args.device2_label, args.device2_root),
    ]

    all_rows = []
    all_selection = []
    for device in devices:
        rows, selection = _build_hybrid_rows(device, args.route, offloaded_ops)
        all_rows.append(rows)
        all_selection.append(selection)
    hybrid_rows = pd.concat(all_rows, ignore_index=True)
    selection_rows = pd.concat(all_selection, ignore_index=True)

    hybrid_rows.to_csv(out_dir / "hybrid_e2e_error_rows.csv", index=False)
    selection_rows.to_csv(out_dir / "npu_offload_selection_summary.csv", index=False)

    point_frames = []
    summary_rows = []
    scenario_cols = [
        ("CPU baseline", "cpu_relative_error"),
        (f"CPU+NPU {args.offload_profile.replace('_', '-')}", "hybrid_relative_error"),
    ]
    for device_label, group in hybrid_rows.groupby("device_label", sort=False):
        for scenario_name, value_col in scenario_cols:
            points = _cdf_points(group[value_col])
            points["device_label"] = device_label
            points["scenario"] = scenario_name
            point_frames.append(points)
            row = {
                "device_label": device_label,
                "scenario": scenario_name,
                **_summarize(group[value_col], thresholds),
            }
            summary_rows.append(row)

    cdf_points = pd.concat(point_frames, ignore_index=True)
    cdf_points = cdf_points[["device_label", "scenario", "relative_error", "cdf_percent"]]
    cdf_points.to_csv(out_dir / "hybrid_e2e_error_cdf_points.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "hybrid_e2e_error_cdf_summary.csv", index=False)

    hybrid_scenario = f"CPU+NPU {args.offload_profile.replace('_', '-')}"
    _plot_cdf(cdf_points, thresholds, out_dir, args.offload_profile)
    _plot_cdf(
        cdf_points,
        thresholds,
        out_dir,
        args.offload_profile,
        scenario_filter=hybrid_scenario,
        output_stem="hybrid_e2e_error_cdf_cpu_npu_only",
        title="Experimental CPU+NPU E2E Error CDF",
        annotate_metrics=True,
    )
    _write_report(out_dir, args, summary, selection_rows, thresholds)

    manifest = {
        "route": args.route,
        "offload_profile": args.offload_profile,
        "device1_root": str(args.device1_root),
        "device2_root": str(args.device2_root),
        "thresholds": thresholds,
        "output_files": [
            "hybrid_e2e_error_rows.csv",
            "hybrid_e2e_error_cdf_points.csv",
            "hybrid_e2e_error_cdf_summary.csv",
            "npu_operator_category_mapping.csv",
            "npu_offload_selection_summary.csv",
            "hybrid_e2e_error_cdf.png",
            "hybrid_e2e_error_cdf.pdf",
            "hybrid_e2e_error_cdf_cpu_npu_only.png",
            "hybrid_e2e_error_cdf_cpu_npu_only.pdf",
            "HYBRID_CPU_NPU_E2E_CDF_REPORT.md",
        ],
        "note": "Independent post-processing experiment; no CPU mainline model outputs are modified.",
    }
    (out_dir / "hybrid_e2e_error_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
