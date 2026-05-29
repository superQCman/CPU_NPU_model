#!/usr/bin/env python3
"""Experimental CPU+NPU batch/NIP sensitivity plots.

This is a standalone post-processing experiment.  It reads existing CPU full
E2E outputs, applies the CPU+NPU compute-island error replacement used by the
hybrid CDF experiment, and plots normalized E2E sensitivity over batch size and
number of indices per lookup.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrator.experimental.plot_cpu_npu_hybrid_e2e_error_cdf import (  # noqa: E402
    DEFAULT_ROUTE,
    OFFLOAD_PROFILES,
    DeviceInput,
    _build_hybrid_rows,
)


DEVICE_ORDER = ["Device_1", "Device_2"]
DEVICE_COLORS = {"Device_1": "#b65a2a", "Device_2": "#2f5f9e"}
DEVICE_MARKERS = {"Device_1": "o", "Device_2": "s"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device1-root", type=Path, required=True)
    parser.add_argument("--device2-root", type=Path, required=True)
    parser.add_argument("--device1-label", default="Device_1")
    parser.add_argument("--device2-label", default="Device_2")
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--offload-profile", choices=sorted(OFFLOAD_PROFILES), default="compute_island")
    parser.add_argument("--style-reference-dir", type=Path, default=Path(""))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _spearman_corr(x: Iterable[float], y: Iterable[float]) -> float:
    xs = pd.Series(list(x), dtype="float64")
    ys = pd.Series(list(y), dtype="float64")
    valid = xs.notna() & ys.notna()
    if int(valid.sum()) < 2:
        return math.nan
    if xs[valid].nunique() < 2 or ys[valid].nunique() < 2:
        return math.nan
    return float(xs[valid].corr(ys[valid], method="spearman"))


def _nondecreasing_pass_rate(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if pd.notna(v)], dtype=float)
    if len(arr) <= 1:
        return math.nan
    diffs = np.diff(arr)
    return float(np.mean(diffs >= -1e-12))


def _load_hybrid_rows(args: argparse.Namespace) -> pd.DataFrame:
    offloaded_ops = OFFLOAD_PROFILES[args.offload_profile]
    devices = [
        DeviceInput(str(args.device1_label), args.device1_root),
        DeviceInput(str(args.device2_label), args.device2_root),
    ]
    frames: list[pd.DataFrame] = []
    for device in devices:
        rows, _ = _build_hybrid_rows(device, args.route, offloaded_ops)
        frames.append(rows)
    combined = pd.concat(frames, ignore_index=True)
    combined["batch_size"] = pd.to_numeric(combined["batch_size"], errors="coerce")
    combined["num_indices_per_lookup"] = pd.to_numeric(combined["num_indices_per_lookup"], errors="coerce")
    combined["inter_threads"] = pd.to_numeric(combined["inter_threads"], errors="coerce")
    combined["hybrid_predicted_e2e_us"] = pd.to_numeric(combined["hybrid_predicted_e2e_us"], errors="coerce")
    return combined


def _summarize_axis(rows: pd.DataFrame, axis: str) -> pd.DataFrame:
    grouped = (
        rows.dropna(subset=[axis, "hybrid_predicted_e2e_us"])
        .groupby(["device_label", axis], dropna=False)
        .agg(
            combo_count=("combo", "size"),
            mean_hybrid_predicted_e2e_us=("hybrid_predicted_e2e_us", "mean"),
            median_hybrid_relative_error=("hybrid_relative_error", "median"),
            mean_hybrid_relative_error=("hybrid_relative_error", "mean"),
        )
        .reset_index()
        .rename(columns={axis: "axis_value"})
    )
    grouped["sweep_axis"] = axis
    point_frames: list[pd.DataFrame] = []
    for device_label, part in grouped.groupby("device_label", sort=False):
        part = part.sort_values("axis_value").copy()
        if part.empty:
            continue
        baseline = float(part["mean_hybrid_predicted_e2e_us"].iloc[0])
        part["baseline_axis_value"] = float(part["axis_value"].iloc[0])
        part["normalized_hybrid_e2e"] = (
            part["mean_hybrid_predicted_e2e_us"] / baseline if baseline > 0 else np.nan
        )
        point_frames.append(part)
    if not point_frames:
        return pd.DataFrame()
    return pd.concat(point_frames, ignore_index=True)


def _max_change(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (device_label, axis), part in points.groupby(["device_label", "sweep_axis"], sort=False):
        part = part.sort_values("axis_value")
        values = pd.to_numeric(part["mean_hybrid_predicted_e2e_us"], errors="coerce").dropna()
        if values.empty:
            continue
        min_value = float(values.min())
        max_value = float(values.max())
        change_pct = ((max_value - min_value) / min_value * 100.0) if min_value > 0 else math.nan
        min_row = part.loc[part["mean_hybrid_predicted_e2e_us"].idxmin()]
        max_row = part.loc[part["mean_hybrid_predicted_e2e_us"].idxmax()]
        rows.append(
            {
                "device_label": device_label,
                "sweep_axis": axis,
                "baseline_axis_value": float(part["axis_value"].iloc[0]),
                "min_latency_axis_value": float(min_row["axis_value"]),
                "max_latency_axis_value": float(max_row["axis_value"]),
                "min_sweep_axis_value": float(part["axis_value"].min()),
                "max_sweep_axis_value": float(part["axis_value"].max()),
                "max_change_pct": change_pct,
                "point_count": int(len(part)),
                "combo_count_total": int(part["combo_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _audit_axis(rows: pd.DataFrame, points: pd.DataFrame, axis: str) -> pd.DataFrame:
    audit_rows: list[dict[str, object]] = []
    for device_label, part in points.loc[points["sweep_axis"].eq(axis)].groupby("device_label", sort=False):
        ordered = part.sort_values("axis_value")
        audit_rows.append(
            {
                "device_label": device_label,
                "sweep_axis": axis,
                "audit_scope": "aggregated_axis",
                "group_count": 1,
                "point_count": int(len(ordered)),
                "spearman": _spearman_corr(ordered["axis_value"], ordered["mean_hybrid_predicted_e2e_us"]),
                "nondecreasing_pass_rate": _nondecreasing_pass_rate(ordered["mean_hybrid_predicted_e2e_us"]),
            }
        )

    if axis == "batch_size":
        group_keys = ["device_label", "case_id", "source_name", "num_indices_per_lookup", "inter_threads"]
    else:
        group_keys = ["device_label", "case_id", "source_name", "batch_size", "inter_threads"]

    matched_rows: list[dict[str, object]] = []
    for keys, group in rows.dropna(subset=[axis, "hybrid_predicted_e2e_us"]).groupby(group_keys, dropna=False):
        del keys
        ordered = group.sort_values(axis)
        if ordered[axis].nunique() < 3:
            continue
        matched_rows.append(
            {
                "spearman": _spearman_corr(ordered[axis], ordered["hybrid_predicted_e2e_us"]),
                "nondecreasing_pass_rate": _nondecreasing_pass_rate(ordered["hybrid_predicted_e2e_us"]),
                "point_count": int(len(ordered)),
            }
        )
    matched = pd.DataFrame(matched_rows)
    if matched.empty:
        for device_label in DEVICE_ORDER:
            audit_rows.append(
                {
                    "device_label": device_label,
                    "sweep_axis": axis,
                    "audit_scope": "matched_case_thread_groups",
                    "group_count": 0,
                    "point_count": 0,
                    "spearman": math.nan,
                    "nondecreasing_pass_rate": math.nan,
                }
            )
    else:
        for device_label, group in rows.groupby("device_label", sort=False):
            del group
        # Recompute per device to keep the output easy to read.
        for device_label in DEVICE_ORDER:
            device_matched: list[dict[str, object]] = []
            device_rows = rows.loc[rows["device_label"].eq(device_label)]
            for _, group in device_rows.dropna(subset=[axis, "hybrid_predicted_e2e_us"]).groupby(
                group_keys[1:], dropna=False
            ):
                ordered = group.sort_values(axis)
                if ordered[axis].nunique() < 3:
                    continue
                device_matched.append(
                    {
                        "spearman": _spearman_corr(ordered[axis], ordered["hybrid_predicted_e2e_us"]),
                        "nondecreasing_pass_rate": _nondecreasing_pass_rate(ordered["hybrid_predicted_e2e_us"]),
                        "point_count": int(len(ordered)),
                    }
                )
            device_frame = pd.DataFrame(device_matched)
            audit_rows.append(
                {
                    "device_label": device_label,
                    "sweep_axis": axis,
                    "audit_scope": "matched_case_thread_groups",
                    "group_count": int(len(device_frame)),
                    "point_count": int(device_frame["point_count"].sum()) if not device_frame.empty else 0,
                    "spearman": float(device_frame["spearman"].mean()) if not device_frame.empty else math.nan,
                    "nondecreasing_pass_rate": float(device_frame["nondecreasing_pass_rate"].mean())
                    if not device_frame.empty
                    else math.nan,
                }
            )
    return pd.DataFrame(audit_rows)


def _apply_trend_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_axis(
    points: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    axis: str,
    out_dir: Path,
    stem: str,
    title: str,
    xlabel: str,
    annotation: str,
) -> None:
    _apply_trend_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
    subset = points.loc[points["sweep_axis"].eq(axis)].copy()
    for device_label in DEVICE_ORDER:
        part = subset.loc[subset["device_label"].eq(device_label)].sort_values("axis_value")
        if part.empty:
            continue
        ax.plot(
            part["axis_value"],
            part["normalized_hybrid_e2e"],
            marker=DEVICE_MARKERS.get(device_label, "o"),
            color=DEVICE_COLORS.get(device_label, "#2f5f9e"),
            linestyle="-",
            alpha=1.0,
            label=device_label,
        )
        change_part = changes.loc[
            changes["device_label"].eq(device_label) & changes["sweep_axis"].eq(axis)
        ]
        if not change_part.empty:
            row = change_part.iloc[0]
            last = part.iloc[-1]
            ax.annotate(
                f"max change {float(row['max_change_pct']):.1f}%",
                xy=(float(last["axis_value"]), float(last["normalized_hybrid_e2e"])),
                xytext=(8, 0 if device_label == "Device_1" else -18),
                textcoords="offset points",
                ha="left",
                va="center",
                color=DEVICE_COLORS.get(device_label, "0.2"),
                fontsize=6.2,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": DEVICE_COLORS.get(device_label, "0.2"),
                    "alpha": 0.90,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.7,
                    "color": DEVICE_COLORS.get(device_label, "0.2"),
                    "shrinkA": 0.0,
                    "shrinkB": 2.5,
                },
            )
    base_value = float(subset["baseline_axis_value"].dropna().min()) if not subset.empty else math.nan
    if math.isfinite(base_value):
        ax.axvline(base_value, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized scheduled E2E time")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, color="0.85", linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    if annotation:
        ax.text(
            0.5,
            -0.25,
            annotation,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=5.15,
            color="0.25",
            linespacing=1.15,
            clip_on=False,
        )
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text = df.copy().fillna("")
    lines = [
        "| " + " | ".join(str(col) for col in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for row in text.to_numpy():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    args: argparse.Namespace,
    rows: pd.DataFrame,
    changes: pd.DataFrame,
    audit: pd.DataFrame,
    out_dir: Path,
) -> None:
    change_view = changes[
        ["device_label", "sweep_axis", "min_sweep_axis_value", "max_sweep_axis_value", "max_change_pct"]
    ].copy()
    change_view["max_change_pct"] = change_view["max_change_pct"].map(lambda v: f"{float(v):.1f}%")
    audit_view = audit[
        ["device_label", "sweep_axis", "audit_scope", "group_count", "spearman", "nondecreasing_pass_rate"]
    ].copy()
    for col in ["spearman", "nondecreasing_pass_rate"]:
        audit_view[col] = audit_view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
    combo_counts = rows.groupby("device_label").size().reset_index(name="combo_count")

    report = f"""# CPU+NPU Batch/NIP Sensitivity Report

## 实验口径

该实验是独立 CPU+NPU 合成验证分支，不修改 CPU 主线模型、不重新训练、不覆盖已有 CDF 或趋势图。图中展示的是 CPU+NPU `compute_island` 误差替换口径下，E2E 预测随上层负载参数变化的归一化趋势和最大变化百分比，不展示具体耗时。

本实验没有使用真实 CPU+NPU E2E 实测，因此不能解释为 NPU 加速收益；它只用于检查 batch size 和 `num_indices_per_lookup` 增大时，端到端预测趋势是否符合预期。

## 输入覆盖

{_markdown_table(combo_counts)}

## 最大变化百分比

{_markdown_table(change_view)}

## 趋势审计

{_markdown_table(audit_view)}

## 输出文件

- `hybrid_batch_size_sensitivity.png`
- `hybrid_batch_size_sensitivity.pdf`
- `hybrid_nip_sensitivity.png`
- `hybrid_nip_sensitivity.pdf`
- `hybrid_batch_nip_sensitivity_points.csv`
- `hybrid_batch_nip_max_change_pct.csv`
- `hybrid_batch_nip_sensitivity_audit.csv`

## 复现命令

```bash
# Run from the repository root

python calibrator/experimental/plot_cpu_npu_hybrid_batch_nip_sensitivity.py \\
  --device1-root {args.device1_root} \\
  --device2-root {args.device2_root} \\
  --route {args.route} \\
  --offload-profile {args.offload_profile} \\
  --style-reference-dir {args.style_reference_dir} \\
  --out-dir {args.out_dir}
```
"""
    (out_dir / "HYBRID_CPU_NPU_BATCH_NIP_SENSITIVITY_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_hybrid_rows(args)
    batch_points = _summarize_axis(rows, "batch_size")
    nip_points = _summarize_axis(rows, "num_indices_per_lookup")
    points = pd.concat([batch_points, nip_points], ignore_index=True)
    changes = _max_change(points)
    audit = pd.concat(
        [
            _audit_axis(rows, points, "batch_size"),
            _audit_axis(rows, points, "num_indices_per_lookup"),
        ],
        ignore_index=True,
    )

    point_cols = [
        "device_label",
        "sweep_axis",
        "axis_value",
        "baseline_axis_value",
        "normalized_hybrid_e2e",
        "combo_count",
        "median_hybrid_relative_error",
        "mean_hybrid_relative_error",
    ]
    points[point_cols].to_csv(out_dir / "hybrid_batch_nip_sensitivity_points.csv", index=False)
    changes.to_csv(out_dir / "hybrid_batch_nip_max_change_pct.csv", index=False)
    audit.to_csv(out_dir / "hybrid_batch_nip_sensitivity_audit.csv", index=False)

    _plot_axis(
        points,
        changes,
        axis="batch_size",
        out_dir=out_dir,
        stem="hybrid_batch_size_sensitivity",
        title="CPU+NPU Hybrid Batch Size Sensitivity",
        xlabel="Batch size",
        annotation="CPU+NPU compute-island | normalized by minimum batch size per device",
    )
    _plot_axis(
        points,
        changes,
        axis="num_indices_per_lookup",
        out_dir=out_dir,
        stem="hybrid_nip_sensitivity",
        title="CPU+NPU Hybrid Lookup Intensity Sensitivity",
        xlabel="Number of indices per lookup",
        annotation="CPU+NPU compute-island | normalized by minimum NIP per device",
    )
    _write_report(args=args, rows=rows, changes=changes, audit=audit, out_dir=out_dir)

    manifest = {
        "route": args.route,
        "offload_profile": args.offload_profile,
        "device1_root": str(args.device1_root),
        "device2_root": str(args.device2_root),
        "style_reference_dir": str(args.style_reference_dir),
        "combo_counts": rows.groupby("device_label").size().to_dict(),
        "note": "Independent synthetic CPU+NPU sensitivity experiment; no CPU mainline outputs are modified.",
    }
    (out_dir / "hybrid_batch_nip_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
