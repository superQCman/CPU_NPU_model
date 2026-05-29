#!/usr/bin/env python3
"""Dense batch/NIP trend validation using trained v3 model bundles.

This is an independent CPU+NPU experimental branch.  It does not read train/test
rows and does not refit the CPU model.  It loads exported v3 bundles, generates
synthetic DLRM top-level configurations, runs static E2E inference, and plots
normalized sensitivity over batch size and number of indices per lookup.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
)
from calibrator.static_feature_mapping.core import (  # noqa: E402
    FULL_GRAPH_76,
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


DEVICE_ORDER = ["Device_1", "Device_2"]
DEVICE_COLORS = {"Device_1": "#b65a2a", "Device_2": "#2f5f9e"}
DEVICE_MARKERS = {"Device_1": "o", "Device_2": "s"}


@dataclass(frozen=True)
class BundleInput:
    label: str
    path: Path


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"empty integer list: {raw!r}")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device1-bundle",
        type=Path,
        default=Path(
            "outputs/cache_residency_worker_share_v3_full_kunpeng920_20260505/model_bundles/"
            "joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl"
        ),
    )
    parser.add_argument(
        "--device2-bundle",
        type=Path,
        default=Path(
            "outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/"
            "joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl"
        ),
    )
    parser.add_argument("--device1-label", default="Device_1")
    parser.add_argument("--device2-label", default="Device_2")
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--offload-profile", choices=sorted(OFFLOAD_PROFILES), default="compute_island")
    parser.add_argument(
        "--batch-values",
        default="512,768,1024,1280,1536,1792,2048,2560,3072,3584,4096",
    )
    parser.add_argument(
        "--nip-values",
        default="200,400,600,800,1000,1200,1400,1600,1800,2000,2400,2800,3200",
    )
    parser.add_argument("--fixed-batch-size", type=int, default=2048)
    parser.add_argument("--fixed-nip", type=int, default=1600)
    parser.add_argument("--arch-embedding-size", default="65536")
    parser.add_argument("--arch-mlp-bot", default="512-2048-512")
    parser.add_argument("--arch-mlp-top", default="548-2048-1")
    parser.add_argument("--inter-threads", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=3)
    parser.add_argument("--style-reference-dir", type=Path, default=Path(""))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/experimental_cpu_npu_hybrid_batch_nip_model_inference_20260510"),
    )
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
    return float(np.mean(np.diff(arr) >= -1e-12))


def _apply_trend_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _config_for_axis(args: argparse.Namespace, axis: str, value: int) -> DlrmStaticConfig:
    batch = int(value) if axis == "batch_size" else int(args.fixed_batch_size)
    nip = int(value) if axis == "num_indices_per_lookup" else int(args.fixed_nip)
    return DlrmStaticConfig(
        batch_size=batch,
        num_indices_per_lookup=nip,
        arch_embedding_size=str(args.arch_embedding_size),
        arch_mlp_bot=str(args.arch_mlp_bot),
        arch_mlp_top=str(args.arch_mlp_top),
        inter_threads=int(args.inter_threads),
        num_threads=int(args.num_threads),
    )


def _predict_one(bundle: dict[str, Any], config: DlrmStaticConfig, offloaded_ops: set[str]) -> dict[str, Any]:
    op_shapes = build_static_op_shape_table(config)
    full_graph = build_static_feature_frame(config, graph_scope=FULL_GRAPH_76)
    model_input = build_static_feature_frame(config, graph_scope=MODEL_INPUT_51)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        prepared, pred = _predict_with_bundle(bundle, model_input)
    predictions = prepared.copy()
    predictions["pred_us"] = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    predictions["pred_total_us"] = predictions["pred_us"]
    e2e_summary, _ = _schedule_static_e2e(config=config, op_shapes=op_shapes, predictions=predictions)
    offloaded = predictions[predictions["op_type"].astype(str).isin(offloaded_ops)]
    offloaded_pred_sum = float(pd.to_numeric(offloaded["pred_us"], errors="coerce").fillna(0.0).sum())
    predicted_e2e = float(e2e_summary["predicted_e2e_us"])
    return {
        "batch_size": int(config.batch_size),
        "num_indices_per_lookup": int(config.num_indices_per_lookup),
        "inter_threads": int(config.inter_threads),
        "num_threads": int(config.num_threads),
        "model_input_rows": int(len(model_input)),
        "full_graph_rows": int(len(full_graph)),
        "operator_pred_us_sum": float(predictions["pred_us"].sum()),
        "offloaded_operator_pred_us_sum": offloaded_pred_sum,
        "offloaded_operator_pred_share_of_e2e": offloaded_pred_sum / predicted_e2e if predicted_e2e > 0 else math.nan,
        **e2e_summary,
    }


def _run_inference(args: argparse.Namespace) -> pd.DataFrame:
    batch_values = _parse_int_list(args.batch_values)
    nip_values = _parse_int_list(args.nip_values)
    offloaded_ops = OFFLOAD_PROFILES[args.offload_profile]
    devices = [
        BundleInput(str(args.device1_label), args.device1_bundle),
        BundleInput(str(args.device2_label), args.device2_bundle),
    ]
    rows: list[dict[str, Any]] = []
    for device in devices:
        bundle = _load_model_bundle(device.path)
        route = str(bundle.get("route", ""))
        if route and route != str(args.route):
            raise ValueError(f"{device.label} bundle route mismatch: {route!r} != {args.route!r}")
        for axis, values in [("batch_size", batch_values), ("num_indices_per_lookup", nip_values)]:
            for value in values:
                config = _config_for_axis(args, axis, int(value))
                row = _predict_one(bundle, config, offloaded_ops)
                row.update(
                    {
                        "device_label": device.label,
                        "bundle_path": str(device.path),
                        "model_sha256": str(bundle.get("model_sha256", "")),
                        "sweep_axis": axis,
                        "axis_value": int(value),
                        "offload_profile": str(args.offload_profile),
                    }
                )
                rows.append(row)
    points = pd.DataFrame(rows)
    norm_frames: list[pd.DataFrame] = []
    for (device_label, axis), part in points.groupby(["device_label", "sweep_axis"], sort=False):
        part = part.sort_values("axis_value").copy()
        baseline = float(part["predicted_e2e_us"].iloc[0])
        part["baseline_axis_value"] = int(part["axis_value"].iloc[0])
        part["normalized_predicted_e2e"] = part["predicted_e2e_us"] / baseline if baseline > 0 else np.nan
        norm_frames.append(part)
    return pd.concat(norm_frames, ignore_index=True)


def _max_change(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (device_label, axis), part in points.groupby(["device_label", "sweep_axis"], sort=False):
        part = part.sort_values("axis_value")
        values = pd.to_numeric(part["predicted_e2e_us"], errors="coerce").dropna()
        min_value = float(values.min())
        max_value = float(values.max())
        min_row = part.loc[part["predicted_e2e_us"].idxmin()]
        max_row = part.loc[part["predicted_e2e_us"].idxmax()]
        rows.append(
            {
                "device_label": device_label,
                "sweep_axis": axis,
                "baseline_axis_value": int(part["baseline_axis_value"].iloc[0]),
                "min_sweep_axis_value": int(part["axis_value"].min()),
                "max_sweep_axis_value": int(part["axis_value"].max()),
                "min_latency_axis_value": int(min_row["axis_value"]),
                "max_latency_axis_value": int(max_row["axis_value"]),
                "max_change_pct": ((max_value - min_value) / min_value * 100.0) if min_value > 0 else math.nan,
                "point_count": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _audit(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (device_label, axis), part in points.groupby(["device_label", "sweep_axis"], sort=False):
        ordered = part.sort_values("axis_value")
        rows.append(
            {
                "device_label": device_label,
                "sweep_axis": axis,
                "point_count": int(len(ordered)),
                "spearman": _spearman_corr(ordered["axis_value"], ordered["predicted_e2e_us"]),
                "nondecreasing_pass_rate": _nondecreasing_pass_rate(ordered["predicted_e2e_us"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_axis(
    points: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    axis: str,
    title: str,
    xlabel: str,
    stem: str,
    annotation: str,
    out_dir: Path,
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
            part["normalized_predicted_e2e"],
            marker=DEVICE_MARKERS.get(device_label, "o"),
            color=DEVICE_COLORS.get(device_label, "#2f5f9e"),
            linestyle="-",
            alpha=1.0,
            label=device_label,
        )
        change_row = changes.loc[changes["device_label"].eq(device_label) & changes["sweep_axis"].eq(axis)]
        if not change_row.empty:
            row = change_row.iloc[0]
            last = part.iloc[-1]
            if device_label == "Device_1":
                xytext = (-18, 24)
                ha = "right"
                va = "bottom"
            else:
                xytext = (-18, -24)
                ha = "right"
                va = "top"
            ax.annotate(
                f"max change {float(row['max_change_pct']):.1f}%",
                xy=(float(last["axis_value"]), float(last["normalized_predicted_e2e"])),
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
                va=va,
                color=DEVICE_COLORS.get(device_label, "0.2"),
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.22",
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
    ax.text(
        0.5,
        -0.25,
        annotation,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
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


def _write_report(args: argparse.Namespace, points: pd.DataFrame, changes: pd.DataFrame, audit: pd.DataFrame) -> None:
    change_view = changes[
        ["device_label", "sweep_axis", "min_sweep_axis_value", "max_sweep_axis_value", "max_change_pct"]
    ].copy()
    change_view["max_change_pct"] = change_view["max_change_pct"].map(lambda value: f"{float(value):.1f}%")
    audit_view = audit.copy()
    for col in ["spearman", "nondecreasing_pass_rate"]:
        audit_view[col] = audit_view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    report = f"""# CPU+NPU Batch/NIP Dense Model-Inference Sensitivity Report

## 实验口径

该实验是独立 CPU+NPU 合成验证分支。脚本不读取训练/测试 combo，不重新训练模型，而是直接加载已导出的 Device_1 / Device_2 v3 bundle，对更密集的 batch size 和 `num_indices_per_lookup` 采样点执行静态配置推理。

当前没有真实 NPU latency model 和 CPU+NPU E2E 实测，因此本报告只验证训练后 E2E 性能模型对上层负载变化的响应趋势。`offload_profile={args.offload_profile}` 作为实验元数据保留，不用于声称 NPU 加速收益。

## 采样设置

- batch values: `{args.batch_values}`
- NIP values: `{args.nip_values}`
- fixed batch for NIP sweep: `{args.fixed_batch_size}`
- fixed NIP for batch sweep: `{args.fixed_nip}`
- arch embedding size: `{args.arch_embedding_size}`
- arch mlp bot: `{args.arch_mlp_bot}`
- arch mlp top: `{args.arch_mlp_top}`
- inter threads: `{args.inter_threads}`
- num threads: `{args.num_threads}`

## 最大变化百分比

{_markdown_table(change_view)}

## 趋势审计

{_markdown_table(audit_view)}

## 输出文件

- `hybrid_batch_size_model_inference_sensitivity.png`
- `hybrid_batch_size_model_inference_sensitivity.pdf`
- `hybrid_nip_model_inference_sensitivity.png`
- `hybrid_nip_model_inference_sensitivity.pdf`
- `hybrid_batch_nip_model_inference_points.csv`
- `hybrid_batch_nip_model_inference_max_change_pct.csv`
- `hybrid_batch_nip_model_inference_audit.csv`
"""
    (args.out_dir / "HYBRID_CPU_NPU_BATCH_NIP_MODEL_INFERENCE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = _run_inference(args)
    changes = _max_change(points)
    audit = _audit(points)
    points.to_csv(args.out_dir / "hybrid_batch_nip_model_inference_points.csv", index=False)
    changes.to_csv(args.out_dir / "hybrid_batch_nip_model_inference_max_change_pct.csv", index=False)
    audit.to_csv(args.out_dir / "hybrid_batch_nip_model_inference_audit.csv", index=False)
    _plot_axis(
        points,
        changes,
        axis="batch_size",
        title="CPU+NPU Hybrid Batch Size Sensitivity",
        xlabel="Batch size",
        stem="hybrid_batch_size_model_inference_sensitivity",
        annotation="Dense static-config inference | normalized by minimum batch size per device",
        out_dir=args.out_dir,
    )
    _plot_axis(
        points,
        changes,
        axis="num_indices_per_lookup",
        title="CPU+NPU Hybrid Lookup Intensity Sensitivity",
        xlabel="Number of indices per lookup",
        stem="hybrid_nip_model_inference_sensitivity",
        annotation="Dense static-config inference | normalized by minimum NIP per device",
        out_dir=args.out_dir,
    )
    _write_report(args, points, changes, audit)
    manifest = {
        "route": args.route,
        "offload_profile": args.offload_profile,
        "device1_bundle": str(args.device1_bundle),
        "device2_bundle": str(args.device2_bundle),
        "style_reference_dir": str(args.style_reference_dir),
        "batch_values": _parse_int_list(args.batch_values),
        "nip_values": _parse_int_list(args.nip_values),
        "note": "Dense synthetic top-level DLRM config inference; no train/test rows are read.",
    }
    (args.out_dir / "hybrid_batch_nip_model_inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
