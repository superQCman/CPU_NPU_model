#!/usr/bin/env python3
"""Plot relative-error CDFs for graph-aggregated DLRM task classes.

The input is produced by ``calculate_graph_task_mape.py``.  Each input row is
one DLRM execution stage in one workload combination: its member operator
latencies have already been summed before relative error is calculated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASK_CLASS_ORDER = [
    "Embedding lookup + reduction",
    "Bottom MLP",
    "Feature interaction",
    "Top MLP",
]
TASK_CLASS_LABELS = {
    "Embedding lookup + reduction": "Embedding lookup\n+ reduction",
    "Bottom MLP": "Bottom MLP",
    "Feature interaction": "Feature interaction",
    "Top MLP": "Top MLP",
}
TASK_CLASS_COLORS = {
    "Embedding lookup + reduction": "#4C78A8",
    "Bottom MLP": "#F58518",
    "Feature interaction": "#54A24B",
    "Top MLP": "#C77DA8",
}
DEVICE_ORDER = ["Device_1", "Device_2"]


def _parse_thresholds(raw: str) -> list[float]:
    values = sorted({float(token.strip()) for token in raw.split(",") if token.strip()})
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("thresholds must be positive relative-error values")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CDFs for DLRM graph-aggregated task-class relative errors."
    )
    parser.add_argument(
        "--task-prediction-rows",
        required=True,
        type=Path,
        help="dlrm_task_class_prediction_rows.csv emitted by calculate_graph_task_mape.py",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--threshold",
        default="0.10,0.15",
        help="Comma-separated relative-error thresholds. Default: 0.10,0.15.",
    )
    parser.add_argument(
        "--title",
        default="DLRM task-class relative error CDF",
        help="Figure title.",
    )
    args = parser.parse_args()
    if not args.task_prediction_rows.is_file():
        parser.error(f"task prediction rows do not exist: {args.task_prediction_rows}")
    args.thresholds = _parse_thresholds(args.threshold)
    return args


def _load_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"device", "task_class", "actual_us", "pred_us"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    actual = pd.to_numeric(frame["actual_us"], errors="coerce")
    pred = pd.to_numeric(frame["pred_us"], errors="coerce")
    if not (np.isfinite(actual) & np.isfinite(pred) & actual.gt(0) & pred.ge(0)).all():
        raise ValueError(f"{path} contains invalid actual_us or pred_us values")
    frame = frame.assign(relative_error=(pred - actual).abs() / actual.replace(0.0, np.nan))
    frame = frame.loc[
        frame["device"].isin(DEVICE_ORDER)
        & frame["task_class"].isin(TASK_CLASS_ORDER)
        & np.isfinite(frame["relative_error"].to_numpy(dtype=float))
    ].copy()
    if frame.empty:
        raise ValueError(f"{path} has no finite rows for the expected task classes")
    expected = {(device, task) for device in DEVICE_ORDER for task in TASK_CLASS_ORDER}
    observed = set(map(tuple, frame[["device", "task_class"]].drop_duplicates().to_numpy()))
    missing_groups = sorted(expected.difference(observed))
    if missing_groups:
        raise ValueError(f"{path} is missing task-class groups: {missing_groups}")
    return frame


def _cdf_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, len(ordered) + 1, dtype=float) / float(len(ordered)) * 100.0


def _summarize(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for device in DEVICE_ORDER:
        for task_class in TASK_CLASS_ORDER:
            values = frame.loc[
                frame["device"].eq(device) & frame["task_class"].eq(task_class), "relative_error"
            ].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "device": device,
                "task_class": task_class,
                "samples": int(len(values)),
                "mape_pct": float(np.mean(values) * 100.0),
                "median_ape_pct": float(np.percentile(values, 50.0) * 100.0),
                "p90_ape_pct": float(np.percentile(values, 90.0) * 100.0),
                "p95_ape_pct": float(np.percentile(values, 95.0) * 100.0),
                "p99_ape_pct": float(np.percentile(values, 99.0) * 100.0),
                "max_ape_pct": float(np.max(values) * 100.0),
            }
            for threshold in thresholds:
                row[f"cdf_at_{threshold:.2f}_pct"] = float(np.mean(values <= threshold) * 100.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_points(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for device in DEVICE_ORDER:
        for task_class in TASK_CLASS_ORDER:
            values = frame.loc[
                frame["device"].eq(device) & frame["task_class"].eq(task_class), "relative_error"
            ].to_numpy(dtype=float)
            x, y = _cdf_points(values)
            rows.append(
                pd.DataFrame(
                    {
                        "device": device,
                        "task_class": task_class,
                        "relative_error": x,
                        "cdf_percent": y,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _plot(points: pd.DataFrame, summary: pd.DataFrame, *, thresholds: list[float], title: str, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    max_error = float(points["relative_error"].max())
    x_max = max(0.30, np.ceil((max_error + 0.02) / 0.05) * 0.05, max(thresholds) + 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.1), sharey=True)
    for axis, device in zip(axes, DEVICE_ORDER, strict=True):
        device_points = points.loc[points["device"].eq(device)]
        device_summary = summary.loc[summary["device"].eq(device)].set_index("task_class")
        for task_class in TASK_CLASS_ORDER:
            part = device_points.loc[device_points["task_class"].eq(task_class)]
            mape = float(device_summary.loc[task_class, "mape_pct"])
            axis.plot(
                part["relative_error"],
                part["cdf_percent"],
                color=TASK_CLASS_COLORS[task_class],
                linewidth=2.4,
                label=f"{TASK_CLASS_LABELS[task_class].replace(chr(10), ' ')} ({mape:.1f}%)",
            )
        for threshold in thresholds:
            axis.axvline(threshold, color="#7A7A7A", linestyle="--", linewidth=1.0, alpha=0.75)
            axis.text(
                threshold,
                2.5,
                f"{threshold:.0%}",
                rotation=90,
                ha="right",
                va="bottom",
                color="#666666",
                fontsize=11,
            )
        axis.set_title(device.replace("_", " "))
        axis.set_xlim(0.0, x_max)
        axis.set_ylim(0.0, 100.0)
        axis.set_xlabel("Relative error")
        axis.grid(True, linestyle="--", alpha=0.28)
        axis.set_axisbelow(True)
        axis.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94)
    axes[0].set_ylabel("CDF (%)")
    fig.suptitle(title, fontsize=21, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "dlrm_task_class_relative_error_cdf.png", bbox_inches="tight")
    fig.savefig(out_dir / "dlrm_task_class_relative_error_cdf.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: pd.DataFrame, *, source: Path, thresholds: list[float], out_dir: Path) -> None:
    threshold_columns = [f"cdf_at_{threshold:.2f}_pct" for threshold in thresholds]
    lines = [
        "# DLRM 图聚合任务相对误差 CDF",
        "",
        "每一行表示一个 DLRM 执行阶段在一个 workload combination 下的预测结果。"
        "同一阶段内的算子实测耗时和预测耗时先分别求和，再计算相对误差。",
        "",
        f"- 输入：`{source.resolve()}`",
        f"- 误差阈值：`{','.join(f'{value:.2f}' for value in thresholds)}`",
        "",
        "| Device | Task class | Samples | MAPE | P90 APE | P95 APE | "
        + " | ".join(f"CDF @ {value:.0%}" for value in thresholds)
        + " |",
        "| --- | --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in thresholds) + " |",
    ]
    for _, row in summary.iterrows():
        threshold_values = " | ".join(f"{float(row[column]):.1f}%" for column in threshold_columns)
        lines.append(
            f"| {row['device']} | {row['task_class']} | {int(row['samples'])} | "
            f"{float(row['mape_pct']):.2f}% | {float(row['p90_ape_pct']):.2f}% | "
            f"{float(row['p95_ape_pct']):.2f}% | {threshold_values} |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `dlrm_task_class_relative_error_cdf.png`",
            "- `dlrm_task_class_relative_error_cdf.pdf`",
            "- `dlrm_task_class_relative_error_cdf_summary.csv`",
            "- `dlrm_task_class_relative_error_cdf_points.csv`",
            "",
            "## 复现命令",
            "",
            "```bash",
            "python calibrator/plot_task_class_relative_error_cdf.py \\",
            f"  --task-prediction-rows {source.resolve()} \\",
            f"  --threshold {','.join(f'{value:.2f}' for value in thresholds)} \\",
            f"  --out-dir {out_dir.resolve()}",
            "```",
        ]
    )
    (out_dir / "DLRM_TASK_CLASS_RELATIVE_ERROR_CDF_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.task_prediction_rows)
    summary = _summarize(rows, args.thresholds)
    points = _build_points(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "dlrm_task_class_relative_error_cdf_summary.csv", index=False)
    points.to_csv(args.out_dir / "dlrm_task_class_relative_error_cdf_points.csv", index=False)
    _plot(points, summary, thresholds=args.thresholds, title=str(args.title), out_dir=args.out_dir)
    _write_report(summary, source=args.task_prediction_rows, thresholds=args.thresholds, out_dir=args.out_dir)
    (args.out_dir / "dlrm_task_class_relative_error_cdf_metadata.json").write_text(
        json.dumps(
            {
                "input": str(args.task_prediction_rows.resolve()),
                "thresholds": args.thresholds,
                "task_classes": TASK_CLASS_ORDER,
                "devices": DEVICE_ORDER,
                "metric": "abs(sum(pred_us) - sum(actual_us)) / sum(actual_us)",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_csv(index=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
