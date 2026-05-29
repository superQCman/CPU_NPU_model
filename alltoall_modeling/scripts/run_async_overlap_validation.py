#!/usr/bin/env python3
"""Generate async overlap validation artifacts from A2A validation data.

The default validation mode is replay-based: measured All-to-All latencies from
``validation_metrics.csv`` are used as the communication ground truth while the
existing latency model predictions are used as the communication model input.
This validates the async composition and quantifies how single-op error
propagates into overlapped regions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.async_overlap_model import AsyncOverlapParams, predict_async_region
from src.bw_model import ALIGNMENT_HIGH, ALIGNMENT_MID


def group_of_size(size: int) -> str:
    if size % ALIGNMENT_HIGH == 0:
        return "Group1 (128KB)"
    if size % ALIGNMENT_MID == 0:
        return "Group2 (32KB)"
    return "Group3 (unaligned)"


def select_rows(df: pd.DataFrame, max_rows_per_group: int) -> pd.DataFrame:
    rows = []
    d = df.copy()
    d["group"] = d["size"].astype(int).map(group_of_size)
    for group, gd in d.groupby("group", sort=False):
        gd = gd.sort_values("size").reset_index(drop=True)
        if len(gd) <= max_rows_per_group:
            rows.append(gd)
            continue
        idx = np.linspace(0, len(gd) - 1, max_rows_per_group).round().astype(int)
        rows.append(gd.iloc[np.unique(idx)])
    return pd.concat(rows, ignore_index=True).sort_values(["group", "size"]).reset_index(drop=True)


def build_validation_frame(
    comm_df: pd.DataFrame,
    params: AsyncOverlapParams,
    compute_ratios: list[float],
) -> pd.DataFrame:
    records = []
    for _, row in comm_df.iterrows():
        size = int(row["size"])
        group = group_of_size(size)
        comm_true = float(row["latency"])
        comm_pred = float(row["latency_pred"])
        for ratio in compute_ratios:
            compute_us = comm_true * float(ratio)
            obs = predict_async_region(comm_true, compute_us, params)
            pred = predict_async_region(comm_pred, compute_us, params)
            abs_err = abs(pred["async_total_us"] - obs["async_total_us"])
            rel_err = abs_err / max(obs["async_total_us"], 1e-9) * 100.0
            records.append(
                {
                    "size": size,
                    "group": group,
                    "compute_ratio": float(ratio),
                    "compute_us": compute_us,
                    "comm_true_us": comm_true,
                    "comm_pred_us": comm_pred,
                    "async_observed_replay_us": obs["async_total_us"],
                    "async_pred_us": pred["async_total_us"],
                    "visible_comm_observed_us": obs["visible_comm_us"],
                    "visible_comm_pred_us": pred["visible_comm_us"],
                    "hidden_comm_observed_us": obs["hidden_comm_us"],
                    "hidden_comm_pred_us": pred["hidden_comm_us"],
                    "speedup_observed": obs["speedup"],
                    "speedup_pred": pred["speedup"],
                    "overlap_ratio": obs["overlap_ratio"],
                    "hidden_fraction_of_comm": obs["hidden_fraction_of_comm"],
                    "abs_err_us": abs_err,
                    "rel_err_percent": rel_err,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    return {
        "samples": int(len(df)),
        "mape_percent": float(df["rel_err_percent"].mean()),
        "rmse_us": float(np.sqrt(np.mean(df["abs_err_us"] ** 2))),
        "p50_rel_err_percent": float(df["rel_err_percent"].quantile(0.50)),
        "p90_rel_err_percent": float(df["rel_err_percent"].quantile(0.90)),
        "p95_rel_err_percent": float(df["rel_err_percent"].quantile(0.95)),
        "mean_observed_speedup": float(df["speedup_observed"].mean()),
        "mean_hidden_fraction_of_comm": float(df["hidden_fraction_of_comm"].mean()),
    }


def write_params(path: Path, params: AsyncOverlapParams, args: argparse.Namespace) -> None:
    payload = {
        "model_type": "a2a_async_overlap",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_comm_model": "a2a_latency",
        "validation_mode": "replay_from_measured_a2a_validation",
        "scope": "single-node 8 NPU HCCL All-to-All async region",
        "params": params.to_dict(),
        "compute_ratios": [float(x) for x in args.compute_ratios],
        "notes": [
            "eta is a scenario parameter unless calibrated from profiler traces.",
            "Replay validation uses measured single-op communication latency as async ground truth input.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(path: Path, overall: dict, by_group: pd.DataFrame, by_ratio: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Async Overlap Modeling Validation Summary",
        "",
        f"- Run time: {datetime.now().isoformat(timespec='seconds')}",
        "- Validation mode: replay from measured All-to-All validation results",
        f"- Input validation CSV: `{args.validation_metrics}`",
        f"- eta: `{args.eta}`",
        f"- launch overhead: `{args.launch_overhead_us} us`",
        f"- sync overhead: `{args.sync_overhead_us} us`",
        f"- compute ratios: `{', '.join(str(x) for x in args.compute_ratios)}`",
        "",
        "## Overall Metrics",
        "",
        f"- Samples: {overall['samples']}",
        f"- Async total MAPE: {overall['mape_percent']:.3f}%",
        f"- Async total RMSE: {overall['rmse_us']:.3f} us",
        f"- Relative Error P50/P90/P95: {overall['p50_rel_err_percent']:.3f}% / {overall['p90_rel_err_percent']:.3f}% / {overall['p95_rel_err_percent']:.3f}%",
        f"- Mean observed speedup vs sync baseline: {overall['mean_observed_speedup']:.3f}x",
        f"- Mean hidden communication fraction: {overall['mean_hidden_fraction_of_comm']:.3f}",
        "",
        "## Metrics By Group",
        "",
        "| Group | Samples | MAPE(%) | RMSE(us) | P90 Err(%) | Mean Speedup | Hidden Comm Fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_group.iterrows():
        lines.append(
            f"| {row['group']} | {int(row['samples'])} | {row['mape_percent']:.3f} | {row['rmse_us']:.3f} | "
            f"{row['p90_rel_err_percent']:.3f} | {row['mean_observed_speedup']:.3f} | {row['mean_hidden_fraction_of_comm']:.3f} |"
        )
    lines += [
        "",
        "## Metrics By Compute Ratio",
        "",
        "| compute/comm | Samples | MAPE(%) | RMSE(us) | P90 Err(%) | Mean Speedup | Hidden Comm Fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_ratio.iterrows():
        lines.append(
            f"| {row['compute_ratio']:.2f} | {int(row['samples'])} | {row['mape_percent']:.3f} | {row['rmse_us']:.3f} | "
            f"{row['p90_rel_err_percent']:.3f} | {row['mean_observed_speedup']:.3f} | {row['mean_hidden_fraction_of_comm']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This report validates the async overlap formula by replaying measured All-to-All latency points across multiple compute-window sizes.",
        "- The reported prediction error is the propagated error from the single All-to-All latency model into the async region model.",
        "- To calibrate `eta` for a real workload, collect `(T_comm, T_compute_window, T_async_region)` from profiler traces and use `derive_eta` in `src/async_overlap_model.py`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(df: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, gd in df.groupby(key, sort=True):
        row = summarize(gd)
        row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_validation(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for ratio, gd in df.groupby("compute_ratio", sort=True):
        ax.scatter(gd["async_observed_replay_us"], gd["async_pred_us"], s=12, alpha=0.45, label=f"r={ratio:g}")
    lo = min(df["async_observed_replay_us"].min(), df["async_pred_us"].min())
    hi = max(df["async_observed_replay_us"].max(), df["async_pred_us"].max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.2, ls="--", label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Replay observed async total (us)")
    ax.set_ylabel("Predicted async total (us)")
    ax.set_title("Async Region: Predicted vs Replay Observed")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax2 = axes[1]
    ratio_summary = df.groupby("compute_ratio")["hidden_fraction_of_comm"].mean().reset_index()
    ax2.plot(ratio_summary["compute_ratio"], ratio_summary["hidden_fraction_of_comm"], marker="o", lw=2)
    ax2.set_xlabel("Compute window / communication latency")
    ax2.set_ylabel("Mean hidden communication fraction")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Communication Hiding vs Compute Window")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate A2A async overlap model by replaying communication validation data.")
    parser.add_argument("--validation-metrics", default="artifacts/reports/validation_metrics.csv")
    parser.add_argument("--eta", type=float, default=0.85)
    parser.add_argument("--launch-overhead-us", type=float, default=3.0)
    parser.add_argument("--sync-overhead-us", type=float, default=8.0)
    parser.add_argument("--overlap-window-scale", type=float, default=1.0)
    parser.add_argument("--compute-ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--max-rows-per-group", type=int, default=80)
    args = parser.parse_args()

    validation_path = PROJECT_ROOT / args.validation_metrics
    if not validation_path.exists():
        raise FileNotFoundError(f"missing validation metrics: {validation_path}")

    reports_dir = PROJECT_ROOT / "artifacts/reports"
    models_dir = PROJECT_ROOT / "artifacts/models"
    plots_dir = PROJECT_ROOT / "artifacts/plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    comm_df = pd.read_csv(validation_path)
    required = {"size", "latency", "latency_pred"}
    missing = required.difference(comm_df.columns)
    if missing:
        raise ValueError(f"validation metrics missing columns: {sorted(missing)}")

    comm_df = select_rows(comm_df, args.max_rows_per_group)
    params = AsyncOverlapParams(
        eta=args.eta,
        launch_overhead_us=args.launch_overhead_us,
        sync_overhead_us=args.sync_overhead_us,
        overlap_window_scale=args.overlap_window_scale,
    )
    out_df = build_validation_frame(comm_df, params, args.compute_ratios)
    out_csv = reports_dir / "async_overlap_validation.csv"
    out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    overall = summarize(out_df)
    by_group = aggregate(out_df, "group")
    by_ratio = aggregate(out_df, "compute_ratio")
    by_group.to_csv(reports_dir / "async_overlap_metrics_by_group.csv", index=False)
    by_ratio.to_csv(reports_dir / "async_overlap_metrics_by_compute_ratio.csv", index=False)
    write_params(models_dir / "async_overlap_model_params.json", params, args)
    write_summary(reports_dir / "async_overlap_summary.md", overall, by_group, by_ratio, args)
    plot_validation(out_df, plots_dir / "async_overlap_validation.png")

    print("[done] async overlap artifacts generated")
    print(f"  csv: {out_csv}")
    print(f"  report: {reports_dir / 'async_overlap_summary.md'}")


if __name__ == "__main__":
    main()
