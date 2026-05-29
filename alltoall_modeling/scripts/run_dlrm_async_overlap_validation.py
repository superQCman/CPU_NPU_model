#!/usr/bin/env python3
"""Validate async All-to-All hiding with DLRM profiler step traces.

This script consumes Ascend profiler ``step_trace_time.csv`` from an actual DLRM
run. The profiler already reports Computing, Communication and Overlapped time,
so the validation is based on real DLRM execution rather than synthetic replay.
"""

from __future__ import annotations

import argparse
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


DEFAULT_PROFILER_DIR = PROJECT_ROOT / "data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fnum(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_step_trace(path: Path, min_comm_us: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Step", "Computing", "Communication", "Overlapped"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"step trace missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "step": fnum(df["Step"]).astype(int),
            "computing_us": fnum(df["Computing"]),
            "communication_us": fnum(df["Communication"]),
            "overlapped_us": fnum(df["Overlapped"]),
        }
    )
    if "Communication(Not Overlapped)" in df.columns:
        out["communication_not_overlapped_us"] = fnum(df["Communication(Not Overlapped)"])
    else:
        out["communication_not_overlapped_us"] = out["communication_us"] - out["overlapped_us"]

    out = out[out["communication_us"] >= float(min_comm_us)].copy()
    out = out.reset_index(drop=True)
    bound = np.minimum(out["communication_us"].to_numpy(), out["computing_us"].to_numpy())
    out["eta_observed"] = np.where(bound > 0, out["overlapped_us"].to_numpy() / bound, 0.0)
    out["eta_observed"] = out["eta_observed"].clip(0.0, 1.0)
    out["hidden_fraction_of_comm"] = np.where(
        out["communication_us"] > 0,
        out["overlapped_us"] / out["communication_us"],
        0.0,
    )
    out["async_region_observed_us"] = out["computing_us"] + out["communication_us"] - out["overlapped_us"]
    out["sync_region_baseline_us"] = out["computing_us"] + out["communication_us"]
    out["observed_speedup"] = out["sync_region_baseline_us"] / out["async_region_observed_us"].clip(lower=1e-9)
    return out


def count_record_markers(operator_details: Path) -> dict[str, int]:
    if not operator_details.exists():
        return {"alltoall_start_markers": 0, "alltoall_wait_markers": 0}
    start = 0
    wait = 0
    with operator_details.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "DLRM alltoall communication start" in line:
                start += 1
            elif "DLRM alltoall communication wait" in line:
                wait += 1
    return {"alltoall_start_markers": start, "alltoall_wait_markers": wait}


def add_predictions(df: pd.DataFrame, eta: float) -> pd.DataFrame:
    params = AsyncOverlapParams(eta=eta, launch_overhead_us=0.0, sync_overhead_us=0.0)
    rows = []
    for _, row in df.iterrows():
        pred = predict_async_region(row["communication_us"], row["computing_us"], params)
        rows.append(pred)
    pred_df = pd.DataFrame(rows)
    out = df.copy()
    out["eta_model"] = float(eta)
    out["async_region_pred_us"] = pred_df["async_total_us"]
    out["visible_comm_pred_us"] = pred_df["visible_comm_us"]
    out["hidden_comm_pred_us"] = pred_df["hidden_comm_us"]
    out["abs_err_us"] = (out["async_region_pred_us"] - out["async_region_observed_us"]).abs()
    out["rel_err_percent"] = out["abs_err_us"] / out["async_region_observed_us"].clip(lower=1e-9) * 100.0
    return out


def metrics(df: pd.DataFrame) -> dict[str, float]:
    return {
        "steps": int(len(df)),
        "eta_mean": float(df["eta_observed"].mean()),
        "eta_median": float(df["eta_observed"].median()),
        "eta_p10": float(df["eta_observed"].quantile(0.10)),
        "eta_p90": float(df["eta_observed"].quantile(0.90)),
        "comm_mean_us": float(df["communication_us"].mean()),
        "compute_mean_us": float(df["computing_us"].mean()),
        "overlap_mean_us": float(df["overlapped_us"].mean()),
        "hidden_fraction_mean": float(df["hidden_fraction_of_comm"].mean()),
        "speedup_mean": float(df["observed_speedup"].mean()),
        "model_mape_percent": float(df["rel_err_percent"].mean()),
        "model_rmse_us": float(np.sqrt(np.mean(df["abs_err_us"] ** 2))),
        "model_p90_rel_err_percent": float(df["rel_err_percent"].quantile(0.90)),
    }


def write_summary(path: Path, summary: dict, markers: dict, args: argparse.Namespace) -> None:
    lines = [
        "# DLRM Async All-to-All Validation Summary",
        "",
        f"- Run time: {datetime.now().isoformat(timespec='seconds')}",
        "- Validation mode: real DLRM Ascend profiler step trace",
        f"- Profiler dir: `{display_path(Path(args.profiler_dir))}`",
        f"- Step trace: `{args.step_trace}`",
        f"- Min communication filter: `{args.min_comm_us} us`",
        f"- All-to-All start markers: `{markers['alltoall_start_markers']}`",
        f"- All-to-All wait markers: `{markers['alltoall_wait_markers']}`",
        "",
        "## Observed DLRM Overlap",
        "",
        f"- Steps used: {summary['steps']}",
        f"- Mean communication: {summary['comm_mean_us']:.3f} us",
        f"- Mean overlappable compute: {summary['compute_mean_us']:.3f} us",
        f"- Mean overlapped time: {summary['overlap_mean_us']:.3f} us",
        f"- Observed eta mean/median: {summary['eta_mean']:.4f} / {summary['eta_median']:.4f}",
        f"- Observed eta P10/P90: {summary['eta_p10']:.4f} / {summary['eta_p90']:.4f}",
        f"- Mean hidden fraction of communication: {summary['hidden_fraction_mean']:.6f}",
        f"- Mean speedup over sync comm+compute region: {summary['speedup_mean']:.6f}x",
        "",
        "## Fitted Eta Model Check",
        "",
        f"- Model eta: {summary['eta_median']:.4f} (median observed eta)",
        f"- Async-region MAPE: {summary['model_mape_percent']:.3f}%",
        f"- Async-region RMSE: {summary['model_rmse_us']:.3f} us",
        f"- Async-region P90 relative error: {summary['model_p90_rel_err_percent']:.3f}%",
        "",
        "## Interpretation",
        "",
        "- DLRM does launch All-to-All asynchronously and runs bottom MLP before `wait`, so it is a valid real workload for async hiding validation.",
        "- In this profiled case, communication is tens of milliseconds while the bottom MLP compute window is about hundreds of microseconds, so only a tiny fraction of communication can be hidden.",
        "- The calibrated eta is low in this trace, meaning only a small part of the available bottom-MLP window overlaps the communication timeline.",
        "- The hidden fraction of total communication is even lower because `min(T_comm, T_compute)` is dominated by the short compute window.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_params(path: Path, summary: dict, args: argparse.Namespace) -> None:
    payload = {
        "model_type": "dlrm_a2a_async_overlap_calibration",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": display_path(Path(args.profiler_dir)),
        "params": {
            "eta": summary["eta_median"],
            "launch_overhead_us": 0.0,
            "sync_overhead_us": 0.0,
            "overlap_window_scale": 1.0,
        },
        "observed_metrics": summary,
        "notes": [
            "Calibrated from Ascend step_trace_time.csv Computing/Communication/Overlapped columns.",
            "Overhead is set to zero because step_trace aggregate fields already include runtime scheduling effects.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(df["communication_us"], df["computing_us"], c=df["eta_observed"], s=36, cmap="viridis")
    ax.set_xlabel("Communication (us)")
    ax.set_ylabel("Overlappable compute (us)")
    ax.set_title("DLRM Step Trace: Comm vs Compute Window")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Observed eta")

    ax2 = axes[1]
    ax2.scatter(df["async_region_observed_us"], df["async_region_pred_us"], s=36, alpha=0.65)
    lo = min(df["async_region_observed_us"].min(), df["async_region_pred_us"].min())
    hi = max(df["async_region_observed_us"].max(), df["async_region_pred_us"].max())
    ax2.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1.2)
    ax2.set_xlabel("Observed async comm+compute region (us)")
    ax2.set_ylabel("Predicted async comm+compute region (us)")
    ax2.set_title("Median-eta Model Check")
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate async A2A hiding from DLRM profiler step trace.")
    parser.add_argument("--profiler-dir", default=str(DEFAULT_PROFILER_DIR))
    parser.add_argument("--step-trace", default="step_trace_time.csv")
    parser.add_argument("--operator-details", default="operator_details.csv")
    parser.add_argument("--min-comm-us", type=float, default=1000.0)
    args = parser.parse_args()

    profiler_dir = Path(args.profiler_dir)
    step_trace = profiler_dir / args.step_trace
    operator_details = profiler_dir / args.operator_details
    if not step_trace.exists():
        raise FileNotFoundError(f"missing step trace: {step_trace}")

    reports_dir = PROJECT_ROOT / "artifacts/reports"
    models_dir = PROJECT_ROOT / "artifacts/models"
    plots_dir = PROJECT_ROOT / "artifacts/plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    observed = load_step_trace(step_trace, args.min_comm_us)
    if len(observed) == 0:
        raise RuntimeError("no DLRM steps left after communication filter")
    eta = float(observed["eta_observed"].median())
    result = add_predictions(observed, eta)
    summary = metrics(result)
    markers = count_record_markers(operator_details)

    result.to_csv(reports_dir / "dlrm_async_overlap_validation.csv", index=False)
    write_summary(reports_dir / "dlrm_async_overlap_summary.md", summary, markers, args)
    write_params(models_dir / "dlrm_async_overlap_model_params.json", summary, args)
    plot(result, plots_dir / "dlrm_async_overlap_validation.png")

    print("[done] DLRM async overlap validation generated")
    print(f"  report: {reports_dir / 'dlrm_async_overlap_summary.md'}")
    print(f"  csv: {reports_dir / 'dlrm_async_overlap_validation.csv'}")


if __name__ == "__main__":
    main()
