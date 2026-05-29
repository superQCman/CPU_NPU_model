#!/usr/bin/env python3
"""Validate random-access latency residency formulas against pointer-chase data.

This is a read-only post-processing experiment. It compares measured
single-thread pointer-chase latency against three analytical residency
formulas used or discussed by the random_mem model:

* current_interp: current continuous-residency implementation.
* ppt_weighted: explicit PPT weighted-average latency.
* hard_threshold: current non-continuous hard-residency center behavior.

The script does not train, refit, or modify any model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DEFAULT_MODES = ("current_interp", "ppt_weighted", "hard_threshold")
MODEL_PLOT_STEMS = {
    "current_interp": "random_access_latency_model_vs_measured_continuous_residency",
    "ppt_weighted": "random_access_latency_model_vs_measured_weighted_average",
    "hard_threshold": "random_access_latency_model_vs_measured_hard_residency",
}
PTRCHASE_RE = re.compile(
    r"size_bytes=(?P<size_bytes>\d+).*?"
    r"total_loads=(?P<total_loads>\d+).*?"
    r"total_ns=(?P<total_ns>[0-9.]+).*?"
    r"ns_per_load=(?P<ns_per_load>[0-9.]+)"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare random access latency formulas with pointer-chase measurements."
    )
    parser.add_argument("--hw-config", required=True, type=Path)
    parser.add_argument("--ptrchase-root", required=True, type=Path)
    parser.add_argument("--num-threads", type=float, default=1.0)
    parser.add_argument("--active-groups", type=float, default=1.0)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--stable-cv-threshold", type=float, default=15.0)
    parser.add_argument("--target-mape-threshold", type=float, default=20.0)
    parser.add_argument("--extreme-ape-threshold", type=float, default=100.0)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def _parse_size_bytes(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"([0-9.]+)\s*([A-Za-z]*)", text)
    if not match:
        return float("nan")
    number = float(match.group(1))
    suffix = match.group(2).lower()
    factor = {
        "": 1.0,
        "b": 1.0,
        "kib": 1024.0,
        "kb": 1000.0,
        "mib": 1024.0**2,
        "mb": 1000.0**2,
        "gib": 1024.0**3,
        "gb": 1000.0**3,
    }.get(suffix)
    if factor is None:
        return float("nan")
    return number * factor


def _parse_ghz(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"([0-9.]+)\s*([A-Za-z]*)", text)
    if not match:
        return float("nan")
    number = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix in {"ghz", ""}:
        return number
    if suffix == "mhz":
        return number / 1000.0
    return float("nan")


def _nested_get(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_hw(hw_config: Path) -> dict[str, float]:
    payload = yaml.safe_load(hw_config.read_text(encoding="utf-8"))
    cpu_clock_ghz = _parse_ghz(_nested_get(payload, "core.cpu_clock"))
    total_cores = float(_nested_get(payload, "core.total_cores", 1.0) or 1.0)
    l1_size = _parse_size_bytes(_nested_get(payload, "cache.l1d.size"))
    l2_size = _parse_size_bytes(_nested_get(payload, "cache.l2.size"))
    l3_size = _parse_size_bytes(
        _nested_get(payload, "cache.l3_per_die.size", _nested_get(payload, "cache.l3.size"))
    )
    l1_cycles = float(_nested_get(payload, "cache.l1d.response_latency_cycles", float("nan")))
    l2_cycles = float(_nested_get(payload, "cache.l2.response_latency_cycles", float("nan")))
    l3_cycles = float(
        _nested_get(payload, "cache.l3_per_die.response_latency_cycles", _nested_get(payload, "cache.l3.response_latency_cycles", float("nan")))
    )
    local_mem_delay_ns = float(_nested_get(payload, "memory.local_mem_delay_ns", float("nan")))
    required = {
        "cpu_clock_ghz": cpu_clock_ghz,
        "total_cores": total_cores,
        "l1_size_bytes": l1_size,
        "l2_size_bytes": l2_size,
        "l3_size_bytes": l3_size,
        "l1_latency_ns": l1_cycles / max(cpu_clock_ghz, 1e-12),
        "l2_latency_ns": l2_cycles / max(cpu_clock_ghz, 1e-12),
        "l3_latency_ns": l3_cycles / max(cpu_clock_ghz, 1e-12),
        "mem_latency_ns": local_mem_delay_ns,
    }
    missing = [key for key, value in required.items() if not np.isfinite(value) or value <= 0]
    if missing:
        raise ValueError(f"invalid or missing hardware fields in {hw_config}: {missing}")
    return required


def _load_ptrchase_rows(ptrchase_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    files = sorted(ptrchase_root.glob("run_*/ptrchase_sweep.txt"))
    if not files and (ptrchase_root / "ptrchase_sweep.txt").exists():
        files = [ptrchase_root / "ptrchase_sweep.txt"]
    if not files:
        raise FileNotFoundError(f"no ptrchase_sweep.txt found under {ptrchase_root}")
    for path in files:
        run_id = path.parent.name if path.parent != ptrchase_root else "run_1"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = PTRCHASE_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "source_file": str(path),
                    "line_number": lineno,
                    "size_bytes": int(match.group("size_bytes")),
                    "total_loads": int(match.group("total_loads")),
                    "total_ns": float(match.group("total_ns")),
                    "measured_ns_per_load": float(match.group("ns_per_load")),
                }
            )
    if not rows:
        raise ValueError(f"no parseable ptrchase rows found under {ptrchase_root}")
    return pd.DataFrame(rows)


def _aggregate_measurements(raw: pd.DataFrame, stable_cv_threshold: float) -> pd.DataFrame:
    grouped = raw.groupby("size_bytes", sort=True)["measured_ns_per_load"]
    rows: list[dict[str, Any]] = []
    for size_bytes, values in grouped:
        arr = values.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        cv = float(std / mean * 100.0) if mean > 0 else float("inf")
        rows.append(
            {
                "size_bytes": int(size_bytes),
                "measurement_count": int(arr.size),
                "measured_ns_median": float(np.median(arr)),
                "measured_ns_mean": mean,
                "measured_ns_min": float(np.min(arr)),
                "measured_ns_max": float(np.max(arr)),
                "measured_ns_std": std,
                "cv_percent": cv,
                "stable": bool(arr.size >= 3 and cv <= stable_cv_threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("size_bytes").reset_index(drop=True)


def _latency_components(
    working_set_bytes: float,
    hw: dict[str, float],
    *,
    num_threads: float,
    active_groups: float,
) -> dict[str, float]:
    workers = float(np.clip(num_threads, a_min=1.0, a_max=hw["total_cores"]))
    groups = max(float(active_groups), 1.0)
    system_workers = float(np.clip(workers * groups, a_min=1.0, a_max=hw["total_cores"]))
    l3_share = float(np.clip(workers / max(system_workers, 1e-12), a_min=0.0, a_max=1.0))
    l1_eff = hw["l1_size_bytes"] * workers
    l2_eff = hw["l2_size_bytes"] * workers
    l3_eff = hw["l3_size_bytes"] * l3_share
    working_set = max(float(working_set_bytes), 1.0)
    p_l1 = min(l1_eff / working_set, 1.0)
    p_l2 = min(l2_eff / working_set, 1.0)
    p_l3 = min(l3_eff / working_set, 1.0)
    r_l1 = p_l1
    r_l2 = max(p_l2 - p_l1, 0.0)
    r_l3 = max(p_l3 - p_l2, 0.0)
    r_mem = max(1.0 - p_l3, 0.0)
    residency_level = r_l1 + 2.0 * r_l2 + 3.0 * r_l3 + 4.0 * r_mem
    levels = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    latencies = np.asarray(
        [hw["l1_latency_ns"], hw["l2_latency_ns"], hw["l3_latency_ns"], hw["mem_latency_ns"]],
        dtype=float,
    )
    current_interp = float(np.interp(np.clip(residency_level, 1.0, 4.0), levels, latencies))
    ppt_weighted = (
        r_l1 * hw["l1_latency_ns"]
        + r_l2 * hw["l2_latency_ns"]
        + r_l3 * hw["l3_latency_ns"]
        + r_mem * hw["mem_latency_ns"]
    )
    if working_set <= l1_eff:
        hard_optimistic = hw["l1_latency_ns"]
        hard_conservative = hw["l2_latency_ns"]
        hard_level = 1.0
    elif working_set <= l2_eff:
        hard_optimistic = hw["l2_latency_ns"]
        hard_conservative = hw["l3_latency_ns"]
        hard_level = 2.0
    elif working_set <= l3_eff:
        hard_optimistic = hw["l3_latency_ns"]
        hard_conservative = hw["mem_latency_ns"]
        hard_level = 3.0
    else:
        hard_optimistic = hw["mem_latency_ns"]
        hard_conservative = hw["mem_latency_ns"]
        hard_level = 4.0
    hard_threshold = 0.5 * (hard_optimistic + hard_conservative)
    return {
        "n_worker": workers,
        "n_system": system_workers,
        "c_l1_eff_bytes": l1_eff,
        "c_l2_eff_bytes": l2_eff,
        "c_l3_eff_bytes": l3_eff,
        "p_l1": p_l1,
        "p_l2": p_l2,
        "p_l3": p_l3,
        "r_l1": r_l1,
        "r_l2": r_l2,
        "r_l3": r_l3,
        "r_mem": r_mem,
        "continuous_residency_level": residency_level,
        "hard_residency_level": hard_level,
        "current_interp_latency_ns": current_interp,
        "ppt_weighted_latency_ns": ppt_weighted,
        "hard_threshold_latency_ns": hard_threshold,
    }


def _add_predictions(
    measurements: pd.DataFrame,
    hw: dict[str, float],
    *,
    num_threads: float,
    active_groups: float,
    modes: list[str],
    extreme_ape_threshold: float,
) -> pd.DataFrame:
    rows = []
    for row in measurements.to_dict("records"):
        payload = dict(row)
        payload.update(
            _latency_components(
                float(row["size_bytes"]),
                hw,
                num_threads=num_threads,
                active_groups=active_groups,
            )
        )
        measured = float(payload["measured_ns_median"])
        for mode in modes:
            pred_col = f"{mode}_latency_ns"
            pred = float(payload[pred_col])
            ape = abs(pred - measured) / measured * 100.0 if measured > 0 else float("nan")
            signed = (pred - measured) / measured * 100.0 if measured > 0 else float("nan")
            payload[f"{mode}_ape_pct"] = ape
            payload[f"{mode}_signed_error_pct"] = signed
            payload[f"{mode}_extreme_ape"] = bool(np.isfinite(ape) and ape > extreme_ape_threshold)
        rows.append(payload)
    return pd.DataFrame(rows)


def _summary(points: pd.DataFrame, modes: list[str], target_mape_threshold: float) -> pd.DataFrame:
    stable = points[points["stable"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for subset_name, subset in (("stable", stable), ("all", points)):
            values = pd.to_numeric(subset[f"{mode}_ape_pct"], errors="coerce").dropna().to_numpy(dtype=float)
            signed_values = pd.to_numeric(subset[f"{mode}_signed_error_pct"], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                rows.append(
                    {
                        "mode": mode,
                        "subset": subset_name,
                        "row_count": 0,
                        "mape_pct": float("nan"),
                        "median_ape_pct": float("nan"),
                        "p90_ape_pct": float("nan"),
                        "max_ape_pct": float("nan"),
                        "mean_signed_error_pct": float("nan"),
                        "pass_mape_le_threshold": False,
                    }
                )
                continue
            mape = float(np.mean(values))
            rows.append(
                {
                    "mode": mode,
                    "subset": subset_name,
                    "row_count": int(values.size),
                    "mape_pct": mape,
                    "median_ape_pct": float(np.median(values)),
                    "p90_ape_pct": float(np.percentile(values, 90)),
                    "max_ape_pct": float(np.max(values)),
                    "mean_signed_error_pct": float(np.mean(signed_values)) if signed_values.size else float("nan"),
                    "pass_mape_le_threshold": bool(mape <= target_mape_threshold),
                }
            )
    return pd.DataFrame(rows)


def _plot_model_vs_measured(points: pd.DataFrame, modes: list[str], out_dir: Path) -> None:
    stable = points[points["stable"].astype(bool)].copy()
    font = {
        "title": 19,
        "label": 17,
        "tick": 14,
        "legend": 14,
    }
    palette = {
        "current_interp": "#D55E00",
        "ppt_weighted": "#0072B2",
        "hard_threshold": "#009E73",
    }
    for mode in modes:
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        ax.scatter(
            stable["size_bytes"],
            stable["measured_ns_median"],
            color="#222222",
            s=64,
            label="Measured median",
            zorder=4,
        )
        ax.plot(
            stable["size_bytes"],
            stable[f"{mode}_latency_ns"],
            marker="o",
            markersize=7,
            linewidth=2.8,
            color=palette.get(mode),
            label="Model prediction",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Pointer-chase working set size (bytes)", fontsize=font["label"])
        ax.set_ylabel("Random access latency (ns/load)", fontsize=font["label"])
        ax.set_title("Random Access Latency Model Validation", fontsize=font["title"], pad=12)
        ax.tick_params(axis="both", which="major", labelsize=font["tick"])
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(loc="best", fontsize=font["legend"], frameon=True)
        fig.tight_layout()
        stem = MODEL_PLOT_STEMS.get(mode, f"random_access_latency_model_vs_measured_{mode}")
        fig.savefig(out_dir / f"{stem}.png", dpi=350)
        fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)


def _plot_ape(points: pd.DataFrame, modes: list[str], out_dir: Path, target_mape_threshold: float) -> None:
    stable = points[points["stable"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    palette = {
        "current_interp": "#D55E00",
        "ppt_weighted": "#0072B2",
        "hard_threshold": "#009E73",
    }
    labels = {
        "current_interp": "Current continuous interp",
        "ppt_weighted": "PPT weighted average",
        "hard_threshold": "Hard-threshold center",
    }
    for mode in modes:
        ax.plot(
            stable["size_bytes"],
            stable[f"{mode}_ape_pct"],
            marker="o",
            linewidth=2.2,
            color=palette.get(mode),
            label=labels.get(mode, mode),
        )
    ax.axhline(target_mape_threshold, color="#666666", linestyle="--", linewidth=1.4, label=f"{target_mape_threshold:.0f}% target")
    ax.set_xscale("log")
    ax.set_xlabel("Pointer-chase working set size (bytes)")
    ax.set_ylabel("Absolute percentage error (%)")
    ax.set_title("Random Access Latency APE by Working Set")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "random_access_latency_ape_by_size.png", dpi=350)
    fig.savefig(out_dir / "random_access_latency_ape_by_size.pdf")
    plt.close(fig)


def _report(
    *,
    args: argparse.Namespace,
    hw: dict[str, float],
    points: pd.DataFrame,
    summary: pd.DataFrame,
    modes: list[str],
) -> str:
    stable_count = int(points["stable"].sum())
    unstable_count = int((~points["stable"].astype(bool)).sum())
    stable_summary = summary[summary["subset"].eq("stable")].copy()
    best_row = stable_summary.sort_values("mape_pct", na_position="last").head(1)
    best_mode = str(best_row.iloc[0]["mode"]) if not best_row.empty else "n/a"
    best_mape = float(best_row.iloc[0]["mape_pct"]) if not best_row.empty else float("nan")
    status = "PASS" if np.isfinite(best_mape) and best_mape <= args.target_mape_threshold else "SANITY_WARNING"
    extreme_rows = []
    for mode in modes:
        mask = points["stable"].astype(bool) & points[f"{mode}_extreme_ape"].astype(bool)
        if mask.any():
            extreme_rows.append(f"- `{mode}` has `{int(mask.sum())}` stable rows above `{args.extreme_ape_threshold:.1f}%` APE.")
    if unstable_count:
        extreme_rows.append(f"- `{unstable_count}` size points are excluded from main figures because repeats < 3 or CV > `{args.stable_cv_threshold:.1f}%`.")
    warning_text = "\n".join(extreme_rows) if extreme_rows else "- No stable extreme-APE rows under the configured threshold."
    table = stable_summary[
        ["mode", "row_count", "mape_pct", "median_ape_pct", "p90_ape_pct", "max_ape_pct", "mean_signed_error_pct", "pass_mape_le_threshold"]
    ].copy()
    table_md = _markdown_table(table)
    plotted_modes = [
        str(row["mode"])
        for row in stable_summary.to_dict("records")
        if bool(row.get("pass_mape_le_threshold", False))
    ]
    return "\n".join(
        [
            "# Random Access Latency Validation Report",
            "",
            f"Overall status: `{status}`",
            "",
            "## Scope",
            "",
            "This experiment compares analytical random-access average latency against real pointer-chase measurements. It validates `L_touch/L_avg`, not full Gather `T_random` or E2E latency.",
            "",
            "## Code vs PPT consistency",
            "",
            "- Consistent: effective cache capacities use `C_L1,eff=N_worker*C_L1`, `C_L2,eff=N_worker*C_L2`, and `C_L3,eff=N_worker/N_system*C_L3`.",
            "- Consistent: cumulative residency probabilities use `P_L1=min(C_L1,eff/W_set,1)`, `P_L2=min(C_L2,eff/W_set,1)`, and `P_L3=min(C_L3,eff/W_set,1)`.",
            "- Difference: current continuous code maps exclusive shares to a continuous level `R` and then interpolates latency; PPT uses direct weighted latency `r_L1*L1 + r_L2*L2 + r_L3*L3 + r_mem*DRAM`.",
            "- Boundary: the trend plotting path enables `WHITEBOX_CONTINUOUS_CACHE_RESIDENCY=1`; the route code default remains the non-continuous hard-residency path.",
            "",
            "## Inputs",
            "",
            f"- Hardware config: `{args.hw_config}`",
            f"- Pointer-chase root: `{args.ptrchase_root}`",
            f"- N_worker: `{args.num_threads}`",
            f"- G_active: `{args.active_groups}`",
            f"- Stable point rule: repeats >= 3 and CV <= `{args.stable_cv_threshold:.1f}%`",
            "",
            "## Hardware latency anchors",
            "",
            f"- L1: `{hw['l1_latency_ns']:.4f}` ns",
            f"- L2: `{hw['l2_latency_ns']:.4f}` ns",
            f"- L3: `{hw['l3_latency_ns']:.4f}` ns",
            f"- DRAM: `{hw['mem_latency_ns']:.4f}` ns",
            "",
            "## Stable-point summary",
            "",
            table_md,
            "",
            f"Best stable MAPE mode: `{best_mode}` with `{best_mape:.3f}%`.",
            "",
            "Main figures plot only stable points and modes that pass the configured MAPE threshold. Failed modes remain in CSV outputs and the summary table.",
            "",
            "## Warnings",
            "",
            warning_text,
            "",
            "## Outputs",
            "",
            "- `random_access_latency_validation_points.csv`: all measured and predicted points.",
            "- `random_access_latency_validation_summary.csv`: MAPE/P90/max summary by mode.",
            "- `random_access_latency_model_vs_measured_<scheme>.png/pdf`: one stable measured-vs-predicted figure per plotted model scheme.",
            "- `random_access_latency_ape_by_size.png/pdf`: stable APE curves.",
            f"- Plotted model scheme files: `{', '.join(MODEL_PLOT_STEMS.get(mode, mode) for mode in plotted_modes) if plotted_modes else 'none'}`.",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict("records"):
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append("nan" if not math.isfinite(value) else f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    invalid = sorted(set(modes) - set(DEFAULT_MODES))
    if invalid:
        raise ValueError(f"unsupported modes: {invalid}; expected subset of {DEFAULT_MODES}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hw = _load_hw(args.hw_config)
    raw = _load_ptrchase_rows(args.ptrchase_root)
    raw.to_csv(args.out_dir / "random_access_latency_measurement_rows.csv", index=False)
    measurements = _aggregate_measurements(raw, args.stable_cv_threshold)
    points = _add_predictions(
        measurements,
        hw,
        num_threads=args.num_threads,
        active_groups=args.active_groups,
        modes=modes,
        extreme_ape_threshold=args.extreme_ape_threshold,
    )
    summary = _summary(points, modes, args.target_mape_threshold)
    points.to_csv(args.out_dir / "random_access_latency_validation_points.csv", index=False)
    summary.to_csv(args.out_dir / "random_access_latency_validation_summary.csv", index=False)
    stable_summary = summary[summary["subset"].eq("stable")]
    plotted_modes = [
        str(row["mode"])
        for row in stable_summary.to_dict("records")
        if bool(row.get("pass_mape_le_threshold", False))
    ]
    for legacy_name in (
        "random_access_latency_model_vs_measured.png",
        "random_access_latency_model_vs_measured.pdf",
    ):
        legacy_path = args.out_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
    _plot_model_vs_measured(points, plotted_modes, args.out_dir)
    _plot_ape(points, plotted_modes, args.out_dir, args.target_mape_threshold)
    manifest = {
        "hw_config": str(args.hw_config),
        "ptrchase_root": str(args.ptrchase_root),
        "modes": modes,
        "num_threads": args.num_threads,
        "active_groups": args.active_groups,
        "stable_cv_threshold": args.stable_cv_threshold,
        "target_mape_threshold": args.target_mape_threshold,
        "hardware_latency_ns": {
            "l1": hw["l1_latency_ns"],
            "l2": hw["l2_latency_ns"],
            "l3": hw["l3_latency_ns"],
            "dram": hw["mem_latency_ns"],
        },
        "stable_point_count": int(points["stable"].sum()),
        "total_point_count": int(len(points)),
    }
    (args.out_dir / "random_access_latency_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = _report(args=args, hw=hw, points=points, summary=summary, modes=modes)
    (args.out_dir / "RANDOM_ACCESS_LATENCY_VALIDATION_REPORT.md").write_text(report + "\n", encoding="utf-8")
    print(report.splitlines()[0])
    print(f"wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
