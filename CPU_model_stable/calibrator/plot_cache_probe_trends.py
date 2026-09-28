from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrator.whitebox_family_routes import _load_hw_params  # noqa: E402


HW_CONFIGS = {
    "S920X20": PROJECT_ROOT / "calibrator" / "hw_config" / "s920x20_single_numa.yaml",
    "Kunpeng920": PROJECT_ROOT / "calibrator" / "hw_config" / "kunpeng920_single_numa.yaml",
}

LEVELS = (
    ("L1D", "l1d_size_bytes", "l1d_latency_ns_static"),
    ("L2", "l2_size_bytes", "l2_latency_ns_static"),
    ("L3", "l3_size_bytes", "l3_latency_ns_static"),
)

FACTORS_SIZE = tuple(float(value) for value in np.round(np.linspace(0.80, 1.20, 9), 4))
FACTORS_LATENCY = tuple(float(value) for value in np.round(np.linspace(0.80, 1.20, 9), 4))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic cache-sensitive workload predictions.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "cache_analysis"),
    )
    return parser.parse_args()


def _smooth_fit_probability(capacity_bytes: float, working_set_bytes: float) -> float:
    ratio = float(capacity_bytes) / max(float(working_set_bytes), 1.0)
    return float(1.0 / (1.0 + np.exp(-(ratio - 1.0) / 0.075)))


def _expected_cache_latency_us(hw: dict[str, float], working_set_bytes: float, *, size_scales: dict[str, float], latency_scales: dict[str, float]) -> float:
    l1_size = float(hw["l1d_size_bytes"]) * float(size_scales.get("L1D", 1.0))
    l2_size = float(hw["l2_size_bytes"]) * float(size_scales.get("L2", 1.0))
    l3_size = float(hw["l3_size_bytes"]) * float(size_scales.get("L3", 1.0))
    l1_p = _smooth_fit_probability(l1_size, working_set_bytes)
    l2_p = max(_smooth_fit_probability(l2_size, working_set_bytes), l1_p)
    l3_p = max(_smooth_fit_probability(l3_size, working_set_bytes), l2_p)
    p_l2 = max(l2_p - l1_p, 0.0)
    p_l3 = max(l3_p - l2_p, 0.0)
    p_mem = max(1.0 - l3_p, 0.0)
    l1_us = float(hw["l1d_latency_ns_static"]) * float(latency_scales.get("L1D", 1.0)) / 1000.0
    l2_us = float(hw["l2_latency_ns_static"]) * float(latency_scales.get("L2", 1.0)) / 1000.0
    l3_us = float(hw["l3_latency_ns_static"]) * float(latency_scales.get("L3", 1.0)) / 1000.0
    mem_us = float(hw["local_mem_delay_ns"]) / 1000.0
    return l1_p * l1_us + p_l2 * l2_us + p_l3 * l3_us + p_mem * mem_us


def _probe_time_us(hw: dict[str, float], working_set_bytes: float, *, size_scales: dict[str, float], latency_scales: dict[str, float]) -> float:
    touch_count = max(min(float(working_set_bytes) / max(float(hw["cacheline_bytes"]), 1.0), 262144.0), 4096.0)
    cacheline_bytes = float(hw["cacheline_bytes"])
    latency_us = _expected_cache_latency_us(hw, working_set_bytes, size_scales=size_scales, latency_scales=latency_scales)
    random_bw_bytes_per_us = max(float(hw["random_gather_bw_gb_s_1t"]) * 1000.0, 1e-6)
    bandwidth_us = touch_count * cacheline_bytes / random_bw_bytes_per_us
    bdp_parallel = max(random_bw_bytes_per_us * latency_us / cacheline_bytes, 1.0)
    latency_us_total = touch_count * latency_us / np.sqrt(bdp_parallel)
    dispatch_us = 0.35
    return dispatch_us + max(latency_us_total, bandwidth_us) + 0.10 * min(latency_us_total, bandwidth_us)


def _working_set_for_level(hw: dict[str, float], level: str, mode: str) -> float:
    if mode == "size":
        key = {"L1D": "l1d_size_bytes", "L2": "l2_size_bytes", "L3": "l3_size_bytes"}[level]
        return 1.02 * float(hw[key])
    if level == "L1D":
        return 0.80 * float(hw["l1d_size_bytes"])
    if level == "L2":
        return max(4.0 * float(hw["l1d_size_bytes"]), 0.80 * float(hw["l2_size_bytes"]))
    return max(4.0 * float(hw["l2_size_bytes"]), 0.80 * float(hw["l3_size_bytes"]))


def _build_points() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hw_name, path in HW_CONFIGS.items():
        hw = _load_hw_params(path)
        for level, _, _ in LEVELS:
            working_set_size = _working_set_for_level(hw, level, "size")
            base_size = _probe_time_us(hw, working_set_size, size_scales={}, latency_scales={})
            for factor in FACTORS_SIZE:
                rows.append(
                    {
                        "hardware": hw_name,
                        "probe": f"{level} size",
                        "group": "cache_size",
                        "level": level,
                        "relative_parameter": factor,
                        "normalized_time": _probe_time_us(hw, working_set_size, size_scales={level: factor}, latency_scales={}) / base_size,
                        "working_set_bytes": working_set_size,
                    }
                )
            working_set_latency = _working_set_for_level(hw, level, "latency")
            base_latency = _probe_time_us(hw, working_set_latency, size_scales={}, latency_scales={})
            for factor in FACTORS_LATENCY:
                rows.append(
                    {
                        "hardware": hw_name,
                        "probe": f"{level} latency",
                        "group": "cache_latency",
                        "level": level,
                        "relative_parameter": factor,
                        "normalized_time": _probe_time_us(hw, working_set_latency, size_scales={}, latency_scales={level: factor}) / base_latency,
                        "working_set_bytes": working_set_latency,
                    }
                )
    return pd.DataFrame(rows)


def _plot(points: pd.DataFrame, output_root: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.45, 4.6), squeeze=False, constrained_layout=True)
    palette = {"S920X20": "#2f5f9e", "Kunpeng920": "#b65a2a"}
    for row_idx, group in enumerate(("cache_size", "cache_latency")):
        for col_idx, (level, _, _) in enumerate(LEVELS):
            ax = axes[row_idx][col_idx]
            subset = points.loc[points["group"].eq(group) & points["level"].eq(level)].copy()
            for hw_name, hw_points in subset.groupby("hardware", sort=False):
                hw_points = hw_points.sort_values("relative_parameter")
                ax.plot(
                    hw_points["relative_parameter"],
                    hw_points["normalized_time"],
                    marker="o",
                    color=palette.get(str(hw_name), "#2f5f9e"),
                    label=str(hw_name),
                )
                base = hw_points.iloc[(hw_points["relative_parameter"] - 1.0).abs().argsort()[:1]]
                ax.scatter(base["relative_parameter"], base["normalized_time"], s=28, color="red", zorder=5, edgecolor="white", linewidth=0.5)
            ax.axvline(1.0, color="0.55", linestyle="--", linewidth=0.8)
            ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
            ax.grid(True, color="0.85", linewidth=0.7)
            ax.set_title(f"{level} cache {'size' if group == 'cache_size' else 'latency'}")
            ax.set_xlabel(f"Relative {level} {'size' if group == 'cache_size' else 'latency'} (baseline=1)")
            if col_idx == 0:
                ax.set_ylabel("Normalized model-predicted time")
            ax.legend(frameon=False, loc="best")
    fig.savefig(output_root / "cache_controlled_probe_trends.pdf", bbox_inches="tight")
    fig.savefig(output_root / "cache_controlled_probe_trends.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    points = _build_points()
    points.to_csv(output_root / "cache_controlled_probe_points.csv", index=False)
    amplitudes = points.groupby(["hardware", "group", "level"], dropna=False)["normalized_time"].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    summary = amplitudes.groupby(["group", "level"]).median().rename("median_amp").reset_index()
    summary.to_csv(output_root / "cache_controlled_probe_summary.csv", index=False)
    _plot(points, output_root)
    print(f"[cache-probe] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
