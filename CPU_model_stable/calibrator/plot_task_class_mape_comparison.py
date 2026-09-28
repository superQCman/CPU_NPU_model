#!/usr/bin/env python3
"""Plot DLRM task-class MAPE for analytical-scale and complete-model routes.

The script is read-only: it consumes the two task-class summary CSV files
created by ``calculate_graph_task_mape.py`` and does not train or refit a
model.  Device color is kept constant between methods; hatching distinguishes
the analytical-scale route from the complete RF route.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_CLASS_ORDER = [
    "Embedding lookup + reduction",
    "Bottom MLP",
    "Feature interaction",
    "Top MLP",
]
DEVICE_COLORS = {"Device_1": "#C46E2D", "Device_2": "#3F72B8"}
DEVICE_LABELS = {"Device_1": "Device_1", "Device_2": "Device_2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DLRM task-class MAPE for analytical-scale and complete-model routes."
    )
    parser.add_argument("--analytical-summary", required=True, type=Path)
    parser.add_argument("--complete-model-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_summary(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"task_class", "device", "mape_pct"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, ["task_class", "device", "mape_pct"]].copy()
    frame["mape_pct"] = pd.to_numeric(frame["mape_pct"], errors="coerce")
    if not (np.isfinite(frame["mape_pct"]) & frame["mape_pct"].ge(0)).all():
        raise ValueError(f"{path} contains invalid mape_pct values")
    if frame.duplicated(["task_class", "device"]).any():
        raise ValueError(f"{path} contains duplicate task_class/device rows")
    frame["method"] = label
    return frame


def build_comparison(analytical: pd.DataFrame, complete: pd.DataFrame) -> pd.DataFrame:
    expected = {(task_class, device) for task_class in TASK_CLASS_ORDER for device in DEVICE_COLORS}
    analytical_keys = set(map(tuple, analytical[["task_class", "device"]].to_numpy()))
    complete_keys = set(map(tuple, complete[["task_class", "device"]].to_numpy()))
    if analytical_keys != expected:
        raise ValueError(f"analytical summary does not cover the expected task classes: {sorted(analytical_keys ^ expected)}")
    if complete_keys != expected:
        raise ValueError(f"complete-model summary does not cover the expected task classes: {sorted(complete_keys ^ expected)}")

    result = analytical.rename(columns={"mape_pct": "analytical_scale_mape_pct"}).merge(
        complete.rename(columns={"mape_pct": "complete_model_mape_pct"}),
        on=["task_class", "device"],
        suffixes=("_analytical", "_complete"),
        validate="one_to_one",
    )
    result["complete_minus_analytical_pp"] = (
        result["complete_model_mape_pct"] - result["analytical_scale_mape_pct"]
    )
    order = {name: index for index, name in enumerate(TASK_CLASS_ORDER)}
    return (
        result.assign(task_class_order=result["task_class"].map(order))
        .sort_values(["task_class_order", "device"], kind="stable")
        .drop(columns=["task_class_order", "method_analytical", "method_complete"])
        .reset_index(drop=True)
    )


def plot_comparison(comparison: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(TASK_CLASS_ORDER), dtype=float)
    width = 0.18
    offsets = {
        ("Device_1", "analytical"): -1.5 * width,
        ("Device_1", "complete"): -0.5 * width,
        ("Device_2", "analytical"): 0.5 * width,
        ("Device_2", "complete"): 1.5 * width,
    }
    value_columns = {
        "analytical": "analytical_scale_mape_pct",
        "complete": "complete_model_mape_pct",
    }
    labels = {
        ("Device_1", "analytical"): "Analytical + scale / Device_1",
        ("Device_1", "complete"): "Complete model / Device_1",
        ("Device_2", "analytical"): "Analytical + scale / Device_2",
        ("Device_2", "complete"): "Complete model / Device_2",
    }

    fig, ax = plt.subplots(figsize=(15.4, 7.2))
    max_value = 0.0
    for device in ("Device_1", "Device_2"):
        for method in ("analytical", "complete"):
            values: list[float] = []
            for task_class in TASK_CLASS_ORDER:
                row = comparison.loc[
                    comparison["task_class"].eq(task_class) & comparison["device"].eq(device)
                ]
                values.append(float(row.iloc[0][value_columns[method]]))
            max_value = max(max_value, max(values))
            bars = ax.bar(
                x + offsets[(device, method)],
                values,
                width,
                color=DEVICE_COLORS[device],
                alpha=0.52 if method == "analytical" else 1.0,
                hatch="//" if method == "analytical" else None,
                edgecolor="white",
                linewidth=0.7,
                label=labels[(device, method)],
            )
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.24,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12.5,
                )

    ax.axhline(15.0, color="#9B2C2C", linewidth=1.2, linestyle="--", label="15% threshold")
    ax.set_title("DLRM Task-Class MAPE: Analytical + Scale vs Complete Model", fontsize=22)
    ax.set_ylabel("MAPE (%)", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Embedding lookup\n+ reduction", "Bottom MLP", "Feature interaction", "Top MLP"],
        fontsize=16,
    )
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, max(17.0, max_value * 1.28))
    ax.legend(ncol=2, frameon=True, fontsize=12.5, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_dir / "dlrm_task_class_mape_analytical_vs_complete.png", dpi=300)
    fig.savefig(out_dir / "dlrm_task_class_mape_analytical_vs_complete.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analytical = read_summary(args.analytical_summary, "analytical_scale")
    complete = read_summary(args.complete_model_summary, "complete_model")
    comparison = build_comparison(analytical, complete)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out_dir / "dlrm_task_class_mape_analytical_vs_complete.csv", index=False)
    plot_comparison(comparison, args.out_dir)
    print(comparison.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
