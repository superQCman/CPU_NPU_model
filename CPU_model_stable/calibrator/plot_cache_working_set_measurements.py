from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "hardware_calibration" / "raw" / "s920x20_recalib_20260428_3run"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cache_analysis"
PTRCHASE_RE = re.compile(r"size_bytes=(?P<size>\d+).*?ns_per_load=(?P<ns>[-+0-9.]+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot measured cache working-set pointer-chase trends.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def _parse_ptrchase(path: Path, *, run_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = PTRCHASE_RE.search(line)
        if not match:
            continue
        size_bytes = int(match.group("size"))
        rows.append(
            {
                "hardware": "S920X20",
                "run": run_name,
                "size_bytes": size_bytes,
                "size_kib": size_bytes / 1024.0,
                "size_mib": size_bytes / 1024.0 / 1024.0,
                "ns_per_load": float(match.group("ns")),
            }
        )
    return rows


def _load_measurements(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(raw_root.glob("run_*")):
        rows.extend(_parse_ptrchase(run_dir / "ptrchase_sweep.txt", run_name=run_dir.name))
    focus = _parse_ptrchase(raw_root / "l1_focus_ptrchase_10run.txt", run_name="l1_focus")
    rows.extend(focus)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No ptrchase measurements found under {raw_root}")
    return frame


def _plot(frame: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    summary = (
        frame.groupby(["hardware", "size_bytes", "size_kib", "size_mib"], dropna=False)
        .agg(ns_per_load_median=("ns_per_load", "median"), ns_per_load_min=("ns_per_load", "min"), ns_per_load_max=("ns_per_load", "max"), runs=("run", "count"))
        .reset_index()
        .sort_values("size_bytes")
    )
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "lines.markersize": 3.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.7, 3.15), constrained_layout=True)
    ax.plot(summary["size_kib"], summary["ns_per_load_median"], marker="o", color="#2f5f9e", label="S920X20 measured")
    ax.fill_between(
        summary["size_kib"].to_numpy(dtype=float),
        summary["ns_per_load_min"].to_numpy(dtype=float),
        summary["ns_per_load_max"].to_numpy(dtype=float),
        color="#2f5f9e",
        alpha=0.16,
        linewidth=0.0,
        label="run min-max",
    )
    cache_marks = [
        ("L1D", 64.0),
        ("L2", 1280.0),
        ("L3", 70.0 * 1024.0),
    ]
    for label, kib in cache_marks:
        ax.axvline(kib, color="0.45", linestyle="--", linewidth=0.8)
        ax.text(kib, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=6.5, color="0.25")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Pointer-chase working set (KiB, log scale)")
    ax.set_ylabel("Measured latency (ns/load)")
    ax.set_title("Measured Cache Working-Set Sensitivity")
    ax.grid(True, which="both", color="0.86", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / "cache_working_set_measured_trends.pdf", bbox_inches="tight")
    fig.savefig(output_root / "cache_working_set_measured_trends.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    args = _parse_args()
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = _load_measurements(raw_root)
    frame.to_csv(output_root / "cache_working_set_measurements_raw.csv", index=False)
    summary = _plot(frame, output_root)
    summary.to_csv(output_root / "cache_working_set_measurements_summary.csv", index=False)
    print(f"[cache-measured] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
