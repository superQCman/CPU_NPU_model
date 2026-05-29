from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mainline E2E relative-error CDF for one or two devices.")
    parser.add_argument("--device1-combo-metrics")
    parser.add_argument("--device2-combo-metrics")
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--threshold", default="0.10,0.15", help="Comma-separated relative-error thresholds to annotate.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not args.device1_combo_metrics and not args.device2_combo_metrics:
        parser.error("at least one of --device1-combo-metrics or --device2-combo-metrics is required")
    return args


def _parse_thresholds(raw: Any) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("at least one threshold is required")
    return sorted(dict.fromkeys(values))


def _load_device_frame(path: Path, *, device_id: str, label: str, route: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"combo metrics file not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "route" not in frame.columns:
        raise KeyError(f"{path} is missing route column")
    frame = frame[frame["route"].astype(str).eq(str(route))].copy()
    if frame.empty:
        raise ValueError(f"{path} has no rows for route={route!r}")
    if "coverage_ratio" in frame.columns:
        frame["coverage_ratio"] = pd.to_numeric(frame["coverage_ratio"], errors="coerce")
        frame = frame[frame["coverage_ratio"].fillna(0.0).ge(1.0)].copy()
    required = {"actual_e2e_us", "predicted_e2e_us"}
    if required.issubset(set(frame.columns)):
        actual = pd.to_numeric(frame["actual_e2e_us"], errors="coerce")
        pred = pd.to_numeric(frame["predicted_e2e_us"], errors="coerce")
        frame["relative_error"] = (pred - actual).abs() / actual.replace(0.0, np.nan)
    elif "ape" in frame.columns:
        frame["relative_error"] = pd.to_numeric(frame["ape"], errors="coerce")
    else:
        raise KeyError(f"{path} must contain actual/predicted E2E columns or ape")
    frame = frame[np.isfinite(frame["relative_error"].to_numpy(dtype=float))].copy()
    if frame.empty:
        raise ValueError(f"{path} has no finite relative-error rows for route={route!r}")
    frame["device_id"] = device_id
    frame["device_label"] = label
    frame["input_path"] = str(path)
    return frame


def _cdf_points(frame: pd.DataFrame) -> pd.DataFrame:
    values = np.sort(frame["relative_error"].to_numpy(dtype=float))
    n = len(values)
    return pd.DataFrame(
        {
            "relative_error": values,
            "cdf_percent": np.arange(1, n + 1, dtype=float) / float(n) * 100.0,
        }
    )


def _summary_row(frame: pd.DataFrame, *, thresholds: list[float]) -> dict[str, Any]:
    values = frame["relative_error"].to_numpy(dtype=float)
    row = {
        "device_id": str(frame["device_id"].iloc[0]),
        "device_label": str(frame["device_label"].iloc[0]),
        "combo_count": int(len(values)),
        "mape": float(np.mean(values)),
        "median_ape": float(np.percentile(values, 50)),
        "p90_ape": float(np.percentile(values, 90)),
        "p95_ape": float(np.percentile(values, 95)),
        "p99_ape": float(np.percentile(values, 99)),
        "max_ape": float(np.max(values)),
        "input_path": str(frame["input_path"].iloc[0]),
    }
    for threshold in thresholds:
        key = f"cdf_at_{threshold:.2f}_percent"
        row[key] = float(np.mean(values <= float(threshold)) * 100.0)
    if thresholds:
        row["cdf_at_threshold_percent"] = row[f"cdf_at_{thresholds[-1]:.2f}_percent"]
        row["threshold"] = float(thresholds[-1])
    return row


def _plot(points: pd.DataFrame, summary: pd.DataFrame, *, thresholds: list[float], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "Device_1": "#4C78A8",
        "Device_2": "#C77DA8",
    }
    markers = {
        "Device_1": "o",
        "Device_2": "s",
    }
    threshold_label_offsets = {
        (0, "Device_1"): (30, -8),
        (0, "Device_2"): (30, -38),
        (1, "Device_1"): (34, -6),
        (1, "Device_2"): (34, -42),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for _, row in summary.iterrows():
        device_id = str(row["device_id"])
        label = str(row["device_label"])
        part = points[points["device_id"].astype(str).eq(device_id)].copy()
        color = colors.get(device_id, None)
        ax.plot(part["relative_error"], part["cdf_percent"], lw=2.0, label=label, color=color)
        for threshold_index, threshold in enumerate(thresholds):
            key = f"cdf_at_{threshold:.2f}_percent"
            y = float(row[key])
            ax.scatter(
                [threshold],
                [y],
                s=34,
                marker=markers.get(device_id, "o"),
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
            offset = threshold_label_offsets.get((threshold_index, device_id), (8, 0))
            ax.annotate(
                f"{label} @ {threshold:.2f}: {y:.1f}%",
                xy=(threshold, y),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                va="bottom" if offset[1] >= 0 else "top",
                color=color,
                bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": color, "alpha": 0.92},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9, "shrinkA": 0.0, "shrinkB": 3.0},
            )
    metric_lines = ["MAPE"]
    for _, row in summary.iterrows():
        metric_lines.append(f"{row['device_label']}: {float(row['mape']):.2%}")
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
    p99 = float(summary["p99_ape"].max())
    xmax = max(0.30, p99 + 0.02, max(thresholds) + 0.04)
    for threshold_index, threshold in enumerate(thresholds):
        ax.axvline(threshold, color="#777777", ls="--", lw=1.0, alpha=0.75 if threshold_index == len(thresholds) - 1 else 0.45)
        ax.text(
            threshold,
            2.5,
            f"{threshold:.2f}",
            rotation=90,
            ha="right",
            va="bottom",
            color="#666666",
            fontsize=9,
        )
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 100.5)
    ax.set_xlabel("Relative error")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Mainline E2E Relative Error CDF")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "e2e_relative_error_cdf_mainline.png")
    fig.savefig(out_dir / "e2e_relative_error_cdf_mainline.pdf")
    plt.close(fig)


def _write_report(summary: pd.DataFrame, *, route: str, thresholds: list[float], out_dir: Path) -> None:
    threshold_cols = [f"cdf_at_{threshold:.2f}_percent" for threshold in thresholds]
    input_lines = []
    command_lines = ["python calibrator/plot_e2e_relative_error_cdf.py \\"]
    for _, row in summary.iterrows():
        device_id = str(row["device_id"])
        input_path = str(row["input_path"])
        input_lines.append(f"- {device_id} combo metrics: `{input_path}`")
        if device_id == "Device_1":
            command_lines.append(f"  --device1-combo-metrics {input_path} \\")
        elif device_id == "Device_2":
            command_lines.append(f"  --device2-combo-metrics {input_path} \\")
    command_lines.extend(
        [
            f"  --route {route} \\",
            f"  --threshold {','.join(f'{threshold:.2f}' for threshold in thresholds)} \\",
            f"  --out-dir {out_dir}",
        ]
    )
    lines = [
        "# Mainline E2E Relative Error CDF",
        "",
        f"- route: `{route}`",
        f"- thresholds: `{','.join(f'{threshold:.2f}' for threshold in thresholds)}`",
        "- scope: correct-hw full E2E combo-level results only",
        *input_lines,
        "",
        "| Device | Combos | MAPE | Median APE | P90 APE | P95 APE | Max APE | "
        + " | ".join(f"CDF @ {threshold:.2f}" for threshold in thresholds)
        + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        + " | ".join("---:" for _ in thresholds)
        + " |",
    ]
    for _, row in summary.iterrows():
        cdf_values = " | ".join(f"{float(row[col]):.1f}%" for col in threshold_cols)
        lines.append(
            "| {device} | {count} | {mape:.2%} | {median:.2%} | {p90:.2%} | {p95:.2%} | {max_ape:.2%} | {cdf_values} |".format(
                device=row["device_label"],
                count=int(row["combo_count"]),
                mape=float(row["mape"]),
                median=float(row["median_ape"]),
                p90=float(row["p90_ape"]),
                p95=float(row["p95_ape"]),
                max_ape=float(row["max_ape"]),
                cdf_values=cdf_values,
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `e2e_relative_error_cdf_mainline.png`",
            "- `e2e_relative_error_cdf_mainline.pdf`",
            "- `e2e_relative_error_cdf_points.csv`",
            "- `e2e_relative_error_cdf_summary.csv`",
            "",
            "## Reproduce",
            "",
            "```bash",
            "# Run from the repository root",
            "",
            *command_lines,
            "```",
        ]
    )
    (out_dir / "E2E_RELATIVE_ERROR_CDF_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_input_aliases(summary: pd.DataFrame, out_dir: Path) -> None:
    alias_dir = out_dir / "input_aliases"
    alias_dir.mkdir(parents=True, exist_ok=True)
    for _, row in summary.iterrows():
        source = Path(str(row["input_path"]))
        device_id = str(row["device_id"]).lower().replace("_", "")
        alias = alias_dir / f"{device_id}_e2e_pipeline_combo_metrics.csv"
        if source.resolve() == alias.resolve():
            continue
        shutil.copyfile(source, alias)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    route = str(args.route)
    thresholds = _parse_thresholds(args.threshold)
    frames = []
    if args.device1_combo_metrics:
        frames.append(
            _load_device_frame(
                Path(args.device1_combo_metrics),
                device_id="Device_1",
                label="Device_1",
                route=route,
            )
        )
    if args.device2_combo_metrics:
        frames.append(
            _load_device_frame(
                Path(args.device2_combo_metrics),
                device_id="Device_2",
                label="Device_2",
                route=route,
            )
        )
    point_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for frame in frames:
        points = _cdf_points(frame)
        points["device_id"] = str(frame["device_id"].iloc[0])
        points["device_label"] = str(frame["device_label"].iloc[0])
        point_frames.append(points)
        summary_rows.append(_summary_row(frame, thresholds=thresholds))
    all_points = pd.concat(point_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("device_id").reset_index(drop=True)
    all_points.to_csv(out_dir / "e2e_relative_error_cdf_points.csv", index=False)
    summary.to_csv(out_dir / "e2e_relative_error_cdf_summary.csv", index=False)
    _write_input_aliases(summary, out_dir)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "thresholds": thresholds,
        "python": sys.version,
        "outputs": {
            "png": str(out_dir / "e2e_relative_error_cdf_mainline.png"),
            "pdf": str(out_dir / "e2e_relative_error_cdf_mainline.pdf"),
            "points": str(out_dir / "e2e_relative_error_cdf_points.csv"),
            "summary": str(out_dir / "e2e_relative_error_cdf_summary.csv"),
        },
    }
    (out_dir / "e2e_relative_error_cdf_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot(all_points, summary, thresholds=thresholds, out_dir=out_dir)
    _write_report(summary, route=route, thresholds=thresholds, out_dir=out_dir)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary.to_dict(orient="records")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
