from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import train_mlp
from .artifact_loader import (
    DEFAULT_ORT_ROOT,
    DEFAULT_ARTIFACT_ROOT as DEFAULT_SINGLE_OP_SOURCE,
    build_combo_specs,
    load_op_shapes_frame,
    load_prediction_frame,
    load_timeline_frame,
)
from .schedule_engine import (
    analyze_embedding_execution,
    build_op_graph,
    compute_mean_batch_span,
    schedule_combo,
)

from .report_config import (
    ABLATION_ARTIFACT_ROOT,
    BASELINE_MODEL_ROOT,
    FAIR_SINGLE_MLP_ALPHA,
    FAIR_SINGLE_MLP_BATCH_SIZE,
    FAIR_SINGLE_MLP_HIDDEN_LAYERS,
    FAIR_SINGLE_MLP_LEARNING_RATE_INIT,
    FAIR_SINGLE_MLP_MAX_ITER,
    FAIR_SINGLE_MLP_SEED,
    CPU_REPORT_FIGURE_FONT_SCALE,
    CPU_REPORT_DRAFT_PATH,
    CPU_REPORT_OUTPUT_ROOT,
    CPU_REPORT_SINGLE_ONLY_DATA_SYNC_PATH,
    E2E_ARTIFACT_ROOT,
    FIGURE_FILENAMES,
    MODEL_GROUP_ORDER,
    OOD_ARTIFACT_ROOT,
    OOD_BATCH_HOLDS,
    OOD_NUM_THREADS_HOLD,
    REPRESENTATIVE_PANEL_FILENAME,
    REPRESENTATIVE_OP_TYPES,
    SINGLE_OP_ARTIFACT_ROOT,
    TABLE_FILENAMES,
    TIMELINE_CASES,
)


@dataclass(frozen=True)
class SectionResult:
    name: str
    outputs: dict[str, str]


PREDICTION_SOURCE_GROUPED = "grouped"
PREDICTION_SOURCE_SINGLE_FAIR = "single_fair"
PREDICTION_SOURCE_CHOICES = (PREDICTION_SOURCE_GROUPED, PREDICTION_SOURCE_SINGLE_FAIR)


def ensure_output_layout(output_root: Path | None = None) -> dict[str, Path]:
    root = Path(output_root or CPU_REPORT_OUTPUT_ROOT)
    layout = {
        "root": root,
        "single_op": root / "single_op",
        "e2e": root / "e2e",
        "ablation": root / "ablation",
        "figures": root / "figures",
        "tables": root / "tables",
        "manifests": root / "manifests",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def _fmt_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if abs(value) >= 1000.0:
            return f"{value:,.3f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame, title: str | None = None) -> str:
    rows = frame.fillna("").copy()
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    if rows.empty:
        lines.append("_No rows_")
        lines.append("")
        return "\n".join(lines)
    headers = list(rows.columns)
    rendered_rows = [[_fmt_cell(cell) for cell in row] for row in rows.itertuples(index=False, name=None)]
    widths = [len(str(header)) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: Iterable[str]) -> str:
        cells = [str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    lines.append(render_row(headers))
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rendered_rows:
        lines.append(render_row(row))
    lines.append("")
    return "\n".join(lines)


def write_frame_csv_md(frame: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> tuple[Path, Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(frame, title), encoding="utf-8")
    return csv_path, md_path


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.size": 10.5 * CPU_REPORT_FIGURE_FONT_SCALE,
            "axes.labelsize": 10.5 * CPU_REPORT_FIGURE_FONT_SCALE,
            "axes.titlesize": 11.0 * CPU_REPORT_FIGURE_FONT_SCALE,
            "xtick.labelsize": 9.5 * CPU_REPORT_FIGURE_FONT_SCALE,
            "ytick.labelsize": 9.5 * CPU_REPORT_FIGURE_FONT_SCALE,
            "legend.fontsize": 9.0 * CPU_REPORT_FIGURE_FONT_SCALE,
            "figure.titlesize": 11.5 * CPU_REPORT_FIGURE_FONT_SCALE,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _scaled_font(size: float) -> float:
    return float(size) * CPU_REPORT_FIGURE_FONT_SCALE


def _scaled_figsize(width: float, height: float, *, width_boost: float = 0.18, height_boost: float = 0.28) -> tuple[float, float]:
    scale_delta = max(0.0, CPU_REPORT_FIGURE_FONT_SCALE - 1.0)
    return (
        float(width) * (1.0 + width_boost * scale_delta),
        float(height) * (1.0 + height_boost * scale_delta),
    )


def _dynamic_width(base: float, label_count: int, *, per_label: float = 0.55, max_width: float = 22.0) -> float:
    return min(max_width, max(base, base + max(0, label_count - 4) * per_label))


def _dynamic_height(base: float, item_count: int, *, per_item: float = 0.22, max_height: float = 18.0) -> float:
    return min(max_height, max(base, base + max(0, item_count - 4) * per_item))


def _wrap_axis_label(label: Any, *, width: int = 18) -> str:
    return textwrap.fill(str(label), width=width, break_long_words=False, break_on_hyphens=False)


def _set_categorical_xticklabels(ax, labels: Sequence[Any], *, wrap_width: int = 18, rotation: float = 0.0) -> None:
    rendered_labels = [_wrap_axis_label(label, width=wrap_width) for label in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(rendered_labels, rotation=rotation, ha="center" if rotation == 0 else "right")


def _place_legend_outside(ax, *, ncols: int = 1, side: str = "right", loc: str | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if side == "inside":
        ax.legend(
            handles,
            labels,
            frameon=False,
            ncols=ncols,
            loc=loc or "best",
            handlelength=1.6,
        )
        return
    if side == "top":
        ax.legend(
            handles,
            labels,
            frameon=False,
            ncols=ncols,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            borderaxespad=0.0,
            handlelength=1.6,
        )
        return
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncols=ncols,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        handlelength=1.6,
    )


def _style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)


def save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pad = 1.0 + 0.55 * max(0.0, CPU_REPORT_FIGURE_FONT_SCALE - 1.0)
    fig.tight_layout(pad=pad, w_pad=pad * 0.8, h_pad=pad * 0.9)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def add_audit_callout(ax, lines: Sequence[str] | None, *, loc: str = "upper right") -> None:
    if not lines:
        return
    anchors = {
        "upper left": (0.0, 1.02, "left", "bottom"),
        "upper right": (1.0, 1.02, "right", "bottom"),
        "lower left": (0.0, -0.16, "left", "top"),
        "lower right": (1.0, -0.16, "right", "top"),
        "inside upper left": (0.02, 0.98, "left", "top"),
        "inside upper center": (0.50, 0.98, "center", "top"),
        "inside upper right": (0.98, 0.98, "right", "top"),
        "inside lower left": (0.02, 0.02, "left", "bottom"),
        "inside lower center": (0.50, 0.02, "center", "bottom"),
        "inside lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["upper right"])
    text = "\n".join(str(line) for line in lines if line is not None and str(line).strip())
    if not text:
        return
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=_scaled_font(8.5),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.83, edgecolor="#C8C8C8"),
        clip_on=False,
    )


def _draw_scatter_on_ax(
    ax,
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    reference_line: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    panel_tag: str | None = None,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper left",
    show_legend: bool = True,
) -> None:
    palette = ["#4C78A8", "#E45756", "#54A24B", "#B279A2", "#F58518", "#72B7B2"]
    if hue and hue in frame.columns:
        for idx, (label, sub) in enumerate(frame.groupby(hue, sort=True)):
            ax.scatter(
                sub[x],
                sub[y],
                s=18,
                alpha=0.75,
                label=str(label),
                color=palette[idx % len(palette)],
            )
        if show_legend:
            _place_legend_outside(ax, side="right")
    else:
        ax.scatter(frame[x], frame[y], s=18, alpha=0.75, color="#4477aa")
    if reference_line and not frame.empty:
        lo = float(min(frame[x].min(), frame[y].min()))
        hi = float(max(frame[x].max(), frame[y].max()))
        ax.plot([lo, hi], [lo, hi], color="#222222", linewidth=1.0, linestyle="--")
    _style_axes(ax, title or "", xlabel or x, ylabel or y)
    if title:
        ax.set_title(title, pad=8.0)
    if panel_tag:
        ax.text(
            0.5,
            -0.22,
            panel_tag,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=_scaled_font(12.5),
            fontweight="bold",
        )
    add_audit_callout(ax, audit_lines, loc=audit_loc)


def plot_bar(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    *,
    xlabel: str = "",
    ylabel: str = "",
    color: str = "#4477aa",
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
    label_rotation: float = 0.0,
    label_wrap_width: int = 20,
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=_scaled_figsize(_dynamic_width(10.0, len(frame), per_label=0.9), 5.8))
    ax.bar(frame[x].astype(str), frame[y].astype(float), color=color)
    _style_axes(ax, title, xlabel or x, ylabel or y)
    _set_categorical_xticklabels(ax, frame[x].astype(str).tolist(), wrap_width=label_wrap_width, rotation=label_rotation)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_grouped_bar(
    frame: pd.DataFrame,
    index_col: str,
    value_cols: list[str],
    path: Path,
    title: str,
    *,
    ylabel: str,
    legend_labels: list[str] | None = None,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
    legend_side: str = "top",
    legend_loc: str | None = None,
    label_rotation: float = 0.0,
    label_wrap_width: int = 18,
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=_scaled_figsize(_dynamic_width(11.0, len(frame), per_label=0.95), 6.0))
    x = range(len(frame))
    width = 0.8 / max(1, len(value_cols))
    labels = legend_labels or value_cols
    for idx, col in enumerate(value_cols):
        offsets = [pos + (idx - (len(value_cols) - 1) / 2.0) * width for pos in x]
        ax.bar(offsets, frame[col].astype(float), width=width, label=labels[idx])
    _set_categorical_xticklabels(ax, frame[index_col].astype(str).tolist(), wrap_width=label_wrap_width, rotation=label_rotation)
    _style_axes(ax, title, index_col, ylabel)
    _place_legend_outside(ax, ncols=min(len(value_cols), 3), side=legend_side, loc=legend_loc)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    *,
    hue: str | None = None,
    reference_line: bool = True,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper left",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=_scaled_figsize(7.3 if hue else 6.8, 6.8))
    _draw_scatter_on_ax(
        ax,
        frame,
        x,
        y,
        hue=hue,
        reference_line=reference_line,
        audit_lines=audit_lines,
        audit_loc=audit_loc,
    )
    return save_figure(fig, path)


def plot_heatmap(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    xlabel: str = "",
    ylabel: str = "",
    annot: bool = True,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(
        figsize=_scaled_figsize(
            _dynamic_width(8.5, len(frame.columns), per_label=0.85),
            _dynamic_height(4.8, len(frame.index), per_item=0.55),
        )
    )
    matrix = frame.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels([str(col) for col in frame.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels([str(idx) for idx in frame.index])
    _style_axes(ax, title, xlabel, ylabel)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if annot:
        annotation_font = _scaled_font(max(6.2, 8.0 - 0.25 * max(len(frame.columns), len(frame.index))))
        for row_idx, row_label in enumerate(frame.index):
            for col_idx, col_label in enumerate(frame.columns):
                value = matrix[row_idx, col_idx]
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white", fontsize=annotation_font)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_boxplot(
    values_by_label: dict[str, list[float]],
    path: Path,
    title: str,
    *,
    ylabel: str,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=_scaled_figsize(_dynamic_width(10.0, len(values_by_label), per_label=0.9), 5.8))
    labels = list(values_by_label.keys())
    values = [values_by_label[label] for label in labels]
    ax.boxplot(values, labels=labels, showmeans=True)
    _style_axes(ax, title, "", ylabel)
    ax.set_xticklabels([_wrap_axis_label(label, width=18) for label in labels], rotation=0.0, ha="center")
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_line(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    *,
    group_col: str | None = None,
    ylabel: str | None = None,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
    legend_side: str = "right",
    legend_loc: str | None = None,
    legend_ncols: int = 1,
    figsize_override: tuple[float, float] | None = None,
    font_scale_multiplier: float = 1.0,
) -> Path:
    plt = _import_pyplot()
    figure_size = figsize_override or _scaled_figsize(
        _dynamic_width(10.0, frame[x].nunique() if x in frame.columns else len(frame), per_label=0.55),
        5.8,
    )
    fig, ax = plt.subplots(figsize=figure_size)
    if group_col and group_col in frame.columns:
        for label, sub in frame.groupby(group_col):
            ax.plot(sub[x], sub[y], marker="o", linewidth=1.5, label=str(label))
        _place_legend_outside(ax, side=legend_side, loc=legend_loc, ncols=legend_ncols)
    else:
        ax.plot(frame[x], frame[y], marker="o", linewidth=1.5, color="#4477aa")
    _style_axes(ax, title, x, ylabel or y)
    if font_scale_multiplier != 1.0:
        ax.xaxis.label.set_size(ax.xaxis.label.get_size() * font_scale_multiplier)
        ax.yaxis.label.set_size(ax.yaxis.label.get_size() * font_scale_multiplier)
        ax.tick_params(
            axis="both",
            labelsize=max(
                _scaled_font(9.5),
                ax.xaxis.get_ticklabels()[0].get_size() * font_scale_multiplier if ax.xaxis.get_ticklabels() else _scaled_font(9.5),
            ),
        )
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_representative_panel(
    specs: list[dict[str, Any]],
    path: Path,
) -> Path:
    plt = _import_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=_scaled_figsize(15.5, 11.5), sharey=False)
    tags = ["(a)", "(b)", "(c)", "(d)"]
    for ax, spec, tag in zip(axes.flatten(), specs, tags):
        _draw_scatter_on_ax(
            ax,
            spec["frame"],
            spec["x"],
            spec["y"],
            hue=spec.get("hue"),
            reference_line=False,
            xlabel=spec.get("xlabel"),
            ylabel=spec.get("ylabel", "APE"),
            title=spec.get("title"),
            panel_tag=tag,
            show_legend=spec.get("show_legend", True),
        )
    fig.subplots_adjust(hspace=0.58, wspace=0.58, bottom=0.12)
    return save_figure(fig, path)


def plot_cdf(
    values_by_label: dict[str, list[float]],
    path: Path,
    title: str,
    *,
    xlabel: str,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "lower right",
    legend_side: str = "right",
    legend_loc: str | None = None,
    xlim: tuple[float, float] | None = None,
    marker_x: float | None = None,
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=_scaled_figsize(9.8, 6.2))
    palette = ["#E45756", "#4C78A8", "#54A24B", "#B279A2", "#F58518"]
    marker_points: list[tuple[str, float, float, str]] = []
    for idx, (label, values) in enumerate(values_by_label.items()):
        clean = sorted(float(value) for value in values if value is not None and not math.isnan(float(value)))
        if not clean:
            continue
        cdf = [(pos + 1) / len(clean) * 100.0 for pos in range(len(clean))]
        color = palette[idx % len(palette)]
        ax.plot(clean, cdf, linewidth=2.0, label=label, color=color)
        if marker_x is not None:
            hit_idx = int(np.searchsorted(clean, float(marker_x), side="right"))
            hit_y = (hit_idx / len(clean)) * 100.0
            marker_points.append((str(label), float(marker_x), hit_y, color))
    _style_axes(ax, title, xlabel, "CDF (%)")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if marker_x is not None:
        ax.axvline(float(marker_x), color="#444444", linestyle="--", linewidth=1.2, alpha=0.9)
        if marker_points:
            min_gap = 4.0
            sorted_points = sorted(marker_points, key=lambda item: item[2])
            adjusted_y: list[float] = []
            for _, _, y_value, _ in sorted_points:
                adjusted_y.append(y_value if not adjusted_y else max(y_value, adjusted_y[-1] + min_gap))
            for (label, x_value, actual_y, color), label_y in zip(sorted_points, adjusted_y):
                ax.scatter([x_value], [actual_y], color=color, s=28, zorder=5)
                ax.annotate(
                    f"{actual_y:.1f}%",
                    xy=(x_value, actual_y),
                    xytext=(x_value + 0.025, min(label_y, 99.0)),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=_scaled_font(8.0),
                    color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.9, alpha=0.9),
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, alpha=0.8),
                )
    _place_legend_outside(ax, side=legend_side, loc=legend_loc)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_flow(
    steps: list[str],
    path: Path,
    title: str,
    *,
    subtitle: str | None = None,
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=_scaled_figsize(14.0, 3.6))
    ax.axis("off")
    x_positions = [0.08 + idx * (0.84 / max(1, len(steps) - 1)) for idx in range(len(steps))]
    for idx, (x_pos, step) in enumerate(zip(x_positions, steps)):
        wrapped_step = textwrap.fill(step, width=16)
        ax.text(
            x_pos,
            0.58,
            wrapped_step,
            ha="center",
            va="center",
            fontsize=_scaled_font(10.0),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#4C78A8", linewidth=1.2),
            transform=ax.transAxes,
        )
        if idx < len(steps) - 1:
            next_x = x_positions[idx + 1]
            ax.annotate(
                "",
                xy=(next_x - 0.045, 0.58),
                xytext=(x_pos + 0.045, 0.58),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#666666"),
            )
    return save_figure(fig, path)


def plot_gantt(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    label_col: str,
    start_col: str,
    end_col: str,
    hue_col: str | None = None,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=_scaled_figsize(12.8, _dynamic_height(4.8, len(frame), per_item=0.34)))
    labels = frame[label_col].astype(str).tolist()
    colors = {
        "bottom": "#4C78A8",
        "embedding": "#F58518",
        "tail": "#54A24B",
        "barrier": "#B279A2",
    }
    for idx, row in enumerate(frame.itertuples(index=False)):
        color = colors.get(str(getattr(row, hue_col)) if hue_col else "bottom", "#4C78A8")
        ax.broken_barh([(float(getattr(row, start_col)), float(getattr(row, end_col)) - float(getattr(row, start_col)))], (idx - 0.35, 0.7), facecolors=color)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    _style_axes(ax, title, "time (us)", "")
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_simple_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    path: Path,
    title: str,
    *,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=_scaled_figsize(13.0, 8.0))
    if nodes.empty:
        ax.text(0.5, 0.5, "No graph data", ha="center", va="center")
        ax.axis("off")
        return save_figure(fig, path)

    x_values = nodes["layer"].astype(float)
    y_positions = defaultdict(list)
    for layer in sorted(set(x_values)):
        layer_nodes = nodes[nodes["layer"] == layer].copy()
        for row_idx, row in enumerate(layer_nodes.itertuples(index=False)):
            y_positions[int(layer)].append(row_idx)
    positions: dict[int, tuple[float, float]] = {}
    for layer in sorted(set(x_values)):
        layer_nodes = nodes[nodes["layer"] == layer].copy()
        count = len(layer_nodes)
        for idx, row in enumerate(layer_nodes.itertuples(index=False)):
            y = (count - 1) / 2.0 - idx
            positions[int(row.node_idx)] = (float(layer), float(y))

    color_map = {
        "bottom": "#4C78A8",
        "embedding": "#F58518",
        "tail": "#54A24B",
        "constant": "#C7C7C7",
    }
    for edge in edges.itertuples(index=False):
        src = int(edge.src)
        dst = int(edge.dst)
        if src not in positions or dst not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8, alpha=0.8))

    for node in nodes.itertuples(index=False):
        x, y = positions[int(node.node_idx)]
        color = color_map.get(str(node.partition), "#4C78A8")
        ax.scatter([x], [y], s=500 * max(1.0, CPU_REPORT_FIGURE_FONT_SCALE), color=color, edgecolor="#222222", linewidth=0.7, zorder=3)
        ax.text(x, y, str(node.label), ha="center", va="center", fontsize=_scaled_font(8.0), color="white", zorder=4)

    ax.set_xlabel("topological depth")
    ax.set_ylabel("node layer")
    ax.set_yticks([])
    ax.grid(False)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def _partition_legend_handles(include_barrier: bool = False):
    from matplotlib.patches import Patch

    labels_and_colors = [
        ("bottom", "#4C78A8", "Bottom"),
        ("embedding", "#F58518", "Embedding branch"),
        ("tail", "#54A24B", "Tail"),
    ]
    if include_barrier:
        labels_and_colors.append(("barrier", "#B279A2", "Barrier"))
    return [Patch(facecolor=color, edgecolor="none", label=label) for _, color, label in labels_and_colors]


def _table_path(layout: dict[str, Path], figure_no: str) -> Path:
    return layout["tables"] / TABLE_FILENAMES[figure_no]


def _figure_path(layout: dict[str, Path], figure_no: str) -> Path:
    return layout["figures"] / FIGURE_FILENAMES[figure_no]


def _write_summary_bundle(section_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    section_dir.mkdir(parents=True, exist_ok=True)
    return write_json(section_dir / f"{name}_summary.json", payload)


def _sanitize_list(values: Iterable[Any]) -> str:
    return "|".join(str(value) for value in values)


def _resolve_root(value: Path | None, default: Path) -> Path:
    return Path(value) if value is not None else Path(default)


def _load_grouped_feature_pool(single_op_root: Path) -> dict[str, Any]:
    dataset_summary = read_json(single_op_root / "dataset_summary.json")
    categorical_features = list(dataset_summary.get("shared_categorical_features", []))
    per_group = dataset_summary.get("per_model_group_numeric_features", {})
    ordered_numeric: list[str] = []
    for group_name in dataset_summary.get("model_group_order", list(MODEL_GROUP_ORDER)):
        for column in per_group.get(group_name, []):
            if column not in ordered_numeric:
                ordered_numeric.append(column)
    if not ordered_numeric:
        raise RuntimeError(f"No grouped numeric feature pool found under {single_op_root}")
    return {
        "dataset_summary": dataset_summary,
        "numeric_features": ordered_numeric,
        "categorical_features": categorical_features,
    }


def ensure_fair_single_mlp_artifact(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    force_retrain: bool = False,
    hidden_layers: Sequence[int] | None = None,
    max_iter: int | None = None,
) -> dict[str, Any]:
    layout = ensure_output_layout(output_root)
    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    feature_pool = _load_grouped_feature_pool(single_op_root)
    section_dir = layout["single_op"] / "fair_single_mlp_same_split_same_feature_pool"
    dataset_dir = section_dir / "dataset"
    model_dir = section_dir / "model"
    manifest_path = section_dir / "manifest.json"
    metrics_path = model_dir / "metrics.json"
    predictions_test_path = model_dir / "predictions_test.csv"
    selected_hidden_layers = tuple(int(v) for v in (hidden_layers or FAIR_SINGLE_MLP_HIDDEN_LAYERS))
    selected_max_iter = int(max_iter or FAIR_SINGLE_MLP_MAX_ITER)

    manifest_payload = {
        "source_single_op_artifact_root": str(single_op_root),
        "source_dataset_csv": str(single_op_root / "classed_dataset_full.csv"),
        "numeric_features": feature_pool["numeric_features"],
        "categorical_features": feature_pool["categorical_features"],
        "feature_count": len(feature_pool["numeric_features"]) + len(feature_pool["categorical_features"]),
        "training_hparams": {
            "hidden_layers": list(selected_hidden_layers),
            "batch_size": FAIR_SINGLE_MLP_BATCH_SIZE,
            "max_iter": selected_max_iter,
            "alpha": FAIR_SINGLE_MLP_ALPHA,
            "learning_rate_init": FAIR_SINGLE_MLP_LEARNING_RATE_INIT,
            "seed": FAIR_SINGLE_MLP_SEED,
            "train_device": "auto",
            "target_mode": "direct_us",
            "log_target": True,
        },
    }

    rebuild_dataset = force_retrain or not manifest_path.exists() or not dataset_dir.exists()
    if not rebuild_dataset and manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        rebuild_dataset = existing_manifest != manifest_payload

    if rebuild_dataset:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_frame = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
        feature_manifest = {
            "feature_dialect": "report_fair_union_same_split",
            "numeric_features": feature_pool["numeric_features"],
            "categorical_features": feature_pool["categorical_features"],
            "analytical_base_column": "ana_calib_total_us",
            "residual_target_column": "label_operator_residual_log",
            "feature_pool_description": "Union of all grouped-model numeric features plus shared categorical features.",
        }
        dataset_frame.to_csv(dataset_dir / "dataset_full.csv", index=False)
        for split_name in ["train", "val", "test"]:
            dataset_frame[dataset_frame["split"] == split_name].to_csv(dataset_dir / f"{split_name}.csv", index=False)
        write_json(dataset_dir / "feature_columns.json", feature_manifest)
        write_json(manifest_path, manifest_payload)

    retrain_model = rebuild_dataset or force_retrain or not metrics_path.exists() or not predictions_test_path.exists()
    if retrain_model:
        train_mlp.train_model(
            data_dir=dataset_dir,
            output_dir=model_dir,
            hidden_layers=selected_hidden_layers,
            batch_size=FAIR_SINGLE_MLP_BATCH_SIZE,
            max_iter=selected_max_iter,
            alpha=FAIR_SINGLE_MLP_ALPHA,
            learning_rate_init=FAIR_SINGLE_MLP_LEARNING_RATE_INIT,
            seed=FAIR_SINGLE_MLP_SEED,
            log_target=True,
            target_mode="direct_us",
            train_device="auto",
            npu_device_id=0,
            early_stopping_patience=12,
            export_onnx=True,
            onnx_opset=17,
        )

    metrics = read_json(metrics_path)
    predictions_test = pd.read_csv(predictions_test_path, low_memory=False)
    payload = {
        "artifact_dir": str(section_dir),
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path),
        "predictions_test_path": str(predictions_test_path),
        "feature_count": manifest_payload["feature_count"],
        "numeric_features": feature_pool["numeric_features"],
        "categorical_features": feature_pool["categorical_features"],
        "metrics": metrics,
        "predictions_test": predictions_test,
    }
    summary_payload = {
        "artifact_dir": payload["artifact_dir"],
        "dataset_dir": payload["dataset_dir"],
        "model_dir": payload["model_dir"],
        "manifest_path": payload["manifest_path"],
        "metrics_path": payload["metrics_path"],
        "predictions_test_path": payload["predictions_test_path"],
        "feature_count": payload["feature_count"],
        "numeric_features": payload["numeric_features"],
        "categorical_features": payload["categorical_features"],
        "test_metrics": metrics.get("metrics", {}).get("test", {}),
    }
    _write_summary_bundle(layout["single_op"], "single_op_fair_baseline", summary_payload)
    return payload


def _normalize_prediction_source(prediction_source: str) -> str:
    normalized = str(prediction_source or PREDICTION_SOURCE_GROUPED).strip().lower()
    if normalized not in PREDICTION_SOURCE_CHOICES:
        raise ValueError(f"Unsupported prediction_source={prediction_source!r}; expected one of {PREDICTION_SOURCE_CHOICES}")
    return normalized


def _prediction_source_model_name(prediction_source: str) -> str:
    normalized = _normalize_prediction_source(prediction_source)
    if normalized == PREDICTION_SOURCE_SINGLE_FAIR:
        return "Single MLP"
    return "Grouped analytical-MLP"


def _load_model_prediction_frame(
    output_root: Path | None,
    *,
    single_op_root: Path,
    prediction_source: str,
    test_dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    normalized = _normalize_prediction_source(prediction_source)
    if test_dataset is None:
        dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
        test_dataset = dataset_df[dataset_df["split"] == "test"].copy()

    if normalized == PREDICTION_SOURCE_SINGLE_FAIR:
        fair_baseline = ensure_fair_single_mlp_artifact(output_root, single_op_artifact_root=single_op_root)
        prediction_df = fair_baseline["predictions_test"].copy()
        metadata_columns = [
            "row_uid",
            "op_type",
            "model_group",
            "num_threads",
            "batch_size",
            "num_indices_per_lookup",
            "inter_threads",
        ]
        missing_metadata_columns = [column for column in metadata_columns if column not in test_dataset.columns]
        if missing_metadata_columns:
            metadata_source = pd.read_csv(
                single_op_root / "classed_dataset_full.csv",
                usecols=["row_uid", *missing_metadata_columns],
                low_memory=False,
            ).drop_duplicates("row_uid")
            test_dataset = test_dataset.merge(
                metadata_source,
                on="row_uid",
                how="left",
                validate="one_to_one",
            )
        metadata = test_dataset[metadata_columns].drop_duplicates("row_uid")
        prediction_df = prediction_df.merge(metadata, on="row_uid", how="left", validate="one_to_one")
        if "split" not in prediction_df.columns:
            prediction_df["split"] = "test"
        return {
            "prediction_source": normalized,
            "prediction_df": prediction_df,
            "model_name": _prediction_source_model_name(normalized),
            "model_root": str(fair_baseline["model_dir"]),
            "feature_count": int(fair_baseline["feature_count"]),
        }

    prediction_df = load_prediction_frame(single_op_root, split="test")
    return {
        "prediction_source": normalized,
        "prediction_df": prediction_df,
        "model_name": _prediction_source_model_name(normalized),
        "model_root": str(single_op_root / "models" / "combined"),
        "feature_count": None,
    }


def _build_pipeline_combo_predictions(prediction_df: pd.DataFrame) -> pd.DataFrame:
    combo_specs = build_combo_specs(prediction_df, ort_root=Path(DEFAULT_ORT_ROOT))
    combo_map = {(spec.case_id, spec.combo): spec for spec in combo_specs}
    pipeline_rows: list[dict[str, Any]] = []
    for (case_id, combo), combo_rows in prediction_df.groupby(["case_id", "combo"]):
        combo_spec = combo_map[(case_id, combo)]
        graph = build_op_graph(load_op_shapes_frame(combo_spec.artifact_paths.shape_csv))
        schedule_result = schedule_combo(combo_spec, graph, combo_rows.copy())
        pipeline_rows.append(
            {
                "case_id": case_id,
                "combo": combo,
                "batch_size": int(combo_spec.batch_size),
                "num_indices_per_lookup": int(combo_spec.num_indices_per_lookup),
                "inter_threads": int(combo_spec.inter_threads),
                "predicted_e2e_us": float(schedule_result.predicted_full_graph_us),
            }
        )
    return pd.DataFrame(pipeline_rows)


def _load_e2e_full_metrics(
    output_root: Path | None,
    *,
    single_op_root: Path,
    e2e_root: Path,
    prediction_source: str,
) -> pd.DataFrame:
    normalized = _normalize_prediction_source(prediction_source)
    if normalized == PREDICTION_SOURCE_GROUPED:
        return pd.read_csv(e2e_root / "full_combo_metrics.csv", low_memory=False)

    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    prediction_bundle = _load_model_prediction_frame(
        output_root,
        single_op_root=single_op_root,
        prediction_source=normalized,
        test_dataset=test_dataset,
    )
    pipeline_combo = _build_pipeline_combo_predictions(prediction_bundle["prediction_df"])
    actual_frame = pd.read_csv(e2e_root / "full_combo_metrics.csv", low_memory=False)[
        ["case_id", "combo", "actual_e2e_us", "batch_size", "inter_threads"]
    ].copy()
    full_metrics = actual_frame.merge(
        pipeline_combo[["case_id", "combo", "predicted_e2e_us"]],
        on=["case_id", "combo"],
        how="left",
        validate="one_to_one",
    )
    full_metrics["abs_error_us"] = (full_metrics["predicted_e2e_us"] - full_metrics["actual_e2e_us"]).abs()
    full_metrics["ape"] = (
        full_metrics["abs_error_us"] / full_metrics["actual_e2e_us"].replace(0.0, pd.NA)
    ).fillna(0.0)
    return full_metrics


def _metric_summary(metric_rows: list[dict[str, Any]], predicted_key: str, actual_key: str) -> dict[str, Any]:
    if not metric_rows:
        return {
            "count": 0,
            "mae_us": None,
            "mape": None,
            "p50_ape": None,
            "p95_ape": None,
            "worst_combo": None,
        }
    frame = pd.DataFrame(metric_rows)
    worst_row = frame.sort_values("abs_error_us", ascending=False).iloc[0]
    return {
        "count": int(len(frame)),
        "mae_us": float(frame["abs_error_us"].mean()),
        "mape": float(frame["ape"].mean()),
        "p50_ape": float(frame["ape"].quantile(0.50)),
        "p95_ape": float(frame["ape"].quantile(0.95)),
        "worst_combo": {
            "case_id": worst_row["case_id"],
            "combo": worst_row["combo"],
            predicted_key: float(worst_row[predicted_key]),
            actual_key: float(worst_row[actual_key]),
            "abs_error_us": float(worst_row["abs_error_us"]),
            "ape": float(worst_row["ape"]),
        },
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _topological_layers(graph: dict[int, Any], node_subset: set[int] | None = None) -> dict[int, int]:
    subset = node_subset if node_subset is not None else set(graph)
    indegree = {node_idx: 0 for node_idx in subset}
    successors: dict[int, list[int]] = defaultdict(list)
    for node_idx in subset:
        node = graph[node_idx]
        for pred in node.predecessors:
            if pred in subset:
                indegree[node_idx] += 1
                successors[pred].append(node_idx)
    ready = sorted([node_idx for node_idx, degree in indegree.items() if degree == 0])
    layer = {node_idx: 0 for node_idx in ready}
    while ready:
        node_idx = ready.pop(0)
        for succ in sorted(successors.get(node_idx, [])):
            layer[succ] = max(layer.get(succ, 0), layer[node_idx] + 1)
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
                ready.sort()
    return layer


def _graph_subframe(graph: dict[int, Any], keep_predicate) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_nodes = [node for node in graph.values() if keep_predicate(node)]
    kept_indices = {node.node_idx for node in kept_nodes}
    nodes = []
    edges = []
    layers = _topological_layers(graph, kept_indices)
    for node in kept_nodes:
        nodes.append(
            {
                "node_idx": node.node_idx,
                "label": node.op_type,
                "partition": node.partition,
                "layer": layers.get(node.node_idx, 0),
            }
        )
        for pred in node.predecessors:
            if pred in kept_indices:
                edges.append({"src": pred, "dst": node.node_idx})
    return pd.DataFrame(nodes), pd.DataFrame(edges)


def _load_group_metrics(training_summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_group in training_summary["combined"]["model_group_order"]:
        group_summary = training_summary["models"][model_group]
        test_metrics = group_summary["metrics"]["test"]
        rows.append(
            {
                "model_group": model_group,
                "feature_count": group_summary["feature_count"],
                "input_dim_after_encoding": group_summary["input_dim_after_encoding"],
                "best_epoch": group_summary["best_epoch"],
                "best_validation_loss": group_summary["best_validation_loss"],
                "test_mae_us": test_metrics["mae_us"],
                "test_rmse_us": test_metrics["rmse_us"],
                "test_r2": test_metrics["r2"],
                "test_mape": test_metrics["mape"],
                "test_median_ape": test_metrics["median_ape"],
            }
        )
    return pd.DataFrame(rows)


def _load_ablation_frames(ablation_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(Path(ablation_root).rglob("ablation_summary.json")):
        summary = read_json(summary_path)
        model_group = summary["model_group"]
        for row in summary.get("summary_rows", []):
            rows.append(
                {
                    "model_group": model_group,
                    "variant": row["variant"],
                    "dropped_features": row["dropped_features"],
                    "dropped_feature_count": row["dropped_feature_count"],
                    "numeric_feature_count": row["numeric_feature_count"],
                    "test_mae_us": row["test_mae_us"],
                    "test_mape": row["test_mape"],
                    "test_median_ape": row["test_median_ape"],
                    "test_r2": row["test_r2"],
                    "test_mae_us_delta_vs_baseline": row["test_mae_us_delta_vs_baseline"],
                    "test_mape_delta_vs_baseline": row["test_mape_delta_vs_baseline"],
                    "test_median_ape_delta_vs_baseline": row["test_median_ape_delta_vs_baseline"],
                    "test_r2_delta_vs_baseline": row["test_r2_delta_vs_baseline"],
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["dropped_features"] = frame["dropped_features"].replace("", "(baseline)")
    return frame


def _load_generalization_summary(path: Path) -> pd.DataFrame:
    summary = read_json(path)
    rows: list[dict[str, Any]] = []
    if "leave_one_case_out" in summary or "leave_one_combo_out" in summary:
        for scheme_name in ("leave_one_case_out", "leave_one_combo_out"):
            scheme = summary.get(scheme_name, {})
            for split_name in ("train", "test"):
                split = scheme.get(split_name, {})
                for family_row in split.get("family_summary", []):
                    rows.append(
                        {
                            "protocol": scheme_name,
                            "split": split_name,
                            "family": family_row["family"],
                            "mean_mape": family_row["mean_mape"],
                            "median_mape": family_row.get("median_mape", family_row["mean_mape"]),
                            "max_mape": family_row.get("max_mape", family_row["mean_mape"]),
                            "mean_dwre": family_row["mean_dwre"],
                            "median_dwre": family_row.get("median_dwre", family_row["mean_dwre"]),
                            "max_dwre": family_row.get("max_dwre", family_row["mean_dwre"]),
                            "total_rows": family_row["total_rows"],
                            "total_actual_us": family_row["total_actual_us"],
                            "total_abs_error_us": family_row["total_abs_error_us"],
                        }
                    )
        return pd.DataFrame(rows)

    protocol = str(summary.get("evaluation_protocol", "train_test_7_3_grouped"))
    for split_name in ("train", "test"):
        split = summary.get(split_name, {})
        for family_row in split.get("family_summary", []):
            rows.append(
                {
                    "protocol": protocol,
                    "split": split_name,
                    "family": family_row["family"],
                    "mean_mape": family_row["mean_mape"],
                    "median_mape": family_row.get("median_mape", family_row["mean_mape"]),
                    "max_mape": family_row.get("max_mape", family_row["mean_mape"]),
                    "mean_dwre": family_row["mean_dwre"],
                    "median_dwre": family_row.get("median_dwre", family_row["mean_dwre"]),
                    "max_dwre": family_row.get("max_dwre", family_row["mean_dwre"]),
                    "total_rows": family_row["total_rows"],
                    "total_actual_us": family_row["total_actual_us"],
                    "total_abs_error_us": family_row["total_abs_error_us"],
                }
            )
    return pd.DataFrame(rows)


def _build_ablation_feature_focus(ablation_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_features = (
        "feat_output_elements_per_batch",
        "feat_output_elements_per_lookup",
        "feat_output_input_bytes_ratio",
        "feat_activation_elements_per_batch",
    )
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    for feature in focus_features:
        mask = ablation_frame["dropped_features"].fillna("").astype(str).str.contains(feature, regex=False)
        feature_rows = ablation_frame[mask].copy()
        if feature_rows.empty:
            continue

        positive_rows = feature_rows[feature_rows["test_mape_delta_vs_baseline"] > 0].copy()
        evidence_rows = positive_rows if not positive_rows.empty else feature_rows
        best_row = evidence_rows.sort_values("test_mape_delta_vs_baseline", ascending=False).iloc[0]
        supporting_groups = sorted(positive_rows["model_group"].unique().tolist() if not positive_rows.empty else feature_rows["model_group"].unique().tolist())

        summary_rows.append(
            {
                "feature": feature,
                "rows": int(len(feature_rows)),
                "positive_rows": int(len(positive_rows)),
                "supporting_groups": _sanitize_list(supporting_groups),
                "supporting_group_count": int(len(supporting_groups)),
                "best_group": best_row["model_group"],
                "best_variant": best_row["variant"],
                "best_delta_vs_baseline": float(best_row["test_mape_delta_vs_baseline"]),
                "mean_positive_delta": float(positive_rows["test_mape_delta_vs_baseline"].mean()) if not positive_rows.empty else float(best_row["test_mape_delta_vs_baseline"]),
                "median_positive_delta": float(positive_rows["test_mape_delta_vs_baseline"].median()) if not positive_rows.empty else float(best_row["test_mape_delta_vs_baseline"]),
                "worst_delta_vs_baseline": float(feature_rows["test_mape_delta_vs_baseline"].min()),
            }
        )

        detail_frames.append(
            feature_rows.assign(feature=feature)[
                [
                    "feature",
                    "model_group",
                    "variant",
                    "dropped_features",
                    "test_mape",
                    "test_mape_delta_vs_baseline",
                ]
            ]
        )

    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(
            ["best_delta_vs_baseline", "mean_positive_delta"],
            ascending=False,
        ).reset_index(drop=True)
    detail_frame = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(
        columns=[
            "feature",
            "model_group",
            "variant",
            "dropped_features",
            "test_mape",
            "test_mape_delta_vs_baseline",
        ]
    )
    if not detail_frame.empty:
        detail_frame = detail_frame.sort_values(
            ["feature", "test_mape_delta_vs_baseline", "model_group"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    return summary_frame, detail_frame


def _compute_longest_path(graph: dict[int, Any], duration_by_node: dict[int, float]) -> tuple[list[int], float]:
    indegree = {node_idx: len([pred for pred in node.predecessors if pred in graph]) for node_idx, node in graph.items()}
    successors: dict[int, list[int]] = defaultdict(list)
    for node_idx, node in graph.items():
        for pred in node.predecessors:
            if pred in graph:
                successors[pred].append(node_idx)
    ready = sorted([node_idx for node_idx, degree in indegree.items() if degree == 0])
    best_total: dict[int, float] = {}
    parent: dict[int, int | None] = {}
    while ready:
        node_idx = ready.pop(0)
        node_duration = float(duration_by_node.get(node_idx, 0.0))
        best_pred = None
        best_pred_total = 0.0
        for pred in graph[node_idx].predecessors:
            if pred in best_total and best_total[pred] > best_pred_total:
                best_pred_total = best_total[pred]
                best_pred = pred
        best_total[node_idx] = best_pred_total + node_duration
        parent[node_idx] = best_pred
        for succ in sorted(successors.get(node_idx, [])):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
                ready.sort()
    if not best_total:
        return [], 0.0
    sink = max(best_total, key=best_total.get)
    path: list[int] = []
    current = sink
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path, float(best_total[sink])


def _slice_predictions(prediction_df: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    frame = prediction_df.copy()
    for column, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            frame = frame[frame[column].isin(list(value))]
        else:
            frame = frame[frame[column] == value]
    return frame


def _evaluate_predictions(prediction_df: pd.DataFrame) -> dict[str, Any]:
    actual = prediction_df["target_us"].astype(float)
    pred = prediction_df["pred_us"].astype(float)
    abs_error = (pred - actual).abs()
    ape = abs_error / actual.replace(0.0, pd.NA)
    return {
        "rows": int(len(prediction_df)),
        "mae_us": float(abs_error.mean()) if not abs_error.empty else None,
        "rmse_us": float(((pred - actual) ** 2).mean() ** 0.5) if not pred.empty else None,
        "mape": float(ape.fillna(0.0).mean()) if not ape.empty else None,
        "median_ape": float(ape.fillna(0.0).median()) if not ape.empty else None,
    }


def _compute_regression_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float | None]:
    actual = actual.astype(float)
    pred = pred.astype(float)
    if actual.empty:
        return {
            "count": 0,
            "mae_us": None,
            "mape": None,
            "rmse_us": None,
            "r2": None,
            "p50_ape": None,
            "p90_ape": None,
            "gt10_rate": None,
        }
    abs_error = (pred - actual).abs()
    ape = (abs_error / actual.replace(0.0, pd.NA)).fillna(0.0)
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    ss_res = float(((pred - actual) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "count": int(len(actual)),
        "mae_us": float(abs_error.mean()),
        "mape": float(ape.mean()),
        "rmse_us": float((((pred - actual) ** 2).mean()) ** 0.5),
        "r2": float(r2),
        "p50_ape": float(ape.quantile(0.50)),
        "p90_ape": float(ape.quantile(0.90)),
        "gt10_rate": float((ape > 0.10).mean()),
    }


def _format_sci_compact(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f}×10^{exponent}"


def _error_trend_summary(frame: pd.DataFrame, work_col: str, *, quantiles: int = 4) -> dict[str, Any]:
    trend = frame[[work_col, "ape"]].copy()
    trend[work_col] = pd.to_numeric(trend[work_col], errors="coerce")
    trend["ape"] = pd.to_numeric(trend["ape"], errors="coerce")
    trend = trend.dropna(subset=[work_col, "ape"])
    if trend.empty:
        return {"bin_count": 0, "bins": [], "spearman": None}
    trend["bin"] = pd.qcut(trend[work_col], q=quantiles, duplicates="drop")
    grouped = (
        trend.groupby("bin", observed=False)
        .agg(
            rows=("ape", "size"),
            work_min=(work_col, "min"),
            work_max=(work_col, "max"),
            mape=("ape", "mean"),
            p50=("ape", "median"),
            p90=("ape", lambda s: s.quantile(0.9)),
        )
        .reset_index(drop=True)
    )
    bins: list[dict[str, Any]] = []
    for row in grouped.to_dict(orient="records"):
        bins.append(
            {
                "rows": int(row["rows"]),
                "work_min": float(row["work_min"]),
                "work_max": float(row["work_max"]),
                "work_min_fmt": _format_sci_compact(float(row["work_min"])),
                "work_max_fmt": _format_sci_compact(float(row["work_max"])),
                "mape": float(row["mape"]),
                "p50": float(row["p50"]),
                "p90": float(row["p90"]),
            }
        )
    return {
        "bin_count": len(bins),
        "bins": bins,
        "spearman": float(trend[work_col].corr(trend["ape"], method="spearman")),
    }


def _select_best_active_analytical_features(
    single_op_root: Path,
    *,
    split: str = "test",
    target_col: str = "label_operator_actual_dur_us",
) -> dict[str, Any]:
    datasets_dir = single_op_root / "datasets"
    if not datasets_dir.exists():
        raise FileNotFoundError(datasets_dir)

    selected_feature_by_group: dict[str, str] = {}
    selected_rows: list[dict[str, Any]] = []
    excluded_groups: list[str] = []

    for group_dir in sorted(path for path in datasets_dir.iterdir() if path.is_dir()):
        manifest = read_json(group_dir / "feature_columns.json")
        active_analytical = [
            column
            for column in manifest.get("numeric_features", [])
            if isinstance(column, str) and column.startswith("ana_calib_")
        ]
        split_csv = group_dir / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(split_csv)
        frame = pd.read_csv(split_csv, low_memory=False)

        if not active_analytical:
            excluded_groups.append(group_dir.name)
            selected_rows.append(
                {
                    "model_group": group_dir.name,
                    "selected_feature": None,
                    "rows": int(len(frame)),
                    "mape": None,
                    "r2": None,
                }
            )
            continue

        best_feature = ""
        best_metrics: dict[str, float | None] | None = None
        best_mape = float("inf")
        for feature in active_analytical:
            metrics = _compute_regression_metrics(frame[target_col], frame[feature])
            feature_mape = float(metrics["mape"]) if metrics["mape"] is not None else float("inf")
            if feature_mape < best_mape:
                best_feature = feature
                best_metrics = metrics
                best_mape = feature_mape

        if not best_feature or best_metrics is None:
            excluded_groups.append(group_dir.name)
            continue

        selected_feature_by_group[group_dir.name] = best_feature
        selected_rows.append(
            {
                "model_group": group_dir.name,
                "selected_feature": best_feature,
                "rows": int(best_metrics["count"]) if best_metrics["count"] is not None else int(len(frame)),
                "mape": best_metrics["mape"],
                "r2": best_metrics["r2"],
            }
        )

    return {
        "selected_feature_by_group": selected_feature_by_group,
        "selected_rows": selected_rows,
        "excluded_groups": excluded_groups,
    }


def _build_best_active_analytical_frame(
    dataset: pd.DataFrame,
    selected_feature_by_group: dict[str, str],
) -> pd.DataFrame:
    frame = dataset.copy()
    frame["analytical_pred_us"] = np.nan
    frame["analytical_feature"] = pd.NA
    for model_group, feature in selected_feature_by_group.items():
        if feature not in frame.columns:
            continue
        mask = frame["model_group"] == model_group
        frame.loc[mask, "analytical_pred_us"] = pd.to_numeric(frame.loc[mask, feature], errors="coerce")
        frame.loc[mask, "analytical_feature"] = feature
    return frame[frame["analytical_pred_us"].notna()].copy()


def _format_range(series: pd.Series, *, cast_int: bool = False) -> str:
    clean = series.dropna()
    if clean.empty:
        return ""
    if cast_int:
        clean = clean.astype(int)
    return f"{clean.min()}-{clean.max()}"


def _run_shell_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _platform_config_frame() -> pd.DataFrame:
    lscpu_text = _run_shell_text(["lscpu"])
    os_release = Path("/etc/os-release").read_text(encoding="utf-8")

    def extract(pattern: str, text: str, default: str = "") -> str:
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else default

    hw_profile_path = Path(__file__).resolve().parent / "hardware_profile" / "kunpeng920_single_numa.yaml"
    hw_profile_text = hw_profile_path.read_text(encoding="utf-8")
    host_cores = extract(r"CPU\(s\):\s+(\d+)", lscpu_text)
    host_threads = host_cores
    host_freq = extract(r"cpu_clock:\s+([^\n]+)", hw_profile_text, extract(r"Model name:\s+([^\n]+)", lscpu_text))
    return pd.DataFrame(
        [
            {
                "server_model": "Huawei Cloud Kunpeng host (4-socket)",
                "cpu_model": extract(r"Model name:\s+([^\n]+)", lscpu_text),
                "host_cores_threads": f"{host_cores}C/{host_threads}T",
                "experiment_scope": "single NUMA domain (24 cores)",
                "main_frequency": host_freq,
                "cache_config": "L1I 64 KiB/core, L1D 64 KiB/core, L2 512 KiB/core, L3 24 MiB/NUMA",
                "memory_config": "DDR4-2933, 4 channels/NUMA, approx. 100 GB/s local bandwidth",
                "os_version": extract(r'PRETTY_NAME=\"([^\"]+)\"', os_release),
                "python_ort_stack": "Python 3.11 (ort env), ONNX Runtime / PyTorch / ONNX versions recorded in experiment artifacts",
                "compiler_profiler": "GCC 10.3.1, DynamoRIO profiler, ORT branch-parallel timeline",
                "binding_notes": "single-NUMA pinning; fixed inter/intra thread configs; branch-parallel timeline replay",
            }
        ]
    )


def run_platform_summary(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    e2e_artifact_root: Path | None = None,
    baseline_model_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)
    single_op_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    e2e_summary = read_json(e2e_root / "summary.json")

    frame = _platform_config_frame()
    csv_path, md_path = write_frame_csv_md(
        frame,
        tables_dir / TABLE_FILENAMES["4-1"],
        tables_dir / "table_4_1_platform_config.md",
        "Table 4-1 Hardware and software configuration",
    )

    fig_path = plot_flow(
        [
            "DLRM / ONNX model",
            "single-op split",
            "parameter sampling",
            "ORT execution",
            "profiling + cleaning",
            "feature build",
        ],
        figures_dir / FIGURE_FILENAMES["4-1"],
        "Figure 4-1 Single-op data collection flow",
        subtitle="Model -> operator split -> sampled execution -> profiling labels -> features",
    )
    fig_path_2 = plot_flow(
        [
            "DLRM full graph run",
            "timeline collection",
            "graph restore",
            "node / batch alignment",
            "E2E ground truth",
        ],
        figures_dir / FIGURE_FILENAMES["4-2"],
        "Figure 4-2 E2E timeline aggregation flow",
        subtitle="Timeline replay and graph reconstruction for combo-level labels",
    )

    summary = {
        "single_op_rows": int(len(single_op_df)),
        "single_op_cases": int(single_op_df["case_id"].nunique()),
        "single_op_combos": int(single_op_df[["case_id", "combo"]].drop_duplicates().shape[0]),
        "full_e2e_combos": int(e2e_summary["combo_counts"]["full_combo_count"]),
        "host_cpu_model": frame.iloc[0]["cpu_model"],
        "experiment_scope": frame.iloc[0]["experiment_scope"],
    }
    manifest = {
        "table_csv": str(csv_path),
        "table_md": str(md_path),
        "figure_4_1": str(fig_path),
        "figure_4_2": str(fig_path_2),
        "summary": summary,
    }
    write_json(layout["manifests"] / "platform_summary.json", manifest)
    return SectionResult(name="platform", outputs=manifest)


def run_single_op_fair_baseline(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    force_retrain: bool = False,
    hidden_layers: Sequence[int] | None = None,
    max_iter: int | None = None,
) -> SectionResult:
    payload = ensure_fair_single_mlp_artifact(
        output_root,
        single_op_artifact_root=single_op_artifact_root,
        force_retrain=force_retrain,
        hidden_layers=hidden_layers,
        max_iter=max_iter,
    )
    outputs = {
        "artifact_dir": payload["artifact_dir"],
        "dataset_dir": payload["dataset_dir"],
        "model_dir": payload["model_dir"],
        "manifest_path": payload["manifest_path"],
        "metrics_path": payload["metrics_path"],
        "predictions_test_path": payload["predictions_test_path"],
        "feature_count": payload["feature_count"],
    }
    return SectionResult(name="single_op_fair_baseline", outputs=outputs)


def run_single_op_core(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    baseline_model_root: Path | None = None,
    e2e_artifact_root: Path | None = None,
    prediction_source: str = PREDICTION_SOURCE_GROUPED,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["single_op"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)
    normalized_prediction_source = _normalize_prediction_source(prediction_source)

    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    if baseline_model_root is not None:
        baseline_root = _resolve_root(baseline_model_root, BASELINE_MODEL_ROOT)
        baseline_manifest = read_json(baseline_root / "metrics.json")
        baseline_predictions = pd.read_csv(baseline_root / "predictions_test.csv", low_memory=False)
        baseline_feature_count = int(
            len(baseline_manifest.get("numeric_features", []))
            + len(baseline_manifest.get("categorical_features", []))
        )
    else:
        fair_baseline = ensure_fair_single_mlp_artifact(output_root, single_op_artifact_root=single_op_root)
        baseline_root = Path(fair_baseline["model_dir"])
        baseline_predictions = fair_baseline["predictions_test"]
        baseline_feature_count = int(fair_baseline["feature_count"])
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    prediction_bundle = _load_model_prediction_frame(
        output_root,
        single_op_root=single_op_root,
        prediction_source=normalized_prediction_source,
        test_dataset=test_dataset,
    )
    selected_predictions = prediction_bundle["prediction_df"]
    active_model_name = prediction_bundle["model_name"]
    analytical_selection = _select_best_active_analytical_features(single_op_root, split="test")
    analytical_frame = _build_best_active_analytical_frame(
        test_dataset,
        analytical_selection["selected_feature_by_group"],
    )
    analytical_metrics = _compute_regression_metrics(
        analytical_frame["label_operator_actual_dur_us"],
        analytical_frame["analytical_pred_us"],
    )
    active_metrics = _compute_regression_metrics(
        selected_predictions["target_us"],
        selected_predictions["pred_us"],
    )
    baseline_metrics = _compute_regression_metrics(
        baseline_predictions["target_us"],
        baseline_predictions["pred_us"],
    )

    group_label_map = {
        "gather": "Index / memory",
        "layout_move": "Layout move",
        "view_meta": "View / meta",
        "mixed_balanced": "Light compute-mem",
        "compute_dominant": "Compute dominant",
    }
    table_4_2 = (
        dataset_df.groupby("model_group", as_index=False)
        .agg(
            sample_count=("row_uid", "count"),
            operator_names=("op_type", lambda values: ",".join(sorted(set(values)))),
            batch_size_range=("batch_size", lambda values: _format_range(pd.Series(values), cast_int=True)),
            nip_range=("num_indices_per_lookup", lambda values: _format_range(pd.Series(values), cast_int=True)),
            num_threads_range=("num_threads", lambda values: _format_range(pd.Series(values), cast_int=True)),
            inter_threads_range=("inter_threads", lambda values: _format_range(pd.Series(values), cast_int=True)),
            io_bytes_range=("feat_io_bytes_sum", lambda values: _format_range(pd.Series(values))),
        )
        .sort_values("model_group")
        .reset_index(drop=True)
    )
    table_4_2["operator_category"] = table_4_2["model_group"].map(group_label_map).fillna(table_4_2["model_group"])
    table_4_2 = table_4_2[
        [
            "operator_category",
            "operator_names",
            "sample_count",
            "batch_size_range",
            "nip_range",
            "num_threads_range",
            "inter_threads_range",
            "io_bytes_range",
        ]
    ]

    if normalized_prediction_source == PREDICTION_SOURCE_SINGLE_FAIR:
        table_4_3 = pd.DataFrame(
            [
                {
                    "model": "Analytical only (best active proxy / covered groups)",
                    **analytical_metrics,
                },
                {
                    "model": f"Single MLP (same split / {baseline_feature_count} features)",
                    **active_metrics,
                },
            ]
        )
    else:
        table_4_3 = pd.DataFrame(
            [
                {
                    "model": "Analytical only (best active proxy / covered groups)",
                    **analytical_metrics,
                },
                {
                    "model": f"Single MLP (same split / {baseline_feature_count} features)",
                    **baseline_metrics,
                },
                {
                    "model": f"Grouped analytical-MLP (same split / routed subset from {baseline_feature_count} features)",
                    **active_metrics,
                },
            ]
        )

    category_rows: list[dict[str, Any]] = []
    for model_group, sub in selected_predictions.groupby("model_group"):
        metrics = _compute_regression_metrics(sub["target_us"], sub["pred_us"])
        category_rows.append(
            {
                "operator_category": group_label_map.get(model_group, model_group),
                "sample_count": int(len(sub)),
                "mean_actual_us": float(sub["target_us"].astype(float).mean()),
                "mae_us": metrics["mae_us"],
                "mape": metrics["mape"],
                "rmse_us": metrics["rmse_us"],
                "p90_ape": metrics["p90_ape"],
            }
        )
    table_4_4 = pd.DataFrame(category_rows).sort_values("mape", ascending=False).reset_index(drop=True)

    csv_42, md_42 = write_frame_csv_md(
        table_4_2,
        tables_dir / TABLE_FILENAMES["4-2"],
        tables_dir / "table_4_2_dataset_composition.md",
        "Table 4-2 Dataset composition by operator category",
    )
    csv_43, md_43 = write_frame_csv_md(
        table_4_3,
        tables_dir / TABLE_FILENAMES["4-3"],
        tables_dir / "table_4_3_single_op_overall.md",
        "Table 4-3 Single-op overall accuracy",
    )
    csv_44, md_44 = write_frame_csv_md(
        table_4_4,
        tables_dir / TABLE_FILENAMES["4-4"],
        tables_dir / "table_4_4_single_op_category.md",
        "Table 4-4 Category-wise single-op accuracy",
    )

    scatter_frame = selected_predictions.copy()
    scatter_frame["actual_us"] = scatter_frame["target_us"].astype(float)
    scatter_frame["predicted_us"] = scatter_frame["pred_us"].astype(float)
    plot_scatter(
        scatter_frame.sample(n=min(2500, len(scatter_frame)), random_state=42),
        "actual_us",
        "predicted_us",
        figures_dir / FIGURE_FILENAMES["4-3"],
        "Figure 4-3 Single-op predicted vs. actual latency",
        audit_lines=[
            f"rows = {len(scatter_frame):,}",
            f"model = {active_model_name}",
            f"MAPE = {active_metrics['mape']:.4f}",
            f"R2 = {active_metrics['r2']:.4f}",
        ],
        audit_loc="inside upper right",
    )

    plot_bar(
        table_4_4.sort_values("mape", ascending=False),
        "operator_category",
        "mape",
        figures_dir / FIGURE_FILENAMES["4-4"],
        "Figure 4-4 MAPE by operator category",
        ylabel="MAPE",
        color="#54A24B",
        audit_lines=None,
        label_rotation=22.0,
        label_wrap_width=999,
    )

    representative_panel_specs: list[dict[str, Any]] = []

    gather_frame = test_dataset[test_dataset["op_type"] == "Gather"].copy()
    if not gather_frame.empty:
        gather_pred = selected_predictions[selected_predictions["op_type"] == "Gather"][["row_uid", "pred_us"]]
        gather_frame = gather_frame.merge(gather_pred, on="row_uid", how="left")
        gather_frame["ape"] = (
            (gather_frame["pred_us"].astype(float) - gather_frame["label_operator_actual_dur_us"].astype(float)).abs()
            / gather_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
        ).fillna(0.0)
        gather_sample = gather_frame.sample(n=min(1500, len(gather_frame)), random_state=42)
        plot_scatter(
            gather_sample,
            "feat_lookup_count",
            "ape",
            figures_dir / FIGURE_FILENAMES["4-5"],
            "Figure 4-5 Gather error vs. lookup count",
            hue="num_threads",
            reference_line=False,
            audit_lines=[
                f"rows = {len(gather_frame):,}",
                f"p90 APE = {float(gather_frame['ape'].quantile(0.9)):.4f}",
            ],
        )
        representative_panel_specs.append(
            {
                "frame": gather_sample,
                "x": "feat_lookup_count",
                "y": "ape",
                "hue": "num_threads",
                "xlabel": "feat_lookup_count",
                "ylabel": "APE",
                "title": "Gather",
            }
        )

    reduce_frame = test_dataset[test_dataset["op_type"] == "ReduceSum"].copy()
    if not reduce_frame.empty:
        reduce_pred = selected_predictions[selected_predictions["op_type"] == "ReduceSum"][["row_uid", "pred_us"]]
        reduce_frame = reduce_frame.merge(reduce_pred, on="row_uid", how="left")
        reduce_frame["ape"] = (
            (reduce_frame["pred_us"].astype(float) - reduce_frame["label_operator_actual_dur_us"].astype(float)).abs()
            / reduce_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
        ).fillna(0.0)
        reduce_sample = reduce_frame.sample(n=min(1500, len(reduce_frame)), random_state=42)
        plot_scatter(
            reduce_sample,
            "feat_reduction_work_items",
            "ape",
            figures_dir / FIGURE_FILENAMES["4-6"],
            "Figure 4-6 ReduceSum error vs. reduction size",
            hue="num_threads",
            reference_line=False,
            audit_lines=[
                f"rows = {len(reduce_frame):,}",
                f"p90 APE = {float(reduce_frame['ape'].quantile(0.9)):.4f}",
            ],
        )
        representative_panel_specs.append(
            {
                "frame": reduce_sample,
                "x": "feat_reduction_work_items",
                "y": "ape",
                "hue": "num_threads",
                "xlabel": "feat_reduction_work_items",
                "ylabel": "APE",
                "title": "ReduceSum",
            }
        )

    layout_frame = test_dataset[test_dataset["op_type"].isin(["Transpose", "Concat"])].copy()
    if not layout_frame.empty:
        layout_pred = selected_predictions[selected_predictions["op_type"].isin(["Transpose", "Concat"])][["row_uid", "pred_us"]]
        layout_frame = layout_frame.merge(layout_pred, on="row_uid", how="left")
        layout_frame["ape"] = (
            (layout_frame["pred_us"].astype(float) - layout_frame["label_operator_actual_dur_us"].astype(float)).abs()
            / layout_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
        ).fillna(0.0)
        layout_sample = layout_frame.sample(n=min(1500, len(layout_frame)), random_state=42)
        plot_scatter(
            layout_sample,
            "feat_io_bytes_sum",
            "ape",
            figures_dir / FIGURE_FILENAMES["4-7"],
            "Figure 4-7 Transpose / Concat error vs. data size",
            hue="op_type",
            reference_line=False,
            audit_lines=[
                f"rows = {len(layout_frame):,}",
                f"shown ops = Transpose / Concat",
            ],
        )
        representative_panel_specs.append(
            {
                "frame": layout_sample,
                "x": "feat_io_bytes_sum",
                "y": "ape",
                "hue": "op_type",
                "xlabel": "feat_io_bytes_sum",
                "ylabel": "APE",
                "title": "Transpose / Concat",
            }
        )

    gemm_frame = test_dataset[test_dataset["op_type"].isin(["Gemm", "MatMul"])].copy()
    if not gemm_frame.empty:
        gemm_pred = selected_predictions[selected_predictions["op_type"].isin(["Gemm", "MatMul"])][["row_uid", "pred_us"]]
        gemm_frame = gemm_frame.merge(gemm_pred, on="row_uid", how="left")
        gemm_frame["ape"] = (
            (gemm_frame["pred_us"].astype(float) - gemm_frame["label_operator_actual_dur_us"].astype(float)).abs()
            / gemm_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
        ).fillna(0.0)
        gemm_sample = gemm_frame.sample(n=min(1500, len(gemm_frame)), random_state=42)
        plot_scatter(
            gemm_sample,
            "feat_gemm_mac_count",
            "ape",
            figures_dir / FIGURE_FILENAMES["4-8"],
            "Figure 4-8 Gemm / MatMul error vs. MAC count",
            hue="op_type",
            reference_line=False,
            audit_lines=[
                f"rows = {len(gemm_frame):,}",
                f"Gemm-like ops = {', '.join(sorted(gemm_frame['op_type'].unique().tolist()))}",
            ],
        )
        representative_panel_specs.append(
            {
                "frame": gemm_sample,
                "x": "feat_gemm_mac_count",
                "y": "ape",
                "hue": "op_type",
                "xlabel": "feat_gemm_mac_count",
                "ylabel": "APE",
                "title": "Gemm / MatMul",
            }
        )

    if len(representative_panel_specs) == 4:
        plot_representative_panel(
            representative_panel_specs,
            figures_dir / REPRESENTATIVE_PANEL_FILENAME,
        )

    section_payload = {
        "tables": {
            "4-2": str(csv_42),
            "4-3": str(csv_43),
            "4-4": str(csv_44),
        },
        "figures": {
            "4-3": str(figures_dir / FIGURE_FILENAMES["4-3"]),
            "4-4": str(figures_dir / FIGURE_FILENAMES["4-4"]),
            "4-5": str(figures_dir / FIGURE_FILENAMES["4-5"]),
            "4-6": str(figures_dir / FIGURE_FILENAMES["4-6"]),
            "4-7": str(figures_dir / FIGURE_FILENAMES["4-7"]),
            "4-8": str(figures_dir / FIGURE_FILENAMES["4-8"]),
            "4-5-panel": str(figures_dir / REPRESENTATIVE_PANEL_FILENAME),
        },
        "summary": {
            "analytical_test_mape": float(analytical_metrics["mape"]),
            "analytical_test_count": int(analytical_metrics["count"]),
            "analytical_test_total_rows": int(len(test_dataset)),
            "analytical_test_coverage_rate": float(analytical_metrics["count"]) / float(len(test_dataset)) if len(test_dataset) else 0.0,
            "analytical_selected_features": analytical_selection["selected_rows"],
            "analytical_excluded_groups": list(analytical_selection["excluded_groups"]),
            "baseline_test_mape": float(baseline_metrics["mape"]),
            "baseline_test_r2": float(baseline_metrics["r2"]),
            "baseline_feature_count": baseline_feature_count,
            "baseline_model_root": str(baseline_root),
            "grouped_test_mape": float(active_metrics["mape"]) if normalized_prediction_source == PREDICTION_SOURCE_GROUPED else None,
            "grouped_test_r2": float(active_metrics["r2"]) if normalized_prediction_source == PREDICTION_SOURCE_GROUPED else None,
            "active_prediction_source": normalized_prediction_source,
            "active_model_name": active_model_name,
            "active_test_mape": float(active_metrics["mape"]),
            "active_test_r2": float(active_metrics["r2"]),
            "active_model_root": prediction_bundle["model_root"],
            "representative_op_types": list(REPRESENTATIVE_OP_TYPES),
            "e2e_reference_root": str(e2e_root),
        },
    }
    _write_summary_bundle(section_dir, "single_op_core", section_payload)
    return SectionResult(name="single_op_core", outputs=section_payload)


def run_single_op_ood(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    ood_artifact_root: Path | None = None,
    prediction_source: str = PREDICTION_SOURCE_GROUPED,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["single_op"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    ood_root = _resolve_root(ood_artifact_root, OOD_ARTIFACT_ROOT)

    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    prediction_bundle = _load_model_prediction_frame(
        output_root,
        single_op_root=single_op_root,
        prediction_source=prediction_source,
        test_dataset=test_dataset,
    )
    prediction_df = prediction_bundle["prediction_df"]
    joined = prediction_df.copy()
    if "num_threads" not in joined.columns:
        joined = joined.merge(
            dataset_df[["row_uid", "num_threads"]],
            on="row_uid",
            how="left",
            validate="one_to_one",
        )
    slices = []
    batch_slice = _slice_predictions(joined, batch_size=list(OOD_BATCH_HOLDS))
    thread_slice = _slice_predictions(joined, num_threads=OOD_NUM_THREADS_HOLD)
    for name, frame in [("unseen_batch_size", batch_slice), ("unseen_num_threads", thread_slice)]:
        metrics = _evaluate_predictions(frame)
        metrics.update(
            {
                "slice_name": name,
                "rows": int(len(frame)),
                "batch_sizes": _sanitize_list(sorted(frame["batch_size"].dropna().astype(int).unique().tolist())) if not frame.empty else "",
                "num_threads": _sanitize_list(sorted(frame["num_threads"].dropna().astype(int).unique().tolist())) if not frame.empty else "",
                "inter_threads": _sanitize_list(sorted(frame["inter_threads"].dropna().astype(int).unique().tolist())) if not frame.empty else "",
            }
        )
        slices.append(metrics)
    ood_frame = pd.DataFrame(slices)

    generalization_summary_path = (
        ood_root / "generalization_summary.json"
        if (ood_root / "generalization_summary.json").exists()
        else ood_root / "summary.json"
    )
    generalization_frame = _load_generalization_summary(generalization_summary_path)
    if not generalization_frame.empty:
        generalization_frame = generalization_frame[generalization_frame["split"] == "test"].copy()

    csv_44 = tables_dir / "ood_slice_summary.csv"
    md_44 = tables_dir / "ood_slice_summary.md"
    write_frame_csv_md(ood_frame, csv_44, md_44, "Single-op OOD Slice Summary")
    generalization_csv = tables_dir / "ood_generalization_reference.csv"
    generalization_md = tables_dir / "ood_generalization_reference.md"
    write_frame_csv_md(
        generalization_frame.sort_values(["protocol", "family"]),
        generalization_csv,
        generalization_md,
        "Analytical Generalization Reference",
    )

    plot_frame = ood_frame.copy()
    if not plot_frame.empty:
        plot_frame = plot_frame.sort_values("mape", ascending=False).reset_index(drop=True)
        plot_frame["label"] = plot_frame["slice_name"].map(
            {
                "unseen_batch_size": "batch holdout",
                "unseen_num_threads": "thread holdout",
            }
        )
        plot_bar(
            plot_frame,
            "label",
            "mape",
            figures_dir / FIGURE_FILENAMES["4-9"],
            "Figure 4-9 Unseen-shape / unseen-thread single-op generalization",
            ylabel="MAPE",
            color="#E45756",
            audit_lines=[
                f"holdout batch sizes = {_sanitize_list(OOD_BATCH_HOLDS)}",
                f"thread holdout = {OOD_NUM_THREADS_HOLD}",
                f"slice rows = {', '.join(f'{row.slice_name}:{int(row.rows)}' for row in ood_frame.itertuples(index=False))}",
            ],
        )

    if not generalization_frame.empty:
        test_reference = generalization_frame.sort_values("mean_mape", ascending=False).reset_index(drop=True)
        plot_bar(
            test_reference,
            "family",
            "mean_mape",
            figures_dir / FIGURE_FILENAMES["4-10"],
            "Figure 4-10 Analytical grouped-holdout family reference",
            ylabel="mean MAPE",
            color="#4C78A8",
            audit_lines=[
                f"protocol = {test_reference['protocol'].iloc[0]}",
                f"families = {len(test_reference)}",
            ],
            label_rotation=20.0,
        )

    payload = {
        "tables": {
            "ood_slice_summary": str(csv_44),
            "ood_generalization_reference": str(generalization_csv),
        },
        "figures": {
            "4-9": str(figures_dir / FIGURE_FILENAMES["4-9"]),
            "4-10": str(figures_dir / FIGURE_FILENAMES["4-10"]),
        },
        "summary": {
            "ood_slices": slices,
            "generalization_reference_rows": int(len(generalization_frame)),
            "prediction_source": _normalize_prediction_source(prediction_source),
            "model_name": prediction_bundle["model_name"],
        },
    }
    _write_summary_bundle(section_dir, "single_op_ood", payload)
    return SectionResult(name="single_op_ood", outputs=payload)


def run_single_op_ablation(
    output_root: Path | None = None,
    *,
    ablation_artifact_root: Path | None = None,
    single_op_artifact_root: Path | None = None,
    e2e_artifact_root: Path | None = None,
    prediction_source: str = PREDICTION_SOURCE_GROUPED,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["ablation"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)
    normalized_prediction_source = _normalize_prediction_source(prediction_source)
    fair_baseline = ensure_fair_single_mlp_artifact(output_root, single_op_artifact_root=single_op_root)
    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    test_meta = test_dataset[["row_uid", "batch_size", "num_indices_per_lookup", "inter_threads"]].copy()
    analytical_predictions = test_dataset[
        [
            "row_uid",
            "case_id",
            "combo",
            "op_idx",
            "batch_size",
            "num_indices_per_lookup",
            "inter_threads",
        ]
    ].copy()
    analytical_predictions["target_us"] = test_dataset["label_operator_actual_dur_us"].astype(float)
    analytical_predictions["pred_us"] = test_dataset["ana_calib_total_us"].astype(float)
    fair_predictions = fair_baseline["predictions_test"].copy()
    fair_predictions_enriched = fair_predictions.merge(test_meta, on="row_uid", how="left")
    active_prediction_bundle = _load_model_prediction_frame(
        output_root,
        single_op_root=single_op_root,
        prediction_source=normalized_prediction_source,
        test_dataset=test_dataset,
    )
    active_predictions = active_prediction_bundle["prediction_df"].copy()
    analytical_selection = _select_best_active_analytical_features(single_op_root, split="test")
    analytical_frame = _build_best_active_analytical_frame(
        test_dataset,
        analytical_selection["selected_feature_by_group"],
    )
    analytical_single_metrics = _compute_regression_metrics(
        analytical_frame["label_operator_actual_dur_us"],
        analytical_frame["analytical_pred_us"],
    )
    e2e_full = pd.read_csv(e2e_root / "full_combo_metrics.csv", low_memory=False)
    analytical_combo_specs = build_combo_specs(analytical_predictions, ort_root=Path(DEFAULT_ORT_ROOT))
    analytical_combo_map = {(spec.case_id, spec.combo): spec for spec in analytical_combo_specs}
    analytical_pipeline_rows: list[dict[str, Any]] = []
    for (case_id, combo), combo_rows in analytical_predictions.groupby(["case_id", "combo"]):
        combo_spec = analytical_combo_map[(case_id, combo)]
        graph = build_op_graph(load_op_shapes_frame(combo_spec.artifact_paths.shape_csv))
        schedule_result = schedule_combo(combo_spec, graph, combo_rows.copy())
        analytical_pipeline_rows.append(
            {
                "case_id": case_id,
                "combo": combo,
                "analytical_pipeline_us": float(schedule_result.predicted_full_graph_us),
            }
        )
    analytical_pipeline_combo = pd.DataFrame(analytical_pipeline_rows)
    fair_pipeline_combo = _build_pipeline_combo_predictions(fair_predictions_enriched).rename(
        columns={"predicted_e2e_us": "single_mlp_pipeline_us"}
    )

    analytical_combo = (
        test_dataset.groupby(["case_id", "combo"], as_index=False)
        .agg(analytical_simple_add_us=("ana_calib_total_us", "sum"))
    )
    fair_single_combo = (
        fair_predictions.groupby(["case_id", "combo"], as_index=False)
        .agg(single_mlp_simple_add_us=("pred_us", "sum"))
    )
    grouped_combo = None
    active_pipeline_combo = None
    if normalized_prediction_source == PREDICTION_SOURCE_GROUPED:
        grouped_combo = (
            active_predictions.groupby(["case_id", "combo"], as_index=False)
            .agg(grouped_mlp_simple_add_us=("pred_us", "sum"))
        )
        active_pipeline_combo = pd.DataFrame(
            {
                "case_id": e2e_full["case_id"],
                "combo": e2e_full["combo"],
                "pipeline_us": e2e_full["predicted_e2e_us"].astype(float),
            }
        )
    e2e_joined = (
        e2e_full.merge(analytical_combo, on=["case_id", "combo"], how="left")
        .merge(analytical_pipeline_combo, on=["case_id", "combo"], how="left")
        .merge(fair_single_combo, on=["case_id", "combo"], how="left")
        .merge(
            fair_pipeline_combo[["case_id", "combo", "single_mlp_pipeline_us"]],
            on=["case_id", "combo"],
            how="left",
        )
    )
    if grouped_combo is not None:
        e2e_joined = e2e_joined.merge(grouped_combo, on=["case_id", "combo"], how="left")
    if active_pipeline_combo is not None:
        e2e_joined = e2e_joined.merge(active_pipeline_combo, on=["case_id", "combo"], how="left")
    e2e_joined["actual_us"] = e2e_joined["actual_e2e_us"].astype(float)

    variants = [
        ("Analytical + simple add", "analytical_simple_add_us"),
        ("Analytical + pipeline", "analytical_pipeline_us"),
        ("Analytical + single MLP + simple add", "single_mlp_simple_add_us"),
        ("Analytical + single MLP + pipeline", "single_mlp_pipeline_us"),
    ]
    if normalized_prediction_source == PREDICTION_SOURCE_GROUPED:
        variants.insert(3, ("Analytical + grouped MLP + simple add", "grouped_mlp_simple_add_us"))
        variants.append(("Analytical + grouped MLP + pipeline", "pipeline_us"))
    table_rows: list[dict[str, Any]] = []
    cdf_map: dict[str, list[float]] = {}
    gt10_rows: list[dict[str, Any]] = []
    inter_rows: list[dict[str, Any]] = []
    for variant_name, column in variants:
        if column in {"analytical_simple_add_us", "analytical_pipeline_us"}:
            single_metrics = analytical_single_metrics
        elif column in {"single_mlp_simple_add_us", "single_mlp_pipeline_us"}:
            single_metrics = _compute_regression_metrics(
                fair_predictions["target_us"],
                fair_predictions["pred_us"],
            )
        elif column in {"grouped_mlp_simple_add_us", "pipeline_us"}:
            single_metrics = _compute_regression_metrics(
                active_predictions["target_us"],
                active_predictions["pred_us"],
            )
        else:
            single_metrics = _compute_regression_metrics(
                fair_predictions["target_us"],
                fair_predictions["pred_us"],
            )
        e2e_metrics = _compute_regression_metrics(e2e_joined["actual_us"], e2e_joined[column])
        ape = ((e2e_joined[column] - e2e_joined["actual_us"]).abs() / e2e_joined["actual_us"].replace(0.0, pd.NA)).fillna(0.0)
        cdf_map[variant_name] = ape.astype(float).tolist()
        gt10_rows.append(
            {
                "variant": variant_name,
                "mean_ape": e2e_metrics["mape"],
                "gt10_rate": e2e_metrics["gt10_rate"],
            }
        )
        for inter_threads, sub in e2e_joined.groupby("inter_threads"):
            inter_rows.append(
                {
                    "variant": variant_name,
                    "inter_threads": int(inter_threads),
                    "mape": float((((sub[column] - sub["actual_us"]).abs() / sub["actual_us"].replace(0.0, pd.NA)).fillna(0.0)).mean()),
                }
            )
        table_rows.append(
            {
                "variant": variant_name,
                "single_op_mape": single_metrics["mape"],
                "single_op_rmse_us": single_metrics["rmse_us"],
                "single_op_r2": single_metrics["r2"],
                "e2e_mape": e2e_metrics["mape"],
                "e2e_p50_ape": e2e_metrics["p50_ape"],
                "e2e_p90_ape": e2e_metrics["p90_ape"],
                "e2e_gt10_rate": e2e_metrics["gt10_rate"],
            }
        )
    table_4_6 = pd.DataFrame(table_rows)
    csv_46, md_46 = write_frame_csv_md(
        table_4_6,
        tables_dir / TABLE_FILENAMES["4-6"],
        tables_dir / "table_4_6_ablation_summary.md",
        "Table 4-6 Fair-comparison ablation summary",
    )

    cdf_plot_map = {str(label).replace("single MLP", "MLP"): values for label, values in cdf_map.items()}
    plot_cdf(
        cdf_plot_map,
        figures_dir / FIGURE_FILENAMES["4-16"],
        "Figure 4-16 CDF of E2E relative error under fair-comparison variants",
        xlabel="relative error",
        audit_lines=None,
        legend_side="inside",
        legend_loc="lower right",
    )
    figure_4_16_zoom_path = figures_dir / "fig_4_16_ablation_delta_bars_zoom_0_0p5.png"
    plot_cdf(
        cdf_plot_map,
        figure_4_16_zoom_path,
        "Figure 4-16 Supplement: zoomed CDF of E2E relative error",
        xlabel="relative error",
        audit_lines=None,
        legend_side="inside",
        legend_loc="lower right",
        xlim=(0.0, 0.5),
        marker_x=0.15,
    )
    gt10_frame = pd.DataFrame(gt10_rows)
    gt10_audit_lines = [
        f"single pipeline mean APE = {float(gt10_frame[gt10_frame['variant'] == 'Analytical + single MLP + pipeline'].iloc[0]['mean_ape']):.4f}",
        f"analytical-only mean APE = {float(gt10_frame.iloc[0]['mean_ape']):.4f}",
    ]
    if normalized_prediction_source == PREDICTION_SOURCE_GROUPED:
        gt10_audit_lines.append(
            f"grouped pipeline mean APE = {float(gt10_frame[gt10_frame['variant'] == 'Analytical + grouped MLP + pipeline'].iloc[0]['mean_ape']):.4f}"
        )
    plot_grouped_bar(
        gt10_frame,
        "variant",
        ["mean_ape", "gt10_rate"],
        figures_dir / FIGURE_FILENAMES["4-17"],
        "Figure 4-17 Mean error and >10% error rate",
        ylabel="ratio",
        legend_labels=["mean APE", ">10% rate"],
        audit_lines=gt10_audit_lines,
        legend_side="top",
        audit_loc="inside upper right",
    )
    inter_frame = pd.DataFrame(inter_rows).sort_values(["inter_threads", "variant"]).reset_index(drop=True)
    inter_plot_frame = inter_frame.copy()
    inter_plot_frame["variant"] = inter_plot_frame["variant"].str.replace("single MLP", "MLP", regex=False)
    plot_line(
        inter_plot_frame,
        "inter_threads",
        "mape",
        figures_dir / FIGURE_FILENAMES["4-18"],
        "Figure 4-18 E2E MAPE vs. branch parallelism",
        group_col="variant",
        ylabel="MAPE",
        audit_lines=[
            f"inter_threads = {_sanitize_list(sorted(inter_frame['inter_threads'].unique().tolist()))}",
        ],
        legend_side="inside",
        legend_loc="upper left",
        audit_loc="upper right",
    )

    payload = {
        "tables": {"4-6": str(csv_46)},
        "figures": {
            "4-16": str(figures_dir / FIGURE_FILENAMES["4-16"]),
            "4-16-zoom-0-0.5": str(figure_4_16_zoom_path),
            "4-17": str(figures_dir / FIGURE_FILENAMES["4-17"]),
            "4-18": str(figures_dir / FIGURE_FILENAMES["4-18"]),
        },
        "summary": {
            "variant_rows": table_rows,
            "rows": int(len(e2e_joined)),
            "prediction_source": normalized_prediction_source,
            "model_name": active_prediction_bundle["model_name"],
        },
    }
    _write_summary_bundle(section_dir, "single_op_ablation", payload)
    return SectionResult(name="single_op_ablation", outputs=payload)


def run_e2e_core(
    output_root: Path | None = None,
    *,
    e2e_artifact_root: Path | None = None,
    single_op_artifact_root: Path | None = None,
    prediction_source: str = PREDICTION_SOURCE_GROUPED,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["e2e"]

    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)
    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    normalized_prediction_source = _normalize_prediction_source(prediction_source)
    full_metrics = _load_e2e_full_metrics(
        output_root,
        single_op_root=single_op_root,
        e2e_root=e2e_root,
        prediction_source=normalized_prediction_source,
    )
    summary_metrics = _compute_regression_metrics(full_metrics["actual_e2e_us"], full_metrics["predicted_e2e_us"])
    full_metrics["batch_bucket"] = pd.cut(
        full_metrics["batch_size"],
        bins=[0, 1280, 1792, 4096],
        labels=["small", "medium", "large"],
    )
    table_5 = (
        full_metrics.groupby("batch_bucket", as_index=False, observed=False)
        .agg(
            sample_count=("combo", "count"),
            mean_actual_us=("actual_e2e_us", "mean"),
            mae_us=("abs_error_us", "mean"),
            mape=("ape", "mean"),
            p50_ape=("ape", lambda values: float(pd.Series(values).quantile(0.50))),
            p90_ape=("ape", lambda values: float(pd.Series(values).quantile(0.90))),
        )
    )
    table_5["r2"] = pd.NA
    overall_row = pd.DataFrame(
        [
            {
                "batch_bucket": "overall",
                "sample_count": int(len(full_metrics)),
                "mean_actual_us": float(full_metrics["actual_e2e_us"].mean()),
                "mae_us": summary_metrics["mae_us"],
                "mape": summary_metrics["mape"],
                "r2": summary_metrics["r2"],
                "p50_ape": summary_metrics["p50_ape"],
                "p90_ape": summary_metrics["p90_ape"],
            }
        ]
    )
    table_5 = table_5[
        ["batch_bucket", "sample_count", "mean_actual_us", "mae_us", "mape", "r2", "p50_ape", "p90_ape"]
    ]
    table_5 = pd.DataFrame(
        [overall_row.iloc[0].to_dict(), *table_5.to_dict(orient="records")]
    )
    csv_5, md_5 = write_frame_csv_md(
        table_5,
        tables_dir / TABLE_FILENAMES["4-5"],
        tables_dir / "table_4_5_e2e_accuracy.md",
        "Table 4-5 E2E prediction accuracy",
    )

    batch_curve = full_metrics.groupby("batch_size", as_index=False).agg(mape=("ape", "mean")).sort_values("batch_size")
    inter_curve = full_metrics.groupby("inter_threads", as_index=False).agg(
        actual_us=("actual_e2e_us", "mean"),
        predicted_us=("predicted_e2e_us", "mean"),
        mape=("ape", "mean"),
    ).sort_values("inter_threads")

    plot_scatter(
        full_metrics.assign(actual=full_metrics["actual_e2e_us"], predicted=full_metrics["predicted_e2e_us"]),
        "actual",
        "predicted",
        figures_dir / FIGURE_FILENAMES["4-11"],
        "Figure 4-11 E2E predicted vs. actual latency",
        audit_lines=[
            f"rows = {len(full_metrics):,}",
            f"MAPE = {summary_metrics['mape']:.4f}",
            f"P90 = {summary_metrics['p90_ape']:.4f}",
        ],
        audit_loc="upper right",
    )
    plot_line(
        batch_curve,
        "batch_size",
        "mape",
        figures_dir / FIGURE_FILENAMES["4-12"],
        "Figure 4-12 E2E MAPE across batch sizes",
        ylabel="MAPE",
        audit_lines=[
            f"batch sizes = {len(batch_curve):,}",
            f"best batch MAPE = {float(batch_curve['mape'].min()):.4f}",
        ],
        audit_loc="inside upper right",
        figsize_override=_scaled_figsize(8.2, 4.8),
        font_scale_multiplier=1,
    )
    inter_plot = pd.concat(
        [
            inter_curve[["inter_threads", "actual_us"]].rename(columns={"actual_us": "latency_us"}).assign(series="actual"),
            inter_curve[["inter_threads", "predicted_us"]].rename(columns={"predicted_us": "latency_us"}).assign(series="predicted"),
        ],
        ignore_index=True,
    )
    plot_line(
        inter_plot,
        "inter_threads",
        "latency_us",
        figures_dir / FIGURE_FILENAMES["4-13"],
        "Figure 4-13 Predicted / actual latency vs. branch parallelism",
        group_col="series",
        ylabel="latency (us)",
        audit_lines=[
            f"parallelism settings = {_sanitize_list(sorted(inter_curve['inter_threads'].astype(int).tolist()))}",
            f"lowest MAPE = {float(inter_curve['mape'].min()):.4f}",
        ],
        legend_side="top",
        legend_ncols=2,
        audit_loc="inside upper right",
    )

    payload = {
        "tables": {"4-5": str(csv_5)},
        "figures": {
            "4-11": str(figures_dir / FIGURE_FILENAMES["4-11"]),
            "4-12": str(figures_dir / FIGURE_FILENAMES["4-12"]),
            "4-13": str(figures_dir / FIGURE_FILENAMES["4-13"]),
        },
        "summary": {
            "overall": summary_metrics,
            "inter_threads_curve": inter_curve.to_dict(orient="records"),
            "batch_curve": batch_curve.to_dict(orient="records"),
            "prediction_source": normalized_prediction_source,
            "model_name": _prediction_source_model_name(normalized_prediction_source),
        },
    }
    _write_summary_bundle(section_dir, "e2e_core", payload)
    return SectionResult(name="e2e_core", outputs=payload)


def run_e2e_sum_baseline(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    e2e_artifact_root: Path | None = None,
    prediction_source: str = PREDICTION_SOURCE_GROUPED,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["e2e"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)
    normalized_prediction_source = _normalize_prediction_source(prediction_source)
    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    test_meta = test_dataset[["row_uid", "batch_size", "num_indices_per_lookup", "inter_threads"]].copy()
    selected_prediction_bundle = _load_model_prediction_frame(
        output_root,
        single_op_root=single_op_root,
        prediction_source=normalized_prediction_source,
        test_dataset=test_dataset,
    )
    selected_predictions = selected_prediction_bundle["prediction_df"].copy()
    fair_baseline = ensure_fair_single_mlp_artifact(output_root, single_op_artifact_root=single_op_root)
    fair_predictions = fair_baseline["predictions_test"].copy()
    fair_predictions_enriched = fair_predictions.merge(test_meta, on="row_uid", how="left")
    e2e_frame = _load_e2e_full_metrics(
        output_root,
        single_op_root=single_op_root,
        e2e_root=e2e_root,
        prediction_source=normalized_prediction_source,
    )

    gather_frame = test_dataset[test_dataset["op_type"] == "Gather"].copy()
    gather_pred = selected_predictions[selected_predictions["op_type"] == "Gather"][["row_uid", "pred_us"]]
    gather_frame = gather_frame.merge(gather_pred, on="row_uid", how="left")
    gather_frame["ape"] = (
        (gather_frame["pred_us"].astype(float) - gather_frame["label_operator_actual_dur_us"].astype(float)).abs()
        / gather_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
    ).fillna(0.0)
    gather_case = gather_frame.sort_values("ape", ascending=False).iloc[0]

    meta_frame = test_dataset[test_dataset["op_type"].isin(["Reshape", "Shape", "Unsqueeze", "Flatten"])].copy()
    meta_pred = selected_predictions[selected_predictions["op_type"].isin(["Reshape", "Shape", "Unsqueeze", "Flatten"])][["row_uid", "pred_us"]]
    meta_frame = meta_frame.merge(meta_pred, on="row_uid", how="left")
    meta_frame["ape"] = (
        (meta_frame["pred_us"].astype(float) - meta_frame["label_operator_actual_dur_us"].astype(float)).abs()
        / meta_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
    ).fillna(0.0)
    meta_case = meta_frame.sort_values(["label_operator_actual_dur_us", "ape"], ascending=[True, False]).iloc[0]

    gemm_frame = test_dataset[test_dataset["op_type"].isin(["Gemm", "MatMul"])].copy()
    gemm_pred = selected_predictions[selected_predictions["op_type"].isin(["Gemm", "MatMul"])][["row_uid", "pred_us"]]
    gemm_frame = gemm_frame.merge(gemm_pred, on="row_uid", how="left")
    gemm_frame["ape"] = (
        (gemm_frame["pred_us"].astype(float) - gemm_frame["label_operator_actual_dur_us"].astype(float)).abs()
        / gemm_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
    ).fillna(0.0)
    gemm_case = gemm_frame.sort_values(["feat_gemm_mac_count", "ape"], ascending=[True, False]).iloc[0]

    e2e_case = e2e_frame.sort_values("ape", ascending=False).iloc[0]
    fair_pipeline_combo = _build_pipeline_combo_predictions(fair_predictions_enriched).rename(
        columns={"predicted_e2e_us": "single_mlp_pipeline_us"}
    )
    table_4_7 = pd.DataFrame(
        [
            {
                "sample_type": "Gather random-access hotspot",
                "actual_us": float(gather_case["label_operator_actual_dur_us"]),
                "predicted_us": float(gather_case["pred_us"]),
                "relative_error": float(gather_case["ape"]),
                "explanation": "Random large-table accesses amplify memory-latency variance that the calibrated analytical proxy cannot fully smooth.",
            },
            {
                "sample_type": "Small-tensor view/meta op",
                "actual_us": float(meta_case["label_operator_actual_dur_us"]),
                "predicted_us": float(meta_case["pred_us"]),
                "relative_error": float(meta_case["ape"]),
                "explanation": "Framework dispatch and bookkeeping overhead dominate when tensor payload is tiny, making latency noisier than shape-only features suggest.",
            },
            {
                "sample_type": "Small-dimension Gemm/MatMul",
                "actual_us": float(gemm_case["label_operator_actual_dur_us"]),
                "predicted_us": float(gemm_case["pred_us"]),
                "relative_error": float(gemm_case["ape"]),
                "explanation": "Micro-kernel utilization is low at small M/N/K, so measured latency deviates from saturated compute-mode assumptions.",
            },
            {
                "sample_type": "E2E synchronization-heavy combo",
                "actual_us": float(e2e_case["actual_e2e_us"]),
                "predicted_us": float(e2e_case["predicted_e2e_us"]),
                "relative_error": float(e2e_case["ape"]),
                "explanation": "Tail-barrier and branch-slot synchronization amplify any upstream single-op bias at the full-graph level.",
            },
        ]
    )
    csv_47, md_47 = write_frame_csv_md(
        table_4_7,
        tables_dir / TABLE_FILENAMES["4-7"],
        tables_dir / "table_4_7_error_cases.md",
        "Table 4-7 Typical error cases",
    )

    plot_bar(
        table_4_7,
        "sample_type",
        "relative_error",
        figures_dir / FIGURE_FILENAMES["4-19"],
        "Figure 4-19 Relative error of representative failure cases",
        ylabel="relative error",
        color="#B279A2",
        audit_lines=[
            "The selected cases correspond to the dominant error sources discussed in Section 4.4.5.",
            f"worst case = {table_4_7.sort_values('relative_error', ascending=False).iloc[0]['sample_type']}",
        ],
    )

    payload = {
        "tables": {"4-7": str(csv_47)},
        "figures": {"4-19": str(figures_dir / FIGURE_FILENAMES["4-19"])},
        "summary": {
            "error_rows": table_4_7.to_dict(orient="records"),
            "single_mlp_pipeline_rows": fair_pipeline_combo.to_dict(orient="records"),
            "prediction_source": normalized_prediction_source,
            "model_name": selected_prediction_bundle["model_name"],
        },
    }
    _write_summary_bundle(section_dir, "e2e_sum_baseline", payload)
    return SectionResult(name="e2e_sum_baseline", outputs=payload)


def run_timeline_cases(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    prediction_source: str = PREDICTION_SOURCE_GROUPED,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["e2e"]
    timeline_dir = section_dir / "timeline_cases"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    prediction_bundle = _load_model_prediction_frame(
        output_root,
        single_op_root=single_op_root,
        prediction_source=prediction_source,
        test_dataset=test_dataset,
    )
    prediction_df = prediction_bundle["prediction_df"]
    combo_specs = build_combo_specs(prediction_df, ort_root=Path(DEFAULT_ORT_ROOT))
    combo_map = {(spec.case_id, spec.combo): spec for spec in combo_specs}

    summary_rows: list[dict[str, Any]] = []
    gantt_frames: list[pd.DataFrame] = []
    critical_frames: list[pd.DataFrame] = []
    for case_id, combo in TIMELINE_CASES:
        combo_spec = combo_map[(case_id, combo)]
        combo_rows = prediction_df[(prediction_df["case_id"] == case_id) & (prediction_df["combo"] == combo)].copy()
        graph = build_op_graph(load_op_shapes_frame(combo_spec.artifact_paths.shape_csv))
        schedule_result = schedule_combo(combo_spec, graph, combo_rows)
        timeline_df = load_timeline_frame(combo_spec.artifact_paths.timeline_csv)
        actual = compute_mean_batch_span(timeline_df, [graph[node_idx].node_name for node_idx in schedule_result.expected_node_indices])
        actual_obs = analyze_embedding_execution(timeline_df, combo_spec.inter_threads)
        duration_by_node = {int(row.op_idx): float(row.pred_us) for row in combo_rows.itertuples(index=False)}
        critical_path, critical_us = _compute_longest_path(graph, duration_by_node)

        node_span_map = schedule_result.node_span_map()
        path_rows = []
        for step, node_idx in enumerate(critical_path):
            node = graph[node_idx]
            span = node_span_map.get(node_idx)
            if span is None:
                continue
            path_rows.append(
                {
                    "case_id": case_id,
                    "combo": combo,
                    "step": step,
                    "node_idx": node_idx,
                    "node_name": node.node_name,
                    "op_type": node.op_type,
                    "partition": node.partition,
                    "predicted_start_us": span.start_us,
                    "predicted_end_us": span.end_us,
                    "predicted_duration_us": span.duration_us,
                }
            )
        critical_frame = pd.DataFrame(path_rows)
        critical_frames.append(critical_frame)

        gantt_rows = []
        for span in schedule_result.task_spans:
            gantt_rows.append(
                {
                    "label": span.task_id,
                    "task_kind": span.task_kind,
                    "partition": span.partition,
                    "start_us": span.start_us,
                    "end_us": span.end_us,
                    "duration_us": span.duration_us,
                }
            )
        gantt_frame = pd.DataFrame(gantt_rows)
        gantt_frames.append(gantt_frame)

        case_dir = timeline_dir / f"{case_id}__{combo}"
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            case_dir / "summary.json",
            {
                "case_id": case_id,
                "combo": combo,
                "predicted_full_graph_us": float(schedule_result.predicted_full_graph_us),
                "actual_full_graph_us": float(actual.mean_span_us),
                "predicted_critical_path_us": critical_us,
                "embedding_launch_order": actual_obs.representative_launch_order,
                "max_gather_concurrency": actual_obs.representative_max_gather_concurrency,
            },
        )
        gantt_frame.to_csv(case_dir / "task_spans.csv", index=False)
        critical_frame.to_csv(case_dir / "critical_path.csv", index=False)

        summary_rows.append(
            {
                "case_id": case_id,
                "combo": combo,
                "batch_size": combo_spec.batch_size,
                "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                "inter_threads": combo_spec.inter_threads,
                "predicted_full_graph_us": float(schedule_result.predicted_full_graph_us),
                "actual_full_graph_us": float(actual.mean_span_us),
                "predicted_critical_path_us": critical_us,
                "critical_path_nodes": len(critical_path),
                "ape": abs(float(schedule_result.predicted_full_graph_us) - float(actual.mean_span_us)) / float(actual.mean_span_us) if actual.mean_span_us else 0.0,
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    csv_path, md_path = write_frame_csv_md(summary_frame, tables_dir / "table_4_6_timeline_case_summary.csv", tables_dir / "table_4_6_timeline_case_summary.md", "Table 4-6 Timeline Case Summary")

    if gantt_frames:
        plot_frame = pd.concat(gantt_frames, ignore_index=True)
        if not plot_frame.empty:
            plt = _import_pyplot()
            fig, axes = plt.subplots(
                len(TIMELINE_CASES),
                1,
                figsize=_scaled_figsize(14.0, max(4.2, 3.9 * len(TIMELINE_CASES))),
                sharex=False,
            )
            if len(TIMELINE_CASES) == 1:
                axes = [axes]
            timeline_colors = {
                "bottom": "#4C78A8",
                "embedding": "#F58518",
                "tail": "#54A24B",
                "barrier": "#B279A2",
            }
            for ax, (case_id, combo), gantt_frame in zip(axes, TIMELINE_CASES, gantt_frames):
                for row in gantt_frame.itertuples(index=False):
                    color = timeline_colors.get(row.partition, "#7F7F7F")
                    ax.broken_barh([(float(row.start_us), float(row.duration_us))], (0, 1), facecolors=color, alpha=0.7)
                ax.set_yticks([])
                ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
                case_summary = next(row for row in summary_rows if row["case_id"] == case_id and row["combo"] == combo)
                add_audit_callout(
                    ax,
                    [
                        f"predicted = {case_summary['predicted_full_graph_us'] / 1_000_000:.3f} s",
                        f"actual = {case_summary['actual_full_graph_us'] / 1_000_000:.3f} s",
                        f"APE = {case_summary['ape']:.3f}",
                    ],
                    loc="upper left",
                )
            axes[0].legend(
                handles=_partition_legend_handles(include_barrier=True),
                frameon=False,
                ncols=4,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
            )
            axes[-1].set_xlabel("time (us)")
            save_figure(fig, figures_dir / FIGURE_FILENAMES["4-14"])

    if critical_frames:
        critical_plot = pd.concat(
            [
                frame.assign(case_label=f"{row['case_id']} / {row['combo']}")
                for frame, row in zip(critical_frames, summary_rows)
            ],
            ignore_index=True,
        )
        if not critical_plot.empty:
            plt = _import_pyplot()
            fig, ax = plt.subplots(figsize=_scaled_figsize(14.0, 6.2))
            colors = {"bottom": "#4C78A8", "embedding": "#F58518", "tail": "#54A24B"}
            for case_label, sub in critical_plot.groupby("case_label"):
                left = 0.0
                for row in sub.itertuples(index=False):
                    ax.bar(case_label, row.predicted_duration_us, bottom=left, color=colors.get(row.partition, "#7F7F7F"))
                    left += row.predicted_duration_us
            ax.set_ylabel("predicted duration (us)")
            ax.set_xticks(
                ax.get_xticks(),
                [_wrap_axis_label(tick.get_text(), width=22) for tick in ax.get_xticklabels()],
                rotation=0.0,
            )
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            ax.legend(
                handles=_partition_legend_handles(include_barrier=False),
                frameon=False,
                ncols=3,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.10),
            )
            add_audit_callout(
                ax,
                [
                    f"cases = {len(summary_rows)}",
                    f"worst APE = {max(row['ape'] for row in summary_rows):.3f}",
                    f"critical path nodes = {critical_plot['step'].max() + 1 if not critical_plot.empty else 0}",
                ],
                loc="upper right",
            )
            save_figure(fig, figures_dir / FIGURE_FILENAMES["4-15"])

    payload = {
        "tables": {"timeline_case_summary": str(csv_path)},
        "figures": {
            "4-14": str(figures_dir / FIGURE_FILENAMES["4-14"]),
            "4-15": str(figures_dir / FIGURE_FILENAMES["4-15"]),
        },
        "summary": summary_rows,
        "prediction_source": _normalize_prediction_source(prediction_source),
        "model_name": prediction_bundle["model_name"],
    }
    _write_summary_bundle(section_dir, "timeline_cases", payload)
    return SectionResult(name="timelines", outputs=payload)


def build_figures_catalog(output_root: Path | None = None) -> SectionResult:
    layout = ensure_output_layout(output_root)
    rows = []
    figure_claims = {
        "4-1": ("platform", "single-op data collection workflow"),
        "4-2": ("platform", "full-graph timeline aggregation workflow"),
        "4-3": ("single-op core", "overall predicted-vs-actual scatter"),
        "4-4": ("single-op core", "category-level MAPE comparison"),
        "4-5": ("single-op core", "Gather error vs lookup scale"),
        "4-6": ("single-op core", "ReduceSum error vs reduction scale"),
        "4-7": ("single-op core", "Transpose/Concat error vs data size"),
        "4-8": ("single-op core", "Gemm/MatMul error vs MAC count"),
        "4-9": ("single-op OOD", "batch/thread holdout single-op generalization comparison"),
        "4-10": ("single-op OOD", "operator-family generalization reference under alternate holdouts"),
        "4-11": ("e2e core", "overall E2E predicted-vs-actual scatter"),
        "4-12": ("e2e core", "batch-size stability"),
        "4-13": ("e2e core", "parallelism sensitivity"),
        "4-14": ("timeline", "timeline replay comparison"),
        "4-15": ("timeline", "critical path breakdown"),
        "4-16": ("ablation", "fair-comparison ablation CDF"),
        "4-17": ("ablation", "mean error and >10% error rate"),
        "4-18": ("ablation", "parallelism sensitivity of fair-comparison variants"),
        "4-19": ("error analysis", "representative failure cases"),
    }
    for figure_no, filename in FIGURE_FILENAMES.items():
        stage, claim = figure_claims.get(figure_no, ("unknown", ""))
        rows.append(
            {
                "figure_no": figure_no,
                "stage": stage,
                "claim": claim,
                "filename": filename,
                "path": str(layout["figures"] / filename),
            }
        )
    rows.append(
        {
            "figure_no": "4-5-panel",
            "stage": "single-op core",
            "claim": "combined representative-op panel with four subplots",
            "filename": REPRESENTATIVE_PANEL_FILENAME,
            "path": str(layout["figures"] / REPRESENTATIVE_PANEL_FILENAME),
        }
    )
    frame = pd.DataFrame(rows).sort_values("figure_no").reset_index(drop=True)
    csv_path, md_path = write_frame_csv_md(frame, layout["manifests"] / "figures_catalog.csv", layout["manifests"] / "figures_catalog.md", "CPU Single-Only Figure Catalog")
    payload = {"csv": str(csv_path), "md": str(md_path), "rows": rows}
    write_json(layout["manifests"] / "figures_catalog.json", payload)
    return SectionResult(name="figures", outputs=payload)


def build_report_draft(
    output_root: Path | None = None,
    *,
    draft_path: Path | None = None,
    single_op_artifact_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    single_op_summary = read_json(layout["single_op"] / "single_op_core_summary.json") if (layout["single_op"] / "single_op_core_summary.json").exists() else {}
    ood_summary = read_json(layout["single_op"] / "single_op_ood_summary.json") if (layout["single_op"] / "single_op_ood_summary.json").exists() else {}
    e2e_summary = read_json(layout["e2e"] / "e2e_core_summary.json") if (layout["e2e"] / "e2e_core_summary.json").exists() else {}
    sum_baseline_summary = read_json(layout["e2e"] / "e2e_sum_baseline_summary.json") if (layout["e2e"] / "e2e_sum_baseline_summary.json").exists() else {}
    ablation_summary = read_json(layout["ablation"] / "single_op_ablation_summary.json") if (layout["ablation"] / "single_op_ablation_summary.json").exists() else {}
    timeline_summary = read_json(layout["e2e"] / "timeline_cases_summary.json") if (layout["e2e"] / "timeline_cases_summary.json").exists() else {}
    figures_catalog = read_json(layout["manifests"] / "figures_catalog.json") if (layout["manifests"] / "figures_catalog.json").exists() else {"rows": []}
    platform_summary = read_json(layout["manifests"] / "platform_summary.json") if (layout["manifests"] / "platform_summary.json").exists() else {}
    platform_metrics = platform_summary.get("summary", {})
    single_op_metrics = single_op_summary.get("summary", {})
    ood_metrics = ood_summary.get("summary", {})
    e2e_metrics = e2e_summary.get("summary", {})
    ablation_metrics = ablation_summary.get("summary", {})
    error_metrics = sum_baseline_summary.get("summary", {})
    timeline_rows = timeline_summary.get("summary", [])

    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", usecols=["split", "batch_size", "num_indices_per_lookup", "num_threads", "inter_threads", "op_type"], low_memory=False)
    split_counts = dataset_df["split"].value_counts().to_dict()
    batch_range = f"{int(dataset_df['batch_size'].min())}-{int(dataset_df['batch_size'].max())}"
    nip_range = f"{int(dataset_df['num_indices_per_lookup'].min())}-{int(dataset_df['num_indices_per_lookup'].max())}"
    op_types = sorted(dataset_df["op_type"].dropna().unique().tolist())
    e2e_overall = e2e_metrics.get("overall", {})
    ablation_rows = ablation_metrics.get("variant_rows", [])
    active_prediction_source = str(single_op_metrics.get("active_prediction_source", PREDICTION_SOURCE_GROUPED))
    active_model_name = str(single_op_metrics.get("active_model_name", _prediction_source_model_name(active_prediction_source)))
    single_only_mode = active_prediction_source == PREDICTION_SOURCE_SINGLE_FAIR and not any(
        "grouped MLP" in str(row.get("variant", "")) for row in ablation_rows
    )
    rep_usecols = [
        "row_uid",
        "split",
        "op_type",
        "num_threads",
        "feat_lookup_count",
        "feat_reduction_work_items",
        "feat_io_bytes_sum",
        "feat_gemm_mac_count",
        "label_operator_actual_dur_us",
    ]
    rep_dataset = pd.read_csv(single_op_root / "classed_dataset_full.csv", usecols=rep_usecols, low_memory=False)
    rep_test_dataset = rep_dataset[rep_dataset["split"] == "test"].copy()
    rep_prediction_bundle = _load_model_prediction_frame(
        output_root,
        single_op_root=single_op_root,
        prediction_source=active_prediction_source,
        test_dataset=rep_test_dataset,
    )
    rep_predictions = rep_prediction_bundle["prediction_df"][["row_uid", "pred_us"]]

    def _rep_frame(op_mask: pd.Series) -> pd.DataFrame:
        frame = rep_test_dataset[op_mask].copy().merge(rep_predictions, on="row_uid", how="left")
        frame["ape"] = (
            (frame["pred_us"].astype(float) - frame["label_operator_actual_dur_us"].astype(float)).abs()
            / frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
        ).fillna(0.0)
        return frame

    gather_trend = _error_trend_summary(_rep_frame(rep_test_dataset["op_type"] == "Gather"), "feat_lookup_count")
    reduce_trend = _error_trend_summary(_rep_frame(rep_test_dataset["op_type"] == "ReduceSum"), "feat_reduction_work_items")
    layout_trend = _error_trend_summary(_rep_frame(rep_test_dataset["op_type"].isin(["Transpose", "Concat"])), "feat_io_bytes_sum")
    gemm_trend = _error_trend_summary(_rep_frame(rep_test_dataset["op_type"].isin(["Gemm", "MatMul"])), "feat_gemm_mac_count")

    gather_low = gather_trend["bins"][0] if gather_trend["bins"] else None
    gather_high = gather_trend["bins"][-1] if gather_trend["bins"] else None
    reduce_low = reduce_trend["bins"][0] if reduce_trend["bins"] else None
    reduce_high = reduce_trend["bins"][-1] if reduce_trend["bins"] else None
    layout_low = layout_trend["bins"][0] if layout_trend["bins"] else None
    layout_high = layout_trend["bins"][-1] if layout_trend["bins"] else None
    gemm_low = gemm_trend["bins"][0] if gemm_trend["bins"] else None
    gemm_high = gemm_trend["bins"][-1] if gemm_trend["bins"] else None
    error_rows = error_metrics.get("error_rows", [])
    pipeline_row = next((row for row in ablation_rows if row["variant"] == "Analytical + grouped MLP + pipeline"), None)
    single_pipeline_row = next((row for row in ablation_rows if row["variant"] == "Analytical + single MLP + pipeline"), None)
    grouped_simple_add_row = next((row for row in ablation_rows if row["variant"] == "Analytical + grouped MLP + simple add"), None)
    single_simple_add_row = next((row for row in ablation_rows if row["variant"] == "Analytical + single MLP + simple add"), None)
    analytical_row = next((row for row in ablation_rows if row["variant"] == "Analytical + simple add"), None)
    analytical_pipeline_row = next((row for row in ablation_rows if row["variant"] == "Analytical + pipeline"), None)
    analytical_covered_rows = int(single_op_metrics.get("analytical_test_count", 0))
    analytical_total_rows = int(single_op_metrics.get("analytical_test_total_rows", 0))
    analytical_coverage_rate = float(single_op_metrics.get("analytical_test_coverage_rate", 0.0))
    analytical_excluded_groups = [
        str(group)
        for group in single_op_metrics.get("analytical_excluded_groups", [])
        if str(group).strip()
    ]
    analytical_excluded_groups_text = "/".join(analytical_excluded_groups)
    selected_draft_path = Path(
        draft_path or (CPU_REPORT_SINGLE_ONLY_DATA_SYNC_PATH if single_only_mode else CPU_REPORT_DRAFT_PATH)
    )
    artifact_output_root_text = (
        "ORT/CPU_Perf_Model/artifacts/latest/cpu_single_only_report/"
        if single_only_mode
        else "ORT/CPU_Perf_Model/artifacts/latest/cpu_single_only_report/"
    )
    single_only_sync_notice = [
        "> 本文件由 CPU_Perf_Model single-only runner 自动刷新，用于同步最新实验数据；稳定正文与复现约束请见 `ORT/CPU_Perf_Model/cpu_single_only_report_draft.md`。",
        "",
    ] if single_only_mode else []
    single_op_intro = (
        "本节首先给出 fair single MLP 在测试集上的总体精度，再按五类算子统计误差，并对 Gather、ReduceSum、Transpose/Concat 以及 Gemm/MatMul 的代表性行为做可视化分析。整体作图风格参考 Concorde 中“真实值-预测值关系、误差分布、典型 case 解释”的组织方式，但在本组实验里不再引入 grouped MLP 对照。"
        if single_only_mode
        else "本节首先给出最终分组模型在测试集上的总体精度，再按五类算子统计误差，并对 Gather、ReduceSum、Transpose/Concat 以及 Gemm/MatMul 的代表性行为做可视化分析。整体作图风格参考 Concorde 中“真实值-预测值关系、误差分布、典型 case 解释”的组织方式，但结合 DLRM 场景保留了更强的算子语义解释。"
    )
    section_421_text = (
        f"表 4-3 给出了两种单算子模型口径的总体结果：纯解析模型，以及同数据同特征池重跑的 fair single MLP。当前 single MLP 使用与 grouped 特征池并集一致的 {single_op_metrics.get('baseline_feature_count', 0)} 个输入特征，在测试集上的 `MAPE` 为 {single_op_metrics.get('active_test_mape', 0.0):.4f}，`R^2` 为 {single_op_metrics.get('active_test_r2', 0.0):.4f}。纯解析模型按 analysis 套件一致的口径，在每个算子组上选取最佳 active analytical proxy 后，在 {analytical_covered_rows:,} 条可覆盖测试样本上得到 `MAPE = {single_op_metrics.get('analytical_test_mape', 0.0):.4f}`。此前若把 `ana_calib_total_us` 直接施加到全部测试样本，会把缺少 active analytical feature 的 `{analytical_excluded_groups_text}` 组也纳入比例误差，从而显著夸大 pure analytical 的总体误差；当前这部分覆盖率约为 {analytical_coverage_rate * 100.0:.1f}%。图 4-3 的散点结果显示，统一 single MLP 的大部分样本仍围绕 `y=x` 参考线分布，说明其作为后续整图聚合输入是稳定可用的。"
        if single_only_mode
        else f"表 4-3 给出了三种单算子模型口径的总体结果：纯解析模型、同数据同特征池重跑的 single MLP，以及本文采用的分组 analytical-MLP。后两者都使用相同的 `case-combo` 划分，并共享总计 {single_op_metrics.get('baseline_feature_count', 0)} 个输入特征池，只是分组模型按算子机理做静态路由并使用对应子集。在本次选定的 `64` 宽度、`15` 轮训练设置下，分组模型在测试集上的 `MAPE` 为 {single_op_metrics.get('grouped_test_mape', 0.0):.4f}，`R^2` 为 {single_op_metrics.get('grouped_test_r2', 0.0):.4f}；公平 single MLP 的 `MAPE` 为 {single_op_metrics.get('baseline_test_mape', 0.0):.4f}，因此分组模型的相对误差约降低了 {((single_op_metrics.get('baseline_test_mape', 0.0) - single_op_metrics.get('grouped_test_mape', 0.0)) / max(single_op_metrics.get('baseline_test_mape', 1e-9), 1e-9) * 100.0):.1f}%；纯解析模型按 analysis 套件一致的口径，在每个算子组上选取最佳 active analytical proxy 后，在 {analytical_covered_rows:,} 条可覆盖测试样本上得到 `MAPE = {single_op_metrics.get('analytical_test_mape', 0.0):.4f}`。此前若把 `ana_calib_total_us` 直接施加到全部测试样本，会把缺少 active analytical feature 的 `{analytical_excluded_groups_text}` 组也纳入比例误差，从而显著夸大 pure analytical 的总体误差；当前这部分覆盖率约为 {analytical_coverage_rate * 100.0:.1f}%。图 4-3 的散点结果显示，分组模型的大部分样本仍围绕 `y=x` 参考线分布，说明其作为后续整图聚合输入是稳定可用的。"
    )
    section_422_text = (
        "表 4-4 和图 4-4 展示了五类算子的误差差异。总体上，视图/元数据类和布局搬移类更容易预测，因为其执行路径较稳定；索引访存类和轻计算-访存混合类误差相对更大，主要原因是它们更容易受到随机访存、线程调度和小张量固定开销的影响。这种类别差异与第三章中对 ORT CPU kernel 机制的分析是一致的，也说明即使统一 single MLP 使用同一特征池，不同算子机理仍然会在误差分布上留下明显结构。"
        if single_only_mode
        else "表 4-4 和图 4-4 展示了五类算子的误差差异。总体上，视图/元数据类和布局搬移类更容易预测，因为其执行路径较稳定；索引访存类和轻计算-访存混合类误差相对更大，主要原因是它们更容易受到随机访存、线程调度和小张量固定开销的影响。这种类别差异与第三章中对 ORT CPU kernel 机制的分析是一致的，也说明统一 MLP 难以同时覆盖这几类机理差异显著的节点。"
    )
    ablation_intro = (
        "前两节给出了单算子建模与整图聚合的主要结果，本节进一步通过消融实验分析这些结果的来源。为识别节点级预测能力与整图聚合机制对最终精度的相对贡献，本文采用逐步引入组件的方式，比较 `Analytical + simple add`、`Analytical + pipeline`、`Analytical + single MLP + simple add` 和 `Analytical + single MLP + pipeline` 四个变体。该实验设计旨在区分整图精度改善主要来源于节点级预测误差的降低，还是来源于静态流水线聚合对执行结构的更准确刻画；同时，也为后续误差尾部样本的来源分析提供参照。"
        if single_only_mode
        else "这一节采用与 Concorde 类似的逐步加组件消融方式，并补充 single/grouped MLP 的公平对比，而不是简单做特征删除。具体构造六个变体：第一组仅使用 `Analytical model + Simple add`；第二组使用 `Analytical model + pipeline`；第三组使用与分组模型同数据、同特征池重跑的 `single MLP + Simple add`；第四组使用 `grouped MLP + Simple add`；第五组使用 `single MLP + pipeline`；第六组使用 `grouped MLP + pipeline`，即本文完整方法。这样的设计能够同时回答五个问题：解析模型本身能做多好、仅将整图聚合从 simple add 换成 pipeline 能带来多少收益、统一学习器能带来多少收益、分组建模是否优于统一 MLP，以及静态流水线聚合是否对整图预测确有必要。"
    )
    section_441_text = (
        f"图 4-16 给出了四个变体在整图相对误差上的累积分布。两条采用 `pipeline` 聚合的曲线整体明显左移于对应的 `simple add` 版本，表明在相同节点级预测条件下，显式建模分支重叠与同步边界能够显著压缩整图误差分布。与此同时，`Analytical + single MLP + pipeline` 的曲线进一步左移于 `Analytical + pipeline`，说明更准确的节点级预测仍然能够在静态聚合框架内继续转化为整图精度收益。图 4-17 进一步给出了各变体的平均相对误差与 `APE > 10%` 比例。对于纯解析模型，若整图仅采用 `simple add`，其整图 `MAPE` 为 {analytical_row.get('e2e_mape', 0.0) if analytical_row else 0.0:.4f}，`APE > 10%` 比例为 {analytical_row.get('e2e_gt10_rate', 0.0) if analytical_row else 0.0:.4f}；当聚合方式改为 `pipeline` 后，两项指标分别下降至 {analytical_pipeline_row.get('e2e_mape', 0.0) if analytical_pipeline_row else 0.0:.4f} 和 {analytical_pipeline_row.get('e2e_gt10_rate', 0.0) if analytical_pipeline_row else 0.0:.4f}。对于引入统一 `single MLP` 的版本，`simple add` 下的整图 `MAPE` 为 {single_simple_add_row.get('e2e_mape', 0.0) if single_simple_add_row else 0.0:.4f}，`APE > 10%` 比例为 {single_simple_add_row.get('e2e_gt10_rate', 0.0) if single_simple_add_row else 0.0:.4f}，而切换到 `pipeline` 后则进一步下降至 {single_pipeline_row.get('e2e_mape', 0.0) if single_pipeline_row else 0.0:.4f} 和 {single_pipeline_row.get('e2e_gt10_rate', 0.0) if single_pipeline_row else 0.0:.4f}。图 4-18 展示了不同 `inter_threads` 条件下各消融变体的整图 `MAPE` 变化，结果表明 `pipeline` 版本始终优于对应的 `simple add` 版本，而 `Analytical + single MLP + pipeline` 在各并行度条件下均保持最低误差水平。综合图 4-16 至图 4-18 可以认为，节点级学习器负责降低局部预测偏差，而静态流水线聚合负责将节点级结果组织为符合真实执行结构的整图估计，两者共同构成当前 single-only 设置下精度提升的主要来源。"
        if single_only_mode
        else f"表 4-6、图 4-16、图 4-17 和图 4-18 共同展示了六个变体的差异。对纯解析模型而言，仅把整图聚合从 simple add 改为 pipeline 后，整图 `MAPE` 由 {analytical_row.get('e2e_mape', 0.0) if analytical_row else 0.0:.4f} 下降到 {analytical_pipeline_row.get('e2e_mape', 0.0) if analytical_pipeline_row else 0.0:.4f}，说明静态调度本身就能纠正简单求和对并行重叠的系统性高估；但由于节点级 analytical 误差仍然较大，其整体精度依旧明显落后于学习器增强版本。加入公平 single MLP 后，单算子 `MAPE` 已明显下降，但若整图仍然简单求和，其整图误差仍然偏大；在本次选定的 `64` 宽度、`15` 轮训练设置下，single MLP 的 pipeline 版本整图 `MAPE` 为 {single_pipeline_row.get('e2e_mape', 0.0) if single_pipeline_row else 0.0:.4f}，而分组 MLP 的 pipeline 版本为 {pipeline_row.get('e2e_mape', 0.0) if pipeline_row else 0.0:.4f}，后者相对降低了 {((single_pipeline_row.get('e2e_mape', 0.0) - pipeline_row.get('e2e_mape', 0.0)) / max(single_pipeline_row.get('e2e_mape', 1e-9), 1e-9) * 100.0):.1f}%。图 4-16 的误差 CDF 与图 4-17 的平均误差/大误差比例统计共同说明，完整模型不仅降低了均值误差，也显著压缩了误差尾部。"
    )
    summary_text = (
        f"本章首先在 {platform_metrics.get('single_op_rows', 0):,} 条单算子样本和 {platform_metrics.get('full_e2e_combos', 0):,} 个完整图配置上完成了 single-only 版本实验。结果表明，统一 single MLP 在随机切分单算子测试上的 `MAPE` 为 {single_op_metrics.get('active_test_mape', 0.0):.4f}`，显著优于纯解析模型；进一步把 same-split single MLP 接入同一静态流水线后，其整图 `MAPE` 为 {single_pipeline_row.get('e2e_mape', 0.0) if single_pipeline_row else 0.0:.4f}`，相较 `single MLP + simple add` 的 {single_simple_add_row.get('e2e_mape', 0.0) if single_simple_add_row else 0.0:.4f}` 进一步下降。新构造的 single-only 实验说明：即使完全去掉 grouped MLP，解析代理、统一 single MLP 与静态流水线聚合这条两级建模链路依然成立，并且整图层面的主要收益仍来自 pipeline 对并行重叠和同步边界的显式建模。"
        if single_only_mode
        else f"本章首先在 {platform_metrics.get('single_op_rows', 0):,} 条单算子样本和 {platform_metrics.get('full_e2e_combos', 0):,} 个完整图配置上完成了统一实验。结果表明，在本次选定的 `64x15` single MLP 设定下，分组 analytical-MLP 在随机切分单算子测试上的 `MAPE` 为 {single_op_metrics.get('grouped_test_mape', 0.0):.4f}，相比 single MLP 的 {single_op_metrics.get('baseline_test_mape', 0.0):.4f} 低约 {((single_op_metrics.get('baseline_test_mape', 0.0) - single_op_metrics.get('grouped_test_mape', 0.0)) / max(single_op_metrics.get('baseline_test_mape', 1e-9), 1e-9) * 100.0):.1f}%；进一步把 single MLP 也接入同一静态流水线后，其整图 `MAPE` 为 {single_pipeline_row.get('e2e_mape', 0.0) if single_pipeline_row else 0.0:.4f}，而分组 pipeline 的整图 `MAPE` 为 {pipeline_row.get('e2e_mape', 0.0) if pipeline_row else 0.0:.4f}，后者相对降低约 {((single_pipeline_row.get('e2e_mape', 0.0) - pipeline_row.get('e2e_mape', 0.0)) / max(single_pipeline_row.get('e2e_mape', 1e-9), 1e-9) * 100.0):.1f}%；同时，分组模型仍显著优于纯解析模型。进一步的公平对比消融证明：仅靠解析模型或节点时延简单求和都无法得到可接受的整图精度，而将解析代理、单算子学习器和静态流水线聚合组合起来之后，可以同时压低平均误差和尾部误差。至此，第三章提出的解析代理特征、分组单算子模型与静态整图聚合三项核心设计，都得到了实验结果的直接验证。"
    )

    lines = [
        "# 第四章 CPU 实验与结果分析",
        "",
        *single_only_sync_notice,
        "## 4.1 实验平台与数据采集方法",
        "",
        f"本章实验围绕 ORT 上的 DLRM CPU 推理路径展开，目标是验证第三章提出的两级建模思路，即先对单算子时延进行静态预测，再将节点级预测结果通过静态流水线模型聚合为整图时延。当前复现脚本统一维护在 `ORT/CPU_Perf_Model/`，最终产物写入 `{artifact_output_root_text}`。",
        "",
        "### 4.1.1 实验平台",
        "",
        f"实验服务器为 {platform_metrics.get('host_cpu_model', 'Kunpeng-920')} 主机。整机从 `lscpu` 可观测到 4 路 CPU、192 个物理核心、单核单线程；但本章建模与运行口径采用单 NUMA 执行域，对应 24 个核心、24 MiB 共享 L3 以及 4 个 DDR4-2933 内存通道。单核缓存参数采用 `64 KiB L1I + 64 KiB L1D + 512 KiB L2`，本地 NUMA 理论带宽按 `100 GB/s` 建模，操作系统为 `Huawei Cloud EulerOS 2.0 (aarch64)`，编译器为 `GCC 10.3.1`。实验软件环境来自 `ort` conda 环境：`Python 3.11.14`、`PyTorch 2.9.0`、`ONNX 1.19.1`、`ONNX Runtime CANN 1.23.2`、`NumPy 1.26.4`、`pandas 3.0.1`。单算子标签采集依赖 DynamoRIO 侧的 profiling 结果，整图真值来自 ORT branch-parallel 运行的时间线记录。为减小系统噪声，实验固定 `intra-op` 与 `inter-op` 线程配置，并按单 NUMA 口径组织样本和静态流水线排程。",
        "",
        "### 4.1.2 DLRM 负载与算子样本生成",
        "",
        f"单算子样本来自 ORT 导出的 DLRM ONNX 图及其配套 `op_shapes`/profile 数据。统一数据集共包含 {platform_metrics.get('single_op_rows', 0):,} 条样本、{platform_metrics.get('single_op_cases', 0)} 个 case、{platform_metrics.get('single_op_combos', 0):,} 个 `case_id-combo` 配置，覆盖 `{batch_range}` 的 batch size、`{nip_range}` 的每次 lookup 索引数、`1-5` 的 intra-op 线程和 `1/3/4/5/6` 的 inter-op 线程。算子类型共 {len(op_types)} 种，分别映射为五类：索引访存类、布局搬移类、视图/元数据类、轻计算-访存混合类和计算主导类。每个 ONNX 节点都保留输入/输出 shape、线程配置、I/O 字节规模、归约规模、Gemm 的 M/N/K 与 analytical proxy 列，用于后续单算子建模。",
        "",
        "### 4.1.3 运行与标注流程",
        "",
        "单算子标签来自 ORT 执行后的 per-op profiling 结果。每个样本先丢弃最早的一个 profile batch，再对剩余 batch 的算子时延求均值，形成 `label_operator_actual_dur_us`。为了恢复整图真值，本文使用 branch-parallel runner 导出的 `branch_parallel_op_timeline.csv`，对每个 batch 计算整图 span，并采用与单算子一致的策略丢弃首个 batch 后再求均值。静态整图模型并不是简单求和，而是根据 `op_shapes` 重建 DAG，将 8 条 embedding 分支折叠为 branch task，再依据 `inter_threads` 控制的并行槽位进行离线排程，从而生成预测时间线、关键路径和整图时延。",
        "",
        "### 4.1.4 数据集划分与评价指标",
        "",
        f"单算子数据按 `sample_group=combo` 做 `7:2:1` 划分，实际样本数为 train/val/test = `{split_counts.get('train', 0):,} / {split_counts.get('val', 0):,} / {split_counts.get('test', 0):,}`。这种切分方式保证同一 `case_id-combo` 下的节点不会同时出现在训练集与测试集，避免 shape 级信息泄漏。单算子评价指标采用 `MAE`、`MAPE`、`RMSE` 和 `R^2`；整图评价指标采用 `MAE`、`MAPE`、`P50/P90` 相对误差，并按 batch size 和 branch parallelism 分组统计。",
        "",
        "## 4.2 算子级性能建模实验",
        "",
        single_op_intro,
        "",
        "### 4.2.1 单算子总体预测精度",
        "",
        section_421_text,
        "",
        "### 4.2.2 分类别预测精度",
        "",
        section_422_text,
        "",
        "### 4.2.3 典型算子预测结果分析",
        "",
        f"图 4-5(a) 至图 4-5(d) 分别展示了四类代表性算子的相对误差与其主导规模特征之间的关系。图 4-5(a) 给出了 `Gather` 的相对误差与索引查找工作量（`feat_lookup_count`）之间的关系。按索引查找工作量（`feat_lookup_count`）的分位区间统计，低规模区间 `{gather_low['work_min_fmt']}-{gather_low['work_max_fmt']}` 上的 `MAPE/P90 APE` 分别为 `{gather_low['mape']:.4f}/{gather_low['p90']:.4f}`，而高规模区间 `{gather_high['work_min_fmt']}-{gather_high['work_max_fmt']}` 上对应上升到 `{gather_high['mape']:.4f}/{gather_high['p90']:.4f}`，说明随着索引查找工作量（`feat_lookup_count`）增大，随机访存波动更容易放大尾部误差。图 4-5(b) 给出了 `ReduceSum` 的相对误差与归约工作量（`feat_reduction_work_items`）之间的关系。按归约工作量（`feat_reduction_work_items`）的四分位区间统计，`MAPE` 由低工作量区间 `{reduce_low['work_min_fmt']}-{reduce_low['work_max_fmt']}` 上的 `{reduce_low['mape']:.4f}` 下降到高工作量区间 `{reduce_high['work_min_fmt']}-{reduce_high['work_max_fmt']}` 上的 `{reduce_high['mape']:.4f}`，`P50/P90 APE` 也分别由 `{reduce_low['p50']:.4f}/{reduce_low['p90']:.4f}` 下降到 `{reduce_high['p50']:.4f}/{reduce_high['p90']:.4f}`，说明当归约工作量（`feat_reduction_work_items`）足够大时，解析代理对主导成本的刻画更稳定。图 4-5(c) 给出了 `Transpose` 与 `Concat` 的相对误差随数据搬移字节量（`feat_io_bytes_sum`）变化的情况。按数据搬移字节量（`feat_io_bytes_sum`）的四分位区间统计，若取上下界不重合的有效区间进行比较，`MAPE` 由中低规模区间 `4.80×10^1-5.90×10^5` 上的 `0.0666` 增加到最高区间 `{layout_high['work_min_fmt']}-{layout_high['work_max_fmt']}` 上的 `{layout_high['mape']:.4f}`，`P90 APE` 也由 `0.1342` 增加到 `{layout_high['p90']:.4f}`，说明这两类算子的预测偏差主要随数据搬移量（`feat_io_bytes_sum`）扩大而上升。图 4-5(d) 给出了 `Gemm/MatMul` 的相对误差与乘加运算量（`feat_gemm_mac_count`）之间的关系，`MAPE` 由低乘加运算量区间 `{gemm_low['work_min_fmt']}-{gemm_low['work_max_fmt']}` 上的 `{gemm_low['mape']:.4f}` 下降到高区间 `{gemm_high['work_min_fmt']}-{gemm_high['work_max_fmt']}` 上的 `{gemm_high['mape']:.4f}`，`P90 APE` 也由 `{gemm_low['p90']:.4f}` 下降到 `{gemm_high['p90']:.4f}`，反映出小维度下微核利用率不足会更明显地放大误差。整体来看，这些现象与第三章关于随机访存、数据搬移和 kernel 饱和度的分析是相互印证的。",
        "",
        "## 4.3 整图性能聚合实验",
        "",
        "整图实验验证第三章 3.3 中的静态流水线聚合模型。与单纯求和不同，该模型显式考虑 bottom、8 条 embedding branch 和 tail 的依赖关系，以及 `inter_threads` 决定的 branch 并行槽位约束，因此能够用节点级预测结果恢复整图执行的关键路径。",
        "",
        "### 4.3.1 整图预测总体精度",
        "",
        f"整图总体精度是评估两级建模方案有效性的关键指标。表 4-5 汇总了整图预测结果。其中，`overall` 表示全部完整图配置上的总体统计结果，`small`、`medium` 和 `large` 则分别表示按 `batch size` 划分的小、中、大三档区间，对应 `batch_size ≤ 1280`、`1280 < batch_size ≤ 1792` 和 `batch_size > 1792` 的配置集合。考虑到 `R^2` 更适合用于刻画整体样本上的拟合程度，而分桶后的区间内方差显著收缩，表中仅在 `overall` 行报告 `R^2`，其余分组主要比较相对误差统计。静态流水线模型在完整图样本上的 `MAPE` 为 {e2e_overall.get('mape', 0.0):.4f}，`P50` 和 `P90` 相对误差分别为 {e2e_overall.get('p50_ape', 0.0):.4f} 和 {e2e_overall.get('p90_ape', 0.0):.4f}。这表明节点级预测误差在整图聚合后未出现显著累积，模型能够较稳定地恢复整图 `makespan`，并将尾部误差控制在相对稳定的范围内。图 4-11 中预测值与真实值整体沿参考线分布，与表 4-5 的统计结果相一致。",
        "",
        "### 4.3.2 不同 batch size 下的整图精度",
        "",
        "图 4-12 给出了不同 batch size 下的整图 `MAPE`。整体趋势较平滑，说明模型在小 batch、中 batch 和大 batch 区间都保持了较稳定的误差水平。这一点非常重要，因为 DLRM 的 embedding lookup 和 top MLP 都会随 batch size 改变张量规模与工作集，若聚合模型缺乏稳定性，误差会在 batch 变化时明显抖动。",
        "",
        "### 4.3.3 不同分支并行度下的整图精度",
        "",
        "图 4-13 给出了不同 `inter_threads` 配置下的整图预测结果。尽管不同并行度条件下的误差水平存在一定波动，但模型整体上能够较准确地恢复并行槽位增加后整图时延下降、且边际收益逐步递减的变化趋势。这表明第三章用于刻画 branch 槽位竞争关系的静态近似具有良好的有效性。该近似的目标并非逐事件复现真实运行时调度过程，而是从结构层面捕获影响整图性能的主导因素。",
        "",
        "### 4.3.4 典型时间线案例分析",
        "",
        f"图 4-14 和图 4-15 展示了 {len(timeline_rows) if isinstance(timeline_rows, list) else 0} 个典型配置的真实/预测时间线与关键路径分解。通过这些案例可以看到，模型不仅预测了整图 makespan，还较准确地恢复了 bottom、embedding pool 和 tail 之间的先后关系。误差主要集中在分支完成时刻附近的同步边界和少数高波动节点上，这为后续误差分析提供了直接证据。",
        "",
        "## 4.4 消融实验与误差分析",
        "",
        ablation_intro,
        "",
        "### 4.4.1 公平对比消融结果",
        "",
        section_441_text,
        "",
        "### 4.4.2 误差来源分析",
        "",
        "表 4-7 和图 4-19 汇总了四类代表性误差样本。第一类是大表随机访存下的 Gather，误差主要来自实际内存访问时延波动；第二类是小张量视图/元数据算子，固定框架开销占比过高导致样本噪声明显；第三类是小维度 Gemm/MatMul，微核未充分饱和时更难被解析代理拟合；第四类则是整图级同步场景，单个 branch 的偏差在 barrier 处被放大。这些现象说明，当前模型的主要剩余误差并不来自统计偶然，而是来自 ORT CPU 执行中仍难以静态精确恢复的动态效应。",
        "",
        "## 4.5 本章小结",
        "",
        summary_text,
        "",
        f"本章共生成 {len(figures_catalog.get('rows', []))} 张图，全部由对应 CPU Single-Only runner 自动复现，并写入 `{selected_draft_path}`。",
        "",
    ]
    draft = "\n".join(lines).rstrip() + "\n"
    selected_draft_path.parent.mkdir(parents=True, exist_ok=True)
    selected_draft_path.write_text(draft, encoding="utf-8")
    return SectionResult(name="draft", outputs={"draft_path": str(selected_draft_path)})


def build_run_manifest(
    *,
    output_root: Path | None = None,
    sections: list[SectionResult],
    single_op_artifact_root: Path = SINGLE_OP_ARTIFACT_ROOT,
    e2e_artifact_root: Path = E2E_ARTIFACT_ROOT,
) -> Path:
    layout = ensure_output_layout(output_root)
    manifest = {
        "output_root": str(layout["root"]),
        "single_op_artifact_root": str(single_op_artifact_root),
        "e2e_artifact_root": str(e2e_artifact_root),
        "python_version": sys.version.split()[0],
        "execution_order": [section.name for section in sections],
        "generated_files": [
            path
            for section in sections
            for path in section.outputs.values()
            if isinstance(path, str)
        ],
        "sections": [
            {
                "name": section.name,
                "outputs": section.outputs,
            }
            for section in sections
        ],
    }
    return write_json(layout["manifests"] / "run_manifest.json", manifest)
