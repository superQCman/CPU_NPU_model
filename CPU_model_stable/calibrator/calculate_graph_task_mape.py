#!/usr/bin/env python3
"""Calculate graph-task MAPE from existing single-operator predictions.

This is a read-only reporting utility. It neither trains a model nor changes
the prediction values produced by the main performance-model pipeline.

The task boundary is determined only from the static graph and operator type:
Shape, Unsqueeze, Flatten, Reshape, Add, and Mul are assigned to an adjacent
non-small operator. A direct predecessor is preferred; if unavailable, a
direct successor is used. Actual latency and predicted latency are summed with
the selected task before the task-level absolute percentage error is computed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow `python calibrator/calculate_graph_task_mape.py ...` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cpu_perf_model.schedule_engine import build_op_graph


DEFAULT_ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"
SMALL_OP_TYPES = {"Shape", "Unsqueeze", "Flatten", "Reshape", "Add", "Mul"}
ROW_UID_OP_INDEX_RE = re.compile(r"::opidx:([0-9]+)(?=::|$)")
TASK_CLASS_ORDER = [
    "Embedding lookup + reduction",
    "Bottom MLP",
    "Feature interaction",
    "Top MLP",
]
DEVICE_COLORS = {"Device_1": "#C46E2D", "Device_2": "#3F72B8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate existing v3 prediction rows into graph tasks and calculate MAPE."
    )
    parser.add_argument("--graph-shape-csv", required=True, type=Path)
    parser.add_argument("--device1-prediction-rows", required=True, type=Path)
    parser.add_argument("--device2-prediction-rows", required=True, type=Path)
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--task-class-plot-title",
        default="DLRM task-class MAPE after graph aggregation",
        help="Title for the task-class MAPE bar chart.",
    )
    return parser.parse_args()


def parse_op_index(row_uid: object) -> int:
    match = ROW_UID_OP_INDEX_RE.search(str(row_uid))
    if match is None:
        raise ValueError(f"cannot parse op_idx from row_uid={row_uid!r}")
    return int(match.group(1))


def select_task_host(node_idx: int, observed_nodes: set[int], graph: dict[int, object]) -> int:
    """Return a deterministic adjacent data-bearing node for a small operator."""
    node = graph[node_idx]
    predecessors = [
        idx
        for idx in node.predecessors
        if idx in observed_nodes and graph[idx].op_type not in SMALL_OP_TYPES
    ]
    if predecessors:
        # The latest direct predecessor is the closest producer in topological order.
        return max(predecessors)

    successors = [
        idx
        for idx in node.successors
        if idx in observed_nodes and graph[idx].op_type not in SMALL_OP_TYPES
    ]
    if successors:
        # A small node without a data-bearing predecessor belongs to its first consumer.
        return min(successors)
    return node_idx


def read_prediction_rows(path: Path, device: str, route: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"row_uid", "case_id", "source_name", "combo", "node_name", "op_type", "route", "actual_us", "pred_us"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame = frame.loc[frame["route"].eq(route)].copy()
    if frame.empty:
        raise ValueError(f"{path} has no rows for route={route}")
    frame["device"] = device
    frame["op_idx"] = frame["row_uid"].map(parse_op_index)
    for column in ("actual_us", "pred_us"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        valid = np.isfinite(frame[column]) & (frame[column] > 0 if column == "actual_us" else frame[column] >= 0)
        if not valid.all():
            raise ValueError(f"{path} contains invalid {column} values")
    keys = ["case_id", "source_name", "combo", "op_idx"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{path} contains duplicate operator rows within a workload combination")
    return frame


def assign_graph_tasks(frame: pd.DataFrame, graph: dict[int, object]) -> pd.DataFrame:
    unknown_nodes = sorted(set(frame["op_idx"].astype(int)).difference(graph))
    if unknown_nodes:
        raise ValueError(f"prediction rows contain node indices absent from graph: {unknown_nodes}")
    expected_node_names = frame["op_idx"].map(lambda value: str(graph[int(value)].node_name))
    expected_op_types = frame["op_idx"].map(lambda value: str(graph[int(value)].op_type))
    node_name_mismatch = frame["node_name"].astype(str).ne(expected_node_names)
    op_type_mismatch = frame["op_type"].astype(str).ne(expected_op_types)
    if bool(node_name_mismatch.any()) or bool(op_type_mismatch.any()):
        mismatch_count = int((node_name_mismatch | op_type_mismatch).sum())
        raise ValueError(
            f"prediction rows do not match graph_shape_csv for {mismatch_count} nodes; "
            "use the op-shape CSV that belongs to this DLRM graph"
        )

    assigned_parts: list[pd.DataFrame] = []
    group_columns = ["case_id", "source_name", "combo"]
    for _, combo_frame in frame.groupby(group_columns, sort=False, dropna=False):
        work = combo_frame.copy()
        observed_nodes = set(work["op_idx"].astype(int))
        task_hosts: list[int] = []
        attached_flags: list[bool] = []
        for row in work.itertuples(index=False):
            node_idx = int(row.op_idx)
            is_small = str(row.op_type) in SMALL_OP_TYPES
            task_host = select_task_host(node_idx, observed_nodes, graph) if is_small else node_idx
            task_hosts.append(task_host)
            attached_flags.append(bool(is_small and task_host != node_idx))
        work["graph_task_host_op_idx"] = task_hosts
        work["small_op_attached"] = attached_flags
        assigned_parts.append(work)
    return pd.concat(assigned_parts, ignore_index=True)


def aggregate_tasks(assigned: pd.DataFrame, graph: dict[int, object]) -> pd.DataFrame:
    group_columns = ["device", "case_id", "source_name", "combo", "graph_task_host_op_idx"]
    task_rows = (
        assigned.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            actual_us=("actual_us", "sum"),
            pred_us=("pred_us", "sum"),
            member_rows=("row_uid", "count"),
            attached_small_rows=("small_op_attached", "sum"),
        )
        .copy()
    )
    task_rows["host_op_type"] = task_rows["graph_task_host_op_idx"].map(
        lambda value: str(graph[int(value)].op_type)
    )
    task_rows["host_node_name"] = task_rows["graph_task_host_op_idx"].map(
        lambda value: str(graph[int(value)].node_name)
    )
    task_rows["ape"] = (task_rows["pred_us"] - task_rows["actual_us"]).abs() / task_rows["actual_us"].clip(lower=1e-12)
    return task_rows.sort_values(group_columns, kind="stable").reset_index(drop=True)


def task_class_for_node(node_idx: int, graph: dict[int, object]) -> str:
    """Map a DLRM graph node to one contiguous execution-stage class.

    The mapping uses stable node-name prefixes from the exported DLRM graph,
    rather than node indices.  It therefore stays valid when the graph keeps
    the same DLRM subgraph structure but node indices change after re-export.
    """
    node = graph[int(node_idx)]
    node_name = str(node.node_name)
    if node_name.startswith("/emb_l"):
        return "Embedding lookup + reduction"
    if node_name.startswith("/bot_l/"):
        return "Bottom MLP"
    if node_name.startswith("/top_l/"):
        return "Top MLP"
    return "Feature interaction"


def aggregate_task_classes(assigned: pd.DataFrame, graph: dict[int, object]) -> pd.DataFrame:
    """Sum prediction rows into DLRM execution stages before calculating APE.

    A row retains its original measured and predicted latency.  The function
    only changes the reporting unit: all rows in one static execution stage
    and one workload combination are summed before absolute percentage error
    is calculated.  This avoids treating a shape/control node as an isolated
    user-visible task when it is part of an adjacent DLRM execution stage.
    """
    work = assigned.copy()
    work["task_class"] = work["op_idx"].map(lambda value: task_class_for_node(int(value), graph))
    group_columns = ["device", "case_id", "source_name", "combo", "task_class"]
    task_rows = (
        work.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            actual_us=("actual_us", "sum"),
            pred_us=("pred_us", "sum"),
            member_rows=("row_uid", "count"),
            node_indices=("op_idx", lambda values: ",".join(map(str, sorted(set(values))))),
        )
        .copy()
    )
    task_rows["ape"] = (
        (task_rows["pred_us"] - task_rows["actual_us"]).abs()
        / task_rows["actual_us"].clip(lower=1e-12)
    )
    order = {name: index for index, name in enumerate(TASK_CLASS_ORDER)}
    task_rows["task_class_order"] = task_rows["task_class"].map(order).fillna(len(order)).astype(int)
    return task_rows.sort_values(group_columns, kind="stable").reset_index(drop=True)


def summarize_task_classes(task_rows: pd.DataFrame) -> pd.DataFrame:
    """Return unweighted and latency-weighted MAPE for each execution stage."""
    rows: list[dict[str, object]] = []
    for task_class in TASK_CLASS_ORDER:
        for device in ("Device_1", "Device_2"):
            subset = task_rows.loc[
                task_rows["task_class"].eq(task_class) & task_rows["device"].eq(device)
            ].copy()
            if subset.empty:
                continue
            actual_sum = float(subset["actual_us"].sum())
            weighted_mape = (
                float((subset["pred_us"] - subset["actual_us"]).abs().sum()) / actual_sum * 100.0
                if actual_sum > 0.0
                else float("nan")
            )
            mape = float(subset["ape"].mean() * 100.0)
            rows.append(
                {
                    "task_class": task_class,
                    "device": device,
                    "rows": int(len(subset)),
                    "mape_pct": mape,
                    "weighted_mape_pct": weighted_mape,
                    "actual_us_mean": float(subset["actual_us"].mean()),
                    "actual_us_min": float(subset["actual_us"].min()),
                    "mape_below_15pct": bool(mape < 15.0),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order = {name: index for index, name in enumerate(TASK_CLASS_ORDER)}
    return (
        summary.assign(task_class_order=summary["task_class"].map(order))
        .sort_values(["task_class_order", "device"], kind="stable")
        .drop(columns="task_class_order")
        .reset_index(drop=True)
    )


def plot_task_class_mape(summary: pd.DataFrame, out_dir: Path, *, title: str) -> None:
    """Plot task-class MAPE and show the 15% acceptance threshold explicitly."""
    pivot = (
        summary.pivot(index="task_class", columns="device", values="mape_pct")
        .reindex(TASK_CLASS_ORDER)
        .dropna(how="all")
    )
    if pivot.empty:
        raise ValueError("task-class MAPE summary is empty")

    x = list(range(len(pivot.index)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    max_value = float(pivot.to_numpy(dtype=float).max())
    label_offset = max(0.28, max_value * 0.025)
    for offset, device in ((-width / 2.0, "Device_1"), (width / 2.0, "Device_2")):
        values = pivot.get(device, pd.Series(index=pivot.index, dtype=float)).to_numpy(dtype=float)
        bars = ax.bar(
            [value + offset for value in x],
            values,
            width,
            label=device,
            color=DEVICE_COLORS[device],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values, strict=True):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    float(value) + label_offset,
                    f"{float(value):.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.axhline(15.0, color="#9B2C2C", linewidth=1.2, linestyle="--", label="15% threshold")
    ax.set_title(str(title), fontsize=15)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("lookup +", "lookup\n+") for name in pivot.index], fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, max(17.0, 15.0 + label_offset * 3.0, max_value * 1.25))
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "dlrm_task_class_mape_bar.png", dpi=300)
    fig.savefig(out_dir / "dlrm_task_class_mape_bar.pdf")
    plt.close(fig)


def summarize(device: str, assigned: pd.DataFrame, task_rows: pd.DataFrame, route: str, graph_shape_csv: Path) -> dict[str, object]:
    input_rows = int(len(assigned))
    small_rows = assigned["op_type"].isin(SMALL_OP_TYPES)
    attached_rows = int(assigned["small_op_attached"].sum())
    standalone_rows = int(small_rows.sum()) - attached_rows
    baseline_mape = float(
        ((assigned["pred_us"] - assigned["actual_us"]).abs() / assigned["actual_us"].clip(lower=1e-12)).mean()
        * 100.0
    )
    mape = float(task_rows["ape"].mean() * 100.0)
    return {
        "device": device,
        "route": route,
        "input_rows": input_rows,
        "graph_task_rows": int(len(task_rows)),
        "small_operator_rows": int(small_rows.sum()),
        "small_operator_rows_attached": attached_rows,
        "small_operator_rows_standalone": standalone_rows,
        "baseline_row_mape_pct": baseline_mape,
        "mape_pct": mape,
        "accuracy_pct": 100.0 - mape,
        "mape_reduction_pct_points": baseline_mape - mape,
        "graph_shape_csv": str(graph_shape_csv.resolve()),
        "attachment_rule": "direct_non_small_predecessor_max_else_direct_non_small_successor_min",
    }


def main() -> None:
    args = parse_args()
    if not args.graph_shape_csv.exists():
        raise FileNotFoundError(args.graph_shape_csv)
    graph = build_op_graph(pd.read_csv(args.graph_shape_csv))

    input_specs = [
        ("Device_1", args.device1_prediction_rows),
        ("Device_2", args.device2_prediction_rows),
    ]
    assigned_frames: list[pd.DataFrame] = []
    task_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for device, input_path in input_specs:
        prediction_rows = read_prediction_rows(input_path, device, args.route)
        assigned = assign_graph_tasks(prediction_rows, graph)
        task_rows = aggregate_tasks(assigned, graph)
        assigned_frames.append(assigned)
        task_frames.append(task_rows)
        summary_rows.append(summarize(device, assigned, task_rows, args.route, args.graph_shape_csv))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    assignments = pd.concat(assigned_frames, ignore_index=True)
    task_predictions = pd.concat(task_frames, ignore_index=True)
    task_class_predictions = aggregate_task_classes(assignments, graph)
    task_class_summary = summarize_task_classes(task_class_predictions)
    if task_class_summary.empty:
        raise ValueError("no task-class rows were produced")
    if not bool(task_class_summary["mape_below_15pct"].all()):
        failed = task_class_summary.loc[
            ~task_class_summary["mape_below_15pct"], ["task_class", "device", "mape_pct"]
        ]
        print(
            "WARNING: task-class MAPE is at or above 15%: "
            + repr(failed.to_dict(orient="records")), file=sys.stderr,
        )
    summary.to_csv(args.out_dir / "graph_task_mape_summary.csv", index=False)
    assignments.to_csv(args.out_dir / "graph_task_assignment_rows.csv", index=False)
    task_predictions.to_csv(args.out_dir / "graph_task_prediction_rows.csv", index=False)
    task_class_predictions.to_csv(args.out_dir / "dlrm_task_class_prediction_rows.csv", index=False)
    task_class_summary.to_csv(args.out_dir / "dlrm_task_class_mape_summary.csv", index=False)
    (args.out_dir / "dlrm_task_class_mape_method.json").write_text(
        json.dumps(
            {
                "route": str(args.route),
                "task_class_plot_title": str(args.task_class_plot_title),
                "metric": "mean(abs(sum(pred_us) - sum(actual_us)) / sum(actual_us)) * 100",
                "threshold_pct": 15.0,
                "task_classes": {
                    "Embedding lookup + reduction": "node_name starts with /emb_l",
                    "Bottom MLP": "node_name starts with /bot_l/",
                    "Feature interaction": "observed non-embedding, non-bottom, non-top nodes",
                    "Top MLP": "node_name starts with /top_l/",
                },
                "inputs": {
                    "graph_shape_csv": str(args.graph_shape_csv.resolve()),
                    "device1_prediction_rows": str(args.device1_prediction_rows.resolve()),
                    "device2_prediction_rows": str(args.device2_prediction_rows.resolve()),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_task_class_mape(
        task_class_summary,
        args.out_dir,
        title=str(args.task_class_plot_title),
    )

    print(summary.to_csv(index=False), end="")
    print(task_class_summary.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
