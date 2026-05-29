from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from .graph_contract import (
    BranchTask,
    ComboSpec,
    CoverageSummary,
    EmbeddingOrderObservation,
    NodeSpan,
    OpNode,
    ScheduleResult,
    ScheduledTaskSpan,
    TimingSummary,
)

BRANCH_COUNT = 8
BRANCH_FIFO_ORDER = tuple(range(BRANCH_COUNT))
EMBEDDING_OP_ORDER = {"Gather": 0, "Reshape": 1, "ReduceSum": 2}
BRANCH_NAME_RE = re.compile(r"^/emb_l(?P<branch_idx>\d+)/")


@dataclass(frozen=True)
class _TaskDef:
    task_id: str
    task_kind: str
    partition: str
    predecessors: tuple[str, ...]
    node_indices: tuple[int, ...]
    node_names: tuple[str, ...]
    branch_idx: int | None
    duration_us: float
    sort_key: tuple[int, int]


def parse_branch_idx(node_name: str) -> int | None:
    match = BRANCH_NAME_RE.match(node_name)
    if not match:
        return None
    return int(match.group("branch_idx"))


def normalize_timeline_node_name(node_name: str) -> str:
    return node_name[:-12] if node_name.endswith("_kernel_time") else node_name


def _partition_for_node(node_name: str, op_type: str) -> str:
    if op_type == "Constant":
        return "constant"
    if node_name.startswith("/bot_l/"):
        return "bottom"
    branch_idx = parse_branch_idx(node_name)
    if branch_idx is not None and op_type in EMBEDDING_OP_ORDER:
        return "embedding"
    return "tail"


def _parse_producer_idx(producer_node: str) -> int | None:
    if producer_node in {"initializer", "graph_input"}:
        return None
    if pd.isna(producer_node):
        return None
    head = str(producer_node).split(":", 1)[0]
    return int(head) if head.isdigit() else None


def build_op_graph(op_shapes_df: pd.DataFrame) -> dict[int, OpNode]:
    unique_nodes = (
        op_shapes_df[["node_idx", "node_name", "op_type"]]
        .drop_duplicates()
        .sort_values("node_idx")
    )
    op_type_by_idx = dict(zip(unique_nodes["node_idx"], unique_nodes["op_type"]))

    predecessors: dict[int, set[int]] = defaultdict(set)
    successors: dict[int, set[int]] = defaultdict(set)

    input_rows = op_shapes_df[op_shapes_df["tensor_role"] == "input"]
    for row in input_rows.itertuples(index=False):
        pred_idx = _parse_producer_idx(row.producer_node)
        if pred_idx is None:
            continue
        if op_type_by_idx.get(pred_idx) == "Constant":
            continue
        predecessors[row.node_idx].add(pred_idx)
        successors[pred_idx].add(row.node_idx)

    graph: dict[int, OpNode] = {}
    for row in unique_nodes.itertuples(index=False):
        branch_idx = parse_branch_idx(row.node_name)
        graph[row.node_idx] = OpNode(
            node_idx=int(row.node_idx),
            node_name=row.node_name,
            op_type=row.op_type,
            predecessors=tuple(sorted(predecessors.get(row.node_idx, set()))),
            successors=tuple(sorted(successors.get(row.node_idx, set()))),
            partition=_partition_for_node(row.node_name, row.op_type),
            branch_idx=branch_idx,
        )
    return graph


def classify_coverage(graph: dict[int, OpNode], observed_node_indices: set[int]) -> CoverageSummary:
    expected = tuple(sorted(node_idx for node_idx, node in graph.items() if node.op_type != "Constant"))
    observed = tuple(sorted(node_idx for node_idx in observed_node_indices if node_idx in graph and graph[node_idx].op_type != "Constant"))
    missing = tuple(sorted(set(expected) - set(observed)))
    expected_count = len(expected)
    observed_count = len(observed)
    missing_count = len(missing)
    coverage_ratio = observed_count / expected_count if expected_count else 1.0
    return CoverageSummary(
        expected_node_indices=expected,
        observed_node_indices=observed,
        missing_node_indices=missing,
        expected_count=expected_count,
        observed_count=observed_count,
        missing_count=missing_count,
        coverage_ratio=coverage_ratio,
        is_full_graph=(missing_count == 0),
    )


def _build_branch_tasks(graph: dict[int, OpNode]) -> tuple[tuple[BranchTask, ...], dict[int, str]]:
    branch_tasks: list[BranchTask] = []
    node_to_task: dict[int, str] = {}

    for branch_idx in BRANCH_FIFO_ORDER:
        member_nodes = sorted(
            (
                node_idx
                for node_idx, node in graph.items()
                if node.partition == "embedding" and node.branch_idx == branch_idx
            ),
            key=lambda node_idx: (EMBEDDING_OP_ORDER[graph[node_idx].op_type], node_idx),
        )
        if not member_nodes:
            continue
        member_set = set(member_nodes)
        predecessor_node_indices = sorted(
            {
                pred
                for node_idx in member_nodes
                for pred in graph[node_idx].predecessors
                if pred not in member_set and graph[pred].op_type != "Constant"
            }
        )
        successor_node_indices = sorted(
            {
                succ
                for node_idx in member_nodes
                for succ in graph[node_idx].successors
                if succ not in member_set and graph[succ].op_type != "Constant"
            }
        )
        branch_task = BranchTask(
            branch_idx=branch_idx,
            node_indices=tuple(member_nodes),
            node_names=tuple(graph[node_idx].node_name for node_idx in member_nodes),
            predecessor_node_indices=tuple(predecessor_node_indices),
            successor_node_indices=tuple(successor_node_indices),
        )
        branch_tasks.append(branch_task)
        task_id = f"branch:{branch_idx}"
        for node_idx in member_nodes:
            node_to_task[node_idx] = task_id

    return tuple(branch_tasks), node_to_task


def _build_task_defs(
    graph: dict[int, OpNode],
    coverage: CoverageSummary,
    branch_tasks: tuple[BranchTask, ...],
    node_to_task: dict[int, str],
    duration_by_node: dict[int, float],
) -> dict[str, _TaskDef]:
    task_defs: dict[str, _TaskDef] = {}
    bottom_task_ids: list[str] = []

    for node_idx, node in sorted(graph.items()):
        if node.op_type == "Constant" or node.partition == "embedding":
            continue
        node_to_task[node_idx] = f"op:{node_idx}"

    for branch_task in branch_tasks:
        preds = sorted({node_to_task[pred] for pred in branch_task.predecessor_node_indices})
        duration = float(sum(duration_by_node.get(node_idx, 0.0) for node_idx in branch_task.node_indices))
        task_id = f"branch:{branch_task.branch_idx}"
        task_defs[task_id] = _TaskDef(
            task_id=task_id,
            task_kind="branch",
            partition="embedding",
            predecessors=tuple(preds),
            node_indices=branch_task.node_indices,
            node_names=branch_task.node_names,
            branch_idx=branch_task.branch_idx,
            duration_us=duration,
            sort_key=(1, branch_task.branch_idx),
        )

    for node_idx, node in sorted(graph.items()):
        if node.op_type == "Constant" or node.partition == "embedding":
            continue
        task_id = node_to_task[node_idx]
        predecessors = sorted(
            {
                node_to_task[pred]
                for pred in node.predecessors
                if pred in node_to_task and graph[pred].op_type != "Constant"
            }
        )
        partition_rank = 0 if node.partition == "bottom" else 3
        task_defs[task_id] = _TaskDef(
            task_id=task_id,
            task_kind="op",
            partition=node.partition,
            predecessors=tuple(predecessors),
            node_indices=(node_idx,),
            node_names=(node.node_name,),
            branch_idx=None,
            duration_us=float(duration_by_node.get(node_idx, 0.0)),
            sort_key=(partition_rank, node_idx),
        )
        if node.partition == "bottom":
            bottom_task_ids.append(task_id)

    branch_task_ids = [task_id for task_id, task in task_defs.items() if task.task_kind == "branch"]
    barrier_preds = tuple(sorted(set(bottom_task_ids + branch_task_ids)))
    task_defs["barrier:tail"] = _TaskDef(
        task_id="barrier:tail",
        task_kind="barrier",
        partition="barrier",
        predecessors=barrier_preds,
        node_indices=(),
        node_names=(),
        branch_idx=None,
        duration_us=0.0,
        sort_key=(2, 0),
    )

    for task_id, task in list(task_defs.items()):
        if task.partition != "tail":
            continue
        predecessors = tuple(sorted(set(task.predecessors + ("barrier:tail",))))
        task_defs[task_id] = _TaskDef(
            task_id=task.task_id,
            task_kind=task.task_kind,
            partition=task.partition,
            predecessors=predecessors,
            node_indices=task.node_indices,
            node_names=task.node_names,
            branch_idx=task.branch_idx,
            duration_us=task.duration_us,
            sort_key=task.sort_key,
        )

    return task_defs


def _topological_order(task_defs: dict[str, _TaskDef]) -> list[str]:
    indegree = {task_id: len(task.predecessors) for task_id, task in task_defs.items()}
    successors: dict[str, list[str]] = defaultdict(list)
    for task_id, task in task_defs.items():
        for pred in task.predecessors:
            successors[pred].append(task_id)

    ready_heap = [
        (task_defs[task_id].sort_key, task_id)
        for task_id, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready_heap)

    order: list[str] = []
    while ready_heap:
        _, task_id = heapq.heappop(ready_heap)
        order.append(task_id)
        for succ in successors.get(task_id, []):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(ready_heap, (task_defs[succ].sort_key, succ))

    if len(order) != len(task_defs):
        raise ValueError("Task graph contains a cycle")
    return order


def _span_for_nodes(node_spans: dict[int, NodeSpan], node_indices: tuple[int, ...]) -> float:
    if not node_indices:
        return 0.0
    starts = [node_spans[node_idx].start_us for node_idx in node_indices if node_idx in node_spans]
    ends = [node_spans[node_idx].end_us for node_idx in node_indices if node_idx in node_spans]
    if not starts or not ends:
        return 0.0
    return float(max(ends) - min(starts))


def schedule_combo(
    combo_spec: ComboSpec,
    graph: dict[int, OpNode],
    prediction_rows: pd.DataFrame,
) -> ScheduleResult:
    observed_node_indices = set(prediction_rows["op_idx"].astype(int).tolist())
    coverage = classify_coverage(graph, observed_node_indices)
    duration_by_node = {
        int(row.op_idx): float(row.pred_us)
        for row in prediction_rows.itertuples(index=False)
    }

    branch_tasks, node_to_task = _build_branch_tasks(graph)
    task_defs = _build_task_defs(graph, coverage, branch_tasks, node_to_task, duration_by_node)
    topo_order = _topological_order(task_defs)

    slot_count = max(1, int(combo_spec.inter_threads))
    branch_heap: list[tuple[float, int]] = []
    task_spans: list[ScheduledTaskSpan] = []
    node_spans: dict[int, NodeSpan] = {}
    task_end_us: dict[str, float] = {}

    for task_id in topo_order:
        task = task_defs[task_id]
        ready_time = max((task_end_us[pred] for pred in task.predecessors), default=0.0)
        if task.task_kind == "branch":
            if len(branch_heap) < slot_count:
                start_us = ready_time
            else:
                freed_us, _ = heapq.heappop(branch_heap)
                start_us = max(ready_time, freed_us)
        else:
            start_us = ready_time
        end_us = start_us + task.duration_us

        if task.task_kind == "branch":
            heapq.heappush(branch_heap, (end_us, task.branch_idx or 0))
            cursor = start_us
            for node_idx in task.node_indices:
                node = graph[node_idx]
                node_duration = float(duration_by_node.get(node_idx, 0.0))
                node_start = cursor
                node_end = node_start + node_duration
                node_spans[node_idx] = NodeSpan(
                    node_idx=node_idx,
                    node_name=node.node_name,
                    start_us=node_start,
                    end_us=node_end,
                    duration_us=node_duration,
                    task_id=task_id,
                )
                cursor = node_end
        elif task.task_kind == "op":
            node_idx = task.node_indices[0]
            node = graph[node_idx]
            node_spans[node_idx] = NodeSpan(
                node_idx=node_idx,
                node_name=node.node_name,
                start_us=start_us,
                end_us=end_us,
                duration_us=task.duration_us,
                task_id=task_id,
            )

        task_spans.append(
            ScheduledTaskSpan(
                task_id=task_id,
                task_kind=task.task_kind,
                partition=task.partition,
                start_us=start_us,
                end_us=end_us,
                duration_us=task.duration_us,
                node_indices=task.node_indices,
                branch_idx=task.branch_idx,
            )
        )
        task_end_us[task_id] = end_us

    task_span_lookup = {span.task_id: span for span in task_spans}
    bottom_end_us = max(
        (span.end_us for span in task_spans if span.partition == "bottom"),
        default=0.0,
    )
    branch_pool_end_us = max(
        (span.end_us for span in task_spans if span.task_kind == "branch"),
        default=0.0,
    )
    tail_barrier_us = task_span_lookup["barrier:tail"].end_us

    return ScheduleResult(
        combo_spec=combo_spec,
        coverage=coverage,
        branch_tasks=branch_tasks,
        task_spans=tuple(sorted(task_spans, key=lambda span: (span.start_us, span.task_id))),
        node_spans=tuple(sorted(node_spans.values(), key=lambda span: span.node_idx)),
        expected_node_indices=coverage.expected_node_indices,
        observed_node_indices=coverage.observed_node_indices,
        predicted_full_graph_us=_span_for_nodes(node_spans, coverage.expected_node_indices),
        predicted_observed_subgraph_us=_span_for_nodes(node_spans, coverage.observed_node_indices),
        bottom_end_us=bottom_end_us,
        branch_pool_end_us=branch_pool_end_us,
        tail_barrier_us=tail_barrier_us,
    )


def compute_mean_batch_span(
    timeline_df: pd.DataFrame,
    node_names: list[str] | tuple[str, ...] | set[str],
) -> TimingSummary:
    node_name_set = set(node_names)
    if not node_name_set:
        raise ValueError("node_names must not be empty")

    node_rows = timeline_df[timeline_df["normalized_node_name"].isin(node_name_set)].copy()
    if node_rows.empty:
        raise ValueError("Timeline does not contain any requested nodes")

    grouped = (
        node_rows.groupby(["batch_idx", "normalized_node_name"], as_index=False)
        .agg(start_us=("start_us", "min"), end_us=("end_us", "max"))
    )
    spans: list[tuple[int, float]] = []
    incomplete_batches: list[int] = []

    for batch_idx, batch_df in grouped.groupby("batch_idx"):
        present = set(batch_df["normalized_node_name"])
        if present != node_name_set:
            incomplete_batches.append(int(batch_idx))
            continue
        span_us = float(batch_df["end_us"].max() - batch_df["start_us"].min())
        spans.append((int(batch_idx), span_us))

    if not spans:
        raise ValueError("No complete batches were available for the requested nodes")

    spans.sort(key=lambda item: item[0])
    dropped_batch_indices: tuple[int, ...] = ()
    kept_spans = spans
    if len(spans) > 1:
        dropped_batch_indices = (spans[0][0],)
        kept_spans = spans[1:]
    kept_batch_indices = tuple(batch_idx for batch_idx, _ in kept_spans)
    mean_span_us = float(sum(span for _, span in kept_spans) / len(kept_spans))

    return TimingSummary(
        mean_span_us=mean_span_us,
        kept_batch_indices=kept_batch_indices,
        dropped_batch_indices=dropped_batch_indices,
        incomplete_batch_indices=tuple(sorted(incomplete_batches)),
        per_batch_spans=tuple(kept_spans),
    )


def _max_overlap(intervals: list[tuple[float, float]]) -> int:
    events: list[tuple[float, int]] = []
    for start_us, end_us in intervals:
        events.append((start_us, 1))
        events.append((end_us, -1))
    events.sort(key=lambda item: (item[0], 0 if item[1] < 0 else 1))

    active = 0
    max_active = 0
    for _, delta in events:
        active += delta
        max_active = max(max_active, active)
    return max_active


def _representative_value(values: list[tuple[int, ...]] | list[int]) -> tuple[int, ...] | int:
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def analyze_embedding_execution(
    timeline_df: pd.DataFrame,
    inter_threads: int,
) -> EmbeddingOrderObservation:
    gather_rows = timeline_df[
        timeline_df["normalized_node_name"].str.match(r"^/emb_l\d+/Gather$")
    ].copy()
    reduce_rows = timeline_df[
        timeline_df["normalized_node_name"].str.match(r"^/emb_l\d+/ReduceSum$")
    ].copy()

    batch_infos: list[dict[str, object]] = []
    for batch_idx in sorted(set(gather_rows["batch_idx"]).intersection(reduce_rows["batch_idx"])):
        batch_gathers = gather_rows[gather_rows["batch_idx"] == batch_idx].copy()
        batch_reduces = reduce_rows[reduce_rows["batch_idx"] == batch_idx].copy()
        if len(batch_gathers) < BRANCH_COUNT or len(batch_reduces) < BRANCH_COUNT:
            continue

        batch_gathers["branch_idx"] = batch_gathers["normalized_node_name"].map(parse_branch_idx)
        batch_reduces["branch_idx"] = batch_reduces["normalized_node_name"].map(parse_branch_idx)
        gather_info = (
            batch_gathers.groupby("branch_idx", as_index=False)
            .agg(start_us=("start_us", "min"), end_us=("end_us", "max"))
            .sort_values("start_us")
        )
        reduce_info = (
            batch_reduces.groupby("branch_idx", as_index=False)
            .agg(end_us=("end_us", "max"))
            .sort_values("branch_idx")
        )
        if len(gather_info) != BRANCH_COUNT or len(reduce_info) != BRANCH_COUNT:
            continue

        order = tuple(int(branch_idx) for branch_idx in gather_info["branch_idx"].tolist())
        intervals = list(zip(gather_info["start_us"], gather_info["end_us"]))
        max_concurrency = _max_overlap(intervals)

        gather_start_by_branch = {
            int(row.branch_idx): float(row.start_us)
            for row in gather_info.itertuples(index=False)
        }
        reduce_end_by_branch = {
            int(row.branch_idx): float(row.end_us)
            for row in reduce_info.itertuples(index=False)
        }

        handoff_gaps: list[float] = []
        active_heap: list[tuple[float, int]] = []
        for branch_idx in order[:inter_threads]:
            heapq.heappush(active_heap, (reduce_end_by_branch[branch_idx], branch_idx))
        for branch_idx in order[inter_threads:]:
            freed_end_us, _ = heapq.heappop(active_heap)
            handoff_gaps.append(float(gather_start_by_branch[branch_idx] - freed_end_us))
            heapq.heappush(active_heap, (reduce_end_by_branch[branch_idx], branch_idx))

        tail_rows = timeline_df[
            (timeline_df["batch_idx"] == batch_idx) & (timeline_df["task_name"] == "tail")
        ]
        tail_start_gap = None
        if not tail_rows.empty:
            tail_start_gap = float(tail_rows["start_us"].min() - max(reduce_end_by_branch.values()))

        batch_infos.append(
            {
                "batch_idx": int(batch_idx),
                "order": order,
                "max_concurrency": int(max_concurrency),
                "handoff_gaps": tuple(handoff_gaps),
                "tail_start_gap": tail_start_gap,
            }
        )

    if not batch_infos:
        raise ValueError("No complete embedding batches found in timeline")

    batch_infos.sort(key=lambda item: int(item["batch_idx"]))
    kept_infos = batch_infos[1:] if len(batch_infos) > 1 else batch_infos
    orders = [info["order"] for info in kept_infos]
    concurrencies = [int(info["max_concurrency"]) for info in kept_infos]
    flat_handoff_gaps = [
        float(gap)
        for info in kept_infos
        for gap in info["handoff_gaps"]
    ]
    tail_gaps = [
        float(info["tail_start_gap"])
        for info in kept_infos
        if info["tail_start_gap"] is not None
    ]

    return EmbeddingOrderObservation(
        kept_batch_count=len(kept_infos),
        representative_launch_order=_representative_value(orders),  # type: ignore[arg-type]
        all_kept_batches_match_fifo=all(order == BRANCH_FIFO_ORDER for order in orders),
        representative_max_gather_concurrency=int(_representative_value(concurrencies)),  # type: ignore[arg-type]
        all_kept_batches_match_inter_threads=all(value == int(inter_threads) for value in concurrencies),
        handoff_gap_mean_us=(float(sum(flat_handoff_gaps) / len(flat_handoff_gaps)) if flat_handoff_gaps else None),
        handoff_gap_min_us=(float(min(flat_handoff_gaps)) if flat_handoff_gaps else None),
        handoff_gap_max_us=(float(max(flat_handoff_gaps)) if flat_handoff_gaps else None),
        tail_start_gap_mean_us=(float(sum(tail_gaps) / len(tail_gaps)) if tail_gaps else None),
        tail_start_gap_min_us=(float(min(tail_gaps)) if tail_gaps else None),
        tail_start_gap_max_us=(float(max(tail_gaps)) if tail_gaps else None),
    )
