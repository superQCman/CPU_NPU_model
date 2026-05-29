from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComboArtifactPaths:
    case_id: str
    combo: str
    shape_csv: Path
    profile_dir: Path
    timeline_csv: Path


@dataclass(frozen=True)
class ComboSpec:
    case_id: str
    combo: str
    batch_size: int
    num_indices_per_lookup: int
    inter_threads: int
    artifact_paths: ComboArtifactPaths


@dataclass(frozen=True)
class OpNode:
    node_idx: int
    node_name: str
    op_type: str
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]
    partition: str
    branch_idx: int | None = None


@dataclass(frozen=True)
class BranchTask:
    branch_idx: int
    node_indices: tuple[int, ...]
    node_names: tuple[str, ...]
    predecessor_node_indices: tuple[int, ...]
    successor_node_indices: tuple[int, ...]


@dataclass(frozen=True)
class CoverageSummary:
    expected_node_indices: tuple[int, ...]
    observed_node_indices: tuple[int, ...]
    missing_node_indices: tuple[int, ...]
    expected_count: int
    observed_count: int
    missing_count: int
    coverage_ratio: float
    is_full_graph: bool


@dataclass(frozen=True)
class NodeSpan:
    node_idx: int
    node_name: str
    start_us: float
    end_us: float
    duration_us: float
    task_id: str


@dataclass(frozen=True)
class ScheduledTaskSpan:
    task_id: str
    task_kind: str
    partition: str
    start_us: float
    end_us: float
    duration_us: float
    node_indices: tuple[int, ...]
    branch_idx: int | None = None


@dataclass(frozen=True)
class TimingSummary:
    mean_span_us: float
    kept_batch_indices: tuple[int, ...]
    dropped_batch_indices: tuple[int, ...]
    incomplete_batch_indices: tuple[int, ...]
    per_batch_spans: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class EmbeddingOrderObservation:
    kept_batch_count: int
    representative_launch_order: tuple[int, ...]
    all_kept_batches_match_fifo: bool
    representative_max_gather_concurrency: int
    all_kept_batches_match_inter_threads: bool
    handoff_gap_mean_us: float | None
    handoff_gap_min_us: float | None
    handoff_gap_max_us: float | None
    tail_start_gap_mean_us: float | None
    tail_start_gap_min_us: float | None
    tail_start_gap_max_us: float | None


@dataclass(frozen=True)
class ScheduleResult:
    combo_spec: ComboSpec
    coverage: CoverageSummary
    branch_tasks: tuple[BranchTask, ...]
    task_spans: tuple[ScheduledTaskSpan, ...]
    node_spans: tuple[NodeSpan, ...]
    expected_node_indices: tuple[int, ...]
    observed_node_indices: tuple[int, ...]
    predicted_full_graph_us: float
    predicted_observed_subgraph_us: float
    bottom_end_us: float
    branch_pool_end_us: float
    tail_barrier_us: float

    def task_span_map(self) -> dict[str, ScheduledTaskSpan]:
        return {span.task_id: span for span in self.task_spans}

    def node_span_map(self) -> dict[int, NodeSpan]:
        return {span.node_idx: span for span in self.node_spans}
