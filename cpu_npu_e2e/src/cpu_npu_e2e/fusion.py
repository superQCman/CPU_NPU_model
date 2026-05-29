from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from .alltoall_adapter import AllToAllAdapter
from .cpu_adapter import CpuLatencyAdapter, run_cpu_static_model_inference
from .npu_adapter import NpuLatencyAdapter
from .routing import RoutingTable
from .shapes import GraphNode, build_graph, load_op_shapes, tensor_bytes


@dataclass(frozen=True)
class ScheduledNode:
    node_idx: int
    start_us: float
    end_us: float


@dataclass(frozen=True)
class FusionCaseSpec:
    name: str
    shape_csv: Path


COMBO_RE = re.compile(r"bs(?P<batch_size>\d+)_nip(?P<nip>\d+)")


def discover_shape_csv(cpu_model_root: Path) -> Path:
    try:
        return next((cpu_model_root / "pipeline_artifacts").glob("**/op_shapes.csv"))
    except StopIteration as exc:
        raise FileNotFoundError(f"no op_shapes.csv found under {cpu_model_root / 'pipeline_artifacts'}") from exc


def load_case_specs(cases_config: Path, cpu_model_root: Path) -> list[FusionCaseSpec]:
    payload = yaml.safe_load(cases_config.read_text(encoding="utf-8")) or {}
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{cases_config} must contain a non-empty 'cases' list")

    specs: list[FusionCaseSpec] = []
    for idx, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            raise ValueError(f"case #{idx} in {cases_config} must be a mapping")
        name = str(case.get("name", "")).strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"case #{idx} has invalid name {name!r}")
        if "shape_csv" not in case:
            raise ValueError(f"case {name!r} is missing shape_csv")
        shape_csv = Path(str(case["shape_csv"]))
        if not shape_csv.is_absolute():
            shape_csv = cpu_model_root / shape_csv
        if not shape_csv.exists():
            raise FileNotFoundError(f"case {name!r} shape_csv not found: {shape_csv}")
        specs.append(FusionCaseSpec(name=name, shape_csv=shape_csv))
    return specs


def parse_combo_from_path(path: Path) -> tuple[str, str, int | None, int | None]:
    combo = path.parent.name
    case_id = path.parent.parent.name if path.parent.parent else "unknown_case"
    match = COMBO_RE.fullmatch(combo)
    if not match:
        return case_id, combo, None, None
    return case_id, combo, int(match.group("batch_size")), int(match.group("nip"))


def predict_nodes(
    graph: dict[int, GraphNode],
    routing: RoutingTable,
    cpu_adapter: CpuLatencyAdapter,
    npu_adapter: NpuLatencyAdapter,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for node_idx, node in sorted(graph.items()):
        route = routing.route_for(node.op_type)
        if route.backend == "IGNORE":
            pred_us = 0.0
            source = "ignored"
        elif route.backend == "CPU":
            pred_us, source = cpu_adapter.predict_us(node)
        elif route.backend == "NPU":
            pred_us, source = npu_adapter.predict_us(node, route.target_op_type)
        else:
            raise ValueError(f"unsupported backend {route.backend!r} for {node.op_type}")
        rows.append(
            {
                "node_idx": node_idx,
                "node_name": node.node_name,
                "op_type": node.op_type,
                "partition": node.partition,
                "branch_idx": node.branch_idx,
                "route_backend": route.backend,
                "target_op_type": route.target_op_type,
                "pred_us": float(pred_us),
                "prediction_source": source,
                "input_bytes": tensor_bytes(node.inputs),
                "output_bytes": tensor_bytes(node.outputs),
            }
        )
    return pd.DataFrame(rows)


def embedding_message_bytes(graph: dict[int, GraphNode]) -> int:
    total = 0
    for node in graph.values():
        if node.partition != "embedding":
            continue
        has_embedding_successor = any(graph[succ].partition == "embedding" for succ in node.successors if succ in graph)
        if not has_embedding_successor:
            total += tensor_bytes(node.outputs)
    return int(total)


def schedule_subset(
    *,
    graph: dict[int, GraphNode],
    pred_by_node: dict[int, float],
    backend_by_node: dict[int, str],
    node_ids: set[int],
    cpu_slots: int,
    npu_slots: int,
    external_ready: Callable[[int], float],
) -> dict[int, ScheduledNode]:
    remaining = set(node_ids)
    end_times: dict[int, float] = {}
    spans: dict[int, ScheduledNode] = {}
    resource_heaps = {
        "CPU": [0.0] * max(1, int(cpu_slots)),
        "NPU": [0.0] * max(1, int(npu_slots)),
        "IGNORE": [0.0],
    }
    for heap in resource_heaps.values():
        heapq.heapify(heap)

    while remaining:
        ready = [
            node_idx
            for node_idx in sorted(remaining)
            if all(pred not in node_ids or pred in end_times for pred in graph[node_idx].predecessors)
        ]
        if not ready:
            raise ValueError("task graph contains a cycle or unresolved predecessor")
        for node_idx in ready:
            node = graph[node_idx]
            dep_ready = 0.0
            for pred in node.predecessors:
                if pred in end_times:
                    dep_ready = max(dep_ready, end_times[pred])
                else:
                    dep_ready = max(dep_ready, external_ready(pred))
            backend = backend_by_node.get(node_idx, "CPU")
            duration = max(float(pred_by_node.get(node_idx, 0.0)), 0.0)
            if backend == "IGNORE" or duration <= 0:
                start = dep_ready
                end = dep_ready
            else:
                heap = resource_heaps.setdefault(backend, [0.0])
                free_at = heapq.heappop(heap)
                start = max(dep_ready, free_at)
                end = start + duration
                heapq.heappush(heap, end)
            spans[node_idx] = ScheduledNode(node_idx=node_idx, start_us=float(start), end_us=float(end))
            end_times[node_idx] = float(end)
            remaining.remove(node_idx)
    return spans


def run_fusion(
    *,
    cpu_model_root: Path,
    npu_root: Path,
    alltoall_root: Path,
    routing_path: Path,
    output_dir: Path,
    shape_csv: Path | None = None,
    cpu_predictions: Path | None = None,
    cpu_model_bundle: Path | None = None,
    cpu_model_num_threads: int | None = None,
    cpu_model_inter_threads: int | None = None,
    cpu_slots: int = 1,
    npu_slots: int = 1,
    allow_cpu_static_fallback: bool = True,
) -> dict[str, object]:
    if cpu_predictions is not None and cpu_model_bundle is not None:
        raise ValueError("cpu_predictions and cpu_model_bundle are mutually exclusive")
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_csv = shape_csv or discover_shape_csv(cpu_model_root)
    op_shapes = load_op_shapes(shape_csv)
    graph = build_graph(op_shapes)
    routing = RoutingTable.from_yaml(routing_path)
    routing.validate_all(node.op_type for node in graph.values())

    cpu_model_summary: dict[str, object] = {}
    cpu_required_node_ids = {
        idx
        for idx, node in graph.items()
        if routing.route_for(node.op_type).backend == "CPU"
    }
    if cpu_model_bundle is not None:
        cpu_model_result = run_cpu_static_model_inference(
            cpu_model_root=cpu_model_root,
            model_bundle_path=cpu_model_bundle,
            graph=graph,
            shape_csv=shape_csv,
            required_node_ids=cpu_required_node_ids,
            num_threads=cpu_model_num_threads,
            inter_threads=cpu_model_inter_threads,
        )
        cpu_model_result.predictions.to_csv(output_dir / "cpu_operator_predictions.csv", index=False)
        (output_dir / "cpu_model_summary.json").write_text(
            json.dumps(cpu_model_result.summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        cpu_model_summary = cpu_model_result.summary
        cpu_adapter = CpuLatencyAdapter(
            prediction_frame=cpu_model_result.predictions,
            prediction_source="cpu_model_bundle_full_graph_76",
            allow_static_fallback=allow_cpu_static_fallback,
        )
    else:
        cpu_adapter = CpuLatencyAdapter(cpu_predictions, allow_static_fallback=allow_cpu_static_fallback)
    npu_adapter = NpuLatencyAdapter(npu_root)
    alltoall_adapter = AllToAllAdapter(alltoall_root)

    node_df = predict_nodes(graph, routing, cpu_adapter, npu_adapter)
    pred_by_node = dict(zip(node_df["node_idx"].astype(int), node_df["pred_us"].astype(float)))
    backend_by_node = dict(zip(node_df["node_idx"].astype(int), node_df["route_backend"].astype(str)))

    schedulable = {idx for idx, node in graph.items() if node.partition != "constant"}
    non_tail = {idx for idx in schedulable if graph[idx].partition in {"bottom", "embedding"}}
    tail = {idx for idx in schedulable if graph[idx].partition == "tail"}

    first_spans = schedule_subset(
        graph=graph,
        pred_by_node=pred_by_node,
        backend_by_node=backend_by_node,
        node_ids=non_tail,
        cpu_slots=cpu_slots,
        npu_slots=npu_slots,
        external_ready=lambda _idx: 0.0,
    )
    bottom_end_us = max((span.end_us for idx, span in first_spans.items() if graph[idx].partition == "bottom"), default=0.0)
    embedding_end_us = max((span.end_us for idx, span in first_spans.items() if graph[idx].partition == "embedding"), default=0.0)
    message_bytes = embedding_message_bytes(graph)
    a2a = alltoall_adapter.predict_region(message_bytes, compute_window_us=bottom_end_us)
    alltoall_start_us = embedding_end_us
    alltoall_end_us = max(bottom_end_us, alltoall_start_us + a2a.visible_comm_us)

    def external_ready(pred_idx: int) -> float:
        if pred_idx in first_spans:
            if graph[pred_idx].partition == "embedding":
                return alltoall_end_us
            return first_spans[pred_idx].end_us
        return 0.0

    tail_spans = schedule_subset(
        graph=graph,
        pred_by_node=pred_by_node,
        backend_by_node=backend_by_node,
        node_ids=tail,
        cpu_slots=cpu_slots,
        npu_slots=npu_slots,
        external_ready=external_ready,
    )
    spans = {**first_spans, **tail_spans}
    node_df["start_us"] = node_df["node_idx"].map(lambda idx: spans.get(int(idx), ScheduledNode(int(idx), 0.0, 0.0)).start_us)
    node_df["end_us"] = node_df["node_idx"].map(lambda idx: spans.get(int(idx), ScheduledNode(int(idx), 0.0, 0.0)).end_us)
    node_df.to_csv(output_dir / "fused_node_predictions.csv", index=False)

    case_id, combo, batch_size, nip = parse_combo_from_path(shape_csv)
    e2e_us = max([alltoall_end_us, *(span.end_us for span in spans.values())])
    a2a_df = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "combo": combo,
                "message_bytes": a2a.message_bytes,
                "alltoall_start_us": alltoall_start_us,
                "comm_us": a2a.comm_us,
                "compute_window_us": a2a.compute_window_us,
                "visible_comm_us": a2a.visible_comm_us,
                "hidden_comm_us": a2a.hidden_comm_us,
                "hidden_fraction_of_comm": a2a.hidden_fraction_of_comm,
                "eta": a2a.eta,
                "alltoall_end_us": alltoall_end_us,
            }
        ]
    )
    a2a_df.to_csv(output_dir / "alltoall_region_predictions.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "combo": combo,
                "batch_size": batch_size,
                "num_indices_per_lookup": nip,
                "node_count": int(len(schedulable)),
                "bottom_end_us": bottom_end_us,
                "embedding_end_us": embedding_end_us,
                "alltoall_visible_comm_us": a2a.visible_comm_us,
                "tail_end_us": max((span.end_us for idx, span in tail_spans.items()), default=0.0),
                "predicted_e2e_us": e2e_us,
            }
        ]
    )
    metrics_df.to_csv(output_dir / "fused_e2e_combo_metrics.csv", index=False)

    coverage = (
        node_df.groupby(["op_type", "route_backend", "target_op_type"], as_index=False)
        .agg(node_count=("node_idx", "count"), pred_us_sum=("pred_us", "sum"))
        .sort_values(["route_backend", "op_type"])
    )
    coverage.to_csv(output_dir / "routing_coverage.csv", index=False)

    summary = {
        "cpu_model_root": str(cpu_model_root),
        "npu_root": str(npu_root),
        "alltoall_root": str(alltoall_root),
        "routing_path": str(routing_path),
        "shape_csv": str(shape_csv),
        "cpu_predictions": str(cpu_predictions) if cpu_predictions else None,
        "cpu_model_bundle": str(cpu_model_bundle) if cpu_model_bundle else None,
        "cpu_prediction_source": (
            "cpu_model_bundle_full_graph_76"
            if cpu_model_bundle
            else ("cpu_prediction_csv" if cpu_predictions else "cpu_static_fallback")
        ),
        "cpu_required_node_count": int(len(cpu_required_node_ids)),
        "cpu_static_fallback_used": bool(cpu_adapter.uses_static_fallback),
        "cpu_model_summary": cpu_model_summary,
        "cpu_slots": int(cpu_slots),
        "npu_slots": int(npu_slots),
        "case_id": case_id,
        "combo": combo,
        "predicted_e2e_us": float(e2e_us),
        "artifacts": {
            "fused_node_predictions": str(output_dir / "fused_node_predictions.csv"),
            "alltoall_region_predictions": str(output_dir / "alltoall_region_predictions.csv"),
            "fused_e2e_combo_metrics": str(output_dir / "fused_e2e_combo_metrics.csv"),
            "routing_coverage": str(output_dir / "routing_coverage.csv"),
            "cpu_operator_predictions": str(output_dir / "cpu_operator_predictions.csv") if cpu_model_bundle else None,
            "cpu_model_summary": str(output_dir / "cpu_model_summary.json") if cpu_model_bundle else None,
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _case_csv(case_output_dir: Path, filename: str, case_name: str) -> pd.DataFrame:
    df = pd.read_csv(case_output_dir / filename)
    df.insert(0, "e2e_case_name", case_name)
    return df


def run_fusion_cases(
    *,
    cpu_model_root: Path,
    npu_root: Path,
    alltoall_root: Path,
    routing_path: Path,
    output_dir: Path,
    cases_config: Path,
    cpu_predictions: Path | None = None,
    cpu_model_bundle: Path | None = None,
    cpu_model_num_threads: int | None = None,
    cpu_model_inter_threads: int | None = None,
    cpu_slots: int = 1,
    npu_slots: int = 1,
    allow_cpu_static_fallback: bool = True,
) -> dict[str, object]:
    if cpu_predictions is not None and cpu_model_bundle is not None:
        raise ValueError("cpu_predictions and cpu_model_bundle are mutually exclusive")
    output_dir.mkdir(parents=True, exist_ok=True)
    case_specs = load_case_specs(cases_config, cpu_model_root)
    cases_dir = output_dir / "cases"

    per_case_summaries: list[dict[str, object]] = []
    aggregate_files = [
        "fused_node_predictions.csv",
        "alltoall_region_predictions.csv",
        "fused_e2e_combo_metrics.csv",
        "routing_coverage.csv",
    ]
    optional_aggregate_files = [
        "cpu_operator_predictions.csv",
    ]
    aggregate_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in aggregate_files}
    optional_aggregate_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in optional_aggregate_files}

    for spec in case_specs:
        case_output_dir = cases_dir / spec.name
        case_summary = run_fusion(
            cpu_model_root=cpu_model_root,
            npu_root=npu_root,
            alltoall_root=alltoall_root,
            routing_path=routing_path,
            output_dir=case_output_dir,
            shape_csv=spec.shape_csv,
            cpu_predictions=cpu_predictions,
            cpu_model_bundle=cpu_model_bundle,
            cpu_model_num_threads=cpu_model_num_threads,
            cpu_model_inter_threads=cpu_model_inter_threads,
            cpu_slots=cpu_slots,
            npu_slots=npu_slots,
            allow_cpu_static_fallback=allow_cpu_static_fallback,
        )
        per_case_summaries.append(
            {
                "name": spec.name,
                "shape_csv": str(spec.shape_csv),
                "case_id": case_summary["case_id"],
                "combo": case_summary["combo"],
                "predicted_e2e_us": case_summary["predicted_e2e_us"],
                "cpu_prediction_source": case_summary["cpu_prediction_source"],
                "cpu_static_fallback_used": case_summary["cpu_static_fallback_used"],
                "output_dir": str(case_output_dir),
            }
        )
        for filename in aggregate_files:
            aggregate_frames[filename].append(_case_csv(case_output_dir, filename, spec.name))
        for filename in optional_aggregate_files:
            if (case_output_dir / filename).exists():
                optional_aggregate_frames[filename].append(_case_csv(case_output_dir, filename, spec.name))

    for filename, frames in aggregate_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(output_dir / filename, index=False)
    for filename, frames in optional_aggregate_frames.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output_dir / filename, index=False)

    summary = {
        "mode": "cases_config",
        "cpu_model_root": str(cpu_model_root),
        "npu_root": str(npu_root),
        "alltoall_root": str(alltoall_root),
        "routing_path": str(routing_path),
        "cases_config": str(cases_config),
        "cpu_predictions": str(cpu_predictions) if cpu_predictions else None,
        "cpu_model_bundle": str(cpu_model_bundle) if cpu_model_bundle else None,
        "cpu_prediction_source": (
            "cpu_model_bundle_full_graph_76"
            if cpu_model_bundle
            else ("cpu_prediction_csv" if cpu_predictions else "cpu_static_fallback")
        ),
        "cpu_static_fallback_used": any(bool(case["cpu_static_fallback_used"]) for case in per_case_summaries),
        "cpu_slots": int(cpu_slots),
        "npu_slots": int(npu_slots),
        "case_count": len(per_case_summaries),
        "cases": per_case_summaries,
        "artifacts": {
            "cases_dir": str(cases_dir),
            "fused_node_predictions": str(output_dir / "fused_node_predictions.csv"),
            "alltoall_region_predictions": str(output_dir / "alltoall_region_predictions.csv"),
            "fused_e2e_combo_metrics": str(output_dir / "fused_e2e_combo_metrics.csv"),
            "routing_coverage": str(output_dir / "routing_coverage.csv"),
            "cpu_operator_predictions": str(output_dir / "cpu_operator_predictions.csv") if (output_dir / "cpu_operator_predictions.csv").exists() else None,
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
