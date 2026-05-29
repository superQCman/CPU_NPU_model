#!/usr/bin/env python3
"""
Manual branch-parallel DLRM runner.

This runner keeps the front-end CLI close to run_ort_dlrm.py, but executes the
rewritten DLRM graph in a different way:
  - bottom MLP branch
  - emb_l0 .. emb_l7 branches
  - tail graph after branch merge

The branch tasks are executed concurrently via separate ORT sessions, which is a
practical fallback when ORT_PARALLEL still runs the full graph serially.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from onnx import load as onnx_load
from onnx import utils as onnx_utils

from run_ort_dlrm import (
    _force_ops_to_cpu,
    _import_onnxruntime,
    _patch_model_for_cann,
    _replace_loop_with_gather,
    _setup_cann_env,
    dump_op_shapes_to_csv,
    generate_inputs,
    print_statistics,
)


@dataclass
class TaskSpec:
    name: str
    model_path: Path
    session: ort.InferenceSession
    output_name: str
    input_names: List[str]
    lane: int


@dataclass
class TaskRecord:
    batch_idx: int
    task_name: str
    phase: str
    lane: int
    start_us: int
    end_us: int
    dur_us: int


@dataclass
class OpRecord:
    batch_idx: int
    task_name: str
    op_name: str
    node_name: str
    provider: str
    lane: int
    start_us: int
    end_us: int
    dur_us: int


@dataclass
class ProfileMergeRecord:
    batch_idx: int
    task_name: str
    lane: int
    event: dict


@dataclass(frozen=True)
class SplitLayout:
    bottom_input_name: str
    bottom_output_name: str
    emb_input_names: List[str]
    emb_output_names: List[str]
    tail_output_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DLRM branch-parallel ORT runner")
    parser.add_argument("--onnx-path", type=str, default="./dlrm_onnx/dlrm_s_pytorch.onnx")
    parser.add_argument("--use-cann", action="store_true", default=False)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--intra-threads", type=int, default=1, help="per-branch intra-op threads")
    parser.add_argument("--inter-threads", type=int, default=1, help="max concurrent branch tasks")
    parser.add_argument("--parallel-branches", type=int, default=0, help="explicit max concurrent branch tasks; overrides --inter-threads when > 0")
    parser.add_argument("--tail-intra-threads", type=int, default=0, help="tail session intra-op threads; 0 means reuse --intra-threads")
    parser.add_argument("--no-replace-loop", action="store_true", default=False)
    parser.add_argument("--num-indices-per-lookup", type=int, default=0, metavar="N")
    parser.add_argument("--force-cpu-ops", type=str, default="", metavar="OP[,OP,...]")
    parser.add_argument("--enable-profiling", action="store_true", default=False, help="enable per-submodel ORT profiling")
    parser.add_argument("--profile-warmup", action="store_true", default=False, help="include warmup in branch profiling")
    parser.add_argument("--profile-dir", type=str, default="./onnx_operator_analysis/branch_parallel")
    parser.add_argument("--disable-graph-optimizations", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--shape-csv", type=str, default="")
    parser.add_argument("--submodel-dir", type=str, default="./branch_parallel_submodels")
    parser.add_argument("--out-dir", type=str, default="", help="timeline output directory; defaults to --profile-dir")
    parser.add_argument("--verify-full-output", action="store_true", default=False)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir).resolve() if args.out_dir else Path(args.profile_dir).resolve()


def build_providers(use_cann: bool, device_id: int) -> List[object]:
    _setup_cann_env()
    _, available = _import_onnxruntime()

    if use_cann and "CANNExecutionProvider" not in available:
        print("[WARN] CANNExecutionProvider unavailable; falling back to CPUExecutionProvider.")
        use_cann = False

    if not use_cann:
        return ["CPUExecutionProvider"]

    cann_provider_options = {
        "device_id": str(device_id),
        "precision_mode": "force_fp32",
        "op_select_impl_mode": "high_performance",
        "arena_extend_strategy": "kNextPowerOfTwo",
        "enable_cann_graph": "0",
    }
    return [
        ("CANNExecutionProvider", cann_provider_options),
        "CPUExecutionProvider",
    ]


def model_contains_ops(model_path: Path, op_types: Sequence[str]) -> bool:
    target_ops = {op.strip() for op in op_types if op and op.strip()}
    if not target_ops:
        return False
    model = onnx_load(str(model_path))
    return any(node.op_type in target_ops for node in model.graph.node)


def choose_task_providers(
    base_providers: List[object],
    use_cann: bool,
    model_path: Path,
    force_cpu_ops: Sequence[str],
) -> List[object]:
    if not use_cann:
        return base_providers

    forced_ops = {op.strip() for op in force_cpu_ops if op and op.strip()}
    if "Reshape" in forced_ops and model_contains_ops(model_path, ["Reshape"]):
        print(
            f"[FORCE_CPU] {model_path.name} contains Reshape; "
            "using CPUExecutionProvider for this submodel."
        )
        return ["CPUExecutionProvider"]
    return base_providers


def build_full_graph_node_lookup(onnx_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]]]:
    model = onnx_load(str(onnx_path))
    by_name: Dict[str, Dict[str, str]] = {}
    by_name_op: Dict[Tuple[str, str], Dict[str, str]] = {}

    def register(name: str, node_idx: int, op_type: str) -> None:
        if not name:
            return
        payload = {
            "node_index": str(node_idx),
            "node_name": name,
            "op_name": op_type,
        }
        by_name[name] = payload
        by_name_op[(name, op_type)] = payload

    for idx, node in enumerate(model.graph.node):
        register(node.name, idx, node.op_type)
        register(f"fused {node.name}", idx, node.op_type)
    return by_name, by_name_op


def build_session(
    model_path: Path,
    intra_threads: int,
    providers: List[object],
    disable_graph_optimizations: bool,
    enable_profiling: bool,
    profile_prefix: str | None,
) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3
    if disable_graph_optimizations:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if enable_profiling:
        opts.enable_profiling = True
        if profile_prefix:
            opts.profile_file_prefix = profile_prefix
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


def extract_submodel(src: Path, dst: Path, input_names: List[str], output_names: List[str]) -> None:
    if dst.exists():
        try:
            if dst.stat().st_mtime_ns >= src.stat().st_mtime_ns:
                return
        except FileNotFoundError:
            pass
    ensure_dir(dst.parent)
    onnx_utils.extract_model(str(src), str(dst), input_names, output_names)


def _dedupe_keep_order(names: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _sorted_embedding_inputs(runtime_inputs: Sequence[str]) -> List[str]:
    emb_inputs = [name for name in runtime_inputs if name.startswith("indices_")]

    def sort_key(name: str) -> Tuple[int, str]:
        try:
            return (int(name.split("_", 1)[1]), name)
        except (IndexError, ValueError):
            return (10**9, name)

    return sorted(emb_inputs, key=sort_key)


def _preferred_boundary_marker(group_name: str) -> str | None:
    if group_name == "bottom":
        return "/bot_l/"
    if group_name.startswith("indices_"):
        try:
            idx = int(group_name.split("_", 1)[1])
        except (IndexError, ValueError):
            return None
        return f"/emb_l.{idx}/"
    return None


def _require_single_boundary(group_name: str, tensors: Sequence[str]) -> str:
    unique = _dedupe_keep_order(tensors)
    preferred_marker = _preferred_boundary_marker(group_name)
    if preferred_marker:
        preferred = [name for name in unique if preferred_marker in name]
        if preferred:
            unique = preferred
    if len(unique) != 1:
        raise ValueError(f"Expected exactly one boundary tensor for {group_name}, got {unique}")
    return unique[0]


def infer_split_layout(onnx_path: Path) -> SplitLayout:
    model = onnx_load(str(onnx_path))
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    static_tensors: set[str] = set(initializer_names)
    runtime_inputs = [value.name for value in graph.input if value.name not in initializer_names]

    if "dense_x" not in runtime_inputs:
        raise ValueError(f"Could not find dense_x among runtime graph inputs: {runtime_inputs}")

    emb_input_names = _sorted_embedding_inputs(runtime_inputs)
    if not emb_input_names:
        raise ValueError(f"Could not find embedding index inputs in runtime graph inputs: {runtime_inputs}")

    tensor_groups: Dict[str, str] = {"dense_x": "bottom"}
    for name in emb_input_names:
        tensor_groups[name] = name

    node_groups: List[str | None] = []
    consumers: Dict[str, List[int]] = defaultdict(list)
    for idx, node in enumerate(graph.node):
        for inp in node.input:
            if inp:
                consumers[inp].append(idx)

    for node in graph.node:
        dynamic_inputs = [inp for inp in node.input if inp and inp not in static_tensors]
        source_groups = {tensor_groups[inp] for inp in dynamic_inputs if inp in tensor_groups}
        has_unknown_dynamic_input = any(inp not in tensor_groups for inp in dynamic_inputs)
        node_group = next(iter(source_groups)) if len(source_groups) == 1 and not has_unknown_dynamic_input else None
        node_groups.append(node_group)

        if not dynamic_inputs:
            for out in node.output:
                if out:
                    static_tensors.add(out)

        for out in node.output:
            if out:
                if node_group is not None:
                    tensor_groups[out] = node_group
                elif out in tensor_groups:
                    del tensor_groups[out]

    graph_outputs = {value.name for value in graph.output}
    boundary_tensors: Dict[str, List[str]] = defaultdict(list)
    for idx, node in enumerate(graph.node):
        node_group = node_groups[idx]
        if node_group is None:
            continue
        for out in node.output:
            if not out:
                continue
            crosses_group = out in graph_outputs or any(node_groups[consumer_idx] != node_group for consumer_idx in consumers.get(out, []))
            if crosses_group:
                boundary_tensors[node_group].append(out)

    bottom_output_name = _require_single_boundary("bottom", boundary_tensors.get("bottom", []))
    emb_output_names = [
        _require_single_boundary(emb_input_name, boundary_tensors.get(emb_input_name, []))
        for emb_input_name in emb_input_names
    ]

    graph_output_names = [value.name for value in graph.output]
    if len(graph_output_names) != 1:
        raise ValueError(f"Expected exactly one graph output for tail extraction, got {graph_output_names}")

    return SplitLayout(
        bottom_input_name="dense_x",
        bottom_output_name=bottom_output_name,
        emb_input_names=emb_input_names,
        emb_output_names=emb_output_names,
        tail_output_name=graph_output_names[0],
    )


def prepare_submodels(rewritten_onnx: Path, submodel_dir: Path, layout: SplitLayout) -> Tuple[Path, List[Path], Path]:
    stem = rewritten_onnx.stem
    bottom_path = submodel_dir / f"{stem}_bottom.onnx"
    emb_paths = [submodel_dir / f"{stem}_{emb_input_name}.onnx" for emb_input_name in layout.emb_input_names]
    tail_path = submodel_dir / f"{stem}_tail.onnx"

    extract_submodel(rewritten_onnx, bottom_path, [layout.bottom_input_name], [layout.bottom_output_name])
    for emb_input_name, emb_output_name, emb_path in zip(layout.emb_input_names, layout.emb_output_names, emb_paths):
        extract_submodel(rewritten_onnx, emb_path, [emb_input_name], [emb_output_name])
    extract_submodel(
        rewritten_onnx,
        tail_path,
        [layout.bottom_output_name] + layout.emb_output_names,
        [layout.tail_output_name],
    )
    return bottom_path, emb_paths, tail_path


def create_task_specs(
    bottom_path: Path,
    emb_paths: List[Path],
    tail_path: Path,
    layout: SplitLayout,
    args: argparse.Namespace,
    providers: List[object],
    force_cpu_ops: Sequence[str],
    profile_enabled: bool,
) -> Tuple[List[TaskSpec], TaskSpec]:
    branch_specs: List[TaskSpec] = []
    profile_dir = Path(args.profile_dir).resolve()

    def prefix(task_name: str) -> str | None:
        if not profile_enabled:
            return None
        ensure_dir(profile_dir)
        return str(profile_dir / f"{task_name}_profile")

    bottom_sess = build_session(
        bottom_path,
        intra_threads=args.intra_threads,
        providers=choose_task_providers(providers, args.use_cann, bottom_path, force_cpu_ops),
        disable_graph_optimizations=args.disable_graph_optimizations,
        enable_profiling=profile_enabled,
        profile_prefix=prefix("bottom"),
    )
    branch_specs.append(
        TaskSpec(
            name="bottom",
            model_path=bottom_path,
            session=bottom_sess,
            output_name=layout.bottom_output_name,
            input_names=[inp.name for inp in bottom_sess.get_inputs()],
            lane=0,
        )
    )

    for idx, (emb_path, emb_output_name) in enumerate(zip(emb_paths, layout.emb_output_names)):
        sess = build_session(
            emb_path,
            intra_threads=args.intra_threads,
            providers=choose_task_providers(providers, args.use_cann, emb_path, force_cpu_ops),
            disable_graph_optimizations=args.disable_graph_optimizations,
            enable_profiling=profile_enabled,
            profile_prefix=prefix(f"emb_l{idx}"),
        )
        branch_specs.append(
            TaskSpec(
                name=f"emb_l{idx}",
                model_path=emb_path,
                session=sess,
                output_name=emb_output_name,
                input_names=[inp.name for inp in sess.get_inputs()],
                lane=idx + 1,
            )
        )

    tail_threads = args.tail_intra_threads if args.tail_intra_threads > 0 else args.intra_threads
    tail_sess = build_session(
        tail_path,
        intra_threads=tail_threads,
        providers=choose_task_providers(providers, args.use_cann, tail_path, force_cpu_ops),
        disable_graph_optimizations=args.disable_graph_optimizations,
        enable_profiling=profile_enabled,
        profile_prefix=prefix("tail"),
    )
    tail_spec = TaskSpec(
        name="tail",
        model_path=tail_path,
        session=tail_sess,
        output_name=layout.tail_output_name,
        input_names=[inp.name for inp in tail_sess.get_inputs()],
        lane=len(branch_specs),
    )
    return branch_specs, tail_spec


def run_task(spec: TaskSpec, feed: Dict[str, np.ndarray], batch_idx: int) -> Tuple[np.ndarray, TaskRecord]:
    subset = {name: feed[name] for name in spec.input_names}
    start_ns = time.perf_counter_ns()
    outputs = spec.session.run([spec.output_name], subset)
    end_ns = time.perf_counter_ns()
    return (
        outputs[0],
        TaskRecord(
            batch_idx=batch_idx,
            task_name=spec.name,
            phase="branch" if spec.name != "tail" else "tail",
            lane=spec.lane,
            start_us=start_ns // 1000,
            end_us=end_ns // 1000,
            dur_us=(end_ns - start_ns) // 1000,
        ),
    )


def run_branch_stage(
    branch_specs: Sequence[TaskSpec],
    full_feed: Dict[str, np.ndarray],
    batch_idx: int,
    max_parallel_branches: int,
) -> Tuple[Dict[str, np.ndarray], List[TaskRecord]]:
    branch_results: Dict[str, np.ndarray] = {}
    batch_records: List[TaskRecord] = []

    with ThreadPoolExecutor(max_workers=max_parallel_branches) as pool:
        future_to_spec: Dict[Future[Tuple[np.ndarray, TaskRecord]], TaskSpec] = {
            pool.submit(run_task, spec, full_feed, batch_idx): spec
            for spec in branch_specs
        }
        for future, spec in future_to_spec.items():
            output, record = future.result()
            branch_results[spec.output_name] = output
            batch_records.append(record)

    batch_records.sort(key=lambda item: (item.start_us, item.task_name))
    return branch_results, batch_records


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_profile_node_name(name: str) -> str:
    return name[:-12] if name.endswith("_kernel_time") else name


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def format_us(value: int) -> str:
    return f"{value:,} us"


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def hash_color(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    hue = int(digest[:6], 16) % 360
    return f"hsl({hue}, 58%, 68%)"


def summarize_records(records: Sequence[TaskRecord | OpRecord]) -> Tuple[int, int]:
    rows = [
        {
            "start_us": record.start_us,
            "end_us": record.end_us,
            "dur_us": record.dur_us,
            "label": getattr(record, "task_name", ""),
        }
        for record in records
    ]
    summary = summarize_row_timeline(rows, build_occupancy_segments(rows))
    return int(summary["max_concurrency"]), int(summary["total_parallel_us"])


def build_occupancy_segments(rows: Sequence[dict]) -> List[Dict[str, object]]:
    boundaries: List[tuple[int, int, int]] = []
    for idx, row in enumerate(rows):
        boundaries.append((int(row["start_us"]), 0, idx))
        boundaries.append((int(row["end_us"]), 1, idx))
    boundaries.sort()

    active: set[int] = set()
    prev_time: int | None = None
    segments: List[Dict[str, object]] = []

    for time_pt, event_type, idx in boundaries:
        if prev_time is not None and time_pt > prev_time and active:
            active_rows = [rows[i] for i in sorted(active)]
            segments.append(
                {
                    "start_us": prev_time,
                    "end_us": time_pt,
                    "dur_us": time_pt - prev_time,
                    "concurrency": len(active_rows),
                    "labels": "|".join(str(row.get("label", "")) for row in active_rows),
                    "tasks": "|".join(str(row.get("task_name", "")) for row in active_rows if row.get("task_name")),
                    "providers": "|".join(
                        sorted({str(row.get("provider", "")) for row in active_rows if str(row.get("provider", ""))})
                    ),
                }
            )

        prev_time = time_pt
        if event_type == 0:
            active.add(idx)
        else:
            active.discard(idx)

    return segments


def summarize_row_timeline(rows: Sequence[dict], occupancy_segments: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "event_count": 0,
            "provider_counts": {},
            "label_counts": {},
            "task_counts": {},
            "min_ts_us": 0,
            "max_end_us": 0,
            "wall_time_us": 0,
            "total_active_us": 0,
            "total_parallel_us": 0,
            "parallel_pct": 0.0,
            "max_concurrency": 0,
            "avg_concurrency_when_active": 0.0,
        }

    min_ts_us = min(int(row["start_us"]) for row in rows)
    max_end_us = max(int(row["end_us"]) for row in rows)
    total_active_us = sum(int(seg["dur_us"]) for seg in occupancy_segments)
    total_parallel_us = sum(int(seg["dur_us"]) for seg in occupancy_segments if int(seg["concurrency"]) >= 2)
    weighted_concurrency = sum(int(seg["dur_us"]) * int(seg["concurrency"]) for seg in occupancy_segments)

    provider_counts = Counter(str(row.get("provider", "")) for row in rows if str(row.get("provider", "")))
    label_counts = Counter(str(row.get("label", "")) for row in rows if str(row.get("label", "")))
    task_counts = Counter(str(row.get("task_name", "")) for row in rows if str(row.get("task_name", "")))

    return {
        "event_count": len(rows),
        "provider_counts": dict(provider_counts),
        "label_counts": dict(label_counts),
        "task_counts": dict(task_counts),
        "min_ts_us": min_ts_us,
        "max_end_us": max_end_us,
        "wall_time_us": max_end_us - min_ts_us,
        "total_active_us": total_active_us,
        "total_parallel_us": total_parallel_us,
        "parallel_pct": (100.0 * total_parallel_us / total_active_us) if total_active_us else 0.0,
        "max_concurrency": max((int(seg["concurrency"]) for seg in occupancy_segments), default=0),
        "avg_concurrency_when_active": (weighted_concurrency / total_active_us) if total_active_us else 0.0,
    }


def build_overlap_pairs(rows: Sequence[dict]) -> List[Dict[str, object]]:
    boundaries: List[tuple[int, int, int]] = []
    for idx, row in enumerate(rows):
        boundaries.append((int(row["start_us"]), 0, idx))
        boundaries.append((int(row["end_us"]), 1, idx))
    boundaries.sort()

    active: set[int] = set()
    prev_time: int | None = None
    pair_agg = defaultdict(lambda: {"count": 0, "overlap_us": 0})

    for time_pt, event_type, idx in boundaries:
        if prev_time is not None and time_pt > prev_time and len(active) >= 2:
            active_rows = [rows[i] for i in sorted(active)]
            interval_dur = time_pt - prev_time
            active_rows.sort(key=lambda row: str(row.get("label", "")))
            for i in range(len(active_rows)):
                for j in range(i + 1, len(active_rows)):
                    a = active_rows[i]
                    b = active_rows[j]
                    key = (str(a.get("label", "")), str(b.get("label", "")))
                    pair_agg[key]["count"] += 1
                    pair_agg[key]["overlap_us"] += interval_dur

        prev_time = time_pt
        if event_type == 0:
            active.add(idx)
        else:
            active.discard(idx)

    out = []
    for (a_label, b_label), stats in pair_agg.items():
        out.append(
            {
                "label_a": a_label,
                "label_b": b_label,
                "overlap_count": stats["count"],
                "total_overlap_us": stats["overlap_us"],
            }
        )
    out.sort(key=lambda row: (-int(row["total_overlap_us"]), row["label_a"], row["label_b"]))
    return out


def render_counts_table(title: str, counter: Dict[str, int], limit: int = 12) -> str:
    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    if not rows:
        return "<p>No data.</p>"
    body = "\n".join(
        f"<tr><td>{html.escape(name or '(empty)')}</td><td>{count}</td></tr>"
        for name, count in rows
    )
    return (
        f"<section><h3>{html.escape(title)}</h3>"
        "<table><thead><tr><th>Name</th><th>Count</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def render_pair_table(pair_rows: Sequence[Dict[str, object]], limit: int = 15) -> str:
    if not pair_rows:
        return "<p>No overlapping pairs detected.</p>"
    body = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['label_a']))}</td>"
        f"<td>{html.escape(str(row['label_b']))}</td>"
        f"<td>{row['overlap_count']}</td>"
        f"<td>{row['total_overlap_us']}</td>"
        "</tr>"
        for row in pair_rows[:limit]
    )
    return (
        "<table><thead><tr>"
        "<th>label_a</th><th>label_b</th><th>overlap_count</th><th>total_overlap_us</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table>"
    )


def render_segment_table(segments: Sequence[Dict[str, object]], limit: int = 20) -> str:
    parallel_only = [seg for seg in segments if int(seg["concurrency"]) >= 2]
    parallel_only.sort(key=lambda seg: (-int(seg["concurrency"]), -int(seg["dur_us"]), int(seg["start_us"])))
    if not parallel_only:
        return "<p>No concurrency segments detected.</p>"
    body = "\n".join(
        "<tr>"
        f"<td>{seg['concurrency']}</td>"
        f"<td>{seg['start_us']}</td>"
        f"<td>{seg['end_us']}</td>"
        f"<td>{seg['dur_us']}</td>"
        f"<td>{html.escape(truncate_text(str(seg.get('labels', '')), 120))}</td>"
        "</tr>"
        for seg in parallel_only[:limit]
    )
    return (
        "<table><thead><tr>"
        "<th>concurrency</th><th>start_us</th><th>end_us</th><th>dur_us</th><th>labels</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table>"
    )


def render_timeline_html(path: Path, rows: Sequence[dict], lane_labels: Dict[int, str], title: str, source_note: str = "") -> None:
    if not rows:
        path.write_text("<html><body><p>No records.</p></body></html>", encoding="utf-8")
        return

    occupancy_segments = build_occupancy_segments(rows)
    summary = summarize_row_timeline(rows, occupancy_segments)
    pair_rows = build_overlap_pairs(rows)

    min_us = int(summary["min_ts_us"])
    max_us = int(summary["max_end_us"])
    span_us = max(max_us - min_us, 1)

    width = 1600
    left_pad = 150
    right_pad = 20
    top_pad = 24
    lane_height = 22
    plot_width = width - left_pad - right_pad
    lane_count = max(lane_labels.keys()) + 1 if lane_labels else 1
    occupancy_height = 140
    lane_chart_height = max(lane_count * lane_height + 40, 160)
    svg_height = occupancy_height + lane_chart_height + 100

    def x_for(ts_us: int) -> float:
        return left_pad + ((ts_us - min_us) / span_us) * plot_width

    ticks = []
    for idx in range(11):
        value = min_us + int(span_us * idx / 10)
        x = x_for(value)
        ticks.append(
            f'<line x1="{x:.2f}" y1="{top_pad}" x2="{x:.2f}" y2="{svg_height - 30}" class="tick" />'
            f'<text x="{x:.2f}" y="{svg_height - 8}" class="tick-label">{value - min_us} us</text>'
        )

    occupancy_rects = []
    max_concurrency = max(int(summary["max_concurrency"]), 1)
    for seg in occupancy_segments:
        start = int(seg["start_us"])
        end = int(seg["end_us"])
        concurrency = int(seg["concurrency"])
        x = x_for(start)
        width_px = max(x_for(end) - x, 1.0)
        bar_height = (concurrency / max_concurrency) * (occupancy_height - 40)
        y = top_pad + (occupancy_height - 20) - bar_height
        occupancy_rects.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width_px:.2f}" height="{bar_height:.2f}" class="occupancy">'
            f'<title>concurrency={concurrency} start={start}us end={end}us dur={end - start}us labels={html.escape(str(seg.get("labels", "")))}</title></rect>'
        )

    lane_text = []
    for lane, label in sorted(lane_labels.items()):
        y = occupancy_height + 30 + lane * lane_height
        lane_text.append(
            f'<text x="{left_pad - 10}" y="{y + 13}" class="lane-label">{html.escape(label)}</text>'
        )

    rects = []
    for row in rows:
        lane = int(row["lane"])
        label = str(row.get("label", row.get("task_name", row.get("op_name", "item"))))
        x = x_for(int(row["start_us"]))
        width_px = max(x_for(int(row["end_us"])) - x, 2.0)
        y = occupancy_height + 30 + lane * lane_height
        tooltip = " ".join(f"{k}={v}" for k, v in row.items() if k not in {"lane"})
        rects.append(
            f'<g><rect x="{x:.2f}" y="{y:.2f}" width="{width_px:.2f}" height="16" rx="3" fill="{hash_color(label)}" class="event">'
            f'<title>{html.escape(tooltip)}</title></rect>'
            + (
                f'<text x="{x + 3:.2f}" y="{y + 12:.2f}" class="event-label">{html.escape(truncate_text(label, 24))}</text>'
                if width_px >= 38
                else ""
            )
            + "</g>"
        )

    provider_table = render_counts_table("Provider Event Counts", summary["provider_counts"], limit=8)
    task_table = render_counts_table("Top Task Counts", summary["task_counts"], limit=12)
    label_table = render_counts_table("Top Label Counts", summary["label_counts"], limit=12)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      margin: 24px;
      color: #10212b;
      background: #f5f7f9;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0 24px 0;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d6dde3;
      border-radius: 8px;
      padding: 12px 14px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }}
    .card .label {{
      font-size: 12px;
      color: #5b6b76;
      margin-bottom: 6px;
    }}
    .card .value {{
      font-size: 20px;
      font-weight: 700;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d6dde3;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
      overflow-x: auto;
    }}
    svg {{
      background: #fbfcfd;
      border: 1px solid #e0e6eb;
      border-radius: 8px;
    }}
    .tick {{
      stroke: #dbe4ea;
      stroke-width: 1;
    }}
    .tick-label {{
      font-size: 10px;
      text-anchor: middle;
      fill: #6d7c87;
    }}
    .lane-label {{
      font-size: 11px;
      text-anchor: end;
      fill: #6d7c87;
    }}
    .event {{
      stroke: rgba(16, 33, 43, 0.35);
      stroke-width: 0.7;
    }}
    .event-label {{
      font-size: 10px;
      fill: #0e2430;
      pointer-events: none;
    }}
    .occupancy {{
      fill: #3b82f6;
      opacity: 0.55;
    }}
    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(360px, 1fr));
      gap: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #e7edf1;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f4f8fb;
    }}
    .note {{
      color: #5b6b76;
      font-size: 12px;
      margin-top: 10px;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="note">{html.escape(source_note)}</div>

  <div class="summary">
    <div class="card"><div class="label">Events</div><div class="value">{summary['event_count']}</div></div>
    <div class="card"><div class="label">Lane count</div><div class="value">{lane_count}</div></div>
    <div class="card"><div class="label">Max concurrency</div><div class="value">{summary['max_concurrency']}</div></div>
    <div class="card"><div class="label">Parallel active time</div><div class="value">{format_pct(float(summary['parallel_pct']))}</div></div>
    <div class="card"><div class="label">Wall time span</div><div class="value">{format_us(int(summary['wall_time_us']))}</div></div>
    <div class="card"><div class="label">Active time</div><div class="value">{format_us(int(summary['total_active_us']))}</div></div>
    <div class="card"><div class="label">Parallel time</div><div class="value">{format_us(int(summary['total_parallel_us']))}</div></div>
    <div class="card"><div class="label">Avg concurrency when active</div><div class="value">{float(summary['avg_concurrency_when_active']):.2f}</div></div>
  </div>

  <section class="panel">
    <h2>Timeline</h2>
    <svg width="{width}" height="{svg_height}" viewBox="0 0 {width} {svg_height}">
      {''.join(ticks)}
      <text x="{left_pad}" y="16" class="lane-label" text-anchor="start">Concurrency occupancy</text>
      {''.join(occupancy_rects)}
      {''.join(lane_text)}
      {''.join(rects)}
    </svg>
    <div class="note">Top chart shows instantaneous concurrency. Lower chart assigns each event to its logical lane. Overlapping bars indicate real overlap in the branch-parallel execution timeline.</div>
  </section>

  <div class="panel-grid">
    <section class="panel">
      <h2>Top Overlap Pairs</h2>
      {render_pair_table(pair_rows)}
    </section>
    <section class="panel">
      <h2>Top Parallel Segments</h2>
      {render_segment_table(occupancy_segments)}
    </section>
  </div>

  <div class="panel-grid">
    <section class="panel">
      {provider_table}
    </section>
    <section class="panel">
      {task_table}
    </section>
  </div>

  <section class="panel">
    {label_table}
  </section>
</body>
</html>
"""
    ensure_dir(path.parent)
    path.write_text(html_text, encoding="utf-8")


def task_rows(records: Sequence[TaskRecord]) -> List[dict]:
    return [
        {
            "batch_idx": r.batch_idx,
            "task_name": r.task_name,
            "phase": r.phase,
            "lane": r.lane,
            "start_us": r.start_us,
            "end_us": r.end_us,
            "dur_us": r.dur_us,
            "label": r.task_name,
        }
        for r in records
    ]


def load_profile_nodes(profile_path: Path) -> List[dict]:
    with profile_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for ev in data:
        if ev.get("cat") != "Node":
            continue
        args = ev.get("args", {})
        rows.append(
            {
                "node_name": str(ev.get("name", "") or ""),
                "op_name": str(args.get("op_name", "") or "unknown"),
                "provider": str(args.get("provider", "") or ""),
                "ts_us": int(ev.get("ts", 0) or 0),
                "dur_us": int(ev.get("dur", 0) or 0),
            }
        )
    rows.sort(key=lambda row: (row["ts_us"], row["node_name"]))
    return rows


def load_profile_events(profile_path: Path) -> List[dict]:
    with profile_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [ev for ev in data if ev.get("cat") == "Node"]


def align_profile_events_to_runs(task_name: str, lane: int, task_records: Sequence[TaskRecord], profile_nodes: Sequence[dict]) -> List[OpRecord]:
    if not task_records or not profile_nodes:
        return []

    run_count = len(task_records)
    if len(profile_nodes) % run_count != 0:
        print(f"[WARN] profile rows for {task_name} not divisible by run count; skipping op timeline alignment.")
        return []

    chunk = len(profile_nodes) // run_count
    op_records: List[OpRecord] = []
    ordered_runs = sorted(task_records, key=lambda item: item.start_us)

    for run_idx, run_record in enumerate(ordered_runs):
        nodes = profile_nodes[run_idx * chunk : (run_idx + 1) * chunk]
        if not nodes:
            continue
        base_ts = nodes[0]["ts_us"]
        for node in nodes:
            start_us = run_record.start_us + (int(node["ts_us"]) - base_ts)
            end_us = start_us + int(node["dur_us"])
            op_records.append(
                OpRecord(
                    batch_idx=run_record.batch_idx,
                    task_name=task_name,
                    op_name=str(node["op_name"]),
                    node_name=str(node["node_name"]),
                    provider=str(node["provider"]),
                    lane=lane,
                    start_us=start_us,
                    end_us=end_us,
                    dur_us=int(node["dur_us"]),
                )
            )
    return op_records


def build_merged_profile_events(
    profile_paths: Dict[str, Path],
    all_task_records: Sequence[TaskRecord],
    full_graph_by_name: Dict[str, Dict[str, str]],
    full_graph_by_name_op: Dict[Tuple[str, str], Dict[str, str]],
) -> List[dict]:
    task_records_by_name: Dict[str, List[TaskRecord]] = {}
    for record in all_task_records:
        task_records_by_name.setdefault(record.task_name, []).append(record)

    merged: List[ProfileMergeRecord] = []
    for task_name, records in task_records_by_name.items():
        profile_path = profile_paths.get(task_name)
        if not profile_path or not profile_path.exists():
            continue

        profile_events = load_profile_events(profile_path)
        node_events = [ev for ev in profile_events if ev.get("cat") == "Node"]
        run_count = len(records)
        if not node_events or run_count <= 0 or len(node_events) % run_count != 0:
            continue

        chunk = len(node_events) // run_count
        ordered_runs = sorted(records, key=lambda item: item.start_us)

        for run_idx, run_record in enumerate(ordered_runs):
            run_events = node_events[run_idx * chunk : (run_idx + 1) * chunk]
            if not run_events:
                continue
            base_ts = int(run_events[0].get("ts", 0) or 0)

            for ev in run_events:
                event = json.loads(json.dumps(ev))
                args = event.setdefault("args", {})
                raw_name = str(event.get("name", "") or "")
                normalized_name = normalize_profile_node_name(raw_name)
                op_name = str(args.get("op_name", "") or "")

                match = full_graph_by_name_op.get((normalized_name, op_name)) or full_graph_by_name.get(normalized_name)
                if match:
                    args["node_index"] = match["node_index"]
                    args["op_name"] = match["op_name"]

                shifted_ts = run_record.start_us + (int(event.get("ts", 0) or 0) - base_ts)
                event["ts"] = shifted_ts
                merged.append(
                    ProfileMergeRecord(
                        batch_idx=run_record.batch_idx,
                        task_name=task_name,
                        lane=run_record.lane,
                        event=event,
                    )
                )

    merged.sort(key=lambda item: (int(item.event.get("ts", 0) or 0), str(item.event.get("name", ""))))
    return [item.event for item in merged]


def write_merged_profile_json(profile_dir: Path, merged_events: Sequence[dict]) -> Path:
    ensure_dir(profile_dir)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = profile_dir / f"ort_cann_profile_{timestamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(merged_events), f, ensure_ascii=False)
    return path


def op_rows(records: Sequence[OpRecord]) -> List[dict]:
    return [
        {
            "batch_idx": r.batch_idx,
            "task_name": r.task_name,
            "op_name": r.op_name,
            "node_name": r.node_name,
            "provider": r.provider,
            "lane": r.lane,
            "start_us": r.start_us,
            "end_us": r.end_us,
            "dur_us": r.dur_us,
            "label": f"{r.task_name}:{r.op_name}",
        }
        for r in records
    ]


def maybe_verify_output(
    rewritten_path: Path,
    tail_output_name: str,
    providers: List[object],
    args: argparse.Namespace,
    feed: Dict[str, np.ndarray],
    ref_output: np.ndarray,
) -> None:
    sess = build_session(
        rewritten_path,
        intra_threads=args.intra_threads,
        providers=providers,
        disable_graph_optimizations=args.disable_graph_optimizations,
        enable_profiling=False,
        profile_prefix=None,
    )
    outputs = sess.run([tail_output_name], feed)
    np.testing.assert_allclose(outputs[0], ref_output, rtol=1e-5, atol=1e-5)
    print("[VERIFY] branch-parallel output matches the full rewritten model output.")


def maybe_profiled_task_specs(
    args: argparse.Namespace,
    rewritten_path: Path,
    bottom_path: Path,
    emb_paths: List[Path],
    tail_path: Path,
    layout: SplitLayout,
    providers: List[object],
    force_cpu_ops: Sequence[str],
) -> Tuple[List[TaskSpec], TaskSpec]:
    return create_task_specs(
        bottom_path=bottom_path,
        emb_paths=emb_paths,
        tail_path=tail_path,
        layout=layout,
        args=args,
        providers=providers,
        force_cpu_ops=force_cpu_ops,
        profile_enabled=args.enable_profiling,
    )


def recreate_without_profiling_if_needed(
    args: argparse.Namespace,
    rewritten_path: Path,
    bottom_path: Path,
    emb_paths: List[Path],
    tail_path: Path,
    layout: SplitLayout,
    providers: List[object],
    force_cpu_ops: Sequence[str],
) -> Tuple[List[TaskSpec], TaskSpec]:
    if not args.enable_profiling or args.profile_warmup or args.warmup_batches <= 0:
        return maybe_profiled_task_specs(
            args,
            rewritten_path,
            bottom_path,
            emb_paths,
            tail_path,
            layout,
            providers,
            force_cpu_ops,
        )

    warmup_branch_specs, warmup_tail_spec = create_task_specs(
        bottom_path=bottom_path,
        emb_paths=emb_paths,
        tail_path=tail_path,
        layout=layout,
        args=args,
        providers=providers,
        force_cpu_ops=force_cpu_ops,
        profile_enabled=False,
    )
    print("[PROFILE] warmup will not be included in profiling; running warmup on non-profiled sessions.")
    run_parallel_workload(
        args=args,
        rewritten_path=rewritten_path,
        branch_specs=warmup_branch_specs,
        tail_spec=warmup_tail_spec,
        collect_records=False,
        start_batch_idx=-args.warmup_batches,
        run_batches=args.warmup_batches,
    )
    return maybe_profiled_task_specs(
        args,
        rewritten_path,
        bottom_path,
        emb_paths,
        tail_path,
        layout,
        providers,
        force_cpu_ops,
    )


def run_parallel_workload(
    args: argparse.Namespace,
    rewritten_path: Path,
    branch_specs: Sequence[TaskSpec],
    tail_spec: TaskSpec,
    collect_records: bool,
    start_batch_idx: int,
    run_batches: int,
) -> Tuple[List[TaskRecord], List[float], np.ndarray | None, Dict[str, np.ndarray] | None]:
    max_parallel_branches = min(
        args.parallel_branches if args.parallel_branches > 0 else max(1, args.inter_threads),
        len(branch_specs),
    )
    if args.verbose:
        print(f"[PAR] max concurrent branch tasks: {max_parallel_branches}")

    input_session = build_session(
        rewritten_path,
        intra_threads=1,
        providers=["CPUExecutionProvider"],
        disable_graph_optimizations=False,
        enable_profiling=False,
        profile_prefix=None,
    )

    all_records: List[TaskRecord] = []
    latencies_ms: List[float] = []
    last_output: np.ndarray | None = None
    last_feed: Dict[str, np.ndarray] | None = None

    for offset in range(run_batches):
        batch_idx = start_batch_idx + offset
        if batch_idx < 0:
            seed = -batch_idx - 1
        else:
            seed = 1000 + batch_idx

        full_feed = generate_inputs(
            input_session,
            args.batch_size,
            seed=seed,
            onnx_path=str(rewritten_path),
            bag_size=args.num_indices_per_lookup,
        )

        batch_start_ns = time.perf_counter_ns()
        branch_results, batch_records = run_branch_stage(branch_specs, full_feed, batch_idx, max_parallel_branches)
        tail_feed = {name: branch_results[name] for name in tail_spec.input_names}
        tail_output, tail_record = run_task(tail_spec, tail_feed, batch_idx)
        batch_records.append(tail_record)
        batch_end_ns = time.perf_counter_ns()

        if collect_records:
            all_records.extend(batch_records)
            elapsed_ms = (batch_end_ns - batch_start_ns) / 1_000_000
            latencies_ms.append(elapsed_ms)
            max_concurrency, parallel_us = summarize_records(batch_records)
            print(
                f"  Batch {batch_idx:3d}: {elapsed_ms:8.2f} ms  "
                f"max_concurrency={max_concurrency}  parallel_us={parallel_us}"
            )

        last_output = tail_output
        last_feed = full_feed

    return all_records, latencies_ms, last_output, last_feed


def end_profiling(task_specs: Sequence[TaskSpec], tail_spec: TaskSpec) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in list(task_specs) + [tail_spec]:
        profiler = getattr(spec.session, "end_profiling", None)
        if callable(profiler):
            profile_path = Path(profiler())
            out[spec.name] = profile_path
    return out


def build_op_timeline(
    profile_paths: Dict[str, Path],
    all_task_records: Sequence[TaskRecord],
    lane_labels: Dict[int, str],
) -> List[OpRecord]:
    op_records: List[OpRecord] = []
    task_records_by_name: Dict[str, List[TaskRecord]] = {}
    for record in all_task_records:
        task_records_by_name.setdefault(record.task_name, []).append(record)

    for task_name, records in task_records_by_name.items():
        profile_path = profile_paths.get(task_name)
        if not profile_path or not profile_path.exists():
            continue
        lane = records[0].lane
        nodes = load_profile_nodes(profile_path)
        op_records.extend(align_profile_events_to_runs(task_name, lane, records, nodes))
    return op_records


def main() -> None:
    args = parse_args()

    out_dir = choose_output_dir(args)
    profile_dir = Path(args.profile_dir).resolve()
    submodel_dir = Path(args.submodel_dir).resolve()
    ensure_dir(out_dir)
    ensure_dir(submodel_dir)
    ensure_dir(profile_dir)

    providers = build_providers(args.use_cann, args.device_id)

    print("=" * 70)
    print("DLRM Branch-Parallel ORT Runner")
    print("=" * 70)
    print(f"  ONNX 模型     : {args.onnx_path}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Batches       : {args.num_batches}（warmup={args.warmup_batches}）")
    print(f"  Use CANN      : {'启用' if args.use_cann else '禁用（仅 CPU）'}")
    print(f"  Replace Loop  : {'启用' if not args.no_replace_loop else '禁用'}")
    print(f"  Intra threads : {args.intra_threads}")
    print(f"  Inter threads : {args.inter_threads}")
    if args.parallel_branches > 0:
        print(f"  Parallel cap  : {args.parallel_branches}")
    print(f"  Profiling     : {'启用' if args.enable_profiling else '禁用'}")
    print(f"  Graph Opt     : {'禁用' if args.disable_graph_optimizations else '启用（默认）'}")

    onnx_path = Path(args.onnx_path).resolve()
    effective_path = onnx_path
    if args.use_cann:
        effective_path = Path(_patch_model_for_cann(str(effective_path))).resolve()
    if not args.no_replace_loop:
        effective_path = Path(_replace_loop_with_gather(str(effective_path), override_bag_size=args.num_indices_per_lookup)).resolve()
    ops = [t.strip() for t in args.force_cpu_ops.split(",") if t.strip()]
    if args.use_cann and ops:
        effective_path = Path(_force_ops_to_cpu(str(effective_path), ops)).resolve()
    print(f"[ORT] 加载模型: {effective_path}")
    print(f"[PAR] using model: {effective_path}")

    if args.shape_csv:
        dump_op_shapes_to_csv(
            str(effective_path),
            args.shape_csv,
            batch_size=args.batch_size,
            bag_size=args.num_indices_per_lookup,
        )

    layout = infer_split_layout(effective_path)
    print(f"[PAR] inferred bottom split: {layout.bottom_input_name} -> {layout.bottom_output_name}")
    print(f"[PAR] inferred emb splits   : {', '.join(f'{name}->{out}' for name, out in zip(layout.emb_input_names, layout.emb_output_names))}")
    print(f"[PAR] inferred tail output : {layout.tail_output_name}")

    bottom_path, emb_paths, tail_path = prepare_submodels(effective_path, submodel_dir, layout)
    branch_specs, tail_spec = recreate_without_profiling_if_needed(
        args=args,
        rewritten_path=effective_path,
        bottom_path=bottom_path,
        emb_paths=emb_paths,
        tail_path=tail_path,
        layout=layout,
        providers=providers,
        force_cpu_ops=ops,
    )

    warmup_records: List[TaskRecord] = []
    if args.warmup_batches > 0 and (not args.enable_profiling or args.profile_warmup):
        print(f"[PAR] warmup: {args.warmup_batches} batches")
        warmup_records, _, _, _ = run_parallel_workload(
            args=args,
            rewritten_path=effective_path,
            branch_specs=branch_specs,
            tail_spec=tail_spec,
            collect_records=args.enable_profiling and args.profile_warmup,
            start_batch_idx=-args.warmup_batches,
            run_batches=args.warmup_batches,
        )

    print(f"[PAR] inference: {args.num_batches} batches, batch_size={args.batch_size}")
    task_records, latencies_ms, last_output, last_feed = run_parallel_workload(
        args=args,
        rewritten_path=effective_path,
        branch_specs=branch_specs,
        tail_spec=tail_spec,
        collect_records=True,
        start_batch_idx=0,
        run_batches=args.num_batches,
    )

    all_task_records = warmup_records + task_records
    print_statistics(latencies_ms)

    if args.verify_full_output and last_output is not None and last_feed is not None:
        maybe_verify_output(effective_path, layout.tail_output_name, providers, args, last_feed, last_output)

    task_timeline_csv = out_dir / "branch_parallel_timeline.csv"
    task_timeline_html = out_dir / "branch_parallel_timeline.html"
    task_segments_csv = out_dir / "branch_parallel_concurrency_segments.csv"
    task_lane_labels = {spec.lane: spec.name for spec in list(branch_specs) + [tail_spec]}
    task_rows_data = task_rows(all_task_records)
    task_segments = [seg for seg in build_occupancy_segments(task_rows_data) if int(seg["concurrency"]) >= 2]
    write_csv(task_timeline_csv, task_rows_data)
    write_csv(task_segments_csv, task_segments)
    render_timeline_html(
        task_timeline_html,
        task_rows_data,
        task_lane_labels,
        "Branch Parallel Task Timeline",
        source_note=f"Task-level execution from {effective_path}",
    )

    total_max_concurrency, total_parallel_us = summarize_records(all_task_records)
    print(f"[PAR] total task max concurrency : {total_max_concurrency}")
    print(f"[PAR] total task parallel time   : {total_parallel_us} us")
    print(f"[PAR] saved task timeline CSV   : {task_timeline_csv}")
    print(f"[PAR] saved task segments CSV   : {task_segments_csv}")
    print(f"[PAR] saved task timeline HTML  : {task_timeline_html}")

    if args.enable_profiling:
        profile_paths = end_profiling(branch_specs, tail_spec)
        op_timeline = build_op_timeline(profile_paths, all_task_records, task_lane_labels)
        full_graph_by_name, full_graph_by_name_op = build_full_graph_node_lookup(effective_path)
        merged_profile_events = build_merged_profile_events(
            profile_paths=profile_paths,
            all_task_records=all_task_records,
            full_graph_by_name=full_graph_by_name,
            full_graph_by_name_op=full_graph_by_name_op,
        )
        merged_profile_json = write_merged_profile_json(profile_dir, merged_profile_events)
        op_timeline_csv = out_dir / "branch_parallel_op_timeline.csv"
        op_timeline_html = out_dir / "branch_parallel_op_timeline.html"
        op_segments_csv = out_dir / "branch_parallel_op_concurrency_segments.csv"
        op_rows_data = op_rows(op_timeline)
        op_segments = [seg for seg in build_occupancy_segments(op_rows_data) if int(seg["concurrency"]) >= 2]
        write_csv(op_timeline_csv, op_rows_data)
        write_csv(op_segments_csv, op_segments)
        render_timeline_html(
            op_timeline_html,
            op_rows_data,
            task_lane_labels,
            "Branch Parallel Operator Timeline",
            source_note=f"Operator-level timeline reconstructed from per-submodel ORT profiles in {profile_dir}",
        )
        op_max_concurrency, op_parallel_us = summarize_records(op_timeline)
        print(f"[PAR] total op max concurrency   : {op_max_concurrency}")
        print(f"[PAR] total op parallel time     : {op_parallel_us} us")
        print(f"[PAR] saved op timeline CSV     : {op_timeline_csv}")
        print(f"[PAR] saved op segments CSV     : {op_segments_csv}")
        print(f"[PAR] saved op timeline HTML    : {op_timeline_html}")
        print(f"[PAR] merged profile JSON       : {merged_profile_json}")
        for name, profile_path in sorted(profile_paths.items()):
            print(f"[PAR] profile {name:<8}: {profile_path}")


if __name__ == "__main__":
    main()
