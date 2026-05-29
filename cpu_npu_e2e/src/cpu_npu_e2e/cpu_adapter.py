from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .shapes import GraphNode, infer_gemm_mnk, tensor_bytes, tensor_elements


CASE_THREADS_RE = re.compile(r"case_\d+_run_(?P<num_threads>\d+)_(?P<inter_threads>\d+)")


@dataclass(frozen=True)
class CpuStaticConfig:
    batch_size: int
    num_indices_per_lookup: int
    arch_embedding_size: str
    arch_mlp_bot: str
    arch_mlp_top: str
    num_threads: int
    inter_threads: int


@dataclass(frozen=True)
class CpuStaticModelResult:
    predictions: pd.DataFrame
    summary: dict[str, object]


class CpuLatencyAdapter:
    """CPU latency source.

    The formal path reads exported CPU prediction rows. A deterministic static
    fallback is kept for smoke tests and shape-only integration checks when the
    CPU delivery does not include prediction CSVs.
    """

    PRED_COLS = (
        "pred_us",
        "prediction_us",
        "predicted_us",
        "predicted_latency_us",
        "prediction",
        "y_pred",
    )

    def __init__(
        self,
        prediction_csv: str | Path | None = None,
        *,
        prediction_frame: pd.DataFrame | None = None,
        prediction_source: str = "cpu_prediction_csv",
        allow_static_fallback: bool = True,
    ) -> None:
        if prediction_csv is not None and prediction_frame is not None:
            raise ValueError("prediction_csv and prediction_frame are mutually exclusive")
        self.prediction_csv = Path(prediction_csv) if prediction_csv else None
        self.allow_static_fallback = bool(allow_static_fallback)
        self._pred_by_idx: dict[int, float] = {}
        self.prediction_source = prediction_source
        self.prediction_frame = pd.DataFrame()
        self.uses_static_fallback = False
        if self.prediction_csv is not None:
            self.prediction_frame = pd.read_csv(self.prediction_csv)
            self._pred_by_idx = self._load_predictions(self.prediction_frame, str(self.prediction_csv))
        elif prediction_frame is not None:
            self.prediction_frame = prediction_frame.copy()
            self._pred_by_idx = self._load_predictions(self.prediction_frame, "prediction_frame")

    def _load_predictions(self, frame: pd.DataFrame, source: str) -> dict[int, float]:
        pred_col = next((col for col in self.PRED_COLS if col in frame.columns), None)
        if pred_col is None:
            raise ValueError(f"{source} has no supported prediction column: {self.PRED_COLS}")
        if "op_idx" in frame.columns:
            op_idx = pd.to_numeric(frame["op_idx"], errors="coerce")
        elif "node_idx" in frame.columns:
            op_idx = pd.to_numeric(frame["node_idx"], errors="coerce")
        elif "row_uid" in frame.columns:
            op_idx = frame["row_uid"].astype(str).str.extract(r"opidx:(\d+)")[0].astype(float)
        else:
            raise ValueError(f"{source} must contain op_idx, node_idx, or row_uid")
        pred = pd.to_numeric(frame[pred_col], errors="coerce")
        out: dict[int, float] = {}
        for idx, value in zip(op_idx, pred):
            if np.isfinite(idx) and np.isfinite(value):
                out[int(idx)] = max(float(value), 0.0)
        if not out:
            raise ValueError(f"{source} did not contain usable predictions")
        return out

    def predict_us(self, node: GraphNode) -> tuple[float, str]:
        if node.node_idx in self._pred_by_idx:
            return self._pred_by_idx[node.node_idx], self.prediction_source
        if not self.allow_static_fallback:
            raise KeyError(f"missing CPU prediction for node_idx={node.node_idx}")
        self.uses_static_fallback = True
        return self._static_estimate_us(node), "cpu_static_fallback"

    def _static_estimate_us(self, node: GraphNode) -> float:
        input_bytes = tensor_bytes(node.inputs)
        output_bytes = tensor_bytes(node.outputs)
        elems = max(tensor_elements(node.outputs), tensor_elements(node.inputs), 1)
        bytes_rw = max(input_bytes + output_bytes, 1)
        dispatch_us = 1.5
        stream_bw_bytes_per_us = 60_000.0
        random_bw_bytes_per_us = 18_000.0
        flops_per_us = 220_000.0

        if node.op_type in {"Shape", "Flatten", "Unsqueeze", "Reshape"}:
            return dispatch_us + min(bytes_rw / stream_bw_bytes_per_us, 3.0)
        if node.op_type in {"Gemm", "MatMul"}:
            m, k, n = infer_gemm_mnk(node)
            flops = max(2.0 * m * k * n, float(elems))
            return dispatch_us + max(bytes_rw / stream_bw_bytes_per_us, flops / flops_per_us)
        if node.op_type == "Gather":
            return dispatch_us + bytes_rw / random_bw_bytes_per_us
        if node.op_type == "ReduceSum":
            return dispatch_us + max(bytes_rw / stream_bw_bytes_per_us, elems / flops_per_us)
        return dispatch_us + max(bytes_rw / stream_bw_bytes_per_us, elems / flops_per_us)


def _output_dims(node: GraphNode) -> tuple[int, ...]:
    if not node.outputs:
        raise ValueError(f"{node.node_idx}:{node.node_name} has no output tensor")
    return node.outputs[0].dims


def _input_dims(node: GraphNode, port_idx: int) -> tuple[int, ...]:
    for spec in node.inputs:
        if int(spec.port_idx) == int(port_idx):
            return spec.dims
    raise ValueError(f"{node.node_idx}:{node.node_name} has no input tensor at port {port_idx}")


def _node_by_name(graph: dict[int, GraphNode], name: str) -> GraphNode:
    for node in graph.values():
        if node.node_name == name:
            return node
    raise KeyError(f"node {name!r} not found in graph")


def _parse_threads(shape_csv: Path, num_threads: int | None, inter_threads: int | None) -> tuple[int, int]:
    parsed_num = num_threads
    parsed_inter = inter_threads
    for part in reversed(shape_csv.parts):
        match = CASE_THREADS_RE.fullmatch(part)
        if match:
            parsed_num = parsed_num if parsed_num is not None else int(match.group("num_threads"))
            parsed_inter = parsed_inter if parsed_inter is not None else int(match.group("inter_threads"))
            break
    return int(parsed_num if parsed_num is not None else 1), int(parsed_inter if parsed_inter is not None else 1)


def infer_cpu_static_config_from_graph(
    graph: dict[int, GraphNode],
    *,
    shape_csv: Path,
    num_threads: int | None = None,
    inter_threads: int | None = None,
) -> CpuStaticConfig:
    embedding_sizes: list[int] = []
    batch_size: int | None = None
    num_indices: int | None = None
    for branch_idx in range(8):
        gather = _node_by_name(graph, f"/emb_l{branch_idx}/Gather")
        table_shape = _input_dims(gather, 0)
        if len(table_shape) < 2:
            raise ValueError(f"{gather.node_name} table shape is not rank-2: {table_shape}")
        embedding_sizes.append(int(table_shape[0]))
        reshape = _node_by_name(graph, f"/emb_l{branch_idx}/Reshape")
        reshape_out = _output_dims(reshape)
        if len(reshape_out) != 3:
            raise ValueError(f"{reshape.node_name} output shape is not rank-3: {reshape_out}")
        batch_size = int(reshape_out[0]) if batch_size is None else batch_size
        num_indices = int(reshape_out[1]) if num_indices is None else num_indices

    bot0 = _node_by_name(graph, "/bot_l/bot_l.0/Gemm")
    bot2 = _node_by_name(graph, "/bot_l/bot_l.2/Gemm")
    dense_dim = int(_input_dims(bot0, 0)[1])
    bot_hidden = int(_output_dims(bot0)[1])
    bot_out = int(_output_dims(bot2)[1])

    top0 = _node_by_name(graph, "/top_l/top_l.0/Gemm")
    top2 = _node_by_name(graph, "/top_l/top_l.2/Gemm")
    top4 = _node_by_name(graph, "/top_l/top_l.4/Gemm")
    top_dims = [int(_output_dims(top0)[1]), int(_output_dims(top2)[1]), int(_output_dims(top4)[1])]
    parsed_num_threads, parsed_inter_threads = _parse_threads(shape_csv, num_threads, inter_threads)

    if batch_size is None or num_indices is None:
        raise ValueError("failed to infer batch_size and num_indices_per_lookup from embedding reshape nodes")

    return CpuStaticConfig(
        batch_size=int(batch_size),
        num_indices_per_lookup=int(num_indices),
        arch_embedding_size="-".join(str(value) for value in embedding_sizes),
        arch_mlp_bot=f"{dense_dim}-{bot_hidden}-{bot_out}",
        arch_mlp_top="-".join(str(value) for value in top_dims),
        num_threads=parsed_num_threads,
        inter_threads=parsed_inter_threads,
    )


def _import_cpu_static_inference(cpu_model_root: Path):
    root = str(cpu_model_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from calibrator.static_feature_mapping.core import (  # type: ignore
        FULL_GRAPH_76,
        DlrmStaticConfig,
        build_static_feature_frame,
    )
    from calibrator.static_feature_mapping.predict_from_config import (  # type: ignore
        _bundle_op_thread_scale_enabled,
        _load_model_bundle,
        _predict_with_bundle,
    )

    return FULL_GRAPH_76, DlrmStaticConfig, build_static_feature_frame, _load_model_bundle, _predict_with_bundle, _bundle_op_thread_scale_enabled


def _validate_prediction_alignment(
    *,
    graph: dict[int, GraphNode],
    predictions: pd.DataFrame,
    required_node_ids: set[int],
) -> None:
    meta = {
        int(row.op_idx): (str(row.node_name), str(row.op_type))
        for row in predictions[["op_idx", "node_name", "op_type"]].itertuples(index=False)
    }
    missing = sorted(idx for idx in required_node_ids if idx not in meta)
    if missing:
        raise KeyError(f"CPU model bundle prediction is missing required CPU nodes: {missing}")
    mismatches: list[str] = []
    for idx, node in sorted(graph.items()):
        if idx not in meta:
            continue
        pred_name, pred_op = meta[idx]
        if pred_name != node.node_name or pred_op != node.op_type:
            mismatches.append(f"{idx}: graph=({node.node_name},{node.op_type}) bundle=({pred_name},{pred_op})")
    if mismatches:
        raise ValueError("CPU model bundle full_graph_76 does not align with op_shapes.csv: " + "; ".join(mismatches[:5]))


def run_cpu_static_model_inference(
    *,
    cpu_model_root: Path,
    model_bundle_path: Path,
    graph: dict[int, GraphNode],
    shape_csv: Path,
    required_node_ids: set[int],
    num_threads: int | None = None,
    inter_threads: int | None = None,
) -> CpuStaticModelResult:
    if not model_bundle_path.exists():
        raise FileNotFoundError(f"CPU model bundle not found: {model_bundle_path}")
    config = infer_cpu_static_config_from_graph(
        graph,
        shape_csv=shape_csv,
        num_threads=num_threads,
        inter_threads=inter_threads,
    )
    (
        full_graph_scope,
        dlrm_static_config_cls,
        build_static_feature_frame,
        load_model_bundle,
        predict_with_bundle,
        bundle_op_thread_scale_enabled,
    ) = _import_cpu_static_inference(cpu_model_root)
    cpu_config = dlrm_static_config_cls(
        batch_size=config.batch_size,
        num_indices_per_lookup=config.num_indices_per_lookup,
        arch_embedding_size=config.arch_embedding_size,
        arch_mlp_bot=config.arch_mlp_bot,
        arch_mlp_top=config.arch_mlp_top,
        inter_threads=config.inter_threads,
        num_threads=config.num_threads,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        bundle = load_model_bundle(model_bundle_path)
        full_graph = build_static_feature_frame(cpu_config, graph_scope=full_graph_scope)
        prepared, pred = predict_with_bundle(bundle, full_graph)

    predictions = prepared.copy()
    predictions["pred_us"] = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    predictions["pred_total_us"] = predictions["pred_us"]
    predictions["cpu_prediction_source"] = "cpu_model_bundle_full_graph_76"
    _validate_prediction_alignment(graph=graph, predictions=predictions, required_node_ids=required_node_ids)
    required_predictions = predictions[predictions["op_idx"].astype(int).isin(required_node_ids)]
    summary = {
        "cpu_prediction_source": "cpu_model_bundle_full_graph_76",
        "cpu_model_bundle": str(model_bundle_path),
        "graph_scope": str(full_graph_scope),
        "full_graph_rows": int(len(predictions)),
        "required_cpu_node_count": int(len(required_node_ids)),
        "required_cpu_pred_us_sum": float(required_predictions["pred_us"].sum()),
        "operator_pred_us_sum": float(predictions["pred_us"].sum()),
        "model_sha256": str(bundle.get("model_sha256", "")),
        "route": str(bundle.get("route", "")),
        "op_thread_scale_enabled": bool(bundle_op_thread_scale_enabled(bundle)),
        "config": {
            "batch_size": config.batch_size,
            "num_indices_per_lookup": config.num_indices_per_lookup,
            "arch_embedding_size": config.arch_embedding_size,
            "arch_mlp_bot": config.arch_mlp_bot,
            "arch_mlp_top": config.arch_mlp_top,
            "num_threads": config.num_threads,
            "inter_threads": config.inter_threads,
        },
    }
    return CpuStaticModelResult(predictions=predictions, summary=summary)
