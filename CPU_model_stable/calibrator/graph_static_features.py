from __future__ import annotations

import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NODE_REF_RE = re.compile(r"^\s*(\d+):([^:]+):([^:]+)\s*$")
EMB_SCOPE_RE = re.compile(r"^/emb_l\d+/")

EXEC_MODE_ORDER = [
    "metadata_view",
    "tiny_shape_arith",
    "indexed_block_copy",
    "payload_concat",
    "reduction_layout",
    "transpose_generic",
    "gemm_like",
]
EXEC_MODE_TO_CODE = {name: index + 1 for index, name in enumerate(EXEC_MODE_ORDER)}

SUBMODEL_ROLE_ORDER = [
    "unknown",
    "embedding_branch",
    "mlp_bot",
    "interaction",
    "tail",
    "mlp_top",
]
SUBMODEL_ROLE_TO_CODE = {name: index for index, name in enumerate(SUBMODEL_ROLE_ORDER)}

OP_TYPE_ORDER = [
    "Unknown",
    "Constant",
    "Shape",
    "Unsqueeze",
    "Flatten",
    "Reshape",
    "Gather",
    "ReduceSum",
    "Concat",
    "Transpose",
    "Add",
    "Mul",
    "MatMul",
    "Gemm",
    "Relu",
    "Sigmoid",
]
OP_TYPE_TO_CODE = {name: index for index, name in enumerate(OP_TYPE_ORDER)}

GRAPH_FEATURE_COLUMNS = [
    "pred_op_type_primary",
    "succ_op_type_primary",
    "pred_exec_mode_primary",
    "succ_exec_mode_primary",
    "pred_op_type_primary_code",
    "succ_op_type_primary_code",
    "pred_exec_mode_primary_code",
    "succ_exec_mode_primary_code",
    "pred_count",
    "succ_count",
    "cross_scope_pred_count",
    "cross_scope_succ_count",
    "topo_pos_norm",
    "reverse_topo_pos_norm",
    "distance_to_output",
    "distance_to_nearest_merge",
    "distance_to_nearest_concat",
    "distance_to_nearest_reduce",
    "distance_to_nearest_gather",
    "nearest_merge_fanin",
    "graph_feature_mapped_flag",
]

for hop in ("1hop", "2hop"):
    for exec_mode in EXEC_MODE_ORDER:
        GRAPH_FEATURE_COLUMNS.append(f"graph_{hop}_{exec_mode}_count")


def _normalized_node_name(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return int(default)


def _parse_node_ref(value: Any) -> tuple[int, str, str] | None:
    text = str(value or "").strip()
    if not text or text in {"initializer", "graph_input", "graph_output", "nan", "None"}:
        return None
    match = NODE_REF_RE.match(text)
    if match is None:
        return None
    return int(match.group(1)), str(match.group(2)).strip(), str(match.group(3)).strip()


def _parse_consumer_nodes(value: Any) -> list[tuple[int, str, str]]:
    text = str(value or "").strip()
    if not text or text in {"nan", "None"}:
        return []
    refs: list[tuple[int, str, str]] = []
    for token in text.split(","):
        parsed = _parse_node_ref(token)
        if parsed is not None:
            refs.append(parsed)
    return refs


def _graph_scope(node_name: str, op_type: str) -> str:
    if EMB_SCOPE_RE.match(node_name):
        return node_name.split("/", 2)[1]
    if "/bot_l/" in node_name:
        return "bot_l"
    if "/top_l/" in node_name:
        return "top_l"
    segments = [segment for segment in node_name.split("/") if segment]
    if not segments:
        return str(op_type or "unknown")
    return str(segments[0])


def _exec_mode_from_op_type(op_type: str) -> str:
    text = str(op_type or "")
    if text in {"Shape", "Unsqueeze", "Flatten", "Reshape", "Constant"}:
        return "metadata_view"
    if text in {"Add", "Mul"}:
        return "tiny_shape_arith"
    if text == "Gather":
        return "indexed_block_copy"
    if text == "Concat":
        return "payload_concat"
    if text == "ReduceSum":
        return "reduction_layout"
    if text == "Transpose":
        return "transpose_generic"
    if text in {"Gemm", "MatMul"}:
        return "gemm_like"
    return "metadata_view"


def _role_from_node_name(node_name: str, op_type: str) -> str:
    if EMB_SCOPE_RE.match(node_name):
        return "embedding_branch"
    if "/bot_l/" in node_name:
        return "mlp_bot"
    if "/top_l/" in node_name:
        return "mlp_top"
    if op_type in {"Gather", "MatMul", "Transpose", "Add", "Mul", "Shape", "Unsqueeze", "Reshape", "Flatten"}:
        return "interaction"
    return "tail"


def _code(mapping: dict[str, int], value: str, default: int = 0) -> float:
    return float(mapping.get(str(value), int(default)))


def _distance_to_predicate(
    start_name: str,
    adjacency: dict[str, set[str]],
    *,
    node_index_map: dict[str, int],
    predicate: Any,
) -> int:
    if predicate(start_name):
        return 0
    visited = {start_name}
    queue: deque[tuple[str, int]] = deque([(start_name, 0)])
    while queue:
        current, distance = queue.popleft()
        next_names = sorted(adjacency.get(current, set()), key=lambda item: node_index_map.get(item, 10**9))
        for next_name in next_names:
            if next_name in visited:
                continue
            next_distance = distance + 1
            if predicate(next_name):
                return int(next_distance)
            visited.add(next_name)
            queue.append((next_name, next_distance))
    return -1


def _distance_to_output(
    start_name: str,
    adjacency: dict[str, set[str]],
    *,
    node_index_map: dict[str, int],
    sink_nodes: set[str],
) -> int:
    return _distance_to_predicate(
        start_name,
        adjacency,
        node_index_map=node_index_map,
        predicate=lambda name: name in sink_nodes,
    )


@lru_cache(maxsize=2048)
def graph_feature_table_for_shape_csv(shape_csv_text: str) -> pd.DataFrame:
    path = Path(shape_csv_text)
    if not path.exists():
        return pd.DataFrame(columns=["op_idx", "node_name_normalized", "submodel_role_static", "submodel_role_static_code", *GRAPH_FEATURE_COLUMNS])

    frame = pd.read_csv(
        path,
        usecols=[
            "node_idx",
            "node_name",
            "op_type",
            "tensor_role",
            "producer_node",
            "consumer_nodes",
        ],
        low_memory=False,
    )
    return graph_feature_table_from_op_shapes_frame(frame)


def graph_feature_table_from_op_shapes_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "node_idx",
        "node_name",
        "op_type",
        "tensor_role",
        "producer_node",
        "consumer_nodes",
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"op shape frame is missing graph columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["node_name_normalized"] = frame["node_name"].astype(str).map(_normalized_node_name)
    frame["node_idx"] = pd.to_numeric(frame["node_idx"], errors="coerce").fillna(-1).astype(int)

    node_rows: list[dict[str, Any]] = []
    for (node_idx, node_name_norm), group in frame.groupby(["node_idx", "node_name_normalized"], sort=True):
        if int(node_idx) < 0:
            continue
        node_name = str(node_name_norm)
        op_type = str(group["op_type"].dropna().astype(str).iloc[0]) if not group["op_type"].dropna().empty else "Unknown"
        preds: set[str] = set()
        succs: set[str] = set()
        pred_types: dict[str, str] = {}
        succ_types: dict[str, str] = {}
        for _, row in group.iterrows():
            if str(row.get("tensor_role", "")).strip().lower() == "input":
                parsed = _parse_node_ref(row.get("producer_node"))
                if parsed is not None:
                    _, pred_name, pred_op = parsed
                    preds.add(pred_name)
                    pred_types[pred_name] = pred_op
            if str(row.get("tensor_role", "")).strip().lower() == "output":
                for _, succ_name, succ_op in _parse_consumer_nodes(row.get("consumer_nodes")):
                    succs.add(succ_name)
                    succ_types[succ_name] = succ_op
        node_rows.append(
            {
                "op_idx": int(node_idx),
                "node_name_normalized": node_name,
                "op_type": op_type,
                "exec_mode": _exec_mode_from_op_type(op_type),
                "scope": _graph_scope(node_name, op_type),
                "submodel_role_static": _role_from_node_name(node_name, op_type),
                "preds": preds,
                "succs": succs,
                "pred_types": pred_types,
                "succ_types": succ_types,
            }
        )

    if not node_rows:
        return pd.DataFrame(columns=["op_idx", "node_name_normalized", "submodel_role_static", "submodel_role_static_code", *GRAPH_FEATURE_COLUMNS])

    node_table = pd.DataFrame(node_rows).sort_values("op_idx").reset_index(drop=True)
    node_name_to_idx = {str(row["node_name_normalized"]): int(row["op_idx"]) for _, row in node_table.iterrows()}
    node_name_to_scope = {str(row["node_name_normalized"]): str(row["scope"]) for _, row in node_table.iterrows()}
    node_name_to_op = {str(row["node_name_normalized"]): str(row["op_type"]) for _, row in node_table.iterrows()}
    node_name_to_mode = {str(row["node_name_normalized"]): str(row["exec_mode"]) for _, row in node_table.iterrows()}
    node_name_to_fanin = {str(row["node_name_normalized"]): max(len(set(row["preds"])), 0) for _, row in node_table.iterrows()}
    adjacency = {str(row["node_name_normalized"]): set(str(item) for item in row["succs"]) for _, row in node_table.iterrows()}
    max_idx = max(int(node_table["op_idx"].max()), 1)
    sink_nodes = {name for name, succs in adjacency.items() if not succs}

    exec_mode_index = {name: idx for idx, name in enumerate(EXEC_MODE_ORDER)}
    records: list[dict[str, Any]] = []
    for _, row in node_table.iterrows():
        node_name = str(row["node_name_normalized"])
        current_scope = str(row["scope"])
        preds = sorted(set(row["preds"]), key=lambda item: node_name_to_idx.get(str(item), -1))
        succs = sorted(set(row["succs"]), key=lambda item: node_name_to_idx.get(str(item), 10**9))

        pred_primary = preds[-1] if preds else ""
        succ_primary = succs[0] if succs else ""

        pred_count = float(len(preds))
        succ_count = float(len(succs))
        cross_scope_pred_count = float(sum(1 for name in preds if node_name_to_scope.get(str(name), "") != current_scope))
        cross_scope_succ_count = float(sum(1 for name in succs if node_name_to_scope.get(str(name), "") != current_scope))

        topo_pos_norm = float(int(row["op_idx"]) / max_idx)
        reverse_topo_pos_norm = float((max_idx - int(row["op_idx"])) / max_idx)
        distance_to_output = float(_distance_to_output(node_name, adjacency, node_index_map=node_name_to_idx, sink_nodes=sink_nodes))
        distance_to_nearest_merge = float(
            _distance_to_predicate(
                node_name,
                adjacency,
                node_index_map=node_name_to_idx,
                predicate=lambda name: node_name_to_fanin.get(str(name), 0) > 1,
            )
        )
        distance_to_nearest_concat = float(
            _distance_to_predicate(
                node_name,
                adjacency,
                node_index_map=node_name_to_idx,
                predicate=lambda name: node_name_to_op.get(str(name), "") == "Concat",
            )
        )
        distance_to_nearest_reduce = float(
            _distance_to_predicate(
                node_name,
                adjacency,
                node_index_map=node_name_to_idx,
                predicate=lambda name: node_name_to_op.get(str(name), "") == "ReduceSum",
            )
        )
        distance_to_nearest_gather = float(
            _distance_to_predicate(
                node_name,
                adjacency,
                node_index_map=node_name_to_idx,
                predicate=lambda name: node_name_to_op.get(str(name), "") == "Gather",
            )
        )

        nearest_merge_fanin = 0.0
        if distance_to_nearest_merge >= 0.0:
            visited = {node_name}
            queue: deque[str] = deque([node_name])
            found_merge = ""
            while queue and not found_merge:
                current = queue.popleft()
                next_names = sorted(adjacency.get(current, set()), key=lambda item: node_name_to_idx.get(item, 10**9))
                for next_name in next_names:
                    if next_name in visited:
                        continue
                    if node_name_to_fanin.get(next_name, 0) > 1:
                        found_merge = next_name
                        break
                    visited.add(next_name)
                    queue.append(next_name)
            if found_merge:
                nearest_merge_fanin = float(node_name_to_fanin.get(found_merge, 0))

        hop1 = [node_name_to_mode.get(name, "metadata_view") for name in [*preds, *succs]]
        hop2_names: set[str] = set()
        for neighbor in [*preds, *succs]:
            hop2_names.update(adjacency.get(neighbor, set()))
        hop2_names.difference_update({node_name, *preds, *succs})
        hop2 = [node_name_to_mode.get(name, "metadata_view") for name in sorted(hop2_names, key=lambda item: node_name_to_idx.get(item, 10**9))]

        record: dict[str, Any] = {
            "op_idx": int(row["op_idx"]),
            "node_name_normalized": node_name,
            "submodel_role_static": str(row["submodel_role_static"]),
            "submodel_role_static_code": _code(SUBMODEL_ROLE_TO_CODE, str(row["submodel_role_static"])),
            "pred_op_type_primary": node_name_to_op.get(pred_primary, "Unknown") if pred_primary else "Unknown",
            "succ_op_type_primary": node_name_to_op.get(succ_primary, "Unknown") if succ_primary else "Unknown",
            "pred_exec_mode_primary": node_name_to_mode.get(pred_primary, "metadata_view") if pred_primary else "metadata_view",
            "succ_exec_mode_primary": node_name_to_mode.get(succ_primary, "metadata_view") if succ_primary else "metadata_view",
            "pred_op_type_primary_code": _code(OP_TYPE_TO_CODE, node_name_to_op.get(pred_primary, "Unknown") if pred_primary else "Unknown"),
            "succ_op_type_primary_code": _code(OP_TYPE_TO_CODE, node_name_to_op.get(succ_primary, "Unknown") if succ_primary else "Unknown"),
            "pred_exec_mode_primary_code": _code(EXEC_MODE_TO_CODE, node_name_to_mode.get(pred_primary, "metadata_view") if pred_primary else "metadata_view"),
            "succ_exec_mode_primary_code": _code(EXEC_MODE_TO_CODE, node_name_to_mode.get(succ_primary, "metadata_view") if succ_primary else "metadata_view"),
            "pred_count": pred_count,
            "succ_count": succ_count,
            "cross_scope_pred_count": cross_scope_pred_count,
            "cross_scope_succ_count": cross_scope_succ_count,
            "topo_pos_norm": topo_pos_norm,
            "reverse_topo_pos_norm": reverse_topo_pos_norm,
            "distance_to_output": distance_to_output,
            "distance_to_nearest_merge": distance_to_nearest_merge,
            "distance_to_nearest_concat": distance_to_nearest_concat,
            "distance_to_nearest_reduce": distance_to_nearest_reduce,
            "distance_to_nearest_gather": distance_to_nearest_gather,
            "nearest_merge_fanin": nearest_merge_fanin,
            "graph_feature_mapped_flag": 1.0,
        }
        hop_counts = {
            "1hop": np.zeros(len(EXEC_MODE_ORDER), dtype=float),
            "2hop": np.zeros(len(EXEC_MODE_ORDER), dtype=float),
        }
        for hop_name, values in (("1hop", hop1), ("2hop", hop2)):
            for item in values:
                hop_counts[hop_name][exec_mode_index.get(str(item), 0)] += 1.0
            for exec_mode, index in exec_mode_index.items():
                record[f"graph_{hop_name}_{exec_mode}_count"] = float(hop_counts[hop_name][index])
        records.append(record)

    return pd.DataFrame(records)
