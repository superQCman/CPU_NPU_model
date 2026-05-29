from __future__ import annotations

import ast
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .graph_static_features import GRAPH_FEATURE_COLUMNS, graph_feature_table_for_shape_csv
except ImportError:
    from graph_static_features import GRAPH_FEATURE_COLUMNS, graph_feature_table_for_shape_csv  # type: ignore


STATIC_FEATURE_CONTRACT_VERSION = 3
STATIC_FEATURE_CONTRACT_VARIANT = "v3_static_exec_state"

RUNTIME_OBSERVED_PREFIXES = ("cpu_", "local_ctx_", "combo_")
RUNTIME_OBSERVED_EXACT = {"comp_feat_pressure_threads"}
RUNTIME_OBSERVED_EXCLUDE = {"combo"}

STATIC_CATEGORICAL_FEATURES = [
    "cfg_runner_mode",
    "cfg_backend_family",
    "feat_primary_dtype_class",
    "feat_broadcast_class",
    "feat_transpose_mode_class",
    "feat_reduction_layout_class",
    "feat_shape_metadata_class",
    "submodel_role_static",
    "exec_region_static",
    "branch_lifetime_class",
    "producer_materialization_state",
    "consumer_materialization_state",
    "reduce_partial_shape_static",
    "pred_op_type_primary",
    "succ_op_type_primary",
    "pred_exec_mode_primary",
    "succ_exec_mode_primary",
]

STATIC_NUMERIC_FEATURES = [
    "cfg_runner_is_branch_parallel",
    "cfg_parallel_branches",
    "cfg_tail_intra_threads",
    "cfg_disable_graph_optimizations",
    "cfg_force_cpu_ops_enabled",
    "feat_primary_dtype_is_int64",
    "feat_primary_dtype_is_float",
    "feat_primary_dtype_is_bool",
    "feat_primary_dtype_class_code",
    "feat_broadcast_same_flag",
    "feat_broadcast_scalar_flag",
    "feat_broadcast_generic_flag",
    "feat_broadcast_class_code",
    "feat_concat_input_count",
    "feat_gather_index_count_static",
    "feat_gather_tail_block_bytes",
    "feat_transpose_reshape_like_flag",
    "feat_transpose_single_axis_flag",
    "feat_transpose_generic_flag",
    "feat_transpose_mode_class_code",
    "feat_reduction_middle_axis_flag",
    "feat_reduction_rank_collapse_flag",
    "feat_reduction_keepdims_flag",
    "feat_reduction_layout_class_code",
    "feat_shape_query_flag",
    "feat_rank_insert_flag",
    "feat_flatten_view_flag",
    "feat_reshape_view_or_copy_flag",
    "feat_shape_metadata_class_code",
    "feat_concat_shape_flag",
    "feat_tiny_shape_arith_flag",
    "submodel_role_static_code",
    "exec_region_static_code",
    "branch_lifetime_class_code",
    "producer_materialization_state_code",
    "consumer_materialization_state_code",
    "reduce_partial_shape_static_code",
    "materialize_required_flag",
    "merge_sync_obligation_flag",
    "merge_fanin_static",
    "phase_barrier_count_static",
    "worker_shard_count_static",
    "exact_input_tensor_bytes_sum",
    "exact_output_tensor_bytes_sum",
    "exact_read_bytes",
    "exact_write_bytes",
    "feat_graph_fanin",
    "feat_graph_fanout",
    "feat_graph_branch_fanout",
    "feat_graph_merge_fanin",
    *GRAPH_FEATURE_COLUMNS,
]

_ENV_LINE_RE = re.compile(r"^\[ENV\]\s+([A-Z0-9_]+)=(.*)$")

DTYPE_BYTES = {
    "float16": 2.0,
    "float32": 4.0,
    "float64": 8.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "uint8": 1.0,
    "int16": 2.0,
    "uint16": 2.0,
    "int32": 4.0,
    "uint32": 4.0,
    "int64": 8.0,
    "uint64": 8.0,
    "bool": 1.0,
}

_PRIMARY_DTYPE_CODE = {"other": 0.0, "int64": 1.0, "float": 2.0, "bool": 3.0}
_BROADCAST_CLASS_CODE = {"not_applicable": 0.0, "same_shape": 1.0, "scalar_like": 2.0, "generic_broadcast": 3.0}
_TRANSPOSE_MODE_CLASS_CODE = {"not_applicable": 0.0, "unknown": 1.0, "reshape_like": 2.0, "single_axis": 3.0, "generic": 4.0}
_REDUCTION_LAYOUT_CLASS_CODE = {"not_applicable": 0.0, "unknown": 1.0, "middle_axis_collapse": 2.0, "keepdims_like": 3.0, "rank_collapse": 4.0}
_SHAPE_METADATA_CLASS_CODE = {
    "not_applicable": 0.0,
    "shape_query": 1.0,
    "rank_insert": 2.0,
    "flatten_view": 3.0,
    "reshape_view_or_copy": 4.0,
    "concat_shape": 5.0,
}
_SUBMODEL_ROLE_CODE = {
    "unknown": 0.0,
    "embedding_branch": 1.0,
    "mlp_bot": 2.0,
    "interaction": 3.0,
    "tail": 4.0,
    "mlp_top": 5.0,
}
_EXEC_REGION_CODE = {
    "unknown": 0.0,
    "metadata_control": 1.0,
    "embedding_private": 2.0,
    "interaction_merge": 3.0,
    "tail_serial": 4.0,
    "top_mlp": 5.0,
}
_BRANCH_LIFETIME_CODE = {
    "serial": 0.0,
    "merge_imminent": 1.0,
    "short_lived": 2.0,
    "long_lived": 3.0,
}
_MATERIALIZATION_STATE_CODE = {
    "unknown": 0.0,
    "source_input": 1.0,
    "view_like": 2.0,
    "materialized": 3.0,
}
_REDUCE_PARTIAL_SHAPE_CODE = {
    "not_applicable": 0.0,
    "scalar_like": 1.0,
    "vector_like": 2.0,
    "tensor_like": 3.0,
    "collapsed_rank": 4.0,
}


def _normalized_node_name(value: Any) -> str:
    return str(value or "").strip()


def _is_runtime_observed_column(column: str) -> bool:
    text = str(column)
    if text in RUNTIME_OBSERVED_EXCLUDE:
        return False
    if text in RUNTIME_OBSERVED_EXACT:
        return True
    return any(text.startswith(prefix) for prefix in RUNTIME_OBSERVED_PREFIXES)


def _series_text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    return frame.get(column, pd.Series(default, index=frame.index)).fillna(default).astype(str)


def _series_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce").fillna(default).astype(float)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float(default)


def _parse_env_lines(log_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not log_path.exists():
        return env
    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _ENV_LINE_RE.match(raw_line.strip())
        if match:
            env[str(match.group(1))] = str(match.group(2)).strip()
    return env


@lru_cache(maxsize=32)
def _resolved_source_manifest_root(dataset_root_text: str) -> str:
    dataset_root = Path(dataset_root_text).resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        split_manifest_path = dataset_root / "split_manifest.json"
        if split_manifest_path.exists():
            try:
                split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return ""
            input_dataset_root = Path(str(split_payload.get("input_dataset_root", "")).strip()).resolve()
            input_manifest_path = input_dataset_root / "manifest.json"
            if input_manifest_path.exists():
                return str(input_dataset_root)
        return ""
    return str(dataset_root)


@lru_cache(maxsize=32)
def _source_manifest_payload(dataset_root_text: str) -> dict[str, Any]:
    resolved = _resolved_source_manifest_root(dataset_root_text)
    if not resolved:
        return {}
    manifest_path = Path(resolved) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(manifest)


@lru_cache(maxsize=32)
def _static_source_config_map(dataset_root_text: str) -> dict[str, dict[str, Any]]:
    manifest = _source_manifest_payload(dataset_root_text)
    if not manifest:
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for spec in manifest.get("case_specs", []):
        source_name = str(spec.get("source_name", "")).strip()
        if not source_name:
            continue
        env = _parse_env_lines(Path(str(spec.get("log_path", "")).strip()))
        runner_mode = str(env.get("RUNNER_MODE", "unknown")).strip() or "unknown"
        mapping[source_name] = {
            "cfg_runner_mode": runner_mode,
            "cfg_runner_is_branch_parallel": 1.0 if runner_mode == "branch_parallel" else 0.0,
            "cfg_parallel_branches": float(_safe_int(env.get("PARALLEL_BRANCHES", 0), 0)),
            "cfg_tail_intra_threads": float(_safe_int(env.get("TAIL_INTRA_THREADS", 0), 0)),
            "cfg_disable_graph_optimizations": float(_safe_int(env.get("DISABLE_GRAPH_OPTIMIZATIONS", 0), 0)),
            "cfg_force_cpu_ops_enabled": float(_safe_int(env.get("FORCE_CPU_OPS", 1), 1)),
        }
    return mapping


@lru_cache(maxsize=32)
def _shape_csv_lookup_map(dataset_root_text: str) -> dict[tuple[str, int, int], str]:
    manifest = _source_manifest_payload(dataset_root_text)
    if not manifest:
        return {}
    mapping: dict[tuple[str, int, int], str] = {}
    for spec in manifest.get("case_specs", []):
        source_name = str(spec.get("source_name", "")).strip()
        sweep_dir = Path(str(spec.get("sweep_dir", "")).strip())
        if not source_name or not sweep_dir.exists():
            continue
        sweep_summary_path = sweep_dir / "sweep_summary.csv"
        if not sweep_summary_path.exists():
            continue
        summary = pd.read_csv(
            sweep_summary_path,
            usecols=["batch_size", "num_indices_per_lookup", "shape_csv"],
            low_memory=False,
        )
        for _, row in summary.iterrows():
            batch_size = _safe_int(row.get("batch_size"), 0)
            num_indices = _safe_int(row.get("num_indices_per_lookup"), 0)
            shape_csv = Path(str(row.get("shape_csv", "")).strip())
            if not shape_csv.exists():
                continue
            mapping[(source_name, batch_size, num_indices)] = str(shape_csv.resolve())
    return mapping


def _parse_typed_tensors(value: Any) -> list[tuple[str, list[int]]]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text in {"nan", "None", "[]", "unknown"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    tensors: list[tuple[str, list[int]]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        for dtype, dims in entry.items():
            if isinstance(dims, (list, tuple)):
                clean_dims: list[int] = []
                for dim in dims:
                    try:
                        clean_dims.append(int(dim))
                    except (TypeError, ValueError):
                        continue
                tensors.append((str(dtype).strip().lower(), clean_dims))
    return tensors


def _element_count(dims: list[int]) -> int:
    if not dims:
        return 1
    result = 1
    for dim in dims:
        result *= max(int(dim), 1)
    return int(result)


def _tensor_bytes(specs: list[tuple[str, list[int]]]) -> float:
    total = 0.0
    for dtype, dims in specs:
        total += float(_element_count(dims)) * float(DTYPE_BYTES.get(str(dtype).strip().lower(), 4.0))
    return float(max(total, 0.0))


def _submodel_role_from_row(node_name: str, op_type: str) -> str:
    text = str(node_name).strip()
    if text.startswith("/emb_l"):
        return "embedding_branch"
    if "/bot_l/" in text:
        return "mlp_bot"
    if "/top_l/" in text:
        return "mlp_top"
    if str(op_type) in {"Gather", "MatMul", "Transpose", "Add", "Mul", "Shape", "Unsqueeze", "Reshape", "Flatten"}:
        return "interaction"
    return "tail"


def _is_materialized(op_type: str, *, reshape_view_or_copy: bool, transpose_reshape_like: bool, concat_shape: bool) -> float:
    if op_type == "Shape":
        return 0.0
    if op_type == "Unsqueeze":
        return 0.0
    if op_type == "Flatten":
        return 0.0
    if op_type == "Reshape":
        return 0.0 if reshape_view_or_copy else 1.0
    if op_type == "Transpose":
        return 0.0 if transpose_reshape_like else 1.0
    if op_type == "Concat":
        return 0.0 if concat_shape else 1.0
    return 1.0


def _exact_io_bytes(op_type: str, input_bytes: float, output_bytes: float, *, materialized: float) -> tuple[float, float]:
    if op_type == "Shape":
        return 0.0, float(max(output_bytes, 0.0))
    if op_type in {"Unsqueeze", "Flatten"}:
        return 0.0, 0.0
    if op_type == "Reshape" and materialized < 0.5:
        return 0.0, 0.0
    if op_type == "Concat" and output_bytes <= 1024.0:
        return 0.0, 0.0
    if materialized < 0.5:
        return 0.0, 0.0
    return float(max(input_bytes, 0.0)), float(max(output_bytes, 0.0))


def _exec_region_from_state(
    *,
    op_type: str,
    submodel_role: str,
    materialized: bool,
    merge_sync_obligation: bool,
    pred_count: float,
    exec_mode_hint: str,
) -> str:
    if exec_mode_hint == "metadata_view" and not materialized:
        return "metadata_control"
    if submodel_role == "embedding_branch":
        return "embedding_private"
    if submodel_role in {"mlp_bot", "mlp_top"}:
        return "top_mlp"
    if merge_sync_obligation or pred_count > 1.0 or submodel_role == "interaction":
        return "interaction_merge"
    if op_type in {"Shape", "Unsqueeze", "Flatten"} and not materialized:
        return "metadata_control"
    return "tail_serial"


def _branch_lifetime_class(*, exec_region: str, distance_to_merge: float) -> str:
    if exec_region != "embedding_private":
        return "serial"
    if distance_to_merge <= 1.0:
        return "merge_imminent"
    if distance_to_merge <= 4.0:
        return "short_lived"
    return "long_lived"


def _materialization_state_from_mode(exec_mode: str, *, pred_or_succ_count: float) -> str:
    if pred_or_succ_count <= 0.0:
        return "source_input"
    if exec_mode in {"metadata_view", "tiny_shape_arith"}:
        return "view_like"
    if exec_mode in {"indexed_block_copy", "payload_concat", "reduction_layout", "transpose_generic", "gemm_like"}:
        return "materialized"
    return "unknown"


def _reduce_partial_shape_class(
    *,
    op_type: str,
    reduction_layout_class: str,
    output_rank: float,
    output_elements: float,
) -> str:
    if op_type != "ReduceSum":
        return "not_applicable"
    if reduction_layout_class == "rank_collapse":
        return "collapsed_rank"
    if output_elements <= 1.0:
        return "scalar_like"
    if output_rank <= 1.0:
        return "vector_like"
    return "tensor_like"


def _dtype_class(dtype: str) -> str:
    text = str(dtype).lower()
    if text.startswith("int64") or text == "int64":
        return "int64"
    if text.startswith("float") or text.startswith("bfloat"):
        return "float"
    if text == "bool":
        return "bool"
    return "other"


def _broadcast_class(input_specs: list[tuple[str, list[int]]], op_type: str) -> str:
    if op_type not in {"Add", "Mul"} or len(input_specs) < 2:
        return "not_applicable"
    lhs = input_specs[0][1]
    rhs = input_specs[1][1]
    if lhs == rhs:
        return "same_shape"
    if _element_count(lhs) <= 1 or _element_count(rhs) <= 1:
        return "scalar_like"
    max_rank = max(len(lhs), len(rhs))
    lhs_pad = [1] * (max_rank - len(lhs)) + lhs
    rhs_pad = [1] * (max_rank - len(rhs)) + rhs
    if all(a == b or a == 1 or b == 1 for a, b in zip(lhs_pad, rhs_pad, strict=False)):
        return "generic_broadcast"
    return "generic_broadcast"


def _changed_positions(lhs: list[int], rhs: list[int]) -> list[int]:
    max_rank = max(len(lhs), len(rhs))
    lhs_pad = [1] * (max_rank - len(lhs)) + lhs
    rhs_pad = [1] * (max_rank - len(rhs)) + rhs
    return [index for index, (a, b) in enumerate(zip(lhs_pad, rhs_pad, strict=False)) if a != b]


def _transpose_mode(input_dims: list[int], output_dims: list[int], op_type: str) -> str:
    if op_type != "Transpose":
        return "not_applicable"
    if not input_dims or not output_dims:
        return "unknown"
    if sorted(input_dims) != sorted(output_dims):
        return "unknown"
    if input_dims == output_dims:
        return "reshape_like"
    changed = _changed_positions(input_dims, output_dims)
    if not changed:
        return "reshape_like"
    if all(
        ([1] * (max(len(input_dims), len(output_dims)) - len(input_dims)) + input_dims)[idx] == 1
        or ([1] * (max(len(input_dims), len(output_dims)) - len(output_dims)) + output_dims)[idx] == 1
        for idx in changed
    ):
        return "reshape_like"
    if len(changed) <= 2:
        return "single_axis"
    return "generic"


def _reduction_layout(input_dims: list[int], output_dims: list[int], op_type: str) -> str:
    if op_type != "ReduceSum":
        return "not_applicable"
    input_rank = len(input_dims)
    output_rank = len(output_dims)
    if input_rank == 3 and output_rank == 2 and len(output_dims) == 2:
        if input_dims[0] == output_dims[0] and input_dims[-1] == output_dims[-1]:
            return "middle_axis_collapse"
    if output_rank == input_rank:
        return "keepdims_like"
    if output_rank < input_rank:
        return "rank_collapse"
    return "unknown"


def _shape_metadata_class(op_type: str, primary_dtype_class: str, output_size: float) -> str:
    if op_type == "Shape":
        return "shape_query"
    if op_type == "Unsqueeze":
        return "rank_insert"
    if op_type == "Flatten":
        return "flatten_view"
    if op_type == "Reshape":
        return "reshape_view_or_copy"
    if op_type == "Concat" and primary_dtype_class == "int64" and output_size <= 512.0:
        return "concat_shape"
    return "not_applicable"


def _static_topology_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "node_scope" not in work.columns:
        work["static_is_embedding_branch"] = "no"
        work["static_parallel_branch_count"] = 1.0
        work["static_distance_to_merge"] = 0.0
        return work
    node_scope = _series_text(work, "node_scope", "").str.lower()
    is_embedding_branch = node_scope.str.match(r"^emb_l[0-9]+$")
    work["static_is_embedding_branch"] = np.where(is_embedding_branch.to_numpy(dtype=bool), "yes", "no")
    work["static_parallel_branch_count"] = 1.0
    work["static_distance_to_merge"] = 0.0
    if not {"case_id", "combo", "op_idx"}.issubset(work.columns):
        return work
    op_idx = pd.to_numeric(work["op_idx"], errors="coerce")
    for _, group in work.groupby(["case_id", "combo"], sort=False):
        group_scope = group["node_scope"].fillna("").astype(str).str.lower()
        emb_mask = group_scope.str.match(r"^emb_l[0-9]+$").to_numpy(dtype=bool)
        branch_scopes = sorted({value for value in group_scope.loc[emb_mask].tolist() if value})
        branch_count = float(max(len(branch_scopes), 1))
        work.loc[group.index, "static_parallel_branch_count"] = pd.Series(
            np.where(emb_mask, branch_count, 1.0),
            index=group.index,
            dtype=float,
        )
        if not emb_mask.any():
            continue
        group_op_idx = pd.to_numeric(group["op_idx"], errors="coerce").fillna(np.nan).to_numpy(dtype=float)
        merge_candidates = group_op_idx[~emb_mask]
        merge_candidates = merge_candidates[~np.isnan(merge_candidates)]
        if len(merge_candidates) == 0:
            continue
        merge_op_idx = float(np.min(merge_candidates))
        distances = np.clip(merge_op_idx - np.nan_to_num(group_op_idx, nan=merge_op_idx), a_min=0.0, a_max=None)
        work.loc[group.index, "static_distance_to_merge"] = pd.Series(
            np.where(emb_mask, distances, 0.0),
            index=group.index,
            dtype=float,
        )
    return work


def _merge_graph_feature_table(
    work: pd.DataFrame,
    graph_table: pd.DataFrame,
    *,
    default_graph_values: dict[str, Any],
) -> pd.DataFrame:
    if graph_table.empty or not {"op_idx", "node_name_normalized"}.issubset(graph_table.columns):
        return work
    if not {"op_idx", "node_name_normalized"}.issubset(work.columns):
        return work

    local = work.copy()
    local["op_idx"] = pd.to_numeric(local["op_idx"], errors="coerce").fillna(-1).astype(int)
    local["node_name_normalized"] = local["node_name_normalized"].astype(str).map(_normalized_node_name)

    graph_features = graph_table.copy()
    graph_features["op_idx"] = pd.to_numeric(graph_features["op_idx"], errors="coerce").fillna(-1).astype(int)
    graph_features["node_name_normalized"] = graph_features["node_name_normalized"].astype(str).map(_normalized_node_name)

    merged = local.merge(
        graph_features,
        on=["op_idx", "node_name_normalized"],
        how="left",
        suffixes=("", "_graph"),
    )
    fallback_graph = graph_features.drop_duplicates(subset=["node_name_normalized"]).drop(columns=["op_idx"], errors="ignore")
    missing_mask = merged["graph_feature_mapped_flag"].isna() | pd.to_numeric(
        merged["graph_feature_mapped_flag"],
        errors="coerce",
    ).fillna(0.0).le(0.0)
    if missing_mask.any() and not fallback_graph.empty:
        fallback = merged.loc[missing_mask, ["node_name_normalized"]].merge(
            fallback_graph,
            on="node_name_normalized",
            how="left",
            suffixes=("", "_fb"),
        )
        for column in fallback.columns:
            if column == "node_name_normalized":
                continue
            target_column = column[:-6] if column.endswith("_graph") else column
            if target_column in merged.columns:
                merged.loc[missing_mask, target_column] = fallback[column].to_numpy()
            else:
                merged.loc[missing_mask, column] = fallback[column].to_numpy()

    graph_merge_columns = [
        "submodel_role_static",
        "submodel_role_static_code",
        *GRAPH_FEATURE_COLUMNS,
    ]
    for column in graph_merge_columns:
        graph_column = f"{column}_graph"
        if graph_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[graph_column].where(merged[graph_column].notna(), merged[column])
            else:
                merged[column] = merged[graph_column]
            merged = merged.drop(columns=[graph_column], errors="ignore")
    for column, default in default_graph_values.items():
        merged[column] = merged[column].fillna(default)
    merged["submodel_role_static"] = merged["submodel_role_static"].fillna("unknown").astype(str)
    merged["submodel_role_static_code"] = pd.to_numeric(
        merged["submodel_role_static_code"],
        errors="coerce",
    ).fillna(0.0).astype(float)
    return merged


def augment_static_feature_frame(
    frame: pd.DataFrame,
    *,
    dataset_root: Path | None = None,
    graph_feature_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = _static_topology_columns(frame.copy())
    resolved_dataset_root = dataset_root
    if resolved_dataset_root is None:
        attr_root = work.attrs.get("dataset_root", "")
        if attr_root:
            resolved_dataset_root = Path(str(attr_root))
    if "row_uid" in work.columns:
        work["row_uid"] = work["row_uid"].astype(str)
    if "node_name_normalized" not in work.columns:
        work["node_name_normalized"] = _series_text(work, "node_name", "").map(_normalized_node_name)
    else:
        work["node_name_normalized"] = _series_text(work, "node_name_normalized", "").map(_normalized_node_name)

    dataset_root_text = str(resolved_dataset_root.resolve()) if resolved_dataset_root is not None else ""
    config_map = _static_source_config_map(dataset_root_text) if resolved_dataset_root is not None else {}
    shape_csv_lookup = _shape_csv_lookup_map(dataset_root_text) if resolved_dataset_root is not None else {}
    source_name = _series_text(work, "source_name", "")
    backend_family = _series_text(work, "backend_family", "cpu").replace({"": "cpu"})

    def _config_numeric(column: str, key: str, default: float) -> pd.Series:
        mapped = pd.to_numeric(
            source_name.map(lambda value: config_map.get(str(value), {}).get(key, np.nan)),
            errors="coerce",
        )
        existing = _series_numeric(work, column, default)
        return mapped.where(~mapped.isna(), existing).fillna(default).astype(float)

    def _config_text(column: str, key: str, default: str) -> pd.Series:
        mapped = source_name.map(lambda value: str(config_map.get(str(value), {}).get(key, ""))).astype(object)
        mapped = mapped.where(mapped.ne(""), np.nan)
        existing = _series_text(work, column, default)
        return mapped.where(mapped.notna(), existing).fillna(default).astype(str)

    work["cfg_runner_mode"] = _config_text("cfg_runner_mode", "cfg_runner_mode", "unknown")
    work["cfg_runner_is_branch_parallel"] = _config_numeric(
        "cfg_runner_is_branch_parallel",
        "cfg_runner_is_branch_parallel",
        1.0,
    )
    work["cfg_parallel_branches"] = _config_numeric("cfg_parallel_branches", "cfg_parallel_branches", 0.0)
    work["cfg_tail_intra_threads"] = _config_numeric("cfg_tail_intra_threads", "cfg_tail_intra_threads", 0.0)
    work["cfg_disable_graph_optimizations"] = _config_numeric(
        "cfg_disable_graph_optimizations",
        "cfg_disable_graph_optimizations",
        0.0,
    )
    work["cfg_force_cpu_ops_enabled"] = _config_numeric(
        "cfg_force_cpu_ops_enabled",
        "cfg_force_cpu_ops_enabled",
        1.0,
    )
    work["cfg_backend_family"] = _series_text(work, "cfg_backend_family", backend_family.astype(str).iloc[0] if len(work) else "cpu")
    work["cfg_backend_family"] = work["cfg_backend_family"].replace({"": np.nan}).fillna(backend_family.astype(str))

    op_type = _series_text(work, "op_type", "")
    output_size = _series_numeric(work, "output_size", 0.0)

    primary_dtype_class: list[str] = []
    broadcast_class: list[str] = []
    concat_input_count: list[float] = []
    gather_index_count: list[float] = []
    gather_tail_block_bytes: list[float] = []
    transpose_mode_class: list[str] = []
    reduction_layout_class: list[str] = []
    shape_metadata_class: list[str] = []
    tiny_shape_arith_flag: list[float] = []
    materialize_required_flag: list[float] = []
    exact_input_tensor_bytes_sum: list[float] = []
    exact_output_tensor_bytes_sum: list[float] = []
    exact_read_bytes: list[float] = []
    exact_write_bytes: list[float] = []
    submodel_role_static: list[str] = []
    exec_region_static: list[str] = []
    branch_lifetime_class: list[str] = []
    producer_materialization_state: list[str] = []
    consumer_materialization_state: list[str] = []
    reduce_partial_shape_static: list[str] = []
    merge_fanin_static: list[float] = []
    phase_barrier_count_static: list[float] = []
    worker_shard_count_static: list[float] = []
    graph_fanin: list[float] = []
    graph_fanout: list[float] = []
    graph_branch_fanout: list[float] = []
    graph_merge_fanin: list[float] = []

    input_values = work.get("input_type_shape", pd.Series("", index=work.index)).tolist()
    output_values = work.get("output_type_shape", pd.Series("", index=work.index)).tolist()
    node_names = _series_text(work, "node_name", "").tolist()
    branch_counts = _series_numeric(work, "static_parallel_branch_count", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    is_embedding_branch = _series_text(work, "static_is_embedding_branch", "no").eq("yes").to_numpy(dtype=bool)

    for idx, (raw_input, raw_output, current_op, current_output_size, current_node_name) in enumerate(
        zip(input_values, output_values, op_type.tolist(), output_size.tolist(), node_names, strict=False)
    ):
        input_specs = _parse_typed_tensors(raw_input)
        output_specs = _parse_typed_tensors(raw_output)
        input_primary = input_specs[0] if input_specs else ("unknown", [])
        output_primary = output_specs[0] if output_specs else input_primary
        dtype_class = _dtype_class(output_primary[0] if output_specs else input_primary[0])
        out_elems = _element_count(output_primary[1])
        input_bytes = _tensor_bytes(input_specs)
        output_bytes = _tensor_bytes(output_specs)

        primary_dtype_class.append(dtype_class)
        broadcast_class.append(_broadcast_class(input_specs, str(current_op)))
        concat_inputs = float(len(input_specs))
        concat_input_count.append(concat_inputs)

        gather_indices = 0.0
        if str(current_op) == "Gather" and len(input_specs) >= 2:
            gather_indices = float(_element_count(input_specs[1][1]))
        gather_index_count.append(gather_indices)
        gather_tail_block_bytes.append(float(current_output_size) / gather_indices if gather_indices > 0 else 0.0)

        transpose_mode_class.append(_transpose_mode(input_primary[1], output_primary[1], str(current_op)))
        reduction_layout_class.append(_reduction_layout(input_primary[1], output_primary[1], str(current_op)))
        shape_metadata_class.append(_shape_metadata_class(str(current_op), dtype_class, float(current_output_size)))

        tiny_shape_arith = (
            str(current_op) in {"Add", "Mul"}
            and dtype_class == "int64"
            and out_elems <= 128
            and float(current_output_size) <= 1024.0
        )
        tiny_shape_arith_flag.append(1.0 if tiny_shape_arith else 0.0)
        materialized = _is_materialized(
            str(current_op),
            reshape_view_or_copy=bool(str(current_op) == "Reshape"),
            transpose_reshape_like=bool(transpose_mode_class[-1] == "reshape_like"),
            concat_shape=bool(shape_metadata_class[-1] == "concat_shape"),
        )
        materialize_required_flag.append(float(materialized))
        exact_input_tensor_bytes_sum.append(float(input_bytes))
        exact_output_tensor_bytes_sum.append(float(output_bytes))
        read_bytes, write_bytes = _exact_io_bytes(str(current_op), input_bytes, output_bytes, materialized=float(materialized))
        exact_read_bytes.append(float(read_bytes))
        exact_write_bytes.append(float(write_bytes))
        submodel_role_static.append(_submodel_role_from_row(str(current_node_name), str(current_op)))

        graph_fanin.append(float(max(len(input_specs), 1)))
        graph_fanout.append(float(branch_counts[idx] if is_embedding_branch[idx] else 1.0))
        graph_branch_fanout.append(float(branch_counts[idx] if is_embedding_branch[idx] else 1.0))
        graph_merge_fanin.append(float(concat_inputs if str(current_op) == "Concat" else 1.0))

    work["feat_primary_dtype_class"] = pd.Series(primary_dtype_class, index=work.index, dtype="object")
    work["feat_primary_dtype_is_int64"] = work["feat_primary_dtype_class"].eq("int64").astype(float)
    work["feat_primary_dtype_is_float"] = work["feat_primary_dtype_class"].eq("float").astype(float)
    work["feat_primary_dtype_is_bool"] = work["feat_primary_dtype_class"].eq("bool").astype(float)
    work["feat_primary_dtype_class_code"] = work["feat_primary_dtype_class"].map(_PRIMARY_DTYPE_CODE).fillna(0.0).astype(float)

    work["feat_broadcast_class"] = pd.Series(broadcast_class, index=work.index, dtype="object")
    work["feat_broadcast_same_flag"] = work["feat_broadcast_class"].eq("same_shape").astype(float)
    work["feat_broadcast_scalar_flag"] = work["feat_broadcast_class"].eq("scalar_like").astype(float)
    work["feat_broadcast_generic_flag"] = work["feat_broadcast_class"].eq("generic_broadcast").astype(float)
    work["feat_broadcast_class_code"] = work["feat_broadcast_class"].map(_BROADCAST_CLASS_CODE).fillna(0.0).astype(float)

    work["feat_concat_input_count"] = pd.Series(concat_input_count, index=work.index, dtype=float)
    work["feat_gather_index_count_static"] = pd.Series(gather_index_count, index=work.index, dtype=float)
    work["feat_gather_tail_block_bytes"] = pd.Series(gather_tail_block_bytes, index=work.index, dtype=float)

    work["feat_transpose_mode_class"] = pd.Series(transpose_mode_class, index=work.index, dtype="object")
    work["feat_transpose_reshape_like_flag"] = work["feat_transpose_mode_class"].eq("reshape_like").astype(float)
    work["feat_transpose_single_axis_flag"] = work["feat_transpose_mode_class"].eq("single_axis").astype(float)
    work["feat_transpose_generic_flag"] = work["feat_transpose_mode_class"].eq("generic").astype(float)
    work["feat_transpose_mode_class_code"] = work["feat_transpose_mode_class"].map(_TRANSPOSE_MODE_CLASS_CODE).fillna(0.0).astype(float)

    work["feat_reduction_layout_class"] = pd.Series(reduction_layout_class, index=work.index, dtype="object")
    work["feat_reduction_middle_axis_flag"] = work["feat_reduction_layout_class"].eq("middle_axis_collapse").astype(float)
    work["feat_reduction_rank_collapse_flag"] = work["feat_reduction_layout_class"].eq("rank_collapse").astype(float)
    work["feat_reduction_keepdims_flag"] = work["feat_reduction_layout_class"].eq("keepdims_like").astype(float)
    work["feat_reduction_layout_class_code"] = work["feat_reduction_layout_class"].map(_REDUCTION_LAYOUT_CLASS_CODE).fillna(0.0).astype(float)

    work["feat_shape_metadata_class"] = pd.Series(shape_metadata_class, index=work.index, dtype="object")
    work["feat_shape_query_flag"] = work["feat_shape_metadata_class"].eq("shape_query").astype(float)
    work["feat_rank_insert_flag"] = work["feat_shape_metadata_class"].eq("rank_insert").astype(float)
    work["feat_flatten_view_flag"] = work["feat_shape_metadata_class"].eq("flatten_view").astype(float)
    work["feat_reshape_view_or_copy_flag"] = work["feat_shape_metadata_class"].eq("reshape_view_or_copy").astype(float)
    work["feat_concat_shape_flag"] = work["feat_shape_metadata_class"].eq("concat_shape").astype(float)
    work["feat_shape_metadata_class_code"] = work["feat_shape_metadata_class"].map(_SHAPE_METADATA_CLASS_CODE).fillna(0.0).astype(float)
    work["feat_tiny_shape_arith_flag"] = pd.Series(tiny_shape_arith_flag, index=work.index, dtype=float)
    work["materialize_required_flag"] = pd.Series(materialize_required_flag, index=work.index, dtype=float)
    work["exact_input_tensor_bytes_sum"] = pd.Series(exact_input_tensor_bytes_sum, index=work.index, dtype=float)
    work["exact_output_tensor_bytes_sum"] = pd.Series(exact_output_tensor_bytes_sum, index=work.index, dtype=float)
    work["exact_read_bytes"] = pd.Series(exact_read_bytes, index=work.index, dtype=float)
    work["exact_write_bytes"] = pd.Series(exact_write_bytes, index=work.index, dtype=float)
    work["submodel_role_static"] = pd.Series(submodel_role_static, index=work.index, dtype="object")
    work["submodel_role_static_code"] = work["submodel_role_static"].map(_SUBMODEL_ROLE_CODE).fillna(0.0).astype(float)

    work["feat_graph_fanin"] = pd.Series(graph_fanin, index=work.index, dtype=float)
    work["feat_graph_fanout"] = pd.Series(graph_fanout, index=work.index, dtype=float)
    work["feat_graph_branch_fanout"] = pd.Series(graph_branch_fanout, index=work.index, dtype=float)
    work["feat_graph_merge_fanin"] = pd.Series(graph_merge_fanin, index=work.index, dtype=float)

    default_graph_values: dict[str, Any] = {
        "pred_op_type_primary": "Unknown",
        "succ_op_type_primary": "Unknown",
        "pred_exec_mode_primary": "metadata_view",
        "succ_exec_mode_primary": "metadata_view",
        "pred_op_type_primary_code": 0.0,
        "succ_op_type_primary_code": 0.0,
        "pred_exec_mode_primary_code": 0.0,
        "succ_exec_mode_primary_code": 0.0,
        "pred_count": 0.0,
        "succ_count": 0.0,
        "cross_scope_pred_count": 0.0,
        "cross_scope_succ_count": 0.0,
        "topo_pos_norm": 0.0,
        "reverse_topo_pos_norm": 0.0,
        "distance_to_output": -1.0,
        "distance_to_nearest_merge": -1.0,
        "distance_to_nearest_concat": -1.0,
        "distance_to_nearest_reduce": -1.0,
        "distance_to_nearest_gather": -1.0,
        "nearest_merge_fanin": 0.0,
        "graph_feature_mapped_flag": 0.0,
    }
    for hop in ("1hop", "2hop"):
        for exec_mode_name in (
            "metadata_view",
            "tiny_shape_arith",
            "indexed_block_copy",
            "payload_concat",
            "reduction_layout",
            "transpose_generic",
            "gemm_like",
        ):
            default_graph_values[f"graph_{hop}_{exec_mode_name}_count"] = 0.0
    for column, default in default_graph_values.items():
        work[column] = pd.Series(default, index=work.index)

    if graph_feature_table is not None:
        work = _merge_graph_feature_table(
            work,
            graph_feature_table,
            default_graph_values=default_graph_values,
        ).sort_index()
    elif shape_csv_lookup and {"source_name", "batch_size", "num_indices_per_lookup", "op_idx", "node_name_normalized"}.issubset(work.columns):
        merged_frames: list[pd.DataFrame] = []
        for (current_source, batch_size_value, num_indices_value), group in work.groupby(
            ["source_name", "batch_size", "num_indices_per_lookup"],
            sort=False,
        ):
            shape_csv = shape_csv_lookup.get((str(current_source), _safe_int(batch_size_value, 0), _safe_int(num_indices_value, 0)))
            if not shape_csv:
                merged_frames.append(group.copy())
                continue
            graph_table = graph_feature_table_for_shape_csv(shape_csv)
            merged = _merge_graph_feature_table(
                group.copy(),
                graph_table,
                default_graph_values=default_graph_values,
            )
            merged_frames.append(merged)
        if merged_frames:
            work = pd.concat(merged_frames, ignore_index=False).sort_index()

    work["merge_sync_obligation_flag"] = (
        work["node_name"].astype(str).str.contains("/Concat", regex=False)
        | (pd.to_numeric(work["pred_count"], errors="coerce").fillna(0.0) > 1.0)
        | (
            _series_text(work, "static_is_embedding_branch", "no").eq("yes")
            & _series_numeric(work, "static_distance_to_merge", 0.0).le(1.0)
        )
    ).astype(float)

    pred_count = pd.to_numeric(work["pred_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    nearest_merge_fanin = pd.to_numeric(work["nearest_merge_fanin"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    graph_merge_fanin = pd.to_numeric(work["feat_graph_merge_fanin"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    static_branch_count = pd.to_numeric(work["static_parallel_branch_count"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    distance_to_merge = pd.to_numeric(work["distance_to_nearest_merge"], errors="coerce").fillna(-1.0).to_numpy(dtype=float)
    branch_limit = pd.to_numeric(work["cfg_parallel_branches"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    inter_threads = pd.to_numeric(work["inter_threads"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    num_threads = pd.to_numeric(work["num_threads"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    effective_branch_limit = np.where(branch_limit > 0.0, branch_limit, inter_threads)
    tail_intra_threads = pd.to_numeric(work["cfg_tail_intra_threads"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    merge_sync = pd.to_numeric(work["merge_sync_obligation_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
    materialized = pd.to_numeric(work["materialize_required_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
    pred_exec_mode = work["pred_exec_mode_primary"].astype(str).to_numpy(dtype=str)
    succ_exec_mode = work["succ_exec_mode_primary"].astype(str).to_numpy(dtype=str)
    op_values = work["op_type"].astype(str).to_numpy(dtype=str)
    submodel_roles = work["submodel_role_static"].astype(str).to_numpy(dtype=str)
    output_ranks = pd.to_numeric(work.get("output_rank", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    output_sizes = pd.to_numeric(work.get("output_size", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    output_elements = np.maximum(output_sizes / np.clip(_series_numeric(work, "output_element_bytes", 4.0).to_numpy(dtype=float), a_min=1.0, a_max=None), 1.0)
    reduction_layout_values = work["feat_reduction_layout_class"].astype(str).to_numpy(dtype=str)

    for idx in range(len(work)):
        merge_fanin = max(float(nearest_merge_fanin[idx]), float(graph_merge_fanin[idx]), float(pred_count[idx]), 1.0 if merge_sync[idx] else 0.0, 1.0)
        exec_mode_hint = str(pred_exec_mode[idx]) if str(pred_exec_mode[idx]) else "metadata_view"
        region = _exec_region_from_state(
            op_type=str(op_values[idx]),
            submodel_role=str(submodel_roles[idx]),
            materialized=bool(materialized[idx]),
            merge_sync_obligation=bool(merge_sync[idx]),
            pred_count=float(pred_count[idx]),
            exec_mode_hint=str(exec_mode_hint),
        )
        barriers = 0.0
        if region == "interaction_merge" or merge_sync[idx]:
            barriers = float(max(int(math.ceil(math.log2(max(merge_fanin, 1.0)))), 1))
        lifetime = _branch_lifetime_class(exec_region=region, distance_to_merge=float(distance_to_merge[idx]))
        producer_state = _materialization_state_from_mode(str(pred_exec_mode[idx]), pred_or_succ_count=float(pred_count[idx]))
        consumer_state = _materialization_state_from_mode(
            str(succ_exec_mode[idx]),
            pred_or_succ_count=float(pd.to_numeric(work["succ_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[idx]),
        )
        if region == "embedding_private":
            session_threads = float(num_threads[idx])
        elif region in {"interaction_merge", "tail_serial", "top_mlp", "metadata_control"} and float(tail_intra_threads[idx]) > 0.0:
            session_threads = float(tail_intra_threads[idx])
        else:
            session_threads = float(num_threads[idx])
        worker_shards = max(int(math.ceil(max(session_threads, 1.0))), 1)
        partial_shape = _reduce_partial_shape_class(
            op_type=str(op_values[idx]),
            reduction_layout_class=str(reduction_layout_values[idx]),
            output_rank=float(output_ranks[idx]),
            output_elements=float(output_elements[idx]),
        )
        exec_region_static.append(region)
        branch_lifetime_class.append(lifetime)
        producer_materialization_state.append(producer_state)
        consumer_materialization_state.append(consumer_state)
        reduce_partial_shape_static.append(partial_shape)
        merge_fanin_static.append(float(merge_fanin))
        phase_barrier_count_static.append(float(barriers))
        worker_shard_count_static.append(float(worker_shards))

    work["exec_region_static"] = pd.Series(exec_region_static, index=work.index, dtype="object")
    work["exec_region_static_code"] = work["exec_region_static"].map(_EXEC_REGION_CODE).fillna(0.0).astype(float)
    work["branch_lifetime_class"] = pd.Series(branch_lifetime_class, index=work.index, dtype="object")
    work["branch_lifetime_class_code"] = work["branch_lifetime_class"].map(_BRANCH_LIFETIME_CODE).fillna(0.0).astype(float)
    work["producer_materialization_state"] = pd.Series(producer_materialization_state, index=work.index, dtype="object")
    work["producer_materialization_state_code"] = work["producer_materialization_state"].map(_MATERIALIZATION_STATE_CODE).fillna(0.0).astype(float)
    work["consumer_materialization_state"] = pd.Series(consumer_materialization_state, index=work.index, dtype="object")
    work["consumer_materialization_state_code"] = work["consumer_materialization_state"].map(_MATERIALIZATION_STATE_CODE).fillna(0.0).astype(float)
    work["reduce_partial_shape_static"] = pd.Series(reduce_partial_shape_static, index=work.index, dtype="object")
    work["reduce_partial_shape_static_code"] = work["reduce_partial_shape_static"].map(_REDUCE_PARTIAL_SHAPE_CODE).fillna(0.0).astype(float)
    work["merge_fanin_static"] = pd.Series(merge_fanin_static, index=work.index, dtype=float)
    work["phase_barrier_count_static"] = pd.Series(phase_barrier_count_static, index=work.index, dtype=float)
    work["worker_shard_count_static"] = pd.Series(worker_shard_count_static, index=work.index, dtype=float)
    return work


def sanitize_feature_manifest(payload: dict[str, Any], *, available_columns: list[str] | None = None) -> dict[str, Any]:
    cleaned = dict(payload)
    available = set(available_columns or [])

    def _filtered(values: list[Any], additions: list[str] | None = None) -> list[str]:
        merged: list[str] = []
        for value in values:
            column = str(value)
            if _is_runtime_observed_column(column):
                continue
            if available and column not in available:
                continue
            merged.append(column)
        if additions:
            for column in additions:
                if available and column not in available:
                    continue
                if column not in merged:
                    merged.append(column)
        return merged

    categorical = _filtered(list(cleaned.get("categorical_features", [])), STATIC_CATEGORICAL_FEATURES)
    numeric = _filtered(list(cleaned.get("numeric_features", [])), STATIC_NUMERIC_FEATURES)
    analysis_numeric = _filtered(list(cleaned.get("analysis_numeric_features", [])), STATIC_NUMERIC_FEATURES)
    analytical = _filtered(list(cleaned.get("analytical_feature_columns", [])))
    all_features = _filtered(list(cleaned.get("all_features", [])), [*STATIC_CATEGORICAL_FEATURES, *STATIC_NUMERIC_FEATURES])

    cleaned["categorical_features"] = categorical
    cleaned["numeric_features"] = numeric
    cleaned["analysis_numeric_features"] = analysis_numeric
    cleaned["analytical_feature_columns"] = analytical
    cleaned["all_features"] = all_features
    cleaned["feature_contract_variant"] = STATIC_FEATURE_CONTRACT_VARIANT
    cleaned["static_feature_contract_version"] = int(STATIC_FEATURE_CONTRACT_VERSION)
    return cleaned


def apply_static_contract_to_output_root(output_root: Path) -> dict[str, Any]:
    csv_names = ("dataset_full", "train", "val", "test")
    patched_files: list[str] = []
    row_counts: dict[str, int] = {}
    row_uid_sets: dict[str, set[str]] = {}

    for name in csv_names:
        csv_path = output_root / f"{name}.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        before_count = int(len(frame))
        before_uids = set(frame["row_uid"].astype(str).tolist()) if "row_uid" in frame.columns else set()
        augmented = augment_static_feature_frame(frame, dataset_root=output_root)
        if len(augmented) != before_count:
            raise ValueError(f"Static feature augmentation changed row count for {csv_path}")
        after_uids = set(augmented["row_uid"].astype(str).tolist()) if "row_uid" in augmented.columns else set()
        if before_uids != after_uids:
            raise ValueError(f"Static feature augmentation changed row_uid universe for {csv_path}")
        augmented.to_csv(csv_path, index=False)
        patched_files.append(csv_path.name)
        row_counts[name] = before_count
        row_uid_sets[name] = after_uids

    feature_manifest_path = output_root / "feature_columns.json"
    if feature_manifest_path.exists():
        payload = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        available_columns: list[str] = []
        dataset_full_path = output_root / "dataset_full.csv"
        if dataset_full_path.exists():
            available_columns = pd.read_csv(dataset_full_path, nrows=1, low_memory=False).columns.tolist()
        sanitized = sanitize_feature_manifest(payload, available_columns=available_columns)
        feature_manifest_path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        patched_files.append(feature_manifest_path.name)

    return {
        "patched_files": patched_files,
        "feature_contract_variant": STATIC_FEATURE_CONTRACT_VARIANT,
        "static_feature_contract_version": int(STATIC_FEATURE_CONTRACT_VERSION),
        "row_counts": {key: int(value) for key, value in row_counts.items()},
        "row_uid_counts": {key: int(len(value)) for key, value in row_uid_sets.items()},
    }
