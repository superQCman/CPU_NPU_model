from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pandas.errors import EmptyDataError


COMBO_RE = re.compile(r"bs(?P<batch>\d+)_nip(?P<nip>\d+)")
TRACE_NODE_RE = re.compile(r"_kernel_time$")
TRACE_OP_RE = re.compile(r"^(?P<op_idx>\d+)")

DTYPE_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "uint16": 2,
    "int16": 2,
    "float": 4,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "double": 8,
    "float64": 8,
    "uint64": 8,
    "int64": 8,
}

HARDWARE_PROFILE_PATH = Path(__file__).resolve().parent / "hardware_profile" / "kunpeng920_single_numa.yaml"


def normalize_node_name(node_name: str | float | int | None) -> str:
    text = "" if node_name is None else str(node_name).strip()
    if not text or text.lower() == "nan":
        return ""
    return TRACE_NODE_RE.sub("", text)


def extract_node_scope(node_name: str | float | int | None) -> str:
    text = normalize_node_name(node_name)
    if not text:
        return "unknown"
    stripped = text.strip("/")
    if not stripped:
        return "root"
    return stripped.split("/", 1)[0]


def split_combo(combo: str | None) -> tuple[int | None, int | None]:
    if not combo:
        return None, None
    match = COMBO_RE.search(str(combo))
    if not match:
        return None, None
    return int(match.group("batch")), int(match.group("nip"))


def parse_trace_op_idx(trace_op_name: str | float | int | None) -> int | None:
    if trace_op_name is None:
        return None
    match = TRACE_OP_RE.search(str(trace_op_name).strip())
    if not match:
        return None
    return int(match.group("op_idx"))


def parse_shape_dims(value: str | float | int | None) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []

    dims: list[int] = []
    for dim in parsed:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        elif isinstance(dim, float) and dim.is_integer() and dim > 0:
            dims.append(int(dim))
    return dims


def _shape_entries(value: str | float | int | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict) or not item:
            continue
        dtype, dims = next(iter(item.items()))
        if not isinstance(dims, list):
            continue
        clean_dims: list[int] = []
        dynamic_dims = 0
        for dim in dims:
            if isinstance(dim, int) and dim > 0:
                clean_dims.append(dim)
            elif isinstance(dim, float) and dim.is_integer() and dim > 0:
                clean_dims.append(int(dim))
            else:
                dynamic_dims += 1
        out.append(
            {
                "dtype": str(dtype),
                "dims": clean_dims,
                "rank": len(clean_dims),
                "dynamic_dims": dynamic_dims,
            }
        )
    return out


def _entry_num_elements(entry: dict[str, Any]) -> float:
    dims = entry.get("dims", [])
    if not dims:
        return 0.0
    product = 1
    for dim in dims:
        product *= int(dim)
    return float(product)


def _entry_request_elements(entry: dict[str, Any]) -> float:
    dims = [int(dim) for dim in entry.get("dims", [])]
    if not dims:
        return 1.0 if int(entry.get("dynamic_dims", 0) or 0) == 0 else 0.0
    product = 1
    for dim in dims:
        product *= int(dim)
    return float(product)


def _infer_gather_request_rows(input_entries: list[dict[str, Any]]) -> float:
    if len(input_entries) < 2:
        return 0.0
    return max(_entry_request_elements(input_entries[1]), 0.0)


def _shape_features(shape_text: str | float | int | None, prefix: str) -> dict[str, float]:
    entries = _shape_entries(shape_text)
    if not entries:
        return {
            f"feat_{prefix}_tensor_count": 0.0,
            f"feat_{prefix}_rank_sum": 0.0,
            f"feat_{prefix}_rank_max": 0.0,
            f"feat_{prefix}_elements_sum": 0.0,
            f"feat_{prefix}_elements_max": 0.0,
            f"feat_{prefix}_bytes_sum": 0.0,
            f"feat_{prefix}_dynamic_dims": 0.0,
        }

    ranks = [float(entry["rank"]) for entry in entries]
    elems = [_entry_num_elements(entry) for entry in entries]
    bytes_list = [
        elem_count * DTYPE_SIZES.get(str(entry["dtype"]).lower(), 4)
        for entry, elem_count in zip(entries, elems)
    ]
    dynamic_dims = [float(entry["dynamic_dims"]) for entry in entries]
    return {
        f"feat_{prefix}_tensor_count": float(len(entries)),
        f"feat_{prefix}_rank_sum": float(sum(ranks)),
        f"feat_{prefix}_rank_max": float(max(ranks)),
        f"feat_{prefix}_elements_sum": float(sum(elems)),
        f"feat_{prefix}_elements_max": float(max(elems)),
        f"feat_{prefix}_bytes_sum": float(sum(bytes_list)),
        f"feat_{prefix}_dynamic_dims": float(sum(dynamic_dims)),
    }


def _safe_ratio_series(numerator: pd.Series, denominator: pd.Series, floor: float = 1.0) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0).clip(lower=floor)
    return num / den


def _product(values: list[int]) -> float:
    if not values:
        return 0.0
    product = 1
    for value in values:
        product *= int(value)
    return float(product)


def _infer_gemm_mnk(
    input_entries: list[dict[str, Any]],
    output_entries: list[dict[str, Any]],
) -> tuple[float, float, float]:
    if not input_entries or not output_entries:
        return 0.0, 0.0, 0.0

    a_dims = [int(dim) for dim in input_entries[0].get("dims", [])]
    b_dims = [int(dim) for dim in input_entries[1].get("dims", [])] if len(input_entries) > 1 else []
    c_dims = [int(dim) for dim in output_entries[0].get("dims", [])]
    if not c_dims:
        return 0.0, 0.0, 0.0

    n_dim = float(c_dims[-1])
    m_dim = float(int(_product(c_dims[:-1])) if len(c_dims) > 1 else 1)
    k_dim = float(a_dims[-1]) if a_dims else 0.0

    if len(b_dims) >= 2:
        if float(b_dims[0]) == n_dim:
            k_dim = float(b_dims[1])
        elif float(b_dims[-1]) == n_dim:
            k_dim = float(b_dims[-2])

    return m_dim, n_dim, k_dim


def _infer_reduction_dims(input_dims: list[int], output_dims: list[int]) -> list[int]:
    if not input_dims or input_dims == output_dims:
        return []

    reduced: list[int] = []
    out_idx = 0
    for dim in input_dims:
        if out_idx < len(output_dims) and dim == output_dims[out_idx]:
            out_idx += 1
        else:
            reduced.append(int(dim))
    if out_idx == len(output_dims):
        return reduced

    return [int(dim) for idx, dim in enumerate(input_dims) if idx >= len(output_dims) or output_dims[idx] != dim]


def _infer_reduction_axis_positions(input_dims: list[int], output_dims: list[int]) -> list[int]:
    if not input_dims or input_dims == output_dims:
        return []

    positions: list[int] = []
    out_idx = 0
    for idx, dim in enumerate(input_dims):
        if out_idx < len(output_dims) and dim == output_dims[out_idx]:
            out_idx += 1
        else:
            positions.append(idx)
    if out_idx == len(output_dims):
        return positions

    return [idx for idx, dim in enumerate(input_dims) if idx >= len(output_dims) or output_dims[idx] != dim]


def serialize_shape_entries(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""

    work = rows.copy()
    work["port_idx"] = pd.to_numeric(work["port_idx"], errors="coerce")
    work = work.sort_values(["port_idx", "tensor_name"], kind="stable")
    entries: list[dict[str, list[Any]]] = []
    for _, row in work.iterrows():
        entries.append(
            {
                str(row.get("dtype", "")).strip(): parse_shape_dims(row.get("shape")),
            }
        )
    return repr(entries) if entries else ""


def load_op_shapes_for_combo(op_shapes_csv: Path) -> pd.DataFrame:
    if not op_shapes_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(op_shapes_csv)
    required = {"node_idx", "node_name", "op_type", "tensor_role", "port_idx", "dtype", "shape"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["node_idx", "node_name", "op_type"], dropna=False):
        node_idx, node_name, op_type = keys
        input_rows = group[group["tensor_role"].astype(str).str.strip().eq("input")]
        output_rows = group[group["tensor_role"].astype(str).str.strip().eq("output")]
        rows.append(
            {
                "op_idx": pd.to_numeric(node_idx, errors="coerce"),
                "node_name_normalized": normalize_node_name(node_name),
                "op_type": "" if pd.isna(op_type) else str(op_type),
                "input_type_shape_op_shapes": serialize_shape_entries(input_rows),
                "output_type_shape_op_shapes": serialize_shape_entries(output_rows),
            }
        )
    return pd.DataFrame(rows)


def merge_op_shapes_into_feature_rows(df: pd.DataFrame, op_shapes_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or op_shapes_df.empty:
        out = df.copy()
        out["input_type_shape"] = ""
        out["output_type_shape"] = ""
        return out

    out = df.copy()
    idx_rows = op_shapes_df.dropna(subset=["op_idx"]).copy()
    if not idx_rows.empty:
        idx_rows["op_idx"] = pd.to_numeric(idx_rows["op_idx"], errors="coerce").astype("Int64")
        out = out.merge(
            idx_rows[["op_idx", "op_type", "input_type_shape_op_shapes", "output_type_shape_op_shapes"]],
            on=["op_idx", "op_type"],
            how="left",
        )
    else:
        out["input_type_shape_op_shapes"] = ""
        out["output_type_shape_op_shapes"] = ""

    name_rows = op_shapes_df[
        ["node_name_normalized", "op_type", "input_type_shape_op_shapes", "output_type_shape_op_shapes"]
    ].drop_duplicates(subset=["node_name_normalized", "op_type"], keep="first")
    out = out.merge(
        name_rows.rename(
            columns={
                "input_type_shape_op_shapes": "input_type_shape_op_shapes_by_name",
                "output_type_shape_op_shapes": "output_type_shape_op_shapes_by_name",
            }
        ),
        on=["node_name_normalized", "op_type"],
        how="left",
    )

    for target in ["input_type_shape", "output_type_shape"]:
        idx_col = f"{target}_op_shapes"
        name_col = f"{target}_op_shapes_by_name"
        existing = out[target].fillna("").astype(str) if target in out.columns else pd.Series("", index=out.index)
        existing = existing.where(~existing.str.lower().eq("nan"), "")
        idx_values = out[idx_col].fillna("").astype(str)
        name_values = out[name_col].fillna("").astype(str)
        merged = existing.where(existing.str.len() > 0, idx_values)
        out[target] = merged.where(merged.str.len() > 0, name_values)

    drop_columns = [
        "input_type_shape_op_shapes",
        "output_type_shape_op_shapes",
        "input_type_shape_op_shapes_by_name",
        "output_type_shape_op_shapes_by_name",
    ]
    return out.drop(columns=[column for column in drop_columns if column in out.columns])


def ensure_runtime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    defaults = {
        "batch_size": 0,
        "num_indices_per_lookup": 0,
        "total_instructions": 0,
        "total_loads": 0,
        "total_stores": 0,
        "num_threads": 1,
        "dur_us": np.nan,
        "output_size": 0.0,
        "activation_size": 0.0,
        "parameter_size": 0.0,
    }
    source_map = {
        "dur_us": "cpu_dur_us_avg",
        "output_size": "cpu_output_size_avg",
        "activation_size": "cpu_activation_size_avg",
        "parameter_size": "cpu_parameter_size_avg",
    }
    for target, source in source_map.items():
        if target not in out.columns or out[target].isna().all():
            if source in out.columns:
                out[target] = out[source]
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default

    numeric_columns = [
        "batch_size",
        "num_indices_per_lookup",
        "total_instructions",
        "total_loads",
        "total_stores",
        "num_threads",
        "dur_us",
        "output_size",
        "activation_size",
        "parameter_size",
        "reuse_time_mean",
        "reuse_distance_mean",
        "reuse_distance_unique_cache_lines_per_k_accesses",
        "opc_branch_ratio",
        "opc_fp_math_ratio",
        "opc_load_ratio",
        "opc_math_ratio",
        "opc_simd_ratio",
        "opc_store_ratio",
        "cpu_dur_us_avg",
        "cpu_output_size_avg",
        "cpu_activation_size_avg",
        "cpu_parameter_size_avg",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_runtime_columns(df)

    input_shape_rows = out["input_type_shape"].map(lambda value: _shape_features(value, "input"))
    output_shape_rows = out["output_type_shape"].map(lambda value: _shape_features(value, "output"))
    input_shape_entries = out["input_type_shape"].map(_shape_entries)
    output_shape_entries = out["output_type_shape"].map(_shape_entries)

    input_shape_df = pd.DataFrame(list(input_shape_rows))
    output_shape_df = pd.DataFrame(list(output_shape_rows))
    out = pd.concat([out.reset_index(drop=True), input_shape_df, output_shape_df], axis=1)

    batch_size = out["batch_size"].fillna(0).clip(lower=0)
    batch_size_safe = batch_size.clip(lower=1)
    output_bytes = out["output_size"].fillna(0).clip(lower=0)
    activation_bytes = out["activation_size"].fillna(0).clip(lower=0)
    parameter_bytes = out["parameter_size"].fillna(0).clip(lower=0)
    op_type = out["op_type"].fillna("").astype(str)
    gemm_mask = op_type.isin(["Gemm", "MatMul"])
    reduction_mask = op_type.eq("ReduceSum")
    gather_mask = op_type.eq("Gather")

    output_elements = output_bytes / 4.0
    activation_elements = activation_bytes / 4.0
    parameter_elements = parameter_bytes / 4.0
    output_elements_per_batch = output_elements / batch_size_safe
    activation_elements_per_batch = activation_elements / batch_size_safe
    configured_lookup_count = batch_size * out["num_indices_per_lookup"].fillna(0).clip(lower=0)
    inferred_lookup_count = pd.Series(
        [_infer_gather_request_rows(entries) for entries in input_shape_entries],
        index=out.index,
        dtype=float,
    )
    lookup_count = inferred_lookup_count.where(inferred_lookup_count > 0.0, configured_lookup_count)
    lookup_count_safe = lookup_count.clip(lower=1)
    io_bytes = output_bytes + activation_bytes + parameter_bytes

    gemm_shape_rows = [
        _infer_gemm_mnk(inp, outp) for inp, outp in zip(input_shape_entries, output_shape_entries)
    ]
    gemm_shape_df = pd.DataFrame(gemm_shape_rows, columns=["feat_gemm_m", "feat_gemm_n", "feat_gemm_k"]).fillna(0.0)

    reduction_rows: list[dict[str, float]] = []
    for inp, outp in zip(input_shape_entries, output_shape_entries):
        input_dims = [int(dim) for dim in inp[0].get("dims", [])] if inp else []
        output_dims = [int(dim) for dim in outp[0].get("dims", [])] if outp else []
        reduced_dims = _infer_reduction_dims(input_dims, output_dims)
        reduced_positions = _infer_reduction_axis_positions(input_dims, output_dims)
        trailing_positions: list[int] = []
        reduced_position_set = set(reduced_positions)
        for idx in range(len(input_dims) - 1, -1, -1):
            if idx in reduced_position_set:
                trailing_positions.append(idx)
            else:
                break
        reduction_rows.append(
            {
                "feat_reduction_axes_count": float(len(reduced_dims)),
                "feat_reduction_axes_product": _product(reduced_dims) if reduced_dims else 0.0,
                "feat_reduction_input_rank": float(len(input_dims)),
                "feat_reduction_output_rank": float(len(output_dims)),
                "feat_reduction_trailing_fraction": (
                    float(len(trailing_positions)) / float(len(reduced_positions))
                    if reduced_positions
                    else 0.0
                ),
            }
        )
    reduction_df = pd.DataFrame(reduction_rows).fillna(0.0)
    shape_features_df = pd.concat([gemm_shape_df, reduction_df], axis=1)

    for column in ["feat_gemm_m", "feat_gemm_n", "feat_gemm_k"]:
        shape_features_df[column] = np.where(gemm_mask, shape_features_df[column].fillna(0.0), 0.0)
    for column in [
        "feat_reduction_axes_count",
        "feat_reduction_axes_product",
        "feat_reduction_input_rank",
        "feat_reduction_output_rank",
        "feat_reduction_trailing_fraction",
    ]:
        shape_features_df[column] = np.where(reduction_mask, shape_features_df[column].fillna(0.0), 0.0)

    out = pd.concat([out.reset_index(drop=True), shape_features_df.reset_index(drop=True)], axis=1)

    gemm_mac_count = out["feat_gemm_m"].fillna(0.0) * out["feat_gemm_n"].fillna(0.0) * out["feat_gemm_k"].fillna(0.0)
    gemm_mac_count = gemm_mac_count.where(gemm_mac_count > 0.0, output_elements * activation_elements_per_batch)

    reduction_factor = _safe_ratio_series(activation_elements, output_elements)
    reduction_factor = reduction_factor.where(
        out["feat_reduction_axes_product"].fillna(0.0).le(0.0),
        out["feat_reduction_axes_product"],
    )
    reduction_work_items = (activation_elements - output_elements).clip(lower=0.0)
    reduction_work_items = reduction_work_items.where(
        out["feat_reduction_axes_product"].fillna(0.0).le(0.0),
        output_elements * np.clip(out["feat_reduction_axes_product"].fillna(0.0) - 1.0, a_min=0.0, a_max=None),
    )

    feat_threads_effective = out["num_threads"].fillna(1).clip(lower=1)
    feat_memory_ops = out["total_loads"].fillna(0) + out["total_stores"].fillna(0)
    feat_working_set_bytes = output_bytes + activation_bytes + parameter_bytes
    engineered_df = pd.DataFrame(
        {
            "feat_memory_ops": feat_memory_ops,
            "feat_working_set_bytes": feat_working_set_bytes,
            "feat_threads_effective": feat_threads_effective,
            "feat_insts_per_thread": out["total_instructions"].fillna(0) / feat_threads_effective,
            "feat_memops_per_inst": feat_memory_ops.fillna(0) / out["total_instructions"].clip(lower=1),
            "feat_output_elements": output_elements,
            "feat_activation_elements": activation_elements,
            "feat_parameter_elements": parameter_elements,
            "feat_lookup_count": np.where(gather_mask, lookup_count, 0.0),
            "feat_output_elements_per_lookup": np.where(
                gather_mask,
                _safe_ratio_series(output_elements, lookup_count_safe),
                0.0,
            ),
            "feat_output_elements_per_batch": output_elements_per_batch,
            "feat_activation_elements_per_batch": activation_elements_per_batch,
            "feat_activation_per_output_element": _safe_ratio_series(activation_elements, output_elements),
            "feat_gemm_mac_count": np.where(gemm_mask, gemm_mac_count, 0.0),
            "feat_gemm_bytes_per_mac": np.where(
                gemm_mask,
                _safe_ratio_series(io_bytes, gemm_mac_count),
                0.0,
            ),
            "feat_reduction_factor": np.where(reduction_mask, reduction_factor, 0.0),
            "feat_reduction_work_items": np.where(reduction_mask, reduction_work_items, 0.0),
            "feat_io_bytes_sum": io_bytes,
            "feat_output_input_bytes_ratio": _safe_ratio_series(
                out["feat_output_bytes_sum"].fillna(output_bytes),
                out["feat_input_bytes_sum"].fillna(0.0),
            ),
        },
        index=out.index,
    )
    out = pd.concat([out.reset_index(drop=True), engineered_df.reset_index(drop=True)], axis=1)
    return out


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (data or {}).items():
        full_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_profile_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    size_match = re.fullmatch(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>KiB|MiB|GiB|KB|MB|GB)", text)
    if size_match:
        scale = {
            "KB": 1000.0,
            "MB": 1000.0**2,
            "GB": 1000.0**3,
            "KiB": 1024.0,
            "MiB": 1024.0**2,
            "GiB": 1024.0**3,
        }[size_match.group("unit")]
        return float(size_match.group("value")) * scale

    freq_match = re.fullmatch(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)", text)
    if freq_match:
        scale = {
            "Hz": 1e-9,
            "kHz": 1e-6,
            "MHz": 1e-3,
            "GHz": 1.0,
        }[freq_match.group("unit")]
        return float(freq_match.group("value")) * scale

    numeric = _safe_float(text, default=np.nan)
    if np.isfinite(numeric):
        return numeric
    return text


def _sanitize_identity_token(value: Any, default: str = "unknown") -> str:
    text = str(value).strip().lower()
    if not text:
        return default
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def load_hardware_profile(profile_path: Path | None = None) -> dict[str, Any]:
    resolved_path = profile_path or HARDWARE_PROFILE_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Hardware profile was not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def infer_hardware_identity(profile: dict[str, Any] | None) -> dict[str, str]:
    payload = profile or {}
    core = payload.get("core", {}) or {}
    instruction = payload.get("instruction", {}) or {}
    profile_name = str(payload.get("profile_name", "")).strip() or "unknown_profile"
    vendor = str(core.get("vendor", "")).strip() or "unknown_vendor"
    model_name = str(core.get("model_name", "")).strip() or "unknown_model"
    isa_family = str(instruction.get("isa_family", "")).strip() or str(core.get("architecture", "")).strip() or "unknown_isa"
    simd_name = str(instruction.get("simd_name", "")).strip() or "unknown_simd"
    backend_family = str(payload.get("backend_family", "")).strip() or "ort_cpu"
    hardware_family = "_".join(
        [
            _sanitize_identity_token(vendor, "unknown_vendor"),
            _sanitize_identity_token(model_name, "unknown_model"),
            _sanitize_identity_token(isa_family, "unknown_isa"),
            _sanitize_identity_token(simd_name, "unknown_simd"),
        ]
    )
    return {
        "hardware_id": _sanitize_identity_token(profile_name, "unknown_hw"),
        "hardware_family": hardware_family,
        "hardware_profile_name": profile_name,
        "backend_family": _sanitize_identity_token(backend_family, "ort_cpu"),
    }


def load_hardware_identity(profile_path: Path | None = None) -> dict[str, str]:
    return infer_hardware_identity(load_hardware_profile(profile_path))


def load_hardware_features(profile_path: Path | None = None) -> dict[str, float]:
    profile = load_hardware_profile(profile_path)
    flat = _flatten_dict(profile)
    out: dict[str, float] = {}
    for key, value in flat.items():
        if key.startswith("paper_cross_check") or key.startswith("host_cross_check") or key in {"profile_name", "source_config", "source_paper", "notes"}:
            continue
        normalized = _normalize_profile_value(value)
        if isinstance(normalized, (int, float)) and normalized is not None and np.isfinite(float(normalized)):
            out[f"hw_{key}"] = float(normalized)
    return out


def add_operator_hardware_context(
    df: pd.DataFrame,
    hardware_features: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    hw = hardware_features or load_hardware_features()
    for key, value in hw.items():
        if key not in out.columns:
            out[key] = float(value)

    threads = pd.to_numeric(
        out["feat_threads_effective"] if "feat_threads_effective" in out.columns else out.get("num_threads", 1.0),
        errors="coerce",
    ).fillna(1.0).clip(lower=1.0)
    total_cores = pd.to_numeric(out.get("hw_core_total_cores", threads), errors="coerce").fillna(threads).clip(lower=1.0)
    active_cores = np.minimum(threads.to_numpy(dtype=float), total_cores.to_numpy(dtype=float))
    out["hw_core_active_cores"] = active_cores
    out["hw_core_active_core_fraction"] = active_cores / total_cores.clip(lower=1.0)

    for cache_column, out_column in [
        ("hw_cache_l1d_size", "hw_cache_l1d_active_bytes"),
        ("hw_cache_l2_size", "hw_cache_l2_active_bytes"),
    ]:
        if cache_column in out.columns:
            cache_size = pd.to_numeric(out[cache_column], errors="coerce").fillna(0.0)
            out[out_column] = cache_size * active_cores

    if "hw_cache_l3_per_die_size" in out.columns:
        l3_size = pd.to_numeric(out["hw_cache_l3_per_die_size"], errors="coerce").fillna(0.0)
        cores_per_die = pd.to_numeric(out.get("hw_core_cores_per_die", total_cores), errors="coerce").fillna(total_cores).clip(lower=1.0)
        total_dies = np.ceil(total_cores.to_numpy(dtype=float) / cores_per_die.to_numpy(dtype=float))
        active_dies = np.ceil(active_cores / cores_per_die.to_numpy(dtype=float))
        active_dies = np.minimum(active_dies, total_dies)
        out["hw_cache_l3_active_dies"] = active_dies
        out["hw_cache_l3_active_bytes"] = l3_size * active_dies
    return out


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def add_analytical_hardware_software_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    feat_io_bytes_sum = _numeric_series(out, "feat_io_bytes_sum", 0.0).clip(lower=0.0)
    feat_output_elements = _numeric_series(out, "feat_output_elements", 0.0).clip(lower=0.0)
    active_cores = _numeric_series(out, "hw_core_active_cores", 1.0).clip(lower=1.0)
    cpu_clock_ghz = _numeric_series(out, "hw_core_cpu_clock", 1.0).clip(lower=1e-6)

    throughput_fma = _numeric_series(
        out,
        "hw_instruction_fp_throughput_per_cycle_vector_sp_fma",
        np.nan,
    )
    throughput_mul = _numeric_series(
        out,
        "hw_instruction_fp_throughput_per_cycle_vector_sp_mul",
        0.0,
    )
    vector_sp_throughput = throughput_fma.where(throughput_fma > 0.0, throughput_mul).clip(lower=0.0)
    simd_width_bits = _numeric_series(out, "hw_instruction_simd_width_bits", 128.0).clip(lower=32.0)
    peak_fp32_ops_per_us = vector_sp_throughput * (simd_width_bits / 32.0) * 2.0 * cpu_clock_ghz * 1e3 * active_cores

    compute_ops = pd.Series(0.0, index=out.index, dtype=float)
    op_type = out.get("op_type", pd.Series("", index=out.index)).fillna("").astype(str)
    gemm_like_mask = op_type.isin(["Gemm", "MatMul"])
    reduction_mask = op_type.eq("ReduceSum")
    elementwise_mask = op_type.isin(["Relu", "Add", "Mul"])
    sigmoid_mask = op_type.eq("Sigmoid")

    gemm_compute_ops = (
        2.0
        * _numeric_series(out, "feat_gemm_m", 0.0).clip(lower=0.0)
        * _numeric_series(out, "feat_gemm_n", 0.0).clip(lower=0.0)
        * _numeric_series(out, "feat_gemm_k", 0.0).clip(lower=0.0)
    )
    compute_ops = compute_ops.where(~gemm_like_mask, gemm_compute_ops)
    compute_ops = compute_ops.where(
        ~reduction_mask,
        _numeric_series(out, "feat_reduction_work_items", 0.0).clip(lower=0.0),
    )
    compute_ops = compute_ops.where(~elementwise_mask, feat_output_elements)
    compute_ops = compute_ops.where(~sigmoid_mask, 4.0 * feat_output_elements)

    mem_bandwidth_bytes_per_us = _numeric_series(out, "hw_memory_bandwidth_gb_s_total", 0.0).clip(lower=1e-6) * 1e3
    compute_time_us = compute_ops / peak_fp32_ops_per_us.clip(lower=1e-6)
    mem_bw_time_us = feat_io_bytes_sum / mem_bandwidth_bytes_per_us
    roofline_base_us = np.maximum(
        compute_time_us.to_numpy(dtype=float),
        mem_bw_time_us.to_numpy(dtype=float),
    )

    l1_ratio = _numeric_series(out, "hw_ratio_working_set_to_l1d_active_bytes", np.inf).replace([np.inf, -np.inf], np.inf)
    l2_ratio = _numeric_series(out, "hw_ratio_working_set_to_l2_active_bytes", np.inf).replace([np.inf, -np.inf], np.inf)
    l3_ratio = _numeric_series(out, "hw_ratio_working_set_to_l3_active_bytes", np.inf).replace([np.inf, -np.inf], np.inf)
    cache_fit_level = np.select(
        [
            l1_ratio.to_numpy(dtype=float) <= 1.0,
            l2_ratio.to_numpy(dtype=float) <= 1.0,
            l3_ratio.to_numpy(dtype=float) <= 1.0,
        ],
        [1.0, 2.0, 3.0],
        default=4.0,
    )

    l1_latency_ns = _numeric_series(out, "hw_cache_l1d_response_latency_cycles", 0.0) / cpu_clock_ghz
    l2_latency_ns = _numeric_series(out, "hw_cache_l2_response_latency_cycles", 0.0) / cpu_clock_ghz
    l3_latency_ns = _numeric_series(out, "hw_cache_l3_per_die_response_latency_cycles", 0.0) / cpu_clock_ghz
    dram_latency_ns = _numeric_series(out, "hw_memory_local_mem_delay_ns", 0.0).clip(lower=0.0)
    expected_latency_ns = np.select(
        [
            cache_fit_level == 1.0,
            cache_fit_level == 2.0,
            cache_fit_level == 3.0,
        ],
        [
            l1_latency_ns.to_numpy(dtype=float),
            l2_latency_ns.to_numpy(dtype=float),
            l3_latency_ns.to_numpy(dtype=float),
        ],
        default=dram_latency_ns.to_numpy(dtype=float),
    )

    cacheline_bytes = _numeric_series(out, "hw_cache_cacheline_bytes", 64.0).clip(lower=1.0)
    access_units = feat_io_bytes_sum / cacheline_bytes
    latency_proxy_us = (
        access_units.to_numpy(dtype=float)
        * expected_latency_ns
        / 1000.0
        / active_cores.to_numpy(dtype=float)
    )

    arithmetic_intensity = compute_ops / feat_io_bytes_sum.clip(lower=1.0)
    ridge_point_ops_per_byte = peak_fp32_ops_per_us / mem_bandwidth_bytes_per_us
    ridge_gap = arithmetic_intensity / ridge_point_ops_per_byte.clip(lower=1e-6)

    out["ana_cache_fit_level"] = cache_fit_level.astype(float)
    out["ana_expected_latency_ns"] = expected_latency_ns.astype(float)
    out["ana_compute_ops"] = compute_ops.to_numpy(dtype=float)
    out["ana_mem_bw_time_us"] = mem_bw_time_us.to_numpy(dtype=float)
    out["ana_latency_proxy_us"] = latency_proxy_us.astype(float)
    out["ana_ridge_gap"] = ridge_gap.to_numpy(dtype=float)
    out["ana_roofline_base_us"] = roofline_base_us.astype(float)
    out["ana_base_us"] = np.maximum(roofline_base_us, 1e-3).astype(float)
    return out


def _robust_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(clean.mean())


def aggregate_branch_op_timeline(op_timeline_df: pd.DataFrame) -> pd.DataFrame:
    if op_timeline_df.empty:
        return pd.DataFrame()

    df = op_timeline_df.copy()
    df["node_name_normalized"] = df["node_name"].map(normalize_node_name)
    df["op_type"] = df["op_name"].fillna("Unknown").astype(str)
    df["task_name"] = df["task_name"].fillna("unknown").astype(str)
    df["lane"] = pd.to_numeric(df.get("lane"), errors="coerce").fillna(0.0)
    df["start_us"] = pd.to_numeric(df.get("start_us"), errors="coerce")
    df["end_us"] = pd.to_numeric(df.get("end_us"), errors="coerce")
    df["dur_us"] = pd.to_numeric(df.get("dur_us"), errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df[
        (df["node_name_normalized"] != "")
        & df["start_us"].notna()
        & df["end_us"].notna()
        & (df["dur_us"] > 0.0)
    ].copy()
    if df.empty:
        return pd.DataFrame()

    events = df.to_dict(orient="records")
    overlap_us = [0.0] * len(events)
    weighted_other_active = [0.0] * len(events)
    weighted_other_tasks = [0.0] * len(events)
    cross_task_overlap_us = [0.0] * len(events)
    same_op_overlap_us = [0.0] * len(events)

    boundaries: list[tuple[float, int, int]] = []
    for idx, event in enumerate(events):
        boundaries.append((float(event["start_us"]), 1, idx))
        boundaries.append((float(event["end_us"]), 0, idx))
    boundaries.sort()

    active: set[int] = set()
    active_task_counts: Counter[str] = Counter()
    active_op_counts: Counter[str] = Counter()
    prev_time: float | None = None

    for time_us, event_type, idx in boundaries:
        if prev_time is not None and time_us > prev_time and active:
            interval = time_us - prev_time
            active_count = len(active)
            active_task_total = len(active_task_counts)
            if active_count > 1:
                for active_idx in active:
                    event = events[active_idx]
                    other_active = active_count - 1
                    overlap_us[active_idx] += interval
                    weighted_other_active[active_idx] += other_active * interval

                    other_tasks = active_task_total - 1
                    weighted_other_tasks[active_idx] += max(other_tasks, 0) * interval
                    if other_tasks > 0:
                        cross_task_overlap_us[active_idx] += interval

                    same_op_other = active_op_counts[str(event["op_type"])] - 1
                    if same_op_other > 0:
                        same_op_overlap_us[active_idx] += interval

        prev_time = time_us
        event = events[idx]
        if event_type == 1:
            active.add(idx)
            active_task_counts[str(event["task_name"])] += 1
            active_op_counts[str(event["op_type"])] += 1
        else:
            active.discard(idx)
            active_task_counts[str(event["task_name"])] -= 1
            if active_task_counts[str(event["task_name"])] <= 0:
                active_task_counts.pop(str(event["task_name"]), None)
            active_op_counts[str(event["op_type"])] -= 1
            if active_op_counts[str(event["op_type"])] <= 0:
                active_op_counts.pop(str(event["op_type"]), None)

    per_node_samples: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for idx, event in enumerate(events):
        dur = max(float(event["dur_us"]), 1e-9)
        per_node_samples[(str(event["node_name_normalized"]), str(event["op_type"]))].append(
            {
                "overlap_ratio": overlap_us[idx] / dur,
                "cross_task_overlap_ratio": cross_task_overlap_us[idx] / dur,
                "same_op_overlap_ratio": same_op_overlap_us[idx] / dur,
                "mean_other_active": weighted_other_active[idx] / dur,
                "mean_other_tasks": weighted_other_tasks[idx] / dur,
            }
        )

    rows: list[dict[str, Any]] = []
    for (node_name_normalized, op_type), samples in sorted(per_node_samples.items()):
        row = {
            "node_name_normalized": node_name_normalized,
            "op_type": op_type,
            "local_ctx_overlap_ratio_mean": _robust_mean(pd.Series([sample["overlap_ratio"] for sample in samples])),
            "local_ctx_cross_task_overlap_ratio_mean": _robust_mean(
                pd.Series([sample["cross_task_overlap_ratio"] for sample in samples])
            ),
            "local_ctx_same_op_overlap_ratio_mean": _robust_mean(
                pd.Series([sample["same_op_overlap_ratio"] for sample in samples])
            ),
            "local_ctx_mean_other_active_mean": _robust_mean(
                pd.Series([sample["mean_other_active"] for sample in samples])
            ),
            "local_ctx_mean_other_tasks_mean": _robust_mean(
                pd.Series([sample["mean_other_tasks"] for sample in samples])
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_concurrency_summary(segments_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    if segments_df.empty or "dur_us" not in segments_df.columns or "concurrency" not in segments_df.columns:
        return {
            f"{prefix}_parallel_fraction": 0.0,
            f"{prefix}_weighted_mean_parallel_concurrency": 1.0,
        }

    df = segments_df.copy()
    df["dur_us"] = pd.to_numeric(df["dur_us"], errors="coerce").fillna(0.0)
    df["concurrency"] = pd.to_numeric(df["concurrency"], errors="coerce").fillna(0.0)
    df["start_us"] = pd.to_numeric(df.get("start_us"), errors="coerce")
    df["end_us"] = pd.to_numeric(df.get("end_us"), errors="coerce")
    df = df[df["dur_us"] > 0].copy()
    if df.empty:
        return {
            f"{prefix}_parallel_fraction": 0.0,
            f"{prefix}_weighted_mean_parallel_concurrency": 1.0,
        }

    parallel_only = df[df["concurrency"] >= 2]
    parallel_us = float(parallel_only["dur_us"].sum()) if not parallel_only.empty else 0.0
    if {"start_us", "end_us"}.issubset(df.columns):
        makespan = float(df["end_us"].max() - df["start_us"].min())
    else:
        makespan = float(df["dur_us"].sum())
    parallel_weighted_mean = (
        float(np.average(parallel_only["concurrency"], weights=parallel_only["dur_us"]))
        if not parallel_only.empty
        else 1.0
    )
    return {
        f"{prefix}_parallel_fraction": 0.0 if makespan <= 0 else parallel_us / makespan,
        f"{prefix}_weighted_mean_parallel_concurrency": parallel_weighted_mean,
    }


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def build_stage2_candidate_feature_frame(
    df: pd.DataFrame,
    combo_profile_dir: Path,
    hardware_profile_path: Path | None = None,
) -> pd.DataFrame:
    out = add_operator_hardware_context(df, load_hardware_features(hardware_profile_path))

    op_timeline_path = combo_profile_dir / "branch_parallel_op_timeline.csv"
    task_segments_path = combo_profile_dir / "branch_parallel_concurrency_segments.csv"
    op_segments_path = combo_profile_dir / "branch_parallel_op_concurrency_segments.csv"

    if op_timeline_path.exists():
        local_ctx_df = aggregate_branch_op_timeline(pd.read_csv(op_timeline_path, low_memory=False))
        out = out.merge(local_ctx_df, on=["node_name_normalized", "op_type"], how="left")
    else:
        for column in [
            "local_ctx_overlap_ratio_mean",
            "local_ctx_cross_task_overlap_ratio_mean",
            "local_ctx_same_op_overlap_ratio_mean",
            "local_ctx_mean_other_active_mean",
            "local_ctx_mean_other_tasks_mean",
        ]:
            out[column] = np.nan

    combo_features: dict[str, float] = {}
    if task_segments_path.exists():
        combo_features.update(weighted_concurrency_summary(safe_read_csv(task_segments_path), "combo_task"))
    if op_segments_path.exists():
        combo_features.update(weighted_concurrency_summary(safe_read_csv(op_segments_path), "combo_op"))
    for key, value in combo_features.items():
        out[key] = float(value)

    feat_threads_effective = pd.to_numeric(out.get("feat_threads_effective", out.get("num_threads", 1.0)), errors="coerce").fillna(1.0).clip(lower=1.0)
    feat_working_set_bytes = pd.to_numeric(out.get("feat_working_set_bytes", out.get("feat_io_bytes_sum", 0.0)), errors="coerce").fillna(0.0).clip(lower=0.0)
    total_cores = pd.to_numeric(out.get("hw_core_total_cores", 1.0), errors="coerce").fillna(1.0).clip(lower=1.0)
    local_active_ops = 1.0 + pd.to_numeric(out.get("local_ctx_mean_other_active_mean", 0.0), errors="coerce").fillna(0.0)

    out["hw_ratio_threads_to_total_cores"] = feat_threads_effective / total_cores
    out["hw_ratio_working_set_to_l1d_active_bytes"] = _safe_ratio_series(
        feat_working_set_bytes,
        pd.to_numeric(out.get("hw_cache_l1d_active_bytes", 0.0), errors="coerce").fillna(0.0),
    )
    out["hw_ratio_working_set_to_l2_active_bytes"] = _safe_ratio_series(
        feat_working_set_bytes,
        pd.to_numeric(out.get("hw_cache_l2_active_bytes", 0.0), errors="coerce").fillna(0.0),
    )
    out["hw_ratio_working_set_to_l3_active_bytes"] = _safe_ratio_series(
        feat_working_set_bytes,
        pd.to_numeric(out.get("hw_cache_l3_active_bytes", 0.0), errors="coerce").fillna(0.0),
    )
    out["comp_feat_pressure_threads"] = feat_threads_effective * local_active_ops
    out["comp_feat_pressure_ws_to_l2_ratio"] = _safe_ratio_series(
        feat_working_set_bytes * local_active_ops,
        pd.to_numeric(out.get("hw_cache_l2_active_bytes", 0.0), errors="coerce").fillna(0.0),
    )
    out["comp_feat_pressure_ws_to_l3_ratio"] = _safe_ratio_series(
        feat_working_set_bytes * local_active_ops,
        pd.to_numeric(out.get("hw_cache_l3_active_bytes", 0.0), errors="coerce").fillna(0.0),
    )
    return add_analytical_hardware_software_features(out)
