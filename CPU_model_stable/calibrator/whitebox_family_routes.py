from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytical_base import run as base_run

try:
    import yaml
except ImportError:  # pragma: no cover - deployment portability
    yaml = None

try:
    from .ml_semantic_routes import (
        WHITEBOX_PRIOR_ONLY_ROUTE,
        WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
        WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
        WHITEBOX_ORT_KERNEL_RF_ROUTE,
        WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
        WHITEBOX_WORKLOAD_BENCHMARK_ROUTE,
        family_bucket_for_op_type,
    )
    from .static_feature_contract import augment_static_feature_frame
    from .whitebox_perf_feature_rf import (
        RF_GROUP_GEMM,
        RF_GROUP_INDEXED_COPY,
        RF_GROUP_METADATA,
        RF_GROUP_PAYLOAD_CONCAT,
        RF_GROUP_REDUCTION,
        RF_GROUP_TINY_SHAPE,
        RF_GROUP_TRANSPOSE,
        BENCHMARK_RF_FEATURE_LIMIT,
        MONOTONIC_POSITIVE_ANCHOR_FEATURES,
        RF_FEATURE_LIMIT,
        fit_family_feature_rf,
        predict_family_feature_rf,
        rf_group_feature_schema,
        rf_group_spec,
    )
except ImportError:
    from ml_semantic_routes import (  # type: ignore
        WHITEBOX_PRIOR_ONLY_ROUTE,
        WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
        WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
        WHITEBOX_ORT_KERNEL_RF_ROUTE,
        WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
        WHITEBOX_WORKLOAD_BENCHMARK_ROUTE,
        family_bucket_for_op_type,
    )
    from static_feature_contract import augment_static_feature_frame  # type: ignore
    from whitebox_perf_feature_rf import (  # type: ignore
        RF_GROUP_GEMM,
        RF_GROUP_INDEXED_COPY,
        RF_GROUP_METADATA,
        RF_GROUP_PAYLOAD_CONCAT,
        RF_GROUP_REDUCTION,
        RF_GROUP_TINY_SHAPE,
        RF_GROUP_TRANSPOSE,
        BENCHMARK_RF_FEATURE_LIMIT,
        MONOTONIC_POSITIVE_ANCHOR_FEATURES,
        RF_FEATURE_LIMIT,
        fit_family_feature_rf,
        predict_family_feature_rf,
        rf_group_feature_schema,
        rf_group_spec,
    )


OP_THREAD_SCALE_KIND = "op_type_num_threads_median_ratio"
OP_THREAD_SCALE_SCHEMA_VERSION = 1
OP_THREAD_SCALE_DEFAULT = 1.0
OP_THREAD_SCALE_MIN_ROWS_DEFAULT = 4


HARD_POLICIES = {
    "batch_size",
    "num_indices_per_lookup",
    "arch_embedding_size",
    "arch_mlp_bot",
    "inter_threads_decoupled_v2",
    "num_threads_decoupled_v2",
    "joint_6d_tuple",
}

PURE_ZERO_FIT_MODE = "pure_zero_fit"
WHITEBOX_FEATURE_ML_MODE = "whitebox_perf_feature_rf"
WHITEBOX_PHYSICS_GUARDED_RF_MODE = "whitebox_perf_feature_rf_physics_guarded_v3"
WHITEBOX_ORT_KERNEL_RF_MODE = "whitebox_perf_feature_rf_ort_kernel_v3"
WHITEBOX_EMBEDDING_MEM_RF_MODE = "whitebox_perf_feature_rf_embedding_mem_v4"
WHITEBOX_GATHER_SLICE_RF_MODE = "whitebox_perf_feature_rf_gather_slice_v5"
WHITEBOX_GATHER_SLICE_COMPACT_RF_MODE = "whitebox_perf_feature_rf_gather_slice_v5_compact_probe"
WHITEBOX_WORKLOAD_BENCHMARK_MODE = "whitebox_perf_feature_rf_workload_benchmark"
WHITEBOX_MODULE_SCHEMA_VERSION = 5
PHYSICS_GUARDED_SYNTHETIC_WEIGHT_COLUMN = "__physics_guarded_weight"
PHYSICS_GUARDED_SYNTHETIC_INDICATOR_COLUMN = "__physics_guarded_synthetic"
PHYSICS_GUARDED_SYNTHETIC_SOURCE_INDEX_COLUMN = "__physics_guarded_source_index"
PHYSICS_GUARDED_COUNTERFACTUAL_WEIGHT = 0.25
PHYSICS_GUARD_DELTA_CAP_MULTIPLIER = 2.0
PHYSICS_GUARD_DELTA_TOL_US = 1e-6
PHYSICS_GUARDED_EXCLUDED_FEATURES = {
    "share_descriptor_center",
    "share_stream_mem_center",
    "share_random_mem_center",
    "share_compute_center",
    "share_reorder_center",
    "share_sync_center",
    "share_dispatch_center",
    "bottleneck_margin_center",
    "launch_vs_useful_ratio_center",
    "random_vs_stream_ratio_center",
    "mem_vs_compute_ratio_center",
    "gather_slice_vs_current_stream_ratio",
}
PHYSICS_GUARDED_COUNTERFACTUAL_FIELD_FACTORS = (
    ("cache.l1d.size", 0.8),
    ("cache.l1d.size", 1.2),
    ("cache.l2.size", 0.8),
    ("cache.l2.size", 1.2),
    ("cache.l3_per_die.size", 0.8),
    ("cache.l3_per_die.size", 1.2),
    ("cache.l1d.response_latency_cycles", 0.8),
    ("cache.l1d.response_latency_cycles", 1.2),
    ("cache.l2.response_latency_cycles", 0.8),
    ("cache.l2.response_latency_cycles", 1.2),
    ("cache.l3_per_die.response_latency_cycles", 0.8),
    ("cache.l3_per_die.response_latency_cycles", 1.2),
    ("memory.copy_bw_gb_s_1t", 0.8),
    ("memory.copy_bw_gb_s_1t", 1.2),
    ("memory.copy_bw_gb_s_saturated", 0.8),
    ("memory.copy_bw_gb_s_saturated", 1.2),
    ("memory.random_gather_bw_gb_s_1t", 0.8),
    ("memory.random_gather_bw_gb_s_1t", 1.2),
    ("memory.random_gather_bw_gb_s_saturated", 0.8),
    ("memory.random_gather_bw_gb_s_saturated", 1.2),
    ("memory.local_mem_delay_ns", 0.8),
    ("memory.local_mem_delay_ns", 1.2),
    ("core.cpu_clock", 0.8),
    ("core.cpu_clock", 1.2),
    ("instruction.simd_width_bits", 0.8),
    ("instruction.simd_width_bits", 1.2),
    ("instruction.fp_throughput_per_cycle.vector_sp_fma", 0.8),
    ("instruction.fp_throughput_per_cycle.vector_sp_fma", 1.2),
    ("instruction.fp_latency_cycles.vector_sp_fma", 0.8),
    ("instruction.fp_latency_cycles.vector_sp_fma", 1.2),
    ("pipeline.fetch_width", 0.8),
    ("pipeline.fetch_width", 1.2),
    ("pipeline.decode_width", 0.8),
    ("pipeline.decode_width", 1.2),
    ("pipeline.rename_width", 0.8),
    ("pipeline.rename_width", 1.2),
    ("pipeline.dispatch_width", 0.8),
    ("pipeline.dispatch_width", 1.2),
    ("pipeline.issue_width", 0.8),
    ("pipeline.issue_width", 1.2),
    ("pipeline.commit_width", 0.8),
    ("pipeline.commit_width", 1.2),
)
MODULE_NAMES = [
    "descriptor",
    "stream_mem",
    "random_mem",
    "compute",
    "reorder",
    "sync",
    "dispatch",
]
DISABLED_ANALYTICAL_MODULES = {
    "descriptor",
    "reorder",
    "sync",
}

MODULE_COLUMNS = [f"svc_{name}_us" for name in MODULE_NAMES]
BOTTLENECK_ORDER = list(MODULE_COLUMNS)
MODULE_TO_SHARE = {f"svc_{name}_us": f"share_{name}" for name in MODULE_NAMES}
GATHER_SLICE_COMPACT_FEATURE_SET_NAME = "compact_29_keep_thread_slice_ratio"
GATHER_SLICE_COMPACT_INDEXED_FEATURES = [
    "whitebox_center_us",
    "whitebox_band_width_us",
    "whitebox_exec_center_us",
    "whitebox_serial_overhead_center_us",
    "share_stream_mem_center",
    "share_random_mem_center",
    "share_sync_center",
    "launch_vs_useful_ratio_center",
    "random_vs_stream_ratio_center",
    "feat_lookup_count",
    "gather_ort_block_size_bytes",
    "gather_ort_task_count",
    "gather_ort_index_scan_count",
    "gather_ort_grain_bytes",
    "gather_effective_head_touch_count",
    "gather_slice_vs_current_stream_ratio",
    "gather_slice_underpredict_gap_us",
    "num_threads",
    "log_num_threads",
    "effective_worker_count",
    "active_parallel_groups",
    "thread_pressure_ratio",
    "parallel_exposure_factor",
    "is_single_intra",
    "worker_shard_count_static",
    "submodel_role_static_code",
    "exec_region_static_code",
    "static_parallel_branch_count",
    "static_distance_to_merge",
]
GATHER_SLICE_COMPACT_REMOVED_FEATURE_GROUPS = {
    "redundant_byte_scale": [
        "exact_input_tensor_bytes_sum",
        "exact_output_tensor_bytes_sum",
        "exact_read_bytes",
        "exact_write_bytes",
        "output_size",
    ],
    "ort_payload_redundancy": [
        "gather_ort_dest_payload_bytes",
        "gather_tail_copy_bytes",
        "gather_ort_source_payload_bytes",
    ],
    "single_thread_secondary": ["is_single_inter"],
    "static_low_frequency": ["branch_lifetime_class_code", "phase_barrier_count_static"],
    "slice_raw_diagnostics": ["gather_slice_copy_bw_gb_s", "gather_slice_stream_us"],
}
REQUIRED_WHITEBOX_HW_FIELDS = [
    "memory.copy_bw_gb_s_1t",
    "memory.copy_bw_gb_s_saturated",
    "memory.random_gather_bw_gb_s_1t",
    "memory.random_gather_bw_gb_s_saturated",
    "runtime.parallel_launch_overhead_us_intra",
    "runtime.parallel_launch_overhead_us_inter",
]

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

_SHAPE_SEGMENT_RE = re.compile(r":\s*\[(.*?)\]")
_TYPED_TENSOR_RE = re.compile(r"\{'([^']+)':\s*\[(.*?)\]\}")

EXEC_MODE_METADATA = "metadata_view_ops"
EXEC_MODE_TINY_SHAPE_ARITH = "tiny_shape_arith_ops"
EXEC_MODE_INDEXED_BLOCK_COPY = "indexed_block_copy_ops"
EXEC_MODE_PAYLOAD_CONCAT = "payload_concat_ops"
EXEC_MODE_SHAPE_CONCAT = "shape_concat_ops"
EXEC_MODE_TRANSPOSE = "transpose_generic_ops"
EXEC_MODE_REDUCTION = "reduction_layout_ops"
EXEC_MODE_GEMM = "gemm_like_ops"
EXEC_MODE_OTHER = "other_ops"

EXEC_MODE_TO_FAMILY = {
    EXEC_MODE_METADATA: "meta_like",
    EXEC_MODE_TINY_SHAPE_ARITH: "meta_like",
    EXEC_MODE_SHAPE_CONCAT: "meta_like",
    EXEC_MODE_INDEXED_BLOCK_COPY: "gather_like",
    EXEC_MODE_PAYLOAD_CONCAT: "copy_like",
    EXEC_MODE_TRANSPOSE: "copy_like",
    EXEC_MODE_REDUCTION: "reduce_like",
    EXEC_MODE_GEMM: "gemm_like",
}

EXEC_MODE_TO_RF_GROUP = {
    EXEC_MODE_METADATA: RF_GROUP_METADATA,
    EXEC_MODE_TINY_SHAPE_ARITH: RF_GROUP_TINY_SHAPE,
    EXEC_MODE_SHAPE_CONCAT: RF_GROUP_METADATA,
    EXEC_MODE_INDEXED_BLOCK_COPY: RF_GROUP_INDEXED_COPY,
    EXEC_MODE_PAYLOAD_CONCAT: RF_GROUP_PAYLOAD_CONCAT,
    EXEC_MODE_TRANSPOSE: RF_GROUP_TRANSPOSE,
    EXEC_MODE_REDUCTION: RF_GROUP_REDUCTION,
    EXEC_MODE_GEMM: RF_GROUP_GEMM,
}

RF_GROUP_OPTYPE_PRIOR_FALLBACK = {
    RF_GROUP_METADATA: {"Shape"},
}

WHITEBOX_WORKLOAD_BENCHMARK_VARIANTS = ("upper", "compact")
WHITEBOX_WORKLOAD_BENCHMARK_LIMITS = {
    "upper": int(BENCHMARK_RF_FEATURE_LIMIT),
    "compact": int(RF_FEATURE_LIMIT),
}

ANALYTICAL_PHYSICAL_LITERAL_ALLOWLIST = {
    "0",
    "0.0",
    "0.25",
    "0.5",
    "0.8",
    "1",
    "1.0",
    "1.2",
    "1e-06",
    "1e-09",
    "2",
    "2.0",
    "3",
    "3.0",
    "4",
    "4.0",
    "5",
    "8.0",
    "16.0",
    "20.0",
    "32.0",
    "64.0",
    "100.0",
    "128.0",
    "512.0",
    "1000.0",
    "1024.0",
}

_ANALYTICAL_PHYSICAL_AUDIT_DONE = False


def _random_memory_latency_composition_mode() -> str:
    value = os.environ.get("WHITEBOX_RANDOM_MEM_LATENCY_COMPOSITION", "max").strip().lower()
    if value in {"add", "additive", "sum"}:
        return "additive"
    return "max"


def _random_memory_latency_tail_alpha() -> float:
    raw_value = os.environ.get("WHITEBOX_RANDOM_MEM_LATENCY_TAIL_ALPHA", "0").strip()
    try:
        return float(np.clip(float(raw_value), a_min=0.0, a_max=None))
    except ValueError:
        return 0.0


def _compute_latency_tail_alpha() -> float:
    raw_value = os.environ.get("WHITEBOX_COMPUTE_LATENCY_TAIL_ALPHA", "0").strip()
    try:
        return float(np.clip(float(raw_value), a_min=0.0, a_max=None))
    except ValueError:
        return 0.0


def _freeze_override_cache_latency() -> bool:
    value = os.environ.get("WHITEBOX_OVERRIDE_FREEZE_CACHE_LATENCY", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _use_static_cache_residency_ratios() -> bool:
    value = os.environ.get("WHITEBOX_USE_STATIC_CACHE_RESIDENCY_RATIOS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _continuous_cache_residency() -> bool:
    value = os.environ.get("WHITEBOX_CONTINUOUS_CACHE_RESIDENCY", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _assert_physical_constant_audit() -> None:
    global _ANALYTICAL_PHYSICAL_AUDIT_DONE
    if _ANALYTICAL_PHYSICAL_AUDIT_DONE:
        return
    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    cutoff_line = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fit_whitebox_family_prior_bundle":
            cutoff_line = int(node.lineno)
            break
    if cutoff_line is None:
        raise RuntimeError("physical constant audit could not locate analytical boundary")

    literals: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Constant(self, node: ast.Constant) -> None:  # type: ignore[override]
            if getattr(node, "lineno", cutoff_line) >= cutoff_line:
                return
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                literals.add(repr(node.value))

    _Visitor().visit(module)
    undocumented = sorted(literals.difference(ANALYTICAL_PHYSICAL_LITERAL_ALLOWLIST))
    if undocumented:
        raise RuntimeError(
            "Analytical prior contains undocumented numeric literals without a physical/structural justification: "
            + ", ".join(undocumented)
        )
    _ANALYTICAL_PHYSICAL_AUDIT_DONE = True


def _nested_get(payload: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _nested_set(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    node: Any = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict):
            return
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    if isinstance(node, dict):
        node[parts[-1]] = value


def _parse_quantity(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("GHz"):
        return float(text[:-3])
    if text.endswith("MHz"):
        return float(text[:-3]) / 1000.0
    if text.endswith("ns"):
        return float(text[:-2])
    if text.endswith("KiB"):
        return float(text[:-3]) * 1024.0
    if text.endswith("MiB"):
        return float(text[:-3]) * 1024.0 * 1024.0
    if text.endswith("GiB"):
        return float(text[:-3]) * 1024.0 * 1024.0 * 1024.0
    return float(text)


def _load_hw_payload(hw_config: Path | dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(hw_config, dict):
        return dict(hw_config), "<override_dict>"
    if yaml is None:
        return {}, str(Path(hw_config).resolve())
    path = Path(hw_config).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Hardware config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload, str(path)


def _scaled_hw_payload(payload: dict[str, Any], field: str, factor: float) -> dict[str, Any] | None:
    original = _nested_get(payload, field)
    numeric = _parse_quantity(original)
    if numeric is None or not np.isfinite(numeric) or numeric <= 0.0:
        return None
    mutated = copy.deepcopy(payload)
    _nested_set(mutated, field, float(numeric) * float(factor))
    return mutated


def _parse_gather_slice_copy_bw_points(payload: dict[str, Any]) -> tuple[list[dict[str, float]], float]:
    config = _nested_get(payload, "memory.gather_slice_copy_bw", {})
    if not isinstance(config, dict):
        return [], 0.0
    measured_slices = float(_parse_quantity(config.get("measured_slices")) or 0.0)
    points: list[dict[str, float]] = []
    raw_points = config.get("points", [])
    if not isinstance(raw_points, list):
        return points, measured_slices
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        block_size = float(_parse_quantity(item.get("block_size_bytes")) or 0.0)
        num_threads = float(_parse_quantity(item.get("num_threads")) or 0.0)
        write_gb_s = float(_parse_quantity(item.get("write_gb_s")) or 0.0)
        if block_size <= 0.0 or num_threads <= 0.0 or write_gb_s <= 0.0:
            continue
        points.append(
            {
                "block_size_bytes": block_size,
                "num_threads": num_threads,
                "write_gb_s": write_gb_s,
            }
        )
    return points, measured_slices


def _derive_random_bw_from_latency(payload: dict[str, Any], total_cores: float, stream_bw_total: float, cacheline_bytes: float) -> tuple[float, float]:
    mem_delay_ns = float(_parse_quantity(_nested_get(payload, "memory.local_mem_delay_ns", 100.0)) or 100.0)
    random_bw_1t = max(cacheline_bytes / max(mem_delay_ns, 1e-6), 1e-6)
    random_bw_sat = min(max(random_bw_1t * max(total_cores, 1.0), 1e-6), max(stream_bw_total, random_bw_1t))
    return random_bw_1t, random_bw_sat


def _derive_launch_overheads(payload: dict[str, Any], total_cores: float, cpu_clock: float) -> tuple[float, float]:
    l3_cycles = float(
        _parse_quantity(_nested_get(payload, "cache.l3_per_die.response_latency_cycles"))
        or _parse_quantity(_nested_get(payload, "cache.l3.response_latency_cycles"))
        or 100.0
    )
    barrier_roundtrip_us = max((2.0 * max(total_cores, 1.0) * l3_cycles) / max(cpu_clock * 1000.0, 1e-6), 1e-6)
    return barrier_roundtrip_us, barrier_roundtrip_us


def _load_hw_params(hw_config: Path | dict[str, Any], *, require_measured_whitebox: bool = False) -> dict[str, Any]:
    payload, source = _load_hw_payload(hw_config)

    total_cores = float(_parse_quantity(_nested_get(payload, "core.total_cores", 1.0)) or 1.0)
    cpu_clock = float(_parse_quantity(_nested_get(payload, "core.cpu_clock", 1.0)) or 1.0)
    simd_width_bits = float(_parse_quantity(_nested_get(payload, "instruction.simd_width_bits", 128.0)) or 128.0)
    fetch_width = float(_parse_quantity(_nested_get(payload, "pipeline.fetch_width", 1.0)) or 1.0)
    decode_width = float(_parse_quantity(_nested_get(payload, "pipeline.decode_width", fetch_width)) or fetch_width)
    rename_width = float(_parse_quantity(_nested_get(payload, "pipeline.rename_width", decode_width)) or decode_width)
    dispatch_width = float(_parse_quantity(_nested_get(payload, "pipeline.dispatch_width", rename_width)) or rename_width)
    issue_width = float(_parse_quantity(_nested_get(payload, "pipeline.issue_width", dispatch_width)) or dispatch_width)
    commit_width = float(_parse_quantity(_nested_get(payload, "pipeline.commit_width", issue_width)) or issue_width)

    fp_tp_add = float(_parse_quantity(_nested_get(payload, "instruction.fp_throughput_per_cycle.vector_sp_add", 1.0)) or 1.0)
    fp_tp_mul = float(_parse_quantity(_nested_get(payload, "instruction.fp_throughput_per_cycle.vector_sp_mul", fp_tp_add)) or fp_tp_add)
    fp_tp_fma = float(_parse_quantity(_nested_get(payload, "instruction.fp_throughput_per_cycle.vector_sp_fma", max(fp_tp_add, fp_tp_mul))) or max(fp_tp_add, fp_tp_mul))
    fp_lat_add = float(_parse_quantity(_nested_get(payload, "instruction.fp_latency_cycles.vector_sp_add", 1.0)) or 1.0)
    fp_lat_mul = float(_parse_quantity(_nested_get(payload, "instruction.fp_latency_cycles.vector_sp_mul", fp_lat_add)) or fp_lat_add)
    fp_lat_fma = float(_parse_quantity(_nested_get(payload, "instruction.fp_latency_cycles.vector_sp_fma", max(fp_lat_add, fp_lat_mul))) or max(fp_lat_add, fp_lat_mul))
    stream_bw_total = float(_parse_quantity(_nested_get(payload, "memory.bandwidth_gb_s_total", 1.0)) or 1.0)
    cacheline_bytes = float(
        _parse_quantity(_nested_get(payload, "cache.cacheline_bytes"))
        or _parse_quantity(_nested_get(payload, "memory.cacheline_bytes"))
        or 64.0
    )
    local_mem_delay_ns = float(_parse_quantity(_nested_get(payload, "memory.local_mem_delay_ns", 100.0)) or 100.0)

    copy_bw_sat = _parse_quantity(_nested_get(payload, "memory.copy_bw_gb_s_saturated"))
    copy_bw_1t = _parse_quantity(_nested_get(payload, "memory.copy_bw_gb_s_1t"))
    random_bw_sat = _parse_quantity(_nested_get(payload, "memory.random_gather_bw_gb_s_saturated"))
    random_bw_1t = _parse_quantity(_nested_get(payload, "memory.random_gather_bw_gb_s_1t"))
    launch_intra = _parse_quantity(_nested_get(payload, "runtime.parallel_launch_overhead_us_intra"))
    launch_inter = _parse_quantity(_nested_get(payload, "runtime.parallel_launch_overhead_us_inter"))

    l1d_size = float(_parse_quantity(_nested_get(payload, "cache.l1d.size", 64.0 * 1024.0)) or 64.0 * 1024.0)
    l2_size = float(_parse_quantity(_nested_get(payload, "cache.l2.size", 512.0 * 1024.0)) or 512.0 * 1024.0)
    l3_size = float(
        _parse_quantity(_nested_get(payload, "cache.l3_per_die.size"))
        or _parse_quantity(_nested_get(payload, "cache.l3.size"))
        or 8.0 * 1024.0 * 1024.0
    )
    l1_latency = float(_parse_quantity(_nested_get(payload, "cache.l1d.response_latency_cycles", 8.0)) or 8.0)
    l2_latency = float(_parse_quantity(_nested_get(payload, "cache.l2.response_latency_cycles", 20.0)) or 20.0)
    l3_latency = float(
        _parse_quantity(_nested_get(payload, "cache.l3_per_die.response_latency_cycles"))
        or _parse_quantity(_nested_get(payload, "cache.l3.response_latency_cycles"))
        or 100.0
    )
    l1_latency_ns = float(l1_latency / max(cpu_clock, 1e-6))
    l2_latency_ns = float(l2_latency / max(cpu_clock, 1e-6))
    l3_latency_ns = float(l3_latency / max(cpu_clock, 1e-6))

    missing_whitebox_fields = [
        dotted_key
        for dotted_key, parsed in (
            ("memory.copy_bw_gb_s_1t", copy_bw_1t),
            ("memory.copy_bw_gb_s_saturated", copy_bw_sat),
            ("memory.random_gather_bw_gb_s_1t", random_bw_1t),
            ("memory.random_gather_bw_gb_s_saturated", random_bw_sat),
            ("runtime.parallel_launch_overhead_us_intra", launch_intra),
            ("runtime.parallel_launch_overhead_us_inter", launch_inter),
        )
        if parsed is None
    ]
    if require_measured_whitebox and missing_whitebox_fields:
        raise ValueError(
            f"Whitebox routes require measured hardware fields in {source}: {', '.join(missing_whitebox_fields)}"
        )

    derived_copy_bw_1t = max(stream_bw_total / max(total_cores, 1.0), 1e-6)
    derived_random_bw_1t, derived_random_bw_sat = _derive_random_bw_from_latency(payload, total_cores, stream_bw_total, cacheline_bytes)
    derived_launch_intra, derived_launch_inter = _derive_launch_overheads(payload, total_cores, cpu_clock)
    gather_slice_copy_bw_points, gather_slice_copy_measured_slices = _parse_gather_slice_copy_bw_points(payload)

    return {
        "hw_config_path": source,
        "payload": payload,
        "total_cores": total_cores,
        "cpu_clock_ghz": cpu_clock,
        "simd_width_bits": simd_width_bits,
        "fetch_width": fetch_width,
        "decode_width": decode_width,
        "rename_width": rename_width,
        "dispatch_width": dispatch_width,
        "issue_width": issue_width,
        "commit_width": commit_width,
        "fp_throughput_vector_sp_add": fp_tp_add,
        "fp_throughput_vector_sp_mul": fp_tp_mul,
        "fp_throughput_vector_sp_fma": fp_tp_fma,
        "fp_latency_vector_sp_add": fp_lat_add,
        "fp_latency_vector_sp_mul": fp_lat_mul,
        "fp_latency_vector_sp_fma": fp_lat_fma,
        "cacheline_bytes": cacheline_bytes,
        "copy_bw_gb_s_1t": float(copy_bw_1t if copy_bw_1t is not None else derived_copy_bw_1t),
        "copy_bw_gb_s_saturated": float(copy_bw_sat if copy_bw_sat is not None else stream_bw_total),
        "random_gather_bw_gb_s_1t": float(random_bw_1t if random_bw_1t is not None else derived_random_bw_1t),
        "random_gather_bw_gb_s_saturated": float(random_bw_sat if random_bw_sat is not None else derived_random_bw_sat),
        "parallel_launch_overhead_us_intra": float(launch_intra if launch_intra is not None else derived_launch_intra),
        "parallel_launch_overhead_us_inter": float(launch_inter if launch_inter is not None else derived_launch_inter),
        "gather_slice_copy_bw_points": gather_slice_copy_bw_points,
        "gather_slice_copy_measured_slices": float(gather_slice_copy_measured_slices),
        "l1d_size_bytes": l1d_size,
        "l2_size_bytes": l2_size,
        "l3_size_bytes": l3_size,
        "l1d_latency_cycles": l1_latency,
        "l2_latency_cycles": l2_latency,
        "l3_latency_cycles": l3_latency,
        "l1d_latency_ns_static": l1_latency_ns,
        "l2_latency_ns_static": l2_latency_ns,
        "l3_latency_ns_static": l3_latency_ns,
        "local_mem_delay_ns": local_mem_delay_ns,
        "missing_whitebox_fields": missing_whitebox_fields,
    }


def _series_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce").fillna(default).astype(float)


def _series_text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    return frame.get(column, pd.Series(default, index=frame.index)).fillna(default).astype(str)


def _bytes_per_us_from_gb_s(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value.astype(float).clip(lower=1e-6) * 1000.0
    if isinstance(value, np.ndarray):
        return np.clip(value.astype(float), a_min=1e-6, a_max=None) * 1000.0
    return max(float(value) * 1000.0, 1e-6)


def _shape_tensors(value: Any) -> list[list[int]]:
    text = str(value).strip()
    if not text or text in {"nan", "None", "[]", "unknown"}:
        return []
    tensors: list[list[int]] = []
    for segment in _SHAPE_SEGMENT_RE.findall(text):
        dims: list[int] = []
        for token in segment.split(","):
            token = token.strip()
            if token and token.lstrip("-").isdigit():
                dims.append(int(token))
        if dims:
            tensors.append(dims)
    return tensors


def _shape_dims(value: Any) -> list[int]:
    tensors = _shape_tensors(value)
    return tensors[0] if tensors else []


def _shape_feature_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    input_ranks: list[float] = []
    output_ranks: list[float] = []
    shape_tokens: list[float] = []
    input_tensors: list[float] = []
    output_tensors: list[float] = []
    reshape_same_rank: list[float] = []
    input_primary_elements: list[float] = []
    output_primary_elements: list[float] = []
    input_dim_sum: list[float] = []
    output_dim_sum: list[float] = []
    input_dim_max: list[float] = []
    output_dim_max: list[float] = []
    for input_shape, output_shape in zip(
        frame.get("input_type_shape", pd.Series("", index=frame.index)).tolist(),
        frame.get("output_type_shape", pd.Series("", index=frame.index)).tolist(),
        strict=False,
    ):
        input_parsed = _shape_tensors(input_shape)
        output_parsed = _shape_tensors(output_shape)
        input_rank = len(input_parsed[0]) if input_parsed else 0
        output_rank = len(output_parsed[0]) if output_parsed else 0
        input_ranks.append(float(input_rank))
        output_ranks.append(float(output_rank))
        shape_tokens.append(float(sum(len(tensor) for tensor in input_parsed) + sum(len(tensor) for tensor in output_parsed)))
        input_tensors.append(float(len(input_parsed)))
        output_tensors.append(float(len(output_parsed)))
        reshape_same_rank.append(1.0 if input_rank == output_rank and input_rank > 0 else 0.0)
        input_primary = input_parsed[0] if input_parsed else []
        output_primary = output_parsed[0] if output_parsed else []
        input_primary_elements.append(float(np.prod(input_primary, dtype=float)) if input_primary else 0.0)
        output_primary_elements.append(float(np.prod(output_primary, dtype=float)) if output_primary else 0.0)
        input_dim_sum.append(float(sum(input_primary)) if input_primary else 0.0)
        output_dim_sum.append(float(sum(output_primary)) if output_primary else 0.0)
        input_dim_max.append(float(max(input_primary)) if input_primary else 0.0)
        output_dim_max.append(float(max(output_primary)) if output_primary else 0.0)
    return {
        "input_rank": np.asarray(input_ranks, dtype=float),
        "output_rank": np.asarray(output_ranks, dtype=float),
        "shape_token_count": np.asarray(shape_tokens, dtype=float),
        "input_tensor_count": np.asarray(input_tensors, dtype=float),
        "output_tensor_count": np.asarray(output_tensors, dtype=float),
        "reshape_same_rank_flag": np.asarray(reshape_same_rank, dtype=float),
        "input_primary_elements": np.asarray(input_primary_elements, dtype=float),
        "output_primary_elements": np.asarray(output_primary_elements, dtype=float),
        "input_dim_sum": np.asarray(input_dim_sum, dtype=float),
        "output_dim_sum": np.asarray(output_dim_sum, dtype=float),
        "input_dim_max": np.asarray(input_dim_max, dtype=float),
        "output_dim_max": np.asarray(output_dim_max, dtype=float),
    }


def _parse_perm_sequence(value: Any) -> list[int] | None:
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)):
        return None
    numbers: list[int] = []
    for item in parsed:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            return None
    return numbers if numbers else None


def _perm_feature_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ranks: list[float] = []
    fixed_point_ratio: list[float] = []
    displacement_norm: list[float] = []
    inversion_ratio: list[float] = []
    moved_axes: list[float] = []
    inversion_count: list[float] = []
    for perm_value in frame.get("attr_perm_json", pd.Series("", index=frame.index)).tolist():
        perm = _parse_perm_sequence(perm_value)
        if not perm:
            ranks.append(0.0)
            fixed_point_ratio.append(1.0)
            displacement_norm.append(0.0)
            inversion_ratio.append(0.0)
            moved_axes.append(0.0)
            inversion_count.append(0.0)
            continue
        length = len(perm)
        fixed = sum(1 for index, axis in enumerate(perm) if index == axis)
        displacement = float(sum(abs(index - axis) for index, axis in enumerate(perm)))
        inversions = 0
        for index, axis in enumerate(perm):
            inversions += sum(1 for later in perm[index + 1 :] if later < axis)
        total_pairs = max(length * (length - 1) // 2, 1)
        fixed_ratio = fixed / max(length, 1)
        disp_norm = displacement / max(length * length, 1)
        inv_ratio = inversions / total_pairs
        ranks.append(float(length))
        fixed_point_ratio.append(float(fixed_ratio))
        displacement_norm.append(float(disp_norm))
        inversion_ratio.append(float(inv_ratio))
        moved_axes.append(float(max(length - fixed, 0)))
        inversion_count.append(float(inversions))
    return {
        "perm_rank": np.asarray(ranks, dtype=float),
        "perm_fixed_point_ratio": np.asarray(fixed_point_ratio, dtype=float),
        "perm_displacement_norm": np.asarray(displacement_norm, dtype=float),
        "perm_inversion_ratio": np.asarray(inversion_ratio, dtype=float),
        "perm_moved_axes": np.asarray(moved_axes, dtype=float),
        "perm_inversion_count": np.asarray(inversion_count, dtype=float),
    }


def _typed_tensor_specs(value: Any) -> list[tuple[str, list[int]]]:
    text = str(value).strip()
    if not text or text in {"nan", "None", "[]", "unknown"}:
        return []
    specs: list[tuple[str, list[int]]] = []
    for dtype, dims_blob in _TYPED_TENSOR_RE.findall(text):
        dims: list[int] = []
        for token in dims_blob.split(","):
            token = token.strip()
            if token and token.lstrip("-").isdigit():
                dims.append(int(token))
        specs.append((dtype.strip(), dims))
    return specs


def _primary_tensor_element_bytes(frame: pd.DataFrame, column: str, default: float = 4.0) -> np.ndarray:
    values: list[float] = []
    for raw in frame.get(column, pd.Series("", index=frame.index)).tolist():
        specs = _typed_tensor_specs(raw)
        if not specs:
            values.append(float(default))
            continue
        values.append(float(DTYPE_BYTES.get(specs[0][0], default)))
    return np.asarray(values, dtype=float)


def _parse_numeric_tokens(value: Any) -> list[float]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text in {"unknown", "None", "nan"}:
        return []
    numbers: list[float] = []
    for token in text.split("-"):
        token = token.strip()
        if not token:
            continue
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    return numbers


def _arch_reducer_values(series: pd.Series | Any, *, reducer: str) -> np.ndarray:
    values = pd.Series(series, copy=False) if isinstance(series, pd.Series) else pd.Series(series)
    reduced_values: list[float] = []
    for value in values.tolist():
        numbers = _parse_numeric_tokens(value)
        if not numbers:
            reduced_values.append(0.0)
            continue
        reduced = float(np.mean(numbers) if reducer == "mean" else np.sum(numbers))
        reduced_values.append(max(reduced, 0.0))
    return np.asarray(reduced_values, dtype=float)


def _effective_worker_count(frame: pd.DataFrame, hw_params: dict[str, Any]) -> np.ndarray:
    num_threads = _series_numeric(frame, "num_threads", 1.0).to_numpy(dtype=float)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    return np.clip(num_threads, a_min=1.0, a_max=total_cores)


def _branch_parallel_limit(frame: pd.DataFrame) -> np.ndarray:
    configured = _series_numeric(frame, "cfg_parallel_branches", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    inter_threads = _series_numeric(frame, "inter_threads", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    return np.where(configured > 0.0, configured, inter_threads)


def _execution_modes(frame: pd.DataFrame) -> pd.Series:
    op_type = _series_text(frame, "op_type", "")
    reshape_like = _series_numeric(frame, "feat_reshape_view_or_copy_flag", 0.0).gt(0.5)
    tiny_shape_arith = _series_numeric(frame, "feat_tiny_shape_arith_flag", 0.0).gt(0.5)
    concat_shape = _series_numeric(frame, "feat_concat_shape_flag", 0.0).gt(0.5)
    primary_dtype_is_int64 = _series_numeric(frame, "feat_primary_dtype_is_int64", 0.0).gt(0.5)
    embedding_branch = _series_text(frame, "static_is_embedding_branch", "no").eq("yes")
    shape_gather = op_type.eq("Gather") & primary_dtype_is_int64 & ~embedding_branch

    modes = pd.Series(EXEC_MODE_OTHER, index=frame.index, dtype=object)
    modes.loc[op_type.isin(["Shape", "Unsqueeze", "Flatten"])] = EXEC_MODE_METADATA
    modes.loc[op_type.eq("Reshape") & reshape_like] = EXEC_MODE_METADATA
    modes.loc[op_type.isin(["Add", "Mul"]) & tiny_shape_arith] = EXEC_MODE_TINY_SHAPE_ARITH
    modes.loc[op_type.eq("Gather")] = EXEC_MODE_INDEXED_BLOCK_COPY
    modes.loc[shape_gather] = EXEC_MODE_METADATA
    modes.loc[op_type.eq("Concat") & concat_shape] = EXEC_MODE_SHAPE_CONCAT
    modes.loc[op_type.eq("Concat") & ~concat_shape] = EXEC_MODE_PAYLOAD_CONCAT
    modes.loc[op_type.eq("Transpose")] = EXEC_MODE_TRANSPOSE
    modes.loc[op_type.eq("ReduceSum")] = EXEC_MODE_REDUCTION
    modes.loc[op_type.isin(["Gemm", "MatMul"])] = EXEC_MODE_GEMM
    return modes.astype(str)


def _active_parallel_groups(frame: pd.DataFrame) -> np.ndarray:
    branch_limit = _branch_parallel_limit(frame)
    branch_count = _series_numeric(frame, "static_parallel_branch_count", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    is_embedding = _series_text(frame, "static_is_embedding_branch", "no").eq("yes").to_numpy(dtype=bool)
    groups = np.where(is_embedding, np.minimum(branch_limit, branch_count), 1.0)
    return np.clip(groups, a_min=1.0, a_max=None)


def _thread_pressure_ratio(frame: pd.DataFrame, hw_params: dict[str, Any]) -> np.ndarray:
    num_threads = _series_numeric(frame, "num_threads", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    active_groups = _active_parallel_groups(frame)
    total = max(float(hw_params["total_cores"]), 1.0)
    return np.clip((num_threads * active_groups) / total, a_min=0.0, a_max=None)


def _parallel_exposure_factor(frame: pd.DataFrame, hw_params: dict[str, Any]) -> np.ndarray:
    workers = _effective_worker_count(frame, hw_params)
    groups = _active_parallel_groups(frame)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    total_parallel_units = np.maximum(workers * groups, 1.0)
    topology = _capacity_fraction(total_parallel_units, total_cores)
    distance_to_merge = _series_numeric(frame, "static_distance_to_merge", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    branch_count = _series_numeric(frame, "static_parallel_branch_count", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    branch_limit = _branch_parallel_limit(frame)
    embedding_flag = _series_text(frame, "static_is_embedding_branch", "no").eq("yes").to_numpy(dtype=bool)
    merge_fanin = np.maximum(np.minimum(branch_limit, branch_count) - 1.0, 1.0)
    merge_decay = np.clip(distance_to_merge / np.maximum(distance_to_merge + merge_fanin, 1.0), a_min=0.0, a_max=1.0)
    branch_boost = np.clip((branch_count - 1.0) / np.maximum(branch_count, 1.0), a_min=0.0, a_max=1.0)
    exposure = np.where(embedding_flag, np.maximum(topology, merge_decay * branch_boost), topology)
    exposure = np.where((workers <= 1.0 + 1e-9) & (groups <= 1.0 + 1e-9), 0.0, exposure)
    return np.clip(exposure, a_min=0.0, a_max=1.0)


def _capacity_fraction(active_units: np.ndarray, total_units: float | np.ndarray) -> np.ndarray:
    active = np.asarray(active_units, dtype=float)
    total = np.asarray(total_units, dtype=float)
    total = np.clip(total, a_min=1.0, a_max=None)
    denominator = np.maximum(total - 1.0, 1.0)
    fraction = (active - 1.0) / denominator
    fraction = np.where(total <= 1.0 + 1e-9, 0.0, fraction)
    return np.clip(fraction, a_min=0.0, a_max=1.0)


def _linear_capacity_curve(low: float, high: float, fraction: np.ndarray) -> np.ndarray:
    low_f = float(min(low, high))
    high_f = float(max(low, high))
    return np.clip(low_f + np.clip(fraction, 0.0, 1.0) * (high_f - low_f), a_min=low_f, a_max=high_f)


def _worker_capacity_curve_gb_s(worker_units: np.ndarray, low: float, high: float, hw_params: dict[str, Any]) -> np.ndarray:
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    units = np.clip(np.asarray(worker_units, dtype=float), a_min=1.0, a_max=total_cores)
    return _linear_capacity_curve(float(low), float(high), _capacity_fraction(units, total_cores))


def _worker_parallel_units(frame: pd.DataFrame, hw_params: dict[str, Any]) -> np.ndarray:
    workers = _effective_worker_count(frame, hw_params)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    return np.clip(workers, a_min=1.0, a_max=total_cores)


def _branch_contention_share_triplet(frame: pd.DataFrame, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workers = _effective_worker_count(frame, hw_params)
    groups = _active_parallel_groups(frame)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    total_parallel_units = np.clip(workers * groups, a_min=1.0, a_max=total_cores)
    contention_fraction = _capacity_fraction(total_parallel_units, total_cores)
    optimistic = np.ones(len(frame), dtype=float)
    center = 1.0 + contention_fraction
    conservative = 1.0 + np.maximum(groups - 1.0, 0.0) * contention_fraction
    conservative = np.maximum(conservative, center)
    return optimistic, center, conservative


def _family_from_exec_mode(exec_mode: pd.Series, op_type: pd.Series) -> pd.Series:
    mapped = exec_mode.astype(str).map(EXEC_MODE_TO_FAMILY)
    fallback = op_type.astype(str).map(family_bucket_for_op_type).fillna("other").astype(str)
    return mapped.fillna(fallback).astype(str)


def _rf_group_from_exec_mode(exec_mode: pd.Series) -> pd.Series:
    return exec_mode.astype(str).map(EXEC_MODE_TO_RF_GROUP).fillna("").astype(str)


def _copy_bw_triplet_gb_s(frame: pd.DataFrame, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    worker_units = _worker_parallel_units(frame, hw_params)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    low = float(hw_params["copy_bw_gb_s_1t"])
    high = float(hw_params["copy_bw_gb_s_saturated"])
    base_bw = _linear_capacity_curve(low, high, _capacity_fraction(worker_units, total_cores))
    share_opt, share_center, share_cons = _branch_contention_share_triplet(frame, hw_params)
    optimistic = base_bw / np.clip(share_opt, a_min=1.0, a_max=None)
    center = base_bw / np.clip(share_center, a_min=1.0, a_max=None)
    conservative = base_bw / np.clip(share_cons, a_min=1.0, a_max=None)
    return optimistic, center, conservative


def _gather_slice_copy_bw_gb_s(frame: pd.DataFrame, hw_params: dict[str, Any]) -> np.ndarray:
    _, fallback_center, _ = _copy_bw_triplet_gb_s(frame, hw_params)
    points = hw_params.get("gather_slice_copy_bw_points", [])
    if not isinstance(points, list) or not points:
        return fallback_center

    point_frame = pd.DataFrame(points)
    required = {"block_size_bytes", "num_threads", "write_gb_s"}
    if point_frame.empty or not required.issubset(set(point_frame.columns)):
        return fallback_center
    point_frame = point_frame.dropna(subset=list(required)).copy()
    if point_frame.empty:
        return fallback_center

    block_size = _series_numeric(frame, "gather_ort_block_size_bytes", 0.0).to_numpy(dtype=float)
    thread_units = _series_numeric(frame, "num_threads", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    task_count = _series_numeric(frame, "gather_ort_task_count", 0.0).to_numpy(dtype=float)
    blocks = np.asarray(sorted(point_frame["block_size_bytes"].astype(float).unique()), dtype=float)
    if blocks.size == 0:
        return fallback_center

    active_mask = (block_size > 0.0) & (task_count > 0.0)
    nearest_idx = np.abs(block_size[:, None] - blocks[None, :]).argmin(axis=1)
    result = np.asarray(fallback_center, dtype=float).copy()
    for block_idx, block_value in enumerate(blocks):
        row_mask = (nearest_idx == block_idx) & active_mask
        if not np.any(row_mask):
            continue
        block_points = point_frame.loc[point_frame["block_size_bytes"].astype(float).eq(float(block_value))].sort_values(
            "num_threads"
        )
        thread_points = block_points["num_threads"].astype(float).to_numpy(dtype=float)
        bw_points = block_points["write_gb_s"].astype(float).to_numpy(dtype=float)
        if thread_points.size == 0:
            continue
        interpolated = np.interp(thread_units[row_mask], thread_points, bw_points, left=bw_points[0], right=bw_points[-1])
        result[row_mask] = np.clip(interpolated, a_min=1e-6, a_max=None)
    return result


def _gather_unique_ratio(frame: pd.DataFrame) -> np.ndarray:
    lookup_count = _series_numeric(frame, "feat_lookup_count", np.nan).replace(0.0, np.nan)
    unique_count = _series_numeric(frame, "gather_unique_idx_count", np.nan).fillna(lookup_count)
    ratio = (unique_count / lookup_count).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0, upper=1.0)
    return ratio.to_numpy(dtype=float)


def _gather_row_bytes_per_lookup(frame: pd.DataFrame) -> np.ndarray:
    lookup_count = _series_numeric(frame, "feat_lookup_count", np.nan).replace(0.0, np.nan)
    output_size = _series_numeric(frame, "output_size", 0.0)
    row_bytes = (output_size / lookup_count).replace([np.inf, -np.inf], np.nan)
    fallback = np.maximum(output_size.to_numpy(dtype=float), 8.0)
    return row_bytes.fillna(pd.Series(fallback, index=frame.index, dtype=float)).clip(lower=8.0).to_numpy(dtype=float)


def _cache_residency_level(frame: pd.DataFrame, working_set_bytes: np.ndarray, hw_params: dict[str, Any]) -> np.ndarray:
    active_groups = np.clip(_active_parallel_groups(frame), a_min=1.0, a_max=None)
    levels = np.full(len(frame), 4.0, dtype=float)
    working_set = np.clip(np.asarray(working_set_bytes, dtype=float), a_min=1.0, a_max=None)
    workers = _effective_worker_count(frame, hw_params)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    system_workers = np.clip(workers * active_groups, a_min=1.0, a_max=total_cores)
    l3_worker_share = np.clip(workers / np.maximum(system_workers, 1.0), a_min=0.0, a_max=1.0)
    l1_active = float(hw_params["l1d_size_bytes"]) * workers
    l2_active = float(hw_params["l2_size_bytes"]) * workers
    l3_active = float(hw_params["l3_size_bytes"]) * l3_worker_share
    if _continuous_cache_residency():
        p_l1 = np.minimum(l1_active / working_set, 1.0)
        p_l2 = np.minimum(l2_active / working_set, 1.0)
        p_l3 = np.minimum(l3_active / working_set, 1.0)
        p_l2_only = np.maximum(p_l2 - p_l1, 0.0)
        p_l3_only = np.maximum(p_l3 - p_l2, 0.0)
        p_mem = np.maximum(1.0 - p_l3, 0.0)
        return p_l1 + 2.0 * p_l2_only + 3.0 * p_l3_only + 4.0 * p_mem
    levels = np.where(working_set <= l1_active, 1.0, levels)
    levels = np.where((levels > 1.0) & (working_set <= l2_active), 2.0, levels)
    levels = np.where((levels > 2.0) & (working_set <= l3_active), 3.0, levels)
    return levels


def _random_bw_triplet_gb_s(frame: pd.DataFrame, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    worker_units = _worker_parallel_units(frame, hw_params)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    low = float(hw_params["random_gather_bw_gb_s_1t"])
    high = float(hw_params["random_gather_bw_gb_s_saturated"])
    base_bw = _linear_capacity_curve(low, high, _capacity_fraction(worker_units, total_cores))
    share_opt, share_center, share_cons = _branch_contention_share_triplet(frame, hw_params)
    optimistic = base_bw / np.clip(share_opt, a_min=1.0, a_max=None)
    center = base_bw / np.clip(share_center, a_min=1.0, a_max=None)
    conservative = base_bw / np.clip(share_cons, a_min=1.0, a_max=None)
    unique_ratio = _gather_unique_ratio(frame)
    row_bytes = _gather_row_bytes_per_lookup(frame)
    row_cacheline_spans = np.clip(row_bytes / max(float(hw_params["cacheline_bytes"]), 1.0), a_min=1.0, a_max=None)
    working_set_bytes = np.maximum(
        _series_numeric(frame, "gather_working_set_eff_bytes", 0.0).to_numpy(dtype=float),
        _series_numeric(frame, "feat_lookup_count", 0.0).to_numpy(dtype=float) * row_bytes * np.clip(unique_ratio, a_min=0.0, a_max=1.0),
    )
    working_set_bytes = np.where(working_set_bytes > 0.0, working_set_bytes, row_bytes)
    residency = _cache_residency_level(frame, working_set_bytes, hw_params)
    return optimistic, center, conservative, unique_ratio, row_cacheline_spans, residency


def _latency_per_touch_us_by_level(level: np.ndarray, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l1_us = float(hw_params["l1d_latency_ns_static"]) / 1000.0
    l2_us = float(hw_params["l2_latency_ns_static"]) / 1000.0
    l3_us = float(hw_params["l3_latency_ns_static"]) / 1000.0
    mem_us = float(hw_params["local_mem_delay_ns"]) / 1000.0
    if _continuous_cache_residency():
        level_arr = np.clip(np.asarray(level, dtype=float), a_min=1.0, a_max=4.0)
        xp = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
        fp = np.asarray([l1_us, l2_us, l3_us, mem_us], dtype=float)
        optimistic = np.interp(np.clip(level_arr - 0.5, a_min=1.0, a_max=4.0), xp, fp)
        center = np.interp(level_arr, xp, fp)
        conservative = np.interp(np.clip(level_arr + 0.5, a_min=1.0, a_max=4.0), xp, fp)
        return optimistic.astype(float), center.astype(float), conservative.astype(float)
    optimistic = np.where(level <= 1.0, l1_us, np.where(level <= 2.0, l2_us, np.where(level <= 3.0, l3_us, mem_us)))
    conservative = np.where(level <= 1.0, l2_us, np.where(level <= 2.0, l3_us, mem_us))
    center = 0.5 * (optimistic + conservative)
    return optimistic.astype(float), center.astype(float), conservative.astype(float)


def _op_throughput_per_cycle(frame: pd.DataFrame, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    op_type = _series_text(frame, "op_type", "").to_numpy(dtype=str)
    add_tp = float(hw_params["fp_throughput_vector_sp_add"])
    mul_tp = float(hw_params["fp_throughput_vector_sp_mul"])
    fma_tp = float(hw_params["fp_throughput_vector_sp_fma"])
    optimistic = np.full(len(frame), max(add_tp, mul_tp, fma_tp), dtype=float)
    conservative = np.full(len(frame), min(add_tp, mul_tp, fma_tp), dtype=float)
    center = np.full(len(frame), (add_tp + mul_tp + fma_tp) / 3.0, dtype=float)
    add_like = np.isin(op_type, ["Add", "ReduceSum"])
    mul_like = op_type == "Mul"
    fma_like = np.isin(op_type, ["Gemm", "MatMul"])
    optimistic = np.where(add_like, add_tp, optimistic)
    center = np.where(add_like, add_tp, center)
    conservative = np.where(add_like, add_tp, conservative)
    optimistic = np.where(mul_like, mul_tp, optimistic)
    center = np.where(mul_like, mul_tp, center)
    conservative = np.where(mul_like, mul_tp, conservative)
    optimistic = np.where(fma_like, fma_tp, optimistic)
    center = np.where(fma_like, fma_tp, center)
    conservative = np.where(fma_like, fma_tp, conservative)
    return optimistic, center, conservative


def _op_latency_cycles(frame: pd.DataFrame, hw_params: dict[str, Any]) -> np.ndarray:
    op_type = _series_text(frame, "op_type", "").to_numpy(dtype=str)
    add_lat = float(hw_params["fp_latency_vector_sp_add"])
    mul_lat = float(hw_params["fp_latency_vector_sp_mul"])
    fma_lat = float(hw_params["fp_latency_vector_sp_fma"])
    center = np.full(len(frame), (add_lat + mul_lat + fma_lat) / 3.0, dtype=float)
    center = np.where(np.isin(op_type, ["Add", "ReduceSum"]), add_lat, center)
    center = np.where(op_type == "Mul", mul_lat, center)
    center = np.where(np.isin(op_type, ["Gemm", "MatMul"]), fma_lat, center)
    return np.clip(center, a_min=0.0, a_max=None)


def _compute_latency_tail_triplet_us(
    frame: pd.DataFrame,
    hw_params: dict[str, Any],
    item_count: np.ndarray,
    depth_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = _compute_latency_tail_alpha()
    zeros = np.zeros(len(frame), dtype=float)
    if alpha <= 0.0:
        return zeros, zeros, zeros
    simd_lane_items = max(float(hw_params["simd_width_bits"]) / 32.0, 1.0)
    vector_ops = np.maximum(np.asarray(item_count, dtype=float) / simd_lane_items, 1.0)
    latency_depth = np.sqrt(vector_ops)
    if depth_hint is not None:
        latency_depth = np.maximum(latency_depth, np.asarray(depth_hint, dtype=float))
    worker_units = np.sqrt(np.clip(_worker_parallel_units(frame, hw_params), a_min=1.0, a_max=None))
    latency_cycles = _op_latency_cycles(frame, hw_params)
    center = alpha * latency_cycles * latency_depth / np.clip(float(hw_params["cpu_clock_ghz"]) * 1000.0 * worker_units, a_min=1e-6, a_max=None)
    center = np.clip(center, a_min=0.0, a_max=None)
    return 0.5 * center, center, 2.0 * center


def _add_compute_latency_tail(
    frame: pd.DataFrame,
    hw_params: dict[str, Any],
    compute_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    item_count: np.ndarray,
    depth_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tail_triplet = _compute_latency_tail_triplet_us(frame, hw_params, item_count, depth_hint)
    return tuple(a + b for a, b in zip(compute_triplet, tail_triplet, strict=False))


def _fp_capacity_triplet_items_per_us(
    frame: pd.DataFrame,
    hw_params: dict[str, Any],
    *,
    use_simd_lane_items: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workers = _effective_worker_count(frame, hw_params)
    if use_simd_lane_items:
        # Hardware calibration reports vector instruction throughput. The
        # compute prior converts it to scalar FP32 lane throughput so
        # compute-bound Gemm/MatMul is not capped at one lane per vector op.
        simd_scale = max(float(hw_params["simd_width_bits"]) / 32.0, 1.0)
    else:
        simd_scale = max(float(hw_params["simd_width_bits"]) / 128.0, 0.5)
    tp_optimistic, tp_center, tp_conservative = _op_throughput_per_cycle(frame, hw_params)
    cpu_clock = float(hw_params["cpu_clock_ghz"])
    optimistic = np.clip(tp_optimistic * simd_scale * workers * cpu_clock * 1000.0, a_min=1e-6, a_max=None)
    center = np.clip(tp_center * simd_scale * workers * cpu_clock * 1000.0, a_min=1e-6, a_max=None)
    conservative = np.clip(tp_conservative * simd_scale * workers * cpu_clock * 1000.0, a_min=1e-6, a_max=None)
    return optimistic, center, conservative


def _frontend_capacity_triplet_tokens_per_us(hw_params: dict[str, Any]) -> tuple[float, float, float]:
    widths = [
        float(hw_params["fetch_width"]),
        float(hw_params["decode_width"]),
        float(hw_params["rename_width"]),
        float(hw_params["dispatch_width"]),
    ]
    optimistic = max(widths) * float(hw_params["cpu_clock_ghz"]) * 1000.0
    conservative = min(widths) * float(hw_params["cpu_clock_ghz"]) * 1000.0
    center = float(np.mean(widths)) * float(hw_params["cpu_clock_ghz"]) * 1000.0
    return max(optimistic, 1e-6), max(center, 1e-6), max(conservative, 1e-6)


def _reorder_capacity_triplet_tokens_per_us(hw_params: dict[str, Any]) -> tuple[float, float, float]:
    widths = [
        float(hw_params["issue_width"]),
        float(hw_params["dispatch_width"]),
        float(hw_params["commit_width"]),
    ]
    optimistic = max(widths) * float(hw_params["cpu_clock_ghz"]) * 1000.0
    conservative = min(widths) * float(hw_params["cpu_clock_ghz"]) * 1000.0
    center = float(np.mean(widths)) * float(hw_params["cpu_clock_ghz"]) * 1000.0
    return max(optimistic, 1e-6), max(center, 1e-6), max(conservative, 1e-6)


def _latency_parallelism_triplet(frame: pd.DataFrame, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workers = _effective_worker_count(frame, hw_params)
    widths = np.asarray(
        [
            float(hw_params["issue_width"]),
            float(hw_params["dispatch_width"]),
            float(hw_params["commit_width"]),
        ],
        dtype=float,
    )
    optimistic = np.clip(workers * max(float(widths.max()), 1.0), a_min=1.0, a_max=None)
    center = np.clip(workers * max(float(widths.mean()), 1.0), a_min=1.0, a_max=None)
    conservative = np.clip(workers * max(float(widths.min()), 1.0), a_min=1.0, a_max=None)
    return optimistic, center, conservative


def _launch_cost_triplet_us(frame: pd.DataFrame, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workers = _effective_worker_count(frame, hw_params)
    intra = float(hw_params["parallel_launch_overhead_us_intra"])
    inter = float(hw_params["parallel_launch_overhead_us_inter"])
    dispatch_width = max(float(hw_params["dispatch_width"]), 1.0)
    active_groups = _active_parallel_groups(frame)
    worker_span = np.clip(workers, a_min=1.0, a_max=None)
    group_span = np.clip(active_groups, a_min=1.0, a_max=None)
    group_excess = np.maximum(group_span - 1.0, 0.0)
    per_event_lower = intra / np.clip(dispatch_width * worker_span * group_span, a_min=1.0, a_max=None)
    per_event_center = intra / np.clip(dispatch_width * worker_span, a_min=1.0, a_max=None)
    per_event_center = per_event_center + group_excess * inter / np.clip(dispatch_width * worker_span * group_span, a_min=1.0, a_max=None)
    per_event_upper = intra / np.clip(worker_span, a_min=1.0, a_max=None)
    per_event_upper = per_event_upper + group_excess * inter / np.clip(worker_span * group_span, a_min=1.0, a_max=None)
    per_event_upper = np.maximum(per_event_upper, per_event_center)
    return np.clip(per_event_lower, 0.0, None), np.clip(per_event_center, 0.0, None), np.clip(per_event_upper, 0.0, None)


def _inline_control_cost_triplet_us(frame: pd.DataFrame, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intra = float(hw_params["parallel_launch_overhead_us_intra"])
    dispatch_width = max(float(hw_params["dispatch_width"]), 1.0)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    share_opt, share_center, share_cons = _branch_contention_share_triplet(frame, hw_params)
    base_cost = intra / max(dispatch_width * total_cores, 1.0)
    lower = np.full(len(frame), base_cost, dtype=float) * np.clip(share_opt, a_min=1.0, a_max=None)
    center = np.full(len(frame), base_cost, dtype=float) * np.clip(share_center, a_min=1.0, a_max=None)
    upper = np.full(len(frame), base_cost, dtype=float) * np.clip(share_cons, a_min=1.0, a_max=None)
    upper = np.maximum(upper, center)
    return np.clip(lower, 0.0, None), np.clip(center, 0.0, None), np.clip(upper, 0.0, None)


def _reduction_tree_depth(frame: pd.DataFrame) -> np.ndarray:
    axes = _series_numeric(frame, "feat_reduction_axes_product", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    return np.ceil(np.log2(np.maximum(axes, 1.0)))


def _module_interval_column(module_name: str, bound: str) -> str:
    return f"svc_{module_name}_{bound}_us"


def _module_band_column(module_name: str) -> str:
    return f"svc_{module_name}_band_width_us"


def _share_column(module_name: str, bound: str) -> str:
    return f"share_{module_name}_{bound}"


def _service_from_capacity(demand: np.ndarray, optimistic_capacity: np.ndarray, center_capacity: np.ndarray, conservative_capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demand_arr = np.clip(np.asarray(demand, dtype=float), a_min=0.0, a_max=None)
    lower = demand_arr / np.clip(np.asarray(optimistic_capacity, dtype=float), a_min=1e-6, a_max=None)
    center = demand_arr / np.clip(np.asarray(center_capacity, dtype=float), a_min=1e-6, a_max=None)
    upper = demand_arr / np.clip(np.asarray(conservative_capacity, dtype=float), a_min=1e-6, a_max=None)
    return np.clip(lower, 0.0, None), np.clip(center, 0.0, None), np.clip(upper, 0.0, None)


def _service_from_cost(events: np.ndarray, lower_cost: np.ndarray, center_cost: np.ndarray, upper_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events_arr = np.clip(np.asarray(events, dtype=float), a_min=0.0, a_max=None)
    lower = events_arr * np.asarray(lower_cost, dtype=float)
    center = events_arr * np.asarray(center_cost, dtype=float)
    upper = events_arr * np.asarray(upper_cost, dtype=float)
    return np.clip(lower, 0.0, None), np.clip(center, 0.0, None), np.clip(upper, 0.0, None)


def _metadata_touch_triplet_us(events: np.ndarray, hw_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    touch_events = np.clip(np.asarray(events, dtype=float), a_min=0.0, a_max=None)
    lower_cost = np.full(len(touch_events), float(hw_params["l1d_latency_ns_static"]) / 1000.0, dtype=float)
    center_cost = np.full(len(touch_events), float(hw_params["l2_latency_ns_static"]) / 1000.0, dtype=float)
    upper_cost = np.full(len(touch_events), float(hw_params["l3_latency_ns_static"]) / 1000.0, dtype=float)
    return _service_from_cost(touch_events, lower_cost, center_cost, upper_cost)


def _set_module_interval(work: pd.DataFrame, mask: pd.Series, module_name: str, lower: np.ndarray, center: np.ndarray, upper: np.ndarray) -> None:
    work.loc[mask, _module_interval_column(module_name, "lower")] = np.asarray(lower, dtype=float)
    work.loc[mask, _module_interval_column(module_name, "center")] = np.asarray(center, dtype=float)
    work.loc[mask, _module_interval_column(module_name, "upper")] = np.asarray(upper, dtype=float)


def _zero_disabled_analytical_modules(work: pd.DataFrame) -> None:
    for module_name in DISABLED_ANALYTICAL_MODULES:
        for bound in ("lower", "center", "upper"):
            work[_module_interval_column(module_name, bound)] = 0.0
        work[f"svc_{module_name}_us"] = 0.0


def _shape_complexity_tokens(work: pd.DataFrame) -> np.ndarray:
    return (
        work["shape_token_count"].to_numpy(dtype=float)
        + work["input_tensor_count"].to_numpy(dtype=float)
        + work["output_tensor_count"].to_numpy(dtype=float)
        + work["input_rank"].to_numpy(dtype=float)
        + work["output_rank"].to_numpy(dtype=float)
        + work["perm_rank"].to_numpy(dtype=float)
    )


def _element_count_from_bytes(byte_series: pd.Series | np.ndarray, element_bytes: np.ndarray, default_count: float = 1.0) -> np.ndarray:
    values = byte_series.to_numpy(dtype=float) if isinstance(byte_series, pd.Series) else np.asarray(byte_series, dtype=float)
    count = values / np.clip(np.asarray(element_bytes, dtype=float), a_min=1.0, a_max=None)
    return np.clip(count, a_min=default_count, a_max=None)


def _service_share_columns(work: pd.DataFrame) -> None:
    for bound in ("lower", "center", "upper"):
        service_matrix = np.stack(
            [pd.to_numeric(work[_module_interval_column(name, bound)], errors="coerce").fillna(0.0).to_numpy(dtype=float) for name in MODULE_NAMES],
            axis=1,
        )
        sum_svc = np.clip(service_matrix.sum(axis=1), a_min=1e-6, a_max=None)
        max_svc = np.max(service_matrix, axis=1)
        for index, module_name in enumerate(MODULE_NAMES):
            work[_share_column(module_name, bound)] = service_matrix[:, index] / sum_svc
        if bound == "center":
            order = np.argsort(service_matrix, axis=1)
            top2 = service_matrix[np.arange(len(service_matrix)), order[:, -2]] if service_matrix.shape[1] > 1 else np.zeros(len(service_matrix), dtype=float)
            work["sum_svc_us"] = sum_svc
            work["sum_svc_center_us"] = sum_svc
            work["max_svc_us"] = max_svc
            work["max_svc_center_us"] = max_svc
            work["max_to_sum_ratio_center"] = max_svc / sum_svc
            work["max_to_sum_ratio"] = work["max_to_sum_ratio_center"]
            work["bottleneck_id_center"] = np.argmax(service_matrix, axis=1).astype(float)
            work["bottleneck_id"] = work["bottleneck_id_center"]
            work["bottleneck_margin_center"] = (max_svc - top2) / np.clip(max_svc, a_min=1e-6, a_max=None)
            work["bottleneck_margin"] = work["bottleneck_margin_center"]
            useful = np.maximum(
                sum_svc - pd.to_numeric(work[_module_interval_column("dispatch", "center")], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                0.0,
            )
            dispatch_center = pd.to_numeric(work[_module_interval_column("dispatch", "center")], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            random_center = pd.to_numeric(work[_module_interval_column("random_mem", "center")], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            stream_center = pd.to_numeric(work[_module_interval_column("stream_mem", "center")], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            compute_center = pd.to_numeric(work[_module_interval_column("compute", "center")], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            sync_center = pd.to_numeric(work[_module_interval_column("sync", "center")], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            reorder_center = pd.to_numeric(work[_module_interval_column("reorder", "center")], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            work["launch_vs_useful_ratio_center"] = dispatch_center / np.clip(useful, a_min=1e-6, a_max=None)
            work["launch_vs_useful_ratio"] = work["launch_vs_useful_ratio_center"]
            work["random_vs_stream_ratio_center"] = random_center / np.clip(stream_center, a_min=1e-6, a_max=None)
            work["random_vs_stream_ratio"] = work["random_vs_stream_ratio_center"]
            work["sync_vs_compute_ratio_center"] = sync_center / np.clip(compute_center, a_min=1e-6, a_max=None)
            work["sync_vs_compute_ratio"] = work["sync_vs_compute_ratio_center"]
            mem_total = stream_center + random_center
            work["mem_vs_compute_ratio_center"] = mem_total / np.clip(compute_center, a_min=1e-6, a_max=None)
            work["mem_vs_compute_ratio"] = work["mem_vs_compute_ratio_center"]
            work["reorder_vs_stream_ratio_center"] = reorder_center / np.clip(stream_center, a_min=1e-6, a_max=None)
            work["reorder_vs_stream_ratio"] = work["reorder_vs_stream_ratio_center"]
            work["launch_window_utilization_center"] = useful / sum_svc
            work["launch_window_utilization"] = work["launch_window_utilization_center"]
            for module_name in MODULE_NAMES:
                alias_column = MODULE_TO_SHARE[f"svc_{module_name}_us"]
                work[alias_column] = work[_share_column(module_name, "center")]
        else:
            work[f"sum_svc_{bound}_us"] = sum_svc
            work[f"max_svc_{bound}_us"] = max_svc
    for module_name in MODULE_NAMES:
        work[_module_band_column(module_name)] = np.maximum(
            pd.to_numeric(work[_module_interval_column(module_name, "upper")], errors="coerce").fillna(0.0)
            - pd.to_numeric(work[_module_interval_column(module_name, "lower")], errors="coerce").fillna(0.0),
            0.0,
        )


def _compose_total_intervals(work: pd.DataFrame) -> None:
    op_type = _series_text(work, "op_type", "")
    family = _series_text(work, "whitebox_prior_family", "")
    exec_mode = _series_text(work, "whitebox_exec_mode", EXEC_MODE_OTHER)
    exec_region = _series_text(work, "exec_region_static", "unknown")
    materialize_required = _series_numeric(work, "materialize_required_flag", 0.0).gt(0.5).to_numpy(dtype=bool)
    merge_sync_obligation = _series_numeric(work, "merge_sync_obligation_flag", 0.0).to_numpy(dtype=float)
    phase_barriers = _series_numeric(work, "phase_barrier_count_static", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    merge_fanin = _series_numeric(work, "merge_fanin_static", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    reduce_inter_branch_merge_bytes = _series_numeric(work, "reduce_inter_branch_merge_bytes", 0.0).clip(lower=0.0).to_numpy(dtype=float)

    composition_mode = pd.Series("other_execmax_dispatch", index=work.index, dtype=object)

    for bound in ("lower", "center", "upper"):
        descriptor = pd.to_numeric(work[_module_interval_column("descriptor", bound)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        stream = pd.to_numeric(work[_module_interval_column("stream_mem", bound)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        random = pd.to_numeric(work[_module_interval_column("random_mem", bound)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        compute = pd.to_numeric(work[_module_interval_column("compute", bound)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        reorder = pd.to_numeric(work[_module_interval_column("reorder", bound)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sync = pd.to_numeric(work[_module_interval_column("sync", bound)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dispatch = pd.to_numeric(work[_module_interval_column("dispatch", bound)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        launch_floor = pd.to_numeric(work[f"dispatch_floor_us_{bound}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        copy_bw = pd.to_numeric(work[f"eff_copy_bw_gb_s_{bound}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # Mainline recovery: keep the stronger single-layer analytical
        # composition as the formal prior.
        exec_term = np.maximum.reduce([stream, compute, descriptor])
        serial_term = dispatch.copy()

        # Shadow path: keep the three-layer state/phase composition only as
        # diagnostics. It must not change the formal whitebox_* outputs until
        # phase resource accounting is validated.
        kernel_term = np.maximum.reduce([stream, compute, descriptor])
        state_term = dispatch.copy()
        phase_term = np.zeros(len(work), dtype=float)
        phase_stream_term = np.zeros(len(work), dtype=float)
        phase_barrier_term = np.zeros(len(work), dtype=float)

        meta_mask = exec_mode.eq(EXEC_MODE_METADATA).to_numpy(dtype=bool)
        tiny_mask = exec_mode.eq(EXEC_MODE_TINY_SHAPE_ARITH).to_numpy(dtype=bool)
        gather_mask = exec_mode.eq(EXEC_MODE_INDEXED_BLOCK_COPY).to_numpy(dtype=bool)
        reduce_mask = exec_mode.eq(EXEC_MODE_REDUCTION).to_numpy(dtype=bool)
        gemm_mask = exec_mode.eq(EXEC_MODE_GEMM).to_numpy(dtype=bool)
        concat_shape_mask = exec_mode.eq(EXEC_MODE_SHAPE_CONCAT).to_numpy(dtype=bool)
        concat_payload_mask = exec_mode.eq(EXEC_MODE_PAYLOAD_CONCAT).to_numpy(dtype=bool)
        concat_mask = concat_shape_mask | concat_payload_mask
        transpose_mask = exec_mode.eq(EXEC_MODE_TRANSPOSE).to_numpy(dtype=bool)
        ewise_mask = (family.eq("ewise_like") & ~exec_mode.isin([EXEC_MODE_TINY_SHAPE_ARITH])).to_numpy(dtype=bool)
        other_mask = ~(meta_mask | tiny_mask | gather_mask | reduce_mask | ewise_mask | gemm_mask | concat_mask | transpose_mask)

        meta_materialize_mask = meta_mask & materialize_required
        meta_view_mask = meta_mask & ~materialize_required

        exec_term[meta_view_mask] = descriptor[meta_view_mask]
        serial_term[meta_view_mask] = dispatch[meta_view_mask]
        kernel_term[meta_view_mask] = descriptor[meta_view_mask]
        state_term[meta_view_mask] = dispatch[meta_view_mask]
        composition_mode.loc[meta_view_mask] = "meta_descriptor_dispatch"

        exec_term[meta_materialize_mask] = descriptor[meta_materialize_mask] + stream[meta_materialize_mask]
        serial_term[meta_materialize_mask] = dispatch[meta_materialize_mask]
        kernel_term[meta_materialize_mask] = descriptor[meta_materialize_mask] + stream[meta_materialize_mask]
        state_term[meta_materialize_mask] = dispatch[meta_materialize_mask]
        composition_mode.loc[meta_materialize_mask] = "meta_materialize_stream_descriptor_dispatch"

        exec_term[tiny_mask] = np.maximum(descriptor[tiny_mask], compute[tiny_mask])
        serial_term[tiny_mask] = dispatch[tiny_mask]
        kernel_term[tiny_mask] = np.maximum(descriptor[tiny_mask], compute[tiny_mask])
        state_term[tiny_mask] = dispatch[tiny_mask]
        composition_mode.loc[tiny_mask] = "tiny_shape_descriptor_compute_dispatch"

        exec_term[gather_mask] = np.maximum(random[gather_mask], stream[gather_mask])
        serial_term[gather_mask] = sync[gather_mask] + dispatch[gather_mask]
        kernel_term[gather_mask] = np.maximum(random[gather_mask], stream[gather_mask])
        state_term[gather_mask] = sync[gather_mask] + dispatch[gather_mask]
        composition_mode.loc[gather_mask] = "gather_overlap_sync_dispatch"

        exec_term[reduce_mask] = np.maximum(stream[reduce_mask], compute[reduce_mask])
        serial_term[reduce_mask] = reorder[reduce_mask] + sync[reduce_mask] + dispatch[reduce_mask]
        kernel_term[reduce_mask] = np.maximum(stream[reduce_mask], compute[reduce_mask]) + reorder[reduce_mask]
        state_term[reduce_mask] = sync[reduce_mask] + dispatch[reduce_mask]
        composition_mode.loc[reduce_mask] = "reduce_overlap_reorder_sync_dispatch"

        exec_term[concat_payload_mask] = stream[concat_payload_mask] + descriptor[concat_payload_mask]
        serial_term[concat_payload_mask] = sync[concat_payload_mask] + dispatch[concat_payload_mask]
        kernel_term[concat_payload_mask] = stream[concat_payload_mask] + descriptor[concat_payload_mask]
        state_term[concat_payload_mask] = sync[concat_payload_mask] + dispatch[concat_payload_mask]
        composition_mode.loc[concat_payload_mask] = "concat_stream_descriptor_sync_dispatch"

        exec_term[concat_shape_mask] = descriptor[concat_shape_mask]
        serial_term[concat_shape_mask] = dispatch[concat_shape_mask]
        kernel_term[concat_shape_mask] = descriptor[concat_shape_mask]
        state_term[concat_shape_mask] = dispatch[concat_shape_mask]
        composition_mode.loc[concat_shape_mask] = "shape_concat_descriptor_dispatch"

        exec_term[transpose_mask] = np.maximum(stream[transpose_mask], reorder[transpose_mask]) + descriptor[transpose_mask]
        serial_term[transpose_mask] = dispatch[transpose_mask]
        kernel_term[transpose_mask] = np.maximum(stream[transpose_mask], reorder[transpose_mask]) + descriptor[transpose_mask]
        state_term[transpose_mask] = dispatch[transpose_mask]
        composition_mode.loc[transpose_mask] = "transpose_overlap_descriptor_dispatch"

        ewise_only_mask = ewise_mask & ~transpose_mask & ~concat_mask
        exec_term[ewise_only_mask] = np.maximum(stream[ewise_only_mask], compute[ewise_only_mask])
        serial_term[ewise_only_mask] = dispatch[ewise_only_mask]
        kernel_term[ewise_only_mask] = np.maximum(stream[ewise_only_mask], compute[ewise_only_mask])
        state_term[ewise_only_mask] = dispatch[ewise_only_mask]
        composition_mode.loc[ewise_only_mask] = "ewise_overlap_dispatch"

        gemm_only_mask = gemm_mask & ~transpose_mask & ~concat_mask
        exec_term[gemm_only_mask] = np.maximum(stream[gemm_only_mask], compute[gemm_only_mask])
        serial_term[gemm_only_mask] = reorder[gemm_only_mask] + dispatch[gemm_only_mask]
        kernel_term[gemm_only_mask] = np.maximum(stream[gemm_only_mask], compute[gemm_only_mask]) + reorder[gemm_only_mask]
        state_term[gemm_only_mask] = dispatch[gemm_only_mask]
        composition_mode.loc[gemm_only_mask] = "gemm_overlap_reorder_dispatch"

        exec_term[other_mask] = np.maximum.reduce([stream[other_mask], compute[other_mask], descriptor[other_mask]])
        serial_term[other_mask] = dispatch[other_mask]
        kernel_term[other_mask] = np.maximum.reduce([stream[other_mask], compute[other_mask], descriptor[other_mask]])
        state_term[other_mask] = dispatch[other_mask]

        merge_phase_mask = exec_region.eq("interaction_merge").to_numpy(dtype=bool)
        metadata_phase_mask = exec_region.eq("metadata_control").to_numpy(dtype=bool)
        embedding_phase_mask = exec_region.eq("embedding_private").to_numpy(dtype=bool)
        phase_events = phase_barriers * merge_sync_obligation
        merge_handoff_events = np.maximum(merge_fanin - 1.0, 0.0) * merge_sync_obligation

        control_merge_mask = (meta_mask | tiny_mask | concat_mask) & (merge_phase_mask | metadata_phase_mask)
        phase_barrier_term[control_merge_mask] = (
            phase_events[control_merge_mask] + merge_handoff_events[control_merge_mask]
        ) * launch_floor[control_merge_mask]

        reduction_merge_mask = reduce_mask & (embedding_phase_mask | merge_phase_mask)
        phase_barrier_term[reduction_merge_mask] = phase_events[reduction_merge_mask] * launch_floor[reduction_merge_mask]
        if np.any(reduction_merge_mask):
            phase_stream_triplet = _service_from_capacity(
                reduce_inter_branch_merge_bytes[reduction_merge_mask],
                _bytes_per_us_from_gb_s(copy_bw[reduction_merge_mask]),
                _bytes_per_us_from_gb_s(copy_bw[reduction_merge_mask]),
                _bytes_per_us_from_gb_s(copy_bw[reduction_merge_mask]),
            )
            phase_stream_term[reduction_merge_mask] = phase_stream_triplet[1]

        gather_merge_mask = gather_mask & (embedding_phase_mask | merge_phase_mask)
        phase_barrier_term[gather_merge_mask] = phase_events[gather_merge_mask] * launch_floor[gather_merge_mask]

        phase_term = phase_stream_term + phase_barrier_term

        state_adjusted = kernel_term + state_term
        phase_adjusted = state_adjusted + phase_term
        total_term = exec_term + serial_term

        work[f"whitebox_kernel_only_{bound}_us"] = np.clip(kernel_term, a_min=0.0, a_max=None)
        work[f"whitebox_state_adjusted_{bound}_us"] = np.clip(state_adjusted, a_min=0.0, a_max=None)
        work[f"whitebox_phase_adjusted_{bound}_us"] = np.clip(phase_adjusted, a_min=0.0, a_max=None)
        work[f"whitebox_phase_serial_overhead_{bound}_us"] = np.clip(phase_term, a_min=0.0, a_max=None)
        work[f"whitebox_phase_stream_overhead_{bound}_us"] = np.clip(phase_stream_term, a_min=0.0, a_max=None)
        work[f"whitebox_phase_barrier_overhead_{bound}_us"] = np.clip(phase_barrier_term, a_min=0.0, a_max=None)
        work[f"whitebox_exec_{bound}_us"] = np.clip(exec_term, a_min=0.0, a_max=None)
        work[f"whitebox_serial_overhead_{bound}_us"] = np.clip(serial_term, a_min=0.0, a_max=None)
        work[f"whitebox_{bound}_us"] = np.clip(total_term, a_min=0.0, a_max=None)

    work["whitebox_composition_mode"] = composition_mode.astype(str)
    work["whitebox_band_width_us"] = np.maximum(
        pd.to_numeric(work["whitebox_upper_us"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["whitebox_lower_us"], errors="coerce").fillna(0.0),
        0.0,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float(np.nanmedian(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else float("nan")
    ordered = np.argsort(values[valid])
    sorted_values = values[valid][ordered]
    sorted_weights = weights[valid][ordered]
    cutoff = 0.5 * float(np.sum(sorted_weights))
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")])


def _normalize_op_thread_scale_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not config:
        return None
    coefficients: dict[str, dict[str, float]] = {}
    for op_name, per_thread in dict(config.get("coefficients", {})).items():
        op_key = str(op_name)
        coefficients[op_key] = {}
        for thread_key, value in dict(per_thread or {}).items():
            try:
                scale = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(scale) and scale > 0.0:
                coefficients[op_key][str(int(float(thread_key)))] = scale
    return {
        "schema_version": int(config.get("schema_version", OP_THREAD_SCALE_SCHEMA_VERSION)),
        "scale_kind": str(config.get("scale_kind", OP_THREAD_SCALE_KIND)),
        "default_scale": float(config.get("default_scale", OP_THREAD_SCALE_DEFAULT)),
        "min_group_rows": int(config.get("min_group_rows", OP_THREAD_SCALE_MIN_ROWS_DEFAULT)),
        "target_column": str(config.get("target_column", base_run.TARGET_US_COLUMN)),
        "baseline_column": str(config.get("baseline_column", "whitebox_center_us")),
        "fit_source": dict(config.get("fit_source", {})),
        "coefficients": coefficients,
    }


def load_op_thread_scale_config(path: Path | str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized = _normalize_op_thread_scale_config(config)
    if normalized is None:
        raise ValueError(f"Empty op-thread scale config: {path}")
    if normalized["scale_kind"] != OP_THREAD_SCALE_KIND:
        raise ValueError(f"Unsupported op-thread scale kind: {normalized['scale_kind']}")
    return normalized


def save_op_thread_scale_config(config: dict[str, Any], path: Path | str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def op_thread_scale_coefficients_frame(config: dict[str, Any] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    normalized = _normalize_op_thread_scale_config(config)
    if not normalized:
        return pd.DataFrame()
    for op_name, per_thread in sorted(normalized["coefficients"].items()):
        for thread_key, scale in sorted(per_thread.items(), key=lambda item: int(item[0])):
            rows.append({"op_type": op_name, "num_threads": int(thread_key), "scale": float(scale)})
    return pd.DataFrame(rows)


def _op_thread_scale_for_frame(work: pd.DataFrame, config: dict[str, Any] | None) -> pd.Series:
    normalized = _normalize_op_thread_scale_config(config)
    default_scale = float(normalized.get("default_scale", OP_THREAD_SCALE_DEFAULT)) if normalized else OP_THREAD_SCALE_DEFAULT
    scale = pd.Series(default_scale, index=work.index, dtype=float)
    if not normalized:
        return scale
    coefficients = dict(normalized.get("coefficients", {}))
    op_series = _series_text(work, "op_type", "")
    thread_series = _series_numeric(work, "num_threads", 1.0).round().clip(lower=1.0).astype(int).astype(str)
    for op_name, per_thread in coefficients.items():
        op_mask = op_series.eq(str(op_name))
        if not op_mask.any():
            continue
        mapped = thread_series.loc[op_mask].map({str(key): float(value) for key, value in dict(per_thread).items()})
        scale.loc[op_mask] = mapped.fillna(default_scale).astype(float)
    return scale.replace([np.inf, -np.inf], default_scale).fillna(default_scale).clip(lower=1e-9)


def _apply_op_thread_scale(work: pd.DataFrame, config: dict[str, Any] | None) -> None:
    scale = _op_thread_scale_for_frame(work, config)
    work["whitebox_op_thread_scale"] = scale
    if not config:
        work["whitebox_op_thread_scale_source"] = "off"
        return
    work["whitebox_op_thread_scale_source"] = OP_THREAD_SCALE_KIND
    for column in (
        "whitebox_lower_us",
        "whitebox_center_us",
        "whitebox_upper_us",
        "pred_total_lower_us",
        "pred_total_us",
        "pred_total_upper_us",
        "whitebox_prior_us",
        "pred_mem_us",
        "pred_compute_us",
        "pred_sync_us",
        "pred_launch_us",
    ):
        if column not in work.columns:
            continue
        unscaled_column = f"{column.removesuffix('_us')}_unscaled_us" if column.endswith("_us") else f"{column}_unscaled"
        if unscaled_column not in work.columns:
            work[unscaled_column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0) * scale
    work["whitebox_band_width_us"] = np.maximum(
        pd.to_numeric(work["whitebox_upper_us"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["whitebox_lower_us"], errors="coerce").fillna(0.0),
        0.0,
    )


def fit_op_thread_scale_config(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    prior_variant: str = "legacy_v2",
    min_group_rows: int = OP_THREAD_SCALE_MIN_ROWS_DEFAULT,
) -> dict[str, Any]:
    unscaled_bundle = fit_whitebox_family_prior_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        prior_variant=prior_variant,
        op_thread_scale_config=None,
    )
    prepared = add_whitebox_family_prior_columns(fit_frame, unscaled_bundle)
    target_column = base_run.TARGET_US_COLUMN
    actual = pd.to_numeric(prepared[target_column], errors="coerce")
    baseline = pd.to_numeric(prepared["whitebox_center_us"], errors="coerce")
    ratio = actual / baseline.clip(lower=1e-9)
    fit_table = pd.DataFrame(
        {
            "op_type": _series_text(prepared, "op_type", ""),
            "num_threads": _series_numeric(prepared, "num_threads", 1.0).round().clip(lower=1.0).astype(int),
            "ratio": ratio,
            "weight": actual.clip(lower=1e-9),
        }
    )
    fit_table = fit_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio", "weight"])
    fit_table = fit_table.loc[(fit_table["ratio"] > 0.0) & (fit_table["weight"] > 0.0)].copy()
    coefficients: dict[str, dict[str, float]] = {}
    group_rows: list[dict[str, Any]] = []
    for (op_name, thread_count), group in fit_table.groupby(["op_type", "num_threads"], sort=True):
        rows = int(len(group))
        if rows < int(min_group_rows):
            continue
        scale = _weighted_median(group["ratio"].to_numpy(dtype=float), group["weight"].to_numpy(dtype=float))
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        coefficients.setdefault(str(op_name), {})[str(int(thread_count))] = float(scale)
        group_rows.append(
            {
                "op_type": str(op_name),
                "num_threads": int(thread_count),
                "rows": rows,
                "scale": float(scale),
            }
        )
    return {
        "schema_version": OP_THREAD_SCALE_SCHEMA_VERSION,
        "scale_kind": OP_THREAD_SCALE_KIND,
        "default_scale": OP_THREAD_SCALE_DEFAULT,
        "min_group_rows": int(min_group_rows),
        "target_column": target_column,
        "baseline_column": "whitebox_center_us",
        "fit_source": {
            "policy": str(policy or ""),
            "prior_variant": str(prior_variant),
            "rows": int(len(fit_frame)),
            "groups": int(len(group_rows)),
            "hw_config_path": str(Path(hw_config).resolve()),
        },
        "coefficients": coefficients,
    }


def _prepare_base_columns(
    frame: pd.DataFrame,
    hw_params: dict[str, Any],
    *,
    prior_variant: str = "legacy_v2",
) -> pd.DataFrame:
    work = augment_static_feature_frame(frame.copy())
    op_type = _series_text(work, "op_type", "")
    work["whitebox_exec_mode"] = _execution_modes(work)
    work["whitebox_prior_family"] = _family_from_exec_mode(work["whitebox_exec_mode"], op_type)
    work["whitebox_rf_group"] = _rf_group_from_exec_mode(work["whitebox_exec_mode"])
    work["whitebox_prior_source"] = "physically_constrained_interval_prior"
    work["whitebox_residual_source"] = "prior_only"

    shape_features = _shape_feature_arrays(work)
    perm_features = _perm_feature_arrays(work)
    for column, values in {**shape_features, **perm_features}.items():
        work[column] = values
    work["rank_delta"] = np.abs(work["output_rank"].to_numpy(dtype=float) - work["input_rank"].to_numpy(dtype=float))
    work["output_element_bytes"] = _primary_tensor_element_bytes(work, "output_type_shape", default=4.0)
    work["input_element_bytes"] = _primary_tensor_element_bytes(work, "input_type_shape", default=4.0)
    work["exact_io_bytes"] = _series_numeric(work, "exact_read_bytes", 0.0) + _series_numeric(work, "exact_write_bytes", 0.0)

    num_threads = _series_numeric(work, "num_threads", 1.0).clip(lower=1.0)
    inter_threads = _series_numeric(work, "inter_threads", 1.0).clip(lower=1.0)
    total_threads = (num_threads * inter_threads).clip(lower=1.0)
    work["total_threads"] = total_threads
    work["threads_per_core"] = total_threads.to_numpy(dtype=float) / max(float(hw_params["total_cores"]), 1.0)
    work["log_num_threads"] = np.log1p(num_threads.to_numpy(dtype=float))
    work["log_inter_threads"] = np.log1p(inter_threads.to_numpy(dtype=float))
    work["log_total_threads"] = np.log1p(total_threads.to_numpy(dtype=float))
    work["is_single_intra"] = num_threads.le(1.0).astype(float)
    work["is_single_inter"] = inter_threads.le(1.0).astype(float)
    work["static_is_embedding_branch_flag"] = _series_text(work, "static_is_embedding_branch", "no").eq("yes").astype(float)
    work["attr_axis_value"] = _series_numeric(work, "attr_axis", 0.0)
    for feature_name, target_op in {
        "is_concat_op": "Concat",
        "is_transpose_op": "Transpose",
        "is_add_op": "Add",
        "is_mul_op": "Mul",
        "is_sigmoid_op": "Sigmoid",
        "is_reshape_op": "Reshape",
        "is_shape_op": "Shape",
        "is_flatten_op": "Flatten",
        "is_unsqueeze_op": "Unsqueeze",
        "is_gather_op": "Gather",
        "is_reduce_sum_op": "ReduceSum",
        "is_gemm_op": "Gemm",
        "is_matmul_op": "MatMul",
    }.items():
        work[feature_name] = op_type.eq(target_op).astype(float)

    effective_workers = _effective_worker_count(work, hw_params)
    work["effective_worker_count"] = effective_workers
    work["branch_parallel_limit"] = _branch_parallel_limit(work)
    work["thread_pressure_ratio"] = _thread_pressure_ratio(work, hw_params)
    work["active_parallel_groups"] = _active_parallel_groups(work)
    work["parallel_exposure_factor"] = _parallel_exposure_factor(work, hw_params)
    total_cores = max(float(hw_params["total_cores"]), 1.0)
    worker_excess = np.maximum(effective_workers - 1.0, 0.0)
    branch_groups = np.clip(work["active_parallel_groups"].to_numpy(dtype=float), a_min=1.0, a_max=None)
    branch_task_recovery = np.maximum(branch_groups - 1.0, 0.0) / total_cores
    work["dispatch_event_count"] = np.clip((worker_excess / total_cores) + branch_task_recovery, a_min=0.0, a_max=None)
    work["sync_event_scale"] = np.clip(branch_task_recovery, a_min=0.0, a_max=None)
    gather_unique_ratio = _gather_unique_ratio(work)
    work["gather_unique_ratio"] = gather_unique_ratio
    work["arch_embedding_size_mean"] = _arch_reducer_values(_series_text(work, "arch_embedding_size", "unknown"), reducer="mean")
    work["arch_mlp_bot_mean"] = _arch_reducer_values(_series_text(work, "arch_mlp_bot", "unknown"), reducer="mean")
    work["arch_mlp_bot_sum"] = _arch_reducer_values(_series_text(work, "arch_mlp_bot", "unknown"), reducer="sum")
    output_size = _series_numeric(work, "output_size", 0.0).to_numpy(dtype=float)
    gather_index_count = _series_numeric(work, "feat_gather_index_count_static", 0.0).to_numpy(dtype=float)
    gather_tail_block_bytes = _series_numeric(work, "feat_gather_tail_block_bytes", 0.0).to_numpy(dtype=float)
    cacheline_bytes = max(float(hw_params["cacheline_bytes"]), 1.0)
    per_lookup_block = np.where(
        gather_tail_block_bytes > 0.0,
        gather_tail_block_bytes,
        np.where(gather_index_count > 0.0, output_size / np.clip(gather_index_count, a_min=1.0, a_max=None), 0.0),
    )
    submodel_role_static = _series_text(work, "submodel_role_static", "unknown").to_numpy(dtype=str)
    embedding_role = np.isin(submodel_role_static, ["embedding_branch"])
    gather_unique_rows_total = np.where(
        gather_index_count > 0.0,
        np.maximum(gather_index_count * np.clip(gather_unique_ratio, a_min=0.0, a_max=1.0), 1.0),
        0.0,
    )
    gather_unique_rows_per_worker = np.where(
        gather_unique_rows_total > 0.0,
        gather_unique_rows_total / np.clip(effective_workers, a_min=1.0, a_max=None),
        0.0,
    )
    work["gather_unique_rows_total"] = gather_unique_rows_total
    work["gather_unique_rows_per_worker"] = gather_unique_rows_per_worker
    work["gather_effective_head_touch_count"] = np.where(
        embedding_role,
        gather_unique_rows_per_worker,
        gather_unique_rows_total,
    )
    # Gather head latency is paid by distinct row-start touches; the row
    # payload itself is accounted for by the contiguous tail copy path.
    work["gather_head_cacheline_touches"] = work["gather_effective_head_touch_count"].to_numpy(dtype=float)
    work["gather_tail_copy_bytes"] = np.maximum(_series_numeric(work, "exact_write_bytes", 0.0).to_numpy(dtype=float), 0.0)
    gather_task_count = np.where(
        per_lookup_block > 0.0,
        output_size / np.clip(per_lookup_block, a_min=1.0, a_max=None),
        gather_index_count,
    )
    gather_m_count = np.where(
        gather_index_count > 0.0,
        gather_task_count / np.clip(gather_index_count, a_min=1.0, a_max=None),
        0.0,
    )
    work["gather_ort_task_count"] = np.clip(gather_task_count, a_min=0.0, a_max=None)
    work["gather_ort_m_count"] = np.clip(gather_m_count, a_min=0.0, a_max=None)
    work["gather_ort_block_size_bytes"] = np.clip(per_lookup_block, a_min=0.0, a_max=None)
    work["gather_ort_source_payload_bytes"] = np.maximum(_series_numeric(work, "exact_read_bytes", 0.0).to_numpy(dtype=float), 0.0)
    work["gather_ort_dest_payload_bytes"] = work["gather_tail_copy_bytes"].to_numpy(dtype=float)
    work["gather_ort_index_scan_count"] = np.clip(gather_index_count, a_min=0.0, a_max=None)
    work["gather_ort_grain_bytes"] = np.clip(per_lookup_block, a_min=1.0, a_max=None)

    copy_bw_lower, copy_bw_center, copy_bw_upper = _copy_bw_triplet_gb_s(work, hw_params)
    random_bw_lower, random_bw_center, random_bw_upper, unique_ratio, row_cacheline_spans, residency = _random_bw_triplet_gb_s(work, hw_params)
    fp_lower, fp_center, fp_upper = _fp_capacity_triplet_items_per_us(
        work,
        hw_params,
        use_simd_lane_items=True,
    )
    launch_lower, launch_center, launch_upper = _launch_cost_triplet_us(work, hw_params)
    descriptor_tp_lower, descriptor_tp_center, descriptor_tp_upper = _frontend_capacity_triplet_tokens_per_us(hw_params)
    reorder_tp_lower, reorder_tp_center, reorder_tp_upper = _reorder_capacity_triplet_tokens_per_us(hw_params)

    work["eff_copy_bw_gb_s_lower"] = copy_bw_lower
    work["eff_copy_bw_gb_s_center"] = copy_bw_center
    work["eff_copy_bw_gb_s_upper"] = copy_bw_upper
    work["eff_copy_bw_gb_s"] = copy_bw_center
    work["effective_copy_bw_gb_s"] = copy_bw_center
    gather_slice_bw = _gather_slice_copy_bw_gb_s(work, hw_params)
    gather_slice_stream_us = work["gather_tail_copy_bytes"].to_numpy(dtype=float) / np.clip(
        _bytes_per_us_from_gb_s(gather_slice_bw),
        a_min=1e-6,
        a_max=None,
    )
    work["gather_slice_copy_bw_gb_s"] = gather_slice_bw
    work["gather_slice_stream_us"] = np.clip(gather_slice_stream_us, a_min=0.0, a_max=None)
    work["gather_slice_vs_current_stream_ratio"] = 1.0
    work["gather_slice_underpredict_gap_us"] = 0.0

    work["eff_random_bw_gb_s_lower"] = random_bw_lower
    work["eff_random_bw_gb_s_center"] = random_bw_center
    work["eff_random_bw_gb_s_upper"] = random_bw_upper
    work["eff_random_bw_gb_s"] = random_bw_center
    work["effective_random_gather_bw_gb_s"] = random_bw_center
    work["gather_unique_ratio"] = unique_ratio
    work["gather_row_cacheline_spans"] = row_cacheline_spans
    work["cache_residency_level"] = residency

    work["eff_fp_items_per_us_lower"] = fp_lower
    work["eff_fp_items_per_us_center"] = fp_center
    work["eff_fp_items_per_us_upper"] = fp_upper
    work["eff_fp_items_per_us"] = fp_center

    work["dispatch_floor_us_lower"] = launch_lower
    work["dispatch_floor_us_center"] = launch_center
    work["dispatch_floor_us_upper"] = launch_upper
    work["dispatch_floor_us"] = launch_center

    work["descriptor_tokens_per_us_lower"] = descriptor_tp_lower
    work["descriptor_tokens_per_us_center"] = descriptor_tp_center
    work["descriptor_tokens_per_us_upper"] = descriptor_tp_upper
    work["reorder_tokens_per_us_lower"] = reorder_tp_lower
    work["reorder_tokens_per_us_center"] = reorder_tp_center
    work["reorder_tokens_per_us_upper"] = reorder_tp_upper
    work["reduction_tree_depth"] = _reduction_tree_depth(work)
    work["reduction_intra_tree_depth"] = np.ceil(np.log2(np.maximum(effective_workers, 1.0)))
    active_groups = np.clip(work["active_parallel_groups"].to_numpy(dtype=float), a_min=1.0, a_max=None)
    merge_sync_obligation = _series_numeric(work, "merge_sync_obligation_flag", 0.0).to_numpy(dtype=float)
    exact_output_bytes = np.maximum(
        _series_numeric(work, "exact_output_tensor_bytes_sum", 0.0).to_numpy(dtype=float),
        _series_numeric(work, "exact_write_bytes", 0.0).to_numpy(dtype=float),
    )
    intra_partial_write = np.maximum(effective_workers - 1.0, 0.0) / np.maximum(effective_workers, 1.0)
    inter_merge_write = (np.maximum(active_groups - 1.0, 0.0) / active_groups) * merge_sync_obligation
    work["reduce_partial_read_bytes"] = np.maximum(_series_numeric(work, "exact_read_bytes", 0.0).to_numpy(dtype=float), 0.0)
    work["reduce_partial_write_bytes"] = exact_output_bytes * intra_partial_write
    work["reduce_inter_branch_merge_bytes"] = exact_output_bytes * inter_merge_write
    work["reduction_inter_tree_depth"] = np.ceil(np.log2(np.maximum(active_groups, 1.0))) * merge_sync_obligation
    work["reduce_sync_rounds"] = work["reduction_intra_tree_depth"].to_numpy(dtype=float) + work["reduction_inter_tree_depth"].to_numpy(dtype=float)
    work["reduce_write_amplification"] = 1.0 + intra_partial_write + inter_merge_write
    reduction_layout = _series_text(work, "feat_reduction_layout_class", "unknown").to_numpy(dtype=str)
    reduce_fast_kind_code = np.zeros(len(work), dtype=float)
    reduce_fast_kind_code = np.where(reduction_layout == "middle_axis_collapse", 16.0, reduce_fast_kind_code)
    reduce_fast_kind_code = np.where(reduction_layout == "keepdims_like", 4.0, reduce_fast_kind_code)
    reduce_fast_kind_code = np.where(reduction_layout == "rank_collapse", 8.0, reduce_fast_kind_code)
    output_element_bytes = np.clip(_series_numeric(work, "output_element_bytes", 4.0).to_numpy(dtype=float), a_min=1.0, a_max=None)
    reduce_output_elements = np.maximum(exact_output_bytes / output_element_bytes, 1.0)
    batch_size = np.maximum(_series_numeric(work, "batch_size", 0.0).to_numpy(dtype=float), 0.0)
    reduce_parallel_extent = np.where(
        reduction_layout == "middle_axis_collapse",
        np.maximum(batch_size, 1.0),
        reduce_output_elements,
    )
    work["reduce_ort_fast_kind_code"] = reduce_fast_kind_code
    work["reduce_ort_input_read_bytes"] = np.maximum(_series_numeric(work, "exact_read_bytes", 0.0).to_numpy(dtype=float), 0.0)
    work["reduce_ort_output_write_bytes"] = exact_output_bytes
    work["reduce_ort_stream_bytes"] = work["reduce_ort_input_read_bytes"].to_numpy(dtype=float) + work["reduce_ort_output_write_bytes"].to_numpy(dtype=float)
    work["reduce_ort_grain_tasks"] = np.clip(reduce_parallel_extent, a_min=1.0, a_max=None)
    work["reduce_ort_parallel_axis_extent"] = np.clip(reduce_parallel_extent, a_min=1.0, a_max=None)

    zero_columns = {
        _module_interval_column(module_name, bound): np.zeros(len(work), dtype=float)
        for module_name in MODULE_NAMES
        for bound in ("lower", "center", "upper")
    }
    zero_columns.update({f"svc_{module_name}_us": np.zeros(len(work), dtype=float) for module_name in MODULE_NAMES})
    work = pd.concat([work, pd.DataFrame(zero_columns, index=work.index)], axis=1)
    return work


def add_whitebox_family_prior_columns(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    override_hw_config: Path | dict[str, Any] | None = None,
) -> pd.DataFrame:
    _assert_physical_constant_audit()
    if override_hw_config is not None:
        hw_params = _load_hw_params(override_hw_config, require_measured_whitebox=False)
        reference_hw = dict(bundle["hw_params"])
        if _freeze_override_cache_latency():
            for key in ("l1d_latency_ns_static", "l2_latency_ns_static", "l3_latency_ns_static"):
                hw_params[key] = float(reference_hw[key])
    else:
        hw_params = dict(bundle["hw_params"])
    prior_variant = str(bundle.get("prior_variant", "legacy_v2"))
    use_ort_kernel_v3 = prior_variant == "ort_kernel_v3"
    work = _prepare_base_columns(frame, hw_params, prior_variant=prior_variant)
    op_type = _series_text(work, "op_type", "")
    exec_mode = _series_text(work, "whitebox_exec_mode", EXEC_MODE_OTHER)

    exact_read_bytes = _series_numeric(work, "exact_read_bytes", 0.0)
    exact_write_bytes = _series_numeric(work, "exact_write_bytes", 0.0)
    io_bytes = exact_read_bytes + exact_write_bytes
    output_size = _series_numeric(work, "exact_output_tensor_bytes_sum", 0.0)
    output_size = output_size.where(output_size.gt(0.0), _series_numeric(work, "output_size", 0.0))
    output_elements = _element_count_from_bytes(output_size, work["output_element_bytes"].to_numpy(dtype=float))
    shape_tokens = _shape_complexity_tokens(work)
    active_groups = work["active_parallel_groups"].to_numpy(dtype=float)
    reduction_depth = work["reduction_tree_depth"].to_numpy(dtype=float)
    input_tensors = work["input_tensor_count"].to_numpy(dtype=float)
    output_rank = work["output_rank"].to_numpy(dtype=float)
    rank_delta = work["rank_delta"].to_numpy(dtype=float)
    perm_rank = work["perm_rank"].to_numpy(dtype=float)
    perm_moved_axes = work["perm_moved_axes"].to_numpy(dtype=float)
    perm_inversion_count = work["perm_inversion_count"].to_numpy(dtype=float)
    thread_pressure = work["thread_pressure_ratio"].to_numpy(dtype=float)
    parallel_exposure = work["parallel_exposure_factor"].to_numpy(dtype=float)
    sync_event_scale = work["sync_event_scale"].to_numpy(dtype=float)
    merge_sync_obligation = _series_numeric(work, "merge_sync_obligation_flag", 0.0).to_numpy(dtype=float)
    exec_region_static = _series_text(work, "exec_region_static", "unknown").to_numpy(dtype=str)
    merge_fanin_static = _series_numeric(work, "merge_fanin_static", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    phase_barrier_count_static = _series_numeric(work, "phase_barrier_count_static", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    worker_shard_count_static = _series_numeric(work, "worker_shard_count_static", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    materialize_required = _series_numeric(work, "materialize_required_flag", 0.0).to_numpy(dtype=float)
    reduce_write_amplification = _series_numeric(work, "reduce_write_amplification", 1.0).to_numpy(dtype=float)
    reduction_intra_tree_depth = _series_numeric(work, "reduction_intra_tree_depth", 0.0).to_numpy(dtype=float)
    reduction_inter_tree_depth = _series_numeric(work, "reduction_inter_tree_depth", 0.0).to_numpy(dtype=float)
    reduce_sync_rounds = _series_numeric(work, "reduce_sync_rounds", 0.0).to_numpy(dtype=float)
    reduce_partial_read_bytes = _series_numeric(work, "reduce_partial_read_bytes", 0.0).to_numpy(dtype=float)
    reduce_partial_write_bytes = _series_numeric(work, "reduce_partial_write_bytes", 0.0).to_numpy(dtype=float)
    reduce_inter_branch_merge_bytes = _series_numeric(work, "reduce_inter_branch_merge_bytes", 0.0).to_numpy(dtype=float)
    gather_head_cacheline_touches = _series_numeric(work, "gather_head_cacheline_touches", 0.0).to_numpy(dtype=float)
    gather_unique_rows_total = _series_numeric(work, "gather_unique_rows_total", 0.0).to_numpy(dtype=float)
    gather_unique_rows_per_worker = _series_numeric(work, "gather_unique_rows_per_worker", 0.0).to_numpy(dtype=float)
    gather_tail_copy_bytes = _series_numeric(work, "gather_tail_copy_bytes", 0.0).to_numpy(dtype=float)
    gather_ort_source_payload_bytes = _series_numeric(work, "gather_ort_source_payload_bytes", 0.0).to_numpy(dtype=float)
    gather_ort_dest_payload_bytes = _series_numeric(work, "gather_ort_dest_payload_bytes", 0.0).to_numpy(dtype=float)
    gather_ort_task_count = _series_numeric(work, "gather_ort_task_count", 0.0).to_numpy(dtype=float)
    gather_ort_grain_bytes = _series_numeric(work, "gather_ort_grain_bytes", 1.0).to_numpy(dtype=float)
    reduce_ort_stream_bytes = _series_numeric(work, "reduce_ort_stream_bytes", 0.0).to_numpy(dtype=float)
    reduce_ort_grain_tasks = _series_numeric(work, "reduce_ort_grain_tasks", 1.0).to_numpy(dtype=float)
    gather_effective_head_touch_count = _series_numeric(work, "gather_effective_head_touch_count", 0.0).to_numpy(dtype=float)
    effective_worker_count = _series_numeric(work, "effective_worker_count", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    num_threads = _series_numeric(work, "num_threads", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    inter_threads = _series_numeric(work, "inter_threads", 1.0).clip(lower=1.0).to_numpy(dtype=float)
    dispatch_events = work["dispatch_event_count"].to_numpy(dtype=float)
    meta_tokens = np.maximum(shape_tokens + rank_delta + work["attr_axis_value"].to_numpy(dtype=float).clip(min=0.0), 1.0)
    primary_dtype_is_int64 = _series_numeric(work, "feat_primary_dtype_is_int64", 0.0).to_numpy(dtype=float)
    tiny_shape_arith = _series_numeric(work, "feat_tiny_shape_arith_flag", 0.0).to_numpy(dtype=float)
    concat_shape_flag = _series_numeric(work, "feat_concat_shape_flag", 0.0).to_numpy(dtype=float)
    transpose_reshape_like_flag = _series_numeric(work, "feat_transpose_reshape_like_flag", 0.0).to_numpy(dtype=float)
    transpose_single_axis_flag = _series_numeric(work, "feat_transpose_single_axis_flag", 0.0).to_numpy(dtype=float)
    transpose_generic_flag = _series_numeric(work, "feat_transpose_generic_flag", 0.0).to_numpy(dtype=float)
    reduction_middle_axis_flag = _series_numeric(work, "feat_reduction_middle_axis_flag", 0.0).to_numpy(dtype=float)
    reduction_rank_collapse_flag = _series_numeric(work, "feat_reduction_rank_collapse_flag", 0.0).to_numpy(dtype=float)
    reduction_keepdims_flag = _series_numeric(work, "feat_reduction_keepdims_flag", 0.0).to_numpy(dtype=float)
    gather_index_count_static = np.maximum(_series_numeric(work, "feat_gather_index_count_static", 0.0).to_numpy(dtype=float), 0.0)
    gather_tail_block_bytes = np.maximum(_series_numeric(work, "feat_gather_tail_block_bytes", 0.0).to_numpy(dtype=float), 0.0)
    graph_branch_fanout = np.maximum(_series_numeric(work, "feat_graph_branch_fanout", 1.0).to_numpy(dtype=float), 1.0)
    submodel_role_static = _series_text(work, "submodel_role_static", "unknown").to_numpy(dtype=str)

    copy_lower = work["eff_copy_bw_gb_s_lower"].to_numpy(dtype=float)
    copy_center = work["eff_copy_bw_gb_s_center"].to_numpy(dtype=float)
    copy_upper = work["eff_copy_bw_gb_s_upper"].to_numpy(dtype=float)
    random_lower = work["eff_random_bw_gb_s_lower"].to_numpy(dtype=float)
    random_center = work["eff_random_bw_gb_s_center"].to_numpy(dtype=float)
    random_upper = work["eff_random_bw_gb_s_upper"].to_numpy(dtype=float)
    branch_share_lower, branch_share_center, branch_share_upper = _branch_contention_share_triplet(work, hw_params)
    isolated_copy_lower = copy_lower * np.clip(branch_share_lower, a_min=1.0, a_max=None)
    isolated_copy_center = copy_center * np.clip(branch_share_center, a_min=1.0, a_max=None)
    isolated_copy_upper = copy_upper * np.clip(branch_share_upper, a_min=1.0, a_max=None)
    isolated_random_lower = random_lower * np.clip(branch_share_lower, a_min=1.0, a_max=None)
    isolated_random_center = random_center * np.clip(branch_share_center, a_min=1.0, a_max=None)
    isolated_random_upper = random_upper * np.clip(branch_share_upper, a_min=1.0, a_max=None)
    kernel_worker_units = np.where(
        np.isin(exec_region_static, ["embedding_private"]),
        np.maximum(effective_worker_count, num_threads),
        effective_worker_count,
    )
    copy_curve_current = _worker_capacity_curve_gb_s(
        effective_worker_count,
        float(hw_params["copy_bw_gb_s_1t"]),
        float(hw_params["copy_bw_gb_s_saturated"]),
        hw_params,
    )
    copy_curve_kernel = _worker_capacity_curve_gb_s(
        kernel_worker_units,
        float(hw_params["copy_bw_gb_s_1t"]),
        float(hw_params["copy_bw_gb_s_saturated"]),
        hw_params,
    )
    copy_kernel_scale = copy_curve_kernel / np.clip(copy_curve_current, a_min=1e-6, a_max=None)
    kernel_copy_lower = isolated_copy_lower * copy_kernel_scale
    kernel_copy_center = isolated_copy_center * copy_kernel_scale
    kernel_copy_upper = isolated_copy_upper * copy_kernel_scale
    random_curve_current = _worker_capacity_curve_gb_s(
        effective_worker_count,
        float(hw_params["random_gather_bw_gb_s_1t"]),
        float(hw_params["random_gather_bw_gb_s_saturated"]),
        hw_params,
    )
    random_curve_kernel = _worker_capacity_curve_gb_s(
        kernel_worker_units,
        float(hw_params["random_gather_bw_gb_s_1t"]),
        float(hw_params["random_gather_bw_gb_s_saturated"]),
        hw_params,
    )
    random_kernel_scale = random_curve_kernel / np.clip(random_curve_current, a_min=1e-6, a_max=None)
    kernel_random_lower = isolated_random_lower * random_kernel_scale
    kernel_random_center = isolated_random_center * random_kernel_scale
    kernel_random_upper = isolated_random_upper * random_kernel_scale
    fp_lower = work["eff_fp_items_per_us_lower"].to_numpy(dtype=float)
    fp_center = work["eff_fp_items_per_us_center"].to_numpy(dtype=float)
    fp_upper = work["eff_fp_items_per_us_upper"].to_numpy(dtype=float)
    launch_lower = work["dispatch_floor_us_lower"].to_numpy(dtype=float)
    launch_center = work["dispatch_floor_us_center"].to_numpy(dtype=float)
    launch_upper = work["dispatch_floor_us_upper"].to_numpy(dtype=float)
    inline_launch_lower, inline_launch_center, inline_launch_upper = _inline_control_cost_triplet_us(work, hw_params)
    desc_tp_lower = work["descriptor_tokens_per_us_lower"].to_numpy(dtype=float)
    desc_tp_center = work["descriptor_tokens_per_us_center"].to_numpy(dtype=float)
    desc_tp_upper = work["descriptor_tokens_per_us_upper"].to_numpy(dtype=float)
    reorder_tp_lower = work["reorder_tokens_per_us_lower"].to_numpy(dtype=float)
    reorder_tp_center = work["reorder_tokens_per_us_center"].to_numpy(dtype=float)
    reorder_tp_upper = work["reorder_tokens_per_us_upper"].to_numpy(dtype=float)

    meta_mask = exec_mode.eq(EXEC_MODE_METADATA)
    if meta_mask.any():
        meta_rows = work.loc[meta_mask].copy()
        meta_desc = np.maximum(
            meta_rows["input_tensor_count"].to_numpy(dtype=float)
            + meta_rows["output_tensor_count"].to_numpy(dtype=float),
            1.0,
        )
        meta_op = op_type.loc[meta_mask].to_numpy(dtype=str)
        reshape_mask = meta_op == "Reshape"
        shape_mask = meta_op == "Shape"
        unsqueeze_mask = meta_op == "Unsqueeze"
        flatten_mask = meta_op == "Flatten"
        meta_roles = submodel_role_static[meta_mask.to_numpy()]
        tail_like = np.isin(meta_roles, ["interaction", "tail", "mlp_bot", "mlp_top"])
        meta_desc = np.where(
            reshape_mask,
            np.maximum(
                meta_rows["input_tensor_count"].to_numpy(dtype=float)
                + meta_rows["output_tensor_count"].to_numpy(dtype=float)
                + meta_rows["input_rank"].to_numpy(dtype=float)
                + meta_rows["output_rank"].to_numpy(dtype=float)
                + np.abs(meta_rows["rank_delta"].to_numpy(dtype=float)),
                1.0,
            ),
            meta_desc,
        )
        meta_desc = np.where(
            shape_mask,
            np.maximum(
                np.ones(int(meta_mask.sum()), dtype=float)
                + meta_rows["merge_sync_obligation_flag"].to_numpy(dtype=float),
                1.0,
            ),
            meta_desc,
        )
        meta_desc = np.where(shape_mask, np.maximum(meta_desc + tail_like.astype(float), 1.0), meta_desc)
        meta_desc = np.where(
            unsqueeze_mask,
            np.maximum(
                meta_rows["input_tensor_count"].to_numpy(dtype=float)
                + meta_rows["output_tensor_count"].to_numpy(dtype=float)
                + meta_rows["output_rank"].to_numpy(dtype=float)
                + np.ones(int(meta_mask.sum()), dtype=float)
                + np.abs(meta_rows["rank_delta"].to_numpy(dtype=float)),
                1.0,
            ),
            meta_desc,
        )
        meta_desc = np.where(
            flatten_mask,
            np.maximum(
                meta_rows["input_tensor_count"].to_numpy(dtype=float)
                + meta_rows["output_tensor_count"].to_numpy(dtype=float)
                + meta_rows["input_rank"].to_numpy(dtype=float)
                + meta_rows["output_rank"].to_numpy(dtype=float)
                + meta_rows["rank_delta"].to_numpy(dtype=float)
                + meta_rows["reshape_same_rank_flag"].to_numpy(dtype=float),
                1.0,
            ),
            meta_desc,
        )
        desc_lower, desc_center, desc_upper = _metadata_touch_triplet_us(meta_desc, hw_params)
        reshape_or_flatten_mask = reshape_mask | flatten_mask
        meta_stream_bytes = (
            np.maximum(meta_rows["exact_read_bytes"].to_numpy(dtype=float), 0.0)
            + np.maximum(meta_rows["exact_write_bytes"].to_numpy(dtype=float), 0.0)
        ) * meta_rows["materialize_required_flag"].to_numpy(dtype=float)
        meta_stream_lower = np.zeros(int(meta_mask.sum()), dtype=float)
        meta_stream_center = np.zeros(int(meta_mask.sum()), dtype=float)
        meta_stream_upper = np.zeros(int(meta_mask.sum()), dtype=float)
        if reshape_or_flatten_mask.any():
            stream_triplet = _service_from_capacity(
                meta_stream_bytes[reshape_or_flatten_mask],
                _bytes_per_us_from_gb_s(copy_lower[meta_mask.to_numpy()][reshape_or_flatten_mask]),
                _bytes_per_us_from_gb_s(copy_center[meta_mask.to_numpy()][reshape_or_flatten_mask]),
                _bytes_per_us_from_gb_s(copy_upper[meta_mask.to_numpy()][reshape_or_flatten_mask]),
            )
            meta_stream_lower[reshape_or_flatten_mask] = stream_triplet[0]
            meta_stream_center[reshape_or_flatten_mask] = stream_triplet[1]
            meta_stream_upper[reshape_or_flatten_mask] = stream_triplet[2]
        meta_dispatch_events = np.ones(int(meta_mask.sum()), dtype=float)
        meta_dispatch_events = meta_dispatch_events + meta_rows["merge_sync_obligation_flag"].to_numpy(dtype=float)
        launch_triplet = _service_from_cost(
            meta_dispatch_events,
            np.zeros(int(meta_mask.sum()), dtype=float),
            inline_launch_center[meta_mask.to_numpy()],
            launch_upper[meta_mask.to_numpy()],
        )
        if shape_mask.any():
            shape_inline = inline_launch_lower[meta_mask.to_numpy()][shape_mask]
            launch_lower_meta = np.asarray(launch_triplet[0], dtype=float)
            launch_center_meta = np.asarray(launch_triplet[1], dtype=float)
            launch_upper_meta = np.asarray(launch_triplet[2], dtype=float)
            launch_lower_meta[shape_mask] = shape_inline
            launch_center_meta[shape_mask] = shape_inline
            launch_upper_meta[shape_mask] = shape_inline
            launch_triplet = (launch_lower_meta, launch_center_meta, launch_upper_meta)
        _set_module_interval(work, meta_mask, "stream_mem", meta_stream_lower, meta_stream_center, meta_stream_upper)
        _set_module_interval(work, meta_mask, "descriptor", desc_lower, desc_center, desc_upper)
        _set_module_interval(work, meta_mask, "dispatch", *launch_triplet)
        work.loc[meta_mask, "whitebox_prior_source"] = "meta_descriptor_dispatch_interval"

    gather_mask = exec_mode.eq(EXEC_MODE_INDEXED_BLOCK_COPY)
    if gather_mask.any():
        gather_mask_arr = gather_mask.to_numpy(dtype=bool)
        latency_parallel_lower, latency_parallel_center, latency_parallel_upper = _latency_parallelism_triplet(work, hw_params)
        gather_roles = submodel_role_static[gather_mask_arr]
        embedding_gather = np.isin(gather_roles, ["embedding_branch"])
        latency_parallel_lower = np.where(
            embedding_gather,
            latency_parallel_lower[gather_mask_arr] / np.clip(effective_worker_count[gather_mask_arr], a_min=1.0, a_max=None),
            latency_parallel_lower[gather_mask_arr],
        )
        latency_parallel_center = np.where(
            embedding_gather,
            latency_parallel_center[gather_mask_arr] / np.clip(effective_worker_count[gather_mask_arr], a_min=1.0, a_max=None),
            latency_parallel_center[gather_mask_arr],
        )
        latency_parallel_upper = np.where(
            embedding_gather,
            latency_parallel_upper[gather_mask_arr] / np.clip(effective_worker_count[gather_mask_arr], a_min=1.0, a_max=None),
            latency_parallel_upper[gather_mask_arr],
        )
        latency_lower, latency_center, latency_upper = _latency_per_touch_us_by_level(
            work["cache_residency_level"].to_numpy(dtype=float),
            hw_params,
        )
        cacheline_bytes = max(float(hw_params["cacheline_bytes"]), 1.0)
        random_head_bytes = gather_head_cacheline_touches * cacheline_bytes
        gather_random_bw_lower = random_lower
        gather_random_bw_center = random_center
        gather_random_bw_upper = random_upper
        gather_copy_bw_lower = copy_lower
        gather_copy_bw_center = copy_center
        gather_copy_bw_upper = copy_upper
        if use_ort_kernel_v3:
            random_head_bytes = np.maximum(random_head_bytes, gather_ort_source_payload_bytes)
            stream_tail_bytes = gather_ort_dest_payload_bytes
            head_touch_count = np.maximum(
                np.maximum(gather_effective_head_touch_count[gather_mask_arr], 1.0),
                gather_ort_task_count[gather_mask_arr],
            )
            gather_random_bw_lower = kernel_random_lower
            gather_random_bw_center = kernel_random_center
            gather_random_bw_upper = kernel_random_upper
            gather_copy_bw_lower = kernel_copy_lower
            gather_copy_bw_center = kernel_copy_center
            gather_copy_bw_upper = kernel_copy_upper
        else:
            stream_tail_bytes = gather_tail_copy_bytes
            head_touch_count = np.maximum(gather_effective_head_touch_count[gather_mask_arr], 1.0)
        bdp_parallel_lower = _bytes_per_us_from_gb_s(gather_random_bw_lower[gather_mask_arr]) * latency_lower[gather_mask_arr] / cacheline_bytes
        bdp_parallel_center = _bytes_per_us_from_gb_s(gather_random_bw_center[gather_mask_arr]) * latency_center[gather_mask_arr] / cacheline_bytes
        bdp_parallel_upper = _bytes_per_us_from_gb_s(gather_random_bw_upper[gather_mask_arr]) * latency_upper[gather_mask_arr] / cacheline_bytes
        latency_parallel_lower = np.maximum(latency_parallel_lower, np.clip(bdp_parallel_lower, a_min=1.0, a_max=None))
        latency_parallel_center = np.maximum(latency_parallel_center, np.clip(bdp_parallel_center, a_min=1.0, a_max=None))
        latency_parallel_upper = np.maximum(latency_parallel_upper, np.clip(bdp_parallel_upper, a_min=1.0, a_max=None))
        head_bw_triplet = (
            random_head_bytes[gather_mask_arr]
            / np.clip(_bytes_per_us_from_gb_s(gather_random_bw_lower[gather_mask_arr]), a_min=1e-6, a_max=None),
            random_head_bytes[gather_mask_arr]
            / np.clip(_bytes_per_us_from_gb_s(gather_random_bw_center[gather_mask_arr]), a_min=1e-6, a_max=None),
            random_head_bytes[gather_mask_arr]
            / np.clip(_bytes_per_us_from_gb_s(gather_random_bw_upper[gather_mask_arr]), a_min=1e-6, a_max=None),
        )
        stream_triplet = _service_from_capacity(
            stream_tail_bytes[gather_mask_arr],
            _bytes_per_us_from_gb_s(gather_copy_bw_lower[gather_mask_arr]),
            _bytes_per_us_from_gb_s(gather_copy_bw_center[gather_mask_arr]),
            _bytes_per_us_from_gb_s(gather_copy_bw_upper[gather_mask_arr]),
        )
        lat_triplet = (
            head_touch_count * latency_lower[gather_mask_arr] / np.clip(latency_parallel_lower, a_min=1.0, a_max=None),
            head_touch_count * latency_center[gather_mask_arr] / np.clip(latency_parallel_center, a_min=1.0, a_max=None),
            head_touch_count * latency_upper[gather_mask_arr] / np.clip(latency_parallel_upper, a_min=1.0, a_max=None),
        )
        work.loc[gather_mask, "random_bw_component_lower_us"] = head_bw_triplet[0]
        work.loc[gather_mask, "random_bw_component_center_us"] = head_bw_triplet[1]
        work.loc[gather_mask, "random_bw_component_upper_us"] = head_bw_triplet[2]
        work.loc[gather_mask, "random_latency_component_lower_us"] = lat_triplet[0]
        work.loc[gather_mask, "random_latency_component_center_us"] = lat_triplet[1]
        work.loc[gather_mask, "random_latency_component_upper_us"] = lat_triplet[2]
        latency_composition_mode = _random_memory_latency_composition_mode()
        latency_tail_alpha = _random_memory_latency_tail_alpha()
        if latency_composition_mode == "additive":
            random_triplet = tuple(a + b for a, b in zip(head_bw_triplet, lat_triplet, strict=False))
        elif latency_tail_alpha > 0.0:
            random_triplet = tuple(
                np.maximum(a, b) + latency_tail_alpha * np.minimum(a, b)
                for a, b in zip(head_bw_triplet, lat_triplet, strict=False)
            )
        else:
            random_triplet = tuple(np.maximum(a, b) for a, b in zip(head_bw_triplet, lat_triplet, strict=False))
        work.loc[gather_mask, "random_mem_latency_composition"] = latency_composition_mode
        work.loc[gather_mask, "random_mem_latency_tail_alpha"] = latency_tail_alpha
        sync_events = np.zeros(int(gather_mask.sum()), dtype=float)
        sync_events = np.where(
            embedding_gather,
            np.maximum(graph_branch_fanout[gather_mask_arr] - 1.0, 0.0)
            * merge_sync_obligation[gather_mask_arr]
            / np.maximum(active_groups[gather_mask_arr], 1.0),
            np.maximum(active_groups[gather_mask_arr] - 1.0, 0.0) * sync_event_scale[gather_mask_arr],
        )
        sync_events = sync_events + (
            np.maximum(graph_branch_fanout[gather_mask_arr] - 1.0, 0.0)
            * merge_sync_obligation[gather_mask_arr]
            / np.maximum(active_groups[gather_mask_arr], 1.0)
        )
        if use_ort_kernel_v3:
            schedule_pressure = np.log1p(
                gather_ort_task_count[gather_mask_arr]
                / np.clip(effective_worker_count[gather_mask_arr], a_min=1.0, a_max=None)
            )
            grain_penalty = np.log1p(
                cacheline_bytes
                / np.clip(gather_ort_grain_bytes[gather_mask_arr], a_min=1.0, a_max=None)
            )
            sync_events = sync_events + schedule_pressure * grain_penalty * np.maximum(thread_pressure[gather_mask_arr], 0.0)
        sync_triplet = _service_from_cost(
            sync_events,
            launch_lower[gather_mask_arr],
            launch_center[gather_mask_arr],
            launch_upper[gather_mask_arr],
        )
        launch_triplet = _service_from_cost(
            dispatch_events[gather_mask_arr],
            launch_lower[gather_mask_arr],
            launch_center[gather_mask_arr],
            launch_upper[gather_mask_arr],
        )
        _set_module_interval(work, gather_mask, "stream_mem", *stream_triplet)
        _set_module_interval(work, gather_mask, "random_mem", *random_triplet)
        _set_module_interval(work, gather_mask, "sync", *sync_triplet)
        _set_module_interval(work, gather_mask, "dispatch", *launch_triplet)
        work.loc[gather_mask, "whitebox_prior_source"] = (
            "gather_ort_slice_memcpy_interval_v3" if use_ort_kernel_v3 else "gather_random_stream_latency_bandwidth_interval"
        )

    reduce_mask = exec_mode.eq(EXEC_MODE_REDUCTION)
    if reduce_mask.any():
        reduce_mask_arr = reduce_mask.to_numpy(dtype=bool)
        reduction_items = np.maximum(
            exact_read_bytes.to_numpy(dtype=float)[reduce_mask_arr]
            / np.clip(work["input_element_bytes"].to_numpy(dtype=float)[reduce_mask_arr], a_min=1.0, a_max=None),
            output_elements[reduce_mask_arr],
        )
        reduce_read_bytes = reduce_partial_read_bytes[reduce_mask_arr]
        reduce_partial_write = reduce_partial_write_bytes[reduce_mask_arr]
        reduce_merge_write = reduce_inter_branch_merge_bytes[reduce_mask_arr]
        reduce_copy_bw_lower = copy_lower
        reduce_copy_bw_center = copy_center
        reduce_copy_bw_upper = copy_upper
        if use_ort_kernel_v3:
            stream_bytes = reduce_ort_stream_bytes[reduce_mask_arr]
            reduce_copy_bw_lower = kernel_copy_lower
            reduce_copy_bw_center = kernel_copy_center
            reduce_copy_bw_upper = kernel_copy_upper
        else:
            stream_bytes = reduce_read_bytes + reduce_partial_write + reduce_merge_write
        stream_triplet = _service_from_capacity(
            stream_bytes,
            _bytes_per_us_from_gb_s(reduce_copy_bw_lower[reduce_mask_arr]),
            _bytes_per_us_from_gb_s(reduce_copy_bw_center[reduce_mask_arr]),
            _bytes_per_us_from_gb_s(reduce_copy_bw_upper[reduce_mask_arr]),
        )
        compute_triplet = _service_from_capacity(
            reduction_items,
            fp_lower[reduce_mask_arr],
            fp_center[reduce_mask_arr],
            fp_upper[reduce_mask_arr],
        )
        compute_triplet = _add_compute_latency_tail(
            work.loc[reduce_mask],
            hw_params,
            compute_triplet,
            reduction_items,
            reduction_depth[reduce_mask_arr] + reduction_intra_tree_depth[reduce_mask_arr],
        )
        reduction_axes_count = np.maximum(_series_numeric(work.loc[reduce_mask], "feat_reduction_axes_count", 1.0).to_numpy(dtype=float), 1.0)
        if use_ort_kernel_v3:
            reorder_events = np.maximum(
                reduction_axes_count
                + reduction_middle_axis_flag[reduce_mask_arr] * np.maximum(output_rank[reduce_mask_arr], 1.0)
                + reduction_rank_collapse_flag[reduce_mask_arr] * np.maximum(rank_delta[reduce_mask_arr], 1.0)
                + np.log1p(reduce_ort_grain_tasks[reduce_mask_arr]),
                1.0,
            )
        else:
            reorder_events = np.maximum(
                reduction_depth[reduce_mask_arr] * reduction_axes_count
                + thread_pressure[reduce_mask_arr],
                1.0,
            )
            reorder_events = reorder_events + reduction_intra_tree_depth[reduce_mask_arr]
            reorder_events = reorder_events + reduction_inter_tree_depth[reduce_mask_arr]
            reorder_events = reorder_events + reduction_middle_axis_flag[reduce_mask_arr] * np.maximum(output_rank[reduce_mask_arr], 1.0)
            reorder_events = reorder_events + reduction_rank_collapse_flag[reduce_mask_arr] * np.maximum(rank_delta[reduce_mask_arr], 1.0)
        reorder_triplet = _service_from_capacity(
            reorder_events,
            reorder_tp_lower[reduce_mask_arr],
            reorder_tp_center[reduce_mask_arr],
            reorder_tp_upper[reduce_mask_arr],
        )
        if use_ort_kernel_v3:
            parallel_grain = reduce_ort_grain_tasks[reduce_mask_arr]
            intra_sync_events = np.where(
                parallel_grain > effective_worker_count[reduce_mask_arr] * 2.0,
                np.maximum(thread_pressure[reduce_mask_arr], 0.0),
                0.0,
            )
            inter_sync_events = merge_sync_obligation[reduce_mask_arr] * np.maximum(active_groups[reduce_mask_arr] - 1.0, 0.0)
            sync_events = intra_sync_events + inter_sync_events + reduction_keepdims_flag[reduce_mask_arr] * merge_sync_obligation[reduce_mask_arr]
        else:
            intra_sync_events = reduction_intra_tree_depth[reduce_mask_arr] * np.maximum(thread_pressure[reduce_mask_arr], 0.0)
            inter_sync_events = reduction_inter_tree_depth[reduce_mask_arr] * merge_sync_obligation[reduce_mask_arr]
            sync_events = intra_sync_events + inter_sync_events + reduction_keepdims_flag[reduce_mask_arr] * merge_sync_obligation[reduce_mask_arr]
        sync_triplet = _service_from_cost(
            sync_events,
            launch_lower[reduce_mask_arr],
            launch_center[reduce_mask_arr],
            launch_upper[reduce_mask_arr],
        )
        launch_triplet = _service_from_cost(
            dispatch_events[reduce_mask_arr],
            launch_lower[reduce_mask_arr],
            launch_center[reduce_mask_arr],
            launch_upper[reduce_mask_arr],
        )
        _set_module_interval(work, reduce_mask, "stream_mem", *stream_triplet)
        _set_module_interval(work, reduce_mask, "compute", *compute_triplet)
        _set_module_interval(work, reduce_mask, "reorder", *reorder_triplet)
        _set_module_interval(work, reduce_mask, "sync", *sync_triplet)
        _set_module_interval(work, reduce_mask, "dispatch", *launch_triplet)
        work.loc[reduce_mask, "whitebox_prior_source"] = (
            "reduce_ort_fast_reduce_interval_v3" if use_ort_kernel_v3 else "reduce_stream_compute_tree_interval"
        )

    concat_payload_mask = exec_mode.eq(EXEC_MODE_PAYLOAD_CONCAT)
    if concat_payload_mask.any():
        concat_mask = concat_payload_mask
        read_bytes = np.maximum(exact_read_bytes.to_numpy(dtype=float)[concat_mask.to_numpy()], 0.0)
        write_bytes = np.maximum(exact_write_bytes.to_numpy(dtype=float)[concat_mask.to_numpy()], 0.0)
        copy_window_bytes = np.maximum(read_bytes, write_bytes)
        stream_triplet = _service_from_capacity(
            copy_window_bytes,
            _bytes_per_us_from_gb_s(copy_lower[concat_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_center[concat_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_upper[concat_mask.to_numpy()]),
        )
        descriptor_events = np.maximum(
            input_tensors[concat_mask.to_numpy()]
            + output_rank[concat_mask.to_numpy()] + 1.0,
            1.0,
        )
        descriptor_triplet = _service_from_capacity(
            descriptor_events,
            desc_tp_lower[concat_mask.to_numpy()],
            desc_tp_center[concat_mask.to_numpy()],
            desc_tp_upper[concat_mask.to_numpy()],
        )
        sync_triplet = _service_from_cost(
            np.maximum(input_tensors[concat_mask.to_numpy()] - 1.0, 0.0)
            * sync_event_scale[concat_mask.to_numpy()]
            * merge_sync_obligation[concat_mask.to_numpy()],
            launch_lower[concat_mask.to_numpy()],
            launch_center[concat_mask.to_numpy()],
            launch_upper[concat_mask.to_numpy()],
        )
        launch_triplet = _service_from_cost(
            dispatch_events[concat_mask.to_numpy()],
            launch_lower[concat_mask.to_numpy()],
            launch_center[concat_mask.to_numpy()],
            launch_upper[concat_mask.to_numpy()],
        )
        _set_module_interval(work, concat_mask, "stream_mem", *stream_triplet)
        _set_module_interval(work, concat_mask, "descriptor", *descriptor_triplet)
        _set_module_interval(work, concat_mask, "sync", *sync_triplet)
        _set_module_interval(work, concat_mask, "dispatch", *launch_triplet)
        work.loc[concat_mask, "whitebox_prior_source"] = "concat_stream_metadata_interval"

    concat_shape_mask = exec_mode.eq(EXEC_MODE_SHAPE_CONCAT)
    if concat_shape_mask.any():
        descriptor_events = np.maximum(
            input_tensors[concat_shape_mask.to_numpy()]
            + output_rank[concat_shape_mask.to_numpy()]
            + np.abs(work["attr_axis_value"].to_numpy(dtype=float)[concat_shape_mask.to_numpy()])
            + np.log1p(output_elements[concat_shape_mask.to_numpy()]),
            1.0,
        )
        descriptor_triplet = _service_from_capacity(
            descriptor_events,
            desc_tp_lower[concat_shape_mask.to_numpy()],
            desc_tp_center[concat_shape_mask.to_numpy()],
            desc_tp_upper[concat_shape_mask.to_numpy()],
        )
        shape_concat_dispatch = np.ones(int(concat_shape_mask.sum()), dtype=float)
        launch_triplet = _service_from_cost(
            shape_concat_dispatch,
            np.zeros(int(concat_shape_mask.sum()), dtype=float),
            inline_launch_center[concat_shape_mask.to_numpy()],
            launch_upper[concat_shape_mask.to_numpy()],
        )
        _set_module_interval(work, concat_shape_mask, "descriptor", *descriptor_triplet)
        _set_module_interval(work, concat_shape_mask, "dispatch", *launch_triplet)
        work.loc[concat_shape_mask, "whitebox_prior_source"] = "shape_concat_descriptor_interval"

    tiny_shape_mask = exec_mode.eq(EXEC_MODE_TINY_SHAPE_ARITH)
    if tiny_shape_mask.any():
        tiny_descriptor_events = np.maximum(
            input_tensors[tiny_shape_mask.to_numpy()]
            + output_rank[tiny_shape_mask.to_numpy()]
            + np.abs(rank_delta[tiny_shape_mask.to_numpy()])
            + 1.0,
            1.0,
        )
        tiny_compute_items = np.maximum(output_elements[tiny_shape_mask.to_numpy()], 1.0)
        tiny_descriptor_triplet = _metadata_touch_triplet_us(tiny_descriptor_events, hw_params)
        tiny_compute_triplet = _service_from_capacity(
            tiny_compute_items,
            reorder_tp_lower[tiny_shape_mask.to_numpy()],
            reorder_tp_center[tiny_shape_mask.to_numpy()],
            reorder_tp_upper[tiny_shape_mask.to_numpy()],
        )
        tiny_compute_triplet = _add_compute_latency_tail(
            work.loc[tiny_shape_mask],
            hw_params,
            tiny_compute_triplet,
            tiny_compute_items,
        )
        tiny_dispatch_events = np.ones(int(tiny_shape_mask.sum()), dtype=float)
        tiny_launch_triplet = _service_from_cost(
            tiny_dispatch_events,
            np.zeros(int(tiny_shape_mask.sum()), dtype=float),
            inline_launch_center[tiny_shape_mask.to_numpy()],
            launch_upper[tiny_shape_mask.to_numpy()],
        )
        _set_module_interval(work, tiny_shape_mask, "descriptor", *tiny_descriptor_triplet)
        _set_module_interval(work, tiny_shape_mask, "compute", *tiny_compute_triplet)
        _set_module_interval(work, tiny_shape_mask, "dispatch", *tiny_launch_triplet)
        work.loc[tiny_shape_mask, "whitebox_prior_source"] = "tiny_shape_descriptor_compute_interval"

    ewise_mask = work["whitebox_prior_family"].eq("ewise_like") & ~tiny_shape_mask
    if ewise_mask.any():
        stream_triplet = _service_from_capacity(
            io_bytes.to_numpy(dtype=float)[ewise_mask.to_numpy()],
            _bytes_per_us_from_gb_s(copy_lower[ewise_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_center[ewise_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_upper[ewise_mask.to_numpy()]),
        )
        ewise_compute_items = output_elements[ewise_mask.to_numpy()] * np.maximum(input_tensors[ewise_mask.to_numpy()] - 1.0, 1.0)
        compute_triplet = _service_from_capacity(
            ewise_compute_items,
            fp_lower[ewise_mask.to_numpy()],
            fp_center[ewise_mask.to_numpy()],
            fp_upper[ewise_mask.to_numpy()],
        )
        compute_triplet = _add_compute_latency_tail(
            work.loc[ewise_mask],
            hw_params,
            compute_triplet,
            ewise_compute_items,
        )
        launch_triplet = _service_from_cost(
            dispatch_events[ewise_mask.to_numpy()],
            launch_lower[ewise_mask.to_numpy()],
            launch_center[ewise_mask.to_numpy()],
            launch_upper[ewise_mask.to_numpy()],
        )
        _set_module_interval(work, ewise_mask, "stream_mem", *stream_triplet)
        _set_module_interval(work, ewise_mask, "compute", *compute_triplet)
        _set_module_interval(work, ewise_mask, "dispatch", *launch_triplet)
        work.loc[ewise_mask, "whitebox_prior_source"] = "ewise_stream_compute_interval"

    transpose_mask = exec_mode.eq(EXEC_MODE_TRANSPOSE)
    if transpose_mask.any():
        transpose_is_reshape_like = transpose_reshape_like_flag[transpose_mask.to_numpy()] > 0.5
        transpose_is_single_axis = transpose_single_axis_flag[transpose_mask.to_numpy()] > 0.5
        read_bytes = np.maximum(exact_read_bytes.to_numpy(dtype=float)[transpose_mask.to_numpy()], 0.0)
        write_bytes = np.maximum(exact_write_bytes.to_numpy(dtype=float)[transpose_mask.to_numpy()], 0.0)
        copy_window_bytes = np.maximum(read_bytes, write_bytes)
        copy_window_bytes = np.where(transpose_is_reshape_like, 0.0, copy_window_bytes)
        stream_triplet = _service_from_capacity(
            copy_window_bytes,
            _bytes_per_us_from_gb_s(copy_lower[transpose_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_center[transpose_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_upper[transpose_mask.to_numpy()]),
        )
        descriptor_events = np.maximum(meta_tokens[transpose_mask.to_numpy()] + perm_rank[transpose_mask.to_numpy()], 1.0)
        descriptor_events = np.where(
            transpose_is_reshape_like,
            np.maximum(descriptor_events + shape_tokens[transpose_mask.to_numpy()], 1.0),
            descriptor_events,
        )
        descriptor_triplet = _service_from_capacity(
            descriptor_events,
            desc_tp_lower[transpose_mask.to_numpy()],
            desc_tp_center[transpose_mask.to_numpy()],
            desc_tp_upper[transpose_mask.to_numpy()],
        )
        reorder_events = np.maximum(perm_moved_axes[transpose_mask.to_numpy()] + perm_inversion_count[transpose_mask.to_numpy()], 1.0)
        reorder_events = np.where(transpose_is_reshape_like, 0.0, reorder_events)
        reorder_triplet = _service_from_capacity(
            reorder_events,
            reorder_tp_lower[transpose_mask.to_numpy()],
            reorder_tp_center[transpose_mask.to_numpy()],
            reorder_tp_upper[transpose_mask.to_numpy()],
        )
        launch_triplet = _service_from_cost(
            dispatch_events[transpose_mask.to_numpy()],
            launch_lower[transpose_mask.to_numpy()],
            launch_center[transpose_mask.to_numpy()],
            launch_upper[transpose_mask.to_numpy()],
        )
        _set_module_interval(work, transpose_mask, "stream_mem", *stream_triplet)
        _set_module_interval(work, transpose_mask, "descriptor", *descriptor_triplet)
        _set_module_interval(work, transpose_mask, "reorder", *reorder_triplet)
        _set_module_interval(work, transpose_mask, "dispatch", *launch_triplet)
        work.loc[transpose_mask, "whitebox_prior_source"] = "transpose_stream_perm_interval"

    gemm_mask = exec_mode.eq(EXEC_MODE_GEMM)
    if gemm_mask.any():
        mac = np.maximum(_series_numeric(work.loc[gemm_mask], "feat_gemm_mac_count", 0.0).to_numpy(dtype=float), output_elements[gemm_mask.to_numpy()])
        compute_triplet = _service_from_capacity(
            2.0 * mac,
            fp_lower[gemm_mask.to_numpy()],
            fp_center[gemm_mask.to_numpy()],
            fp_upper[gemm_mask.to_numpy()],
        )
        gemm_avg_k = mac / np.clip(output_elements[gemm_mask.to_numpy()], a_min=1.0, a_max=None)
        compute_triplet = _add_compute_latency_tail(
            work.loc[gemm_mask],
            hw_params,
            compute_triplet,
            2.0 * mac,
            np.log1p(gemm_avg_k),
        )
        stream_triplet = _service_from_capacity(
            io_bytes.to_numpy(dtype=float)[gemm_mask.to_numpy()],
            _bytes_per_us_from_gb_s(copy_lower[gemm_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_center[gemm_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_upper[gemm_mask.to_numpy()]),
        )
        reorder_events = np.maximum(input_tensors[gemm_mask.to_numpy()] + output_rank[gemm_mask.to_numpy()], 1.0)
        reorder_triplet = _service_from_capacity(
            reorder_events,
            reorder_tp_lower[gemm_mask.to_numpy()],
            reorder_tp_center[gemm_mask.to_numpy()],
            reorder_tp_upper[gemm_mask.to_numpy()],
        )
        launch_triplet = _service_from_cost(
            dispatch_events[gemm_mask.to_numpy()],
            launch_lower[gemm_mask.to_numpy()],
            launch_center[gemm_mask.to_numpy()],
            launch_upper[gemm_mask.to_numpy()],
        )
        _set_module_interval(work, gemm_mask, "stream_mem", *stream_triplet)
        _set_module_interval(work, gemm_mask, "compute", *compute_triplet)
        _set_module_interval(work, gemm_mask, "reorder", *reorder_triplet)
        _set_module_interval(work, gemm_mask, "dispatch", *launch_triplet)
        work.loc[gemm_mask, "whitebox_prior_source"] = "gemm_stream_compute_interval"

    covered_mask = meta_mask | gather_mask | reduce_mask | concat_payload_mask | concat_shape_mask | tiny_shape_mask | ewise_mask | transpose_mask | gemm_mask
    other_mask = ~covered_mask
    if other_mask.any():
        generic_stream = _service_from_capacity(
            io_bytes.to_numpy(dtype=float)[other_mask.to_numpy()] * np.maximum(materialize_required[other_mask.to_numpy()], 0.0),
            _bytes_per_us_from_gb_s(copy_lower[other_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_center[other_mask.to_numpy()]),
            _bytes_per_us_from_gb_s(copy_upper[other_mask.to_numpy()]),
        )
        generic_descriptor = _service_from_capacity(
            np.maximum(meta_tokens[other_mask.to_numpy()], 1.0),
            desc_tp_lower[other_mask.to_numpy()],
            desc_tp_center[other_mask.to_numpy()],
            desc_tp_upper[other_mask.to_numpy()],
        )
        generic_compute = _service_from_capacity(
            output_elements[other_mask.to_numpy()],
            fp_lower[other_mask.to_numpy()],
            fp_center[other_mask.to_numpy()],
            fp_upper[other_mask.to_numpy()],
        )
        generic_compute = _add_compute_latency_tail(
            work.loc[other_mask],
            hw_params,
            generic_compute,
            output_elements[other_mask.to_numpy()],
        )
        launch_triplet = _service_from_cost(
            dispatch_events[other_mask.to_numpy()],
            launch_lower[other_mask.to_numpy()],
            launch_center[other_mask.to_numpy()],
            launch_upper[other_mask.to_numpy()],
        )
        _set_module_interval(work, other_mask, "stream_mem", *generic_stream)
        _set_module_interval(work, other_mask, "descriptor", *generic_descriptor)
        _set_module_interval(work, other_mask, "compute", *generic_compute)
        _set_module_interval(work, other_mask, "dispatch", *launch_triplet)
        work.loc[other_mask, "whitebox_prior_source"] = "other_physical_interval_fallback"

    _zero_disabled_analytical_modules(work)

    for module_name in MODULE_NAMES:
        work[f"svc_{module_name}_us"] = pd.to_numeric(work[_module_interval_column(module_name, "center")], errors="coerce").fillna(0.0)

    current_stream = pd.to_numeric(work["svc_stream_mem_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gather_slice_stream = _series_numeric(work, "gather_slice_stream_us", 0.0).to_numpy(dtype=float)
    work["gather_slice_vs_current_stream_ratio"] = np.where(
        current_stream > 0.0,
        gather_slice_stream / np.clip(current_stream, a_min=1e-6, a_max=None),
        1.0,
    )
    work["gather_slice_underpredict_gap_us"] = np.clip(gather_slice_stream - current_stream, a_min=0.0, a_max=None)

    _service_share_columns(work)
    _compose_total_intervals(work)
    work["pred_mem_us"] = pd.to_numeric(work["svc_stream_mem_us"], errors="coerce").fillna(0.0) + pd.to_numeric(work["svc_random_mem_us"], errors="coerce").fillna(0.0)
    work["pred_compute_us"] = (
        pd.to_numeric(work["svc_descriptor_us"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["svc_compute_us"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["svc_reorder_us"], errors="coerce").fillna(0.0)
    )
    work["pred_sync_us"] = pd.to_numeric(work["svc_sync_us"], errors="coerce").fillna(0.0)
    work["pred_launch_us"] = pd.to_numeric(work["svc_dispatch_us"], errors="coerce").fillna(0.0)
    work["pred_total_lower_us"] = work["whitebox_lower_us"]
    work["pred_total_us"] = work["whitebox_center_us"]
    work["pred_total_upper_us"] = work["whitebox_upper_us"]
    work["whitebox_prior_us"] = work["whitebox_center_us"]
    _apply_op_thread_scale(work, bundle.get("op_thread_scale_config"))
    work["module_schema_version"] = int(WHITEBOX_MODULE_SCHEMA_VERSION)
    work["whitebox_hw_config_path"] = str(hw_params.get("hw_config_path", bundle.get("hw_config_path", "")))
    work["whitebox_missing_hw_fields"] = ",".join(str(value) for value in hw_params.get("missing_whitebox_fields", []))
    return work


def _sanitize_training_summary(summary: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in summary.items():
        if key in {"model", "preprocess_state"}:
            continue
        if isinstance(value, np.generic):
            cleaned[str(key)] = value.item()
        else:
            cleaned[str(key)] = value
    cleaned["model_level"] = "rf_group"
    cleaned["whitebox_prior_source"] = "physically_constrained_interval_prior"
    return cleaned


def _model_sha256(family_bundles: dict[str, dict[str, Any]]) -> str:
    payload: dict[str, Any] = {}
    for family_name, bundle in sorted(family_bundles.items()):
        payload[str(family_name)] = {
            "enabled": bool(bundle.get("enabled", False)),
            "baseline_column": str(bundle.get("baseline_column", "whitebox_center_us")),
            "preprocess_state": bundle.get("preprocess_state"),
            "model": bundle.get("model"),
        }
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(blob).hexdigest()


def fit_whitebox_family_prior_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    prior_variant: str = "legacy_v2",
    op_thread_scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del fit_frame
    _assert_physical_constant_audit()
    hw_params = _load_hw_params(hw_config, require_measured_whitebox=False)
    normalized_scale_config = _normalize_op_thread_scale_config(op_thread_scale_config)
    return {
        "route": WHITEBOX_PRIOR_ONLY_ROUTE,
        "route_whitebox_mode": PURE_ZERO_FIT_MODE,
        "fit_data_dependent": False,
        "policy": str(policy or ""),
        "prior_variant": str(prior_variant),
        "hw_config_path": str(Path(hw_config).resolve()),
        "hw_params": hw_params,
        "required_hw_fields": list(REQUIRED_WHITEBOX_HW_FIELDS),
        "module_schema_version": int(WHITEBOX_MODULE_SCHEMA_VERSION),
        "op_thread_scale_enabled": bool(normalized_scale_config),
        "op_thread_scale_config": normalized_scale_config,
    }


def benchmark_variant_route_name(variant: str) -> str:
    variant_key = str(variant).strip().lower()
    if variant_key not in WHITEBOX_WORKLOAD_BENCHMARK_LIMITS:
        raise ValueError(f"Unsupported benchmark variant: {variant}")
    return f"{WHITEBOX_WORKLOAD_BENCHMARK_ROUTE}[{variant_key}]"


def _validation_groups_by_policy() -> dict[str, list[str]]:
    return {
        "batch_size": ["batch_size"],
        "num_indices_per_lookup": ["num_indices_per_lookup"],
        "arch_embedding_size": ["arch_embedding_size"],
        "arch_mlp_bot": ["arch_mlp_bot"],
        "inter_threads_decoupled_v2": ["inter_threads", "combo"],
        "num_threads_decoupled_v2": ["num_threads", "combo"],
        "joint_6d_tuple": ["joint_6d_tuple_key"],
    }


def _physics_guarded_counterfactual_training_frame(
    fit_frame: pd.DataFrame,
    prior_bundle: dict[str, Any],
    fit_prior: pd.DataFrame,
) -> pd.DataFrame:
    real = fit_prior.copy().reset_index(drop=True)
    actual = pd.to_numeric(real[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    prior0 = pd.to_numeric(real["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    residual0 = actual - prior0
    source_index = np.arange(len(real), dtype=int)
    real[PHYSICS_GUARDED_SYNTHETIC_INDICATOR_COLUMN] = False
    real[PHYSICS_GUARDED_SYNTHETIC_SOURCE_INDEX_COLUMN] = source_index
    real[PHYSICS_GUARDED_SYNTHETIC_WEIGHT_COLUMN] = 1.0
    real["physics_guarded_counterfactual_field"] = "__real__"
    real["physics_guarded_counterfactual_factor"] = 1.0

    base_payload = dict(prior_bundle.get("hw_params", {}).get("payload", {}))
    raw = fit_frame.reset_index(drop=True)
    chunks = [real]
    seen: set[str] = set()
    for field, factor in PHYSICS_GUARDED_COUNTERFACTUAL_FIELD_FACTORS:
        payload = _scaled_hw_payload(base_payload, field, float(factor))
        if payload is None:
            continue
        key = json.dumps(payload, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        mutated = add_whitebox_family_prior_columns(raw, prior_bundle, override_hw_config=payload).reset_index(drop=True)
        prior_h = pd.to_numeric(mutated["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mutated[base_run.TARGET_US_COLUMN] = np.clip(prior_h + residual0, a_min=1e-6, a_max=None)
        mutated[PHYSICS_GUARDED_SYNTHETIC_INDICATOR_COLUMN] = True
        mutated[PHYSICS_GUARDED_SYNTHETIC_SOURCE_INDEX_COLUMN] = source_index
        mutated[PHYSICS_GUARDED_SYNTHETIC_WEIGHT_COLUMN] = float(PHYSICS_GUARDED_COUNTERFACTUAL_WEIGHT)
        mutated["physics_guarded_counterfactual_field"] = str(field)
        mutated["physics_guarded_counterfactual_factor"] = float(factor)
        chunks.append(mutated)
    return pd.concat(chunks, ignore_index=True, sort=False)


def _fit_whitebox_rf_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None,
    route_name: str,
    route_whitebox_mode: str,
    ml_input_source: str,
    ml_raw_sw_features_used: bool,
    allow_workload_features: bool,
    feature_limit: int,
    workload_feature_groups: set[str] | None = None,
    benchmark_variant: str = "",
    prior_variant: str = "legacy_v2",
    prediction_clip_by_group: dict[str, float] | None = None,
    sample_weight_by_group: dict[str, str] | None = None,
    feature_limit_by_group: dict[str, int] | None = None,
    feature_include_by_group: dict[str, list[str]] | None = None,
    feature_exclude_by_group: dict[str, set[str]] | None = None,
    feature_set_name: str = "",
    op_thread_scale_config: dict[str, Any] | None = None,
    target_mode_override_by_group: dict[str, str] | None = None,
    monotonic_feature_columns: set[str] | None = None,
    physics_guarded_training: bool = False,
) -> dict[str, Any]:
    validation_groups_by_policy = _validation_groups_by_policy()
    prior_bundle = fit_whitebox_family_prior_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        prior_variant=prior_variant,
        op_thread_scale_config=op_thread_scale_config,
    )
    fit_prior = add_whitebox_family_prior_columns(fit_frame, prior_bundle)
    fit_training = (
        _physics_guarded_counterfactual_training_frame(fit_frame, prior_bundle, fit_prior)
        if physics_guarded_training
        else fit_prior
    )
    rf_group_bundles: dict[str, dict[str, Any]] = {}
    training_summaries: list[dict[str, Any]] = []
    candidate_feature_rows: list[dict[str, Any]] = []
    pruned_feature_rows: list[dict[str, Any]] = []
    group_feature_count_rows: list[dict[str, Any]] = []
    rf_groups = [
        RF_GROUP_METADATA,
        RF_GROUP_TINY_SHAPE,
        RF_GROUP_INDEXED_COPY,
        RF_GROUP_PAYLOAD_CONCAT,
        RF_GROUP_REDUCTION,
        RF_GROUP_TRANSPOSE,
        RF_GROUP_GEMM,
    ]
    for group_name in rf_groups:
        family_frame = fit_training.loc[fit_training["whitebox_rf_group"].astype(str).eq(group_name)].copy()
        group_feature_limit = int(dict(feature_limit_by_group or {}).get(str(group_name), int(feature_limit)))
        feature_schema = rf_group_feature_schema(
            group_name,
            family_frame if not family_frame.empty else fit_training,
            allow_workload_features=allow_workload_features,
            workload_feature_groups=workload_feature_groups,
            feature_limit=group_feature_limit,
            feature_include=dict(feature_include_by_group or {}).get(str(group_name)),
            feature_exclude=dict(feature_exclude_by_group or {}).get(str(group_name)),
            target_mode_override=dict(target_mode_override_by_group or {}).get(str(group_name)),
        )
        bundle = fit_family_feature_rf(
            family_frame,
            family_name=str(group_name),
            feature_columns=list(feature_schema["feature_columns"]),
            log1p_columns=list(feature_schema["log1p_columns"]),
            target_column=base_run.TARGET_US_COLUMN,
            baseline_column="whitebox_center_us",
            n_estimators=512,
            max_depth=16,
            min_samples_leaf=4,
            max_features="sqrt",
            min_rows=96,
            min_relative_improvement=0.0,
            seed=0,
            validation_group_columns=validation_groups_by_policy.get(str(policy or ""), []),
            target_mode=str(feature_schema["target_mode"]),
            prediction_clip_k=float(dict(prediction_clip_by_group or {}).get(str(group_name), 1.0)),
            sample_weight_column=(
                PHYSICS_GUARDED_SYNTHETIC_WEIGHT_COLUMN
                if physics_guarded_training
                else dict(sample_weight_by_group or {}).get(str(group_name))
            ),
            monotonic_feature_columns=monotonic_feature_columns,
            synthetic_indicator_column=PHYSICS_GUARDED_SYNTHETIC_INDICATOR_COLUMN if physics_guarded_training else None,
            synthetic_source_index_column=PHYSICS_GUARDED_SYNTHETIC_SOURCE_INDEX_COLUMN if physics_guarded_training else None,
        )
        bundle["rf_group"] = str(group_name)
        bundle["fallback_group"] = feature_schema.get("fallback_group")
        bundle["candidate_feature_rows"] = list(feature_schema["candidate_rows"])
        bundle["selected_feature_rows"] = list(feature_schema["selected_rows"])
        bundle["selected_feature_columns"] = list(feature_schema["feature_columns"])
        bundle["feature_schema_sha256"] = str(feature_schema["schema_sha256"])
        bundle["benchmark_variant"] = str(benchmark_variant or "")
        rf_group_bundles[str(group_name)] = bundle
        candidate_feature_rows.extend(
            [
                {
                    "rf_group": str(group_name),
                    "rows": int(len(family_frame)),
                    "benchmark_variant": str(benchmark_variant or ""),
                    "feature_set_name": str(feature_set_name or ""),
                    **row,
                }
                for row in feature_schema["candidate_rows"]
            ]
        )
        pruned_feature_rows.extend(
            [
                {
                    "rf_group": str(group_name),
                    "rows": int(len(family_frame)),
                    "benchmark_variant": str(benchmark_variant or ""),
                    "feature_set_name": str(feature_set_name or ""),
                    **row,
                }
                for row in feature_schema["selected_rows"]
            ]
        )
        group_feature_count_rows.append(
            {
                "rf_group": str(group_name),
                "rows": int(len(family_frame)),
                "feature_count": int(feature_schema["feature_count"]),
                "feature_limit": int(feature_schema.get("feature_limit", group_feature_limit)),
                "allow_workload_features": bool(feature_schema.get("allow_workload_features", allow_workload_features)),
                "enabled": bool(bundle.get("enabled", False)),
                "reason": str(bundle.get("reason", "")),
                "fallback_group": str(feature_schema.get("fallback_group") or ""),
                "target_mode": str(feature_schema.get("target_mode", "")),
                "schema_sha256": str(feature_schema.get("schema_sha256", "")),
                "benchmark_variant": str(benchmark_variant or ""),
                "prior_variant": str(prior_variant),
                "prediction_clip_k": float(dict(prediction_clip_by_group or {}).get(str(group_name), 1.0)),
                "sample_weight_column": str(dict(sample_weight_by_group or {}).get(str(group_name), "")),
                "feature_set_name": str(feature_set_name or ""),
                "physics_guarded_training": bool(physics_guarded_training),
            }
        )
        training_summaries.append(_sanitize_training_summary(bundle))

    return {
        "route": str(route_name),
        "route_whitebox_mode": str(route_whitebox_mode),
        "fit_data_dependent": True,
        "policy": str(policy or ""),
        "prior_bundle": prior_bundle,
        "rf_group_bundles": rf_group_bundles,
        "family_bundles": rf_group_bundles,
        "model_sha256": _model_sha256(rf_group_bundles),
        "train_backend": "sklearn_random_forest",
        "train_env": "cpu",
        "ml_input_source": str(ml_input_source),
        "ml_raw_sw_features_used": bool(ml_raw_sw_features_used),
        "ml_raw_hw_params_used": False,
        "hw_refit_required": False,
        "benchmark_variant": str(benchmark_variant or ""),
        "prior_variant": str(prior_variant),
        "feature_limit": int(feature_limit),
        "feature_set_name": str(feature_set_name or ""),
        "physics_guarded_training": bool(physics_guarded_training),
        "physics_guard_delta_cap_multiplier": float(PHYSICS_GUARD_DELTA_CAP_MULTIPLIER),
        "family_mlp_enabled_count": int(sum(1 for bundle in rf_group_bundles.values() if bool(bundle.get("enabled", False)))),
        "family_training_summaries": training_summaries,
        "rf_candidate_feature_rows": candidate_feature_rows,
        "rf_pruned_feature_rows": pruned_feature_rows,
        "rf_group_feature_count_rows": group_feature_count_rows,
        "module_schema_version": int(WHITEBOX_MODULE_SCHEMA_VERSION),
    }

def fit_whitebox_family_simple_rf_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    op_thread_scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _fit_whitebox_rf_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        route_name=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
        route_whitebox_mode=WHITEBOX_FEATURE_ML_MODE,
        ml_input_source="whitebox_interval_modules_v4_rf_groups",
        ml_raw_sw_features_used=False,
        allow_workload_features=False,
        feature_limit=int(RF_FEATURE_LIMIT),
        op_thread_scale_config=op_thread_scale_config,
    )


def fit_whitebox_family_physics_guarded_rf_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    op_thread_scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rf_groups = [
        RF_GROUP_METADATA,
        RF_GROUP_TINY_SHAPE,
        RF_GROUP_INDEXED_COPY,
        RF_GROUP_PAYLOAD_CONCAT,
        RF_GROUP_REDUCTION,
        RF_GROUP_TRANSPOSE,
        RF_GROUP_GEMM,
    ]
    return _fit_whitebox_rf_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        route_name=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
        route_whitebox_mode=WHITEBOX_PHYSICS_GUARDED_RF_MODE,
        ml_input_source="whitebox_interval_modules_v4_physics_guarded_rf_groups",
        ml_raw_sw_features_used=False,
        allow_workload_features=False,
        feature_limit=int(RF_FEATURE_LIMIT),
        feature_exclude_by_group={group: set() for group in rf_groups},
        target_mode_override_by_group={},
        monotonic_feature_columns=set(MONOTONIC_POSITIVE_ANCHOR_FEATURES),
        physics_guarded_training=True,
        op_thread_scale_config=op_thread_scale_config,
    )


def fit_whitebox_family_ort_kernel_rf_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    op_thread_scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _fit_whitebox_rf_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        route_name=WHITEBOX_ORT_KERNEL_RF_ROUTE,
        route_whitebox_mode=WHITEBOX_ORT_KERNEL_RF_MODE,
        ml_input_source="whitebox_ort_kernel_modules_v3_rf_groups",
        ml_raw_sw_features_used=True,
        allow_workload_features=False,
        workload_feature_groups={RF_GROUP_INDEXED_COPY, RF_GROUP_REDUCTION},
        feature_limit=int(RF_FEATURE_LIMIT),
        prior_variant="ort_kernel_v3",
        op_thread_scale_config=op_thread_scale_config,
    )


def fit_whitebox_family_embedding_mem_rf_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    op_thread_scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _fit_whitebox_rf_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        route_name=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
        route_whitebox_mode=WHITEBOX_EMBEDDING_MEM_RF_MODE,
        ml_input_source="whitebox_interval_modules_v4_group_clip_rf_groups",
        ml_raw_sw_features_used=False,
        allow_workload_features=False,
        feature_limit=int(RF_FEATURE_LIMIT),
        prior_variant="legacy_v2",
        prediction_clip_by_group={
            RF_GROUP_INDEXED_COPY: 1.5,
            RF_GROUP_REDUCTION: 3.5,
            RF_GROUP_GEMM: 4.0,
        },
        op_thread_scale_config=op_thread_scale_config,
    )


def fit_whitebox_family_gather_slice_rf_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    op_thread_scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = _fit_whitebox_rf_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        route_name=WHITEBOX_GATHER_SLICE_RF_ROUTE,
        route_whitebox_mode=WHITEBOX_GATHER_SLICE_RF_MODE,
        ml_input_source="whitebox_gather_slice_v5_rf_groups",
        ml_raw_sw_features_used=True,
        allow_workload_features=False,
        workload_feature_groups={RF_GROUP_INDEXED_COPY},
        feature_limit=45,
        prior_variant="legacy_v2",
        prediction_clip_by_group={
            RF_GROUP_INDEXED_COPY: 2.5,
            RF_GROUP_REDUCTION: 3.5,
            RF_GROUP_GEMM: 4.0,
        },
        sample_weight_by_group={
            RF_GROUP_INDEXED_COPY: base_run.TARGET_US_COLUMN,
        },
        op_thread_scale_config=op_thread_scale_config,
    )
    bundle["gather_slice_post_correction_alpha"] = 0.5
    return bundle


def fit_whitebox_family_gather_slice_compact_rf_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    op_thread_scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    indexed_feature_set = list(dict.fromkeys(GATHER_SLICE_COMPACT_INDEXED_FEATURES))
    bundle = _fit_whitebox_rf_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        route_name=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
        route_whitebox_mode=WHITEBOX_GATHER_SLICE_COMPACT_RF_MODE,
        ml_input_source="whitebox_gather_slice_v5_compact_probe_rf_groups",
        ml_raw_sw_features_used=True,
        allow_workload_features=False,
        workload_feature_groups={RF_GROUP_INDEXED_COPY},
        feature_limit=45,
        feature_limit_by_group={RF_GROUP_INDEXED_COPY: len(indexed_feature_set)},
        feature_include_by_group={RF_GROUP_INDEXED_COPY: indexed_feature_set},
        feature_exclude_by_group={},
        feature_set_name=GATHER_SLICE_COMPACT_FEATURE_SET_NAME,
        prior_variant="legacy_v2",
        prediction_clip_by_group={
            RF_GROUP_INDEXED_COPY: 2.5,
            RF_GROUP_REDUCTION: 3.5,
            RF_GROUP_GEMM: 4.0,
        },
        sample_weight_by_group={
            RF_GROUP_INDEXED_COPY: base_run.TARGET_US_COLUMN,
        },
        op_thread_scale_config=op_thread_scale_config,
    )
    bundle["gather_slice_post_correction_alpha"] = 0.5
    bundle["compact_feature_set_name"] = GATHER_SLICE_COMPACT_FEATURE_SET_NAME
    bundle["compact_removed_feature_groups"] = GATHER_SLICE_COMPACT_REMOVED_FEATURE_GROUPS
    return bundle


def fit_whitebox_family_workload_rf_benchmark_bundle(
    fit_frame: pd.DataFrame,
    hw_config: Path,
    *,
    policy: str | None = None,
    variant: str = "upper",
    op_thread_scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_key = str(variant).strip().lower()
    if variant_key not in WHITEBOX_WORKLOAD_BENCHMARK_LIMITS:
        raise ValueError(f"Unsupported benchmark variant: {variant}")
    return _fit_whitebox_rf_bundle(
        fit_frame,
        hw_config,
        policy=policy,
        route_name=benchmark_variant_route_name(variant_key),
        route_whitebox_mode=WHITEBOX_WORKLOAD_BENCHMARK_MODE,
        ml_input_source="whitebox_interval_modules_v4_rf_groups_workload_benchmark",
        ml_raw_sw_features_used=True,
        allow_workload_features=True,
        feature_limit=int(WHITEBOX_WORKLOAD_BENCHMARK_LIMITS[variant_key]),
        benchmark_variant=variant_key,
        op_thread_scale_config=op_thread_scale_config,
    )


def predict_whitebox_family_prior(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    override_hw_config: Path | dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    prepared = add_whitebox_family_prior_columns(frame, bundle, override_hw_config=override_hw_config)
    pred = pd.to_numeric(prepared["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=True)
    return prepared, np.clip(pred, a_min=0.0, a_max=None)


def _set_whitebox_rf_prediction_outputs(prepared: pd.DataFrame, pred: np.ndarray, bundle: dict[str, Any]) -> None:
    pred = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    prior = np.clip(
        pd.to_numeric(prepared["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        a_min=1e-6,
        a_max=None,
    )
    ratio = pred / prior
    for column in ("pred_mem_us", "pred_compute_us", "pred_sync_us", "pred_launch_us"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0).to_numpy(dtype=float) * ratio
    prepared["pred_total_us"] = pred
    prepared["pred_total_lower_us"] = pd.to_numeric(prepared["whitebox_lower_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float) * ratio
    prepared["pred_total_upper_us"] = pd.to_numeric(prepared["whitebox_upper_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float) * ratio
    prepared["model_sha256"] = str(bundle.get("model_sha256", ""))


def _predict_whitebox_family_rf_raw(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    override_hw_config: Path | dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    prepared = add_whitebox_family_prior_columns(frame, bundle["prior_bundle"], override_hw_config=override_hw_config)
    pred = pd.to_numeric(prepared["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=True)
    rf_group_bundles = dict(bundle.get("rf_group_bundles", bundle.get("family_bundles", {})))
    for group_name, model_bundle in rf_group_bundles.items():
        mask = prepared["whitebox_rf_group"].astype(str).eq(str(group_name))
        fallback_ops = RF_GROUP_OPTYPE_PRIOR_FALLBACK.get(str(group_name), set())
        if fallback_ops:
            mask = mask & ~prepared["op_type"].astype(str).isin(fallback_ops)
        if not mask.any():
            continue
        target_bundle = model_bundle if bool(model_bundle.get("enabled", False)) else None
        residual_source = f"rf_residual_{group_name}"
        if target_bundle is None:
            fallback_group = str(model_bundle.get("fallback_group", "") or "")
            fallback_bundle = rf_group_bundles.get(fallback_group, {})
            if fallback_group and bool(fallback_bundle.get("enabled", False)):
                target_bundle = fallback_bundle
                residual_source = f"rf_fallback_{fallback_group}"
        if target_bundle is None:
            continue
        family_pred = predict_family_feature_rf(target_bundle, prepared.loc[mask].copy())
        pred[mask.to_numpy(dtype=bool)] = np.asarray(family_pred, dtype=float)
        prepared.loc[mask, "whitebox_residual_source"] = residual_source

    if str(bundle.get("route", "")) in {WHITEBOX_GATHER_SLICE_RF_ROUTE, WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE}:
        alpha = float(bundle.get("gather_slice_post_correction_alpha", 0.0))
        if alpha > 0.0 and "gather_slice_underpredict_gap_us" in prepared.columns:
            correction_mask = (
                prepared["whitebox_rf_group"].astype(str).eq(RF_GROUP_INDEXED_COPY)
                & prepared["static_is_embedding_branch"].astype(str).eq("yes")
            )
            if correction_mask.any():
                correction = (
                    pd.to_numeric(prepared.loc[correction_mask, "gather_slice_underpredict_gap_us"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                pred[correction_mask.to_numpy(dtype=bool)] = pred[correction_mask.to_numpy(dtype=bool)] + alpha * correction
                prepared.loc[correction_mask, "whitebox_residual_source"] = (
                    prepared.loc[correction_mask, "whitebox_residual_source"].astype(str) + "_gather_slice_alpha"
                )

    pred = np.clip(pred, a_min=0.0, a_max=None)
    _set_whitebox_rf_prediction_outputs(prepared, pred, bundle)
    return prepared, pred


def predict_whitebox_family_simple_rf(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    override_hw_config: Path | dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    return _predict_whitebox_family_rf_raw(bundle, frame, override_hw_config=override_hw_config)


def predict_whitebox_family_physics_guarded_rf(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    override_hw_config: Path | dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if override_hw_config is None:
        prepared, pred = _predict_whitebox_family_rf_raw(bundle, frame)
        prepared["physics_guard_action"] = "baseline_no_override"
        prepared["physics_guard_applied"] = False
        return prepared, pred

    base_prepared, base_pred = _predict_whitebox_family_rf_raw(bundle, frame)
    target_prepared, target_pred = _predict_whitebox_family_rf_raw(bundle, frame, override_hw_config=override_hw_config)
    base_prior = pd.to_numeric(base_prepared["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    target_prior = pd.to_numeric(target_prepared["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_pred_arr = np.asarray(base_pred, dtype=float)
    target_pred_arr = np.asarray(target_pred, dtype=float)
    prior_delta = target_prior - base_prior
    rf_delta = target_pred_arr - base_pred_arr
    tol = float(PHYSICS_GUARD_DELTA_TOL_US)
    conflict = (np.abs(prior_delta) > tol) & (prior_delta * rf_delta < -tol)
    cap_mask = (
        (np.abs(prior_delta) > tol)
        & (prior_delta * rf_delta >= 0.0)
        & (np.abs(rf_delta) > float(PHYSICS_GUARD_DELTA_CAP_MULTIPLIER) * np.abs(prior_delta))
    )
    residual_frozen = target_prior + (base_pred_arr - base_prior)
    capped = base_pred_arr + np.sign(prior_delta) * float(PHYSICS_GUARD_DELTA_CAP_MULTIPLIER) * np.abs(prior_delta)
    guarded_pred = target_pred_arr.copy()
    guarded_pred[conflict] = residual_frozen[conflict]
    guarded_pred[cap_mask] = capped[cap_mask]
    guarded_pred = np.clip(guarded_pred, a_min=0.0, a_max=None)
    action = np.full(len(target_prepared), "none", dtype=object)
    action[conflict] = "direction_conflict_residual_frozen"
    action[cap_mask & ~conflict] = "delta_cap"
    target_prepared["physics_guard_action"] = action
    target_prepared["physics_guard_applied"] = conflict | cap_mask
    target_prepared["physics_guard_raw_pred_us"] = target_pred_arr
    target_prepared["physics_guard_base_pred_us"] = base_pred_arr
    target_prepared["physics_guard_base_prior_us"] = base_prior
    target_prepared["physics_guard_target_prior_us"] = target_prior
    target_prepared["physics_guard_prior_delta_us"] = prior_delta
    target_prepared["physics_guard_raw_delta_us"] = rf_delta
    target_prepared["physics_guard_final_delta_us"] = guarded_pred - base_pred_arr
    _set_whitebox_rf_prediction_outputs(target_prepared, guarded_pred, bundle)
    return target_prepared, guarded_pred
