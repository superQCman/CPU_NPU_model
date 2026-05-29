from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..feature_contract import TARGET_COLUMN  # noqa: E402
from ..feature_engineering import (  # noqa: E402
    _infer_gemm_mnk,
    _infer_gather_request_rows,
    _shape_entries,
    add_analytical_hardware_software_features,
    add_engineered_features,
    add_operator_hardware_context,
    load_hardware_features,
)
from .. import analytical_helpers as analytical_eval  # noqa: E402

try:  # noqa: E402
    from .contracts import (
        ANALYTICAL_FEATURE_COLUMNS,
        ANALYTICAL_FEATURE_DESCRIPTIONS,
        CALIBRATED_FAMILIES,
        DEFAULT_INPUT_CSV,
        DEFAULT_OUTPUT_DIR,
        FEATURE_DESCRIPTIONS,
        GENERIC_MEMORY_OP_TYPES,
        GENERIC_MIXED_OP_TYPES,
        HEAVY_FAMILIES,
        OP_AWARE_LIGHT_OP_TYPES,
        OP_CLASS_ORDER,
        OP_TYPE_CLASS_MAP,
        resolve_op_class,
    )
except ImportError:  # noqa: E402
    from contracts import (
        ANALYTICAL_FEATURE_COLUMNS,
        ANALYTICAL_FEATURE_DESCRIPTIONS,
        CALIBRATED_FAMILIES,
        DEFAULT_INPUT_CSV,
        DEFAULT_OUTPUT_DIR,
        FEATURE_DESCRIPTIONS,
        GENERIC_MEMORY_OP_TYPES,
        GENERIC_MIXED_OP_TYPES,
        HEAVY_FAMILIES,
        OP_AWARE_LIGHT_OP_TYPES,
        OP_CLASS_ORDER,
        OP_TYPE_CLASS_MAP,
        resolve_op_class,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build calibrated analytical features and export deployable analytical artifacts.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Input dataset_full.csv. Defaults to dataset_all_no_trace/dataset_full.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for calibrated analytical features.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Coordinate-descent passes used when fitting heavy-family parameters on the full dataset.",
    )
    parser.add_argument(
        "--hardware-profile",
        default="",
        help="Optional hardware profile YAML used when rebuilding hardware-derived features.",
    )
    return parser.parse_args()


def _drop_project_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[column for column in frame.columns if column.startswith(("feat_", "ana_", "hw_"))],
        errors="ignore",
    ).copy()


def rebuild_local_features(frame: pd.DataFrame, hardware_profile_path: Path | None = None) -> pd.DataFrame:
    work = _drop_project_derived_columns(frame)
    work = add_engineered_features(work)
    hardware_features = load_hardware_features(hardware_profile_path)
    work = add_operator_hardware_context(work, hardware_features=hardware_features)
    work = add_analytical_hardware_software_features(work)
    work["op_class"] = work.get("op_type", pd.Series("", index=work.index)).map(resolve_op_class)
    return work


def infer_batched_matmul_dims(
    input_entries: list[dict[str, Any]],
    output_entries: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    return analytical_eval.infer_batched_matmul_dims(input_entries, output_entries)


def _calibrated_family_for_row(row: pd.Series) -> str:
    op_type = str(row.get("op_type", "")).strip()
    if op_type in CALIBRATED_FAMILIES:
        return op_type
    return ""


def prepare_heavy_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    heavy_source = frame[frame["op_type"].astype(str).isin(CALIBRATED_FAMILIES)].copy()
    if heavy_source.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, row in heavy_source.iterrows():
        family = _calibrated_family_for_row(row)
        if not family:
            continue
        input_entries = _shape_entries(row.get("input_type_shape"))
        output_entries = _shape_entries(row.get("output_type_shape"))
        input_bytes_sum = float(sum(analytical_eval.entry_num_bytes(entry) for entry in input_entries))
        output_bytes_sum = float(sum(analytical_eval.entry_num_bytes(entry) for entry in output_entries))
        output_entry = output_entries[0] if output_entries else {}
        output_dims = [int(dim) for dim in output_entry.get("dims", [])]
        output_dtype_bytes = analytical_eval.dtype_size(str(output_entry.get("dtype", "float32")))

        gemm_m, gemm_n, gemm_k = _infer_gemm_mnk(input_entries, output_entries)
        batch_count, matmul_m, matmul_n, matmul_k = infer_batched_matmul_dims(input_entries, output_entries)

        configured_request_rows = max(analytical_eval.safe_float(row.get("feat_lookup_count"), 0.0), 0.0)
        request_rows = max(_infer_gather_request_rows(input_entries), 0.0)
        if request_rows <= 0.0:
            request_rows = configured_request_rows
        row_bytes = analytical_eval.safe_float(row.get("output_size"), 0.0) / max(request_rows, 1.0)
        cacheline = max(analytical_eval.safe_float(row.get("hw_cache_cacheline_bytes"), 64.0), 1.0)
        cachelines_per_row = math.ceil(row_bytes / cacheline) if row_bytes > 0.0 else 0

        gather_table_rows = 0.0
        if input_entries:
            dims0 = [int(dim) for dim in input_entries[0].get("dims", [])]
            if dims0:
                gather_table_rows = float(max(dims0[0], 0))
        gather_requested_rows_capped = min(request_rows, gather_table_rows) if gather_table_rows > 0.0 else request_rows
        gather_unique_rows_est = (
            gather_table_rows * (1.0 - math.exp(-request_rows / gather_table_rows))
            if gather_table_rows > 0.0
            else request_rows
        )
        gather_unique_rows_est = min(max(gather_unique_rows_est, 0.0), gather_requested_rows_capped)
        gather_src_unique_bytes = gather_unique_rows_est * row_bytes
        gather_src_fit_level = analytical_eval.fit_level(gather_src_unique_bytes, row)
        src_latency_us = analytical_eval.latency_from_level(gather_src_fit_level, row)

        transpose_prefix_blocks = 1.0
        if len(output_dims) > 1:
            prefix = 1
            for dim in output_dims[:-1]:
                prefix *= int(dim)
            transpose_prefix_blocks = float(prefix)

        transpose_suffix_block_bytes = float(output_dims[-1] * output_dtype_bytes) if output_dims else output_bytes_sum
        transpose_suffix_fit_level = analytical_eval.fit_level(transpose_suffix_block_bytes, row)
        transpose_stride_latency_us = analytical_eval.latency_from_level(transpose_suffix_fit_level, row)

        reduce_acc_bytes_per_thread = max(analytical_eval.safe_float(row.get("output_size"), 0.0), 0.0) / max(
            analytical_eval.safe_float(row.get("num_threads"), 1.0),
            1.0,
        )
        reduce_acc_fit_level = analytical_eval.fit_level(reduce_acc_bytes_per_thread, row)
        reduce_acc_latency_us = analytical_eval.latency_from_level(reduce_acc_fit_level, row)
        issue_slots = analytical_eval.issue_slots_per_us(row)
        add_lat_us = analytical_eval.add_latency_us(row)

        records.append(
            {
                "row_uid": row["row_uid"],
                "case_id": str(row.get("case_id", "")),
                "combo": str(row.get("combo", "")),
                "family": family,
                "actual_us": analytical_eval.safe_float(row.get(TARGET_COLUMN), 0.0),
                "num_threads": max(analytical_eval.safe_float(row.get("num_threads"), 1.0), 1.0),
                "output_size": max(analytical_eval.safe_float(row.get("output_size"), 0.0), 0.0),
                "activation_size": max(analytical_eval.safe_float(row.get("activation_size"), 0.0), 0.0),
                "parameter_size": max(analytical_eval.safe_float(row.get("parameter_size"), 0.0), 0.0),
                "feat_io_bytes_sum": max(analytical_eval.safe_float(row.get("feat_io_bytes_sum"), 0.0), 0.0),
                "output_elements": output_bytes_sum / max(float(output_dtype_bytes), 1.0),
                "feat_lookup_count": request_rows,
                "feat_reduction_axes_product": max(analytical_eval.safe_float(row.get("feat_reduction_axes_product"), 0.0), 0.0),
                "feat_reduction_work_items": max(analytical_eval.safe_float(row.get("feat_reduction_work_items"), 0.0), 0.0),
                "input_count": float(len(input_entries)),
                "input_bytes_sum": input_bytes_sum,
                "output_bytes_sum": output_bytes_sum,
                "copy_chunk_mean_bytes": input_bytes_sum / max(float(len(input_entries)), 1.0),
                "cacheline_bytes": cacheline,
                "issue_slots_per_us": issue_slots,
                "gather_row_bytes": row_bytes,
                "gather_cachelines_per_row": float(cachelines_per_row),
                "gather_src_latency_us": src_latency_us,
                "gather_table_rows": gather_table_rows,
                "gather_requested_rows_capped": gather_requested_rows_capped,
                "gather_unique_rows_est": gather_unique_rows_est,
                "gather_src_unique_bytes": gather_src_unique_bytes,
                "gather_src_fit_level": float(gather_src_fit_level),
                "gemm_m": float(gemm_m),
                "gemm_n": float(gemm_n),
                "gemm_k": float(gemm_k),
                "matmul_batch_count": float(batch_count),
                "matmul_m": float(matmul_m),
                "matmul_n": float(matmul_n),
                "matmul_k": float(matmul_k),
                "transpose_prefix_blocks": transpose_prefix_blocks,
                "transpose_stride_latency_us": transpose_stride_latency_us,
                "reduce_acc_bytes_per_thread": reduce_acc_bytes_per_thread,
                "reduce_acc_fit_level": float(reduce_acc_fit_level),
                "reduce_acc_latency_us": reduce_acc_latency_us,
                "bw_peak_bytes_per_us": max(analytical_eval.safe_float(row.get("hw_memory_bandwidth_gb_s_total"), 100.0), 1e-6) * 1e3,
                "hw_core_cpu_clock": max(analytical_eval.safe_float(row.get("hw_core_cpu_clock"), 2.6), 1e-6),
                "hw_cache_l1d_response_latency_cycles": max(
                    analytical_eval.safe_float(row.get("hw_cache_l1d_response_latency_cycles"), 1.0),
                    1e-6,
                ),
                "peak_add_ops_per_us": analytical_eval.peak_add_ops_per_us(row),
                "peak_fma_ops_per_us": analytical_eval.peak_fma_ops_per_us(row),
            }
        )
    return pd.DataFrame(records)


def _concat_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chunk = frame["copy_chunk_mean_bytes"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    stream = (frame["input_bytes_sum"] + frame["output_size"]).to_numpy(dtype=float)
    input_count = frame["input_count"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_copy_inf"], 1e-6)
    bw_eff = analytical_eval.effective_bandwidth(chunk, bw_inf, params["tau_copy_start"])
    t_stream = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    cacheline = frame["cacheline_bytes"].to_numpy(dtype=float)
    issue_slots = frame["issue_slots_per_us"].to_numpy(dtype=float)
    t_issue = (stream / np.clip(cacheline, a_min=1.0, a_max=None)) / np.clip(issue_slots, a_min=1e-6, a_max=None)
    mem_us = np.maximum(t_stream, t_issue)
    overhead_us = input_count * max(params["tau_dispatch"], 0.0)
    compute_us = np.zeros(len(frame), dtype=float)
    total_us = mem_us + overhead_us
    return total_us, mem_us, compute_us, overhead_us


def _reduce_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inner = frame["feat_reduction_axes_product"].to_numpy(dtype=float)
    stream = (frame["activation_size"] + frame["output_size"]).to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    peak_add = frame["peak_add_ops_per_us"].to_numpy(dtype=float)
    add_ops = frame["feat_reduction_work_items"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_copy_inf"], 1e-6) * max(params["kappa_reduce"], 1e-6)
    bw_eff = analytical_eval.effective_bandwidth(inner, bw_inf, params["tau_reduce_start"])
    mem_us = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    compute_us = add_ops / np.clip(peak_add, a_min=1e-6, a_max=None)
    overhead_us = np.zeros(len(frame), dtype=float)
    total_us = np.maximum(mem_us, compute_us)
    return total_us, mem_us, compute_us, overhead_us


def _gather_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_bytes = frame["gather_row_bytes"].to_numpy(dtype=float)
    request_rows = frame["feat_lookup_count"].to_numpy(dtype=float)
    stream = (2.0 * frame["output_size"] + 8.0 * frame["feat_lookup_count"]).to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    cachelines_per_row = frame["gather_cachelines_per_row"].to_numpy(dtype=float)
    threads = frame["num_threads"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_gather_inf"], 1e-6)
    bw_eff = analytical_eval.effective_bandwidth(row_bytes, bw_inf, params["tau_gather_row_start"])
    t_bw = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    src_rows = request_rows
    lat_src_us = frame["gather_src_latency_us"].to_numpy(dtype=float)
    t_src = src_rows * cachelines_per_row * lat_src_us / np.clip(
        threads * max(params["m_gather"], 1e-6),
        a_min=1e-6,
        a_max=None,
    )
    cpu_clock_ghz = np.clip(
        pd.to_numeric(frame.get("hw_core_cpu_clock", pd.Series(2.6, index=frame.index)), errors="coerce").fillna(2.6).to_numpy(dtype=float),
        a_min=1e-6,
        a_max=None,
    )
    l1_latency_cycles = np.clip(
        pd.to_numeric(
            frame.get("hw_cache_l1d_response_latency_cycles", pd.Series(1.0, index=frame.index)),
            errors="coerce",
        ).fillna(1.0).to_numpy(dtype=float),
        a_min=1e-6,
        a_max=None,
    )
    l1_latency_us = l1_latency_cycles / cpu_clock_ghz / 1000.0
    tau_floor = 8.0 * np.power(
        np.clip(lat_src_us / np.clip(l1_latency_us, a_min=1e-9, a_max=None), a_min=1.0, a_max=None),
        0.75,
    )
    mem_us = np.maximum.reduce([t_bw, t_src, tau_floor])
    total_us = mem_us.copy()
    compute_us = np.zeros(len(frame), dtype=float)
    overhead_us = np.zeros(len(frame), dtype=float)
    return total_us, mem_us, compute_us, overhead_us


def _gemm_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m_dim = frame["gemm_m"].to_numpy(dtype=float)
    n_dim = frame["gemm_n"].to_numpy(dtype=float)
    k_dim = frame["gemm_k"].to_numpy(dtype=float)
    flops = 2.0 * m_dim * n_dim * k_dim
    mem_bytes = frame["feat_io_bytes_sum"].to_numpy(dtype=float)
    peak_fma = frame["peak_fma_ops_per_us"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    rho_eff = (
        max(params["rho_fma_inf"], 1e-6)
        * m_dim / np.clip(m_dim + max(params["M50"], 0.0), a_min=1e-6, a_max=None)
        * n_dim / np.clip(n_dim + max(params["N50"], 0.0), a_min=1e-6, a_max=None)
        * k_dim / np.clip(k_dim + max(params["K50"], 0.0), a_min=1e-6, a_max=None)
    )
    compute_us = flops / np.clip(peak_fma * rho_eff, a_min=1e-6, a_max=None)
    mem_us = mem_bytes / np.clip(bw_peak, a_min=1e-6, a_max=None)
    total_us = np.maximum(mem_us, compute_us)
    overhead_us = np.zeros(len(frame), dtype=float)
    return total_us, mem_us, compute_us, overhead_us


def _matmul_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_count = frame["matmul_batch_count"].to_numpy(dtype=float)
    m_dim = frame["matmul_m"].to_numpy(dtype=float)
    n_dim = frame["matmul_n"].to_numpy(dtype=float)
    k_dim = frame["matmul_k"].to_numpy(dtype=float)
    flops = 2.0 * batch_count * m_dim * n_dim * k_dim
    peak_fma = frame["peak_fma_ops_per_us"].to_numpy(dtype=float)
    threads = frame["num_threads"].to_numpy(dtype=float)
    occ_ref = max(params["occ_ref"], 1e-6)
    rho_eff = (
        max(params["rho_tiny_inf"], 1e-6)
        * np.minimum(m_dim / occ_ref, 1.0)
        * np.minimum(n_dim / occ_ref, 1.0)
        * k_dim / np.clip(k_dim + max(params["K50_tiny"], 0.0), a_min=1e-6, a_max=None)
    )
    compute_us = flops / np.clip(peak_fma * rho_eff, a_min=1e-6, a_max=None)
    overhead_us = np.ceil(batch_count / np.clip(threads, a_min=1.0, a_max=None)) * max(params["tau_micro"], 0.0)
    mem_us = np.zeros(len(frame), dtype=float)
    total_us = compute_us + overhead_us
    return total_us, mem_us, compute_us, overhead_us


def _transpose_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_bytes = frame["output_size"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    prefix_blocks = frame["transpose_prefix_blocks"].to_numpy(dtype=float)
    threads = frame["num_threads"].to_numpy(dtype=float)
    lat_us = frame["transpose_stride_latency_us"].to_numpy(dtype=float)
    copy_us = 2.0 * out_bytes / np.clip(
        bw_peak * max(params["rho_copy_inf"], 1e-6),
        a_min=1e-6,
        a_max=None,
    )
    stride_us = prefix_blocks * lat_us / np.clip(
        np.power(np.clip(threads, a_min=1.0, a_max=None), max(params["eta_stride"], 0.0))
        * max(params["m_stride"], 1e-6),
        a_min=1e-6,
        a_max=None,
    )
    mem_us = copy_us + stride_us
    total_us = mem_us.copy()
    compute_us = np.zeros(len(frame), dtype=float)
    overhead_us = np.zeros(len(frame), dtype=float)
    return total_us, mem_us, compute_us, overhead_us


def _relu_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mem_us = analytical_eval.predict_relu(frame, params)
    total_us = mem_us.copy()
    compute_us = np.zeros(len(frame), dtype=float)
    overhead_us = np.zeros(len(frame), dtype=float)
    return total_us, mem_us, compute_us, overhead_us


def _add_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stream = frame["feat_io_bytes_sum"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_add_inf"], 1e-6)
    bw_eff = analytical_eval.effective_bandwidth(stream, bw_inf, params["tau_add_start"])
    mem_us = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    overhead_us = np.full(len(frame), max(params["tau_add"], 0.0), dtype=float)
    compute_us = np.zeros(len(frame), dtype=float)
    total_us = mem_us + overhead_us
    return total_us, mem_us, compute_us, overhead_us


def _mul_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stream = frame["feat_io_bytes_sum"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_mul_inf"], 1e-6)
    bw_eff = analytical_eval.effective_bandwidth(stream, bw_inf, params["tau_mul_start"])
    mem_us = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    overhead_us = np.full(len(frame), max(params["tau_mul"], 0.0), dtype=float)
    compute_us = np.zeros(len(frame), dtype=float)
    total_us = mem_us + overhead_us
    return total_us, mem_us, compute_us, overhead_us


def _sigmoid_components(frame: pd.DataFrame, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stream = frame["feat_io_bytes_sum"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    output_elements = frame["output_elements"].to_numpy(dtype=float)
    peak_add = frame["peak_add_ops_per_us"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_sigmoid_inf"], 1e-6)
    bw_eff = analytical_eval.effective_bandwidth(stream, bw_inf, params["tau_sigmoid_start"])
    mem_us = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    compute_us = output_elements / np.clip(
        peak_add * max(params["rho_sigmoid_compute"], 1e-9),
        a_min=1e-6,
        a_max=None,
    )
    overhead_us = np.full(len(frame), max(params["tau_sigmoid"], 0.0), dtype=float)
    total_us = overhead_us + np.maximum(mem_us, compute_us)
    return total_us, mem_us, compute_us, overhead_us


def predict_heavy_family_components(
    frame: pd.DataFrame,
    family: str,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if family == "Concat":
        return _concat_components(frame, params)
    if family == "ReduceSum":
        return _reduce_components(frame, params)
    if family == "Gather":
        return _gather_components(frame, params)
    if family == "Gemm":
        return _gemm_components(frame, params)
    if family == "MatMul":
        return _matmul_components(frame, params)
    if family == "Transpose":
        return _transpose_components(frame, params)
    if family == "Relu":
        return _relu_components(frame, params)
    if family == "Add":
        return _add_components(frame, params)
    if family == "Mul":
        return _mul_components(frame, params)
    if family == "Sigmoid":
        return _sigmoid_components(frame, params)
    raise KeyError(family)


def _peak_add_ops_per_us_vector(frame: pd.DataFrame) -> np.ndarray:
    throughput = pd.to_numeric(
        frame.get("hw_instruction_fp_throughput_per_cycle_vector_sp_add", 2.0),
        errors="coerce",
    ).fillna(2.0)
    lanes = pd.to_numeric(frame.get("hw_instruction_simd_width_bits", 128.0), errors="coerce").fillna(128.0) / 32.0
    cpu_clock = pd.to_numeric(frame.get("hw_core_cpu_clock", 2.6), errors="coerce").fillna(2.6)
    active_cores = pd.to_numeric(frame.get("hw_core_active_cores", frame.get("num_threads", 1.0)), errors="coerce").fillna(1.0)
    return (throughput.clip(lower=1e-6) * lanes.clip(lower=1.0) * cpu_clock.clip(lower=1e-6) * 1e3 * active_cores.clip(lower=1.0)).to_numpy(dtype=float)


def analytical_superfamily_name(family: str) -> str:
    return "gemm_like" if str(family) in {"Gemm", "MatMul"} else str(family)


def analytical_regime_name(family: str) -> str:
    if str(family) == "Gemm":
        return "large_gemm_saturation"
    if str(family) == "MatMul":
        return "tiny_batched_occ"
    return "default"


def add_calibrated_analytical_columns(
    rebuilt_df: pd.DataFrame,
    heavy_prepared_df: pd.DataFrame,
    params: dict[str, float],
) -> pd.DataFrame:
    out = rebuilt_df.copy()
    out["ana_calib_total_us"] = 0.0
    out["ana_calib_mem_us"] = 0.0
    out["ana_calib_compute_us"] = 0.0
    out["ana_calib_overhead_us"] = 0.0
    out["ana_calib_family"] = "unassigned"
    out["ana_calib_superfamily"] = "unassigned"
    out["ana_calib_regime"] = "unassigned"

    if not heavy_prepared_df.empty:
        heavy_parts: list[pd.DataFrame] = []
        for family in CALIBRATED_FAMILIES:
            family_df = heavy_prepared_df[heavy_prepared_df["family"] == family].copy()
            if family_df.empty:
                continue
            total_us, mem_us, compute_us, overhead_us = predict_heavy_family_components(family_df, family, params)
            heavy_parts.append(
                pd.DataFrame(
                    {
                        "row_uid": family_df["row_uid"].astype(str),
                        "ana_calib_total_us": total_us,
                        "ana_calib_mem_us": mem_us,
                        "ana_calib_compute_us": compute_us,
                        "ana_calib_overhead_us": overhead_us,
                        "ana_calib_family": family,
                    }
                )
            )
        if heavy_parts:
            heavy_features = pd.concat(heavy_parts, ignore_index=True)
            out = out.merge(heavy_features, on="row_uid", how="left", suffixes=("", "__heavy"))
            for column in [
                "ana_calib_total_us",
                "ana_calib_mem_us",
                "ana_calib_compute_us",
                "ana_calib_overhead_us",
                "ana_calib_family",
            ]:
                heavy_column = f"{column}__heavy"
                if heavy_column in out.columns:
                    if column == "ana_calib_family":
                        out[column] = np.where(out[heavy_column].notna(), out[heavy_column], out[column])
                    else:
                        out[column] = pd.to_numeric(out[heavy_column], errors="coerce").fillna(out[column])
                    out = out.drop(columns=[heavy_column])

    memory_mask = out["op_type"].astype(str).isin(GENERIC_MEMORY_OP_TYPES) & out["ana_calib_family"].eq("unassigned")
    out.loc[memory_mask, "ana_calib_total_us"] = pd.to_numeric(
        out.loc[memory_mask, "ana_mem_bw_time_us"],
        errors="coerce",
    ).fillna(0.0)
    out.loc[memory_mask, "ana_calib_mem_us"] = out.loc[memory_mask, "ana_calib_total_us"]
    out.loc[memory_mask, "ana_calib_compute_us"] = 0.0
    out.loc[memory_mask, "ana_calib_overhead_us"] = 0.0
    out.loc[memory_mask, "ana_calib_family"] = "generic_memory"

    mixed_mask = out["op_type"].astype(str).isin(GENERIC_MIXED_OP_TYPES) & out["ana_calib_family"].eq("unassigned")
    if mixed_mask.any():
        mem_us = pd.to_numeric(out.loc[mixed_mask, "ana_mem_bw_time_us"], errors="coerce").fillna(0.0)
        compute_ops = pd.to_numeric(out.loc[mixed_mask, "ana_compute_ops"], errors="coerce").fillna(0.0)
        peak_add = _peak_add_ops_per_us_vector(out.loc[mixed_mask])
        compute_us = compute_ops.to_numpy(dtype=float) / np.clip(peak_add, a_min=1e-6, a_max=None)
        total_us = np.maximum(mem_us.to_numpy(dtype=float), compute_us)
        out.loc[mixed_mask, "ana_calib_total_us"] = total_us
        out.loc[mixed_mask, "ana_calib_mem_us"] = mem_us.to_numpy(dtype=float)
        out.loc[mixed_mask, "ana_calib_compute_us"] = compute_us
        out.loc[mixed_mask, "ana_calib_overhead_us"] = 0.0
        out.loc[mixed_mask, "ana_calib_family"] = "generic_mixed"

    unassigned_mask = out["ana_calib_family"].eq("unassigned")
    if unassigned_mask.any():
        fallback_total = pd.to_numeric(out.loc[unassigned_mask, "ana_mem_bw_time_us"], errors="coerce").fillna(0.0)
        out.loc[unassigned_mask, "ana_calib_total_us"] = fallback_total
        out.loc[unassigned_mask, "ana_calib_mem_us"] = fallback_total
        out.loc[unassigned_mask, "ana_calib_compute_us"] = 0.0
        out.loc[unassigned_mask, "ana_calib_overhead_us"] = 0.0
        out.loc[unassigned_mask, "ana_calib_family"] = "generic_fallback"

    out["ana_calib_superfamily"] = out["ana_calib_family"].map(analytical_superfamily_name)
    out["ana_calib_regime"] = out["ana_calib_family"].map(analytical_regime_name)

    for column in ANALYTICAL_FEATURE_COLUMNS:
        if column in {"ana_calib_family", "ana_calib_superfamily", "ana_calib_regime"}:
            out[column] = out[column].fillna("generic_fallback").astype(str)
        elif column == "op_class":
            out[column] = out[column].fillna("mixed_balanced").astype(str)
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    return out


def build_full_feature_artifacts(
    input_csv: Path,
    output_dir: Path,
    *,
    passes: int,
    hardware_profile_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(input_csv, low_memory=False)
    rebuilt_df = rebuild_local_features(raw_df, hardware_profile_path=hardware_profile_path)
    heavy_prepared_df = prepare_heavy_prediction_frame(rebuilt_df)
    full_params = analytical_eval.calibrate_params(heavy_prepared_df, passes=passes)
    feature_df = add_calibrated_analytical_columns(rebuilt_df, heavy_prepared_df, full_params)

    feature_columns = [
        "row_uid",
        "case_id",
        "combo",
        "split",
        "op_type",
        "op_class",
        "ana_calib_family",
        "ana_calib_superfamily",
        "ana_calib_regime",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ana_calib_compute_us",
        "ana_calib_overhead_us",
        TARGET_COLUMN,
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
    ]
    feature_export = feature_df[[column for column in feature_columns if column in feature_df.columns]].copy()
    feature_csv = output_dir / "analytical_features_full.csv"
    feature_export.to_csv(feature_csv, index=False)

    deploy_dir = output_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    params_json = deploy_dir / "full_data_parameters.json"
    with params_json.open("w", encoding="utf-8") as handle:
        json.dump(full_params, handle, indent=2, ensure_ascii=False)

    manifest = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "hardware_profile_path": "" if hardware_profile_path is None else str(hardware_profile_path),
        "heavy_families": list(HEAVY_FAMILIES),
        "calibrated_families": list(CALIBRATED_FAMILIES),
        "op_aware_light_families": list(OP_AWARE_LIGHT_OP_TYPES),
        "generic_proxy_op_types": {
            "memory_pure": list(GENERIC_MEMORY_OP_TYPES),
            "mixed_balanced": list(GENERIC_MIXED_OP_TYPES),
        },
        "op_type_class_map": dict(OP_TYPE_CLASS_MAP),
        "op_class_order": list(OP_CLASS_ORDER),
        "analytical_feature_columns": list(ANALYTICAL_FEATURE_COLUMNS),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "analytical_feature_descriptions": ANALYTICAL_FEATURE_DESCRIPTIONS,
        "artifacts": {
            "deploy_parameters_json": str(params_json),
            "feature_csv": str(feature_csv),
        },
        "row_count": int(len(feature_export)),
        "heavy_row_count": int(len(heavy_prepared_df)),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    args = parse_args()
    hardware_profile_path = Path(args.hardware_profile) if args.hardware_profile else None
    manifest = build_full_feature_artifacts(
        Path(args.input_csv),
        Path(args.output_dir),
        passes=args.passes,
        hardware_profile_path=hardware_profile_path,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
