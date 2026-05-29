from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .feature_engineering import (
    DTYPE_SIZES,
    _infer_gemm_mnk,
    _infer_gather_request_rows,
    _shape_entries,
    add_operator_hardware_context,
    load_hardware_features,
)


HEAVY_FAMILY_ORDER = ["Gather", "ReduceSum", "Gemm", "MatMul", "Transpose", "Concat"]
LIGHT_FAMILY_ORDER = ["Relu", "Add", "Mul", "Sigmoid"]
FAMILY_ORDER = HEAVY_FAMILY_ORDER + LIGHT_FAMILY_ORDER
COPY_FAMILIES = ["Concat", "ReduceSum", "Transpose"]


def dtype_size(dtype_name: str | None) -> int:
    text = "" if dtype_name is None else str(dtype_name).strip().lower()
    return int(DTYPE_SIZES.get(text, 4))


def entry_num_elements(entry: dict[str, Any]) -> int:
    dims = [int(dim) for dim in entry.get("dims", [])]
    if not dims:
        return 0
    product = 1
    for dim in dims:
        product *= int(dim)
    return int(product)


def entry_num_bytes(entry: dict[str, Any]) -> float:
    return float(entry_num_elements(entry) * dtype_size(str(entry.get("dtype", ""))))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    target = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.clip(target, a_min=1e-9, a_max=None)
    return float(np.mean(np.abs(pred - target) / denominator))


def duration_weighted_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    target = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    numerator = float(np.sum(np.abs(pred - target)))
    denominator = float(np.sum(np.clip(target, a_min=1e-9, a_max=None)))
    return numerator / denominator if denominator > 0.0 else 0.0


def fit_level(working_set_bytes: float, row: pd.Series) -> int:
    ws = max(safe_float(working_set_bytes, 0.0), 0.0)
    l1_bytes = max(safe_float(row.get("hw_cache_l1d_active_bytes"), 0.0), 0.0)
    l2_bytes = max(safe_float(row.get("hw_cache_l2_active_bytes"), 0.0), 0.0)
    l3_bytes = max(safe_float(row.get("hw_cache_l3_active_bytes"), 0.0), 0.0)
    if ws <= max(l1_bytes, 1.0):
        return 1
    if ws <= max(l2_bytes, 1.0):
        return 2
    if ws <= max(l3_bytes, 1.0):
        return 3
    return 4


def latency_from_level(level: int, row: pd.Series) -> float:
    cpu_clock_ghz = max(safe_float(row.get("hw_core_cpu_clock"), 2.6), 1e-6)
    l1_us = safe_float(row.get("hw_cache_l1d_response_latency_cycles"), 1.0) / cpu_clock_ghz / 1000.0
    l2_us = safe_float(row.get("hw_cache_l2_response_latency_cycles"), 10.0) / cpu_clock_ghz / 1000.0
    l3_us = safe_float(row.get("hw_cache_l3_per_die_response_latency_cycles"), 20.0) / cpu_clock_ghz / 1000.0
    mem_us = safe_float(row.get("hw_memory_local_mem_delay_ns"), 90.0) / 1000.0
    if int(level) == 1:
        return l1_us
    if int(level) == 2:
        return l2_us
    if int(level) == 3:
        return l3_us
    return mem_us


def fit_latency_us(working_set_bytes: float, row: pd.Series) -> float:
    return latency_from_level(fit_level(working_set_bytes, row), row)


def issue_slots_per_us(row: pd.Series) -> float:
    widths = [
        safe_float(row.get("hw_pipeline_fetch_width"), 4.0),
        safe_float(row.get("hw_pipeline_decode_width"), 4.0),
        safe_float(row.get("hw_pipeline_rename_width"), 4.0),
        safe_float(row.get("hw_pipeline_dispatch_width"), 4.0),
        safe_float(row.get("hw_pipeline_issue_width"), 4.0),
        safe_float(row.get("hw_pipeline_commit_width"), 4.0),
    ]
    pipeline_width = max(min(widths), 1.0)
    cpu_clock = max(safe_float(row.get("hw_core_cpu_clock"), 2.6), 1e-6)
    active_cores = max(
        min(safe_float(row.get("num_threads"), 1.0), safe_float(row.get("hw_core_total_cores"), 24.0)),
        1.0,
    )
    return pipeline_width * cpu_clock * 1e3 * active_cores


def add_latency_us(row: pd.Series) -> float:
    cpu_clock = max(safe_float(row.get("hw_core_cpu_clock"), 2.6), 1e-6)
    return safe_float(row.get("hw_instruction_fp_latency_cycles_vector_sp_add"), 3.0) / cpu_clock / 1000.0


def peak_add_ops_per_us(row: pd.Series) -> float:
    throughput = max(safe_float(row.get("hw_instruction_fp_throughput_per_cycle_vector_sp_add"), 2.0), 1e-6)
    lanes = max(safe_float(row.get("hw_instruction_simd_width_bits"), 128.0) / 32.0, 1.0)
    cpu_clock = max(safe_float(row.get("hw_core_cpu_clock"), 2.6), 1e-6)
    active_cores = max(
        min(safe_float(row.get("num_threads"), 1.0), safe_float(row.get("hw_core_total_cores"), 24.0)),
        1.0,
    )
    return throughput * lanes * cpu_clock * 1e3 * active_cores


def peak_fma_ops_per_us(row: pd.Series) -> float:
    throughput = max(safe_float(row.get("hw_instruction_fp_throughput_per_cycle_vector_sp_fma"), 2.0), 1e-6)
    lanes = max(safe_float(row.get("hw_instruction_simd_width_bits"), 128.0) / 32.0, 1.0)
    cpu_clock = max(safe_float(row.get("hw_core_cpu_clock"), 2.6), 1e-6)
    active_cores = max(
        min(safe_float(row.get("num_threads"), 1.0), safe_float(row.get("hw_core_total_cores"), 24.0)),
        1.0,
    )
    return throughput * lanes * 2.0 * cpu_clock * 1e3 * active_cores


def analytical_family_name(row: pd.Series) -> str:
    op_type = str(row.get("op_type", ""))
    node_name = str(row.get("node_name", ""))
    if op_type == "Gather" and safe_float(row.get("output_size"), 0.0) > 1e8:
        return "Gather"
    if op_type == "ReduceSum" and safe_float(row.get("activation_size"), 0.0) > 1e8:
        return "ReduceSum"
    if op_type == "Gemm" and safe_float(row.get("label_operator_actual_dur_us"), 0.0) > 1000.0:
        return "Gemm"
    if node_name == "/MatMul":
        return "MatMul"
    if node_name == "/Transpose":
        return "Transpose"
    if node_name == "/Concat":
        return "Concat"
    if op_type in LIGHT_FAMILY_ORDER:
        return op_type
    return ""


def infer_batched_matmul_dims(
    input_entries: list[dict[str, Any]],
    output_entries: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    if len(input_entries) < 2 or not output_entries:
        return 0.0, 0.0, 0.0, 0.0
    a_dims = [int(dim) for dim in input_entries[0].get("dims", [])]
    b_dims = [int(dim) for dim in input_entries[1].get("dims", [])]
    c_dims = [int(dim) for dim in output_entries[0].get("dims", [])]
    if len(a_dims) < 3 or len(b_dims) < 3 or len(c_dims) < 3:
        return 0.0, 0.0, 0.0, 0.0
    batch_count = float(c_dims[0])
    m_dim = float(c_dims[-2])
    n_dim = float(c_dims[-1])
    k_dim = float(a_dims[-1])
    if len(b_dims) >= 2 and b_dims[-1] == int(n_dim):
        k_dim = float(b_dims[-2])
    return batch_count, m_dim, n_dim, k_dim


def default_params() -> dict[str, float]:
    return {
        "rho_copy_inf": 0.18,
        "tau_copy_start": 0.0,
        "tau_dispatch": 20.0,
        "kappa_reduce": 0.5,
        "tau_reduce_start": 0.0,
        "rho_gather_inf": 0.12,
        "tau_gather_row_start": 128.0 / (100.0 * 1e3 * 0.12),
        "m_gather": 4.0,
        "rho_fma_inf": 0.55,
        "M50": 32.0,
        "N50": 0.0,
        "K50": 64.0,
        "occ_ref": 16.0,
        "rho_tiny_inf": 0.30,
        "K50_tiny": 0.0,
        "tau_micro": 0.0,
        "m_stride": 8.0,
        "eta_stride": 0.0,
        "rho_relu_inf": 0.02,
        "tau_relu_start": 0.0,
        "rho_add_inf": 0.05,
        "tau_add_start": 0.0,
        "tau_add": 8.0,
        "rho_mul_inf": 0.04,
        "tau_mul_start": 0.0,
        "tau_mul": 12.0,
        "rho_sigmoid_inf": 0.04,
        "tau_sigmoid_start": 0.0,
        "tau_sigmoid": 20.0,
        "rho_sigmoid_compute": 1e-4,
    }


PARAM_GRID: dict[str, list[float]] = {
    "rho_copy_inf": [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30],
    "tau_copy_start": [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032],
    "tau_dispatch": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0],
    "kappa_reduce": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00],
    "tau_reduce_start": [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016],
    "rho_gather_inf": [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26],
    "tau_gather_row_start": [0.0, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024, 0.032],
    "m_gather": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
    "rho_fma_inf": [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80],
    "M50": [0.0, 8.0, 16.0, 32.0, 64.0, 128.0],
    "N50": [0.0, 8.0, 16.0, 32.0, 64.0],
    "K50": [0.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    "occ_ref": [4.0, 8.0, 12.0, 16.0, 24.0, 32.0],
    "rho_tiny_inf": [0.08, 0.12, 0.18, 0.24, 0.30, 0.40, 0.50, 0.60],
    "K50_tiny": [0.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    "tau_micro": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    "m_stride": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.50, 0.75, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0],
    "eta_stride": [0.0, 0.25, 0.5, 0.75, 1.0],
    "rho_relu_inf": [0.008, 0.010, 0.012, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080, 0.10, 0.12, 0.15, 0.18, 0.20, 0.24, 0.30],
    "tau_relu_start": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "rho_add_inf": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18],
    "tau_add_start": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
    "tau_add": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0],
    "rho_mul_inf": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18],
    "tau_mul_start": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
    "tau_mul": [0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0],
    "rho_sigmoid_inf": [0.008, 0.010, 0.012, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080],
    "tau_sigmoid_start": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
    "tau_sigmoid": [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0],
    "rho_sigmoid_compute": [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 1.5e-2, 2e-2],
}


def effective_bandwidth(size_bytes: np.ndarray, bw_inf_bytes_per_us: np.ndarray, tau_start_us: float) -> np.ndarray:
    size = np.clip(size_bytes.astype(float), a_min=0.0, a_max=None)
    bw_inf = np.clip(bw_inf_bytes_per_us.astype(float), a_min=1e-6, a_max=None)
    tau = max(float(tau_start_us), 0.0)
    return bw_inf * size / np.clip(bw_inf * tau + size, a_min=1e-6, a_max=None)


def predict_concat(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    chunk = frame["copy_chunk_mean_bytes"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    stream = (frame["input_bytes_sum"] + frame["output_size"]).to_numpy(dtype=float)
    input_count = frame["input_count"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_copy_inf"], 1e-6)
    bw_eff = effective_bandwidth(chunk, bw_inf, params["tau_copy_start"])
    t_stream = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    cacheline = frame["cacheline_bytes"].to_numpy(dtype=float)
    issue_slots = frame["issue_slots_per_us"].to_numpy(dtype=float)
    t_issue = (stream / np.clip(cacheline, a_min=1.0, a_max=None)) / np.clip(issue_slots, a_min=1e-6, a_max=None)
    mem_us = np.maximum(t_stream, t_issue)
    return mem_us + input_count * max(params["tau_dispatch"], 0.0)


def predict_reduce(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    inner = frame["feat_reduction_axes_product"].to_numpy(dtype=float)
    stream = (frame["activation_size"] + frame["output_size"]).to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    peak_add = frame["peak_add_ops_per_us"].to_numpy(dtype=float)
    add_ops = frame["feat_reduction_work_items"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_copy_inf"], 1e-6) * max(params["kappa_reduce"], 1e-6)
    bw_eff = effective_bandwidth(inner, bw_inf, params["tau_reduce_start"])
    mem_us = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    compute_us = add_ops / np.clip(peak_add, a_min=1e-6, a_max=None)
    return np.maximum(mem_us, compute_us)


def predict_gather(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    row_bytes = frame["gather_row_bytes"].to_numpy(dtype=float)
    request_rows = frame["feat_lookup_count"].to_numpy(dtype=float)
    stream = (2.0 * frame["output_size"] + 8.0 * frame["feat_lookup_count"]).to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    cachelines_per_row = frame["gather_cachelines_per_row"].to_numpy(dtype=float)
    threads = frame["num_threads"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_gather_inf"], 1e-6)
    bw_eff = effective_bandwidth(row_bytes, bw_inf, params["tau_gather_row_start"])
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
        pd.to_numeric(frame.get("hw_cache_l1d_response_latency_cycles", pd.Series(1.0, index=frame.index)), errors="coerce").fillna(1.0).to_numpy(dtype=float),
        a_min=1e-6,
        a_max=None,
    )
    l1_latency_us = l1_latency_cycles / cpu_clock_ghz / 1000.0
    tau_floor = 8.0 * np.power(
        np.clip(lat_src_us / np.clip(l1_latency_us, a_min=1e-9, a_max=None), a_min=1.0, a_max=None),
        0.75,
    )
    return np.maximum.reduce([t_bw, t_src, tau_floor])


def predict_gemm(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
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
    t_comp = flops / np.clip(peak_fma * rho_eff, a_min=1e-6, a_max=None)
    t_mem = mem_bytes / np.clip(bw_peak, a_min=1e-6, a_max=None)
    return np.maximum(t_comp, t_mem)


def predict_matmul(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
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
    t_comp = flops / np.clip(peak_fma * rho_eff, a_min=1e-6, a_max=None)
    t_launch = np.ceil(batch_count / np.clip(threads, a_min=1.0, a_max=None)) * max(params["tau_micro"], 0.0)
    return t_comp + t_launch


def predict_transpose(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    out_bytes = frame["output_size"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    prefix_blocks = frame["transpose_prefix_blocks"].to_numpy(dtype=float)
    threads = frame["num_threads"].to_numpy(dtype=float)
    lat_us = frame["transpose_stride_latency_us"].to_numpy(dtype=float)
    copy_us = 2.0 * out_bytes / np.clip(bw_peak * max(params["rho_copy_inf"], 1e-6), a_min=1e-6, a_max=None)
    stride_us = prefix_blocks * lat_us / np.clip(
        np.power(np.clip(threads, a_min=1.0, a_max=None), max(params["eta_stride"], 0.0)) * max(params["m_stride"], 1e-6),
        a_min=1e-6,
        a_max=None,
    )
    return copy_us + stride_us


def predict_relu(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    stream = frame["feat_io_bytes_sum"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_relu_inf"], 1e-6)
    bw_eff = effective_bandwidth(stream, bw_inf, params["tau_relu_start"])
    return stream / np.clip(bw_eff, a_min=1e-6, a_max=None)


def predict_add(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    stream = frame["feat_io_bytes_sum"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_add_inf"], 1e-6)
    bw_eff = effective_bandwidth(stream, bw_inf, params["tau_add_start"])
    return max(params["tau_add"], 0.0) + stream / np.clip(bw_eff, a_min=1e-6, a_max=None)


def predict_mul(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    stream = frame["feat_io_bytes_sum"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_mul_inf"], 1e-6)
    bw_eff = effective_bandwidth(stream, bw_inf, params["tau_mul_start"])
    return max(params["tau_mul"], 0.0) + stream / np.clip(bw_eff, a_min=1e-6, a_max=None)


def predict_sigmoid(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    stream = frame["feat_io_bytes_sum"].to_numpy(dtype=float)
    bw_peak = frame["bw_peak_bytes_per_us"].to_numpy(dtype=float)
    output_elements = frame["output_elements"].to_numpy(dtype=float)
    peak_add = frame["peak_add_ops_per_us"].to_numpy(dtype=float)
    bw_inf = bw_peak * max(params["rho_sigmoid_inf"], 1e-6)
    bw_eff = effective_bandwidth(stream, bw_inf, params["tau_sigmoid_start"])
    t_mem = stream / np.clip(bw_eff, a_min=1e-6, a_max=None)
    t_compute = output_elements / np.clip(peak_add * max(params["rho_sigmoid_compute"], 1e-9), a_min=1e-6, a_max=None)
    return max(params["tau_sigmoid"], 0.0) + np.maximum(t_mem, t_compute)


def predict_family(frame: pd.DataFrame, family: str, params: dict[str, float]) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=float)
    if family == "Concat":
        return predict_concat(frame, params)
    if family == "ReduceSum":
        return predict_reduce(frame, params)
    if family == "Gather":
        return predict_gather(frame, params)
    if family == "Gemm":
        return predict_gemm(frame, params)
    if family == "MatMul":
        return predict_matmul(frame, params)
    if family == "Transpose":
        return predict_transpose(frame, params)
    if family == "Relu":
        return predict_relu(frame, params)
    if family == "Add":
        return predict_add(frame, params)
    if family == "Mul":
        return predict_mul(frame, params)
    if family == "Sigmoid":
        return predict_sigmoid(frame, params)
    raise KeyError(family)


def family_mape(frame: pd.DataFrame, family: str, params: dict[str, float]) -> float:
    family_df = frame[frame["family"] == family].copy()
    if family_df.empty:
        return float("nan")
    pred = predict_family(family_df, family, params)
    return mape(family_df["actual_us"].to_numpy(dtype=float), pred)


def coordinate_search(param_names: list[str], objective: Any, params: dict[str, float], passes: int) -> dict[str, float]:
    tuned = dict(params)
    best_score = objective(tuned)
    for _ in range(max(int(passes), 1)):
        improved = False
        for name in param_names:
            local_best_value = tuned[name]
            local_best_score = best_score
            for candidate in PARAM_GRID[name]:
                trial = dict(tuned)
                trial[name] = float(candidate)
                score = objective(trial)
                if not np.isfinite(score):
                    continue
                if score + 1e-12 < local_best_score:
                    local_best_score = score
                    local_best_value = float(candidate)
            if local_best_value != tuned[name]:
                tuned[name] = local_best_value
                best_score = local_best_score
                improved = True
        if not improved:
            break
    return tuned


def calibrate_params(train_df: pd.DataFrame, passes: int) -> dict[str, float]:
    params = default_params()

    def copy_objective(trial: dict[str, float]) -> float:
        scores = [
            family_mape(train_df[train_df["family"].isin(COPY_FAMILIES)], family, trial)
            for family in COPY_FAMILIES
            if not train_df[train_df["family"] == family].empty
        ]
        clean = [score for score in scores if np.isfinite(score)]
        return float(np.mean(clean)) if clean else float("nan")

    def family_objective(family: str):
        return lambda trial: family_mape(train_df, family, trial)

    for _ in range(max(int(passes), 1)):
        before = dict(params)
        params = coordinate_search(["rho_copy_inf", "tau_copy_start"], copy_objective, params, passes=1)
        params = coordinate_search(["tau_dispatch"], family_objective("Concat"), params, passes=1)
        params = coordinate_search(["kappa_reduce", "tau_reduce_start"], family_objective("ReduceSum"), params, passes=1)
        params = coordinate_search(["rho_gather_inf", "tau_gather_row_start", "m_gather"], family_objective("Gather"), params, passes=1)
        params = coordinate_search(["rho_fma_inf", "M50", "N50", "K50"], family_objective("Gemm"), params, passes=1)
        params = coordinate_search(["occ_ref", "rho_tiny_inf", "K50_tiny", "tau_micro"], family_objective("MatMul"), params, passes=1)
        params = coordinate_search(["m_stride", "eta_stride"], family_objective("Transpose"), params, passes=1)
        params = coordinate_search(["rho_relu_inf", "tau_relu_start"], family_objective("Relu"), params, passes=1)
        params = coordinate_search(["rho_add_inf", "tau_add_start", "tau_add"], family_objective("Add"), params, passes=1)
        params = coordinate_search(["rho_mul_inf", "tau_mul_start", "tau_mul"], family_objective("Mul"), params, passes=1)
        params = coordinate_search(["rho_sigmoid_inf", "tau_sigmoid_start", "tau_sigmoid", "rho_sigmoid_compute"], family_objective("Sigmoid"), params, passes=1)
        if params == before:
            break
    return params


def family_metric_rows(frame: pd.DataFrame, params: dict[str, float], split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_df = frame[frame["family"] == family].copy()
        if family_df.empty:
            continue
        pred = predict_family(family_df, family, params)
        rows.append(
            {
                "split": split_name,
                "family": family,
                "row_count": int(len(family_df)),
                "mape": mape(family_df["actual_us"].to_numpy(dtype=float), pred),
                "dwre": duration_weighted_relative_error(family_df["actual_us"].to_numpy(dtype=float), pred),
                "actual_sum_us": float(family_df["actual_us"].sum()),
                "abs_error_sum_us": float(np.sum(np.abs(pred - family_df["actual_us"].to_numpy(dtype=float)))),
                "actual_mean_us": float(family_df["actual_us"].mean()),
                "pred_mean_us": float(np.mean(pred)),
            }
        )
    return rows


def summarize_metrics(metrics_df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    split_df = metrics_df[metrics_df["split"] == split_name].copy()
    if split_df.empty:
        return {}
    family_summary = (
        split_df.groupby("family", as_index=False)
        .agg(
            mean_mape=("mape", "mean"),
            median_mape=("mape", "median"),
            max_mape=("mape", "max"),
            mean_dwre=("dwre", "mean"),
            median_dwre=("dwre", "median"),
            max_dwre=("dwre", "max"),
            total_rows=("row_count", "sum"),
            total_actual_us=("actual_sum_us", "sum"),
            total_abs_error_us=("abs_error_sum_us", "sum"),
        )
        .sort_values("family")
    )
    weighted = float(np.average(split_df["mape"], weights=split_df["row_count"]))
    duration_weighted = float(split_df["abs_error_sum_us"].sum() / max(split_df["actual_sum_us"].sum(), 1e-9))
    return {
        "split": split_name,
        "family_summary": family_summary.to_dict(orient="records"),
        "weighted_family_mape": weighted,
        "duration_weighted_relative_error": duration_weighted,
    }


def prepare_heavy_slice(input_csv: Path, hardware_profile_path: Path | None = None) -> pd.DataFrame:
    dataset = pd.read_csv(input_csv, low_memory=False)
    dataset = add_operator_hardware_context(dataset, load_hardware_features(hardware_profile_path))
    dataset["family"] = dataset.apply(analytical_family_name, axis=1)
    dataset = dataset[dataset["family"] != ""].copy()
    dataset["actual_us"] = pd.to_numeric(dataset["label_operator_actual_dur_us"], errors="coerce").fillna(0.0)
    records: list[dict[str, Any]] = []
    for _, row in dataset.iterrows():
        input_entries = _shape_entries(row.get("input_type_shape"))
        output_entries = _shape_entries(row.get("output_type_shape"))
        input_bytes_sum = float(sum(entry_num_bytes(entry) for entry in input_entries))
        output_bytes_sum = float(sum(entry_num_bytes(entry) for entry in output_entries))
        output_entry = output_entries[0] if output_entries else {}
        output_dims = [int(dim) for dim in output_entry.get("dims", [])]
        output_dtype_bytes = dtype_size(str(output_entry.get("dtype", "float32")))
        gemm_m, gemm_n, gemm_k = _infer_gemm_mnk(input_entries, output_entries)
        batch_count, matmul_m, matmul_n, matmul_k = infer_batched_matmul_dims(input_entries, output_entries)
        configured_request_rows = max(safe_float(row.get("feat_lookup_count"), 0.0), 0.0)
        request_rows = max(_infer_gather_request_rows(input_entries), 0.0)
        if request_rows <= 0.0:
            request_rows = configured_request_rows
        row_bytes = safe_float(row.get("output_size"), 0.0) / max(request_rows, 1.0)
        cacheline = max(safe_float(row.get("hw_cache_cacheline_bytes"), 64.0), 1.0)
        cachelines_per_row = math.ceil(row_bytes / cacheline) if row_bytes > 0.0 else 0
        gather_table_rows = 0.0
        if input_entries:
            dims0 = [int(dim) for dim in input_entries[0].get("dims", [])]
            if dims0:
                gather_table_rows = float(max(dims0[0], 0))
        gather_requested_rows_capped = min(request_rows, gather_table_rows) if gather_table_rows > 0.0 else request_rows
        gather_unique_rows_est = gather_table_rows * (1.0 - math.exp(-request_rows / gather_table_rows)) if gather_table_rows > 0.0 else request_rows
        gather_unique_rows_est = min(max(gather_unique_rows_est, 0.0), gather_requested_rows_capped)
        gather_src_unique_bytes = gather_unique_rows_est * row_bytes
        gather_src_fit_level = fit_level(gather_src_unique_bytes, row)
        src_latency_us = latency_from_level(gather_src_fit_level, row)
        transpose_prefix_blocks = 1.0
        if len(output_dims) > 1:
            prefix = 1
            for dim in output_dims[:-1]:
                prefix *= int(dim)
            transpose_prefix_blocks = float(prefix)
        transpose_suffix_block_bytes = float(output_dims[-1] * output_dtype_bytes) if output_dims else output_bytes_sum
        transpose_suffix_fit_level = fit_level(transpose_suffix_block_bytes, row)
        transpose_stride_latency_us = latency_from_level(transpose_suffix_fit_level, row)
        reduce_acc_bytes_per_thread = max(safe_float(row.get("output_size"), 0.0), 0.0) / max(safe_float(row.get("num_threads"), 1.0), 1.0)
        reduce_acc_fit_level = fit_level(reduce_acc_bytes_per_thread, row)
        reduce_acc_latency_us = latency_from_level(reduce_acc_fit_level, row)
        issue_slots = issue_slots_per_us(row)
        records.append(
            {
                "row_uid": row["row_uid"],
                "sample_group": str(row.get("sample_group", "")),
                "case_id": str(row.get("case_id", "")),
                "combo": str(row.get("combo", "")),
                "node_name": str(row.get("node_name", "")),
                "family": str(row["family"]),
                "actual_us": safe_float(row.get("actual_us"), 0.0),
                "num_threads": max(safe_float(row.get("num_threads"), 1.0), 1.0),
                "output_size": max(safe_float(row.get("output_size"), 0.0), 0.0),
                "activation_size": max(safe_float(row.get("activation_size"), 0.0), 0.0),
                "parameter_size": max(safe_float(row.get("parameter_size"), 0.0), 0.0),
                "feat_io_bytes_sum": max(safe_float(row.get("feat_io_bytes_sum"), 0.0), 0.0),
                "output_elements": output_bytes_sum / max(float(output_dtype_bytes), 1.0),
                "feat_lookup_count": request_rows,
                "feat_reduction_axes_product": max(safe_float(row.get("feat_reduction_axes_product"), 0.0), 0.0),
                "feat_reduction_work_items": max(safe_float(row.get("feat_reduction_work_items"), 0.0), 0.0),
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
                "bw_peak_bytes_per_us": max(safe_float(row.get("hw_memory_bandwidth_gb_s_total"), 100.0), 1e-6) * 1e3,
                "hw_core_cpu_clock": max(safe_float(row.get("hw_core_cpu_clock"), 2.6), 1e-6),
                "hw_cache_l1d_response_latency_cycles": max(safe_float(row.get("hw_cache_l1d_response_latency_cycles"), 1.0), 1e-6),
                "peak_add_ops_per_us": peak_add_ops_per_us(row),
                "peak_fma_ops_per_us": peak_fma_ops_per_us(row),
            }
        )
    return pd.DataFrame(records)
