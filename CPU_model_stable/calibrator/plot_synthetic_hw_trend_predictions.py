from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrator.hw_generalization_audit import (  # noqa: E402
    DEFAULT_BINDING_CONFIG,
    DatasetBinding,
    _fit_route_bundle,
    _load_bindings,
    _load_hw_payload,
    _nested_get,
    _nested_set,
    _predict_route,
    _prepare_semantic_frames,
    _scale_scalar,
)
from calibrator.static_feature_mapping.core import (  # noqa: E402
    FULL_GRAPH_76,
    MODEL_INPUT_51,
    DlrmStaticConfig,
    build_static_feature_frame,
    build_static_op_shape_table,
)
from calibrator.whitebox_family_routes import (  # noqa: E402
    WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
    WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
    _load_hw_params,
    fit_op_thread_scale_config,
)
from cpu_perf_model.graph_contract import ComboArtifactPaths, ComboSpec  # noqa: E402
from cpu_perf_model.schedule_engine import build_op_graph, schedule_combo  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "hw_generalization_sensitivity_contcache_simple_rf_v2_20260429"
    / "trend_figures_synthetic_model"
)

SENSITIVITY_ENV_DEFAULTS = {
    "WHITEBOX_RANDOM_MEM_LATENCY_TAIL_ALPHA": "0.10",
    "WHITEBOX_COMPUTE_LATENCY_TAIL_ALPHA": "1.0",
    "WHITEBOX_OVERRIDE_FREEZE_CACHE_LATENCY": "0",
    "WHITEBOX_USE_STATIC_CACHE_RESIDENCY_RATIOS": "0",
    "WHITEBOX_CONTINUOUS_CACHE_RESIDENCY": "1",
}


@dataclass(frozen=True)
class TrendSpec:
    key: str
    group: str
    title: str
    x_label: str
    fields: tuple[str, ...]
    relation: str
    factors: tuple[float, ...]
    workload_kind: str
    op_types: tuple[str, ...]
    aggregate_label: str
    target_cache_level: str = ""


@dataclass(frozen=True)
class WorkloadVariant:
    key: str
    label: str
    scale: float


FACTOR_CACHE_SIZE = tuple(float(value) for value in np.round(np.linspace(0.65, 1.35, 15), 4))
FACTOR_COST = tuple(float(value) for value in np.round(np.linspace(0.75, 1.25, 11), 4))
FACTOR_CAPACITY = tuple(float(value) for value in np.round(np.linspace(0.70, 1.30, 13), 4))
FACTOR_WIDE_CAPACITY = tuple(float(value) for value in np.round(np.linspace(0.60, 1.60, 13), 4))
FACTOR_FIXED_CACHE_SIZE = (0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00)
FACTOR_FIXED_GENERAL = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.70, 1.85, 2.00)


GLOBAL_FIXED_DLRM_CONFIGS: dict[str, dict[str, Any]] = {
    "small": {
        "fixed_workload_id": "DLRM-S",
        "batch_size": 1024,
        "arch_mlp_bot": "256-1024-256",
        "arch_mlp_top": "292-1024-1",
    },
    "base": {
        "fixed_workload_id": "DLRM-M",
        "batch_size": 2048,
        "arch_mlp_bot": "512-2048-512",
        "arch_mlp_top": "548-2048-1",
    },
    "large": {
        "fixed_workload_id": "DLRM-L",
        "batch_size": 4096,
        "arch_mlp_bot": "1024-4096-1024",
        "arch_mlp_top": "1060-4096-1",
    },
}


WORKER_SWEEP_POINTS = (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 60)


TREND_SPECS: tuple[TrendSpec, ...] = (
    TrendSpec("active_parallel_workers", "core", "Active parallel workers", "Active parallel workers", (), "capacity", WORKER_SWEEP_POINTS, "core_parallel", ("Gather", "ReduceSum", "Gemm", "MatMul"), "synthetic saturated parallel DLRM-like graph"),
    TrendSpec("l1d_size", "cache_size", "L1D cache size", "Relative L1D size", ("cache.l1d.size",), "capacity", FACTOR_CACHE_SIZE, "cache_size_l1", ("Gather",), "synthetic Gather near L1", "L1D"),
    TrendSpec("l2_size", "cache_size", "L2 cache size", "Relative L2 size", ("cache.l2.size",), "capacity", FACTOR_CACHE_SIZE, "cache_size_l2", ("Gather",), "synthetic Gather near L2", "L2"),
    TrendSpec("l3_size", "cache_size", "L3 cache size", "Relative L3 size", ("cache.l3_per_die.size",), "capacity", FACTOR_CACHE_SIZE, "cache_size_l3", ("Gather",), "synthetic Gather near L3", "L3"),
    TrendSpec("l1d_latency", "cache_latency", "L1D cache latency", "Relative L1D latency", ("cache.l1d.response_latency_cycles",), "cost", FACTOR_COST, "cache_latency_l1", ("Gather",), "synthetic Gather L1 resident", "L1D"),
    TrendSpec("l2_latency", "cache_latency", "L2 cache latency", "Relative L2 latency", ("cache.l2.response_latency_cycles",), "cost", FACTOR_COST, "cache_latency_l2", ("Gather",), "synthetic Gather L2 resident", "L2"),
    TrendSpec("l3_latency", "cache_latency", "L3 cache latency", "Relative L3 latency", ("cache.l3_per_die.response_latency_cycles",), "cost", FACTOR_COST, "cache_latency_l3", ("Gather",), "synthetic Gather L3 resident", "L3"),
    TrendSpec("copy_bw", "memory", "Sequential memory bandwidth", "Relative copy bandwidth", ("memory.copy_bw_gb_s_1t", "memory.copy_bw_gb_s_saturated"), "capacity", FACTOR_CAPACITY, "stream_copy", ("Concat", "Transpose", "Reshape"), "synthetic stream-copy subgraph"),
    TrendSpec("random_gather_bw", "memory", "Random gather bandwidth", "Relative gather bandwidth", ("memory.random_gather_bw_gb_s_1t", "memory.random_gather_bw_gb_s_saturated"), "capacity", FACTOR_CAPACITY, "memory_random", ("Gather",), "synthetic random Gather"),
    TrendSpec("mem_latency", "memory", "DRAM access latency", "Relative DRAM latency", ("memory.local_mem_delay_ns",), "cost", FACTOR_COST, "memory_latency", ("Gather",), "synthetic DRAM-resident Gather"),
    TrendSpec("cpu_clock", "compute_pipeline", "CPU frequency", "Relative CPU frequency", ("core.cpu_clock",), "capacity", FACTOR_COST, "compute_gemm", ("Gemm", "MatMul"), "synthetic Gemm/MatMul"),
    TrendSpec("simd_width", "compute_pipeline", "SIMD width", "Relative SIMD width", ("instruction.simd_width_bits",), "capacity", FACTOR_COST, "compute_gemm", ("Gemm", "MatMul"), "synthetic Gemm/MatMul"),
    TrendSpec("fma_throughput", "compute_pipeline", "Vector FMA throughput", "Relative FMA throughput", ("instruction.fp_throughput_per_cycle.vector_sp_fma",), "capacity", FACTOR_COST, "compute_gemm", ("Gemm", "MatMul"), "synthetic Gemm/MatMul"),
    TrendSpec("fma_latency", "compute_pipeline", "Vector FMA latency", "Relative FMA latency", ("instruction.fp_latency_cycles.vector_sp_fma",), "cost", FACTOR_COST, "compute_gemm", ("Gemm", "MatMul"), "synthetic Gemm/MatMul"),
    TrendSpec(
        "pipeline_width",
        "compute_pipeline",
        "Pipeline width",
        "Relative pipeline width",
        (
            "pipeline.fetch_width",
            "pipeline.decode_width",
            "pipeline.rename_width",
            "pipeline.dispatch_width",
            "pipeline.issue_width",
            "pipeline.commit_width",
        ),
        "capacity",
        FACTOR_COST,
        "metadata_pipeline",
        ("Shape", "Reshape", "Unsqueeze", "Flatten", "Concat"),
        "synthetic metadata/control subgraph",
    ),
    TrendSpec("full_compute_bundle", "bundle", "Compute bundle", "Relative compute bundle", ("core.cpu_clock", "instruction.simd_width_bits", "instruction.fp_throughput_per_cycle.vector_sp_fma"), "capacity", FACTOR_COST, "compute_gemm", ("Gemm", "MatMul"), "synthetic Gemm/MatMul"),
    TrendSpec("full_memory_bundle", "bundle", "Memory bundle", "Relative memory bundle", ("memory.copy_bw_gb_s_1t", "memory.copy_bw_gb_s_saturated", "memory.random_gather_bw_gb_s_1t", "memory.random_gather_bw_gb_s_saturated"), "capacity", FACTOR_WIDE_CAPACITY, "memory_random", ("Gather",), "synthetic memory-heavy Gather"),
)

GROUP_TO_FIGURE = {
    "core": ("synthetic_core_trends", "Synthetic Core Capacity Sensitivity"),
    "cache_size": ("synthetic_cache_size_trends", "Synthetic Cache Size Sensitivity"),
    "cache_latency": ("synthetic_cache_latency_trends", "Synthetic Cache Latency Sensitivity"),
    "memory": ("synthetic_memory_trends", "Synthetic Memory Sensitivity"),
    "compute_pipeline": ("synthetic_compute_pipeline_trends", "Synthetic Compute And Pipeline Sensitivity"),
    "bundle": ("synthetic_bundle_trends", "Synthetic Bundle Sensitivity"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot synthetic-workload hardware trend predictions for whitebox_family_prior_plus_simple_rf_v2.",
    )
    parser.add_argument("--binding-config", default=str(DEFAULT_BINDING_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--policy", default="")
    parser.add_argument("--datasets", default="s920x20_local,kunpeng920_remote")
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--op-thread-scale-mode", default="fit", choices=["off", "fit"])
    parser.add_argument(
        "--bundle-source",
        default="fit",
        choices=["fit", "binding"],
        help="fit retrains the route from binding data; binding loads each dataset's model_bundle from the binding config.",
    )
    parser.add_argument("--route", default=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)
    parser.add_argument("--plot-source", default="route_pred", choices=["route_pred", "whitebox_prior"])
    parser.add_argument("--prediction-scope", default="operator_group", choices=["operator_group", "e2e_pipeline"])
    parser.add_argument("--groups", default="", help="Comma-separated trend groups to run; empty means all groups.")
    parser.add_argument(
        "--cache-size-working-set-scale",
        type=float,
        default=1.03,
        help="Synthetic cache-size workload working-set multiplier relative to the target cache level.",
    )
    parser.add_argument(
        "--workload-variants",
        default="base",
        help="Comma-separated workload size variants to draw in each panel. Supported: small,base,large.",
    )
    parser.add_argument(
        "--workload-design",
        default="per_trend_probe",
        choices=["per_trend_probe", "global_fixed_dlrm"],
        help="Synthetic workload design. per_trend_probe preserves existing per-trend probes; global_fixed_dlrm reuses fixed DLRM-S/M/L configs across all trends.",
    )
    parser.add_argument(
        "--hardware-factor-profile",
        default="current",
        choices=["current", "wide_fixed"],
        help="Hardware sweep factor profile. wide_fixed uses wider ranges for fixed-workload bottleneck/inflection sweeps.",
    )
    parser.add_argument(
        "--hide-device-label",
        action="store_true",
        help="Hide Device_1/Device_2 prefixes in plot legends; useful when plotting a single dataset alias.",
    )
    return parser.parse_args()


def _apply_sensitivity_env_defaults() -> dict[str, str]:
    for key, value in SENSITIVITY_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    return {key: str(os.environ.get(key, "")) for key in SENSITIVITY_ENV_DEFAULTS}


def _mutate_payload(base_payload: dict[str, Any], spec: TrendSpec, factor: float) -> tuple[dict[str, Any], str, str]:
    payload = json.loads(json.dumps(base_payload))
    if spec.key == "active_parallel_workers":
        value = int(round(float(factor)))
        original = _nested_get(payload, "core.total_cores")
        _nested_set(payload, "core.total_cores", value)
        return (
            payload,
            json.dumps([original], ensure_ascii=False),
            json.dumps([value], ensure_ascii=False),
        )
    original_values: list[Any] = []
    mutated_values: list[Any] = []
    for field in spec.fields:
        original = _nested_get(payload, field)
        mutated = _scale_scalar(original, factor)
        _nested_set(payload, field, mutated)
        original_values.append(original)
        mutated_values.append(mutated)
    return payload, json.dumps(original_values, ensure_ascii=False), json.dumps(mutated_values, ensure_ascii=False)


def _load_bundle_from_binding(binding: DatasetBinding, route: str) -> tuple[dict[str, Any], Path]:
    if str(route) != WHITEBOX_PHYSICS_GUARDED_RF_ROUTE:
        raise ValueError(
            "--bundle-source=binding currently supports only "
            f"{WHITEBOX_PHYSICS_GUARDED_RF_ROUTE}, got {route!r}"
        )
    bundle_path = getattr(binding, "model_bundle", None)
    if bundle_path is None:
        raise ValueError(
            f"Dataset binding {binding.alias!r} has no model_bundle. "
            "Add model_bundle to the binding config or use --bundle-source=fit."
        )
    bundle_path = Path(bundle_path).resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"model_bundle not found for {binding.alias}: {bundle_path}")
    if bundle_path.is_dir():
        raise FileNotFoundError(f"model_bundle must be a pickle file, got directory: {bundle_path}")
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"model_bundle must contain a dict, got {type(bundle).__name__}: {bundle_path}")
    bundle_route = str(bundle.get("route", "")).strip()
    if bundle_route != str(route):
        raise ValueError(f"model_bundle route mismatch for {binding.alias}: expected {route!r}, got {bundle_route!r}")
    return bundle, bundle_path


def _bundle_export_hw_config_path(bundle: dict[str, Any]) -> str:
    export_metadata = bundle.get("export_metadata", {})
    if isinstance(export_metadata, dict):
        return str(export_metadata.get("hw_config_path", ""))
    return ""


def _bundle_hw_config_mismatch(bundle: dict[str, Any], binding_hw_config: Path) -> bool:
    export_hw = _bundle_export_hw_config_path(bundle)
    if not export_hw:
        return False
    try:
        return Path(export_hw).resolve() != Path(binding_hw_config).resolve()
    except OSError:
        return True


def _workload_variants(requested: str, workload_design: str = "per_trend_probe") -> tuple[WorkloadVariant, ...]:
    if str(workload_design) == "global_fixed_dlrm":
        variants = {
            "small": WorkloadVariant("small", "DLRM-S", 1.0),
            "base": WorkloadVariant("base", "DLRM-M", 1.0),
            "large": WorkloadVariant("large", "DLRM-L", 1.0),
        }
    else:
        variants = {
            "small": WorkloadVariant("small", "small", 0.70),
            "base": WorkloadVariant("base", "base", 1.00),
            "large": WorkloadVariant("large", "large", 1.45),
        }
    keys = [item.strip() for item in str(requested).split(",") if item.strip()]
    if not keys:
        keys = ["base"]
    unknown = [key for key in keys if key not in variants]
    if unknown:
        raise ValueError(f"Unsupported workload variants: {unknown}. Supported: {sorted(variants)}")
    return tuple(variants[key] for key in keys)


def _specs_for_factor_profile(specs: tuple[TrendSpec, ...], hardware_factor_profile: str) -> tuple[TrendSpec, ...]:
    if str(hardware_factor_profile) == "current":
        return specs
    if str(hardware_factor_profile) != "wide_fixed":
        raise ValueError(f"Unsupported hardware factor profile: {hardware_factor_profile}")
    profiled: list[TrendSpec] = []
    for spec in specs:
        if spec.key == "active_parallel_workers":
            profiled.append(spec)
            continue
        factors = FACTOR_FIXED_CACHE_SIZE if spec.group == "cache_size" else FACTOR_FIXED_GENERAL
        profiled.append(replace(spec, factors=factors))
    return tuple(profiled)


def _embedding_rows(rows: int) -> str:
    value = max(int(rows), 128)
    return "-".join(str(value) for _ in range(8))


def _threading_for_hw(hw: dict[str, float]) -> tuple[int, int]:
    cores = max(int(round(float(hw.get("total_cores", 1.0)))), 1)
    inter_threads = min(8, max(1, cores))
    intra_threads = max(1, cores // inter_threads)
    return intra_threads, inter_threads


def _worker_sweep_values(hw: dict[str, float]) -> tuple[float, ...]:
    cores = max(int(round(float(hw.get("total_cores", 1.0)))), 1)
    values = sorted(set(int(value) for value in WORKER_SWEEP_POINTS) | {cores})
    return tuple(float(value) for value in values)


def _threading_for_active_workers(hw: dict[str, float], target_workers: int | float | None) -> tuple[int, int, int]:
    baseline_cores = max(int(round(float(hw.get("total_cores", 1.0)))), 1)
    target = int(round(float(target_workers))) if target_workers is not None else baseline_cores
    target = max(target, 1)
    design_cores = max(baseline_cores, target)
    best: tuple[int, int, int, tuple[int, int, int]] | None = None
    for inter_threads in range(min(8, target), 0, -1):
        intra_threads = int(math.ceil(float(target) / float(inter_threads)))
        active_workers = min(design_cores, intra_threads * inter_threads)
        overshoot = max(active_workers - target, 0)
        exact_penalty = 0 if active_workers == target else 1
        score = (exact_penalty, overshoot, -inter_threads)
        candidate = (intra_threads, inter_threads, active_workers, score)
        if best is None or candidate[3] < best[3]:
            best = candidate
    if best is None:
        return 1, 1, 1
    return best[0], best[1], best[2]


def _active_workers_from_config(
    hw: dict[str, float],
    config: DlrmStaticConfig,
    active_worker_count: int | float | None,
) -> float:
    baseline_cores = max(int(round(float(hw.get("total_cores", 1.0)))), 1)
    design_cores = baseline_cores
    if active_worker_count is not None:
        design_cores = max(design_cores, int(round(float(active_worker_count))))
    return float(min(float(design_cores), float(config.num_threads) * float(config.inter_threads)))


def _arch_embedding_rows(lookup_count: int) -> str:
    rows = max(int(math.ceil(float(lookup_count) * 4.0)), 128)
    return "-".join(str(rows) for _ in range(8))


def _config_from_lookup_count(
    *,
    lookup_count: int,
    bot_out: int,
    hw: dict[str, float],
    dense_dim: int = 16,
    bot_hidden: int = 64,
    top_hidden: int = 64,
    single_thread: bool = False,
) -> DlrmStaticConfig:
    lookup_count = max(int(lookup_count), 1)
    batch_guess = int(2 ** round(math.log2(max(math.sqrt(float(lookup_count)), 1.0))))
    batch = int(np.clip(batch_guess, 1, 8192))
    num_indices = max(int(math.ceil(float(lookup_count) / float(batch))), 1)
    if single_thread:
        intra_threads, inter_threads = 1, 1
    else:
        intra_threads, inter_threads = _threading_for_hw(hw)
    return DlrmStaticConfig(
        batch_size=batch,
        num_indices_per_lookup=num_indices,
        arch_embedding_size=_arch_embedding_rows(batch * num_indices),
        arch_mlp_bot=f"{int(dense_dim)}-{int(bot_hidden)}-{int(bot_out)}",
        arch_mlp_top=f"{int(bot_out) + 36}-{int(top_hidden)}-1",
        inter_threads=inter_threads,
        num_threads=intra_threads,
    )


def _global_fixed_dlrm_config(workload_variant: WorkloadVariant) -> DlrmStaticConfig:
    variant_key = str(workload_variant.key)
    if variant_key not in GLOBAL_FIXED_DLRM_CONFIGS:
        raise ValueError(f"Unsupported global fixed DLRM variant: {variant_key}")
    spec = GLOBAL_FIXED_DLRM_CONFIGS[variant_key]
    return DlrmStaticConfig(
        batch_size=int(spec["batch_size"]),
        num_indices_per_lookup=16,
        arch_embedding_size=_embedding_rows(65536),
        arch_mlp_bot=str(spec["arch_mlp_bot"]),
        arch_mlp_top=str(spec["arch_mlp_top"]),
        inter_threads=8,
        num_threads=3,
    )


def _fixed_workload_id(workload_design: str, workload_variant: WorkloadVariant) -> str:
    if str(workload_design) != "global_fixed_dlrm":
        return ""
    return str(GLOBAL_FIXED_DLRM_CONFIGS[str(workload_variant.key)]["fixed_workload_id"])


def _config_for_target_working_set(
    hw: dict[str, float],
    target_bytes: float,
    *,
    bot_out: int = 64,
    single_thread: bool = False,
) -> DlrmStaticConfig:
    row_bytes = max(float(bot_out) * 4.0, 8.0)
    lookup_count = int(max(round(float(target_bytes) / row_bytes), 1))
    return _config_from_lookup_count(lookup_count=lookup_count, bot_out=bot_out, hw=hw, single_thread=single_thread)


def _capacity_bytes(hw: dict[str, float], level: str) -> float:
    if level == "L1D":
        return float(hw["l1d_size_bytes"])
    if level == "L2":
        return float(hw["l2_size_bytes"])
    if level == "L3":
        return float(hw["l3_size_bytes"])
    raise ValueError(f"Unsupported cache level: {level}")


def _active_cache_capacity_bytes(hw: dict[str, float], level: str) -> float:
    _, inter_threads = _threading_for_hw(hw)
    return _capacity_bytes(hw, level) / max(float(inter_threads), 1.0)


def _workload_config(spec: TrendSpec, hw: dict[str, float], *, cache_size_working_set_scale: float = 1.03) -> DlrmStaticConfig:
    return _workload_config_for_variant(
        spec,
        hw,
        workload_variant=WorkloadVariant("base", "base", 1.0),
        cache_size_working_set_scale=cache_size_working_set_scale,
        workload_design="per_trend_probe",
    )


def _workload_config_for_variant(
    spec: TrendSpec,
    hw: dict[str, float],
    *,
    workload_variant: WorkloadVariant,
    cache_size_working_set_scale: float = 1.03,
    workload_design: str = "per_trend_probe",
    active_worker_count: int | float | None = None,
) -> DlrmStaticConfig:
    variant = str(workload_variant.key)
    if str(workload_design) == "global_fixed_dlrm":
        return _global_fixed_dlrm_config(workload_variant)
    if spec.workload_kind == "core_parallel":
        if spec.key == "active_parallel_workers":
            intra_threads, inter_threads, _ = _threading_for_active_workers(hw, active_worker_count)
        else:
            intra_threads, inter_threads = _threading_for_hw(hw)
        batch = {"small": 4096, "base": 8192, "large": 16384}[variant]
        num_indices = {"small": 64, "base": 96, "large": 128}[variant]
        bot = {"small": "512-2048-512", "base": "1024-4096-1024", "large": "2048-8192-2048"}[variant]
        top = {"small": "548-2048-1", "base": "1060-4096-1", "large": "2084-8192-1"}[variant]
        return DlrmStaticConfig(
            batch_size=batch,
            num_indices_per_lookup=num_indices,
            arch_embedding_size=_arch_embedding_rows(batch * num_indices),
            arch_mlp_bot=bot,
            arch_mlp_top=top,
            inter_threads=inter_threads,
            num_threads=intra_threads,
        )
    if spec.workload_kind == "cache_size_l1":
        scale = {"small": 1.10, "base": cache_size_working_set_scale, "large": 1.80}[variant]
        return _config_for_target_working_set(hw, scale * _capacity_bytes(hw, "L1D"), bot_out=2, single_thread=True)
    if spec.workload_kind == "cache_size_l2":
        scale = {"small": 1.10, "base": cache_size_working_set_scale, "large": 1.80}[variant]
        return _config_for_target_working_set(hw, scale * _capacity_bytes(hw, "L2"), bot_out=2, single_thread=True)
    if spec.workload_kind == "cache_size_l3":
        scale = {"small": 1.10, "base": cache_size_working_set_scale, "large": 1.80}[variant]
        return _config_for_target_working_set(hw, scale * _capacity_bytes(hw, "L3"), bot_out=2, single_thread=True)
    if spec.workload_kind == "cache_latency_l1":
        scale = {"small": 0.45, "base": 0.70, "large": 0.95}[variant]
        return _config_for_target_working_set(hw, scale * _capacity_bytes(hw, "L1D"), bot_out=2, single_thread=True)
    if spec.workload_kind == "cache_latency_l2":
        scale = {"small": 0.45, "base": 0.70, "large": 0.95}[variant]
        target = max(2.5 * _capacity_bytes(hw, "L1D"), scale * _capacity_bytes(hw, "L2"))
        return _config_for_target_working_set(hw, target, bot_out=2, single_thread=True)
    if spec.workload_kind == "cache_latency_l3":
        scale = {"small": 0.45, "base": 0.70, "large": 0.95}[variant]
        target = max(2.5 * _capacity_bytes(hw, "L2"), scale * _capacity_bytes(hw, "L3"))
        return _config_for_target_working_set(hw, target, bot_out=2, single_thread=True)
    if spec.workload_kind == "memory_random":
        scale = {"small": 1.40, "base": 2.20, "large": 3.00}[variant]
        return _config_for_target_working_set(hw, scale * _active_cache_capacity_bytes(hw, "L3"), bot_out=2)
    if spec.workload_kind == "memory_latency":
        scale = {"small": 1.10, "base": 1.60, "large": 2.20}[variant]
        return _config_for_target_working_set(hw, scale * _capacity_bytes(hw, "L3"), bot_out=2, single_thread=True)
    if spec.workload_kind == "stream_copy":
        intra_threads, inter_threads = _threading_for_hw(hw)
        batch = {"small": 2048, "base": 4096, "large": 8192}[variant]
        rows = {"small": 32768, "base": 65536, "large": 131072}[variant]
        return DlrmStaticConfig(
            batch_size=batch,
            num_indices_per_lookup=16,
            arch_embedding_size=_embedding_rows(rows),
            arch_mlp_bot="64-256-128",
            arch_mlp_top="164-256-1",
            inter_threads=inter_threads,
            num_threads=intra_threads,
        )
    if spec.workload_kind == "compute_gemm":
        intra_threads, inter_threads = _threading_for_hw(hw)
        if variant == "small":
            batch, bot, top = 1024, "256-1024-256", "292-1024-1"
        elif variant == "large":
            batch, bot, top = 4096, "1024-4096-1024", "1060-4096-1"
        else:
            batch, bot, top = 2048, "512-2048-512", "548-2048-1"
        return DlrmStaticConfig(
            batch_size=batch,
            num_indices_per_lookup=16,
            arch_embedding_size="65536-65536-65536-65536-65536-65536-65536-65536",
            arch_mlp_bot=bot,
            arch_mlp_top=top,
            inter_threads=inter_threads,
            num_threads=intra_threads,
        )
    if spec.workload_kind == "metadata_pipeline":
        intra_threads, inter_threads = _threading_for_hw(hw)
        batch = {"small": 16, "base": 32, "large": 64}[variant]
        rows = {"small": 512, "base": 1024, "large": 2048}[variant]
        return DlrmStaticConfig(
            batch_size=batch,
            num_indices_per_lookup=4,
            arch_embedding_size=_embedding_rows(rows),
            arch_mlp_bot="16-32-32",
            arch_mlp_top="68-64-1",
            inter_threads=inter_threads,
            num_threads=intra_threads,
        )
    raise ValueError(f"Unsupported workload kind: {spec.workload_kind}")


def _specialize_frame_for_trend(spec: TrendSpec, frame: pd.DataFrame) -> pd.DataFrame:
    return frame


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _gather_working_set_summary(frame: pd.DataFrame, hw: dict[str, float]) -> dict[str, float]:
    if frame.empty or "op_type" not in frame.columns:
        return {
            "gather_working_set_bytes_median": float("nan"),
            "gather_working_set_bytes_max": float("nan"),
            "ratio_working_set_to_l1d": float("nan"),
            "ratio_working_set_to_l2": float("nan"),
            "ratio_working_set_to_l3": float("nan"),
        }
    gather = frame.loc[frame["op_type"].astype(str).eq("Gather")].copy()
    if gather.empty:
        return {
            "gather_working_set_bytes_median": float("nan"),
            "gather_working_set_bytes_max": float("nan"),
            "ratio_working_set_to_l1d": float("nan"),
            "ratio_working_set_to_l2": float("nan"),
            "ratio_working_set_to_l3": float("nan"),
        }
    lookup_count = _numeric_column(gather, "feat_lookup_count", np.nan).replace(0.0, np.nan)
    output_size = _numeric_column(gather, "output_size", 0.0)
    row_bytes = (output_size / lookup_count).replace([np.inf, -np.inf], np.nan)
    row_bytes = row_bytes.fillna(np.maximum(output_size, 8.0)).clip(lower=8.0)
    unique_count = _numeric_column(gather, "gather_unique_idx_count", np.nan).fillna(lookup_count)
    unique_ratio = (unique_count / lookup_count).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0, upper=1.0)
    static_working_set = _numeric_column(gather, "gather_working_set_eff_bytes", 0.0).fillna(0.0)
    working_set = np.maximum(
        static_working_set.to_numpy(dtype=float),
        (
            _numeric_column(gather, "feat_lookup_count", 0.0).fillna(0.0)
            * row_bytes
            * unique_ratio
        ).to_numpy(dtype=float),
    )
    working_set = working_set[np.isfinite(working_set) & (working_set > 0.0)]
    if working_set.size == 0:
        median = float("nan")
        maximum = float("nan")
    else:
        median = float(np.nanmedian(working_set))
        maximum = float(np.nanmax(working_set))
    return {
        "gather_working_set_bytes_median": median,
        "gather_working_set_bytes_max": maximum,
        "ratio_working_set_to_l1d": float(median / max(float(hw["l1d_size_bytes"]), 1.0)) if np.isfinite(median) else float("nan"),
        "ratio_working_set_to_l2": float(median / max(float(hw["l2_size_bytes"]), 1.0)) if np.isfinite(median) else float("nan"),
        "ratio_working_set_to_l3": float(median / max(float(hw["l3_size_bytes"]), 1.0)) if np.isfinite(median) else float("nan"),
    }


def _target_frame(
    spec: TrendSpec,
    hw: dict[str, float],
    *,
    workload_variant: WorkloadVariant = WorkloadVariant("base", "base", 1.0),
    cache_size_working_set_scale: float = 1.03,
    workload_design: str = "per_trend_probe",
    hardware_factor_profile: str = "current",
    active_worker_count: int | float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = _workload_config_for_variant(
        spec,
        hw,
        workload_variant=workload_variant,
        cache_size_working_set_scale=cache_size_working_set_scale,
        workload_design=workload_design,
        active_worker_count=active_worker_count,
    )
    frame = _specialize_frame_for_trend(spec, build_static_feature_frame(config, graph_scope=MODEL_INPUT_51))
    mask = frame["op_type"].astype(str).isin(spec.op_types)
    if spec.workload_kind.startswith("cache_") or spec.workload_kind in {"memory_random", "memory_latency"}:
        mask = mask & frame["submodel_role_static"].astype(str).eq("embedding_branch")
    if not mask.any():
        raise RuntimeError(f"Synthetic workload for {spec.key} has no target op types: {spec.op_types}")
    selected = frame.loc[mask].copy().reset_index(drop=True)
    metadata = {
        "workload_variant": str(workload_variant.key),
        "workload_variant_label": str(workload_variant.label),
        "workload_variant_scale": float(workload_variant.scale),
        "batch_size": int(config.batch_size),
        "num_indices_per_lookup": int(config.num_indices_per_lookup),
        "arch_embedding_size": str(config.arch_embedding_size),
        "arch_mlp_bot": str(config.arch_mlp_bot),
        "arch_mlp_top": str(config.arch_mlp_top),
        "num_threads": int(config.num_threads),
        "inter_threads": int(config.inter_threads),
        "selected_ops": ",".join(str(value) for value in spec.op_types),
        "selected_rows": int(len(selected)),
        "selected_output_bytes": float(pd.to_numeric(selected.get("exact_output_tensor_bytes_sum", 0.0), errors="coerce").fillna(0.0).sum()),
        "selected_read_bytes": float(pd.to_numeric(selected.get("exact_read_bytes", 0.0), errors="coerce").fillna(0.0).sum()),
        "selected_write_bytes": float(pd.to_numeric(selected.get("exact_write_bytes", 0.0), errors="coerce").fillna(0.0).sum()),
        "selected_lookup_count": float(pd.to_numeric(selected.get("feat_lookup_count", 0.0), errors="coerce").fillna(0.0).sum()),
        "cache_size_working_set_scale": float(cache_size_working_set_scale) if spec.group == "cache_size" else np.nan,
        "workload_design": str(workload_design),
        "fixed_workload_id": _fixed_workload_id(workload_design, workload_variant),
        "hardware_factor_profile": str(hardware_factor_profile),
        "target_active_parallel_workers": float(active_worker_count) if active_worker_count is not None else np.nan,
        "active_parallel_workers": _active_workers_from_config(hw, config, active_worker_count),
        **_gather_working_set_summary(selected, hw),
    }
    return selected, metadata


def _e2e_frame(
    spec: TrendSpec,
    hw: dict[str, float],
    *,
    workload_variant: WorkloadVariant = WorkloadVariant("base", "base", 1.0),
    cache_size_working_set_scale: float = 1.03,
    workload_design: str = "per_trend_probe",
    hardware_factor_profile: str = "current",
    active_worker_count: int | float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, DlrmStaticConfig, dict[str, Any]]:
    config = _workload_config_for_variant(
        spec,
        hw,
        workload_variant=workload_variant,
        cache_size_working_set_scale=cache_size_working_set_scale,
        workload_design=workload_design,
        active_worker_count=active_worker_count,
    )
    frame = _specialize_frame_for_trend(spec, build_static_feature_frame(config, graph_scope=FULL_GRAPH_76))
    op_shapes = build_static_op_shape_table(config)
    non_constant = frame.loc[~frame["op_type"].astype(str).eq("Constant")].copy()
    metadata = {
        "workload_variant": str(workload_variant.key),
        "workload_variant_label": str(workload_variant.label),
        "workload_variant_scale": float(workload_variant.scale),
        "batch_size": int(config.batch_size),
        "num_indices_per_lookup": int(config.num_indices_per_lookup),
        "arch_embedding_size": str(config.arch_embedding_size),
        "arch_mlp_bot": str(config.arch_mlp_bot),
        "arch_mlp_top": str(config.arch_mlp_top),
        "num_threads": int(config.num_threads),
        "inter_threads": int(config.inter_threads),
        "selected_ops": "full_graph_scheduled",
        "selected_rows": int(len(non_constant)),
        "selected_output_bytes": float(pd.to_numeric(non_constant.get("exact_output_tensor_bytes_sum", 0.0), errors="coerce").fillna(0.0).sum()),
        "selected_read_bytes": float(pd.to_numeric(non_constant.get("exact_read_bytes", 0.0), errors="coerce").fillna(0.0).sum()),
        "selected_write_bytes": float(pd.to_numeric(non_constant.get("exact_write_bytes", 0.0), errors="coerce").fillna(0.0).sum()),
        "selected_lookup_count": float(pd.to_numeric(non_constant.get("feat_lookup_count", 0.0), errors="coerce").fillna(0.0).sum()),
        "full_graph_rows": int(len(frame)),
        "scheduled_non_constant_rows": int(len(non_constant)),
        "schedule_model": "dlrm_static_dag_branch_pool_barrier_tail",
        "cache_size_working_set_scale": float(cache_size_working_set_scale) if spec.group == "cache_size" else np.nan,
        "workload_design": str(workload_design),
        "fixed_workload_id": _fixed_workload_id(workload_design, workload_variant),
        "hardware_factor_profile": str(hardware_factor_profile),
        "target_active_parallel_workers": float(active_worker_count) if active_worker_count is not None else np.nan,
        "active_parallel_workers": _active_workers_from_config(hw, config, active_worker_count),
        **_gather_working_set_summary(non_constant, hw),
    }
    return frame, op_shapes, config, metadata


def _trend_pass(values: np.ndarray, relation: str, *, tol: float = 1e-6) -> bool:
    diff = np.diff(np.asarray(values, dtype=float))
    if relation == "capacity":
        return bool(np.all(diff <= tol))
    if relation == "cost":
        return bool(np.all(diff >= -tol))
    return False


def _sweep_values_for_spec(spec: TrendSpec, hw: dict[str, float]) -> tuple[float, ...]:
    if spec.key == "active_parallel_workers":
        return _worker_sweep_values(hw)
    return tuple(float(value) for value in spec.factors)


def _project_monotonic_values(values: np.ndarray, relation: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if relation == "capacity":
        return np.minimum.accumulate(arr)
    if relation == "cost":
        return np.maximum.accumulate(arr)
    return arr.copy()


def _aggregate_prediction(prepared: pd.DataFrame, pred: np.ndarray, plot_source: str) -> tuple[float, float, dict[str, float]]:
    route_sum = float(np.asarray(pred, dtype=float).sum())
    prior_sum = float(pd.to_numeric(prepared["whitebox_center_us"], errors="coerce").fillna(0.0).sum())
    modules = {}
    for module in ("descriptor", "stream_mem", "random_mem", "compute", "reorder", "sync", "dispatch"):
        column = f"svc_{module}_us"
        modules[module] = float(pd.to_numeric(prepared.get(column, 0.0), errors="coerce").fillna(0.0).sum())
    selected = route_sum if plot_source == "route_pred" else prior_sum
    return selected, prior_sum, modules


def _aggregate_scheduled_prediction(
    *,
    prepared: pd.DataFrame,
    pred: np.ndarray,
    plot_source: str,
    config: DlrmStaticConfig,
    op_shapes: pd.DataFrame,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    work = prepared.copy()
    route_pred = np.asarray(pred, dtype=float)
    prior_pred = pd.to_numeric(work["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    selected = route_pred if plot_source == "route_pred" else prior_pred
    work["pred_us"] = np.clip(selected, a_min=0.0, a_max=None)
    prediction_rows = work.loc[~work["op_type"].astype(str).eq("Constant"), ["op_idx", "node_name", "op_type", "pred_us"]].copy()
    graph = build_op_graph(op_shapes)
    artifact_paths = ComboArtifactPaths(
        case_id=str(config.case_id),
        combo=str(config.combo),
        shape_csv=Path("."),
        profile_dir=Path("."),
        timeline_csv=Path("."),
    )
    combo_spec = ComboSpec(
        case_id=str(config.case_id),
        combo=str(config.combo),
        batch_size=int(config.batch_size),
        num_indices_per_lookup=int(config.num_indices_per_lookup),
        inter_threads=int(config.inter_threads),
        artifact_paths=artifact_paths,
    )
    schedule_result = schedule_combo(combo_spec, graph, prediction_rows)

    prior_work = work.copy()
    prior_work["pred_us"] = np.clip(prior_pred, a_min=0.0, a_max=None)
    prior_schedule = schedule_combo(
        combo_spec,
        graph,
        prior_work.loc[~prior_work["op_type"].astype(str).eq("Constant"), ["op_idx", "node_name", "op_type", "pred_us"]].copy(),
    )

    modules = {}
    for module in ("descriptor", "stream_mem", "random_mem", "compute", "reorder", "sync", "dispatch"):
        column = f"svc_{module}_us"
        modules[module] = float(pd.to_numeric(work.get(column, 0.0), errors="coerce").fillna(0.0).sum())
    schedule_meta = {
        "scheduled_pred_full_graph_us": float(schedule_result.predicted_full_graph_us),
        "scheduled_pred_observed_subgraph_us": float(schedule_result.predicted_observed_subgraph_us),
        "scheduled_prior_full_graph_us": float(prior_schedule.predicted_full_graph_us),
        "coverage_ratio": float(schedule_result.coverage.coverage_ratio),
        "coverage_missing_count": float(schedule_result.coverage.missing_count),
        "bottom_end_us": float(schedule_result.bottom_end_us),
        "branch_pool_end_us": float(schedule_result.branch_pool_end_us),
        "tail_barrier_us": float(schedule_result.tail_barrier_us),
    }
    return (
        float(schedule_result.predicted_full_graph_us),
        float(prior_schedule.predicted_full_graph_us),
        modules,
        schedule_meta,
    )


def _annotation_lines(parts: list[str]) -> str:
    if len(parts) <= 2:
        return " | ".join(parts)
    return " | ".join(parts[:2]) + "\n" + " | ".join(parts[2:])


def _first_numeric(frame: pd.DataFrame, column: str, default: float) -> float:
    if column not in frame.columns:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float(default)
    return float(values.iloc[0])


def _panel_workload_annotation(spec: TrendSpec, subset: pd.DataFrame) -> str:
    base_cache_scale = _first_numeric(subset, "cache_size_working_set_scale", 1.03)
    level = spec.target_cache_level or "cache"
    if spec.key == "active_parallel_workers":
        label_by_variant = {"small": "S", "base": "B", "large": "L"}
        parts: list[str] = []
        for variant in ("small", "base", "large"):
            rows = subset.loc[subset["workload_variant"].astype(str).eq(variant)] if "workload_variant" in subset.columns else pd.DataFrame()
            if rows.empty:
                continue
            first = rows.iloc[0]
            batch = int(float(first.get("batch_size", 0)))
            nip = int(float(first.get("num_indices_per_lookup", 0)))
            mlp = str(first.get("arch_mlp_bot", ""))
            parts.append(f"{label_by_variant[variant]}: bs={batch}, nip={nip}, MLP={mlp}")
        if parts:
            return "\n".join(parts)
        return _annotation_lines(["S/B/L: workers swept to 60"])
    if spec.workload_kind.startswith("cache_size_"):
        return _annotation_lines(
            [
                f"S: WS~1.10x {level}",
                f"B: WS~{base_cache_scale:.2f}x {level}",
                f"L: WS~1.80x {level}",
            ]
        )
    if spec.workload_kind.startswith("cache_latency_"):
        return _annotation_lines(
            [
                f"S: resident WS~0.45x {level}",
                f"B: resident WS~0.70x {level}",
                f"L: resident WS~0.95x {level}",
            ]
        )
    if spec.workload_kind == "stream_copy":
        return _annotation_lines(
            [
                "S: stream copy bs=2048, rows=32768",
                "B: bs=4096, rows=65536",
                "L: bs=8192, rows=131072",
            ]
        )
    if spec.workload_kind == "memory_random":
        return _annotation_lines(
            [
                "S: Gather WS~1.40x active L3",
                "B: WS~2.20x active L3",
                "L: WS~3.00x active L3",
            ]
        )
    if spec.workload_kind == "memory_latency":
        return _annotation_lines(
            [
                "S: DRAM Gather WS~1.10x L3",
                "B: WS~1.60x L3",
                "L: WS~2.20x L3",
            ]
        )
    if spec.workload_kind == "compute_gemm":
        return _annotation_lines(
            [
                "S: Gemm bs=1024, MLP=256-1024-256",
                "B: bs=2048, MLP=512-2048-512",
                "L: bs=4096, MLP=1024-4096-1024",
            ]
        )
    if spec.workload_kind == "metadata_pipeline":
        return _annotation_lines(
            [
                "S: metadata bs=16, rows=512",
                "B: bs=32, rows=1024",
                "L: bs=64, rows=2048",
            ]
        )
    return ""


def _plot_group(
    group: str,
    specs: list[TrendSpec],
    points: pd.DataFrame,
    output_root: Path,
    plot_source: str,
    route: str,
    bundle_source: str,
    prediction_scope: str,
    hide_device_label: bool = False,
) -> None:
    figure_name, _ = GROUP_TO_FIGURE[group]
    n = len(specs)
    cols = 3 if n >= 3 else n
    rows = int(math.ceil(n / max(cols, 1)))
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if group == "core" and n == 1:
        figsize = (7.2, 4.8)
    else:
        figsize = (3.15 * cols, 2.58 * rows)
    centered_two_panel_bottom = cols == 3 and rows == 2 and n == 5
    if centered_two_panel_bottom:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(2, 6)
        axes_flat = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[0, 4:6]),
            fig.add_subplot(grid[1, 1:3]),
            fig.add_subplot(grid[1, 3:5]),
        ]
        extra_axes: list[Any] = []
    else:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, constrained_layout=True)
        axes_flat = list(axes.flat)
        extra_axes = list(axes.flat[n:])
    palette = {"s920x20_local": "#2f5f9e", "kunpeng920_remote": "#b65a2a"}
    variant_style = {
        "small": ("--", "s", 0.85),
        "base": ("-", "o", 1.0),
        "large": (":", "^", 0.9),
    }
    for panel_idx, (ax, spec) in enumerate(zip(axes_flat, specs, strict=False)):
        subset = points.loc[points["trend_key"].astype(str).eq(spec.key)].copy()
        if subset.empty:
            ax.text(0.5, 0.5, "No synthetic points", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if "workload_variant" not in subset.columns:
            subset["workload_variant"] = "base"
            subset["workload_variant_label"] = "base"
        for (alias, variant), case_points in subset.groupby(["dataset_alias", "workload_variant"], sort=False):
            case_points = case_points.sort_values("relative_parameter")
            machine = "Device_2" if str(alias) == "s920x20_local" else "Device_1"
            variant_label = str(case_points.get("workload_variant_label", pd.Series([variant])).iloc[0])
            linestyle, marker, alpha = variant_style.get(str(variant), ("-", "o", 1.0))
            label = variant_label if hide_device_label else f"{machine} {variant_label}"
            ax.plot(
                case_points["relative_parameter"],
                case_points["normalized_pred"],
                marker=marker,
                color=palette.get(str(alias), "#2f5f9e"),
                linestyle=linestyle,
                alpha=alpha,
                label=label,
            )
            base = case_points.iloc[(case_points["relative_parameter"] - 1.0).abs().argsort()[:1]]
            if not base.empty:
                ax.scatter(base["relative_parameter"], base["normalized_pred"], s=28, color="red", zorder=5, edgecolor="white", linewidth=0.5)
        ax.axvline(1.0, color="0.55", linestyle="--", linewidth=0.8)
        ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_title(spec.title)
        if spec.key == "active_parallel_workers":
            ax.set_xlabel(spec.x_label)
        else:
            ax.set_xlabel(f"{spec.x_label} (baseline=1)")
        annotation = _panel_workload_annotation(spec, subset)
        if annotation:
            ax.text(
                0.5,
                -0.36,
                annotation,
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=5.15,
                color="0.25",
                linespacing=1.15,
                clip_on=False,
            )
        if panel_idx % cols == 0:
            ylabel = "Normalized scheduled E2E time" if prediction_scope == "e2e_pipeline" else "Normalized predicted workload time"
            ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(True, color="0.85", linewidth=0.7)
        ax.legend(frameon=False, loc="best")
    for ax in extra_axes:
        ax.set_axis_off()
    if prediction_scope == "e2e_pipeline":
        stem = f"synthetic_e2e_{figure_name.removeprefix('synthetic_')}_{plot_source}"
    else:
        stem = f"{figure_name}_{plot_source}"
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    if group == "compute_pipeline" and n == 5:
        active_parallel_spec = next((spec for spec in TREND_SPECS if spec.key == "active_parallel_workers"), None)
        pipeline_width_spec = next((spec for spec in specs if spec.key == "pipeline_width"), None)
        parallel_pipeline_specs = [
            spec
            for spec in (active_parallel_spec, pipeline_width_spec)
            if spec is not None and points["trend_key"].astype(str).eq(spec.key).any()
        ]
        _plot_group_subset(
            specs[:3],
            points,
            output_root,
            plot_source,
            prediction_scope,
            f"{stem}_compute_capacity",
            cols=3,
            hide_device_label=hide_device_label,
        )
        if parallel_pipeline_specs:
            _plot_group_subset(
                parallel_pipeline_specs,
                points,
                output_root,
                plot_source,
                prediction_scope,
                f"{stem}_latency_pipeline",
                cols=2,
                hide_device_label=hide_device_label,
            )


def _plot_group_subset(
    specs: list[TrendSpec],
    points: pd.DataFrame,
    output_root: Path,
    plot_source: str,
    prediction_scope: str,
    stem: str,
    *,
    cols: int,
    hide_device_label: bool = False,
) -> None:
    rows = int(math.ceil(len(specs) / max(cols, 1)))
    figsize = (3.15 * cols, 2.58 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, constrained_layout=True)
    palette = {"s920x20_local": "#2f5f9e", "kunpeng920_remote": "#b65a2a"}
    variant_style = {
        "small": ("--", "s", 0.85),
        "base": ("-", "o", 1.0),
        "large": (":", "^", 0.9),
    }
    for panel_idx, (ax, spec) in enumerate(zip(axes.flat, specs, strict=False)):
        subset = points.loc[points["trend_key"].astype(str).eq(spec.key)].copy()
        if subset.empty:
            ax.text(0.5, 0.5, "No synthetic points", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if "workload_variant" not in subset.columns:
            subset["workload_variant"] = "base"
            subset["workload_variant_label"] = "base"
        for (alias, variant), case_points in subset.groupby(["dataset_alias", "workload_variant"], sort=False):
            case_points = case_points.sort_values("relative_parameter")
            machine = "Device_2" if str(alias) == "s920x20_local" else "Device_1"
            variant_label = str(case_points.get("workload_variant_label", pd.Series([variant])).iloc[0])
            linestyle, marker, alpha = variant_style.get(str(variant), ("-", "o", 1.0))
            label = variant_label if hide_device_label else f"{machine} {variant_label}"
            ax.plot(
                case_points["relative_parameter"],
                case_points["normalized_pred"],
                marker=marker,
                color=palette.get(str(alias), "#2f5f9e"),
                linestyle=linestyle,
                alpha=alpha,
                label=label,
            )
            base = case_points.iloc[(case_points["relative_parameter"] - 1.0).abs().argsort()[:1]]
            if not base.empty:
                ax.scatter(
                    base["relative_parameter"],
                    base["normalized_pred"],
                    s=28,
                    color="red",
                    zorder=5,
                    edgecolor="white",
                    linewidth=0.5,
                )
        ax.axvline(1.0, color="0.55", linestyle="--", linewidth=0.8)
        ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_title(spec.title)
        if spec.key == "active_parallel_workers":
            ax.set_xlabel(spec.x_label)
        else:
            ax.set_xlabel(f"{spec.x_label} (baseline=1)")
        annotation = _panel_workload_annotation(spec, subset)
        if annotation:
            ax.text(
                0.5,
                -0.36,
                annotation,
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=5.15,
                color="0.25",
                linespacing=1.15,
                clip_on=False,
            )
        if panel_idx % cols == 0:
            ylabel = "Normalized scheduled E2E time" if prediction_scope == "e2e_pipeline" else "Normalized predicted workload time"
            ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(True, color="0.85", linewidth=0.7)
        ax.legend(frameon=False, loc="best")
    for ax in axes.flat[len(specs):]:
        ax.set_axis_off()
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _run_for_binding(
    *,
    binding: DatasetBinding,
    config: dict[str, Any],
    policy: str,
    scale: int,
    passes: int,
    op_thread_scale_mode: str,
    plot_source: str,
    route: str,
    bundle_source: str,
    prediction_scope: str,
    specs: tuple[TrendSpec, ...],
    workload_variants: tuple[WorkloadVariant, ...],
    cache_size_working_set_scale: float,
    workload_design: str,
    hardware_factor_profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bundle_path: Path | None = None
    if str(bundle_source) == "binding":
        print(f"[synthetic-trend] load_bundle {binding.alias}", flush=True)
        bundle, bundle_path = _load_bundle_from_binding(binding, str(route))
        if str(op_thread_scale_mode) == "fit":
            print(
                f"[synthetic-trend] note {binding.alias}: --op-thread-scale-mode is ignored when --bundle-source=binding",
                flush=True,
            )
    else:
        print(f"[synthetic-trend] fit {binding.alias}", flush=True)
        fit_frame, _, _, _ = _prepare_semantic_frames(
            binding=binding,
            config=config,
            policy=policy,
            hw_config=binding.hw_config,
            scale=scale,
            passes=passes,
        )
        op_thread_scale_config = None
        if op_thread_scale_mode == "fit":
            op_thread_scale_config = fit_op_thread_scale_config(fit_frame, binding.hw_config, policy=policy)
        bundle = _fit_route_bundle(str(route), fit_frame, binding.hw_config, policy, op_thread_scale_config)
    base_payload = _load_hw_payload(binding.hw_config)
    hw = _load_hw_params(binding.hw_config)
    bundle_meta = {
        "bundle_source": str(bundle_source),
        "model_bundle_path": str(bundle_path or ""),
        "model_sha256": str(bundle.get("model_sha256", "")),
        "model_bundle_export_hw_config_path": _bundle_export_hw_config_path(bundle),
        "model_bundle_hw_config_mismatch": bool(_bundle_hw_config_mismatch(bundle, binding.hw_config)),
    }
    if bool(bundle_meta["model_bundle_hw_config_mismatch"]):
        print(
            f"[synthetic-trend] warning {binding.alias}: binding hw_config differs from bundle export hw_config",
            flush=True,
        )

    case_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    for spec in specs:
        for workload_variant in workload_variants:
            active_worker_sweep = spec.key == "active_parallel_workers"
            sweep_values = _sweep_values_for_spec(spec, hw)
            initial_worker_count = sweep_values[0] if active_worker_sweep and sweep_values else None
            if prediction_scope == "e2e_pipeline":
                frame, op_shapes, workload_config, workload_meta = _e2e_frame(
                    spec,
                    hw,
                    workload_variant=workload_variant,
                    cache_size_working_set_scale=cache_size_working_set_scale,
                    workload_design=workload_design,
                    hardware_factor_profile=hardware_factor_profile,
                    active_worker_count=initial_worker_count,
                )
            else:
                frame, workload_meta = _target_frame(
                    spec,
                    hw,
                    workload_variant=workload_variant,
                    cache_size_working_set_scale=cache_size_working_set_scale,
                    workload_design=workload_design,
                    hardware_factor_profile=hardware_factor_profile,
                    active_worker_count=initial_worker_count,
                )
                op_shapes = pd.DataFrame()
                workload_config = None
            print(
                f"[synthetic-trend] sweep {binding.alias} {spec.key} variant={workload_variant.key} rows={len(frame)}",
                flush=True,
            )
            factor_predictions: list[float] = []
            factor_priors: list[float] = []
            factor_modules: list[dict[str, float]] = []
            factor_schedule_meta: list[dict[str, float]] = []
            factor_workload_meta: list[dict[str, Any]] = []
            factor_axis_values: list[float] = []
            mutated_values: list[str] = []
            original_value = ""
            for factor in sweep_values:
                if active_worker_sweep:
                    if prediction_scope == "e2e_pipeline":
                        frame, op_shapes, workload_config, workload_meta = _e2e_frame(
                            spec,
                            hw,
                            workload_variant=workload_variant,
                            cache_size_working_set_scale=cache_size_working_set_scale,
                            workload_design=workload_design,
                            hardware_factor_profile=hardware_factor_profile,
                            active_worker_count=float(factor),
                        )
                    else:
                        frame, workload_meta = _target_frame(
                            spec,
                            hw,
                            workload_variant=workload_variant,
                            cache_size_working_set_scale=cache_size_working_set_scale,
                            workload_design=workload_design,
                            hardware_factor_profile=hardware_factor_profile,
                            active_worker_count=float(factor),
                        )
                        op_shapes = pd.DataFrame()
                        workload_config = None
                payload, original_value, mutated_value = _mutate_payload(base_payload, spec, float(factor))
                prepared, pred = _predict_route(str(route), bundle, frame, override_hw_config=payload)
                if "physics_guard_action" in prepared.columns:
                    actions = prepared["physics_guard_action"].fillna("none").astype(str)
                    applied = (
                        pd.to_numeric(prepared.get("physics_guard_applied", False), errors="coerce")
                        .fillna(0)
                        .astype(bool)
                    )
                    for action, count in actions.value_counts(dropna=False).items():
                        action_mask = actions == str(action)
                        guard_rows.append(
                            {
                                "dataset_alias": binding.alias,
                                "actual_hardware": binding.actual_hardware,
                                "route": str(route),
                                "trend_key": spec.key,
                                "trend_group": spec.group,
                                "workload_variant": str(workload_variant.key),
                                "workload_variant_label": str(workload_variant.label),
                                "workload_design": str(workload_design),
                                "fixed_workload_id": _fixed_workload_id(workload_design, workload_variant),
                                "hardware_factor_profile": str(hardware_factor_profile),
                                **bundle_meta,
                                "relative_parameter": float(factor),
                                "guard_action": str(action),
                                "rows": int(count),
                                "guard_applied_rows": int(applied[action_mask].sum()),
                            }
                        )
                if prediction_scope == "e2e_pipeline":
                    if workload_config is None:
                        raise RuntimeError("E2E workload config is missing")
                    selected_pred, prior_pred, modules, schedule_meta = _aggregate_scheduled_prediction(
                        prepared=prepared,
                        pred=pred,
                        plot_source=str(plot_source),
                        config=workload_config,
                        op_shapes=op_shapes,
                    )
                else:
                    selected_pred, prior_pred, modules = _aggregate_prediction(prepared, pred, str(plot_source))
                    schedule_meta = {}
                factor_predictions.append(float(selected_pred))
                factor_priors.append(float(prior_pred))
                factor_modules.append(modules)
                factor_schedule_meta.append(schedule_meta)
                factor_workload_meta.append(dict(workload_meta))
                factor_axis_values.append(
                    float(workload_meta.get("active_parallel_workers", factor)) if active_worker_sweep else float(factor)
                )
                mutated_values.append(mutated_value)
            factors = np.asarray(factor_axis_values, dtype=float)
            raw_values = np.asarray(factor_predictions, dtype=float)
            use_aggregate_projection = str(route) == WHITEBOX_PHYSICS_GUARDED_RF_ROUTE and str(plot_source) == "route_pred"
            values = _project_monotonic_values(raw_values, spec.relation) if use_aggregate_projection else raw_values
            baseline_idx = int(np.argmin(np.abs(factors - 1.0)))
            baseline = max(float(values[baseline_idx]), 1e-9)
            norm = values / baseline
            amplitude = float(np.nanmax(norm) - np.nanmin(norm))
            monotonic_pass = _trend_pass(norm, spec.relation)
            modules_at_base = factor_modules[baseline_idx]
            schedule_at_base = factor_schedule_meta[baseline_idx]
            workload_meta_at_base = factor_workload_meta[baseline_idx] if factor_workload_meta else dict(workload_meta)
            dominant_module = max(modules_at_base.items(), key=lambda item: item[1])[0] if modules_at_base else ""
            case_row = {
                "trend_key": spec.key,
                "trend_group": spec.group,
                "trend_title": spec.title,
                "dataset_alias": binding.alias,
                "actual_hardware": binding.actual_hardware,
                "route": str(route),
                "hw_config": str(binding.hw_config),
                **bundle_meta,
                "plot_source": str(plot_source),
                "prediction_scope": str(prediction_scope),
                "aggregate_label": spec.aggregate_label,
                "workload_kind": spec.workload_kind,
                "target_cache_level": spec.target_cache_level,
                "baseline_pred_us": float(baseline),
                "baseline_raw_pred_us": float(raw_values[baseline_idx]),
                "baseline_prior_us": float(factor_priors[baseline_idx]),
                "normalized_amplitude": amplitude,
                "monotonic_pass": bool(monotonic_pass),
                "aggregate_monotonic_projection": bool(use_aggregate_projection),
                "aggregate_projection_max_abs_delta_us": float(np.nanmax(np.abs(values - raw_values))) if len(values) else 0.0,
                "dominant_module_at_baseline": dominant_module,
                "original_value": original_value,
                **workload_meta_at_base,
            }
            for module, value in modules_at_base.items():
                case_row[f"baseline_svc_{module}_us"] = float(value)
            for key, value in schedule_at_base.items():
                case_row[f"baseline_{key}"] = float(value)
            case_rows.append(case_row)
            for factor, pred_us, raw_pred_us, prior_us, mutated, modules, schedule_meta, point_workload_meta in zip(
                factors,
                values,
                raw_values,
                factor_priors,
                mutated_values,
                factor_modules,
                factor_schedule_meta,
                factor_workload_meta,
                strict=False,
            ):
                point = {
                    "trend_key": spec.key,
                    "trend_group": spec.group,
                    "trend_title": spec.title,
                    "dataset_alias": binding.alias,
                    "actual_hardware": binding.actual_hardware,
                    "route": str(route),
                    **bundle_meta,
                    "plot_source": str(plot_source),
                    "prediction_scope": str(prediction_scope),
                    "aggregate_label": spec.aggregate_label,
                    "workload_kind": spec.workload_kind,
                    "workload_variant": str(workload_variant.key),
                    "workload_variant_label": str(workload_variant.label),
                    "workload_variant_scale": float(workload_variant.scale),
                    "workload_design": str(workload_design),
                    "fixed_workload_id": _fixed_workload_id(workload_design, workload_variant),
                    "hardware_factor_profile": str(hardware_factor_profile),
                    "relative_parameter": float(factor),
                    "pred_us": float(pred_us),
                    "raw_pred_us": float(raw_pred_us),
                    "whitebox_prior_us": float(prior_us),
                    "residual_vs_prior_us": float(pred_us - prior_us),
                    "raw_residual_vs_prior_us": float(raw_pred_us - prior_us),
                    "normalized_pred": float(pred_us / baseline),
                    "aggregate_monotonic_projection": bool(use_aggregate_projection),
                    "aggregate_projection_delta_us": float(pred_us - raw_pred_us),
                    "baseline_pred_us": float(baseline),
                    "original_value": original_value,
                    "mutated_value": mutated,
                    "relation": spec.relation,
                }
                for key, value in point_workload_meta.items():
                    point.setdefault(key, value)
                for module, value in modules.items():
                    point[f"svc_{module}_us"] = float(value)
                for key, value in schedule_meta.items():
                    point[key] = float(value)
                point_rows.append(point)
    return case_rows, point_rows, guard_rows


def main() -> None:
    args = _parse_args()
    env_used = _apply_sensitivity_env_defaults()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.binding_config).resolve()
    config, all_bindings = _load_bindings(config_path)
    policy = str(args.policy or config.get("policy") or "joint_6d_tuple")
    requested = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    bindings = {alias: all_bindings[alias] for alias in requested}
    requested_groups = {item.strip() for item in str(args.groups).split(",") if item.strip()}
    specs = tuple(spec for spec in TREND_SPECS if not requested_groups or spec.group in requested_groups)
    specs = _specs_for_factor_profile(specs, str(args.hardware_factor_profile))
    if not specs:
        raise ValueError(f"No trend specs matched --groups={args.groups!r}")
    workload_variants = _workload_variants(str(args.workload_variants), str(args.workload_design))

    all_case_rows: list[dict[str, Any]] = []
    all_point_rows: list[dict[str, Any]] = []
    all_guard_rows: list[dict[str, Any]] = []
    for binding in bindings.values():
        case_rows, point_rows, guard_rows = _run_for_binding(
            binding=binding,
            config=config,
            policy=policy,
            scale=int(args.scale),
            passes=int(args.passes),
            op_thread_scale_mode=str(args.op_thread_scale_mode),
            plot_source=str(args.plot_source),
            route=str(args.route),
            bundle_source=str(args.bundle_source),
            prediction_scope=str(args.prediction_scope),
            specs=specs,
            workload_variants=workload_variants,
            cache_size_working_set_scale=float(args.cache_size_working_set_scale),
            workload_design=str(args.workload_design),
            hardware_factor_profile=str(args.hardware_factor_profile),
        )
        all_case_rows.extend(case_rows)
        all_point_rows.extend(point_rows)
        all_guard_rows.extend(guard_rows)

    case_frame = pd.DataFrame(all_case_rows)
    point_frame = pd.DataFrame(all_point_rows)
    guard_frame = pd.DataFrame(all_guard_rows)
    name_prefix = "synthetic_e2e_trend" if str(args.prediction_scope) == "e2e_pipeline" else "synthetic_trend"
    case_frame.to_csv(output_root / f"{name_prefix}_case_index_{args.plot_source}.csv", index=False)
    point_frame.to_csv(output_root / f"{name_prefix}_sweep_points_{args.plot_source}.csv", index=False)
    guard_frame.to_csv(output_root / "hardware_response_guard_audit.csv", index=False)

    workload_columns = [
        "trend_group",
        "trend_key",
        "trend_title",
        "dataset_alias",
        "actual_hardware",
        "bundle_source",
        "model_bundle_path",
        "model_sha256",
        "model_bundle_export_hw_config_path",
        "model_bundle_hw_config_mismatch",
        "workload_variant",
        "workload_variant_label",
        "workload_kind",
        "workload_design",
        "fixed_workload_id",
        "hardware_factor_profile",
        "target_cache_level",
        "batch_size",
        "num_indices_per_lookup",
        "arch_embedding_size",
        "arch_mlp_bot",
        "arch_mlp_top",
        "num_threads",
        "inter_threads",
        "target_active_parallel_workers",
        "active_parallel_workers",
        "selected_rows",
        "selected_output_bytes",
        "selected_read_bytes",
        "selected_write_bytes",
        "selected_lookup_count",
        "full_graph_rows",
        "scheduled_non_constant_rows",
        "schedule_model",
        "cache_size_working_set_scale",
        "gather_working_set_bytes_median",
        "gather_working_set_bytes_max",
        "ratio_working_set_to_l1d",
        "ratio_working_set_to_l2",
        "ratio_working_set_to_l3",
        "baseline_pred_us",
        "baseline_prior_us",
        "dominant_module_at_baseline",
    ]
    workload_manifest = case_frame[[column for column in workload_columns if column in case_frame.columns]].drop_duplicates()
    workload_manifest.to_csv(output_root / f"{name_prefix}_workload_manifest_{args.plot_source}.csv", index=False)

    hw_sweep_columns = [
        "trend_group",
        "trend_key",
        "trend_title",
        "dataset_alias",
        "actual_hardware",
        "bundle_source",
        "model_bundle_path",
        "model_sha256",
        "model_bundle_export_hw_config_path",
        "model_bundle_hw_config_mismatch",
        "workload_variant",
        "workload_design",
        "fixed_workload_id",
        "hardware_factor_profile",
        "relative_parameter",
        "target_active_parallel_workers",
        "active_parallel_workers",
        "relation",
        "original_value",
        "mutated_value",
    ]
    hw_sweep_manifest = point_frame[[column for column in hw_sweep_columns if column in point_frame.columns]].drop_duplicates()
    hw_sweep_manifest.to_csv(output_root / f"{name_prefix}_hardware_sweep_manifest_{args.plot_source}.csv", index=False)
    hw_range_summary = pd.DataFrame(
        [
            {
                "trend_group": spec.group,
                "trend_key": spec.key,
                "trend_title": spec.title,
                "fields": json.dumps(list(spec.fields), ensure_ascii=False),
                "relation": spec.relation,
                "factor_min": float(min(spec.factors)),
                "factor_max": float(max(spec.factors)),
                "factor_count": int(len(spec.factors)),
                "workload_kind": spec.workload_kind,
                "target_cache_level": spec.target_cache_level,
                "hardware_factor_profile": str(args.hardware_factor_profile),
            }
            for spec in specs
        ]
    )
    hw_range_summary.to_csv(output_root / f"{name_prefix}_hardware_range_summary_{args.plot_source}.csv", index=False)
    (output_root / f"{name_prefix}_workload_and_hw_manifest_{args.plot_source}.json").write_text(
        json.dumps(
            {
                "workload_manifest": workload_manifest.to_dict(orient="records"),
                "hardware_sweep_manifest": hw_sweep_manifest.to_dict(orient="records"),
                "hardware_range_summary": hw_range_summary.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for group in GROUP_TO_FIGURE:
        group_specs = [spec for spec in specs if spec.group == group]
        if not group_specs:
            continue
        _plot_group(
            group,
            group_specs,
            point_frame,
            output_root,
            str(args.plot_source),
            str(args.route),
            str(args.bundle_source),
            str(args.prediction_scope),
            hide_device_label=bool(args.hide_device_label),
        )

    summary = (
        case_frame.groupby(["trend_group", "trend_key"], dropna=False)
        .agg(
            cases=("dataset_alias", "count"),
            pass_rate=("monotonic_pass", "mean"),
            median_amp=("normalized_amplitude", "median"),
            min_amp=("normalized_amplitude", "min"),
            max_amp=("normalized_amplitude", "max"),
        )
        .reset_index()
        if not case_frame.empty
        else pd.DataFrame()
    )
    summary.to_csv(output_root / f"{name_prefix}_summary_{args.plot_source}.csv", index=False)
    (output_root / f"{name_prefix}_env_{args.plot_source}.json").write_text(
        json.dumps(
            {
                "route": str(args.route),
                "bundle_source": str(args.bundle_source),
                "plot_source": str(args.plot_source),
                "prediction_scope": str(args.prediction_scope),
                "policy": policy,
                "env": env_used,
                "binding_config": str(config_path),
                "datasets": requested,
                "bundle_manifest": (
                    case_frame[
                        [
                            column
                            for column in (
                                "dataset_alias",
                                "actual_hardware",
                                "bundle_source",
                                "model_bundle_path",
                                "model_sha256",
                                "model_bundle_export_hw_config_path",
                                "model_bundle_hw_config_mismatch",
                                "hw_config",
                            )
                            if column in case_frame.columns
                        ]
                    ]
                    .drop_duplicates()
                    .to_dict(orient="records")
                    if not case_frame.empty
                    else []
                ),
                "groups": sorted(requested_groups) if requested_groups else "all",
                "workload_variants": [
                    {"key": variant.key, "label": variant.label, "scale": variant.scale}
                    for variant in workload_variants
                ],
                "workload_design": str(args.workload_design),
                "hardware_factor_profile": str(args.hardware_factor_profile),
                "trend_specs": [
                    {
                        "key": spec.key,
                        "group": spec.group,
                        "title": spec.title,
                        "fields": list(spec.fields),
                        "relation": spec.relation,
                        "factors": list(spec.factors),
                        "workload_kind": spec.workload_kind,
                        "target_cache_level": spec.target_cache_level,
                    }
                    for spec in specs
                ],
                "cache_size_working_set_scale": float(args.cache_size_working_set_scale),
                "note": "Synthetic software workloads are generated from static DLRM/operator shapes. bundle_source=fit uses measured rows to fit the requested route; bundle_source=binding loads each model_bundle from the binding config and does not refit RF. e2e_pipeline scope schedules the full static DLRM DAG with branch-pool/barrier/tail timing instead of summing selected operators.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[synthetic-trend] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
