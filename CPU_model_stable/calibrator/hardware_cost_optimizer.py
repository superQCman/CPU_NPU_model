from __future__ import annotations

import argparse
import concurrent.futures
import copy
import itertools
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

try:
    import yaml
except ImportError:  # pragma: no cover - fail fast in main.
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrator.ml_semantic_routes import (  # noqa: E402
    WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
    WHITEBOX_PRIOR_ONLY_ROUTE,
)
from calibrator.static_feature_mapping.core import (  # noqa: E402
    MODEL_INPUT_51,
    DlrmStaticConfig,
    build_static_feature_frame,
    build_static_op_shape_table,
)
from calibrator.whitebox_family_routes import (  # noqa: E402
    PHYSICS_GUARD_DELTA_CAP_MULTIPLIER,
    PHYSICS_GUARD_DELTA_TOL_US,
    _load_hw_params,
    _predict_whitebox_family_rf_raw,
    _set_whitebox_rf_prediction_outputs,
    predict_whitebox_family_physics_guarded_rf,
    predict_whitebox_family_prior,
)
from cpu_perf_model.graph_contract import ComboArtifactPaths, ComboSpec  # noqa: E402
from cpu_perf_model.schedule_engine import build_op_graph, schedule_combo  # noqa: E402


BUNDLE_WEIGHTS = {
    "core": 0.25,
    "pipeline": 0.10,
    "vector": 0.15,
    "memory": 0.30,
    "cache": 0.20,
}

BUNDLE_GROUPS = ["core", "pipeline", "vector", "memory", "cache"]

HARDWARE_FIELDS = [
    "core.total_cores",
    "core.cpu_clock",
    "pipeline.fetch_width",
    "pipeline.decode_width",
    "pipeline.rename_width",
    "pipeline.dispatch_width",
    "pipeline.issue_width",
    "pipeline.commit_width",
    "instruction.simd_width_bits",
    "instruction.fp_throughput_per_cycle.vector_sp_add",
    "instruction.fp_throughput_per_cycle.vector_sp_mul",
    "instruction.fp_throughput_per_cycle.vector_sp_fma",
    "instruction.fp_latency_cycles.vector_sp_add",
    "instruction.fp_latency_cycles.vector_sp_mul",
    "instruction.fp_latency_cycles.vector_sp_fma",
    "memory.bandwidth_gb_s_total",
    "memory.copy_bw_gb_s_1t",
    "memory.copy_bw_gb_s_saturated",
    "memory.random_gather_bw_gb_s_1t",
    "memory.random_gather_bw_gb_s_saturated",
    "memory.local_mem_delay_ns",
    "cache.cacheline_bytes",
    "cache.l1d.size",
    "cache.l2.size",
    "cache.l3_per_die.size",
    "cache.l1d.response_latency_cycles",
    "cache.l2.response_latency_cycles",
    "cache.l3_per_die.response_latency_cycles",
]


@dataclass(frozen=True)
class BundleLevel:
    group: str
    level_id: str
    description: str
    values: dict[str, Any]
    cost_values: dict[str, float] | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    weight: float
    config: DlrmStaticConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Surrogate-cost constrained hardware search using existing trained "
            "whitebox_family_prior_plus_physics_guarded_rf_v3 bundles only."
        ),
    )
    parser.add_argument("--device1-bundle", default="", help="Existing Kunpeng920 v3 pickle bundle.")
    parser.add_argument("--device2-bundle", default="", help="Existing S920X20 v3 pickle bundle.")
    parser.add_argument("--device1-hw-config", required=True)
    parser.add_argument("--device2-hw-config", required=True)
    parser.add_argument("--device1-op-thread-scale-config", default="")
    parser.add_argument("--device2-op-thread-scale-config", default="")
    parser.add_argument("--budgets", default="1.0,1.1,1.2,1.3,1.5,1.75,2.0,device2")
    parser.add_argument("--workload-set", default="dlrm_s_m_l", choices=["dlrm_s_m_l"])
    parser.add_argument("--search-algorithm", default="grid", choices=["grid", "nsga2", "both"])
    parser.add_argument(
        "--cost-model",
        default="surrogate",
        choices=["surrogate", "reference_v2"],
        help=(
            "surrogate keeps the legacy relative proxy cost. reference_v2 uses the "
            "two-point customer parameter cost constraints from --cost-constraints."
        ),
    )
    parser.add_argument(
        "--cost-constraints",
        default="calibrator/configs/hardware_cost_constraints_v2.json",
        help="JSON hardware parameter cost constraints used by --cost-model reference_v2.",
    )
    parser.add_argument(
        "--pair-extrapolation-mode",
        default="allow",
        choices=["allow", "forbid"],
        help=(
            "For reference_v2, allow or forbid candidate parameter values outside "
            "the two customer-provided cost reference points. forbid is the "
            "conservative report-ready mode."
        ),
    )
    parser.add_argument(
        "--budget-unit",
        default="relative",
        choices=["relative", "wan"],
        help="Interpret numeric --budgets as relative cost or reference cost in 万元.",
    )
    parser.add_argument("--nsga2-population", type=int, default=64)
    parser.add_argument("--nsga2-generations", type=int, default=30)
    parser.add_argument("--nsga2-seed", type=int, default=20260503)
    parser.add_argument("--nsga2-crossover-prob", type=float, default=0.90)
    parser.add_argument("--nsga2-mutation-prob", type=float, default=0.20)
    parser.add_argument("--prediction-workers", type=int, default=1)
    parser.add_argument(
        "--grid-profile",
        default="full",
        choices=["full", "smoke"],
        help="Candidate grid size. full enumerates the complete bundled grid; smoke uses a small representative grid for validation.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-refit", action="store_true", help="Required: forbid model fitting inside optimizer.")
    parser.add_argument(
        "--allow-prior-only-fallback",
        action="store_true",
        help=(
            "If v3 bundles are missing, run lower-confidence prior-only optimization "
            "with optional op-thread scale configs. This does not use RF."
        ),
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for hardware cost optimization")
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for hardware cost optimization")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _nested_get(payload: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _nested_set(payload: dict[str, Any], dotted: str, value: Any) -> None:
    node: Any = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = node.get(part) if isinstance(node, dict) else None
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _parse_quantity(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    suffixes = (
        ("GHz", 1.0),
        ("MHz", 1.0 / 1000.0),
        ("ns", 1.0),
        ("KiB", 1024.0),
        ("MiB", 1024.0 * 1024.0),
        ("GiB", 1024.0 * 1024.0 * 1024.0),
    )
    for suffix, multiplier in suffixes:
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * multiplier
    return float(text)


def _numeric(payload: dict[str, Any], dotted: str, default: float = float("nan")) -> float:
    try:
        value = _parse_quantity(_nested_get(payload, dotted))
    except Exception:
        value = float("nan")
    return float(default if not np.isfinite(value) else value)


def _load_pickle_bundle(path_text: str, *, expected_route: str) -> dict[str, Any]:
    if not str(path_text).strip():
        raise FileNotFoundError(
            "model bundle not found; existing CSV summaries cannot restore RF weights. "
            "Run is aborted because --no-refit is enforced: <empty path>"
        )
    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(
            "model bundle not found; existing CSV summaries cannot restore RF weights. "
            f"Run is aborted because --no-refit is enforced: {path}"
        )
    if path.is_dir():
        raise FileNotFoundError(
            "model bundle path points to a directory, not a pickle file. "
            f"Run is aborted because --no-refit is enforced: {path}"
        )
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"model bundle must contain a dict, got {type(bundle).__name__}: {path}")
    route = str(bundle.get("route", "")).strip()
    if route != expected_route:
        raise ValueError(f"bundle route must be {expected_route}, got {route!r}: {path}")
    if not str(bundle.get("model_sha256", "")).strip():
        print(f"[hardware-cost] warning: bundle has empty model_sha256: {path}", file=sys.stderr)
    return bundle


def _load_scale_config(path_text: str) -> dict[str, Any] | None:
    if not str(path_text).strip():
        return None
    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(f"op-thread scale config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _prior_only_bundle(hw_config: Path, scale_config_path: str, *, label: str) -> dict[str, Any]:
    scale_config = _load_scale_config(scale_config_path)
    return {
        "route": WHITEBOX_PRIOR_ONLY_ROUTE,
        "route_whitebox_mode": "prior_only_no_refit_fallback",
        "fit_data_dependent": False,
        "policy": "hardware_cost_optimizer",
        "prior_variant": "legacy_v2",
        "hw_config_path": str(hw_config.resolve()),
        "hw_params": _load_hw_params(hw_config, require_measured_whitebox=False),
        "op_thread_scale_enabled": bool(scale_config),
        "op_thread_scale_config": scale_config,
        "model_sha256": f"prior_only_fallback_{label}",
    }


def _load_prediction_bundles(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], bool]:
    if not bool(args.no_refit):
        raise ValueError("--no-refit is required. This optimizer must not train or refit models.")
    try:
        device1_bundle = _load_pickle_bundle(str(args.device1_bundle), expected_route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE)
        device2_bundle = _load_pickle_bundle(str(args.device2_bundle), expected_route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE)
        return device1_bundle, device2_bundle, False
    except FileNotFoundError:
        if not bool(args.allow_prior_only_fallback):
            raise
        print(
            "[hardware-cost] warning: v3 bundle missing; using lower-confidence prior-only fallback.",
            file=sys.stderr,
        )
        device1_hw = Path(args.device1_hw_config).resolve()
        device2_hw = Path(args.device2_hw_config).resolve()
        return (
            _prior_only_bundle(device1_hw, str(args.device1_op_thread_scale_config), label="device1"),
            _prior_only_bundle(device2_hw, str(args.device2_op_thread_scale_config), label="device2"),
            True,
        )


def _ratio_capacity(value: float, baseline: float, exponent: float = 1.10) -> float:
    return float(max(value, 1e-9) / max(baseline, 1e-9)) ** float(exponent)


def _ratio_latency(value: float, baseline: float, exponent: float = 1.10) -> float:
    return float(max(baseline, 1e-9) / max(value, 1e-9)) ** float(exponent)


def _mean(values: list[float]) -> float:
    clean = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(clean)) if clean else 1.0


def _geom_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if np.isfinite(v) and float(v) > 0.0]
    if not clean:
        return 1.0
    return float(np.exp(np.mean(np.log(np.asarray(clean, dtype=float)))))


def _cost_components(payload: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    core_cost = _mean(
        [
            _ratio_capacity(_numeric(payload, "core.total_cores"), _numeric(baseline, "core.total_cores"), 1.05),
            _ratio_capacity(_numeric(payload, "core.cpu_clock"), _numeric(baseline, "core.cpu_clock"), 1.80),
        ]
    )
    pipeline_fields = [
        "pipeline.fetch_width",
        "pipeline.decode_width",
        "pipeline.rename_width",
        "pipeline.dispatch_width",
        "pipeline.issue_width",
        "pipeline.commit_width",
    ]
    pipeline_cost = _mean(
        [_ratio_capacity(_numeric(payload, field), _numeric(baseline, field), 1.25) for field in pipeline_fields]
    )
    vector_capacity_fields = [
        "instruction.simd_width_bits",
        "instruction.fp_throughput_per_cycle.vector_sp_add",
        "instruction.fp_throughput_per_cycle.vector_sp_mul",
        "instruction.fp_throughput_per_cycle.vector_sp_fma",
    ]
    vector_latency_fields = [
        "instruction.fp_latency_cycles.vector_sp_add",
        "instruction.fp_latency_cycles.vector_sp_mul",
        "instruction.fp_latency_cycles.vector_sp_fma",
    ]
    vector_cost = _mean(
        [_ratio_capacity(_numeric(payload, field), _numeric(baseline, field), 1.20) for field in vector_capacity_fields]
        + [_ratio_latency(_numeric(payload, field), _numeric(baseline, field), 1.20) for field in vector_latency_fields]
    )
    memory_capacity_fields = [
        "memory.bandwidth_gb_s_total",
        "memory.copy_bw_gb_s_1t",
        "memory.copy_bw_gb_s_saturated",
        "memory.random_gather_bw_gb_s_1t",
        "memory.random_gather_bw_gb_s_saturated",
    ]
    memory_cost = _mean(
        [_ratio_capacity(_numeric(payload, field), _numeric(baseline, field), 1.15) for field in memory_capacity_fields]
        + [
            _ratio_latency(
                _numeric(payload, "memory.local_mem_delay_ns"),
                _numeric(baseline, "memory.local_mem_delay_ns"),
                1.25,
            )
        ]
    )
    cache_size_fields = ["cache.l1d.size", "cache.l2.size", "cache.l3_per_die.size"]
    cache_latency_fields = [
        "cache.l1d.response_latency_cycles",
        "cache.l2.response_latency_cycles",
        "cache.l3_per_die.response_latency_cycles",
    ]
    cache_cost = _mean(
        [_ratio_capacity(_numeric(payload, field), _numeric(baseline, field), 1.10) for field in cache_size_fields]
        + [_ratio_latency(_numeric(payload, field), _numeric(baseline, field), 1.10) for field in cache_latency_fields]
    )
    total = (
        BUNDLE_WEIGHTS["core"] * core_cost
        + BUNDLE_WEIGHTS["pipeline"] * pipeline_cost
        + BUNDLE_WEIGHTS["vector"] * vector_cost
        + BUNDLE_WEIGHTS["memory"] * memory_cost
        + BUNDLE_WEIGHTS["cache"] * cache_cost
    )
    return {
        "cost_core": float(core_cost),
        "cost_pipeline": float(pipeline_cost),
        "cost_vector": float(vector_cost),
        "cost_memory": float(memory_cost),
        "cost_cache": float(cache_cost),
        "surrogate_cost": float(total),
    }


def _load_cost_constraints(path_text: str) -> dict[str, Any]:
    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(f"hardware cost constraints not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"hardware cost constraints must be a JSON object: {path}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError(f"hardware cost constraints must contain non-empty parameters: {path}")
    seen: set[str] = set()
    for entry in parameters:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid parameter entry in {path}: {entry!r}")
        pid = str(entry.get("id", "")).strip()
        if not pid:
            raise ValueError(f"parameter entry is missing id in {path}: {entry!r}")
        if pid in seen:
            raise ValueError(f"duplicate parameter id in {path}: {pid}")
        seen.add(pid)
        if "field" not in entry or "group" not in entry or "cost_points" not in entry:
            raise ValueError(f"parameter entry is missing required fields in {path}: {pid}")
        points = entry.get("cost_points")
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError(f"parameter {pid} must have exactly two cost_points")
    return payload


def _constraint_entries(constraints: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not constraints:
        return []
    return [dict(item) for item in constraints.get("parameters", [])]


def _model_value_to_cost_unit(payload: dict[str, Any], entry: dict[str, Any]) -> float:
    value = _numeric(payload, str(entry["field"]))
    divisor = float(entry.get("model_value_divisor", 1.0) or 1.0)
    return float(value / divisor)


def _baseline_cost_values(
    baseline: dict[str, Any],
    constraints: dict[str, Any] | None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for entry in _constraint_entries(constraints):
        pid = str(entry["id"])
        if "baseline_value" in entry:
            values[pid] = float(entry["baseline_value"])
        else:
            values[pid] = _model_value_to_cost_unit(baseline, entry)
    return values


def _parameter_q(value: float, direction: str) -> float:
    value = float(value)
    if value <= 0.0:
        return float("nan")
    if str(direction) == "smaller_is_costlier":
        return 1.0 / value
    return value


def _parameter_reference_cost_wan(entry: dict[str, Any], value: float) -> float:
    points = entry["cost_points"]
    p1, p2 = points[0], points[1]
    x1 = float(p1["value"])
    x2 = float(p2["value"])
    k1 = float(p1["cost_wan"])
    k2 = float(p2["cost_wan"])
    direction = str(entry.get("direction", "larger_is_costlier"))
    q = _parameter_q(float(value), direction)
    q1 = _parameter_q(x1, direction)
    q2 = _parameter_q(x2, direction)
    if not all(np.isfinite(v) and v > 0.0 for v in (q, q1, q2, k1, k2)):
        return float("nan")
    if abs(q2 - q1) <= 1e-12:
        return float(k1)
    gamma = float(np.log(k2 / k1) / np.log(q2 / q1))
    return float(k1 * (q / q1) ** gamma)


def _pair_bounds(entry: dict[str, Any]) -> tuple[float, float]:
    points = entry["cost_points"]
    values = [float(point["value"]) for point in points]
    return float(min(values)), float(max(values))


def _cost_components_reference_v2(
    payload: dict[str, Any],
    baseline: dict[str, Any],
    constraints: dict[str, Any],
    *,
    cost_values: dict[str, float] | None = None,
    pair_extrapolation_mode: str = "allow",
) -> dict[str, Any]:
    baseline_values = _baseline_cost_values(baseline, constraints)
    supplied = dict(cost_values or {})
    group_delta_values: dict[str, list[float]] = {group: [] for group in BUNDLE_GROUPS}
    parameter_rows: list[dict[str, Any]] = []
    baseline_cost_wan = float(constraints.get("baseline_cost_wan", 8.0))

    for entry in _constraint_entries(constraints):
        pid = str(entry["id"])
        group = str(entry["group"])
        value = float(supplied.get(pid, _model_value_to_cost_unit(payload, entry)))
        ref_value = float(baseline_values.get(pid, _model_value_to_cost_unit(baseline, entry)))
        cost_wan = _parameter_reference_cost_wan(entry, value)
        ref_cost_wan = _parameter_reference_cost_wan(entry, ref_value)
        anchor_cost_wan = max(float(ref_cost_wan), baseline_cost_wan) if np.isfinite(ref_cost_wan) else baseline_cost_wan
        relative_cost = float(cost_wan / anchor_cost_wan) if np.isfinite(cost_wan) and anchor_cost_wan > 0.0 else float("nan")
        delta_wan = float(cost_wan - anchor_cost_wan) if np.isfinite(cost_wan) and np.isfinite(anchor_cost_wan) else float("nan")
        positive_delta_wan = float(max(delta_wan, 0.0)) if np.isfinite(delta_wan) else float("nan")
        include = bool(entry.get("cost_included", True))
        if include and group in group_delta_values and np.isfinite(positive_delta_wan):
            group_delta_values[group].append(positive_delta_wan)
        min_value = float(entry.get("range_min", -float("inf")))
        max_value = float(entry.get("range_max", float("inf")))
        pair_min, pair_max = _pair_bounds(entry)
        pair_extrapolated = bool(value < pair_min - 1e-12 or value > pair_max + 1e-12)
        parameter_rows.append(
            {
                "parameter_id": pid,
                "field": str(entry["field"]),
                "group": group,
                "unit": str(entry.get("unit", "")),
                "value": value,
                "baseline_value": ref_value,
                "reference_cost_wan": cost_wan,
                "baseline_reference_cost_wan": ref_cost_wan,
                "anchor_cost_wan": anchor_cost_wan,
                "relative_cost": relative_cost,
                "cost_delta_wan": delta_wan,
                "positive_delta_wan": positive_delta_wan,
                "range_min": min_value,
                "range_max": max_value,
                "pair_min": pair_min,
                "pair_max": pair_max,
                "pair_extrapolated": pair_extrapolated,
                "in_range": bool(value >= min_value - 1e-12 and value <= max_value + 1e-12),
                "optimized": bool(entry.get("optimized", True)),
                "strict_doc_value": bool(entry.get("strict_doc_value", False)),
                "cost_included": include,
                "direction": str(entry.get("direction", "larger_is_costlier")),
                "pair_extrapolation_mode": str(pair_extrapolation_mode),
            }
        )

    group_deltas = {
        group: float(np.sqrt(np.sum(np.square(values)))) if values else 0.0
        for group, values in group_delta_values.items()
    }
    cost_wan = float(baseline_cost_wan + sum(group_deltas.values()))
    relative_total = float(cost_wan / baseline_cost_wan) if baseline_cost_wan > 0.0 else float("nan")
    return {
        "cost_core": float(1.0 + group_deltas.get("core", 0.0) / baseline_cost_wan),
        "cost_pipeline": float(1.0 + group_deltas.get("pipeline", 0.0) / baseline_cost_wan),
        "cost_vector": float(1.0 + group_deltas.get("vector", 0.0) / baseline_cost_wan),
        "cost_memory": float(1.0 + group_deltas.get("memory", 0.0) / baseline_cost_wan),
        "cost_cache": float(1.0 + group_deltas.get("cache", 0.0) / baseline_cost_wan),
        "delta_core_wan": float(group_deltas.get("core", 0.0)),
        "delta_pipeline_wan": float(group_deltas.get("pipeline", 0.0)),
        "delta_vector_wan": float(group_deltas.get("vector", 0.0)),
        "delta_memory_wan": float(group_deltas.get("memory", 0.0)),
        "delta_cache_wan": float(group_deltas.get("cache", 0.0)),
        "relative_cost": relative_total,
        "cost_wan": cost_wan,
        "surrogate_cost": relative_total,
        "_parameter_costs": parameter_rows,
    }


def _cost_components_for_model(
    payload: dict[str, Any],
    baseline: dict[str, Any],
    *,
    cost_model: str,
    constraints: dict[str, Any] | None,
    cost_values: dict[str, float] | None = None,
    pair_extrapolation_mode: str = "allow",
) -> dict[str, Any]:
    if str(cost_model) == "reference_v2":
        if constraints is None:
            raise ValueError("--cost-model reference_v2 requires --cost-constraints")
        return _cost_components_reference_v2(
            payload,
            baseline,
            constraints,
            cost_values=cost_values,
            pair_extrapolation_mode=pair_extrapolation_mode,
        )
    cost = _cost_components(payload, baseline)
    cost["relative_cost"] = float(cost["surrogate_cost"])
    cost["cost_wan"] = float("nan")
    cost["_parameter_costs"] = []
    return cost


def _set_many(payload: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(payload)
    for field, value in values.items():
        _nested_set(mutated, field, value)
    return mutated


def _interp(device1: dict[str, Any], device2: dict[str, Any], field: str, alpha: float) -> float:
    a = _numeric(device1, field)
    b = _numeric(device2, field)
    return float(a + (b - a) * float(alpha))


def _scale_from(device: dict[str, Any], fields: list[str], factor: float) -> dict[str, float]:
    return {field: float(_numeric(device, field) * float(factor)) for field in fields}


def _build_bundle_levels(device1: dict[str, Any], device2: dict[str, Any]) -> dict[str, list[BundleLevel]]:
    pipeline_fields = [
        "pipeline.fetch_width",
        "pipeline.decode_width",
        "pipeline.rename_width",
        "pipeline.dispatch_width",
        "pipeline.issue_width",
        "pipeline.commit_width",
    ]
    vector_fields = [
        "instruction.simd_width_bits",
        "instruction.fp_throughput_per_cycle.vector_sp_add",
        "instruction.fp_throughput_per_cycle.vector_sp_mul",
        "instruction.fp_throughput_per_cycle.vector_sp_fma",
        "instruction.fp_latency_cycles.vector_sp_add",
        "instruction.fp_latency_cycles.vector_sp_mul",
        "instruction.fp_latency_cycles.vector_sp_fma",
    ]
    memory_capacity_fields = [
        "memory.bandwidth_gb_s_total",
        "memory.copy_bw_gb_s_1t",
        "memory.copy_bw_gb_s_saturated",
        "memory.random_gather_bw_gb_s_1t",
        "memory.random_gather_bw_gb_s_saturated",
    ]
    cache_size_fields = ["cache.l1d.size", "cache.l2.size", "cache.l3_per_die.size"]
    cache_latency_fields = [
        "cache.l1d.response_latency_cycles",
        "cache.l2.response_latency_cycles",
        "cache.l3_per_die.response_latency_cycles",
    ]

    def core_level(level_id: str, cores: int, clock: float, desc: str) -> BundleLevel:
        return BundleLevel(
            "core",
            level_id,
            desc,
            {
                "core.total_cores": int(cores),
                "core.cores_per_socket": int(cores),
                "core.cores_per_die": int(cores),
                "core.cpu_clock": float(clock),
            },
        )

    core_levels = [
        core_level("core_d1_24c", 24, _numeric(device1, "core.cpu_clock"), "Device_1 baseline core/frequency"),
        core_level("core_32c_mid", 32, 2.75, "Intermediate 32-core SKU-like point"),
        core_level("core_d2_40c", 40, _numeric(device2, "core.cpu_clock"), "Device_2 baseline core/frequency"),
        core_level("core_48c_plus", 48, 3.00, "Conservative enhanced 48-core point"),
        core_level("core_64c_plus", 64, 3.05, "High-core enhanced point"),
    ]

    pipeline_levels = [
        BundleLevel("pipeline", "pipe_d1", "Device_1 pipeline width", {f: _numeric(device1, f) for f in pipeline_fields}),
        BundleLevel(
            "pipeline",
            "pipe_mid",
            "Midpoint pipeline width",
            {f: _interp(device1, device2, f, 0.5) for f in pipeline_fields},
        ),
        BundleLevel("pipeline", "pipe_d2", "Device_2 pipeline width", {f: _numeric(device2, f) for f in pipeline_fields}),
        BundleLevel("pipeline", "pipe_enhanced", "Enhanced pipeline width", _scale_from(device2, pipeline_fields, 1.10)),
    ]

    vector_mid = {f: _interp(device1, device2, f, 0.5) for f in vector_fields}
    vector_enhanced = {f: _numeric(device2, f) for f in vector_fields}
    for field in [
        "instruction.fp_throughput_per_cycle.vector_sp_add",
        "instruction.fp_throughput_per_cycle.vector_sp_mul",
        "instruction.fp_throughput_per_cycle.vector_sp_fma",
    ]:
        vector_enhanced[field] = float(vector_enhanced[field] * 1.15)
    for field in [
        "instruction.fp_latency_cycles.vector_sp_add",
        "instruction.fp_latency_cycles.vector_sp_mul",
        "instruction.fp_latency_cycles.vector_sp_fma",
    ]:
        vector_enhanced[field] = float(vector_enhanced[field] * 0.85)
    vector_levels = [
        BundleLevel("vector", "vec_d1", "Device_1 SIMD/FP execution", {f: _numeric(device1, f) for f in vector_fields}),
        BundleLevel("vector", "vec_mid", "Intermediate SIMD/FP execution", vector_mid),
        BundleLevel("vector", "vec_d2", "Device_2 SIMD/FP execution", {f: _numeric(device2, f) for f in vector_fields}),
        BundleLevel("vector", "vec_enhanced", "Enhanced SIMD/FP execution", vector_enhanced),
    ]

    def memory_level(level_id: str, source: dict[str, Any], factor: float, latency_factor: float, desc: str) -> BundleLevel:
        values = _scale_from(source, memory_capacity_fields, factor)
        values["memory.local_mem_delay_ns"] = float(_numeric(source, "memory.local_mem_delay_ns") * latency_factor)
        return BundleLevel("memory", level_id, desc, values)

    memory_levels = [
        memory_level("mem_d1", device1, 1.0, 1.0, "Device_1 memory subsystem"),
        memory_level("mem_d1_125", device1, 1.25, 0.97, "Device_1 memory bandwidth +25%"),
        memory_level("mem_d1_150", device1, 1.50, 0.95, "Device_1 memory bandwidth +50%"),
        memory_level("mem_d2", device2, 1.0, 1.0, "Device_2 memory subsystem"),
        memory_level("mem_d2_120", device2, 1.20, 0.95, "Device_2 memory bandwidth +20%"),
        memory_level("mem_d2_140", device2, 1.40, 0.90, "Device_2 memory bandwidth +40%"),
    ]

    def cache_values(
        *,
        l1_size: float,
        l2_size: float,
        l3_size: float,
        l1_lat: float,
        l2_lat: float,
        l3_lat: float,
    ) -> dict[str, float]:
        return {
            "cache.cacheline_bytes": 64,
            "cache.l1d.size": float(l1_size),
            "cache.l2.size": float(l2_size),
            "cache.l3_per_die.size": float(l3_size),
            "cache.l1d.response_latency_cycles": float(l1_lat),
            "cache.l2.response_latency_cycles": float(l2_lat),
            "cache.l3_per_die.response_latency_cycles": float(l3_lat),
        }

    d1_cache = cache_values(
        l1_size=_numeric(device1, "cache.l1d.size"),
        l2_size=_numeric(device1, "cache.l2.size"),
        l3_size=_numeric(device1, "cache.l3_per_die.size"),
        l1_lat=_numeric(device1, "cache.l1d.response_latency_cycles"),
        l2_lat=_numeric(device1, "cache.l2.response_latency_cycles"),
        l3_lat=_numeric(device1, "cache.l3_per_die.response_latency_cycles"),
    )
    d2_cache = cache_values(
        l1_size=_numeric(device2, "cache.l1d.size"),
        l2_size=_numeric(device2, "cache.l2.size"),
        l3_size=_numeric(device2, "cache.l3_per_die.size"),
        l1_lat=_numeric(device2, "cache.l1d.response_latency_cycles"),
        l2_lat=_numeric(device2, "cache.l2.response_latency_cycles"),
        l3_lat=_numeric(device2, "cache.l3_per_die.response_latency_cycles"),
    )
    cache_levels = [
        BundleLevel("cache", "cache_d1", "Device_1 cache hierarchy", d1_cache),
        BundleLevel(
            "cache",
            "cache_l2_up",
            "L2 enlarged from Device_1",
            cache_values(
                l1_size=d1_cache["cache.l1d.size"],
                l2_size=768 * 1024,
                l3_size=32 * 1024 * 1024,
                l1_lat=d1_cache["cache.l1d.response_latency_cycles"],
                l2_lat=d1_cache["cache.l2.response_latency_cycles"] * 1.08,
                l3_lat=d1_cache["cache.l3_per_die.response_latency_cycles"] * 1.03,
            ),
        ),
        BundleLevel(
            "cache",
            "cache_l3_up",
            "L3 enlarged from Device_1",
            cache_values(
                l1_size=d1_cache["cache.l1d.size"],
                l2_size=d1_cache["cache.l2.size"],
                l3_size=48 * 1024 * 1024,
                l1_lat=d1_cache["cache.l1d.response_latency_cycles"],
                l2_lat=d1_cache["cache.l2.response_latency_cycles"],
                l3_lat=d1_cache["cache.l3_per_die.response_latency_cycles"] * 1.05,
            ),
        ),
        BundleLevel("cache", "cache_d2", "Device_2 cache hierarchy", d2_cache),
        BundleLevel(
            "cache",
            "cache_large_penalty",
            "Large L2/L3 with latency penalty",
            cache_values(
                l1_size=d2_cache["cache.l1d.size"],
                l2_size=1536 * 1024,
                l3_size=96 * 1024 * 1024,
                l1_lat=d2_cache["cache.l1d.response_latency_cycles"],
                l2_lat=d2_cache["cache.l2.response_latency_cycles"] * 1.08,
                l3_lat=d2_cache["cache.l3_per_die.response_latency_cycles"] * 1.08,
            ),
        ),
        BundleLevel(
            "cache",
            "cache_large_fast",
            "Large L2/L3 with latency improvement",
            cache_values(
                l1_size=d2_cache["cache.l1d.size"],
                l2_size=1536 * 1024,
                l3_size=96 * 1024 * 1024,
                l1_lat=d2_cache["cache.l1d.response_latency_cycles"] * 0.95,
                l2_lat=d2_cache["cache.l2.response_latency_cycles"] * 0.95,
                l3_lat=d2_cache["cache.l3_per_die.response_latency_cycles"] * 0.95,
            ),
        ),
    ]

    return {
        "core": core_levels,
        "pipeline": pipeline_levels,
        "vector": vector_levels,
        "memory": memory_levels,
        "cache": cache_levels,
    }


def _pid_values(pids: list[str], value: float) -> dict[str, float]:
    return {pid: float(value) for pid in pids}


def _build_reference_v2_bundle_levels(device1: dict[str, Any], device2: dict[str, Any]) -> dict[str, list[BundleLevel]]:
    pipeline_fields = [
        "pipeline.fetch_width",
        "pipeline.decode_width",
        "pipeline.rename_width",
        "pipeline.dispatch_width",
        "pipeline.issue_width",
        "pipeline.commit_width",
    ]
    pipeline_pids = ["x3", "x4", "x5", "x6", "x7", "x8"]
    vector_throughput_fields = [
        "instruction.fp_throughput_per_cycle.vector_sp_add",
        "instruction.fp_throughput_per_cycle.vector_sp_mul",
        "instruction.fp_throughput_per_cycle.vector_sp_fma",
    ]
    vector_latency_fields = [
        "instruction.fp_latency_cycles.vector_sp_add",
        "instruction.fp_latency_cycles.vector_sp_mul",
        "instruction.fp_latency_cycles.vector_sp_fma",
    ]

    def core_level(level_id: str, cores: int, clock_pred: float, clock_cost: float, desc: str) -> BundleLevel:
        return BundleLevel(
            "core",
            level_id,
            desc,
            {
                "core.total_cores": int(cores),
                "core.cores_per_socket": int(cores),
                "core.cores_per_die": int(cores),
                "core.cpu_clock": float(clock_pred),
            },
            {"x1": float(cores), "x2": float(clock_cost)},
        )

    core_levels = [
        core_level("core_d1_24c", 24, _numeric(device1, "core.cpu_clock"), 2.6, "Device_1 baseline core/frequency"),
        core_level("core_28c_2675", 28, 2.675, 2.675, "28-core 2.675GHz reference interpolation"),
        core_level("core_32c_2750", 32, 2.750, 2.750, "32-core 2.750GHz reference interpolation"),
        core_level("core_36c_2825", 36, 2.825, 2.825, "36-core 2.825GHz reference interpolation"),
        core_level("core_d2_40c", 40, _numeric(device2, "core.cpu_clock"), 2.9, "Device_2 baseline core/frequency"),
        core_level("core_48c_3ghz", 48, 3.0, 3.0, "48-core 3.0GHz reference SKU"),
        core_level("core_64c_3ghz", 64, 3.0, 3.0, "64-core 3.0GHz reference SKU"),
    ]

    def pipeline_level(level_id: str, pred_width: float | None, cost_width: float, desc: str, source: dict[str, Any] | None = None) -> BundleLevel:
        if source is not None:
            values = {field: _numeric(source, field) for field in pipeline_fields}
        else:
            values = {field: float(pred_width) for field in pipeline_fields}
        return BundleLevel("pipeline", level_id, desc, values, _pid_values(pipeline_pids, float(cost_width)))

    pipeline_levels = [
        pipeline_level("pipe_d1", None, 6.0, "Device_1 measured pipeline proxy; costed as 6-wide reference", source=device1),
        pipeline_level("pipe_d2", None, 6.42, "Device_2 measured pipeline proxy; costed as 6.42-wide reference", source=device2),
        pipeline_level("pipe_w6", 6.0, 6.0, "6-wide reference pipeline"),
        pipeline_level("pipe_w614", 6.14, 6.14, "6.14-wide reference pipeline interpolation"),
        pipeline_level("pipe_w628", 6.28, 6.28, "6.28-wide reference pipeline interpolation"),
        pipeline_level("pipe_w642", 6.42, 6.42, "6.42-wide reference pipeline"),
        pipeline_level("pipe_w8", 8.0, 8.0, "8-wide reference pipeline"),
    ]

    def vector_level(
        level_id: str,
        *,
        simd: float,
        add_tp: float,
        mul_tp: float,
        fma_tp: float,
        desc: str,
        source_latency: dict[str, Any] | None = None,
        source_all: dict[str, Any] | None = None,
    ) -> BundleLevel:
        if source_all is not None:
            values = {
                "instruction.simd_width_bits": _numeric(source_all, "instruction.simd_width_bits"),
                **{field: _numeric(source_all, field) for field in vector_throughput_fields},
                **{field: _numeric(source_all, field) for field in vector_latency_fields},
            }
        else:
            latency_source = source_latency or device2
            values = {
                "instruction.simd_width_bits": float(simd),
                "instruction.fp_throughput_per_cycle.vector_sp_add": float(add_tp),
                "instruction.fp_throughput_per_cycle.vector_sp_mul": float(mul_tp),
                "instruction.fp_throughput_per_cycle.vector_sp_fma": float(fma_tp),
                **{field: _numeric(latency_source, field) for field in vector_latency_fields},
            }
        return BundleLevel(
            "vector",
            level_id,
            desc,
            values,
            {"x9": float(simd), "x10": float(add_tp), "x11": float(mul_tp), "x12": float(fma_tp)},
        )

    vector_levels = [
        vector_level("vec_d1", simd=128, add_tp=2, mul_tp=2, fma_tp=2, desc="Device_1 SIMD/FP execution", source_all=device1),
        vector_level("vec_160_tp2", simd=160, add_tp=2, mul_tp=2, fma_tp=2, desc="160-bit SIMD interpolation with 2 op/cycle throughput"),
        vector_level("vec_192_tp2", simd=192, add_tp=2, mul_tp=2, fma_tp=2, desc="192-bit SIMD interpolation with 2 op/cycle throughput"),
        vector_level("vec_224_tp2", simd=224, add_tp=2, mul_tp=2, fma_tp=2, desc="224-bit SIMD interpolation with 2 op/cycle throughput"),
        vector_level("vec_d2", simd=256, add_tp=2, mul_tp=2, fma_tp=2, desc="Device_2 SIMD/FP execution", source_all=device2),
        vector_level("vec_512_tp2", simd=512, add_tp=2, mul_tp=2, fma_tp=2, desc="512-bit SIMD with 2 op/cycle throughput"),
        vector_level("vec_1024_tp4", simd=1024, add_tp=4, mul_tp=4, fma_tp=4, desc="1024-bit SIMD with 4 op/cycle throughput"),
        vector_level("vec_2048_tp4", simd=2048, add_tp=4, mul_tp=4, fma_tp=4, desc="2048-bit SIMD with 4 op/cycle throughput"),
    ]

    def memory_level(
        level_id: str,
        *,
        total: float,
        copy_1t: float,
        copy_sat: float,
        gather_1t: float,
        gather_sat: float,
        latency_ns: float,
        desc: str,
        source_all: dict[str, Any] | None = None,
        cost_values: dict[str, float] | None = None,
    ) -> BundleLevel:
        if source_all is not None:
            values = {
                "memory.bandwidth_gb_s_total": _numeric(source_all, "memory.bandwidth_gb_s_total"),
                "memory.copy_bw_gb_s_1t": _numeric(source_all, "memory.copy_bw_gb_s_1t"),
                "memory.copy_bw_gb_s_saturated": _numeric(source_all, "memory.copy_bw_gb_s_saturated"),
                "memory.random_gather_bw_gb_s_1t": _numeric(source_all, "memory.random_gather_bw_gb_s_1t"),
                "memory.random_gather_bw_gb_s_saturated": _numeric(source_all, "memory.random_gather_bw_gb_s_saturated"),
                "memory.local_mem_delay_ns": _numeric(source_all, "memory.local_mem_delay_ns"),
            }
        else:
            values = {
                "memory.bandwidth_gb_s_total": float(total),
                "memory.copy_bw_gb_s_1t": float(copy_1t),
                "memory.copy_bw_gb_s_saturated": float(copy_sat),
                "memory.random_gather_bw_gb_s_1t": float(gather_1t),
                "memory.random_gather_bw_gb_s_saturated": float(gather_sat),
                "memory.local_mem_delay_ns": float(latency_ns),
            }
        default_cost_values = {
            "x13": float(total),
            "x14": float(copy_1t),
            "x15": float(copy_sat),
            "x16": float(gather_1t),
            "x17": float(gather_sat),
            "x18": float(latency_ns),
        }
        if cost_values:
            default_cost_values.update({key: float(value) for key, value in cost_values.items()})
        return BundleLevel("memory", level_id, desc, values, default_cost_values)

    memory_levels = [
        memory_level(
            "mem_d1",
            total=40,
            copy_1t=5,
            copy_sat=30,
            gather_1t=1,
            gather_sat=4,
            latency_ns=90,
            desc="Device_1 memory subsystem",
            source_all=device1,
            cost_values={"x13": 40, "x14": 5, "x15": 30, "x16": 1, "x17": 4, "x18": 90},
        ),
        memory_level(
            "mem_d2",
            total=_numeric(device2, "memory.bandwidth_gb_s_total"),
            copy_1t=_numeric(device2, "memory.copy_bw_gb_s_1t"),
            copy_sat=_numeric(device2, "memory.copy_bw_gb_s_saturated"),
            gather_1t=_numeric(device2, "memory.random_gather_bw_gb_s_1t"),
            gather_sat=_numeric(device2, "memory.random_gather_bw_gb_s_saturated"),
            latency_ns=_numeric(device2, "memory.local_mem_delay_ns"),
            desc="Device_2 memory subsystem",
            source_all=device2,
        ),
        memory_level("mem_55", total=55, copy_1t=12.5, copy_sat=57.5, gather_1t=1.375, gather_sat=8, latency_ns=87.5, desc="55GB/s reference memory interpolation"),
        memory_level("mem_70", total=70, copy_1t=20, copy_sat=85, gather_1t=1.75, gather_sat=12, latency_ns=85, desc="70GB/s reference memory interpolation"),
        memory_level("mem_85", total=85, copy_1t=27.5, copy_sat=112.5, gather_1t=2.125, gather_sat=16, latency_ns=82.5, desc="85GB/s reference memory interpolation"),
        memory_level("mem_100", total=100, copy_1t=35, copy_sat=140, gather_1t=2.5, gather_sat=20, latency_ns=80, desc="100GB/s reference memory subsystem"),
        memory_level("mem_200", total=200, copy_1t=50, copy_sat=300, gather_1t=4.0, gather_sat=30, latency_ns=70, desc="200GB/s enhanced memory subsystem"),
        memory_level("mem_400", total=400, copy_1t=80, copy_sat=800, gather_1t=8.0, gather_sat=40, latency_ns=50, desc="400GB/s high-end memory subsystem"),
    ]

    def cache_values(l1_kib: float, l2_kib: float, l3_mib: float, l1_lat: float, l2_lat: float, l3_lat: float) -> dict[str, float]:
        return {
            "cache.cacheline_bytes": 64,
            "cache.l1d.size": float(l1_kib) * 1024.0,
            "cache.l2.size": float(l2_kib) * 1024.0,
            "cache.l3_per_die.size": float(l3_mib) * 1024.0 * 1024.0,
            "cache.l1d.response_latency_cycles": float(l1_lat),
            "cache.l2.response_latency_cycles": float(l2_lat),
            "cache.l3_per_die.response_latency_cycles": float(l3_lat),
        }

    def cache_level(
        level_id: str,
        *,
        l1_kib: float,
        l2_kib: float,
        l3_mib: float,
        l1_lat: float,
        l2_lat: float,
        l3_lat: float,
        desc: str,
        source_all: dict[str, Any] | None = None,
        cost_values: dict[str, float] | None = None,
    ) -> BundleLevel:
        values = (
            {
                "cache.cacheline_bytes": _numeric(source_all, "cache.cacheline_bytes"),
                "cache.l1d.size": _numeric(source_all, "cache.l1d.size"),
                "cache.l2.size": _numeric(source_all, "cache.l2.size"),
                "cache.l3_per_die.size": _numeric(source_all, "cache.l3_per_die.size"),
                "cache.l1d.response_latency_cycles": _numeric(source_all, "cache.l1d.response_latency_cycles"),
                "cache.l2.response_latency_cycles": _numeric(source_all, "cache.l2.response_latency_cycles"),
                "cache.l3_per_die.response_latency_cycles": _numeric(source_all, "cache.l3_per_die.response_latency_cycles"),
            }
            if source_all is not None
            else cache_values(l1_kib, l2_kib, l3_mib, l1_lat, l2_lat, l3_lat)
        )
        default_cost_values = {
            "x19": 64,
            "x20": float(l1_kib),
            "x21": float(l2_kib),
            "x22": float(l3_mib),
            "x23": float(l1_lat),
            "x24": float(l2_lat),
            "x25": float(l3_lat),
        }
        if cost_values:
            default_cost_values.update({key: float(value) for key, value in cost_values.items()})
        return BundleLevel("cache", level_id, desc, values, default_cost_values)

    cache_levels = [
        cache_level(
            "cache_d1",
            l1_kib=64,
            l2_kib=512,
            l3_mib=24,
            l1_lat=13,
            l2_lat=30,
            l3_lat=126,
            desc="Device_1 cache hierarchy",
            source_all=device1,
            cost_values={"x19": 64, "x20": 64, "x21": 512, "x22": 24, "x23": 13, "x24": 30, "x25": 126},
        ),
        cache_level(
            "cache_d2",
            l1_kib=64,
            l2_kib=1280,
            l3_mib=70,
            l1_lat=_numeric(device2, "cache.l1d.response_latency_cycles"),
            l2_lat=_numeric(device2, "cache.l2.response_latency_cycles"),
            l3_lat=_numeric(device2, "cache.l3_per_die.response_latency_cycles"),
            desc="Device_2 cache hierarchy",
            source_all=device2,
            cost_values={
                "x19": 64,
                "x20": 64,
                "x21": 1280,
                "x22": 70,
                "x23": _numeric(device2, "cache.l1d.response_latency_cycles"),
                "x24": _numeric(device2, "cache.l2.response_latency_cycles"),
                "x25": _numeric(device2, "cache.l3_per_die.response_latency_cycles"),
            },
        ),
        cache_level(
            "cache_30m_interp",
            l1_kib=80,
            l2_kib=640,
            l3_mib=30,
            l1_lat=12.5,
            l2_lat=28.75,
            l3_lat=121.25,
            desc="30MiB LLC cache latency-penalty interpolation",
        ),
        cache_level(
            "cache_36m_interp",
            l1_kib=96,
            l2_kib=768,
            l3_mib=36,
            l1_lat=12,
            l2_lat=27.5,
            l3_lat=116.5,
            desc="36MiB LLC cache latency-penalty interpolation",
        ),
        cache_level(
            "cache_42m_interp",
            l1_kib=112,
            l2_kib=896,
            l3_mib=42,
            l1_lat=11.5,
            l2_lat=26.25,
            l3_lat=111.75,
            desc="42MiB LLC cache latency-penalty interpolation",
        ),
        cache_level(
            "cache_48m_fast",
            l1_kib=64,
            l2_kib=1024,
            l3_mib=48,
            l1_lat=11,
            l2_lat=25,
            l3_lat=107,
            desc="48MiB LLC enlarged-cache latency-penalty reference",
        ),
        cache_level(
            "cache_96m_fast",
            l1_kib=128,
            l2_kib=1024,
            l3_mib=96,
            l1_lat=11,
            l2_lat=25,
            l3_lat=107,
            desc="96MiB LLC enlarged-cache latency-penalty reference",
        ),
        cache_level(
            "cache_160m_fast",
            l1_kib=128,
            l2_kib=4096,
            l3_mib=160,
            l1_lat=11,
            l2_lat=25,
            l3_lat=107,
            desc="160MiB LLC large-cache latency-penalty reference",
        ),
    ]

    return {
        "core": core_levels,
        "pipeline": pipeline_levels,
        "vector": vector_levels,
        "memory": memory_levels,
        "cache": cache_levels,
    }


def _bundle_levels_for_profile(
    device1: dict[str, Any],
    device2: dict[str, Any],
    *,
    grid_profile: str,
    cost_model: str = "surrogate",
) -> dict[str, list[BundleLevel]]:
    levels = _build_reference_v2_bundle_levels(device1, device2) if str(cost_model) == "reference_v2" else _build_bundle_levels(device1, device2)
    if str(grid_profile) == "smoke":
        levels = {
            "core": [levels["core"][0], levels["core"][2], levels["core"][-1]],
            "pipeline": [levels["pipeline"][0], levels["pipeline"][2]],
            "vector": [levels["vector"][0], levels["vector"][2]],
            "memory": [levels["memory"][0], levels["memory"][3], levels["memory"][-1]],
            "cache": [levels["cache"][0], levels["cache"][3], levels["cache"][-1]],
        }
    elif str(grid_profile) != "full":
        raise ValueError(f"Unsupported grid profile: {grid_profile}")
    return levels


def _candidate_grid(
    device1: dict[str, Any],
    device2: dict[str, Any],
    *,
    grid_profile: str,
    cost_model: str,
    cost_constraints: dict[str, Any] | None,
    pair_extrapolation_mode: str = "allow",
) -> list[dict[str, Any]]:
    levels = _bundle_levels_for_profile(device1, device2, grid_profile=grid_profile, cost_model=cost_model)
    rows: list[dict[str, Any]] = []
    baseline_cost_values = _baseline_cost_values(device1, cost_constraints) if str(cost_model) == "reference_v2" else {}
    indexed_levels = [[(idx, level) for idx, level in enumerate(levels[group])] for group in BUNDLE_GROUPS]
    for index, combo in enumerate(
        itertools.product(*indexed_levels)
    ):
        payload = copy.deepcopy(device1)
        cost_values = dict(baseline_cost_values)
        gene = tuple(int(idx) for idx, _ in combo)
        row: dict[str, Any] = {
            "candidate_id": f"cand_{index:05d}",
            "gene": "|".join(str(item) for item in gene),
        }
        for group, idx in zip(BUNDLE_GROUPS, gene, strict=True):
            row[f"{group}_idx"] = int(idx)
        for _, level in combo:
            payload = _set_many(payload, level.values)
            if level.cost_values:
                cost_values.update({key: float(value) for key, value in level.cost_values.items()})
            row[f"{level.group}_level"] = level.level_id
            row[f"{level.group}_description"] = level.description
        cost = _cost_components_for_model(
            payload,
            device1,
            cost_model=str(cost_model),
            constraints=cost_constraints,
            cost_values=cost_values,
            pair_extrapolation_mode=pair_extrapolation_mode,
        )
        if str(cost_model) == "reference_v2":
            out_of_range = [
                item
                for item in cost.get("_parameter_costs", [])
                if bool(item.get("optimized", True)) and not bool(item.get("in_range", False))
            ]
            if out_of_range:
                continue
            if str(pair_extrapolation_mode) == "forbid":
                pair_extrapolated = [
                    item
                    for item in cost.get("_parameter_costs", [])
                    if bool(item.get("optimized", True))
                    and bool(item.get("cost_included", True))
                    and bool(item.get("pair_extrapolated", False))
                ]
                if pair_extrapolated:
                    continue
        row.update(cost)
        row["cost_model"] = str(cost_model)
        for field in HARDWARE_FIELDS:
            row[field] = _nested_get(payload, field, "")
        row["payload"] = payload
        rows.append(row)
    return rows


def _write_candidate_range_audit(out_dir: Path, parameter_cost_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not parameter_cost_rows:
        audit = pd.DataFrame()
        audit.to_csv(out_dir / "hardware_cost_candidate_range_audit.csv", index=False)
        return audit
    frame = pd.DataFrame(parameter_cost_rows)
    grouped_rows: list[dict[str, Any]] = []
    for (parameter_id, field), group in frame.groupby(["parameter_id", "field"], sort=True):
        values = pd.to_numeric(group["value"], errors="coerce")
        in_range = group["in_range"].astype(bool)
        pair_extrapolated = (
            group["pair_extrapolated"].astype(bool)
            if "pair_extrapolated" in group.columns
            else pd.Series([False] * len(group), index=group.index)
        )
        grouped_rows.append(
            {
                "parameter_id": parameter_id,
                "field": field,
                "group": str(group["group"].iloc[0]),
                "range_min": float(pd.to_numeric(group["range_min"], errors="coerce").iloc[0]),
                "range_max": float(pd.to_numeric(group["range_max"], errors="coerce").iloc[0]),
                "pair_min": float(pd.to_numeric(group["pair_min"], errors="coerce").iloc[0])
                if "pair_min" in group.columns
                else float("nan"),
                "pair_max": float(pd.to_numeric(group["pair_max"], errors="coerce").iloc[0])
                if "pair_max" in group.columns
                else float("nan"),
                "min_value": float(values.min()),
                "max_value": float(values.max()),
                "candidate_count": int(group["candidate_id"].astype(str).nunique()),
                "out_of_range_count": int((~in_range).sum()),
                "pair_extrapolated_count": int(pair_extrapolated.sum()),
                "all_in_range": bool(in_range.all()),
                "all_within_pair_points": bool((~pair_extrapolated).all()),
                "optimized": bool(group["optimized"].astype(bool).iloc[0]),
                "cost_included": bool(group["cost_included"].astype(bool).iloc[0]),
            }
        )
    audit = pd.DataFrame(grouped_rows)
    audit.to_csv(out_dir / "hardware_cost_candidate_range_audit.csv", index=False)
    return audit

def _workloads() -> list[WorkloadSpec]:
    return [
        WorkloadSpec(
            "DLRM-S",
            0.25,
            DlrmStaticConfig(
                batch_size=1024,
                num_indices_per_lookup=16,
                arch_embedding_size="-".join(["65536"] * 8),
                arch_mlp_bot="256-1024-256",
                arch_mlp_top="292-1024-1",
                inter_threads=8,
                num_threads=3,
            ),
        ),
        WorkloadSpec(
            "DLRM-M",
            0.50,
            DlrmStaticConfig(
                batch_size=2048,
                num_indices_per_lookup=16,
                arch_embedding_size="-".join(["65536"] * 8),
                arch_mlp_bot="512-2048-512",
                arch_mlp_top="548-2048-1",
                inter_threads=8,
                num_threads=3,
            ),
        ),
        WorkloadSpec(
            "DLRM-L",
            0.25,
            DlrmStaticConfig(
                batch_size=4096,
                num_indices_per_lookup=16,
                arch_embedding_size="-".join(["65536"] * 8),
                arch_mlp_bot="1024-4096-1024",
                arch_mlp_top="1060-4096-1",
                inter_threads=8,
                num_threads=3,
            ),
        ),
    ]


def _schedule_e2e(config: DlrmStaticConfig, op_shapes: pd.DataFrame, predictions: pd.DataFrame) -> float:
    graph = build_op_graph(op_shapes)
    combo_spec = ComboSpec(
        case_id=str(config.case_id),
        combo=str(config.combo),
        batch_size=int(config.batch_size),
        num_indices_per_lookup=int(config.num_indices_per_lookup),
        inter_threads=int(config.inter_threads),
        artifact_paths=ComboArtifactPaths(
            case_id=str(config.case_id),
            combo=str(config.combo),
            shape_csv=Path("static_op_shapes_full_graph_76.csv"),
            profile_dir=Path("."),
            timeline_csv=Path(""),
        ),
    )
    prediction_rows = predictions[["op_idx", "pred_us"]].copy()
    prediction_rows["op_idx"] = prediction_rows["op_idx"].astype(int)
    prediction_rows["pred_us"] = pd.to_numeric(prediction_rows["pred_us"], errors="coerce").fillna(0.0)
    result = schedule_combo(combo_spec, graph, prediction_rows)
    return float(result.predicted_observed_subgraph_us)


def _prepare_prediction_cache(
    *,
    bundle: dict[str, Any],
    model_input: pd.DataFrame,
) -> dict[str, Any]:
    route = str(bundle.get("route", "")).strip()
    if route != WHITEBOX_PHYSICS_GUARDED_RF_ROUTE:
        return {}
    base_prepared, base_pred = _predict_whitebox_family_rf_raw(bundle, model_input)
    return {
        "base_prepared": base_prepared,
        "base_pred": np.asarray(base_pred, dtype=float),
        "base_prior": pd.to_numeric(base_prepared["whitebox_center_us"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
    }


def _predict_with_cached_guard(
    *,
    bundle: dict[str, Any],
    model_input: pd.DataFrame,
    payload: dict[str, Any],
    cache: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    if not cache:
        return predict_whitebox_family_physics_guarded_rf(bundle, model_input, override_hw_config=payload)
    target_prepared, target_pred = _predict_whitebox_family_rf_raw(bundle, model_input, override_hw_config=payload)
    base_pred = np.asarray(cache["base_pred"], dtype=float)
    base_prior = np.asarray(cache["base_prior"], dtype=float)
    target_prior = pd.to_numeric(target_prepared["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    target_pred_arr = np.asarray(target_pred, dtype=float)
    prior_delta = target_prior - base_prior
    rf_delta = target_pred_arr - base_pred
    tol = float(PHYSICS_GUARD_DELTA_TOL_US)
    conflict = (np.abs(prior_delta) > tol) & (prior_delta * rf_delta < -tol)
    cap_mask = (
        (np.abs(prior_delta) > tol)
        & (prior_delta * rf_delta >= 0.0)
        & (np.abs(rf_delta) > float(PHYSICS_GUARD_DELTA_CAP_MULTIPLIER) * np.abs(prior_delta))
    )
    residual_frozen = target_prior + (base_pred - base_prior)
    capped = base_pred + np.sign(prior_delta) * float(PHYSICS_GUARD_DELTA_CAP_MULTIPLIER) * np.abs(prior_delta)
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
    target_prepared["physics_guard_base_pred_us"] = base_pred
    target_prepared["physics_guard_base_prior_us"] = base_prior
    target_prepared["physics_guard_target_prior_us"] = target_prior
    target_prepared["physics_guard_prior_delta_us"] = prior_delta
    target_prepared["physics_guard_raw_delta_us"] = rf_delta
    target_prepared["physics_guard_final_delta_us"] = guarded_pred - base_pred
    _set_whitebox_rf_prediction_outputs(target_prepared, guarded_pred, bundle)
    return target_prepared, guarded_pred


def _predict_latency(
    *,
    bundle: dict[str, Any],
    model_input: pd.DataFrame,
    op_shapes: pd.DataFrame,
    config: DlrmStaticConfig,
    payload: dict[str, Any],
    prediction_cache: dict[str, Any],
) -> float:
    route = str(bundle.get("route", "")).strip()
    if route == WHITEBOX_PHYSICS_GUARDED_RF_ROUTE:
        prepared, pred = _predict_with_cached_guard(
            bundle=bundle,
            model_input=model_input,
            payload=payload,
            cache=prediction_cache,
        )
    elif route == WHITEBOX_PRIOR_ONLY_ROUTE:
        prepared, pred = predict_whitebox_family_prior(bundle, model_input, override_hw_config=payload)
    else:
        raise ValueError(f"unsupported bundle route: {route}")
    predictions = prepared.copy()
    predictions["pred_us"] = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    return _schedule_e2e(config, op_shapes, predictions)


def _evaluate_candidate_row(
    *,
    row: dict[str, Any],
    device1_bundle: dict[str, Any],
    device2_bundle: dict[str, Any],
    prepared_workloads: dict[str, tuple[WorkloadSpec, pd.DataFrame, pd.DataFrame]],
    prediction_caches: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = row["payload"]
    candidate_id = str(row["candidate_id"])
    weighted_terms: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    for workload_id, (workload, op_shapes, model_input) in prepared_workloads.items():
        d1_us = _predict_latency(
            bundle=device1_bundle,
            model_input=model_input,
            op_shapes=op_shapes,
            config=workload.config,
            payload=payload,
            prediction_cache=prediction_caches[("Device_1", workload_id)],
        )
        d2_us = _predict_latency(
            bundle=device2_bundle,
            model_input=model_input,
            op_shapes=op_shapes,
            config=workload.config,
            payload=payload,
            prediction_cache=prediction_caches[("Device_2", workload_id)],
        )
        ensemble_mean = float(np.mean([d1_us, d2_us]))
        ensemble_min = float(min(d1_us, d2_us))
        ensemble_max = float(max(d1_us, d2_us))
        weighted_terms.append(float(workload.weight) * ensemble_mean)
        prediction_rows.extend(
            [
                {
                    "candidate_id": candidate_id,
                    "workload_id": workload_id,
                    "workload_weight": float(workload.weight),
                    "model_device": "Device_1",
                    "predicted_e2e_us": d1_us,
                    "ensemble_mean_us": ensemble_mean,
                    "ensemble_min_us": ensemble_min,
                    "ensemble_max_us": ensemble_max,
                },
                {
                    "candidate_id": candidate_id,
                    "workload_id": workload_id,
                    "workload_weight": float(workload.weight),
                    "model_device": "Device_2",
                    "predicted_e2e_us": d2_us,
                    "ensemble_mean_us": ensemble_mean,
                    "ensemble_min_us": ensemble_min,
                    "ensemble_max_us": ensemble_max,
                },
            ]
        )
    return prediction_rows, {"candidate_id": candidate_id, "objective_e2e_us": float(sum(weighted_terms))}


def _evaluate_candidate_rows(
    *,
    candidate_rows: list[dict[str, Any]],
    device1_bundle: dict[str, Any],
    device2_bundle: dict[str, Any],
    prepared_workloads: dict[str, tuple[WorkloadSpec, pd.DataFrame, pd.DataFrame]],
    prediction_caches: dict[tuple[str, str], dict[str, Any]],
    prediction_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workers = max(int(prediction_workers), 1)
    prediction_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []

    def evaluate(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _evaluate_candidate_row(
            row=row,
            device1_bundle=device1_bundle,
            device2_bundle=device2_bundle,
            prepared_workloads=prepared_workloads,
            prediction_caches=prediction_caches,
        )

    if workers == 1 or len(candidate_rows) <= 1:
        for idx, row in enumerate(candidate_rows, start=1):
            if idx == 1 or idx % 100 == 0 or idx == len(candidate_rows):
                print(f"[hardware-cost] candidate {idx}/{len(candidate_rows)}", flush=True)
            pred_rows, objective = evaluate(row)
            prediction_rows.extend(pred_rows)
            objective_rows.append(objective)
    else:
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate, row) for row in candidate_rows]
            for future in concurrent.futures.as_completed(futures):
                pred_rows, objective = future.result()
                prediction_rows.extend(pred_rows)
                objective_rows.append(objective)
                completed += 1
                if completed == 1 or completed % 100 == 0 or completed == len(candidate_rows):
                    print(f"[hardware-cost] candidate {completed}/{len(candidate_rows)}", flush=True)

    prediction_frame = pd.DataFrame(prediction_rows)
    if not prediction_frame.empty:
        prediction_frame = prediction_frame.sort_values(
            ["candidate_id", "workload_id", "model_device"],
            kind="mergesort",
        ).reset_index(drop=True)
    objective_frame = pd.DataFrame(objective_rows)
    if not objective_frame.empty:
        objective_frame = objective_frame.sort_values(["candidate_id"], kind="mergesort").reset_index(drop=True)
    return prediction_frame, objective_frame


def _parse_budgets(
    text: str,
    *,
    device2_cost: float,
    device2_cost_wan: float,
    budget_unit: str,
    baseline_cost_wan: float,
) -> list[dict[str, Any]]:
    budgets: list[dict[str, Any]] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if item.lower() == "device2":
            budget_relative = float(device2_cost)
            budget_wan = float(device2_cost_wan if np.isfinite(device2_cost_wan) else device2_cost * baseline_cost_wan)
        else:
            value = float(item)
            if str(budget_unit) == "wan":
                budget_wan = value
                budget_relative = value / max(float(baseline_cost_wan), 1e-12)
            else:
                budget_relative = value
                budget_wan = value * float(baseline_cost_wan)
        budgets.append(
            {
                "budget_label": item,
                "budget": float(budget_wan if str(budget_unit) == "wan" else budget_relative),
                "budget_relative": float(budget_relative),
                "budget_wan": float(budget_wan),
                "budget_unit": str(budget_unit),
            }
        )
    return budgets


def _pareto_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["surrogate_cost", "objective_e2e_us", "candidate_id"]).reset_index(drop=True)
    keep: list[bool] = []
    best = float("inf")
    for value in pd.to_numeric(ordered["objective_e2e_us"], errors="coerce").fillna(float("inf")):
        value_f = float(value)
        keep.append(value_f < best - 1e-9)
        best = min(best, value_f)
    return ordered.loc[keep].reset_index(drop=True)


def _best_by_budget(
    result_frame: pd.DataFrame,
    budgets_text: str,
    *,
    device2_cost: float,
    device2_cost_wan: float,
    budget_unit: str,
    baseline_cost_wan: float,
) -> pd.DataFrame:
    budget_rows: list[dict[str, Any]] = []
    for budget_row in _parse_budgets(
        str(budgets_text),
        device2_cost=device2_cost,
        device2_cost_wan=device2_cost_wan,
        budget_unit=budget_unit,
        baseline_cost_wan=baseline_cost_wan,
    ):
        if str(budget_unit) == "wan" and "cost_wan" in result_frame.columns:
            feasible = result_frame[result_frame["cost_wan"] <= float(budget_row["budget_wan"])].copy()
        else:
            feasible = result_frame[result_frame["surrogate_cost"] <= float(budget_row["budget_relative"])].copy()
        if feasible.empty:
            budget_rows.append({**budget_row, "candidate_id": "", "feasible": False})
            continue
        best = feasible.sort_values(["objective_e2e_us", "surrogate_cost", "candidate_id"]).iloc[0].to_dict()
        best.update(budget_row)
        best["feasible"] = True
        budget_rows.append(best)
    return pd.DataFrame(budget_rows)


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def _fast_non_dominated_sort(genes: list[tuple[int, ...]], objectives: dict[tuple[int, ...], tuple[float, float]]) -> list[list[tuple[int, ...]]]:
    dominates_map: dict[tuple[int, ...], list[tuple[int, ...]]] = {gene: [] for gene in genes}
    domination_count: dict[tuple[int, ...], int] = {gene: 0 for gene in genes}
    fronts: list[list[tuple[int, ...]]] = [[]]
    for p in genes:
        for q in genes:
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominates_map[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    index = 0
    while index < len(fronts) and fronts[index]:
        next_front: list[tuple[int, ...]] = []
        for p in fronts[index]:
            for q in dominates_map[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        index += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def _crowding_distance(front: list[tuple[int, ...]], objectives: dict[tuple[int, ...], tuple[float, float]]) -> dict[tuple[int, ...], float]:
    distance = {gene: 0.0 for gene in front}
    if len(front) <= 2:
        return {gene: float("inf") for gene in front}
    for objective_index in (0, 1):
        ordered = sorted(front, key=lambda gene: objectives[gene][objective_index])
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        min_value = objectives[ordered[0]][objective_index]
        max_value = objectives[ordered[-1]][objective_index]
        span = max(max_value - min_value, 1e-12)
        for idx in range(1, len(ordered) - 1):
            if np.isinf(distance[ordered[idx]]):
                continue
            prev_value = objectives[ordered[idx - 1]][objective_index]
            next_value = objectives[ordered[idx + 1]][objective_index]
            distance[ordered[idx]] += float((next_value - prev_value) / span)
    return distance


def _rank_and_crowding(
    genes: list[tuple[int, ...]],
    objectives: dict[tuple[int, ...], tuple[float, float]],
) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], float], list[list[tuple[int, ...]]]]:
    fronts = _fast_non_dominated_sort(genes, objectives)
    ranks: dict[tuple[int, ...], int] = {}
    distances: dict[tuple[int, ...], float] = {}
    for rank, front in enumerate(fronts):
        front_distance = _crowding_distance(front, objectives)
        for gene in front:
            ranks[gene] = rank
            distances[gene] = front_distance[gene]
    return ranks, distances, fronts


def _select_survivors(
    genes: list[tuple[int, ...]],
    objectives: dict[tuple[int, ...], tuple[float, float]],
    population_size: int,
) -> list[tuple[int, ...]]:
    _, _, fronts = _rank_and_crowding(genes, objectives)
    survivors: list[tuple[int, ...]] = []
    for front in fronts:
        if len(survivors) + len(front) <= population_size:
            survivors.extend(sorted(front))
            continue
        distance = _crowding_distance(front, objectives)
        needed = population_size - len(survivors)
        survivors.extend(sorted(front, key=lambda gene: (-distance[gene], gene))[:needed])
        break
    return survivors


def _run_nsga2(
    *,
    candidate_rows: list[dict[str, Any]],
    candidate_frame: pd.DataFrame,
    device1_bundle: dict[str, Any],
    device2_bundle: dict[str, Any],
    prepared_workloads: dict[str, tuple[WorkloadSpec, pd.DataFrame, pd.DataFrame]],
    prediction_caches: dict[tuple[str, str], dict[str, Any]],
    prediction_workers: int,
    population_size: int,
    generations: int,
    seed: int,
    crossover_prob: float,
    mutation_prob: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_by_gene = {
        tuple(int(row[f"{group}_idx"]) for group in BUNDLE_GROUPS): row
        for row in candidate_rows
    }
    available_genes = sorted(candidate_by_gene)
    candidate_meta_by_id = {
        str(row["candidate_id"]): {
            key: value
            for key, value in row.items()
            if key not in {"payload", "_parameter_costs"}
        }
        for row in candidate_rows
    }
    max_indices = [
        int(candidate_frame[f"{group}_idx"].max())
        for group in BUNDLE_GROUPS
    ]
    rng = np.random.default_rng(int(seed))
    search_space_size = len(available_genes)
    population_size = min(max(int(population_size), 4), search_space_size)
    generations = max(int(generations), 1)

    eval_objectives: dict[tuple[int, ...], dict[str, Any]] = {}
    eval_predictions: dict[tuple[int, ...], list[dict[str, Any]]] = {}

    def evaluate_genes(genes: list[tuple[int, ...]]) -> None:
        missing = sorted({gene for gene in genes if gene not in eval_objectives})
        if not missing:
            return
        rows = [candidate_by_gene[gene] for gene in missing]
        predictions, objectives = _evaluate_candidate_rows(
            candidate_rows=rows,
            device1_bundle=device1_bundle,
            device2_bundle=device2_bundle,
            prepared_workloads=prepared_workloads,
            prediction_caches=prediction_caches,
            prediction_workers=prediction_workers,
        )
        objective_by_id = {str(row["candidate_id"]): row for row in objectives.to_dict(orient="records")}
        pred_by_id: dict[str, list[dict[str, Any]]] = {}
        for row in predictions.to_dict(orient="records"):
            pred_by_id.setdefault(str(row["candidate_id"]), []).append(row)
        for gene in missing:
            candidate_id = str(candidate_by_gene[gene]["candidate_id"])
            objective = objective_by_id[candidate_id]
            eval_objectives[gene] = objective
            eval_predictions[gene] = pred_by_id.get(candidate_id, [])

    def objective_tuple(gene: tuple[int, ...]) -> tuple[float, float]:
        row = candidate_by_gene[gene]
        objective = eval_objectives[gene]
        return float(row["surrogate_cost"]), float(objective["objective_e2e_us"])

    def random_gene() -> tuple[int, ...]:
        return available_genes[int(rng.integers(0, len(available_genes)))]

    seed_population = [tuple(0 for _ in BUNDLE_GROUPS), tuple(max_indices)]
    population: list[tuple[int, ...]] = [gene for gene in seed_population if gene in candidate_by_gene]
    while len(set(population)) < population_size:
        population.append(random_gene())
        population = list(dict.fromkeys(population))
    population = population[:population_size]

    generation_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    def tournament(ranks: dict[tuple[int, ...], int], distance: dict[tuple[int, ...], float]) -> tuple[int, ...]:
        a, b = rng.choice(len(population), size=2, replace=False)
        gene_a = population[int(a)]
        gene_b = population[int(b)]
        key_a = (ranks[gene_a], -distance[gene_a], gene_a)
        key_b = (ranks[gene_b], -distance[gene_b], gene_b)
        return gene_a if key_a < key_b else gene_b

    for generation in range(generations + 1):
        evaluate_genes(population)
        objectives = {gene: objective_tuple(gene) for gene in population}
        ranks, distance, fronts = _rank_and_crowding(population, objectives)
        front0 = fronts[0] if fronts else []
        best_latency_gene = min(population, key=lambda gene: (objectives[gene][1], objectives[gene][0], gene))
        best_cost_gene = min(population, key=lambda gene: (objectives[gene][0], objectives[gene][1], gene))
        generation_rows.append(
            {
                "generation": generation,
                "population_size": len(population),
                "front0_size": len(front0),
                "unique_evaluated": len(eval_objectives),
                "best_cost": objectives[best_cost_gene][0],
                "best_latency_us": objectives[best_latency_gene][1],
                "front0_min_cost": min((objectives[gene][0] for gene in front0), default=float("nan")),
                "front0_min_latency_us": min((objectives[gene][1] for gene in front0), default=float("nan")),
            }
        )
        for gene in population:
            trace_rows.append(
                {
                    "generation": generation,
                    "candidate_id": str(candidate_by_gene[gene]["candidate_id"]),
                    "gene": "|".join(str(item) for item in gene),
                    "rank": ranks[gene],
                    "crowding_distance": distance[gene],
                    "surrogate_cost": objectives[gene][0],
                    "objective_e2e_us": objectives[gene][1],
                }
            )
        if generation == generations:
            break
        offspring: list[tuple[int, ...]] = []
        while len(offspring) < population_size:
            parent_a = tournament(ranks, distance)
            parent_b = tournament(ranks, distance)
            child_a = list(parent_a)
            child_b = list(parent_b)
            if float(rng.random()) < float(crossover_prob):
                for dim in range(len(BUNDLE_GROUPS)):
                    if bool(rng.integers(0, 2)):
                        child_a[dim], child_b[dim] = child_b[dim], child_a[dim]
            for child in (child_a, child_b):
                for dim, upper in enumerate(max_indices):
                    if float(rng.random()) < float(mutation_prob):
                        child[dim] = int(rng.integers(0, upper + 1))
                child_gene = tuple(int(v) for v in child)
                if child_gene not in candidate_by_gene:
                    child_gene = random_gene()
                offspring.append(child_gene)
                if len(offspring) >= population_size:
                    break
        combined = list(dict.fromkeys(population + offspring))
        evaluate_genes(combined)
        combined_objectives = {gene: objective_tuple(gene) for gene in combined}
        population = _select_survivors(combined, combined_objectives, population_size)

    evaluated_genes = sorted(eval_objectives)
    objective_rows = []
    prediction_rows = []
    for gene in evaluated_genes:
        candidate_id = str(candidate_by_gene[gene]["candidate_id"])
        objective_rows.append({**candidate_meta_by_id[candidate_id], **eval_objectives[gene]})
        prediction_rows.extend(eval_predictions[gene])
    result_frame = pd.DataFrame(objective_rows)
    if not result_frame.empty:
        result_frame = result_frame.sort_values(["objective_e2e_us", "surrogate_cost", "candidate_id"]).reset_index(drop=True)
    prediction_frame = pd.DataFrame(prediction_rows)
    if not prediction_frame.empty:
        prediction_frame = prediction_frame.sort_values(
            ["candidate_id", "workload_id", "model_device"],
            kind="mergesort",
        ).reset_index(drop=True)
    pareto = _pareto_frontier(result_frame) if not result_frame.empty else pd.DataFrame()
    return (
        prediction_frame,
        result_frame,
        pareto,
        pd.DataFrame(generation_rows),
        pd.DataFrame(trace_rows),
    )


def _pareto_has_dominated_points(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    rows = frame[["surrogate_cost", "objective_e2e_us"]].to_numpy(dtype=float)
    for i, row_i in enumerate(rows):
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            if _dominates((float(row_j[0]), float(row_j[1])), (float(row_i[0]), float(row_i[1]))):
                return True
    return False


def _budget_latency_not_monotone(best_by_budget: pd.DataFrame) -> bool:
    if best_by_budget.empty or "feasible" not in best_by_budget.columns:
        return False
    feasible = best_by_budget[best_by_budget["feasible"].astype(bool)].copy()
    if feasible.empty:
        return False
    feasible = feasible.sort_values(["budget", "budget_label"]).reset_index(drop=True)
    values = pd.to_numeric(feasible["objective_e2e_us"], errors="coerce").fillna(float("inf")).to_numpy(dtype=float)
    return bool(np.any(np.diff(values) > 1e-9))


def _candidate_warning_rows(
    *,
    candidate_frame: pd.DataFrame,
    baseline: dict[str, Any],
    candidate_ids: set[str],
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not candidate_ids or candidate_frame.empty or "candidate_id" not in candidate_frame.columns:
        return rows
    subset = candidate_frame[candidate_frame["candidate_id"].astype(str).isin(candidate_ids)].copy()
    for _, row in subset.iterrows():
        candidate_id = str(row["candidate_id"])
        mem_total = float(row.get("memory.bandwidth_gb_s_total", float("nan")))
        gather_sat = float(row.get("memory.random_gather_bw_gb_s_saturated", float("nan")))
        base_mem_total = _numeric(baseline, "memory.bandwidth_gb_s_total")
        base_gather_sat = _numeric(baseline, "memory.random_gather_bw_gb_s_saturated")
        if mem_total > base_mem_total * 1.10 and gather_sat < base_gather_sat * 0.95:
            rows.append(
                {
                    "source": source,
                    "severity": "SANITY_WARNING",
                    "candidate_id": candidate_id,
                    "check": "memory_bandwidth_random_gather_inconsistent",
                    "details": "sequential memory bandwidth is higher than baseline while random gather bandwidth is lower.",
                }
            )
        l2_size = float(row.get("cache.l2.size", float("nan")))
        l3_size = float(row.get("cache.l3_per_die.size", float("nan")))
        l2_lat = float(row.get("cache.l2.response_latency_cycles", float("nan")))
        l3_lat = float(row.get("cache.l3_per_die.response_latency_cycles", float("nan")))
        cache_desc = str(row.get("cache_description", "")).lower()
        if (
            (l2_size > _numeric(baseline, "cache.l2.size") or l3_size > _numeric(baseline, "cache.l3_per_die.size"))
            and (l2_lat > _numeric(baseline, "cache.l2.response_latency_cycles") or l3_lat > _numeric(baseline, "cache.l3_per_die.response_latency_cycles"))
            and not any(token in cache_desc for token in ("penalty", "device_2", "enlarged", "large"))
        ):
            rows.append(
                {
                    "source": source,
                    "severity": "SANITY_WARNING",
                    "candidate_id": candidate_id,
                    "check": "cache_size_latency_tradeoff_unexplained",
                    "details": "cache size and latency both increase, but the cache level description does not mark it as a latency-penalty tradeoff.",
                }
            )
    return rows


def _write_sanity_audit(
    *,
    out_dir: Path,
    baseline: dict[str, Any],
    grid_result: pd.DataFrame,
    grid_pareto: pd.DataFrame,
    grid_best: pd.DataFrame,
    nsga_result: pd.DataFrame | None,
    nsga_pareto: pd.DataFrame | None,
    nsga_best: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_global(source: str, severity: str, check: str, details: str) -> None:
        rows.append({"source": source, "severity": severity, "candidate_id": "", "check": check, "details": details})

    if _pareto_has_dominated_points(grid_pareto):
        add_global("grid", "SANITY_FAIL", "pareto_dominated_point", "Grid Pareto frontier contains a dominated point.")
    if _budget_latency_not_monotone(grid_best):
        add_global("grid", "SANITY_FAIL", "budget_best_not_monotone", "Grid best latency increases when budget increases.")
    selected_ids = set(grid_pareto.get("candidate_id", pd.Series(dtype=str)).astype(str))
    selected_ids.update(grid_best.get("candidate_id", pd.Series(dtype=str)).astype(str))
    selected_ids.discard("")
    rows.extend(
        _candidate_warning_rows(
            candidate_frame=grid_result,
            baseline=baseline,
            candidate_ids=selected_ids,
            source="grid",
        )
    )

    if nsga_result is not None and nsga_pareto is not None and nsga_best is not None and not nsga_result.empty:
        if _pareto_has_dominated_points(nsga_pareto):
            add_global("nsga2", "SANITY_FAIL", "pareto_dominated_point", "NSGA-II Pareto frontier contains a dominated point.")
        if _budget_latency_not_monotone(nsga_best):
            add_global("nsga2", "SANITY_FAIL", "budget_best_not_monotone", "NSGA-II best latency increases when budget increases.")
        nsga_ids = set(nsga_pareto.get("candidate_id", pd.Series(dtype=str)).astype(str))
        nsga_ids.update(nsga_best.get("candidate_id", pd.Series(dtype=str)).astype(str))
        nsga_ids.discard("")
        rows.extend(
            _candidate_warning_rows(
                candidate_frame=nsga_result,
                baseline=baseline,
                candidate_ids=nsga_ids,
                source="nsga2",
            )
        )

    if not rows:
        rows.append(
            {
                "source": "all",
                "severity": "SANITY_PASS",
                "candidate_id": "",
                "check": "all_checks",
                "details": "No dominated-frontier, budget-monotonicity, or selected-candidate physical consistency issue detected.",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "optimization_sanity_audit.csv", index=False)
    return audit


def _plot_pareto(
    *,
    out_dir: Path,
    grid_result: pd.DataFrame,
    grid_pareto: pd.DataFrame,
    grid_best: pd.DataFrame,
    nsga_result: pd.DataFrame | None = None,
    nsga_pareto: pd.DataFrame | None = None,
    nsga_best: pd.DataFrame | None = None,
    cost_x_col: str = "surrogate_cost",
    cost_x_label: str = "Surrogate cost (Device_1 baseline = 1.0)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x_series: list[pd.Series] = []
    for frame in (grid_result, grid_pareto, grid_best, nsga_result, nsga_pareto, nsga_best):
        if frame is not None and not frame.empty and cost_x_col in frame.columns:
            x_series.append(pd.to_numeric(frame[cost_x_col], errors="coerce"))
        if frame is not None and not frame.empty:
            if cost_x_col == "cost_wan" and "budget_wan" in frame.columns:
                x_series.append(pd.to_numeric(frame["budget_wan"], errors="coerce"))
            elif cost_x_col != "cost_wan" and "budget_relative" in frame.columns:
                x_series.append(pd.to_numeric(frame["budget_relative"], errors="coerce"))
    if x_series:
        finite_x = pd.concat(x_series, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
        x_min = float(finite_x.min()) if not finite_x.empty else float("nan")
        x_max = float(finite_x.max()) if not finite_x.empty else float("nan")
    else:
        x_min = float("nan")
        x_max = float("nan")
    budget_line_max = x_max * 1.03 if np.isfinite(x_max) else float("inf")
    if not grid_result.empty:
        ax.scatter(
            grid_result[cost_x_col],
            grid_result["objective_e2e_us"],
            s=16,
            alpha=0.25,
            color="#7f7f7f",
            label="Grid candidates",
            zorder=1,
        )
    if not grid_pareto.empty:
        ordered = grid_pareto.sort_values(cost_x_col)
        ax.plot(
            ordered[cost_x_col],
            ordered["objective_e2e_us"],
            marker="o",
            linewidth=1.8,
            color="#1f77b4",
            label="Grid Pareto",
            zorder=3,
        )
    if nsga_result is not None and not nsga_result.empty:
        ax.scatter(
            nsga_result[cost_x_col],
            nsga_result["objective_e2e_us"],
            s=24,
            alpha=0.35,
            color="#ff7f0e",
            label="NSGA-II evaluated",
            zorder=2,
        )
    if nsga_pareto is not None and not nsga_pareto.empty:
        ordered = nsga_pareto.sort_values(cost_x_col)
        ax.plot(
            ordered[cost_x_col],
            ordered["objective_e2e_us"],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color="#d62728",
            label="NSGA-II Pareto",
            alpha=0.75,
            zorder=4,
        )
    for label, frame, line_color in (
        ("Grid budget best", grid_best, "#2ca02c"),
        ("NSGA-II budget best", nsga_best, "#9467bd"),
    ):
        if frame is None or frame.empty or "feasible" not in frame.columns:
            continue
        feasible = frame[frame["feasible"].astype(bool)].copy()
        if feasible.empty:
            continue
        for _, row in feasible.iterrows():
            budget = float(row["budget_wan"] if cost_x_col == "cost_wan" and "budget_wan" in row else row["budget"])
            if budget > budget_line_max:
                continue
            ax.axvline(budget, color=line_color, linestyle=":", linewidth=0.7, alpha=0.30, zorder=0)
    for label, frame, face_color, edge_color in (
        ("Grid budget best", grid_best, "#8dd35f", "#145a14"),
        ("NSGA-II budget best", nsga_best, "#ffd92f", "#4b0082"),
    ):
        if frame is None or frame.empty or "feasible" not in frame.columns:
            continue
        feasible = frame[frame["feasible"].astype(bool)].copy()
        if feasible.empty:
            continue
        if cost_x_col == "cost_wan" and "budget_wan" in feasible.columns:
            budget_x = pd.to_numeric(feasible["budget_wan"], errors="coerce")
        elif "budget_relative" in feasible.columns:
            budget_x = pd.to_numeric(feasible["budget_relative"], errors="coerce")
        else:
            budget_x = pd.to_numeric(feasible.get("budget", feasible[cost_x_col]), errors="coerce")
        actual_x = pd.to_numeric(feasible[cost_x_col], errors="coerce")
        y = pd.to_numeric(feasible["objective_e2e_us"], errors="coerce")
        for actual, budget, y_value in zip(actual_x, budget_x, y, strict=False):
            if not all(np.isfinite(v) for v in (actual, budget, y_value)):
                continue
            if abs(float(actual) - float(budget)) > 1e-9:
                ax.plot(
                    [float(actual), float(budget)],
                    [float(y_value), float(y_value)],
                    linestyle="-",
                    linewidth=0.8,
                    color=edge_color,
                    alpha=0.40,
                    zorder=8,
                )
        ax.scatter(
            actual_x,
            y,
            marker="D",
            s=42,
            color=face_color,
            edgecolor=edge_color,
            linewidth=0.8,
            alpha=0.85,
            label=f"{label} actual cost",
            zorder=9,
        )
        ax.scatter(
            budget_x,
            y,
            marker="*",
            s=260,
            color=face_color,
            edgecolor=edge_color,
            linewidth=1.1,
            label=label,
            zorder=10,
        )
    ax.set_xlabel(cost_x_label)
    ax.set_ylabel("Weighted E2E latency (us)")
    ax.set_title("Cost-performance Pareto Frontier")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        margin = max((x_max - x_min) * 0.06, x_max * 0.01)
        ax.set_xlim(x_min - margin, x_max + margin)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"pareto_frontier_cost_latency.{suffix}", dpi=220)
    plt.close(fig)


def _write_report(
    *,
    out_dir: Path,
    mode: str,
    search_algorithm: str,
    prediction_workers: int,
    cost_model: str,
    budget_unit: str,
    cost_constraints_path: str,
    pair_extrapolation_mode: str = "allow",
    device1_bundle_path: str,
    device2_bundle_path: str,
    device1_bundle: dict[str, Any],
    device2_bundle: dict[str, Any],
    device2_cost: float,
    device2_cost_wan: float | None = None,
    best_by_budget: pd.DataFrame,
    pareto: pd.DataFrame,
    sanity_audit: pd.DataFrame,
    candidate_range_audit: pd.DataFrame | None = None,
    nsga_best_by_budget: pd.DataFrame | None = None,
    nsga_pareto: pd.DataFrame | None = None,
    nsga_generation_summary: pd.DataFrame | None = None,
) -> None:
    title = "硬件成本优化报告"
    if mode == "prior_only_fallback":
        title += "（LOWER-CONFIDENCE PRIOR-ONLY FALLBACK）"
    hidden_columns = {"payload", "_parameter_costs"}

    def display_frame(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
        if frame.empty:
            return "No feasible candidate."
        visible = frame.drop(columns=[col for col in hidden_columns if col in frame.columns], errors="ignore")
        preferred_columns = [
            "candidate_id",
            "core_level",
            "pipeline_level",
            "vector_level",
            "memory_level",
            "cache_level",
            "relative_cost",
            "cost_wan",
            "objective_e2e_us",
            "budget_label",
            "budget_relative",
            "budget_wan",
            "budget_unit",
            "feasible",
        ]
        selected_columns = [col for col in preferred_columns if col in visible.columns]
        if selected_columns:
            visible = visible[selected_columns]
        if len(visible) > int(max_rows):
            visible = visible.head(int(max_rows)).copy()
        return visible.to_string(index=False)

    grid_summary = "Grid was not run." if str(search_algorithm) == "nsga2" and best_by_budget.empty else display_frame(best_by_budget)
    lines = [
        f"# {title}",
        "",
        "## 运行口径",
        "",
        "- `refit=false`，优化阶段不重新训练 RF，也不重新拟合 op-thread scale。",
        "- 候选硬件只通过 `override_hw_config` 进入预测路径。",
        f"- prediction mode: `{mode}`",
        f"- search algorithm: `{search_algorithm}`",
        f"- prediction workers: `{int(prediction_workers)}`",
        f"- Device_1 bundle: `{device1_bundle_path}`",
        f"- Device_1 model_sha256: `{device1_bundle.get('model_sha256', '')}`",
        f"- Device_2 bundle: `{device2_bundle_path}`",
        f"- Device_2 model_sha256: `{device2_bundle.get('model_sha256', '')}`",
        f"- cost model: `{cost_model}`",
        f"- budget unit: `{budget_unit}`",
        f"- cost constraints: `{cost_constraints_path}`",
        f"- pair extrapolation mode: `{pair_extrapolation_mode}`",
        (
            f"- Device_2 reference cost: `{float(device2_cost_wan):.6f} wan`"
            if cost_model == "reference_v2"
            else f"- Device_2 relative cost: `{device2_cost:.6f}`"
        ),
        "",
        "## 成本函数",
        "",
        (
            "当前报告使用需求方参数约束 v2 的 reference cost model。每个参数先用两点幂律函数拟合参考成本曲线，"
            "再转成相对 Device_1 基准的正向增量成本。组内使用 L2 范数聚合增量，整机成本为 8w 基准加各组增量。"
            "当 `pair_extrapolation_mode=forbid` 时，候选参数不得超出需求方两个参考成本点覆盖的区间。"
            if cost_model == "reference_v2"
            else "当前报告使用旧 surrogate relative cost。"
        ),
        "",
        "$$\\Delta_g=\\sqrt{\\sum_{p\\in P_g}\\max(k_p(x_p)-A_p,0)^2},\\quad C_{wan}(h)=8w+\\sum_g\\Delta_g$$"
        if cost_model == "reference_v2"
        else "$$C(h)=\\sum_{g\\in G}W_gC_g(h_g)$$",
        "",
        "| 硬件参数组 | Aggregation |",
        "|---|---|",
        *[
            f"| {key} | L2 norm of positive parameter cost deltas |"
            for key in BUNDLE_WEIGHTS
        ],
        "",
        (
            "`x13 memory.bandwidth_gb_s_total` 严格保留文档中的 `{40, 6w; 100, 8.5w}`，这是当前确认到的唯一非 8w 低点。"
            if cost_model == "reference_v2"
            else "该成本函数是 surrogate relative cost，不是采购报价。"
        ),
        "",
        "Pair 对描述的是单参数变化下的整机参考成本，因此本报告不把 25 个参数成本直接相加。",
        "",
        "## 参数范围审计",
        "",
        (
            "- range status: `PASS`，所有参与优化的候选参数均在 v2 约束范围内。"
            if candidate_range_audit is not None
            and not candidate_range_audit.empty
            and bool(candidate_range_audit["all_in_range"].astype(bool).all())
            else "- status: `N/A`，未生成参数范围审计。"
        ),
        (
            "- pair status: `PASS`，所有参与优化的候选参数均在 Pair 参考点区间内。"
            if str(pair_extrapolation_mode) == "forbid"
            and candidate_range_audit is not None
            and not candidate_range_audit.empty
            and "all_within_pair_points" in candidate_range_audit.columns
            and bool(candidate_range_audit["all_within_pair_points"].astype(bool).all())
            else "- pair status: `ALLOW`，候选允许对 Pair 参考点之外做成本外推。"
        ),
        "- audit file: `hardware_cost_candidate_range_audit.csv`",
        "",
        "## Grid 每个预算档的最优候选",
        "",
        "```text",
        grid_summary,
        "```",
        "",
        "## Pareto 前沿摘要",
        "",
        f"- Pareto candidates: {len(pareto)}",
        f"- Best latency on frontier: {pareto['objective_e2e_us'].min():.6f} us" if not pareto.empty else "- Best latency on frontier: n/a",
        "",
        "## NSGA-II 摘要",
        "",
        (
            "```text\n"
            + display_frame(nsga_best_by_budget)
            + "\n```"
            if nsga_best_by_budget is not None and not nsga_best_by_budget.empty
            else "NSGA-II was not run."
        ),
        "",
        (
            f"- NSGA-II Pareto candidates: {len(nsga_pareto)}"
            if nsga_pareto is not None and not nsga_pareto.empty
            else "- NSGA-II Pareto candidates: n/a"
        ),
        (
            f"- NSGA-II evaluated candidates: {int(nsga_generation_summary['unique_evaluated'].max())}"
            if nsga_generation_summary is not None and not nsga_generation_summary.empty
            else "- NSGA-II evaluated candidates: n/a"
        ),
        "",
        "## 合理性审计",
        "",
        "```text",
        sanity_audit.to_string(index=False) if not sanity_audit.empty else "No sanity audit rows.",
        "```",
        "",
        "## 可视化",
        "",
        "- `pareto_frontier_cost_latency.png`",
        "- `pareto_frontier_cost_latency.pdf`",
        "",
    ]
    out_dir.joinpath("HARDWARE_COST_OPTIMIZATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_model = str(args.cost_model)
    pair_extrapolation_mode = str(args.pair_extrapolation_mode)
    cost_constraints = _load_cost_constraints(str(args.cost_constraints)) if cost_model == "reference_v2" else None
    baseline_cost_wan = float(cost_constraints.get("baseline_cost_wan", 8.0)) if cost_constraints else 1.0

    device1_hw_path = Path(args.device1_hw_config).resolve()
    device2_hw_path = Path(args.device2_hw_config).resolve()
    device1_payload = _load_yaml(device1_hw_path)
    device2_payload = _load_yaml(device2_hw_path)
    device1_bundle, device2_bundle, fallback = _load_prediction_bundles(args)
    mode = "prior_only_fallback" if fallback else "physics_guarded_rf_v3"

    candidate_rows = _candidate_grid(
        device1_payload,
        device2_payload,
        grid_profile=str(args.grid_profile),
        cost_model=cost_model,
        cost_constraints=cost_constraints,
        pair_extrapolation_mode=pair_extrapolation_mode,
    )
    parameter_cost_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        for param_row in row.get("_parameter_costs", []):
            parameter_cost_rows.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "gene": str(row["gene"]),
                    **param_row,
                }
            )
    candidate_frame = pd.DataFrame(
        [{k: v for k, v in row.items() if k not in {"payload", "_parameter_costs"}} for row in candidate_rows]
    )
    if parameter_cost_rows:
        pd.DataFrame(parameter_cost_rows).to_csv(out_dir / "candidate_parameter_costs.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "candidate_parameter_costs.csv", index=False)
    candidate_range_audit = _write_candidate_range_audit(out_dir, parameter_cost_rows)
    if cost_constraints:
        pd.DataFrame(_constraint_entries(cost_constraints)).to_csv(
            out_dir / "hardware_cost_constraints_v2_resolved.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(out_dir / "hardware_cost_constraints_v2_resolved.csv", index=False)

    device2_exact = candidate_frame[
        candidate_frame["core_level"].eq("core_d2_40c")
        & candidate_frame["pipeline_level"].eq("pipe_d2")
        & candidate_frame["vector_level"].eq("vec_d2")
        & candidate_frame["memory_level"].eq("mem_d2")
        & candidate_frame["cache_level"].eq("cache_d2")
    ]
    if not device2_exact.empty:
        device2_cost = float(device2_exact.iloc[0]["surrogate_cost"])
        device2_cost_wan = float(device2_exact.iloc[0].get("cost_wan", float("nan")))
    else:
        device2_costs = _cost_components_for_model(
            device2_payload,
            device1_payload,
            cost_model=cost_model,
            constraints=cost_constraints,
            pair_extrapolation_mode=pair_extrapolation_mode,
        )
        device2_cost = float(device2_costs["surrogate_cost"])
        device2_cost_wan = float(device2_costs.get("cost_wan", float("nan")))
    candidate_frame.to_csv(out_dir / "candidate_hardware_grid.csv", index=False)

    workloads = _workloads()
    prepared_workloads: dict[str, tuple[WorkloadSpec, pd.DataFrame, pd.DataFrame]] = {}
    for workload in workloads:
        op_shapes = build_static_op_shape_table(workload.config)
        model_input = build_static_feature_frame(workload.config, graph_scope=MODEL_INPUT_51)
        prepared_workloads[workload.workload_id] = (workload, op_shapes, model_input)
    prediction_caches: dict[tuple[str, str], dict[str, Any]] = {}
    for workload_id, (_, _, model_input) in prepared_workloads.items():
        prediction_caches[("Device_1", workload_id)] = _prepare_prediction_cache(
            bundle=device1_bundle,
            model_input=model_input,
        )
        prediction_caches[("Device_2", workload_id)] = _prepare_prediction_cache(
            bundle=device2_bundle,
            model_input=model_input,
        )

    search_algorithm = str(args.search_algorithm)
    prediction_workers = max(int(args.prediction_workers), 1)
    started_at = time.perf_counter()

    run_grid = search_algorithm in {"grid", "both"}
    run_nsga2 = search_algorithm in {"nsga2", "both"}

    if run_grid:
        prediction_frame, objective_frame = _evaluate_candidate_rows(
            candidate_rows=candidate_rows,
            device1_bundle=device1_bundle,
            device2_bundle=device2_bundle,
            prepared_workloads=prepared_workloads,
            prediction_caches=prediction_caches,
            prediction_workers=prediction_workers,
        )
        prediction_frame.to_csv(out_dir / "candidate_predictions.csv", index=False)
        result_frame = candidate_frame.merge(objective_frame, on="candidate_id", how="left")
        result_frame = result_frame.sort_values(["objective_e2e_us", "surrogate_cost", "candidate_id"]).reset_index(drop=True)
        result_frame.to_csv(out_dir / "candidate_objectives.csv", index=False)
        pareto = _pareto_frontier(result_frame)
        best_by_budget = _best_by_budget(
            result_frame,
            str(args.budgets),
            device2_cost=device2_cost,
            device2_cost_wan=device2_cost_wan,
            budget_unit=str(args.budget_unit),
            baseline_cost_wan=baseline_cost_wan,
        )
    else:
        prediction_frame = pd.DataFrame()
        result_frame = pd.DataFrame()
        pareto = pd.DataFrame()
        best_by_budget = pd.DataFrame()
        pd.DataFrame().to_csv(out_dir / "candidate_predictions.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "candidate_objectives.csv", index=False)
    pareto.to_csv(out_dir / "pareto_frontier.csv", index=False)
    best_by_budget.to_csv(out_dir / "best_by_budget.csv", index=False)

    nsga_prediction_frame: pd.DataFrame | None = None
    nsga_result_frame: pd.DataFrame | None = None
    nsga_pareto: pd.DataFrame | None = None
    nsga_best_by_budget: pd.DataFrame | None = None
    nsga_generation_summary: pd.DataFrame | None = None
    nsga_trace: pd.DataFrame | None = None
    if run_nsga2:
        (
            nsga_prediction_frame,
            nsga_result_frame,
            nsga_pareto,
            nsga_generation_summary,
            nsga_trace,
        ) = _run_nsga2(
            candidate_rows=candidate_rows,
            candidate_frame=candidate_frame,
            device1_bundle=device1_bundle,
            device2_bundle=device2_bundle,
            prepared_workloads=prepared_workloads,
            prediction_caches=prediction_caches,
            prediction_workers=prediction_workers,
            population_size=int(args.nsga2_population),
            generations=int(args.nsga2_generations),
            seed=int(args.nsga2_seed),
            crossover_prob=float(args.nsga2_crossover_prob),
            mutation_prob=float(args.nsga2_mutation_prob),
        )
        nsga_prediction_frame.to_csv(out_dir / "nsga2_candidate_predictions.csv", index=False)
        nsga_result_frame.to_csv(out_dir / "nsga2_candidate_objectives.csv", index=False)
        nsga_pareto.to_csv(out_dir / "nsga2_pareto_frontier.csv", index=False)
        nsga_best_by_budget = _best_by_budget(
            nsga_result_frame,
            str(args.budgets),
            device2_cost=device2_cost,
            device2_cost_wan=device2_cost_wan,
            budget_unit=str(args.budget_unit),
            baseline_cost_wan=baseline_cost_wan,
        )
        nsga_best_by_budget.to_csv(out_dir / "nsga2_best_by_budget.csv", index=False)
        nsga_generation_summary.to_csv(out_dir / "nsga2_generation_summary.csv", index=False)
        nsga_trace.to_csv(out_dir / "nsga2_search_trace.csv", index=False)

    sanity_audit = _write_sanity_audit(
        out_dir=out_dir,
        baseline=device1_payload,
        grid_result=result_frame if not result_frame.empty else candidate_frame,
        grid_pareto=pareto,
        grid_best=best_by_budget,
        nsga_result=nsga_result_frame,
        nsga_pareto=nsga_pareto,
        nsga_best=nsga_best_by_budget,
    )
    _plot_pareto(
        out_dir=out_dir,
        grid_result=result_frame if not result_frame.empty else pd.DataFrame(),
        grid_pareto=pareto,
        grid_best=best_by_budget,
        nsga_result=nsga_result_frame,
        nsga_pareto=nsga_pareto,
        nsga_best=nsga_best_by_budget,
        cost_x_col="cost_wan" if cost_model == "reference_v2" else "surrogate_cost",
        cost_x_label="Reference cost (wan RMB, Device_1 baseline = 8w)"
        if cost_model == "reference_v2"
        else "Surrogate cost (Device_1 baseline = 1.0)",
    )

    primary_best = best_by_budget if run_grid else (nsga_best_by_budget if nsga_best_by_budget is not None else pd.DataFrame())
    warning_ids = set(
        sanity_audit[
            sanity_audit["severity"].astype(str).isin(["SANITY_WARNING", "SANITY_FAIL"])
            & sanity_audit["candidate_id"].astype(str).ne("")
        ]["candidate_id"].astype(str)
    )
    feasible_best = (
        primary_best[primary_best["feasible"].astype(bool)].copy()
        if not primary_best.empty and "feasible" in primary_best.columns
        else pd.DataFrame()
    )
    if not feasible_best.empty:
        eligible = feasible_best[~feasible_best["candidate_id"].astype(str).isin(warning_ids)].copy()
        chosen = eligible.iloc[-1] if not eligible.empty else feasible_best.iloc[-1]
        best_candidate_id = str(chosen["candidate_id"])
        best_payload = next(row["payload"] for row in candidate_rows if str(row["candidate_id"]) == best_candidate_id)
        _write_yaml(out_dir / "best_candidate_hw_config.yaml", best_payload)
    elapsed_s = float(time.perf_counter() - started_at)

    manifest = {
        "refit": False,
        "mode": mode,
        "route": WHITEBOX_PHYSICS_GUARDED_RF_ROUTE if not fallback else WHITEBOX_PRIOR_ONLY_ROUTE,
        "cost_model": cost_model,
        "cost_constraints": str(Path(args.cost_constraints).resolve()) if cost_constraints else "",
        "budget_unit": str(args.budget_unit),
        "pair_extrapolation_mode": pair_extrapolation_mode,
        "device1_bundle": str(args.device1_bundle),
        "device2_bundle": str(args.device2_bundle),
        "device1_hw_config": str(device1_hw_path),
        "device2_hw_config": str(device2_hw_path),
        "device2_surrogate_cost": device2_cost,
        "device2_cost_wan": device2_cost_wan,
        "baseline_cost_wan": baseline_cost_wan,
        "bundle_weights": BUNDLE_WEIGHTS,
        "candidate_count": int(len(candidate_frame)),
        "grid_evaluated_count": int(len(result_frame)),
        "nsga2_evaluated_count": int(len(nsga_result_frame)) if nsga_result_frame is not None else 0,
        "grid_profile": str(args.grid_profile),
        "search_algorithm": search_algorithm,
        "prediction_workers": int(prediction_workers),
        "elapsed_seconds": elapsed_s,
        "nsga2": {
            "population": int(args.nsga2_population),
            "generations": int(args.nsga2_generations),
            "seed": int(args.nsga2_seed),
            "crossover_prob": float(args.nsga2_crossover_prob),
            "mutation_prob": float(args.nsga2_mutation_prob),
        },
        "workloads": [
            {
                "workload_id": workload.workload_id,
                "weight": float(workload.weight),
                "batch_size": int(workload.config.batch_size),
                "num_indices_per_lookup": int(workload.config.num_indices_per_lookup),
                "arch_embedding_size": str(workload.config.arch_embedding_size),
                "arch_mlp_bot": str(workload.config.arch_mlp_bot),
                "arch_mlp_top": str(workload.config.arch_mlp_top),
                "inter_threads": int(workload.config.inter_threads),
                "num_threads": int(workload.config.num_threads),
            }
            for workload in workloads
        ],
    }
    (out_dir / "optimization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_report(
        out_dir=out_dir,
        mode=mode,
        search_algorithm=search_algorithm,
        prediction_workers=prediction_workers,
        cost_model=cost_model,
        budget_unit=str(args.budget_unit),
        cost_constraints_path=str(args.cost_constraints) if cost_constraints else "",
        pair_extrapolation_mode=pair_extrapolation_mode,
        candidate_range_audit=candidate_range_audit,
        device1_bundle_path=str(args.device1_bundle),
        device2_bundle_path=str(args.device2_bundle),
        device1_bundle=device1_bundle,
        device2_bundle=device2_bundle,
        device2_cost=device2_cost,
        device2_cost_wan=device2_cost_wan,
        best_by_budget=best_by_budget,
        pareto=pareto,
        sanity_audit=sanity_audit,
        nsga_best_by_budget=nsga_best_by_budget,
        nsga_pareto=nsga_pareto,
        nsga_generation_summary=nsga_generation_summary,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "candidate_count": len(candidate_frame),
                "grid_evaluated_count": int(len(result_frame)),
                "nsga2_evaluated_count": int(len(nsga_result_frame)) if nsga_result_frame is not None else 0,
                "mode": mode,
                "elapsed_seconds": elapsed_s,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
