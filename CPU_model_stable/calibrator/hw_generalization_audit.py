from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - fail fast in main.
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BINDING_CONFIG = SCRIPT_DIR / "configs" / "hw_generalization_bindings.json"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "hw_generalization_audit"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytical_base import run as base_run  # noqa: E402
from calibrator import run as cal_run  # noqa: E402
from calibrator.pipeline_eval import evaluate_source_test_pipeline  # noqa: E402
from calibrator.whitebox_family_routes import (  # noqa: E402
    WHITEBOX_GATHER_SLICE_RF_ROUTE,
    WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
    WHITEBOX_PRIOR_ONLY_ROUTE,
    WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
    fit_op_thread_scale_config,
    fit_whitebox_family_gather_slice_rf_bundle,
    fit_whitebox_family_physics_guarded_rf_bundle,
    fit_whitebox_family_prior_bundle,
    fit_whitebox_family_simple_rf_bundle,
    predict_whitebox_family_physics_guarded_rf,
    predict_whitebox_family_prior,
    predict_whitebox_family_simple_rf,
)


SERVICE_MODULES = ["descriptor", "stream_mem", "random_mem", "compute", "reorder", "sync", "dispatch"]
QUANTITY_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(.*)$")


@dataclass(frozen=True)
class DatasetBinding:
    alias: str
    data_root: Path
    hw_config: Path
    actual_hardware: str
    metadata_note: str = ""
    model_bundle: Path | None = None


@dataclass(frozen=True)
class PerturbationSpec:
    field: str
    group: str
    relation: str
    expected_modules: tuple[str, ...]
    value_kind: str = "scalar"


PARAMETER_SPECS: list[PerturbationSpec] = [
    PerturbationSpec("core.total_cores", "core_capacity", "capacity_up_decreases_time", ("stream_mem", "random_mem", "compute", "dispatch", "sync")),
    PerturbationSpec("core.cpu_clock", "clock", "capacity_up_decreases_time", ("descriptor", "compute", "reorder")),
    PerturbationSpec("pipeline.fetch_width", "frontend", "capacity_up_decreases_time", ("descriptor",)),
    PerturbationSpec("pipeline.decode_width", "frontend", "capacity_up_decreases_time", ("descriptor",)),
    PerturbationSpec("pipeline.rename_width", "frontend", "capacity_up_decreases_time", ("descriptor",)),
    PerturbationSpec("pipeline.dispatch_width", "frontend_dispatch", "capacity_up_decreases_time", ("descriptor", "reorder", "dispatch", "sync")),
    PerturbationSpec("pipeline.issue_width", "reorder", "capacity_up_decreases_time", ("reorder", "random_mem")),
    PerturbationSpec("pipeline.commit_width", "reorder", "capacity_up_decreases_time", ("reorder", "random_mem")),
    PerturbationSpec("instruction.simd_width_bits", "compute", "capacity_up_decreases_time", ("compute",)),
    PerturbationSpec("instruction.fp_throughput_per_cycle.vector_sp_add", "compute", "capacity_up_decreases_time", ("compute",)),
    PerturbationSpec("instruction.fp_throughput_per_cycle.vector_sp_mul", "compute", "capacity_up_decreases_time", ("compute",)),
    PerturbationSpec("instruction.fp_throughput_per_cycle.vector_sp_fma", "compute", "capacity_up_decreases_time", ("compute",)),
    PerturbationSpec("memory.bandwidth_gb_s_total", "stream_memory_fallback", "capacity_up_decreases_time", ("stream_mem",)),
    PerturbationSpec("memory.copy_bw_gb_s_1t", "stream_memory", "capacity_up_decreases_time", ("stream_mem",)),
    PerturbationSpec("memory.copy_bw_gb_s_saturated", "stream_memory", "capacity_up_decreases_time", ("stream_mem",)),
    PerturbationSpec("memory.random_gather_bw_gb_s_1t", "random_memory", "capacity_up_decreases_time", ("random_mem",)),
    PerturbationSpec("memory.random_gather_bw_gb_s_saturated", "random_memory", "capacity_up_decreases_time", ("random_mem",)),
    PerturbationSpec("memory.local_mem_delay_ns", "memory_latency", "cost_up_increases_time", ("random_mem",)),
    PerturbationSpec("runtime.parallel_launch_overhead_us_intra", "launch", "cost_up_increases_time", ("dispatch", "sync")),
    PerturbationSpec("runtime.parallel_launch_overhead_us_inter", "launch", "cost_up_increases_time", ("dispatch", "sync")),
    PerturbationSpec("cache.cacheline_bytes", "cacheline", "capacity_up_decreases_time", ("random_mem",)),
    PerturbationSpec("cache.l1d.size", "cache_capacity", "capacity_up_decreases_time", ("random_mem",)),
    PerturbationSpec("cache.l2.size", "cache_capacity", "capacity_up_decreases_time", ("random_mem",)),
    PerturbationSpec("cache.l3_per_die.size", "cache_capacity", "capacity_up_decreases_time", ("random_mem",)),
    PerturbationSpec("cache.l1d.response_latency_cycles", "cache_latency", "cost_up_increases_time", ("random_mem", "descriptor")),
    PerturbationSpec("cache.l2.response_latency_cycles", "cache_latency", "cost_up_increases_time", ("random_mem", "descriptor")),
    PerturbationSpec("cache.l3_per_die.response_latency_cycles", "cache_latency", "cost_up_increases_time", ("random_mem", "descriptor")),
    PerturbationSpec("memory.gather_slice_copy_bw.points", "gather_slice_memory", "capacity_up_decreases_time", ("stream_mem",), "gather_slice_points"),
]

NEGATIVE_CONTROL_SPECS: list[PerturbationSpec] = [
    PerturbationSpec("core.vendor", "identity_negative_control", "no_effect", ()),
    PerturbationSpec("core.model_name", "identity_negative_control", "no_effect", ()),
    PerturbationSpec("core.sockets", "topology_negative_control", "no_effect", ()),
    PerturbationSpec("core.numa_nodes", "topology_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l1d.assoc", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l2.assoc", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l3_per_die.assoc", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l1d.shared_by_cpus", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l2.shared_by_cpus", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l3_per_die.shared_by_cpus", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l1d.tag_latency_cycles", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l1d.data_latency_cycles", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l2.tag_latency_cycles", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l2.data_latency_cycles", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l3_per_die.tag_latency_cycles", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("cache.l3_per_die.data_latency_cycles", "cache_negative_control", "no_effect", ()),
    PerturbationSpec("pipeline.wb_width", "pipeline_negative_control", "no_effect", ()),
    PerturbationSpec("instruction.fp_latency_cycles.vector_sp_add", "instruction_latency_negative_control", "no_effect", ()),
    PerturbationSpec("instruction.fp_latency_cycles.vector_sp_mul", "instruction_latency_negative_control", "no_effect", ()),
    PerturbationSpec("instruction.fp_latency_cycles.vector_sp_fma", "instruction_latency_negative_control", "no_effect", ()),
    PerturbationSpec("host_cross_check.lscpu_model_name", "audit_metadata_negative_control", "no_effect", ()),
    PerturbationSpec("paper_cross_check.cpu_model", "audit_metadata_negative_control", "no_effect", ()),
]

BUNDLE_FIELDS: dict[str, list[str]] = {
    "compute_bundle": [
        "core.cpu_clock",
        "instruction.simd_width_bits",
        "instruction.fp_throughput_per_cycle.vector_sp_add",
        "instruction.fp_throughput_per_cycle.vector_sp_mul",
        "instruction.fp_throughput_per_cycle.vector_sp_fma",
    ],
    "stream_memory_bundle": [
        "memory.bandwidth_gb_s_total",
        "memory.copy_bw_gb_s_1t",
        "memory.copy_bw_gb_s_saturated",
    ],
    "random_memory_bundle": [
        "memory.random_gather_bw_gb_s_1t",
        "memory.random_gather_bw_gb_s_saturated",
        "memory.local_mem_delay_ns",
        "cache.l1d.response_latency_cycles",
        "cache.l2.response_latency_cycles",
        "cache.l3_per_die.response_latency_cycles",
    ],
    "cache_bundle": [
        "cache.cacheline_bytes",
        "cache.l1d.size",
        "cache.l2.size",
        "cache.l3_per_die.size",
        "cache.l1d.response_latency_cycles",
        "cache.l2.response_latency_cycles",
        "cache.l3_per_die.response_latency_cycles",
    ],
    "launch_bundle": [
        "runtime.parallel_launch_overhead_us_intra",
        "runtime.parallel_launch_overhead_us_inter",
        "pipeline.dispatch_width",
    ],
}


def _json_dump(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path_from_config(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_bindings(path: Path) -> tuple[dict[str, Any], dict[str, DatasetBinding]]:
    payload = _read_json(path)
    base_dir = path.parent
    result: dict[str, DatasetBinding] = {}
    for alias, item in dict(payload.get("datasets", {})).items():
        result[str(alias)] = DatasetBinding(
            alias=str(alias),
            data_root=_path_from_config(str(item["data_root"]), base_dir=base_dir),
            hw_config=_path_from_config(str(item["hw_config"]), base_dir=base_dir),
            actual_hardware=str(item.get("actual_hardware", alias)),
            metadata_note=str(item.get("metadata_note", "")),
            model_bundle=(
                _path_from_config(str(item["model_bundle"]), base_dir=base_dir)
                if str(item.get("model_bundle", "")).strip()
                else None
            ),
        )
    if not result:
        raise ValueError(f"No datasets found in binding config: {path}")
    return payload, result


def _split_name(config: dict[str, Any], policy: str) -> str:
    return str(config.get("split_name") or f"source_kunpeng920_dataset3_unseen_{policy}")


def _split_root(binding: DatasetBinding, config: dict[str, Any], policy: str) -> Path:
    return binding.data_root / _split_name(config, policy)


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)


def _value_counts(path: Path, column: str, *, limit: int = 8) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, usecols=[column], low_memory=False)
    except Exception:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype(str).value_counts(dropna=False).head(limit).items()}


def _load_hw_payload(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for hardware generalization audit")
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Hardware config is not a mapping: {path}")
    return payload


def _nested_get(payload: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _nested_set(payload: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = payload
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _scale_scalar(value: Any, factor: float) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return max(int(round(float(value) * factor)), 1)
    if isinstance(value, float):
        return float(value) * factor
    if isinstance(value, str):
        match = QUANTITY_RE.match(value)
        if not match:
            return value
        number = float(match.group(1)) * factor
        suffix = match.group(2)
        if abs(number - round(number)) < 1e-9 and not suffix.strip():
            return str(int(round(number)))
        return f"{number:.6g}{suffix}"
    return value


def _numeric_for_direction(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = QUANTITY_RE.match(value)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None
    if isinstance(value, list):
        write_values = [
            _numeric_for_direction(item.get("write_gb_s"))
            for item in value
            if isinstance(item, dict) and "write_gb_s" in item
        ]
        write_values = [item for item in write_values if item is not None]
        if write_values:
            return float(np.mean(write_values))
    return None


def _direction_from_values(original: Any, mutated: Any, fallback: str) -> str:
    original_num = _numeric_for_direction(original)
    mutated_num = _numeric_for_direction(mutated)
    if original_num is None or mutated_num is None:
        return fallback
    if mutated_num > original_num:
        return "up"
    if mutated_num < original_num:
        return "down"
    return "same"


def _negative_control_value(value: Any) -> Any:
    if isinstance(value, str):
        return f"{value}__negctrl"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return int(value) + 1
    if isinstance(value, float):
        return float(value) * 1.2 + 1e-9
    return value


def _scale_gather_slice_points(payload: dict[str, Any], factor: float) -> bool:
    points = _nested_get(payload, "memory.gather_slice_copy_bw.points", [])
    if not isinstance(points, list) or not points:
        return False
    changed = False
    for point in points:
        if isinstance(point, dict) and "write_gb_s" in point:
            point["write_gb_s"] = _scale_scalar(point["write_gb_s"], factor)
            changed = True
    return changed


def _mutated_payload(
    base_payload: dict[str, Any],
    spec: PerturbationSpec,
    *,
    factor: float | None = None,
    swap_payload: dict[str, Any] | None = None,
    negative_control: bool = False,
) -> tuple[dict[str, Any], Any, Any, bool]:
    payload = json.loads(json.dumps(base_payload))
    original = _nested_get(payload, spec.field)
    if spec.value_kind == "gather_slice_points" and factor is not None:
        changed = _scale_gather_slice_points(payload, factor)
        return payload, "points", f"write_gb_s*{factor:g}", changed
    if swap_payload is not None:
        swapped = _nested_get(swap_payload, spec.field)
        if swapped is None:
            return payload, original, None, False
        _nested_set(payload, spec.field, swapped)
        return payload, original, swapped, swapped != original
    if negative_control:
        mutated = _negative_control_value(original)
    elif factor is not None:
        mutated = _scale_scalar(original, factor)
    else:
        mutated = original
    _nested_set(payload, spec.field, mutated)
    return payload, original, mutated, mutated != original


def _service_column(frame: pd.DataFrame, module: str) -> pd.Series:
    for column in (f"svc_{module}_us", f"svc_{module}_center_us"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _prediction_summary(
    *,
    dataset_alias: str,
    actual_hardware: str,
    route: str,
    field_name: str,
    field_group: str,
    perturbation_kind: str,
    direction: str,
    relation: str,
    expected_modules: tuple[str, ...],
    original_value: Any,
    mutated_value: Any,
    reference_prepared: pd.DataFrame,
    reference_pred: np.ndarray,
    mutated_prepared: pd.DataFrame,
    mutated_pred: np.ndarray,
    weights_before: str,
    weights_after: str,
    ml_raw_hw_params_used: bool = False,
) -> dict[str, Any]:
    ref_pred = np.asarray(reference_pred, dtype=float)
    mut_pred = np.asarray(mutated_pred, dtype=float)
    pred_delta = mut_pred - ref_pred
    module_delta: dict[str, float] = {}
    abs_module_delta_total = 0.0
    expected_abs_delta = 0.0
    expected_signed = np.zeros(len(mutated_prepared), dtype=float)
    for module in SERVICE_MODULES:
        delta = _service_column(mutated_prepared, module).to_numpy(dtype=float) - _service_column(reference_prepared, module).to_numpy(dtype=float)
        abs_sum = float(np.abs(delta).sum())
        module_delta[f"svc_{module}_delta_us"] = float(delta.sum())
        abs_module_delta_total += abs_sum
        if module in expected_modules:
            expected_abs_delta += abs_sum
            expected_signed += delta

    tolerance = 1e-6
    if expected_modules:
        signal = expected_signed
    else:
        signal = pred_delta
    applicable_mask = np.abs(signal) > tolerance
    applicable_rows = int(applicable_mask.sum())

    if relation == "no_effect":
        pass_rate = 1.0 if float(np.max(np.abs(pred_delta))) <= tolerance and abs_module_delta_total <= tolerance else 0.0
        monotonic_pass = bool(pass_rate >= 1.0)
    elif applicable_rows == 0:
        pass_rate = 1.0
        monotonic_pass = True
    else:
        applicable_signal = signal[applicable_mask]
        if relation == "capacity_up_decreases_time":
            if direction in {"up", "target_swap", "bundle_swap"}:
                ok = applicable_signal <= tolerance
            else:
                ok = applicable_signal >= -tolerance
        elif relation == "cost_up_increases_time":
            if direction in {"up", "target_swap", "bundle_swap"}:
                ok = applicable_signal >= -tolerance
            else:
                ok = applicable_signal <= tolerance
        else:
            ok = np.ones_like(applicable_signal, dtype=bool)
        pass_rate = float(np.mean(ok)) if len(ok) else 1.0
        monotonic_pass = bool(pass_rate >= 0.95)

    target_share = float(expected_abs_delta / abs_module_delta_total) if abs_module_delta_total > 0.0 else 1.0
    dominant_module = ""
    if module_delta:
        dominant_module = max(SERVICE_MODULES, key=lambda name: abs(module_delta[f"svc_{name}_delta_us"]))
    return {
        "dataset_alias": dataset_alias,
        "actual_hardware": actual_hardware,
        "route": route,
        "field_name": field_name,
        "field_group": field_group,
        "perturbation_kind": perturbation_kind,
        "direction": direction,
        "relation": relation,
        "expected_modules": ",".join(expected_modules),
        "original_value": json.dumps(original_value, ensure_ascii=False, sort_keys=True) if isinstance(original_value, (dict, list)) else str(original_value),
        "mutated_value": json.dumps(mutated_value, ensure_ascii=False, sort_keys=True) if isinstance(mutated_value, (dict, list)) else str(mutated_value),
        "rows": int(len(mutated_prepared)),
        "applicable_rows": applicable_rows,
        "mean_pred_delta_us": float(np.mean(pred_delta)) if len(pred_delta) else 0.0,
        "mean_abs_pred_delta_us": float(np.mean(np.abs(pred_delta))) if len(pred_delta) else 0.0,
        "max_abs_pred_delta_us": float(np.max(np.abs(pred_delta))) if len(pred_delta) else 0.0,
        "total_abs_svc_delta_us": float(abs_module_delta_total),
        "target_module_share": target_share,
        "dominant_module": dominant_module,
        "monotonic_pass_rate": pass_rate,
        "monotonic_pass": monotonic_pass,
        "weights_sha256_before": str(weights_before),
        "weights_sha256_after": str(weights_after),
        "weights_unchanged": str(weights_before) == str(weights_after),
        "ml_raw_hw_params_used": bool(ml_raw_hw_params_used),
        **module_delta,
    }


def _bool_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    normalized = series.astype(str).str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    numeric = pd.to_numeric(series, errors="coerce")
    values = normalized.fillna(numeric).fillna(0.0)
    return float(values.mean())


def _module_sensitivity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_alias",
        "actual_hardware",
        "route",
        "field_name",
        "field_group",
        "perturbation_kind",
        "direction",
        "relation",
        "module",
        "is_expected_module",
        "signed_svc_delta_us",
        "abs_svc_delta_us",
        "module_abs_delta_share",
        "target_module_share",
        "monotonic_pass",
        "ml_raw_hw_params_used",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        expected = {value for value in str(item.get("expected_modules", "")).split(",") if value}
        module_abs = {
            module: abs(float(pd.to_numeric(pd.Series([item.get(f"svc_{module}_delta_us", 0.0)]), errors="coerce").fillna(0.0).iloc[0]))
            for module in SERVICE_MODULES
        }
        total_abs = float(sum(module_abs.values()))
        for module in SERVICE_MODULES:
            signed_delta = float(pd.to_numeric(pd.Series([item.get(f"svc_{module}_delta_us", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            rows.append(
                {
                    "dataset_alias": item.get("dataset_alias", ""),
                    "actual_hardware": item.get("actual_hardware", ""),
                    "route": item.get("route", ""),
                    "field_name": item.get("field_name", ""),
                    "field_group": item.get("field_group", ""),
                    "perturbation_kind": item.get("perturbation_kind", ""),
                    "direction": item.get("direction", ""),
                    "relation": item.get("relation", ""),
                    "module": module,
                    "is_expected_module": module in expected,
                    "signed_svc_delta_us": signed_delta,
                    "abs_svc_delta_us": module_abs[module],
                    "module_abs_delta_share": float(module_abs[module] / total_abs) if total_abs > 0.0 else 0.0,
                    "target_module_share": float(pd.to_numeric(pd.Series([item.get("target_module_share", 1.0)]), errors="coerce").fillna(1.0).iloc[0]),
                    "monotonic_pass": item.get("monotonic_pass", False),
                    "ml_raw_hw_params_used": item.get("ml_raw_hw_params_used", False),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _fit_route_bundle(route: str, fit_frame: pd.DataFrame, hw_config: Path, policy: str, op_thread_scale_config: dict[str, Any] | None) -> dict[str, Any]:
    if route == WHITEBOX_PRIOR_ONLY_ROUTE:
        return fit_whitebox_family_prior_bundle(
            fit_frame,
            hw_config,
            policy=policy,
            op_thread_scale_config=op_thread_scale_config,
        )
    if route == WHITEBOX_PRIOR_SIMPLE_RF_ROUTE:
        return fit_whitebox_family_simple_rf_bundle(
            fit_frame,
            hw_config,
            policy=policy,
            op_thread_scale_config=op_thread_scale_config,
        )
    if route == WHITEBOX_PHYSICS_GUARDED_RF_ROUTE:
        return fit_whitebox_family_physics_guarded_rf_bundle(
            fit_frame,
            hw_config,
            policy=policy,
            op_thread_scale_config=op_thread_scale_config,
        )
    if route == WHITEBOX_GATHER_SLICE_RF_ROUTE:
        return fit_whitebox_family_gather_slice_rf_bundle(
            fit_frame,
            hw_config,
            policy=policy,
            op_thread_scale_config=op_thread_scale_config,
        )
    raise ValueError(f"Unsupported perturbation route: {route}")


def _predict_route(route: str, bundle: dict[str, Any], frame: pd.DataFrame, override_hw_config: dict[str, Any] | Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    if route == WHITEBOX_PRIOR_ONLY_ROUTE:
        return predict_whitebox_family_prior(bundle, frame, override_hw_config=override_hw_config)
    if route == WHITEBOX_PHYSICS_GUARDED_RF_ROUTE:
        return predict_whitebox_family_physics_guarded_rf(bundle, frame, override_hw_config=override_hw_config)
    return predict_whitebox_family_simple_rf(bundle, frame, override_hw_config=override_hw_config)


def _model_hash(route: str, bundle: dict[str, Any]) -> str:
    if route == WHITEBOX_PRIOR_ONLY_ROUTE:
        return str(bundle.get("hw_config_path", "prior_only"))
    return str(bundle.get("model_sha256", ""))


def _dataset_binding_rows(config: dict[str, Any], bindings: dict[str, DatasetBinding], policy: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for binding in bindings.values():
        split_root = _split_root(binding, config, policy)
        split_manifest = _read_json(split_root / "split_manifest.json") if (split_root / "split_manifest.json").exists() else {}
        index_path = split_root / "pipeline_artifact_index.csv"
        shape_ok = 0
        timeline_ok = 0
        index_rows = 0
        if index_path.exists():
            index = pd.read_csv(index_path, low_memory=False)
            index_rows = int(len(index))
            if "shape_csv" in index.columns:
                shape_ok = int(sum(bool(value) and (split_root / str(value)).exists() for value in index["shape_csv"].fillna("")))
            if "timeline_csv" in index.columns:
                timeline_ok = int(sum(bool(value) and (split_root / str(value)).exists() for value in index["timeline_csv"].fillna("")))
        metadata_counts = _value_counts(split_root / "dataset_full.csv", "hardware_id")
        profile_counts = _value_counts(split_root / "dataset_full.csv", "hardware_profile_name")
        stale_metadata = bool(metadata_counts and binding.actual_hardware not in metadata_counts)
        rows.append(
            {
                "dataset_alias": binding.alias,
                "actual_hardware": binding.actual_hardware,
                "data_root": str(binding.data_root),
                "split_root": str(split_root),
                "hw_config": str(binding.hw_config),
                "data_root_exists": binding.data_root.exists(),
                "split_root_exists": split_root.exists(),
                "split_manifest_exists": (split_root / "split_manifest.json").exists(),
                "split_manifest_sha256": _sha256_file(split_root / "split_manifest.json"),
                "dataset_scope": str(split_manifest.get("dataset_scope", "")),
                "phase_scope": str(split_manifest.get("phase_scope", "")),
                "metadata_note": binding.metadata_note,
                "csv_hardware_id_counts": json.dumps(metadata_counts, sort_keys=True),
                "csv_hardware_profile_name_counts": json.dumps(profile_counts, sort_keys=True),
                "stale_hardware_metadata": stale_metadata,
                "dataset_full_rows": _row_count(split_root / "dataset_full.csv"),
                "train_rows": _row_count(split_root / "train.csv"),
                "test_rows": _row_count(split_root / "test.csv"),
                "e2e_train_rows": _row_count(split_root / "e2e_train.csv"),
                "e2e_test_raw_rows": _row_count(split_root / "e2e_test_raw.csv"),
                "pipeline_artifact_index_rows": index_rows,
                "shape_csv_actual_exists": shape_ok,
                "timeline_csv_actual_exists": timeline_ok,
            }
        )
    return pd.DataFrame(rows)


def _hardware_inventory_rows(bindings: dict[str, DatasetBinding]) -> pd.DataFrame:
    consumed = {spec.field: spec for spec in PARAMETER_SPECS}
    negative = {spec.field: spec for spec in NEGATIVE_CONTROL_SPECS}
    rows: list[dict[str, Any]] = []
    all_fields = sorted(set(consumed) | set(negative))
    for binding in bindings.values():
        payload = _load_hw_payload(binding.hw_config)
        for field in all_fields:
            spec = consumed.get(field) or negative[field]
            value = _nested_get(payload, field)
            rows.append(
                {
                    "dataset_alias": binding.alias,
                    "actual_hardware": binding.actual_hardware,
                    "hw_config": str(binding.hw_config),
                    "field_name": field,
                    "field_group": spec.group,
                    "inventory_role": "positive_control" if field in consumed else "negative_control",
                    "relation": spec.relation,
                    "expected_modules": ",".join(spec.expected_modules),
                    "present": value is not None,
                    "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value),
                }
            )
    return pd.DataFrame(rows)


def _run_calibrator(
    *,
    python_bin: Path,
    artifact_root: Path,
    binding: DatasetBinding,
    bound_hw_config: Path,
    alternate_hw_config: Path,
    policy: str,
    routes: str,
    scales: str,
    passes: int,
    op_thread_scale_mode: str,
    force: bool,
) -> Path:
    output_root = artifact_root
    marker = output_root / "combined_summary.csv"
    if marker.exists() and not force:
        print(f"[hwgen] reuse existing run: {output_root}", flush=True)
        return output_root
    cmd = [
        str(python_bin),
        str(SCRIPT_DIR / "run.py"),
        "--policy",
        policy,
        "--datasets",
        "source",
        "--data-root",
        str(binding.data_root),
        "--artifact-root",
        str(output_root),
        "--source-hw-config",
        str(bound_hw_config),
        "--target-hw-config",
        str(alternate_hw_config),
        "--routes",
        routes,
        "--scales",
        scales,
        "--passes",
        str(int(passes)),
    ]
    if op_thread_scale_mode != "off":
        cmd.extend(["--op-thread-scale-mode", op_thread_scale_mode])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'calibrator'}:{env.get('PYTHONPATH', '')}"
    print("[hwgen] run " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    return output_root


def _read_run_outputs(root: Path, *, dataset_alias: str, run_type: str, actual_hardware: str, bound_hw_config: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _read(name: str) -> pd.DataFrame:
        path = root / name
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path, low_memory=False)
        frame.insert(0, "dataset_alias", dataset_alias)
        frame.insert(1, "actual_hardware", actual_hardware)
        frame.insert(2, "run_type", run_type)
        frame.insert(3, "bound_hw_config", str(bound_hw_config))
        frame.insert(4, "run_artifact_root", str(root))
        return frame

    return _read("combined_summary.csv"), _read("e2e_pipeline_summary.csv")


def _compare_wrong_hw(singleop: pd.DataFrame, e2e: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "metric_scope",
        "dataset_alias",
        "route",
        "scale",
        "correct_mape",
        "wrong_mape",
        "delta_mape_wrong_minus_correct",
        "correct_p90",
        "wrong_p90",
        "delta_p90_wrong_minus_correct",
        "correct_hw_config",
        "wrong_hw_config",
        "wrong_hw_not_better",
    ]
    for metric_scope, frame, mape_col, p90_col in [
        ("singleop", singleop, "mape", "p90"),
        ("e2e_pipeline", e2e, "e2e_mape", "e2e_p90"),
    ]:
        if frame.empty or mape_col not in frame.columns:
            continue
        keys = ["dataset_alias", "route", "scale"]
        for key, group in frame.groupby(keys, dropna=False):
            correct = group.loc[group["run_type"].astype(str).eq("correct_hw")]
            wrong = group.loc[group["run_type"].astype(str).eq("wrong_hw")]
            if correct.empty or wrong.empty:
                continue
            c = correct.sort_values(mape_col).iloc[0]
            w = wrong.sort_values(mape_col).iloc[0]
            rows.append(
                {
                    "metric_scope": metric_scope,
                    "dataset_alias": key[0],
                    "route": key[1],
                    "scale": key[2],
                    "correct_mape": float(c[mape_col]),
                    "wrong_mape": float(w[mape_col]),
                    "delta_mape_wrong_minus_correct": float(w[mape_col]) - float(c[mape_col]),
                    "correct_p90": float(c[p90_col]) if p90_col in c else np.nan,
                    "wrong_p90": float(w[p90_col]) if p90_col in w else np.nan,
                    "delta_p90_wrong_minus_correct": (float(w[p90_col]) - float(c[p90_col])) if p90_col in c and p90_col in w else np.nan,
                    "correct_hw_config": str(c.get("bound_hw_config", "")),
                    "wrong_hw_config": str(w.get("bound_hw_config", "")),
                    "wrong_hw_not_better": bool(float(w[mape_col]) >= float(c[mape_col])),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _prepare_semantic_frames(
    *,
    binding: DatasetBinding,
    config: dict[str, Any],
    policy: str,
    hw_config: Path,
    scale: int,
    passes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    split_root = _split_root(binding, config, policy)
    splits = cal_run._load_raw_splits(split_root, include_pipeline_splits=True)
    fit_pool = cal_run._fit_pool(splits)
    sampled = cal_run._sample_fit_subset(fit_pool, int(scale), seed=0)
    test_raw = splits.get("test_raw", splits["test"])
    pipeline_test_raw = splits.get("e2e_test_raw", test_raw)
    params = base_run._build_analytical_params(sampled, hw_config, passes=int(passes))
    fit_semantic = cal_run.prepare_semantic_static_frame(sampled, hw_config, params)
    test_semantic = cal_run.prepare_semantic_static_frame(test_raw, hw_config, params)
    pipeline_test_semantic = cal_run.prepare_semantic_static_frame(pipeline_test_raw, hw_config, params)
    return fit_semantic, test_semantic, pipeline_test_semantic, split_root


def _perturbation_audit(
    *,
    artifact_root: Path,
    config: dict[str, Any],
    bindings: dict[str, DatasetBinding],
    policy: str,
    routes: list[str],
    scale: int,
    passes: int,
    op_thread_scale_mode: str,
) -> None:
    all_specs = [*PARAMETER_SPECS, *NEGATIVE_CONTROL_SPECS]
    param_rows: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    e2e_summary_frames: list[pd.DataFrame] = []
    e2e_combo_frames: list[pd.DataFrame] = []

    aliases = list(bindings)
    if len(aliases) < 2:
        print("[hwgen] perturbation audit needs at least two dataset bindings for swaps", flush=True)
    for alias, binding in bindings.items():
        alternate_binding = next((bindings[name] for name in aliases if name != alias), binding)
        base_payload = _load_hw_payload(binding.hw_config)
        alternate_payload = _load_hw_payload(alternate_binding.hw_config)
        print(f"[hwgen] perturbation prepare {alias}", flush=True)
        fit_frame, test_frame, pipeline_frame, split_root = _prepare_semantic_frames(
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

        for route in routes:
            print(f"[hwgen] perturbation fit {alias} {route}", flush=True)
            bundle = _fit_route_bundle(route, fit_frame, binding.hw_config, policy, op_thread_scale_config)
            reference_prepared, reference_pred = _predict_route(route, bundle, test_frame)
            reference_pipe_prepared, reference_pipe_pred = _predict_route(route, bundle, pipeline_frame)
            weights_hash = _model_hash(route, bundle)
            ml_raw_hw_params_used = bool(bundle.get("ml_raw_hw_params_used", False))

            swapped_prepared, swapped_pred = _predict_route(route, bundle, test_frame, override_hw_config=alternate_payload)
            frozen_rows.append(
                _prediction_summary(
                    dataset_alias=alias,
                    actual_hardware=binding.actual_hardware,
                    route=route,
                    field_name="__full_hardware_swap__",
                    field_group="full_hardware_swap",
                    perturbation_kind="frozen_swap",
                    direction="target_swap",
                    relation="mixed",
                    expected_modules=tuple(SERVICE_MODULES),
                    original_value=str(binding.hw_config),
                    mutated_value=str(alternate_binding.hw_config),
                    reference_prepared=reference_prepared,
                    reference_pred=reference_pred,
                    mutated_prepared=swapped_prepared,
                    mutated_pred=swapped_pred,
                    weights_before=weights_hash,
                    weights_after=_model_hash(route, bundle),
                    ml_raw_hw_params_used=ml_raw_hw_params_used,
                )
            )

            swapped_pipe_prepared, swapped_pipe_pred = _predict_route(route, bundle, pipeline_frame, override_hw_config=alternate_payload)
            for label, prepared, pred in [
                ("reference", reference_pipe_prepared, reference_pipe_pred),
                ("frozen_swap", swapped_pipe_prepared, swapped_pipe_pred),
            ]:
                prediction_rows = cal_run._prediction_error_frame(
                    policy=policy,
                    dataset=f"{alias}_{label}",
                    route=route,
                    scale=scale,
                    prepared_frame=prepared,
                    pred=pred,
                )
                e2e_summary, e2e_combo, _, _, _ = evaluate_source_test_pipeline(
                    prediction_rows=prediction_rows,
                    split_root=split_root,
                    policy=policy,
                    dataset=f"{alias}_{label}",
                )
                if not e2e_summary.empty:
                    e2e_summary.insert(0, "dataset_alias", alias)
                    e2e_summary.insert(1, "actual_hardware", binding.actual_hardware)
                    e2e_summary.insert(2, "perturbation_kind", label)
                    e2e_summary_frames.append(e2e_summary)
                if not e2e_combo.empty:
                    e2e_combo.insert(0, "dataset_alias", alias)
                    e2e_combo.insert(1, "actual_hardware", binding.actual_hardware)
                    e2e_combo.insert(2, "perturbation_kind", label)
                    e2e_combo_frames.append(e2e_combo)

            for spec in all_specs:
                mutations: list[tuple[str, str, dict[str, Any], Any, Any, bool]] = []
                if spec.relation == "no_effect":
                    payload, original, mutated, changed = _mutated_payload(base_payload, spec, negative_control=True)
                    mutations.append(("negative_control", "mutated", payload, original, mutated, changed))
                else:
                    for factor in (0.8, 0.9, 1.1, 1.2):
                        direction = "up" if factor > 1.0 else "down"
                        payload, original, mutated, changed = _mutated_payload(base_payload, spec, factor=factor)
                        mutations.append((f"factor_{factor:g}", direction, payload, original, mutated, changed))
                    payload, original, mutated, changed = _mutated_payload(base_payload, spec, swap_payload=alternate_payload)
                    mutations.append(("target_swap", _direction_from_values(original, mutated, "target_swap"), payload, original, mutated, changed))

                for perturbation_kind, direction, payload, original, mutated, changed in mutations:
                    if not changed:
                        param_rows.append(
                            {
                                "dataset_alias": alias,
                                "actual_hardware": binding.actual_hardware,
                                "route": route,
                                "field_name": spec.field,
                                "field_group": spec.group,
                                "perturbation_kind": perturbation_kind,
                                "direction": direction,
                                "relation": spec.relation,
                                "expected_modules": ",".join(spec.expected_modules),
                                "original_value": str(original),
                                "mutated_value": str(mutated),
                                "rows": int(len(test_frame)),
                                "applicable_rows": 0,
                                "mean_pred_delta_us": 0.0,
                                "mean_abs_pred_delta_us": 0.0,
                                "max_abs_pred_delta_us": 0.0,
                                "total_abs_svc_delta_us": 0.0,
                                "target_module_share": 1.0,
                                "dominant_module": "",
                                "monotonic_pass_rate": 1.0,
                                "monotonic_pass": True,
                                "weights_sha256_before": weights_hash,
                                "weights_sha256_after": weights_hash,
                                "weights_unchanged": True,
                                "ml_raw_hw_params_used": ml_raw_hw_params_used,
                                "skipped_reason": "field_missing_or_no_change",
                            }
                        )
                        continue
                    mutated_prepared, mutated_pred = _predict_route(route, bundle, test_frame, override_hw_config=payload)
                    row = _prediction_summary(
                        dataset_alias=alias,
                        actual_hardware=binding.actual_hardware,
                        route=route,
                        field_name=spec.field,
                        field_group=spec.group,
                        perturbation_kind=perturbation_kind,
                        direction=direction,
                        relation=spec.relation,
                        expected_modules=spec.expected_modules,
                        original_value=original,
                        mutated_value=mutated,
                        reference_prepared=reference_prepared,
                        reference_pred=reference_pred,
                        mutated_prepared=mutated_prepared,
                        mutated_pred=mutated_pred,
                        weights_before=weights_hash,
                        weights_after=_model_hash(route, bundle),
                        ml_raw_hw_params_used=ml_raw_hw_params_used,
                    )
                    row["skipped_reason"] = ""
                    param_rows.append(row)

            for bundle_name, fields in BUNDLE_FIELDS.items():
                payload = json.loads(json.dumps(base_payload))
                changed_fields: list[str] = []
                for field in fields:
                    swapped = _nested_get(alternate_payload, field)
                    if swapped is not None:
                        _nested_set(payload, field, swapped)
                        changed_fields.append(field)
                if not changed_fields:
                    continue
                mutated_prepared, mutated_pred = _predict_route(route, bundle, test_frame, override_hw_config=payload)
                param_rows.append(
                    _prediction_summary(
                        dataset_alias=alias,
                        actual_hardware=binding.actual_hardware,
                        route=route,
                        field_name=bundle_name,
                        field_group="bundle_swap",
                        perturbation_kind="bundle_swap",
                        direction="bundle_swap",
                        relation="mixed",
                        expected_modules=tuple(SERVICE_MODULES),
                        original_value=",".join(changed_fields),
                        mutated_value=f"from:{alternate_binding.actual_hardware}",
                        reference_prepared=reference_prepared,
                        reference_pred=reference_pred,
                        mutated_prepared=mutated_prepared,
                        mutated_pred=mutated_pred,
                        weights_before=weights_hash,
                        weights_after=_model_hash(route, bundle),
                        ml_raw_hw_params_used=ml_raw_hw_params_used,
                    )
                )

    param_frame = pd.DataFrame(param_rows)
    frozen_frame = pd.DataFrame(frozen_rows)
    _csv_write(artifact_root / "parameter_perturbation_audit.csv", param_frame)
    _csv_write(artifact_root / "frozen_swap_audit.csv", frozen_frame)
    module_frame = _module_sensitivity_frame(param_frame)
    _csv_write(artifact_root / "module_sensitivity_audit.csv", module_frame)
    if not param_frame.empty:
        analytical = param_frame.loc[param_frame["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)].reset_index(drop=True)
        e2e = param_frame.loc[~param_frame["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)].reset_index(drop=True)
        _csv_write(artifact_root / "analytical_param_perturbation.csv", analytical)
        _csv_write(artifact_root / "analytical_hw_perturbation_audit.csv", analytical)
        _csv_write(artifact_root / "e2e_frozen_param_perturbation.csv", e2e)
        for alias in bindings:
            alias_analytical = analytical.loc[analytical["dataset_alias"].astype(str).eq(alias)].reset_index(drop=True)
            alias_module = module_frame.loc[module_frame["dataset_alias"].astype(str).eq(alias)].reset_index(drop=True)
            run_root = artifact_root / "runs" / f"{alias}_correct_hw"
            _csv_write(run_root / "analytical_hw_perturbation_audit.csv", alias_analytical)
            _csv_write(run_root / "module_sensitivity_audit.csv", alias_module)
    if e2e_summary_frames:
        _csv_write(artifact_root / "frozen_swap_e2e_summary.csv", pd.concat(e2e_summary_frames, ignore_index=True))
    if e2e_combo_frames:
        _csv_write(artifact_root / "frozen_swap_e2e_combo_metrics.csv", pd.concat(e2e_combo_frames, ignore_index=True))


def _claim_matrix(artifact_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    analytical_path = artifact_root / "analytical_param_perturbation.csv"
    e2e_path = artifact_root / "e2e_frozen_param_perturbation.csv"
    wrong_path = artifact_root / "wrong_hw_ablation.csv"
    if analytical_path.exists():
        analytical = pd.read_csv(analytical_path, low_memory=False)
        pos = analytical.loc[analytical["relation"].astype(str).ne("no_effect")]
        neg = analytical.loc[analytical["relation"].astype(str).eq("no_effect")]
        rows.append(
            {
                "claim": "Analytical parameters have expected monotonic physical response",
                "evidence": "analytical_param_perturbation.csv",
                "metric": "monotonic_pass_rate",
                "value": _bool_mean(pos.get("monotonic_pass", pd.Series(dtype=float))) if not pos.empty else np.nan,
                "pass": bool((_bool_mean(pos.get("monotonic_pass", pd.Series(dtype=float))) >= 0.95) if not pos.empty else False),
            }
        )
        rows.append(
            {
                "claim": "Negative-control hardware metadata fields do not affect predictions",
                "evidence": "analytical_param_perturbation.csv",
                "metric": "max_abs_pred_delta_us",
                "value": float(pd.to_numeric(neg.get("max_abs_pred_delta_us", pd.Series(dtype=float)), errors="coerce").fillna(0.0).max()) if not neg.empty else np.nan,
                "pass": bool((pd.to_numeric(neg.get("max_abs_pred_delta_us", pd.Series(dtype=float)), errors="coerce").fillna(0.0).max() <= 1e-6) if not neg.empty else False),
            }
        )
    if e2e_path.exists():
        e2e = pd.read_csv(e2e_path, low_memory=False)
        rows.append(
            {
                "claim": "E2E frozen predictions inherit Analytical hardware response",
                "evidence": "e2e_frozen_param_perturbation.csv",
                "metric": "monotonic_pass",
                "value": _bool_mean(e2e.get("monotonic_pass", pd.Series(dtype=float))) if not e2e.empty else np.nan,
                "pass": bool((_bool_mean(e2e.get("monotonic_pass", pd.Series(dtype=float))) >= 0.90) if not e2e.empty else False),
            }
        )
        rows.append(
            {
                "claim": "RF stage does not consume raw hardware fields directly",
                "evidence": "e2e_frozen_param_perturbation.csv",
                "metric": "ml_raw_hw_params_used",
                "value": _bool_mean(e2e.get("ml_raw_hw_params_used", pd.Series(dtype=float))) if not e2e.empty else np.nan,
                "pass": bool((_bool_mean(e2e.get("ml_raw_hw_params_used", pd.Series(dtype=float))) == 0.0) if not e2e.empty else False),
            }
        )
    if wrong_path.exists():
        wrong = pd.read_csv(wrong_path, low_memory=False)
        rows.append(
            {
                "claim": "Correct hardware binding is not worse than wrong-hardware binding",
                "evidence": "wrong_hw_ablation.csv",
                "metric": "wrong_hw_not_better",
                "value": _bool_mean(wrong.get("wrong_hw_not_better", pd.Series(dtype=float))) if not wrong.empty else np.nan,
                "pass": bool((_bool_mean(wrong.get("wrong_hw_not_better", pd.Series(dtype=float))) >= 0.80) if not wrong.empty else False),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hardware generalization audits for Analytical and E2E whitebox routes.")
    parser.add_argument("--binding-config", default=str(DEFAULT_BINDING_CONFIG))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--policy", default="")
    parser.add_argument("--datasets", default="", help="Comma-separated binding aliases. Defaults to all aliases in the binding config.")
    parser.add_argument(
        "--routes",
        default=f"{WHITEBOX_PRIOR_ONLY_ROUTE},{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE},{WHITEBOX_PHYSICS_GUARDED_RF_ROUTE},{WHITEBOX_GATHER_SLICE_RF_ROUTE}",
        help="Routes passed to calibrator/run.py and used by perturbation audit where supported.",
    )
    parser.add_argument(
        "--perturbation-routes",
        default=f"{WHITEBOX_PRIOR_ONLY_ROUTE},{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE},{WHITEBOX_PHYSICS_GUARDED_RF_ROUTE},{WHITEBOX_GATHER_SLICE_RF_ROUTE}",
        help="Comma-separated routes for in-process perturbation audit. Supported: prior, simple_rf_v2, physics_guarded_rf_v3, gather_slice_rf_v5.",
    )
    parser.add_argument("--scales", default="100")
    parser.add_argument("--perturbation-scale", type=int, default=100)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--op-thread-scale-mode", default="fit", choices=["off", "fit"])
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-perturbations", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun baseline calibrator jobs even if outputs already exist.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.binding_config).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    config, all_bindings = _load_bindings(config_path)
    policy = str(args.policy or config.get("policy") or "joint_6d_tuple")
    requested_aliases = [value.strip() for value in str(args.datasets).split(",") if value.strip()]
    bindings = {alias: all_bindings[alias] for alias in (requested_aliases or list(all_bindings))}
    if len(bindings) < 2:
        raise ValueError("At least two dataset bindings are required for wrong-hw and frozen-swap audits.")

    _json_dump(artifact_root / "resolved_binding_config.json", {
        "binding_config": str(config_path),
        "policy": policy,
        "datasets": {
            alias: {
                "data_root": str(binding.data_root),
                "hw_config": str(binding.hw_config),
                "actual_hardware": binding.actual_hardware,
                "metadata_note": binding.metadata_note,
            }
            for alias, binding in bindings.items()
        },
    })
    binding_frame = _dataset_binding_rows(config, bindings, policy)
    _csv_write(artifact_root / "hardware_dataset_binding.csv", binding_frame)
    _csv_write(artifact_root / "hardware_parameter_inventory.csv", _hardware_inventory_rows(bindings))

    singleop_frames: list[pd.DataFrame] = []
    e2e_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    aliases = list(bindings)
    if not args.skip_baselines:
        for alias, binding in bindings.items():
            alternate = next(bindings[name] for name in aliases if name != alias)
            for run_type, hw_config, alt_config in [
                ("correct_hw", binding.hw_config, alternate.hw_config),
                ("wrong_hw", alternate.hw_config, binding.hw_config),
            ]:
                run_root = artifact_root / "runs" / f"{alias}_{run_type}"
                _run_calibrator(
                    python_bin=Path(args.python_bin).resolve(),
                    artifact_root=run_root,
                    binding=binding,
                    bound_hw_config=hw_config,
                    alternate_hw_config=alt_config,
                    policy=policy,
                    routes=str(args.routes),
                    scales=str(args.scales),
                    passes=int(args.passes),
                    op_thread_scale_mode=str(args.op_thread_scale_mode),
                    force=bool(args.force),
                )
                singleop, e2e = _read_run_outputs(
                    run_root,
                    dataset_alias=alias,
                    run_type=run_type,
                    actual_hardware=binding.actual_hardware,
                    bound_hw_config=hw_config,
                )
                if not singleop.empty:
                    singleop_frames.append(singleop)
                if not e2e.empty:
                    e2e_frames.append(e2e)
                run_rows.append(
                    {
                        "dataset_alias": alias,
                        "actual_hardware": binding.actual_hardware,
                        "run_type": run_type,
                        "data_root": str(binding.data_root),
                        "bound_hw_config": str(hw_config),
                        "alternate_hw_config": str(alt_config),
                        "artifact_root": str(run_root),
                    }
                )

    if run_rows:
        _csv_write(artifact_root / "baseline_run_index.csv", pd.DataFrame(run_rows))
    if singleop_frames:
        singleop = pd.concat(singleop_frames, ignore_index=True)
        _csv_write(artifact_root / "baseline_combined_summary.csv", singleop)
    elif args.skip_baselines and (artifact_root / "baseline_combined_summary.csv").exists():
        singleop = pd.read_csv(artifact_root / "baseline_combined_summary.csv", low_memory=False)
    else:
        singleop = pd.DataFrame()
    if e2e_frames:
        e2e = pd.concat(e2e_frames, ignore_index=True)
        _csv_write(artifact_root / "baseline_e2e_pipeline_summary.csv", e2e)
    elif args.skip_baselines and (artifact_root / "baseline_e2e_pipeline_summary.csv").exists():
        e2e = pd.read_csv(artifact_root / "baseline_e2e_pipeline_summary.csv", low_memory=False)
    else:
        e2e = pd.DataFrame()
    wrong_hw = _compare_wrong_hw(singleop, e2e)
    _csv_write(artifact_root / "wrong_hw_ablation.csv", wrong_hw)

    if not args.skip_perturbations:
        perturbation_routes = [value.strip() for value in str(args.perturbation_routes).split(",") if value.strip()]
        _perturbation_audit(
            artifact_root=artifact_root,
            config=config,
            bindings=bindings,
            policy=policy,
            routes=perturbation_routes,
            scale=int(args.perturbation_scale),
            passes=int(args.passes),
            op_thread_scale_mode=str(args.op_thread_scale_mode),
        )

    claim_matrix = _claim_matrix(artifact_root)
    _csv_write(artifact_root / "generalization_claim_matrix.csv", claim_matrix)
    print(f"[hwgen] wrote audit artifacts to {artifact_root}", flush=True)


if __name__ == "__main__":
    main()
