#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - remote portability fallback
    yaml = None

try:
    import reproduce_hw_profile as base
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from hardware_calibration.scripts import reproduce_hw_profile as base  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = ROOT_DIR.parent
HW_CONFIG_DIR = PROJECT_ROOT / "calibrator" / "hw_config"
PROBE_PROFILE_AUTO = "auto"
FIELD_SOURCE_MANUAL = "manual_template"


STABILITY_THRESHOLDS = {
    "core": 5.0,
    "cache": 8.0,
    "memory": 10.0,
    "runtime": 15.0,
}


METRIC_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("cpu_clock_ghz", ("derived", "cpu_clock_ghz"), "core"),
    ("issue_width", ("derived", "pipeline", "issue_width"), "core"),
    ("scalar_dp_add_cycles", ("derived", "instruction", "fp_latency_cycles", "scalar_dp_add"), "core"),
    ("scalar_dp_mul_cycles", ("derived", "instruction", "fp_latency_cycles", "scalar_dp_mul"), "core"),
    ("scalar_dp_fma_cycles", ("derived", "instruction", "fp_latency_cycles", "scalar_dp_fma"), "core"),
    ("vector_sp_add_cycles", ("derived", "instruction", "fp_latency_cycles", "vector_sp_add"), "core"),
    ("vector_sp_mul_cycles", ("derived", "instruction", "fp_latency_cycles", "vector_sp_mul"), "core"),
    ("vector_sp_fma_cycles", ("derived", "instruction", "fp_latency_cycles", "vector_sp_fma"), "core"),
    ("vector_sp_add_tp", ("derived", "instruction", "fp_throughput_per_cycle", "vector_sp_add"), "core"),
    ("vector_sp_mul_tp", ("derived", "instruction", "fp_throughput_per_cycle", "vector_sp_mul"), "core"),
    ("vector_sp_fma_tp", ("derived", "instruction", "fp_throughput_per_cycle", "vector_sp_fma"), "core"),
    ("bandwidth_gb_s_total", ("derived", "memory", "bandwidth_gb_s_total"), "memory"),
    ("copy_bw_gb_s_1t", ("derived", "memory", "copy_bw_gb_s_1t"), "memory"),
    ("copy_bw_gb_s_saturated", ("derived", "memory", "copy_bw_gb_s_saturated"), "memory"),
    ("random_gather_bw_gb_s_1t", ("derived", "memory", "random_gather_bw_gb_s_1t"), "memory"),
    ("random_gather_bw_gb_s_saturated", ("derived", "memory", "random_gather_bw_gb_s_saturated"), "memory"),
    ("parallel_launch_overhead_us_intra", ("derived", "runtime", "parallel_launch_overhead_us_intra"), "runtime"),
    ("parallel_launch_overhead_us_inter", ("derived", "runtime", "parallel_launch_overhead_us_inter"), "runtime"),
]


REQUIRED_MODEL_FIELDS = [
    "core.total_cores",
    "core.cpu_clock",
    "cache.l1d.size",
    "cache.l2.size",
    "cache.l3_per_die.size",
    "cache.l1d.response_latency_cycles",
    "cache.l2.response_latency_cycles",
    "cache.l3_per_die.response_latency_cycles",
    "cache.cacheline_bytes",
    "memory.bandwidth_gb_s_total",
    "memory.local_mem_delay_ns",
    "memory.copy_bw_gb_s_1t",
    "memory.copy_bw_gb_s_saturated",
    "memory.random_gather_bw_gb_s_1t",
    "memory.random_gather_bw_gb_s_saturated",
    "runtime.parallel_launch_overhead_us_intra",
    "runtime.parallel_launch_overhead_us_inter",
    "pipeline.fetch_width",
    "pipeline.decode_width",
    "pipeline.rename_width",
    "pipeline.dispatch_width",
    "pipeline.issue_width",
    "pipeline.commit_width",
    "instruction.simd_width_bits",
    "instruction.fp_latency_cycles.vector_sp_add",
    "instruction.fp_latency_cycles.vector_sp_mul",
    "instruction.fp_latency_cycles.vector_sp_fma",
    "instruction.fp_throughput_per_cycle.vector_sp_add",
    "instruction.fp_throughput_per_cycle.vector_sp_mul",
    "instruction.fp_throughput_per_cycle.vector_sp_fma",
]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read or write hardware YAML")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write candidate YAML")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _get_nested(payload: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _get_summary_value(summary: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = summary
    for part in path:
        value = value[part]
    return float(value)


def _finite_number(value: Any) -> bool:
    try:
        numeric = base._parse_quantity(value) if isinstance(value, str) else float(value)
    except Exception:
        return False
    return math.isfinite(float(numeric)) and float(numeric) > 0.0


def _stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"median": math.nan, "mean": math.nan, "min": math.nan, "max": math.nan, "std": math.nan, "cv_percent": math.inf}
    mean = statistics.mean(finite)
    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    cv = 100.0 * std / mean if abs(mean) > 1e-12 else 0.0
    return {
        "median": statistics.median(finite),
        "mean": mean,
        "min": min(finite),
        "max": max(finite),
        "std": std,
        "cv_percent": cv,
    }


def _format_binary_size(size_bytes: float) -> str:
    value = int(round(float(size_bytes)))
    if value % (1024 * 1024) == 0:
        return f"{value // (1024 * 1024)}MiB"
    if value % 1024 == 0:
        return f"{value // 1024}KiB"
    return str(value)


def _read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return default


def _run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def _parse_size_bytes(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            return float(base._parse_size(value.strip()))
        return float(value)
    except Exception:
        return float(default)


def _parse_cpu_list_len(cpu_list: str) -> int:
    try:
        return len(base._parse_cpu_list(str(cpu_list)))
    except Exception:
        return 0


def _discover_lscpu_payload() -> dict[str, str]:
    try:
        text = _run_text(["lscpu"])
    except Exception:
        return {}
    payload: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _discover_numactl_payload(numa_node: int) -> dict[str, Any]:
    try:
        text = _run_text(["numactl", "-H"])
    except Exception:
        return {"cpu_list_text": "", "cpus": [], "numa_mem_mb": 0}
    cpu_match = re.search(rf"node {int(numa_node)} cpus:\s+(.+)", text)
    size_match = re.search(rf"node {int(numa_node)} size:\s+([0-9]+) MB", text)
    cpu_list_text = cpu_match.group(1).strip() if cpu_match else ""
    cpus = base._parse_cpu_list(cpu_list_text) if cpu_list_text else []
    return {
        "cpu_list_text": cpu_list_text,
        "cpus": cpus,
        "numa_mem_mb": int(size_match.group(1)) if size_match else 0,
    }


def _discover_cache_sysfs(cpu: int) -> dict[str, dict[str, Any]]:
    cache_root = Path(f"/sys/devices/system/cpu/cpu{int(cpu)}/cache")
    result: dict[str, dict[str, Any]] = {}
    for index_dir in sorted(cache_root.glob("index*")):
        level = _read_text(index_dir / "level")
        ctype = _read_text(index_dir / "type").lower()
        if not level:
            continue
        if level == "1" and ctype == "data":
            key = "l1d"
        elif level == "1" and ctype == "instruction":
            key = "l1i"
        elif level == "2":
            key = "l2"
        elif level == "3":
            key = "l3_per_die"
        else:
            continue
        shared_cpu_list = _read_text(index_dir / "shared_cpu_list")
        result[key] = {
            "size": int(_parse_size_bytes(_read_text(index_dir / "size"), 0.0)),
            "assoc": int(float(_read_text(index_dir / "ways_of_associativity", "0") or 0)),
            "cacheline_bytes": int(float(_read_text(index_dir / "coherency_line_size", "64") or 64)),
            "shared_cpu_list": shared_cpu_list,
            "shared_by_cpus": _parse_cpu_list_len(shared_cpu_list) or 1,
        }
    return result


def _round_probe_size(size: float) -> int:
    return _rounded_probe_size(max(float(size), 64.0))


def _auto_ptrchase_sizes(cache: dict[str, dict[str, Any]], numa_mem_mb: int) -> tuple[list[int], dict[str, int]]:
    l1 = float(cache.get("l1d", {}).get("size") or 64 * 1024)
    l2 = float(cache.get("l2", {}).get("size") or 512 * 1024)
    l3 = float(cache.get("l3_per_die", {}).get("size") or 8 * 1024 * 1024)
    numa_bytes = float(max(int(numa_mem_mb), 1)) * 1024.0 * 1024.0

    l1_target = _round_probe_size(0.75 * l1)
    l2_target = _round_probe_size(max(0.75 * l2, 1.25 * l1))
    l3_target = _round_probe_size(max(0.75 * l3, 1.25 * l2))
    dram_target = _round_probe_size(max(256.0 * 1024.0 * 1024.0, 4.0 * l3))
    dram_target = min(dram_target, _round_probe_size(max(numa_bytes * 0.20, l3 * 2.0)))
    if dram_target <= l3:
        dram_target = _round_probe_size(l3 * 2.0)

    candidates: set[int] = set()
    for factor in (0.25, 0.50, 0.70, 0.75, 0.90, 1.10):
        candidates.add(_round_probe_size(l1 * factor))
    for value in (1.25 * l1, 0.50 * l2, 0.75 * l2, 0.90 * l2, 1.10 * l2, 1.25 * l2):
        candidates.add(_round_probe_size(value))
    for value in (1.25 * l2, 0.35 * l3, 0.50 * l3, 0.75 * l3, 0.90 * l3, l3, 1.25 * l3):
        candidates.add(_round_probe_size(value))
    for value in (2.0 * l3, 4.0 * l3, 256.0 * 1024.0 * 1024.0, dram_target):
        candidates.add(_round_probe_size(value))
    candidates.update({l1_target, l2_target, l3_target, dram_target})
    max_probe = max(dram_target, _round_probe_size(2.0 * l3))
    sizes = sorted(size for size in candidates if 64 <= size <= max_probe)
    selected = {"l1": l1_target, "l2": l2_target, "l3": l3_target, "dram": dram_target}
    return sizes, selected


def _auto_stream_elements(cache: dict[str, dict[str, Any]], numa_mem_mb: int) -> int:
    l3 = float(cache.get("l3_per_die", {}).get("size") or 8 * 1024 * 1024)
    numa_bytes = float(max(int(numa_mem_mb), 1)) * 1024.0 * 1024.0
    target_bytes = max(8.0 * l3, 512.0 * 1024.0 * 1024.0)
    target_bytes = min(target_bytes, max(128.0 * 1024.0 * 1024.0, 0.15 * numa_bytes))
    elements = int(target_bytes / 24.0)  # stream triad allocates three double arrays.
    return max(4_000_000, min(elements, 100_000_000))


def _auto_throughput_spec(lscpu: dict[str, str], build_dir: Path) -> tuple[str, dict[str, str], str]:
    flags = str(lscpu.get("Flags", "")).lower().split()
    sve_file = Path("/proc/sys/abi/sve_default_vector_length")
    if "sve" in flags and sve_file.exists():
        return (
            "sve",
            {"vector_sp_add": "fadds", "vector_sp_mul": "fmuls", "vector_sp_fma": "fmas"},
            "lscpu flags contain sve and /proc/sys/abi/sve_default_vector_length exists",
        )
    return (
        "neon",
        {"vector_sp_add": "fadd4s", "vector_sp_mul": "fmul4s", "vector_sp_fma": "fma4s"},
        "SVE unavailable or not enabled; using NEON throughput microbenchmark",
    )


def _cpu_clock_fallback_from_system(lscpu: dict[str, str]) -> tuple[float, str]:
    cpuinfo_max_freq = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    if cpuinfo_max_freq.exists():
        try:
            return float(cpuinfo_max_freq.read_text(encoding="utf-8").strip()) / 1e6, "cpuinfo_max_freq"
        except Exception:
            pass
    if "CPU max MHz" in lscpu:
        try:
            return float(lscpu["CPU max MHz"]) / 1000.0, "lscpu CPU max MHz"
        except Exception:
            pass
    return 0.0, "unavailable"


def _prepare_profile_spec(args: argparse.Namespace, output_dir: Path, template: dict[str, Any] | None) -> dict[str, Any]:
    probe_profile = str(args.probe_profile or PROBE_PROFILE_AUTO)
    if probe_profile != PROBE_PROFILE_AUTO:
        if probe_profile not in base.PROFILE_SPECS:
            raise ValueError(f"unknown --probe-profile {probe_profile!r}; expected auto or one of {sorted(base.PROFILE_SPECS)}")
        spec = deepcopy(base.PROFILE_SPECS[probe_profile])
        base.PROFILE_SPECS[args.profile_name] = spec
        plan = {
            "profile_name": args.profile_name,
            "probe_profile": probe_profile,
            "mode": "compatibility_profile_clone",
            "source_profile_spec": probe_profile,
            "ptrchase_sizes": spec.get("ptrchase_sizes", []),
            "selected_latency_sizes": spec.get("selected_latency_sizes", {}),
            "stream_elements": spec.get("stream_elements"),
            "stream_iterations": spec.get("stream_iterations"),
            "throughput_mode": spec.get("throughput_mode"),
            "throughput_ops": spec.get("throughput_ops", {}),
        }
        _write_auto_probe_plan(plan, output_dir)
        return plan

    lscpu = _discover_lscpu_payload()
    numa = _discover_numactl_payload(int(args.numa_node))
    cpu = int(args.cpu) if args.cpu is not None else int(numa.get("cpus", [0])[0] if numa.get("cpus") else 0)
    threads = int(args.threads) if args.threads is not None else int(len(numa.get("cpus", [])) or 1)
    cache = _discover_cache_sysfs(cpu)
    ptrchase_sizes, selected_latency_sizes = _auto_ptrchase_sizes(cache, int(numa.get("numa_mem_mb", 0) or 0))
    throughput_mode, throughput_ops, throughput_reason = _auto_throughput_spec(lscpu, Path(args.build_dir).resolve())
    fallback_clock, fallback_clock_source = _cpu_clock_fallback_from_system(lscpu)
    stream_elements = _auto_stream_elements(cache, int(numa.get("numa_mem_mb", 0) or 0))
    spec = {
        "fallback_cpu_clock_ghz": fallback_clock,
        "fallback_cpu_clock_source": fallback_clock_source,
        "ptrchase_sizes": ptrchase_sizes,
        "extra_mem_sizes": [],
        "selected_latency_sizes": selected_latency_sizes,
        "throughput_mode": throughput_mode,
        "throughput_ops": throughput_ops,
        "stream_iterations": 5,
        "stream_elements": stream_elements,
        "auto_generated": True,
    }
    base.PROFILE_SPECS[args.profile_name] = spec
    plan = {
        "profile_name": args.profile_name,
        "probe_profile": PROBE_PROFILE_AUTO,
        "mode": "auto_generated_from_host",
        "numa_node": int(args.numa_node),
        "cpu": cpu,
        "threads": threads,
        "numa": numa,
        "lscpu_architecture": lscpu.get("Architecture", ""),
        "lscpu_flags_used_for_isa": lscpu.get("Flags", ""),
        "cache_sysfs": cache,
        "ptrchase_sizes": ptrchase_sizes,
        "selected_latency_sizes": selected_latency_sizes,
        "stream_elements": stream_elements,
        "stream_iterations": spec["stream_iterations"],
        "throughput_mode": throughput_mode,
        "throughput_ops": throughput_ops,
        "throughput_selection_reason": throughput_reason,
        "cpu_clock_fallback_ghz": fallback_clock,
        "cpu_clock_fallback_source": fallback_clock_source,
        "template_used": bool(template),
    }
    _write_auto_probe_plan(plan, output_dir)
    return plan


def _write_auto_probe_plan(plan: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "auto_probe_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# Auto Probe Plan: {plan.get('profile_name', '')}",
        "",
        f"- probe profile: `{plan.get('probe_profile', '')}`",
        f"- mode: `{plan.get('mode', '')}`",
        f"- throughput mode: `{plan.get('throughput_mode', '')}`",
        f"- throughput ops: `{json.dumps(plan.get('throughput_ops', {}), ensure_ascii=False)}`",
        f"- stream elements: `{plan.get('stream_elements', '')}`",
        f"- stream iterations: `{plan.get('stream_iterations', '')}`",
        f"- selected latency sizes: `{json.dumps(plan.get('selected_latency_sizes', {}), ensure_ascii=False)}`",
        f"- ptrchase sizes: `{plan.get('ptrchase_sizes', [])}`",
    ]
    if plan.get("probe_profile") == PROBE_PROFILE_AUTO:
        lines.extend(
            [
                f"- numa node: `{plan.get('numa_node')}`",
                f"- cpu: `{plan.get('cpu')}`",
                f"- threads: `{plan.get('threads')}`",
                f"- throughput selection: {plan.get('throughput_selection_reason', '')}",
            ]
        )
    (output_dir / "auto_probe_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rounded_probe_size(size: float) -> int:
    rounded = int(round(float(size) / 64.0) * 64)
    return max(64, rounded)


def _dynamic_ptrchase_sizes(template: dict[str, Any] | None, profile: str) -> list[int]:
    base_sizes = list(base.PROFILE_SPECS[profile]["ptrchase_sizes"])
    if not template:
        return sorted(set(int(size) for size in base_sizes))
    l1 = float(base._parse_quantity(_get_nested(template, "cache.l1d.size", 64 * 1024)) or 64 * 1024)
    l2 = float(base._parse_quantity(_get_nested(template, "cache.l2.size", 512 * 1024)) or 512 * 1024)
    l3 = float(base._parse_quantity(_get_nested(template, "cache.l3_per_die.size", 8 * 1024 * 1024)) or 8 * 1024 * 1024)
    candidates = set(int(size) for size in base_sizes)
    for factor in (0.50, 0.70, 0.75, 0.90):
        candidates.add(_rounded_probe_size(l1 * factor))
    for value in (l1 * 1.25, l2 * 0.50, l2 * 0.75, l2 * 0.90):
        candidates.add(_rounded_probe_size(value))
    for value in (l2 * 1.25, l3 * 0.50, l3 * 0.75, l3):
        candidates.add(_rounded_probe_size(value))
    for value in (l3 * 1.25, max(64 * 1024 * 1024, l3 * 2.0), max(256 * 1024 * 1024, l3 * 4.0)):
        candidates.add(_rounded_probe_size(value))
    return sorted(size for size in candidates if size > 0)


def _patch_profile_probe_sizes(profile: str, template: dict[str, Any] | None) -> None:
    spec = dict(base.PROFILE_SPECS[profile])
    spec["ptrchase_sizes"] = _dynamic_ptrchase_sizes(template, profile)
    spec["extra_mem_sizes"] = []
    base.PROFILE_SPECS[profile] = spec


def _collect_run_dirs(args: argparse.Namespace, expected_yaml: Path, template: dict[str, Any] | None) -> list[Path]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing_root:
        source = Path(args.reuse_existing_root).resolve()
        run_dirs = sorted(path for path in source.glob("run_*") if path.is_dir())
        if not run_dirs:
            raise RuntimeError(f"no run_* directories found under {source}")
        return run_dirs

    _patch_profile_probe_sizes(args.profile_name, template)
    run_dirs: list[Path] = []
    build_dir = Path(args.build_dir).resolve()
    for run_idx in range(1, args.repeats + 1):
        run_dir = output_dir / f"run_{run_idx}"
        if run_dir.exists() and args.overwrite_existing_runs:
            shutil.rmtree(run_dir)
        elif run_dir.exists() and not args.keep_existing_runs:
            raise RuntimeError(f"run directory already exists: {run_dir}; choose an empty output dir or pass --overwrite-existing-runs")
        run_dir.mkdir(parents=True, exist_ok=True)
        capture_args = SimpleNamespace(
            profile=args.profile_name,
            numa_node=args.numa_node,
            cpu=args.cpu,
            threads=args.threads,
            ptrchase_target_loads=args.ptrchase_target_loads,
            ptrchase_min_rounds=args.ptrchase_min_rounds,
        )
        base._capture(capture_args, expected_yaml, run_dir, build_dir)
        run_dirs.append(run_dir)
    return run_dirs


def _derive_run_summaries(profile: str, numa_node: int, run_dirs: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    derive_args = SimpleNamespace(profile=profile, numa_node=numa_node)
    return [(run_dir, base._derive_summary(derive_args, run_dir)) for run_dir in run_dirs]


def _metric_stats(run_summaries: list[tuple[Path, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metric_name, path, group in METRIC_SPECS:
        values: list[float] = []
        missing_runs: list[str] = []
        for run_dir, summary in run_summaries:
            try:
                values.append(_get_summary_value(summary, path))
            except Exception:
                missing_runs.append(str(run_dir))
        stats = _stats(values)
        stats.update(
            {
                "metric": metric_name,
                "group": group,
                "threshold_cv_percent": STABILITY_THRESHOLDS[group],
                "passed": not missing_runs and stats["cv_percent"] <= STABILITY_THRESHOLDS[group],
                "missing_runs": missing_runs,
                "values": values,
            }
        )
        result[metric_name] = stats
    return result


def _probe_candidates(
    profile: str,
    numa_node: int,
    run_dirs: list[Path],
    first_summary: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    derive_args = SimpleNamespace(profile=profile, numa_node=numa_node)
    ptrchase_by_run = [(run_dir, base._parse_ptrchase(run_dir)) for run_dir in run_dirs]
    all_sizes = sorted(set().union(*(set(values) for _, values in ptrchase_by_run)))

    l1_size = float(first_summary["observed"]["cache"]["l1d"]["size"])
    l2_size = float(first_summary["observed"]["cache"]["l2"]["size"])
    l3_size = float(first_summary["observed"]["cache"]["l3_per_die"]["size"])
    level_rules = {
        "l1": {"legal": lambda size: size < l1_size, "target": 0.75 * l1_size},
        "l2": {"legal": lambda size: l1_size < size < l2_size, "target": 0.75 * l2_size},
        "l3": {"legal": lambda size: l2_size < size <= l3_size, "target": 0.75 * l3_size},
        "dram": {"legal": lambda size: size > l3_size, "target": max(256.0 * 1024.0 * 1024.0, 4.0 * l3_size)},
    }

    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for level, rule in level_rules.items():
        is_legal = rule["legal"]
        target_size = float(rule["target"])
        level_rows: list[dict[str, Any]] = []
        for size in all_sizes:
            values = [ptrchase.get(size) for _, ptrchase in ptrchase_by_run if size in ptrchase]
            complete = len(values) == len(ptrchase_by_run)
            stats = _stats([float(value) for value in values if value is not None])
            legal = bool(is_legal(size))
            stable = complete and legal and stats["cv_percent"] <= STABILITY_THRESHOLDS["cache"]
            row = {
                "level": level,
                "target_size_bytes": int(round(target_size)),
                "size_bytes": int(size),
                "size_label": _format_binary_size(size),
                "median_ns": stats["median"],
                "mean_ns": stats["mean"],
                "min_ns": stats["min"],
                "max_ns": stats["max"],
                "cv_percent": stats["cv_percent"],
                "complete": complete,
                "legal": legal,
                "stable": stable,
                "selected": False,
                "selection_reason": "",
            }
            rows.append(row)
            if legal and complete:
                level_rows.append(row)
        stable_rows = [row for row in level_rows if row["stable"]]
        candidates = stable_rows or level_rows
        if not candidates:
            selected[level] = {"level": level, "passed": False, "error": "no legal complete pointer-chase candidate"}
            continue
        chosen = sorted(
            candidates,
            key=lambda row: (
                abs(math.log(max(float(row["size_bytes"]), 1.0) / max(target_size, 1.0))),
                float(row["cv_percent"]),
            ),
        )[0]
        chosen["selected"] = True
        chosen["selection_reason"] = "stable_nearest_representative_probe" if chosen["stable"] else "nearest_representative_probe_but_failed_stability"
        selected[level] = dict(chosen)
        selected[level]["passed"] = bool(chosen["stable"])

    # Keep selected markers visible in the original rows list.
    selected_keys = {(item["level"], item.get("size_bytes")) for item in selected.values() if "size_bytes" in item}
    for row in rows:
        if (row["level"], row["size_bytes"]) in selected_keys:
            row["selected"] = True
            row["selection_reason"] = selected[row["level"]].get("selection_reason", "")
    return selected, rows


def _first_lscpu_payload(run_dir: Path) -> dict[str, str]:
    return base._parse_lscpu(run_dir / "lscpu.txt")


def _first_numactl_payload(run_dir: Path, numa_node: int) -> dict[str, Any]:
    return base._parse_numactl(run_dir / "numactl_H.txt", numa_node)


def _build_candidate_yaml(
    *,
    profile: str,
    template: dict[str, Any],
    target_yaml: Path,
    run_summaries: list[tuple[Path, dict[str, Any]]],
    metric_stats: dict[str, dict[str, Any]],
    selected_probes: dict[str, dict[str, Any]],
    output_dir: Path,
    numa_node: int,
) -> dict[str, Any]:
    first_run, first_summary = run_summaries[0]
    lscpu = _first_lscpu_payload(first_run)
    numactl = _first_numactl_payload(first_run, numa_node)
    payload = deepcopy(template)
    now = _now_stamp()

    def metric(name: str) -> float:
        return float(metric_stats[name]["median"])

    clock_ghz = metric("cpu_clock_ghz")

    def probe_ns(level: str, fallback_field: str, fallback_is_cycles: bool = False) -> float:
        probe = selected_probes.get(level, {})
        try:
            value = float(probe.get("median_ns"))
            if math.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
        fallback = _get_nested(template, fallback_field)
        if fallback is None:
            return math.nan
        parsed = float(base._parse_quantity(fallback) if isinstance(fallback, str) else fallback)
        return parsed / max(clock_ghz, 1e-9) if fallback_is_cycles else parsed

    def probe_size(level: str) -> int | None:
        probe = selected_probes.get(level, {})
        try:
            return int(probe["size_bytes"])
        except Exception:
            return None

    l1_ns = probe_ns("l1", "cache.l1d.response_latency_cycles", fallback_is_cycles=True)
    l2_ns = probe_ns("l2", "cache.l2.response_latency_cycles", fallback_is_cycles=True)
    l3_ns = probe_ns("l3", "cache.l3_per_die.response_latency_cycles", fallback_is_cycles=True)
    dram_ns = probe_ns("dram", "memory.local_mem_delay_ns", fallback_is_cycles=False)
    l1_cycles = l1_ns * clock_ghz
    l2_cycles = l2_ns * clock_ghz
    l3_cycles = l3_ns * clock_ghz
    l1_size = int(first_summary["observed"]["cache"]["l1d"]["size"])
    l1i_size = int(first_summary["observed"]["cache"]["l1i"]["size"])
    l2_size = int(first_summary["observed"]["cache"]["l2"]["size"])
    l3_size = int(first_summary["observed"]["cache"]["l3_per_die"]["size"])
    total_cores = int(first_summary["observed"]["core"]["total_cores"])
    threads_per_core = int(float(lscpu.get("Thread(s) per core", _get_nested(payload, "core.threads_per_core", 1))))
    issue_width = metric("issue_width")

    payload["profile_name"] = profile
    payload["source_config"] = str(target_yaml.relative_to(PROJECT_ROOT) if target_yaml.is_relative_to(PROJECT_ROOT) else target_yaml)
    notes = [str(item) for item in payload.get("notes", [])] if isinstance(payload.get("notes"), list) else []
    notes.append(f"Auto-calibrated by hardware_calibration/scripts/auto_calibrate_hw_config.py at {now}; candidate source {output_dir}.")
    payload["notes"] = notes

    payload["scope"] = {
        "execution_scope": "single_numa",
        "locality": "numa_local",
        "numa_node_index": int(numa_node),
        "numa_cpu_list": str(numactl["cpu_list_text"]),
        "numa_mem_mb": int(numactl["numa_mem_mb"]),
    }
    payload["core"] = {
        "architecture": lscpu.get("Architecture", _get_nested(payload, "core.architecture", "unknown")),
        "vendor": lscpu.get("Vendor ID", _get_nested(payload, "core.vendor", "unknown")),
        "model_name": lscpu.get("Model name") or lscpu.get("Model") or _get_nested(payload, "core.model_name", profile),
        "sockets": int(float(lscpu.get("Socket(s)", _get_nested(payload, "core.sockets", 1)))),
        "total_cores": total_cores,
        "cores_per_socket": int(float(lscpu.get("Core(s) per socket", total_cores))),
        "threads_per_core": threads_per_core,
        "numa_nodes": int(float(lscpu.get("NUMA node(s)", _get_nested(payload, "core.numa_nodes", 1)))),
        "cores_per_die": total_cores,
        "cpu_clock": f"{metric('cpu_clock_ghz'):.6f}GHz",
    }

    existing_cache = payload.get("cache", {}) if isinstance(payload.get("cache"), dict) else {}
    l1d_assoc = int(first_summary["observed"]["cache"]["l1d"]["assoc"])
    l1i_assoc = int(first_summary["observed"]["cache"]["l1i"]["assoc"])
    l2_assoc = int(first_summary["observed"]["cache"]["l2"]["assoc"])
    l3_assoc = int(first_summary["observed"]["cache"]["l3_per_die"]["assoc"])
    cacheline = int(first_summary["observed"]["cache"]["l1d"].get("cacheline_bytes", 64) or 64)
    l3_cacheline = int(first_summary["observed"]["cache"]["l3_per_die"].get("cacheline_bytes", cacheline) or cacheline)
    payload["cache"] = {
        "l1i": {
            "size": _format_binary_size(l1i_size),
            "assoc": l1i_assoc,
            "shared_by_cpus": 1,
            "tag_latency_cycles": round(l1_cycles, 6),
            "data_latency_cycles": round(l1_cycles, 6),
            "response_latency_cycles": round(l1_cycles, 6),
        },
        "l1d": {
            "size": _format_binary_size(l1_size),
            "assoc": l1d_assoc,
            "shared_by_cpus": 1,
            "tag_latency_cycles": round(l1_cycles, 6),
            "data_latency_cycles": round(l1_cycles, 6),
            "response_latency_cycles": round(l1_cycles, 6),
        },
        "l2": {
            "size": _format_binary_size(l2_size),
            "assoc": l2_assoc,
            "shared_by_cpus": 1,
            "tag_latency_cycles": round(l2_cycles, 6),
            "data_latency_cycles": round(l2_cycles, 6),
            "response_latency_cycles": round(l2_cycles, 6),
        },
        "l3_per_die": {
            "size": _format_binary_size(l3_size),
            "assoc": l3_assoc,
            "shared_by_cpus": total_cores,
            "tag_latency_cycles": round(l3_cycles, 6),
            "data_latency_cycles": round(l3_cycles, 6),
            "response_latency_cycles": round(l3_cycles, 6),
        },
        "cacheline_bytes": cacheline,
    }
    payload["cache"]["coherency_line_bytes_l3"] = l3_cacheline

    memory = deepcopy(payload.get("memory", {})) if isinstance(payload.get("memory"), dict) else {}
    memory.update(
        {
            "numa_nodes": 1,
            "bandwidth_gb_s_per_socket": round(metric("bandwidth_gb_s_total"), 6),
            "bandwidth_gb_s_total": round(metric("bandwidth_gb_s_total"), 6),
            "local_mem_delay_ns": round(dram_ns, 6),
            "copy_bw_gb_s_1t": round(metric("copy_bw_gb_s_1t"), 6),
            "copy_bw_gb_s_saturated": round(metric("copy_bw_gb_s_saturated"), 6),
            "random_gather_bw_gb_s_1t": round(metric("random_gather_bw_gb_s_1t"), 6),
            "random_gather_bw_gb_s_saturated": round(metric("random_gather_bw_gb_s_saturated"), 6),
        }
    )
    payload["memory"] = memory

    payload["runtime"] = {
        "parallel_launch_overhead_us_intra": round(metric("parallel_launch_overhead_us_intra"), 6),
        "parallel_launch_overhead_us_inter": round(metric("parallel_launch_overhead_us_inter"), 6),
    }
    payload["pipeline"] = {
        "fetch_width": round(issue_width, 6),
        "decode_width": round(issue_width, 6),
        "rename_width": round(issue_width, 6),
        "dispatch_width": round(issue_width, 6),
        "issue_width": round(issue_width, 6),
        "wb_width": round(issue_width, 6),
        "commit_width": round(issue_width, 6),
    }

    simd_width = int(round(metric_stats.get("simd_width_bits", {}).get("median", first_summary["derived"]["instruction"]["simd_width_bits"])))
    simd_width = int(round(first_summary["derived"]["instruction"]["simd_width_bits"])) if simd_width <= 0 else simd_width
    instruction = deepcopy(payload.get("instruction", {})) if isinstance(payload.get("instruction"), dict) else {}
    instruction.update(
        {
            "simd_width_bits": simd_width,
            "fp_latency_cycles": {
                "vector_sp_add": round(metric("vector_sp_add_cycles"), 6),
                "vector_sp_mul": round(metric("vector_sp_mul_cycles"), 6),
                "vector_sp_fma": round(metric("vector_sp_fma_cycles"), 6),
                "scalar_dp_add": round(metric("scalar_dp_add_cycles"), 6),
                "scalar_dp_mul": round(metric("scalar_dp_mul_cycles"), 6),
                "scalar_dp_fma": round(metric("scalar_dp_fma_cycles"), 6),
            },
            "fp_throughput_per_cycle": {
                "vector_sp_add": round(metric("vector_sp_add_tp"), 6),
                "vector_sp_mul": round(metric("vector_sp_mul_tp"), 6),
                "vector_sp_fma": round(metric("vector_sp_fma_tp"), 6),
            },
        }
    )
    instruction.setdefault("simd_name", "SVE" if simd_width > 128 else "NEON")
    instruction.setdefault("isa_family", "ARMv8-A+SVE" if simd_width > 128 else "ARMv8-A")
    payload["instruction"] = instruction

    host_cross = deepcopy(payload.get("host_cross_check", {})) if isinstance(payload.get("host_cross_check"), dict) else {}
    host_cross.update(
        {
            "auto_calibrated": True,
            "auto_calibration_timestamp": now,
            "auto_calibration_output_dir": str(output_dir),
            "calibration_summary_path": str(output_dir / "convergence_summary.json"),
            "selected_probe_report_path": str(output_dir / "selected_probe_report.csv"),
            "lscpu_architecture": lscpu.get("Architecture"),
            "lscpu_vendor": lscpu.get("Vendor ID"),
            "lscpu_model_name": lscpu.get("Model name") or lscpu.get("Model"),
            "lscpu_sockets": int(float(lscpu.get("Socket(s)", 1))),
            "lscpu_cores_per_socket": int(float(lscpu.get("Core(s) per socket", total_cores))),
            "lscpu_threads_per_core": threads_per_core,
            "lscpu_numa_nodes": int(float(lscpu.get("NUMA node(s)", 1))),
            "target_numa_node_index": int(numa_node),
            "target_numa_core_count": total_cores,
            "target_numa_mem_mb": int(numactl["numa_mem_mb"]),
            "target_numa_cpu_clock_ghz": round(metric("cpu_clock_ghz"), 6),
            "measured_issue_ipc": round(issue_width, 6),
            "measured_stream_triad_gb_s_threads": total_cores,
            "measured_stream_triad_gb_s": round(metric("bandwidth_gb_s_total"), 6),
            "measured_copy_bw_gb_s_1_thread": round(metric("copy_bw_gb_s_1t"), 6),
            f"measured_copy_bw_gb_s_{total_cores}_threads": round(metric("copy_bw_gb_s_saturated"), 6),
            "measured_random_gather_bw_gb_s_1_thread": round(metric("random_gather_bw_gb_s_1t"), 6),
            f"measured_random_gather_bw_gb_s_{total_cores}_threads": round(metric("random_gather_bw_gb_s_saturated"), 6),
            f"measured_parallel_launch_overhead_us_intra_{total_cores}_threads": round(metric("parallel_launch_overhead_us_intra"), 6),
            f"measured_parallel_launch_overhead_us_inter_{total_cores}_threads": round(metric("parallel_launch_overhead_us_inter"), 6),
            "measured_ptrchase_l1_ns_selected": round(l1_ns, 6),
            "measured_ptrchase_l1_size_bytes_selected": probe_size("l1"),
            "measured_ptrchase_l2_ns_selected": round(l2_ns, 6),
            "measured_ptrchase_l2_size_bytes_selected": probe_size("l2"),
            "measured_ptrchase_l3_ns_selected": round(l3_ns, 6),
            "measured_ptrchase_l3_size_bytes_selected": probe_size("l3"),
            "measured_ptrchase_mem_ns_selected": round(dram_ns, 6),
            "measured_ptrchase_mem_size_bytes_selected": probe_size("dram"),
        }
    )
    for metric_name, stats in metric_stats.items():
        host_cross[f"{metric_name}_median"] = round(float(stats["median"]), 6) if math.isfinite(float(stats["median"])) else None
        host_cross[f"{metric_name}_cv_percent"] = round(float(stats["cv_percent"]), 4) if math.isfinite(float(stats["cv_percent"])) else None
    payload["host_cross_check"] = host_cross
    return payload


def _field_source_rows(
    *,
    candidate: dict[str, Any],
    run_summaries: list[tuple[Path, dict[str, Any]]],
    selected_probes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    first_summary = run_summaries[0][1] if run_summaries else {}
    clock_source = str(_get_nested(first_summary, "derived.cpu_clock_source", ""))
    benchmark_fields = {
        "memory.bandwidth_gb_s_total",
        "memory.copy_bw_gb_s_1t",
        "memory.copy_bw_gb_s_saturated",
        "memory.random_gather_bw_gb_s_1t",
        "memory.random_gather_bw_gb_s_saturated",
        "runtime.parallel_launch_overhead_us_intra",
        "runtime.parallel_launch_overhead_us_inter",
        "instruction.fp_latency_cycles.vector_sp_add",
        "instruction.fp_latency_cycles.vector_sp_mul",
        "instruction.fp_latency_cycles.vector_sp_fma",
        "instruction.fp_throughput_per_cycle.vector_sp_add",
        "instruction.fp_throughput_per_cycle.vector_sp_mul",
        "instruction.fp_throughput_per_cycle.vector_sp_fma",
    }
    scalar_benchmark_fields = {
        "instruction.fp_latency_cycles.scalar_dp_add",
        "instruction.fp_latency_cycles.scalar_dp_mul",
        "instruction.fp_latency_cycles.scalar_dp_fma",
    }
    discovered_fields = {
        "core.total_cores",
        "cache.cacheline_bytes",
        "cache.l1d.size",
        "cache.l2.size",
        "cache.l3_per_die.size",
        "instruction.simd_width_bits",
    }
    pipeline_fields = {
        "pipeline.fetch_width",
        "pipeline.decode_width",
        "pipeline.rename_width",
        "pipeline.dispatch_width",
        "pipeline.issue_width",
        "pipeline.wb_width",
        "pipeline.commit_width",
    }
    cache_latency_levels = {
        "cache.l1d.response_latency_cycles": "l1",
        "cache.l2.response_latency_cycles": "l2",
        "cache.l3_per_die.response_latency_cycles": "l3",
        "memory.local_mem_delay_ns": "dram",
    }
    fields = list(dict.fromkeys(REQUIRED_MODEL_FIELDS + [
        "pipeline.wb_width",
        *sorted(scalar_benchmark_fields),
        "cache.coherency_line_bytes_l3",
        "core.vendor",
        "core.model_name",
        "core.sockets",
        "core.numa_nodes",
    ]))
    rows: list[dict[str, Any]] = []
    for field in fields:
        value = _get_nested(candidate, field)
        source = FIELD_SOURCE_MANUAL
        detail = ""
        if field == "core.cpu_clock":
            source = "measured_benchmark" if "perf cycles/task-clock" in clock_source else "discovered_sysfs"
            detail = clock_source
        elif field in benchmark_fields or field in scalar_benchmark_fields:
            source = "measured_benchmark"
        elif field in discovered_fields or field.startswith("core.") or field == "cache.coherency_line_bytes_l3":
            source = "discovered_sysfs"
        elif field in pipeline_fields:
            source = "derived_proxy"
            detail = "derived from measured issue_width proxy"
        elif field in cache_latency_levels:
            level = cache_latency_levels[field]
            probe = selected_probes.get(level, {})
            if "median_ns" in probe:
                source = "measured_benchmark"
                detail = f"selected ptrchase {level} size={probe.get('size_bytes')} stable={probe.get('passed')}"
        rows.append(
            {
                "field": field,
                "value": value,
                "source": source,
                "detail": detail,
            }
        )
    return rows


def _write_field_source_report(rows: list[dict[str, Any]], output_dir: Path) -> None:
    columns = ["field", "value", "source", "detail"]
    with (output_dir / "field_source_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _gate_rows(
    *,
    args: argparse.Namespace,
    candidate: dict[str, Any],
    metric_stats: dict[str, dict[str, Any]],
    selected_probes: dict[str, dict[str, Any]],
    field_source_rows: list[dict[str, Any]],
    candidate_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, group: str = "gate") -> None:
        rows.append({"name": name, "group": group, "passed": bool(passed), "detail": detail})

    add("promotion_repeat_count", (not args.promote_if_pass) or args.repeats >= 3 or bool(args.reuse_existing_root), f"repeats={args.repeats}, reuse_existing={bool(args.reuse_existing_root)}")
    add("topology_total_cores", _finite_number(_get_nested(candidate, "core.total_cores")), str(_get_nested(candidate, "core.total_cores")), "topology")
    add("topology_numa_cpu_list", bool(_get_nested(candidate, "scope.numa_cpu_list", "")), str(_get_nested(candidate, "scope.numa_cpu_list", "")), "topology")
    add("topology_simd_width", _finite_number(_get_nested(candidate, "instruction.simd_width_bits")), str(_get_nested(candidate, "instruction.simd_width_bits")), "topology")
    for field in ("cache.l1d.size", "cache.l2.size", "cache.l3_per_die.size"):
        add(f"required_{field}", _finite_number(_get_nested(candidate, field)), str(_get_nested(candidate, field)), "required_fields")
    for field in REQUIRED_MODEL_FIELDS:
        add(f"required_{field}", _finite_number(_get_nested(candidate, field)), str(_get_nested(candidate, field)), "required_fields")
    source_by_field = {str(row.get("field")): str(row.get("source")) for row in field_source_rows}
    manual_required = [
        field
        for field in REQUIRED_MODEL_FIELDS
        if source_by_field.get(field, FIELD_SOURCE_MANUAL) == FIELD_SOURCE_MANUAL
    ]
    add(
        "no_manual_template_for_required_whitebox_fields",
        not manual_required,
        "manual_template_fields=" + ",".join(manual_required),
        "field_sources",
    )
    for level, probe in selected_probes.items():
        add(f"probe_{level}_legal_stable", bool(probe.get("passed")), json.dumps(probe, ensure_ascii=False), "cache_probe")
    for metric_name, stats in metric_stats.items():
        passed = bool(stats["passed"])
        detail = f"median={stats['median']}, cv={stats['cv_percent']:.4f}, threshold={stats['threshold_cv_percent']}, missing={len(stats['missing_runs'])}"
        add(f"stable_{metric_name}", passed, detail, str(stats["group"]))
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from calibrator.whitebox_family_routes import _load_hw_params

        _load_hw_params(candidate_path, require_measured_whitebox=True)
        add("load_hw_params", True, "candidate is readable by _load_hw_params", "model_contract")
    except Exception as exc:  # pragma: no cover - import/runtime environment dependent
        add("load_hw_params", False, repr(exc), "model_contract")
    return rows


def _write_gate_reports(rows: list[dict[str, Any]], output_dir: Path) -> bool:
    ok = all(bool(row["passed"]) for row in rows)
    payload = {"ok": ok, "rows": rows}
    (output_dir / "calibration_gate_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Calibration Gate Report", "", f"- overall: `{'PASS' if ok else 'FAIL'}`", "", "| Group | Gate | Pass | Detail |", "| --- | --- | --- | --- |"]
    for row in rows:
        detail = str(row["detail"]).replace("\n", " ")
        if len(detail) > 220:
            detail = detail[:217] + "..."
        lines.append(f"| `{row['group']}` | `{row['name']}` | `{'yes' if row['passed'] else 'no'}` | {detail} |")
    (output_dir / "calibration_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


def _write_probe_report(rows: list[dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "selected_probe_report.csv"
    columns = ["level", "target_size_bytes", "size_bytes", "size_label", "median_ns", "mean_ns", "min_ns", "max_ns", "cv_percent", "complete", "legal", "stable", "selected", "selection_reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_yaml_diff_report(old: dict[str, Any], new: dict[str, Any], output_dir: Path) -> None:
    keys = REQUIRED_MODEL_FIELDS + [
        "pipeline.wb_width",
        "instruction.fp_latency_cycles.scalar_dp_add",
        "instruction.fp_latency_cycles.scalar_dp_mul",
        "instruction.fp_latency_cycles.scalar_dp_fma",
    ]
    lines = ["# YAML Diff Report", "", "| Field | Old | New | Delta % |", "| --- | ---: | ---: | ---: |"]
    for key in keys:
        old_value = _get_nested(old, key)
        new_value = _get_nested(new, key)
        delta_text = ""
        try:
            old_num = float(base._parse_quantity(old_value) if isinstance(old_value, str) else old_value)
            new_num = float(base._parse_quantity(new_value) if isinstance(new_value, str) else new_value)
            delta_text = f"{100.0 * (new_num - old_num) / max(abs(old_num), 1e-9):.3f}"
        except Exception:
            delta_text = ""
        lines.append(f"| `{key}` | `{old_value}` | `{new_value}` | `{delta_text}` |")
    (output_dir / "yaml_diff_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_convergence_summary(
    run_summaries: list[tuple[Path, dict[str, Any]]],
    metric_stats: dict[str, dict[str, Any]],
    selected_probes: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    summary = deepcopy(run_summaries[0][1])
    for metric_name, path, _ in METRIC_SPECS:
        target = summary
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = metric_stats[metric_name]["median"]
    clock = metric_stats["cpu_clock_ghz"]["median"]
    summary["selected_probes"] = {
        level: {
            "size_bytes": item.get("size_bytes"),
            "size_label": item.get("size_label"),
            "median_ns": item.get("median_ns"),
            "cv_percent": item.get("cv_percent"),
            "passed": item.get("passed"),
        }
        for level, item in selected_probes.items()
    }
    if all(level in selected_probes and "median_ns" in selected_probes[level] for level in ("l1", "l2", "l3", "dram")):
        summary["derived"]["cache_response_latency_ns"] = {
            "l1": selected_probes["l1"]["median_ns"],
            "l2": selected_probes["l2"]["median_ns"],
            "l3": selected_probes["l3"]["median_ns"],
            "dram": selected_probes["dram"]["median_ns"],
        }
        summary["derived"]["cache_response_latency_cycles"] = {
            "l1": selected_probes["l1"]["median_ns"] * clock,
            "l2": selected_probes["l2"]["median_ns"] * clock,
            "l3": selected_probes["l3"]["median_ns"] * clock,
        }
        summary["derived"]["memory"]["local_mem_delay_ns"] = selected_probes["dram"]["median_ns"]
    stats_payload = {
        "metrics": metric_stats,
        "runs": [{"run_dir": str(run_dir)} for run_dir, _ in run_summaries],
    }
    (output_dir / "convergence_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "convergence_stats.json").write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _promote_if_requested(args: argparse.Namespace, target_yaml: Path, candidate_path: Path, gate_ok: bool, output_dir: Path) -> None:
    record = {
        "requested": bool(args.promote_if_pass),
        "promoted": False,
        "gate_ok": gate_ok,
        "target_yaml": str(target_yaml),
        "candidate_yaml": str(candidate_path),
        "backup_yaml": None,
    }
    if args.promote_if_pass and gate_ok:
        stamp = _now_stamp()
        backup = target_yaml.with_name(f"{target_yaml.name}.bak.{stamp}")
        if target_yaml.exists():
            shutil.copy2(target_yaml, backup)
            record["backup_yaml"] = str(backup)
        target_yaml.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, target_yaml)
        record["promoted"] = True
        record["promoted_at"] = stamp
    (output_dir / "promotion_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically capture, gate, and optionally promote a calibrated hardware YAML.")
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--probe-profile", default=PROBE_PROFILE_AUTO, choices=[PROBE_PROFILE_AUTO, *sorted(base.PROFILE_SPECS)])
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--promote-if-pass", action="store_true")
    parser.add_argument("--reuse-existing-root", default=None)
    parser.add_argument("--keep-existing-runs", action="store_true")
    parser.add_argument("--overwrite-existing-runs", action="store_true")
    parser.add_argument("--expected-yaml", default=None)
    parser.add_argument("--template-yaml", default=None)
    parser.add_argument("--build-dir", default=str(ROOT_DIR / "build"))
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--ptrchase-target-loads", type=int, default=600000000)
    parser.add_argument("--ptrchase-min-rounds", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_yaml = Path(args.expected_yaml).resolve() if args.expected_yaml else (HW_CONFIG_DIR / f"{args.profile_name}.yaml").resolve()
    template_path = Path(args.template_yaml).resolve() if args.template_yaml else target_yaml
    if args.template_yaml and yaml is not None and not template_path.exists():
        raise FileNotFoundError(f"template YAML does not exist: {template_path}")
    template = _load_yaml(template_path) if yaml is not None and template_path.exists() else {}
    _prepare_profile_spec(args, output_dir, template)

    run_dirs = _collect_run_dirs(args, target_yaml, template)
    run_summaries = _derive_run_summaries(args.profile_name, args.numa_node, run_dirs)
    metric_stats = _metric_stats(run_summaries)
    selected_probes, probe_rows = _probe_candidates(args.profile_name, args.numa_node, run_dirs, run_summaries[0][1])
    _write_probe_report(probe_rows, output_dir)
    _write_convergence_summary(run_summaries, metric_stats, selected_probes, output_dir)

    if yaml is None:
        rows = [
            {
                "name": "pyyaml_available",
                "group": "dependency",
                "passed": False,
                "detail": "PyYAML is required for candidate YAML generation and promotion; JSON evidence was still generated.",
            }
        ]
        _write_gate_reports(rows, output_dir)
        _promote_if_requested(args, target_yaml, output_dir / "candidate_hw_config.yaml", False, output_dir)
        return 1

    candidate = _build_candidate_yaml(
        profile=args.profile_name,
        template=template,
        target_yaml=target_yaml,
        run_summaries=run_summaries,
        metric_stats=metric_stats,
        selected_probes=selected_probes,
        output_dir=output_dir,
        numa_node=args.numa_node,
    )
    candidate_path = output_dir / "candidate_hw_config.yaml"
    _write_yaml(candidate_path, candidate)
    _write_yaml_diff_report(template, candidate, output_dir)
    field_sources = _field_source_rows(candidate=candidate, run_summaries=run_summaries, selected_probes=selected_probes)
    _write_field_source_report(field_sources, output_dir)

    rows = _gate_rows(args=args, candidate=candidate, metric_stats=metric_stats, selected_probes=selected_probes, field_source_rows=field_sources, candidate_path=candidate_path)
    gate_ok = _write_gate_reports(rows, output_dir)
    _promote_if_requested(args, target_yaml, candidate_path, gate_ok, output_dir)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
