#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - remote portability fallback
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
HW_CONFIG_DIR = ROOT_DIR.parent / "calibrator" / "hw_config"


PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "kunpeng920_single_numa": {
        "fallback_cpu_clock_ghz": 2.6,
        "ptrchase_sizes": [32768, 49152, 65536, 131072, 262144, 393216, 524288, 786432, 1048576, 2097152, 8388608, 16777216, 25165824],
        "extra_mem_sizes": [(67108864, 95), (268435456, 24)],
        "selected_latency_sizes": {"l1": 49152, "l2": 262144, "l3": 16777216, "dram": 268435456},
        "throughput_mode": "neon",
        "throughput_ops": {"vector_sp_add": "fadd4s", "vector_sp_mul": "fmul4s", "vector_sp_fma": "fma4s"},
        "stream_iterations": 5,
        "stream_elements": 50000000,
    },
    "s920x20_single_numa": {
        "fallback_cpu_clock_ghz": 2.9,
        "ptrchase_sizes": [32768, 49152, 65536, 131072, 262144, 524288, 1048576, 1310720, 2097152, 8388608, 33554432, 67108864, 73400320, 268435456],
        "extra_mem_sizes": [],
        "selected_latency_sizes": {"l1": 49152, "l2": 1048576, "l3": 73400320, "dram": 268435456},
        "throughput_mode": "sve",
        "throughput_ops": {"vector_sp_add": "fadds", "vector_sp_mul": "fmuls", "vector_sp_fma": "fmas"},
        "stream_iterations": 5,
        "stream_elements": 50000000,
    },
}


COMPARE_SPECS: list[tuple[str, str, float, float]] = [
    ("core.total_cores", "observed.core.total_cores", 0.0, 0.0),
    ("core.cpu_clock", "derived.cpu_clock_ghz", 0.0, 0.05),
    ("cache.l1d.size", "observed.cache.l1d.size", 0.0, 0.0),
    ("cache.l2.size", "observed.cache.l2.size", 0.0, 0.0),
    ("cache.l3_per_die.size", "observed.cache.l3_per_die.size", 0.0, 0.0),
    ("cache.l1d.response_latency_cycles", "derived.cache_response_latency_cycles.l1", 0.0, 0.12),
    ("cache.l2.response_latency_cycles", "derived.cache_response_latency_cycles.l2", 0.0, 0.12),
    ("cache.l3_per_die.response_latency_cycles", "derived.cache_response_latency_cycles.l3", 0.0, 0.12),
    ("memory.bandwidth_gb_s_total", "derived.memory.bandwidth_gb_s_total", 0.0, 0.1),
    ("memory.local_mem_delay_ns", "derived.memory.local_mem_delay_ns", 0.0, 0.15),
    ("pipeline.issue_width", "derived.pipeline.issue_width", 0.0, 0.05),
    ("instruction.simd_width_bits", "derived.instruction.simd_width_bits", 0.0, 0.0),
    ("instruction.fp_latency_cycles.vector_sp_add", "derived.instruction.fp_latency_cycles.vector_sp_add", 0.0, 0.05),
    ("instruction.fp_latency_cycles.vector_sp_mul", "derived.instruction.fp_latency_cycles.vector_sp_mul", 0.0, 0.05),
    ("instruction.fp_latency_cycles.vector_sp_fma", "derived.instruction.fp_latency_cycles.vector_sp_fma", 0.0, 0.05),
    ("instruction.fp_latency_cycles.scalar_dp_add", "derived.instruction.fp_latency_cycles.scalar_dp_add", 0.0, 0.05),
    ("instruction.fp_latency_cycles.scalar_dp_mul", "derived.instruction.fp_latency_cycles.scalar_dp_mul", 0.0, 0.05),
    ("instruction.fp_latency_cycles.scalar_dp_fma", "derived.instruction.fp_latency_cycles.scalar_dp_fma", 0.0, 0.05),
    ("instruction.fp_throughput_per_cycle.vector_sp_add", "derived.instruction.fp_throughput_per_cycle.vector_sp_add", 0.0, 0.08),
    ("instruction.fp_throughput_per_cycle.vector_sp_mul", "derived.instruction.fp_throughput_per_cycle.vector_sp_mul", 0.0, 0.08),
    ("instruction.fp_throughput_per_cycle.vector_sp_fma", "derived.instruction.fp_throughput_per_cycle.vector_sp_fma", 0.0, 0.08),
]


def _run(cmd: list[str], *, stdout_path: Path | None = None, stderr_path: Path | None = None) -> None:
    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path else None
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path else None
    try:
        subprocess.run(cmd, check=True, stdout=stdout_handle, stderr=stderr_handle)
    finally:
        if stdout_handle:
            stdout_handle.close()
        if stderr_handle:
            stderr_handle.close()


def _parse_size(text: str) -> float:
    text = text.strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMG]i?B?|[KMG])", text)
    if not match:
        return float(text)
    value = float(match.group(1))
    unit = match.group(2)
    scale = {
        "K": 1024.0,
        "KB": 1000.0,
        "KiB": 1024.0,
        "M": 1024.0**2,
        "MB": 1000.0**2,
        "MiB": 1024.0**2,
        "G": 1024.0**3,
        "GB": 1000.0**3,
        "GiB": 1024.0**3,
    }[unit]
    return value * scale


def _parse_quantity(text: str) -> float:
    stripped = text.strip()
    if stripped.endswith("GHz"):
        return float(stripped[:-3])
    if stripped.endswith("MHz"):
        return float(stripped[:-3]) / 1000.0
    return _parse_size(stripped)


def _parse_cpu_list(cpu_list: str) -> list[int]:
    cpus: list[int] = []
    for part in cpu_list.replace(",", " ").split():
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not available in this Python environment")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            raise KeyError(dotted_key)
        value = value[part]
    return value


def _parse_perf_value(path: Path, event_substring: str) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        cols = line.split(",")
        if len(cols) < 3 or event_substring not in cols[2]:
            continue
        return float(cols[0])
    raise ValueError(f"{event_substring} not found in {path}")


def _parse_perf_cycles(path: Path) -> float:
    return _parse_perf_value(path, "cycles")


def _parse_perf_task_clock_seconds(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        cols = line.split(",")
        if len(cols) < 3 or "task-clock" not in cols[2]:
            continue
        value = float(cols[0])
        unit = cols[1].strip()
        if unit == "sec":
            return value
        if unit == "msec":
            return value * 1e-3
        if unit == "usec":
            return value * 1e-6
        if unit == "nsec":
            return value * 1e-9
        raise ValueError(f"unsupported task-clock unit {unit} in {path}")
    raise ValueError(f"task-clock not found in {path}")


def _parse_stdout_counts(path: Path) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for part in path.read_text(encoding="utf-8").strip().split():
        if "=" in part:
            key, value = part.split("=", 1)
            tokens[key] = value
    return tokens


def _parse_lscpu_cache(path: Path) -> dict[str, dict[str, float]]:
    rows = path.read_text(encoding="utf-8").splitlines()
    header = rows[0].split()
    result: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        cols = row.split()
        if not cols:
            continue
        name = cols[0]
        result[name] = {
            "one_size_bytes": _parse_size(cols[1]),
            "ways": float(cols[3]),
            "level": float(cols[5]),
            "coherency_size": _parse_size(cols[7]) if len(cols) > 7 else 0.0,
        }
    return result


def _parse_lscpu(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _parse_numactl(path: Path, numa_node: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    cpu_match = re.search(rf"node {numa_node} cpus:\s+(.+)", text)
    size_match = re.search(rf"node {numa_node} size:\s+([0-9]+) MB", text)
    if not cpu_match or not size_match:
        raise ValueError("failed to parse numactl output")
    cpu_list_text = cpu_match.group(1).strip()
    cpus = _parse_cpu_list(cpu_list_text)
    return {
        "cpu_list_text": cpu_list_text,
        "cpus": cpus,
        "numa_mem_mb": int(size_match.group(1)),
    }


def _capture(args: argparse.Namespace, expected_yaml: Path, raw_dir: Path, build_dir: Path) -> None:
    spec = PROFILE_SPECS[args.profile]
    raw_dir.mkdir(parents=True, exist_ok=True)
    _run([str(SCRIPT_DIR / "build_benchmarks.sh"), str(build_dir)])

    _run(["lscpu"], stdout_path=raw_dir / "lscpu.txt")
    _run(["numactl", "-H"], stdout_path=raw_dir / "numactl_H.txt")
    _run(["lscpu", "-B", "-C"], stdout_path=raw_dir / "lscpu_cache.txt")
    _run(["perf", "--version"], stdout_path=raw_dir / "perf_version.txt")

    lscpu_payload = _parse_lscpu(raw_dir / "lscpu.txt")
    numactl_payload = _parse_numactl(raw_dir / "numactl_H.txt", args.numa_node)
    cpu = args.cpu if args.cpu is not None else numactl_payload["cpus"][0]
    threads = args.threads if args.threads is not None else len(numactl_payload["cpus"])

    mem_type_lines = []
    for path in sorted(Path("/sys/devices/system/edac/mc").glob("mc*/csrow*/mem_type")):
        mem_type_lines.append(f"{path}:{path.read_text(encoding='utf-8').strip()}")
    (raw_dir / "edac_mem_type.txt").write_text("\n".join(mem_type_lines) + ("\n" if mem_type_lines else ""), encoding="utf-8")

    size_lines = []
    for path in sorted(Path("/sys/devices/system/edac/mc").glob("mc*/csrow*/size_mb")):
        size_lines.append(f"{path}:{path.read_text(encoding='utf-8').strip()}")
    (raw_dir / "edac_size_mb.txt").write_text("\n".join(size_lines) + ("\n" if size_lines else ""), encoding="utf-8")

    ptrchase_lines: list[str] = []
    for size in spec["ptrchase_sizes"]:
        rounds = max(int(args.ptrchase_target_loads / (size // 64)), args.ptrchase_min_rounds)
        cmd = ["numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "aarch64_cpu_calib"), "ptrchase", str(cpu), str(size), str(rounds)]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        ptrchase_lines.append(completed.stdout.strip())
    (raw_dir / "ptrchase_sweep.txt").write_text("\n".join(ptrchase_lines) + "\n", encoding="utf-8")

    for size, rounds in spec["extra_mem_sizes"]:
        _run(
            ["numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "aarch64_cpu_calib"), "ptrchase", str(cpu), str(size), str(rounds)],
            stdout_path=raw_dir / f"ptrchase_mem_{size // (1024 * 1024)}m.txt",
        )

    _run(
        ["perf", "stat", "-x,", "-e", "cycles,task-clock,instructions", str(build_dir / "aarch64_cpu_calib"), "issue", str(cpu), "200000000"],
        stdout_path=raw_dir / "issue_width.stdout.txt",
        stderr_path=raw_dir / "issue_width.perf.csv",
    )

    latency_ops = [("scalar_fadd64", ["scalar_chain", str(cpu), "fadd64", "50000000"]),
                   ("scalar_fmul64", ["scalar_chain", str(cpu), "fmul64", "50000000"]),
                   ("scalar_fma64", ["scalar_chain", str(cpu), "fma64", "50000000"]),
                   ("neon_fadd4s", ["neon_chain", str(cpu), "fadd4s", "50000000"]),
                   ("neon_fmul4s", ["neon_chain", str(cpu), "fmul4s", "50000000"]),
                   ("neon_fma4s", ["neon_chain", str(cpu), "fma4s", "50000000"])]
    for stem, subcmd in latency_ops:
        _run(
            ["perf", "stat", "-x,", "-e", "cycles,instructions", str(build_dir / "aarch64_cpu_calib"), *subcmd],
            stdout_path=raw_dir / f"{stem}.stdout.txt",
            stderr_path=raw_dir / f"{stem}.perf.csv",
        )

    throughput_mode = spec["throughput_mode"]
    if throughput_mode == "sve":
        sve_path = build_dir / "aarch64_sve_fp_throughput"
        if not sve_path.exists():
            raise FileNotFoundError("SVE throughput binary was not built")
        _run(["bash", "-lc", "cat /proc/sys/abi/sve_default_vector_length"], stdout_path=raw_dir / "sve_default_vector_length.txt")
        throughput_binary = sve_path
        prefix = "sve"
    else:
        throughput_binary = build_dir / "aarch64_neon_fp_throughput"
        prefix = "neon"

    for key, op in spec["throughput_ops"].items():
        stem = f"{prefix}_throughput_{op}"
        _run(
            ["perf", "stat", "-x,", "-e", "cycles,instructions", str(throughput_binary), str(cpu), op, "50000000"],
            stdout_path=raw_dir / f"{stem}.stdout.txt",
            stderr_path=raw_dir / f"{stem}.perf.csv",
        )

    cpuinfo_max_freq = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    if cpuinfo_max_freq.exists():
        (raw_dir / "cpuinfo_max_freq.txt").write_text(cpuinfo_max_freq.read_text(encoding="utf-8"), encoding="utf-8")

    _run(
        ["env", f"OMP_NUM_THREADS={threads}", "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "numa_stream_omp"), str(spec["stream_elements"]), str(spec["stream_iterations"]), "3.0"],
        stdout_path=raw_dir / "stream_triad.txt",
    )
    _run(
        ["env", "OMP_NUM_THREADS=1", "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "copy_bw_omp"), str(spec["stream_elements"]), str(spec["stream_iterations"])],
        stdout_path=raw_dir / "copy_bw_1t.txt",
    )
    _run(
        ["env", f"OMP_NUM_THREADS={threads}", "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "copy_bw_omp"), str(spec["stream_elements"]), str(spec["stream_iterations"])],
        stdout_path=raw_dir / "copy_bw_saturated.txt",
    )
    _run(
        ["env", "OMP_NUM_THREADS=1", "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "random_gather_bw_omp"), str(spec["stream_elements"] // 2), str(spec["stream_iterations"])],
        stdout_path=raw_dir / "random_gather_bw_1t.txt",
    )
    _run(
        ["env", f"OMP_NUM_THREADS={threads}", "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "random_gather_bw_omp"), str(spec["stream_elements"] // 2), str(spec["stream_iterations"])],
        stdout_path=raw_dir / "random_gather_bw_saturated.txt",
    )
    _run(
        ["env", f"OMP_NUM_THREADS={threads}", "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "thread_launch_overhead_omp"), "parallel_for_empty", "10000"],
        stdout_path=raw_dir / "thread_launch_overhead_intra.txt",
    )
    _run(
        ["env", f"OMP_NUM_THREADS={threads}", "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}", str(build_dir / "thread_launch_overhead_omp"), "task_spawn_empty", "5000"],
        stdout_path=raw_dir / "thread_launch_overhead_inter.txt",
    )


def _parse_ptrchase(raw_dir: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    for line in (raw_dir / "ptrchase_sweep.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = _parse_stdout_counts_from_line(line)
        result[int(parts["size_bytes"])] = float(parts["ns_per_load"])
    for extra_file in raw_dir.glob("ptrchase_mem_*m.txt"):
        line = extra_file.read_text(encoding="utf-8").strip()
        if line:
            parts = _parse_stdout_counts_from_line(line)
            result[int(parts["size_bytes"])] = float(parts["ns_per_load"])
    return result


def _parse_stdout_counts_from_line(line: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for part in line.split():
        if "=" in part:
            key, value = part.split("=", 1)
            tokens[key] = value
    return tokens


def _parse_optional_keyed_metric(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = _parse_stdout_counts_from_line(line)
        if key in parts:
            return float(parts[key])
    return None


def _derive_summary(args: argparse.Namespace, raw_dir: Path) -> dict[str, Any]:
    spec = PROFILE_SPECS[args.profile]
    lscpu_payload = _parse_lscpu(raw_dir / "lscpu.txt")
    cache_payload = _parse_lscpu_cache(raw_dir / "lscpu_cache.txt")
    numactl_payload = _parse_numactl(raw_dir / "numactl_H.txt", args.numa_node)
    ptrchase = _parse_ptrchase(raw_dir)

    issue_cycles = _parse_perf_cycles(raw_dir / "issue_width.perf.csv")
    try:
        cpu_clock_ghz = issue_cycles / _parse_perf_task_clock_seconds(raw_dir / "issue_width.perf.csv") / 1e9
        cpu_clock_source = "perf cycles/task-clock on issue loop"
    except ValueError:
        cpuinfo_max_freq = raw_dir / "cpuinfo_max_freq.txt"
        if cpuinfo_max_freq.exists():
            cpu_clock_ghz = float(cpuinfo_max_freq.read_text(encoding="utf-8").strip()) / 1e6
            cpu_clock_source = "cpuinfo_max_freq"
        elif "CPU max MHz" in lscpu_payload:
            cpu_clock_ghz = float(lscpu_payload["CPU max MHz"]) / 1000.0
            cpu_clock_source = "lscpu CPU max MHz"
        else:
            cpu_clock_ghz = float(spec.get("fallback_cpu_clock_ghz", 0.0))
            if cpu_clock_ghz <= 0.0:
                raise ValueError("failed to derive CPU clock from perf, sysfs, lscpu, or profile fallback")
            cpu_clock_source = str(spec.get("fallback_cpu_clock_source", "profile fallback"))
    issue_counts = _parse_stdout_counts(raw_dir / "issue_width.stdout.txt")
    issue_width = float(issue_counts["adds_per_iter"]) * float(issue_counts["iterations"]) / issue_cycles

    def latency_cycles(stem: str) -> float:
        cycles = _parse_perf_cycles(raw_dir / f"{stem}.perf.csv")
        tokens = _parse_stdout_counts(raw_dir / f"{stem}.stdout.txt")
        ops = float(tokens["ops_per_iter"]) * float(tokens["iterations"])
        return cycles / ops

    throughput_mode = spec["throughput_mode"]
    throughput_prefix = "sve" if throughput_mode == "sve" else "neon"

    def throughput_per_cycle(op: str) -> float:
        stem = f"{throughput_prefix}_throughput_{op}"
        cycles = _parse_perf_cycles(raw_dir / f"{stem}.perf.csv")
        tokens = _parse_stdout_counts(raw_dir / f"{stem}.stdout.txt")
        vector_insts = float(tokens["vector_insts_per_iter"]) * float(tokens["iterations"])
        return vector_insts / cycles

    stream_best = 0.0
    for line in (raw_dir / "stream_triad.txt").read_text(encoding="utf-8").splitlines():
        if "best_gb_s=" in line:
            stream_best = float(_parse_stdout_counts_from_line(line)["best_gb_s"])
            break
    if stream_best <= 0.0:
        raise ValueError("failed to parse stream bandwidth")
    copy_bw_1t = _parse_optional_keyed_metric(raw_dir / "copy_bw_1t.txt", "best_gb_s")
    copy_bw_saturated = _parse_optional_keyed_metric(raw_dir / "copy_bw_saturated.txt", "best_gb_s")
    random_gather_bw_1t = _parse_optional_keyed_metric(raw_dir / "random_gather_bw_1t.txt", "best_gb_s")
    random_gather_bw_saturated = _parse_optional_keyed_metric(raw_dir / "random_gather_bw_saturated.txt", "best_gb_s")
    launch_intra = _parse_optional_keyed_metric(raw_dir / "thread_launch_overhead_intra.txt", "avg_launch_us")
    launch_inter = _parse_optional_keyed_metric(raw_dir / "thread_launch_overhead_inter.txt", "avg_launch_us")

    simd_width_bits = 256.0 if throughput_mode == "sve" else 128.0
    if throughput_mode == "sve" and (raw_dir / "sve_default_vector_length.txt").exists():
        simd_width_bits = float((raw_dir / "sve_default_vector_length.txt").read_text(encoding="utf-8").strip()) * 8.0

    derived = {
        "profile_name": args.profile,
        "observed": {
            "core": {
                "architecture": lscpu_payload["Architecture"],
                "vendor": lscpu_payload["Vendor ID"],
                "total_cores": len(numactl_payload["cpus"]),
                "numa_mem_mb": numactl_payload["numa_mem_mb"],
                "numa_cpu_list": numactl_payload["cpu_list_text"],
            },
            "cache": {
                "l1d": {
                    "size": int(cache_payload["L1d"]["one_size_bytes"]),
                    "assoc": int(cache_payload["L1d"]["ways"]),
                    "cacheline_bytes": int(cache_payload["L1d"].get("coherency_size", 0.0) or 64),
                },
                "l1i": {
                    "size": int(cache_payload["L1i"]["one_size_bytes"]),
                    "assoc": int(cache_payload["L1i"]["ways"]),
                    "cacheline_bytes": int(cache_payload["L1i"].get("coherency_size", 0.0) or 64),
                },
                "l2": {
                    "size": int(cache_payload["L2"]["one_size_bytes"]),
                    "assoc": int(cache_payload["L2"]["ways"]),
                    "cacheline_bytes": int(cache_payload["L2"].get("coherency_size", 0.0) or 64),
                },
                "l3_per_die": {
                    "size": int(cache_payload["L3"]["one_size_bytes"]),
                    "assoc": int(cache_payload["L3"]["ways"]),
                    "cacheline_bytes": int(cache_payload["L3"].get("coherency_size", 0.0) or 64),
                },
            },
        },
        "selected_probes": spec["selected_latency_sizes"],
        "derived": {
            "cpu_clock_ghz": cpu_clock_ghz,
            "cpu_clock_source": cpu_clock_source,
            "cache_response_latency_ns": {
                "l1": ptrchase[spec["selected_latency_sizes"]["l1"]],
                "l2": ptrchase[spec["selected_latency_sizes"]["l2"]],
                "l3": ptrchase[spec["selected_latency_sizes"]["l3"]],
                "dram": ptrchase[spec["selected_latency_sizes"]["dram"]],
            },
            "cache_response_latency_cycles": {
                "l1": ptrchase[spec["selected_latency_sizes"]["l1"]] * cpu_clock_ghz,
                "l2": ptrchase[spec["selected_latency_sizes"]["l2"]] * cpu_clock_ghz,
                "l3": ptrchase[spec["selected_latency_sizes"]["l3"]] * cpu_clock_ghz,
            },
            "memory": {
                "bandwidth_gb_s_total": stream_best,
                "local_mem_delay_ns": ptrchase[spec["selected_latency_sizes"]["dram"]],
            },
            "runtime": {},
            "pipeline": {"issue_width": issue_width},
            "instruction": {
                "simd_width_bits": simd_width_bits,
                "fp_latency_cycles": {
                    "scalar_dp_add": latency_cycles("scalar_fadd64"),
                    "scalar_dp_mul": latency_cycles("scalar_fmul64"),
                    "scalar_dp_fma": latency_cycles("scalar_fma64"),
                    "vector_sp_add": latency_cycles("neon_fadd4s"),
                    "vector_sp_mul": latency_cycles("neon_fmul4s"),
                    "vector_sp_fma": latency_cycles("neon_fma4s"),
                },
                "fp_throughput_per_cycle": {
                    "vector_sp_add": throughput_per_cycle(spec["throughput_ops"]["vector_sp_add"]),
                    "vector_sp_mul": throughput_per_cycle(spec["throughput_ops"]["vector_sp_mul"]),
                    "vector_sp_fma": throughput_per_cycle(spec["throughput_ops"]["vector_sp_fma"]),
                },
            },
        },
    }
    if copy_bw_1t is not None:
        derived["derived"]["memory"]["copy_bw_gb_s_1t"] = copy_bw_1t
    if copy_bw_saturated is not None:
        derived["derived"]["memory"]["copy_bw_gb_s_saturated"] = copy_bw_saturated
    if random_gather_bw_1t is not None:
        derived["derived"]["memory"]["random_gather_bw_gb_s_1t"] = random_gather_bw_1t
    if random_gather_bw_saturated is not None:
        derived["derived"]["memory"]["random_gather_bw_gb_s_saturated"] = random_gather_bw_saturated
    if launch_intra is not None:
        derived["derived"]["runtime"]["parallel_launch_overhead_us_intra"] = launch_intra
    if launch_inter is not None:
        derived["derived"]["runtime"]["parallel_launch_overhead_us_inter"] = launch_inter
    return derived


def _compare(expected_yaml: Path, summary: dict[str, Any]) -> dict[str, Any]:
    expected = _load_yaml(expected_yaml)
    comparisons: list[dict[str, Any]] = []
    ok = True
    for yaml_key, summary_key, abs_tol, rel_tol in COMPARE_SPECS:
        expected_value = _nested_get(expected, yaml_key)
        summary_value = _nested_get(summary, summary_key)
        passed = True
        normalized_expected = expected_value
        if isinstance(expected_value, str):
            try:
                normalized_expected = _parse_quantity(expected_value)
            except Exception:
                normalized_expected = expected_value
        if isinstance(normalized_expected, str):
            passed = normalized_expected == summary_value
        else:
            expected_num = float(normalized_expected)
            summary_num = float(summary_value)
            delta = abs(summary_num - expected_num)
            rel = delta / max(abs(expected_num), 1e-9)
            passed = delta <= abs_tol or rel <= rel_tol
        ok = ok and passed
        comparisons.append(
            {
                "yaml_key": yaml_key,
                "summary_key": summary_key,
                "expected": expected_value,
                "observed": summary_value,
                "passed": passed,
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
            }
        )
    return {"ok": ok, "comparisons": comparisons}


def _write_markdown(summary: dict[str, Any], compare: dict[str, Any] | None, output_path: Path) -> None:
    lines = [
        f"# Reproduced Calibration Summary: {summary['profile_name']}",
        "",
        "## Core Results",
        "",
        f"- single NUMA cores: `{summary['observed']['core']['total_cores']}`",
        f"- CPU clock GHz: `{summary['derived']['cpu_clock_ghz']:.6f}` via `{summary['derived']['cpu_clock_source']}`",
        f"- L1 / L2 / L3 response proxy ns: `{summary['derived']['cache_response_latency_ns']['l1']:.4f}` / `{summary['derived']['cache_response_latency_ns']['l2']:.4f}` / `{summary['derived']['cache_response_latency_ns']['l3']:.4f}`",
        f"- DRAM proxy ns: `{summary['derived']['memory']['local_mem_delay_ns']:.4f}`",
        f"- issue width proxy: `{summary['derived']['pipeline']['issue_width']:.6f}`",
        f"- SIMD width bits: `{summary['derived']['instruction']['simd_width_bits']:.0f}`",
        f"- vector throughput add/mul/fma: `{summary['derived']['instruction']['fp_throughput_per_cycle']['vector_sp_add']:.6f}` / `{summary['derived']['instruction']['fp_throughput_per_cycle']['vector_sp_mul']:.6f}` / `{summary['derived']['instruction']['fp_throughput_per_cycle']['vector_sp_fma']:.6f}`",
        f"- stream bandwidth GB/s: `{summary['derived']['memory']['bandwidth_gb_s_total']:.4f}`",
    ]
    if "copy_bw_gb_s_1t" in summary["derived"]["memory"]:
        lines.append(f"- copy bandwidth GB/s (1t / sat): `{summary['derived']['memory']['copy_bw_gb_s_1t']:.4f}` / `{summary['derived']['memory']['copy_bw_gb_s_saturated']:.4f}`")
    if "random_gather_bw_gb_s_1t" in summary["derived"]["memory"]:
        lines.append(f"- random gather bandwidth GB/s (1t / sat): `{summary['derived']['memory']['random_gather_bw_gb_s_1t']:.4f}` / `{summary['derived']['memory']['random_gather_bw_gb_s_saturated']:.4f}`")
    if "parallel_launch_overhead_us_intra" in summary["derived"]["runtime"]:
        lines.append(f"- thread launch overhead us (intra / inter): `{summary['derived']['runtime']['parallel_launch_overhead_us_intra']:.4f}` / `{summary['derived']['runtime'].get('parallel_launch_overhead_us_inter', 0.0):.4f}`")
    if compare is not None:
        lines.extend(["", "## YAML Check", "", f"- overall: `{'PASS' if compare['ok'] else 'FAIL'}`", "", "| Key | Expected | Observed | Pass |", "| --- | ---: | ---: | --- |"])
        for row in compare["comparisons"]:
            lines.append(f"| `{row['yaml_key']}` | `{row['expected']}` | `{row['observed']}` | `{'yes' if row['passed'] else 'no'}` |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture and summarize host hardware calibration evidence.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_SPECS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-yaml", default=None)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--build-dir", default=str(ROOT_DIR / "build"))
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--ptrchase-target-loads", type=int, default=600000000)
    parser.add_argument("--ptrchase-min-rounds", type=int, default=200)
    args = parser.parse_args()

    raw_dir = Path(args.output_dir).resolve()
    build_dir = Path(args.build_dir).resolve()
    expected_yaml = Path(args.expected_yaml).resolve() if args.expected_yaml else (HW_CONFIG_DIR / f"{args.profile}.yaml").resolve()

    if not args.skip_capture:
        _capture(args, expected_yaml, raw_dir, build_dir)

    summary = _derive_summary(args, raw_dir)
    compare = _compare(expected_yaml, summary) if yaml is not None and expected_yaml.exists() else None

    (raw_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if compare is not None:
        (raw_dir / "yaml_comparison.json").write_text(json.dumps(compare, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(summary, compare, raw_dir / "calibration_summary.md")

    if yaml is None:
        print("warning: PyYAML not available; skipped YAML comparison", file=sys.stderr)

    if compare is not None and not compare["ok"]:
        print(json.dumps(compare, indent=2, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
