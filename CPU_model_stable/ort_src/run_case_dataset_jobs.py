#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from calibrator.machine_topology import available_numa_nodes_csv, write_machine_config
except ImportError:
    available_numa_nodes_csv = None
    write_machine_config = None

DEFAULT_DISCOVER_GLOB = "case_*_run_*_*.sh"
DEFAULT_AVAILABLE_NUMAS = "auto"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "data" / "case_dataset_scheduler_logs"
SWEEP_SCRIPT = SCRIPT_DIR / "run_ort_sweep_extensible_no_trace.sh"

SCRIPT_ENV_PATTERN = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)\\$")
CASE_VARIANT_PATTERN = re.compile(r"^(case_\d+)(?:_|$)")

CSV_TO_ENV = {
    "onnx_variant": "ONNX_VARIANT",
    "intra_threads": "INTRA_THREADS",
    "inter_threads": "INTER_THREADS",
    "numa_node": "NUMA_NODE",
    "batch_start": "BATCH_START",
    "batch_end": "BATCH_END",
    "batch_step": "BATCH_STEP",
    "num_indices_start": "NUM_INDICES_START",
    "num_indices_end": "NUM_INDICES_END",
    "num_indices_step": "NUM_INDICES_STEP",
    "num_batches": "NUM_BATCHES",
    "warmup_batches": "WARMUP_BATCHES",
    "runner_mode": "RUNNER_MODE",
    "use_cann": "USE_CANN",
    "resume": "RESUME",
    "preserve_generated_onnx": "PRESERVE_GENERATED_ONNX",
    "parallel_branches": "PARALLEL_BRANCHES",
    "tail_intra_threads": "TAIL_INTRA_THREADS",
    "verify_full_output": "VERIFY_FULL_OUTPUT",
    "force_cpu_ops": "FORCE_CPU_OPS",
    "stop_after_infer": "STOP_AFTER_INFER",
    "profile_warmup": "PROFILE_WARMUP",
    "disable_graph_optimizations": "DISABLE_GRAPH_OPTIMIZATIONS",
    "use_numactl": "USE_NUMACTL",
    "generated_onnx_tmp_root": "GENERATED_ONNX_TMP_ROOT",
    "out_root": "OUT_ROOT",
    "feature_dataset_root": "FEATURE_DATASET_ROOT",
    "feature_subset_root": "FEATURE_SUBSET_ROOT",
}


@dataclass
class Job:
    name: str
    source: str
    env: Dict[str, str] = field(default_factory=dict)
    numa_node: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ORT dataset extraction scheduler with one-job-per-NUMA execution.")
    parser.add_argument("--discover-glob", default=DEFAULT_DISCOVER_GLOB, help="Glob for existing case scripts to discover.")
    parser.add_argument("--no-discover-case-scripts", action="store_true", help="Disable auto-discovery of existing case scripts.")
    parser.add_argument("--jobs-file", action="append", default=[], help="CSV file with extra jobs to schedule.")
    parser.add_argument("--jobs", default="", help="Comma-separated job names to include.")
    parser.add_argument("--exclude-jobs", default="", help="Comma-separated job names to skip.")
    parser.add_argument(
        "--available-numas",
        default=DEFAULT_AVAILABLE_NUMAS,
        help="Comma-separated NUMA nodes usable for scheduling, or 'auto' to detect local NUMA nodes.",
    )
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT), help="Root directory for scheduler logs and summary.")
    parser.add_argument("--machine-config-json", default="", help="Optional path to write detected machine topology JSON.")
    parser.add_argument("--skip-generated-onnx", action="store_true", help="Do not retain generated_onnx outputs; use a temporary working directory and clean it after each combo.")
    parser.add_argument("--generated-onnx-tmp-root", default="/tmp/ort_generated_onnx", help="Temporary root used when --skip-generated-onnx is enabled.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep scheduling remaining jobs after a failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned jobs without running them.")
    return parser.parse_args()


def _clean_value(raw: str) -> str:
    value = raw.strip()
    if value.endswith("\\"):
        value = value[:-1].rstrip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _derive_variant_from_name(job_name: str) -> str:
    match = CASE_VARIANT_PATTERN.match(job_name)
    return match.group(1) if match else ""


def _derive_roots(job_name: str) -> Dict[str, str]:
    artifact_root = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data"))).resolve()
    return {
        "OUT_ROOT": str(artifact_root / f"sweep_runs_extensible_{job_name}"),
        "FEATURE_DATASET_ROOT": str(artifact_root / f"features_extensible_{job_name}"),
        "FEATURE_SUBSET_ROOT": str(artifact_root / f"features_extensible_{job_name}_selected"),
    }


def _is_enabled(text: str) -> bool:
    return str(text).strip().lower() not in {"", "0", "false", "no", "off"}


def _load_jobs_from_case_scripts(pattern: str) -> List[Job]:
    jobs: List[Job] = []
    for path in sorted(SCRIPT_DIR.glob(pattern)):
        if not path.is_file():
            continue
        env: Dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            match = SCRIPT_ENV_PATTERN.match(line)
            if not match:
                continue
            env[match.group(1)] = _clean_value(match.group(2))
        if not env:
            continue
        job_name = path.stem
        jobs.append(Job(name=job_name, source=str(path), env=env, numa_node=env.get("NUMA_NODE", "")))
    return jobs


def _load_jobs_from_csv(csv_path: Path) -> List[Job]:
    jobs: List[Job] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            if "enabled" in row and not _is_enabled(row.get("enabled", "1")):
                continue
            job_name = (row.get("job_name") or "").strip()
            if not job_name:
                raise ValueError(f"Missing job_name in {csv_path}")
            env = _derive_roots(job_name)
            variant = (row.get("onnx_variant") or "").strip() or _derive_variant_from_name(job_name)
            if variant:
                env["ONNX_VARIANT"] = variant
            for column, env_name in CSV_TO_ENV.items():
                raw = (row.get(column) or "").strip()
                if raw:
                    env[env_name] = raw
            extra_env = (row.get("extra_env") or "").strip()
            if extra_env:
                for chunk in extra_env.split(";"):
                    piece = chunk.strip()
                    if not piece:
                        continue
                    if "=" not in piece:
                        raise ValueError(f"Invalid extra_env entry in {csv_path}: {piece}")
                    key, value = piece.split("=", 1)
                    env[key.strip()] = value.strip()
            jobs.append(Job(name=job_name, source=str(csv_path), env=env, numa_node=env.get("NUMA_NODE", "")))
    return jobs


def _filter_jobs(jobs: Iterable[Job], include: set[str], exclude: set[str]) -> List[Job]:
    filtered: List[Job] = []
    for job in jobs:
        if include and job.name not in include:
            continue
        if job.name in exclude:
            continue
        filtered.append(job)
    return filtered


def _remap_unavailable_numa(requested_numa: str, available_numas: List[str]) -> str:
    try:
        requested_index = int(requested_numa)
    except ValueError as exc:
        raise ValueError(f"Requested NUMA node is not numeric: {requested_numa}") from exc
    ordered_numas = sorted(available_numas, key=int)
    return ordered_numas[requested_index % len(ordered_numas)]


def _assign_auto_numas(jobs: List[Job], available_numas: List[str]) -> List[tuple[str, str, str]]:
    if not available_numas:
        raise ValueError("No available NUMA nodes configured.")
    loads = {numa: 0 for numa in available_numas}
    remapped: List[tuple[str, str, str]] = []
    for job in jobs:
        if job.numa_node:
            if job.numa_node not in loads:
                original_numa = job.numa_node
                job.numa_node = _remap_unavailable_numa(original_numa, available_numas)
                job.env["NUMA_NODE"] = job.numa_node
                remapped.append((job.name, original_numa, job.numa_node))
            loads[job.numa_node] += 1
    for job in jobs:
        if job.numa_node:
            continue
        chosen = min(available_numas, key=lambda value: (loads[value], int(value)))
        job.numa_node = chosen
        job.env["NUMA_NODE"] = chosen
        loads[chosen] += 1
    return remapped


def _job_env(job: Job, args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(job.env)
    env.setdefault("RUNNER_MODE", "branch_parallel")
    env.setdefault("RESUME", "1")
    env.setdefault("PRESERVE_GENERATED_ONNX", "1")
    if args.skip_generated_onnx:
        env["PRESERVE_GENERATED_ONNX"] = "0"
        env["GENERATED_ONNX_TMP_ROOT"] = args.generated_onnx_tmp_root
    env["NUMA_NODE"] = job.numa_node
    env.setdefault("ONNX_VARIANT", _derive_variant_from_name(job.name))
    roots = _derive_roots(job.name)
    for key, value in roots.items():
        env.setdefault(key, value)
    return env


def _build_summary_path(log_root: Path) -> Path:
    return log_root / "scheduler_summary.csv"


def _run_job(job: Job, args: argparse.Namespace, log_root: Path) -> dict[str, str]:
    env = _job_env(job, args)
    job_log = log_root / f"{job.name}.log"
    start_ts = datetime.now()
    status = "SKIPPED"
    return_code = 0
    with job_log.open("w", encoding="utf-8") as handle:
        handle.write(f"[SCHED] job={job.name}\n")
        handle.write(f"[SCHED] source={job.source}\n")
        handle.write(f"[SCHED] numa_node={job.numa_node}\n")
        for key in sorted(env):
            if key in {"PWD", "OLDPWD", "SHLVL", "_"}:
                continue
            if key not in job.env and key not in {
                "NUMA_NODE",
                "ONNX_VARIANT",
                "OUT_ROOT",
                "FEATURE_DATASET_ROOT",
                "FEATURE_SUBSET_ROOT",
                "PRESERVE_GENERATED_ONNX",
                "GENERATED_ONNX_TMP_ROOT",
                "RUNNER_MODE",
                "RESUME",
            }:
                continue
            handle.write(f"[ENV] {key}={env[key]}\n")
        handle.write("[SCHED] command=bash run_ort_sweep_extensible_no_trace.sh\n")
        handle.flush()
        process = subprocess.Popen(
            ["bash", str(SWEEP_SCRIPT)],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()
        status = "OK" if return_code == 0 else "FAILED"
    end_ts = datetime.now()
    return {
        "job_name": job.name,
        "source": job.source,
        "numa_node": job.numa_node,
        "status": status,
        "return_code": str(return_code),
        "start_time": start_ts.isoformat(timespec="seconds"),
        "end_time": end_ts.isoformat(timespec="seconds"),
        "duration_sec": f"{(end_ts - start_ts).total_seconds():.1f}",
        "log_path": str(job_log),
        "out_root": env.get("OUT_ROOT", ""),
        "feature_dataset_root": env.get("FEATURE_DATASET_ROOT", ""),
        "preserve_generated_onnx": env.get("PRESERVE_GENERATED_ONNX", ""),
    }


def _write_summary(path: Path, rows: List[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_name",
        "source",
        "numa_node",
        "status",
        "return_code",
        "start_time",
        "end_time",
        "duration_sec",
        "log_path",
        "out_root",
        "feature_dataset_root",
        "preserve_generated_onnx",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = _parse_args()
    if not SWEEP_SCRIPT.is_file():
        raise FileNotFoundError(f"Sweep script not found: {SWEEP_SCRIPT}")

    available_numas_text = str(args.available_numas).strip()
    if available_numas_text.lower() == "auto":
        available_numas_text = available_numa_nodes_csv() if available_numa_nodes_csv is not None else "0"
    available_numas = [value.strip() for value in available_numas_text.split(",") if value.strip()]
    if args.machine_config_json and write_machine_config is not None:
        write_machine_config(Path(args.machine_config_json).resolve())
    include = {value.strip() for value in str(args.jobs).split(",") if value.strip()}
    exclude = {value.strip() for value in str(args.exclude_jobs).split(",") if value.strip()}

    jobs: List[Job] = []
    if not args.no_discover_case_scripts:
        jobs.extend(_load_jobs_from_case_scripts(args.discover_glob))
    for raw in args.jobs_file:
        jobs.extend(_load_jobs_from_csv(Path(raw).resolve()))

    if not jobs:
        raise ValueError("No jobs were discovered. Provide --jobs-file or enable case-script discovery.")

    deduped: Dict[str, Job] = {}
    for job in jobs:
        deduped[job.name] = job
    selected = _filter_jobs(deduped.values(), include=include, exclude=exclude)
    if not selected:
        raise ValueError("No jobs matched the requested filters.")

    numa_remaps = _assign_auto_numas(selected, available_numas)
    selected = sorted(selected, key=lambda job: (int(job.numa_node), job.name))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_root).resolve() / timestamp
    log_root.mkdir(parents=True, exist_ok=True)

    print("[PLAN] unified case dataset scheduler", flush=True)
    if numa_remaps:
        remap_counts = Counter((old, new) for _, old, new in numa_remaps)
        remap_items = sorted(remap_counts.items(), key=lambda item: (int(item[0][0]), int(item[0][1])))
        remap_text = ", ".join(
            f"{old}->{new} x{count}" for (old, new), count in remap_items
        )
        print(f"[PLAN] remapped unavailable NUMA requests: {remap_text}", flush=True)
    for job in selected:
        env = _job_env(job, args)
        print(
            f"  - job={job.name} source={job.source} numa={job.numa_node} "
            f"variant={env.get('ONNX_VARIANT','')} intra={env.get('INTRA_THREADS','')} "
            f"inter={env.get('INTER_THREADS','')} out={env.get('OUT_ROOT','')}",
            flush=True,
        )
    print(f"[PLAN] logs={log_root}", flush=True)

    if args.dry_run:
        return 0

    by_numa: Dict[str, List[Job]] = {}
    for job in selected:
        by_numa.setdefault(job.numa_node, []).append(job)

    results: List[dict[str, str]] = []
    results_lock = threading.Lock()
    stop_event = threading.Event()
    failure_seen = threading.Event()

    def worker(numa_node: str, jobs_for_numa: List[Job]) -> None:
        for job in jobs_for_numa:
            if stop_event.is_set():
                break
            print(f"[RUN ] numa={numa_node} job={job.name}", flush=True)
            result = _run_job(job, args, log_root)
            with results_lock:
                results.append(result)
            print(
                f"[DONE] numa={numa_node} job={job.name} status={result['status']} "
                f"duration_sec={result['duration_sec']} log={result['log_path']}",
                flush=True,
            )
            if result["status"] != "OK":
                failure_seen.set()
                if not args.continue_on_error:
                    stop_event.set()
                    break

    threads: List[threading.Thread] = []
    for numa_node in sorted(by_numa, key=int):
        thread = threading.Thread(target=worker, args=(numa_node, by_numa[numa_node]), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    results_sorted = sorted(results, key=lambda row: (int(row["numa_node"]), row["start_time"], row["job_name"]))
    summary_path = _build_summary_path(log_root)
    _write_summary(summary_path, results_sorted)
    print(f"[SUMMARY] {summary_path}", flush=True)
    return 1 if failure_seen.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
