#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import reproduce_hw_profile as base


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
HW_CONFIG_DIR = ROOT_DIR.parent / "calibrator" / "hw_config"


METRIC_SPECS: list[tuple[str, tuple[str, ...]]] = [
    ("cpu_clock_ghz", ("derived", "cpu_clock_ghz")),
    ("l1_ns", ("derived", "cache_response_latency_ns", "l1")),
    ("l2_ns", ("derived", "cache_response_latency_ns", "l2")),
    ("l3_ns", ("derived", "cache_response_latency_ns", "l3")),
    ("dram_ns", ("derived", "memory", "local_mem_delay_ns")),
    ("bandwidth_gb_s_total", ("derived", "memory", "bandwidth_gb_s_total")),
    ("issue_width", ("derived", "pipeline", "issue_width")),
    ("scalar_dp_add_cycles", ("derived", "instruction", "fp_latency_cycles", "scalar_dp_add")),
    ("scalar_dp_mul_cycles", ("derived", "instruction", "fp_latency_cycles", "scalar_dp_mul")),
    ("scalar_dp_fma_cycles", ("derived", "instruction", "fp_latency_cycles", "scalar_dp_fma")),
    ("vector_sp_add_cycles", ("derived", "instruction", "fp_latency_cycles", "vector_sp_add")),
    ("vector_sp_mul_cycles", ("derived", "instruction", "fp_latency_cycles", "vector_sp_mul")),
    ("vector_sp_fma_cycles", ("derived", "instruction", "fp_latency_cycles", "vector_sp_fma")),
    ("vector_sp_add_tp", ("derived", "instruction", "fp_throughput_per_cycle", "vector_sp_add")),
    ("vector_sp_mul_tp", ("derived", "instruction", "fp_throughput_per_cycle", "vector_sp_mul")),
    ("vector_sp_fma_tp", ("derived", "instruction", "fp_throughput_per_cycle", "vector_sp_fma")),
]


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for part in path:
        value = value[part]
    return value


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def _collect_run_summaries(args: argparse.Namespace, expected_yaml: Path) -> list[tuple[Path, dict[str, Any]]]:
    runs_root = Path(args.output_dir).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing_root:
        source_root = Path(args.reuse_existing_root).resolve()
        run_dirs = sorted(path for path in source_root.glob("run_*") if path.is_dir())
    else:
        run_dirs = []
        for run_idx in range(1, args.repeats + 1):
            run_dir = runs_root / f"run_{run_idx}"
            if run_dir.exists() and not args.keep_existing_runs:
                shutil.rmtree(run_dir)
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "reproduce_hw_profile.py"),
                "--profile",
                args.profile,
                "--output-dir",
                str(run_dir),
                "--numa-node",
                str(args.numa_node),
                "--build-dir",
                str(args.build_dir),
            ]
            if args.cpu is not None:
                cmd.extend(["--cpu", str(args.cpu)])
            if args.threads is not None:
                cmd.extend(["--threads", str(args.threads)])
            cmd.extend(
                [
                    "--ptrchase-target-loads",
                    str(args.ptrchase_target_loads),
                    "--ptrchase-min-rounds",
                    str(args.ptrchase_min_rounds),
                    "--expected-yaml",
                    str(Path("/tmp/nonexistent_expected_hw_profile.yaml")),
                ]
            )
            subprocess.run(cmd, check=True)
            run_dirs.append(run_dir)

    derive_args = SimpleNamespace(profile=args.profile, numa_node=args.numa_node)
    summaries = [(run_dir, base._derive_summary(derive_args, run_dir)) for run_dir in run_dirs]
    if not summaries:
        raise RuntimeError("no run_* directories were found for convergence aggregation")
    return summaries


def _build_converged_summary(run_summaries: list[tuple[Path, dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    first_summary = deepcopy(run_summaries[0][1])
    stats: dict[str, Any] = {"metrics": {}, "runs": []}

    for run_dir, summary in run_summaries:
        stats["runs"].append(
            {
                "run_dir": str(run_dir),
                "cpu_clock_ghz": summary["derived"]["cpu_clock_ghz"],
                "l1_ns": summary["derived"]["cache_response_latency_ns"]["l1"],
                "l2_ns": summary["derived"]["cache_response_latency_ns"]["l2"],
                "l3_ns": summary["derived"]["cache_response_latency_ns"]["l3"],
                "dram_ns": summary["derived"]["memory"]["local_mem_delay_ns"],
                "bandwidth_gb_s_total": summary["derived"]["memory"]["bandwidth_gb_s_total"],
                "issue_width": summary["derived"]["pipeline"]["issue_width"],
            }
        )

    median_summary = deepcopy(first_summary)
    median_summary["derived"]["cpu_clock_source"] = f"{len(run_summaries)}-run median from repeat_reproduce_hw_profile.py"

    for metric_name, path in METRIC_SPECS:
        values = [_get_nested(summary, path) for _, summary in run_summaries]
        median_value = statistics.median(values)
        mean_value = statistics.mean(values)
        std_value = statistics.pstdev(values)
        cv_percent = 100.0 * std_value / mean_value if mean_value else 0.0
        stats["metrics"][metric_name] = {
            "median": median_value,
            "mean": mean_value,
            "min": min(values),
            "max": max(values),
            "std": std_value,
            "cv_percent": cv_percent,
            "stable_cv_le_5pct": cv_percent <= 5.0,
            "stable_cv_le_2pct": cv_percent <= 2.0,
        }
        _set_nested(median_summary, path, median_value)

    clock = median_summary["derived"]["cpu_clock_ghz"]
    median_summary["derived"]["cache_response_latency_cycles"]["l1"] = median_summary["derived"]["cache_response_latency_ns"]["l1"] * clock
    median_summary["derived"]["cache_response_latency_cycles"]["l2"] = median_summary["derived"]["cache_response_latency_ns"]["l2"] * clock
    median_summary["derived"]["cache_response_latency_cycles"]["l3"] = median_summary["derived"]["cache_response_latency_ns"]["l3"] * clock

    stats["summary"] = {
        "repeats": len(run_summaries),
        "stable_metric_count_cv_le_2pct": sum(1 for item in stats["metrics"].values() if item["stable_cv_le_2pct"]),
        "stable_metric_count_cv_le_5pct": sum(1 for item in stats["metrics"].values() if item["stable_cv_le_5pct"]),
    }
    return median_summary, stats


def _write_markdown(converged_summary: dict[str, Any], stats: dict[str, Any], compare: dict[str, Any] | None, output_path: Path) -> None:
    lines = [
        f"# Convergence Summary: {converged_summary['profile_name']}",
        "",
        f"- repeats: `{stats['summary']['repeats']}`",
        f"- stable metrics with `cv <= 2%`: `{stats['summary']['stable_metric_count_cv_le_2pct']}`",
        f"- stable metrics with `cv <= 5%`: `{stats['summary']['stable_metric_count_cv_le_5pct']}`",
        "",
        "## Aggregated Metrics",
        "",
        "| Metric | Median | Mean | Min | Max | CV % | Stable <= 5% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric_name, metric_stats in stats["metrics"].items():
        lines.append(
            f"| `{metric_name}` | `{metric_stats['median']:.6f}` | `{metric_stats['mean']:.6f}` | `{metric_stats['min']:.6f}` | `{metric_stats['max']:.6f}` | `{metric_stats['cv_percent']:.2f}` | `{'yes' if metric_stats['stable_cv_le_5pct'] else 'no'}` |"
        )

    lines.extend(
        [
            "",
            "## Per-Run Core Metrics",
            "",
            "| Run | CPU GHz | L1 ns | L2 ns | L3 ns | DRAM ns | BW GB/s | Issue |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["runs"]:
        lines.append(
            f"| `{Path(row['run_dir']).name}` | `{row['cpu_clock_ghz']:.6f}` | `{row['l1_ns']:.4f}` | `{row['l2_ns']:.4f}` | `{row['l3_ns']:.4f}` | `{row['dram_ns']:.4f}` | `{row['bandwidth_gb_s_total']:.4f}` | `{row['issue_width']:.6f}` |"
        )

    if compare is not None:
        lines.extend(["", "## YAML Check", "", f"- overall: `{'PASS' if compare['ok'] else 'FAIL'}`"])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or aggregate repeated host hardware calibration captures.")
    parser.add_argument("--profile", required=True, choices=sorted(base.PROFILE_SPECS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-yaml", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reuse-existing-root", default=None)
    parser.add_argument("--keep-existing-runs", action="store_true")
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--build-dir", default=str(ROOT_DIR / "build"))
    parser.add_argument("--ptrchase-target-loads", type=int, default=600000000)
    parser.add_argument("--ptrchase-min-rounds", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_yaml = Path(args.expected_yaml).resolve() if args.expected_yaml else (HW_CONFIG_DIR / f"{args.profile}.yaml").resolve()

    run_summaries = _collect_run_summaries(args, expected_yaml)
    converged_summary, stats = _build_converged_summary(run_summaries)
    compare = base._compare(expected_yaml, converged_summary) if base.yaml is not None and expected_yaml.exists() else None

    (output_dir / "convergence_summary.json").write_text(json.dumps(converged_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "convergence_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if compare is not None:
        (output_dir / "yaml_comparison.json").write_text(json.dumps(compare, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(converged_summary, stats, compare, output_dir / "convergence_summary.md")

    if base.yaml is None:
        print("warning: PyYAML not available; skipped YAML comparison", file=sys.stderr)

    if compare is not None and not compare["ok"]:
        print(json.dumps(compare, indent=2, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
