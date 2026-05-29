from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _parse_lscpu_key_value(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _parse_cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for piece in str(text).split(","):
        part = piece.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(left)
                end = int(right)
            except ValueError:
                continue
            cpus.extend(range(start, end + 1))
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                continue
    return sorted(dict.fromkeys(cpus))


def _read_sysfs_numa_nodes() -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    root = Path("/sys/devices/system/node")
    for node_dir in sorted(root.glob("node[0-9]*")):
        node = node_dir.name.removeprefix("node")
        cpulist_path = node_dir / "cpulist"
        meminfo_path = node_dir / "meminfo"
        cpulist = cpulist_path.read_text(encoding="utf-8").strip() if cpulist_path.exists() else ""
        mem_mb = None
        if meminfo_path.exists():
            match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo_path.read_text(encoding="utf-8", errors="ignore"))
            if match:
                mem_mb = int(match.group(1)) // 1024
        nodes[node] = {
            "node": int(node),
            "cpu_list": cpulist,
            "cpu_count": len(_parse_cpu_list(cpulist)),
            "mem_mb": mem_mb,
        }
    return nodes


def detect_machine_topology() -> dict[str, Any]:
    lscpu_text = _run_text(["lscpu"])
    lscpu = _parse_lscpu_key_value(lscpu_text)
    numa_nodes = _read_sysfs_numa_nodes()
    if not numa_nodes:
        count = int(lscpu.get("NUMA node(s)", "1") or 1)
        for idx in range(max(count, 1)):
            cpu_list = lscpu.get(f"NUMA node{idx} CPU(s)", "")
            numa_nodes[str(idx)] = {
                "node": idx,
                "cpu_list": cpu_list,
                "cpu_count": len(_parse_cpu_list(cpu_list)),
                "mem_mb": None,
            }
    available_nodes = sorted(numa_nodes, key=lambda value: int(value))
    online_cpus = os.cpu_count() or 1
    return {
        "project_root": str(PROJECT_ROOT),
        "architecture": lscpu.get("Architecture", ""),
        "vendor_id": lscpu.get("Vendor ID", ""),
        "model_name": lscpu.get("Model name", ""),
        "cpu_max_mhz": lscpu.get("CPU max MHz", ""),
        "cpu_min_mhz": lscpu.get("CPU min MHz", ""),
        "socket_count": _int_or_none(lscpu.get("Socket(s)")),
        "cores_per_socket": _int_or_none(lscpu.get("Core(s) per socket")),
        "threads_per_core": _int_or_none(lscpu.get("Thread(s) per core")),
        "logical_cpu_count": _int_or_none(lscpu.get("CPU(s)")) or online_cpus,
        "online_cpu_count": online_cpus,
        "numa_node_count": len(available_nodes),
        "available_numa_nodes": [int(value) for value in available_nodes],
        "numa_nodes": [numa_nodes[value] for value in available_nodes],
        "l1d_cache": lscpu.get("L1d cache", ""),
        "l1i_cache": lscpu.get("L1i cache", ""),
        "l2_cache": lscpu.get("L2 cache", ""),
        "l3_cache": lscpu.get("L3 cache", ""),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def available_numa_nodes_csv() -> str:
    topology = detect_machine_topology()
    nodes = topology.get("available_numa_nodes") or [0]
    return ",".join(str(int(node)) for node in nodes)


def write_machine_config(path: Path) -> dict[str, Any]:
    topology = detect_machine_topology()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(topology, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return topology


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect local CPU/NUMA topology for portable E2E data generation.")
    parser.add_argument("--available-numas", action="store_true", help="Print only the comma-separated NUMA list.")
    parser.add_argument("--write-json", type=Path, help="Write the full detected topology JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.write_json:
        topology = write_machine_config(args.write_json)
    else:
        topology = detect_machine_topology()
    if args.available_numas:
        print(",".join(str(int(node)) for node in topology.get("available_numa_nodes", [0])))
    else:
        print(json.dumps(topology, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
