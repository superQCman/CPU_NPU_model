#!/usr/bin/env python3
"""
Build a per-operator training dataset by aligning:
  1. op_shapes_*.csv         (canonical node_idx / node_name / op_type)
  2. *_cpu_thread_detail.csv (full-model ORT profiling, repeated over batches)

This variant intentionally does not merge DynamoRIO trace features.

Outputs:
  - aligned CPU-thread detail CSV with canonical node_name from op_shapes
  - aggregated CPU-thread per-node CSV
  - final one-row-per-op feature CSV for training
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CPU_NUMERIC_FIELDS = [
    "dur_us",
    "main_Distribution",
    "main_DistributionEnqueue",
    "main_Run",
    "main_Wait",
    "main_WaitRevoke",
    "num_sub_threads",
    "active_sub_threads",
    "actual_threads_used",
    "total_sub_runs",
    "sub_max_runs",
    "output_size",
    "activation_size",
    "parameter_size",
]


def read_csv(path: Path, *, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        print(f"[WARN] optional CSV not found, continuing without it: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_profile_node_name(node_name: str) -> str:
    suffix = "_kernel_time"
    if node_name.endswith(suffix):
        return node_name[: -len(suffix)]
    return node_name


def parse_bool(value: str) -> int:
    text = str(value).strip().lower()
    return 1 if text in {"1", "true", "yes"} else 0


def parse_float(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def load_op_shape_nodes(op_shapes_path: Path) -> List[Dict[str, str]]:
    nodes: List[Dict[str, str]] = []
    seen = set()
    for row in read_csv(op_shapes_path):
        key = (row["node_idx"], row["node_name"], row["op_type"])
        if key in seen:
            continue
        seen.add(key)
        nodes.append(
            {
                "node_idx": row["node_idx"],
                "node_name": row["node_name"],
                "op_type": row["op_type"],
            }
        )
    nodes.sort(key=lambda row: int(row["node_idx"]))
    return nodes


def build_op_shape_lookup(
    nodes: Iterable[Dict[str, str]],
) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_idx: Dict[str, Dict[str, str]] = {}
    by_name_op: Dict[Tuple[str, str], Dict[str, str]] = {}
    by_name: Dict[str, Dict[str, str]] = {}
    for node in nodes:
        by_idx[node["node_idx"]] = node
        by_name_op[(node["node_name"], node["op_type"])] = node
        by_name[node["node_name"]] = node
    return by_idx, by_name_op, by_name


def align_cpu_detail(
    cpu_rows: List[Dict[str, str]],
    by_idx: Dict[str, Dict[str, str]],
    by_name_op: Dict[Tuple[str, str], Dict[str, str]],
    by_name: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    aligned_rows: List[Dict[str, str]] = []
    unmatched_rows: List[Dict[str, str]] = []

    for row in cpu_rows:
        raw_node_name = row.get("node_name", "")
        normalized_name = normalize_profile_node_name(raw_node_name)
        profile_idx = str(row.get("node_index", "")).strip()
        op_type = row.get("op_name", "").strip()

        match = None
        match_method = ""

        idx_match = by_idx.get(profile_idx)
        if idx_match and (not op_type or idx_match["op_type"] == op_type):
            match = idx_match
            match_method = "node_index"
        elif (normalized_name, op_type) in by_name_op:
            match = by_name_op[(normalized_name, op_type)]
            match_method = "node_name_op_type"
        elif normalized_name in by_name:
            match = by_name[normalized_name]
            match_method = "node_name"
        elif idx_match:
            match = idx_match
            match_method = "node_index_fallback"

        aligned = dict(row)
        aligned["profile_node_name_raw"] = raw_node_name
        aligned["profile_node_name_normalized"] = normalized_name
        aligned["profile_node_index_raw"] = profile_idx

        if match:
            aligned["node_name"] = match["node_name"]
            aligned["node_index"] = match["node_idx"]
            aligned["op_type"] = match["op_type"]
            aligned["match_method"] = match_method
        else:
            aligned["node_name"] = normalized_name
            aligned["node_index"] = profile_idx
            aligned["op_type"] = op_type
            aligned["match_method"] = "unmatched"
            unmatched_rows.append(
                {
                    "profile_node_index_raw": profile_idx,
                    "profile_node_name_raw": raw_node_name,
                    "profile_node_name_normalized": normalized_name,
                    "op_type": op_type,
                }
            )

        aligned_rows.append(aligned)

    return aligned_rows, unmatched_rows


def aggregate_cpu_rows(aligned_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in aligned_rows:
        key = (str(row.get("node_index", "")).strip(), row.get("node_name", ""), row.get("op_type", ""))
        groups[key].append(row)

    aggregated_rows: List[Dict[str, str]] = []
    for (node_idx, node_name, op_type), rows in sorted(groups.items(), key=lambda item: int(item[0][0])):
        out: Dict[str, str] = {
            "node_idx": node_idx,
            "node_name": node_name,
            "op_type": op_type,
            "call_count": str(len(rows)),
            "provider": rows[0].get("provider", ""),
            "cpu_profile_match_methods": "|".join(sorted(set(row.get("match_method", "") for row in rows if row.get("match_method")))),
        }

        main_cores = sorted(
            {
                str(int(core))
                for row in rows
                for core in [parse_float(row.get("main_core", ""))]
                if core is not None and core >= 0
            },
            key=int,
        )
        sub_cores = sorted(
            {
                token
                for row in rows
                for token in row.get("sub_cores", "").split("|")
                if token
            },
            key=int,
        )

        out["main_cores"] = "|".join(main_cores)
        out["sub_cores"] = "|".join(sub_cores)

        main_thread_used_count = sum(parse_bool(row.get("main_thread_used", "")) for row in rows)
        out["main_thread_used_count"] = str(main_thread_used_count)
        out["main_thread_used_pct"] = format_number(100.0 * main_thread_used_count / len(rows))

        for field in CPU_NUMERIC_FIELDS:
            values = [parse_float(row.get(field, "")) for row in rows]
            numeric_values = [value for value in values if value is not None]
            if not numeric_values:
                continue

            out[f"{field}_avg"] = format_number(sum(numeric_values) / len(numeric_values))
            out[f"{field}_min"] = format_number(min(numeric_values))
            out[f"{field}_max"] = format_number(max(numeric_values))
            if field == "dur_us":
                out[f"{field}_sum"] = format_number(sum(numeric_values))

        aggregated_rows.append(out)

    return aggregated_rows


def build_cpu_lookup(cpu_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {str(row.get("node_idx", "")).strip(): row for row in cpu_rows if str(row.get("node_idx", "")).strip()}


def build_final_rows(
    nodes: List[Dict[str, str]],
    cpu_lookup: Dict[str, Dict[str, str]],
    batch_size: int | None,
    num_indices: int | None,
    arch_embedding_size: str,
    arch_mlp_bot: str,
    arch_mlp_top: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for node in nodes:
        node_idx = node["node_idx"]
        cpu_row = cpu_lookup.get(node_idx, {})

        row: Dict[str, str] = {
            "batch_size": "" if batch_size is None else str(batch_size),
            "num_indices_per_lookup": "" if num_indices is None else str(num_indices),
            "arch_embedding_size": arch_embedding_size,
            "arch_mlp_bot": arch_mlp_bot,
            "arch_mlp_top": arch_mlp_top,
            "node_idx": node_idx,
            "node_name": node["node_name"],
            "op_type": node["op_type"],
            "has_cpu_profile": "1" if cpu_row else "0",
            "cpu_profile_missing_reason": "",
        }

        if not cpu_row:
            if node["op_type"] == "Constant":
                row["cpu_profile_missing_reason"] = "constant_or_no_thread_stats"
            else:
                row["cpu_profile_missing_reason"] = "no_cpu_thread_stats"

        if cpu_row:
            for key, value in cpu_row.items():
                if key in {"node_idx", "node_name", "op_type"}:
                    continue
                row[f"cpu_{key}"] = value

        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-op training feature CSV for one sweep combo without trace features.")
    parser.add_argument("--op-shapes", required=True, help="Path to op_shapes_{batch}_{nip}.csv")
    parser.add_argument("--cpu-detail", required=True, help="Path to *_cpu_thread_detail.csv")
    parser.add_argument("--out", required=True, help="Path to the final merged feature CSV")
    parser.add_argument("--aligned-cpu-detail-out", required=True, help="Path to write aligned CPU detail CSV")
    parser.add_argument("--cpu-agg-out", required=True, help="Path to write per-node aggregated CPU CSV")
    parser.add_argument("--unmatched-out", required=True, help="Path to write unmatched CPU rows CSV")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size metadata")
    parser.add_argument("--num-indices-per-lookup", type=int, default=None, help="Optional num-indices metadata")
    parser.add_argument("--arch-embedding-size", default="", help="Optional DLRM arch_embedding_size metadata")
    parser.add_argument("--arch-mlp-bot", default="", help="Optional DLRM arch_mlp_bot metadata")
    parser.add_argument("--arch-mlp-top", default="", help="Optional DLRM arch_mlp_top metadata")
    args = parser.parse_args()

    op_shapes_path = Path(args.op_shapes)
    cpu_detail_path = Path(args.cpu_detail)

    nodes = load_op_shape_nodes(op_shapes_path)
    by_idx, by_name_op, by_name = build_op_shape_lookup(nodes)

    cpu_rows = read_csv(cpu_detail_path, required=False)
    aligned_rows, unmatched_rows = align_cpu_detail(cpu_rows, by_idx, by_name_op, by_name)
    cpu_agg_rows = aggregate_cpu_rows(aligned_rows)

    cpu_lookup = build_cpu_lookup(cpu_agg_rows)
    final_rows = build_final_rows(
        nodes,
        cpu_lookup,
        batch_size=args.batch_size,
        num_indices=args.num_indices_per_lookup,
        arch_embedding_size=args.arch_embedding_size,
        arch_mlp_bot=args.arch_mlp_bot,
        arch_mlp_top=args.arch_mlp_top,
    )

    write_csv(aligned_rows, Path(args.aligned_cpu_detail_out))
    write_csv(cpu_agg_rows, Path(args.cpu_agg_out))
    write_csv(unmatched_rows, Path(args.unmatched_out))
    write_csv(final_rows, Path(args.out))

    matched = len(aligned_rows) - len(unmatched_rows)
    print(f"Loaded op_shapes nodes : {len(nodes)}")
    print(f"Loaded cpu detail rows : {len(cpu_rows)}")
    print(f"Aligned cpu detail rows: {matched} matched, {len(unmatched_rows)} unmatched")
    print(f"Aggregated cpu rows    : {len(cpu_agg_rows)}")
    print(f"Final feature rows     : {len(final_rows)}")
    print(f"Saved aligned detail   : {args.aligned_cpu_detail_out}")
    print(f"Saved cpu aggregate    : {args.cpu_agg_out}")
    print(f"Saved unmatched rows   : {args.unmatched_out}")
    print(f"Saved final dataset    : {args.out}")


if __name__ == "__main__":
    main()
