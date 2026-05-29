#!/usr/bin/env python3
"""
Select a compact no-trace feature subset from merged ORT training feature CSVs.

This variant intentionally excludes all DynamoRIO-trace-derived columns.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


FIXED_COLUMNS: List[Tuple[str, str]] = [
    ("batch_size", "batch_size"),
    ("num_indices_per_lookup", "num_indices_per_lookup"),
    ("arch_embedding_size", "arch_embedding_size"),
    ("arch_mlp_bot", "arch_mlp_bot"),
    ("arch_mlp_top", "arch_mlp_top"),
    ("node_name", "node_name"),
    ("op_type", "op_type"),
    ("input_type_shape", "input_type_shape"),
    ("output_type_shape", "output_type_shape"),
    ("num_threads", "num_threads"),
    ("output_size", "cpu_output_size_avg"),
    ("activation_size", "cpu_activation_size_avg"),
    ("parameter_size", "cpu_parameter_size_avg"),
]

DUR_SOURCE_BY_MODE = {
    "avg": "cpu_dur_us_avg",
    "min": "cpu_dur_us_min",
    "max": "cpu_dur_us_max",
    "sum": "cpu_dur_us_sum",
}

SHAPE_FIELDS = ("input_type_shape", "output_type_shape")
SIZE_FIELDS = ("output_size", "activation_size", "parameter_size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a compact no-trace feature subset from merged ORT feature CSVs."
    )
    parser.add_argument("--input", required=True, help="Input feature CSV file or directory containing bs*_nip*.csv files.")
    parser.add_argument("--output", required=True, help="Output CSV file or output directory.")
    parser.add_argument("--pattern", default="bs*_nip*.csv", help="Glob used when --input is a directory. Default: %(default)s")
    parser.add_argument(
        "--dur-source",
        choices=sorted(DUR_SOURCE_BY_MODE),
        default="avg",
        help="Which aggregated CPU duration column to expose as dur_us. Default: %(default)s",
    )
    return parser.parse_args()


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


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


def build_column_plan(fieldnames: Sequence[str], dur_source: str) -> List[Tuple[str, str]]:
    plan: List[Tuple[str, str]] = list(FIXED_COLUMNS)
    plan.append(("dur_us", DUR_SOURCE_BY_MODE[dur_source]))
    return [(out_name, source_name) for out_name, source_name in plan if source_name in fieldnames or out_name in SHAPE_FIELDS or out_name in SIZE_FIELDS]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    ordered: List[Path] = []
    seen = set()
    for path in paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(path)
    return ordered


def choose_profile_roots(input_path: Path) -> List[Path]:
    ort_root = get_repo_root() / "ORT"
    env_candidates = []
    for env_name in ("SELECT_FEATURE_SUBSET_PROFILE_ROOT", "PROFILE_ROOT"):
        value = os.environ.get(env_name, "").strip()
        if value:
            env_candidates.append(Path(value))

    dataset_part = next(
        (
            part
            for part in input_path.parts
            if part.startswith("features_extensible") or part.startswith("features")
        ),
        "",
    )

    candidates: List[Path] = []
    if dataset_part.startswith("features_extensible"):
        suffix = dataset_part[len("features_extensible"):]
        if suffix:
            candidates.append(ort_root / f"sweep_runs_extensible{suffix}" / "onnx_profiles")
        candidates.append(ort_root / "sweep_runs_extensible" / "onnx_profiles")
    else:
        suffix = dataset_part[len("features"):] if dataset_part.startswith("features") else ""
        if suffix:
            candidates.append(ort_root / f"sweep_runs{suffix}" / "onnx_profiles")
        candidates.append(ort_root / "sweep_runs" / "onnx_profiles")

    return dedupe_paths([*env_candidates, *candidates])


def find_aux_detail_csv(input_path: Path) -> Path | None:
    combo = input_path.stem
    for root in choose_profile_roots(input_path):
        combo_dir = root / combo
        if not combo_dir.is_dir():
            continue
        aligned = sorted(combo_dir.glob("*_cpu_thread_detail_aligned.csv"))
        if aligned:
            return aligned[-1]
        detail = sorted(combo_dir.glob("*_cpu_thread_detail.csv"))
        if detail:
            return detail[-1]
    return None


def collect_distinct_values(rows: Iterable[Dict[str, str]], field: str) -> List[str]:
    values: List[str] = []
    seen = set()
    for row in rows:
        value = str(row.get(field, "")).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def build_aux_feature_lookup(input_path: Path) -> Tuple[Dict[str, Dict[str, str]], Path | None]:
    detail_path = find_aux_detail_csv(input_path)
    if detail_path is None:
        return {}, None

    with detail_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        node_idx = str(row.get("node_index", "")).strip()
        if node_idx:
            grouped_rows[node_idx].append(row)

    lookup: Dict[str, Dict[str, str]] = {}
    for node_idx, node_rows in grouped_rows.items():
        out: Dict[str, str] = {}
        for field in SHAPE_FIELDS:
            values = collect_distinct_values(node_rows, field)
            if values:
                out[field] = values[0] if len(values) == 1 else " || ".join(values)

        for field in SIZE_FIELDS:
            numeric_values = [
                value
                for value in (parse_float(row.get(field, "")) for row in node_rows)
                if value is not None
            ]
            if numeric_values:
                out[field] = format_number(sum(numeric_values) / len(numeric_values))

        if out:
            lookup[node_idx] = out

    return lookup, detail_path


def process_file(input_csv: Path, output_csv: Path, dur_source: str) -> None:
    header = read_header(input_csv)
    if not header:
        raise RuntimeError(f"Input CSV is empty: {input_csv}")

    column_plan = build_column_plan(header, dur_source)
    aux_lookup, aux_path = build_aux_feature_lookup(input_csv)

    rows_out: List[Dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out_row: Dict[str, str] = {}
            node_idx = str(row.get("node_idx", "")).strip()
            aux_row = aux_lookup.get(node_idx, {})
            for out_name, source_name in column_plan:
                value = row.get(source_name, "")
                if (not value) and out_name in aux_row:
                    value = aux_row[out_name]
                out_row[out_name] = value
            rows_out.append(out_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[out_name for out_name, _ in column_plan])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[ OK ] {input_csv} -> {output_csv} ({len(rows_out)} rows)")
    if aux_path is not None:
        print(f"[INFO] backfilled shape/size fields from {aux_path}")


def iter_input_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(path for path in input_path.glob(pattern) if path.is_file())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    input_files = iter_input_files(input_path, args.pattern)
    if not input_files:
        raise RuntimeError(f"No input files matched under {input_path}")

    if input_path.is_file():
        process_file(input_path, output_path, args.dur_source)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for input_csv in input_files:
        process_file(input_csv, output_path / input_csv.name, args.dur_source)


if __name__ == "__main__":
    main()
