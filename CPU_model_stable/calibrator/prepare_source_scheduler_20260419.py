from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpu_perf_model.dataset_builder import build_dataset  # noqa: E402

from prepare_common import (  # noqa: E402
    TRAIN_ONLY_RATIOS,
    _json_dump,
    _patch_shape_backfill,
)
from pipeline_eval import materialize_pipeline_artifact_index  # noqa: E402
from static_feature_contract import apply_static_contract_to_output_root  # noqa: E402
from scheduler_20260419_registry import (  # noqa: E402
    DEFAULT_LOGS_ROOT,
    manifest_sources_from_specs,
    parse_scheduler_logs,
    static_source_metadata_map,
)


DEFAULTS = json.loads((SCRIPT_DIR / "configs" / "defaults.json").read_text(encoding="utf-8"))
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = (PROJECT_ROOT / DEFAULTS["source_scheduler_20260419"]["source_output_root"]).resolve()
DEFAULT_SOURCE_HW = (SCRIPT_DIR / DEFAULTS["default_assets"]["source_hw"]).resolve()
DEFAULT_LOGS_DIR = (PROJECT_ROOT / DEFAULTS["source_scheduler_20260419"]["logs_root"]).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the canonical source raw dataset from case_dataset_scheduler_logs/20260419_185414."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--source-hw-config", default=str(DEFAULT_SOURCE_HW))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-files-per-case", type=int, default=0)
    return parser.parse_args()


def _patch_static_source_metadata(
    output_root: Path,
    *,
    specs: list[dict[str, Any]],
    source_hw: Path,
) -> dict[str, Any]:
    source_map = static_source_metadata_map(specs)
    touched_files: list[str] = []
    validation: dict[str, Any] = {}
    static_columns = [
        "case_id",
        "num_threads",
        "inter_threads",
        "arch_embedding_size",
        "arch_mlp_bot",
        "arch_mlp_top",
    ]

    for split_name in ("dataset_full", "train", "val", "test"):
        csv_path = output_root / f"{split_name}.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        if frame.empty:
            for column in static_columns:
                if column not in frame.columns:
                    frame[column] = pd.Series(dtype="object")
            frame.to_csv(csv_path, index=False)
            touched_files.append(csv_path.name)
            continue

        frame["source_name"] = frame["source_name"].astype(str)
        missing = sorted(set(frame["source_name"].unique()) - set(source_map))
        if missing:
            raise ValueError(f"Missing static scheduler metadata for source_name values: {missing}")

        for column in static_columns:
            frame[column] = frame["source_name"].map(lambda value, key=column: source_map[str(value)][key])
        frame["num_threads"] = pd.to_numeric(frame["num_threads"], errors="raise").astype(int)
        frame["inter_threads"] = pd.to_numeric(frame["inter_threads"], errors="raise").astype(int)
        frame.to_csv(csv_path, index=False)
        touched_files.append(csv_path.name)

        if split_name == "dataset_full":
            validation = {
                "rows": int(len(frame)),
                "case_ids": sorted(frame["case_id"].astype(str).unique().tolist()),
                "source_names": sorted(frame["source_name"].astype(str).unique().tolist()),
                "arch_embedding_size_values": sorted(frame["arch_embedding_size"].astype(str).unique().tolist()),
                "arch_mlp_bot_values": sorted(frame["arch_mlp_bot"].astype(str).unique().tolist()),
                "arch_mlp_top_values": sorted(frame["arch_mlp_top"].astype(str).unique().tolist()),
                "num_threads_values": sorted(int(value) for value in frame["num_threads"].dropna().unique().tolist()),
                "inter_threads_values": sorted(int(value) for value in frame["inter_threads"].dropna().unique().tolist()),
                "source_hw_config": str(source_hw),
            }
    return {"patched_files": touched_files, "validation": validation}


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.head(0).copy()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    logs_root = Path(args.logs_root).resolve()
    source_hw = Path(args.source_hw_config).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    specs = parse_scheduler_logs(logs_root)
    print(f"[prepare_source_scheduler] logs_root={logs_root}", flush=True)
    print(f"[prepare_source_scheduler] discovered_cases={len(specs)}", flush=True)

    source_manifest_path = output_root / DEFAULTS["source_scheduler_20260419"]["source_manifest_filename"]
    _json_dump(source_manifest_path, {"sources": manifest_sources_from_specs(specs)})
    print(f"[prepare_source_scheduler] source_manifest={source_manifest_path}", flush=True)

    frames: list[pd.DataFrame] = []
    per_source_build_summaries: list[dict[str, Any]] = []
    feature_manifest_payload: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="source_scheduler_20260419_", dir="/tmp") as temp_dir:
        temp_root = Path(temp_dir)
        for index, spec in enumerate(specs, start=1):
            case_root = temp_root / str(spec["source_name"])
            case_root.mkdir(parents=True, exist_ok=True)
            case_manifest_path = case_root / "source_manifest.json"
            _json_dump(case_manifest_path, {"sources": manifest_sources_from_specs([spec])})
            print(
                f"[prepare_source_scheduler] building {index}/{len(specs)} "
                f"{spec['case_id']} from {Path(spec['feature_root']).name}",
                flush=True,
            )
            summary = build_dataset(
                output_dir=case_root,
                case_pattern="*",
                selected_case_ids=None,
                source_manifest=case_manifest_path,
                max_files_per_case=int(args.max_files_per_case),
                group_column="sample_group",
                seed=int(args.seed),
                ratios=dict(TRAIN_ONLY_RATIOS),
                feature_dialect="auto",
                hardware_profile_path=source_hw,
                drop_first_profile_batch=True,
                profile_instability_metric="last2_range_ratio",
                profile_instability_threshold=0.20,
                disable_profile_stability_filter=False,
            )
            frame = pd.read_csv(case_root / "dataset_full.csv", low_memory=False)
            frames.append(frame)
            per_source_build_summaries.append(
                {
                    "case_id": str(spec["case_id"]),
                    "source_name": str(spec["source_name"]),
                    "rows": int(len(frame)),
                    "build_dataset_summary": summary,
                }
            )
            if feature_manifest_payload is None:
                feature_manifest_payload = json.loads((case_root / "feature_columns.json").read_text(encoding="utf-8"))

    if not frames:
        raise RuntimeError("No rows were assembled from the scheduler-backed source cases")

    dataset = pd.concat(frames, ignore_index=True, sort=False)
    dataset = dataset.sort_values(["split", "source_name", "combo", "op_idx", "trace_op_name"], kind="stable").reset_index(drop=True)
    dataset.to_csv(output_root / "dataset_full.csv", index=False)
    dataset.to_csv(output_root / "train.csv", index=False)
    _empty_like(dataset).to_csv(output_root / "val.csv", index=False)
    _empty_like(dataset).to_csv(output_root / "test.csv", index=False)
    if feature_manifest_payload is not None:
        _json_dump(output_root / "feature_columns.json", feature_manifest_payload)

    patch_summary = _patch_static_source_metadata(output_root, specs=specs, source_hw=source_hw)
    shape_backfill_summary = _patch_shape_backfill(output_root)
    manifest = {
        "phase_scope": DEFAULTS["source_scheduler_20260419"]["phase_scope"],
        "dataset_name": "source_kunpeng920_dataset3_scheduler_20260419_raw",
        "scheduler_logs_root": str(logs_root),
        "source_hw_config": str(source_hw),
        "source_manifest_json": str(source_manifest_path),
        "case_specs": [
            {
                "case_family": str(spec["case_family"]),
                "case_id": str(spec["case_id"]),
                "source_name": str(spec["source_name"]),
                "source_mode": str(spec["source_mode"]),
                "num_threads": int(spec["num_threads"]),
                "inter_threads": int(spec["inter_threads"]),
                "arch_embedding_size": str(spec["arch_embedding_size"]),
                "arch_mlp_bot": str(spec["arch_mlp_bot"]),
                "arch_mlp_top": str(spec["arch_mlp_top"]),
                "feature_root": str(Path(spec["feature_root"]).resolve()),
                "sweep_dir": str(Path(spec["sweep_dir"]).resolve()),
                "log_path": str(Path(spec["log_path"]).resolve()),
            }
            for spec in specs
        ],
        "combined_rows_pre_patch": int(len(dataset)),
        "per_source_build_summaries": per_source_build_summaries,
        "explicit_static_source_patch": patch_summary,
        "shape_backfill_patch": shape_backfill_summary,
    }
    _json_dump(output_root / "manifest.json", manifest)
    static_contract_summary = apply_static_contract_to_output_root(output_root)
    manifest["static_feature_contract_patch"] = static_contract_summary
    _, pipeline_index_summary = materialize_pipeline_artifact_index(output_root)
    manifest["pipeline_artifact_index"] = pipeline_index_summary
    _json_dump(output_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
