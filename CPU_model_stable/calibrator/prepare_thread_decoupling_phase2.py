from __future__ import annotations

import argparse
import json
import sys
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
    filter_specs_by_case_families,
    manifest_sources_from_specs,
    parse_scheduler_logs,
    static_source_metadata_map,
)


DEFAULTS = json.loads((SCRIPT_DIR / "configs" / "defaults.json").read_text(encoding="utf-8"))
DEFAULT_OUTPUT_ROOT = (SCRIPT_DIR / DEFAULTS["thread_decoupling_phase2"]["source_output_root"]).resolve()
DEFAULT_SOURCE_HW = (SCRIPT_DIR / DEFAULTS["default_assets"]["source_hw"]).resolve()
DEFAULT_LOGS_DIR = DEFAULT_LOGS_ROOT
PHASE2_INCLUDED_CASE_FAMILIES = {"case_7", "case_8", "case_10"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare source-only multicase thread-decoupling phase2 raw dataset.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--source-hw-config", default=str(DEFAULT_SOURCE_HW))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-files-per-case", type=int, default=0)
    return parser.parse_args()


def _patch_static_source_metadata(
    output_root: Path,
    *,
    case_specs: list[dict[str, Any]],
    source_hw: Path,
) -> dict[str, Any]:
    source_map = static_source_metadata_map(case_specs)
    touched_files: list[str] = []
    validation: dict[str, Any] = {}
    for split_name in ("dataset_full", "train", "val", "test"):
        csv_path = output_root / f"{split_name}.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        if frame.empty:
            frame["inter_threads"] = pd.Series(dtype=float)
            frame.to_csv(csv_path, index=False)
            touched_files.append(csv_path.name)
            continue
        frame["source_name"] = frame["source_name"].astype(str)
        missing = sorted(set(frame["source_name"].unique()) - set(source_map))
        if missing:
            raise ValueError(f"Failed to assign explicit static metadata for source_name values: {missing}")
        frame["case_id"] = frame["source_name"].map(lambda value: source_map[str(value)]["case_id"])
        frame["num_threads"] = frame["source_name"].map(lambda value: source_map[str(value)]["num_threads"])
        frame["inter_threads"] = frame["source_name"].map(lambda value: source_map[str(value)]["inter_threads"])
        frame["arch_embedding_size"] = frame["source_name"].map(lambda value: source_map[str(value)]["arch_embedding_size"])
        frame["arch_mlp_bot"] = frame["source_name"].map(lambda value: source_map[str(value)]["arch_mlp_bot"])
        frame["arch_mlp_top"] = frame["source_name"].map(lambda value: source_map[str(value)]["arch_mlp_top"])
        frame["num_threads"] = pd.to_numeric(frame["num_threads"], errors="raise").astype(int)
        frame["inter_threads"] = frame["inter_threads"].astype(int)
        frame.to_csv(csv_path, index=False)
        touched_files.append(csv_path.name)
        if split_name == "dataset_full":
            validation = {
                "rows": int(len(frame)),
                "case_ids": sorted(frame["case_id"].astype(str).unique().tolist()),
                "source_names": sorted(frame["source_name"].astype(str).unique().tolist()),
                "inter_threads_values": sorted(int(value) for value in frame["inter_threads"].dropna().unique().tolist()),
                "num_threads_values": sorted(int(value) for value in pd.to_numeric(frame["num_threads"], errors="coerce").dropna().unique().tolist()),
                "source_hw_config": str(source_hw),
            }
    return {"patched_files": touched_files, "validation": validation}


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    logs_root = Path(args.logs_root).resolve()
    source_hw = Path(args.source_hw_config).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_specs = parse_scheduler_logs(logs_root)
    case_specs = filter_specs_by_case_families(all_specs, PHASE2_INCLUDED_CASE_FAMILIES)
    for spec in case_specs:
        feature_root = Path(spec["feature_root"]).resolve()
        sweep_dir = Path(spec["sweep_dir"]).resolve()
        if not feature_root.exists():
            raise FileNotFoundError(f"Phase2 feature root does not exist: {feature_root}")
        if not sweep_dir.exists():
            raise FileNotFoundError(f"Phase2 sweep dir does not exist: {sweep_dir}")

    print(f"[prepare_phase2] output_root={output_root}", flush=True)
    source_manifest_path = output_root / DEFAULTS["thread_decoupling_phase2"]["source_manifest_filename"]
    _json_dump(source_manifest_path, {"sources": manifest_sources_from_specs(case_specs)})
    print(f"[prepare_phase2] source_manifest={source_manifest_path}", flush=True)

    summary = build_dataset(
        output_dir=output_root,
        case_pattern="*",
        selected_case_ids=None,
        source_manifest=source_manifest_path,
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
    patch_summary = _patch_static_source_metadata(output_root, case_specs=case_specs, source_hw=source_hw)
    shape_backfill_summary = _patch_shape_backfill(output_root)
    manifest = {
        "phase_scope": DEFAULTS["thread_decoupling_phase2"]["phase_scope"],
        "dataset_name": "source_kunpeng920_dataset3_thread_decoupling_phase2_raw",
        "scheduler_logs_root": str(logs_root),
        "source_hw_config": str(source_hw),
        "source_manifest_json": str(source_manifest_path),
        "case_specs": [
            {
                "case_id": spec["case_id"],
                "source_name": spec["source_name"],
                "source_mode": spec["source_mode"],
                "num_threads": int(spec["num_threads"]),
                "inter_threads": int(spec["inter_threads"]),
                "feature_root": str(Path(spec["feature_root"]).resolve()),
                "sweep_dir": str(Path(spec["sweep_dir"]).resolve()),
                "log_path": str(Path(spec["log_path"]).resolve()),
            }
            for spec in case_specs
        ],
        "build_dataset_summary": summary,
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
