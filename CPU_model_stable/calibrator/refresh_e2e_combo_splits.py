from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_splits import (
    JOINT_6D_AXIS,
    JOINT_6D_POLICY,
    OUTPUT_ROOT,
    POLICY_CHOICES,
    PHASE2_POLICIES,
    SOURCE_DATASET_NAME,
    SOURCE_RAW_ROOT,
    PHASE2_SOURCE_ROOT,
    PHASE2_SCOPE,
    SOURCE_SCHEDULER_SCOPE,
    _format_value,
    _path_is_under,
    _resolve_phase_scope,
    _selected_policies,
    _source_root_for_policy,
)
from pipeline_eval import (
    E2E_FULL_DATASET_CSV_NAME,
    E2E_FULL_GROUPS_CSV_NAME,
    E2E_FULL_MANIFEST_JSON_NAME,
    build_e2e_combo_split_frames,
    load_or_materialize_e2e_combo_full_dataset,
    materialize_e2e_combo_full_dataset,
    materialize_split_pipeline_snapshot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh combo-complete E2E train/test splits for source policies.",
    )
    parser.add_argument("--policy", default="all", choices=POLICY_CHOICES)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--target-test-ratio", type=float, default=0.30)
    parser.add_argument(
        "--e2e-workers",
        type=int,
        default=0,
        help="Workers for raw E2E combo materialization. Use 0 for auto capped at 32, 1 for serial.",
    )
    parser.add_argument(
        "--force-rebuild-raw-e2e",
        action="store_true",
        help="Delete stale raw E2E cache files before rebuilding.",
    )
    parser.add_argument(
        "--dataset-scope",
        default="source",
        choices=["source"],
        help="Current script only supports source policies.",
    )
    return parser.parse_args()


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _group_case_combo_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_name", "case_id", "combo", "row_count"])
    grouped = (
        frame[["source_name", "case_id", "combo", "row_uid"]]
        .groupby(["source_name", "case_id", "combo"], as_index=False)
        .agg(row_count=("row_uid", "count"))
        .sort_values(["source_name", "combo"], kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def _main_axis_for_policy(policy: str) -> tuple[str, str | None]:
    if policy == "inter_threads_decoupled_v2":
        return "inter_threads", "num_threads"
    if policy == "num_threads_decoupled_v2":
        return "num_threads", "inter_threads"
    if policy == JOINT_6D_POLICY:
        return JOINT_6D_AXIS, None
    return policy, None


def _cleanup_raw_e2e_cache(dataset_root: Path) -> None:
    for name in [
        E2E_FULL_DATASET_CSV_NAME,
        E2E_FULL_GROUPS_CSV_NAME,
        E2E_FULL_MANIFEST_JSON_NAME,
    ]:
        path = dataset_root / name
        if path.exists():
            path.unlink()


def _cleanup_split_e2e_files(split_root: Path) -> None:
    for name in [
        "e2e_train.csv",
        "e2e_train_case_combo_groups.csv",
        "e2e_test_raw.csv",
        "e2e_test_raw_case_combo_groups.csv",
    ]:
        path = split_root / name
        if path.exists():
            path.unlink()


def _replace_split_summary(
    split_root: Path,
    *,
    e2e_train_frame: pd.DataFrame,
    e2e_test_raw_frame: pd.DataFrame,
    e2e_summary: dict[str, Any],
) -> None:
    summary_path = split_root / "split_summary.csv"
    summary = pd.read_csv(summary_path, low_memory=False)
    summary = summary.loc[~summary["split"].isin(["e2e_train", "e2e_test_raw"])].copy()
    e2e_rows = pd.DataFrame(
        [
            {
                "split": "e2e_train",
                "rows": int(len(e2e_train_frame)),
                "case_combo_groups": int(_group_case_combo_rows(e2e_train_frame).shape[0]),
                "case_count": int(e2e_train_frame["case_id"].astype(str).nunique()),
                "ratio": float(e2e_summary.get("combo_train_ratio", 0.0)),
            },
            {
                "split": "e2e_test_raw",
                "rows": int(len(e2e_test_raw_frame)),
                "case_combo_groups": int(_group_case_combo_rows(e2e_test_raw_frame).shape[0]),
                "case_count": int(e2e_test_raw_frame["case_id"].astype(str).nunique()),
                "ratio": float(e2e_summary.get("combo_test_ratio", 0.0)),
            },
        ]
    )
    summary = pd.concat([summary, e2e_rows], ignore_index=True)
    summary.to_csv(summary_path, index=False)


def _update_split_manifest(
    split_root: Path,
    *,
    source_root: Path,
    main_axis: str,
    snapshot: dict[str, Any],
    e2e_train_frame: pd.DataFrame,
    e2e_test_raw_frame: pd.DataFrame,
    e2e_summary: dict[str, Any],
) -> None:
    manifest_path = split_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "pipeline_metadata_available": bool(snapshot.get("pipeline_metadata_available", False)),
            "pipeline_metadata_reason": snapshot.get("pipeline_metadata_reason", ""),
            "pipeline_artifact_index_csv": snapshot.get("pipeline_artifact_index_csv"),
            "pipeline_artifact_index_json": snapshot.get("pipeline_artifact_index_json"),
            "pipeline_artifact_index_rows": int(snapshot.get("pipeline_artifact_index_rows", 0)),
            "pipeline_combo_manifest_csv": snapshot.get("pipeline_combo_manifest_csv"),
            "pipeline_combo_manifest_rows": int(snapshot.get("pipeline_combo_manifest_rows", 0)),
            "e2e_test_raw_available": True,
            "e2e_split_mode": str(e2e_summary.get("split_mode", "combo_stratified")),
            "e2e_graph_scope": str(e2e_summary.get("graph_scope", "")),
            "e2e_target_test_ratio": float(e2e_summary.get("target_test_ratio", 0.0)),
            "e2e_main_axis": main_axis,
            "e2e_main_axis_test_values": [
                _format_value(value) for value in e2e_summary.get("main_axis_test_values", [])
            ],
            "e2e_train_rows": int(len(e2e_train_frame)),
            "e2e_train_case_combo_groups": int(_group_case_combo_rows(e2e_train_frame).shape[0]),
            "e2e_train_combo_ratio": float(e2e_summary.get("combo_train_ratio", 0.0)),
            "e2e_test_rows": int(len(e2e_test_raw_frame)),
            "e2e_test_ratio": float(e2e_summary.get("combo_test_ratio", 0.0)),
            "e2e_test_case_combo_groups": int(_group_case_combo_rows(e2e_test_raw_frame).shape[0]),
            "e2e_total_combo_count": int(e2e_summary.get("combo_count_total", 0)),
            "e2e_train_combo_count": int(e2e_summary.get("combo_count_train", 0)),
            "e2e_selected_combo_count": int(e2e_summary.get("combo_count_test", 0)),
            "input_dataset_root": str(source_root),
        }
    )
    _json_dump(manifest_path, manifest)


def _validate_policy_source_root(policy: str, source_root: Path) -> None:
    phase_scope = _resolve_phase_scope(source_root)
    if policy in PHASE2_POLICIES:
        if source_root != PHASE2_SOURCE_ROOT:
            raise ValueError(f"Phase2 policy {policy} must use {PHASE2_SOURCE_ROOT}, got {source_root}")
        if phase_scope != PHASE2_SCOPE:
            raise ValueError(f"Phase2 policy {policy} requires phase_scope={PHASE2_SCOPE}, got {phase_scope}")
    else:
        if source_root != SOURCE_RAW_ROOT:
            raise ValueError(f"Source policy {policy} must use {SOURCE_RAW_ROOT}, got {source_root}")
        if phase_scope != SOURCE_SCHEDULER_SCOPE:
            raise ValueError(
                f"Source policy {policy} requires phase_scope={SOURCE_SCHEDULER_SCOPE}, got {phase_scope}"
            )


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    e2e_workers = int(args.e2e_workers)
    if _path_is_under(output_root, output_root / "_invalid_legacy_splits_20260420"):
        raise ValueError(f"Output root cannot point into archived legacy splits: {output_root}")

    selected_policies = _selected_policies(str(args.policy))
    prepared_raw_roots: set[Path] = set()
    e2e_full_cache: dict[Path, tuple[pd.DataFrame, dict[str, Any]]] = {}

    for policy in selected_policies:
        source_root = _source_root_for_policy(policy)
        _validate_policy_source_root(policy, source_root)
        if source_root not in prepared_raw_roots:
            if args.force_rebuild_raw_e2e:
                print(f"[refresh-e2e] cleanup raw root cache: {source_root}", flush=True)
                _cleanup_raw_e2e_cache(source_root)
                dataset, summary = materialize_e2e_combo_full_dataset(source_root, max_workers=e2e_workers)
            else:
                dataset, summary = load_or_materialize_e2e_combo_full_dataset(source_root, max_workers=e2e_workers)
            if dataset.empty:
                raise RuntimeError(f"failed to materialize raw E2E dataset for {source_root}: {summary}")
            print(
                f"[refresh-e2e] raw root={source_root.name} rows={len(dataset)} combos={summary.get('combo_count', 0)} "
                f"graph_scope={summary.get('graph_scope', '')} e2e_workers={e2e_workers}",
                flush=True,
            )
            e2e_full_cache[source_root] = (dataset, summary)
            prepared_raw_roots.add(source_root)
        else:
            print(f"[refresh-e2e] reuse prepared raw root cache: {source_root}", flush=True)

        e2e_full_frame, e2e_full_summary = e2e_full_cache[source_root]
        main_axis, _aux_axis = _main_axis_for_policy(policy)
        split_root = output_root / f"{SOURCE_DATASET_NAME}_unseen_{policy}"
        if not split_root.exists():
            raise FileNotFoundError(f"split root does not exist for {policy}: {split_root}")

        print(f"[refresh-e2e] policy={policy} split_root={split_root}", flush=True)
        _cleanup_split_e2e_files(split_root)
        e2e_train_frame, e2e_test_raw_frame, e2e_summary = build_e2e_combo_split_frames(
            source_root,
            base_frame=pd.DataFrame(),
            policy=policy,
            main_axis=main_axis,
            target_test_ratio=float(args.target_test_ratio),
            max_workers=e2e_workers,
            full_frame=e2e_full_frame,
            full_summary=e2e_full_summary,
        )
        if e2e_train_frame.empty or e2e_test_raw_frame.empty:
            raise RuntimeError(f"E2E split generation failed for {policy}: {e2e_summary}")

        e2e_train_frame.to_csv(split_root / "e2e_train.csv", index=False)
        _group_case_combo_rows(e2e_train_frame).to_csv(split_root / "e2e_train_case_combo_groups.csv", index=False)
        e2e_test_raw_frame.to_csv(split_root / "e2e_test_raw.csv", index=False)
        _group_case_combo_rows(e2e_test_raw_frame).to_csv(
            split_root / "e2e_test_raw_case_combo_groups.csv",
            index=False,
        )
        snapshot = materialize_split_pipeline_snapshot(
            split_root=split_root,
            input_dataset_root=source_root,
            frame=pd.DataFrame(),
            extra_frames=[e2e_train_frame, e2e_test_raw_frame],
        )
        _replace_split_summary(
            split_root,
            e2e_train_frame=e2e_train_frame,
            e2e_test_raw_frame=e2e_test_raw_frame,
            e2e_summary=e2e_summary,
        )
        _update_split_manifest(
            split_root,
            source_root=source_root,
            main_axis=main_axis,
            snapshot=snapshot,
            e2e_train_frame=e2e_train_frame,
            e2e_test_raw_frame=e2e_test_raw_frame,
            e2e_summary=e2e_summary,
        )

        op_counts = e2e_test_raw_frame["op_type"].value_counts().head(20).to_dict()
        print(
            f"[refresh-e2e] policy={policy} train_rows={len(e2e_train_frame)} "
            f"test_rows={len(e2e_test_raw_frame)} combo_test_ratio={e2e_summary.get('combo_test_ratio', 0.0):.4f} "
            f"graph_scope={e2e_summary.get('graph_scope', '')} ops={op_counts}",
            flush=True,
        )

    print("[refresh-e2e] all requested policies refreshed", flush=True)


if __name__ == "__main__":
    main()
