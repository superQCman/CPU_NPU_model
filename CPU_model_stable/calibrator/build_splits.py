from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULTS = json.loads((SCRIPT_DIR / "configs" / "defaults.json").read_text(encoding="utf-8"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpu_perf_model.classed.build_classed_dataset import build_static_thread_columns  # noqa: E402
from cpu_perf_model.feature_contract import TARGET_COLUMN  # noqa: E402
from pipeline_eval import (  # noqa: E402
    JOINT_6D_AXIS,
    JOINT_6D_COLUMNS,
    JOINT_6D_POLICY,
    attach_joint_6d_tuple_key,
    backfill_joint_6d_static_columns,
    build_e2e_combo_split_frames,
    materialize_split_pipeline_snapshot,
)
from static_feature_contract import augment_static_feature_frame  # noqa: E402


SOURCE_RAW_ROOT = (PROJECT_ROOT / DEFAULTS["source_scheduler_20260419"]["source_output_root"]).resolve()
LEGACY_SOURCE_RAW_ROOT = (SCRIPT_DIR / DEFAULTS["default_assets"]["source_dataset"]).resolve()
TARGET_RAW_ROOT = (SCRIPT_DIR / DEFAULTS["default_assets"]["target_dataset"]).resolve()
PHASE2_SOURCE_ROOT = (PROJECT_ROOT / DEFAULTS["thread_decoupling_phase2"]["source_output_root"]).resolve()
LEGACY_PHASE2_SOURCE_ROOT = (SCRIPT_DIR / DEFAULTS["thread_decoupling_phase2"]["source_output_root"]).resolve()
OUTPUT_ROOT = SCRIPT_DIR / "data"
TARGET_TEST_RATIO = float(DEFAULTS["target_test_ratio"])
RETAINED_RATIO = dict(DEFAULTS.get("retained_ratio", {}))
RETAINED_TARGET_TEST_RATIO = float(RETAINED_RATIO.get("target_test_ratio", TARGET_TEST_RATIO))
RETAINED_LOWER_BOUND = float(RETAINED_RATIO.get("lower_bound", 0.27))
RETAINED_UPPER_BOUND = float(RETAINED_RATIO.get("upper_bound", 0.33))

SOURCE_DATASET_NAME = "source_kunpeng920_dataset3"
TARGET_DATASET_NAME = "target_s920x20_dataset_all_78910_2_nodrop"
LEGACY_POLICIES = list(DEFAULTS["split_policies"])
PHASE2_POLICIES = list(DEFAULTS.get("phase2_split_policies", []))
JOINT_POLICIES = list(DEFAULTS.get("joint_split_policies", [JOINT_6D_POLICY]))
FORMAL_POLICIES = [*LEGACY_POLICIES, *PHASE2_POLICIES]
POLICY_CHOICES = ["all", "phase2", *LEGACY_POLICIES, *PHASE2_POLICIES, *JOINT_POLICIES]
PHASE2_SCOPE = str(DEFAULTS["thread_decoupling_phase2"]["phase_scope"])
PHASE2_SOURCE_ROOT_NAME = Path(DEFAULTS["thread_decoupling_phase2"]["source_output_root"]).name
SOURCE_SCHEDULER_SCOPE = str(DEFAULTS["source_scheduler_20260419"]["phase_scope"])
LEGACY_INVALID_CASE_IDS = {"case_8_4_4", "case_9_4_4", "case_10_4_6"}
INVALID_LEGACY_ROOT = OUTPUT_ROOT / "_invalid_legacy_splits_20260420"
CASE_ID_THREAD_RE = re.compile(r"^case_\d+_(?P<num_threads>\d+)_(?P<inter_threads>\d+)$")
SOURCE_NAME_THREAD_RE = re.compile(r"^case_\d+_run_(?P<num_threads>\d+)_(?P<inter_threads>\d+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unseen-axis train/test splits for cross-hardware calibrator experiments.")
    parser.add_argument("--source-root", "--raw-source-root", dest="source_root", default=str(SOURCE_RAW_ROOT))
    parser.add_argument("--target-root", "--raw-target-root", dest="target_root", default=str(TARGET_RAW_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--policy", default="all", choices=POLICY_CHOICES)
    parser.add_argument("--dataset-scope", default="source", choices=["source", "target", "both"])
    parser.add_argument("--target-test-ratio", type=float, default=TARGET_TEST_RATIO)
    parser.add_argument("--case-id", default=str(DEFAULTS["default_case_holdout"]))
    parser.add_argument(
        "--e2e-workers",
        type=int,
        default=0,
        help="Workers for raw E2E combo materialization when the raw cache is missing. 0 uses auto.",
    )
    parser.add_argument(
        "--e2e-small-divisor",
        type=int,
        default=1,
        help="If >1, keep about 1/N complete E2E pipeline combos per train/test side.",
    )
    parser.add_argument(
        "--e2e-small-seed",
        type=int,
        default=int(DEFAULTS.get(JOINT_6D_POLICY, {}).get("small_dataset_seed", 20260425)),
        help="Stable seed for E2E small-combo sampling.",
    )
    return parser.parse_args()


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _format_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if float(value).is_integer():
            return int(value)
        return float(value)
    text = str(value)
    if text.isdigit():
        return int(text)
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _sorted_unique_values(series: pd.Series) -> list[Any]:
    unique = [_format_value(value) for value in series.dropna().tolist()]
    return sorted(dict.fromkeys(unique))


def _load_full_frame(dataset_root: Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root).resolve()
    frame = pd.read_csv(dataset_root / "dataset_full.csv", low_memory=False)
    frame["row_uid"] = frame["row_uid"].astype(str)
    frame = _attach_static_thread_columns(frame, dataset_root=dataset_root)
    static_columns = {
        "static_is_embedding_branch",
        "static_parallel_branch_count",
        "static_distance_to_merge",
        "arch_embedding_size",
        "arch_mlp_bot",
        "arch_mlp_top",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "num_threads",
    }
    if static_columns.issubset(set(frame.columns)):
        return frame
    augmented = augment_static_feature_frame(frame, dataset_root=dataset_root)
    augmented = _attach_static_thread_columns(augmented, dataset_root=dataset_root)
    return augmented


def _parse_thread_pair(text: Any, pattern: re.Pattern[str]) -> tuple[int | None, int | None]:
    if pd.isna(text):
        return None, None
    match = pattern.fullmatch(str(text).strip())
    if match is None:
        return None, None
    return int(match.group("num_threads")), int(match.group("inter_threads"))


def _manifest_thread_map(dataset_root: Path) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    manifest = _load_manifest(dataset_root)
    by_source: dict[str, dict[str, int]] = {}
    by_case: dict[str, dict[str, int]] = {}
    for spec in manifest.get("case_specs", []):
        try:
            values = {
                "num_threads": int(spec["num_threads"]),
                "inter_threads": int(spec["inter_threads"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        source_name = str(spec.get("source_name", "")).strip()
        case_id = str(spec.get("case_id", "")).strip()
        if source_name:
            by_source[source_name] = values
        if case_id:
            by_case[case_id] = values
    return by_source, by_case


def _series_from_pair(
    frame: pd.DataFrame,
    *,
    source_column: str,
    pattern: re.Pattern[str],
    thread_column: str,
) -> pd.Series:
    if source_column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index)
    pair_index = 0 if thread_column == "num_threads" else 1
    return frame[source_column].map(lambda value: _parse_thread_pair(value, pattern)[pair_index])


def _canonical_thread_candidates(frame: pd.DataFrame, dataset_root: Path, thread_column: str) -> list[tuple[str, pd.Series]]:
    by_source, by_case = _manifest_thread_map(dataset_root)
    candidates: list[tuple[str, pd.Series]] = []
    if "source_name" in frame.columns:
        candidates.append(
            (
                "manifest_source_name",
                frame["source_name"].map(lambda value: by_source.get(str(value), {}).get(thread_column)),
            )
        )
    if "case_id" in frame.columns:
        candidates.append(
            (
                "manifest_case_id",
                frame["case_id"].map(lambda value: by_case.get(str(value), {}).get(thread_column)),
            )
        )
        candidates.append(
            (
                "case_id",
                _series_from_pair(frame, source_column="case_id", pattern=CASE_ID_THREAD_RE, thread_column=thread_column),
            )
        )
    if "source_name" in frame.columns:
        candidates.append(
            (
                "source_name",
                _series_from_pair(
                    frame,
                    source_column="source_name",
                    pattern=SOURCE_NAME_THREAD_RE,
                    thread_column=thread_column,
                ),
            )
        )
    if thread_column == "inter_threads" and {"row_uid", "case_id", "source_name", "combo"}.issubset(frame.columns):
        try:
            resolved = build_static_thread_columns(frame[["row_uid", "case_id", "source_name", "combo"]])
            resolved = frame[["row_uid"]].merge(resolved, on="row_uid", how="left", validate="one_to_one")
            candidates.append(("legacy_static_resolver", resolved["inter_threads"]))
        except Exception:
            pass
    if thread_column in frame.columns:
        candidates.append(("existing_column", frame[thread_column]))
    return candidates


def _choose_canonical_thread_column(
    frame: pd.DataFrame,
    dataset_root: Path,
    thread_column: str,
) -> tuple[pd.Series, pd.Series, list[dict[str, Any]]]:
    candidates = _canonical_thread_candidates(frame, dataset_root, thread_column)
    values = pd.Series(pd.NA, index=frame.index, dtype="object")
    sources = pd.Series("", index=frame.index, dtype="object")
    conflicts: list[dict[str, Any]] = []
    canonical_priority = {
        "manifest_source_name",
        "manifest_case_id",
        "case_id",
        "source_name",
        "legacy_static_resolver",
    }
    canonical_values: dict[str, pd.Series] = {
        name: pd.to_numeric(series, errors="coerce") for name, series in candidates if name in canonical_priority
    }
    for idx in frame.index:
        observed: list[tuple[str, int]] = []
        for name, series in canonical_values.items():
            value = series.loc[idx]
            if pd.notna(value):
                observed.append((name, int(value)))
        unique_values = sorted({value for _, value in observed})
        if len(unique_values) > 1:
            conflicts.append(
                {
                    "row_uid": str(frame.at[idx, "row_uid"]) if "row_uid" in frame.columns else str(idx),
                    "source_name": str(frame.at[idx, "source_name"]) if "source_name" in frame.columns else "",
                    "case_id": str(frame.at[idx, "case_id"]) if "case_id" in frame.columns else "",
                    "thread_column": thread_column,
                    "candidates": ";".join(f"{name}={value}" for name, value in observed),
                }
            )
        for name, series in candidates:
            value = pd.to_numeric(series.loc[idx], errors="coerce")
            if pd.notna(value):
                values.loc[idx] = int(value)
                sources.loc[idx] = name
                break
    return values, sources, conflicts


def _build_static_thread_audit(
    frame: pd.DataFrame,
    *,
    original_num_threads: pd.Series,
    original_inter_threads: pd.Series,
    canonical_num_threads: pd.Series,
    canonical_inter_threads: pd.Series,
    num_sources: pd.Series,
    inter_sources: pd.Series,
) -> pd.DataFrame:
    audit = pd.DataFrame(
        {
            "source_name": frame["source_name"].astype(str) if "source_name" in frame.columns else "",
            "case_id": frame["case_id"].astype(str) if "case_id" in frame.columns else "",
            "original_num_threads": pd.to_numeric(original_num_threads, errors="coerce"),
            "canonical_num_threads": pd.to_numeric(canonical_num_threads, errors="coerce"),
            "num_threads_source": num_sources,
            "num_threads_overridden": pd.to_numeric(original_num_threads, errors="coerce")
            != pd.to_numeric(canonical_num_threads, errors="coerce"),
            "original_inter_threads": pd.to_numeric(original_inter_threads, errors="coerce"),
            "canonical_inter_threads": pd.to_numeric(canonical_inter_threads, errors="coerce"),
            "inter_threads_source": inter_sources,
            "inter_threads_overridden": pd.to_numeric(original_inter_threads, errors="coerce")
            != pd.to_numeric(canonical_inter_threads, errors="coerce"),
        }
    )
    group_columns = ["source_name", "case_id"]
    aggregations = {
        "original_num_threads": "first",
        "canonical_num_threads": "first",
        "num_threads_source": "first",
        "num_threads_overridden": "max",
        "original_inter_threads": "first",
        "canonical_inter_threads": "first",
        "inter_threads_source": "first",
        "inter_threads_overridden": "max",
    }
    return audit.groupby(group_columns, as_index=False, dropna=False).agg(aggregations)


def _attach_static_thread_columns(frame: pd.DataFrame, *, dataset_root: Path) -> pd.DataFrame:
    base = frame.copy()
    for column in ["num_threads", "inter_threads"]:
        if column not in base.columns:
            base[column] = pd.NA
    original_num = base["num_threads"].copy()
    original_inter = base["inter_threads"].copy()
    canonical_num, num_sources, num_conflicts = _choose_canonical_thread_column(base, dataset_root, "num_threads")
    canonical_inter, inter_sources, inter_conflicts = _choose_canonical_thread_column(base, dataset_root, "inter_threads")
    conflicts = num_conflicts + inter_conflicts
    if conflicts:
        preview = conflicts[:10]
        raise AssertionError(f"Conflicting canonical static thread metadata under {dataset_root}: {preview}")
    base["num_threads"] = pd.to_numeric(canonical_num, errors="coerce")
    base["inter_threads"] = pd.to_numeric(canonical_inter, errors="coerce")
    audit = _build_static_thread_audit(
        base,
        original_num_threads=original_num,
        original_inter_threads=original_inter,
        canonical_num_threads=base["num_threads"],
        canonical_inter_threads=base["inter_threads"],
        num_sources=num_sources,
        inter_sources=inter_sources,
    )
    base.attrs["static_thread_audit"] = audit
    return base


def _attach_static_inter_threads(frame: pd.DataFrame) -> pd.DataFrame:
    if "inter_threads" in frame.columns:
        base = frame.copy()
        base["inter_threads"] = pd.to_numeric(base["inter_threads"], errors="coerce")
        return base
    base = frame.copy()
    inter = build_static_thread_columns(base[["row_uid", "case_id", "source_name", "combo"]])
    merged = base.merge(inter, on="row_uid", how="left", validate="one_to_one")
    merged["inter_threads"] = pd.to_numeric(merged["inter_threads"], errors="coerce")
    return merged


def _resolve_phase_scope(dataset_root: Path) -> str:
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        scope = str(payload.get("phase_scope", "")).strip()
        if scope:
            return scope
    if dataset_root.name == PHASE2_SOURCE_ROOT_NAME:
        return PHASE2_SCOPE
    return "default"


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _group_axis_table(frame: pd.DataFrame, axis: str) -> pd.DataFrame:
    required = list(dict.fromkeys(["case_id", "combo", axis, "row_uid"]))
    group = (
        frame[required]
        .groupby(["case_id", "combo"], as_index=False)
        .agg(
            axis_nunique=(axis, "nunique"),
            axis_value=(axis, "first"),
            row_count=("row_uid", "count"),
        )
    )
    if not group[group["axis_nunique"] != 1].empty:
        raise AssertionError(f"Found case+combo groups with multiple {axis} values")
    return group[["case_id", "combo", "axis_value", "row_count"]].copy()


def _counts_by_value(frame: pd.DataFrame, axis: str) -> pd.Series:
    return frame.groupby(axis)["row_uid"].count().sort_index()


def _apply_policy_split(frame: pd.DataFrame, *, axis: str, test_values: list[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if axis == "case_id":
        test_mask = frame["case_id"].astype(str).isin([str(value) for value in test_values])
    else:
        test_mask = frame[axis].isin(test_values)
    train = frame.loc[~test_mask].copy().reset_index(drop=True)
    test = frame.loc[test_mask].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise AssertionError(f"Axis {axis} produced an empty train or test split")
    return train, test


def _downsample_rows(frame: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    if target_rows >= len(frame):
        return frame.copy().reset_index(drop=True)
    if target_rows <= 0:
        raise ValueError(f"target_rows must be positive, got {target_rows}")
    ordered = frame.copy()
    ordered["_stable_rank"] = ordered["row_uid"].astype(str).map(
        lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()
    )
    ordered = ordered.sort_values(["_stable_rank", "row_uid"]).head(int(target_rows)).drop(columns=["_stable_rank"])
    return ordered.reset_index(drop=True)


def _retained_split(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    *,
    target_ratio: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    total_rows = max(len(train_raw) + len(test_raw), 1)
    raw_test_ratio = float(len(test_raw) / total_rows)
    raw_train_ratio = float(len(train_raw) / total_rows)
    if lower_bound <= raw_test_ratio <= upper_bound:
        return train_raw.copy().reset_index(drop=True), test_raw.copy().reset_index(drop=True), {
            "sampling_policy": "keep_all_within_band",
            "raw_train_rows": int(len(train_raw)),
            "raw_test_rows": int(len(test_raw)),
            "raw_train_ratio": raw_train_ratio,
            "raw_test_ratio": raw_test_ratio,
        }

    if raw_test_ratio < lower_bound:
        target_train_rows = int(round(len(test_raw) * (1.0 - target_ratio) / max(target_ratio, 1e-9)))
        target_train_rows = max(1, min(target_train_rows, len(train_raw)))
        retained_train = _downsample_rows(train_raw, target_train_rows)
        retained_test = test_raw.copy().reset_index(drop=True)
    else:
        target_test_rows = int(round(len(train_raw) * target_ratio / max(1.0 - target_ratio, 1e-9)))
        target_test_rows = max(1, min(target_test_rows, len(test_raw)))
        retained_train = train_raw.copy().reset_index(drop=True)
        retained_test = _downsample_rows(test_raw, target_test_rows)

    retained_total = max(len(retained_train) + len(retained_test), 1)
    return retained_train, retained_test, {
        "sampling_policy": "downsample_larger_side_to_7_3",
        "raw_train_rows": int(len(train_raw)),
        "raw_test_rows": int(len(test_raw)),
        "raw_train_ratio": raw_train_ratio,
        "raw_test_ratio": raw_test_ratio,
        "retained_train_rows": int(len(retained_train)),
        "retained_test_rows": int(len(retained_test)),
        "retained_train_ratio": float(len(retained_train) / retained_total),
        "retained_test_ratio": float(len(retained_test) / retained_total),
    }


def _split_group_tables(train_frame: pd.DataFrame, test_frame: pd.DataFrame, axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_groups = _group_axis_table(train_frame, axis)
    test_groups = _group_axis_table(test_frame, axis)
    return train_groups, test_groups


def _e2e_case_combo_groups(frame: pd.DataFrame) -> pd.DataFrame:
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


def _e2e_fit_subset(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    target = pd.to_numeric(work.get(TARGET_COLUMN, pd.Series(pd.NA, index=work.index)), errors="coerce")
    stable = pd.to_numeric(work.get("profile_is_stable", pd.Series(1.0, index=work.index)), errors="coerce").fillna(1.0)
    mask = target.notna() & (target > 0.0) & stable.gt(0.5)
    return work.loc[mask].copy().reset_index(drop=True)


def _excluded_value_set(axis: str) -> set[str]:
    return {str(value) for value in DEFAULTS.get("excluded_test_values", {}).get(axis, [])}


def _filter_candidate_values(values: list[Any], axis: str) -> list[Any]:
    excluded = _excluded_value_set(axis)
    if not excluded:
        return values
    filtered = [value for value in values if str(_format_value(value)) not in excluded]
    if not filtered:
        raise ValueError(f"All candidate values for axis {axis} were excluded from held-out candidates")
    return filtered


def _select_joint_6d_holdout_values(frame: pd.DataFrame, *, target_ratio: float) -> tuple[list[Any], dict[str, Any]]:
    if JOINT_6D_AXIS not in frame.columns:
        raise KeyError(f"Missing {JOINT_6D_AXIS}; call attach_joint_6d_tuple_key before selecting joint holdout")
    counts = frame.groupby(JOINT_6D_AXIS, dropna=False)["row_uid"].count().reset_index(name="row_count")
    if len(counts) <= 1:
        raise ValueError(f"Need at least two {JOINT_6D_AXIS} values for a train/test split")
    counts["stable_hash"] = counts[JOINT_6D_AXIS].astype(str).map(
        lambda value: hashlib.sha1(f"{JOINT_6D_POLICY}::{value}".encode("utf-8")).hexdigest()
    )
    ordered = counts.sort_values(["stable_hash", JOINT_6D_AXIS], kind="stable").reset_index(drop=True)
    total_rows = int(ordered["row_count"].sum())
    target_rows = int(round(total_rows * float(target_ratio)))
    target_rows = max(1, min(target_rows, max(total_rows - 1, 1)))
    selected: list[str] = []
    selected_rows = 0
    for row in ordered.itertuples(index=False):
        if selected and selected_rows >= target_rows:
            break
        selected.append(str(getattr(row, JOINT_6D_AXIS)))
        selected_rows += int(row.row_count)
    if len(selected) > 1:
        last_value = selected[-1]
        last_rows = int(ordered.loc[ordered[JOINT_6D_AXIS].astype(str).eq(last_value), "row_count"].iloc[0])
        without_last = selected_rows - last_rows
        if without_last > 0 and abs(without_last - target_rows) < abs(selected_rows - target_rows):
            selected.pop()
            selected_rows = without_last
    selected_set = set(selected)
    train_key_count = int(len(ordered) - len(selected_set))
    return selected, {
        "selection_method": "joint_6d_tuple_stable_hash",
        "main_axis": JOINT_6D_AXIS,
        "joint_columns": list(JOINT_6D_COLUMNS),
        "joint_key_count": int(len(ordered)),
        "train_joint_key_count": train_key_count,
        "test_joint_key_count": int(len(selected_set)),
        "source_test_ratio": float(selected_rows / max(total_rows, 1)),
        "test_ratio": float(selected_rows / max(total_rows, 1)),
        "objective": float(abs((selected_rows / max(total_rows, 1)) - float(target_ratio))),
    }


def _axis_overlap_payload(train_frame: pd.DataFrame, test_frame: pd.DataFrame, axis: str) -> dict[str, Any]:
    train_values = {str(_format_value(value)) for value in train_frame[axis].dropna().tolist()}
    test_values = {str(_format_value(value)) for value in test_frame[axis].dropna().tolist()}
    overlap = sorted(train_values.intersection(test_values))
    return {
        "axis": axis,
        "train_unique": int(len(train_values)),
        "test_unique": int(len(test_values)),
        "overlap_count": int(len(overlap)),
        "overlap_sample": "|".join(overlap[:20]),
    }


def _write_joint_6d_tuple_audit(output_root: Path, train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    rows.append({"audit_type": "joint_key", **_axis_overlap_payload(train_raw, test_raw, JOINT_6D_AXIS)})
    for axis in JOINT_6D_COLUMNS:
        rows.append({"audit_type": "single_axis", **_axis_overlap_payload(train_raw, test_raw, axis)})
    pd.DataFrame(rows).to_csv(output_root / "joint_6d_tuple_audit.csv", index=False)


def _write_small_e2e_audit(
    output_root: Path,
    *,
    e2e_summary: dict[str, Any],
    e2e_train_groups: pd.DataFrame,
    e2e_test_groups: pd.DataFrame,
) -> None:
    divisor = int(e2e_summary.get("small_dataset_divisor", 1))
    if divisor <= 1:
        return

    def _node_stats(groups: pd.DataFrame) -> dict[str, Any]:
        if groups.empty:
            return {"min_rows_per_combo": 0, "max_rows_per_combo": 0, "mean_rows_per_combo": 0.0}
        row_count = pd.to_numeric(groups["row_count"], errors="coerce").fillna(0)
        return {
            "min_rows_per_combo": int(row_count.min()),
            "max_rows_per_combo": int(row_count.max()),
            "mean_rows_per_combo": float(row_count.mean()),
        }

    combined_count = int(len(e2e_train_groups) + len(e2e_test_groups))
    rows = [
        {
            "scope": "combined",
            "small_dataset_divisor": divisor,
            "small_dataset_seed": int(e2e_summary.get("small_dataset_seed", 0)),
            "full_combo_count": int(e2e_summary.get("full_combo_count_total", 0)),
            "small_combo_count": combined_count,
            "small_combo_fraction": float(e2e_summary.get("small_dataset_combo_fraction", 0.0)),
            **_node_stats(pd.concat([e2e_train_groups, e2e_test_groups], ignore_index=True, sort=False)),
        },
        {
            "scope": "e2e_train",
            "small_dataset_divisor": divisor,
            "small_dataset_seed": int(e2e_summary.get("small_dataset_seed", 0)),
            "full_combo_count": int(e2e_summary.get("full_combo_count_train", 0)),
            "small_combo_count": int(len(e2e_train_groups)),
            "small_combo_fraction": float(e2e_summary.get("small_dataset_train_combo_fraction", 0.0)),
            **_node_stats(e2e_train_groups),
        },
        {
            "scope": "e2e_test_raw",
            "small_dataset_divisor": divisor,
            "small_dataset_seed": int(e2e_summary.get("small_dataset_seed", 0)),
            "full_combo_count": int(e2e_summary.get("full_combo_count_test", 0)),
            "small_combo_count": int(len(e2e_test_groups)),
            "small_combo_fraction": float(e2e_summary.get("small_dataset_test_combo_fraction", 0.0)),
            **_node_stats(e2e_test_groups),
        },
    ]
    pd.DataFrame(rows).to_csv(output_root / "small_e2e_audit.csv", index=False)


def _shared_candidate_values(source_counts: pd.Series, target_counts: pd.Series, axis: str) -> list[Any]:
    shared = sorted(set(source_counts.index).intersection(set(target_counts.index)))
    return _filter_candidate_values([_format_value(value) for value in shared], axis)


def _single_dataset_best_subset(
    counts: pd.Series,
    *,
    axis: str,
    target_ratio: float,
) -> tuple[list[Any], dict[str, Any]]:
    candidate_values = _filter_candidate_values([_format_value(value) for value in counts.index.tolist()], axis)
    if not candidate_values:
        raise ValueError(f"No candidate values to build a holdout subset for axis {axis}")
    total_rows = max(float(counts.sum()), 1.0)
    if len(candidate_values) > 20 or axis in {"batch_size", "num_indices_per_lookup"}:
        beam: list[tuple[float, tuple[Any, ...]]] = [(0.0, tuple())]
        mean_count = max(int(round(len(candidate_values) * target_ratio)), 1)
        beam_size = 400
        for value in candidate_values:
            weight = float(counts.loc[value]) / total_rows
            candidates: list[tuple[float, tuple[Any, ...]]] = []
            for ratio, chosen in beam:
                candidates.append((ratio, chosen))
                candidates.append((ratio + weight, chosen + (value,)))
            scored: list[tuple[float, int, int, float, tuple[Any, ...]]] = []
            seen: set[tuple[float, int]] = set()
            for ratio, chosen in candidates:
                key = (round(ratio, 6), len(chosen))
                if key in seen:
                    continue
                seen.add(key)
                scored.append((abs(ratio - target_ratio), abs(len(chosen) - mean_count), len(chosen), ratio, chosen))
            scored.sort()
            beam = [(ratio, chosen) for _, _, _, ratio, chosen in scored[:beam_size]]
        best = sorted(
            [(abs(ratio - target_ratio), abs(len(chosen) - mean_count), len(chosen), ratio, chosen) for ratio, chosen in beam if chosen]
        )[0]
        chosen = list(best[4])
        return chosen, {
            "selection_method": "single_dataset_beam_search",
            "candidate_values": [_format_value(value) for value in candidate_values],
            "test_ratio": float(best[3]),
            "objective": float(best[0]),
        }
    best: tuple[float, int, tuple[Any, ...], float] | None = None
    for subset_size in range(1, len(candidate_values) + 1):
        for chosen in itertools.combinations(candidate_values, subset_size):
            ratio = float(counts.loc[list(chosen)].sum()) / total_rows
            record = (abs(ratio - target_ratio), len(chosen), tuple(chosen), ratio)
            if best is None or record < best:
                best = record
    assert best is not None
    chosen = list(best[2])
    return chosen, {
        "selection_method": "single_dataset_search",
        "candidate_values": [_format_value(value) for value in candidate_values],
        "test_ratio": float(best[3]),
        "objective": float(best[0]),
    }


def _bruteforce_best_subset(
    source_counts: pd.Series,
    target_counts: pd.Series,
    *,
    axis: str,
    target_ratio: float,
) -> tuple[list[Any], dict[str, Any]]:
    candidate_values = _shared_candidate_values(source_counts, target_counts, axis)
    if not candidate_values:
        raise ValueError(f"No shared values to build a holdout subset for axis {axis}")
    best: tuple[float, float, int, tuple[Any, ...], float, float] | None = None
    source_total = max(float(source_counts.sum()), 1.0)
    target_total = max(float(target_counts.sum()), 1.0)
    for subset_size in range(1, len(candidate_values) + 1):
        for chosen in itertools.combinations(candidate_values, subset_size):
            source_ratio = float(source_counts.loc[list(chosen)].sum()) / source_total
            target_ratio_observed = float(target_counts.loc[list(chosen)].sum()) / target_total
            record = (
                max(abs(source_ratio - target_ratio), abs(target_ratio_observed - target_ratio)),
                abs(source_ratio - target_ratio_observed),
                len(chosen),
                tuple(chosen),
                source_ratio,
                target_ratio_observed,
            )
            if best is None or record < best:
                best = record
    assert best is not None
    chosen = list(best[3])
    return chosen, {
        "selection_method": "bruteforce",
        "shared_values": [_format_value(value) for value in candidate_values],
        "source_test_ratio": float(best[4]),
        "target_test_ratio": float(best[5]),
        "objective": float(best[0]),
    }


def _beam_best_subset(
    source_counts: pd.Series,
    target_counts: pd.Series,
    *,
    axis: str,
    target_ratio: float,
    beam_size: int = 400,
) -> tuple[list[Any], dict[str, Any]]:
    candidate_values = _shared_candidate_values(source_counts, target_counts, axis)
    if not candidate_values:
        raise ValueError(f"No shared values to build a holdout subset for axis {axis}")
    source_total = max(float(source_counts.sum()), 1.0)
    target_total = max(float(target_counts.sum()), 1.0)
    beam: list[tuple[float, float, tuple[Any, ...]]] = [(0.0, 0.0, tuple())]
    mean_count = max(int(round(len(candidate_values) * target_ratio)), 1)
    for value in candidate_values:
        source_weight = float(source_counts.loc[value]) / source_total
        target_weight = float(target_counts.loc[value]) / target_total
        candidates: list[tuple[float, float, tuple[Any, ...]]] = []
        for source_ratio, target_ratio_observed, chosen in beam:
            candidates.append((source_ratio, target_ratio_observed, chosen))
            candidates.append((source_ratio + source_weight, target_ratio_observed + target_weight, chosen + (value,)))
        scored: list[tuple[float, float, int, int, float, float, tuple[Any, ...]]] = []
        seen: set[tuple[float, float, int]] = set()
        for source_ratio, target_ratio_observed, chosen in candidates:
            key = (round(source_ratio, 6), round(target_ratio_observed, 6), len(chosen))
            if key in seen:
                continue
            seen.add(key)
            scored.append(
                (
                    max(abs(source_ratio - target_ratio), abs(target_ratio_observed - target_ratio)),
                    abs(source_ratio - target_ratio_observed),
                    abs(len(chosen) - mean_count),
                    len(chosen),
                    source_ratio,
                    target_ratio_observed,
                    chosen,
                )
            )
        scored.sort()
        beam = [(source_ratio, target_ratio_observed, chosen) for _, _, _, _, source_ratio, target_ratio_observed, chosen in scored[:beam_size]]
    best = sorted(
        [
            (
                max(abs(source_ratio - target_ratio), abs(target_ratio_observed - target_ratio)),
                abs(source_ratio - target_ratio_observed),
                abs(len(chosen) - mean_count),
                len(chosen),
                source_ratio,
                target_ratio_observed,
                chosen,
            )
            for source_ratio, target_ratio_observed, chosen in beam
            if chosen
        ]
    )[0]
    return list(best[6]), {
        "selection_method": "beam_search",
        "shared_values": [_format_value(value) for value in candidate_values],
        "source_test_ratio": float(best[4]),
        "target_test_ratio": float(best[5]),
        "objective": float(best[0]),
    }


def _select_decoupled_holdout_values(frame: pd.DataFrame, *, policy: str, target_ratio: float) -> tuple[list[Any], dict[str, Any]]:
    if policy == "inter_threads_decoupled_v2":
        main_axis = "inter_threads"
        aux_axis = "num_threads"
        candidate_values = [
            value
            for value in _sorted_unique_values(frame[main_axis])
            if value in {int(item) for item in DEFAULTS["thread_decoupling_phase2"]["held_out_inter_threads"]}
        ]
        required_train_aux = set()
        fixed_holdout_values = list(candidate_values)
    elif policy == "num_threads_decoupled_v2":
        main_axis = "num_threads"
        aux_axis = "inter_threads"
        candidate_values = [
            value
            for value in _sorted_unique_values(frame[main_axis])
            if value in {int(item) for item in DEFAULTS["thread_decoupling_phase2"]["held_out_num_threads"]}
        ]
        required_train_aux = {
            int(value) for value in DEFAULTS["thread_decoupling_phase2"]["required_train_inter_threads"]
        }
        fixed_holdout_values = list(candidate_values)
    else:
        raise ValueError(f"Unsupported decoupled policy: {policy}")

    full_batch_count = max(int(frame["batch_size"].nunique()), 1)
    full_nip_count = max(int(frame["num_indices_per_lookup"].nunique()), 1)
    if fixed_holdout_values:
        test_mask = frame[main_axis].isin(fixed_holdout_values)
        train_frame = frame.loc[~test_mask].copy().reset_index(drop=True)
        test_frame = frame.loc[test_mask].copy().reset_index(drop=True)
        if train_frame.empty or test_frame.empty:
            raise ValueError(f"Fixed decoupled holdout for policy {policy} produced an empty split")
        aux_train_values = set(_sorted_unique_values(train_frame[aux_axis]))
        aux_test_values = set(_sorted_unique_values(test_frame[aux_axis]))
        if not aux_test_values.issubset(aux_train_values):
            raise ValueError(
                f"Fixed decoupled holdout for policy {policy} violates aux-axis overlap: "
                f"test {sorted(aux_test_values)} vs train {sorted(aux_train_values)}"
            )
        if required_train_aux and not required_train_aux.issubset({int(value) for value in aux_train_values}):
            raise ValueError(
                f"Fixed decoupled holdout for policy {policy} does not preserve required aux-axis train values: "
                f"{sorted(required_train_aux)}"
            )
        batch_coverage = float(test_frame["batch_size"].nunique()) / full_batch_count
        nip_coverage = float(test_frame["num_indices_per_lookup"].nunique()) / full_nip_count
        return list(fixed_holdout_values), {
            "selection_method": "fixed_decoupled_values",
            "main_axis": main_axis,
            "aux_axis": aux_axis,
            "main_axis_holdout_values": [_format_value(value) for value in fixed_holdout_values],
            "candidate_values": [_format_value(value) for value in candidate_values],
            "test_ratio": float(len(test_frame) / max(len(frame), 1)),
            "batch_coverage": batch_coverage,
            "nip_coverage": nip_coverage,
            "aux_axis_train_values": [_format_value(value) for value in sorted(aux_train_values)],
            "aux_axis_test_values": [_format_value(value) for value in sorted(aux_test_values)],
        }

    best: tuple[float, int, float, float, tuple[Any, ...], dict[str, Any]] | None = None
    for subset_size in range(1, len(candidate_values) + 1):
        for chosen in itertools.combinations(candidate_values, subset_size):
            if main_axis == "case_id":
                test_mask = frame["case_id"].astype(str).isin([str(value) for value in chosen])
            else:
                test_mask = frame[main_axis].isin(list(chosen))
            train_frame = frame.loc[~test_mask].copy().reset_index(drop=True)
            test_frame = frame.loc[test_mask].copy().reset_index(drop=True)
            if train_frame.empty or test_frame.empty:
                continue
            aux_train_values = set(_sorted_unique_values(train_frame[aux_axis]))
            aux_test_values = set(_sorted_unique_values(test_frame[aux_axis]))
            if not aux_test_values.issubset(aux_train_values):
                continue
            if required_train_aux and not required_train_aux.issubset({int(value) for value in aux_train_values}):
                continue
            test_ratio = float(len(test_frame) / max(len(frame), 1))
            batch_coverage = float(test_frame["batch_size"].nunique()) / full_batch_count
            nip_coverage = float(test_frame["num_indices_per_lookup"].nunique()) / full_nip_count
            payload = {
                "selection_method": "decoupled_search",
                "main_axis": main_axis,
                "aux_axis": aux_axis,
                "main_axis_holdout_values": [_format_value(value) for value in chosen],
                "candidate_values": [_format_value(value) for value in candidate_values],
                "test_ratio": test_ratio,
                "batch_coverage": batch_coverage,
                "nip_coverage": nip_coverage,
                "aux_axis_train_values": [_format_value(value) for value in sorted(aux_train_values)],
                "aux_axis_test_values": [_format_value(value) for value in sorted(aux_test_values)],
            }
            record = (
                abs(test_ratio - target_ratio),
                len(chosen),
                -batch_coverage,
                -nip_coverage,
                tuple(_format_value(value) for value in chosen),
                payload,
            )
            if best is None or record < best:
                best = record
    if best is None:
        raise ValueError(f"Could not find a decoupled holdout subset for policy {policy}")
    return list(best[4]), best[5]


def _select_holdout_values(
    *,
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame | None,
    policy: str,
    target_ratio: float,
    case_id: str,
) -> tuple[list[Any], dict[str, Any]]:
    if policy in PHASE2_POLICIES:
        return _select_decoupled_holdout_values(source_frame, policy=policy, target_ratio=target_ratio)
    if policy == JOINT_6D_POLICY:
        if target_frame is not None:
            raise ValueError(f"{JOINT_6D_POLICY} is source-only and cannot be built with target splits")
        return _select_joint_6d_holdout_values(source_frame, target_ratio=target_ratio)

    axis = policy
    if axis == "case_id":
        source_counts = _counts_by_value(source_frame, axis)
        if target_frame is None:
            source_cases = sorted(_format_value(value) for value in source_counts.index.tolist())
            if case_id not in {str(value) for value in source_cases}:
                raise ValueError(f"Requested case holdout {case_id} is not present in source dataset")
            source_ratio = float(source_counts.loc[case_id]) / max(float(source_counts.sum()), 1.0)
            return [case_id], {
                "selection_method": "fixed_case",
                "source_cases": [str(value) for value in source_cases],
                "source_test_ratio": source_ratio,
                "objective": float(abs(source_ratio - target_ratio)),
            }
        target_counts = _counts_by_value(target_frame, axis)
        shared_cases = sorted(set(source_counts.index).intersection(set(target_counts.index)))
        if case_id not in {str(value) for value in shared_cases}:
            raise ValueError(f"Requested case holdout {case_id} is not shared by source and target")
        source_ratio = float(source_counts.loc[case_id]) / max(float(source_counts.sum()), 1.0)
        target_ratio_observed = float(target_counts.loc[case_id]) / max(float(target_counts.sum()), 1.0)
        return [case_id], {
            "selection_method": "fixed_case",
            "shared_cases": [str(_format_value(value)) for value in shared_cases],
            "source_test_ratio": source_ratio,
            "target_test_ratio": target_ratio_observed,
            "objective": float(max(abs(source_ratio - target_ratio), abs(target_ratio_observed - target_ratio))),
        }

    source_counts = _counts_by_value(source_frame, axis)
    if target_frame is None:
        if axis == "batch_size":
            return _single_dataset_best_subset(source_counts, axis=axis, target_ratio=target_ratio)
        return _single_dataset_best_subset(source_counts, axis=axis, target_ratio=target_ratio)
    target_counts = _counts_by_value(target_frame, axis)
    if axis == "batch_size":
        return _beam_best_subset(source_counts, target_counts, axis=axis, target_ratio=target_ratio)
    return _bruteforce_best_subset(source_counts, target_counts, axis=axis, target_ratio=target_ratio)


def _write_dataset_split(
    *,
    dataset_name: str,
    input_dataset_root: Path,
    dataset_scope: str,
    phase_scope: str,
    policy: str,
    axis: str,
    main_axis: str,
    aux_axis: str | None,
    test_values: list[Any],
    frame: pd.DataFrame,
    output_root: Path,
    selection_stats: dict[str, Any],
    e2e_train_frame: pd.DataFrame | None = None,
    e2e_test_raw_frame: pd.DataFrame | None = None,
    e2e_split_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_manifest = _load_manifest(input_dataset_root)
    train_raw, test_raw = _apply_policy_split(frame, axis=main_axis, test_values=test_values)
    train_frame, test_frame, retained_stats = _retained_split(
        train_raw,
        test_raw,
        target_ratio=RETAINED_TARGET_TEST_RATIO,
        lower_bound=RETAINED_LOWER_BOUND,
        upper_bound=RETAINED_UPPER_BOUND,
    )
    train_raw_groups, test_raw_groups = _split_group_tables(train_raw, test_raw, main_axis)
    train_groups, test_groups = _split_group_tables(train_frame, test_frame, main_axis)
    static_thread_audit = frame.attrs.get("static_thread_audit")
    e2e_manifest: dict[str, Any] = {
        "available": False,
        "split_mode": "",
        "graph_scope": "",
        "main_axis": main_axis,
        "target_test_ratio": 0.0,
        "combo_train_ratio": 0.0,
        "combo_test_ratio": 0.0,
        "train_rows": 0,
        "train_combo_groups": 0,
        "test_raw_rows": 0,
        "test_raw_combo_groups": 0,
        "total_combo_count": 0,
        "train_combo_count": 0,
        "test_combo_count": 0,
        "test_case_ids": [],
        "main_axis_test_values": [],
        "small_dataset_divisor": 1,
        "small_dataset_seed": 0,
        "small_dataset_combo_fraction": 1.0,
        "small_dataset_train_combo_fraction": 1.0,
        "small_dataset_test_combo_fraction": 1.0,
        "full_combo_count_total": 0,
        "full_combo_count_train": 0,
        "full_combo_count_test": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    train_raw.to_csv(output_root / "train_raw.csv", index=False)
    test_raw.to_csv(output_root / "test_raw.csv", index=False)
    train_frame.to_csv(output_root / "train.csv", index=False)
    test_frame.to_csv(output_root / "test.csv", index=False)
    frame.to_csv(output_root / "dataset_full.csv", index=False)
    train_raw_groups.to_csv(output_root / "train_raw_case_combo_groups.csv", index=False)
    test_raw_groups.to_csv(output_root / "test_raw_case_combo_groups.csv", index=False)
    train_groups.to_csv(output_root / "train_case_combo_groups.csv", index=False)
    test_groups.to_csv(output_root / "test_case_combo_groups.csv", index=False)
    static_thread_audit_rows = 0
    static_thread_overrides = 0
    if isinstance(static_thread_audit, pd.DataFrame) and not static_thread_audit.empty:
        static_thread_audit.to_csv(output_root / "static_thread_audit.csv", index=False)
        static_thread_audit_rows = int(len(static_thread_audit))
        static_thread_overrides = int(
            static_thread_audit["num_threads_overridden"].astype(bool).sum()
            + static_thread_audit["inter_threads_overridden"].astype(bool).sum()
        )
    e2e_extra_frames: list[pd.DataFrame] = []
    if e2e_train_frame is not None and not e2e_train_frame.empty:
        e2e_train_groups = _e2e_case_combo_groups(e2e_train_frame)
        e2e_train_frame.to_csv(output_root / "e2e_train.csv", index=False)
        e2e_train_groups.to_csv(output_root / "e2e_train_case_combo_groups.csv", index=False)
        e2e_extra_frames.append(e2e_train_frame)
    else:
        e2e_train_groups = pd.DataFrame(columns=["source_name", "case_id", "combo", "row_count"])
    if e2e_test_raw_frame is not None and e2e_split_summary is not None and not e2e_test_raw_frame.empty:
        e2e_test_groups = _e2e_case_combo_groups(e2e_test_raw_frame)
        e2e_test_raw_frame.to_csv(output_root / "e2e_test_raw.csv", index=False)
        e2e_test_groups.to_csv(output_root / "e2e_test_raw_case_combo_groups.csv", index=False)
        e2e_extra_frames.append(e2e_test_raw_frame)
        e2e_manifest = {
            "available": True,
            "split_mode": str(e2e_split_summary.get("split_mode", "combo_stratified")),
            "graph_scope": str(e2e_split_summary.get("graph_scope", "")),
            "main_axis": main_axis,
            "target_test_ratio": float(e2e_split_summary.get("target_test_ratio", 0.0)),
            "combo_train_ratio": float(e2e_split_summary.get("combo_train_ratio", 0.0)),
            "combo_test_ratio": float(e2e_split_summary.get("combo_test_ratio", 0.0)),
            "train_rows": int(len(e2e_train_frame)) if e2e_train_frame is not None else 0,
            "train_combo_groups": int(len(e2e_train_groups)),
            "test_raw_rows": int(len(e2e_test_raw_frame)),
            "test_raw_combo_groups": int(len(e2e_test_groups)),
            "total_combo_count": int(e2e_split_summary.get("combo_count_total", 0)),
            "train_combo_count": int(e2e_split_summary.get("combo_count_train", 0)),
            "test_combo_count": int(e2e_split_summary.get("combo_count_test", 0)),
            "test_case_ids": sorted(e2e_test_raw_frame["case_id"].astype(str).unique().tolist()),
            "main_axis_test_values": [_format_value(value) for value in e2e_split_summary.get("main_axis_test_values", [])],
            "small_dataset_divisor": int(e2e_split_summary.get("small_dataset_divisor", 1)),
            "small_dataset_seed": int(e2e_split_summary.get("small_dataset_seed", 0)),
            "small_dataset_combo_fraction": float(e2e_split_summary.get("small_dataset_combo_fraction", 1.0)),
            "small_dataset_train_combo_fraction": float(
                e2e_split_summary.get("small_dataset_train_combo_fraction", 1.0)
            ),
            "small_dataset_test_combo_fraction": float(e2e_split_summary.get("small_dataset_test_combo_fraction", 1.0)),
            "full_combo_count_total": int(e2e_split_summary.get("full_combo_count_total", 0)),
            "full_combo_count_train": int(e2e_split_summary.get("full_combo_count_train", 0)),
            "full_combo_count_test": int(e2e_split_summary.get("full_combo_count_test", 0)),
        }
    if policy == JOINT_6D_POLICY:
        _write_joint_6d_tuple_audit(output_root, train_raw, test_raw)
    _write_small_e2e_audit(
        output_root,
        e2e_summary=e2e_manifest,
        e2e_train_groups=e2e_train_groups,
        e2e_test_groups=e2e_test_groups if "e2e_test_groups" in locals() else pd.DataFrame(),
    )
    snapshot_base_frame = (
        pd.DataFrame()
        if e2e_manifest["available"] and int(e2e_manifest.get("small_dataset_divisor", 1)) > 1
        else frame
    )
    pipeline_snapshot = materialize_split_pipeline_snapshot(
        split_root=output_root,
        input_dataset_root=input_dataset_root,
        frame=snapshot_base_frame,
        extra_frames=e2e_extra_frames,
    )

    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": int(len(train_frame)),
                "case_combo_groups": int(len(train_groups)),
                "case_count": int(train_frame["case_id"].nunique()),
                "ratio": float(len(train_frame) / max(len(train_frame) + len(test_frame), 1)),
            },
            {
                "split": "test",
                "rows": int(len(test_frame)),
                "case_combo_groups": int(len(test_groups)),
                "case_count": int(test_frame["case_id"].nunique()),
                "ratio": float(len(test_frame) / max(len(train_frame) + len(test_frame), 1)),
            },
            {
                "split": "train_raw",
                "rows": int(len(train_raw)),
                "case_combo_groups": int(len(train_raw_groups)),
                "case_count": int(train_raw["case_id"].nunique()),
                "ratio": float(len(train_raw) / max(len(frame), 1)),
            },
            {
                "split": "test_raw",
                "rows": int(len(test_raw)),
                "case_combo_groups": int(len(test_raw_groups)),
                "case_count": int(test_raw["case_id"].nunique()),
                "ratio": float(len(test_raw) / max(len(frame), 1)),
            },
        ]
    )
    if e2e_manifest["available"]:
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "split": "e2e_train",
                            "rows": int(e2e_manifest["train_rows"]),
                            "case_combo_groups": int(e2e_manifest["train_combo_groups"]),
                            "case_count": int(e2e_train_frame["case_id"].nunique()) if e2e_train_frame is not None else 0,
                            "ratio": float(e2e_manifest["combo_train_ratio"]),
                        },
                        {
                            "split": "e2e_test_raw",
                            "rows": int(e2e_manifest["test_raw_rows"]),
                            "case_combo_groups": int(e2e_manifest["test_raw_combo_groups"]),
                            "case_count": len(e2e_manifest["test_case_ids"]),
                            "ratio": float(e2e_manifest["combo_test_ratio"]),
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
    summary.to_csv(output_root / "split_summary.csv", index=False)

    aux_train_values = _sorted_unique_values(train_frame[aux_axis]) if aux_axis else []
    aux_test_values = _sorted_unique_values(test_frame[aux_axis]) if aux_axis else []
    manifest = {
        "dataset_name": dataset_name,
        "input_dataset_root": str(input_dataset_root),
        "dataset_scope": dataset_scope,
        "phase_scope": phase_scope,
        "evaluation_scope": "source_only" if dataset_scope == "source" else "mixed",
        "scheduler_logs_root": source_manifest.get("scheduler_logs_root"),
        "policy": policy,
        "axis": axis,
        "main_axis": main_axis,
        "aux_axis": aux_axis,
        "test_values": [_format_value(value) for value in test_values],
        "main_axis_holdout_values": [_format_value(value) for value in test_values],
        "aux_axis_train_values": [_format_value(value) for value in aux_train_values],
        "aux_axis_test_values": [_format_value(value) for value in aux_test_values],
        "train_case_ids": sorted(train_frame["case_id"].astype(str).unique().tolist()),
        "test_case_ids": sorted(test_frame["case_id"].astype(str).unique().tolist()),
        "raw_train_case_ids": sorted(train_raw["case_id"].astype(str).unique().tolist()),
        "raw_test_case_ids": sorted(test_raw["case_id"].astype(str).unique().tolist()),
        "total_rows": int(len(frame)),
        "raw_train_rows": int(len(train_raw)),
        "raw_test_rows": int(len(test_raw)),
        "raw_train_ratio": float(len(train_raw) / max(len(frame), 1)),
        "raw_test_ratio": float(len(test_raw) / max(len(frame), 1)),
        "retained_train_rows": int(len(train_frame)),
        "retained_test_rows": int(len(test_frame)),
        "retained_train_ratio": float(len(train_frame) / max(len(train_frame) + len(test_frame), 1)),
        "retained_test_ratio": float(len(test_frame) / max(len(train_frame) + len(test_frame), 1)),
        "train_case_combo_groups": int(len(train_groups)),
        "test_case_combo_groups": int(len(test_groups)),
        "train_raw_case_combo_groups": int(len(train_raw_groups)),
        "test_raw_case_combo_groups": int(len(test_raw_groups)),
        "sampling_policy": retained_stats["sampling_policy"],
        "selection_stats": selection_stats,
        "pipeline_metadata_available": bool(pipeline_snapshot.get("pipeline_metadata_available", False)),
        "pipeline_metadata_reason": pipeline_snapshot.get("pipeline_metadata_reason", ""),
        "pipeline_artifact_index_csv": pipeline_snapshot.get("pipeline_artifact_index_csv"),
        "pipeline_artifact_index_json": pipeline_snapshot.get("pipeline_artifact_index_json"),
        "pipeline_artifact_index_rows": int(pipeline_snapshot.get("pipeline_artifact_index_rows", 0)),
        "pipeline_combo_manifest_csv": pipeline_snapshot.get("pipeline_combo_manifest_csv"),
        "pipeline_combo_manifest_rows": int(pipeline_snapshot.get("pipeline_combo_manifest_rows", 0)),
        "static_thread_audit_csv": str((output_root / "static_thread_audit.csv").resolve())
        if static_thread_audit_rows
        else "",
        "static_thread_audit_rows": static_thread_audit_rows,
        "static_thread_overrides": static_thread_overrides,
        "e2e_test_raw_available": bool(e2e_manifest["available"]),
        "e2e_split_mode": e2e_manifest["split_mode"],
        "e2e_graph_scope": e2e_manifest.get("graph_scope", ""),
        "e2e_target_test_ratio": float(e2e_manifest["target_test_ratio"]),
        "e2e_main_axis": e2e_manifest["main_axis"],
        "e2e_main_axis_test_values": e2e_manifest["main_axis_test_values"],
        "e2e_train_rows": int(e2e_manifest["train_rows"]),
        "e2e_train_case_combo_groups": int(e2e_manifest["train_combo_groups"]),
        "e2e_train_combo_ratio": float(e2e_manifest["combo_train_ratio"]),
        "e2e_test_rows": int(e2e_manifest["test_raw_rows"]),
        "e2e_test_ratio": float(e2e_manifest["combo_test_ratio"]),
        "e2e_test_case_combo_groups": int(e2e_manifest["test_raw_combo_groups"]),
        "e2e_total_combo_count": int(e2e_manifest["total_combo_count"]),
        "e2e_train_combo_count": int(e2e_manifest["train_combo_count"]),
        "e2e_selected_combo_count": int(e2e_manifest["test_combo_count"]),
        "small_dataset_divisor": int(e2e_manifest.get("small_dataset_divisor", 1)),
        "small_dataset_seed": int(e2e_manifest.get("small_dataset_seed", 0)),
        "small_dataset_combo_fraction": float(e2e_manifest.get("small_dataset_combo_fraction", 1.0)),
        "small_dataset_train_combo_fraction": float(e2e_manifest.get("small_dataset_train_combo_fraction", 1.0)),
        "small_dataset_test_combo_fraction": float(e2e_manifest.get("small_dataset_test_combo_fraction", 1.0)),
        "e2e_full_combo_count": int(e2e_manifest.get("full_combo_count_total", 0)),
        "e2e_full_train_combo_count": int(e2e_manifest.get("full_combo_count_train", 0)),
        "e2e_full_test_combo_count": int(e2e_manifest.get("full_combo_count_test", 0)),
    }
    _json_dump(output_root / "split_manifest.json", manifest)
    return manifest


def _selected_policies(policy_arg: str) -> list[str]:
    if policy_arg == "all":
        return list(FORMAL_POLICIES)
    if policy_arg == "phase2":
        return list(PHASE2_POLICIES)
    return [str(policy_arg)]


def _aggregate_source_row(policy: str, source_manifest: dict[str, Any], source_phase_scope: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "dataset": "source",
        "rows": source_manifest["total_rows"],
        "raw_train_rows": source_manifest["raw_train_rows"],
        "raw_test_rows": source_manifest["raw_test_rows"],
        "raw_train_ratio": source_manifest["raw_train_ratio"],
        "raw_test_ratio": source_manifest["raw_test_ratio"],
        "retained_train_rows": source_manifest["retained_train_rows"],
        "retained_test_rows": source_manifest["retained_test_rows"],
        "retained_train_ratio": source_manifest["retained_train_ratio"],
        "retained_test_ratio": source_manifest["retained_test_ratio"],
        "e2e_train_rows": source_manifest["e2e_train_rows"],
        "e2e_train_ratio": source_manifest["e2e_train_combo_ratio"],
        "e2e_test_rows": source_manifest["e2e_test_rows"],
        "e2e_test_ratio": source_manifest["e2e_test_ratio"],
        "e2e_test_combo_groups": source_manifest["e2e_test_case_combo_groups"],
        "e2e_total_combo_count": source_manifest["e2e_total_combo_count"],
        "e2e_train_combo_count": source_manifest["e2e_train_combo_count"],
        "e2e_selected_combo_count": source_manifest["e2e_selected_combo_count"],
        "small_dataset_divisor": source_manifest["small_dataset_divisor"],
        "small_dataset_combo_fraction": source_manifest["small_dataset_combo_fraction"],
        "sampling_policy": source_manifest["sampling_policy"],
        "phase_scope": source_phase_scope,
    }


def _source_root_for_policy(policy: str) -> Path:
    if policy in PHASE2_POLICIES:
        return PHASE2_SOURCE_ROOT
    return SOURCE_RAW_ROOT


def _is_invalid_legacy_split_dir(split_dir: Path) -> bool:
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    phase_scope = str(manifest.get("phase_scope", "")).strip()
    test_cases = {str(value) for value in manifest.get("test_case_ids", [])}
    train_cases = {str(value) for value in manifest.get("train_case_ids", [])}
    if phase_scope in {"", "default"}:
        return True
    if LEGACY_INVALID_CASE_IDS.intersection(test_cases.union(train_cases)):
        return True
    return False


def _archive_invalid_legacy_source_splits(output_root: Path) -> None:
    INVALID_LEGACY_ROOT.mkdir(parents=True, exist_ok=True)
    for split_dir in sorted(output_root.glob(f"{SOURCE_DATASET_NAME}_unseen_*")):
        if not split_dir.is_dir():
            continue
        if not _is_invalid_legacy_split_dir(split_dir):
            continue
        destination = INVALID_LEGACY_ROOT / split_dir.name
        if destination.exists():
            suffix = 1
            while (INVALID_LEGACY_ROOT / f"{split_dir.name}_{suffix}").exists():
                suffix += 1
            destination = INVALID_LEGACY_ROOT / f"{split_dir.name}_{suffix}"
        shutil.move(str(split_dir), str(destination))


def main() -> None:
    args = _parse_args()
    requested_source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()
    output_root = Path(args.output_root).resolve()
    selected_policies = _selected_policies(str(args.policy))
    target_phase_scope = _resolve_phase_scope(target_root) if str(args.dataset_scope) in {"target", "both"} else "not_loaded"
    target_frame = None
    if str(args.dataset_scope) in {"target", "both"}:
        target_frame = _load_full_frame(target_root)
    source_frame_cache: dict[Path, pd.DataFrame] = {}
    source_phase_scope_cache: dict[Path, str] = {}

    if str(args.dataset_scope) in {"source", "both"}:
        _archive_invalid_legacy_source_splits(output_root)
    accepted_source_roots = {SOURCE_RAW_ROOT, PHASE2_SOURCE_ROOT, LEGACY_SOURCE_RAW_ROOT, LEGACY_PHASE2_SOURCE_ROOT}
    if str(args.dataset_scope) in {"source", "both"} and requested_source_root not in accepted_source_roots:
        raise ValueError(
            "v2 fixes source raw roots by policy; --source-root must be either "
            f"{SOURCE_RAW_ROOT} or {PHASE2_SOURCE_ROOT}, got {requested_source_root}"
        )
    if str(args.dataset_scope) == "target" and any(policy in PHASE2_POLICIES for policy in selected_policies):
        raise ValueError("Decoupled thread policies are source-only and cannot be built with --dataset-scope target")
    if str(args.dataset_scope) == "both" and any(policy in JOINT_POLICIES for policy in selected_policies):
        raise ValueError(f"{JOINT_6D_POLICY} source and target splits must be built separately")

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_manifest: dict[str, Any] = {
        "requested_source_root": str(requested_source_root),
        "legacy_source_root": str(SOURCE_RAW_ROOT),
        "legacy_calibrator_source_root": str(LEGACY_SOURCE_RAW_ROOT),
        "phase2_source_root": str(PHASE2_SOURCE_ROOT),
        "legacy_calibrator_phase2_source_root": str(LEGACY_PHASE2_SOURCE_ROOT),
        "target_root": str(target_root),
        "output_root": str(output_root),
        "dataset_scope": str(args.dataset_scope),
        "target_test_ratio": float(args.target_test_ratio),
        "case_id": str(args.case_id),
        "target_phase_scope": target_phase_scope,
        "policies": {},
    }

    for policy in selected_policies:
        source_root = _source_root_for_policy(policy)
        if _path_is_under(source_root, INVALID_LEGACY_ROOT):
            raise ValueError(f"Source root cannot point into archived invalid legacy splits: {source_root}")
        if policy == JOINT_6D_POLICY and str(args.dataset_scope) == "source":
            source_phase_scope = source_phase_scope_cache.get(source_root)
            if source_phase_scope is None:
                source_phase_scope = _resolve_phase_scope(source_root)
                source_phase_scope_cache[source_root] = source_phase_scope
            if source_phase_scope != SOURCE_SCHEDULER_SCOPE:
                raise ValueError(
                    f"Joint source-only policy {policy} requires phase_scope={SOURCE_SCHEDULER_SCOPE}, "
                    f"got {source_phase_scope} from {source_root}"
                )
            if source_root != SOURCE_RAW_ROOT:
                raise ValueError(f"Joint source-only policy {policy} must use {SOURCE_RAW_ROOT}, got {source_root}")
            main_axis = JOINT_6D_AXIS
            aux_axis = None
            source_out = output_root / f"{SOURCE_DATASET_NAME}_unseen_{policy}"
            e2e_train_frame, e2e_test_raw_frame, e2e_split_summary = build_e2e_combo_split_frames(
                source_root,
                base_frame=pd.DataFrame(),
                policy=policy,
                main_axis=main_axis,
                target_test_ratio=float(args.target_test_ratio),
                max_workers=int(args.e2e_workers),
                small_divisor=int(args.e2e_small_divisor),
                small_seed=int(args.e2e_small_seed),
            )
            if e2e_train_frame.empty or e2e_test_raw_frame.empty:
                raise RuntimeError(f"Joint E2E split generation failed for {policy}: {e2e_split_summary}")
            e2e_train_frame = backfill_joint_6d_static_columns(e2e_train_frame, source_root)
            e2e_test_raw_frame = backfill_joint_6d_static_columns(e2e_test_raw_frame, source_root)
            e2e_train_frame = attach_joint_6d_tuple_key(e2e_train_frame)
            e2e_test_raw_frame = attach_joint_6d_tuple_key(e2e_test_raw_frame)
            split_frame = pd.concat([e2e_train_frame, e2e_test_raw_frame], ignore_index=True, sort=False)
            split_frame = _attach_static_thread_columns(split_frame, dataset_root=source_root)
            static_thread_audit = split_frame.attrs.get("static_thread_audit")
            split_frame = attach_joint_6d_tuple_key(split_frame)
            if isinstance(static_thread_audit, pd.DataFrame):
                split_frame.attrs["static_thread_audit"] = static_thread_audit
            test_values = sorted(e2e_test_raw_frame[JOINT_6D_AXIS].astype(str).unique().tolist())
            selection_stats = {
                "selection_method": "joint_6d_tuple_e2e_small_holdout",
                "main_axis": JOINT_6D_AXIS,
                "joint_columns": list(JOINT_6D_COLUMNS),
                "source_test_ratio": float(e2e_split_summary.get("combo_test_ratio", 0.0)),
                "test_ratio": float(e2e_split_summary.get("combo_test_ratio", 0.0)),
                "full_combo_count_total": int(e2e_split_summary.get("full_combo_count_total", 0)),
                "full_combo_count_train": int(e2e_split_summary.get("full_combo_count_train", 0)),
                "full_combo_count_test": int(e2e_split_summary.get("full_combo_count_test", 0)),
                "small_dataset_divisor": int(e2e_split_summary.get("small_dataset_divisor", 1)),
                "small_dataset_seed": int(e2e_split_summary.get("small_dataset_seed", 0)),
                "small_dataset_combo_fraction": float(e2e_split_summary.get("small_dataset_combo_fraction", 1.0)),
            }
            source_manifest = _write_dataset_split(
                dataset_name=SOURCE_DATASET_NAME,
                input_dataset_root=source_root,
                dataset_scope=str(args.dataset_scope),
                phase_scope=source_phase_scope,
                policy=policy,
                axis=policy,
                main_axis=main_axis,
                aux_axis=aux_axis,
                test_values=test_values,
                frame=split_frame,
                output_root=source_out,
                selection_stats=selection_stats,
                e2e_train_frame=e2e_train_frame,
                e2e_test_raw_frame=e2e_test_raw_frame,
                e2e_split_summary=e2e_split_summary,
            )
            policy_manifest: dict[str, Any] = {
                "source_input_dataset_root": str(source_root),
                "source_phase_scope": source_phase_scope,
                "test_values": [_format_value(value) for value in test_values],
                "selection_stats": selection_stats,
                "source_output_root": str(source_out),
            }
            aggregate_manifest["policies"][policy] = policy_manifest
            aggregate_rows.append(_aggregate_source_row(policy, source_manifest, source_phase_scope))
            continue
        if policy == JOINT_6D_POLICY and str(args.dataset_scope) == "target":
            if target_frame is None:
                raise AssertionError("target_frame must be loaded when building target joint_6d_tuple splits")
            main_axis = JOINT_6D_AXIS
            aux_axis = None
            target_joint_frame = target_frame
            if JOINT_6D_AXIS not in target_joint_frame.columns:
                target_joint_frame = attach_joint_6d_tuple_key(target_joint_frame)
                target_frame = target_joint_frame
            test_values, selection_stats = _select_joint_6d_holdout_values(
                target_joint_frame,
                target_ratio=float(args.target_test_ratio),
            )
            target_out = output_root / f"{TARGET_DATASET_NAME}_unseen_{policy}"
            target_manifest = _write_dataset_split(
                dataset_name=TARGET_DATASET_NAME,
                input_dataset_root=target_root,
                dataset_scope=str(args.dataset_scope),
                phase_scope=target_phase_scope,
                policy=policy,
                axis=policy,
                main_axis=main_axis,
                aux_axis=aux_axis,
                test_values=test_values,
                frame=target_joint_frame,
                output_root=target_out,
                selection_stats=selection_stats,
            )
            policy_manifest = {
                "target_input_dataset_root": str(target_root),
                "target_phase_scope": target_phase_scope,
                "test_values": [_format_value(value) for value in test_values],
                "selection_stats": selection_stats,
                "target_output_root": str(target_out),
            }
            aggregate_manifest["policies"][policy] = policy_manifest
            aggregate_rows.append(
                {
                    "policy": policy,
                    "dataset": "target",
                    "rows": target_manifest["total_rows"],
                    "raw_train_rows": target_manifest["raw_train_rows"],
                    "raw_test_rows": target_manifest["raw_test_rows"],
                    "raw_train_ratio": target_manifest["raw_train_ratio"],
                    "raw_test_ratio": target_manifest["raw_test_ratio"],
                    "retained_train_rows": target_manifest["retained_train_rows"],
                    "retained_test_rows": target_manifest["retained_test_rows"],
                    "retained_train_ratio": target_manifest["retained_train_ratio"],
                    "retained_test_ratio": target_manifest["retained_test_ratio"],
                    "sampling_policy": target_manifest["sampling_policy"],
                    "phase_scope": target_phase_scope,
                }
            )
            continue
        if source_root not in source_frame_cache:
            source_frame_cache[source_root] = _load_full_frame(source_root)
            source_phase_scope_cache[source_root] = _resolve_phase_scope(source_root)
        source_frame = source_frame_cache[source_root]
        source_phase_scope = source_phase_scope_cache[source_root]
        if policy in LEGACY_POLICIES or policy in JOINT_POLICIES:
            if source_phase_scope != SOURCE_SCHEDULER_SCOPE:
                raise ValueError(
                    f"General source-only policy {policy} requires phase_scope={SOURCE_SCHEDULER_SCOPE}, "
                    f"got {source_phase_scope} from {source_root}"
                )
            if source_root != SOURCE_RAW_ROOT:
                raise ValueError(
                    f"General source-only policy {policy} must use scheduler-backed source root {SOURCE_RAW_ROOT}, got {source_root}"
                )
        if policy in JOINT_POLICIES and JOINT_6D_AXIS not in source_frame.columns:
            source_frame = attach_joint_6d_tuple_key(source_frame)
            source_frame_cache[source_root] = source_frame
        if policy in PHASE2_POLICIES:
            if source_phase_scope != PHASE2_SCOPE:
                raise ValueError(
                    f"Phase2 decoupled policy {policy} requires phase_scope={PHASE2_SCOPE}, "
                    f"got {source_phase_scope} from {source_root}"
                )
            if source_root != PHASE2_SOURCE_ROOT:
                raise ValueError(f"Phase2 decoupled policy {policy} must use {PHASE2_SOURCE_ROOT}, got {source_root}")
        if policy in PHASE2_POLICIES:
            if policy == "inter_threads_decoupled_v2":
                main_axis = "inter_threads"
                aux_axis = "num_threads"
            else:
                main_axis = "num_threads"
                aux_axis = "inter_threads"
        elif policy == JOINT_6D_POLICY:
            main_axis = JOINT_6D_AXIS
            aux_axis = None
        else:
            main_axis = policy
            aux_axis = None

        test_values, selection_stats = _select_holdout_values(
            source_frame=source_frame,
            target_frame=target_frame if str(args.dataset_scope) == "both" else None,
            policy=policy,
            target_ratio=float(args.target_test_ratio),
            case_id=str(args.case_id),
        )
        policy_manifest: dict[str, Any] = {
            "source_input_dataset_root": str(source_root),
            "source_phase_scope": source_phase_scope,
            "test_values": [_format_value(value) for value in test_values],
            "selection_stats": selection_stats,
        }
        if str(args.dataset_scope) in {"source", "both"}:
            source_out = output_root / f"{SOURCE_DATASET_NAME}_unseen_{policy}"
            e2e_train_frame, e2e_test_raw_frame, e2e_split_summary = build_e2e_combo_split_frames(
                source_root,
                base_frame=source_frame,
                policy=policy,
                main_axis=main_axis,
                target_test_ratio=float(args.target_test_ratio),
                max_workers=int(args.e2e_workers),
                small_divisor=int(args.e2e_small_divisor),
                small_seed=int(args.e2e_small_seed),
            )
            source_manifest = _write_dataset_split(
                dataset_name=SOURCE_DATASET_NAME,
                input_dataset_root=source_root,
                dataset_scope=str(args.dataset_scope),
                phase_scope=source_phase_scope,
                policy=policy,
                axis=policy,
                main_axis=main_axis,
                aux_axis=aux_axis,
                test_values=test_values,
                frame=source_frame,
                output_root=source_out,
                selection_stats=selection_stats,
                e2e_train_frame=e2e_train_frame if not e2e_train_frame.empty else None,
                e2e_test_raw_frame=e2e_test_raw_frame if not e2e_test_raw_frame.empty else None,
                e2e_split_summary=e2e_split_summary if e2e_split_summary.get("available") else None,
            )
            policy_manifest["source_output_root"] = str(source_out)
            aggregate_rows.append(
                {
                    "policy": policy,
                    "dataset": "source",
                    "rows": source_manifest["total_rows"],
                    "raw_train_rows": source_manifest["raw_train_rows"],
                    "raw_test_rows": source_manifest["raw_test_rows"],
                    "raw_train_ratio": source_manifest["raw_train_ratio"],
                    "raw_test_ratio": source_manifest["raw_test_ratio"],
                    "retained_train_rows": source_manifest["retained_train_rows"],
                    "retained_test_rows": source_manifest["retained_test_rows"],
                    "retained_train_ratio": source_manifest["retained_train_ratio"],
                    "retained_test_ratio": source_manifest["retained_test_ratio"],
                    "e2e_train_rows": source_manifest["e2e_train_rows"],
                    "e2e_train_ratio": source_manifest["e2e_train_combo_ratio"],
                    "e2e_test_rows": source_manifest["e2e_test_rows"],
                    "e2e_test_ratio": source_manifest["e2e_test_ratio"],
                    "e2e_test_combo_groups": source_manifest["e2e_test_case_combo_groups"],
                    "e2e_total_combo_count": source_manifest["e2e_total_combo_count"],
                    "e2e_train_combo_count": source_manifest["e2e_train_combo_count"],
                    "e2e_selected_combo_count": source_manifest["e2e_selected_combo_count"],
                    "small_dataset_divisor": source_manifest["small_dataset_divisor"],
                    "small_dataset_combo_fraction": source_manifest["small_dataset_combo_fraction"],
                    "sampling_policy": source_manifest["sampling_policy"],
                    "phase_scope": source_phase_scope,
                }
            )

        if str(args.dataset_scope) in {"target", "both"}:
            if target_frame is None:
                raise AssertionError("target_frame must be loaded when dataset_scope includes target")
            target_out = output_root / f"{TARGET_DATASET_NAME}_unseen_{policy}"
            target_manifest = _write_dataset_split(
                dataset_name=TARGET_DATASET_NAME,
                input_dataset_root=target_root,
                dataset_scope=str(args.dataset_scope),
                phase_scope=target_phase_scope,
                policy=policy,
                axis=policy,
                main_axis=main_axis,
                aux_axis=aux_axis,
                test_values=test_values,
                frame=target_frame,
                output_root=target_out,
                selection_stats=selection_stats,
            )
            policy_manifest["target_output_root"] = str(target_out)
            aggregate_rows.append(
                {
                    "policy": policy,
                    "dataset": "target",
                    "rows": target_manifest["total_rows"],
                    "raw_train_rows": target_manifest["raw_train_rows"],
                    "raw_test_rows": target_manifest["raw_test_rows"],
                    "raw_train_ratio": target_manifest["raw_train_ratio"],
                    "raw_test_ratio": target_manifest["raw_test_ratio"],
                    "retained_train_rows": target_manifest["retained_train_rows"],
                    "retained_test_rows": target_manifest["retained_test_rows"],
                    "retained_train_ratio": target_manifest["retained_train_ratio"],
                    "retained_test_ratio": target_manifest["retained_test_ratio"],
                    "sampling_policy": target_manifest["sampling_policy"],
                    "phase_scope": target_phase_scope,
                }
            )
        aggregate_manifest["policies"][policy] = policy_manifest

    pd.DataFrame(aggregate_rows).sort_values(["policy", "dataset"]).to_csv(output_root / "split_summary.csv", index=False)
    _json_dump(output_root / "split_manifest.json", aggregate_manifest)


if __name__ == "__main__":
    main()
