from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from cpu_perf_model.artifact_loader import (
        format_combo,
        load_op_shapes_frame,
        load_timeline_frame,
    )
    from cpu_perf_model.dataset_builder import build_rows_for_feature_csv, resolve_source_entries
    from cpu_perf_model.feature_contract import ANALYTICAL_RESIDUAL_TARGET_COLUMN, TARGET_COLUMN
    from cpu_perf_model.feature_engineering import normalize_node_name
    from cpu_perf_model.graph_contract import ComboArtifactPaths, ComboSpec
    from cpu_perf_model.schedule_engine import (
        analyze_embedding_execution,
        build_op_graph,
        compute_mean_batch_span,
        schedule_combo,
    )
except ImportError:
    from cpu_perf_model.artifact_loader import (  # type: ignore
        format_combo,
        load_op_shapes_frame,
        load_timeline_frame,
    )
    from cpu_perf_model.dataset_builder import (  # type: ignore
        build_rows_for_feature_csv,
        resolve_source_entries,
    )
    from cpu_perf_model.feature_contract import (  # type: ignore
        ANALYTICAL_RESIDUAL_TARGET_COLUMN,
        TARGET_COLUMN,
    )
    from cpu_perf_model.feature_engineering import normalize_node_name  # type: ignore
    from cpu_perf_model.graph_contract import ComboArtifactPaths, ComboSpec  # type: ignore
    from cpu_perf_model.schedule_engine import (  # type: ignore
        analyze_embedding_execution,
        build_op_graph,
        compute_mean_batch_span,
        schedule_combo,
    )


PIPELINE_INDEX_CSV_NAME = "pipeline_artifact_index.csv"
PIPELINE_INDEX_JSON_NAME = "pipeline_artifact_index.json"
PIPELINE_COMBO_MANIFEST_CSV_NAME = "pipeline_combo_manifest.csv"
E2E_FULL_DATASET_CSV_NAME = "e2e_combo_full_dataset_full.csv"
E2E_FULL_GROUPS_CSV_NAME = "e2e_combo_full_case_combo_groups.csv"
E2E_FULL_MANIFEST_JSON_NAME = "e2e_combo_full_manifest.json"
E2E_PROFILE_INSTABILITY_METRIC = "last2_range_ratio"
E2E_PROFILE_INSTABILITY_THRESHOLD = 0.20
E2E_GRAPH_SCOPE_OBSERVABLE = "observable_profiled_nodes"
E2E_MATERIALIZE_AUTO_WORKER_CAP = 32
JOINT_6D_POLICY = "joint_6d_tuple"
JOINT_6D_AXIS = "joint_6d_tuple_key"
JOINT_6D_COLUMNS = (
    "batch_size",
    "num_indices_per_lookup",
    "arch_embedding_size",
    "arch_mlp_bot",
    "inter_threads",
    "num_threads",
)


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _as_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _joint_value_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "na"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    text = str(value).strip()
    return text if text else "na"


def joint_6d_tuple_key(
    *,
    batch_size: Any,
    num_indices_per_lookup: Any,
    arch_embedding_size: Any,
    arch_mlp_bot: Any,
    inter_threads: Any,
    num_threads: Any,
) -> str:
    return "|".join(
        [
            f"bs={_joint_value_text(batch_size)}",
            f"nip={_joint_value_text(num_indices_per_lookup)}",
            f"emb={_joint_value_text(arch_embedding_size)}",
            f"bot={_joint_value_text(arch_mlp_bot)}",
            f"inter={_joint_value_text(inter_threads)}",
            f"num={_joint_value_text(num_threads)}",
        ]
    )


def attach_joint_6d_tuple_key(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in JOINT_6D_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns for {JOINT_6D_AXIS}: {missing}")
    work = frame.copy()
    work[JOINT_6D_AXIS] = [
        joint_6d_tuple_key(
            batch_size=row.batch_size,
            num_indices_per_lookup=row.num_indices_per_lookup,
            arch_embedding_size=row.arch_embedding_size,
            arch_mlp_bot=row.arch_mlp_bot,
            inter_threads=row.inter_threads,
            num_threads=row.num_threads,
        )
        for row in work[
            [
                "batch_size",
                "num_indices_per_lookup",
                "arch_embedding_size",
                "arch_mlp_bot",
                "inter_threads",
                "num_threads",
            ]
        ].itertuples(index=False)
    ]
    return work


def backfill_joint_6d_static_columns(frame: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    static_map = _source_static_map(Path(dataset_root).resolve())

    def _mapped(source_name: Any, column: str) -> Any:
        return static_map.get(str(source_name), {}).get(column)

    for column in ["arch_embedding_size", "arch_mlp_bot", "arch_mlp_top"]:
        if column not in work.columns:
            work[column] = pd.NA
        mapped = work["source_name"].map(lambda value, c=column: _mapped(value, c))
        text = work[column].astype("string")
        missing = work[column].isna() | text.str.strip().str.lower().isin({"", "unknown", "nan", "none"})
        work.loc[missing, column] = mapped.loc[missing]

    for column in ["inter_threads", "num_threads"]:
        if column not in work.columns:
            work[column] = pd.NA
        mapped = work["source_name"].map(lambda value, c=column: _mapped(value, c))
        numeric = pd.to_numeric(work[column], errors="coerce")
        mapped_numeric = pd.to_numeric(mapped, errors="coerce")
        work[column] = mapped_numeric.where(mapped_numeric.notna(), numeric)

    if "batch_size" not in work.columns or "num_indices_per_lookup" not in work.columns:
        parsed = work["combo"].map(_parse_combo)
        if "batch_size" not in work.columns:
            work["batch_size"] = parsed.map(lambda value: value[0])
        if "num_indices_per_lookup" not in work.columns:
            work["num_indices_per_lookup"] = parsed.map(lambda value: value[1])
    return work


def _as_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _path_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return str(Path(text).resolve())


PIPELINE_ARTIFACT_PATH_COLUMNS = (
    "sweep_dir",
    "shape_csv",
    "profile_dir",
    "timeline_csv",
    "effective_onnx_path",
    "log_dir",
)


def _resolve_pipeline_artifact_path(value: Any, *, base_root: Path) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = base_root / path
    return str(path.resolve())


def _refresh_pipeline_artifact_exists(index_df: pd.DataFrame) -> pd.DataFrame:
    work = index_df.copy()
    if "shape_csv" in work.columns:
        work["shape_csv_exists"] = work["shape_csv"].map(lambda value: bool(str(value).strip()) and Path(str(value)).exists())
    if "timeline_csv" in work.columns:
        work["timeline_csv_exists"] = work["timeline_csv"].map(
            lambda value: bool(str(value).strip()) and Path(str(value)).exists()
        )
    if "effective_onnx_path" in work.columns:
        work["effective_onnx_exists"] = work["effective_onnx_path"].map(
            lambda value: bool(str(value).strip()) and Path(str(value)).exists()
        )
    return work


def _resolve_pipeline_artifact_index(index_df: pd.DataFrame, *, base_root: Path) -> pd.DataFrame:
    if index_df.empty:
        return index_df.copy()
    work = index_df.copy()
    for column in PIPELINE_ARTIFACT_PATH_COLUMNS:
        if column in work.columns:
            work[column] = work[column].map(lambda value, root=base_root: _resolve_pipeline_artifact_path(value, base_root=root))
    return _refresh_pipeline_artifact_exists(work)


def pipeline_index_key(source_name: Any, combo: Any) -> str:
    return f"{str(source_name)}::{str(combo)}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _safe_ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").abs().clip(lower=1e-9)
    return pd.to_numeric(numerator, errors="coerce") / denom


def _timeline_alias_target_name(node_name: Any) -> str:
    normalized = normalize_node_name(node_name)
    if not normalized:
        return ""
    if normalized.startswith("fused "):
        return normalized[len("fused ") :]
    if normalized == "/MatMul/MatmulTransposeFusion/":
        return "/MatMul"
    return normalized


def _profile_timeline_path_for_combo(case_meta: dict[str, Any], combo: str) -> Path:
    if "profile_root" in case_meta:
        return Path(case_meta["profile_root"]) / combo / "branch_parallel_op_timeline.csv"
    return Path(case_meta["sweep_dir"]) / "onnx_profiles" / combo / "branch_parallel_op_timeline.csv"


def _load_observable_profile_label_stats(
    case_meta: dict[str, Any],
    combo: str,
    *,
    drop_first_profile_batch: bool,
) -> pd.DataFrame:
    timeline_path = _profile_timeline_path_for_combo(case_meta, combo)
    if not timeline_path.exists():
        return pd.DataFrame(columns=["node_name_normalized"])

    timeline_df = pd.read_csv(timeline_path, usecols=["batch_idx", "node_name", "dur_us"])
    timeline_df["batch_idx"] = pd.to_numeric(timeline_df["batch_idx"], errors="coerce")
    timeline_df["dur_us"] = pd.to_numeric(timeline_df["dur_us"], errors="coerce")
    timeline_df = timeline_df[timeline_df["batch_idx"].notna() & timeline_df["dur_us"].notna()].copy()
    if timeline_df.empty:
        return pd.DataFrame(columns=["node_name_normalized"])

    timeline_df["batch_idx"] = timeline_df["batch_idx"].astype(int)
    timeline_df["node_name_normalized"] = timeline_df["node_name"].map(_timeline_alias_target_name)
    timeline_df = timeline_df[timeline_df["node_name_normalized"] != ""].copy()
    if timeline_df.empty:
        return pd.DataFrame(columns=["node_name_normalized"])

    by_batch = (
        timeline_df.groupby(["node_name_normalized", "batch_idx"], dropna=False)["dur_us"]
        .sum()
        .reset_index()
    )
    all_batches = sorted(by_batch["batch_idx"].dropna().unique().tolist())
    dropped_batches = all_batches[:1] if drop_first_profile_batch else []
    kept_batches = [batch for batch in all_batches if batch not in dropped_batches]

    all_stats = by_batch.groupby("node_name_normalized", dropna=False)["dur_us"].agg(
        profile_batch_count_total="count",
        profile_label_all_batch_mean_us="mean",
    )

    kept_stats = (
        by_batch[by_batch["batch_idx"].isin(kept_batches)]
        .groupby("node_name_normalized", dropna=False)["dur_us"]
        .agg(
            profile_batch_count_kept="count",
            profile_label_kept_batch_mean_us="mean",
            profile_label_kept_batch_median_us="median",
        )
    )

    pivot = by_batch.pivot_table(
        index="node_name_normalized",
        columns="batch_idx",
        values="dur_us",
        aggfunc="first",
    ).sort_index(axis=1)

    stats = all_stats.join(kept_stats, how="left").reset_index()
    stats["profile_batch_count_kept"] = pd.to_numeric(
        stats["profile_batch_count_kept"],
        errors="coerce",
    ).fillna(0).astype(int)
    stats["profile_dropped_batch_indices"] = ",".join(str(batch) for batch in dropped_batches)
    stats["profile_kept_batch_indices"] = ",".join(str(batch) for batch in kept_batches)

    if kept_batches:
        kept_values = pivot[kept_batches].apply(pd.to_numeric, errors="coerce")
        kept_mean = kept_values.mean(axis=1)
        kept_std = kept_values.std(axis=1, ddof=0)
        kept_min = kept_values.min(axis=1)
        kept_max = kept_values.max(axis=1)
        kept_metrics = pd.DataFrame(index=pivot.index)
        kept_metrics.index.name = "node_name_normalized"
        kept_metrics["profile_kept_abs_diff_us"] = kept_max - kept_min
        kept_metrics["profile_kept_range_ratio"] = _safe_ratio_series(
            kept_metrics["profile_kept_abs_diff_us"],
            kept_mean,
        )
        kept_metrics["profile_kept_cv"] = _safe_ratio_series(kept_std, kept_mean)
        stats = stats.merge(kept_metrics.reset_index(), on="node_name_normalized", how="left")
    else:
        stats["profile_kept_abs_diff_us"] = np.nan
        stats["profile_kept_range_ratio"] = np.nan
        stats["profile_kept_cv"] = np.nan

    if len(kept_batches) >= 2:
        last2_batches = kept_batches[-2:]
        last2_left = pd.to_numeric(pivot[last2_batches[0]], errors="coerce")
        last2_right = pd.to_numeric(pivot[last2_batches[1]], errors="coerce")
        pair_mean = (last2_left + last2_right) / 2.0
        pair_std = pd.concat([last2_left, last2_right], axis=1).std(axis=1, ddof=0)
        last2_metrics = pd.DataFrame(index=pivot.index)
        last2_metrics.index.name = "node_name_normalized"
        last2_metrics["profile_last2_abs_diff_us"] = (last2_left - last2_right).abs()
        last2_metrics["profile_last2_range_ratio"] = _safe_ratio_series(
            last2_metrics["profile_last2_abs_diff_us"],
            pair_mean,
        )
        last2_metrics["profile_last2_cv"] = _safe_ratio_series(pair_std, pair_mean)
        stats = stats.merge(last2_metrics.reset_index(), on="node_name_normalized", how="left")
    else:
        stats["profile_last2_abs_diff_us"] = np.nan
        stats["profile_last2_range_ratio"] = np.nan
        stats["profile_last2_cv"] = np.nan

    return stats


def _build_rows_for_e2e_observable_feature_csv(
    feature_csv: Path,
    case_meta: dict[str, Any],
    *,
    hardware_profile_path: Path | None,
    drop_first_profile_batch: bool,
) -> pd.DataFrame:
    frame = build_rows_for_feature_csv(
        feature_csv,
        case_meta,
        hardware_profile_path=hardware_profile_path,
        drop_first_profile_batch=drop_first_profile_batch,
    )
    if frame.empty:
        return frame

    combo = str(feature_csv.stem)
    observable_stats = _load_observable_profile_label_stats(
        case_meta,
        combo,
        drop_first_profile_batch=drop_first_profile_batch,
    )
    if observable_stats.empty:
        return frame.head(0).copy()

    observable_columns = [
        "node_name_normalized",
        "profile_batch_count_total",
        "profile_label_all_batch_mean_us",
        "profile_batch_count_kept",
        "profile_label_kept_batch_mean_us",
        "profile_label_kept_batch_median_us",
        "profile_dropped_batch_indices",
        "profile_kept_batch_indices",
        "profile_kept_abs_diff_us",
        "profile_kept_range_ratio",
        "profile_kept_cv",
        "profile_last2_abs_diff_us",
        "profile_last2_range_ratio",
        "profile_last2_cv",
    ]
    frame = frame.drop(columns=[col for col in observable_columns if col in frame.columns and col != "node_name_normalized"])
    frame = frame.merge(observable_stats[observable_columns], on="node_name_normalized", how="left")
    frame[TARGET_COLUMN] = pd.to_numeric(frame.get("profile_label_kept_batch_mean_us"), errors="coerce")
    ana_base_us = pd.to_numeric(frame.get("ana_base_us"), errors="coerce").fillna(1e-3).clip(lower=1e-3)
    frame[ANALYTICAL_RESIDUAL_TARGET_COLUMN] = np.log(frame[TARGET_COLUMN].clip(lower=1e-9) / ana_base_us)
    frame["profile_instability_metric_name"] = E2E_PROFILE_INSTABILITY_METRIC
    frame["profile_instability_value"] = pd.to_numeric(
        frame.get(f"profile_{E2E_PROFILE_INSTABILITY_METRIC}", pd.Series(pd.NA, index=frame.index)),
        errors="coerce",
    )
    frame["profile_is_stable"] = (
        frame["profile_instability_value"].isna()
        | (frame["profile_instability_value"] <= float(E2E_PROFILE_INSTABILITY_THRESHOLD))
    )
    frame["e2e_graph_scope"] = E2E_GRAPH_SCOPE_OBSERVABLE
    frame = frame.loc[frame[TARGET_COLUMN].notna() & (frame[TARGET_COLUMN] > 0.0)].copy()
    return frame.reset_index(drop=True)


def _stable_truth_mask(frame: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(frame.get(TARGET_COLUMN, pd.Series(pd.NA, index=frame.index)), errors="coerce")
    stable = pd.to_numeric(frame.get("profile_is_stable", pd.Series(1.0, index=frame.index)), errors="coerce").fillna(1.0)
    return target.notna() & (target > 0.0) & stable.gt(0.5)


def _group_case_combo_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_name", "case_id", "combo", "row_count"])
    work = frame[["source_name", "case_id", "combo", "row_uid"]].copy()
    grouped = (
        work.groupby(["source_name", "case_id", "combo"], as_index=False)
        .agg(row_count=("row_uid", "count"))
        .sort_values(["source_name", "combo"], kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def _resolve_source_hw_config(dataset_root: Path) -> Path | None:
    manifest = _read_json(dataset_root / "manifest.json")
    path_text = str(manifest.get("source_hw_config", "")).strip()
    if not path_text:
        return None
    path = Path(path_text).resolve()
    if not path.exists():
        return None
    return path


def _resolve_e2e_materialize_workers(max_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if max_workers is None:
        return 1
    requested = int(max_workers)
    if requested < 0:
        raise ValueError(f"E2E materialize worker count must be >= 0, got {max_workers}")
    if requested == 0:
        requested = min(os.cpu_count() or 1, E2E_MATERIALIZE_AUTO_WORKER_CAP)
    return max(1, min(requested, task_count))


def _e2e_combo_row_for_frame(feature_csv: Path, case_meta: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    combo = str(feature_csv.stem)
    return {
        "source_name": str(case_meta["source_name"]),
        "case_id": str(case_meta["case_id"]),
        "combo": combo,
        "rows": int(len(frame)),
        "fit_rows": int(_stable_truth_mask(frame).sum()),
        "batch_size": int(pd.to_numeric(frame.get("batch_size"), errors="coerce").dropna().iloc[0]),
        "num_indices_per_lookup": int(
            pd.to_numeric(frame.get("num_indices_per_lookup"), errors="coerce").dropna().iloc[0]
        ),
    }


def _build_e2e_combo_full_task(
    task: tuple[Path, dict[str, Any], Path | None],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    feature_csv, case_meta, hardware_profile_path = task
    frame = _build_rows_for_e2e_observable_feature_csv(
        feature_csv,
        case_meta,
        hardware_profile_path=hardware_profile_path,
        drop_first_profile_batch=True,
    )
    if frame.empty:
        return frame, None
    frame["split"] = "e2e_full"
    return frame, _e2e_combo_row_for_frame(feature_csv, case_meta, frame)


def build_e2e_combo_full_dataset(
    dataset_root: Path,
    *,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = Path(dataset_root).resolve()
    source_manifest_path = dataset_root / "source_manifest.json"
    if not source_manifest_path.exists():
        return pd.DataFrame(), {
            "dataset_root": str(dataset_root),
            "available": False,
            "reason": "source_manifest_missing",
            "rows": 0,
            "combo_count": 0,
        }

    case_entries = resolve_source_entries(source_manifest_path)
    if not case_entries:
        return pd.DataFrame(), {
            "dataset_root": str(dataset_root),
            "available": False,
            "reason": "no_case_entries",
            "rows": 0,
            "combo_count": 0,
        }

    hardware_profile_path = _resolve_source_hw_config(dataset_root)
    template_columns: list[str] = []
    template_path = dataset_root / "dataset_full.csv"
    if template_path.exists():
        template_columns = pd.read_csv(template_path, nrows=0).columns.tolist()

    frames: list[pd.DataFrame] = []
    combo_rows: list[dict[str, Any]] = []
    source_combo_counts: dict[str, int] = {}
    tasks: list[tuple[Path, dict[str, Any], Path | None]] = []
    for case_meta in case_entries:
        case_dir = Path(case_meta["case_dir"]).resolve()
        feature_glob = str(case_meta.get("feature_glob", "*.csv")).strip() or "*.csv"
        feature_csvs = sorted(case_dir.glob(feature_glob))
        source_combo_counts[str(case_meta["source_name"])] = len(feature_csvs)
        for feature_csv in feature_csvs:
            tasks.append((feature_csv, case_meta, hardware_profile_path))

    worker_count = _resolve_e2e_materialize_workers(max_workers, len(tasks))
    print(
        f"[pipeline_eval] e2e_full_materialize_start dataset_root={dataset_root.name} "
        f"combos={len(tasks)} workers={worker_count}",
        flush=True,
    )

    if worker_count == 1:
        for processed_combo_count, task in enumerate(tasks, start=1):
            frame, combo_row = _build_e2e_combo_full_task(task)
            if not frame.empty:
                frames.append(frame)
            if combo_row is not None:
                combo_rows.append(combo_row)
            if processed_combo_count % 50 == 0 or processed_combo_count == len(tasks):
                print(
                    f"[pipeline_eval] e2e_full_materialize dataset_root={dataset_root.name} "
                    f"combos={processed_combo_count}/{len(tasks)}",
                    flush=True,
                )
    else:
        with cf.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_build_e2e_combo_full_task, task) for task in tasks]
            for processed_combo_count, future in enumerate(cf.as_completed(futures), start=1):
                frame, combo_row = future.result()
                if not frame.empty:
                    frames.append(frame)
                if combo_row is not None:
                    combo_rows.append(combo_row)
                if processed_combo_count % 50 == 0 or processed_combo_count == len(tasks):
                    print(
                        f"[pipeline_eval] e2e_full_materialize dataset_root={dataset_root.name} "
                        f"combos={processed_combo_count}/{len(tasks)}",
                        flush=True,
                    )

    if not frames:
        return pd.DataFrame(), {
            "dataset_root": str(dataset_root),
            "available": False,
            "reason": "no_feature_rows",
            "rows": 0,
            "combo_count": 0,
            "materialize_task_count": int(len(tasks)),
            "materialize_workers": int(worker_count),
        }

    dataset = pd.concat(frames, ignore_index=True, sort=False)
    dataset = dataset.sort_values(["source_name", "combo", "op_idx", "node_name"], kind="stable").reset_index(drop=True)
    if template_columns:
        for column in template_columns:
            if column not in dataset.columns:
                dataset[column] = pd.NA
        extra_columns = [column for column in dataset.columns if column not in template_columns]
        dataset = dataset[template_columns + extra_columns]

    combo_df = pd.DataFrame(combo_rows).sort_values(["source_name", "batch_size", "num_indices_per_lookup"], kind="stable")
    summary = {
        "dataset_root": str(dataset_root),
        "available": True,
        "reason": "",
        "rows": int(len(dataset)),
        "combo_count": int(len(combo_df)),
        "fit_rows": int(_stable_truth_mask(dataset).sum()),
        "mean_rows_per_combo": float(combo_df["rows"].mean()) if not combo_df.empty else 0.0,
        "mean_fit_rows_per_combo": float(combo_df["fit_rows"].mean()) if not combo_df.empty else 0.0,
        "source_combo_counts": source_combo_counts,
        "materialize_task_count": int(len(tasks)),
        "materialize_workers": int(worker_count),
        "profile_instability_metric": E2E_PROFILE_INSTABILITY_METRIC,
        "profile_instability_threshold": float(E2E_PROFILE_INSTABILITY_THRESHOLD),
        "graph_scope": E2E_GRAPH_SCOPE_OBSERVABLE,
    }
    return dataset, summary


def materialize_e2e_combo_full_dataset(
    dataset_root: Path,
    *,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = Path(dataset_root).resolve()
    dataset, summary = build_e2e_combo_full_dataset(dataset_root, max_workers=max_workers)
    if dataset.empty:
        _json_dump(dataset_root / E2E_FULL_MANIFEST_JSON_NAME, summary)
        return dataset, summary
    dataset.to_csv(dataset_root / E2E_FULL_DATASET_CSV_NAME, index=False)
    _group_case_combo_rows(dataset).to_csv(dataset_root / E2E_FULL_GROUPS_CSV_NAME, index=False)
    summary = {
        **summary,
        "dataset_csv": str((dataset_root / E2E_FULL_DATASET_CSV_NAME).resolve()),
        "groups_csv": str((dataset_root / E2E_FULL_GROUPS_CSV_NAME).resolve()),
    }
    _json_dump(dataset_root / E2E_FULL_MANIFEST_JSON_NAME, summary)
    return dataset, summary


def load_or_materialize_e2e_combo_full_dataset(
    dataset_root: Path,
    *,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = Path(dataset_root).resolve()
    csv_path = dataset_root / E2E_FULL_DATASET_CSV_NAME
    manifest_path = dataset_root / E2E_FULL_MANIFEST_JSON_NAME
    if csv_path.exists() and manifest_path.exists():
        manifest = _read_json(manifest_path)
        if str(manifest.get("graph_scope", "")).strip() == E2E_GRAPH_SCOPE_OBSERVABLE:
            return pd.read_csv(csv_path, low_memory=False), manifest
    return materialize_e2e_combo_full_dataset(dataset_root, max_workers=max_workers)


def _source_static_map(dataset_root: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(dataset_root / "manifest.json")
    static_map: dict[str, dict[str, Any]] = {}
    for spec in manifest.get("case_specs", []):
        source_name = str(spec.get("source_name", "")).strip()
        if not source_name:
            continue
        static_map[source_name] = {
            "case_id": str(spec.get("case_id", "")).strip(),
            "inter_threads": _as_int(spec.get("inter_threads")),
            "num_threads": _as_int(spec.get("num_threads")),
            "arch_embedding_size": str(spec.get("arch_embedding_size", "")).strip(),
            "arch_mlp_bot": str(spec.get("arch_mlp_bot", "")).strip(),
            "arch_mlp_top": str(spec.get("arch_mlp_top", "")).strip(),
        }
    return static_map


_COMBO_RE = re.compile(r"^bs(?P<batch>\d+)_nip(?P<nip>\d+)$")


def _parse_combo(combo: str) -> tuple[int | None, int | None]:
    match = _COMBO_RE.fullmatch(str(combo).strip())
    if match is None:
        return None, None
    return int(match.group("batch")), int(match.group("nip"))


def _combo_axis_value(
    *,
    combo: str,
    case_meta: dict[str, Any],
    static_meta: dict[str, Any],
    main_axis: str,
) -> Any:
    batch_size, num_indices = _parse_combo(combo)
    if main_axis == JOINT_6D_AXIS:
        return joint_6d_tuple_key(
            batch_size=batch_size,
            num_indices_per_lookup=num_indices,
            arch_embedding_size=static_meta.get("arch_embedding_size"),
            arch_mlp_bot=static_meta.get("arch_mlp_bot"),
            inter_threads=static_meta.get("inter_threads"),
            num_threads=static_meta.get("num_threads"),
        )
    if main_axis == "batch_size":
        return batch_size
    if main_axis == "num_indices_per_lookup":
        return num_indices
    if main_axis == "case_id":
        return str(case_meta.get("case_id", ""))
    if main_axis in {"inter_threads", "num_threads"}:
        return _as_int(static_meta.get(main_axis))
    if main_axis in {"arch_embedding_size", "arch_mlp_bot", "arch_mlp_top"}:
        return str(static_meta.get(main_axis, "")).strip()
    return static_meta.get(main_axis)


def _load_or_build_pipeline_artifact_index(dataset_root: Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root).resolve()
    csv_path = dataset_root / PIPELINE_INDEX_CSV_NAME
    if csv_path.exists():
        return _resolve_pipeline_artifact_index(pd.read_csv(csv_path, low_memory=False), base_root=dataset_root)
    index_df, _ = materialize_pipeline_artifact_index(dataset_root)
    return index_df


def _combo_meta_frame(dataset_root: Path, *, main_axis: str) -> pd.DataFrame:
    dataset_root = Path(dataset_root).resolve()
    index_df = _load_or_build_pipeline_artifact_index(dataset_root)
    if index_df.empty:
        return pd.DataFrame(
            columns=[
                "pipeline_index_key",
                "source_name",
                "case_id",
                "combo",
                "batch_size",
                "num_indices_per_lookup",
                "inter_threads",
                "num_threads",
                "axis_value",
                "axis_value_norm",
            ]
        )
    meta = index_df[
        [
            "pipeline_index_key",
            "source_name",
            "case_id",
            "combo",
            "batch_size",
            "num_indices_per_lookup",
            "inter_threads",
            "num_threads",
        ]
    ].drop_duplicates().copy()
    static_map = _source_static_map(dataset_root)
    if main_axis in meta.columns:
        meta["axis_value"] = meta[main_axis]
    elif main_axis == JOINT_6D_AXIS:
        meta["axis_value"] = [
            _combo_axis_value(
                combo=row.combo,
                case_meta={"case_id": row.case_id},
                static_meta={
                    **static_map.get(str(row.source_name), {}),
                    "inter_threads": row.inter_threads
                    if not pd.isna(row.inter_threads)
                    else static_map.get(str(row.source_name), {}).get("inter_threads"),
                    "num_threads": row.num_threads
                    if not pd.isna(row.num_threads)
                    else static_map.get(str(row.source_name), {}).get("num_threads"),
                },
                main_axis=main_axis,
            )
            for row in meta.itertuples(index=False)
        ]
    elif main_axis in {"arch_embedding_size", "arch_mlp_bot", "arch_mlp_top"}:
        meta["axis_value"] = meta["source_name"].map(lambda value: static_map.get(str(value), {}).get(main_axis))
    elif main_axis == "case_id":
        meta["axis_value"] = meta["case_id"]
    else:
        raise KeyError(f"Unsupported E2E combo split axis: {main_axis}")
    meta["axis_value_norm"] = meta["axis_value"].map(_normalized_text)
    return meta.sort_values(["axis_value_norm", "source_name", "combo"], kind="stable").reset_index(drop=True)


def _select_joint_axis_combo_holdout_keys(
    combo_meta: pd.DataFrame,
    *,
    policy: str,
    main_axis: str,
    target_test_ratio: float,
) -> tuple[set[str], dict[str, Any]]:
    buckets = (
        combo_meta.groupby("axis_value_norm", dropna=False)
        .agg(combo_count=("pipeline_index_key", "nunique"))
        .reset_index()
    )
    buckets["stable_hash"] = buckets["axis_value_norm"].astype(str).map(
        lambda value: hashlib.sha1(f"{policy}::{main_axis}::{value}".encode("utf-8")).hexdigest()
    )
    ordered = buckets.sort_values(["stable_hash", "axis_value_norm"], kind="stable").reset_index(drop=True)
    total_combo_count = int(ordered["combo_count"].sum())
    target_combo_count = int(round(total_combo_count * float(target_test_ratio)))
    target_combo_count = max(1, min(target_combo_count, max(total_combo_count - 1, 1)))
    selected_values: list[str] = []
    selected_combo_count = 0
    for row in ordered.itertuples(index=False):
        if selected_values and selected_combo_count >= target_combo_count:
            break
        selected_values.append(str(row.axis_value_norm))
        selected_combo_count += int(row.combo_count)
    if len(selected_values) > 1:
        last_value = selected_values[-1]
        last_count = int(ordered.loc[ordered["axis_value_norm"].astype(str).eq(last_value), "combo_count"].iloc[0])
        without_last = selected_combo_count - last_count
        if without_last > 0 and abs(without_last - target_combo_count) < abs(selected_combo_count - target_combo_count):
            selected_values.pop()
            selected_combo_count = without_last
    selected_value_set = set(selected_values)
    selected = combo_meta.loc[
        combo_meta["axis_value_norm"].astype(str).isin(selected_value_set), "pipeline_index_key"
    ].astype(str)
    selected_keys = set(selected.tolist())
    combo_count_test = int(len(selected_keys))
    combo_count_train = int(total_combo_count - combo_count_test)
    bucket_rows = [
        {
            "axis_value": str(row.axis_value_norm),
            "combo_count": int(row.combo_count),
            "test_combo_count": int(row.combo_count) if str(row.axis_value_norm) in selected_value_set else 0,
            "train_combo_count": 0 if str(row.axis_value_norm) in selected_value_set else int(row.combo_count),
        }
        for row in ordered.itertuples(index=False)
    ]
    return selected_keys, {
        "available": True,
        "reason": "",
        "split_mode": "joint_6d_tuple_holdout",
        "main_axis": main_axis,
        "target_test_ratio": float(target_test_ratio),
        "combo_count_total": total_combo_count,
        "combo_count_test": combo_count_test,
        "combo_count_train": combo_count_train,
        "combo_test_ratio": float(combo_count_test / max(total_combo_count, 1)),
        "combo_train_ratio": float(combo_count_train / max(total_combo_count, 1)),
        "axis_bucket_count": int(len(bucket_rows)),
        "axis_bucket_rows": bucket_rows,
    }


def _select_combo_holdout_keys(
    dataset_root: Path,
    *,
    policy: str,
    main_axis: str,
    target_test_ratio: float,
) -> tuple[set[str], dict[str, Any]]:
    combo_meta = _combo_meta_frame(dataset_root, main_axis=main_axis)
    print(
        f"[pipeline_eval] select_combo_holdout_keys dataset_root={Path(dataset_root).name} "
        f"policy={policy} main_axis={main_axis} combos={len(combo_meta)}",
        flush=True,
    )
    if combo_meta.empty:
        return set(), {
            "available": False,
            "reason": "no_pipeline_combo_metadata",
            "combo_count_total": 0,
            "combo_count_test": 0,
            "combo_count_train": 0,
            "combo_test_ratio": 0.0,
            "combo_train_ratio": 0.0,
            "main_axis": main_axis,
            "target_test_ratio": float(target_test_ratio),
        }

    if main_axis == JOINT_6D_AXIS:
        return _select_joint_axis_combo_holdout_keys(
            combo_meta,
            policy=policy,
            main_axis=main_axis,
            target_test_ratio=target_test_ratio,
        )

    selected_keys: list[str] = []
    bucket_rows: list[dict[str, Any]] = []
    for axis_value_norm, bucket in combo_meta.groupby("axis_value_norm", sort=True):
        ordered = bucket.copy()
        ordered["stable_hash"] = ordered.apply(
            lambda row: hashlib.sha1(
                f"{policy}::{main_axis}::{row['source_name']}::{row['combo']}".encode("utf-8")
            ).hexdigest(),
            axis=1,
        )
        ordered = ordered.sort_values(["stable_hash", "source_name", "combo"], kind="stable").reset_index(drop=True)
        bucket_size = int(len(ordered))
        if bucket_size <= 1:
            test_count = 0
        else:
            test_count = int(round(bucket_size * float(target_test_ratio)))
            test_count = max(test_count, 1)
            test_count = min(test_count, bucket_size - 1)
        selected_keys.extend(ordered.head(test_count)["pipeline_index_key"].astype(str).tolist())
        bucket_rows.append(
            {
                "axis_value": axis_value_norm,
                "combo_count": bucket_size,
                "test_combo_count": int(test_count),
                "train_combo_count": int(bucket_size - test_count),
            }
        )

    total_combo_count = int(len(combo_meta))
    test_key_set = set(selected_keys)
    combo_count_test = int(len(test_key_set))
    combo_count_train = int(total_combo_count - combo_count_test)
    return test_key_set, {
        "available": True,
        "reason": "",
        "main_axis": main_axis,
        "target_test_ratio": float(target_test_ratio),
        "combo_count_total": total_combo_count,
        "combo_count_test": combo_count_test,
        "combo_count_train": combo_count_train,
        "combo_test_ratio": float(combo_count_test / max(total_combo_count, 1)),
        "combo_train_ratio": float(combo_count_train / max(total_combo_count, 1)),
        "axis_bucket_count": int(len(bucket_rows)),
        "axis_bucket_rows": bucket_rows,
    }


def _pipeline_key_series(frame: pd.DataFrame) -> pd.Series:
    return frame["source_name"].astype(str) + "::" + frame["combo"].astype(str)


def _stable_small_e2e_score(source_name: Any, combo: Any, *, seed: int, split_name: str) -> int:
    text = f"{int(seed)}|{split_name}|{source_name}|{combo}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _select_small_e2e_combo_keys(
    frame: pd.DataFrame,
    *,
    divisor: int,
    seed: int,
    split_name: str,
) -> set[str]:
    groups = _group_case_combo_rows(frame)
    if groups.empty:
        return set()
    groups = groups.copy()
    groups["pipeline_index_key"] = _pipeline_key_series(groups)
    if int(divisor) <= 1:
        return set(groups["pipeline_index_key"].astype(str).tolist())
    groups["_score"] = [
        _stable_small_e2e_score(row.source_name, row.combo, seed=seed, split_name=split_name)
        for row in groups.itertuples(index=False)
    ]
    target_total = max(1, int(round(len(groups) / float(divisor))))
    allocations: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    grouped = list(groups.groupby("case_id", sort=True))
    for case_id, case_group in grouped:
        quota = len(case_group) / float(divisor)
        take = int(math.floor(quota))
        if len(case_group) > 0:
            take = max(1, take)
        take = min(take, len(case_group))
        allocations[str(case_id)] = take
        fractions.append((quota - math.floor(quota), str(case_id)))

    current = sum(allocations.values())
    if current < target_total:
        for _, case_id in sorted(fractions, reverse=True):
            if current >= target_total:
                break
            case_size = int((groups["case_id"].astype(str) == case_id).sum())
            if allocations[case_id] < case_size:
                allocations[case_id] += 1
                current += 1
    elif current > target_total:
        for _, case_id in sorted(fractions):
            if current <= target_total:
                break
            if allocations[case_id] > 1:
                allocations[case_id] -= 1
                current -= 1

    selected: set[str] = set()
    for case_id, case_group in grouped:
        take = allocations[str(case_id)]
        selected.update(
            case_group.sort_values("_score", kind="stable")
            .head(take)["pipeline_index_key"]
            .astype(str)
            .tolist()
        )
    return selected


def _materialize_combo_key_frame(
    dataset_root: Path,
    *,
    selected_keys: set[str],
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = Path(dataset_root).resolve()
    template_path = dataset_root / "dataset_full.csv"
    if template_path.exists():
        template_columns = pd.read_csv(template_path, nrows=0).columns.tolist()
        use_columns = list(dict.fromkeys(template_columns + ["source_name", "combo"]))
        raw_dataset = pd.read_csv(template_path, low_memory=False, usecols=use_columns)
        if not raw_dataset.empty and {"source_name", "combo"}.issubset(raw_dataset.columns):
            raw_dataset["pipeline_index_key"] = (
                raw_dataset["source_name"].astype(str) + "::" + raw_dataset["combo"].astype(str)
            )
            frame = raw_dataset.loc[raw_dataset["pipeline_index_key"].astype(str).isin(selected_keys)].drop(
                columns=["pipeline_index_key"]
            )
            if not frame.empty:
                frame = frame.sort_values(["source_name", "combo", "op_idx", "node_name"], kind="stable").reset_index(
                    drop=True
                )
                if "split" in frame.columns:
                    frame["split"] = split_name
                print(
                    f"[pipeline_eval] materialize_combo_key_frame_fastpath split={split_name} "
                    f"dataset_root={dataset_root.name} rows={len(frame)} combos={frame[['source_name', 'combo']].drop_duplicates().shape[0]}",
                    flush=True,
                )
                return frame, {
                    "available": True,
                    "reason": "",
                    "rows": int(len(frame)),
                    "combo_count": int(frame[["source_name", "combo"]].drop_duplicates().shape[0]),
                }

    source_manifest_path = dataset_root / "source_manifest.json"
    if not source_manifest_path.exists():
        return pd.DataFrame(), {
            "available": False,
            "reason": "source_manifest_missing",
            "rows": 0,
            "combo_count": 0,
        }

    case_entries = resolve_source_entries(source_manifest_path)
    hardware_profile_path = _resolve_source_hw_config(dataset_root)
    template_columns: list[str] = []
    if template_path.exists():
        template_columns = pd.read_csv(template_path, nrows=0).columns.tolist()

    by_source: dict[str, set[str]] = {}
    for key in selected_keys:
        source_name, _, combo = str(key).partition("::")
        if not source_name or not combo:
            continue
        by_source.setdefault(source_name, set()).add(combo)
    print(
        f"[pipeline_eval] materialize_combo_key_frame_start split={split_name} "
        f"dataset_root={dataset_root.name} selected_combos={len(selected_keys)} sources={len(by_source)}",
        flush=True,
    )

    tasks: list[tuple[Path, dict[str, Any]]] = []
    for case_meta in case_entries:
        source_name = str(case_meta["source_name"])
        selected_combos = by_source.get(source_name, set())
        if not selected_combos:
            continue
        case_dir = Path(case_meta["case_dir"]).resolve()
        feature_glob = str(case_meta.get("feature_glob", "*.csv")).strip() or "*.csv"
        for feature_csv in sorted(case_dir.glob(feature_glob)):
            combo = str(feature_csv.stem)
            if combo not in selected_combos:
                continue
            tasks.append((feature_csv, case_meta))

    def _build_task(task: tuple[Path, dict[str, Any]]) -> pd.DataFrame:
        feature_csv, case_meta = task
        frame = build_rows_for_feature_csv(
            feature_csv,
            case_meta,
            hardware_profile_path=hardware_profile_path,
            drop_first_profile_batch=True,
        )
        if frame.empty:
            return frame
        frame["profile_instability_metric_name"] = E2E_PROFILE_INSTABILITY_METRIC
        frame["profile_instability_value"] = pd.to_numeric(
            frame.get(f"profile_{E2E_PROFILE_INSTABILITY_METRIC}", pd.Series(pd.NA, index=frame.index)),
            errors="coerce",
        )
        frame["profile_is_stable"] = (
            frame["profile_instability_value"].isna()
            | (frame["profile_instability_value"] <= float(E2E_PROFILE_INSTABILITY_THRESHOLD))
        )
        frame["split"] = split_name
        return frame

    frames: list[pd.DataFrame] = []
    max_workers = max(1, min(8, len(tasks)))
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_build_task, task) for task in tasks]
        for processed_combo_count, future in enumerate(cf.as_completed(futures), start=1):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
            if processed_combo_count % 100 == 0 or processed_combo_count == len(tasks):
                print(
                    f"[pipeline_eval] materialize_combo_key_frame split={split_name} "
                    f"dataset_root={dataset_root.name} combos={processed_combo_count}/{len(tasks)}",
                    flush=True,
                )

    if not frames:
        return pd.DataFrame(), {
            "available": False,
            "reason": "no_materialized_feature_rows",
            "rows": 0,
            "combo_count": 0,
        }

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = frame.sort_values(["source_name", "combo", "op_idx", "node_name"], kind="stable").reset_index(drop=True)
    if template_columns:
        for column in template_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        extra_columns = [column for column in frame.columns if column not in template_columns]
        frame = frame[template_columns + extra_columns]
    return frame, {
        "available": True,
        "reason": "",
        "rows": int(len(frame)),
        "combo_count": int(frame[["source_name", "combo"]].drop_duplicates().shape[0]),
    }


def build_e2e_combo_split_frames(
    dataset_root: Path,
    *,
    base_frame: pd.DataFrame,
    policy: str,
    main_axis: str,
    target_test_ratio: float,
    max_workers: int | None = None,
    full_frame: pd.DataFrame | None = None,
    full_summary: dict[str, Any] | None = None,
    small_divisor: int = 1,
    small_seed: int = 20260425,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    test_keys, split_summary = _select_combo_holdout_keys(
        dataset_root,
        policy=policy,
        main_axis=main_axis,
        target_test_ratio=target_test_ratio,
    )
    if not test_keys:
        return pd.DataFrame(), pd.DataFrame(), split_summary

    if full_frame is not None and not full_frame.empty:
        work = full_frame
        resolved_full_summary = full_summary or {
            "available": True,
            "reason": "",
            "graph_scope": E2E_GRAPH_SCOPE_OBSERVABLE,
        }
    else:
        work, resolved_full_summary = load_or_materialize_e2e_combo_full_dataset(dataset_root, max_workers=max_workers)
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            **split_summary,
            "train_rows": 0,
            "train_case_combo_groups": 0,
            "test_raw_rows": 0,
            "test_raw_combo_count": 0,
            "materialize_available": bool(resolved_full_summary.get("available", False)),
            "materialize_reason": str(resolved_full_summary.get("reason", "")),
            "graph_scope": str(resolved_full_summary.get("graph_scope", "")),
            "main_axis_test_values": [],
        }
    key_series = work["source_name"].astype(str) + "::" + work["combo"].astype(str)
    train_frame = work.loc[~key_series.isin(test_keys)].copy().reset_index(drop=True)
    test_raw_frame = work.loc[key_series.isin(test_keys)].copy().reset_index(drop=True)
    if not test_raw_frame.empty:
        test_raw_frame = test_raw_frame.reset_index(drop=True)

    full_train_groups = _group_case_combo_rows(train_frame)
    full_test_groups = _group_case_combo_rows(test_raw_frame)
    full_train_combo_count = int(len(full_train_groups))
    full_test_combo_count = int(len(full_test_groups))
    full_combo_count = int(full_train_combo_count + full_test_combo_count)
    resolved_small_divisor = max(int(small_divisor), 1)
    if resolved_small_divisor > 1:
        train_small_keys = _select_small_e2e_combo_keys(
            train_frame,
            divisor=resolved_small_divisor,
            seed=int(small_seed),
            split_name="e2e_train",
        )
        test_small_keys = _select_small_e2e_combo_keys(
            test_raw_frame,
            divisor=resolved_small_divisor,
            seed=int(small_seed),
            split_name="e2e_test_raw",
        )
        train_frame = train_frame.loc[_pipeline_key_series(train_frame).isin(train_small_keys)].copy().reset_index(drop=True)
        test_raw_frame = test_raw_frame.loc[_pipeline_key_series(test_raw_frame).isin(test_small_keys)].copy().reset_index(drop=True)

    train_groups = _group_case_combo_rows(train_frame)
    test_groups = _group_case_combo_rows(test_raw_frame)
    train_combo_count = int(len(train_groups))
    test_combo_count = int(len(test_groups))
    combo_count = int(train_combo_count + test_combo_count)
    actual_test_keys = set(_pipeline_key_series(test_raw_frame).astype(str).tolist())
    split_summary = {
        **split_summary,
        "train_rows": int(len(train_frame)),
        "train_case_combo_groups": train_combo_count,
        "test_raw_rows": int(len(test_raw_frame)),
        "test_raw_combo_count": test_combo_count,
        "combo_count_total": combo_count,
        "combo_count_train": train_combo_count,
        "combo_count_test": test_combo_count,
        "combo_train_ratio": float(train_combo_count / max(combo_count, 1)),
        "combo_test_ratio": float(test_combo_count / max(combo_count, 1)),
        "full_combo_count_total": full_combo_count,
        "full_combo_count_train": full_train_combo_count,
        "full_combo_count_test": full_test_combo_count,
        "small_dataset_divisor": resolved_small_divisor,
        "small_dataset_seed": int(small_seed),
        "small_dataset_combo_fraction": float(combo_count / max(full_combo_count, 1)),
        "small_dataset_train_combo_fraction": float(train_combo_count / max(full_train_combo_count, 1)),
        "small_dataset_test_combo_fraction": float(test_combo_count / max(full_test_combo_count, 1)),
        "materialize_available": bool(resolved_full_summary.get("available", False)),
        "materialize_reason": str(resolved_full_summary.get("reason", "")),
        "graph_scope": str(resolved_full_summary.get("graph_scope", E2E_GRAPH_SCOPE_OBSERVABLE)),
        "main_axis_test_values": sorted(
            {
                _normalized_text(value)
                for value in _combo_meta_frame(dataset_root, main_axis=main_axis)
                .loc[lambda df: df["pipeline_index_key"].astype(str).isin(actual_test_keys), "axis_value"]
                .tolist()
            }
        ),
    }
    return train_frame, test_raw_frame, split_summary


def build_e2e_test_raw_frame(
    dataset_root: Path,
    *,
    main_axis: str,
    test_values: list[Any],
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = Path(dataset_root).resolve()
    full_frame, full_summary = load_or_materialize_e2e_combo_full_dataset(dataset_root, max_workers=max_workers)
    if full_frame.empty:
        return pd.DataFrame(), {
            "available": bool(full_summary.get("available", False)),
            "reason": str(full_summary.get("reason", "")),
            "rows": 0,
            "combo_count": 0,
            "combo_ratio": 0.0,
        }

    def _normalized(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        return text

    normalized_test_values = {_normalized(value) for value in test_values}
    combo_meta = _combo_meta_frame(dataset_root, main_axis=main_axis)
    selected_keys = set(
        combo_meta.loc[
            combo_meta["axis_value"].astype(str).isin(normalized_test_values),
            "pipeline_index_key",
        ].astype(str).tolist()
    )
    if not selected_keys:
        return pd.DataFrame(), {
            "available": True,
            "reason": "no_matching_combos",
            "rows": 0,
            "combo_count": 0,
            "combo_ratio": 0.0,
        }

    work = full_frame.copy()
    work["pipeline_index_key"] = work["source_name"].astype(str) + "::" + work["combo"].astype(str)
    frame = work.loc[work["pipeline_index_key"].astype(str).isin(selected_keys)].drop(columns=["pipeline_index_key"]).reset_index(drop=True)
    if not frame.empty and "split" in frame.columns:
        frame["split"] = "e2e_test_raw"
    return frame, {
        "available": True,
        "reason": "",
        "rows": int(len(frame)),
        "combo_count": int(frame[["source_name", "combo"]].drop_duplicates().shape[0]),
        "combo_ratio": float(len(selected_keys) / max(combo_meta.shape[0], 1)),
        "total_combo_count": int(combo_meta.shape[0]),
        "selected_combo_count": int(len(selected_keys)),
        "test_values": [value for value in test_values],
        "graph_scope": str(full_summary.get("graph_scope", E2E_GRAPH_SCOPE_OBSERVABLE)),
    }


def build_pipeline_artifact_index(dataset_root: Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root).resolve()
    source_manifest_path = dataset_root / "source_manifest.json"
    payload = _read_json(source_manifest_path)
    sources = payload.get("sources", [])
    if not sources:
        return _empty_frame(
            [
                "pipeline_index_key",
                "source_name",
                "combo",
                "case_id",
                "batch_size",
                "num_indices_per_lookup",
                "inter_threads",
                "num_threads",
                "sweep_dir",
                "shape_csv",
                "profile_dir",
                "timeline_csv",
                "effective_onnx_path",
                "log_dir",
                "shape_csv_exists",
                "timeline_csv_exists",
                "effective_onnx_exists",
            ]
        )

    static_map = _source_static_map(dataset_root)
    rows: list[dict[str, Any]] = []
    for source in sources:
        source_name = str(source.get("source_name", "")).strip()
        if not source_name:
            continue
        source_case_id = str(source.get("case_id", "")).strip()
        source_static = static_map.get(source_name, {})
        sweep_dir = Path(str(source.get("sweep_dir", ""))).resolve()
        sweep_summary_path = sweep_dir / "sweep_summary.csv"
        if not sweep_summary_path.exists():
            continue
        sweep_df = pd.read_csv(sweep_summary_path, low_memory=False)
        if "batch_size" not in sweep_df.columns or "num_indices_per_lookup" not in sweep_df.columns:
            continue
        for row in sweep_df.itertuples(index=False):
            batch_size = _as_int(getattr(row, "batch_size", None))
            nip = _as_int(getattr(row, "num_indices_per_lookup", None))
            if batch_size is None or nip is None:
                continue
            combo = format_combo(batch_size, nip)
            profile_dir = _path_text(getattr(row, "profile_dir", ""))
            timeline_csv = str((Path(profile_dir) / "branch_parallel_op_timeline.csv").resolve()) if profile_dir else ""
            shape_csv = _path_text(getattr(row, "shape_csv", ""))
            effective_onnx_path = _path_text(getattr(row, "effective_onnx_path", ""))
            log_dir = _path_text(getattr(row, "log_dir", ""))
            case_id = source_static.get("case_id") or source_case_id
            rows.append(
                {
                    "pipeline_index_key": pipeline_index_key(source_name, combo),
                    "source_name": source_name,
                    "combo": combo,
                    "case_id": str(case_id),
                    "batch_size": batch_size,
                    "num_indices_per_lookup": nip,
                    "inter_threads": source_static.get("inter_threads"),
                    "num_threads": source_static.get("num_threads"),
                    "sweep_dir": str(sweep_dir),
                    "shape_csv": shape_csv,
                    "profile_dir": profile_dir,
                    "timeline_csv": timeline_csv,
                    "effective_onnx_path": effective_onnx_path,
                    "log_dir": log_dir,
                    "shape_csv_exists": bool(shape_csv and Path(shape_csv).exists()),
                    "timeline_csv_exists": bool(timeline_csv and Path(timeline_csv).exists()),
                    "effective_onnx_exists": bool(effective_onnx_path and Path(effective_onnx_path).exists()),
                }
            )

    index_df = pd.DataFrame(rows)
    if index_df.empty:
        return _empty_frame(
            [
                "pipeline_index_key",
                "source_name",
                "combo",
                "case_id",
                "batch_size",
                "num_indices_per_lookup",
                "inter_threads",
                "num_threads",
                "sweep_dir",
                "shape_csv",
                "profile_dir",
                "timeline_csv",
                "effective_onnx_path",
                "log_dir",
                "shape_csv_exists",
                "timeline_csv_exists",
                "effective_onnx_exists",
            ]
        )

    index_df = index_df.drop_duplicates(subset=["pipeline_index_key"], keep="first")
    index_df = index_df.sort_values(["source_name", "batch_size", "num_indices_per_lookup"]).reset_index(drop=True)
    return index_df


def write_pipeline_artifact_index(output_root: Path, index_df: pd.DataFrame) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ordered = index_df.copy()
    if not ordered.empty:
        ordered = ordered.sort_values(["source_name", "batch_size", "num_indices_per_lookup"]).reset_index(drop=True)
    ordered.to_csv(output_root / PIPELINE_INDEX_CSV_NAME, index=False)
    _json_dump(
        output_root / PIPELINE_INDEX_JSON_NAME,
        {
            "row_count": int(len(ordered)),
            "rows": ordered.to_dict(orient="records"),
        },
    )
    return {
        "pipeline_artifact_index_csv": str((output_root / PIPELINE_INDEX_CSV_NAME).resolve()),
        "pipeline_artifact_index_json": str((output_root / PIPELINE_INDEX_JSON_NAME).resolve()),
        "pipeline_artifact_index_rows": int(len(ordered)),
    }


def materialize_pipeline_artifact_index(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = Path(dataset_root).resolve()
    index_df = build_pipeline_artifact_index(dataset_root)
    summary = write_pipeline_artifact_index(dataset_root, index_df)
    return index_df, summary


def build_pipeline_combo_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "source_name",
        "case_id",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required split columns for pipeline combo manifest: {missing}")
    manifest = frame[required].drop_duplicates().copy()
    manifest["pipeline_index_key"] = manifest["source_name"].astype(str) + "::" + manifest["combo"].astype(str)
    manifest = manifest.sort_values(["source_name", "batch_size", "num_indices_per_lookup"]).reset_index(drop=True)
    return manifest


def materialize_split_pipeline_snapshot(
    *,
    split_root: Path,
    input_dataset_root: Path,
    frame: pd.DataFrame,
    extra_frames: list[pd.DataFrame] | None = None,
) -> dict[str, Any]:
    split_root = Path(split_root).resolve()
    input_dataset_root = Path(input_dataset_root).resolve()
    source_manifest_path = input_dataset_root / "source_manifest.json"
    if not source_manifest_path.exists():
        return {
            "pipeline_metadata_available": False,
            "pipeline_metadata_reason": "source_manifest_missing",
            "pipeline_artifact_index_rows": 0,
            "pipeline_combo_manifest_rows": 0,
        }

    raw_index_path = input_dataset_root / PIPELINE_INDEX_CSV_NAME
    if raw_index_path.exists():
        raw_index = pd.read_csv(raw_index_path, low_memory=False)
    else:
        raw_index, _ = materialize_pipeline_artifact_index(input_dataset_root)
    manifest_frames: list[pd.DataFrame] = []
    if frame is not None and not frame.empty:
        manifest_frames.append(frame)
    for extra in extra_frames or []:
        if extra is not None and not extra.empty:
            manifest_frames.append(extra)
    if not manifest_frames:
        return {
            "pipeline_metadata_available": False,
            "pipeline_metadata_reason": "no_split_rows",
            "pipeline_artifact_index_rows": 0,
            "pipeline_combo_manifest_rows": 0,
        }
    combo_manifest = build_pipeline_combo_manifest(pd.concat(manifest_frames, ignore_index=True, sort=False))
    keys = set(combo_manifest["pipeline_index_key"].astype(str).tolist())
    split_index = raw_index.loc[raw_index["pipeline_index_key"].astype(str).isin(keys)].copy()
    split_index = split_index.sort_values(["source_name", "batch_size", "num_indices_per_lookup"]).reset_index(drop=True)
    missing_keys = sorted(keys - set(split_index["pipeline_index_key"].astype(str).tolist()))
    if missing_keys:
        preview = ", ".join(missing_keys[:10])
        raise KeyError(f"Missing pipeline artifact index rows for split combos: {preview}")
    snapshot_summary = write_pipeline_artifact_index(split_root, split_index)
    combo_manifest.to_csv(split_root / PIPELINE_COMBO_MANIFEST_CSV_NAME, index=False)
    snapshot_summary.update(
        {
            "pipeline_metadata_available": True,
            "pipeline_combo_manifest_csv": str((split_root / PIPELINE_COMBO_MANIFEST_CSV_NAME).resolve()),
            "pipeline_combo_manifest_rows": int(len(combo_manifest)),
        }
    )
    return snapshot_summary


def load_pipeline_artifact_index(split_root: Path) -> pd.DataFrame:
    split_root = Path(split_root).resolve()
    path = split_root / PIPELINE_INDEX_CSV_NAME
    if not path.exists():
        return _empty_frame(
            [
                "pipeline_index_key",
                "source_name",
                "combo",
                "case_id",
                "batch_size",
                "num_indices_per_lookup",
                "inter_threads",
                "num_threads",
                "sweep_dir",
                "shape_csv",
                "profile_dir",
                "timeline_csv",
                "effective_onnx_path",
                "log_dir",
                "shape_csv_exists",
                "timeline_csv_exists",
                "effective_onnx_exists",
            ]
        )
    return _resolve_pipeline_artifact_index(pd.read_csv(path, low_memory=False), base_root=split_root)


def load_pipeline_combo_manifest(split_root: Path) -> pd.DataFrame:
    path = Path(split_root).resolve() / PIPELINE_COMBO_MANIFEST_CSV_NAME
    if not path.exists():
        return _empty_frame(
            [
                "source_name",
                "case_id",
                "combo",
                "batch_size",
                "num_indices_per_lookup",
                "inter_threads",
                "pipeline_index_key",
            ]
        )
    return pd.read_csv(path, low_memory=False)


def pipeline_artifact_index_audit(*, policy: str, dataset: str, split_root: Path) -> pd.DataFrame:
    index_df = load_pipeline_artifact_index(split_root)
    if index_df.empty:
        return _empty_frame(
            [
                "policy",
                "dataset",
                "source_name",
                "combo",
                "shape_csv_exists",
                "timeline_csv_exists",
                "effective_onnx_exists",
            ]
        )
    audit = index_df[
        ["source_name", "combo", "shape_csv_exists", "timeline_csv_exists", "effective_onnx_exists"]
    ].copy()
    audit.insert(0, "dataset", str(dataset))
    audit.insert(0, "policy", str(policy))
    return audit.sort_values(["source_name", "combo"]).reset_index(drop=True)


def _is_portable_split_root(split_root: Path) -> bool:
    split_root = Path(split_root).resolve()
    return (
        (split_root / "portable_manifest.json").exists()
        or (split_root.parent / "portable_manifest.json").exists()
        or (split_root / "portable_artifacts").exists()
    )


def _validate_required_pipeline_artifacts(index_df: pd.DataFrame, *, split_root: Path) -> None:
    if index_df.empty or not _is_portable_split_root(split_root):
        return
    missing = index_df[
        ~index_df.get("shape_csv_exists", pd.Series(False, index=index_df.index)).astype(bool)
        | ~index_df.get("timeline_csv_exists", pd.Series(False, index=index_df.index)).astype(bool)
    ].copy()
    if missing.empty:
        return
    preview_cols = ["source_name", "combo", "shape_csv", "timeline_csv", "shape_csv_exists", "timeline_csv_exists"]
    preview_cols = [column for column in preview_cols if column in missing.columns]
    preview = missing[preview_cols].head(20).to_dict(orient="records")
    raise FileNotFoundError(
        f"Portable E2E split is missing required pipeline artifacts under {split_root}: "
        f"{len(missing)} rows missing shape_csv or timeline_csv. Preview={preview}"
    )


def _format_launch_order(order: tuple[int, ...]) -> str:
    return "->".join(str(item) for item in order)


def _metric_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "combo_count_full": 0,
            "e2e_mae_us": None,
            "e2e_mape": None,
            "e2e_p90": None,
            "e2e_median": None,
        }
    return {
        "combo_count_full": int(len(frame)),
        "e2e_mae_us": float(frame["abs_error_us"].mean()),
        "e2e_mape": float(frame["ape"].mean()),
        "e2e_p90": float(frame["ape"].quantile(0.90)),
        "e2e_median": float(frame["ape"].quantile(0.50)),
    }


def evaluate_source_test_pipeline(
    *,
    prediction_rows: pd.DataFrame,
    split_root: Path,
    policy: str,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_root = Path(split_root).resolve()
    split_manifest = _read_json(split_root / "split_manifest.json")
    e2e_graph_scope = str(split_manifest.get("e2e_graph_scope", "")).strip()
    summary_columns = [
        "policy",
        "dataset",
        "route",
        "scale",
        "combo_count_full",
        "combo_count_partial",
        "coverage_ratio_mean",
        "e2e_mae_us",
        "e2e_mape",
        "e2e_p90",
        "e2e_median",
    ]
    combo_columns = [
        "policy",
        "dataset",
        "route",
        "scale",
        "metric_scope",
        "case_id",
        "source_name",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "expected_count",
        "observed_count",
        "coverage_ratio",
        "predicted_e2e_us",
        "actual_e2e_us",
        "abs_error_us",
        "ape",
        "bottom_end_us",
        "branch_pool_end_us",
        "tail_barrier_us",
    ]
    partial_columns = [
        "policy",
        "dataset",
        "route",
        "scale",
        "metric_scope",
        "reason",
        "case_id",
        "source_name",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "expected_count",
        "observed_count",
        "missing_count",
        "coverage_ratio",
        "predicted_observed_subgraph_us",
        "actual_observed_subgraph_us",
        "abs_error_us",
        "ape",
        "missing_node_indices",
        "missing_node_names",
    ]
    embedding_columns = [
        "policy",
        "dataset",
        "route",
        "scale",
        "case_id",
        "source_name",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "kept_batch_count",
        "launch_order",
        "matches_fifo",
        "max_gather_concurrency",
        "matches_inter_threads",
        "handoff_gap_mean_us",
        "handoff_gap_min_us",
        "handoff_gap_max_us",
        "tail_start_gap_mean_us",
        "tail_start_gap_min_us",
        "tail_start_gap_max_us",
    ]

    if prediction_rows.empty:
        return (
            _empty_frame(summary_columns),
            _empty_frame(combo_columns),
            _empty_frame(partial_columns),
            _empty_frame(embedding_columns),
            pipeline_artifact_index_audit(policy=policy, dataset=dataset, split_root=split_root),
        )

    work = prediction_rows.copy()
    if "combo" not in work.columns:
        work["combo"] = work["row_uid"].astype(str).str.extract(r"^[^:]+::([^:]+)::opidx:")
    work["op_idx"] = work["row_uid"].astype(str).str.extract(r"opidx:(\d+)").astype(int)
    index_df = load_pipeline_artifact_index(split_root)
    combo_manifest = load_pipeline_combo_manifest(split_root)
    audit = pipeline_artifact_index_audit(policy=policy, dataset=dataset, split_root=split_root)
    _validate_required_pipeline_artifacts(index_df, split_root=split_root)
    if index_df.empty or combo_manifest.empty:
        return (
            _empty_frame(summary_columns),
            _empty_frame(combo_columns),
            _empty_frame(partial_columns),
            _empty_frame(embedding_columns),
            audit,
        )

    combo_lookup = combo_manifest.merge(
        index_df,
        on=["pipeline_index_key", "source_name", "combo"],
        how="left",
        suffixes=("", "_index"),
    )
    combo_lookup = combo_lookup.set_index("pipeline_index_key", drop=False)

    full_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    grouped_items = list(work.groupby(["route", "scale", "source_name", "combo"], sort=True))

    def _evaluate_combo_group(
        item: tuple[tuple[Any, Any, Any, Any], pd.DataFrame]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        (route, scale, source_name, combo), group = item
        local_full_rows: list[dict[str, Any]] = []
        local_partial_rows: list[dict[str, Any]] = []
        local_embedding_rows: list[dict[str, Any]] = []
        key = pipeline_index_key(source_name, combo)
        if key not in combo_lookup.index:
            local_partial_rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "route": str(route),
                    "scale": str(scale),
                    "metric_scope": "missing_pipeline_index",
                    "reason": "pipeline_index_missing",
                    "case_id": str(group["case_id"].iloc[0]) if "case_id" in group.columns else "",
                    "source_name": str(source_name),
                    "combo": str(combo),
                    "batch_size": _as_int(group["batch_size"].iloc[0]) if "batch_size" in group.columns else None,
                    "num_indices_per_lookup": _as_int(group["num_indices_per_lookup"].iloc[0]) if "num_indices_per_lookup" in group.columns else None,
                    "inter_threads": _as_int(group["inter_threads"].iloc[0]) if "inter_threads" in group.columns else None,
                    "expected_count": None,
                    "observed_count": int(len(group)),
                    "missing_count": None,
                    "coverage_ratio": None,
                    "predicted_observed_subgraph_us": None,
                    "actual_observed_subgraph_us": None,
                    "abs_error_us": None,
                    "ape": None,
                    "missing_node_indices": "",
                    "missing_node_names": "",
                }
            )
            return local_full_rows, local_partial_rows, local_embedding_rows

        meta = combo_lookup.loc[key]
        case_id = str(meta.get("case_id", ""))
        batch_size = _as_int(meta.get("batch_size"))
        nip = _as_int(meta.get("num_indices_per_lookup"))
        group_inter_threads = (
            _as_int(group["inter_threads"].iloc[0])
            if "inter_threads" in group.columns and not group.empty
            else None
        )
        inter_threads = (
            _as_int(meta.get("inter_threads_index"))
            or _as_int(meta.get("inter_threads"))
            or group_inter_threads
            or 1
        )
        shape_csv = str(meta.get("shape_csv", "")).strip()
        timeline_csv = str(meta.get("timeline_csv", "")).strip()
        profile_dir = str(meta.get("profile_dir", "")).strip()
        if not shape_csv or not Path(shape_csv).exists():
            local_partial_rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "route": str(route),
                    "scale": str(scale),
                    "metric_scope": "missing_pipeline_artifact",
                    "reason": "shape_csv_missing",
                    "case_id": case_id,
                    "source_name": str(source_name),
                    "combo": str(combo),
                    "batch_size": batch_size,
                    "num_indices_per_lookup": nip,
                    "inter_threads": inter_threads,
                    "expected_count": None,
                    "observed_count": int(len(group)),
                    "missing_count": None,
                    "coverage_ratio": None,
                    "predicted_observed_subgraph_us": None,
                    "actual_observed_subgraph_us": None,
                    "abs_error_us": None,
                    "ape": None,
                    "missing_node_indices": "",
                    "missing_node_names": "",
                }
            )
            return local_full_rows, local_partial_rows, local_embedding_rows
        if not timeline_csv or not Path(timeline_csv).exists():
            local_partial_rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "route": str(route),
                    "scale": str(scale),
                    "metric_scope": "missing_pipeline_artifact",
                    "reason": "timeline_csv_missing",
                    "case_id": case_id,
                    "source_name": str(source_name),
                    "combo": str(combo),
                    "batch_size": batch_size,
                    "num_indices_per_lookup": nip,
                    "inter_threads": inter_threads,
                    "expected_count": None,
                    "observed_count": int(len(group)),
                    "missing_count": None,
                    "coverage_ratio": None,
                    "predicted_observed_subgraph_us": None,
                    "actual_observed_subgraph_us": None,
                    "abs_error_us": None,
                    "ape": None,
                    "missing_node_indices": "",
                    "missing_node_names": "",
                }
            )
            return local_full_rows, local_partial_rows, local_embedding_rows

        combo_spec = ComboSpec(
            case_id=case_id,
            combo=str(combo),
            batch_size=int(batch_size or 0),
            num_indices_per_lookup=int(nip or 0),
            inter_threads=int(inter_threads),
            artifact_paths=ComboArtifactPaths(
                case_id=case_id,
                combo=str(combo),
                shape_csv=Path(shape_csv),
                profile_dir=Path(profile_dir) if profile_dir else Path(timeline_csv).parent,
                timeline_csv=Path(timeline_csv),
            ),
        )

        try:
            combo_pred = group[["op_idx", "pred_us"]].copy()
            op_shapes_df = load_op_shapes_frame(shape_csv)
            graph = build_op_graph(op_shapes_df)
            schedule_result = schedule_combo(combo_spec, graph, combo_pred)
            timeline_df = load_timeline_frame(timeline_csv)
            expected_node_names = [graph[node_idx].node_name for node_idx in schedule_result.expected_node_indices]
            observed_node_names = [graph[node_idx].node_name for node_idx in schedule_result.observed_node_indices]
            embedding_observation = analyze_embedding_execution(timeline_df, combo_spec.inter_threads)
        except Exception as exc:
            local_partial_rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "route": str(route),
                    "scale": str(scale),
                    "metric_scope": "pipeline_eval_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "case_id": case_id,
                    "source_name": str(source_name),
                    "combo": str(combo),
                    "batch_size": batch_size,
                    "num_indices_per_lookup": nip,
                    "inter_threads": inter_threads,
                    "expected_count": None,
                    "observed_count": int(len(group)),
                    "missing_count": None,
                    "coverage_ratio": None,
                    "predicted_observed_subgraph_us": None,
                    "actual_observed_subgraph_us": None,
                    "abs_error_us": None,
                    "ape": None,
                    "missing_node_indices": "",
                    "missing_node_names": "",
                }
            )
            return local_full_rows, local_partial_rows, local_embedding_rows
        local_embedding_rows.append(
            {
                "policy": str(policy),
                "dataset": str(dataset),
                "route": str(route),
                "scale": str(scale),
                "case_id": case_id,
                "source_name": str(source_name),
                "combo": str(combo),
                "batch_size": batch_size,
                "num_indices_per_lookup": nip,
                "inter_threads": inter_threads,
                "kept_batch_count": embedding_observation.kept_batch_count,
                "launch_order": _format_launch_order(embedding_observation.representative_launch_order),
                "matches_fifo": embedding_observation.all_kept_batches_match_fifo,
                "max_gather_concurrency": embedding_observation.representative_max_gather_concurrency,
                "matches_inter_threads": embedding_observation.all_kept_batches_match_inter_threads,
                "handoff_gap_mean_us": embedding_observation.handoff_gap_mean_us,
                "handoff_gap_min_us": embedding_observation.handoff_gap_min_us,
                "handoff_gap_max_us": embedding_observation.handoff_gap_max_us,
                "tail_start_gap_mean_us": embedding_observation.tail_start_gap_mean_us,
                "tail_start_gap_min_us": embedding_observation.tail_start_gap_min_us,
                "tail_start_gap_max_us": embedding_observation.tail_start_gap_max_us,
            }
        )

        try:
            actual_observed = compute_mean_batch_span(timeline_df, observed_node_names)
        except Exception as exc:
            local_partial_rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "route": str(route),
                    "scale": str(scale),
                    "metric_scope": "timeline_eval_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "case_id": case_id,
                    "source_name": str(source_name),
                    "combo": str(combo),
                    "batch_size": batch_size,
                    "num_indices_per_lookup": nip,
                    "inter_threads": inter_threads,
                    "expected_count": int(schedule_result.coverage.expected_count),
                    "observed_count": int(schedule_result.coverage.observed_count),
                    "missing_count": int(schedule_result.coverage.missing_count),
                    "coverage_ratio": float(schedule_result.coverage.coverage_ratio),
                    "predicted_observed_subgraph_us": None,
                    "actual_observed_subgraph_us": None,
                    "abs_error_us": None,
                    "ape": None,
                    "missing_node_indices": "|".join(str(node_idx) for node_idx in schedule_result.coverage.missing_node_indices),
                    "missing_node_names": "|".join(graph[node_idx].node_name for node_idx in schedule_result.coverage.missing_node_indices),
                }
            )
            return local_full_rows, local_partial_rows, local_embedding_rows
        if e2e_graph_scope == E2E_GRAPH_SCOPE_OBSERVABLE:
            predicted_us = float(schedule_result.predicted_observed_subgraph_us)
            actual_us = float(actual_observed.mean_span_us)
            local_full_rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "route": str(route),
                    "scale": str(scale),
                    "metric_scope": "observable_profiled_e2e",
                    "case_id": case_id,
                    "source_name": str(source_name),
                    "combo": str(combo),
                    "batch_size": batch_size,
                    "num_indices_per_lookup": nip,
                    "inter_threads": inter_threads,
                    "expected_count": int(len(schedule_result.observed_node_indices)),
                    "observed_count": int(len(schedule_result.observed_node_indices)),
                    "coverage_ratio": 1.0,
                    "predicted_e2e_us": predicted_us,
                    "actual_e2e_us": actual_us,
                    "abs_error_us": abs(predicted_us - actual_us),
                    "ape": abs(predicted_us - actual_us) / actual_us if actual_us else 0.0,
                    "bottom_end_us": float(schedule_result.bottom_end_us),
                    "branch_pool_end_us": float(schedule_result.branch_pool_end_us),
                    "tail_barrier_us": float(schedule_result.tail_barrier_us),
                }
            )
        elif schedule_result.coverage.is_full_graph:
            try:
                actual_full = compute_mean_batch_span(timeline_df, expected_node_names)
            except Exception as exc:
                local_partial_rows.append(
                    {
                        "policy": str(policy),
                        "dataset": str(dataset),
                        "route": str(route),
                        "scale": str(scale),
                        "metric_scope": "timeline_eval_error",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "case_id": case_id,
                        "source_name": str(source_name),
                        "combo": str(combo),
                        "batch_size": batch_size,
                        "num_indices_per_lookup": nip,
                        "inter_threads": inter_threads,
                        "expected_count": int(schedule_result.coverage.expected_count),
                        "observed_count": int(schedule_result.coverage.observed_count),
                        "missing_count": int(schedule_result.coverage.missing_count),
                        "coverage_ratio": float(schedule_result.coverage.coverage_ratio),
                        "predicted_observed_subgraph_us": float(schedule_result.predicted_observed_subgraph_us),
                        "actual_observed_subgraph_us": float(actual_observed.mean_span_us),
                        "abs_error_us": None,
                        "ape": None,
                        "missing_node_indices": "",
                        "missing_node_names": "",
                    }
                )
                return local_full_rows, local_partial_rows, local_embedding_rows
            predicted_us = float(schedule_result.predicted_full_graph_us)
            actual_us = float(actual_full.mean_span_us)
            local_full_rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "route": str(route),
                    "scale": str(scale),
                    "metric_scope": "full_graph_e2e",
                    "case_id": case_id,
                    "source_name": str(source_name),
                    "combo": str(combo),
                    "batch_size": batch_size,
                    "num_indices_per_lookup": nip,
                    "inter_threads": inter_threads,
                    "expected_count": int(schedule_result.coverage.expected_count),
                    "observed_count": int(schedule_result.coverage.observed_count),
                    "coverage_ratio": float(schedule_result.coverage.coverage_ratio),
                    "predicted_e2e_us": predicted_us,
                    "actual_e2e_us": actual_us,
                    "abs_error_us": abs(predicted_us - actual_us),
                    "ape": abs(predicted_us - actual_us) / actual_us if actual_us else 0.0,
                    "bottom_end_us": float(schedule_result.bottom_end_us),
                    "branch_pool_end_us": float(schedule_result.branch_pool_end_us),
                    "tail_barrier_us": float(schedule_result.tail_barrier_us),
                }
            )
        else:
            predicted_us = float(schedule_result.predicted_observed_subgraph_us)
            actual_us = float(actual_observed.mean_span_us)
            missing_names = [graph[node_idx].node_name for node_idx in schedule_result.coverage.missing_node_indices]
            local_partial_rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "route": str(route),
                    "scale": str(scale),
                    "metric_scope": "observed_subgraph_non_e2e",
                    "reason": "partial_combo",
                    "case_id": case_id,
                    "source_name": str(source_name),
                    "combo": str(combo),
                    "batch_size": batch_size,
                    "num_indices_per_lookup": nip,
                    "inter_threads": inter_threads,
                    "expected_count": int(schedule_result.coverage.expected_count),
                    "observed_count": int(schedule_result.coverage.observed_count),
                    "missing_count": int(schedule_result.coverage.missing_count),
                    "coverage_ratio": float(schedule_result.coverage.coverage_ratio),
                    "predicted_observed_subgraph_us": predicted_us,
                    "actual_observed_subgraph_us": actual_us,
                    "abs_error_us": abs(predicted_us - actual_us),
                    "ape": abs(predicted_us - actual_us) / actual_us if actual_us else 0.0,
                    "missing_node_indices": "|".join(str(node_idx) for node_idx in schedule_result.coverage.missing_node_indices),
                    "missing_node_names": "|".join(missing_names),
                }
            )
        return local_full_rows, local_partial_rows, local_embedding_rows

    max_workers = max(1, min(8, len(grouped_items)))
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_evaluate_combo_group, item) for item in grouped_items]
        for processed_count, future in enumerate(cf.as_completed(futures), start=1):
            local_full_rows, local_partial_rows, local_embedding_rows = future.result()
            if local_full_rows:
                full_rows.extend(local_full_rows)
            if local_partial_rows:
                partial_rows.extend(local_partial_rows)
            if local_embedding_rows:
                embedding_rows.extend(local_embedding_rows)
            if processed_count % 100 == 0 or processed_count == len(grouped_items):
                print(
                    f"[pipeline_eval] evaluate_source_test_pipeline policy={policy} dataset={dataset} "
                    f"combos={processed_count}/{len(grouped_items)}",
                    flush=True,
                )

    full_df = pd.DataFrame(full_rows, columns=combo_columns)
    partial_df = pd.DataFrame(partial_rows, columns=partial_columns)
    embedding_df = pd.DataFrame(embedding_rows, columns=embedding_columns)

    summary_rows: list[dict[str, Any]] = []
    summary_groups = work[["route", "scale"]].drop_duplicates().sort_values(["route", "scale"])
    for row in summary_groups.itertuples(index=False):
        full_subset = full_df.loc[(full_df["route"] == row.route) & (full_df["scale"] == row.scale)].reset_index(drop=True)
        partial_subset = partial_df.loc[(partial_df["route"] == row.route) & (partial_df["scale"] == row.scale)].reset_index(drop=True)
        coverage_values = []
        if not full_subset.empty:
            coverage_values.extend(full_subset["coverage_ratio"].dropna().tolist())
        if not partial_subset.empty:
            coverage_values.extend(partial_subset["coverage_ratio"].dropna().tolist())
        metrics = _metric_summary(full_subset)
        summary_rows.append(
            {
                "policy": str(policy),
                "dataset": str(dataset),
                "route": str(row.route),
                "scale": str(row.scale),
                "combo_count_full": int(metrics["combo_count_full"]),
                "combo_count_partial": int(len(partial_subset)),
                "coverage_ratio_mean": float(sum(coverage_values) / len(coverage_values)) if coverage_values else None,
                "e2e_mae_us": metrics["e2e_mae_us"],
                "e2e_mape": metrics["e2e_mape"],
                "e2e_p90": metrics["e2e_p90"],
                "e2e_median": metrics["e2e_median"],
            }
        )

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    return (
        summary_df.sort_values(["route", "scale"]).reset_index(drop=True) if not summary_df.empty else _empty_frame(summary_columns),
        full_df.sort_values(["route", "scale", "case_id", "batch_size", "num_indices_per_lookup"]).reset_index(drop=True)
        if not full_df.empty
        else _empty_frame(combo_columns),
        partial_df.sort_values(["route", "scale", "case_id", "batch_size", "num_indices_per_lookup"]).reset_index(drop=True)
        if not partial_df.empty
        else _empty_frame(partial_columns),
        embedding_df.sort_values(["route", "scale", "case_id", "batch_size", "num_indices_per_lookup"]).reset_index(drop=True)
        if not embedding_df.empty
        else _empty_frame(embedding_columns),
        audit,
    )


def best_e2e_pipeline_by_policy(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    ordered = summary.copy()
    ordered["scale_num"] = ordered["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    ordered = ordered.sort_values(["policy", "dataset", "route", "e2e_mape", "e2e_p90", "scale_num"], na_position="last")
    best = ordered.groupby(["policy", "dataset", "route"], as_index=False).first()
    return best.drop(columns=["scale_num"]).reset_index(drop=True)
