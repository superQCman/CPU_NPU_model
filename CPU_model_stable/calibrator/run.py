from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

try:
    import yaml
except ImportError:  # pragma: no cover - deployment portability
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPLIT_AUGMENT_CACHE_ROOT = Path(
    os.environ.get(
        "CROSS_HW_AUGMENT_CACHE_ROOT",
        str(Path(tempfile.gettempdir()) / "cross_hw_v2_augmented_split_cache"),
    )
)
E2E_AUGMENT_WORKER_ENV = "CROSS_HW_V2_E2E_AUGMENT_WORKERS"
E2E_AUGMENT_AUTO_WORKER_CAP = 8
E2E_AUGMENT_PARALLEL_MIN_ROWS = 50_000
SPLIT_AUGMENT_CACHE_DEPENDENCIES = [
    SCRIPT_DIR / "static_feature_contract.py",
    SCRIPT_DIR / "graph_static_features.py",
]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytical_base import run as base_run
try:
    from .ml_semantic_routes import (
        ML_FAMILY_RF_ROUTE,
        ML_RF_ROUTE,
        ML_RF_SHAPE_TREE_ROUTE,
        WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_RF_ROUTE,
        WHITEBOX_ORT_KERNEL_RF_ROUTE,
        WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
        WHITEBOX_WORKLOAD_BENCHMARK_ROUTE,
        family_bucket_for_op_type,
        fit_semantic_thread_family_bundle,
        fit_semantic_thread_bundle,
        predict_semantic_thread_family_bundle,
        predict_semantic_thread_bundle,
        prepare_semantic_static_frame,
        route_stage,
    )
    from .static_feature_contract import augment_static_feature_frame
    from .whitebox_family_routes import (
        HARD_POLICIES as WHITEBOX_HARD_POLICIES,
        WHITEBOX_PRIOR_ONLY_ROUTE,
        WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
        WHITEBOX_EMBEDDING_MEM_RF_MODE,
        WHITEBOX_PHYSICS_GUARDED_RF_MODE,
        WHITEBOX_GATHER_SLICE_COMPACT_RF_MODE,
        WHITEBOX_GATHER_SLICE_RF_MODE,
        WHITEBOX_ORT_KERNEL_RF_MODE,
        WHITEBOX_WORKLOAD_BENCHMARK_MODE,
        WHITEBOX_WORKLOAD_BENCHMARK_VARIANTS,
        benchmark_variant_route_name,
        fit_op_thread_scale_config,
        fit_whitebox_family_embedding_mem_rf_bundle,
        fit_whitebox_family_gather_slice_compact_rf_bundle,
        fit_whitebox_family_gather_slice_rf_bundle,
        fit_whitebox_family_ort_kernel_rf_bundle,
        fit_whitebox_family_physics_guarded_rf_bundle,
        fit_whitebox_family_prior_bundle,
        fit_whitebox_family_simple_rf_bundle,
        fit_whitebox_family_workload_rf_benchmark_bundle,
        load_op_thread_scale_config,
        op_thread_scale_coefficients_frame,
        predict_whitebox_family_prior,
        predict_whitebox_family_physics_guarded_rf,
        predict_whitebox_family_simple_rf,
        save_op_thread_scale_config,
    )
    from .pipeline_eval import (
        best_e2e_pipeline_by_policy,
        evaluate_source_test_pipeline,
    )
except ImportError:
    from ml_semantic_routes import (
        ML_FAMILY_RF_ROUTE,
        ML_RF_ROUTE,
        ML_RF_SHAPE_TREE_ROUTE,
        WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
        WHITEBOX_GATHER_SLICE_RF_ROUTE,
        WHITEBOX_ORT_KERNEL_RF_ROUTE,
        WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
        WHITEBOX_WORKLOAD_BENCHMARK_ROUTE,
        family_bucket_for_op_type,
        fit_semantic_thread_family_bundle,
        fit_semantic_thread_bundle,
        predict_semantic_thread_family_bundle,
        predict_semantic_thread_bundle,
        prepare_semantic_static_frame,
        route_stage,
    )
    from static_feature_contract import augment_static_feature_frame  # type: ignore
    from whitebox_family_routes import (
        HARD_POLICIES as WHITEBOX_HARD_POLICIES,
        WHITEBOX_PRIOR_ONLY_ROUTE,
        WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
        WHITEBOX_EMBEDDING_MEM_RF_MODE,
        WHITEBOX_PHYSICS_GUARDED_RF_MODE,
        WHITEBOX_GATHER_SLICE_COMPACT_RF_MODE,
        WHITEBOX_GATHER_SLICE_RF_MODE,
        WHITEBOX_ORT_KERNEL_RF_MODE,
        WHITEBOX_WORKLOAD_BENCHMARK_MODE,
        WHITEBOX_WORKLOAD_BENCHMARK_VARIANTS,
        benchmark_variant_route_name,
        fit_op_thread_scale_config,
        fit_whitebox_family_embedding_mem_rf_bundle,
        fit_whitebox_family_gather_slice_compact_rf_bundle,
        fit_whitebox_family_gather_slice_rf_bundle,
        fit_whitebox_family_ort_kernel_rf_bundle,
        fit_whitebox_family_physics_guarded_rf_bundle,
        fit_whitebox_family_prior_bundle,
        fit_whitebox_family_simple_rf_bundle,
        fit_whitebox_family_workload_rf_benchmark_bundle,
        load_op_thread_scale_config,
        op_thread_scale_coefficients_frame,
        predict_whitebox_family_prior,
        predict_whitebox_family_physics_guarded_rf,
        predict_whitebox_family_simple_rf,
        save_op_thread_scale_config,
    )
    from pipeline_eval import (  # type: ignore
        best_e2e_pipeline_by_policy,
        evaluate_source_test_pipeline,
    )


DEFAULTS = json.loads((SCRIPT_DIR / "configs" / "defaults.json").read_text(encoding="utf-8"))
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_ARTIFACT_ROOT = SCRIPT_DIR / "artifacts" / "latest"
DEFAULT_SOURCE_HW = (SCRIPT_DIR / DEFAULTS["default_assets"]["source_hw"]).resolve()
DEFAULT_TARGET_HW = (SCRIPT_DIR / DEFAULTS["default_assets"]["target_hw"]).resolve()
DEFAULT_SCALES = [int(value) for value in DEFAULTS["default_scales"]]
DEFAULT_ROUTES = list(DEFAULTS["routes"])
WHITEBOX_ROUTES = list(DEFAULTS.get("whitebox_routes", []))
HARD_POLICY_EXTRA_ROUTES = list(DEFAULTS.get("hard_policy_routes", [ML_FAMILY_RF_ROUTE, *WHITEBOX_ROUTES]))
BENCHMARK_ROUTE_BASES = [WHITEBOX_WORKLOAD_BENCHMARK_ROUTE]
HARD_POLICY_ONLY_ROUTES = list(
    dict.fromkeys(
        [
            *WHITEBOX_ROUTES,
            WHITEBOX_ORT_KERNEL_RF_ROUTE,
            WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
            WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
            WHITEBOX_GATHER_SLICE_RF_ROUTE,
            WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
            *BENCHMARK_ROUTE_BASES,
        ]
    )
)
LEGACY_POLICIES = list(DEFAULTS["split_policies"])
PHASE2_POLICIES = list(DEFAULTS.get("phase2_split_policies", []))
JOINT_POLICIES = list(DEFAULTS.get("joint_split_policies", ["joint_6d_tuple"]))
FORMAL_POLICIES = [*LEGACY_POLICIES, *PHASE2_POLICIES]
POLICY_CHOICES = ["all", "phase2", *LEGACY_POLICIES, *PHASE2_POLICIES, *JOINT_POLICIES]
SOURCE_DATASET_NAME = "source_kunpeng920_dataset3"
TARGET_DATASET_NAME = "target_s920x20_dataset_all_78910_2_nodrop"
SOURCE_SCHEDULER_SCOPE = str(DEFAULTS["source_scheduler_20260419"]["phase_scope"])
PHASE2_SCOPE = str(DEFAULTS["thread_decoupling_phase2"]["phase_scope"])
SOURCE_RAW_ROOT = (PROJECT_ROOT / DEFAULTS["source_scheduler_20260419"]["source_output_root"]).resolve()
LEGACY_SOURCE_RAW_ROOT = (SCRIPT_DIR / DEFAULTS["default_assets"]["source_dataset"]).resolve()
PHASE2_SOURCE_ROOT = (SCRIPT_DIR / DEFAULTS["thread_decoupling_phase2"]["source_output_root"]).resolve()
LEGACY_INVALID_CASE_IDS = {"case_8_4_4", "case_9_4_4", "case_10_4_6"}
INVALID_LEGACY_ROOT = (DEFAULT_DATA_ROOT / "_invalid_legacy_splits_20260420").resolve()
LEGACY_V3_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "cross_hw_unseen_axis_calibrator"
    / "artifacts"
    / "whitebox_zero_fit_formal_parallel_20260421_v3"
).resolve()
SOURCE_OPERATOR_BASELINE_SUMMARY = (
    SCRIPT_DIR
    / "artifacts"
    / "joint_6d_tuple_small_1_16_v5_gather_slice_e2e_test"
    / "summary.csv"
).resolve()
NOISE_METRIC_COLUMNS = [
    "profile_kept_cv",
    "profile_kept_range_ratio",
    "profile_last2_cv",
    "profile_last2_range_ratio",
]


def _whitebox_route_metadata(route: str) -> tuple[str, bool]:
    if route == WHITEBOX_PRIOR_ONLY_ROUTE:
        return ("pure_zero_fit", False)
    if route == WHITEBOX_PRIOR_SIMPLE_RF_ROUTE:
        return ("whitebox_perf_feature_rf", True)
    if route == WHITEBOX_PHYSICS_GUARDED_RF_ROUTE:
        return (WHITEBOX_PHYSICS_GUARDED_RF_MODE, True)
    if route == WHITEBOX_ORT_KERNEL_RF_ROUTE:
        return (WHITEBOX_ORT_KERNEL_RF_MODE, True)
    if route == WHITEBOX_EMBEDDING_MEM_RF_ROUTE:
        return (WHITEBOX_EMBEDDING_MEM_RF_MODE, True)
    if route == WHITEBOX_GATHER_SLICE_RF_ROUTE:
        return (WHITEBOX_GATHER_SLICE_RF_MODE, True)
    if route == WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE:
        return (WHITEBOX_GATHER_SLICE_COMPACT_RF_MODE, True)
    if str(route).startswith(f"{WHITEBOX_WORKLOAD_BENCHMARK_ROUTE}["):
        return (WHITEBOX_WORKLOAD_BENCHMARK_MODE, True)
    return ("not_whitebox", False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-hardware unseen-axis analytical calibrator experiments.")
    parser.add_argument("--policy", default="all", choices=POLICY_CHOICES)
    parser.add_argument("--datasets", default="source", choices=["source", "target", "both"])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--source-hw-config", default=str(DEFAULT_SOURCE_HW))
    parser.add_argument("--target-hw-config", default=str(DEFAULT_TARGET_HW))
    parser.add_argument("--scales", default=",".join(str(value) for value in DEFAULT_SCALES))
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--op-thread-scale-mode",
        default="off",
        choices=["off", "fit", "load"],
        help="Apply an op_type x num_threads analytical prior scale: off, fit from the current fit split, or load from JSON.",
    )
    parser.add_argument(
        "--op-thread-scale-config",
        default="",
        help="JSON config path used when --op-thread-scale-mode=load.",
    )
    parser.add_argument(
        "--op-thread-scale-output",
        default="",
        help="JSON output path used when --op-thread-scale-mode=fit; defaults to artifact_root/op_thread_scale_config_<policy>_<dataset>_<scale>.json.",
    )
    parser.add_argument(
        "--op-thread-scale-min-rows",
        type=int,
        default=4,
        help="Minimum rows per op_type x num_threads group when fitting op-thread scale.",
    )
    parser.add_argument(
        "--enable-scaling-audit",
        action="store_true",
        help="Run the expensive family_scaling_audit prepass before evaluation.",
    )
    parser.add_argument(
        "--skip-e2e-pipeline",
        action="store_true",
        help="Skip E2E/pipeline split loading and scheduling; run only single-op train/test metrics.",
    )
    parser.add_argument(
        "--enable-root-cause-oracles",
        action="store_true",
        help="Run E2E operator oracle diagnostics for Gather/ReduceSum prediction replacement.",
    )
    parser.add_argument(
        "--export-model-bundles",
        action="store_true",
        help="Export trained model bundles for offline static DLRM-config inference.",
    )
    parser.add_argument(
        "--model-bundle-output-dir",
        default="",
        help="Directory for exported model bundles; defaults to artifact_root/model_bundles.",
    )
    return parser.parse_args()


def _selected_policies(policy_arg: str) -> list[str]:
    if policy_arg == "all":
        return list(FORMAL_POLICIES)
    if policy_arg == "phase2":
        return list(PHASE2_POLICIES)
    return [str(policy_arg)]


def _selected_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "both":
        return ["source", "target"]
    return [str(dataset_arg)]


def _progress(
    stage: str,
    *,
    policy: str | None = None,
    dataset: str | None = None,
    scale: int | None = None,
    detail: str | None = None,
) -> None:
    parts = [time.strftime("%H:%M:%S"), stage]
    if policy:
        parts.append(f"policy={policy}")
    if dataset:
        parts.append(f"dataset={dataset}")
    if scale is not None:
        parts.append(f"scale={int(scale)}")
    if detail:
        parts.append(str(detail))
    print("[progress] " + " | ".join(parts), flush=True)


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _stat_fingerprint(path: Path) -> str:
    if not path.exists():
        return f"{path.resolve()}:missing"
    stat = path.stat()
    return f"{path.resolve()}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}"


def _augmented_split_cache_path(dataset_root: Path, split: str) -> Path:
    parts = [
        _stat_fingerprint(dataset_root / f"{split}.csv"),
        _stat_fingerprint(dataset_root / "split_manifest.json"),
        *[_stat_fingerprint(path) for path in SPLIT_AUGMENT_CACHE_DEPENDENCIES],
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]
    cache_dir = SPLIT_AUGMENT_CACHE_ROOT / dataset_root.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{split}_{digest}.pkl"


def _resolve_e2e_augment_workers(row_count: int, group_count: int) -> int:
    if row_count < E2E_AUGMENT_PARALLEL_MIN_ROWS or group_count <= 1:
        return 1
    raw_value = str(os.environ.get(E2E_AUGMENT_WORKER_ENV, "0")).strip()
    if raw_value in {"", "0", "auto"}:
        requested = min(os.cpu_count() or 1, E2E_AUGMENT_AUTO_WORKER_CAP)
    else:
        requested = int(raw_value)
    if requested < 0:
        raise ValueError(f"{E2E_AUGMENT_WORKER_ENV} must be >= 0, got {raw_value}")
    if requested == 0:
        requested = min(os.cpu_count() or 1, E2E_AUGMENT_AUTO_WORKER_CAP)
    return max(1, min(int(requested), int(group_count)))


def _e2e_augment_chunks(frame: pd.DataFrame, worker_count: int) -> list[pd.DataFrame]:
    if worker_count <= 1:
        return [frame.copy()]
    groups = frame.groupby(["source_name", "combo"], sort=False).indices
    buckets: list[list[int]] = [[] for _ in range(worker_count)]
    bucket_sizes = [0 for _ in range(worker_count)]
    for _, indices in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        bucket_idx = min(range(worker_count), key=lambda idx: bucket_sizes[idx])
        index_list = [int(value) for value in indices]
        buckets[bucket_idx].extend(index_list)
        bucket_sizes[bucket_idx] += len(index_list)
    return [frame.iloc[sorted(indices)].copy() for indices in buckets if indices]


def _augment_static_split_chunk(task: tuple[pd.DataFrame, str]) -> pd.DataFrame:
    chunk, dataset_root_text = task
    return augment_static_feature_frame(chunk, dataset_root=Path(dataset_root_text))


def _augment_split_frame(frame: pd.DataFrame, *, dataset_root: Path, split: str) -> pd.DataFrame:
    if not split.startswith("e2e_") or not {"source_name", "combo"}.issubset(frame.columns):
        return augment_static_feature_frame(frame, dataset_root=dataset_root)

    group_count = int(frame[["source_name", "combo"]].drop_duplicates().shape[0])
    worker_count = _resolve_e2e_augment_workers(len(frame), group_count)
    if worker_count <= 1:
        return augment_static_feature_frame(frame, dataset_root=dataset_root)

    ordered = frame.copy()
    ordered["__augment_order"] = np.arange(len(ordered), dtype=np.int64)
    chunks = _e2e_augment_chunks(ordered, worker_count)
    _progress(
        "augment_split_parallel_start",
        dataset=str(dataset_root.name),
        detail=f"{split}:rows={len(frame)} combos={group_count} workers={len(chunks)}",
    )
    augmented_chunks: list[pd.DataFrame] = []
    with cf.ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(_augment_static_split_chunk, (chunk, str(dataset_root.resolve())))
            for chunk in chunks
        ]
        for processed_count, future in enumerate(cf.as_completed(futures), start=1):
            augmented_chunks.append(future.result())
            _progress(
                "augment_split_parallel_progress",
                dataset=str(dataset_root.name),
                detail=f"{split}:chunks={processed_count}/{len(chunks)}",
            )
    augmented = pd.concat(augmented_chunks, ignore_index=True, sort=False)
    if "__augment_order" in augmented.columns:
        augmented = augmented.sort_values("__augment_order", kind="stable").drop(columns=["__augment_order"]).reset_index(drop=True)
    return augmented


def _effective_routes_for_policy(policy: str, requested_routes: list[str]) -> list[str]:
    base_routes = [route for route in requested_routes if route]
    if policy not in WHITEBOX_HARD_POLICIES:
        return list(dict.fromkeys(route for route in base_routes if route not in HARD_POLICY_ONLY_ROUTES))
    if WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE in base_routes and WHITEBOX_GATHER_SLICE_RF_ROUTE not in base_routes:
        base_routes.append(WHITEBOX_GATHER_SLICE_RF_ROUTE)
    if WHITEBOX_GATHER_SLICE_RF_ROUTE in base_routes and WHITEBOX_EMBEDDING_MEM_RF_ROUTE not in base_routes:
        base_routes.append(WHITEBOX_EMBEDDING_MEM_RF_ROUTE)
    return list(dict.fromkeys([*base_routes, *HARD_POLICY_EXTRA_ROUTES]))


def _benchmark_variant_routes() -> list[str]:
    return [benchmark_variant_route_name(variant) for variant in WHITEBOX_WORKLOAD_BENCHMARK_VARIANTS]


def _is_benchmark_route(route: str) -> bool:
    return str(route).startswith(f"{WHITEBOX_WORKLOAD_BENCHMARK_ROUTE}[")


def _v3_vs_v2_gap_report(current_combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in FORMAL_POLICIES:
        legacy_summary_path = LEGACY_V3_ARTIFACT_ROOT / policy / "summary.csv"
        if not legacy_summary_path.exists():
            continue
        try:
            legacy_summary = pd.read_csv(legacy_summary_path, low_memory=False)
        except Exception:
            continue
        legacy_route = legacy_summary.loc[
            legacy_summary["route"].astype(str).eq("whitebox_family_prior_plus_simple_rf")
        ].copy()
        if legacy_route.empty:
            continue
        legacy_best = legacy_route.sort_values(["mape", "p90"]).iloc[0]

        current_rf = current_combined.loc[
            current_combined["policy"].astype(str).eq(policy)
            & current_combined["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)
        ].copy()
        current_prior = current_combined.loc[
            current_combined["policy"].astype(str).eq(policy)
            & current_combined["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)
        ].copy()
        if current_rf.empty or current_prior.empty:
            continue
        current_rf_best = current_rf.sort_values(["mape", "p90"]).iloc[0]
        current_prior_best = current_prior.sort_values(["mape", "p90"]).iloc[0]
        rows.append(
            {
                "policy": str(policy),
                "v3_best_scale": str(legacy_best.get("scale", "")),
                "v3_best_mape": float(legacy_best.get("mape", 0.0)),
                "v3_best_p90": float(legacy_best.get("p90", 0.0)),
                "v2_best_scale": str(current_rf_best.get("scale", "")),
                "v2_best_mape": float(current_rf_best.get("mape", 0.0)),
                "v2_best_p90": float(current_rf_best.get("p90", 0.0)),
                "prior_only_best_scale": str(current_prior_best.get("scale", "")),
                "prior_only_best_mape": float(current_prior_best.get("mape", 0.0)),
                "prior_only_best_p90": float(current_prior_best.get("p90", 0.0)),
                "v3_minus_v2_gap_pp": float((float(current_rf_best.get("mape", 0.0)) - float(legacy_best.get("mape", 0.0))) * 100.0),
                "v2_rf_gain_pp": float((float(current_prior_best.get("mape", 0.0)) - float(current_rf_best.get("mape", 0.0))) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def _mainline_vs_shadow_summary(summary: pd.DataFrame, prior_gap: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or prior_gap.empty:
        return pd.DataFrame()
    prior_summary = summary.loc[summary["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)].copy()
    if prior_summary.empty:
        return pd.DataFrame()
    prior_summary["scale_num"] = prior_summary["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    prior_best = (
        prior_summary.sort_values(["policy", "dataset", "mape", "p90", "scale_num"])
        .groupby(["policy", "dataset"], as_index=False)
        .first()
        .rename(
            columns={
                "scale": "prior_only_best_scale",
                "mape": "prior_only_mape",
                "p90": "prior_only_p90",
            }
        )
    )
    final_summary = summary.loc[summary["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)].copy()
    final_best = pd.DataFrame()
    if not final_summary.empty:
        final_summary["scale_num"] = final_summary["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        final_best = (
            final_summary.sort_values(["policy", "dataset", "mape", "p90", "scale_num"])
            .groupby(["policy", "dataset"], as_index=False)
            .first()
            .rename(
                columns={
                    "scale": "final_best_scale",
                    "mape": "final_mape",
                    "p90": "final_p90",
                }
            )
        )
    shadow = prior_gap.loc[
        prior_gap["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)
        & prior_gap["slice_kind"].astype(str).eq("overall")
    ].copy()
    if shadow.empty:
        return pd.DataFrame()
    shadow = shadow.rename(columns={"scale": "prior_only_best_scale"})
    merged = prior_best.merge(
        shadow[
            [
                "policy",
                "dataset",
                "prior_only_best_scale",
                "kernel_only_mape",
                "state_adjusted_mape",
                "phase_adjusted_mape",
                "final_prior_mape",
                "state_gain_pp",
                "phase_hypothesis_gain_pp",
                "composition_gain_pp",
                "remaining_gap_pp",
            ]
        ],
        on=["policy", "dataset", "prior_only_best_scale"],
        how="left",
    )
    if not final_best.empty:
        merged = merged.merge(
            final_best[["policy", "dataset", "final_best_scale", "final_mape", "final_p90"]],
            on=["policy", "dataset"],
            how="left",
        )
    return merged[
        [
            "policy",
            "dataset",
            "prior_only_best_scale",
            "prior_only_mape",
            "prior_only_p90",
            "kernel_only_mape",
            "state_adjusted_mape",
            "phase_adjusted_mape",
            "final_prior_mape",
            "state_gain_pp",
            "phase_hypothesis_gain_pp",
            "composition_gain_pp",
            "remaining_gap_pp",
            *(
                ["final_best_scale", "final_mape", "final_p90"]
                if {"final_best_scale", "final_mape", "final_p90"}.issubset(set(merged.columns))
                else []
            ),
        ]
    ].reset_index(drop=True)


def _prediction_error_frame(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
    pred: pd.Series | list[float] | np.ndarray,
) -> pd.DataFrame:
    keep_columns = [
        "row_uid",
        "case_id",
        "source_name",
        "combo",
        "op_type",
        "node_name",
        "trace_op_name",
        "family_bucket",
        "exec_region_static",
        "submodel_role_static",
        "whitebox_prior_family",
        "whitebox_rf_group",
        "whitebox_center_us",
        "whitebox_lower_us",
        "whitebox_upper_us",
        "svc_stream_mem_center_us",
        "gather_slice_copy_bw_gb_s",
        "gather_slice_stream_us",
        "gather_slice_vs_current_stream_ratio",
        "gather_slice_underpredict_gap_us",
        "profile_kept_cv",
        "profile_kept_range_ratio",
        "profile_last2_cv",
        "profile_last2_range_ratio",
    ]
    available = [column for column in keep_columns if column in prepared_frame.columns]
    work = prepared_frame.loc[:, available].copy()
    work["policy"] = str(policy)
    work["dataset"] = str(dataset)
    work["route"] = str(route)
    work["scale"] = f"{int(scale)}%"
    actual = pd.to_numeric(prepared_frame[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred_arr = pd.to_numeric(pd.Series(pred, index=prepared_frame.index), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ape = np.abs(pred_arr - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    work["actual_us"] = actual
    work["pred_us"] = pred_arr
    work["ape"] = ape
    for column in NOISE_METRIC_COLUMNS:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work.reset_index(drop=True)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = np.clip(weights[sorter], a_min=0.0, a_max=None)
    total = float(sorted_weights.sum())
    if total <= 0.0:
        return float(np.quantile(sorted_values, q))
    cumulative = np.cumsum(sorted_weights)
    target = float(q) * total
    return float(np.interp(target, cumulative, sorted_values))


def _weighted_error_stats(ape: np.ndarray, actual_us: np.ndarray) -> dict[str, float]:
    ape_arr = np.asarray(ape, dtype=float)
    actual_arr = np.asarray(actual_us, dtype=float)
    if ape_arr.size == 0:
        return {
            "weighted_mape": 0.0,
            "weighted_p90": 0.0,
            "weighted_median": 0.0,
            "actual_us_sum": 0.0,
        }
    weights = np.clip(actual_arr, a_min=0.0, a_max=None)
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(ape_arr, dtype=float)
    return {
        "weighted_mape": float(np.average(ape_arr, weights=weights)),
        "weighted_p90": _weighted_quantile(ape_arr, weights, 0.90),
        "weighted_median": _weighted_quantile(ape_arr, weights, 0.50),
        "actual_us_sum": float(np.clip(actual_arr, a_min=0.0, a_max=None).sum()),
    }


def _mainline_baseline_summary(
    best: pd.DataFrame,
    analytical_validation: pd.DataFrame,
    v3_gap_report: pd.DataFrame,
) -> pd.DataFrame:
    final_best = best.loc[best["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)].copy()
    prior_best = best.loc[best["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)].copy()
    if final_best.empty and prior_best.empty:
        return pd.DataFrame()
    final_best = final_best.rename(
        columns={
            "scale": "final_best_scale",
            "mape": "final_mape",
            "p90": "final_p90",
        }
    )
    prior_best = prior_best.rename(
        columns={
            "scale": "prior_best_scale",
            "mape": "prior_mape",
            "p90": "prior_p90",
        }
    )
    baseline = final_best.merge(
        prior_best[["policy", "dataset", "prior_best_scale", "prior_mape", "prior_p90"]],
        on=["policy", "dataset"],
        how="outer",
    )
    if not analytical_validation.empty:
        validation = analytical_validation.loc[
            analytical_validation["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE),
            [
                "policy",
                "dataset",
                "scale",
                "coverage_rate",
                "policy_spearman",
                "stable_bottleneck_rate",
                "analytical_hw_pass_rate",
            ],
        ].rename(columns={"scale": "prior_validation_scale"})
        baseline = baseline.merge(validation, on=["policy", "dataset"], how="left")
    if not v3_gap_report.empty:
        baseline = baseline.merge(
            v3_gap_report[["policy", "v3_best_mape", "v3_best_p90", "v3_minus_v2_gap_pp"]],
            on="policy",
            how="left",
        )
    if "final_mape" in baseline.columns:
        baseline["final_gap_to_15_pp"] = (pd.to_numeric(baseline["final_mape"], errors="coerce").fillna(0.0) - 0.15) * 100.0
        baseline["final_gap_to_10_pp"] = (pd.to_numeric(baseline["final_mape"], errors="coerce").fillna(0.0) - 0.10) * 100.0
    if "prior_mape" in baseline.columns and "final_mape" in baseline.columns:
        baseline["rf_gain_pp"] = (
            pd.to_numeric(baseline["prior_mape"], errors="coerce").fillna(0.0)
            - pd.to_numeric(baseline["final_mape"], errors="coerce").fillna(0.0)
        ) * 100.0
    numeric_columns = [column for column in baseline.columns if pd.api.types.is_numeric_dtype(baseline[column])]
    if not baseline.empty and numeric_columns:
        average_row = {"policy": "__avg__", "dataset": "source"}
        for column in numeric_columns:
            average_row[column] = float(pd.to_numeric(baseline[column], errors="coerce").dropna().mean())
        baseline = pd.concat([baseline, pd.DataFrame([average_row])], ignore_index=True)
    return baseline.reset_index(drop=True)


def _mainline_blocker_dashboard(
    breakdown: pd.DataFrame,
    phase_error: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final_breakdown = breakdown.loc[breakdown["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)].copy()
    if not final_breakdown.empty:
        final_breakdown["rank_key"] = pd.to_numeric(final_breakdown["ape_sum_share"], errors="coerce").fillna(0.0)
        ordered = final_breakdown.sort_values(
            ["policy", "dataset", "rank_key", "mape"],
            ascending=[True, True, False, False],
        )
        for (policy, dataset), group in ordered.groupby(["policy", "dataset"], sort=True):
            for rank, (_, item) in enumerate(group.head(top_k).iterrows(), start=1):
                rows.append(
                    {
                        "policy": str(policy),
                        "dataset": str(dataset),
                        "blocker_source": "final_family_op",
                        "rank": int(rank),
                        "family_bucket": str(item.get("family_bucket", "")),
                        "exec_region_static": "",
                        "op_type": str(item.get("op_type", "")),
                        "rows": int(item.get("rows", 0)),
                        "mape": float(item.get("mape", 0.0)),
                        "p90": float(item.get("p90", 0.0)),
                        "ape_sum_share": float(item.get("ape_sum_share", 0.0)),
                    }
                )
    prior_phase = phase_error.loc[
        phase_error["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)
        & phase_error["slice_kind"].astype(str).eq("exec_region_op")
    ].copy()
    if not prior_phase.empty:
        prior_phase["rank_key"] = pd.to_numeric(prior_phase["ape_sum_share"], errors="coerce").fillna(0.0)
        ordered = prior_phase.sort_values(
            ["policy", "dataset", "rank_key", "mape"],
            ascending=[True, True, False, False],
        )
        for (policy, dataset), group in ordered.groupby(["policy", "dataset"], sort=True):
            for rank, (_, item) in enumerate(group.head(top_k).iterrows(), start=1):
                rows.append(
                    {
                        "policy": str(policy),
                        "dataset": str(dataset),
                        "blocker_source": "prior_exec_region_op",
                        "rank": int(rank),
                        "family_bucket": "",
                        "exec_region_static": str(item.get("exec_region_static", "")),
                        "op_type": str(item.get("op_type", "")),
                        "rows": int(item.get("rows", 0)),
                        "mape": float(item.get("mape", 0.0)),
                        "p90": float(item.get("p90", 0.0)),
                        "ape_sum_share": float(item.get("ape_sum_share", 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def _weighted_blocker_dashboard(
    breakdown: pd.DataFrame,
    phase_error: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _weighted_contrib(frame: pd.DataFrame) -> pd.Series:
        weighted_mape = pd.to_numeric(frame.get("weighted_mape", 0.0), errors="coerce").fillna(0.0)
        actual_share = pd.to_numeric(frame.get("actual_us_share", 0.0), errors="coerce").fillna(0.0)
        return weighted_mape * actual_share

    final_breakdown = breakdown.loc[breakdown["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)].copy()
    if not final_breakdown.empty:
        final_breakdown["weighted_total_error_contrib"] = _weighted_contrib(final_breakdown)
        ordered = final_breakdown.sort_values(
            ["policy", "dataset", "weighted_total_error_contrib", "weighted_mape", "mape"],
            ascending=[True, True, False, False, False],
        )
        for (policy, dataset), group in ordered.groupby(["policy", "dataset"], sort=True):
            for rank, (_, item) in enumerate(group.head(top_k).iterrows(), start=1):
                rows.append(
                    {
                        "policy": str(policy),
                        "dataset": str(dataset),
                        "blocker_source": "final_family_op_weighted",
                        "rank": int(rank),
                        "family_bucket": str(item.get("family_bucket", "")),
                        "exec_region_static": "",
                        "op_type": str(item.get("op_type", "")),
                        "rows": int(item.get("rows", 0)),
                        "mape": float(item.get("mape", 0.0)),
                        "p90": float(item.get("p90", 0.0)),
                        "weighted_mape": float(item.get("weighted_mape", 0.0)),
                        "weighted_p90": float(item.get("weighted_p90", 0.0)),
                        "actual_us_share": float(item.get("actual_us_share", 0.0)),
                        "weighted_total_error_contrib": float(item.get("weighted_total_error_contrib", 0.0)),
                        "ape_sum_share": float(item.get("ape_sum_share", 0.0)),
                    }
                )

    prior_phase = phase_error.loc[
        phase_error["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)
        & phase_error["slice_kind"].astype(str).eq("exec_region_op")
    ].copy()
    if not prior_phase.empty:
        prior_phase["weighted_total_error_contrib"] = _weighted_contrib(prior_phase)
        ordered = prior_phase.sort_values(
            ["policy", "dataset", "weighted_total_error_contrib", "weighted_mape", "mape"],
            ascending=[True, True, False, False, False],
        )
        for (policy, dataset), group in ordered.groupby(["policy", "dataset"], sort=True):
            for rank, (_, item) in enumerate(group.head(top_k).iterrows(), start=1):
                rows.append(
                    {
                        "policy": str(policy),
                        "dataset": str(dataset),
                        "blocker_source": "prior_exec_region_op_weighted",
                        "rank": int(rank),
                        "family_bucket": "",
                        "exec_region_static": str(item.get("exec_region_static", "")),
                        "op_type": str(item.get("op_type", "")),
                        "rows": int(item.get("rows", 0)),
                        "mape": float(item.get("mape", 0.0)),
                        "p90": float(item.get("p90", 0.0)),
                        "weighted_mape": float(item.get("weighted_mape", 0.0)),
                        "weighted_p90": float(item.get("weighted_p90", 0.0)),
                        "actual_us_share": float(item.get("actual_us_share", 0.0)),
                        "weighted_total_error_contrib": float(item.get("weighted_total_error_contrib", 0.0)),
                        "ape_sum_share": float(item.get("ape_sum_share", 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def _subsystem_target_scorecard(
    breakdown: pd.DataFrame,
    phase_error: pd.DataFrame,
) -> pd.DataFrame:
    def _subsystem_name(op_type: str) -> str:
        op = str(op_type)
        if op == "Gather":
            return "Gather"
        if op == "ReduceSum":
            return "ReduceSum"
        if op in {"Shape", "Reshape", "Unsqueeze", "Flatten"}:
            return "metadata_control"
        return "other"

    rows: list[dict[str, Any]] = []

    final_breakdown = breakdown.loc[breakdown["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)].copy()
    if not final_breakdown.empty:
        final_breakdown["subsystem"] = final_breakdown["op_type"].astype(str).map(_subsystem_name)
        final_breakdown = final_breakdown.loc[final_breakdown["subsystem"].isin(["Gather", "ReduceSum", "metadata_control"])].copy()
        final_breakdown["weighted_total_error_contrib"] = (
            pd.to_numeric(final_breakdown.get("weighted_mape", 0.0), errors="coerce").fillna(0.0)
            * pd.to_numeric(final_breakdown.get("actual_us_share", 0.0), errors="coerce").fillna(0.0)
        )
        grouped = (
            final_breakdown.groupby(["policy", "dataset", "subsystem"], as_index=False)
            .agg(
                rows=("rows", "sum"),
                actual_us_share=("actual_us_share", "sum"),
                ape_sum_share=("ape_sum_share", "sum"),
                weighted_total_error_contrib=("weighted_total_error_contrib", "sum"),
            )
        )
        grouped["weighted_mape"] = grouped["weighted_total_error_contrib"] / np.clip(
            pd.to_numeric(grouped["actual_us_share"], errors="coerce").fillna(0.0),
            a_min=1e-12,
            a_max=None,
        )
        grouped["route_scope"] = "final_route"
        rows.extend(grouped.to_dict(orient="records"))

    prior_phase = phase_error.loc[
        phase_error["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)
        & phase_error["slice_kind"].astype(str).eq("exec_region_op")
    ].copy()
    if not prior_phase.empty:
        prior_phase["subsystem"] = prior_phase["op_type"].astype(str).map(_subsystem_name)
        prior_phase = prior_phase.loc[prior_phase["subsystem"].isin(["Gather", "ReduceSum", "metadata_control"])].copy()
        prior_phase["weighted_total_error_contrib"] = (
            pd.to_numeric(prior_phase.get("weighted_mape", 0.0), errors="coerce").fillna(0.0)
            * pd.to_numeric(prior_phase.get("actual_us_share", 0.0), errors="coerce").fillna(0.0)
        )
        grouped = (
            prior_phase.groupby(["policy", "dataset", "subsystem"], as_index=False)
            .agg(
                rows=("rows", "sum"),
                actual_us_share=("actual_us_share", "sum"),
                ape_sum_share=("ape_sum_share", "sum"),
                weighted_total_error_contrib=("weighted_total_error_contrib", "sum"),
            )
        )
        grouped["weighted_mape"] = grouped["weighted_total_error_contrib"] / np.clip(
            pd.to_numeric(grouped["actual_us_share"], errors="coerce").fillna(0.0),
            a_min=1e-12,
            a_max=None,
        )
        grouped["route_scope"] = "prior_exec_region_op"
        rows.extend(grouped.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result[
        [
            "policy",
            "dataset",
            "route_scope",
            "subsystem",
            "rows",
            "actual_us_share",
            "weighted_mape",
            "weighted_total_error_contrib",
            "ape_sum_share",
        ]
    ].sort_values(["policy", "dataset", "route_scope", "weighted_total_error_contrib"], ascending=[True, True, True, False])
    return result.reset_index(drop=True)


def _noise_by_policy(row_errors: pd.DataFrame) -> pd.DataFrame:
    if row_errors.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (policy, dataset, route, scale), group in row_errors.groupby(["policy", "dataset", "route", "scale"], sort=True):
        row = {
            "policy": str(policy),
            "dataset": str(dataset),
            "route": str(route),
            "scale": str(scale),
            "rows": int(len(group)),
            "mape": float(pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).mean()),
            "p90": float(pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).quantile(0.90)),
        }
        ape_series = pd.to_numeric(group["ape"], errors="coerce")
        for metric in NOISE_METRIC_COLUMNS:
            values = pd.to_numeric(group.get(metric, pd.Series(np.nan, index=group.index)), errors="coerce")
            valid = values.dropna()
            row[f"{metric}_coverage"] = float(values.notna().mean()) if len(values) else 0.0
            row[f"{metric}_median"] = float(valid.median()) if not valid.empty else np.nan
            row[f"{metric}_p90"] = float(valid.quantile(0.90)) if not valid.empty else np.nan
            row[f"{metric}_corr_to_ape"] = float(values.corr(ape_series, method="spearman")) if valid.size >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _noise_by_op_region(row_errors: pd.DataFrame) -> pd.DataFrame:
    if row_errors.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    final_rows = row_errors.loc[row_errors["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)].copy()
    if final_rows.empty:
        final_rows = row_errors.copy()
    total_ape = final_rows.groupby(["policy", "dataset", "scale"], as_index=False)["ape"].sum().rename(columns={"ape": "total_ape"})
    grouped = final_rows.groupby(["policy", "dataset", "scale", "exec_region_static", "op_type"], dropna=False, sort=True)
    for keys, group in grouped:
        policy, dataset, scale, exec_region, op_type = keys
        row = {
            "policy": str(policy),
            "dataset": str(dataset),
            "scale": str(scale),
            "exec_region_static": str(exec_region),
            "op_type": str(op_type),
            "rows": int(len(group)),
            "mape": float(pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).mean()),
            "p90": float(pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).quantile(0.90)),
        }
        for metric in NOISE_METRIC_COLUMNS:
            values = pd.to_numeric(group.get(metric, pd.Series(np.nan, index=group.index)), errors="coerce").dropna()
            row[f"{metric}_median"] = float(values.median()) if not values.empty else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.merge(total_ape, on=["policy", "dataset", "scale"], how="left")
    grouped_ape = (
        final_rows.groupby(["policy", "dataset", "scale", "exec_region_static", "op_type"], dropna=False, sort=True)["ape"]
        .sum()
        .reset_index()
        .rename(columns={"ape": "group_ape"})
    )
    result = result.merge(grouped_ape, on=["policy", "dataset", "scale", "exec_region_static", "op_type"], how="left")
    result["ape_sum_share"] = pd.to_numeric(result["group_ape"], errors="coerce").fillna(0.0) / np.clip(
        pd.to_numeric(result["total_ape"], errors="coerce").fillna(0.0),
        a_min=1e-12,
        a_max=None,
    )
    result = result.drop(columns=["total_ape", "group_ape"])
    return result.reset_index(drop=True)


def _global_filter_sweep(row_errors: pd.DataFrame) -> pd.DataFrame:
    if row_errors.empty:
        return pd.DataFrame()
    final_rows = row_errors.loc[row_errors["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE)].copy()
    if final_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []

    def _append_rule_rows(
        *,
        scheme: str,
        filter_metric: str,
        threshold: float | None,
        evaluation_mode: str,
        working: pd.DataFrame,
        use_weights: bool,
    ) -> None:
        for (policy, dataset, scale), group in working.groupby(["policy", "dataset", "scale"], sort=True):
            ape = pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            weights = np.ones(len(group), dtype=float)
            if use_weights:
                metric_values = pd.to_numeric(group[filter_metric], errors="coerce").fillna(float(threshold or 0.0)).to_numpy(dtype=float)
                denom = max(float(threshold or 0.0), 1e-6)
                weights = 1.0 / (1.0 + (metric_values / denom))
            row_retention = float(len(group) / max(int(base_counts[(policy, dataset, scale)]), 1))
            rows.append(
                {
                    "policy": str(policy),
                    "dataset": str(dataset),
                    "scale": str(scale),
                    "scheme": str(scheme),
                    "evaluation_mode": str(evaluation_mode),
                    "filter_metric": str(filter_metric),
                    "threshold": float(threshold) if threshold is not None else np.nan,
                    "rows_kept": int(len(group)),
                    "row_retention": row_retention,
                    "mape": float(np.average(ape, weights=weights)) if len(ape) else 0.0,
                    "p90": _weighted_quantile(ape, weights, 0.90) if len(ape) else 0.0,
                }
            )

    base_counts = {
        (policy, dataset, scale): int(len(group))
        for (policy, dataset, scale), group in final_rows.groupby(["policy", "dataset", "scale"], sort=True)
    }
    _append_rule_rows(
        scheme="no_filter",
        filter_metric="",
        threshold=None,
        evaluation_mode="baseline",
        working=final_rows,
        use_weights=False,
    )
    quantiles = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    for metric in ("profile_kept_cv", "profile_kept_range_ratio"):
        values = pd.to_numeric(final_rows[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        thresholds = sorted({float(values.quantile(q)) for q in quantiles})
        for threshold in thresholds:
            filtered = final_rows.loc[pd.to_numeric(final_rows[metric], errors="coerce").fillna(np.inf) <= float(threshold)].copy()
            if not filtered.empty:
                _append_rule_rows(
                    scheme="global_drop",
                    filter_metric=metric,
                    threshold=float(threshold),
                    evaluation_mode="posthoc_test_subset",
                    working=filtered,
                    use_weights=False,
                )
            _append_rule_rows(
                scheme="global_train_weight",
                filter_metric=metric,
                threshold=float(threshold),
                evaluation_mode="posthoc_test_weight_proxy",
                working=final_rows,
                use_weights=True,
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    baseline = result.loc[result["scheme"].astype(str).eq("no_filter"), ["policy", "dataset", "scale", "mape", "p90"]].rename(
        columns={"mape": "baseline_mape", "p90": "baseline_p90"}
    )
    result = result.merge(baseline, on=["policy", "dataset", "scale"], how="left")
    result["mape_delta_pp"] = (pd.to_numeric(result["baseline_mape"], errors="coerce").fillna(0.0) - pd.to_numeric(result["mape"], errors="coerce").fillna(0.0)) * 100.0
    result["p90_delta_pp"] = (pd.to_numeric(result["baseline_p90"], errors="coerce").fillna(0.0) - pd.to_numeric(result["p90"], errors="coerce").fillna(0.0)) * 100.0
    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in result.groupby(["dataset", "scale", "scheme", "evaluation_mode", "filter_metric", "threshold"], dropna=False, sort=True):
        dataset, scale, scheme, evaluation_mode, filter_metric, threshold = keys
        aggregate_rows.append(
            {
                "policy": "__avg__",
                "dataset": str(dataset),
                "scale": str(scale),
                "scheme": str(scheme),
                "evaluation_mode": str(evaluation_mode),
                "filter_metric": str(filter_metric),
                "threshold": float(threshold) if pd.notna(threshold) else np.nan,
                "rows_kept": int(group["rows_kept"].sum()),
                "row_retention": float(pd.to_numeric(group["row_retention"], errors="coerce").fillna(0.0).mean()),
                "mape": float(pd.to_numeric(group["mape"], errors="coerce").fillna(0.0).mean()),
                "p90": float(pd.to_numeric(group["p90"], errors="coerce").fillna(0.0).mean()),
                "baseline_mape": float(pd.to_numeric(group["baseline_mape"], errors="coerce").fillna(0.0).mean()),
                "baseline_p90": float(pd.to_numeric(group["baseline_p90"], errors="coerce").fillna(0.0).mean()),
                "mape_delta_pp": float(pd.to_numeric(group["mape_delta_pp"], errors="coerce").fillna(0.0).mean()),
                "p90_delta_pp": float(pd.to_numeric(group["p90_delta_pp"], errors="coerce").fillna(0.0).mean()),
                "worst_policy_mape_delta_pp": float(pd.to_numeric(group["mape_delta_pp"], errors="coerce").fillna(0.0).min()),
                "min_row_retention": float(pd.to_numeric(group["row_retention"], errors="coerce").fillna(0.0).min()),
            }
        )
    result = pd.concat([result, pd.DataFrame(aggregate_rows)], ignore_index=True)
    return result.reset_index(drop=True)


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _bundle_op_thread_scale_enabled(bundle: dict[str, Any]) -> bool:
    prior_bundle = bundle.get("prior_bundle", {}) if isinstance(bundle.get("prior_bundle", {}), dict) else {}
    prior_config = prior_bundle.get("op_thread_scale_config") if isinstance(prior_bundle, dict) else None
    top_config = bundle.get("op_thread_scale_config")
    return bool(
        bundle.get("op_thread_scale_enabled", False)
        or prior_bundle.get("op_thread_scale_enabled", False)
        or top_config
        or prior_config
    )


def _export_static_inference_bundle(
    *,
    bundle: dict[str, Any],
    output_dir: Path,
    policy: str,
    dataset: str,
    scale: int,
    route: str,
    hw_config: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{policy}_{dataset}_{int(scale)}_{route}.pkl"
    output_path = output_dir / filename
    payload = dict(bundle)
    payload["static_inference_contract_version"] = 1
    payload["export_metadata"] = {
        "policy": str(policy),
        "dataset": str(dataset),
        "scale": f"{int(scale)}%",
        "route": str(route),
        "hw_config_path": str(Path(hw_config).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "model_sha256": str(bundle.get("model_sha256", "")),
        "op_thread_scale_enabled": _bundle_op_thread_scale_enabled(bundle),
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "policy": str(policy),
        "dataset": str(dataset),
        "scale": f"{int(scale)}%",
        "route": str(route),
        "bundle_path": str(output_path),
        "model_sha256": str(bundle.get("model_sha256", "")),
        "op_thread_scale_enabled": _bundle_op_thread_scale_enabled(bundle),
        "hw_config_path": str(Path(hw_config).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
    }


def _load_raw_splits(dataset_root: Path, *, include_pipeline_splits: bool = True) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    split_names = ["train", "test"]
    if include_pipeline_splits and (dataset_root / "e2e_train.csv").exists():
        split_names.append("e2e_train")
    if include_pipeline_splits and (dataset_root / "test_raw.csv").exists():
        split_names.append("test_raw")
    if include_pipeline_splits and (dataset_root / "e2e_test_raw.csv").exists():
        split_names.append("e2e_test_raw")
    for split in split_names:
        csv_path = dataset_root / f"{split}.csv"
        cache_path = _augmented_split_cache_path(dataset_root, split)
        if cache_path.exists():
            try:
                _progress("augment_split_cache_hit", dataset=str(dataset_root.name), detail=f"{split}:{cache_path.name}")
                frame = pd.read_pickle(cache_path)
                frame.attrs["dataset_root"] = str(dataset_root)
                splits[split] = frame
                continue
            except Exception as exc:
                _progress(
                    "augment_split_cache_invalid",
                    dataset=str(dataset_root.name),
                    detail=f"{split}:{cache_path.name}:{type(exc).__name__}",
                )
        _progress("read_split_csv_start", dataset=str(dataset_root.name), detail=f"{split}:{csv_path.name}")
        frame = pd.read_csv(csv_path, low_memory=False)
        _progress("read_split_csv_done", dataset=str(dataset_root.name), detail=f"{split}:rows={len(frame)}")
        frame[base_run.TARGET_US_COLUMN] = pd.to_numeric(frame[base_run.TARGET_COLUMN], errors="coerce").fillna(0.0)
        frame["row_uid"] = frame["row_uid"].astype(str)
        _progress("augment_split_start", dataset=str(dataset_root.name), detail=f"{split}:rows={len(frame)}")
        frame = _augment_split_frame(frame, dataset_root=dataset_root, split=split)
        _progress("augment_split_done", dataset=str(dataset_root.name), detail=f"{split}:rows={len(frame)}")
        frame.attrs["dataset_root"] = str(dataset_root)
        frame.to_pickle(cache_path)
        _progress("augment_split_cache_write", dataset=str(dataset_root.name), detail=f"{split}:{cache_path.name}")
        splits[split] = frame
    return splits


def _load_split_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "split_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _manifest_path(manifest: dict[str, Any], key: str, dataset_root: Path) -> Path:
    raw = str(manifest.get(key, "") or "").strip()
    if not raw:
        return dataset_root.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def _is_portable_split_manifest(dataset_root: Path, manifest: dict[str, Any]) -> bool:
    return bool(manifest.get("portable_dataset", False)) or (dataset_root / "portable_artifacts").exists()


def _validate_split_manifest(policy: str, dataset_name: str, dataset_root: Path, manifest: dict[str, Any]) -> None:
    if not manifest:
        raise ValueError(f"Missing split manifest for {dataset_root}")
    if dataset_name != "source":
        return
    phase_scope = str(manifest.get("phase_scope", "")).strip()
    input_dataset_root = _manifest_path(manifest, "input_dataset_root", dataset_root)
    portable_split = _is_portable_split_manifest(dataset_root, manifest)
    if _path_is_under(input_dataset_root, INVALID_LEGACY_ROOT):
        raise ValueError(f"Source policy {policy} points into archived invalid legacy splits: {input_dataset_root}")
    all_cases = {
        *[str(value) for value in manifest.get("train_case_ids", [])],
        *[str(value) for value in manifest.get("test_case_ids", [])],
        *[str(value) for value in manifest.get("raw_train_case_ids", [])],
        *[str(value) for value in manifest.get("raw_test_case_ids", [])],
    }
    if LEGACY_INVALID_CASE_IDS.intersection(all_cases):
        raise ValueError(f"Invalid legacy split detected in {dataset_root}: {sorted(LEGACY_INVALID_CASE_IDS.intersection(all_cases))}")
    if portable_split:
        if policy in [*LEGACY_POLICIES, *JOINT_POLICIES] and phase_scope != SOURCE_SCHEDULER_SCOPE:
            raise ValueError(
                f"Portable source policy {policy} requires phase_scope={SOURCE_SCHEDULER_SCOPE}, got {phase_scope} from {dataset_root}"
            )
        if policy in PHASE2_POLICIES and phase_scope != PHASE2_SCOPE:
            raise ValueError(f"Portable phase2 policy {policy} requires phase_scope={PHASE2_SCOPE}, got {phase_scope}")
        return
    if policy in LEGACY_POLICIES:
        if phase_scope != SOURCE_SCHEDULER_SCOPE:
            raise ValueError(
                f"General source-only policy {policy} requires phase_scope={SOURCE_SCHEDULER_SCOPE}, got {phase_scope} from {dataset_root}"
            )
        if input_dataset_root != SOURCE_RAW_ROOT:
            raise ValueError(
                f"General source-only policy {policy} must use {SOURCE_RAW_ROOT}, got {input_dataset_root}"
            )
    if policy in JOINT_POLICIES:
        if phase_scope != SOURCE_SCHEDULER_SCOPE:
            raise ValueError(
                f"Joint source-only policy {policy} requires phase_scope={SOURCE_SCHEDULER_SCOPE}, got {phase_scope} from {dataset_root}"
            )
        if input_dataset_root != SOURCE_RAW_ROOT:
            raise ValueError(f"Joint source-only policy {policy} must use {SOURCE_RAW_ROOT}, got {input_dataset_root}")
    if policy in PHASE2_POLICIES:
        if phase_scope != PHASE2_SCOPE:
            raise ValueError(f"Phase2 policy {policy} requires phase_scope={PHASE2_SCOPE}, got {phase_scope} from {dataset_root}")
        if input_dataset_root != PHASE2_SOURCE_ROOT:
            raise ValueError(f"Phase2 policy {policy} must use {PHASE2_SOURCE_ROOT}, got {input_dataset_root}")


def _fit_pool(raw_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return raw_splits["train"].drop_duplicates(subset=["row_uid"]).reset_index(drop=True)


def _scale_to_rows(total_rows: int, scale: int) -> int:
    return max(int(round(int(total_rows) * float(scale) / 100.0)), 1)


def _sample_fit_subset(frame: pd.DataFrame, scale: int, seed: int, *, salt: int = 0) -> pd.DataFrame:
    if int(scale) >= 100:
        return frame.copy().reset_index(drop=True)
    return base_run._sample_fit_subset_rows(
        frame,
        target_rows=_scale_to_rows(len(frame), int(scale)),
        seed=seed,
        salt=int(scale) * 100 + int(salt),
    )


def _summary_row(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    fit_rows: int,
    selected_family: str,
    metrics: dict[str, Any],
    analytical_metrics: dict[str, Any] | None = None,
    train_backend: str = "",
    train_env: str = "",
    ml_input_source: str = "",
    ml_raw_sw_features_used: bool = False,
    ml_raw_hw_params_used: bool = False,
    hw_refit_required: bool = False,
    family_mlp_enabled_count: int = 0,
    source_input_dataset_root: str = "",
    source_phase_scope: str = "",
) -> dict[str, Any]:
    del analytical_metrics
    whitebox_mode, fit_data_dependent = _whitebox_route_metadata(str(route))
    return {
        "policy": policy,
        "dataset": dataset,
        "route": route,
        "stage_used": route_stage(route),
        "route_whitebox_mode": whitebox_mode,
        "fit_data_dependent": bool(fit_data_dependent),
        "scale": f"{int(scale)}%",
        "fit_rows": int(fit_rows),
        "selected_family": selected_family,
        "mape": float(metrics["mape"]),
        "p90": float(metrics["p90"]),
        "median": float(metrics["median"]),
        "eval_rows": int(metrics["eval_rows"]),
        "eval_scope": str(metrics["eval_scope"]),
        "train_backend": str(train_backend),
        "train_env": str(train_env),
        "ml_input_source": str(ml_input_source),
        "ml_raw_sw_features_used": bool(ml_raw_sw_features_used),
        "ml_raw_hw_params_used": bool(ml_raw_hw_params_used),
        "hw_refit_required": bool(hw_refit_required),
        "family_mlp_enabled_count": int(family_mlp_enabled_count),
        "source_input_dataset_root": str(source_input_dataset_root),
        "source_phase_scope": str(source_phase_scope),
    }


def _group_numeric_series(group: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in group.columns:
        return pd.to_numeric(group[column], errors="coerce").fillna(float(default))
    return pd.Series(float(default), index=group.index, dtype=float)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_universe_guard_rows(
    *,
    policy: str,
    dataset: str,
    dataset_root: Path,
    split_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset_full_path = dataset_root / "dataset_full.csv"
    if not dataset_full_path.exists():
        return []
    dataset_full = pd.read_csv(dataset_full_path, usecols=["row_uid"], low_memory=False)
    row_uid_digest = _sha256_text("\n".join(dataset_full["row_uid"].astype(str).tolist()))
    input_dataset_root = Path(str(split_manifest.get("input_dataset_root", ""))).resolve() if split_manifest.get("input_dataset_root") else Path()
    source_manifest_path = input_dataset_root / "manifest.json" if str(input_dataset_root) else Path()
    split_manifest_path = dataset_root / "split_manifest.json"
    return [
        {
            "policy": policy,
            "dataset": dataset,
            "dataset_root": str(dataset_root),
            "source_manifest_path": str(source_manifest_path) if source_manifest_path else "",
            "split_manifest_path": str(split_manifest_path),
            "source_manifest_sha256": _sha256_file(source_manifest_path) if source_manifest_path else "",
            "split_manifest_sha256": _sha256_file(split_manifest_path),
            "dataset_full_rows": int(len(dataset_full)),
            "dataset_full_row_uid_sha256": str(row_uid_digest),
        }
    ]


def _graph_feature_mapping_rows(
    *,
    policy: str,
    dataset: str,
    split_name: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    mapped = pd.to_numeric(frame.get("graph_feature_mapped_flag", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    return [
        {
            "policy": policy,
            "dataset": dataset,
            "split": split_name,
            "rows": int(len(frame)),
            "mapped_rows": int(mapped.gt(0.5).sum()),
            "mapping_coverage": float(mapped.gt(0.5).mean()) if len(mapped) else 0.0,
        }
    ]


def _prior_to_rf_gain_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prior_frame: pd.DataFrame,
    prior_pred: np.ndarray,
    rf_frame: pd.DataFrame,
    rf_pred: np.ndarray,
) -> list[dict[str, Any]]:
    prior_work = prior_frame.copy()
    rf_work = rf_frame.copy()
    prior_work["prior_pred"] = np.asarray(prior_pred, dtype=float)
    rf_work["rf_pred"] = np.asarray(rf_pred, dtype=float)
    merged = prior_work[["row_uid", "op_type", "whitebox_rf_group", base_run.TARGET_US_COLUMN, "prior_pred"]].merge(
        rf_work[["row_uid", "rf_pred"]],
        on="row_uid",
        how="inner",
    )
    actual = pd.to_numeric(merged[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    prior_ape = np.abs(pd.to_numeric(merged["prior_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float) - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    rf_ape = np.abs(pd.to_numeric(merged["rf_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float) - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    merged["prior_ape"] = prior_ape
    merged["rf_ape"] = rf_ape
    rows: list[dict[str, Any]] = []
    for (rf_group, op_type), group in merged.groupby(["whitebox_rf_group", "op_type"], sort=True):
        prior_mape = float(pd.to_numeric(group["prior_ape"], errors="coerce").fillna(0.0).mean())
        rf_mape = float(pd.to_numeric(group["rf_ape"], errors="coerce").fillna(0.0).mean())
        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "route": route,
                "scale": f"{int(scale)}%",
                "rf_group": str(rf_group),
                "op_type": str(op_type),
                "rows": int(len(group)),
                "prior_only_mape": prior_mape,
                "rf_mape": rf_mape,
                "absolute_gain_pp": float((prior_mape - rf_mape) * 100.0),
                "relative_gain": float((prior_mape - rf_mape) / max(prior_mape, 1e-9)),
            }
        )
    return rows


def _spearman_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 2:
        return 0.0
    actual_series = pd.Series(np.asarray(actual, dtype=float))
    pred_series = pd.Series(np.asarray(pred, dtype=float))
    value = actual_series.corr(pred_series, method="spearman")
    if pd.isna(value):
        return 0.0
    return float(value)


def _analytical_interval_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    actual = pd.to_numeric(prepared_frame[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lower = pd.to_numeric(prepared_frame["whitebox_lower_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    center = pd.to_numeric(prepared_frame["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    upper = pd.to_numeric(prepared_frame["whitebox_upper_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    covered = (actual >= lower - 1e-6) & (actual <= upper + 1e-6)
    relative_width = (upper - lower) / np.clip(actual, a_min=1e-6, a_max=None)
    return [
        {
            "policy": policy,
            "dataset": dataset,
            "route": route,
            "stage_used": route_stage(route),
            "scale": f"{int(scale)}%",
            "scope": "overall",
            "rows": int(len(prepared_frame)),
            "coverage_rate": float(np.mean(covered)) if len(covered) else 0.0,
            "median_relative_width": float(np.median(relative_width)) if len(relative_width) else 0.0,
            "mean_relative_width": float(np.mean(relative_width)) if len(relative_width) else 0.0,
            "center_spearman": _spearman_corr(actual, center),
        }
    ]


def _analytical_trend_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    work = prepared_frame.copy()
    if "family_bucket" not in work.columns:
        work["family_bucket"] = work["op_type"].astype(str).map(family_bucket_for_op_type).astype(str)
    actual = pd.to_numeric(work[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    center = pd.to_numeric(work["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rows: list[dict[str, Any]] = [
        {
            "policy": policy,
            "dataset": dataset,
            "route": route,
            "stage_used": route_stage(route),
            "scale": f"{int(scale)}%",
            "slice_kind": "policy",
            "family_bucket": "",
            "op_type": "",
            "rows": int(len(work)),
            "spearman": _spearman_corr(actual, center),
        }
    ]
    for (family_bucket, op_type), group in work.groupby(["family_bucket", "op_type"], sort=True):
        group_actual = pd.to_numeric(group[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        group_center = pd.to_numeric(group["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "route": route,
                "stage_used": route_stage(route),
                "scale": f"{int(scale)}%",
                "slice_kind": "family_op",
                "family_bucket": str(family_bucket),
                "op_type": str(op_type),
                "rows": int(len(group)),
                "spearman": _spearman_corr(group_actual, group_center),
            }
        )
    return rows


def _analytical_bottleneck_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    lower_cols = [f"svc_{name}_lower_us" for name in ("descriptor", "stream_mem", "random_mem", "compute", "reorder", "sync", "dispatch")]
    center_cols = [f"svc_{name}_center_us" for name in ("descriptor", "stream_mem", "random_mem", "compute", "reorder", "sync", "dispatch")]
    upper_cols = [f"svc_{name}_upper_us" for name in ("descriptor", "stream_mem", "random_mem", "compute", "reorder", "sync", "dispatch")]
    lower = np.stack([pd.to_numeric(prepared_frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float) for column in lower_cols], axis=1)
    center = np.stack([pd.to_numeric(prepared_frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float) for column in center_cols], axis=1)
    upper = np.stack([pd.to_numeric(prepared_frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float) for column in upper_cols], axis=1)
    clear_mask = pd.to_numeric(prepared_frame["max_to_sum_ratio_center"], errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 0.40
    center_id = np.argmax(center, axis=1)
    stable = (center_id == np.argmax(lower, axis=1)) & (center_id == np.argmax(upper, axis=1))
    clear_rows = int(clear_mask.sum())
    stable_rate = float(np.mean(stable[clear_mask])) if clear_rows else 0.0
    return [
        {
            "policy": policy,
            "dataset": dataset,
            "route": route,
            "stage_used": route_stage(route),
            "scale": f"{int(scale)}%",
            "rows": int(len(prepared_frame)),
            "clear_bottleneck_rows": clear_rows,
            "stable_bottleneck_rate": stable_rate,
        }
    ]


def _analytical_state_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    required = {
        "exec_region_static",
        "submodel_role_static",
        "branch_lifetime_class",
        "producer_materialization_state",
        "consumer_materialization_state",
        "reduce_partial_shape_static",
        "whitebox_center_us",
    }
    if not required.issubset(set(prepared_frame.columns)):
        return []
    actual = pd.to_numeric(prepared_frame[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    center = pd.to_numeric(prepared_frame["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lower = pd.to_numeric(prepared_frame["whitebox_lower_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    upper = pd.to_numeric(prepared_frame["whitebox_upper_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ape = np.abs(center - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    coverage = (actual >= lower - 1e-6) & (actual <= upper + 1e-6)
    work = prepared_frame.copy()
    work["ape"] = ape
    work["covered"] = coverage.astype(float)
    rows: list[dict[str, Any]] = []
    for state_name in (
        "exec_region_static",
        "submodel_role_static",
        "branch_lifetime_class",
        "producer_materialization_state",
        "consumer_materialization_state",
        "reduce_partial_shape_static",
    ):
        for state_value, group in work.groupby(state_name, sort=True):
            group_actual = pd.to_numeric(group[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0)
            group_center = pd.to_numeric(group["whitebox_center_us"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "policy": policy,
                    "dataset": dataset,
                    "route": route,
                    "stage_used": route_stage(route),
                    "scale": f"{int(scale)}%",
                    "state_name": str(state_name),
                    "state_value": str(state_value),
                    "rows": int(len(group)),
                    "actual_median_us": float(group_actual.median()),
                    "prior_median_us": float(group_center.median()),
                    "prior_mape": float(pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).mean()),
                    "coverage_rate": float(pd.to_numeric(group["covered"], errors="coerce").fillna(0.0).mean()),
                }
            )
    return rows


def _phase_error_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
    pred: pd.Series | list[float] | np.ndarray,
) -> list[dict[str, Any]]:
    if "exec_region_static" not in prepared_frame.columns:
        return []
    work = prepared_frame.copy()
    actual = pd.to_numeric(work[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred_arr = pd.to_numeric(pd.Series(pred, index=work.index), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ape = np.abs(pred_arr - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    work["ape"] = ape
    work["actual_us"] = actual
    total_ape = max(float(ape.sum()), 1e-12)
    total_actual_us = max(float(np.clip(actual, a_min=0.0, a_max=None).sum()), 1e-12)
    rows: list[dict[str, Any]] = []
    overall_weighted = _weighted_error_stats(ape, actual)
    rows.append(
        {
            "policy": policy,
            "dataset": dataset,
            "route": route,
            "stage_used": route_stage(route),
            "scale": f"{int(scale)}%",
            "slice_kind": "overall",
            "exec_region_static": "",
            "op_type": "",
            "rows": int(len(work)),
            "mape": float(np.mean(ape)) if len(ape) else 0.0,
            "p90": float(np.quantile(ape, 0.90)) if len(ape) else 0.0,
            "weighted_mape": float(overall_weighted["weighted_mape"]),
            "weighted_p90": float(overall_weighted["weighted_p90"]),
            "weighted_median": float(overall_weighted["weighted_median"]),
            "actual_us_sum": float(overall_weighted["actual_us_sum"]),
            "actual_us_share": 1.0 if len(ape) else 0.0,
            "ape_sum_share": 1.0 if len(ape) else 0.0,
        }
    )
    for exec_region, group in work.groupby("exec_region_static", sort=True):
        values = pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        actual_values = pd.to_numeric(group["actual_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weighted = _weighted_error_stats(values, actual_values)
        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "route": route,
                "stage_used": route_stage(route),
                "scale": f"{int(scale)}%",
                "slice_kind": "exec_region",
                "exec_region_static": str(exec_region),
                "op_type": "",
                "rows": int(len(group)),
                "mape": float(np.mean(values)) if len(values) else 0.0,
                "p90": float(np.quantile(values, 0.90)) if len(values) else 0.0,
                "weighted_mape": float(weighted["weighted_mape"]),
                "weighted_p90": float(weighted["weighted_p90"]),
                "weighted_median": float(weighted["weighted_median"]),
                "actual_us_sum": float(weighted["actual_us_sum"]),
                "actual_us_share": float(weighted["actual_us_sum"] / total_actual_us),
                "ape_sum_share": float(values.sum() / total_ape),
            }
        )
        for op_type, op_group in group.groupby("op_type", sort=True):
            op_values = pd.to_numeric(op_group["ape"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            op_actual = pd.to_numeric(op_group["actual_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            op_weighted = _weighted_error_stats(op_values, op_actual)
            rows.append(
                {
                    "policy": policy,
                    "dataset": dataset,
                    "route": route,
                    "stage_used": route_stage(route),
                    "scale": f"{int(scale)}%",
                    "slice_kind": "exec_region_op",
                    "exec_region_static": str(exec_region),
                    "op_type": str(op_type),
                    "rows": int(len(op_group)),
                    "mape": float(np.mean(op_values)) if len(op_values) else 0.0,
                    "p90": float(np.quantile(op_values, 0.90)) if len(op_values) else 0.0,
                    "weighted_mape": float(op_weighted["weighted_mape"]),
                    "weighted_p90": float(op_weighted["weighted_p90"]),
                    "weighted_median": float(op_weighted["weighted_median"]),
                    "actual_us_sum": float(op_weighted["actual_us_sum"]),
                    "actual_us_share": float(op_weighted["actual_us_sum"] / total_actual_us),
                    "ape_sum_share": float(op_values.sum() / total_ape),
                }
            )
    return rows


def _prior_gap_decomposition_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    required = {
        "whitebox_kernel_only_center_us",
        "whitebox_state_adjusted_center_us",
        "whitebox_phase_adjusted_center_us",
        "whitebox_center_us",
        "exec_region_static",
    }
    if not required.issubset(set(prepared_frame.columns)):
        return []
    work = prepared_frame.copy()
    actual = pd.to_numeric(work[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    kernel = pd.to_numeric(work["whitebox_kernel_only_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    state_adjusted = pd.to_numeric(work["whitebox_state_adjusted_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    phase_adjusted = pd.to_numeric(work["whitebox_phase_adjusted_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    final_prior = pd.to_numeric(work["whitebox_center_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    work["kernel_ape"] = np.abs(kernel - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    work["state_ape"] = np.abs(state_adjusted - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    work["phase_ape"] = np.abs(phase_adjusted - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    work["final_ape"] = np.abs(final_prior - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    rows: list[dict[str, Any]] = []
    slices: list[tuple[str, str, pd.DataFrame]] = [("overall", "", work)]
    slices.extend(("exec_region", str(region), group) for region, group in work.groupby("exec_region_static", sort=True))
    for slice_kind, exec_region, group in slices:
        kernel_mape = float(pd.to_numeric(group["kernel_ape"], errors="coerce").fillna(0.0).mean())
        state_mape = float(pd.to_numeric(group["state_ape"], errors="coerce").fillna(0.0).mean())
        phase_mape = float(pd.to_numeric(group["phase_ape"], errors="coerce").fillna(0.0).mean())
        final_mape = float(pd.to_numeric(group["final_ape"], errors="coerce").fillna(0.0).mean())
        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "route": route,
                "stage_used": route_stage(route),
                "scale": f"{int(scale)}%",
                "slice_kind": slice_kind,
                "exec_region_static": exec_region,
                "rows": int(len(group)),
                "kernel_only_mape": kernel_mape,
                "state_adjusted_mape": state_mape,
                "phase_adjusted_mape": phase_mape,
                "final_prior_mape": final_mape,
                "state_gain_pp": float((kernel_mape - state_mape) * 100.0),
                "phase_hypothesis_gain_pp": float((state_mape - phase_mape) * 100.0),
                "composition_gain_pp": float((state_mape - final_mape) * 100.0),
                "remaining_gap_pp": float(final_mape * 100.0),
            }
        )
    return rows


def _family_breakdown_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
    pred: pd.Series | list[float],
) -> list[dict[str, Any]]:
    work = prepared_frame.copy()
    if "family_bucket" not in work.columns:
        work["family_bucket"] = work["op_type"].astype(str).map(family_bucket_for_op_type).astype(str)
    actual = pd.to_numeric(work[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred_arr = pd.to_numeric(pd.Series(pred, index=work.index), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ape = np.abs(pred_arr - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    work["ape"] = ape
    work["actual_us"] = actual
    total_ape = max(float(ape.sum()), 1e-12)
    total_actual_us = max(float(np.clip(actual, a_min=0.0, a_max=None).sum()), 1e-12)
    rows: list[dict[str, Any]] = []
    for (family_bucket, op_type), group in work.groupby(["family_bucket", "op_type"], sort=True):
        values = pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        actual_values = pd.to_numeric(group["actual_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weighted = _weighted_error_stats(values, actual_values)
        bottleneck = _group_numeric_series(group, "bottleneck_id", -1.0).to_numpy(dtype=float)
        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "route": route,
                "stage_used": route_stage(route),
                "scale": f"{int(scale)}%",
                "family_bucket": str(family_bucket),
                "op_type": str(op_type),
                "rows": int(len(group)),
                "mape": float(np.mean(values)),
                "p90": float(np.quantile(values, 0.90)),
                "median": float(np.quantile(values, 0.50)),
                "weighted_mape": float(weighted["weighted_mape"]),
                "weighted_p90": float(weighted["weighted_p90"]),
                "weighted_median": float(weighted["weighted_median"]),
                "actual_us_sum": float(weighted["actual_us_sum"]),
                "actual_us_share": float(weighted["actual_us_sum"] / total_actual_us),
                "ape_sum_share": float(values.sum() / total_ape),
                "tail_ape_gt_20_share": float(np.mean(values > 0.20)),
                "tail_ape_gt_50_share": float(np.mean(values > 0.50)),
                "bottleneck_id_mode": float(pd.Series(bottleneck).mode().iloc[0]) if len(bottleneck) else -1.0,
            }
        )
    return rows


def _whitebox_source_breakdown_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
    pred: pd.Series | list[float],
) -> list[dict[str, Any]]:
    if "whitebox_prior_source" not in prepared_frame.columns:
        return []
    work = prepared_frame.copy()
    actual = pd.to_numeric(work[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred_arr = pd.to_numeric(pd.Series(pred, index=work.index), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ape = np.abs(pred_arr - actual) / np.clip(actual, a_min=1e-6, a_max=None)
    total_ape = max(float(ape.sum()), 1e-12)
    work["ape"] = ape
    work["whitebox_prior_family"] = work.get("whitebox_prior_family", work["op_type"].astype(str).map(family_bucket_for_op_type).astype(str))
    work["whitebox_residual_source"] = work.get("whitebox_residual_source", pd.Series("prior_only", index=work.index, dtype=object))
    rows: list[dict[str, Any]] = []
    for (family_name, op_type, prior_source, residual_source), group in work.groupby(
        ["whitebox_prior_family", "op_type", "whitebox_prior_source", "whitebox_residual_source"],
        sort=True,
    ):
        values = pd.to_numeric(group["ape"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "route": route,
                "stage_used": route_stage(route),
                "scale": f"{int(scale)}%",
                "whitebox_prior_family": str(family_name),
                "op_type": str(op_type),
                "whitebox_prior_source": str(prior_source),
                "whitebox_residual_source": str(residual_source),
                "rows": int(len(group)),
                "mape": float(np.mean(values)),
                "p90": float(np.quantile(values, 0.90)),
                "median": float(np.quantile(values, 0.50)),
                "ape_sum_share": float(values.sum() / total_ape),
                "prior_us_median": float(pd.to_numeric(group["whitebox_prior_us"], errors="coerce").fillna(0.0).median()),
                "pred_mem_us_median": float(_group_numeric_series(group, "pred_mem_us").median()),
                "pred_compute_us_median": float(_group_numeric_series(group, "pred_compute_us").median()),
                "pred_sync_us_median": float(_group_numeric_series(group, "pred_sync_us").median()),
                "pred_launch_us_median": float(_group_numeric_series(group, "pred_launch_us").median()),
                "pred_total_us_median": float(_group_numeric_series(group, "pred_total_us").median()),
                "effective_copy_bw_gb_s_median": float(_group_numeric_series(group, "effective_copy_bw_gb_s").median()),
                "effective_random_gather_bw_gb_s_median": float(_group_numeric_series(group, "effective_random_gather_bw_gb_s").median()),
                "launch_window_utilization_median": float(_group_numeric_series(group, "launch_window_utilization").median()),
            }
        )
    return rows


def _whitebox_perf_feature_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    prepared_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    required = {"svc_descriptor_us", "svc_stream_mem_us", "svc_random_mem_us", "svc_compute_us", "svc_sync_us", "svc_reorder_us", "svc_dispatch_us"}
    if not required.issubset(set(prepared_frame.columns)):
        return []
    work = prepared_frame.copy()
    work["whitebox_prior_family"] = work.get("whitebox_prior_family", work["op_type"].astype(str).map(family_bucket_for_op_type).astype(str))
    work["whitebox_residual_source"] = work.get("whitebox_residual_source", pd.Series("prior_only", index=work.index, dtype=object))
    rows: list[dict[str, Any]] = []
    for (family_name, op_type, residual_source), group in work.groupby(
        ["whitebox_prior_family", "op_type", "whitebox_residual_source"],
        sort=True,
    ):
        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "route": route,
                "stage_used": route_stage(route),
                "scale": f"{int(scale)}%",
                "whitebox_prior_family": str(family_name),
                "op_type": str(op_type),
                "whitebox_residual_source": str(residual_source),
                "rows": int(len(group)),
                "svc_descriptor_us_median": float(pd.to_numeric(group["svc_descriptor_us"], errors="coerce").fillna(0.0).median()),
                "svc_stream_mem_us_median": float(pd.to_numeric(group["svc_stream_mem_us"], errors="coerce").fillna(0.0).median()),
                "svc_random_mem_us_median": float(pd.to_numeric(group["svc_random_mem_us"], errors="coerce").fillna(0.0).median()),
                "svc_compute_us_median": float(pd.to_numeric(group["svc_compute_us"], errors="coerce").fillna(0.0).median()),
                "svc_sync_us_median": float(pd.to_numeric(group["svc_sync_us"], errors="coerce").fillna(0.0).median()),
                "svc_reorder_us_median": float(pd.to_numeric(group["svc_reorder_us"], errors="coerce").fillna(0.0).median()),
                "svc_dispatch_us_median": float(pd.to_numeric(group["svc_dispatch_us"], errors="coerce").fillna(0.0).median()),
                "sum_svc_us_median": float(_group_numeric_series(group, "sum_svc_us").median()),
                "max_svc_us_median": float(_group_numeric_series(group, "max_svc_us").median()),
                "share_stream_mem_median": float(_group_numeric_series(group, "share_stream_mem").median()),
                "share_random_mem_median": float(_group_numeric_series(group, "share_random_mem").median()),
                "share_compute_median": float(_group_numeric_series(group, "share_compute").median()),
                "share_sync_median": float(_group_numeric_series(group, "share_sync").median()),
                "share_reorder_median": float(_group_numeric_series(group, "share_reorder").median()),
                "share_dispatch_median": float(_group_numeric_series(group, "share_dispatch").median()),
                "eff_copy_bw_gb_s_median": float(_group_numeric_series(group, "eff_copy_bw_gb_s").median()),
                "eff_random_bw_gb_s_median": float(_group_numeric_series(group, "eff_random_bw_gb_s").median()),
                "eff_fp_items_per_us_median": float(_group_numeric_series(group, "eff_fp_items_per_us").median()),
                "launch_vs_useful_ratio_median": float(_group_numeric_series(group, "launch_vs_useful_ratio").median()),
                "mem_vs_compute_ratio_median": float(_group_numeric_series(group, "mem_vs_compute_ratio").median()),
                "bottleneck_id_median": float(_group_numeric_series(group, "bottleneck_id").median()),
            }
        )
    return rows


def _family_mlp_enablement_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in bundle.get("family_training_summaries", []):
        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "route": route,
                "stage_used": route_stage(route),
                "scale": f"{int(scale)}%",
                "family_name": str(entry.get("family_name", "")),
                "model_level": str(entry.get("model_level", "family")),
                "whitebox_prior_source": str(entry.get("whitebox_prior_source", "")),
                "enabled": bool(entry.get("enabled", False)),
                "reason": str(entry.get("reason", "")),
                "rows": int(entry.get("rows", 0)),
                "train_rows": int(entry.get("train_rows", 0)),
                "valid_rows": int(entry.get("valid_rows", 0)),
                "baseline_valid_mape": float(entry.get("baseline_valid_mape", 0.0)),
                "model_valid_mape": float(entry.get("model_valid_mape", 0.0)),
                "relative_improvement": float(entry.get("relative_improvement", 0.0)),
                "best_epoch": int(entry.get("best_epoch", 0)),
                "epochs_trained": int(entry.get("epochs_trained", 0)),
                "model_backend": str(bundle.get("train_backend", "")),
                "train_device": str(entry.get("train_device", bundle.get("train_device", ""))),
                "torch_npu_enabled": bool(entry.get("torch_npu_enabled", False)),
                "n_estimators": int(entry.get("n_estimators", 0)),
                "max_depth": entry.get("max_depth", None),
                "min_samples_leaf": int(entry.get("min_samples_leaf", 0)),
                "max_features": str(entry.get("max_features", "")),
                "feature_count": int(entry.get("feature_count", 0)),
                "top_feature_importances": json.dumps(entry.get("top_feature_importances", []), ensure_ascii=False),
            }
        )
    return rows


def _hw_transfer_audit_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    reference_pred: np.ndarray,
    alternate_hw_config: Path | None,
) -> list[dict[str, Any]]:
    if alternate_hw_config is None or not Path(alternate_hw_config).exists():
        return []
    try:
        transferred_frame, transferred_pred = predict_whitebox_family_simple_rf(
            bundle,
            frame,
            override_hw_config=Path(alternate_hw_config),
        )
        delta = np.abs(np.asarray(reference_pred, dtype=float) - np.asarray(transferred_pred, dtype=float))
        prediction_succeeded = True
        rows = int(len(transferred_frame))
        mean_abs_delta = float(delta.mean()) if len(delta) else 0.0
        max_abs_delta = float(delta.max()) if len(delta) else 0.0
        error = ""
    except Exception as exc:
        prediction_succeeded = False
        rows = 0
        mean_abs_delta = 0.0
        max_abs_delta = 0.0
        error = str(exc)
    return [
        {
            "policy": policy,
            "dataset": dataset,
            "route": route,
            "stage_used": route_stage(route),
            "scale": f"{int(scale)}%",
            "source_hw_config": str(bundle.get("prior_bundle", {}).get("hw_config_path", "")),
            "alternate_hw_config": str(alternate_hw_config),
            "rows": int(rows),
            "weights_reused": True,
            "weights_sha256": str(bundle.get("model_sha256", "")),
            "prediction_succeeded": bool(prediction_succeeded),
            "mean_abs_delta_us": float(mean_abs_delta),
            "max_abs_delta_us": float(max_abs_delta),
            "error": error,
        }
    ]


def _whitebox_invariance_row(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    variant: str,
    reference_pred: pd.Series,
    current_pred: pd.Series,
) -> dict[str, Any]:
    aligned = current_pred.reindex(reference_pred.index)
    diff = np.abs(aligned.to_numpy(dtype=float) - reference_pred.to_numpy(dtype=float))
    denom = np.clip(np.abs(reference_pred.to_numpy(dtype=float)), a_min=1e-6, a_max=None)
    rel = diff / denom
    passed = bool(np.allclose(diff, 0.0, rtol=0.0, atol=0.0))
    return {
        "policy": policy,
        "dataset": dataset,
        "route": route,
        "scale": f"{int(scale)}%",
        "variant": str(variant),
        "rows": int(len(reference_pred)),
        "passed": passed,
        "max_abs_diff_us": float(diff.max()) if len(diff) else 0.0,
        "max_rel_diff": float(rel.max()) if len(rel) else 0.0,
    }


def _whitebox_label_invariance_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    reference_pred: np.ndarray,
    predict_fn: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_series = pd.Series(np.asarray(reference_pred, dtype=float), index=frame["row_uid"].astype(str), dtype=float)
    label_columns = [column for column in (base_run.TARGET_COLUMN, base_run.TARGET_US_COLUMN, "label_operator_actual_dur_us") if column in frame.columns]
    variants: list[tuple[str, pd.DataFrame]] = [("reference", frame.copy())]

    zeroed = frame.copy()
    for column in label_columns:
        zeroed[column] = 0.0
    variants.append(("label_zeroed", zeroed))

    nulled = frame.copy()
    for column in label_columns:
        nulled[column] = np.nan
    variants.append(("label_nan", nulled))

    missing = frame.copy()
    if label_columns:
        missing = missing.drop(columns=label_columns)
    variants.append(("label_missing", missing))

    for variant_name, variant_frame in variants:
        if variant_name == "reference":
            current_series = reference_series
        else:
            _, current_pred = predict_fn(bundle, variant_frame)
            current_series = pd.Series(np.asarray(current_pred, dtype=float), index=frame["row_uid"].astype(str), dtype=float)
        rows.append(
            _whitebox_invariance_row(
                policy=policy,
                dataset=dataset,
                route=route,
                scale=int(scale),
                variant=variant_name,
                reference_pred=reference_series,
                current_pred=current_series,
            )
        )
    return rows


def _yaml_payload(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for module sensitivity audit")
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Hardware config did not parse into a mapping: {path}")
    return payload


def _yaml_nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _yaml_nested_set(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _parse_mutation_quantity(value: Any) -> tuple[float, str]:
    text = str(value).strip()
    for suffix in ("GHz", "MHz"):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]), suffix
    return float(text), ""


def _format_mutation_quantity(value: float, suffix: str) -> Any:
    if suffix:
        return f"{value:.6f}{suffix}"
    if float(value).is_integer():
        return int(round(value))
    return float(value)


def _module_sensitivity_rows(
    *,
    policy: str,
    dataset: str,
    route: str,
    scale: int,
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    reference_prepared: pd.DataFrame,
    reference_pred: np.ndarray,
    hw_config: Path,
    predict_fn: Any,
) -> list[dict[str, Any]]:
    if dataset != "source" or yaml is None:
        return []

    reference_hash = str(bundle.get("model_sha256", ""))
    reference_payload = _yaml_payload(hw_config)
    svc_columns = [
        "svc_descriptor_us",
        "svc_stream_mem_us",
        "svc_random_mem_us",
        "svc_compute_us",
        "svc_reorder_us",
        "svc_sync_us",
        "svc_dispatch_us",
    ]
    specs: list[dict[str, Any]] = [
        {"field": "memory.copy_bw_gb_s_1t", "group": "copy_bw", "expected": ["svc_stream_mem_us"], "up": 1.2, "down": 0.8},
        {"field": "memory.copy_bw_gb_s_saturated", "group": "copy_bw", "expected": ["svc_stream_mem_us"], "up": 1.2, "down": 0.8},
        {"field": "memory.random_gather_bw_gb_s_1t", "group": "random_bw", "expected": ["svc_random_mem_us"], "up": 1.2, "down": 0.8},
        {"field": "memory.random_gather_bw_gb_s_saturated", "group": "random_bw", "expected": ["svc_random_mem_us"], "up": 1.2, "down": 0.8},
        {"field": "instruction.fp_throughput_per_cycle.vector_sp_add", "group": "compute", "expected": ["svc_compute_us"], "up": 1.2, "down": 0.8},
        {"field": "instruction.fp_throughput_per_cycle.vector_sp_mul", "group": "compute", "expected": ["svc_compute_us"], "up": 1.2, "down": 0.8},
        {"field": "instruction.fp_throughput_per_cycle.vector_sp_fma", "group": "compute", "expected": ["svc_compute_us"], "up": 1.2, "down": 0.8},
        {"field": "core.cpu_clock", "group": "compute", "expected": ["svc_compute_us"], "up": 1.1, "down": 0.9},
        {"field": "instruction.simd_width_bits", "group": "compute", "expected": ["svc_compute_us"], "mode": "simd"},
        {"field": "runtime.parallel_launch_overhead_us_intra", "group": "launch", "expected": ["svc_dispatch_us", "svc_sync_us"], "up": 1.2, "down": 0.8},
        {"field": "runtime.parallel_launch_overhead_us_inter", "group": "launch", "expected": ["svc_dispatch_us", "svc_sync_us"], "up": 1.2, "down": 0.8},
    ]

    op_type = reference_prepared.get("op_type", pd.Series("", index=reference_prepared.index)).fillna("").astype(str)
    exec_mode = reference_prepared.get("whitebox_exec_mode", pd.Series("", index=reference_prepared.index)).fillna("").astype(str)
    non_tiny_shape_mask = ~exec_mode.eq("tiny_shape_arith_ops")
    add_like_mask = (op_type.eq("ReduceSum") | (op_type.eq("Add") & non_tiny_shape_mask)).to_numpy(dtype=bool)
    mul_like_mask = (op_type.eq("Mul") & non_tiny_shape_mask).to_numpy(dtype=bool)
    fma_like_mask = op_type.isin(["Gemm", "MatMul"]).to_numpy(dtype=bool)
    compute_like_mask = (op_type.isin(["Add", "Mul", "ReduceSum", "Gemm", "MatMul", "Relu", "Sigmoid"]) & non_tiny_shape_mask).to_numpy(dtype=bool)
    random_bw_component_center = pd.to_numeric(
        reference_prepared.get("random_bw_component_center_us", pd.Series(0.0, index=reference_prepared.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    random_latency_component_center = pd.to_numeric(
        reference_prepared.get("random_latency_component_center_us", pd.Series(0.0, index=reference_prepared.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    random_bw_up_mask = (
        op_type.eq("Gather").to_numpy(dtype=bool)
        & (random_bw_component_center > 1e-6)
        & (random_bw_component_center >= random_latency_component_center - 1e-6)
    )
    random_bw_down_mask = op_type.eq("Gather").to_numpy(dtype=bool)

    applicability_by_field = {
        "instruction.fp_throughput_per_cycle.vector_sp_add": add_like_mask,
        "instruction.fp_throughput_per_cycle.vector_sp_mul": mul_like_mask,
        "instruction.fp_throughput_per_cycle.vector_sp_fma": fma_like_mask,
        "instruction.simd_width_bits": compute_like_mask,
        "core.cpu_clock": compute_like_mask,
    }

    rows: list[dict[str, Any]] = []
    for spec in specs:
        original = _yaml_nested_get(reference_payload, str(spec["field"]))
        if original is None:
            continue
        directions: list[tuple[str, Any]]
        if spec.get("mode") == "simd":
            current = int(float(original))
            candidates = sorted({int(value) for value in (64, 128, 256, 512)})
            lower = max((value for value in candidates if value < current), default=current)
            upper = min((value for value in candidates if value > current), default=current)
            directions = [("up", upper), ("down", lower)]
        else:
            base_value, suffix = _parse_mutation_quantity(original)
            directions = [
                ("up", _format_mutation_quantity(base_value * float(spec["up"]), suffix)),
                ("down", _format_mutation_quantity(base_value * float(spec["down"]), suffix)),
            ]

        for direction, mutated_value in directions:
            if mutated_value == original:
                continue
            applicable_mask = applicability_by_field.get(str(spec["field"]), np.ones(len(reference_prepared), dtype=bool))
            if str(spec["group"]) == "random_bw":
                if direction == "up":
                    applicable_mask = applicable_mask & random_bw_up_mask
                else:
                    applicable_mask = applicable_mask & random_bw_down_mask
            applicable_rows = int(np.count_nonzero(applicable_mask))
            if applicable_rows == 0:
                rows.append(
                    {
                        "policy": policy,
                        "dataset": dataset,
                        "route": route,
                        "stage_used": route_stage(route),
                        "scale": f"{int(scale)}%",
                        "field_name": str(spec["field"]),
                        "field_group": str(spec["group"]),
                        "direction": str(direction),
                        "original_value": str(original),
                        "mutated_value": str(mutated_value),
                        "expected_modules": ",".join(spec["expected"]),
                        "weights_sha256_before": reference_hash,
                        "weights_sha256_after": str(bundle.get("model_sha256", "")),
                        "weights_unchanged": reference_hash == str(bundle.get("model_sha256", "")),
                        "prediction_succeeded": True,
                        "rows": int(len(reference_prepared)),
                        "applicable": False,
                        "applicable_rows": 0,
                        "mean_abs_pred_delta_us": 0.0,
                        "max_abs_pred_delta_us": 0.0,
                        "total_abs_svc_delta_us": 0.0,
                        "target_module_share": 1.0,
                        "max_non_target_delta_us": 0.0,
                        "dominant_module": "",
                        "svc_descriptor_delta_us": 0.0,
                        "svc_stream_mem_delta_us": 0.0,
                        "svc_random_mem_delta_us": 0.0,
                        "svc_compute_delta_us": 0.0,
                        "svc_reorder_delta_us": 0.0,
                        "svc_sync_delta_us": 0.0,
                        "svc_dispatch_delta_us": 0.0,
                        "expectation_pass": True,
                        "monotonic_pass": True,
                        "error": "",
                    }
                )
                continue
            payload = json.loads(json.dumps(reference_payload))
            _yaml_nested_set(payload, str(spec["field"]), mutated_value)
            try:
                perturbed_prepared, perturbed_pred = predict_fn(bundle, frame, override_hw_config=payload)
                delta_by_module = {
                    column: float(
                        np.abs(
                            pd.to_numeric(perturbed_prepared[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)[applicable_mask]
                            - pd.to_numeric(reference_prepared[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)[applicable_mask]
                        ).sum()
                    )
                    for column in svc_columns
                }
                total_delta = sum(delta_by_module.values())
                target_delta = sum(delta_by_module[column] for column in spec["expected"])
                non_target_delta = [delta_by_module[column] for column in svc_columns if column not in spec["expected"]]
                max_non_target_delta = max(non_target_delta) if non_target_delta else 0.0
                if total_delta <= 1e-12:
                    target_share = 1.0
                    max_non_target_delta = 0.0
                    dominant_module = ""
                else:
                    target_share = float(target_delta / max(total_delta, 1e-9))
                    dominant_module = max(delta_by_module.items(), key=lambda item: item[1])[0] if delta_by_module else ""

                ref_target = reference_prepared.loc[applicable_mask, spec["expected"]].sum(axis=1).to_numpy(dtype=float)
                pert_target = perturbed_prepared.loc[applicable_mask, spec["expected"]].sum(axis=1).to_numpy(dtype=float)
                if spec["group"] == "launch":
                    if direction == "up":
                        monotonic_pass = bool(np.all(pert_target >= ref_target - 1e-6))
                    else:
                        monotonic_pass = bool(np.all(pert_target <= ref_target + 1e-6))
                else:
                    if direction == "up":
                        monotonic_pass = bool(np.all(pert_target <= ref_target + 1e-6))
                    else:
                        monotonic_pass = bool(np.all(pert_target >= ref_target - 1e-6))

                if total_delta <= 1e-12:
                    expectation_pass = True
                elif len(spec["expected"]) == 1:
                    expectation_pass = bool(target_share >= 0.80 and max_non_target_delta <= target_delta * 0.10 + 1e-9)
                else:
                    expectation_pass = bool(target_share >= 0.80 and max_non_target_delta <= target_delta * 0.20 + 1e-9)

                prediction_delta = np.abs(np.asarray(reference_pred, dtype=float)[applicable_mask] - np.asarray(perturbed_pred, dtype=float)[applicable_mask])
                rows.append(
                    {
                        "policy": policy,
                        "dataset": dataset,
                        "route": route,
                        "stage_used": route_stage(route),
                        "scale": f"{int(scale)}%",
                        "field_name": str(spec["field"]),
                        "field_group": str(spec["group"]),
                        "direction": str(direction),
                        "original_value": str(original),
                        "mutated_value": str(mutated_value),
                        "expected_modules": ",".join(spec["expected"]),
                        "weights_sha256_before": reference_hash,
                        "weights_sha256_after": str(bundle.get("model_sha256", "")),
                        "weights_unchanged": reference_hash == str(bundle.get("model_sha256", "")),
                        "prediction_succeeded": True,
                        "rows": int(len(perturbed_prepared)),
                        "applicable": True,
                        "applicable_rows": applicable_rows,
                        "mean_abs_pred_delta_us": float(prediction_delta.mean()) if len(prediction_delta) else 0.0,
                        "max_abs_pred_delta_us": float(prediction_delta.max()) if len(prediction_delta) else 0.0,
                        "total_abs_svc_delta_us": float(total_delta),
                        "target_module_share": float(target_share),
                        "max_non_target_delta_us": float(max_non_target_delta),
                        "dominant_module": dominant_module,
                        "svc_descriptor_delta_us": float(delta_by_module["svc_descriptor_us"]),
                        "svc_stream_mem_delta_us": float(delta_by_module["svc_stream_mem_us"]),
                        "svc_random_mem_delta_us": float(delta_by_module["svc_random_mem_us"]),
                        "svc_compute_delta_us": float(delta_by_module["svc_compute_us"]),
                        "svc_reorder_delta_us": float(delta_by_module["svc_reorder_us"]),
                        "svc_sync_delta_us": float(delta_by_module["svc_sync_us"]),
                        "svc_dispatch_delta_us": float(delta_by_module["svc_dispatch_us"]),
                        "expectation_pass": expectation_pass,
                        "monotonic_pass": monotonic_pass,
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "policy": policy,
                        "dataset": dataset,
                        "route": route,
                        "stage_used": route_stage(route),
                        "scale": f"{int(scale)}%",
                        "field_name": str(spec["field"]),
                        "field_group": str(spec["group"]),
                        "direction": str(direction),
                        "original_value": str(original),
                        "mutated_value": str(mutated_value),
                        "expected_modules": ",".join(spec["expected"]),
                        "weights_sha256_before": reference_hash,
                        "weights_sha256_after": str(bundle.get("model_sha256", "")),
                        "weights_unchanged": reference_hash == str(bundle.get("model_sha256", "")),
                        "prediction_succeeded": False,
                        "rows": 0,
                        "applicable": applicable_rows > 0,
                        "applicable_rows": applicable_rows,
                        "mean_abs_pred_delta_us": 0.0,
                        "max_abs_pred_delta_us": 0.0,
                        "total_abs_svc_delta_us": 0.0,
                        "target_module_share": 0.0,
                        "max_non_target_delta_us": 0.0,
                        "dominant_module": "",
                        "svc_descriptor_delta_us": 0.0,
                        "svc_stream_mem_delta_us": 0.0,
                        "svc_random_mem_delta_us": 0.0,
                        "svc_compute_delta_us": 0.0,
                        "svc_reorder_delta_us": 0.0,
                        "svc_sync_delta_us": 0.0,
                        "svc_dispatch_delta_us": 0.0,
                        "expectation_pass": False,
                        "monotonic_pass": False,
                        "error": str(exc),
                    }
                )
    return rows


def _family_scaling_audit_for_root(
    *,
    dataset_root: Path,
    dataset_name: str,
    hw_config: Path,
    passes: int,
) -> pd.DataFrame:
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    phase_scope = str(manifest.get("phase_scope", "unknown"))
    frame = pd.read_csv(dataset_root / "dataset_full.csv", low_memory=False)
    params = base_run._build_analytical_params(frame, hw_config, passes=passes)
    prepared = prepare_semantic_static_frame(frame, hw_config, params)
    axes = ["num_threads", "inter_threads"] if phase_scope == PHASE2_SCOPE else ["batch_size", "num_indices_per_lookup", "arch_embedding_size", "arch_mlp_bot"]
    rows: list[dict[str, Any]] = []
    prepared["ana_ratio"] = (
        pd.to_numeric(prepared[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0)
        / pd.to_numeric(prepared["ana_calib_total_us"], errors="coerce").fillna(0.0).clip(lower=1e-6)
    )
    for axis in axes:
        if axis not in prepared.columns:
            continue
        for (axis_value, family_bucket, op_type), group in prepared.groupby([axis, "family_bucket", "op_type"], sort=True):
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_root": str(dataset_root),
                    "phase_scope": phase_scope,
                    "axis": axis,
                    "axis_value": axis_value,
                    "family_bucket": str(family_bucket),
                    "op_type": str(op_type),
                    "rows": int(len(group)),
                    "actual_median_us": float(pd.to_numeric(group[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).median()),
                    "analytical_prior_median_us": float(pd.to_numeric(group["ana_calib_total_us"], errors="coerce").fillna(0.0).median()),
                    "actual_over_analytical_median": float(pd.to_numeric(group["ana_ratio"], errors="coerce").fillna(0.0).median()),
                    "actual_p90_us": float(pd.to_numeric(group[base_run.TARGET_US_COLUMN], errors="coerce").fillna(0.0).quantile(0.90)),
                    "analytical_prior_p90_us": float(pd.to_numeric(group["ana_calib_total_us"], errors="coerce").fillna(0.0).quantile(0.90)),
                    "actual_over_analytical_p90": float(pd.to_numeric(group["ana_ratio"], errors="coerce").fillna(0.0).quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def _plot_metric(summary: pd.DataFrame, *, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for (dataset, route), group in summary.groupby(["dataset", "route"], sort=True):
        ordered = group.copy()
        ordered["scale_num"] = ordered["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        ordered = ordered.sort_values("scale_num")
        ax.plot(ordered["scale_num"], ordered[metric], marker="o", label=f"{dataset}:{route}")
    ax.set_xlabel("Fit Scale (%)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs fit scale")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _evaluate_dataset(
    *,
    policy: str,
    dataset_name: str,
    dataset_root: Path,
    artifact_root: Path,
    hw_config: Path,
    alternate_hw_config: Path | None,
    scales: list[int],
    routes: list[str],
    passes: int,
    seed: int,
    enable_e2e_pipeline: bool,
    op_thread_scale_mode: str = "off",
    op_thread_scale_config: dict[str, Any] | None = None,
    op_thread_scale_min_rows: int = 4,
    op_thread_scale_output: Path | None = None,
    export_model_bundles: bool = False,
    model_bundle_output_dir: Path | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, Any],
]:
    _progress("dataset_eval_start", policy=policy, dataset=dataset_name, detail=str(dataset_root))
    split_manifest = _load_split_manifest(dataset_root)
    _validate_split_manifest(policy, dataset_name, dataset_root, split_manifest)
    _progress("load_raw_splits", policy=policy, dataset=dataset_name)
    raw_splits = _load_raw_splits(dataset_root, include_pipeline_splits=enable_e2e_pipeline)
    _progress("load_raw_splits_done", policy=policy, dataset=dataset_name)
    for split_frame in raw_splits.values():
        split_frame.attrs["dataset_root"] = str(dataset_root)
    fit_pool = _fit_pool(raw_splits)
    fit_pool.attrs["dataset_root"] = str(dataset_root)
    pipeline_fit_pool = raw_splits.get("e2e_train", fit_pool)
    pipeline_fit_pool.attrs["dataset_root"] = str(dataset_root)
    test_raw = raw_splits["test"]
    test_raw.attrs["dataset_root"] = str(dataset_root)
    pipeline_test_raw = raw_splits.get("e2e_test_raw", raw_splits.get("test_raw", test_raw))
    pipeline_test_raw.attrs["dataset_root"] = str(dataset_root)
    pipeline_requires_alt_fit = pipeline_fit_pool is not fit_pool
    rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    whitebox_rows: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []
    perf_feature_rows: list[dict[str, Any]] = []
    mlp_enablement_rows: list[dict[str, Any]] = []
    hw_transfer_rows: list[dict[str, Any]] = []
    module_sensitivity_rows: list[dict[str, Any]] = []
    analytical_interval_rows: list[dict[str, Any]] = []
    analytical_trend_rows: list[dict[str, Any]] = []
    analytical_bottleneck_rows: list[dict[str, Any]] = []
    analytical_state_rows: list[dict[str, Any]] = []
    phase_error_rows: list[dict[str, Any]] = []
    prior_gap_rows: list[dict[str, Any]] = []
    rf_candidate_feature_rows: list[dict[str, Any]] = []
    rf_pruned_feature_rows: list[dict[str, Any]] = []
    rf_group_feature_count_rows: list[dict[str, Any]] = []
    prior_to_rf_gain_rows: list[dict[str, Any]] = []
    torch_training_summary_rows: list[dict[str, Any]] = []
    model_bundle_manifest_rows: list[dict[str, Any]] = []
    prediction_error_frames: list[pd.DataFrame] = []
    pipeline_prediction_error_frames: list[pd.DataFrame] = []
    op_thread_scale_paths: list[str] = []
    _progress("row_universe_guard_start", policy=policy, dataset=dataset_name)
    row_universe_guard_rows = _row_universe_guard_rows(
        policy=policy,
        dataset=dataset_name,
        dataset_root=dataset_root,
        split_manifest=split_manifest,
    )
    _progress("row_universe_guard_done", policy=policy, dataset=dataset_name)
    graph_mapping_train = raw_splits["train"]
    graph_mapping_test = raw_splits["test"]
    _progress("graph_mapping_audit_ready", policy=policy, dataset=dataset_name)
    graph_feature_mapping_rows = [
        *_graph_feature_mapping_rows(policy=policy, dataset=dataset_name, split_name="train", frame=graph_mapping_train),
        *_graph_feature_mapping_rows(policy=policy, dataset=dataset_name, split_name="test", frame=graph_mapping_test),
    ]
    source_input_dataset_root = str(split_manifest.get("input_dataset_root", "")) if dataset_name == "source" else ""
    source_phase_scope = str(split_manifest.get("phase_scope", "")) if dataset_name == "source" else ""
    for scale in scales:
        _progress("scale_start", policy=policy, dataset=dataset_name, scale=scale, detail=f"fit_pool_rows={len(fit_pool)}")
        sampled = _sample_fit_subset(fit_pool, scale, seed, salt=sum(ord(ch) for ch in dataset_name))
        sampled.attrs["dataset_root"] = str(dataset_root)
        base_run._assert_no_leak(sampled, test_raw)
        pipeline_sampled = sampled
        if pipeline_requires_alt_fit:
            pipeline_sampled = _sample_fit_subset(
                pipeline_fit_pool,
                scale,
                seed,
                salt=sum(ord(ch) for ch in dataset_name) + 9973,
            )
            pipeline_sampled.attrs["dataset_root"] = str(dataset_root)
            base_run._assert_no_leak(pipeline_sampled, pipeline_test_raw)
        _progress("prepare_analytical_params", policy=policy, dataset=dataset_name, scale=scale, detail=f"sampled_rows={len(sampled)}")
        params = base_run._build_analytical_params(sampled, hw_config, passes=passes)
        fit_prepared = base_run._prepare_strict_semantic_free_frame(sampled, hw_config, params)
        test_prepared = base_run._prepare_strict_semantic_free_frame(test_raw, hw_config, params)
        pipeline_params = params
        pipeline_fit_prepared = fit_prepared
        if pipeline_requires_alt_fit:
            pipeline_params = base_run._build_analytical_params(pipeline_sampled, hw_config, passes=passes)
            pipeline_fit_prepared = base_run._prepare_strict_semantic_free_frame(pipeline_sampled, hw_config, pipeline_params)
        pipeline_test_prepared = (
            base_run._prepare_strict_semantic_free_frame(pipeline_test_raw, hw_config, pipeline_params)
            if pipeline_test_raw is not test_raw or pipeline_requires_alt_fit
            else test_prepared
        )
        analytical_metrics = None
        bundle = base_run._fit_calibrator_bundle(fit_prepared, mode="analytical")
        selected_family = str(bundle.get("model_family", base_run.DEFAULT_CALIBRATOR_FAMILY))
        pred = base_run._predict_calibrator_bundle(bundle, test_prepared)
        chain_metrics = base_run._prepared_metrics(test_prepared, pred)
        if base_run.EXP1_BASE_ROUTE in routes:
            rows.append(
                _summary_row(
                    policy=policy,
                    dataset=dataset_name,
                    route=base_run.EXP1_BASE_ROUTE,
                    scale=int(scale),
                    fit_rows=len(sampled),
                    selected_family=selected_family,
                    metrics=chain_metrics,
                    analytical_metrics=analytical_metrics,
                    source_input_dataset_root=source_input_dataset_root,
                    source_phase_scope=source_phase_scope,
                )
            )
            breakdown_rows.extend(
                _family_breakdown_rows(
                    policy=policy,
                    dataset=dataset_name,
                    route=base_run.EXP1_BASE_ROUTE,
                    scale=int(scale),
                    prepared_frame=test_prepared,
                    pred=pred,
                )
            )

        shape_tree_pred = base_run._predict_shape_tree_plus_problem_lookup(fit_prepared, test_prepared, pred)
        shape_tree_metrics = base_run._prepared_metrics(test_prepared, shape_tree_pred)
        if base_run.SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE in routes:
            rows.append(
                _summary_row(
                    policy=policy,
                    dataset=dataset_name,
                    route=base_run.SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
                    scale=int(scale),
                    fit_rows=len(sampled),
                    selected_family=base_run._shape_tree_plus_selected_family(selected_family),
                    metrics=shape_tree_metrics,
                    analytical_metrics=analytical_metrics,
                    source_input_dataset_root=source_input_dataset_root,
                    source_phase_scope=source_phase_scope,
                )
            )
            breakdown_rows.extend(
                _family_breakdown_rows(
                    policy=policy,
                    dataset=dataset_name,
                    route=base_run.SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
                    scale=int(scale),
                    prepared_frame=test_prepared,
                    pred=shape_tree_pred,
                )
            )
        if any(
            route in routes
            for route in (
                ML_RF_ROUTE,
                ML_RF_SHAPE_TREE_ROUTE,
                ML_FAMILY_RF_ROUTE,
                WHITEBOX_PRIOR_ONLY_ROUTE,
                WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                WHITEBOX_ORT_KERNEL_RF_ROUTE,
                WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                WHITEBOX_GATHER_SLICE_RF_ROUTE,
            )
        ):
            _progress("prepare_semantic_frames", policy=policy, dataset=dataset_name, scale=scale)
            fit_semantic = prepare_semantic_static_frame(sampled, hw_config, params)
            test_semantic = prepare_semantic_static_frame(test_raw, hw_config, params)
            pipeline_fit_semantic = fit_semantic
            if pipeline_requires_alt_fit:
                pipeline_fit_semantic = prepare_semantic_static_frame(pipeline_sampled, hw_config, pipeline_params)
            pipeline_test_semantic = (
                prepare_semantic_static_frame(pipeline_test_raw, hw_config, pipeline_params)
                if pipeline_test_raw is not test_raw or pipeline_requires_alt_fit
                else test_semantic
            )
            current_op_thread_scale_config = op_thread_scale_config
            if op_thread_scale_mode == "fit":
                scale_fit_frame = pipeline_fit_semantic if pipeline_requires_alt_fit else fit_semantic
                current_op_thread_scale_config = fit_op_thread_scale_config(
                    scale_fit_frame,
                    hw_config,
                    policy=policy,
                    min_group_rows=int(op_thread_scale_min_rows),
                )
                scale_config_path = (
                    op_thread_scale_output
                    if op_thread_scale_output is not None
                    else artifact_root / f"op_thread_scale_config_{policy}_{dataset_name}_{int(scale)}.json"
                )
                save_op_thread_scale_config(current_op_thread_scale_config, scale_config_path)
                op_thread_scale_paths.append(str(scale_config_path))
                coeff_frame = op_thread_scale_coefficients_frame(current_op_thread_scale_config)
                if not coeff_frame.empty:
                    coeff_frame.to_csv(scale_config_path.with_suffix(".csv"), index=False)
                _progress(
                    "fit_op_thread_scale_done",
                    policy=policy,
                    dataset=dataset_name,
                    scale=scale,
                    detail=f"groups={current_op_thread_scale_config.get('fit_source', {}).get('groups', 0)} path={scale_config_path}",
                )
            if any(route in routes for route in (ML_RF_ROUTE, ML_RF_SHAPE_TREE_ROUTE, ML_FAMILY_RF_ROUTE)):
                ml_bundle = fit_semantic_thread_bundle(fit_semantic, model_family="rf")
                ml_pred = predict_semantic_thread_bundle(ml_bundle, test_semantic)
                ml_metrics = base_run._prepared_metrics(test_semantic, ml_pred)
                if ML_RF_ROUTE in routes:
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=ML_RF_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="semantic_thread_rf",
                            metrics=ml_metrics,
                            analytical_metrics=analytical_metrics,
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=ML_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_semantic,
                            pred=ml_pred,
                        )
                    )
                ml_shape_pred = base_run._predict_shape_tree_plus_problem_lookup(fit_semantic, test_semantic, ml_pred)
                ml_shape_metrics = base_run._prepared_metrics(test_semantic, ml_shape_pred)
                if ML_RF_SHAPE_TREE_ROUTE in routes:
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=ML_RF_SHAPE_TREE_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="semantic_thread_rf+shape_tree_rf_plus_lookup",
                            metrics=ml_shape_metrics,
                            analytical_metrics=analytical_metrics,
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=ML_RF_SHAPE_TREE_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_semantic,
                            pred=ml_shape_pred,
                        )
                    )
            if ML_FAMILY_RF_ROUTE in routes:
                family_bundle = fit_semantic_thread_family_bundle(fit_semantic)
                family_pred = predict_semantic_thread_family_bundle(family_bundle, test_semantic)
                family_metrics = base_run._prepared_metrics(test_semantic, family_pred)
                rows.append(
                    _summary_row(
                        policy=policy,
                        dataset=dataset_name,
                        route=ML_FAMILY_RF_ROUTE,
                        scale=int(scale),
                        fit_rows=len(sampled),
                        selected_family="semantic_thread_family_rf",
                        metrics=family_metrics,
                        analytical_metrics=analytical_metrics,
                        source_input_dataset_root=source_input_dataset_root,
                        source_phase_scope=source_phase_scope,
                    )
                )
                breakdown_rows.extend(
                    _family_breakdown_rows(
                        policy=policy,
                        dataset=dataset_name,
                        route=ML_FAMILY_RF_ROUTE,
                        scale=int(scale),
                        prepared_frame=test_semantic,
                        pred=family_pred,
                        )
                    )
            if policy in WHITEBOX_HARD_POLICIES and any(
                route in routes
                for route in (
                    WHITEBOX_PRIOR_ONLY_ROUTE,
                    WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                    WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                    WHITEBOX_ORT_KERNEL_RF_ROUTE,
                    WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                    WHITEBOX_GATHER_SLICE_RF_ROUTE,
                    WHITEBOX_WORKLOAD_BENCHMARK_ROUTE,
                )
            ):
                if WHITEBOX_PRIOR_ONLY_ROUTE in routes:
                    _progress("fit_whitebox_prior", policy=policy, dataset=dataset_name, scale=scale)
                    whitebox_prior_bundle = fit_whitebox_family_prior_bundle(
                        fit_semantic,
                        hw_config,
                        policy=policy,
                        op_thread_scale_config=current_op_thread_scale_config,
                    )
                    if dataset_name == "source" and whitebox_prior_bundle.get("hw_params", {}).get("missing_whitebox_fields"):
                        raise ValueError(
                            f"Whitebox source evaluation requires measured fields in {hw_config}: "
                            f"{', '.join(whitebox_prior_bundle['hw_params']['missing_whitebox_fields'])}"
                        )
                    test_whitebox, whitebox_pred = predict_whitebox_family_prior(whitebox_prior_bundle, test_semantic)
                    whitebox_prior_metrics = base_run._prepared_metrics(test_whitebox, whitebox_pred)
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="whitebox_family_prior_only_v2",
                            metrics=whitebox_prior_metrics,
                            analytical_metrics=analytical_metrics,
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                            pred=whitebox_pred,
                        )
                    )
                    prediction_error_frames.append(
                        _prediction_error_frame(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                            pred=whitebox_pred,
                        )
                    )
                    if pipeline_test_semantic is not test_semantic:
                        pipeline_whitebox_prior_bundle = (
                            fit_whitebox_family_prior_bundle(
                                pipeline_fit_semantic,
                                hw_config,
                                policy=policy,
                                op_thread_scale_config=current_op_thread_scale_config,
                            )
                            if pipeline_requires_alt_fit
                            else whitebox_prior_bundle
                        )
                        pipeline_test_whitebox, pipeline_whitebox_pred = predict_whitebox_family_prior(
                            pipeline_whitebox_prior_bundle,
                            pipeline_test_semantic,
                        )
                        pipeline_prediction_error_frames.append(
                            _prediction_error_frame(
                                policy=policy,
                                dataset=dataset_name,
                                route=WHITEBOX_PRIOR_ONLY_ROUTE,
                                scale=int(scale),
                                prepared_frame=pipeline_test_whitebox,
                                pred=pipeline_whitebox_pred,
                            )
                        )
                    analytical_interval_rows.extend(
                        _analytical_interval_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                        )
                    )
                    analytical_trend_rows.extend(
                        _analytical_trend_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                        )
                    )
                    analytical_bottleneck_rows.extend(
                        _analytical_bottleneck_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                        )
                    )
                    analytical_state_rows.extend(
                        _analytical_state_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                        )
                    )
                    phase_error_rows.extend(
                        _phase_error_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                            pred=whitebox_pred,
                        )
                    )
                    prior_gap_rows.extend(
                        _prior_gap_decomposition_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                        )
                    )
                    module_sensitivity_rows.extend(
                        _module_sensitivity_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_prior_bundle,
                            frame=test_semantic,
                            reference_prepared=test_whitebox,
                            reference_pred=whitebox_pred,
                            hw_config=hw_config,
                            predict_fn=predict_whitebox_family_prior,
                        )
                    )
                if WHITEBOX_PRIOR_SIMPLE_RF_ROUTE in routes:
                    _progress("fit_whitebox_simple_rf", policy=policy, dataset=dataset_name, scale=scale)
                    whitebox_rf_bundle = fit_whitebox_family_simple_rf_bundle(
                        fit_semantic,
                        hw_config,
                        policy=policy,
                        op_thread_scale_config=current_op_thread_scale_config,
                    )
                    rf_candidate_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_rf_bundle.get("rf_candidate_feature_rows", [])
                        ]
                    )
                    rf_pruned_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_rf_bundle.get("rf_pruned_feature_rows", [])
                        ]
                    )
                    rf_group_feature_count_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_rf_bundle.get("rf_group_feature_count_rows", [])
                        ]
                    )
                    prior_for_gain_frame, prior_for_gain_pred = predict_whitebox_family_prior(whitebox_rf_bundle["prior_bundle"], test_semantic)
                    test_whitebox_rf, whitebox_rf_pred = predict_whitebox_family_simple_rf(whitebox_rf_bundle, test_semantic)
                    whitebox_rf_metrics = base_run._prepared_metrics(test_whitebox_rf, whitebox_rf_pred)
                    prior_to_rf_gain_rows.extend(
                        _prior_to_rf_gain_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prior_frame=prior_for_gain_frame,
                            prior_pred=prior_for_gain_pred,
                            rf_frame=test_whitebox_rf,
                            rf_pred=whitebox_rf_pred,
                        )
                    )
                    invariance_rows.extend(
                        _whitebox_label_invariance_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_rf_bundle,
                            frame=test_semantic,
                            reference_pred=whitebox_rf_pred,
                            predict_fn=predict_whitebox_family_simple_rf,
                        )
                    )
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="whitebox_family_prior_plus_simple_rf_v2",
                            metrics=whitebox_rf_metrics,
                            analytical_metrics=analytical_metrics,
                            train_backend=str(whitebox_rf_bundle.get("train_backend", "")),
                            train_env=str(whitebox_rf_bundle.get("train_env", "")),
                            ml_input_source=str(whitebox_rf_bundle.get("ml_input_source", "")),
                            ml_raw_sw_features_used=bool(whitebox_rf_bundle.get("ml_raw_sw_features_used", False)),
                            ml_raw_hw_params_used=bool(whitebox_rf_bundle.get("ml_raw_hw_params_used", False)),
                            hw_refit_required=bool(whitebox_rf_bundle.get("hw_refit_required", False)),
                            family_mlp_enabled_count=int(whitebox_rf_bundle.get("family_mlp_enabled_count", 0)),
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_rf,
                            pred=whitebox_rf_pred,
                        )
                    )
                    prediction_error_frames.append(
                        _prediction_error_frame(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_rf,
                            pred=whitebox_rf_pred,
                        )
                    )
                    if pipeline_test_semantic is not test_semantic:
                        pipeline_whitebox_rf_bundle = (
                            fit_whitebox_family_simple_rf_bundle(
                                pipeline_fit_semantic,
                                hw_config,
                                policy=policy,
                                op_thread_scale_config=current_op_thread_scale_config,
                            )
                            if pipeline_requires_alt_fit
                            else whitebox_rf_bundle
                        )
                        pipeline_test_whitebox_rf, pipeline_whitebox_rf_pred = predict_whitebox_family_simple_rf(
                            pipeline_whitebox_rf_bundle,
                            pipeline_test_semantic,
                        )
                        pipeline_prediction_error_frames.append(
                            _prediction_error_frame(
                                policy=policy,
                                dataset=dataset_name,
                                route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                                scale=int(scale),
                                prepared_frame=pipeline_test_whitebox_rf,
                                pred=pipeline_whitebox_rf_pred,
                            )
                        )
                    analytical_interval_rows.extend(
                        _analytical_interval_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_rf,
                        )
                    )
                    analytical_trend_rows.extend(
                        _analytical_trend_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_rf,
                        )
                    )
                    analytical_bottleneck_rows.extend(
                        _analytical_bottleneck_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_rf,
                        )
                    )
                    analytical_state_rows.extend(
                        _analytical_state_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_rf,
                        )
                    )
                    whitebox_rows.extend(
                        _whitebox_source_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_rf,
                            pred=whitebox_rf_pred,
                        )
                    )
                    perf_feature_rows.extend(
                        _whitebox_perf_feature_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_rf,
                        )
                    )
                    mlp_enablement_rows.extend(
                        _family_mlp_enablement_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_rf_bundle,
                        )
                    )
                    hw_transfer_rows.extend(
                        _hw_transfer_audit_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_rf_bundle,
                            frame=test_semantic,
                            reference_pred=whitebox_rf_pred,
                            alternate_hw_config=alternate_hw_config,
                        )
                    )
                    module_sensitivity_rows.extend(
                        _module_sensitivity_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_rf_bundle,
                            frame=test_semantic,
                            reference_prepared=test_whitebox_rf,
                            reference_pred=whitebox_rf_pred,
                            hw_config=hw_config,
                            predict_fn=predict_whitebox_family_simple_rf,
                        )
                    )
                    torch_training_summary_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                "train_backend": str(whitebox_rf_bundle.get("train_backend", "")),
                                "train_env": str(whitebox_rf_bundle.get("train_env", "")),
                                "train_device": str(whitebox_rf_bundle.get("train_device", "")),
                                "npu_device_id": int(whitebox_rf_bundle.get("npu_device_id", 0)),
                                "family_mlp_enabled_count": int(whitebox_rf_bundle.get("family_mlp_enabled_count", 0)),
                                "family_training_summaries": whitebox_rf_bundle.get("family_training_summaries", []),
                            }
                        ]
                    )
                if WHITEBOX_PHYSICS_GUARDED_RF_ROUTE in routes:
                    _progress("fit_whitebox_physics_guarded_rf_v3", policy=policy, dataset=dataset_name, scale=scale)
                    physics_guarded_bundle = fit_whitebox_family_physics_guarded_rf_bundle(
                        fit_semantic,
                        hw_config,
                        policy=policy,
                        op_thread_scale_config=current_op_thread_scale_config,
                    )
                    rf_candidate_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in physics_guarded_bundle.get("rf_candidate_feature_rows", [])
                        ]
                    )
                    rf_pruned_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in physics_guarded_bundle.get("rf_pruned_feature_rows", [])
                        ]
                    )
                    rf_group_feature_count_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in physics_guarded_bundle.get("rf_group_feature_count_rows", [])
                        ]
                    )
                    prior_for_guarded_frame, prior_for_guarded_pred = predict_whitebox_family_prior(
                        physics_guarded_bundle["prior_bundle"],
                        test_semantic,
                    )
                    test_guarded_rf, guarded_rf_pred = predict_whitebox_family_physics_guarded_rf(
                        physics_guarded_bundle,
                        test_semantic,
                    )
                    guarded_rf_metrics = base_run._prepared_metrics(test_guarded_rf, guarded_rf_pred)
                    prior_to_rf_gain_rows.extend(
                        _prior_to_rf_gain_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            prior_frame=prior_for_guarded_frame,
                            prior_pred=prior_for_guarded_pred,
                            rf_frame=test_guarded_rf,
                            rf_pred=guarded_rf_pred,
                        )
                    )
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="whitebox_family_prior_plus_physics_guarded_rf_v3",
                            metrics=guarded_rf_metrics,
                            analytical_metrics=analytical_metrics,
                            train_backend=str(physics_guarded_bundle.get("train_backend", "")),
                            train_env=str(physics_guarded_bundle.get("train_env", "")),
                            ml_input_source=str(physics_guarded_bundle.get("ml_input_source", "")),
                            ml_raw_sw_features_used=bool(physics_guarded_bundle.get("ml_raw_sw_features_used", False)),
                            ml_raw_hw_params_used=bool(physics_guarded_bundle.get("ml_raw_hw_params_used", False)),
                            hw_refit_required=bool(physics_guarded_bundle.get("hw_refit_required", False)),
                            family_mlp_enabled_count=int(physics_guarded_bundle.get("family_mlp_enabled_count", 0)),
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_guarded_rf,
                            pred=guarded_rf_pred,
                        )
                    )
                    prediction_error_frames.append(
                        _prediction_error_frame(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_guarded_rf,
                            pred=guarded_rf_pred,
                        )
                    )
                    export_guarded_bundle = physics_guarded_bundle
                    if pipeline_test_semantic is not test_semantic:
                        pipeline_guarded_bundle = (
                            fit_whitebox_family_physics_guarded_rf_bundle(
                                pipeline_fit_semantic,
                                hw_config,
                                policy=policy,
                                op_thread_scale_config=current_op_thread_scale_config,
                            )
                            if pipeline_requires_alt_fit
                            else physics_guarded_bundle
                        )
                        export_guarded_bundle = pipeline_guarded_bundle
                        pipeline_test_guarded_rf, pipeline_guarded_rf_pred = predict_whitebox_family_physics_guarded_rf(
                            pipeline_guarded_bundle,
                            pipeline_test_semantic,
                        )
                        pipeline_prediction_error_frames.append(
                            _prediction_error_frame(
                                policy=policy,
                                dataset=dataset_name,
                                route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                                scale=int(scale),
                                prepared_frame=pipeline_test_guarded_rf,
                                pred=pipeline_guarded_rf_pred,
                            )
                        )
                    if bool(export_model_bundles):
                        export_dir = model_bundle_output_dir if model_bundle_output_dir is not None else artifact_root / "model_bundles"
                        model_bundle_manifest_rows.append(
                            _export_static_inference_bundle(
                                bundle=export_guarded_bundle,
                                output_dir=export_dir,
                                policy=policy,
                                dataset=dataset_name,
                                scale=int(scale),
                                route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                                hw_config=hw_config,
                                dataset_root=dataset_root,
                            )
                        )
                    whitebox_rows.extend(
                        _whitebox_source_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_guarded_rf,
                            pred=guarded_rf_pred,
                        )
                    )
                    perf_feature_rows.extend(
                        _whitebox_perf_feature_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_guarded_rf,
                        )
                    )
                    mlp_enablement_rows.extend(
                        _family_mlp_enablement_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            bundle=physics_guarded_bundle,
                        )
                    )
                    hw_transfer_rows.extend(
                        _hw_transfer_audit_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            bundle=physics_guarded_bundle,
                            frame=test_semantic,
                            reference_pred=guarded_rf_pred,
                            alternate_hw_config=alternate_hw_config,
                        )
                    )
                    module_sensitivity_rows.extend(
                        _module_sensitivity_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                            scale=int(scale),
                            bundle=physics_guarded_bundle,
                            frame=test_semantic,
                            reference_prepared=test_guarded_rf,
                            reference_pred=guarded_rf_pred,
                            hw_config=hw_config,
                            predict_fn=predict_whitebox_family_physics_guarded_rf,
                        )
                    )
                    torch_training_summary_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                "train_backend": str(physics_guarded_bundle.get("train_backend", "")),
                                "train_env": str(physics_guarded_bundle.get("train_env", "")),
                                "train_device": str(physics_guarded_bundle.get("train_device", "")),
                                "npu_device_id": int(physics_guarded_bundle.get("npu_device_id", 0)),
                                "family_mlp_enabled_count": int(physics_guarded_bundle.get("family_mlp_enabled_count", 0)),
                                "family_training_summaries": physics_guarded_bundle.get("family_training_summaries", []),
                            }
                        ]
                    )
                if WHITEBOX_ORT_KERNEL_RF_ROUTE in routes:
                    _progress("fit_whitebox_ort_kernel_rf_v3", policy=policy, dataset=dataset_name, scale=scale)
                    whitebox_ort_bundle = fit_whitebox_family_ort_kernel_rf_bundle(
                        fit_semantic,
                        hw_config,
                        policy=policy,
                        op_thread_scale_config=current_op_thread_scale_config,
                    )
                    rf_candidate_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_ORT_KERNEL_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_ort_bundle.get("rf_candidate_feature_rows", [])
                        ]
                    )
                    rf_pruned_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_ORT_KERNEL_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_ort_bundle.get("rf_pruned_feature_rows", [])
                        ]
                    )
                    rf_group_feature_count_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_ORT_KERNEL_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_ort_bundle.get("rf_group_feature_count_rows", [])
                        ]
                    )
                    prior_for_gain_frame, prior_for_gain_pred = predict_whitebox_family_prior(
                        whitebox_ort_bundle["prior_bundle"], test_semantic
                    )
                    test_whitebox_ort, whitebox_ort_pred = predict_whitebox_family_simple_rf(whitebox_ort_bundle, test_semantic)
                    whitebox_ort_metrics = base_run._prepared_metrics(test_whitebox_ort, whitebox_ort_pred)
                    prior_to_rf_gain_rows.extend(
                        _prior_to_rf_gain_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            prior_frame=prior_for_gain_frame,
                            prior_pred=prior_for_gain_pred,
                            rf_frame=test_whitebox_ort,
                            rf_pred=whitebox_ort_pred,
                        )
                    )
                    invariance_rows.extend(
                        _whitebox_label_invariance_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_ort_bundle,
                            frame=test_semantic,
                            reference_pred=whitebox_ort_pred,
                            predict_fn=predict_whitebox_family_simple_rf,
                        )
                    )
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="whitebox_family_prior_plus_ort_kernel_rf_v3",
                            metrics=whitebox_ort_metrics,
                            analytical_metrics=analytical_metrics,
                            train_backend=str(whitebox_ort_bundle.get("train_backend", "")),
                            train_env=str(whitebox_ort_bundle.get("train_env", "")),
                            ml_input_source=str(whitebox_ort_bundle.get("ml_input_source", "")),
                            ml_raw_sw_features_used=bool(whitebox_ort_bundle.get("ml_raw_sw_features_used", False)),
                            ml_raw_hw_params_used=bool(whitebox_ort_bundle.get("ml_raw_hw_params_used", False)),
                            hw_refit_required=bool(whitebox_ort_bundle.get("hw_refit_required", False)),
                            family_mlp_enabled_count=int(whitebox_ort_bundle.get("family_mlp_enabled_count", 0)),
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_ort,
                            pred=whitebox_ort_pred,
                        )
                    )
                    prediction_error_frames.append(
                        _prediction_error_frame(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_ort,
                            pred=whitebox_ort_pred,
                        )
                    )
                    if pipeline_test_semantic is not test_semantic:
                        pipeline_whitebox_ort_bundle = (
                            fit_whitebox_family_ort_kernel_rf_bundle(
                                pipeline_fit_semantic,
                                hw_config,
                                policy=policy,
                                op_thread_scale_config=current_op_thread_scale_config,
                            )
                            if pipeline_requires_alt_fit
                            else whitebox_ort_bundle
                        )
                        pipeline_test_whitebox_ort, pipeline_whitebox_ort_pred = predict_whitebox_family_simple_rf(
                            pipeline_whitebox_ort_bundle,
                            pipeline_test_semantic,
                        )
                        pipeline_prediction_error_frames.append(
                            _prediction_error_frame(
                                policy=policy,
                                dataset=dataset_name,
                                route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                                scale=int(scale),
                                prepared_frame=pipeline_test_whitebox_ort,
                                pred=pipeline_whitebox_ort_pred,
                            )
                        )
                    whitebox_rows.extend(
                        _whitebox_source_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_ort,
                            pred=whitebox_ort_pred,
                        )
                    )
                    perf_feature_rows.extend(
                        _whitebox_perf_feature_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_ort,
                        )
                    )
                    mlp_enablement_rows.extend(
                        _family_mlp_enablement_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_ort_bundle,
                        )
                    )
                    hw_transfer_rows.extend(
                        _hw_transfer_audit_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_ort_bundle,
                            frame=test_semantic,
                            reference_pred=whitebox_ort_pred,
                            alternate_hw_config=alternate_hw_config,
                        )
                    )
                    module_sensitivity_rows.extend(
                        _module_sensitivity_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_ORT_KERNEL_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_ort_bundle,
                            frame=test_semantic,
                            reference_prepared=test_whitebox_ort,
                            reference_pred=whitebox_ort_pred,
                            hw_config=hw_config,
                            predict_fn=predict_whitebox_family_simple_rf,
                        )
                    )
                    torch_training_summary_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_ORT_KERNEL_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                "train_backend": str(whitebox_ort_bundle.get("train_backend", "")),
                                "train_env": str(whitebox_ort_bundle.get("train_env", "")),
                                "train_device": str(whitebox_ort_bundle.get("train_device", "")),
                                "npu_device_id": int(whitebox_ort_bundle.get("npu_device_id", 0)),
                                "family_mlp_enabled_count": int(whitebox_ort_bundle.get("family_mlp_enabled_count", 0)),
                                "family_training_summaries": whitebox_ort_bundle.get("family_training_summaries", []),
                            }
                        ]
                    )
                if WHITEBOX_EMBEDDING_MEM_RF_ROUTE in routes:
                    _progress("fit_whitebox_embedding_mem_rf_v4", policy=policy, dataset=dataset_name, scale=scale)
                    whitebox_v4_bundle = fit_whitebox_family_embedding_mem_rf_bundle(
                        fit_semantic,
                        hw_config,
                        policy=policy,
                        op_thread_scale_config=current_op_thread_scale_config,
                    )
                    rf_candidate_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v4_bundle.get("rf_candidate_feature_rows", [])
                        ]
                    )
                    rf_pruned_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v4_bundle.get("rf_pruned_feature_rows", [])
                        ]
                    )
                    rf_group_feature_count_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v4_bundle.get("rf_group_feature_count_rows", [])
                        ]
                    )
                    prior_for_gain_frame, prior_for_gain_pred = predict_whitebox_family_prior(
                        whitebox_v4_bundle["prior_bundle"], test_semantic
                    )
                    test_whitebox_v4, whitebox_v4_pred = predict_whitebox_family_simple_rf(whitebox_v4_bundle, test_semantic)
                    whitebox_v4_metrics = base_run._prepared_metrics(test_whitebox_v4, whitebox_v4_pred)
                    prior_to_rf_gain_rows.extend(
                        _prior_to_rf_gain_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prior_frame=prior_for_gain_frame,
                            prior_pred=prior_for_gain_pred,
                            rf_frame=test_whitebox_v4,
                            rf_pred=whitebox_v4_pred,
                        )
                    )
                    invariance_rows.extend(
                        _whitebox_label_invariance_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_v4_bundle,
                            frame=test_semantic,
                            reference_pred=whitebox_v4_pred,
                            predict_fn=predict_whitebox_family_simple_rf,
                        )
                    )
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="whitebox_family_prior_plus_embedding_mem_rf_v4",
                            metrics=whitebox_v4_metrics,
                            analytical_metrics=analytical_metrics,
                            train_backend=str(whitebox_v4_bundle.get("train_backend", "")),
                            train_env=str(whitebox_v4_bundle.get("train_env", "")),
                            ml_input_source=str(whitebox_v4_bundle.get("ml_input_source", "")),
                            ml_raw_sw_features_used=bool(whitebox_v4_bundle.get("ml_raw_sw_features_used", False)),
                            ml_raw_hw_params_used=bool(whitebox_v4_bundle.get("ml_raw_hw_params_used", False)),
                            hw_refit_required=bool(whitebox_v4_bundle.get("hw_refit_required", False)),
                            family_mlp_enabled_count=int(whitebox_v4_bundle.get("family_mlp_enabled_count", 0)),
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v4,
                            pred=whitebox_v4_pred,
                        )
                    )
                    prediction_error_frames.append(
                        _prediction_error_frame(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v4,
                            pred=whitebox_v4_pred,
                        )
                    )
                    if pipeline_test_semantic is not test_semantic:
                        pipeline_whitebox_v4_bundle = (
                            fit_whitebox_family_embedding_mem_rf_bundle(
                                pipeline_fit_semantic,
                                hw_config,
                                policy=policy,
                                op_thread_scale_config=current_op_thread_scale_config,
                            )
                            if pipeline_requires_alt_fit
                            else whitebox_v4_bundle
                        )
                        pipeline_test_whitebox_v4, pipeline_whitebox_v4_pred = predict_whitebox_family_simple_rf(
                            pipeline_whitebox_v4_bundle,
                            pipeline_test_semantic,
                        )
                        pipeline_prediction_error_frames.append(
                            _prediction_error_frame(
                                policy=policy,
                                dataset=dataset_name,
                                route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                                scale=int(scale),
                                prepared_frame=pipeline_test_whitebox_v4,
                                pred=pipeline_whitebox_v4_pred,
                            )
                        )
                    analytical_interval_rows.extend(
                        _analytical_interval_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v4,
                        )
                    )
                    analytical_trend_rows.extend(
                        _analytical_trend_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v4,
                        )
                    )
                    analytical_bottleneck_rows.extend(
                        _analytical_bottleneck_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v4,
                        )
                    )
                    analytical_state_rows.extend(
                        _analytical_state_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v4,
                        )
                    )
                    whitebox_rows.extend(
                        _whitebox_source_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v4,
                            pred=whitebox_v4_pred,
                        )
                    )
                    perf_feature_rows.extend(
                        _whitebox_perf_feature_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v4,
                        )
                    )
                    mlp_enablement_rows.extend(
                        _family_mlp_enablement_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_v4_bundle,
                        )
                    )
                    hw_transfer_rows.extend(
                        _hw_transfer_audit_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_v4_bundle,
                            frame=test_semantic,
                            reference_pred=whitebox_v4_pred,
                            alternate_hw_config=alternate_hw_config,
                        )
                    )
                    module_sensitivity_rows.extend(
                        _module_sensitivity_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_v4_bundle,
                            frame=test_semantic,
                            reference_prepared=test_whitebox_v4,
                            reference_pred=whitebox_v4_pred,
                            hw_config=hw_config,
                            predict_fn=predict_whitebox_family_simple_rf,
                        )
                    )
                    torch_training_summary_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                "train_backend": str(whitebox_v4_bundle.get("train_backend", "")),
                                "train_env": str(whitebox_v4_bundle.get("train_env", "")),
                                "train_device": str(whitebox_v4_bundle.get("train_device", "")),
                                "npu_device_id": int(whitebox_v4_bundle.get("npu_device_id", 0)),
                                "family_mlp_enabled_count": int(whitebox_v4_bundle.get("family_mlp_enabled_count", 0)),
                                "family_training_summaries": whitebox_v4_bundle.get("family_training_summaries", []),
                            }
                        ]
                    )
                if WHITEBOX_GATHER_SLICE_RF_ROUTE in routes:
                    _progress("fit_whitebox_gather_slice_rf_v5", policy=policy, dataset=dataset_name, scale=scale)
                    whitebox_v5_bundle = fit_whitebox_family_gather_slice_rf_bundle(
                        fit_semantic,
                        hw_config,
                        policy=policy,
                        op_thread_scale_config=current_op_thread_scale_config,
                    )
                    rf_candidate_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_GATHER_SLICE_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v5_bundle.get("rf_candidate_feature_rows", [])
                        ]
                    )
                    rf_pruned_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_GATHER_SLICE_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v5_bundle.get("rf_pruned_feature_rows", [])
                        ]
                    )
                    rf_group_feature_count_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_GATHER_SLICE_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v5_bundle.get("rf_group_feature_count_rows", [])
                        ]
                    )
                    prior_for_gain_frame, prior_for_gain_pred = predict_whitebox_family_prior(
                        whitebox_v5_bundle["prior_bundle"], test_semantic
                    )
                    test_whitebox_v5, whitebox_v5_pred = predict_whitebox_family_simple_rf(whitebox_v5_bundle, test_semantic)
                    whitebox_v5_metrics = base_run._prepared_metrics(test_whitebox_v5, whitebox_v5_pred)
                    prior_to_rf_gain_rows.extend(
                        _prior_to_rf_gain_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prior_frame=prior_for_gain_frame,
                            prior_pred=prior_for_gain_pred,
                            rf_frame=test_whitebox_v5,
                            rf_pred=whitebox_v5_pred,
                        )
                    )
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="whitebox_family_prior_plus_gather_slice_rf_v5",
                            metrics=whitebox_v5_metrics,
                            analytical_metrics=analytical_metrics,
                            train_backend=str(whitebox_v5_bundle.get("train_backend", "")),
                            train_env=str(whitebox_v5_bundle.get("train_env", "")),
                            ml_input_source=str(whitebox_v5_bundle.get("ml_input_source", "")),
                            ml_raw_sw_features_used=bool(whitebox_v5_bundle.get("ml_raw_sw_features_used", False)),
                            ml_raw_hw_params_used=bool(whitebox_v5_bundle.get("ml_raw_hw_params_used", False)),
                            hw_refit_required=bool(whitebox_v5_bundle.get("hw_refit_required", False)),
                            family_mlp_enabled_count=int(whitebox_v5_bundle.get("family_mlp_enabled_count", 0)),
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5,
                            pred=whitebox_v5_pred,
                        )
                    )
                    prediction_error_frames.append(
                        _prediction_error_frame(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5,
                            pred=whitebox_v5_pred,
                        )
                    )
                    if pipeline_test_semantic is not test_semantic:
                        pipeline_whitebox_v5_bundle = (
                            fit_whitebox_family_gather_slice_rf_bundle(
                                pipeline_fit_semantic,
                                hw_config,
                                policy=policy,
                                op_thread_scale_config=current_op_thread_scale_config,
                            )
                            if pipeline_requires_alt_fit
                            else whitebox_v5_bundle
                        )
                        pipeline_test_whitebox_v5, pipeline_whitebox_v5_pred = predict_whitebox_family_simple_rf(
                            pipeline_whitebox_v5_bundle,
                            pipeline_test_semantic,
                        )
                        pipeline_prediction_error_frames.append(
                            _prediction_error_frame(
                                policy=policy,
                                dataset=dataset_name,
                                route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                                scale=int(scale),
                                prepared_frame=pipeline_test_whitebox_v5,
                                pred=pipeline_whitebox_v5_pred,
                            )
                        )
                    analytical_interval_rows.extend(
                        _analytical_interval_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5,
                        )
                    )
                    analytical_trend_rows.extend(
                        _analytical_trend_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5,
                        )
                    )
                    analytical_bottleneck_rows.extend(
                        _analytical_bottleneck_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5,
                        )
                    )
                    analytical_state_rows.extend(
                        _analytical_state_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5,
                        )
                    )
                    whitebox_rows.extend(
                        _whitebox_source_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5,
                            pred=whitebox_v5_pred,
                        )
                    )
                    perf_feature_rows.extend(
                        _whitebox_perf_feature_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5,
                        )
                    )
                    mlp_enablement_rows.extend(
                        _family_mlp_enablement_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_v5_bundle,
                        )
                    )
                    torch_training_summary_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_GATHER_SLICE_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                "train_backend": str(whitebox_v5_bundle.get("train_backend", "")),
                                "train_env": str(whitebox_v5_bundle.get("train_env", "")),
                                "train_device": str(whitebox_v5_bundle.get("train_device", "")),
                                "npu_device_id": int(whitebox_v5_bundle.get("npu_device_id", 0)),
                                "family_mlp_enabled_count": int(whitebox_v5_bundle.get("family_mlp_enabled_count", 0)),
                                "family_training_summaries": whitebox_v5_bundle.get("family_training_summaries", []),
                            }
                        ]
                    )
                if WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE in routes:
                    _progress("fit_whitebox_gather_slice_rf_v5_compact", policy=policy, dataset=dataset_name, scale=scale)
                    whitebox_v5_compact_bundle = fit_whitebox_family_gather_slice_compact_rf_bundle(
                        fit_semantic,
                        hw_config,
                        policy=policy,
                        op_thread_scale_config=current_op_thread_scale_config,
                    )
                    rf_candidate_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v5_compact_bundle.get("rf_candidate_feature_rows", [])
                        ]
                    )
                    rf_pruned_feature_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v5_compact_bundle.get("rf_pruned_feature_rows", [])
                        ]
                    )
                    rf_group_feature_count_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                **row,
                            }
                            for row in whitebox_v5_compact_bundle.get("rf_group_feature_count_rows", [])
                        ]
                    )
                    compact_prior_frame, compact_prior_pred = predict_whitebox_family_prior(
                        whitebox_v5_compact_bundle["prior_bundle"], test_semantic
                    )
                    test_whitebox_v5_compact, whitebox_v5_compact_pred = predict_whitebox_family_simple_rf(
                        whitebox_v5_compact_bundle,
                        test_semantic,
                    )
                    whitebox_v5_compact_metrics = base_run._prepared_metrics(
                        test_whitebox_v5_compact,
                        whitebox_v5_compact_pred,
                    )
                    prior_to_rf_gain_rows.extend(
                        _prior_to_rf_gain_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prior_frame=compact_prior_frame,
                            prior_pred=compact_prior_pred,
                            rf_frame=test_whitebox_v5_compact,
                            rf_pred=whitebox_v5_compact_pred,
                        )
                    )
                    rows.append(
                        _summary_row(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            fit_rows=len(sampled),
                            selected_family="whitebox_family_prior_plus_gather_slice_rf_v5_compact_probe",
                            metrics=whitebox_v5_compact_metrics,
                            analytical_metrics=analytical_metrics,
                            train_backend=str(whitebox_v5_compact_bundle.get("train_backend", "")),
                            train_env=str(whitebox_v5_compact_bundle.get("train_env", "")),
                            ml_input_source=str(whitebox_v5_compact_bundle.get("ml_input_source", "")),
                            ml_raw_sw_features_used=bool(whitebox_v5_compact_bundle.get("ml_raw_sw_features_used", False)),
                            ml_raw_hw_params_used=bool(whitebox_v5_compact_bundle.get("ml_raw_hw_params_used", False)),
                            hw_refit_required=bool(whitebox_v5_compact_bundle.get("hw_refit_required", False)),
                            family_mlp_enabled_count=int(whitebox_v5_compact_bundle.get("family_mlp_enabled_count", 0)),
                            source_input_dataset_root=source_input_dataset_root,
                            source_phase_scope=source_phase_scope,
                        )
                    )
                    breakdown_rows.extend(
                        _family_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5_compact,
                            pred=whitebox_v5_compact_pred,
                        )
                    )
                    prediction_error_frames.append(
                        _prediction_error_frame(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5_compact,
                            pred=whitebox_v5_compact_pred,
                        )
                    )
                    if pipeline_test_semantic is not test_semantic:
                        pipeline_whitebox_v5_compact_bundle = (
                            fit_whitebox_family_gather_slice_compact_rf_bundle(
                                pipeline_fit_semantic,
                                hw_config,
                                policy=policy,
                                op_thread_scale_config=current_op_thread_scale_config,
                            )
                            if pipeline_requires_alt_fit
                            else whitebox_v5_compact_bundle
                        )
                        pipeline_test_whitebox_v5_compact, pipeline_whitebox_v5_compact_pred = predict_whitebox_family_simple_rf(
                            pipeline_whitebox_v5_compact_bundle,
                            pipeline_test_semantic,
                        )
                        pipeline_prediction_error_frames.append(
                            _prediction_error_frame(
                                policy=policy,
                                dataset=dataset_name,
                                route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                                scale=int(scale),
                                prepared_frame=pipeline_test_whitebox_v5_compact,
                                pred=pipeline_whitebox_v5_compact_pred,
                            )
                        )
                    analytical_interval_rows.extend(
                        _analytical_interval_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5_compact,
                        )
                    )
                    analytical_trend_rows.extend(
                        _analytical_trend_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5_compact,
                        )
                    )
                    analytical_bottleneck_rows.extend(
                        _analytical_bottleneck_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5_compact,
                        )
                    )
                    analytical_state_rows.extend(
                        _analytical_state_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5_compact,
                        )
                    )
                    whitebox_rows.extend(
                        _whitebox_source_breakdown_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5_compact,
                            pred=whitebox_v5_compact_pred,
                        )
                    )
                    perf_feature_rows.extend(
                        _whitebox_perf_feature_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox_v5_compact,
                        )
                    )
                    mlp_enablement_rows.extend(
                        _family_mlp_enablement_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                            scale=int(scale),
                            bundle=whitebox_v5_compact_bundle,
                        )
                    )
                    torch_training_summary_rows.extend(
                        [
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "route": WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                                "scale": f"{int(scale)}%",
                                "train_backend": str(whitebox_v5_compact_bundle.get("train_backend", "")),
                                "train_env": str(whitebox_v5_compact_bundle.get("train_env", "")),
                                "train_device": str(whitebox_v5_compact_bundle.get("train_device", "")),
                                "npu_device_id": int(whitebox_v5_compact_bundle.get("npu_device_id", 0)),
                                "family_mlp_enabled_count": int(whitebox_v5_compact_bundle.get("family_mlp_enabled_count", 0)),
                                "family_training_summaries": whitebox_v5_compact_bundle.get("family_training_summaries", []),
                            }
                        ]
                    )
                if WHITEBOX_WORKLOAD_BENCHMARK_ROUTE in routes:
                    benchmark_variants = list(WHITEBOX_WORKLOAD_BENCHMARK_VARIANTS)
                    for benchmark_variant in benchmark_variants:
                        benchmark_route = benchmark_variant_route_name(benchmark_variant)
                        benchmark_bundle = fit_whitebox_family_workload_rf_benchmark_bundle(
                            fit_semantic,
                            hw_config,
                            policy=policy,
                            variant=benchmark_variant,
                            op_thread_scale_config=current_op_thread_scale_config,
                        )
                        rf_candidate_feature_rows.extend(
                            [
                                {
                                    "policy": policy,
                                    "dataset": dataset_name,
                                    "route": benchmark_route,
                                    "scale": f"{int(scale)}%",
                                    **row,
                                }
                                for row in benchmark_bundle.get("rf_candidate_feature_rows", [])
                            ]
                        )
                        rf_pruned_feature_rows.extend(
                            [
                                {
                                    "policy": policy,
                                    "dataset": dataset_name,
                                    "route": benchmark_route,
                                    "scale": f"{int(scale)}%",
                                    **row,
                                }
                                for row in benchmark_bundle.get("rf_pruned_feature_rows", [])
                            ]
                        )
                        rf_group_feature_count_rows.extend(
                            [
                                {
                                    "policy": policy,
                                    "dataset": dataset_name,
                                    "route": benchmark_route,
                                    "scale": f"{int(scale)}%",
                                    **row,
                                }
                                for row in benchmark_bundle.get("rf_group_feature_count_rows", [])
                            ]
                        )
                        prior_for_gain_frame, prior_for_gain_pred = predict_whitebox_family_prior(
                            benchmark_bundle["prior_bundle"], test_semantic
                        )
                        test_whitebox_benchmark, whitebox_benchmark_pred = predict_whitebox_family_simple_rf(
                            benchmark_bundle, test_semantic
                        )
                        whitebox_benchmark_metrics = base_run._prepared_metrics(test_whitebox_benchmark, whitebox_benchmark_pred)
                        prior_to_rf_gain_rows.extend(
                            _prior_to_rf_gain_rows(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                prior_frame=prior_for_gain_frame,
                                prior_pred=prior_for_gain_pred,
                                rf_frame=test_whitebox_benchmark,
                                rf_pred=whitebox_benchmark_pred,
                            )
                        )
                        invariance_rows.extend(
                            _whitebox_label_invariance_rows(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                bundle=benchmark_bundle,
                                frame=test_semantic,
                                reference_pred=whitebox_benchmark_pred,
                                predict_fn=predict_whitebox_family_simple_rf,
                            )
                        )
                        rows.append(
                            _summary_row(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                fit_rows=len(sampled),
                                selected_family=f"{WHITEBOX_WORKLOAD_BENCHMARK_ROUTE}[{benchmark_variant}]",
                                metrics=whitebox_benchmark_metrics,
                                analytical_metrics=analytical_metrics,
                                train_backend=str(benchmark_bundle.get("train_backend", "")),
                                train_env=str(benchmark_bundle.get("train_env", "")),
                                ml_input_source=str(benchmark_bundle.get("ml_input_source", "")),
                                ml_raw_sw_features_used=bool(benchmark_bundle.get("ml_raw_sw_features_used", False)),
                                ml_raw_hw_params_used=bool(benchmark_bundle.get("ml_raw_hw_params_used", False)),
                                hw_refit_required=bool(benchmark_bundle.get("hw_refit_required", False)),
                                family_mlp_enabled_count=int(benchmark_bundle.get("family_mlp_enabled_count", 0)),
                                source_input_dataset_root=source_input_dataset_root,
                                source_phase_scope=source_phase_scope,
                            )
                        )
                        prediction_error_frames.append(
                            _prediction_error_frame(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                prepared_frame=test_whitebox_benchmark,
                                pred=whitebox_benchmark_pred,
                            )
                        )
                        if pipeline_test_semantic is not test_semantic:
                            pipeline_benchmark_bundle = (
                                fit_whitebox_family_workload_rf_benchmark_bundle(
                                    pipeline_fit_semantic,
                                    hw_config,
                                    policy=policy,
                                    variant=benchmark_variant,
                                    op_thread_scale_config=current_op_thread_scale_config,
                                )
                                if pipeline_requires_alt_fit
                                else benchmark_bundle
                            )
                            pipeline_test_whitebox_benchmark, pipeline_whitebox_benchmark_pred = predict_whitebox_family_simple_rf(
                                pipeline_benchmark_bundle,
                                pipeline_test_semantic,
                            )
                            pipeline_prediction_error_frames.append(
                                _prediction_error_frame(
                                    policy=policy,
                                    dataset=dataset_name,
                                    route=benchmark_route,
                                    scale=int(scale),
                                    prepared_frame=pipeline_test_whitebox_benchmark,
                                    pred=pipeline_whitebox_benchmark_pred,
                                )
                            )
                        breakdown_rows.extend(
                            _family_breakdown_rows(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                prepared_frame=test_whitebox_benchmark,
                                pred=whitebox_benchmark_pred,
                            )
                        )
                        whitebox_rows.extend(
                            _whitebox_source_breakdown_rows(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                prepared_frame=test_whitebox_benchmark,
                                pred=whitebox_benchmark_pred,
                            )
                        )
                        perf_feature_rows.extend(
                            _whitebox_perf_feature_rows(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                prepared_frame=test_whitebox_benchmark,
                            )
                        )
                        mlp_enablement_rows.extend(
                            _family_mlp_enablement_rows(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                bundle=benchmark_bundle,
                            )
                        )
                        hw_transfer_rows.extend(
                            _hw_transfer_audit_rows(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                bundle=benchmark_bundle,
                                frame=test_semantic,
                                reference_pred=whitebox_benchmark_pred,
                                alternate_hw_config=alternate_hw_config,
                            )
                        )
                        module_sensitivity_rows.extend(
                            _module_sensitivity_rows(
                                policy=policy,
                                dataset=dataset_name,
                                route=benchmark_route,
                                scale=int(scale),
                                bundle=benchmark_bundle,
                                frame=test_semantic,
                                reference_prepared=test_whitebox_benchmark,
                                reference_pred=whitebox_benchmark_pred,
                                hw_config=hw_config,
                                predict_fn=predict_whitebox_family_simple_rf,
                            )
                        )
                        torch_training_summary_rows.extend(
                            [
                                {
                                    "policy": policy,
                                    "dataset": dataset_name,
                                    "route": benchmark_route,
                                    "scale": f"{int(scale)}%",
                                    "train_backend": str(benchmark_bundle.get("train_backend", "")),
                                    "train_env": str(benchmark_bundle.get("train_env", "")),
                                    "train_device": str(benchmark_bundle.get("train_device", "")),
                                    "npu_device_id": int(benchmark_bundle.get("npu_device_id", 0)),
                                    "family_mlp_enabled_count": int(benchmark_bundle.get("family_mlp_enabled_count", 0)),
                                    "family_training_summaries": benchmark_bundle.get("family_training_summaries", []),
                                }
                            ]
                        )
                if WHITEBOX_PRIOR_ONLY_ROUTE in routes:
                    perf_feature_rows.extend(
                        _whitebox_perf_feature_rows(
                            policy=policy,
                            dataset=dataset_name,
                            route=WHITEBOX_PRIOR_ONLY_ROUTE,
                            scale=int(scale),
                            prepared_frame=test_whitebox,
                        )
                    )
        _progress("scale_done", policy=policy, dataset=dataset_name, scale=scale, detail=f"summary_rows={len(rows)}")
    _progress("dataset_eval_done", policy=policy, dataset=dataset_name, detail=f"summary_rows={len(rows)}")
    return (
        pd.DataFrame(rows),
        pd.DataFrame(breakdown_rows),
        pd.DataFrame(whitebox_rows),
        pd.DataFrame(invariance_rows),
        pd.DataFrame(perf_feature_rows),
        pd.DataFrame(mlp_enablement_rows),
        pd.DataFrame(hw_transfer_rows),
        pd.DataFrame(module_sensitivity_rows),
        pd.DataFrame(analytical_interval_rows),
        pd.DataFrame(analytical_trend_rows),
        pd.DataFrame(analytical_bottleneck_rows),
        pd.DataFrame(analytical_state_rows),
        pd.DataFrame(phase_error_rows),
        pd.DataFrame(prior_gap_rows),
        pd.DataFrame(rf_candidate_feature_rows),
        pd.DataFrame(rf_pruned_feature_rows),
        pd.DataFrame(rf_group_feature_count_rows),
        pd.DataFrame(prior_to_rf_gain_rows),
        pd.DataFrame(row_universe_guard_rows),
        pd.DataFrame(graph_feature_mapping_rows),
        pd.concat(prediction_error_frames, ignore_index=True) if prediction_error_frames else pd.DataFrame(),
        pd.concat(pipeline_prediction_error_frames, ignore_index=True) if pipeline_prediction_error_frames else pd.DataFrame(),
        torch_training_summary_rows,
        {
            "split_root": str(dataset_root),
            "split_manifest_path": str(dataset_root / "split_manifest.json"),
            "split_manifest": split_manifest,
            "op_thread_scale_paths": list(dict.fromkeys(op_thread_scale_paths)),
            "model_bundle_manifest_rows": model_bundle_manifest_rows,
        },
    )


def _best_by_policy(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.copy()
    ordered["scale_num"] = ordered["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    ordered = ordered.sort_values(["policy", "dataset", "route", "mape", "p90", "scale_num"]).reset_index(drop=True)
    best = ordered.groupby(["policy", "dataset", "route"], as_index=False).first()
    return best.drop(columns=["scale_num"]).reset_index(drop=True)


def _route_metric_delta_rows(
    singleop_summary: pd.DataFrame,
    e2e_summary: pd.DataFrame,
    *,
    candidate_route: str = WHITEBOX_ORT_KERNEL_RF_ROUTE,
    candidate_label: str = "v3",
    mape_gate_pp: float = 5.0,
    singleop_mape_gate_pp: float | None = None,
) -> pd.DataFrame:
    keys = ["policy", "dataset", "scale"]
    rows: list[dict[str, Any]] = []
    if not singleop_summary.empty:
        single = singleop_summary.loc[
            singleop_summary["route"].astype(str).isin([WHITEBOX_PRIOR_SIMPLE_RF_ROUTE, candidate_route])
        ].copy()
        if not single.empty:
            wide = single.pivot_table(
                index=keys,
                columns="route",
                values=["mape", "p90", "median"],
                aggfunc="first",
            )
            wide.columns = [f"{metric}__{route}" for metric, route in wide.columns]
            wide = wide.reset_index()
            for _, row in wide.iterrows():
                if f"mape__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}" not in row or f"mape__{candidate_route}" not in row:
                    continue
                candidate_mape = float(row.get(f"mape__{candidate_route}", np.nan))
                candidate_p90 = float(row.get(f"p90__{candidate_route}", np.nan))
                rows.append(
                    {
                        "metric_scope": "singleop",
                        "policy": str(row["policy"]),
                        "dataset": str(row["dataset"]),
                        "scale": str(row["scale"]),
                        "v2_mape": float(row.get(f"mape__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}", np.nan)),
                        f"{candidate_label}_mape": candidate_mape,
                        "mape_delta_pp": float(
                            (
                                row.get(f"mape__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}", np.nan)
                                - row.get(f"mape__{candidate_route}", np.nan)
                            )
                            * 100.0
                        ),
                        "v2_p90": float(row.get(f"p90__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}", np.nan)),
                        f"{candidate_label}_p90": candidate_p90,
                        "p90_delta_pp": float(
                            (
                                row.get(f"p90__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}", np.nan)
                                - row.get(f"p90__{candidate_route}", np.nan)
                            )
                            * 100.0
                        ),
                        "base_route": WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                        "candidate_route": candidate_route,
                    }
                )
    if not e2e_summary.empty:
        e2e = e2e_summary.loc[
            e2e_summary["route"].astype(str).isin([WHITEBOX_PRIOR_SIMPLE_RF_ROUTE, candidate_route])
        ].copy()
        if not e2e.empty:
            wide = e2e.pivot_table(
                index=keys,
                columns="route",
                values=["e2e_mape", "e2e_p90", "e2e_median"],
                aggfunc="first",
            )
            wide.columns = [f"{metric}__{route}" for metric, route in wide.columns]
            wide = wide.reset_index()
            for _, row in wide.iterrows():
                if f"e2e_mape__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}" not in row or f"e2e_mape__{candidate_route}" not in row:
                    continue
                candidate_mape = float(row.get(f"e2e_mape__{candidate_route}", np.nan))
                candidate_p90 = float(row.get(f"e2e_p90__{candidate_route}", np.nan))
                rows.append(
                    {
                        "metric_scope": "e2e_pipeline",
                        "policy": str(row["policy"]),
                        "dataset": str(row["dataset"]),
                        "scale": str(row["scale"]),
                        "v2_mape": float(row.get(f"e2e_mape__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}", np.nan)),
                        f"{candidate_label}_mape": candidate_mape,
                        "mape_delta_pp": float(
                            (
                                row.get(f"e2e_mape__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}", np.nan)
                                - row.get(f"e2e_mape__{candidate_route}", np.nan)
                            )
                            * 100.0
                        ),
                        "v2_p90": float(row.get(f"e2e_p90__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}", np.nan)),
                        f"{candidate_label}_p90": candidate_p90,
                        "p90_delta_pp": float(
                            (
                                row.get(f"e2e_p90__{WHITEBOX_PRIOR_SIMPLE_RF_ROUTE}", np.nan)
                                - row.get(f"e2e_p90__{candidate_route}", np.nan)
                            )
                            * 100.0
                        ),
                        "base_route": WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                        "candidate_route": candidate_route,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        singleop_gate = float(mape_gate_pp if singleop_mape_gate_pp is None else singleop_mape_gate_pp)
        gates = np.where(result["metric_scope"].astype(str).eq("singleop"), singleop_gate, float(mape_gate_pp))
        result["acceptance_status"] = np.where(
            (pd.to_numeric(result["mape_delta_pp"], errors="coerce").fillna(-np.inf) >= gates)
            & (pd.to_numeric(result["p90_delta_pp"], errors="coerce").fillna(-np.inf) >= -2.0),
            "pass_gate",
            "reject_keep_v2_default",
        )
    return result


def _operator_oracle_prediction_rows(
    prediction_rows: pd.DataFrame,
    *,
    base_routes: list[str],
) -> pd.DataFrame:
    required = {"route", "op_type", "actual_us", "pred_us"}
    if prediction_rows.empty or not required.issubset(set(prediction_rows.columns)):
        return pd.DataFrame()
    base = prediction_rows.loc[prediction_rows["route"].astype(str).isin(base_routes)].copy()
    if base.empty:
        return pd.DataFrame()
    oracle_specs = [
        ("Gather", {"Gather"}),
        ("ReduceSum", {"ReduceSum"}),
        ("Gather_ReduceSum", {"Gather", "ReduceSum"}),
    ]
    frames: list[pd.DataFrame] = []
    for oracle_name, op_types in oracle_specs:
        work = base.copy()
        mask = work["op_type"].astype(str).isin(op_types)
        if not mask.any():
            continue
        work.loc[mask, "pred_us"] = pd.to_numeric(work.loc[mask, "actual_us"], errors="coerce").fillna(0.0)
        actual = pd.to_numeric(work["actual_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        work["ape"] = np.abs(pred - actual) / np.clip(actual, a_min=1e-6, a_max=None)
        work["oracle_scope"] = oracle_name
        work["base_route"] = work["route"].astype(str)
        work["route"] = work["base_route"] + "__oracle_" + oracle_name
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _dominant_component_for_op(perf_feature: pd.DataFrame, *, route: str, op_type: str, scale: str) -> str:
    if perf_feature.empty:
        return ""
    subset = perf_feature.loc[
        perf_feature["route"].astype(str).eq(str(route))
        & perf_feature["op_type"].astype(str).eq(str(op_type))
        & perf_feature["scale"].astype(str).eq(str(scale))
    ].copy()
    if subset.empty:
        return ""
    component_columns = [
        "svc_descriptor_us_median",
        "svc_stream_mem_us_median",
        "svc_random_mem_us_median",
        "svc_compute_us_median",
        "svc_sync_us_median",
        "svc_reorder_us_median",
        "svc_dispatch_us_median",
    ]
    values: dict[str, float] = {}
    for column in component_columns:
        if column in subset.columns:
            values[column] = float(pd.to_numeric(subset[column], errors="coerce").fillna(0.0).median())
    if not values:
        return ""
    return max(values.items(), key=lambda item: item[1])[0].replace("svc_", "").replace("_us_median", "")


def _root_cause_oracle_rows(
    current_e2e_summary: pd.DataFrame,
    oracle_e2e_summary: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    perf_feature: pd.DataFrame,
) -> pd.DataFrame:
    if current_e2e_summary.empty or oracle_e2e_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, current in current_e2e_summary.iterrows():
        base_route = str(current.get("route", ""))
        if base_route not in {
            WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
            WHITEBOX_ORT_KERNEL_RF_ROUTE,
            WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
            WHITEBOX_GATHER_SLICE_RF_ROUTE,
        }:
            continue
        policy = str(current.get("policy", ""))
        dataset = str(current.get("dataset", ""))
        scale = str(current.get("scale", ""))
        current_mape = float(current.get("e2e_mape", np.nan))
        for op_type, oracle_scope in (("Gather", "Gather"), ("ReduceSum", "ReduceSum"), ("Gather+ReduceSum", "Gather_ReduceSum")):
            oracle_route = f"{base_route}__oracle_{oracle_scope}"
            oracle = oracle_e2e_summary.loc[
                oracle_e2e_summary["policy"].astype(str).eq(policy)
                & oracle_e2e_summary["dataset"].astype(str).eq(dataset)
                & oracle_e2e_summary["scale"].astype(str).eq(scale)
                & oracle_e2e_summary["route"].astype(str).eq(oracle_route)
            ]
            if oracle.empty:
                continue
            oracle_mape = float(oracle.iloc[0].get("e2e_mape", np.nan))
            delta_pp = float((current_mape - oracle_mape) * 100.0)
            op_key = "Gather" if op_type.startswith("Gather") else "ReduceSum"
            op_subset = prediction_rows.loc[
                prediction_rows["route"].astype(str).eq(base_route)
                & prediction_rows["op_type"].astype(str).eq(op_key)
                & prediction_rows["scale"].astype(str).eq(scale)
            ]
            current_singleop_mape = (
                float(pd.to_numeric(op_subset["ape"], errors="coerce").fillna(0.0).mean()) if not op_subset.empty else np.nan
            )
            rows.append(
                {
                    "op_type": op_type,
                    "policy": policy,
                    "dataset": dataset,
                    "route": base_route,
                    "scale": scale,
                    "current_mape": current_mape,
                    "oracle_mape": oracle_mape,
                    "delta_pp": delta_pp,
                    "current_singleop_mape": current_singleop_mape,
                    "dominant_component": _dominant_component_for_op(perf_feature, route=base_route, op_type=op_key, scale=scale),
                    "root_cause_label": "primary_e2e_driver" if delta_pp >= 5.0 else "not_primary_e2e_driver",
                }
            )
    return pd.DataFrame(rows)


def _merge_e2e_route_metadata(e2e_summary: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if e2e_summary.empty:
        return e2e_summary.copy()
    route_meta_columns = [
        "policy",
        "dataset",
        "route",
        "scale",
        "stage_used",
        "route_whitebox_mode",
        "fit_data_dependent",
        "train_backend",
        "train_env",
        "ml_input_source",
        "ml_raw_sw_features_used",
        "ml_raw_hw_params_used",
        "hw_refit_required",
        "family_mlp_enabled_count",
    ]
    available = [column for column in route_meta_columns if column in summary.columns]
    if not {"policy", "dataset", "route", "scale"}.issubset(set(available)):
        return e2e_summary.copy()
    meta = summary[available].drop_duplicates(subset=["policy", "dataset", "route", "scale"])
    merged = e2e_summary.merge(meta, on=["policy", "dataset", "route", "scale"], how="left")
    ordered_columns = [
        "policy",
        "dataset",
        "route",
        "stage_used",
        "route_whitebox_mode",
        "fit_data_dependent",
        "train_backend",
        "train_env",
        "ml_input_source",
        "ml_raw_sw_features_used",
        "ml_raw_hw_params_used",
        "hw_refit_required",
        "family_mlp_enabled_count",
        "scale",
        "combo_count_full",
        "combo_count_partial",
        "coverage_ratio_mean",
        "e2e_mae_us",
        "e2e_mape",
        "e2e_p90",
        "e2e_median",
    ]
    tail_columns = [column for column in merged.columns if column not in ordered_columns]
    return merged[[*ordered_columns, *tail_columns]]


def _singleop_optype_threshold_audit(prediction_errors: pd.DataFrame, *, threshold: float = 0.15) -> pd.DataFrame:
    required = {"policy", "dataset", "route", "scale", "op_type", "actual_us", "pred_us", "ape"}
    if prediction_errors.empty or not required.issubset(set(prediction_errors.columns)):
        return pd.DataFrame()
    work = prediction_errors.copy()
    work["actual_us"] = pd.to_numeric(work["actual_us"], errors="coerce").fillna(0.0)
    work["pred_us"] = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0)
    work["ape"] = pd.to_numeric(work["ape"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(["policy", "dataset", "route", "scale", "op_type"], dropna=False):
        actual = group["actual_us"].to_numpy(dtype=float)
        pred = group["pred_us"].to_numpy(dtype=float)
        ape = np.abs(pred - actual) / np.clip(actual, a_min=1e-6, a_max=None)
        mape = float(np.mean(ape)) if len(ape) else 0.0
        p90 = float(np.quantile(ape, 0.90)) if len(ape) else 0.0
        rows.append(
            {
                "policy": str(key[0]),
                "dataset": str(key[1]),
                "route": str(key[2]),
                "scale": str(key[3]),
                "op_type": str(key[4]),
                "row_count": int(len(group)),
                "actual_us_sum": float(actual.sum()),
                "pred_us_sum": float(pred.sum()),
                "mape": mape,
                "p90": p90,
                "threshold_mape": float(threshold),
                "pass_mape_threshold": bool(mape <= float(threshold)),
            }
        )
    return pd.DataFrame(rows)


def _analytical_quality_scorecard(
    analytical_interval: pd.DataFrame,
    analytical_bottleneck: pd.DataFrame,
    module_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(base: dict[str, Any], metric_name: str, value: float, target: float) -> None:
        rows.append(
            {
                **base,
                "metric_name": metric_name,
                "value": float(value),
                "target": float(target),
                "delta_to_target": float(value - target),
                "pass_target": bool(value >= target),
            }
        )

    if not analytical_interval.empty:
        interval = analytical_interval.loc[analytical_interval["scope"].astype(str).eq("overall")].copy()
        for _, row in interval.iterrows():
            base = {
                "policy": str(row.get("policy", "")),
                "dataset": str(row.get("dataset", "")),
                "route": str(row.get("route", "")),
                "scale": str(row.get("scale", "")),
            }
            add_row(base, "coverage", float(row.get("coverage_rate", np.nan)), 0.80)
            add_row(base, "policy_spearman", float(row.get("center_spearman", np.nan)), 0.75)

    if not module_sensitivity.empty:
        perturb = module_sensitivity.copy()
        if "applicable" in perturb.columns:
            perturb = perturb.loc[perturb["applicable"].astype(bool)].copy()
        if not perturb.empty:
            perturb["quality_pass"] = (
                perturb.get("prediction_succeeded", pd.Series(False, index=perturb.index)).astype(bool)
                & perturb.get("expectation_pass", pd.Series(False, index=perturb.index)).astype(bool)
                & perturb.get("monotonic_pass", pd.Series(False, index=perturb.index)).astype(bool)
            )
            for key, group in perturb.groupby(["policy", "dataset", "route", "scale"], dropna=False):
                base = {
                    "policy": str(key[0]),
                    "dataset": str(key[1]),
                    "route": str(key[2]),
                    "scale": str(key[3]),
                }
                add_row(base, "hw_perturb_pass_rate", float(group["quality_pass"].mean()), 0.95)

    if not analytical_bottleneck.empty:
        for _, row in analytical_bottleneck.iterrows():
            base = {
                "policy": str(row.get("policy", "")),
                "dataset": str(row.get("dataset", "")),
                "route": str(row.get("route", "")),
                "scale": str(row.get("scale", "")),
            }
            add_row(base, "stable_bottleneck_rate", float(row.get("stable_bottleneck_rate", np.nan)), 0.70)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["stage_used"] = result["route"].astype(str).map(route_stage)
    return result


def _e2e_operator_signed_error_by_combo(prediction_errors: pd.DataFrame) -> pd.DataFrame:
    required = {"policy", "dataset", "route", "scale", "case_id", "source_name", "combo", "op_type", "actual_us", "pred_us"}
    if prediction_errors.empty or not required.issubset(set(prediction_errors.columns)):
        return pd.DataFrame()
    work = prediction_errors.copy()
    work["actual_us"] = pd.to_numeric(work["actual_us"], errors="coerce").fillna(0.0)
    work["pred_us"] = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0)
    group_cols = ["policy", "dataset", "route", "scale", "case_id", "source_name", "combo", "op_type"]
    grouped = (
        work.groupby(group_cols, dropna=False)
        .agg(
            row_count=("op_type", "size"),
            actual_us_sum=("actual_us", "sum"),
            pred_us_sum=("pred_us", "sum"),
            mean_ape=("ape", "mean"),
        )
        .reset_index()
    )
    grouped["signed_error_us"] = grouped["pred_us_sum"] - grouped["actual_us_sum"]
    grouped["abs_signed_error_us"] = grouped["signed_error_us"].abs()
    combo_cols = ["policy", "dataset", "route", "scale", "case_id", "source_name", "combo"]
    totals = (
        grouped.groupby(combo_cols, dropna=False)
        .agg(combo_actual_us_sum=("actual_us_sum", "sum"), combo_abs_signed_error_us_sum=("abs_signed_error_us", "sum"))
        .reset_index()
    )
    grouped = grouped.merge(totals, on=combo_cols, how="left")
    grouped["signed_error_share_of_combo_actual"] = grouped["signed_error_us"] / grouped["combo_actual_us_sum"].clip(lower=1e-6)
    grouped["abs_signed_error_share"] = grouped["abs_signed_error_us"] / grouped["combo_abs_signed_error_us_sum"].clip(lower=1e-6)
    return grouped


def _e2e_tail_operator_attribution(operator_signed: pd.DataFrame, e2e_combo: pd.DataFrame) -> pd.DataFrame:
    required_signed = {"policy", "dataset", "route", "scale", "case_id", "source_name", "combo", "op_type", "signed_error_us"}
    required_combo = {"policy", "dataset", "route", "scale", "case_id", "source_name", "combo", "ape"}
    if operator_signed.empty or e2e_combo.empty:
        return pd.DataFrame()
    if not required_signed.issubset(set(operator_signed.columns)) or not required_combo.issubset(set(e2e_combo.columns)):
        return pd.DataFrame()
    combo = e2e_combo.copy()
    combo["ape"] = pd.to_numeric(combo["ape"], errors="coerce").fillna(0.0)
    thresholds = (
        combo.groupby(["policy", "dataset", "route", "scale"], dropna=False)["ape"]
        .quantile(0.90)
        .reset_index()
        .rename(columns={"ape": "tail_threshold_ape"})
    )
    tail = combo.merge(thresholds, on=["policy", "dataset", "route", "scale"], how="left")
    tail = tail.loc[tail["ape"] >= tail["tail_threshold_ape"]].copy()
    if tail.empty:
        return pd.DataFrame()
    join_cols = ["policy", "dataset", "route", "scale", "case_id", "source_name", "combo"]
    merged = operator_signed.merge(
        tail[[*join_cols, "ape", "tail_threshold_ape", "predicted_e2e_us", "actual_e2e_us"]],
        on=join_cols,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    merged["abs_signed_error_us"] = pd.to_numeric(merged["signed_error_us"], errors="coerce").fillna(0.0).abs()
    rows = (
        merged.groupby(["policy", "dataset", "route", "scale", "op_type"], dropna=False)
        .agg(
            tail_combo_count=("combo", "nunique"),
            tail_threshold_ape=("tail_threshold_ape", "max"),
            tail_combo_ape_mean=("ape", "mean"),
            signed_error_us_sum=("signed_error_us", "sum"),
            abs_signed_error_us_sum=("abs_signed_error_us", "sum"),
            actual_us_sum=("actual_us_sum", "sum"),
            pred_us_sum=("pred_us_sum", "sum"),
        )
        .reset_index()
    )
    denom = rows.groupby(["policy", "dataset", "route", "scale"], dropna=False)["abs_signed_error_us_sum"].transform("sum")
    rows["tail_abs_error_share"] = rows["abs_signed_error_us_sum"] / denom.clip(lower=1e-6)
    return rows


def _route_pair_metric_delta_rows(
    singleop_summary: pd.DataFrame,
    e2e_summary: pd.DataFrame,
    *,
    base_route: str,
    candidate_route: str,
    base_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    keys = ["policy", "dataset", "scale"]
    rows: list[dict[str, Any]] = []
    scopes = [
        ("singleop", singleop_summary, "mape", "p90", "median"),
        ("e2e_pipeline", e2e_summary, "e2e_mape", "e2e_p90", "e2e_median"),
    ]
    for scope_name, frame, mape_col, p90_col, median_col in scopes:
        if frame.empty:
            continue
        subset = frame.loc[frame["route"].astype(str).isin([base_route, candidate_route])].copy()
        if subset.empty:
            continue
        wide = subset.pivot_table(index=keys, columns="route", values=[mape_col, p90_col, median_col], aggfunc="first")
        wide.columns = [f"{metric}__{route}" for metric, route in wide.columns]
        wide = wide.reset_index()
        for _, item in wide.iterrows():
            base_mape_key = f"{mape_col}__{base_route}"
            candidate_mape_key = f"{mape_col}__{candidate_route}"
            if base_mape_key not in item or candidate_mape_key not in item:
                continue
            base_mape = float(item.get(base_mape_key, np.nan))
            candidate_mape = float(item.get(candidate_mape_key, np.nan))
            base_p90 = float(item.get(f"{p90_col}__{base_route}", np.nan))
            candidate_p90 = float(item.get(f"{p90_col}__{candidate_route}", np.nan))
            base_median = float(item.get(f"{median_col}__{base_route}", np.nan))
            candidate_median = float(item.get(f"{median_col}__{candidate_route}", np.nan))
            rows.append(
                {
                    "metric_scope": scope_name,
                    "policy": str(item["policy"]),
                    "dataset": str(item["dataset"]),
                    "scale": str(item["scale"]),
                    "base_route": str(base_route),
                    "candidate_route": str(candidate_route),
                    f"{base_label}_mape": base_mape,
                    f"{candidate_label}_mape": candidate_mape,
                    "mape_delta_pp": float((base_mape - candidate_mape) * 100.0),
                    f"{base_label}_p90": base_p90,
                    f"{candidate_label}_p90": candidate_p90,
                    "p90_delta_pp": float((base_p90 - candidate_p90) * 100.0),
                    f"{base_label}_median": base_median,
                    f"{candidate_label}_median": candidate_median,
                    "median_delta_pp": float((base_median - candidate_median) * 100.0),
                }
            )
    return pd.DataFrame(rows)


def _route_pair_operator_delta_rows(
    operator_signed: pd.DataFrame,
    tail_attribution: pd.DataFrame,
    *,
    base_route: str,
    candidate_route: str,
    base_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        ("operator_signed_all", operator_signed),
        ("operator_signed_tail", tail_attribution),
    ]
    for scope_name, source in sources:
        if source.empty:
            continue
        required = {"policy", "dataset", "route", "scale", "op_type", "signed_error_us_sum"}
        work = source.copy()
        if "signed_error_us_sum" not in work.columns and "signed_error_us" in work.columns:
            work["signed_error_us_sum"] = pd.to_numeric(work["signed_error_us"], errors="coerce").fillna(0.0)
        if not required.issubset(set(work.columns)):
            continue
        if "actual_us_sum" not in work.columns:
            work["actual_us_sum"] = np.nan
        if "pred_us_sum" not in work.columns:
            work["pred_us_sum"] = np.nan
        subset = work.loc[
            work["route"].astype(str).isin([base_route, candidate_route])
            & work["op_type"].astype(str).isin(["Gather", "ReduceSum"])
        ].copy()
        if subset.empty:
            continue
        grouped = (
            subset.groupby(["policy", "dataset", "route", "scale", "op_type"], dropna=False)
            .agg(
                signed_error_us_sum=("signed_error_us_sum", "sum"),
                actual_us_sum=("actual_us_sum", "sum"),
                pred_us_sum=("pred_us_sum", "sum"),
            )
            .reset_index()
        )
        wide = grouped.pivot_table(
            index=["policy", "dataset", "scale", "op_type"],
            columns="route",
            values=["signed_error_us_sum", "actual_us_sum", "pred_us_sum"],
            aggfunc="first",
        )
        wide.columns = [f"{metric}__{route}" for metric, route in wide.columns]
        wide = wide.reset_index()
        for _, item in wide.iterrows():
            base_signed_key = f"signed_error_us_sum__{base_route}"
            candidate_signed_key = f"signed_error_us_sum__{candidate_route}"
            if base_signed_key not in item or candidate_signed_key not in item:
                continue
            base_signed = float(item.get(base_signed_key, np.nan))
            candidate_signed = float(item.get(candidate_signed_key, np.nan))
            base_under = abs(min(base_signed, 0.0))
            candidate_under = abs(min(candidate_signed, 0.0))
            rows.append(
                {
                    "metric_scope": scope_name,
                    "policy": str(item["policy"]),
                    "dataset": str(item["dataset"]),
                    "scale": str(item["scale"]),
                    "op_type": str(item["op_type"]),
                    "base_route": str(base_route),
                    "candidate_route": str(candidate_route),
                    f"{base_label}_signed_error_us_sum": base_signed,
                    f"{candidate_label}_signed_error_us_sum": candidate_signed,
                    "signed_error_delta_us": float(base_signed - candidate_signed),
                    "underprediction_magnitude_reduction_ratio": float(
                        (base_under - candidate_under) / max(base_under, 1e-6)
                    ),
                    f"{base_label}_actual_us_sum": float(item.get(f"actual_us_sum__{base_route}", np.nan)),
                    f"{candidate_label}_actual_us_sum": float(item.get(f"actual_us_sum__{candidate_route}", np.nan)),
                    f"{base_label}_pred_us_sum": float(item.get(f"pred_us_sum__{base_route}", np.nan)),
                    f"{candidate_label}_pred_us_sum": float(item.get(f"pred_us_sum__{candidate_route}", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _route_metric_lookup(frame: pd.DataFrame, route: str, key: tuple[str, str, str], *, e2e: bool) -> dict[str, float]:
    if frame.empty:
        return {"mape": float("nan"), "p90": float("nan")}
    policy, dataset, scale = key
    subset = frame.loc[
        frame["policy"].astype(str).eq(policy)
        & frame["dataset"].astype(str).eq(dataset)
        & frame["scale"].astype(str).eq(scale)
        & frame["route"].astype(str).eq(route)
    ]
    if subset.empty:
        return {"mape": float("nan"), "p90": float("nan")}
    row = subset.iloc[0]
    mape_col = "e2e_mape" if e2e else "mape"
    p90_col = "e2e_p90" if e2e else "p90"
    return {
        "mape": float(row.get(mape_col, np.nan)),
        "p90": float(row.get(p90_col, np.nan)),
    }


def _indexed_copy_selected_features(
    rf_pruned: pd.DataFrame,
    route: str,
    key: tuple[str, str, str],
) -> list[str]:
    if rf_pruned.empty:
        return []
    policy, dataset, scale = key
    subset = rf_pruned.loc[
        rf_pruned["policy"].astype(str).eq(policy)
        & rf_pruned["dataset"].astype(str).eq(dataset)
        & rf_pruned["scale"].astype(str).eq(scale)
        & rf_pruned["route"].astype(str).eq(route)
        & rf_pruned["rf_group"].astype(str).eq("indexed_block_copy_rf")
    ]
    if subset.empty or "feature_name" not in subset.columns:
        return []
    return subset["feature_name"].astype(str).drop_duplicates().tolist()


def _gather_tail_underprediction(
    tail_attribution: pd.DataFrame,
    route: str,
    key: tuple[str, str, str],
) -> float:
    if tail_attribution.empty:
        return float("nan")
    policy, dataset, scale = key
    subset = tail_attribution.loc[
        tail_attribution["policy"].astype(str).eq(policy)
        & tail_attribution["dataset"].astype(str).eq(dataset)
        & tail_attribution["scale"].astype(str).eq(scale)
        & tail_attribution["route"].astype(str).eq(route)
        & tail_attribution["op_type"].astype(str).eq("Gather")
    ]
    if subset.empty or "signed_error_us_sum" not in subset.columns:
        return float("nan")
    signed = float(pd.to_numeric(subset.iloc[0].get("signed_error_us_sum", np.nan), errors="coerce"))
    return abs(min(signed, 0.0))


def _indexed_copy_feature_ablation_rows(
    singleop_summary: pd.DataFrame,
    e2e_summary: pd.DataFrame,
    rf_pruned: pd.DataFrame,
    tail_attribution: pd.DataFrame,
) -> pd.DataFrame:
    if singleop_summary.empty and e2e_summary.empty:
        return pd.DataFrame()
    base_route = WHITEBOX_GATHER_SLICE_RF_ROUTE
    compact_route = WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE
    key_frames = []
    for frame in (singleop_summary, e2e_summary):
        if not frame.empty and {"policy", "dataset", "scale", "route"}.issubset(set(frame.columns)):
            key_frames.append(
                frame.loc[frame["route"].astype(str).isin([base_route, compact_route]), ["policy", "dataset", "scale"]]
            )
    if not key_frames:
        return pd.DataFrame()
    keys = (
        pd.concat(key_frames, ignore_index=True)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        key_tuple = (str(key[0]), str(key[1]), str(key[2]))
        base_features = _indexed_copy_selected_features(rf_pruned, base_route, key_tuple)
        compact_features = _indexed_copy_selected_features(rf_pruned, compact_route, key_tuple)
        base_under = _gather_tail_underprediction(tail_attribution, base_route, key_tuple)
        compact_under = _gather_tail_underprediction(tail_attribution, compact_route, key_tuple)
        for route, feature_set_name, features in [
            (base_route, "full_v5_45", base_features),
            (compact_route, "compact_probe", compact_features),
        ]:
            if route == compact_route and not features:
                continue
            single = _route_metric_lookup(singleop_summary, route, key_tuple, e2e=False)
            e2e = _route_metric_lookup(e2e_summary, route, key_tuple, e2e=True)
            removed = sorted(set(base_features) - set(features)) if route == compact_route else []
            route_under = compact_under if route == compact_route else base_under
            reduction_ratio = (
                (base_under - route_under) / max(base_under, 1e-6)
                if np.isfinite(base_under) and np.isfinite(route_under)
                else float("nan")
            )
            e2e_mape = float(e2e["mape"])
            rows.append(
                {
                    "policy": key_tuple[0],
                    "dataset": key_tuple[1],
                    "scale": key_tuple[2],
                    "route": route,
                    "feature_set_name": feature_set_name,
                    "removed_features": ",".join(removed),
                    "feature_count": int(len(features)),
                    "singleop_mape": single["mape"],
                    "singleop_p90": single["p90"],
                    "e2e_mape": e2e_mape,
                    "e2e_p90": e2e["p90"],
                    "gather_tail_underprediction_reduction_ratio": float(reduction_ratio),
                    "acceptance_status": "pass" if np.isfinite(e2e_mape) and e2e_mape <= 0.1000 else "fail",
                }
            )
    return pd.DataFrame(rows)


def _e2e_gather_correction_oracle_rows(e2e_combo: pd.DataFrame, operator_signed: pd.DataFrame) -> pd.DataFrame:
    required_combo = {"policy", "dataset", "route", "scale", "case_id", "source_name", "combo", "predicted_e2e_us", "actual_e2e_us"}
    required_signed = {"policy", "dataset", "route", "scale", "case_id", "source_name", "combo", "op_type", "signed_error_us"}
    if e2e_combo.empty or operator_signed.empty:
        return pd.DataFrame()
    if not required_combo.issubset(set(e2e_combo.columns)) or not required_signed.issubset(set(operator_signed.columns)):
        return pd.DataFrame()
    combo = e2e_combo.copy()
    combo["predicted_e2e_us"] = pd.to_numeric(combo["predicted_e2e_us"], errors="coerce").fillna(0.0)
    combo["actual_e2e_us"] = pd.to_numeric(combo["actual_e2e_us"], errors="coerce").fillna(0.0)
    join_cols = ["policy", "dataset", "route", "scale", "case_id", "source_name", "combo"]
    gather_signed = operator_signed.loc[operator_signed["op_type"].astype(str).eq("Gather"), [*join_cols, "signed_error_us"]].copy()
    gather_signed["signed_error_us"] = pd.to_numeric(gather_signed["signed_error_us"], errors="coerce").fillna(0.0)
    merged = combo.merge(gather_signed, on=join_cols, how="left")
    merged["signed_error_us"] = pd.to_numeric(merged["signed_error_us"], errors="coerce").fillna(0.0)
    correction_specs = [("base", np.nan), ("gather_oracle", 1.0), ("underprediction_alpha_0.25", 0.25), ("underprediction_alpha_0.5", 0.5), ("underprediction_alpha_0.75", 0.75), ("underprediction_alpha_1.0", 1.0)]
    rows: list[dict[str, Any]] = []
    for correction_scope, alpha in correction_specs:
        work = merged.copy()
        if correction_scope == "base":
            corrected = work["predicted_e2e_us"].to_numpy(dtype=float)
        elif correction_scope == "gather_oracle":
            corrected = work["predicted_e2e_us"].to_numpy(dtype=float) - work["signed_error_us"].to_numpy(dtype=float)
        else:
            under = np.minimum(work["signed_error_us"].to_numpy(dtype=float), 0.0)
            corrected = work["predicted_e2e_us"].to_numpy(dtype=float) - float(alpha) * under
        actual = work["actual_e2e_us"].to_numpy(dtype=float)
        work["corrected_ape"] = np.abs(corrected - actual) / np.clip(actual, a_min=1e-6, a_max=None)
        grouped = (
            work.groupby(["policy", "dataset", "route", "scale"], dropna=False)
            .agg(
                combo_count=("combo", "size"),
                e2e_mape=("corrected_ape", "mean"),
                e2e_p90=("corrected_ape", lambda values: float(np.quantile(pd.to_numeric(values, errors="coerce").fillna(0.0), 0.90))),
                e2e_median=("corrected_ape", "median"),
                gather_signed_error_us_sum=("signed_error_us", "sum"),
            )
            .reset_index()
        )
        grouped["correction_scope"] = correction_scope
        grouped["alpha"] = alpha
        rows.extend(grouped.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _gather_slice_prior_ablation_rows(prediction_errors: pd.DataFrame) -> pd.DataFrame:
    required = {
        "policy",
        "dataset",
        "route",
        "scale",
        "op_type",
        "actual_us",
        "pred_us",
        "whitebox_center_us",
        "svc_stream_mem_center_us",
        "gather_slice_stream_us",
    }
    if prediction_errors.empty or not required.issubset(set(prediction_errors.columns)):
        return pd.DataFrame()
    base = prediction_errors.loc[prediction_errors["op_type"].astype(str).eq("Gather")].copy()
    if base.empty:
        return pd.DataFrame()
    for column in ["actual_us", "pred_us", "whitebox_center_us", "svc_stream_mem_center_us", "gather_slice_stream_us"]:
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)
    direct_slice = np.clip(base["whitebox_center_us"] - base["svc_stream_mem_center_us"] + base["gather_slice_stream_us"], a_min=0.0, a_max=None)
    schemes = {
        "v4_prior_center": base["whitebox_center_us"].to_numpy(dtype=float),
        "direct_slice_replacement": direct_slice,
        "gated_slice_floor": np.maximum(base["whitebox_center_us"].to_numpy(dtype=float), direct_slice),
        "route_rf_prediction": base["pred_us"].to_numpy(dtype=float),
    }
    frames: list[pd.DataFrame] = []
    actual = base["actual_us"].to_numpy(dtype=float)
    for scheme, pred in schemes.items():
        work = base[["policy", "dataset", "route", "scale", "op_type"]].copy()
        work["scheme"] = scheme
        work["actual_us"] = actual
        work["pred_us"] = np.asarray(pred, dtype=float)
        work["ape"] = np.abs(work["pred_us"].to_numpy(dtype=float) - actual) / np.clip(actual, a_min=1e-6, a_max=None)
        work["signed_error_us"] = work["pred_us"].to_numpy(dtype=float) - actual
        frames.append(work)
    stacked = pd.concat(frames, ignore_index=True)
    return (
        stacked.groupby(["policy", "dataset", "route", "scale", "op_type", "scheme"], dropna=False)
        .agg(
            row_count=("op_type", "size"),
            mape=("ape", "mean"),
            p90=("ape", lambda values: float(np.quantile(pd.to_numeric(values, errors="coerce").fillna(0.0), 0.90))),
            median=("ape", "median"),
            signed_error_us_sum=("signed_error_us", "sum"),
            actual_us_sum=("actual_us", "sum"),
            pred_us_sum=("pred_us", "sum"),
        )
        .reset_index()
    )


def _acceptance_matrix(best: pd.DataFrame) -> pd.DataFrame:
    acceptance = best.copy()
    acceptance["pass_10_20"] = (acceptance["mape"] < 0.10) & (acceptance["p90"] < 0.20)
    acceptance["whitebox_feature_ml_pass_10_20"] = (
        acceptance["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE) & acceptance["pass_10_20"]
    )
    return acceptance[
        [
            "policy",
            "dataset",
            "route",
            "stage_used",
            "route_whitebox_mode",
            "fit_data_dependent",
            "train_backend",
            "train_env",
            "ml_input_source",
            "ml_raw_sw_features_used",
            "ml_raw_hw_params_used",
            "hw_refit_required",
            "family_mlp_enabled_count",
            "scale",
            "mape",
            "p90",
            "median",
            "fit_rows",
            "pass_10_20",
            "whitebox_feature_ml_pass_10_20",
        ]
    ].rename(columns={"scale": "best_scale", "mape": "best_mape", "p90": "best_p90"})


def _comparison_summary_with_source_baseline(combined: pd.DataFrame, selected_policies: list[str]) -> pd.DataFrame:
    frames = [combined.copy()]
    source_policy_set = set(
        combined.loc[combined["dataset"].astype(str).eq("source"), "policy"].astype(str).tolist()
    )
    missing_source_policies = [policy for policy in selected_policies if policy not in source_policy_set]
    if missing_source_policies and SOURCE_OPERATOR_BASELINE_SUMMARY.exists():
        source_baseline = pd.read_csv(SOURCE_OPERATOR_BASELINE_SUMMARY, low_memory=False)
        source_baseline = source_baseline.loc[
            source_baseline["policy"].astype(str).isin(missing_source_policies)
            & source_baseline["dataset"].astype(str).eq("source")
        ].copy()
        if not source_baseline.empty:
            frames.append(source_baseline)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _analytical_vs_rf_error_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (policy, dataset, scale), group in summary.groupby(["policy", "dataset", "scale"], dropna=False, sort=True):
        analytical = group.loc[group["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)].copy()
        rf = group.loc[group["route"].astype(str).eq(WHITEBOX_GATHER_SLICE_RF_ROUTE)].copy()
        if analytical.empty or rf.empty:
            continue
        analytical_row = analytical.iloc[0]
        rf_row = rf.iloc[0]
        row = {
            "policy": str(policy),
            "dataset": str(dataset),
            "scale": str(scale),
            "analytical_route": WHITEBOX_PRIOR_ONLY_ROUTE,
            "rf_route": WHITEBOX_GATHER_SLICE_RF_ROUTE,
            "analytical_mape": float(analytical_row["mape"]),
            "analytical_p90": float(analytical_row["p90"]),
            "analytical_median": float(analytical_row["median"]),
            "rf_mape": float(rf_row["mape"]),
            "rf_p90": float(rf_row["p90"]),
            "rf_median": float(rf_row["median"]),
            "eval_rows": int(rf_row.get("eval_rows", analytical_row.get("eval_rows", 0))),
        }
        row["analytical_minus_rf_mape"] = row["analytical_mape"] - row["rf_mape"]
        row["analytical_minus_rf_p90"] = row["analytical_p90"] - row["rf_p90"]
        row["analytical_minus_rf_median"] = row["analytical_median"] - row["rf_median"]
        rows.append(row)
    return pd.DataFrame(rows)


def _source_vs_target_operator_accuracy_delta(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    route_specs = [
        ("analytical", WHITEBOX_PRIOR_ONLY_ROUTE),
        ("rf", WHITEBOX_GATHER_SLICE_RF_ROUTE),
    ]
    for (policy, scale), group in summary.groupby(["policy", "scale"], dropna=False, sort=True):
        for model_kind, route in route_specs:
            route_group = group.loc[group["route"].astype(str).eq(route)].copy()
            source = route_group.loc[route_group["dataset"].astype(str).eq("source")]
            target = route_group.loc[route_group["dataset"].astype(str).eq("target")]
            if source.empty or target.empty:
                continue
            source_row = source.iloc[0]
            target_row = target.iloc[0]
            rows.append(
                {
                    "policy": str(policy),
                    "scale": str(scale),
                    "model_kind": model_kind,
                    "route": route,
                    "source_mape": float(source_row["mape"]),
                    "target_mape": float(target_row["mape"]),
                    "target_minus_source_mape": float(target_row["mape"]) - float(source_row["mape"]),
                    "source_p90": float(source_row["p90"]),
                    "target_p90": float(target_row["p90"]),
                    "target_minus_source_p90": float(target_row["p90"]) - float(source_row["p90"]),
                    "source_median": float(source_row["median"]),
                    "target_median": float(target_row["median"]),
                    "target_minus_source_median": float(target_row["median"]) - float(source_row["median"]),
                    "source_eval_rows": int(source_row.get("eval_rows", 0)),
                    "target_eval_rows": int(target_row.get("eval_rows", 0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    source_hw = Path(args.source_hw_config).resolve()
    target_hw = Path(args.target_hw_config).resolve()
    scales = [int(value.strip()) for value in str(args.scales).split(",") if value.strip()]
    requested_routes = [value.strip() for value in str(args.routes).split(",") if value.strip()]
    selected_policies = _selected_policies(str(args.policy))
    selected_datasets = _selected_datasets(str(args.datasets))
    enable_e2e_pipeline = not bool(args.skip_e2e_pipeline)
    op_thread_scale_mode = str(args.op_thread_scale_mode)
    if op_thread_scale_mode == "load" and not str(args.op_thread_scale_config).strip():
        raise ValueError("--op-thread-scale-config is required when --op-thread-scale-mode=load")
    loaded_op_thread_scale_config = (
        load_op_thread_scale_config(Path(args.op_thread_scale_config).resolve())
        if op_thread_scale_mode == "load"
        else None
    )
    op_thread_scale_output = Path(args.op_thread_scale_output).resolve() if str(args.op_thread_scale_output).strip() else None
    model_bundle_output_dir = (
        Path(args.model_bundle_output_dir).resolve()
        if str(args.model_bundle_output_dir).strip()
        else artifact_root / "model_bundles"
    )

    all_rows: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
        "source_hw_config": str(source_hw),
        "target_hw_config": str(target_hw),
        "scales": scales,
        "requested_routes": requested_routes,
        "routes": [],
        "policies": {},
        "datasets": selected_datasets,
        "evaluation_scope": "source_only" if selected_datasets == ["source"] else "mixed",
        "e2e_pipeline_enabled": bool(enable_e2e_pipeline),
        "whitebox_prior_activated": False,
        "op_thread_scale": {
            "mode": op_thread_scale_mode,
            "config_path": str(Path(args.op_thread_scale_config).resolve()) if str(args.op_thread_scale_config).strip() else "",
            "output_path": str(op_thread_scale_output) if op_thread_scale_output is not None else "",
            "min_group_rows": int(args.op_thread_scale_min_rows),
        },
        "model_bundle_export": {
            "enabled": bool(args.export_model_bundles),
            "output_dir": str(model_bundle_output_dir),
        },
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    breakdown_frames: list[pd.DataFrame] = []
    whitebox_frames: list[pd.DataFrame] = []
    invariance_frames: list[pd.DataFrame] = []
    perf_feature_frames: list[pd.DataFrame] = []
    family_mlp_frames: list[pd.DataFrame] = []
    hw_transfer_frames: list[pd.DataFrame] = []
    module_sensitivity_frames: list[pd.DataFrame] = []
    analytical_interval_frames: list[pd.DataFrame] = []
    analytical_trend_frames: list[pd.DataFrame] = []
    analytical_bottleneck_frames: list[pd.DataFrame] = []
    analytical_state_frames: list[pd.DataFrame] = []
    phase_error_frames: list[pd.DataFrame] = []
    prior_gap_frames: list[pd.DataFrame] = []
    rf_candidate_feature_frames: list[pd.DataFrame] = []
    rf_pruned_feature_frames: list[pd.DataFrame] = []
    rf_group_feature_count_frames: list[pd.DataFrame] = []
    prior_to_rf_gain_frames: list[pd.DataFrame] = []
    row_universe_guard_frames: list[pd.DataFrame] = []
    graph_feature_mapping_frames: list[pd.DataFrame] = []
    prediction_error_frames: list[pd.DataFrame] = []
    pipeline_prediction_error_frames: list[pd.DataFrame] = []
    pipeline_audit_frames: list[pd.DataFrame] = []
    e2e_pipeline_summary_frames: list[pd.DataFrame] = []
    e2e_pipeline_combo_frames: list[pd.DataFrame] = []
    e2e_pipeline_partial_combo_frames: list[pd.DataFrame] = []
    e2e_pipeline_embedding_frames: list[pd.DataFrame] = []
    e2e_operator_oracle_summary_frames: list[pd.DataFrame] = []
    root_cause_summary_frames: list[pd.DataFrame] = []
    torch_training_summaries: list[dict[str, Any]] = []
    model_bundle_manifest_rows: list[dict[str, Any]] = []
    effective_route_union: list[str] = []

    scaling_audits: list[pd.DataFrame] = []
    _progress(
        "main_start",
        detail=(
            f"policies={','.join(selected_policies)} datasets={','.join(selected_datasets)} "
            f"scales={','.join(str(value) for value in scales)} routes={','.join(requested_routes)}"
        ),
    )
    if bool(args.enable_scaling_audit):
        _progress("family_scaling_audit_start", detail="enabled")
        if "source" in selected_datasets:
            scaling_audits.append(
                _family_scaling_audit_for_root(
                    dataset_root=SOURCE_RAW_ROOT,
                    dataset_name="source_scheduler_20260419",
                    hw_config=source_hw,
                    passes=int(args.passes),
                )
            )
        if "source" in selected_datasets and any(policy in PHASE2_POLICIES for policy in selected_policies):
            scaling_audits.append(
                _family_scaling_audit_for_root(
                    dataset_root=PHASE2_SOURCE_ROOT,
                    dataset_name="source_thread_decoupling_phase2",
                    hw_config=source_hw,
                    passes=int(args.passes),
                )
            )
        if scaling_audits:
            pd.concat(scaling_audits, ignore_index=True).to_csv(artifact_root / "family_scaling_audit.csv", index=False)
        _progress("family_scaling_audit_done", detail=f"rows={sum(len(frame) for frame in scaling_audits)}")
    else:
        _progress("family_scaling_audit_skip", detail="disabled")

    for policy in selected_policies:
        print(f"[policy] {policy}", flush=True)
        policy_root = artifact_root / policy
        policy_root.mkdir(parents=True, exist_ok=True)
        summary_frames: list[pd.DataFrame] = []
        policy_routes = _effective_routes_for_policy(policy, requested_routes)
        effective_route_union.extend(policy_routes)
        policy_manifest: dict[str, Any] = {"requested_routes": requested_routes, "routes": policy_routes}

        if "source" in selected_datasets:
            source_root = data_root / f"{SOURCE_DATASET_NAME}_unseen_{policy}"
            (
                source_summary,
                source_breakdown,
                source_whitebox,
                source_invariance,
                source_perf_feature,
                source_family_mlp,
                source_hw_transfer,
                source_module_sensitivity,
                source_analytical_interval,
                source_analytical_trend,
                source_analytical_bottleneck,
                source_analytical_state,
                source_phase_error,
                source_prior_gap,
                source_rf_candidate_features,
                source_rf_pruned_features,
                source_rf_group_feature_count,
                source_prior_to_rf_gain,
                source_row_universe_guard,
                source_graph_feature_mapping,
                source_prediction_errors,
                source_pipeline_prediction_errors,
                source_torch_summary,
                source_meta,
            ) = _evaluate_dataset(
                policy=policy,
                dataset_name="source",
                dataset_root=source_root,
                artifact_root=artifact_root,
                hw_config=source_hw,
                alternate_hw_config=target_hw,
                scales=scales,
                routes=policy_routes,
                passes=int(args.passes),
                seed=int(args.seed),
                enable_e2e_pipeline=enable_e2e_pipeline,
                op_thread_scale_mode=op_thread_scale_mode,
                op_thread_scale_config=loaded_op_thread_scale_config,
                op_thread_scale_min_rows=int(args.op_thread_scale_min_rows),
                op_thread_scale_output=op_thread_scale_output,
                export_model_bundles=bool(args.export_model_bundles),
                model_bundle_output_dir=model_bundle_output_dir,
            )
            source_summary["policy"] = policy
            source_summary.to_csv(policy_root / "source_summary.csv", index=False)
            summary_frames.append(source_summary)
            policy_manifest["source_dataset_root"] = str(source_root)
            policy_manifest["source_split_root"] = source_meta["split_root"]
            policy_manifest["source_split_manifest_path"] = source_meta["split_manifest_path"]
            policy_manifest["source_input_dataset_root"] = source_meta["split_manifest"].get("input_dataset_root")
            policy_manifest["source_phase_scope"] = source_meta["split_manifest"].get("phase_scope")
            policy_manifest["source_op_thread_scale_paths"] = source_meta.get("op_thread_scale_paths", [])
            model_bundle_manifest_rows.extend(source_meta.get("model_bundle_manifest_rows", []))
            if not source_breakdown.empty:
                source_breakdown.to_csv(policy_root / "source_family_breakdown.csv", index=False)
                breakdown_frames.append(source_breakdown)
            if not source_whitebox.empty:
                source_whitebox.to_csv(policy_root / "source_whitebox_breakdown.csv", index=False)
                whitebox_frames.append(source_whitebox)
            if not source_invariance.empty:
                source_invariance.to_csv(policy_root / "source_whitebox_invariance.csv", index=False)
                invariance_frames.append(source_invariance)
            if not source_perf_feature.empty:
                source_perf_feature.to_csv(policy_root / "source_perf_feature_breakdown.csv", index=False)
                perf_feature_frames.append(source_perf_feature)
            if not source_family_mlp.empty:
                source_family_mlp.to_csv(policy_root / "source_family_mlp_enablement.csv", index=False)
                family_mlp_frames.append(source_family_mlp)
            if not source_hw_transfer.empty:
                source_hw_transfer.to_csv(policy_root / "source_hw_transfer_audit.csv", index=False)
                hw_transfer_frames.append(source_hw_transfer)
            if not source_module_sensitivity.empty:
                source_module_sensitivity.to_csv(policy_root / "source_module_sensitivity.csv", index=False)
                module_sensitivity_frames.append(source_module_sensitivity)
            if not source_analytical_interval.empty:
                source_analytical_interval.to_csv(policy_root / "source_analytical_interval_coverage.csv", index=False)
                analytical_interval_frames.append(source_analytical_interval)
            if not source_analytical_trend.empty:
                source_analytical_trend.to_csv(policy_root / "source_analytical_trend_audit.csv", index=False)
                analytical_trend_frames.append(source_analytical_trend)
            if not source_analytical_bottleneck.empty:
                source_analytical_bottleneck.to_csv(policy_root / "source_analytical_bottleneck_audit.csv", index=False)
                analytical_bottleneck_frames.append(source_analytical_bottleneck)
            if not source_analytical_state.empty:
                source_analytical_state.to_csv(policy_root / "source_analytical_state_audit.csv", index=False)
                analytical_state_frames.append(source_analytical_state)
            if not source_phase_error.empty:
                source_phase_error.to_csv(policy_root / "source_phase_error_audit.csv", index=False)
                phase_error_frames.append(source_phase_error)
            if not source_prior_gap.empty:
                source_prior_gap.to_csv(policy_root / "source_prior_gap_decomposition.csv", index=False)
                prior_gap_frames.append(source_prior_gap)
            if not source_rf_candidate_features.empty:
                source_rf_candidate_features.to_csv(policy_root / "source_rf_candidate_feature_report.csv", index=False)
                rf_candidate_feature_frames.append(source_rf_candidate_features)
            if not source_rf_pruned_features.empty:
                source_rf_pruned_features.to_csv(policy_root / "source_rf_pruned_feature_manifest.csv", index=False)
                rf_pruned_feature_frames.append(source_rf_pruned_features)
            if not source_rf_group_feature_count.empty:
                source_rf_group_feature_count.to_csv(policy_root / "source_rf_group_feature_count.csv", index=False)
                rf_group_feature_count_frames.append(source_rf_group_feature_count)
            if not source_prior_to_rf_gain.empty:
                source_prior_to_rf_gain.to_csv(policy_root / "source_prior_to_rf_gain.csv", index=False)
                prior_to_rf_gain_frames.append(source_prior_to_rf_gain)
            if not source_row_universe_guard.empty:
                source_row_universe_guard.to_csv(policy_root / "source_row_universe_guard.csv", index=False)
                row_universe_guard_frames.append(source_row_universe_guard)
            if not source_graph_feature_mapping.empty:
                source_graph_feature_mapping.to_csv(policy_root / "source_graph_feature_mapping_audit.csv", index=False)
                graph_feature_mapping_frames.append(source_graph_feature_mapping)
            if not source_prediction_errors.empty:
                source_prediction_errors.to_csv(policy_root / "source_prediction_error_rows.csv", index=False)
                prediction_error_frames.append(source_prediction_errors)
            if not source_pipeline_prediction_errors.empty:
                source_pipeline_prediction_errors.to_csv(policy_root / "source_pipeline_prediction_error_rows.csv", index=False)
                pipeline_prediction_error_frames.append(source_pipeline_prediction_errors)
            if enable_e2e_pipeline and (not source_prediction_errors.empty or not source_pipeline_prediction_errors.empty):
                (
                    source_e2e_summary,
                    source_e2e_combo_metrics,
                    source_e2e_partial_combo_metrics,
                    source_e2e_embedding_check,
                    source_pipeline_audit,
                ) = evaluate_source_test_pipeline(
                    prediction_rows=source_pipeline_prediction_errors if not source_pipeline_prediction_errors.empty else source_prediction_errors,
                    split_root=source_root,
                    policy=policy,
                    dataset="source",
                )
                if not source_e2e_summary.empty:
                    source_e2e_summary = _merge_e2e_route_metadata(source_e2e_summary, source_summary)
                source_e2e_summary.to_csv(policy_root / "source_e2e_pipeline_summary.csv", index=False)
                source_e2e_combo_metrics.to_csv(policy_root / "source_e2e_pipeline_combo_metrics.csv", index=False)
                source_e2e_partial_combo_metrics.to_csv(
                    policy_root / "source_e2e_pipeline_partial_combo_metrics.csv",
                    index=False,
                )
                source_e2e_embedding_check.to_csv(
                    policy_root / "source_e2e_pipeline_embedding_order_check.csv",
                    index=False,
                )
                source_pipeline_audit.to_csv(
                    policy_root / "source_e2e_pipeline_artifact_index_audit.csv",
                    index=False,
                )
                if not source_e2e_summary.empty:
                    e2e_pipeline_summary_frames.append(source_e2e_summary)
                if not source_e2e_combo_metrics.empty:
                    e2e_pipeline_combo_frames.append(source_e2e_combo_metrics)
                if bool(args.enable_root_cause_oracles):
                    oracle_prediction_rows = _operator_oracle_prediction_rows(
                        source_pipeline_prediction_errors if not source_pipeline_prediction_errors.empty else source_prediction_errors,
                            base_routes=[
                                WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
                                WHITEBOX_ORT_KERNEL_RF_ROUTE,
                                WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                                WHITEBOX_GATHER_SLICE_RF_ROUTE,
                            ],
                    )
                    if not oracle_prediction_rows.empty:
                        (
                            source_oracle_e2e_summary,
                            source_oracle_e2e_combo_metrics,
                            _source_oracle_e2e_partial_combo_metrics,
                            _source_oracle_e2e_embedding_check,
                            _source_oracle_pipeline_audit,
                        ) = evaluate_source_test_pipeline(
                            prediction_rows=oracle_prediction_rows,
                            split_root=source_root,
                            policy=policy,
                            dataset="source",
                        )
                        source_oracle_e2e_summary.to_csv(policy_root / "source_e2e_operator_oracle_summary.csv", index=False)
                        source_oracle_e2e_combo_metrics.to_csv(policy_root / "source_e2e_operator_oracle_combo_metrics.csv", index=False)
                        root_cause = _root_cause_oracle_rows(
                            current_e2e_summary=source_e2e_summary,
                            oracle_e2e_summary=source_oracle_e2e_summary,
                            prediction_rows=source_pipeline_prediction_errors
                            if not source_pipeline_prediction_errors.empty
                            else source_prediction_errors,
                            perf_feature=source_perf_feature,
                        )
                        root_cause.to_csv(policy_root / "source_root_cause_summary.csv", index=False)
                        if not source_oracle_e2e_summary.empty:
                            e2e_operator_oracle_summary_frames.append(source_oracle_e2e_summary)
                        if not root_cause.empty:
                            root_cause_summary_frames.append(root_cause)
                if not source_e2e_partial_combo_metrics.empty:
                    e2e_pipeline_partial_combo_frames.append(source_e2e_partial_combo_metrics)
                if not source_e2e_embedding_check.empty:
                    e2e_pipeline_embedding_frames.append(source_e2e_embedding_check)
                if not source_pipeline_audit.empty:
                    pipeline_audit_frames.append(source_pipeline_audit)
            if source_torch_summary:
                _json_dump(policy_root / "source_torch_training_summary.json", source_torch_summary)
                torch_training_summaries.extend(source_torch_summary)

        if "target" in selected_datasets:
            target_root = data_root / f"{TARGET_DATASET_NAME}_unseen_{policy}"
            (
                target_summary,
                target_breakdown,
                target_whitebox,
                target_invariance,
                target_perf_feature,
                target_family_mlp,
                target_hw_transfer,
                target_module_sensitivity,
                target_analytical_interval,
                target_analytical_trend,
                target_analytical_bottleneck,
                target_analytical_state,
                target_phase_error,
                target_prior_gap,
                target_rf_candidate_features,
                target_rf_pruned_features,
                target_rf_group_feature_count,
                target_prior_to_rf_gain,
                target_row_universe_guard,
                target_graph_feature_mapping,
                target_prediction_errors,
                target_pipeline_prediction_errors,
                target_torch_summary,
                target_meta,
            ) = _evaluate_dataset(
                policy=policy,
                dataset_name="target",
                dataset_root=target_root,
                artifact_root=artifact_root,
                hw_config=target_hw,
                alternate_hw_config=source_hw,
                scales=scales,
                routes=policy_routes,
                passes=int(args.passes),
                seed=int(args.seed),
                enable_e2e_pipeline=enable_e2e_pipeline,
                op_thread_scale_mode=op_thread_scale_mode,
                op_thread_scale_config=loaded_op_thread_scale_config,
                op_thread_scale_min_rows=int(args.op_thread_scale_min_rows),
                op_thread_scale_output=op_thread_scale_output,
                export_model_bundles=bool(args.export_model_bundles),
                model_bundle_output_dir=model_bundle_output_dir,
            )
            target_summary["policy"] = policy
            target_summary.to_csv(policy_root / "target_summary.csv", index=False)
            summary_frames.append(target_summary)
            policy_manifest["target_dataset_root"] = str(target_root)
            policy_manifest["target_op_thread_scale_paths"] = target_meta.get("op_thread_scale_paths", [])
            policy_manifest["target_split_root"] = target_meta["split_root"]
            policy_manifest["target_split_manifest_path"] = target_meta["split_manifest_path"]
            policy_manifest["target_input_dataset_root"] = target_meta["split_manifest"].get("input_dataset_root")
            policy_manifest["target_phase_scope"] = target_meta["split_manifest"].get("phase_scope")
            model_bundle_manifest_rows.extend(target_meta.get("model_bundle_manifest_rows", []))
            if not target_breakdown.empty:
                target_breakdown.to_csv(policy_root / "target_family_breakdown.csv", index=False)
                breakdown_frames.append(target_breakdown)
            if not target_whitebox.empty:
                target_whitebox.to_csv(policy_root / "target_whitebox_breakdown.csv", index=False)
                whitebox_frames.append(target_whitebox)
            if not target_invariance.empty:
                target_invariance.to_csv(policy_root / "target_whitebox_invariance.csv", index=False)
                invariance_frames.append(target_invariance)
            if not target_perf_feature.empty:
                target_perf_feature.to_csv(policy_root / "target_perf_feature_breakdown.csv", index=False)
                perf_feature_frames.append(target_perf_feature)
            if not target_family_mlp.empty:
                target_family_mlp.to_csv(policy_root / "target_family_mlp_enablement.csv", index=False)
                family_mlp_frames.append(target_family_mlp)
            if not target_hw_transfer.empty:
                target_hw_transfer.to_csv(policy_root / "target_hw_transfer_audit.csv", index=False)
                hw_transfer_frames.append(target_hw_transfer)
            if not target_module_sensitivity.empty:
                target_module_sensitivity.to_csv(policy_root / "target_module_sensitivity.csv", index=False)
                module_sensitivity_frames.append(target_module_sensitivity)
            if not target_analytical_interval.empty:
                target_analytical_interval.to_csv(policy_root / "target_analytical_interval_coverage.csv", index=False)
                analytical_interval_frames.append(target_analytical_interval)
            if not target_analytical_trend.empty:
                target_analytical_trend.to_csv(policy_root / "target_analytical_trend_audit.csv", index=False)
                analytical_trend_frames.append(target_analytical_trend)
            if not target_analytical_bottleneck.empty:
                target_analytical_bottleneck.to_csv(policy_root / "target_analytical_bottleneck_audit.csv", index=False)
                analytical_bottleneck_frames.append(target_analytical_bottleneck)
            if not target_analytical_state.empty:
                target_analytical_state.to_csv(policy_root / "target_analytical_state_audit.csv", index=False)
                analytical_state_frames.append(target_analytical_state)
            if not target_phase_error.empty:
                target_phase_error.to_csv(policy_root / "target_phase_error_audit.csv", index=False)
                phase_error_frames.append(target_phase_error)
            if not target_prior_gap.empty:
                target_prior_gap.to_csv(policy_root / "target_prior_gap_decomposition.csv", index=False)
                prior_gap_frames.append(target_prior_gap)
            if not target_rf_candidate_features.empty:
                target_rf_candidate_features.to_csv(policy_root / "target_rf_candidate_feature_report.csv", index=False)
                rf_candidate_feature_frames.append(target_rf_candidate_features)
            if not target_rf_pruned_features.empty:
                target_rf_pruned_features.to_csv(policy_root / "target_rf_pruned_feature_manifest.csv", index=False)
                rf_pruned_feature_frames.append(target_rf_pruned_features)
            if not target_rf_group_feature_count.empty:
                target_rf_group_feature_count.to_csv(policy_root / "target_rf_group_feature_count.csv", index=False)
                rf_group_feature_count_frames.append(target_rf_group_feature_count)
            if not target_prior_to_rf_gain.empty:
                target_prior_to_rf_gain.to_csv(policy_root / "target_prior_to_rf_gain.csv", index=False)
                prior_to_rf_gain_frames.append(target_prior_to_rf_gain)
            if not target_row_universe_guard.empty:
                target_row_universe_guard.to_csv(policy_root / "target_row_universe_guard.csv", index=False)
                row_universe_guard_frames.append(target_row_universe_guard)
            if not target_graph_feature_mapping.empty:
                target_graph_feature_mapping.to_csv(policy_root / "target_graph_feature_mapping_audit.csv", index=False)
                graph_feature_mapping_frames.append(target_graph_feature_mapping)
            if not target_prediction_errors.empty:
                target_prediction_errors.to_csv(policy_root / "target_prediction_error_rows.csv", index=False)
                prediction_error_frames.append(target_prediction_errors)
            if target_torch_summary:
                _json_dump(policy_root / "target_torch_training_summary.json", target_torch_summary)
                torch_training_summaries.extend(target_torch_summary)

        if not summary_frames:
            raise ValueError("No datasets were selected for evaluation")

        policy_summary = pd.concat(summary_frames, ignore_index=True)
        policy_summary["scale_num"] = policy_summary["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        policy_summary = policy_summary.sort_values(["dataset", "route", "scale_num"]).drop(columns=["scale_num"]).reset_index(drop=True)
        policy_summary.to_csv(policy_root / "summary.csv", index=False)
        _plot_metric(policy_summary, metric="mape", output_path=policy_root / "mape_vs_scale.png")
        _plot_metric(policy_summary, metric="p90", output_path=policy_root / "p90_vs_scale.png")
        manifest["policies"][policy] = policy_manifest
        all_rows.append(policy_summary)

    combined = pd.concat(all_rows, ignore_index=True)
    combined["scale_num"] = combined["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    combined = combined.sort_values(["policy", "dataset", "route", "scale_num"]).drop(columns=["scale_num"]).reset_index(drop=True)
    manifest["routes"] = combined["route"].astype(str).drop_duplicates().tolist()
    manifest["whitebox_prior_activated"] = any(
        route in manifest["routes"]
        for route in [
            *WHITEBOX_ROUTES,
            WHITEBOX_ORT_KERNEL_RF_ROUTE,
            WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
            WHITEBOX_GATHER_SLICE_RF_ROUTE,
            WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
            *_benchmark_variant_routes(),
        ]
    )
    combined.to_csv(artifact_root / "combined_summary.csv", index=False)
    combined.to_csv(artifact_root / "summary.csv", index=False)
    best = _best_by_policy(combined)
    best.to_csv(artifact_root / "best_by_policy.csv", index=False)
    _acceptance_matrix(best).to_csv(artifact_root / "acceptance_matrix.csv", index=False)
    comparison_summary = _comparison_summary_with_source_baseline(combined, selected_policies)
    analytical_vs_rf = _analytical_vs_rf_error_summary(comparison_summary)
    if not analytical_vs_rf.empty:
        analytical_vs_rf.to_csv(artifact_root / "analytical_vs_rf_error_summary.csv", index=False)
    source_target_delta = _source_vs_target_operator_accuracy_delta(comparison_summary)
    if not source_target_delta.empty:
        source_target_delta.to_csv(artifact_root / "source_vs_target_operator_accuracy_delta.csv", index=False)
    v3_gap_report = _v3_vs_v2_gap_report(combined)
    if not v3_gap_report.empty:
        v3_gap_report.to_csv(artifact_root / "v3_vs_v2_gap_report.csv", index=False)
    breakdown = pd.DataFrame()
    if breakdown_frames:
        breakdown = pd.concat(breakdown_frames, ignore_index=True)
        breakdown["scale_num"] = breakdown["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        breakdown = breakdown.sort_values(["policy", "dataset", "route", "scale_num", "family_bucket", "op_type"]).drop(columns=["scale_num"]).reset_index(drop=True)
        breakdown.to_csv(artifact_root / "family_error_audit.csv", index=False)
    if whitebox_frames:
        whitebox = pd.concat(whitebox_frames, ignore_index=True)
        whitebox["scale_num"] = whitebox["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        whitebox = whitebox.sort_values(
            ["policy", "dataset", "route", "scale_num", "whitebox_prior_family", "op_type", "whitebox_prior_source", "whitebox_residual_source"]
        ).drop(columns=["scale_num"]).reset_index(drop=True)
        whitebox.to_csv(artifact_root / "policy_whitebox_source_breakdown.csv", index=False)
    if invariance_frames:
        invariance = pd.concat(invariance_frames, ignore_index=True)
        invariance["scale_num"] = invariance["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        invariance = invariance.sort_values(["policy", "dataset", "route", "scale_num"]).drop(columns=["scale_num"]).reset_index(drop=True)
        invariance.to_csv(artifact_root / "whitebox_invariance_audit.csv", index=False)
    if perf_feature_frames:
        perf_feature = pd.concat(perf_feature_frames, ignore_index=True)
        perf_feature["scale_num"] = perf_feature["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        perf_feature = perf_feature.sort_values(
            ["policy", "dataset", "route", "scale_num", "whitebox_prior_family", "op_type", "whitebox_residual_source"]
        ).drop(columns=["scale_num"]).reset_index(drop=True)
        perf_feature.to_csv(artifact_root / "policy_perf_feature_breakdown.csv", index=False)
    if family_mlp_frames:
        family_mlp = pd.concat(family_mlp_frames, ignore_index=True)
        family_mlp["scale_num"] = family_mlp["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        family_mlp = family_mlp.sort_values(
            ["policy", "dataset", "route", "scale_num", "family_name"]
        ).drop(columns=["scale_num"]).reset_index(drop=True)
        family_mlp.to_csv(artifact_root / "family_mlp_enablement.csv", index=False)
    if rf_candidate_feature_frames:
        rf_candidate = pd.concat(rf_candidate_feature_frames, ignore_index=True)
        rf_candidate["scale_num"] = rf_candidate["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        rf_candidate = rf_candidate.sort_values(["policy", "dataset", "route", "scale_num", "rf_group", "feature_name"]).drop(columns=["scale_num"]).reset_index(drop=True)
        rf_candidate.to_csv(artifact_root / "rf_candidate_feature_report.csv", index=False)
        benchmark_candidate = rf_candidate.loc[rf_candidate["route"].astype(str).map(_is_benchmark_route)].reset_index(drop=True)
        if not benchmark_candidate.empty:
            benchmark_candidate.to_csv(artifact_root / "benchmark_rf_candidate_feature_report.csv", index=False)
    rf_pruned = pd.DataFrame()
    if rf_pruned_feature_frames:
        rf_pruned = pd.concat(rf_pruned_feature_frames, ignore_index=True)
        rf_pruned["scale_num"] = rf_pruned["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        rf_pruned = rf_pruned.sort_values(["policy", "dataset", "route", "scale_num", "rf_group", "feature_name"]).drop(columns=["scale_num"]).reset_index(drop=True)
        rf_pruned.to_csv(artifact_root / "rf_pruned_feature_manifest.csv", index=False)
        benchmark_pruned = rf_pruned.loc[rf_pruned["route"].astype(str).map(_is_benchmark_route)].reset_index(drop=True)
        if not benchmark_pruned.empty:
            benchmark_pruned.loc[benchmark_pruned["route"].astype(str).eq(benchmark_variant_route_name("upper"))].to_csv(
                artifact_root / "benchmark_rf_pruned_feature_manifest_upper.csv", index=False
            )
            benchmark_pruned.loc[benchmark_pruned["route"].astype(str).eq(benchmark_variant_route_name("compact"))].to_csv(
                artifact_root / "benchmark_rf_pruned_feature_manifest_compact.csv", index=False
            )
        compact_indexed_features = rf_pruned.loc[
            rf_pruned["route"].astype(str).eq(WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE)
            & rf_pruned["rf_group"].astype(str).eq("indexed_block_copy_rf")
        ].reset_index(drop=True)
        if not compact_indexed_features.empty:
            compact_indexed_features.to_csv(artifact_root / "indexed_copy_selected_features_v5_compact.csv", index=False)
    if rf_group_feature_count_frames:
        rf_group_count = pd.concat(rf_group_feature_count_frames, ignore_index=True)
        rf_group_count["scale_num"] = rf_group_count["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        rf_group_count = rf_group_count.sort_values(["policy", "dataset", "route", "scale_num", "rf_group"]).drop(columns=["scale_num"]).reset_index(drop=True)
        rf_group_count.to_csv(artifact_root / "rf_group_feature_count.csv", index=False)
        benchmark_group_count = rf_group_count.loc[rf_group_count["route"].astype(str).map(_is_benchmark_route)].reset_index(drop=True)
        if not benchmark_group_count.empty:
            benchmark_group_count.to_csv(artifact_root / "benchmark_rf_group_feature_count.csv", index=False)
    if prior_to_rf_gain_frames:
        prior_gain = pd.concat(prior_to_rf_gain_frames, ignore_index=True)
        prior_gain["scale_num"] = prior_gain["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        prior_gain = prior_gain.sort_values(["policy", "dataset", "route", "scale_num", "rf_group", "op_type"]).drop(columns=["scale_num"]).reset_index(drop=True)
        prior_gain.to_csv(artifact_root / "prior_to_rf_gain.csv", index=False)
        benchmark_gain = prior_gain.loc[prior_gain["route"].astype(str).map(_is_benchmark_route)].reset_index(drop=True)
        if not benchmark_gain.empty:
            benchmark_gain.to_csv(artifact_root / "benchmark_prior_to_rf_gain.csv", index=False)
    if row_universe_guard_frames:
        row_guard = pd.concat(row_universe_guard_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
        row_guard.to_csv(artifact_root / "stage0_row_universe_guard.csv", index=False)
    if graph_feature_mapping_frames:
        graph_mapping = pd.concat(graph_feature_mapping_frames, ignore_index=True)
        graph_mapping = graph_mapping.sort_values(["policy", "dataset", "split"]).reset_index(drop=True)
        graph_mapping.to_csv(artifact_root / "graph_feature_mapping_audit.csv", index=False)
    if hw_transfer_frames:
        hw_transfer = pd.concat(hw_transfer_frames, ignore_index=True)
        hw_transfer["scale_num"] = hw_transfer["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        hw_transfer = hw_transfer.sort_values(["policy", "dataset", "route", "scale_num"]).drop(columns=["scale_num"]).reset_index(drop=True)
        hw_transfer.to_csv(artifact_root / "hw_transfer_audit.csv", index=False)
    analytical_hw_perturbation = pd.DataFrame()
    module_sensitivity_full = pd.DataFrame()
    if module_sensitivity_frames:
        module_sensitivity_full = pd.concat(module_sensitivity_frames, ignore_index=True)
        module_sensitivity_full["scale_num"] = module_sensitivity_full["scale"].astype(str).map(
            lambda value: float(str(value).rstrip("%"))
        )
        module_sensitivity_full = module_sensitivity_full.sort_values(
            ["policy", "dataset", "route", "field_name", "scale_num", "direction"]
        ).drop(columns=["scale_num"]).reset_index(drop=True)
        analytical_hw_perturbation = module_sensitivity_full.loc[
            module_sensitivity_full["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)
        ].reset_index(drop=True)
        if not analytical_hw_perturbation.empty:
            analytical_hw_perturbation.to_csv(artifact_root / "analytical_hw_perturbation_audit.csv", index=False)
        rf_sensitivity = module_sensitivity_full.loc[
            ~module_sensitivity_full["route"].astype(str).eq(WHITEBOX_PRIOR_ONLY_ROUTE)
        ].reset_index(drop=True)
        if not rf_sensitivity.empty:
            rf_sensitivity.to_csv(artifact_root / "module_sensitivity_audit.csv", index=False)
    benchmark_route_set = {WHITEBOX_PRIOR_SIMPLE_RF_ROUTE, *_benchmark_variant_routes()}
    has_benchmark_routes = bool(combined["route"].astype(str).map(_is_benchmark_route).any())
    benchmark_summary = combined.loc[combined["route"].astype(str).isin(benchmark_route_set)].reset_index(drop=True)
    if has_benchmark_routes and not benchmark_summary.empty:
        benchmark_summary.to_csv(artifact_root / "benchmark_route_summary.csv", index=False)
        benchmark_best = best.loc[best["route"].astype(str).isin(benchmark_route_set)].reset_index(drop=True)
        benchmark_best.to_csv(artifact_root / "benchmark_final_best_route_by_policy.csv", index=False)
        formal_best = benchmark_best.loc[
            benchmark_best["route"].astype(str).eq(WHITEBOX_PRIOR_SIMPLE_RF_ROUTE),
            ["policy", "dataset", "scale", "mape", "p90"],
        ].rename(
            columns={
                "scale": "formal_best_scale",
                "mape": "formal_mape",
                "p90": "formal_p90",
            }
        )
        benchmark_delta = formal_best.copy()
        for variant in WHITEBOX_WORKLOAD_BENCHMARK_VARIANTS:
            variant_route = benchmark_variant_route_name(variant)
            variant_best = benchmark_best.loc[
                benchmark_best["route"].astype(str).eq(variant_route),
                ["policy", "dataset", "scale", "mape", "p90"],
            ].rename(
                columns={
                    "scale": f"{variant}_best_scale",
                    "mape": f"{variant}_mape",
                    "p90": f"{variant}_p90",
                }
            )
            benchmark_delta = benchmark_delta.merge(variant_best, on=["policy", "dataset"], how="left")
            if f"{variant}_mape" in benchmark_delta.columns:
                benchmark_delta[f"{variant}_delta_pp"] = benchmark_delta["formal_mape"] - benchmark_delta[f"{variant}_mape"]
            if f"{variant}_p90" in benchmark_delta.columns:
                benchmark_delta[f"{variant}_p90_delta_pp"] = benchmark_delta["formal_p90"] - benchmark_delta[f"{variant}_p90"]
        benchmark_delta.to_csv(artifact_root / "benchmark_vs_formal_delta.csv", index=False)
    analytical_interval = pd.DataFrame()
    if analytical_interval_frames:
        analytical_interval = pd.concat(analytical_interval_frames, ignore_index=True)
        analytical_interval["scale_num"] = analytical_interval["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        analytical_interval = analytical_interval.sort_values(["policy", "dataset", "route", "scale_num"]).drop(columns=["scale_num"]).reset_index(drop=True)
        analytical_interval.to_csv(artifact_root / "analytical_interval_coverage.csv", index=False)
    analytical_trend = pd.DataFrame()
    if analytical_trend_frames:
        analytical_trend = pd.concat(analytical_trend_frames, ignore_index=True)
        analytical_trend["scale_num"] = analytical_trend["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        analytical_trend = analytical_trend.sort_values(["policy", "dataset", "route", "scale_num", "slice_kind", "family_bucket", "op_type"]).drop(columns=["scale_num"]).reset_index(drop=True)
        analytical_trend.to_csv(artifact_root / "analytical_trend_audit.csv", index=False)
    analytical_bottleneck = pd.DataFrame()
    if analytical_bottleneck_frames:
        analytical_bottleneck = pd.concat(analytical_bottleneck_frames, ignore_index=True)
        analytical_bottleneck["scale_num"] = analytical_bottleneck["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        analytical_bottleneck = analytical_bottleneck.sort_values(["policy", "dataset", "route", "scale_num"]).drop(columns=["scale_num"]).reset_index(drop=True)
        analytical_bottleneck.to_csv(artifact_root / "analytical_bottleneck_audit.csv", index=False)
    if analytical_state_frames:
        analytical_state = pd.concat(analytical_state_frames, ignore_index=True)
        analytical_state["scale_num"] = analytical_state["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        analytical_state = analytical_state.sort_values(["policy", "dataset", "route", "scale_num", "state_name", "state_value"]).drop(columns=["scale_num"]).reset_index(drop=True)
        analytical_state.to_csv(artifact_root / "analytical_state_audit.csv", index=False)
    phase_error = pd.DataFrame()
    if phase_error_frames:
        phase_error = pd.concat(phase_error_frames, ignore_index=True)
        phase_error["scale_num"] = phase_error["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        phase_error = phase_error.sort_values(["policy", "dataset", "route", "scale_num", "slice_kind", "exec_region_static", "op_type"]).drop(columns=["scale_num"]).reset_index(drop=True)
        phase_error.to_csv(artifact_root / "phase_error_audit.csv", index=False)
    prior_gap = pd.DataFrame()
    if prior_gap_frames:
        prior_gap = pd.concat(prior_gap_frames, ignore_index=True)
        prior_gap["scale_num"] = prior_gap["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        prior_gap = prior_gap.sort_values(["policy", "dataset", "route", "scale_num", "slice_kind", "exec_region_static"]).drop(columns=["scale_num"]).reset_index(drop=True)
        prior_gap.to_csv(artifact_root / "prior_gap_decomposition.csv", index=False)
        mainline_vs_shadow = _mainline_vs_shadow_summary(combined, prior_gap)
        if not mainline_vs_shadow.empty:
            mainline_vs_shadow.to_csv(artifact_root / "mainline_vs_shadow_summary.csv", index=False)
    validation = pd.DataFrame()
    if not analytical_interval.empty or not analytical_trend.empty or not analytical_bottleneck.empty or not analytical_hw_perturbation.empty:
        validation = analytical_interval.loc[analytical_interval["scope"].astype(str).eq("overall")].copy()
        validation = validation.rename(columns={"center_spearman": "policy_spearman"})
        if not analytical_bottleneck.empty:
            validation = validation.merge(
                analytical_bottleneck[["policy", "dataset", "route", "scale", "clear_bottleneck_rows", "stable_bottleneck_rate"]],
                on=["policy", "dataset", "route", "scale"],
                how="left",
            )
        if not analytical_hw_perturbation.empty:
            perturb = analytical_hw_perturbation.copy()
            if "applicable" in perturb.columns:
                perturb = perturb.loc[perturb["applicable"].astype(bool)].copy()
            perturb["analytical_hw_pass"] = perturb["expectation_pass"].astype(bool) & perturb["monotonic_pass"].astype(bool)
            perturb_summary = (
                perturb.groupby(["policy", "dataset", "route", "scale"], as_index=False)["analytical_hw_pass"]
                .mean()
                .rename(columns={"analytical_hw_pass": "analytical_hw_pass_rate"})
            )
            validation = validation.merge(perturb_summary, on=["policy", "dataset", "route", "scale"], how="left")
        for column in ("coverage_rate", "policy_spearman", "stable_bottleneck_rate", "analytical_hw_pass_rate"):
            if column not in validation.columns:
                validation[column] = np.nan
        validation["coverage_pass"] = validation["coverage_rate"] >= 0.80
        validation["spearman_pass"] = validation["policy_spearman"] >= 0.75
        validation["bottleneck_pass"] = validation["stable_bottleneck_rate"].fillna(0.0) >= 0.70
        validation["hw_perturb_pass"] = validation["analytical_hw_pass_rate"].fillna(0.0) >= 0.95
        validation["overall_pass"] = validation["coverage_pass"] & validation["spearman_pass"] & validation["bottleneck_pass"] & validation["hw_perturb_pass"]
        validation.to_csv(artifact_root / "analytical_validation_summary.csv", index=False)
    analytical_scorecard = _analytical_quality_scorecard(
        analytical_interval,
        analytical_bottleneck,
        module_sensitivity_full,
    )
    if not analytical_scorecard.empty:
        analytical_scorecard["scale_num"] = analytical_scorecard["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        analytical_scorecard = analytical_scorecard.sort_values(
            ["policy", "dataset", "route", "scale_num", "metric_name"]
        ).drop(columns=["scale_num"]).reset_index(drop=True)
        analytical_scorecard.to_csv(artifact_root / "analytical_quality_scorecard.csv", index=False)
    prediction_errors = pd.DataFrame()
    if prediction_error_frames:
        prediction_errors = pd.concat(prediction_error_frames, ignore_index=True)
        prediction_errors["scale_num"] = prediction_errors["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        prediction_errors = prediction_errors.sort_values(["policy", "dataset", "route", "scale_num", "row_uid"]).drop(columns=["scale_num"]).reset_index(drop=True)
        prediction_errors.to_csv(artifact_root / "prediction_error_rows.csv", index=False)
        noise_by_policy = _noise_by_policy(prediction_errors)
        if not noise_by_policy.empty:
            noise_by_policy.to_csv(artifact_root / "noise_by_policy.csv", index=False)
        noise_by_op_region = _noise_by_op_region(prediction_errors)
        if not noise_by_op_region.empty:
            noise_by_op_region.to_csv(artifact_root / "noise_by_op_region.csv", index=False)
        filter_sweep = _global_filter_sweep(prediction_errors)
        if not filter_sweep.empty:
            filter_sweep.to_csv(artifact_root / "global_filter_sweep.csv", index=False)
        singleop_threshold = _singleop_optype_threshold_audit(prediction_errors, threshold=0.15)
        if not singleop_threshold.empty:
            singleop_threshold["scale_num"] = singleop_threshold["scale"].astype(str).map(
                lambda value: float(str(value).rstrip("%"))
            )
            singleop_threshold = singleop_threshold.sort_values(
                ["policy", "dataset", "route", "scale_num", "op_type"]
            ).drop(columns=["scale_num"]).reset_index(drop=True)
            singleop_threshold.to_csv(artifact_root / "singleop_optype_threshold_audit.csv", index=False)
        gather_slice_prior_ablation = _gather_slice_prior_ablation_rows(prediction_errors)
        if not gather_slice_prior_ablation.empty:
            gather_slice_prior_ablation["scale_num"] = gather_slice_prior_ablation["scale"].astype(str).map(
                lambda value: float(str(value).rstrip("%"))
            )
            gather_slice_prior_ablation = gather_slice_prior_ablation.sort_values(
                ["policy", "dataset", "route", "scale_num", "scheme"]
            ).drop(columns=["scale_num"]).reset_index(drop=True)
            gather_slice_prior_ablation.to_csv(artifact_root / "gather_slice_prior_ablation.csv", index=False)
    pipeline_prediction_errors = pd.DataFrame()
    if pipeline_prediction_error_frames:
        pipeline_prediction_errors = pd.concat(pipeline_prediction_error_frames, ignore_index=True)
        pipeline_prediction_errors["scale_num"] = pipeline_prediction_errors["scale"].astype(str).map(
            lambda value: float(str(value).rstrip("%"))
        )
        pipeline_prediction_errors = pipeline_prediction_errors.sort_values(
            ["policy", "dataset", "route", "scale_num", "row_uid"]
        ).drop(columns=["scale_num"]).reset_index(drop=True)
        pipeline_prediction_errors.to_csv(artifact_root / "pipeline_prediction_error_rows.csv", index=False)
    if pipeline_audit_frames:
        pipeline_audit = pd.concat(pipeline_audit_frames, ignore_index=True)
        pipeline_audit = pipeline_audit.sort_values(["policy", "dataset", "source_name", "combo"]).reset_index(drop=True)
        pipeline_audit.to_csv(artifact_root / "pipeline_artifact_index_audit.csv", index=False)
    e2e_summary = pd.DataFrame()
    if e2e_pipeline_summary_frames:
        e2e_summary = pd.concat(e2e_pipeline_summary_frames, ignore_index=True)
        e2e_summary["scale_num"] = e2e_summary["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        e2e_summary = e2e_summary.sort_values(["policy", "dataset", "route", "scale_num"]).drop(columns=["scale_num"]).reset_index(drop=True)
        e2e_summary.to_csv(artifact_root / "e2e_pipeline_summary.csv", index=False)
        best_e2e_pipeline_by_policy(e2e_summary).to_csv(artifact_root / "best_e2e_pipeline_by_policy.csv", index=False)
    v2_vs_v3_delta = _route_metric_delta_rows(combined, e2e_summary)
    if not v2_vs_v3_delta.empty:
        v2_vs_v3_delta.to_csv(artifact_root / "v2_vs_v3_delta.csv", index=False)
    v2_vs_v4_delta = _route_metric_delta_rows(
        combined,
        e2e_summary,
        candidate_route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
        candidate_label="v4",
        mape_gate_pp=1.5,
        singleop_mape_gate_pp=0.0,
    )
    if not v2_vs_v4_delta.empty:
        v2_vs_v4_delta.to_csv(artifact_root / "v2_vs_v4_delta.csv", index=False)
    v4_vs_v5_delta = _route_pair_metric_delta_rows(
        combined,
        e2e_summary,
        base_route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
        candidate_route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
        base_label="v4",
        candidate_label="v5",
    )
    if not v4_vs_v5_delta.empty:
        v4_vs_v5_delta.to_csv(artifact_root / "v4_vs_v5_e2e_delta.csv", index=False)
    v5_vs_compact_delta = _route_pair_metric_delta_rows(
        combined,
        e2e_summary,
        base_route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
        candidate_route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
        base_label="v5",
        candidate_label="v5_compact",
    )
    if not v5_vs_compact_delta.empty:
        v5_vs_compact_delta.to_csv(artifact_root / "v5_vs_v5_compact_delta.csv", index=False)
    if e2e_operator_oracle_summary_frames:
        operator_oracle = pd.concat(e2e_operator_oracle_summary_frames, ignore_index=True)
        operator_oracle["scale_num"] = operator_oracle["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        operator_oracle = operator_oracle.sort_values(["policy", "dataset", "route", "scale_num"]).drop(columns=["scale_num"]).reset_index(drop=True)
        operator_oracle.to_csv(artifact_root / "e2e_operator_oracle_summary.csv", index=False)
    if root_cause_summary_frames:
        root_cause_summary = pd.concat(root_cause_summary_frames, ignore_index=True)
        root_cause_summary["scale_num"] = root_cause_summary["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        root_cause_summary = root_cause_summary.sort_values(["policy", "dataset", "route", "scale_num", "op_type"]).drop(columns=["scale_num"]).reset_index(drop=True)
        root_cause_summary.to_csv(artifact_root / "root_cause_summary.csv", index=False)
    if e2e_pipeline_combo_frames:
        e2e_combo = pd.concat(e2e_pipeline_combo_frames, ignore_index=True)
        e2e_combo["scale_num"] = e2e_combo["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        e2e_combo = e2e_combo.sort_values(["policy", "dataset", "route", "scale_num", "case_id", "batch_size", "num_indices_per_lookup"]).drop(columns=["scale_num"]).reset_index(drop=True)
        e2e_combo.to_csv(artifact_root / "e2e_pipeline_combo_metrics.csv", index=False)
        e2e_prediction_source = pipeline_prediction_errors if not pipeline_prediction_errors.empty else prediction_errors
        e2e_operator_signed = _e2e_operator_signed_error_by_combo(e2e_prediction_source)
        e2e_tail_attribution = pd.DataFrame()
        if not e2e_operator_signed.empty:
            e2e_operator_signed["scale_num"] = e2e_operator_signed["scale"].astype(str).map(
                lambda value: float(str(value).rstrip("%"))
            )
            e2e_operator_signed = e2e_operator_signed.sort_values(
                ["policy", "dataset", "route", "scale_num", "case_id", "source_name", "combo", "op_type"]
            ).drop(columns=["scale_num"]).reset_index(drop=True)
            e2e_operator_signed.to_csv(artifact_root / "e2e_operator_signed_error_by_combo.csv", index=False)
            e2e_tail_attribution = _e2e_tail_operator_attribution(e2e_operator_signed, e2e_combo)
            if not e2e_tail_attribution.empty:
                e2e_tail_attribution["scale_num"] = e2e_tail_attribution["scale"].astype(str).map(
                    lambda value: float(str(value).rstrip("%"))
                )
                e2e_tail_attribution = e2e_tail_attribution.sort_values(
                    ["policy", "dataset", "route", "scale_num", "tail_abs_error_share"],
                    ascending=[True, True, True, True, False],
                ).drop(columns=["scale_num"]).reset_index(drop=True)
                e2e_tail_attribution.to_csv(artifact_root / "e2e_tail_operator_attribution.csv", index=False)
            gather_correction_oracle = _e2e_gather_correction_oracle_rows(e2e_combo, e2e_operator_signed)
            if not gather_correction_oracle.empty:
                gather_correction_oracle["scale_num"] = gather_correction_oracle["scale"].astype(str).map(
                    lambda value: float(str(value).rstrip("%"))
                )
                gather_correction_oracle = gather_correction_oracle.sort_values(
                    ["policy", "dataset", "route", "scale_num", "correction_scope"]
                ).drop(columns=["scale_num"]).reset_index(drop=True)
                gather_correction_oracle.to_csv(artifact_root / "e2e_gather_correction_oracle.csv", index=False)
            v5_operator_delta = _route_pair_operator_delta_rows(
                e2e_operator_signed,
                e2e_tail_attribution if "e2e_tail_attribution" in locals() else pd.DataFrame(),
                base_route=WHITEBOX_EMBEDDING_MEM_RF_ROUTE,
                candidate_route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                base_label="v4",
                candidate_label="v5",
            )
            if not v5_operator_delta.empty:
                v4_vs_v5_with_ops = pd.concat([v4_vs_v5_delta, v5_operator_delta], ignore_index=True, sort=False)
                v4_vs_v5_with_ops.to_csv(artifact_root / "v4_vs_v5_e2e_delta.csv", index=False)
            compact_operator_delta = _route_pair_operator_delta_rows(
                e2e_operator_signed,
                e2e_tail_attribution,
                base_route=WHITEBOX_GATHER_SLICE_RF_ROUTE,
                candidate_route=WHITEBOX_GATHER_SLICE_COMPACT_RF_ROUTE,
                base_label="v5",
                candidate_label="v5_compact",
            )
            if not compact_operator_delta.empty:
                v5_vs_compact_with_ops = pd.concat(
                    [v5_vs_compact_delta, compact_operator_delta],
                    ignore_index=True,
                    sort=False,
                )
                v5_vs_compact_with_ops.to_csv(artifact_root / "v5_vs_v5_compact_delta.csv", index=False)
            indexed_copy_ablation = _indexed_copy_feature_ablation_rows(
                combined,
                e2e_summary,
                rf_pruned,
                e2e_tail_attribution,
            )
            if not indexed_copy_ablation.empty:
                indexed_copy_ablation["scale_num"] = indexed_copy_ablation["scale"].astype(str).map(
                    lambda value: float(str(value).rstrip("%"))
                )
                indexed_copy_ablation = indexed_copy_ablation.sort_values(
                    ["policy", "dataset", "scale_num", "route"]
                ).drop(columns=["scale_num"]).reset_index(drop=True)
                indexed_copy_ablation.to_csv(artifact_root / "indexed_copy_feature_ablation.csv", index=False)
    if e2e_pipeline_partial_combo_frames:
        e2e_partial_combo = pd.concat(e2e_pipeline_partial_combo_frames, ignore_index=True)
        e2e_partial_combo["scale_num"] = e2e_partial_combo["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        e2e_partial_combo = e2e_partial_combo.sort_values(["policy", "dataset", "route", "scale_num", "case_id", "batch_size", "num_indices_per_lookup"]).drop(columns=["scale_num"]).reset_index(drop=True)
        e2e_partial_combo.to_csv(artifact_root / "e2e_pipeline_partial_combo_metrics.csv", index=False)
    if e2e_pipeline_embedding_frames:
        e2e_embedding = pd.concat(e2e_pipeline_embedding_frames, ignore_index=True)
        e2e_embedding["scale_num"] = e2e_embedding["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        e2e_embedding = e2e_embedding.sort_values(["policy", "dataset", "route", "scale_num", "case_id", "batch_size", "num_indices_per_lookup"]).drop(columns=["scale_num"]).reset_index(drop=True)
        e2e_embedding.to_csv(artifact_root / "e2e_pipeline_embedding_order_check.csv", index=False)
    mainline_baseline = _mainline_baseline_summary(best, validation, v3_gap_report)
    if not mainline_baseline.empty:
        mainline_baseline.to_csv(artifact_root / "mainline_baseline_summary.csv", index=False)
    blocker_dashboard = _mainline_blocker_dashboard(breakdown, phase_error)
    if not blocker_dashboard.empty:
        blocker_dashboard.to_csv(artifact_root / "mainline_blocker_dashboard.csv", index=False)
    weighted_blocker_dashboard = _weighted_blocker_dashboard(breakdown, phase_error)
    if not weighted_blocker_dashboard.empty:
        weighted_blocker_dashboard.to_csv(artifact_root / "weighted_blocker_dashboard.csv", index=False)
    subsystem_target_scorecard = _subsystem_target_scorecard(breakdown, phase_error)
    if not subsystem_target_scorecard.empty:
        subsystem_target_scorecard.to_csv(artifact_root / "subsystem_target_scorecard.csv", index=False)
    if torch_training_summaries:
        _json_dump(artifact_root / "torch_training_summary.json", torch_training_summaries)
    if model_bundle_manifest_rows:
        model_bundle_manifest = pd.DataFrame(model_bundle_manifest_rows)
        model_bundle_manifest["scale_num"] = model_bundle_manifest["scale"].astype(str).map(
            lambda value: float(str(value).rstrip("%"))
        )
        model_bundle_manifest = model_bundle_manifest.sort_values(
            ["policy", "dataset", "route", "scale_num"]
        ).drop(columns=["scale_num"]).reset_index(drop=True)
        model_bundle_manifest.to_csv(artifact_root / "model_bundle_manifest.csv", index=False)
    _json_dump(artifact_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
