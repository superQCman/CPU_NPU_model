from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpu_perf_model import analytical_helpers
from cpu_perf_model.analytical import build_analytical_features
from cpu_perf_model.feature_contract import TARGET_COLUMN
from generalization_benchmarks.dataset3_context_proxy_study import (
    BASELINE_NUMERIC_FEATURES,
    DERIVED_CONTEXT_CATEGORICAL,
    DERIVED_CONTEXT_NUMERIC,
    RAW_CONTEXT_NUMERIC_FEATURES,
    _prepare_context_augmented_frame,
)


DEFAULTS_PATH = SCRIPT_DIR / "configs" / "defaults.json"
DEFAULTS = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))

DEFAULT_SCALES = [int(value) for value in DEFAULTS["default_scales"]]
GRAPH_SEMANTIC_COLUMNS = set(DEFAULTS["graph_semantic_columns"])
NO_ANALYTICAL_OPS = set(DEFAULTS["no_analytical_op_types"])
NON_STATIC_CONCURRENCY_FEATURES = {
    "local_ctx_same_op_overlap_ratio_mean",
    "local_ctx_overlap_ratio_mean",
    "local_ctx_cross_task_overlap_ratio_mean",
    "local_ctx_mean_other_active_mean",
    "local_ctx_mean_other_tasks_mean",
    "combo_task_parallel_fraction",
    "combo_task_weighted_mean_parallel_concurrency",
    "combo_op_parallel_fraction",
    "combo_op_weighted_mean_parallel_concurrency",
    "comp_feat_pressure_threads",
    "comp_feat_pressure_ws_to_l2_ratio",
    "comp_feat_pressure_ws_to_l3_ratio",
}
TREE_PARAMS = dict(DEFAULTS["tree_params"])
CALIBRATOR_FAMILY_SPECS = dict(DEFAULTS.get("calibrator_families", {}))
CALIBRATOR_SELECTION = dict(DEFAULTS.get("calibrator_selection", {}))
DEFAULT_CALIBRATOR_FAMILY = str(CALIBRATOR_SELECTION.get("default_family", "dt"))
SMALL_SCALE_FAMILY_SEARCH_MAX = int(CALIBRATOR_SELECTION.get("small_scale_search_max_scale", 5))
SMALL_SCALE_FAMILY_CANDIDATES = [str(value) for value in CALIBRATOR_SELECTION.get("candidate_families", ["dt", "et", "rf"])]
ANALYTICAL_PER_OP_FAMILY_SEARCH_MAX = int(CALIBRATOR_SELECTION.get("analytical_per_op_small_scale_search_max_scale", SMALL_SCALE_FAMILY_SEARCH_MAX))
ANALYTICAL_PER_OP_CANDIDATES = [
    str(value) for value in CALIBRATOR_SELECTION.get("analytical_per_op_candidate_families", SMALL_SCALE_FAMILY_CANDIDATES)
]
SAMPLER_CONFIG = dict(DEFAULTS.get("sampler", {}))
SAMPLER_SEARCH_MAX_SCALE = int(SAMPLER_CONFIG.get("search_max_scale", 5))
SAMPLER_CANDIDATE_COUNT = int(SAMPLER_CONFIG.get("candidate_count", 10))
SAMPLER_CANDIDATE_SEED_STRIDE = int(SAMPLER_CONFIG.get("candidate_seed_stride", 97))

TARGET_US_COLUMN = "target_us"
SOURCE_BASE_PRED_COLUMN = "source_base_pred_us"
SOURCE_BASE_LOG_COLUMN = "source_base_log1p_us"
DEFAULT_EXPERIMENTS = ("compare_strict", "exp1_optimal", "exp2_warm_refit", "exp3_frozen_analytical")
SEMANTIC_FREE_CATEGORICAL = ["op_type"] + list(DERIVED_CONTEXT_CATEGORICAL[1:])
SEMANTIC_FREE_NUMERIC = list(
    dict.fromkeys(
        column
        for column in (list(BASELINE_NUMERIC_FEATURES) + list(RAW_CONTEXT_NUMERIC_FEATURES) + list(DERIVED_CONTEXT_NUMERIC))
        if column not in NON_STATIC_CONCURRENCY_FEATURES
    )
)
PER_OP_COMMON_CATEGORICAL = list(DERIVED_CONTEXT_CATEGORICAL[1:])
PER_OP_COMMON_NUMERIC = [
    "batch_size",
    "num_threads",
    "inter_threads",
    "output_size",
    "activation_size",
    "parameter_size",
    "feat_io_bytes_sum",
    "ctx_group_len",
    "ctx_pos",
    "ctx_pos_frac",
    "ctx_remaining_ops",
    "ctx_prev_output_size",
    "ctx_prev_activation_size",
    "ctx_prev_feat_io_bytes_sum",
    "ctx_prev_num_threads",
    "ctx_next_output_size",
    "ctx_next_activation_size",
    "ctx_next_feat_io_bytes_sum",
    "ctx_next_num_threads",
    "ctx_prev_same_numel",
    "ctx_next_same_numel",
    "ctx_prev_is_heavy",
    "ctx_next_is_heavy",
    "ctx_prev_is_meta",
    "ctx_next_is_meta",
]
PER_OP_NUMERIC_ADDITIONS: dict[str, list[str]] = {
    "Gather": [
        "num_indices_per_lookup",
        "feat_lookup_count",
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ctx_prev_feat_lookup_count",
        "ctx_next_feat_lookup_count",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "Gemm": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
        "ana_calib_total_us",
        "ana_calib_compute_us",
        "ctx_prev_feat_gemm_mac_count",
        "ctx_next_feat_gemm_mac_count",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "MatMul": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
        "ana_calib_total_us",
        "ana_calib_compute_us",
        "ctx_prev_feat_gemm_mac_count",
        "ctx_next_feat_gemm_mac_count",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "ReduceSum": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "feat_reduction_axes_count",
        "feat_reduction_work_items",
        "feat_reduction_axes_product",
        "feat_reduction_input_rank",
        "feat_reduction_output_rank",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ana_calib_compute_us",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "Concat": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "Transpose": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "Add": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "Mul": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "Relu": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "Sigmoid": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ana_calib_compute_us",
        "ctx_prev_ana_calib_total_us",
        "ctx_next_ana_calib_total_us",
    ],
    "Reshape": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
    ],
    "Flatten": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
    ],
    "Unsqueeze": [
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
    ],
    "Shape": [
        "feat_output_input_bytes_ratio",
    ],
}
SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE = "pure_semantic_free_shape_tree_plus_problem_lookup"
EXP1_BASE_ROUTE = "pure_semantic_free_optimal_analytical_plus_calibrator"
EXP2_BASE_ROUTE = "pure_semantic_free_warm_refit_analytical_plus_calibrator"
EXP3_BASE_ROUTE = "pure_semantic_free_frozen_analytical_plus_calibrator"
SHAPE_TREE_PLUS_LOOKUP_META_OPS = ["Reshape", "Flatten", "Unsqueeze"]
SHAPE_TREE_PLUS_LOOKUP_ANALYTICAL_OPS = ["Concat", "Transpose", "Sigmoid"]
SHAPE_TREE_PLUS_LOOKUP_KEY = ["op_type", "input_type_shape", "output_type_shape", "output_size", "activation_size"]
SHAPE_TREE_SPECIALIZED_SPEC = {
    "categorical": ["input_type_shape", "output_type_shape"],
    "numeric": [
        "batch_size",
        "num_indices_per_lookup",
        "activation_size",
        "output_size",
        "ctx_pos",
        "ctx_pos_frac",
        "ctx_remaining_ops",
    ],
}
REFERENCE_COMPARE_SUMMARY = (
    PROJECT_ROOT / "artifacts" / "latest" / "pure_semantic_free_frozen_analytical_vs_direct_tree_same_scale_s920" / "summary.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-hardware Analytical + calibrator experiment runner.")
    parser.add_argument(
        "--experiment",
        default="all",
        choices=(*DEFAULT_EXPERIMENTS, "all"),
        help="Experiment to run.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(SCRIPT_DIR / "artifacts" / "latest"),
        help="Output root for experiment artifacts.",
    )
    parser.add_argument(
        "--scales",
        default=",".join(str(value) for value in DEFAULT_SCALES),
        help="Comma-separated fit scales in percentage points, e.g. 1,5,10.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--source-dataset-root",
        default=str(SCRIPT_DIR / DEFAULTS["default_assets"]["source_dataset"]),
        help="Source dataset root.",
    )
    parser.add_argument(
        "--target-dataset-root",
        default=str(SCRIPT_DIR / DEFAULTS["default_assets"]["target_dataset"]),
        help="Target dataset root.",
    )
    parser.add_argument(
        "--source-hw-config",
        default=str(SCRIPT_DIR / DEFAULTS["default_assets"]["source_hw"]),
        help="Source hardware YAML.",
    )
    parser.add_argument(
        "--target-hw-config",
        default=str(SCRIPT_DIR / DEFAULTS["default_assets"]["target_hw"]),
        help="Target hardware YAML.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Coordinate-descent passes for Analytical parameter fitting.",
    )
    return parser.parse_args()


def _stable_int_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_scales(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        cleaned = part.strip().rstrip("%")
        if not cleaned:
            continue
        value = int(cleaned)
        if value <= 0 or value > 100:
            raise ValueError(f"Scale must be within (0, 100], got {value}")
        values.append(value)
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError("At least one positive scale is required.")
    return deduped


def _scale_label(scale: int) -> str:
    return f"{int(scale)}%"


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_tree_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copytree(src, dst)


def _copy_file_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _sync_default_assets(source_dataset_root: Path, target_dataset_root: Path, source_hw: Path, target_hw: Path) -> None:
    default_source_dataset = SCRIPT_DIR / DEFAULTS["default_assets"]["source_dataset"]
    default_target_dataset = SCRIPT_DIR / DEFAULTS["default_assets"]["target_dataset"]
    default_source_hw = SCRIPT_DIR / DEFAULTS["default_assets"]["source_hw"]
    default_target_hw = SCRIPT_DIR / DEFAULTS["default_assets"]["target_hw"]

    if source_dataset_root == default_source_dataset:
        _copy_tree_if_missing(PROJECT_ROOT / "data" / "dataset_3", source_dataset_root)
    if target_dataset_root == default_target_dataset:
        _copy_tree_if_missing(PROJECT_ROOT / "artifacts" / "dataset_all_78910_2_nodrop", target_dataset_root)
    if source_hw == default_source_hw:
        _copy_file_if_missing(PROJECT_ROOT / "cpu_perf_model" / "hardware_profile" / "kunpeng920_single_numa.yaml", source_hw)
    if target_hw == default_target_hw:
        _copy_file_if_missing(PROJECT_ROOT / "cpu_perf_model" / "hardware_profile" / "s920x20_single_numa.yaml", target_hw)


def _asset_manifest(source_dataset_root: Path, target_dataset_root: Path, source_hw: Path, target_hw: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "module_root": str(SCRIPT_DIR),
        "assets": {},
    }
    entries = {
        "source_dataset": source_dataset_root,
        "target_dataset": target_dataset_root,
        "source_hw_config": source_hw,
        "target_hw_config": target_hw,
    }
    for name, root in entries.items():
        if root.is_dir():
            files = []
            total_bytes = 0
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                size = int(path.stat().st_size)
                total_bytes += size
                files.append(
                    {
                        "relative_path": str(path.relative_to(root)),
                        "size_bytes": size,
                        "sha256": _sha256(path),
                    }
                )
            payload["assets"][name] = {
                "path": str(root),
                "type": "directory",
                "file_count": len(files),
                "total_bytes": total_bytes,
                "files": files,
            }
        else:
            payload["assets"][name] = {
                "path": str(root),
                "type": "file",
                "size_bytes": int(root.stat().st_size),
                "sha256": _sha256(root),
            }
    return payload


def _write_asset_manifest(source_dataset_root: Path, target_dataset_root: Path, source_hw: Path, target_hw: Path) -> Path:
    manifest_path = SCRIPT_DIR / "asset_manifest.json"
    _json_dump(manifest_path, _asset_manifest(source_dataset_root, target_dataset_root, source_hw, target_hw))
    return manifest_path


def _load_raw_splits(dataset_root: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        csv_path = dataset_root / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")
        frame = pd.read_csv(csv_path, low_memory=False)
        if TARGET_COLUMN not in frame.columns:
            raise KeyError(f"{csv_path} is missing {TARGET_COLUMN}")
        frame[TARGET_US_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").fillna(0.0)
        frame["row_uid"] = frame["row_uid"].astype(str)
        splits[split] = frame
    return splits


def _fit_pool(raw_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat([raw_splits["train"], raw_splits["val"]], ignore_index=True)
    frame = frame.drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
    return frame


def _scale_to_row_count(total_rows: int, scale: int) -> int:
    return int(round(int(total_rows) * float(scale) / 100.0))


def _stratified_target_counts(frame: pd.DataFrame, target_rows: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    total_rows = max(int(len(frame)), 1)
    for op_type, group in frame.groupby("op_type", sort=True):
        count = int(round(int(len(group)) * float(target_rows) / float(total_rows)))
        counts[str(op_type)] = min(len(group), max(count, 1))
    current = int(sum(counts.values()))
    groups = {str(op_type): group for op_type, group in frame.groupby("op_type", sort=True)}
    ordered = sorted(groups, key=lambda name: len(groups[name]), reverse=True)
    while current < int(target_rows):
        progressed = False
        for op_type in ordered:
            if counts[op_type] >= len(groups[op_type]):
                continue
            counts[op_type] += 1
            current += 1
            progressed = True
            if current >= int(target_rows):
                break
        if not progressed:
            break
    while current > int(target_rows):
        progressed = False
        for op_type in reversed(ordered):
            if counts[op_type] <= 1:
                continue
            counts[op_type] -= 1
            current -= 1
            progressed = True
            if current <= int(target_rows):
                break
        if not progressed:
            break
    return counts


def _sample_fit_subset_rows(frame: pd.DataFrame, *, target_rows: int, seed: int, salt: int = 0) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    counts = _stratified_target_counts(frame, target_rows)
    for op_type, group in frame.groupby("op_type", sort=True):
        count = counts[str(op_type)]
        rng = np.random.default_rng(seed + _stable_int_hash(str(op_type)) + int(salt))
        if count >= len(group):
            sampled = group.copy()
        else:
            indices = rng.choice(group.index.to_numpy(dtype=int), size=count, replace=False)
            sampled = group.loc[np.sort(indices)].copy()
        parts.append(sampled)
    sampled_all = pd.concat(parts, ignore_index=True)
    sampled_all = sampled_all.drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
    return sampled_all


def _sample_fit_subset(fit_pool: pd.DataFrame, scale: int, seed: int) -> pd.DataFrame:
    return _sample_fit_subset_rows(
        fit_pool,
        target_rows=_scale_to_row_count(len(fit_pool), scale),
        seed=seed,
        salt=int(scale) * 17,
    )


def _write_sample_rows(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[["row_uid", "op_type"]].copy().to_csv(path, index=False)


def _assert_no_leak(sampled: pd.DataFrame, test_raw: pd.DataFrame) -> None:
    overlap = set(sampled["row_uid"].astype(str)).intersection(set(test_raw["row_uid"].astype(str)))
    if overlap:
        raise AssertionError(f"Train/test leakage detected: {len(overlap)} overlapping row_uids")


def _build_analytical_params(
    raw_frame: pd.DataFrame,
    hw_config: Path,
    *,
    passes: int,
    init_params: dict[str, float] | None = None,
) -> dict[str, float]:
    rebuilt = build_analytical_features.rebuild_local_features(raw_frame, hardware_profile_path=hw_config)
    heavy_prepared = build_analytical_features.prepare_heavy_prediction_frame(rebuilt)
    if init_params is None:
        return analytical_helpers.calibrate_params(heavy_prepared, passes=passes)
    return _warm_start_calibrate_params(heavy_prepared, init_params=init_params, passes=passes)


def _warm_start_calibrate_params(
    heavy_prepared: pd.DataFrame,
    *,
    init_params: dict[str, float],
    passes: int,
) -> dict[str, float]:
    params = analytical_helpers.default_params()
    params.update({key: float(value) for key, value in init_params.items() if key in params})

    def copy_objective(trial: dict[str, float]) -> float:
        scores = [
            analytical_helpers.family_mape(heavy_prepared[heavy_prepared["family"].isin(analytical_helpers.COPY_FAMILIES)], family, trial)
            for family in analytical_helpers.COPY_FAMILIES
            if not heavy_prepared[heavy_prepared["family"] == family].empty
        ]
        scores = [score for score in scores if np.isfinite(score)]
        return float(np.mean(scores)) if scores else float("nan")

    def family_objective(family: str):
        return lambda trial: analytical_helpers.family_mape(heavy_prepared, family, trial)

    for _ in range(max(int(passes), 1)):
        before = dict(params)
        params = analytical_helpers.coordinate_search(["rho_copy_inf", "tau_copy_start"], copy_objective, params, passes=1)
        params = analytical_helpers.coordinate_search(["tau_dispatch"], family_objective("Concat"), params, passes=1)
        params = analytical_helpers.coordinate_search(["kappa_reduce", "tau_reduce_start"], family_objective("ReduceSum"), params, passes=1)
        params = analytical_helpers.coordinate_search(["rho_gather_inf", "tau_gather_row_start", "m_gather"], family_objective("Gather"), params, passes=1)
        params = analytical_helpers.coordinate_search(["rho_fma_inf", "M50", "N50", "K50"], family_objective("Gemm"), params, passes=1)
        params = analytical_helpers.coordinate_search(["occ_ref", "rho_tiny_inf", "K50_tiny", "tau_micro"], family_objective("MatMul"), params, passes=1)
        params = analytical_helpers.coordinate_search(["m_stride", "eta_stride"], family_objective("Transpose"), params, passes=1)
        params = analytical_helpers.coordinate_search(["rho_relu_inf", "tau_relu_start"], family_objective("Relu"), params, passes=1)
        params = analytical_helpers.coordinate_search(["rho_add_inf", "tau_add_start", "tau_add"], family_objective("Add"), params, passes=1)
        params = analytical_helpers.coordinate_search(["rho_mul_inf", "tau_mul_start", "tau_mul"], family_objective("Mul"), params, passes=1)
        params = analytical_helpers.coordinate_search(
            ["rho_sigmoid_inf", "tau_sigmoid_start", "tau_sigmoid", "rho_sigmoid_compute"],
            family_objective("Sigmoid"),
            params,
            passes=1,
        )
        if params == before:
            break
    return params


def _prepare_strict_semantic_free_frame(raw_frame: pd.DataFrame, hw_config: Path, params: dict[str, float]) -> pd.DataFrame:
    rebuilt = build_analytical_features.rebuild_local_features(raw_frame, hardware_profile_path=hw_config)
    heavy_prepared = build_analytical_features.prepare_heavy_prediction_frame(rebuilt)
    feature_df = build_analytical_features.add_calibrated_analytical_columns(rebuilt, heavy_prepared, params)
    feature_df[TARGET_US_COLUMN] = pd.to_numeric(feature_df[TARGET_COLUMN], errors="coerce").fillna(0.0)
    prepared = _prepare_context_augmented_frame(feature_df)
    prepared = prepared.drop(columns=[column for column in GRAPH_SEMANTIC_COLUMNS if column in prepared.columns], errors="ignore")
    for column in GRAPH_SEMANTIC_COLUMNS:
        if column in prepared.columns:
            raise AssertionError(f"Graph semantic column still present after strict filtering: {column}")
    prepared["row_uid"] = prepared["row_uid"].astype(str)
    prepared[TARGET_US_COLUMN] = pd.to_numeric(prepared[TARGET_US_COLUMN], errors="coerce").fillna(0.0)
    prepared["ana_calib_total_us"] = pd.to_numeric(prepared["ana_calib_total_us"], errors="coerce").fillna(0.0)
    return prepared


def _filter_prepared_rows(prepared_full: pd.DataFrame, sampled_raw: pd.DataFrame) -> pd.DataFrame:
    sampled_ids = set(sampled_raw["row_uid"].astype(str))
    filtered = prepared_full[prepared_full["row_uid"].astype(str).isin(sampled_ids)].copy()
    filtered = filtered.drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
    if len(filtered) != int(sampled_raw["row_uid"].astype(str).nunique()):
        raise AssertionError("Prepared subset row count does not match sampled raw subset")
    return filtered


def _load_cached_transfer_small_scale_plan(
    artifact_root: Path,
    *,
    fit_raw: pd.DataFrame,
    small_scales: list[int],
    seed: int,
    target_hw: Path,
) -> dict[int, pd.DataFrame]:
    if not small_scales:
        return {}
    plan_root = artifact_root / "shared_sampling"
    manifest_path = plan_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}
    if int(manifest.get("seed", -1)) != int(seed):
        return {}
    if str(manifest.get("target_hw_config", "")) != str(target_hw):
        return {}
    cached_scales = {int(scale) for scale in manifest.get("scales", [])}
    if not set(int(scale) for scale in small_scales).issubset(cached_scales):
        return {}

    selected: dict[int, pd.DataFrame] = {}
    fit_with_uid = fit_raw.copy()
    fit_with_uid["row_uid"] = fit_with_uid["row_uid"].astype(str)
    for scale in small_scales:
        selected_path = plan_root / f"selected_rows_{_scale_label(scale).replace('%', 'pct')}.csv"
        if not selected_path.exists():
            return {}
        sampled_ids = pd.read_csv(selected_path, usecols=["row_uid"])
        sampled_ids["row_uid"] = sampled_ids["row_uid"].astype(str)
        sampled = fit_with_uid[fit_with_uid["row_uid"].isin(set(sampled_ids["row_uid"]))].copy()
        sampled = sampled.drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
        if len(sampled) != int(sampled_ids["row_uid"].nunique()):
            return {}
        selected[int(scale)] = sampled
    return selected


def _build_transfer_small_scale_plan(
    artifact_root: Path,
    *,
    target_splits: dict[str, pd.DataFrame],
    target_hw: Path,
    source_params: dict[str, float],
    scales: list[int],
    seed: int,
) -> dict[int, pd.DataFrame]:
    small_scales = [int(scale) for scale in scales if int(scale) <= SAMPLER_SEARCH_MAX_SCALE]
    if not small_scales:
        return {}

    fit_raw = _fit_pool(target_splits)
    cached = _load_cached_transfer_small_scale_plan(
        artifact_root,
        fit_raw=fit_raw,
        small_scales=small_scales,
        seed=seed,
        target_hw=target_hw,
    )
    if cached:
        return cached

    plan_root = artifact_root / "shared_sampling"
    plan_root.mkdir(parents=True, exist_ok=True)
    candidate_pool_raw = target_splits["train"].copy()
    proxy_row_ids = set(target_splits["val"]["row_uid"].astype(str))
    fit_prepared_full = _prepare_strict_semantic_free_frame(fit_raw, target_hw, source_params)
    candidate_prepared_full = _filter_prepared_rows(fit_prepared_full, candidate_pool_raw)
    proxy_prepared = fit_prepared_full[fit_prepared_full["row_uid"].astype(str).isin(proxy_row_ids)].copy().reset_index(drop=True)

    reports: list[dict[str, Any]] = []
    selected: dict[int, pd.DataFrame] = {}
    for scale in small_scales:
        target_rows = _scale_to_row_count(len(fit_raw), scale)
        best_key: tuple[float, float, float] | None = None
        best_sampled: pd.DataFrame | None = None
        best_candidate_idx: int | None = None
        for candidate_idx in range(SAMPLER_CANDIDATE_COUNT):
            sampled = _sample_fit_subset_rows(
                candidate_pool_raw,
                target_rows=target_rows,
                seed=seed,
                salt=int(scale) * 1000 + candidate_idx * SAMPLER_CANDIDATE_SEED_STRIDE,
            )
            fit_prepared = _filter_prepared_rows(candidate_prepared_full, sampled)
            bundle = _fit_calibrator_bundle(fit_prepared, mode="analytical")
            proxy_pred = _predict_calibrator_bundle(bundle, proxy_prepared)
            proxy_metrics = _metrics(proxy_prepared[TARGET_US_COLUMN], proxy_pred)
            key = (proxy_metrics["p90"], proxy_metrics["mape"], proxy_metrics["median"])
            reports.append(
                {
                    "scale": _scale_label(scale),
                    "candidate_idx": int(candidate_idx),
                    "fit_rows": int(len(sampled)),
                    "proxy_rows": int(len(proxy_prepared)),
                    "proxy_p90": proxy_metrics["p90"],
                    "proxy_mape": proxy_metrics["mape"],
                    "proxy_median": proxy_metrics["median"],
                    "selected": False,
                }
            )
            if best_key is None or key < best_key:
                best_key = key
                best_sampled = sampled
                best_candidate_idx = int(candidate_idx)
        if best_sampled is None:
            raise AssertionError(f"Failed to build sampling plan for scale {scale}")
        selected[int(scale)] = best_sampled.reset_index(drop=True)
        for row in reports:
            if row["scale"] == _scale_label(scale):
                row["selected"] = int(row["candidate_idx"]) == int(best_candidate_idx)
        _write_sample_rows(plan_root / f"selected_rows_{_scale_label(scale).replace('%', 'pct')}.csv", selected[int(scale)])

    report_df = pd.DataFrame(reports)
    if not report_df.empty:
        report_df["scale_num"] = report_df["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
        report_df = report_df.sort_values(["scale_num", "candidate_idx"]).drop(columns=["scale_num"]).reset_index(drop=True)
        report_df.to_csv(plan_root / "candidate_search_summary.csv", index=False)
    _write_experiment_manifest(
        plan_root / "manifest.json",
        {
            "experiment": "shared_small_scale_sampling_plan",
            "scales": small_scales,
            "seed": seed,
            "search_max_scale": SAMPLER_SEARCH_MAX_SCALE,
            "candidate_count": SAMPLER_CANDIDATE_COUNT,
            "candidate_seed_stride": SAMPLER_CANDIDATE_SEED_STRIDE,
            "candidate_pool_split": "train",
            "proxy_split": "val",
            "target_hw_config": str(target_hw),
        },
    )
    return selected


def _metrics(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    if actual_arr.size == 0:
        return {"mape": float("nan"), "p90": float("nan"), "median": float("nan")}
    denom = np.clip(np.abs(actual_arr), a_min=1e-9, a_max=None)
    ape = np.abs(pred_arr - actual_arr) / denom
    return {
        "mape": float(np.mean(ape)),
        "p90": float(np.quantile(ape, 0.9)),
        "median": float(np.median(ape)),
    }


def _metric_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (float(metrics["p90"]), float(metrics["mape"]), float(metrics["median"]))


def _analytical_active_mask(prepared_frame: pd.DataFrame) -> pd.Series:
    return ~prepared_frame["op_type"].astype(str).isin(NO_ANALYTICAL_OPS)


def _prepared_metrics(
    prepared_frame: pd.DataFrame,
    pred: pd.Series | np.ndarray,
    *,
    scope: str = "all_test_rows",
) -> dict[str, float | int | str]:
    if scope == "all_test_rows":
        mask = pd.Series(True, index=prepared_frame.index)
    elif scope == "analytical_active_only":
        mask = _analytical_active_mask(prepared_frame)
    else:
        raise ValueError(f"Unsupported metric scope: {scope}")
    metrics = _metrics(
        prepared_frame.loc[mask, TARGET_US_COLUMN],
        np.asarray(pred, dtype=float)[mask.to_numpy(dtype=bool)],
    )
    return {
        **metrics,
        "eval_rows": int(mask.sum()),
        "eval_scope": scope,
    }


def _analytical_only_metrics(prepared_frame: pd.DataFrame) -> dict[str, float | int | str]:
    return _prepared_metrics(
        prepared_frame,
        prepared_frame["ana_calib_total_us"],
        scope="analytical_active_only",
    )


def _static_shape_lookup_key(frame: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    return frame[key_columns].fillna("NA").astype(str).agg("||".join, axis=1)


def _apply_static_problem_lookup_overrides(
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    base_pred: np.ndarray,
    *,
    op_types: list[str],
    key_columns: list[str],
) -> np.ndarray:
    pred = np.asarray(base_pred, dtype=float).copy()
    available_key_columns = [column for column in key_columns if column in fit_frame.columns and column in eval_frame.columns]
    if not available_key_columns:
        return pred

    for op_type in op_types:
        fit_sub = fit_frame[fit_frame["op_type"].astype(str) == str(op_type)].copy()
        eval_sub = eval_frame[eval_frame["op_type"].astype(str) == str(op_type)].copy()
        if fit_sub.empty or eval_sub.empty:
            continue
        fit_key = _static_shape_lookup_key(fit_sub, available_key_columns)
        eval_key = _static_shape_lookup_key(eval_sub, available_key_columns)
        if str(op_type) in NO_ANALYTICAL_OPS:
            lookup = fit_sub.assign(_key=fit_key).groupby("_key")[TARGET_US_COLUMN].median()
            values = eval_key.map(lookup)
            mask = eval_sub.index[values.notna()]
            pred[mask] = values.loc[values.notna()].to_numpy(dtype=float)
        else:
            base = np.clip(
                pd.to_numeric(fit_sub["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                a_min=1e-9,
                a_max=None,
            )
            ratio = pd.to_numeric(fit_sub[TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float) / base
            lookup = fit_sub.assign(_key=fit_key, _ratio=ratio).groupby("_key")["_ratio"].median()
            values = eval_key.map(lookup)
            mask = eval_sub.index[values.notna()]
            analytical_base = np.clip(
                pd.to_numeric(eval_frame.loc[mask, "ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                a_min=1e-9,
                a_max=None,
            )
            pred[mask] = analytical_base * values.loc[values.notna()].to_numpy(dtype=float)
    return np.clip(pred, a_min=0.0, a_max=None)


def _specialized_tree_target_transform(frame: pd.DataFrame, op_type: str) -> np.ndarray:
    actual = pd.to_numeric(frame[TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if str(op_type) in NO_ANALYTICAL_OPS:
        return np.log1p(np.clip(actual, a_min=0.0, a_max=None))
    analytical_base = np.clip(
        pd.to_numeric(frame["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        a_min=1e-9,
        a_max=None,
    )
    return np.log(np.clip(actual, a_min=1e-9, a_max=None) / analytical_base)


def _specialized_tree_inverse_transform(frame: pd.DataFrame, op_type: str, pred_target: np.ndarray) -> np.ndarray:
    if str(op_type) in NO_ANALYTICAL_OPS:
        return np.expm1(pred_target)
    analytical_base = np.clip(
        pd.to_numeric(frame["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        a_min=1e-9,
        a_max=None,
    )
    return analytical_base * np.exp(pred_target)


def _fit_specialized_static_tree_prediction(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    op_type: str,
    feature_spec: dict[str, list[str]],
) -> np.ndarray:
    categorical = [column for column in feature_spec.get("categorical", []) if column in train_frame.columns and column in eval_frame.columns]
    numeric = [column for column in feature_spec.get("numeric", []) if column in train_frame.columns and column in eval_frame.columns]
    if not categorical and not numeric:
        actual = pd.to_numeric(train_frame[TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.full(len(eval_frame), float(np.median(actual)) if len(actual) else 0.0, dtype=float)

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    feature_columns = categorical + numeric
    model.fit(train_frame[feature_columns], _specialized_tree_target_transform(train_frame, op_type))
    pred_target = model.predict(eval_frame[feature_columns])
    return np.clip(_specialized_tree_inverse_transform(eval_frame, op_type, pred_target), a_min=0.0, a_max=None)


def _predict_shape_tree_plus_problem_lookup(
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    base_pred: np.ndarray,
) -> np.ndarray:
    pred = _apply_static_problem_lookup_overrides(
        fit_frame,
        eval_frame,
        base_pred,
        op_types=SHAPE_TREE_PLUS_LOOKUP_META_OPS + SHAPE_TREE_PLUS_LOOKUP_ANALYTICAL_OPS,
        key_columns=SHAPE_TREE_PLUS_LOOKUP_KEY,
    )
    shape_fit = fit_frame[fit_frame["op_type"].astype(str) == "Shape"].copy()
    shape_eval = eval_frame[eval_frame["op_type"].astype(str) == "Shape"].copy()
    if not shape_fit.empty and not shape_eval.empty:
        shape_pred = _fit_specialized_static_tree_prediction(
            shape_fit,
            shape_eval,
            op_type="Shape",
            feature_spec=SHAPE_TREE_SPECIALIZED_SPEC,
        )
        pred[shape_eval.index] = shape_pred
    return np.clip(pred, a_min=0.0, a_max=None)


def _shape_tree_plus_selected_family(base_family: Any) -> str:
    family = str(base_family) if pd.notna(base_family) else "unknown"
    return f"{family}+shape_tree_rf_plus_lookup"


def _resolve_feature_columns(
    frame: pd.DataFrame,
    *,
    categorical_candidates: list[str],
    numeric_candidates: list[str],
    include_analytical: bool,
    include_source_base: bool = False,
) -> tuple[list[str], list[str]]:
    categorical = [column for column in categorical_candidates if column in frame.columns and column not in GRAPH_SEMANTIC_COLUMNS]
    numeric: list[str] = []
    for column in numeric_candidates:
        if column not in frame.columns or column in GRAPH_SEMANTIC_COLUMNS:
            continue
        if not include_analytical and "ana_" in column:
            continue
        numeric.append(column)
    if include_source_base:
        for column in (SOURCE_BASE_PRED_COLUMN, SOURCE_BASE_LOG_COLUMN):
            if column in frame.columns:
                numeric.append(column)
    return categorical, list(dict.fromkeys(numeric))


def _semantic_free_feature_columns(frame: pd.DataFrame, *, include_analytical: bool, include_source_base: bool = False) -> tuple[list[str], list[str]]:
    return _resolve_feature_columns(
        frame,
        categorical_candidates=SEMANTIC_FREE_CATEGORICAL,
        numeric_candidates=SEMANTIC_FREE_NUMERIC,
        include_analytical=include_analytical,
        include_source_base=include_source_base,
    )


def _per_op_feature_candidates(op_type: str) -> tuple[list[str], list[str]]:
    numeric_candidates = list(PER_OP_COMMON_NUMERIC)
    numeric_candidates.extend(PER_OP_NUMERIC_ADDITIONS.get(str(op_type), []))
    return list(PER_OP_COMMON_CATEGORICAL), list(dict.fromkeys(numeric_candidates))


def _semantic_free_per_op_feature_columns(
    frame: pd.DataFrame,
    *,
    include_analytical: bool,
    include_source_base: bool = False,
) -> tuple[dict[str, dict[str, list[str]]], list[str], list[str]]:
    per_op_feature_columns: dict[str, dict[str, list[str]]] = {}
    categorical_union: list[str] = []
    numeric_union: list[str] = []
    for op_type in sorted(pd.Series(frame["op_type"], copy=False).dropna().astype(str).unique().tolist()):
        categorical_candidates, numeric_candidates = _per_op_feature_candidates(str(op_type))
        categorical, numeric = _resolve_feature_columns(
            frame,
            categorical_candidates=categorical_candidates,
            numeric_candidates=numeric_candidates,
            include_analytical=include_analytical,
            include_source_base=include_source_base,
        )
        per_op_feature_columns[str(op_type)] = {
            "categorical": categorical,
            "numeric": numeric,
        }
        categorical_union.extend(categorical)
        numeric_union.extend(numeric)
    return (
        per_op_feature_columns,
        list(dict.fromkeys(categorical_union)),
        list(dict.fromkeys(numeric_union)),
    )


def _fit_category_maps(frame: pd.DataFrame, categorical: list[str]) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for column in categorical:
        values = pd.Series(frame[column], copy=False).fillna("__MISSING__").astype(str)
        maps[column] = {value: index for index, value in enumerate(pd.Index(values.unique()))}
    return maps


def _encode_features(frame: pd.DataFrame, categorical: list[str], numeric: list[str], category_maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    encoded = pd.DataFrame(index=frame.index)
    for column in categorical:
        encoded[column] = (
            pd.Series(frame[column], copy=False)
            .fillna("__MISSING__")
            .astype(str)
            .map(category_maps[column])
            .fillna(-1)
            .astype(float)
        )
    for column in numeric:
        encoded[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    return encoded


def _fit_calibrator_bundle(
    fit_frame: pd.DataFrame,
    *,
    mode: str,
    base_prediction: np.ndarray | None = None,
    model_family: str = DEFAULT_CALIBRATOR_FAMILY,
) -> dict[str, Any]:
    family_spec = dict(CALIBRATOR_FAMILY_SPECS.get(model_family, {}))
    family_kind = str(family_spec.get("kind", model_family))
    family_params = dict(family_spec.get("params", TREE_PARAMS if family_kind == "dt" else {}))
    if family_kind == "dt":
        model_ctor = DecisionTreeRegressor
    elif family_kind == "et":
        model_ctor = ExtraTreesRegressor
    elif family_kind == "rf":
        model_ctor = RandomForestRegressor
    else:
        raise KeyError(f"Unsupported calibrator family: {model_family}")
    include_analytical = mode in {"analytical", "delta"}
    include_source_base = mode == "delta"
    fit_work = fit_frame.copy()
    if mode == "delta":
        if base_prediction is None:
            raise ValueError("base_prediction is required for delta mode")
        fit_work[SOURCE_BASE_PRED_COLUMN] = np.asarray(base_prediction, dtype=float)
        fit_work[SOURCE_BASE_LOG_COLUMN] = np.log1p(np.clip(fit_work[SOURCE_BASE_PRED_COLUMN].to_numpy(dtype=float), a_min=0.0, a_max=None))
    per_op_feature_columns, categorical, numeric = _semantic_free_per_op_feature_columns(
        fit_work,
        include_analytical=include_analytical,
        include_source_base=include_source_base,
    )
    category_maps = _fit_category_maps(fit_work, categorical)
    encoded = _encode_features(fit_work, categorical, numeric, category_maps)
    models: dict[str, Any] = {}
    target_modes: dict[str, str] = {}
    op_defaults: dict[str, float] = {}
    global_default_pred = float(np.median(pd.to_numeric(fit_work[TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)))

    for op_type, group in fit_work.groupby("op_type", sort=True):
        model = model_ctor(**family_params)
        group_idx = group.index
        actual = pd.to_numeric(group[TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        op_defaults[str(op_type)] = float(np.median(actual)) if len(actual) else 0.0
        if mode == "analytical" and str(op_type) not in NO_ANALYTICAL_OPS:
            base = np.clip(pd.to_numeric(group["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
            target = np.log(np.clip(actual, a_min=1e-6, a_max=None) / base)
            target_modes[str(op_type)] = "analytical_residual"
        elif mode == "delta":
            base = np.asarray(base_prediction, dtype=float)[group_idx]
            target = actual - base
            target_modes[str(op_type)] = "delta_us"
        else:
            target = np.log1p(np.clip(actual, a_min=0.0, a_max=None))
            target_modes[str(op_type)] = "direct_log"
        feature_spec = per_op_feature_columns.get(str(op_type), {"categorical": categorical, "numeric": numeric})
        feature_columns = feature_spec["categorical"] + feature_spec["numeric"]
        model.fit(encoded.loc[group_idx, feature_columns], target)
        models[str(op_type)] = model

    return {
        "mode": mode,
        "model_family": model_family,
        "model_kind": family_kind,
        "model_params": family_params,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "per_op_feature_columns": per_op_feature_columns,
        "category_maps": category_maps,
        "target_modes": target_modes,
        "models": models,
        "op_defaults": op_defaults,
        "global_default_pred_us": global_default_pred,
        "no_analytical_ops": sorted(NO_ANALYTICAL_OPS),
    }


def _predict_calibrator_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    base_prediction: np.ndarray | None = None,
) -> np.ndarray:
    work = frame.copy()
    if bundle["mode"] == "delta":
        if base_prediction is None:
            raise ValueError("base_prediction is required when predicting with delta bundle")
        work[SOURCE_BASE_PRED_COLUMN] = np.asarray(base_prediction, dtype=float)
        work[SOURCE_BASE_LOG_COLUMN] = np.log1p(np.clip(work[SOURCE_BASE_PRED_COLUMN].to_numpy(dtype=float), a_min=0.0, a_max=None))
    encoded = _encode_features(work, bundle["categorical_columns"], bundle["numeric_columns"], bundle["category_maps"])
    if bundle["mode"] == "delta":
        pred = np.asarray(base_prediction, dtype=float).copy()
    elif bundle["mode"] == "analytical":
        pred = np.array(
            pd.to_numeric(work["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            dtype=float,
            copy=True,
        )
    else:
        pred = np.full(len(work), float(bundle["global_default_pred_us"]), dtype=float)

    per_op_feature_columns = bundle.get("per_op_feature_columns", {})
    for op_type, model in bundle["models"].items():
        mask = work["op_type"].astype(str).eq(str(op_type)).to_numpy()
        if not mask.any():
            continue
        feature_spec = per_op_feature_columns.get(
            str(op_type),
            {"categorical": bundle["categorical_columns"], "numeric": bundle["numeric_columns"]},
        )
        feature_columns = feature_spec["categorical"] + feature_spec["numeric"]
        encoded_part = encoded.loc[mask, feature_columns]
        model_pred = model.predict(encoded_part)
        target_mode = bundle["target_modes"][str(op_type)]
        if target_mode == "analytical_residual":
            base = np.clip(pd.to_numeric(work.loc[mask, "ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
            pred[mask] = base * np.exp(model_pred)
        elif target_mode == "delta_us":
            pred[mask] = pred[mask] + model_pred
        else:
            pred[mask] = np.expm1(model_pred)
    return np.clip(pred, a_min=0.0, a_max=None)


def _save_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def _fit_selected_family_bundle(
    fit_frame: pd.DataFrame,
    proxy_frame: pd.DataFrame,
    *,
    mode: str,
    base_prediction_fit: np.ndarray | None = None,
    base_prediction_proxy: np.ndarray | None = None,
    candidate_families: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    best_bundle: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    best_family: str | None = None
    families = list(candidate_families or SMALL_SCALE_FAMILY_CANDIDATES)
    for family in families:
        bundle = _fit_calibrator_bundle(
            fit_frame,
            mode=mode,
            base_prediction=base_prediction_fit,
            model_family=family,
        )
        proxy_pred = _predict_calibrator_bundle(bundle, proxy_frame, base_prediction=base_prediction_proxy)
        proxy_metrics = _metrics(proxy_frame[TARGET_US_COLUMN], proxy_pred)
        reports.append(
            {
                "family": family,
                "proxy_mape": proxy_metrics["mape"],
                "proxy_p90": proxy_metrics["p90"],
                "proxy_median": proxy_metrics["median"],
                "selected": False,
            }
        )
        key = _metric_key(proxy_metrics)
        if best_key is None or key < best_key:
            best_key = key
            best_bundle = bundle
            best_family = family
    if best_bundle is None or best_family is None:
        raise AssertionError("Failed to select calibrator family")
    for report in reports:
        report["selected"] = report["family"] == best_family
    return best_bundle, {
        "selected_family": best_family,
        "candidates": reports,
    }


def _fit_per_op_selected_family_bundle(
    fit_frame: pd.DataFrame,
    proxy_frame: pd.DataFrame,
    *,
    mode: str,
    candidate_families: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode != "analytical":
        raise ValueError("Per-op family selection is only implemented for analytical mode")

    families = list(candidate_families or ANALYTICAL_PER_OP_CANDIDATES)
    include_analytical = True
    per_op_feature_columns, categorical, numeric = _semantic_free_per_op_feature_columns(
        fit_frame,
        include_analytical=include_analytical,
    )
    category_maps = _fit_category_maps(fit_frame, categorical)
    fit_encoded = _encode_features(fit_frame, categorical, numeric, category_maps)
    proxy_encoded = _encode_features(proxy_frame, categorical, numeric, category_maps)
    models: dict[str, Any] = {}
    target_modes: dict[str, str] = {}
    op_defaults: dict[str, float] = {}
    op_selected_families: dict[str, str] = {}
    reports: list[dict[str, Any]] = []
    global_default_pred = float(np.median(pd.to_numeric(fit_frame[TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)))

    for op_type, group in fit_frame.groupby("op_type", sort=True):
        op_key = str(op_type)
        group_idx = group.index
        actual = pd.to_numeric(group[TARGET_US_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        op_defaults[op_key] = float(np.median(actual)) if len(actual) else 0.0
        if op_key not in NO_ANALYTICAL_OPS:
            base = np.clip(pd.to_numeric(group["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), a_min=1e-6, a_max=None)
            target = np.log(np.clip(actual, a_min=1e-6, a_max=None) / base)
            target_mode = "analytical_residual"
        else:
            target = np.log1p(np.clip(actual, a_min=0.0, a_max=None))
            target_mode = "direct_log"

        proxy_mask = proxy_frame["op_type"].astype(str).eq(op_key).to_numpy()
        feature_spec = per_op_feature_columns.get(op_key, {"categorical": categorical, "numeric": numeric})
        feature_columns = feature_spec["categorical"] + feature_spec["numeric"]
        best_key: tuple[float, float, float] | None = None
        best_family: str | None = None
        best_model: Any | None = None
        for family in families:
            family_spec = dict(CALIBRATOR_FAMILY_SPECS.get(family, {}))
            family_kind = str(family_spec.get("kind", family))
            family_params = dict(family_spec.get("params", TREE_PARAMS if family_kind == "dt" else {}))
            if family_kind == "dt":
                model_ctor = DecisionTreeRegressor
            elif family_kind == "et":
                model_ctor = ExtraTreesRegressor
            elif family_kind == "rf":
                model_ctor = RandomForestRegressor
            else:
                raise KeyError(f"Unsupported calibrator family: {family}")
            model = model_ctor(**family_params)
            model.fit(fit_encoded.loc[group_idx, feature_columns], target)
            if proxy_mask.any():
                proxy_pred_raw = model.predict(proxy_encoded.loc[proxy_mask, feature_columns])
                if target_mode == "analytical_residual":
                    proxy_base = np.clip(
                        pd.to_numeric(proxy_frame.loc[proxy_mask, "ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                        a_min=1e-6,
                        a_max=None,
                    )
                    proxy_pred = proxy_base * np.exp(proxy_pred_raw)
                else:
                    proxy_pred = np.expm1(proxy_pred_raw)
                proxy_metrics = _metrics(proxy_frame.loc[proxy_mask, TARGET_US_COLUMN], proxy_pred)
            else:
                proxy_metrics = {"mape": float("inf"), "p90": float("inf"), "median": float("inf")}
            reports.append(
                {
                    "op_type": op_key,
                    "family": family,
                    "proxy_mape": proxy_metrics["mape"],
                    "proxy_p90": proxy_metrics["p90"],
                    "proxy_median": proxy_metrics["median"],
                    "selected": False,
                }
            )
            key = _metric_key(proxy_metrics)
            if best_key is None or key < best_key:
                best_key = key
                best_family = family
                best_model = model
        if best_model is None or best_family is None:
            raise AssertionError(f"Failed to select per-op family for op_type={op_key}")
        models[op_key] = best_model
        target_modes[op_key] = target_mode
        op_selected_families[op_key] = best_family
        for report in reports:
            if report["op_type"] == op_key and report["family"] == best_family:
                report["selected"] = True

    return {
        "mode": mode,
        "model_family": "per_op_mix",
        "model_kind": "per_op_mixed_family",
        "model_params": {"candidate_families": families},
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "per_op_feature_columns": per_op_feature_columns,
        "category_maps": category_maps,
        "target_modes": target_modes,
        "models": models,
        "op_defaults": op_defaults,
        "op_selected_families": op_selected_families,
        "global_default_pred_us": global_default_pred,
        "no_analytical_ops": sorted(NO_ANALYTICAL_OPS),
    }, {
        "selected_family": "per_op_mix",
        "candidates": reports,
        "op_selected_families": op_selected_families,
    }


def _plot_metric_curves(
    rows: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    series_by: str,
    title: str,
    percent_scale: bool = True,
    reference_lines: list[dict[str, Any]] | None = None,
    x_offsets: dict[str, float] | None = None,
    series_styles: dict[str, dict[str, Any]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_df = rows.copy()
    plot_df["scale_num"] = plot_df["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    xticks = sorted(plot_df["scale_num"].dropna().unique().tolist())
    for name, group in plot_df.groupby(series_by, sort=True):
        ordered = group.sort_values("scale_num")
        x_values = ordered["scale_num"].to_numpy(dtype=float)
        if x_offsets:
            x_values = x_values + float(x_offsets.get(str(name), 0.0))
        values = ordered[metric].to_numpy(dtype=float)
        if percent_scale:
            values = values * 100.0
        style = {"marker": "o", "linewidth": 2}
        if series_styles and str(name) in series_styles:
            style.update(series_styles[str(name)])
        ax.plot(x_values, values, label=str(name), **style)
    for ref in reference_lines or []:
        value = float(ref["value"]) * (100.0 if percent_scale else 1.0)
        ax.axhline(value, linestyle="--", linewidth=1.5, label=str(ref["label"]), color=ref.get("color"))
    ylabel = f"{metric.upper()} (%)" if percent_scale else metric.upper()
    if metric == "p90":
        ylabel = "P90 APE (%)"
    elif metric == "mape":
        ylabel = "MAPE (%)"
    elif metric == "median":
        ylabel = "Median APE (%)"
    ax.set_xlabel("Fit Scale (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xticks:
        ax.set_xticks(xticks)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_compare_rows(rows: pd.DataFrame, output_path: Path, title: str, reference: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True)
    plot_df = rows.copy()
    if "route" not in plot_df.columns:
        plot_df["route"] = "target_chain"
    plot_df["scale_num"] = plot_df["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    for metric, ax in (("mape", axes[0]), ("p90", axes[1])):
        for route, group in plot_df.groupby("route", sort=True):
            ordered = group.sort_values("scale_num")
            ax.plot(ordered["scale_num"], ordered[metric].to_numpy(dtype=float) * 100.0, marker="o", linewidth=2, label=str(route))
        ax.axhline(float(reference[metric]) * 100.0, linestyle="--", linewidth=1.5, color="#555555", label=f"source_reference_{metric}")
        ax.set_xlabel("Fit Scale (%)")
        ax.set_ylabel("MAPE (%)" if metric == "mape" else "P90 APE (%)")
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(2, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _compare_to_reference(summary: pd.DataFrame, output_path: Path) -> None:
    if not REFERENCE_COMPARE_SUMMARY.exists():
        return
    reference = pd.read_csv(REFERENCE_COMPARE_SUMMARY)
    left = summary.copy()
    left["route"] = left["route"].astype(str)
    left["scale"] = left["scale"].astype(str)
    right = reference.copy()
    right["route"] = right["route"].astype(str)
    right["scale"] = right["scale"].astype(str)
    merged = left.merge(right, on=["scale", "route"], suffixes=("_new", "_ref"))
    if merged.empty:
        return
    merged["mape_delta"] = merged["mape_new"] - merged["mape_ref"]
    merged["p90_delta"] = merged["p90_new"] - merged["p90_ref"]
    merged["median_delta"] = merged["median_new"] - merged["median_ref"]
    merged.to_csv(output_path, index=False)


def _write_experiment_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["module_root"] = str(SCRIPT_DIR)
    payload["defaults_path"] = str(DEFAULTS_PATH)
    _json_dump(path, payload)


def _save_scale_artifacts(
    scale_dir: Path,
    *,
    analytical_params: dict[str, float] | None = None,
    calibrator_bundle: dict[str, Any] | None = None,
    metrics_payload: dict[str, Any] | None = None,
) -> None:
    scale_dir.mkdir(parents=True, exist_ok=True)
    if analytical_params is not None:
        _json_dump(scale_dir / "analytical_params.json", analytical_params)
    if calibrator_bundle is not None:
        _save_bundle(scale_dir / "calibrator.joblib", calibrator_bundle)
    if metrics_payload is not None:
        _json_dump(scale_dir / "metrics.json", metrics_payload)


def _train_source_reference(source_splits: dict[str, pd.DataFrame], source_hw: Path, output_root: Path, passes: int) -> dict[str, Any]:
    source_fit_raw = _fit_pool(source_splits)
    source_test_raw = source_splits["test"]
    source_params = _build_analytical_params(source_fit_raw, source_hw, passes=passes)
    source_fit_prepared = _prepare_strict_semantic_free_frame(source_fit_raw, source_hw, source_params)
    source_test_prepared = _prepare_strict_semantic_free_frame(source_test_raw, source_hw, source_params)
    source_bundle = _fit_calibrator_bundle(source_fit_prepared, mode="analytical")
    source_pred = _predict_calibrator_bundle(source_bundle, source_test_prepared)
    source_metrics = _prepared_metrics(source_test_prepared, source_pred)
    analytical_metrics = _analytical_only_metrics(source_test_prepared)

    source_root = output_root / "source_reference"
    source_root.mkdir(parents=True, exist_ok=True)
    _json_dump(source_root / "analytical_params.json", source_params)
    _save_bundle(source_root / "calibrator.joblib", source_bundle)
    reference_payload = {
        "fit_rows": int(len(source_fit_raw)),
        "test_rows": int(len(source_test_raw)),
        "chain_metrics": source_metrics,
        "analytical_only_metrics": analytical_metrics,
    }
    _json_dump(source_root / "metrics.json", reference_payload)
    _json_dump(output_root / "source_reference.json", reference_payload)
    return {
        "params": source_params,
        "bundle": source_bundle,
        "metrics": source_metrics,
        "analytical_metrics": analytical_metrics,
        "fit_prepared": source_fit_prepared,
        "test_prepared": source_test_prepared,
        "fit_raw": source_fit_raw,
        "test_raw": source_test_raw,
    }


def _run_compare_strict(
    artifact_root: Path,
    scales: list[int],
    seed: int,
    passes: int,
    source_splits: dict[str, pd.DataFrame],
    target_splits: dict[str, pd.DataFrame],
    source_hw: Path,
    target_hw: Path,
    shared_sample_plan: dict[int, pd.DataFrame] | None = None,
    source_params_override: dict[str, float] | None = None,
) -> None:
    exp_root = artifact_root / "compare_strict"
    exp_root.mkdir(parents=True, exist_ok=True)

    source_fit_raw = _fit_pool(source_splits)
    target_fit_raw = _fit_pool(target_splits)
    target_test_raw = target_splits["test"]
    source_params = source_params_override or _build_analytical_params(source_fit_raw, source_hw, passes=passes)
    _json_dump(exp_root / "source_frozen_analytical_params.json", source_params)
    target_fit_prepared_full = _prepare_strict_semantic_free_frame(target_fit_raw, target_hw, source_params)
    target_test_prepared = _prepare_strict_semantic_free_frame(target_test_raw, target_hw, source_params)
    target_proxy_ids = set(target_splits["val"]["row_uid"].astype(str))
    target_proxy_prepared = target_fit_prepared_full[target_fit_prepared_full["row_uid"].astype(str).isin(target_proxy_ids)].copy().reset_index(drop=True)
    zero_metrics = _analytical_only_metrics(target_test_prepared)

    rows: list[dict[str, Any]] = [
        {
            "scale": "0%",
            "fit_rows": 0,
            "route": "frozen_analytical_only",
            "selected_family": pd.NA,
            **zero_metrics,
        }
    ]
    family_reports: list[dict[str, Any]] = []
    full_chain_active_rows: list[dict[str, Any]] = []

    for scale in scales:
        scale_label = _scale_label(scale)
        sampled = (shared_sample_plan or {}).get(int(scale))
        if sampled is None:
            sampled = _sample_fit_subset(target_fit_raw, scale, seed)
        _assert_no_leak(sampled, target_test_raw)
        _write_sample_rows(exp_root / f"sampled_rows_{scale_label.replace('%', 'pct')}.csv", sampled)
        trained_analytical_params = _build_analytical_params(sampled, target_hw, passes=passes, init_params=source_params)
        target_test_prepared_trained_analytical = _prepare_strict_semantic_free_frame(target_test_raw, target_hw, trained_analytical_params)
        trained_analytical_metrics = _analytical_only_metrics(target_test_prepared_trained_analytical)
        rows.append(
            {
                "scale": scale_label,
                "fit_rows": int(len(sampled)),
                "route": "pure_semantic_free_trained_analytical_only",
                "selected_family": pd.NA,
                **trained_analytical_metrics,
            }
        )
        fit_prepared = _filter_prepared_rows(target_fit_prepared_full, sampled)
        sampled_ids = set(sampled["row_uid"].astype(str))
        can_select_family = int(scale) <= SMALL_SCALE_FAMILY_SEARCH_MAX and sampled_ids.isdisjoint(target_proxy_ids)

        if int(scale) <= ANALYTICAL_PER_OP_FAMILY_SEARCH_MAX and sampled_ids.isdisjoint(target_proxy_ids):
            analytical_bundle, analytical_selection = _fit_per_op_selected_family_bundle(
                fit_prepared,
                target_proxy_prepared,
                mode="analytical",
                candidate_families=ANALYTICAL_PER_OP_CANDIDATES,
            )
            for report in analytical_selection["candidates"]:
                family_reports.append({"scale": scale_label, "route": "analytical", **report})
            selected_analytical_family = str(analytical_selection["selected_family"])
        elif can_select_family:
            analytical_bundle, analytical_selection = _fit_selected_family_bundle(
                fit_prepared,
                target_proxy_prepared,
                mode="analytical",
            )
            for report in analytical_selection["candidates"]:
                family_reports.append({"scale": scale_label, "route": "analytical", **report})
            selected_analytical_family = str(analytical_selection["selected_family"])
        else:
            analytical_bundle = _fit_calibrator_bundle(fit_prepared, mode="analytical")
            selected_analytical_family = str(analytical_bundle.get("model_family", DEFAULT_CALIBRATOR_FAMILY))
        analytical_pred = _predict_calibrator_bundle(analytical_bundle, target_test_prepared)
        analytical_metrics = _prepared_metrics(target_test_prepared, analytical_pred)
        analytical_active_metrics = _prepared_metrics(target_test_prepared, analytical_pred, scope="analytical_active_only")
        rows.append(
            {
                "scale": scale_label,
                "fit_rows": int(len(sampled)),
                "route": "pure_semantic_free_frozen_analytical_plus_calibrator",
                "selected_family": selected_analytical_family,
                **analytical_metrics,
            }
        )
        shape_tree_plus_pred = _predict_shape_tree_plus_problem_lookup(
            fit_prepared,
            target_test_prepared,
            analytical_pred,
        )
        shape_tree_plus_metrics = _prepared_metrics(target_test_prepared, shape_tree_plus_pred)
        rows.append(
            {
                "scale": scale_label,
                "fit_rows": int(len(sampled)),
                "route": SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
                "selected_family": f"{selected_analytical_family}+shape_tree_rf_plus_lookup",
                **shape_tree_plus_metrics,
            }
        )

        if can_select_family:
            direct_bundle, direct_selection = _fit_selected_family_bundle(
                fit_prepared,
                target_proxy_prepared,
                mode="direct",
            )
            for report in direct_selection["candidates"]:
                family_reports.append({"scale": scale_label, "route": "direct", **report})
            selected_direct_family = str(direct_selection["selected_family"])
        else:
            direct_bundle = _fit_calibrator_bundle(fit_prepared, mode="direct")
            selected_direct_family = str(direct_bundle.get("model_family", DEFAULT_CALIBRATOR_FAMILY))
        direct_pred = _predict_calibrator_bundle(direct_bundle, target_test_prepared)
        direct_metrics = _prepared_metrics(target_test_prepared, direct_pred)
        rows.append(
            {
                "scale": scale_label,
                "fit_rows": int(len(sampled)),
                "route": "pure_semantic_free_direct_hw_context_tree",
                "selected_family": selected_direct_family,
                **direct_metrics,
            }
        )
        full_chain_active_rows.append(
            {
                "scale": scale_label,
                **analytical_active_metrics,
            }
        )

    summary = pd.DataFrame(rows)
    summary["scale_num"] = summary["scale"].map(lambda value: float(str(value).rstrip("%")))
    summary = summary.sort_values(["scale_num", "route"]).drop(columns=["scale_num"]).reset_index(drop=True)
    summary.to_csv(exp_root / "summary.csv", index=False)
    if family_reports:
        report_df = pd.DataFrame(family_reports)
        report_df["scale_num"] = report_df["scale"].map(lambda value: float(str(value).rstrip("%")))
        report_df = report_df.sort_values(["scale_num", "route", "family"]).drop(columns=["scale_num"]).reset_index(drop=True)
        report_df.to_csv(exp_root / "family_search_summary.csv", index=False)

    pivot = summary.pivot(index="scale", columns="route", values=["mape", "p90", "median"])
    pivot.to_csv(exp_root / "pivot.csv")
    compare = summary[
        summary["route"].isin(
            [
                "pure_semantic_free_frozen_analytical_plus_calibrator",
                SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
                "pure_semantic_free_direct_hw_context_tree",
                "pure_semantic_free_trained_analytical_only",
            ]
        )
    ].copy()
    compare_plot = compare[
        compare["route"].isin(
            [
                "pure_semantic_free_frozen_analytical_plus_calibrator",
                SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
                "pure_semantic_free_direct_hw_context_tree",
            ]
        )
    ].copy()
    delta = compare.pivot(index="scale", columns="route", values=["mape", "p90", "median"])
    delta_df = pd.DataFrame({"scale": delta.index})
    delta_df["mape_gap_direct_minus_analytical"] = (
        delta[("mape", "pure_semantic_free_direct_hw_context_tree")] - delta[("mape", "pure_semantic_free_frozen_analytical_plus_calibrator")]
    ).to_numpy()
    delta_df["p90_gap_direct_minus_analytical"] = (
        delta[("p90", "pure_semantic_free_direct_hw_context_tree")] - delta[("p90", "pure_semantic_free_frozen_analytical_plus_calibrator")]
    ).to_numpy()
    delta_df["median_gap_direct_minus_analytical"] = (
        delta[("median", "pure_semantic_free_direct_hw_context_tree")] - delta[("median", "pure_semantic_free_frozen_analytical_plus_calibrator")]
    ).to_numpy()
    delta_df["mape_gap_shape_tree_plus_minus_analytical"] = (
        delta[("mape", SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE)] - delta[("mape", "pure_semantic_free_frozen_analytical_plus_calibrator")]
    ).to_numpy()
    delta_df["p90_gap_shape_tree_plus_minus_analytical"] = (
        delta[("p90", SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE)] - delta[("p90", "pure_semantic_free_frozen_analytical_plus_calibrator")]
    ).to_numpy()
    delta_df["median_gap_shape_tree_plus_minus_analytical"] = (
        delta[("median", SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE)] - delta[("median", "pure_semantic_free_frozen_analytical_plus_calibrator")]
    ).to_numpy()
    full_chain_active_df = pd.DataFrame(full_chain_active_rows).set_index("scale")
    delta_df["mape_gap_trained_analytical_minus_full_chain_active"] = (
        delta[("mape", "pure_semantic_free_trained_analytical_only")] - full_chain_active_df["mape"]
    ).to_numpy()
    delta_df["p90_gap_trained_analytical_minus_full_chain_active"] = (
        delta[("p90", "pure_semantic_free_trained_analytical_only")] - full_chain_active_df["p90"]
    ).to_numpy()
    delta_df["median_gap_trained_analytical_minus_full_chain_active"] = (
        delta[("median", "pure_semantic_free_trained_analytical_only")] - full_chain_active_df["median"]
    ).to_numpy()
    delta_df.to_csv(exp_root / "delta.csv", index=False)

    _plot_metric_curves(
        compare_plot,
        metric="mape",
        output_path=exp_root / "mape_compare.png",
        series_by="route",
        title="Strict semantic_free_context: MAPE",
    )
    _plot_metric_curves(
        compare_plot,
        metric="p90",
        output_path=exp_root / "p90_compare.png",
        series_by="route",
        title="Strict semantic_free_context: P90 APE",
    )
    _compare_to_reference(summary, exp_root / "regression_vs_previous.csv")
    _write_experiment_manifest(
        exp_root / "manifest.json",
        {
            "experiment": "compare_strict",
            "scales": scales,
            "seed": seed,
            "passes": passes,
            "small_scale_family_candidates": SMALL_SCALE_FAMILY_CANDIDATES,
            "source_dataset_root": str(source_splits["train"].attrs.get("dataset_root", "")),
            "target_dataset_root": str(target_splits["train"].attrs.get("dataset_root", "")),
            "source_hw_config": str(source_hw),
            "target_hw_config": str(target_hw),
            "routes": [
                "trained_analytical_only",
                "frozen_analytical_plus_calibrator",
                "shape_tree_plus_problem_lookup",
                "direct_hw_context_tree",
            ],
            "plot_routes": [
                "frozen_analytical_plus_calibrator",
                "shape_tree_plus_problem_lookup",
                "direct_hw_context_tree",
            ],
            "plot_note": "trained_analytical_only is retained in summary.csv and delta.csv but excluded from compare plots because its scale is orders of magnitude larger and would flatten the other curves.",
            "trained_analytical_eval_scope": "analytical_active_only",
            "trained_analytical_note": "trained_analytical_only excludes Reshape/Shape/Flatten/Unsqueeze from metric aggregation because those op types do not use the Analytical model in this framework.",
            "shape_tree_plus_problem_lookup_note": "Branch-only route layered on top of frozen_analytical_plus_calibrator. It keeps the baseline prediction for most ops, overrides Reshape/Flatten/Unsqueeze/Concat/Transpose/Sigmoid with static shape lookups, and replaces Shape with a dedicated static RF expert.",
        },
    )


def _run_exp1_optimal(
    artifact_root: Path,
    scales: list[int],
    seed: int,
    passes: int,
    source_splits: dict[str, pd.DataFrame],
    target_splits: dict[str, pd.DataFrame],
    source_hw: Path,
    target_hw: Path,
) -> None:
    exp_root = artifact_root / "exp1_optimal"
    exp_root.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    dataset_specs = [
        ("source", source_splits, source_hw, exp_root / "source"),
        ("target", target_splits, target_hw, exp_root / "target"),
    ]
    for dataset_name, raw_splits, hw_config, dataset_root in dataset_specs:
        dataset_root.mkdir(parents=True, exist_ok=True)
        fit_raw = _fit_pool(raw_splits)
        test_raw = raw_splits["test"]
        rows: list[dict[str, Any]] = []
        for scale in scales:
            scale_label = _scale_label(scale)
            sampled = _sample_fit_subset(fit_raw, scale, seed)
            _assert_no_leak(sampled, test_raw)
            _write_sample_rows(dataset_root / f"sampled_rows_{scale_label.replace('%', 'pct')}.csv", sampled)
            params = _build_analytical_params(sampled, hw_config, passes=passes)
            fit_prepared_full = _prepare_strict_semantic_free_frame(fit_raw, hw_config, params)
            fit_prepared = _filter_prepared_rows(fit_prepared_full, sampled)
            test_prepared = _prepare_strict_semantic_free_frame(test_raw, hw_config, params)
            bundle = _fit_calibrator_bundle(fit_prepared, mode="analytical")
            selected_family = str(bundle.get("model_family", DEFAULT_CALIBRATOR_FAMILY))
            pred = _predict_calibrator_bundle(bundle, test_prepared)
            chain_metrics = _prepared_metrics(test_prepared, pred)
            shape_tree_plus_pred = _predict_shape_tree_plus_problem_lookup(fit_prepared, test_prepared, pred)
            shape_tree_plus_metrics = _prepared_metrics(test_prepared, shape_tree_plus_pred)
            analytical_metrics = _analytical_only_metrics(test_prepared)
            baseline_row = {
                "scale": scale_label,
                "fit_rows": int(len(sampled)),
                "route": EXP1_BASE_ROUTE,
                "selected_family": selected_family,
                **chain_metrics,
                "analytical_only_mape": analytical_metrics["mape"],
                "analytical_only_p90": analytical_metrics["p90"],
                "analytical_only_median": analytical_metrics["median"],
                "analytical_only_eval_rows": analytical_metrics["eval_rows"],
                "analytical_only_eval_scope": analytical_metrics["eval_scope"],
            }
            shape_tree_row = {
                "scale": scale_label,
                "fit_rows": int(len(sampled)),
                "route": SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
                "selected_family": _shape_tree_plus_selected_family(selected_family),
                **shape_tree_plus_metrics,
                "analytical_only_mape": analytical_metrics["mape"],
                "analytical_only_p90": analytical_metrics["p90"],
                "analytical_only_median": analytical_metrics["median"],
                "analytical_only_eval_rows": analytical_metrics["eval_rows"],
                "analytical_only_eval_scope": analytical_metrics["eval_scope"],
            }
            rows.extend([baseline_row, shape_tree_row])
            combined_rows.extend(
                [
                    {"dataset": dataset_name, **baseline_row},
                    {"dataset": dataset_name, **shape_tree_row},
                ]
            )
            _save_scale_artifacts(
                dataset_root / f"scale_{scale_label.replace('%', 'pct')}",
                analytical_params=params,
                calibrator_bundle=bundle,
                metrics_payload={
                    "baseline_chain_metrics": chain_metrics,
                    "shape_tree_plus_metrics": shape_tree_plus_metrics,
                    "analytical_only_metrics": analytical_metrics,
                },
            )

        summary = pd.DataFrame(rows)
        summary["scale_num"] = summary["scale"].map(lambda value: float(str(value).rstrip("%")))
        summary = summary.sort_values(["scale_num", "route"]).drop(columns=["scale_num"]).reset_index(drop=True)
        summary.to_csv(exp_root / f"{dataset_name}_summary.csv", index=False)
        summary.to_csv(dataset_root / "summary.csv", index=False)
        _plot_metric_curves(summary, metric="mape", output_path=exp_root / f"mape_vs_scale_{dataset_name}.png", series_by="route", title=f"{dataset_name} optimal MAPE")
        _plot_metric_curves(summary, metric="p90", output_path=exp_root / f"p90_vs_scale_{dataset_name}.png", series_by="route", title=f"{dataset_name} optimal P90")

    combined = pd.DataFrame(combined_rows)
    combined["dataset_route"] = combined["dataset"].astype(str) + "::" + combined["route"].astype(str)
    _plot_metric_curves(combined, metric="mape", output_path=exp_root / "combined_mape_vs_scale.png", series_by="dataset_route", title="Source vs target optimal MAPE")
    _plot_metric_curves(combined, metric="p90", output_path=exp_root / "combined_p90_vs_scale.png", series_by="dataset_route", title="Source vs target optimal P90")
    _write_experiment_manifest(
        exp_root / "manifest.json",
        {
            "experiment": "exp1_optimal",
            "scales": scales,
            "seed": seed,
            "passes": passes,
            "routes": [EXP1_BASE_ROUTE, SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE],
            "source_dataset_root": str(source_splits["train"].attrs.get("dataset_root", "")),
            "target_dataset_root": str(target_splits["train"].attrs.get("dataset_root", "")),
            "source_hw_config": str(source_hw),
            "target_hw_config": str(target_hw),
            "shape_tree_plus_problem_lookup_note": "Branch-only route layered on top of the optimal analytical+calibrator chain for each dataset. It keeps the baseline prediction for most ops, overrides Reshape/Flatten/Unsqueeze/Concat/Transpose/Sigmoid with static shape lookups, and replaces Shape with a dedicated static RF expert.",
        },
    )


def _plot_single_series(summary: pd.DataFrame, output_path: Path, metric: str, title: str, *, drop_zero: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ordered = summary.copy()
    if drop_zero:
        ordered = ordered[ordered["scale"].astype(str) != "0%"].copy()
    ordered["scale_num"] = ordered["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    ordered = ordered.sort_values("scale_num")
    if ordered.empty:
        plt.close(fig)
        return
    ax.plot(ordered["scale_num"], ordered[metric].to_numpy(dtype=float) * 100.0, marker="o", linewidth=2)
    ax.set_xlabel("Fit Scale (%)")
    ax.set_ylabel("MAPE (%)" if metric == "mape" else "P90 APE (%)")
    ax.set_title(title)
    ax.set_xticks(sorted(ordered["scale_num"].dropna().unique().tolist()))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _run_exp2_warm_refit(
    artifact_root: Path,
    scales: list[int],
    seed: int,
    passes: int,
    source_splits: dict[str, pd.DataFrame],
    target_splits: dict[str, pd.DataFrame],
    source_hw: Path,
    target_hw: Path,
    shared_sample_plan: dict[int, pd.DataFrame] | None = None,
) -> None:
    exp_root = artifact_root / "exp2_warm_refit"
    exp_root.mkdir(parents=True, exist_ok=True)
    source_reference = _train_source_reference(source_splits, source_hw, exp_root, passes)
    source_params = source_reference["params"]
    source_bundle = source_reference["bundle"]
    source_selected_family = str(source_bundle.get("model_family", DEFAULT_CALIBRATOR_FAMILY))

    target_fit_raw = _fit_pool(target_splits)
    target_test_raw = target_splits["test"]
    zero_test_prepared = _prepare_strict_semantic_free_frame(target_test_raw, target_hw, source_params)
    zero_pred = _predict_calibrator_bundle(source_bundle, zero_test_prepared)
    zero_metrics = _prepared_metrics(zero_test_prepared, zero_pred)
    zero_shape_tree_pred = _predict_shape_tree_plus_problem_lookup(source_reference["fit_prepared"], zero_test_prepared, zero_pred)
    zero_shape_tree_metrics = _prepared_metrics(zero_test_prepared, zero_shape_tree_pred)
    zero_analytical = _analytical_only_metrics(zero_test_prepared)

    rows: list[dict[str, Any]] = [
        {
            "scale": "0%",
            "fit_rows": 0,
            "route": EXP2_BASE_ROUTE,
            "selected_family": source_selected_family,
            **zero_metrics,
            "analytical_only_mape": zero_analytical["mape"],
            "analytical_only_p90": zero_analytical["p90"],
            "analytical_only_median": zero_analytical["median"],
            "analytical_only_eval_rows": zero_analytical["eval_rows"],
            "analytical_only_eval_scope": zero_analytical["eval_scope"],
            "source_base_mape": zero_metrics["mape"],
            "source_base_p90": zero_metrics["p90"],
            "source_base_median": zero_metrics["median"],
        },
        {
            "scale": "0%",
            "fit_rows": 0,
            "route": SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
            "selected_family": _shape_tree_plus_selected_family(source_selected_family),
            **zero_shape_tree_metrics,
            "analytical_only_mape": zero_analytical["mape"],
            "analytical_only_p90": zero_analytical["p90"],
            "analytical_only_median": zero_analytical["median"],
            "analytical_only_eval_rows": zero_analytical["eval_rows"],
            "analytical_only_eval_scope": zero_analytical["eval_scope"],
            "source_base_mape": zero_metrics["mape"],
            "source_base_p90": zero_metrics["p90"],
            "source_base_median": zero_metrics["median"],
        },
    ]
    family_reports: list[dict[str, Any]] = []
    target_proxy_ids = set(target_splits["val"]["row_uid"].astype(str))

    for scale in scales:
        scale_label = _scale_label(scale)
        sampled = (shared_sample_plan or {}).get(int(scale))
        if sampled is None:
            sampled = _sample_fit_subset(target_fit_raw, scale, seed)
        _assert_no_leak(sampled, target_test_raw)
        _write_sample_rows(exp_root / f"sampled_rows_{scale_label.replace('%', 'pct')}.csv", sampled)

        target_params = _build_analytical_params(sampled, target_hw, passes=passes, init_params=source_params)
        fit_prepared_full = _prepare_strict_semantic_free_frame(target_fit_raw, target_hw, target_params)
        fit_prepared = _filter_prepared_rows(fit_prepared_full, sampled)
        test_prepared = _prepare_strict_semantic_free_frame(target_test_raw, target_hw, target_params)
        proxy_prepared = fit_prepared_full[fit_prepared_full["row_uid"].astype(str).isin(target_proxy_ids)].copy().reset_index(drop=True)
        sampled_ids = set(sampled["row_uid"].astype(str))
        can_select_family = int(scale) <= SMALL_SCALE_FAMILY_SEARCH_MAX and sampled_ids.isdisjoint(target_proxy_ids)

        # Tree models do not have a meaningful warm-start path. The reliable protocol
        # is to warm-start only the Analytical parameters from source, then retrain a
        # target-side calibrator on the target subset.
        if int(scale) <= ANALYTICAL_PER_OP_FAMILY_SEARCH_MAX and sampled_ids.isdisjoint(target_proxy_ids):
            target_bundle, selection = _fit_per_op_selected_family_bundle(
                fit_prepared,
                proxy_prepared,
                mode="analytical",
                candidate_families=ANALYTICAL_PER_OP_CANDIDATES,
            )
            for report in selection["candidates"]:
                family_reports.append({"scale": scale_label, **report})
            selected_family = str(selection["selected_family"])
        elif can_select_family:
            target_bundle, selection = _fit_selected_family_bundle(
                fit_prepared,
                proxy_prepared,
                mode="analytical",
            )
            for report in selection["candidates"]:
                family_reports.append({"scale": scale_label, **report})
            selected_family = str(selection["selected_family"])
        else:
            target_bundle = _fit_calibrator_bundle(fit_prepared, mode="analytical")
            selected_family = str(target_bundle.get("model_family", DEFAULT_CALIBRATOR_FAMILY))
        pred = _predict_calibrator_bundle(target_bundle, test_prepared)
        chain_metrics = _prepared_metrics(test_prepared, pred)
        shape_tree_plus_pred = _predict_shape_tree_plus_problem_lookup(fit_prepared, test_prepared, pred)
        shape_tree_plus_metrics = _prepared_metrics(test_prepared, shape_tree_plus_pred)
        analytical_metrics = _analytical_only_metrics(test_prepared)
        baseline_row = {
            "scale": scale_label,
            "fit_rows": int(len(sampled)),
            "route": EXP2_BASE_ROUTE,
            "selected_family": selected_family,
            **chain_metrics,
            "analytical_only_mape": analytical_metrics["mape"],
            "analytical_only_p90": analytical_metrics["p90"],
            "analytical_only_median": analytical_metrics["median"],
            "analytical_only_eval_rows": analytical_metrics["eval_rows"],
            "analytical_only_eval_scope": analytical_metrics["eval_scope"],
            "source_base_mape": zero_metrics["mape"],
            "source_base_p90": zero_metrics["p90"],
            "source_base_median": zero_metrics["median"],
        }
        shape_tree_row = {
            "scale": scale_label,
            "fit_rows": int(len(sampled)),
            "route": SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
            "selected_family": _shape_tree_plus_selected_family(selected_family),
            **shape_tree_plus_metrics,
            "analytical_only_mape": analytical_metrics["mape"],
            "analytical_only_p90": analytical_metrics["p90"],
            "analytical_only_median": analytical_metrics["median"],
            "analytical_only_eval_rows": analytical_metrics["eval_rows"],
            "analytical_only_eval_scope": analytical_metrics["eval_scope"],
            "source_base_mape": zero_metrics["mape"],
            "source_base_p90": zero_metrics["p90"],
            "source_base_median": zero_metrics["median"],
        }
        rows.extend([baseline_row, shape_tree_row])
        _save_scale_artifacts(
            exp_root / f"scale_{scale_label.replace('%', 'pct')}",
            analytical_params=target_params,
            calibrator_bundle=target_bundle,
            metrics_payload={
                "baseline_chain_metrics": chain_metrics,
                "shape_tree_plus_metrics": shape_tree_plus_metrics,
                "analytical_only_metrics": analytical_metrics,
                "source_base_metrics": zero_metrics,
            },
        )

    summary = pd.DataFrame(rows)
    summary["scale_num"] = summary["scale"].map(lambda value: float(str(value).rstrip("%")))
    summary = summary.sort_values(["scale_num", "route"]).drop(columns=["scale_num"]).reset_index(drop=True)
    summary.to_csv(exp_root / "summary.csv", index=False)
    if family_reports:
        report_df = pd.DataFrame(family_reports)
        report_df["scale_num"] = report_df["scale"].map(lambda value: float(str(value).rstrip("%")))
        report_df = report_df.sort_values(["scale_num", "family"]).drop(columns=["scale_num"]).reset_index(drop=True)
        report_df.to_csv(exp_root / "family_search_summary.csv", index=False)
    _plot_metric_curves(summary, metric="mape", output_path=exp_root / "mape_vs_scale.png", series_by="route", title="Warm-refit target MAPE")
    _plot_metric_curves(summary, metric="p90", output_path=exp_root / "p90_vs_scale.png", series_by="route", title="Warm-refit target P90")
    _plot_metric_curves(summary[summary["scale"] != "0%"], metric="mape", output_path=exp_root / "mape_vs_scale_no_zero.png", series_by="route", title="Warm-refit target MAPE (No 0%)")
    _plot_metric_curves(summary[summary["scale"] != "0%"], metric="p90", output_path=exp_root / "p90_vs_scale_no_zero.png", series_by="route", title="Warm-refit target P90 (No 0%)")
    _plot_compare_rows(summary[summary["scale"] != "0%"], exp_root / "target_vs_source_reference.png", "Warm-refit vs source reference", source_reference["metrics"])
    _write_experiment_manifest(
        exp_root / "manifest.json",
        {
            "experiment": "exp2_warm_refit",
            "scales": scales,
            "seed": seed,
            "passes": passes,
            "small_scale_family_candidates": SMALL_SCALE_FAMILY_CANDIDATES,
            "routes": [EXP2_BASE_ROUTE, SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE],
            "source_dataset_root": str(source_splits["train"].attrs.get("dataset_root", "")),
            "target_dataset_root": str(target_splits["train"].attrs.get("dataset_root", "")),
            "source_hw_config": str(source_hw),
            "target_hw_config": str(target_hw),
            "shape_tree_plus_problem_lookup_note": "Branch-only route layered on top of the warm-refit analytical+calibrator chain. It keeps the baseline prediction for most ops, overrides Reshape/Flatten/Unsqueeze/Concat/Transpose/Sigmoid with static shape lookups, and replaces Shape with a dedicated static RF expert.",
        },
    )


def _run_exp3_frozen_analytical(
    artifact_root: Path,
    scales: list[int],
    seed: int,
    passes: int,
    source_splits: dict[str, pd.DataFrame],
    target_splits: dict[str, pd.DataFrame],
    source_hw: Path,
    target_hw: Path,
    shared_sample_plan: dict[int, pd.DataFrame] | None = None,
) -> None:
    exp_root = artifact_root / "exp3_frozen_analytical"
    exp_root.mkdir(parents=True, exist_ok=True)
    source_reference = _train_source_reference(source_splits, source_hw, exp_root, passes)
    source_params = source_reference["params"]
    source_bundle = source_reference["bundle"]
    source_selected_family = str(source_bundle.get("model_family", DEFAULT_CALIBRATOR_FAMILY))

    target_fit_raw = _fit_pool(target_splits)
    target_test_raw = target_splits["test"]
    target_fit_prepared_full = _prepare_strict_semantic_free_frame(target_fit_raw, target_hw, source_params)
    target_test_prepared = _prepare_strict_semantic_free_frame(target_test_raw, target_hw, source_params)
    zero_pred = _predict_calibrator_bundle(source_bundle, target_test_prepared)
    zero_metrics = _prepared_metrics(target_test_prepared, zero_pred)
    zero_shape_tree_pred = _predict_shape_tree_plus_problem_lookup(source_reference["fit_prepared"], target_test_prepared, zero_pred)
    zero_shape_tree_metrics = _prepared_metrics(target_test_prepared, zero_shape_tree_pred)
    zero_analytical = _analytical_only_metrics(target_test_prepared)

    rows: list[dict[str, Any]] = [
        {
            "scale": "0%",
            "fit_rows": 0,
            "route": EXP3_BASE_ROUTE,
            "selected_family": source_selected_family,
            **zero_metrics,
            "analytical_only_mape": zero_analytical["mape"],
            "analytical_only_p90": zero_analytical["p90"],
            "analytical_only_median": zero_analytical["median"],
            "analytical_only_eval_rows": zero_analytical["eval_rows"],
            "analytical_only_eval_scope": zero_analytical["eval_scope"],
            "source_base_mape": zero_metrics["mape"],
            "source_base_p90": zero_metrics["p90"],
            "source_base_median": zero_metrics["median"],
        },
        {
            "scale": "0%",
            "fit_rows": 0,
            "route": SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
            "selected_family": _shape_tree_plus_selected_family(source_selected_family),
            **zero_shape_tree_metrics,
            "analytical_only_mape": zero_analytical["mape"],
            "analytical_only_p90": zero_analytical["p90"],
            "analytical_only_median": zero_analytical["median"],
            "analytical_only_eval_rows": zero_analytical["eval_rows"],
            "analytical_only_eval_scope": zero_analytical["eval_scope"],
            "source_base_mape": zero_metrics["mape"],
            "source_base_p90": zero_metrics["p90"],
            "source_base_median": zero_metrics["median"],
        },
    ]
    family_reports: list[dict[str, Any]] = []
    target_proxy_ids = set(target_splits["val"]["row_uid"].astype(str))

    for scale in scales:
        scale_label = _scale_label(scale)
        sampled = (shared_sample_plan or {}).get(int(scale))
        if sampled is None:
            sampled = _sample_fit_subset(target_fit_raw, scale, seed)
        _assert_no_leak(sampled, target_test_raw)
        _write_sample_rows(exp_root / f"sampled_rows_{scale_label.replace('%', 'pct')}.csv", sampled)

        fit_prepared = _filter_prepared_rows(target_fit_prepared_full, sampled)
        sampled_ids = set(sampled["row_uid"].astype(str))
        can_select_family = int(scale) <= SMALL_SCALE_FAMILY_SEARCH_MAX and sampled_ids.isdisjoint(target_proxy_ids)
        if int(scale) <= ANALYTICAL_PER_OP_FAMILY_SEARCH_MAX and sampled_ids.isdisjoint(target_proxy_ids):
            target_proxy_prepared = target_fit_prepared_full[target_fit_prepared_full["row_uid"].astype(str).isin(target_proxy_ids)].copy().reset_index(drop=True)
            target_bundle, selection = _fit_per_op_selected_family_bundle(
                fit_prepared,
                target_proxy_prepared,
                mode="analytical",
                candidate_families=ANALYTICAL_PER_OP_CANDIDATES,
            )
            for report in selection["candidates"]:
                family_reports.append({"scale": scale_label, **report})
            selected_family = str(selection["selected_family"])
        elif can_select_family:
            target_proxy_prepared = target_fit_prepared_full[target_fit_prepared_full["row_uid"].astype(str).isin(target_proxy_ids)].copy().reset_index(drop=True)
            target_bundle, selection = _fit_selected_family_bundle(
                fit_prepared,
                target_proxy_prepared,
                mode="analytical",
            )
            for report in selection["candidates"]:
                family_reports.append({"scale": scale_label, **report})
            selected_family = str(selection["selected_family"])
        else:
            target_bundle = _fit_calibrator_bundle(fit_prepared, mode="analytical")
            selected_family = str(target_bundle.get("model_family", DEFAULT_CALIBRATOR_FAMILY))
        pred = _predict_calibrator_bundle(target_bundle, target_test_prepared)
        chain_metrics = _prepared_metrics(target_test_prepared, pred)
        shape_tree_plus_pred = _predict_shape_tree_plus_problem_lookup(fit_prepared, target_test_prepared, pred)
        shape_tree_plus_metrics = _prepared_metrics(target_test_prepared, shape_tree_plus_pred)
        analytical_metrics = _analytical_only_metrics(target_test_prepared)
        baseline_row = {
            "scale": scale_label,
            "fit_rows": int(len(sampled)),
            "route": EXP3_BASE_ROUTE,
            "selected_family": selected_family,
            **chain_metrics,
            "analytical_only_mape": analytical_metrics["mape"],
            "analytical_only_p90": analytical_metrics["p90"],
            "analytical_only_median": analytical_metrics["median"],
            "analytical_only_eval_rows": analytical_metrics["eval_rows"],
            "analytical_only_eval_scope": analytical_metrics["eval_scope"],
            "source_base_mape": zero_metrics["mape"],
            "source_base_p90": zero_metrics["p90"],
            "source_base_median": zero_metrics["median"],
        }
        shape_tree_row = {
            "scale": scale_label,
            "fit_rows": int(len(sampled)),
            "route": SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE,
            "selected_family": _shape_tree_plus_selected_family(selected_family),
            **shape_tree_plus_metrics,
            "analytical_only_mape": analytical_metrics["mape"],
            "analytical_only_p90": analytical_metrics["p90"],
            "analytical_only_median": analytical_metrics["median"],
            "analytical_only_eval_rows": analytical_metrics["eval_rows"],
            "analytical_only_eval_scope": analytical_metrics["eval_scope"],
            "source_base_mape": zero_metrics["mape"],
            "source_base_p90": zero_metrics["p90"],
            "source_base_median": zero_metrics["median"],
        }
        rows.extend([baseline_row, shape_tree_row])
        _save_scale_artifacts(
            exp_root / f"scale_{scale_label.replace('%', 'pct')}",
            analytical_params=source_params,
            calibrator_bundle=target_bundle,
            metrics_payload={
                "baseline_chain_metrics": chain_metrics,
                "shape_tree_plus_metrics": shape_tree_plus_metrics,
                "analytical_only_metrics": analytical_metrics,
                "source_base_metrics": zero_metrics,
            },
        )

    summary = pd.DataFrame(rows)
    summary["scale_num"] = summary["scale"].map(lambda value: float(str(value).rstrip("%")))
    summary = summary.sort_values(["scale_num", "route"]).drop(columns=["scale_num"]).reset_index(drop=True)
    summary.to_csv(exp_root / "summary.csv", index=False)
    if family_reports:
        report_df = pd.DataFrame(family_reports)
        report_df["scale_num"] = report_df["scale"].map(lambda value: float(str(value).rstrip("%")))
        report_df = report_df.sort_values(["scale_num", "family"]).drop(columns=["scale_num"]).reset_index(drop=True)
        report_df.to_csv(exp_root / "family_search_summary.csv", index=False)
    _plot_metric_curves(summary, metric="mape", output_path=exp_root / "mape_vs_scale.png", series_by="route", title="Frozen-analytical target MAPE")
    _plot_metric_curves(summary, metric="p90", output_path=exp_root / "p90_vs_scale.png", series_by="route", title="Frozen-analytical target P90")
    _plot_metric_curves(summary[summary["scale"] != "0%"], metric="mape", output_path=exp_root / "mape_vs_scale_no_zero.png", series_by="route", title="Frozen-analytical target MAPE (No 0%)")
    _plot_metric_curves(summary[summary["scale"] != "0%"], metric="p90", output_path=exp_root / "p90_vs_scale_no_zero.png", series_by="route", title="Frozen-analytical target P90 (No 0%)")
    _write_experiment_manifest(
        exp_root / "manifest.json",
        {
            "experiment": "exp3_frozen_analytical",
            "scales": scales,
            "seed": seed,
            "passes": passes,
            "small_scale_family_candidates": SMALL_SCALE_FAMILY_CANDIDATES,
            "routes": [EXP3_BASE_ROUTE, SHAPE_TREE_PLUS_PROBLEM_LOOKUP_ROUTE],
            "source_dataset_root": str(source_splits["train"].attrs.get("dataset_root", "")),
            "target_dataset_root": str(target_splits["train"].attrs.get("dataset_root", "")),
            "source_hw_config": str(source_hw),
            "target_hw_config": str(target_hw),
            "shape_tree_plus_problem_lookup_note": "Branch-only route layered on top of the frozen analytical+calibrator chain. It keeps the baseline prediction for most ops, overrides Reshape/Flatten/Unsqueeze/Concat/Transpose/Sigmoid with static shape lookups, and replaces Shape with a dedicated static RF expert.",
        },
    )


def _write_transfer_overview(artifact_root: Path) -> None:
    compare_path = artifact_root / "compare_strict" / "summary.csv"
    exp2_path = artifact_root / "exp2_warm_refit" / "summary.csv"
    exp3_path = artifact_root / "exp3_frozen_analytical" / "summary.csv"
    if not (compare_path.exists() and exp2_path.exists() and exp3_path.exists()):
        return

    compare = pd.read_csv(compare_path)
    compare = compare[compare["route"].astype(str) == "pure_semantic_free_frozen_analytical_plus_calibrator"].copy()
    compare["experiment"] = "compare_strict"
    exp2 = pd.read_csv(exp2_path).copy()
    if "route" in exp2.columns:
        exp2 = exp2[exp2["route"].astype(str) == EXP2_BASE_ROUTE].copy()
    exp2["experiment"] = "exp2_warm_refit"
    exp3 = pd.read_csv(exp3_path).copy()
    if "route" in exp3.columns:
        exp3 = exp3[exp3["route"].astype(str) == EXP3_BASE_ROUTE].copy()
    exp3["experiment"] = "exp3_frozen_analytical"

    common_columns = ["scale", "fit_rows", "selected_family", "mape", "p90", "median", "experiment"]
    for frame in (compare, exp2, exp3):
        if "selected_family" not in frame.columns:
            frame["selected_family"] = pd.NA
    combined = pd.concat(
        [
            compare[common_columns],
            exp2[common_columns],
            exp3[common_columns],
        ],
        ignore_index=True,
    )
    combined = combined[combined["scale"].astype(str) != "0%"].copy()
    combined["scale_num"] = combined["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    combined = combined.sort_values(["scale_num", "experiment"]).drop(columns=["scale_num"]).reset_index(drop=True)

    overview_root = artifact_root / "transfer_overview"
    overview_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(overview_root / "summary.csv", index=False)
    overview_offsets = {
        "compare_strict": -0.12,
        "exp2_warm_refit": 0.0,
        "exp3_frozen_analytical": 0.12,
    }
    overview_styles = {
        "compare_strict": {
            "linestyle": "--",
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgewidth": 1.8,
            "markersize": 7,
            "zorder": 4,
        },
        "exp2_warm_refit": {
            "linestyle": "-",
            "marker": "s",
            "markersize": 6,
            "zorder": 3,
        },
        "exp3_frozen_analytical": {
            "linestyle": ":",
            "marker": "^",
            "markersize": 6,
            "zorder": 2,
        },
    }
    _plot_metric_curves(
        combined,
        metric="mape",
        output_path=overview_root / "mape_compare.png",
        series_by="experiment",
        title="Transfer comparison: MAPE",
        x_offsets=overview_offsets,
        series_styles=overview_styles,
    )
    _plot_metric_curves(
        combined,
        metric="p90",
        output_path=overview_root / "p90_compare.png",
        series_by="experiment",
        title="Transfer comparison: P90 APE",
        x_offsets=overview_offsets,
        series_styles=overview_styles,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True)
    combined_plot = combined.copy()
    combined_plot["scale_num"] = combined_plot["scale"].astype(str).map(lambda value: float(str(value).rstrip("%")))
    xticks = sorted(combined_plot["scale_num"].dropna().unique().tolist())
    for metric, ax in (("mape", axes[0]), ("p90", axes[1])):
        for experiment, group in combined_plot.groupby("experiment", sort=True):
            ordered = group.sort_values("scale_num")
            style = {"marker": "o", "linewidth": 2}
            style.update(overview_styles.get(str(experiment), {}))
            x_values = ordered["scale_num"].to_numpy(dtype=float) + float(overview_offsets.get(str(experiment), 0.0))
            ax.plot(x_values, ordered[metric].to_numpy(dtype=float) * 100.0, label=str(experiment), **style)
        ax.set_xlabel("Fit Scale (%)")
        ax.set_ylabel("MAPE (%)" if metric == "mape" else "P90 APE (%)")
        if xticks:
            ax.set_xticks(xticks)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Transfer chain comparison")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(2, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(overview_root / "combined_mape_p90.png", dpi=180)
    plt.close(fig)
    _write_experiment_manifest(
        overview_root / "manifest.json",
        {
            "experiment": "transfer_overview",
            "inputs": [str(compare_path), str(exp2_path), str(exp3_path)],
            "note": "Only >0% fit scales are included because compare_strict 0% is analytical-only while exp2/exp3 0% are source-chain zero-shot baselines.",
            "plot_visibility_note": "Small horizontal offsets are applied in overview plots so overlapping compare_strict / exp3 series remain visible.",
        },
    )


def main() -> None:
    args = _parse_args()
    scales = _parse_scales(args.scales)
    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    source_dataset_root = Path(args.source_dataset_root).resolve()
    target_dataset_root = Path(args.target_dataset_root).resolve()
    source_hw = Path(args.source_hw_config).resolve()
    target_hw = Path(args.target_hw_config).resolve()

    _sync_default_assets(source_dataset_root, target_dataset_root, source_hw, target_hw)
    asset_manifest_path = _write_asset_manifest(source_dataset_root, target_dataset_root, source_hw, target_hw)

    source_splits = _load_raw_splits(source_dataset_root)
    target_splits = _load_raw_splits(target_dataset_root)
    for frame in source_splits.values():
        frame.attrs["dataset_root"] = str(source_dataset_root)
    for frame in target_splits.values():
        frame.attrs["dataset_root"] = str(target_dataset_root)

    shared_sample_plan: dict[int, pd.DataFrame] = {}
    source_sampling_params: dict[str, float] | None = None
    if any(experiment in {"compare_strict", "exp2_warm_refit", "exp3_frozen_analytical", "all"} for experiment in ([args.experiment] if args.experiment != "all" else list(DEFAULT_EXPERIMENTS))):
        if any(int(scale) <= SAMPLER_SEARCH_MAX_SCALE for scale in scales):
            source_sampling_params = _build_analytical_params(_fit_pool(source_splits), source_hw, passes=args.passes)
            shared_sample_plan = _build_transfer_small_scale_plan(
                artifact_root,
                target_splits=target_splits,
                target_hw=target_hw,
                source_params=source_sampling_params,
                scales=scales,
                seed=args.seed,
            )

    experiments = DEFAULT_EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    for experiment in experiments:
        if experiment == "compare_strict":
            _run_compare_strict(
                artifact_root,
                scales,
                args.seed,
                args.passes,
                source_splits,
                target_splits,
                source_hw,
                target_hw,
                shared_sample_plan=shared_sample_plan,
                source_params_override=source_sampling_params,
            )
        elif experiment == "exp1_optimal":
            _run_exp1_optimal(artifact_root, scales, args.seed, args.passes, source_splits, target_splits, source_hw, target_hw)
        elif experiment == "exp2_warm_refit":
            _run_exp2_warm_refit(
                artifact_root,
                scales,
                args.seed,
                args.passes,
                source_splits,
                target_splits,
                source_hw,
                target_hw,
                shared_sample_plan=shared_sample_plan,
            )
        elif experiment == "exp3_frozen_analytical":
            _run_exp3_frozen_analytical(
                artifact_root,
                scales,
                args.seed,
                args.passes,
                source_splits,
                target_splits,
                source_hw,
                target_hw,
                shared_sample_plan=shared_sample_plan,
            )
        else:
            raise KeyError(experiment)

    _write_transfer_overview(artifact_root)

    print(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "asset_manifest": str(asset_manifest_path),
                "experiments": list(experiments),
                "scales": scales,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
