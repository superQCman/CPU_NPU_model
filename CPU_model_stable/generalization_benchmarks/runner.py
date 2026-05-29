from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpu_perf_model.analytical import build_analytical_features
from cpu_perf_model.classed.build_classed_dataset import build_classed_dataset_artifacts
from cpu_perf_model.classed.infer_class_models import load_dataset_summary
from cpu_perf_model.classed.train_class_models import train_all_classes
from cpu_perf_model.infer_mlp_onnx import run_inference
from cpu_perf_model.single_fair import train_fair_single_model
from cpu_perf_model.train_mlp import train_model

from .feature_enhancement import (
    ALL_DERIVED_FEATURES,
    ENHANCED_CLASSED_GROUP_FEATURES,
    derive_enhanced_features,
)


SEED = 42
TRAIN_RATIO = 0.7
VAL_FRACTION_OF_TRAIN = 0.1
ANALYTICAL_PASSES = 3
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "latest" / "generalization_benchmarks"
STANDALONE_ROOT = PROJECT_ROOT / "artifacts" / "latest" / "standalone_dataset_merge_ready"
DLRM_ROOT = Path("/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_78910_nodrop")
FIXED_ANALYTICAL_PARAMS_JSON = PROJECT_ROOT / "artifacts" / "latest" / "analytical_model" / "deploy" / "full_data_parameters.json"
BASE_FEATURE_COLUMNS_JSON = STANDALONE_ROOT / "feature_columns.json"

EXPERIMENT_SPECS = {
    "exp1_standalone70_vs_holdout_plus_dlrm_all": {
        "description": "Train on 70% standalone merge-ready; evaluate on standalone holdout and full DLRM.",
    },
    "exp2_union_70_30": {
        "description": "Merge standalone merge-ready and full DLRM, then split 70/30.",
    },
    "exp2_union_70_30_target_push": {
        "description": "Merge standalone merge-ready and full DLRM, then split 70/30; train non-fixed analytical plus routed expert ensemble.",
    },
    "exp3_standalone_only_70_30": {
        "description": "Standalone merge-ready only, split 70/30.",
    },
    "exp2_union_70_30_semantic_free_balanced_fixed_analytical": {
        "description": "Exp2 rerun with fixed analytical parameters, graph-semantic neutralization, and capped train-bucket rebalancing.",
        "fixed_analytical_params_json": str(FIXED_ANALYTICAL_PARAMS_JSON),
        "neutralize_graph_semantics": True,
        "rebalance_train": {
            "label_column": "label_operator_actual_dur_us",
            "label_bins": 8,
            "bucket_cap": 512,
            "bucket_columns": ["source_family", "op_type", "label_bin"],
        },
    },
    "exp2_union_70_30_semantic_free_prebalanced_fixed_analytical": {
        "description": "Exp2 rerun with fixed analytical parameters, graph-semantic neutralization, and split-before matching balance.",
        "fixed_analytical_params_json": str(FIXED_ANALYTICAL_PARAMS_JSON),
        "neutralize_graph_semantics": True,
        "prebalance_dataset": {
            "label_column": "label_operator_actual_dur_us",
            "label_bins": 8,
            "bucket_cap": 512,
            "bucket_columns": ["source_family", "op_type", "label_bin"],
        },
    },
}

MODEL_NAMES = [
    "analytical_reference",
    "original_classed_mlp",
    "original_single_mlp",
    "analytical_residual_single_mlp",
    "enhanced_classed_mlp",
    "family_enhanced",
    "global_enhanced",
    "routed_expert_ensemble",
    "routed_expert_residual_ensemble",
]

BASE_ENSEMBLE_CANDIDATES = [
    "original_single_mlp",
    "original_classed_mlp",
    "enhanced_classed_mlp",
    "family_enhanced",
    "global_enhanced",
]

BASE_ENHANCED_CATEGORICAL_FEATURES = [
    "op_type",
    "node_scope",
    "node_name_normalized",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
]

BASE_ENHANCED_NUMERIC_FEATURES = [
    "batch_size",
    "num_indices_per_lookup",
    "num_threads",
    "inter_threads",
    "output_size",
    "activation_size",
    "parameter_size",
    "feat_io_bytes_sum",
    "feat_lookup_count",
    "ana_calib_mem_us",
    "feat_output_elements_per_batch",
    "ana_calib_total_us",
    "feat_output_input_bytes_ratio",
    "feat_reduction_axes_count",
    "feat_reduction_work_items",
    "feat_reduction_axes_product",
    "feat_reduction_input_rank",
    "feat_reduction_output_rank",
    "feat_gemm_m",
    "feat_gemm_n",
    "feat_gemm_k",
    "feat_gemm_mac_count",
    "feat_gemm_bytes_per_mac",
    "ana_calib_compute_us",
] + list(ALL_DERIVED_FEATURES)

ROUTED_ENSEMBLE_CATEGORICAL_FEATURES = list(BASE_ENHANCED_CATEGORICAL_FEATURES)
ROUTED_ENSEMBLE_NUMERIC_FEATURES = list(dict.fromkeys(
    BASE_ENHANCED_NUMERIC_FEATURES
    + [
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ana_calib_compute_us",
        "blend_candidate_pred_min",
        "blend_candidate_pred_max",
        "blend_candidate_pred_span",
        "blend_candidate_pred_mean",
        "blend_candidate_pred_std",
        "blend_candidate_pred_cv",
        "blend_pred_global_minus_family",
        "blend_pred_classed_minus_single",
        "blend_pred_enhanced_classed_minus_classed",
    ]
    + [f"pred_{name}" for name in BASE_ENSEMBLE_CANDIDATES]
    + ["pred_analytical_reference"]
))

GRAPH_SEMANTIC_CONSTANTS = {
    "node_scope": "no_graph_semantics",
    "node_name_normalized": "no_graph_semantics",
    "arch_embedding_size": "no_arch_semantics",
    "arch_mlp_bot": "no_arch_semantics",
    "arch_mlp_top": "no_arch_semantics",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _ensure_required_columns(
    frame: pd.DataFrame,
    *,
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    for column in categorical_features or []:
        if column not in out.columns:
            out[column] = "missing"
        out[column] = out[column].fillna("missing").astype(str)
    for column in numeric_features or []:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(float)
    return out


def _experiment_config(experiment_id: str) -> dict[str, Any]:
    return dict(EXPERIMENT_SPECS.get(experiment_id, {}))


def _neutralize_graph_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, value in GRAPH_SEMANTIC_CONSTANTS.items():
        out[column] = value
    return out


def _apply_dataset_preprocessing(frame: pd.DataFrame, config: dict[str, Any] | None) -> pd.DataFrame:
    if not config:
        return frame
    out = frame.copy()
    if config.get("neutralize_graph_semantics"):
        out = _neutralize_graph_semantics(out)
    return out


def _derive_source_family(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda value: "standalone" if value.startswith("standalone_") else "dlrm")


def _copy_feature_manifest(output_dir: Path) -> None:
    payload = _read_json(BASE_FEATURE_COLUMNS_JSON)
    _write_json(output_dir / "feature_columns.json", payload)


def _prefix_identity_columns(frame: pd.DataFrame, dataset_alias: str) -> pd.DataFrame:
    out = frame.copy()
    out["benchmark_source_dataset"] = dataset_alias
    out["row_uid"] = dataset_alias + "::" + out["row_uid"].astype(str)
    sample_group = out["sample_group"].astype(str) if "sample_group" in out.columns else out["row_uid"].astype(str)
    out["sample_group"] = dataset_alias + "::" + sample_group
    return out


def _load_base_dataset(dataset_root: Path, dataset_alias: str) -> pd.DataFrame:
    frame = pd.read_csv(dataset_root / "dataset_full.csv", low_memory=False)
    frame = _prefix_identity_columns(frame, dataset_alias)
    frame["benchmark_dataset_alias"] = dataset_alias
    return frame


def _group_split_assign(
    frame: pd.DataFrame,
    *,
    train_ratio: float,
    seed: int,
    group_column: str = "sample_group",
    stratify_column: str | None = "op_type",
) -> tuple[pd.Series, pd.Series]:
    if group_column in frame.columns:
        keep_columns = [group_column] + ([stratify_column] if stratify_column else [])
        group_frame = frame[keep_columns].drop_duplicates(subset=[group_column], keep="first").reset_index(drop=True)
        groups = group_frame[group_column].astype(str)
        stratify = None if not stratify_column else group_frame[stratify_column].astype(str)
        try:
            train_groups, test_groups = train_test_split(
                groups,
                train_size=train_ratio,
                random_state=seed,
                stratify=stratify,
            )
        except ValueError:
            train_groups, test_groups = train_test_split(
                groups,
                train_size=train_ratio,
                random_state=seed,
                shuffle=True,
            )
        train_mask = frame[group_column].astype(str).isin(set(train_groups))
        test_mask = frame[group_column].astype(str).isin(set(test_groups))
        return train_mask, test_mask

    try:
        stratify = None if not stratify_column else frame[stratify_column].astype(str)
        train_idx, test_idx = train_test_split(
            frame.index.to_numpy(),
            train_size=train_ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            frame.index.to_numpy(),
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
        )
    train_mask = frame.index.isin(train_idx)
    test_mask = frame.index.isin(test_idx)
    return train_mask, test_mask


def _assign_train_val_test(
    frame: pd.DataFrame,
    *,
    outer_train_ratio: float,
    seed: int,
    val_fraction_of_train: float,
) -> pd.DataFrame:
    out = frame.copy()
    train_outer_mask, test_mask = _group_split_assign(
        out,
        train_ratio=outer_train_ratio,
        seed=seed,
    )
    out["split"] = "unused"
    out.loc[test_mask, "split"] = "test"
    train_outer = out.loc[train_outer_mask].copy()
    inner_train_mask, val_mask = _group_split_assign(
        train_outer,
        train_ratio=1.0 - val_fraction_of_train,
        seed=seed,
    )
    train_outer.loc[inner_train_mask, "split"] = "train"
    train_outer.loc[val_mask, "split"] = "val"
    out.loc[train_outer.index, "split"] = train_outer["split"]
    if (out["split"] == "unused").any():
        out.loc[out["split"] == "unused", "split"] = "train"
    return out


def _write_dataset_root(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    manifest_payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values("row_uid", kind="stable").reset_index(drop=True)
    frame.to_csv(output_dir / "dataset_full.csv", index=False)
    split_counts: dict[str, int] = {}
    split_group_counts: dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        split_df = frame[frame["split"] == split_name].copy()
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)
        split_counts[split_name] = int(len(split_df))
        split_group_counts[split_name] = int(split_df["sample_group"].astype(str).nunique()) if "sample_group" in split_df.columns else int(len(split_df))
    _copy_feature_manifest(output_dir)
    dataset_summary = {
        "total_rows": int(len(frame)),
        "split_row_counts": split_counts,
        "split_group_counts": split_group_counts,
        "case_count": int(frame["case_id"].astype(str).nunique()) if "case_id" in frame.columns else 0,
        "op_type_counts": {k: int(v) for k, v in frame["op_type"].value_counts().sort_index().to_dict().items()},
        "manifest": manifest_payload,
    }
    _write_json(output_dir / "dataset_summary.json", dataset_summary)
    _write_json(output_dir / "manifest.json", manifest_payload)


def _write_eval_raw_csvs(
    frame: pd.DataFrame,
    eval_root: Path,
    *,
    include_dlrm: bool,
) -> dict[str, Path]:
    eval_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    test_df = frame[frame["split"] == "test"].copy()
    outputs["test"] = eval_root / "test.csv"
    test_df.to_csv(outputs["test"], index=False)
    if include_dlrm:
        standalone_df = test_df[test_df["benchmark_source_dataset"] == "standalone_merge_ready"].copy()
        dlrm_df = test_df[test_df["benchmark_source_dataset"] == "dlrm_original"].copy()
        outputs["test_standalone"] = eval_root / "test_standalone.csv"
        outputs["test_dlrm"] = eval_root / "test_dlrm.csv"
        standalone_df.to_csv(outputs["test_standalone"], index=False)
        dlrm_df.to_csv(outputs["test_dlrm"], index=False)
    return outputs


def _plot_balance_histogram(
    before_train: pd.DataFrame,
    after_train: pd.DataFrame,
    output_path: Path,
    *,
    scope_label: str = "Train",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before_values = np.log10(pd.to_numeric(before_train["label_operator_actual_dur_us"], errors="coerce").clip(lower=1e-6).to_numpy(dtype=float))
    after_values = np.log10(pd.to_numeric(after_train["label_operator_actual_dur_us"], errors="coerce").clip(lower=1e-6).to_numpy(dtype=float))
    bins = np.linspace(float(np.nanmin(np.concatenate([before_values, after_values]))), float(np.nanmax(np.concatenate([before_values, after_values]))), 30)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.hist(before_values, bins=bins, alpha=0.45, density=True, label="Before rebalance", color="#4C78A8")
    ax.hist(after_values, bins=bins, alpha=0.55, density=True, label="After rebalance", color="#F58518")
    ax.set_xlabel("log10(label_operator_actual_dur_us)")
    ax.set_ylabel("Density")
    ax.set_title(f"{scope_label} label distribution before/after balance")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_balance_source_op(
    before_train: pd.DataFrame,
    after_train: pd.DataFrame,
    output_path: Path,
    *,
    scope_label: str = "Train",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for tag, frame in [("Before", before_train), ("After", after_train)]:
        part = (
            frame.assign(source_family=_derive_source_family(frame["source_name"]))
            .groupby(["op_type", "source_family"], as_index=False)
            .size()
        )
        part["stage"] = tag
        frames.append(part)
    plot_df = pd.concat(frames, ignore_index=True, sort=False)
    totals = plot_df.groupby(["stage", "op_type"])["size"].transform("sum").replace(0, 1)
    plot_df["share"] = plot_df["size"] / totals
    op_order = sorted(plot_df["op_type"].astype(str).unique())
    x = np.arange(len(op_order))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.0), sharex=True, sharey=True)
    for ax, stage in zip(axes, ["Before", "After"]):
        stage_df = plot_df[plot_df["stage"] == stage]
        standalone = stage_df[stage_df["source_family"] == "standalone"].set_index("op_type")["share"].reindex(op_order).fillna(0.0)
        dlrm = stage_df[stage_df["source_family"] == "dlrm"].set_index("op_type")["share"].reindex(op_order).fillna(0.0)
        ax.bar(x, dlrm.to_numpy(dtype=float), width=width, label="DLRM", color="#4C78A8")
        ax.bar(x, standalone.to_numpy(dtype=float), width=width, bottom=dlrm.to_numpy(dtype=float), label="Standalone", color="#F58518")
        ax.set_title(f"{stage} balance")
        ax.set_ylabel("Share within op_type")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(op_order, rotation=45, ha="right")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"{scope_label} source composition by op_type before/after balance", fontsize=14.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_balance_pair_heatmap(
    before_train: pd.DataFrame,
    after_train: pd.DataFrame,
    output_path: Path,
    *,
    scope_label: str = "Train",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bs_order = sorted(set(pd.to_numeric(before_train["batch_size"], errors="coerce").dropna().astype(int).tolist()) | set(pd.to_numeric(after_train["batch_size"], errors="coerce").dropna().astype(int).tolist()))
    nip_order = sorted(set(pd.to_numeric(before_train["num_indices_per_lookup"], errors="coerce").dropna().astype(int).tolist()) | set(pd.to_numeric(after_train["num_indices_per_lookup"], errors="coerce").dropna().astype(int).tolist()))

    def pivot(frame: pd.DataFrame) -> pd.DataFrame:
        part = (
            frame.assign(
                batch_size=pd.to_numeric(frame["batch_size"], errors="coerce").astype("Int64"),
                num_indices_per_lookup=pd.to_numeric(frame["num_indices_per_lookup"], errors="coerce").astype("Int64"),
            )
            .groupby(["batch_size", "num_indices_per_lookup"], as_index=False)
            .size()
            .pivot(index="batch_size", columns="num_indices_per_lookup", values="size")
            .reindex(index=bs_order, columns=nip_order, fill_value=0)
        )
        return part.astype(float)

    before_pivot = pivot(before_train)
    after_pivot = pivot(after_train)
    before_values = np.log1p(before_pivot.to_numpy(dtype=float))
    after_values = np.log1p(after_pivot.to_numpy(dtype=float))
    vmax = max(float(np.nanmax(before_values)), float(np.nanmax(after_values)), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5), sharex=True, sharey=True)
    for ax, title, part in zip(axes, ["Before balance", "After balance"], [before_values, after_values]):
        im = ax.imshow(part, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("num_indices_per_lookup")
        ax.set_ylabel("batch_size")
        x_ticks = np.linspace(0, len(nip_order) - 1, min(8, len(nip_order)), dtype=int) if nip_order else np.array([], dtype=int)
        y_ticks = np.linspace(0, len(bs_order) - 1, min(8, len(bs_order)), dtype=int) if bs_order else np.array([], dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(nip_order[idx]) for idx in x_ticks], rotation=45, ha="right")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(bs_order[idx]) for idx in y_ticks])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="log1p(row count)")
    fig.suptitle(f"{scope_label} (batch_size, num_indices_per_lookup) density before/after balance", fontsize=14.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _rebalance_train_distribution(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    label_column: str,
    label_bins: int,
    bucket_cap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_train = frame[frame["split"] == "train"].copy()
    working = before_train.copy()
    working["source_family"] = _derive_source_family(working["source_name"])
    label = pd.to_numeric(working[label_column], errors="coerce").clip(lower=1e-6)
    working["label_bin"] = pd.qcut(np.log10(label), q=label_bins, duplicates="drop", labels=False)
    sampled_parts: list[pd.DataFrame] = []
    bucket_rows: list[dict[str, Any]] = []
    for (source_family, op_type, label_bin), group in working.groupby(["source_family", "op_type", "label_bin"], sort=True):
        target_rows = min(len(group), int(bucket_cap))
        sampled = group.sample(n=target_rows, random_state=seed) if len(group) > target_rows else group.copy()
        sampled_parts.append(sampled.drop(columns=["source_family", "label_bin"], errors="ignore"))
        bucket_rows.append(
            {
                "source_family": source_family,
                "op_type": op_type,
                "label_bin": int(label_bin) if pd.notna(label_bin) else -1,
                "before_rows": int(len(group)),
                "after_rows": int(target_rows),
            }
        )
    after_train = pd.concat(sampled_parts, ignore_index=False, sort=False).sort_index(kind="stable")
    preserved = frame[frame["split"] != "train"].copy()
    balanced = pd.concat([after_train, preserved], ignore_index=False, sort=False)
    balanced = balanced.sort_values("row_uid", kind="stable").reset_index(drop=True)

    plots_root = output_root / "figures"
    _plot_balance_histogram(before_train, after_train, plots_root / "train_label_distribution_before_after.png")
    _plot_balance_source_op(before_train, after_train, plots_root / "train_source_optype_before_after.png")
    _plot_balance_pair_heatmap(before_train, after_train, plots_root / "train_batch_nip_before_after.png")

    bucket_df = pd.DataFrame(bucket_rows).sort_values(["source_family", "op_type", "label_bin"], kind="stable")
    bucket_csv = output_root / "bucket_counts_before_after.csv"
    bucket_df.to_csv(bucket_csv, index=False)
    manifest = {
        "label_column": label_column,
        "label_bins": int(label_bins),
        "bucket_cap": int(bucket_cap),
        "before_train_rows": int(len(before_train)),
        "after_train_rows": int(len(after_train)),
        "before_source_family_counts": {k: int(v) for k, v in _derive_source_family(before_train["source_name"]).value_counts().sort_index().to_dict().items()},
        "after_source_family_counts": {k: int(v) for k, v in _derive_source_family(after_train["source_name"]).value_counts().sort_index().to_dict().items()},
        "bucket_counts_csv": str(bucket_csv),
        "figures": {
            "label_distribution": str(plots_root / "train_label_distribution_before_after.png"),
            "source_optype": str(plots_root / "train_source_optype_before_after.png"),
            "batch_nip_heatmap": str(plots_root / "train_batch_nip_before_after.png"),
        },
    }
    _write_json(output_root / "rebalance_manifest.json", manifest)
    return balanced, manifest


def _prepare_balance_frame(
    frame: pd.DataFrame,
    *,
    label_column: str,
    label_bins: int,
) -> pd.DataFrame:
    working = frame.copy()
    working["_balance_source_family"] = _derive_source_family(working["source_name"])
    label = pd.to_numeric(working[label_column], errors="coerce").clip(lower=1e-6)
    working["_balance_label_bin"] = pd.qcut(np.log10(label), q=label_bins, duplicates="drop", labels=False)
    working["_balance_label_bin"] = pd.to_numeric(working["_balance_label_bin"], errors="coerce").fillna(-1).astype(int)
    return working


def _resolve_balance_bucket_columns(bucket_columns: list[str]) -> list[str]:
    mapping = {
        "source_family": "_balance_source_family",
        "label_bin": "_balance_label_bin",
    }
    return [mapping.get(str(column), str(column)) for column in bucket_columns]


def _cap_frame_by_bucket(
    frame: pd.DataFrame,
    *,
    bucket_columns: list[str],
    bucket_cap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled_parts: list[pd.DataFrame] = []
    bucket_rows: list[dict[str, Any]] = []
    for bucket_idx, (bucket_key, group) in enumerate(frame.groupby(bucket_columns, sort=True, dropna=False)):
        if not isinstance(bucket_key, tuple):
            bucket_key = (bucket_key,)
        target_rows = min(len(group), int(bucket_cap))
        sampled = group.sample(n=target_rows, random_state=seed + bucket_idx) if len(group) > target_rows else group.copy()
        sampled_parts.append(sampled)
        bucket_row = {"before_rows": int(len(group)), "after_rows": int(target_rows)}
        for column_name, value in zip(bucket_columns, bucket_key):
            bucket_row[column_name] = value
        bucket_rows.append(bucket_row)
    balanced = pd.concat(sampled_parts, ignore_index=False, sort=False).sort_index(kind="stable")
    bucket_df = pd.DataFrame(bucket_rows).sort_values(bucket_columns, kind="stable").reset_index(drop=True)
    return balanced, bucket_df


def _assign_train_val_test_by_bucket(
    frame: pd.DataFrame,
    *,
    bucket_columns: list[str],
    outer_train_ratio: float,
    val_fraction_of_train: float,
    seed: int,
) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = "unused"
    for bucket_idx, (_, bucket_df) in enumerate(out.groupby(bucket_columns, sort=True, dropna=False)):
        group_series = bucket_df["sample_group"].astype(str) if "sample_group" in bucket_df.columns else bucket_df.index.astype(str)
        unique_groups = list(dict.fromkeys(group_series.tolist()))
        group_count = len(unique_groups)
        if group_count <= 1:
            out.loc[bucket_df.index, "split"] = "train"
            continue
        if group_count == 2:
            train_group = unique_groups[0]
            test_group = unique_groups[1]
            out.loc[group_series[group_series == train_group].index, "split"] = "train"
            out.loc[group_series[group_series == test_group].index, "split"] = "test"
            continue

        train_outer_mask, test_mask = _group_split_assign(
            bucket_df,
            train_ratio=outer_train_ratio,
            seed=seed + bucket_idx * 2,
            stratify_column=None,
        )
        test_index = test_mask[test_mask].index
        train_outer_index = train_outer_mask[train_outer_mask].index
        out.loc[test_index, "split"] = "test"
        train_outer = bucket_df.loc[train_outer_index].copy()
        inner_groups = train_outer["sample_group"].astype(str).nunique() if "sample_group" in train_outer.columns else int(len(train_outer))
        if inner_groups <= 1:
            out.loc[train_outer.index, "split"] = "train"
            continue
        inner_train_mask, val_mask = _group_split_assign(
            train_outer,
            train_ratio=1.0 - val_fraction_of_train,
            seed=seed + bucket_idx * 2 + 1,
            stratify_column=None,
        )
        out.loc[inner_train_mask[inner_train_mask].index, "split"] = "train"
        out.loc[val_mask[val_mask].index, "split"] = "val"
    if (out["split"] == "unused").any():
        out.loc[out["split"] == "unused", "split"] = "train"
    return out


def _prebalance_dataset_distribution(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    label_column: str,
    label_bins: int,
    bucket_cap: int,
    bucket_columns: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = _prepare_balance_frame(frame, label_column=label_column, label_bins=label_bins)
    resolved_bucket_columns = _resolve_balance_bucket_columns(bucket_columns)
    balanced, bucket_df = _cap_frame_by_bucket(
        working,
        bucket_columns=resolved_bucket_columns,
        bucket_cap=bucket_cap,
        seed=seed,
    )
    plots_root = output_root / "figures"
    _plot_balance_histogram(working, balanced, plots_root / "dataset_label_distribution_before_after.png", scope_label="Dataset")
    _plot_balance_source_op(working, balanced, plots_root / "dataset_source_optype_before_after.png", scope_label="Dataset")
    _plot_balance_pair_heatmap(working, balanced, plots_root / "dataset_batch_nip_before_after.png", scope_label="Dataset")
    bucket_csv = output_root / "bucket_counts_before_after.csv"
    bucket_df.to_csv(bucket_csv, index=False)
    manifest = {
        "label_column": label_column,
        "label_bins": int(label_bins),
        "bucket_cap": int(bucket_cap),
        "bucket_column_names": list(bucket_columns),
        "resolved_bucket_columns": list(resolved_bucket_columns),
        "before_rows": int(len(working)),
        "after_rows": int(len(balanced)),
        "before_source_family_counts": {k: int(v) for k, v in working["_balance_source_family"].value_counts().sort_index().to_dict().items()},
        "after_source_family_counts": {k: int(v) for k, v in balanced["_balance_source_family"].value_counts().sort_index().to_dict().items()},
        "bucket_counts_csv": str(bucket_csv),
        "figures": {
            "label_distribution": str(plots_root / "dataset_label_distribution_before_after.png"),
            "source_optype": str(plots_root / "dataset_source_optype_before_after.png"),
            "batch_nip_heatmap": str(plots_root / "dataset_batch_nip_before_after.png"),
        },
    }
    _write_json(output_root / "rebalance_manifest.json", manifest)
    cleaned = balanced.drop(columns=["_balance_source_family", "_balance_label_bin"], errors="ignore").reset_index(drop=True)
    return cleaned, manifest


def _build_exp1(root: Path, *, preprocess_config: dict[str, Any] | None = None) -> dict[str, Any]:
    standalone = _apply_dataset_preprocessing(_load_base_dataset(STANDALONE_ROOT, "standalone_merge_ready"), preprocess_config)
    dlrm = _apply_dataset_preprocessing(_load_base_dataset(DLRM_ROOT, "dlrm_original"), preprocess_config)
    standalone_split = _assign_train_val_test(
        standalone,
        outer_train_ratio=TRAIN_RATIO,
        seed=SEED,
        val_fraction_of_train=VAL_FRACTION_OF_TRAIN,
    )
    train_root = root / "dataset_train_protocol"
    manifest = {
        "experiment_id": "exp1_standalone70_vs_holdout_plus_dlrm_all",
        "outer_train_ratio": TRAIN_RATIO,
        "val_fraction_of_outer_train": VAL_FRACTION_OF_TRAIN,
        "seed": SEED,
        "train_source": str(STANDALONE_ROOT),
        "external_eval_source": str(DLRM_ROOT),
        "preprocess_config": preprocess_config or {},
        "description": EXPERIMENT_SPECS["exp1_standalone70_vs_holdout_plus_dlrm_all"]["description"],
    }
    _write_dataset_root(standalone_split, train_root, manifest_payload=manifest)

    eval_root = root / "eval_inputs" / "raw"
    eval_root.mkdir(parents=True, exist_ok=True)
    standalone_holdout = standalone_split[standalone_split["split"] == "test"].copy()
    standalone_holdout.to_csv(eval_root / "standalone_holdout.csv", index=False)
    dlrm_all = dlrm.copy()
    dlrm_all["split"] = "test"
    dlrm_all.to_csv(eval_root / "dlrm_all.csv", index=False)
    return {
        "dataset_root": train_root,
        "eval_raw_csvs": {
            "standalone_holdout": eval_root / "standalone_holdout.csv",
            "dlrm_all": eval_root / "dlrm_all.csv",
        },
    }


def _build_union_dataset(
    root: Path,
    experiment_id: str,
    description: str,
    include_dlrm: bool,
    *,
    preprocess_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    standalone = _apply_dataset_preprocessing(_load_base_dataset(STANDALONE_ROOT, "standalone_merge_ready"), preprocess_config)
    frames = [standalone]
    sources = [str(STANDALONE_ROOT)]
    if include_dlrm:
        dlrm = _apply_dataset_preprocessing(_load_base_dataset(DLRM_ROOT, "dlrm_original"), preprocess_config)
        frames.append(dlrm)
        sources.append(str(DLRM_ROOT))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged_split = _assign_train_val_test(
        merged,
        outer_train_ratio=TRAIN_RATIO,
        seed=SEED,
        val_fraction_of_train=VAL_FRACTION_OF_TRAIN,
    )
    dataset_root = root / "dataset"
    manifest = {
        "experiment_id": experiment_id,
        "outer_train_ratio": TRAIN_RATIO,
        "val_fraction_of_outer_train": VAL_FRACTION_OF_TRAIN,
        "seed": SEED,
        "source_roots": sources,
        "preprocess_config": preprocess_config or {},
        "description": description,
    }
    _write_dataset_root(merged_split, dataset_root, manifest_payload=manifest)
    eval_root = root / "eval_inputs" / "raw"
    eval_paths = _write_eval_raw_csvs(merged_split, eval_root, include_dlrm=include_dlrm)
    return {
        "dataset_root": dataset_root,
        "eval_raw_csvs": eval_paths,
    }


def _build_exp2_semantic_free_balanced_fixed_analytical(root: Path) -> dict[str, Any]:
    spec = _experiment_config("exp2_union_70_30_semantic_free_balanced_fixed_analytical")
    preprocess_config = {"neutralize_graph_semantics": bool(spec.get("neutralize_graph_semantics", False))}
    prepared = _build_union_dataset(
        root / "_tmp_unbalanced",
        "exp2_union_70_30_semantic_free_balanced_fixed_analytical",
        spec["description"],
        True,
        preprocess_config=preprocess_config,
    )
    unbalanced_root = Path(prepared["dataset_root"])
    unbalanced_frame = pd.read_csv(unbalanced_root / "dataset_full.csv", low_memory=False)
    rebalance_cfg = dict(spec.get("rebalance_train", {}))
    balancing_root = root / "distribution_control"
    balanced_frame, rebalance_manifest = _rebalance_train_distribution(
        unbalanced_frame,
        output_root=balancing_root,
        label_column=str(rebalance_cfg.get("label_column", "label_operator_actual_dur_us")),
        label_bins=int(rebalance_cfg.get("label_bins", 8)),
        bucket_cap=int(rebalance_cfg.get("bucket_cap", 512)),
        seed=SEED,
    )
    dataset_root = root / "dataset"
    manifest = {
        "experiment_id": "exp2_union_70_30_semantic_free_balanced_fixed_analytical",
        "outer_train_ratio": TRAIN_RATIO,
        "val_fraction_of_outer_train": VAL_FRACTION_OF_TRAIN,
        "seed": SEED,
        "source_roots": [str(STANDALONE_ROOT), str(DLRM_ROOT)],
        "preprocess_config": preprocess_config,
        "rebalance_manifest": str(balancing_root / "rebalance_manifest.json"),
        "description": spec["description"],
    }
    _write_dataset_root(balanced_frame, dataset_root, manifest_payload=manifest)
    eval_root = root / "eval_inputs" / "raw"
    eval_paths = _write_eval_raw_csvs(balanced_frame, eval_root, include_dlrm=True)
    return {
        "dataset_root": dataset_root,
        "dataset_root_unbalanced": str(unbalanced_root),
        "distribution_control_root": str(balancing_root),
        "eval_raw_csvs": eval_paths,
        "preprocess_config": preprocess_config,
        "rebalance_manifest": rebalance_manifest,
    }


def _build_exp2_semantic_free_prebalanced_fixed_analytical(root: Path) -> dict[str, Any]:
    experiment_id = "exp2_union_70_30_semantic_free_prebalanced_fixed_analytical"
    spec = _experiment_config(experiment_id)
    preprocess_config = {"neutralize_graph_semantics": bool(spec.get("neutralize_graph_semantics", False))}
    standalone = _apply_dataset_preprocessing(_load_base_dataset(STANDALONE_ROOT, "standalone_merge_ready"), preprocess_config)
    dlrm = _apply_dataset_preprocessing(_load_base_dataset(DLRM_ROOT, "dlrm_original"), preprocess_config)
    merged = pd.concat([standalone, dlrm], ignore_index=True, sort=False)

    unbalanced_root = root / "_tmp_unbalanced" / "dataset"
    unbalanced_split = _assign_train_val_test(
        merged,
        outer_train_ratio=TRAIN_RATIO,
        seed=SEED,
        val_fraction_of_train=VAL_FRACTION_OF_TRAIN,
    )
    _write_dataset_root(
        unbalanced_split,
        unbalanced_root,
        manifest_payload={
            "experiment_id": experiment_id,
            "description": spec["description"],
            "mode": "prebalance_reference_unbalanced_split",
            "source_roots": [str(STANDALONE_ROOT), str(DLRM_ROOT)],
            "preprocess_config": preprocess_config,
            "seed": SEED,
            "outer_train_ratio": TRAIN_RATIO,
            "val_fraction_of_outer_train": VAL_FRACTION_OF_TRAIN,
        },
    )

    prebalance_cfg = dict(spec.get("prebalance_dataset", {}))
    balancing_root = root / "distribution_control"
    prebalanced_frame, rebalance_manifest = _prebalance_dataset_distribution(
        merged,
        output_root=balancing_root,
        label_column=str(prebalance_cfg.get("label_column", "label_operator_actual_dur_us")),
        label_bins=int(prebalance_cfg.get("label_bins", 8)),
        bucket_cap=int(prebalance_cfg.get("bucket_cap", 512)),
        bucket_columns=[str(value) for value in prebalance_cfg.get("bucket_columns", ["source_family", "op_type", "label_bin"])],
        seed=SEED,
    )
    bucket_columns = _resolve_balance_bucket_columns([str(value) for value in prebalance_cfg.get("bucket_columns", ["source_family", "op_type", "label_bin"])])
    balanced_working = _prepare_balance_frame(
        prebalanced_frame,
        label_column=str(prebalance_cfg.get("label_column", "label_operator_actual_dur_us")),
        label_bins=int(prebalance_cfg.get("label_bins", 8)),
    )
    split_frame = _assign_train_val_test_by_bucket(
        balanced_working,
        bucket_columns=bucket_columns,
        outer_train_ratio=TRAIN_RATIO,
        val_fraction_of_train=VAL_FRACTION_OF_TRAIN,
        seed=SEED,
    ).drop(columns=["_balance_source_family", "_balance_label_bin"], errors="ignore")

    dataset_root = root / "dataset"
    manifest = {
        "experiment_id": experiment_id,
        "outer_train_ratio": TRAIN_RATIO,
        "val_fraction_of_outer_train": VAL_FRACTION_OF_TRAIN,
        "seed": SEED,
        "source_roots": [str(STANDALONE_ROOT), str(DLRM_ROOT)],
        "preprocess_config": preprocess_config,
        "rebalance_manifest": str(balancing_root / "rebalance_manifest.json"),
        "description": spec["description"],
        "balance_mode": "pre_split_matching_balance",
    }
    _write_dataset_root(split_frame, dataset_root, manifest_payload=manifest)
    eval_root = root / "eval_inputs" / "raw"
    eval_paths = _write_eval_raw_csvs(split_frame, eval_root, include_dlrm=True)
    return {
        "dataset_root": dataset_root,
        "dataset_root_unbalanced": str(unbalanced_root),
        "distribution_control_root": str(balancing_root),
        "eval_raw_csvs": eval_paths,
        "preprocess_config": preprocess_config,
        "rebalance_manifest": rebalance_manifest,
    }


def prepare_experiment_datasets(output_root: Path, experiment_id: str | None = None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    selected = [experiment_id] if experiment_id else list(EXPERIMENT_SPECS.keys())
    manifest: dict[str, Any] = {
        "output_root": str(output_root),
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_fraction_of_train": VAL_FRACTION_OF_TRAIN,
        "experiments": {},
    }
    for exp_id in selected:
        exp_root = output_root / exp_id
        spec = _experiment_config(exp_id)
        preprocess_config = {"neutralize_graph_semantics": bool(spec.get("neutralize_graph_semantics", False))}
        if exp_id == "exp1_standalone70_vs_holdout_plus_dlrm_all":
            payload = _build_exp1(exp_root, preprocess_config=preprocess_config)
        elif exp_id == "exp2_union_70_30":
            payload = _build_union_dataset(
                exp_root,
                experiment_id=exp_id,
                description=EXPERIMENT_SPECS[exp_id]["description"],
                include_dlrm=True,
                preprocess_config=preprocess_config,
            )
        elif exp_id == "exp2_union_70_30_target_push":
            payload = _build_union_dataset(
                exp_root,
                experiment_id=exp_id,
                description=EXPERIMENT_SPECS[exp_id]["description"],
                include_dlrm=True,
                preprocess_config=preprocess_config,
            )
        elif exp_id == "exp3_standalone_only_70_30":
            payload = _build_union_dataset(
                exp_root,
                experiment_id=exp_id,
                description=EXPERIMENT_SPECS[exp_id]["description"],
                include_dlrm=False,
                preprocess_config=preprocess_config,
            )
        elif exp_id == "exp2_union_70_30_semantic_free_balanced_fixed_analytical":
            payload = _build_exp2_semantic_free_balanced_fixed_analytical(exp_root)
        elif exp_id == "exp2_union_70_30_semantic_free_prebalanced_fixed_analytical":
            payload = _build_exp2_semantic_free_prebalanced_fixed_analytical(exp_root)
        else:
            raise KeyError(exp_id)
        manifest["experiments"][exp_id] = {
            "dataset_root": str(payload["dataset_root"]),
            "eval_raw_csvs": {name: str(path) for name, path in payload["eval_raw_csvs"].items()},
            "description": EXPERIMENT_SPECS[exp_id]["description"],
            "preprocess_config": preprocess_config,
        }
        if "dataset_root_unbalanced" in payload:
            manifest["experiments"][exp_id]["dataset_root_unbalanced"] = str(payload["dataset_root_unbalanced"])
        if "distribution_control_root" in payload:
            manifest["experiments"][exp_id]["distribution_control_root"] = str(payload["distribution_control_root"])
    _write_json(output_root / "prepared_manifest.json", manifest)
    return manifest


def _train_analytical(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    manifest = build_analytical_features.build_full_feature_artifacts(
        dataset_root / "dataset_full.csv",
        output_dir,
        passes=ANALYTICAL_PASSES,
    )
    _write_json(output_dir / "summary.json", manifest)
    return manifest


def _build_fixed_analytical_artifacts(
    dataset_root: Path,
    output_dir: Path,
    *,
    params_json: Path,
) -> dict[str, Any]:
    if not params_json.exists():
        raise FileNotFoundError(params_json)
    raw_df = pd.read_csv(dataset_root / "dataset_full.csv", low_memory=False)
    params = _read_json(params_json)
    rebuilt_df = build_analytical_features.rebuild_local_features(raw_df)
    heavy_df = build_analytical_features.prepare_heavy_prediction_frame(rebuilt_df)
    feature_df = build_analytical_features.add_calibrated_analytical_columns(rebuilt_df, heavy_df, params)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_csv = output_dir / "analytical_features_full.csv"
    feature_df.to_csv(feature_csv, index=False)
    deploy_dir = output_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    deploy_params = deploy_dir / "full_data_parameters.json"
    deploy_params.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "input_csv": str(dataset_root / "dataset_full.csv"),
        "output_dir": str(output_dir),
        "mode": "fixed_parameters",
        "fixed_parameters_source": str(params_json),
        "artifacts": {
            "deploy_parameters_json": str(deploy_params),
            "feature_csv": str(feature_csv),
        },
        "row_count": int(len(feature_df)),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "summary.json", manifest)
    return manifest


def _merge_raw_with_analytical(raw_frame: pd.DataFrame, analytical_csv: Path) -> pd.DataFrame:
    analytical_df = pd.read_csv(analytical_csv, low_memory=False)
    keep_columns = [column for column in analytical_df.columns if column.startswith("ana_calib_") or column in {"row_uid", "op_class"} or column.startswith("feat_gemm_")]
    analytical_df = analytical_df[keep_columns].copy()
    merged = raw_frame.drop(columns=[column for column in keep_columns if column != "row_uid" and column in raw_frame.columns], errors="ignore").merge(
        analytical_df,
        on="row_uid",
        how="left",
        validate="one_to_one",
    )
    return merged


def _prepare_augmented_dataset(dataset_root: Path, analytical_root: Path, output_csv: Path) -> Path:
    raw = pd.read_csv(dataset_root / "dataset_full.csv", low_memory=False)
    merged = _merge_raw_with_analytical(raw, analytical_root / "analytical_features_full.csv")
    derived = derive_enhanced_features(merged)
    augmented = pd.concat([merged.reset_index(drop=True), derived.reset_index(drop=True)], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_csv, index=False)
    return output_csv


def _build_blend_frame(
    augmented_csv: Path,
    prediction_csvs: dict[str, Path],
    *,
    candidate_models: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(augmented_csv, low_memory=False)
    blend = frame[["row_uid", "op_type", "split", "label_operator_actual_dur_us"]].copy()
    blend["target_us"] = pd.to_numeric(blend["label_operator_actual_dur_us"], errors="coerce").fillna(0.0).astype(float)
    for model_name, path in prediction_csvs.items():
        pred_frame = pd.read_csv(path, low_memory=False)
        part = pred_frame[["row_uid", "pred_us"]].copy()
        part.rename(columns={"pred_us": f"pred_{model_name}"}, inplace=True)
        blend = blend.merge(part, on="row_uid", how="left", validate="one_to_one")
    candidate_pred_columns = [f"pred_{name}" for name in candidate_models if f"pred_{name}" in blend.columns]
    for column in candidate_pred_columns:
        blend[column] = pd.to_numeric(blend[column], errors="coerce").fillna(0.0).astype(float)
    if candidate_pred_columns:
        pred_block = blend[candidate_pred_columns]
        blend["blend_candidate_pred_min"] = pred_block.min(axis=1)
        blend["blend_candidate_pred_max"] = pred_block.max(axis=1)
        blend["blend_candidate_pred_span"] = blend["blend_candidate_pred_max"] - blend["blend_candidate_pred_min"]
        blend["blend_candidate_pred_mean"] = pred_block.mean(axis=1)
        blend["blend_candidate_pred_std"] = pred_block.std(axis=1).fillna(0.0)
        blend["blend_candidate_pred_cv"] = blend["blend_candidate_pred_std"] / blend["blend_candidate_pred_mean"].replace(0.0, 1.0)
    else:
        for column in [
            "blend_candidate_pred_min",
            "blend_candidate_pred_max",
            "blend_candidate_pred_span",
            "blend_candidate_pred_mean",
            "blend_candidate_pred_std",
            "blend_candidate_pred_cv",
        ]:
            blend[column] = 0.0
    blend["blend_pred_global_minus_family"] = (
        pd.to_numeric(blend.get("pred_global_enhanced", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(blend.get("pred_family_enhanced", 0.0), errors="coerce").fillna(0.0)
    )
    blend["blend_pred_classed_minus_single"] = (
        pd.to_numeric(blend.get("pred_original_classed_mlp", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(blend.get("pred_original_single_mlp", 0.0), errors="coerce").fillna(0.0)
    )
    blend["blend_pred_enhanced_classed_minus_classed"] = (
        pd.to_numeric(blend.get("pred_enhanced_classed_mlp", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(blend.get("pred_original_classed_mlp", 0.0), errors="coerce").fillna(0.0)
    )
    augmented = frame.drop(columns=["label_operator_actual_dur_us"], errors="ignore")
    return blend.merge(augmented, on=["row_uid", "op_type", "split"], how="left", validate="one_to_one")


def _fit_router_encoder(
    frame: pd.DataFrame,
    *,
    categorical_features: list[str],
    numeric_features: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    encoded = _ensure_required_columns(frame, categorical_features=categorical_features, numeric_features=numeric_features)
    category_maps: dict[str, dict[str, int]] = {}
    for column in categorical_features:
        values = encoded[column].fillna("missing").astype(str)
        unique_values = sorted(pd.Index(values).unique())
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        category_maps[column] = mapping
        encoded[column] = values.map(mapping).fillna(-1).astype(int)
    return encoded, category_maps


def _apply_router_encoder(
    frame: pd.DataFrame,
    *,
    categorical_features: list[str],
    numeric_features: list[str],
    category_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    encoded = _ensure_required_columns(frame, categorical_features=categorical_features, numeric_features=numeric_features)
    for column in categorical_features:
        mapping = {str(key): int(value) for key, value in category_maps.get(column, {}).items()}
        values = encoded[column].fillna("missing").astype(str)
        encoded[column] = values.map(mapping).fillna(-1).astype(int)
    return encoded


def _classed_infer_custom(
    *,
    classed_data_root: Path,
    model_root: Path,
    input_csv: Path,
    output_csv: Path,
) -> dict[str, Any]:
    dataset_summary = load_dataset_summary(classed_data_root)
    op_type_model_group_map = dict(dataset_summary.get("op_type_model_group_map", {}))
    frame = pd.read_csv(input_csv, low_memory=False)
    frame["model_group"] = frame["op_type"].astype(str).map(op_type_model_group_map)
    rows: list[pd.DataFrame] = []
    work_dir = output_csv.parent / "_tmp_classed_infer"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    for model_group in dataset_summary.get("model_group_order", []):
        group_df = frame[frame["model_group"] == model_group].copy()
        if group_df.empty:
            continue
        group_input = work_dir / f"{model_group}.csv"
        group_output = work_dir / f"{model_group}.pred.csv"
        group_df.to_csv(group_input, index=False)
        run_inference(
            model_dir=model_root / model_group,
            input_csv=group_input,
            output_csv=group_output,
            provider="cpu",
            batch_size=4096,
        )
        pred_df = pd.read_csv(group_output, low_memory=False)
        rows.append(pred_df)
    if not rows:
        raise RuntimeError(f"No routed rows found for {input_csv}")
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = combined.sort_values("row_uid", kind="stable").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    metrics = evaluate_prediction_frame(combined)
    return {
        "output_csv": str(output_csv),
        "metrics": metrics,
    }


def _rebuild_enhanced_classed_dataset(original_classed_root: Path, output_root: Path) -> dict[str, Any]:
    shutil.rmtree(output_root, ignore_errors=True)
    shutil.copytree(original_classed_root, output_root)
    full_df = pd.read_csv(output_root / "classed_dataset_full.csv", low_memory=False)
    full_df = pd.concat([full_df.reset_index(drop=True), derive_enhanced_features(full_df).reset_index(drop=True)], axis=1)
    full_df.to_csv(output_root / "classed_dataset_full.csv", index=False)

    dataset_summary = _read_json(output_root / "dataset_summary.json")
    per_group_numeric = dataset_summary.get("per_model_group_numeric_features", {})
    for model_group in dataset_summary.get("model_group_order", []):
        group_dir = output_root / "datasets" / model_group
        feature_manifest = _read_json(group_dir / "feature_columns.json")
        extra_features = ENHANCED_CLASSED_GROUP_FEATURES.get(model_group, [])
        feature_manifest["numeric_features"] = list(dict.fromkeys(list(feature_manifest.get("numeric_features", [])) + extra_features))
        feature_manifest["analysis_numeric_features"] = list(feature_manifest["numeric_features"])
        _write_json(group_dir / "feature_columns.json", feature_manifest)
        for split_name in ["dataset_full", "train", "val", "test"]:
            split_path = group_dir / f"{split_name}.csv"
            split_df = pd.read_csv(split_path, low_memory=False)
            split_df = pd.concat([split_df.reset_index(drop=True), derive_enhanced_features(split_df).reset_index(drop=True)], axis=1)
            split_df.to_csv(split_path, index=False)
        per_group_numeric[model_group] = feature_manifest["numeric_features"]
    dataset_summary["per_model_group_numeric_features"] = per_group_numeric
    dataset_summary["enhanced_feature_groups"] = ENHANCED_CLASSED_GROUP_FEATURES
    _write_json(output_root / "dataset_summary.json", dataset_summary)
    return dataset_summary


def _build_global_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=SEED,
    )
    return Pipeline([("pre", preprocessor), ("model", model)])


def _fit_global_enhanced(augmented_df: pd.DataFrame, output_root: Path) -> dict[str, Any]:
    categorical = _existing_columns(augmented_df, BASE_ENHANCED_CATEGORICAL_FEATURES)
    numeric = _existing_columns(augmented_df, BASE_ENHANCED_NUMERIC_FEATURES)
    features = categorical + numeric
    train_df = augmented_df[augmented_df["split"] == "train"].copy()
    model = _build_global_pipeline(categorical, numeric)
    y_train = np.log1p(train_df["label_operator_actual_dur_us"].to_numpy(dtype=float))
    with threadpool_limits(limits=1):
        model.fit(train_df[features], y_train)
    output_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_root / "global_enhanced.joblib")
    manifest = {
        "model_type": "hist_gradient_boosting_global",
        "categorical_features": categorical,
        "numeric_features": numeric,
        "artifact": str(output_root / "global_enhanced.joblib"),
        "seed": SEED,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _predict_global_enhanced(model_root: Path, augmented_csv: Path, output_csv: Path) -> dict[str, Any]:
    model = joblib.load(model_root / "global_enhanced.joblib")
    manifest = _read_json(model_root / "manifest.json")
    frame = pd.read_csv(augmented_csv, low_memory=False)
    frame = _ensure_required_columns(
        frame,
        categorical_features=list(manifest["categorical_features"]),
        numeric_features=list(manifest["numeric_features"]),
    )
    pred = np.expm1(model.predict(frame[manifest["categorical_features"] + manifest["numeric_features"]]))
    output = frame[["row_uid", "op_type", "split", "label_operator_actual_dur_us"]].copy()
    output["target_us"] = output["label_operator_actual_dur_us"]
    output["pred_us"] = np.clip(pred, 1e-6, None)
    output["abs_error_us"] = np.abs(output["pred_us"] - output["target_us"])
    output["ape"] = output["abs_error_us"] / np.clip(output["target_us"], 1e-9, None)
    output.to_csv(output_csv, index=False)
    return {"output_csv": str(output_csv), "metrics": evaluate_prediction_frame(output)}


def _fit_family_enhanced(augmented_df: pd.DataFrame, output_root: Path) -> dict[str, Any]:
    numeric = _existing_columns(augmented_df, BASE_ENHANCED_NUMERIC_FEATURES)
    train_df = augmented_df[augmented_df["split"] == "train"].copy()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "model_type": "hist_gradient_boosting_family",
        "numeric_features": numeric,
        "families": {},
        "seed": SEED,
    }
    with threadpool_limits(limits=1):
        for op_type, group in train_df.groupby("op_type", sort=True):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=8,
                max_iter=220,
                min_samples_leaf=12,
                random_state=SEED,
            )
            x = group[numeric]
            y = np.log1p(group["label_operator_actual_dur_us"].to_numpy(dtype=float))
            model.fit(x, y)
            path = output_root / f"{op_type}.joblib"
            joblib.dump(model, path)
            manifest["families"][op_type] = {"artifact": str(path)}
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _predict_family_enhanced(model_root: Path, augmented_csv: Path, output_csv: Path) -> dict[str, Any]:
    manifest = _read_json(model_root / "manifest.json")
    numeric = manifest["numeric_features"]
    frame = pd.read_csv(augmented_csv, low_memory=False)
    frame = _ensure_required_columns(frame, numeric_features=list(numeric))
    rows: list[pd.DataFrame] = []
    for op_type, group in frame.groupby("op_type", sort=True):
        artifact = manifest["families"].get(op_type, {}).get("artifact")
        if not artifact:
            continue
        model = joblib.load(artifact)
        pred = np.expm1(model.predict(group[numeric]))
        part = group[["row_uid", "op_type", "split", "label_operator_actual_dur_us"]].copy()
        part["target_us"] = part["label_operator_actual_dur_us"]
        part["pred_us"] = np.clip(pred, 1e-6, None)
        part["abs_error_us"] = np.abs(part["pred_us"] - part["target_us"])
        part["ape"] = part["abs_error_us"] / np.clip(part["target_us"], 1e-9, None)
        rows.append(part)
    output = pd.concat(rows, ignore_index=True, sort=False).sort_values("row_uid", kind="stable").reset_index(drop=True)
    output.to_csv(output_csv, index=False)
    return {"output_csv": str(output_csv), "metrics": evaluate_prediction_frame(output)}


def _metric_bundle_from_pred(target_us: np.ndarray, pred_us: np.ndarray) -> dict[str, float]:
    clipped_pred = np.clip(np.asarray(pred_us, dtype=float), 1e-6, None)
    target = np.asarray(target_us, dtype=float)
    ape = np.abs(clipped_pred - target) / np.clip(target, 1e-9, None)
    return {
        "rows": int(len(target)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.quantile(ape, 0.5)),
        "p90_ape": float(np.quantile(ape, 0.9)),
        "p95_ape": float(np.quantile(ape, 0.95)),
    }


def _topk_oracle_metrics(group: pd.DataFrame, shortlist: list[str]) -> dict[str, float]:
    ape_columns = [f"ape_{name}" for name in shortlist if f"ape_{name}" in group.columns]
    if not ape_columns:
        return {"rows": int(len(group)), "mape": 0.0, "median_ape": 0.0, "p90_ape": 0.0, "p95_ape": 0.0}
    ape = np.minimum.reduce([pd.to_numeric(group[column], errors="coerce").fillna(0.0).to_numpy(dtype=float) for column in ape_columns])
    return {
        "rows": int(len(group)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.quantile(ape, 0.5)),
        "p90_ape": float(np.quantile(ape, 0.9)),
        "p95_ape": float(np.quantile(ape, 0.95)),
    }


def _choose_best_strategy(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No router strategy candidates were produced.")
    return min(
        candidates,
        key=lambda item: (
            float(item["metrics"]["p90_ape"]),
            float(item["metrics"]["mape"]),
            float(item["metrics"]["median_ape"]),
        ),
    )


def _safe_holdout_split(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(group) < 192:
        return group.copy(), group.copy()
    holdout_size = max(48, int(round(len(group) * 0.2)))
    holdout_size = min(holdout_size, len(group) - 96)
    if holdout_size <= 0:
        return group.copy(), group.copy()
    train_index, holdout_index = train_test_split(
        group.index.to_numpy(),
        test_size=holdout_size,
        random_state=SEED,
        shuffle=True,
    )
    return group.loc[train_index].copy(), group.loc[holdout_index].copy()


def _router_training_weight(group: pd.DataFrame, default_model: str, shortlist: list[str]) -> np.ndarray:
    default_ape = pd.to_numeric(group[f"ape_{default_model}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    best_ape = np.minimum.reduce(
        [pd.to_numeric(group[f"ape_{name}"], errors="coerce").fillna(0.0).to_numpy(dtype=float) for name in shortlist]
    )
    improvement = np.maximum(default_ape - best_ape, 0.0)
    return 1.0 + 6.0 * improvement


def _build_classifier_router_candidate(
    *,
    group_train: pd.DataFrame,
    group_holdout: pd.DataFrame,
    feature_columns: list[str],
    shortlist: list[str],
    default_model: str,
) -> dict[str, Any] | None:
    label_frame = np.stack(
        [pd.to_numeric(group_train[f"ape_{name}"], errors="coerce").fillna(0.0).to_numpy(dtype=float) for name in shortlist],
        axis=1,
    )
    labels = label_frame.argmin(axis=1)
    if len(np.unique(labels)) <= 1:
        return None
    model = RandomForestClassifier(
        n_estimators=320,
        max_depth=12,
        min_samples_leaf=max(10, min(48, len(group_train) // 24)),
        random_state=SEED,
        n_jobs=8,
    )
    weights = _router_training_weight(group_train, default_model, shortlist)
    with threadpool_limits(limits=1):
        model.fit(group_train[feature_columns], labels, sample_weight=weights)
    predicted_labels = model.predict(group_holdout[feature_columns]).astype(int)
    chosen_models = [shortlist[int(label)] for label in predicted_labels]
    pred_us = np.array(
        [float(group_holdout.iloc[idx][f"pred_{model_name}"]) for idx, model_name in enumerate(chosen_models)],
        dtype=float,
    )
    return {
        "strategy": "rf_classifier",
        "artifact_payload": {
            "strategy": "rf_classifier",
            "model": model,
            "shortlist": list(shortlist),
            "feature_columns": list(feature_columns),
            "inverse_label_map": {idx: name for idx, name in enumerate(shortlist)},
        },
        "metrics": _metric_bundle_from_pred(
            pd.to_numeric(group_holdout["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            pred_us,
        ),
    }


def _build_error_router_candidate(
    *,
    group_train: pd.DataFrame,
    group_holdout: pd.DataFrame,
    feature_columns: list[str],
    shortlist: list[str],
    default_model: str,
    blend_mode: bool,
) -> dict[str, Any] | None:
    best_ape = np.minimum.reduce(
        [pd.to_numeric(group_train[f"ape_{name}"], errors="coerce").fillna(0.0).to_numpy(dtype=float) for name in shortlist]
    )
    models: dict[str, HistGradientBoostingRegressor] = {}
    with threadpool_limits(limits=1):
        for name in shortlist:
            candidate_ape = pd.to_numeric(group_train[f"ape_{name}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            weight = 1.0 + 4.0 * np.maximum(candidate_ape - best_ape, 0.0)
            regressor = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=3,
                max_iter=180,
                min_samples_leaf=max(20, min(80, len(group_train) // 18)),
                random_state=SEED,
            )
            regressor.fit(group_train[feature_columns], np.log1p(candidate_ape), sample_weight=weight)
            models[name] = regressor
    predicted_log_ape = np.stack([models[name].predict(group_holdout[feature_columns]) for name in shortlist], axis=1)
    predicted_ape = np.clip(np.expm1(predicted_log_ape), 1e-6, None)
    holdout_pred_block = np.stack(
        [pd.to_numeric(group_holdout[f"pred_{name}"], errors="coerce").fillna(0.0).to_numpy(dtype=float) for name in shortlist],
        axis=1,
    )
    if blend_mode:
        inv_error = 1.0 / predicted_ape
        blend_weights = inv_error / np.clip(inv_error.sum(axis=1, keepdims=True), 1e-9, None)
        pred_us = np.sum(blend_weights * holdout_pred_block, axis=1)
        selected_models = [shortlist[int(idx)] for idx in blend_weights.argmax(axis=1)]
        strategy_name = "error_blend"
    else:
        chosen_index = predicted_ape.argmin(axis=1)
        pred_us = holdout_pred_block[np.arange(len(group_holdout)), chosen_index]
        selected_models = [shortlist[int(idx)] for idx in chosen_index]
        strategy_name = "error_selector"
    return {
        "strategy": strategy_name,
        "artifact_payload": {
            "strategy": strategy_name,
            "shortlist": list(shortlist),
            "feature_columns": list(feature_columns),
            "models": models,
        },
        "metrics": _metric_bundle_from_pred(
            pd.to_numeric(group_holdout["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            pred_us,
        ),
    }


def _predict_router_strategy(
    *,
    group: pd.DataFrame,
    encoded_group: pd.DataFrame,
    family_cfg: dict[str, Any],
    global_default: str,
) -> tuple[list[str], np.ndarray]:
    default_model = str(family_cfg.get("default_model", global_default))
    artifact = family_cfg.get("artifact")
    if not artifact:
        selected_models = [default_model] * len(group)
        pred_us = pd.to_numeric(group.get(f"pred_{default_model}", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return selected_models, pred_us

    router_payload = joblib.load(artifact)
    strategy = str(router_payload.get("strategy", "rf_classifier"))
    shortlist = list(router_payload.get("shortlist", family_cfg.get("shortlist", [default_model])))
    model_features = list(router_payload.get("feature_columns", family_cfg.get("feature_columns", [])))
    feature_frame = encoded_group[model_features]
    pred_block = np.stack(
        [pd.to_numeric(group.get(f"pred_{name}", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) for name in shortlist],
        axis=1,
    )
    if strategy == "rf_classifier":
        inverse_label_map = {int(key): str(value) for key, value in router_payload["inverse_label_map"].items()}
        labels = router_payload["model"].predict(feature_frame).astype(int)
        selected_models = [inverse_label_map[int(label)] for label in labels]
        pred_us = np.array(
            [float(group.iloc[idx][f"pred_{model_name}"]) for idx, model_name in enumerate(selected_models)],
            dtype=float,
        )
        return selected_models, pred_us

    if strategy in {"error_selector", "error_blend"}:
        score_block = np.stack([router_payload["models"][name].predict(feature_frame) for name in shortlist], axis=1)
        predicted_ape = np.clip(np.expm1(score_block), 1e-6, None)
        if strategy == "error_blend":
            inv_error = 1.0 / predicted_ape
            blend_weights = inv_error / np.clip(inv_error.sum(axis=1, keepdims=True), 1e-9, None)
            pred_us = np.sum(blend_weights * pred_block, axis=1)
            selected_models = [shortlist[int(idx)] for idx in blend_weights.argmax(axis=1)]
            return selected_models, pred_us
        chosen_index = predicted_ape.argmin(axis=1)
        selected_models = [shortlist[int(idx)] for idx in chosen_index]
        pred_us = pred_block[np.arange(len(group)), chosen_index]
        return selected_models, pred_us

    selected_models = [default_model] * len(group)
    pred_us = pd.to_numeric(group.get(f"pred_{default_model}", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return selected_models, pred_us


def _route_routed_expert_predictions(
    blend_frame: pd.DataFrame,
    *,
    model_root: Path,
    manifest: dict[str, Any],
    apply_residual: bool,
) -> pd.DataFrame:
    categorical = list(manifest.get("categorical_features", []))
    numeric = list(manifest.get("numeric_features", []))
    encoded = _apply_router_encoder(
        blend_frame,
        categorical_features=categorical,
        numeric_features=numeric,
        category_maps=dict(manifest.get("category_maps", {})),
    )
    global_default = str(manifest.get("global_default_model", BASE_ENSEMBLE_CANDIDATES[0]))
    feature_columns = list(manifest.get("router_feature_columns", []))
    candidate_models = list(manifest.get("candidate_models", []))
    output_rows: list[pd.DataFrame] = []
    for op_type, group in blend_frame.groupby("op_type", sort=True):
        encoded_group = encoded.loc[group.index]
        family_cfg = dict(manifest.get("families", {}).get(str(op_type), {}))
        selected_models, pred_us = _predict_router_strategy(
            group=group,
            encoded_group=encoded_group,
            family_cfg=family_cfg,
            global_default=global_default,
        )
        part = group[["row_uid", "op_type", "split", "target_us"]].copy()
        part["selected_model"] = selected_models
        part["pred_us"] = pred_us
        output_rows.append(part)
    output = pd.concat(output_rows, ignore_index=True, sort=False).sort_values("row_uid", kind="stable").reset_index(drop=True)
    if apply_residual and (model_root / "residual.joblib").exists():
        residual_model = joblib.load(model_root / "residual.joblib")
        residual_features = list(manifest.get("residual_feature_columns", feature_columns))
        model_code_map = {str(name): int(idx) for idx, name in enumerate(candidate_models)}
        residual_frame = blend_frame.merge(
            output[["row_uid", "selected_model", "pred_us"]],
            on="row_uid",
            how="left",
            validate="one_to_one",
        )
        residual_encoded = _apply_router_encoder(
            residual_frame,
            categorical_features=categorical,
            numeric_features=numeric,
            category_maps=dict(manifest.get("category_maps", {})),
        )
        residual_encoded["selected_model_code"] = residual_frame["selected_model"].map(model_code_map).fillna(-1).astype(int)
        residual_encoded["selected_pred_us"] = pd.to_numeric(residual_frame["pred_us"], errors="coerce").fillna(0.0).astype(float)
        residual_delta = residual_model.predict(residual_encoded[residual_features])
        output["pred_us"] = np.exp(np.log(np.clip(output["pred_us"].to_numpy(dtype=float), 1e-6, None)) + residual_delta)
    output["pred_us"] = np.clip(pd.to_numeric(output["pred_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 1e-6, None)
    output["abs_error_us"] = np.abs(output["pred_us"] - output["target_us"])
    output["ape"] = output["abs_error_us"] / np.clip(output["target_us"], 1e-9, None)
    return output


def _train_routed_expert_ensemble(
    *,
    dataset_root: Path,
    analytical_parameters_json: Path,
    output_root: Path,
    model_roots: dict[str, str],
    preprocess_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    support_root = output_root / "support"
    train_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=dataset_root / "train.csv",
        analytical_parameters_json=analytical_parameters_json,
        analytical_output_root=support_root / "analytical_train",
        output_csv=support_root / "train_augmented.csv",
        preprocess_config=preprocess_config,
    )
    train_predictions = _evaluate_base_models_on_augmented_csv(
        augmented_csv=train_aug_csv,
        outputs_dir=support_root / "train_predictions",
        model_roots=model_roots,
        classed_original_data_root=Path(model_roots["classed_data_root"]),
        enhanced_classed_data_root=Path(model_roots["enhanced_classed_data_root"]),
    )
    val_aug_csv = _materialize_eval_augmented_csv(
        raw_csv=dataset_root / "val.csv",
        analytical_parameters_json=analytical_parameters_json,
        analytical_output_root=support_root / "analytical_val",
        output_csv=support_root / "val_augmented.csv",
        preprocess_config=preprocess_config,
    )
    val_predictions = _evaluate_base_models_on_augmented_csv(
        augmented_csv=val_aug_csv,
        outputs_dir=support_root / "val_predictions",
        model_roots=model_roots,
        classed_original_data_root=Path(model_roots["classed_data_root"]),
        enhanced_classed_data_root=Path(model_roots["enhanced_classed_data_root"]),
    )
    train_prediction_csvs = {
        model_name: Path(payload["output_csv"])
        for model_name, payload in train_predictions.items()
        if model_name in set(BASE_ENSEMBLE_CANDIDATES + ["analytical_reference"])
    }
    prediction_csvs = {
        model_name: Path(payload["output_csv"])
        for model_name, payload in val_predictions.items()
        if model_name in set(BASE_ENSEMBLE_CANDIDATES + ["analytical_reference"])
    }
    train_blend = _build_blend_frame(train_aug_csv, train_prediction_csvs, candidate_models=BASE_ENSEMBLE_CANDIDATES)
    val_blend = _build_blend_frame(val_aug_csv, prediction_csvs, candidate_models=BASE_ENSEMBLE_CANDIDATES)
    support_blend = pd.concat([train_blend, val_blend], ignore_index=True, sort=False)
    categorical = _existing_columns(support_blend, ROUTED_ENSEMBLE_CATEGORICAL_FEATURES)
    numeric = _existing_columns(support_blend, ROUTED_ENSEMBLE_NUMERIC_FEATURES)
    encoded, category_maps = _fit_router_encoder(
        support_blend,
        categorical_features=categorical,
        numeric_features=numeric,
    )
    for model_name in BASE_ENSEMBLE_CANDIDATES:
        pred_col = f"pred_{model_name}"
        encoded[f"ape_{model_name}"] = np.abs(
            pd.to_numeric(support_blend[pred_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            - pd.to_numeric(support_blend["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ) / np.clip(pd.to_numeric(support_blend["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 1e-9, None)
    ape_columns = [f"ape_{name}" for name in BASE_ENSEMBLE_CANDIDATES]
    best_idx = encoded[ape_columns].to_numpy(dtype=float).argmin(axis=1)
    encoded["best_model"] = [BASE_ENSEMBLE_CANDIDATES[idx] for idx in best_idx]
    router_feature_columns = list(numeric)
    val_mask = support_blend["split"].astype(str) == "val"
    val_encoded = encoded.loc[val_mask].copy()
    val_blend = support_blend.loc[val_mask].copy()
    global_default_model = min(
        BASE_ENSEMBLE_CANDIDATES,
        key=lambda name: float(np.quantile(val_encoded[f"ape_{name}"].to_numpy(dtype=float), 0.9)),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "model_type": "routed_expert_ensemble",
        "candidate_models": list(BASE_ENSEMBLE_CANDIDATES),
        "categorical_features": categorical,
        "numeric_features": numeric,
        "category_maps": category_maps,
        "router_feature_columns": router_feature_columns,
        "global_default_model": global_default_model,
        "families": {},
        "support": {
            "train_augmented_csv": str(train_aug_csv),
            "train_prediction_csvs": {name: str(path) for name, path in train_prediction_csvs.items()},
            "val_augmented_csv": str(val_aug_csv),
            "val_prediction_csvs": {name: str(path) for name, path in prediction_csvs.items()},
        },
        "seed": SEED,
    }
    for op_type, group in val_encoded.groupby(val_blend["op_type"].astype(str), sort=True):
        fit_group = encoded[(support_blend["op_type"].astype(str) == str(op_type))].copy()
        p90_by_model = {
            name: float(np.quantile(group[f"ape_{name}"].to_numpy(dtype=float), 0.9))
            for name in BASE_ENSEMBLE_CANDIDATES
        }
        default_model = min(p90_by_model, key=p90_by_model.get)
        family_cfg: dict[str, Any] = {
            "default_model": default_model,
            "row_count": int(len(group)),
            "best_model_p90": p90_by_model,
        }
        router_train, router_holdout = _safe_holdout_split(group)
        ranked_models = sorted(
            BASE_ENSEMBLE_CANDIDATES,
            key=lambda name: float(np.mean(router_train[f"ape_{name}"].to_numpy(dtype=float))),
        )
        family_cfg["ranked_models"] = list(ranked_models)
        candidate_payloads: list[dict[str, Any]] = [
            {
                "strategy": "default",
                "artifact_payload": None,
                "metrics": _metric_bundle_from_pred(
                    pd.to_numeric(router_holdout["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                    pd.to_numeric(router_holdout[f"pred_{default_model}"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                ),
                "shortlist": [default_model],
            }
        ]
        for shortlist_size in [2, 3, 4, 5]:
            shortlist = ranked_models[:shortlist_size]
            if len(shortlist) < shortlist_size:
                continue
            shortlist_default = min(
                shortlist,
                key=lambda name: float(np.quantile(router_train[f"ape_{name}"].to_numpy(dtype=float), 0.9)),
            )
            candidate_payloads.append(
                {
                    "strategy": "default",
                    "artifact_payload": None,
                    "metrics": _metric_bundle_from_pred(
                        pd.to_numeric(router_holdout["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                        pd.to_numeric(router_holdout[f"pred_{shortlist_default}"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                    ),
                    "shortlist": shortlist,
                }
            )
            classifier_candidate = _build_classifier_router_candidate(
                group_train=router_train,
                group_holdout=router_holdout,
                feature_columns=router_feature_columns,
                shortlist=shortlist,
                default_model=shortlist_default,
            )
            if classifier_candidate:
                classifier_candidate["shortlist"] = shortlist
                candidate_payloads.append(classifier_candidate)
            error_selector_candidate = _build_error_router_candidate(
                group_train=router_train,
                group_holdout=router_holdout,
                feature_columns=router_feature_columns,
                shortlist=shortlist,
                default_model=shortlist_default,
                blend_mode=False,
            )
            if error_selector_candidate:
                error_selector_candidate["shortlist"] = shortlist
                candidate_payloads.append(error_selector_candidate)
            error_blend_candidate = _build_error_router_candidate(
                group_train=router_train,
                group_holdout=router_holdout,
                feature_columns=router_feature_columns,
                shortlist=shortlist,
                default_model=shortlist_default,
                blend_mode=True,
            )
            if error_blend_candidate:
                error_blend_candidate["shortlist"] = shortlist
                candidate_payloads.append(error_blend_candidate)
        best_strategy = _choose_best_strategy(candidate_payloads)
        family_cfg["selected_strategy"] = str(best_strategy["strategy"])
        family_cfg["strategy_holdout_metrics"] = dict(best_strategy["metrics"])
        family_cfg["holdout_oracle_top2"] = _topk_oracle_metrics(router_holdout, ranked_models[:2])
        family_cfg["holdout_oracle_top3"] = _topk_oracle_metrics(router_holdout, ranked_models[:3])
        family_cfg["holdout_oracle_top4"] = _topk_oracle_metrics(router_holdout, ranked_models[:4])
        family_cfg["holdout_oracle_top5"] = _topk_oracle_metrics(router_holdout, ranked_models[:5])
        family_cfg["shortlist"] = list(best_strategy.get("shortlist", [default_model]))
        family_cfg["feature_columns"] = list(router_feature_columns)
        if best_strategy["strategy"] != "default":
            shortlist = list(best_strategy["shortlist"])
            shortlist_default = min(
                shortlist,
                key=lambda name: float(np.quantile(fit_group[f"ape_{name}"].to_numpy(dtype=float), 0.9)),
            )
            if best_strategy["strategy"] == "rf_classifier":
                classifier_candidate = _build_classifier_router_candidate(
                    group_train=fit_group,
                    group_holdout=fit_group,
                    feature_columns=router_feature_columns,
                    shortlist=shortlist,
                    default_model=shortlist_default,
                )
                retrained_payload = classifier_candidate["artifact_payload"] if classifier_candidate else None
            elif best_strategy["strategy"] == "error_selector":
                error_selector_candidate = _build_error_router_candidate(
                    group_train=fit_group,
                    group_holdout=fit_group,
                    feature_columns=router_feature_columns,
                    shortlist=shortlist,
                    default_model=shortlist_default,
                    blend_mode=False,
                )
                retrained_payload = error_selector_candidate["artifact_payload"] if error_selector_candidate else None
            else:
                error_blend_candidate = _build_error_router_candidate(
                    group_train=fit_group,
                    group_holdout=fit_group,
                    feature_columns=router_feature_columns,
                    shortlist=shortlist,
                    default_model=shortlist_default,
                    blend_mode=True,
                )
                retrained_payload = error_blend_candidate["artifact_payload"] if error_blend_candidate else None
            if retrained_payload:
                artifact = output_root / f"router_{op_type}.joblib"
                joblib.dump(retrained_payload, artifact)
                family_cfg["artifact"] = str(artifact)
        manifest["families"][str(op_type)] = family_cfg

    routed_val = _route_routed_expert_predictions(val_blend, model_root=output_root, manifest=manifest, apply_residual=False)
    residual_train = val_blend.merge(
        routed_val[["row_uid", "selected_model", "pred_us"]],
        on="row_uid",
        how="left",
        validate="one_to_one",
    )
    residual_encoded = _apply_router_encoder(
        residual_train,
        categorical_features=categorical,
        numeric_features=numeric,
        category_maps=category_maps,
    )
    model_code_map = {str(name): int(idx) for idx, name in enumerate(BASE_ENSEMBLE_CANDIDATES)}
    residual_encoded["selected_model_code"] = residual_train["selected_model"].map(model_code_map).fillna(-1).astype(int)
    residual_encoded["selected_pred_us"] = pd.to_numeric(residual_train["pred_us"], errors="coerce").fillna(0.0).astype(float)
    residual_feature_columns = list(router_feature_columns) + ["selected_model_code", "selected_pred_us"]
    residual_target = (
        np.log(np.clip(pd.to_numeric(residual_train["target_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 1e-6, None))
        - np.log(np.clip(pd.to_numeric(residual_train["pred_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 1e-6, None))
    )
    residual_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=25,
        random_state=SEED,
    )
    residual_model.fit(residual_encoded[residual_feature_columns], residual_target)
    residual_artifact = output_root / "residual.joblib"
    joblib.dump(residual_model, residual_artifact)
    manifest["residual_artifact"] = str(residual_artifact)
    manifest["residual_feature_columns"] = residual_feature_columns
    _write_json(output_root / "manifest.json", manifest)

    routed_val.to_csv(output_root / "val_router_predictions.csv", index=False)
    routed_val_residual = _route_routed_expert_predictions(blend, model_root=output_root, manifest=manifest, apply_residual=True)
    routed_val_residual.to_csv(output_root / "val_router_residual_predictions.csv", index=False)
    return {
        "model_type": "routed_expert_ensemble",
        "artifact_root": str(output_root),
        "manifest_json": str(output_root / "manifest.json"),
        "router_val_metrics": evaluate_prediction_frame(routed_val),
        "router_residual_val_metrics": evaluate_prediction_frame(routed_val_residual),
    }


def _predict_routed_expert_ensemble(
    model_root: Path,
    augmented_csv: Path,
    base_prediction_csvs: dict[str, Path],
    output_csv: Path,
    *,
    apply_residual: bool,
) -> dict[str, Any]:
    manifest = _read_json(model_root / "manifest.json")
    blend = _build_blend_frame(
        augmented_csv,
        {
            model_name: base_prediction_csvs[model_name]
            for model_name in BASE_ENSEMBLE_CANDIDATES + ["analytical_reference"]
            if model_name in base_prediction_csvs
        },
        candidate_models=list(manifest.get("candidate_models", BASE_ENSEMBLE_CANDIDATES)),
    )
    routed = _route_routed_expert_predictions(blend, model_root=model_root, manifest=manifest, apply_residual=apply_residual)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    routed.to_csv(output_csv, index=False)
    return {"output_csv": str(output_csv), "metrics": evaluate_prediction_frame(routed)}


def evaluate_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    target = pd.to_numeric(frame["target_us"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(frame["pred_us"], errors="coerce").to_numpy(dtype=float)
    ape = np.abs(pred - target) / np.clip(target, 1e-9, None)
    return {
        "rows": int(len(frame)),
        "mae_us": float(mean_absolute_error(target, pred)),
        "rmse_us": float(np.sqrt(mean_squared_error(target, pred))),
        "r2": float(r2_score(target, pred)) if len(frame) > 1 else 0.0,
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.quantile(ape, 0.9)),
        "p95_ape": float(np.quantile(ape, 0.95)),
    }


def _build_eval_dataset_root(raw_csv: Path, output_root: Path, *, preprocess_config: dict[str, Any] | None = None) -> Path:
    frame = pd.read_csv(raw_csv, low_memory=False)
    frame = _apply_dataset_preprocessing(frame.copy(), preprocess_config)
    frame["split"] = "test"
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "dataset_full.csv", index=False)
    frame.to_csv(output_root / "train.csv", index=False)
    frame.to_csv(output_root / "val.csv", index=False)
    frame.to_csv(output_root / "test.csv", index=False)
    _copy_feature_manifest(output_root)
    _write_json(
        output_root / "manifest.json",
        {
            "source_csv": str(raw_csv),
            "note": "Temporary eval root used for analytical infer.",
        },
    )
    return output_root


def _materialize_eval_augmented_csv(
    *,
    raw_csv: Path,
    analytical_parameters_json: Path,
    analytical_output_root: Path,
    output_csv: Path,
    preprocess_config: dict[str, Any] | None = None,
) -> Path:
    eval_root = _build_eval_dataset_root(raw_csv, analytical_output_root / "_raw_eval", preprocess_config=preprocess_config)
    raw_df = pd.read_csv(eval_root / "dataset_full.csv", low_memory=False)
    analytical_output_root.mkdir(parents=True, exist_ok=True)
    train_params = _read_json(analytical_parameters_json)
    rebuilt_df = build_analytical_features.rebuild_local_features(raw_df)
    heavy_df = build_analytical_features.prepare_heavy_prediction_frame(rebuilt_df)
    feature_df = build_analytical_features.add_calibrated_analytical_columns(rebuilt_df, heavy_df, train_params)
    analytical_csv = analytical_output_root / "analytical_features_full.csv"
    feature_df.to_csv(analytical_csv, index=False)
    merged = _merge_raw_with_analytical(raw_df, analytical_csv)
    augmented = pd.concat([merged.reset_index(drop=True), derive_enhanced_features(merged).reset_index(drop=True)], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_csv, index=False)
    return output_csv


def _evaluate_analytical_reference(augmented_csv: Path, output_csv: Path) -> dict[str, Any]:
    frame = pd.read_csv(augmented_csv, low_memory=False)
    pred = pd.to_numeric(frame["ana_calib_total_us"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    output = frame[["row_uid", "op_type", "split", "label_operator_actual_dur_us"]].copy()
    output["target_us"] = output["label_operator_actual_dur_us"]
    output["pred_us"] = np.clip(pred, 1e-6, None)
    output["abs_error_us"] = np.abs(output["pred_us"] - output["target_us"])
    output["ape"] = output["abs_error_us"] / np.clip(output["target_us"], 1e-9, None)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return {"output_csv": str(output_csv), "metrics": evaluate_prediction_frame(output)}


def _evaluate_base_models_on_augmented_csv(
    *,
    augmented_csv: Path,
    outputs_dir: Path,
    model_roots: dict[str, str],
    classed_original_data_root: Path,
    enhanced_classed_data_root: Path,
) -> dict[str, Any]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    results["analytical_reference"] = _evaluate_analytical_reference(
        augmented_csv,
        outputs_dir / "analytical_reference.csv",
    )
    results["original_single_mlp"] = _evaluate_original_single(
        Path(model_roots["original_single_model_root"]),
        augmented_csv,
        outputs_dir / "original_single_mlp.csv",
    )
    residual_single_root = model_roots.get("analytical_residual_single_model_root", "")
    if residual_single_root:
        results["analytical_residual_single_mlp"] = _evaluate_original_single(
            Path(residual_single_root),
            augmented_csv,
            outputs_dir / "analytical_residual_single_mlp.csv",
        )
    results["original_classed_mlp"] = _classed_infer_custom(
        classed_data_root=classed_original_data_root,
        model_root=Path(model_roots["original_classed_model_root"]),
        input_csv=augmented_csv,
        output_csv=outputs_dir / "original_classed_mlp.csv",
    )
    results["enhanced_classed_mlp"] = _classed_infer_custom(
        classed_data_root=enhanced_classed_data_root,
        model_root=Path(model_roots["enhanced_classed_model_root"]),
        input_csv=augmented_csv,
        output_csv=outputs_dir / "enhanced_classed_mlp.csv",
    )
    results["global_enhanced"] = _predict_global_enhanced(
        Path(model_roots["global_model_root"]),
        augmented_csv,
        outputs_dir / "global_enhanced.csv",
    )
    results["family_enhanced"] = _predict_family_enhanced(
        Path(model_roots["family_model_root"]),
        augmented_csv,
        outputs_dir / "family_enhanced.csv",
    )
    return results


def _evaluate_original_single(model_root: Path, eval_csv: Path, output_csv: Path) -> dict[str, Any]:
    payload = run_inference(
        model_dir=model_root / "model",
        input_csv=eval_csv,
        output_csv=output_csv,
        provider="cpu",
        batch_size=4096,
    )
    return {"output_csv": str(output_csv), "metrics": payload["metrics"]}


def _train_analytical_residual_single_model(*, classed_data_root: Path, output_root: Path) -> dict[str, Any]:
    from cpu_perf_model.single_fair import build_fair_single_dataset

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / "dataset"
    model_dir = output_root / "model"
    dataset_manifest = build_fair_single_dataset(classed_data_root, dataset_dir)
    model_summary = train_model(
        data_dir=dataset_dir,
        output_dir=model_dir,
        hidden_layers=(128, 128, 128, 128, 128, 128),
        batch_size=1024,
        max_iter=200,
        alpha=1e-4,
        learning_rate_init=1e-3,
        seed=SEED,
        log_target=True,
        target_mode="analytical_residual",
        train_device="cpu",
        npu_device_id=0,
        early_stopping_patience=12,
        export_onnx=True,
        onnx_opset=17,
    )
    manifest = {
        "artifact_root": str(output_root),
        "dataset_manifest": dataset_manifest,
        "model_dir": str(model_dir),
        "metrics_json": str(model_dir / "metrics.json"),
        "model_manifest_json": str(model_dir / "model_manifest.json"),
        "predictions_test_csv": str(model_dir / "predictions_test.csv"),
        "feature_count": dataset_manifest["feature_count"],
        "target_mode": "analytical_residual",
        "model_summary": model_summary,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _train_original_models(
    *,
    dataset_root: Path,
    analytical_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    classed_root = output_root / "classed_original_data"
    build_classed_dataset_artifacts(
        input_data_dir=dataset_root,
        analytical_dir=analytical_root,
        output_dir=classed_root,
        feature_branch="with_analytical",
    )
    original_classed_root = output_root / "original_classed_mlp"
    classed_payload = train_all_classes(
        data_root=classed_root,
        model_root=original_classed_root,
        hidden_layers="128,128,128,128,128",
        batch_size=1024,
        max_iter=300,
        alpha=1e-4,
        learning_rate_init=1e-3,
        seed=SEED,
        train_device="cpu",
        npu_device_id=0,
        early_stopping_patience=12,
        log_target=True,
        export_onnx=True,
        onnx_opset=17,
    )
    original_single_root = output_root / "original_single_mlp"
    single_payload = train_fair_single_model(
        classed_data_root=classed_root,
        output_root=original_single_root,
    )
    residual_single_root = output_root / "analytical_residual_single_mlp"
    residual_single_payload = _train_analytical_residual_single_model(
        classed_data_root=classed_root,
        output_root=residual_single_root,
    )
    return {
        "classed_data_root": str(classed_root),
        "original_classed_payload": classed_payload,
        "original_single_payload": single_payload,
        "analytical_residual_single_payload": residual_single_payload,
        "original_classed_model_root": str(original_classed_root),
        "original_single_model_root": str(original_single_root),
        "analytical_residual_single_model_root": str(residual_single_root),
    }


def _train_enhanced_models(
    *,
    dataset_root: Path,
    analytical_parameters_json: Path,
    classed_root: Path,
    augmented_dataset_csv: Path,
    output_root: Path,
    preprocess_config: dict[str, Any] | None = None,
    base_model_roots: dict[str, str] | None = None,
) -> dict[str, Any]:
    enhanced_classed_data_root = output_root / "enhanced_classed_data"
    _rebuild_enhanced_classed_dataset(Path(classed_root), enhanced_classed_data_root)
    enhanced_classed_model_root = output_root / "enhanced_classed_mlp"
    enhanced_classed_payload = train_all_classes(
        data_root=enhanced_classed_data_root,
        model_root=enhanced_classed_model_root,
        hidden_layers="128,128,128,128,128",
        batch_size=1024,
        max_iter=300,
        alpha=1e-4,
        learning_rate_init=1e-3,
        seed=SEED,
        train_device="cpu",
        npu_device_id=0,
        early_stopping_patience=12,
        log_target=True,
        export_onnx=True,
        onnx_opset=17,
    )
    augmented_df = pd.read_csv(augmented_dataset_csv, low_memory=False)
    global_root = output_root / "global_enhanced"
    family_root = output_root / "family_enhanced"
    global_payload = _fit_global_enhanced(augmented_df, global_root)
    family_payload = _fit_family_enhanced(augmented_df, family_root)
    payload = {
        "enhanced_classed_data_root": str(enhanced_classed_data_root),
        "enhanced_classed_payload": enhanced_classed_payload,
        "enhanced_classed_model_root": str(enhanced_classed_model_root),
        "global_payload": global_payload,
        "global_model_root": str(global_root),
        "family_payload": family_payload,
        "family_model_root": str(family_root),
    }
    if base_model_roots:
        routed_root = output_root / "routed_expert_ensemble"
        routed_payload = _train_routed_expert_ensemble(
            dataset_root=dataset_root,
            analytical_parameters_json=analytical_parameters_json,
            output_root=routed_root,
            model_roots={**base_model_roots, **payload},
            preprocess_config=preprocess_config,
        )
        payload["routed_expert_payload"] = routed_payload
        payload["routed_expert_ensemble_model_root"] = str(routed_root)
        payload["routed_expert_residual_model_root"] = str(routed_root)
    return payload


def _evaluate_all_models_for_experiment(
    *,
    experiment_root: Path,
    model_roots: dict[str, str],
    analytical_parameters_json: Path,
    preprocess_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_raw_root = experiment_root / "eval_inputs" / "raw"
    return _evaluate_model_roots_on_raw_csvs(
        raw_csvs={path.stem: path for path in sorted(eval_raw_root.glob("*.csv"))},
        eval_aug_root=experiment_root / "eval_inputs" / "augmented",
        outputs_root=experiment_root / "evaluations",
        model_roots=model_roots,
        analytical_parameters_json=analytical_parameters_json,
        preprocess_config=preprocess_config,
    )


def _evaluate_model_roots_on_raw_csvs(
    *,
    raw_csvs: dict[str, Path],
    eval_aug_root: Path,
    outputs_root: Path,
    model_roots: dict[str, str],
    analytical_parameters_json: Path,
    preprocess_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outputs_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    classed_original_data_root = Path(model_roots["classed_data_root"])
    enhanced_classed_data_root = Path(model_roots["enhanced_classed_data_root"])
    for eval_name, raw_csv in sorted(raw_csvs.items()):
        aug_csv = _materialize_eval_augmented_csv(
            raw_csv=raw_csv,
            analytical_parameters_json=analytical_parameters_json,
            analytical_output_root=eval_aug_root / eval_name / "analytical",
            output_csv=eval_aug_root / f"{eval_name}.csv",
            preprocess_config=preprocess_config,
        )
        eval_result = _evaluate_base_models_on_augmented_csv(
            augmented_csv=aug_csv,
            outputs_dir=outputs_root / eval_name,
            model_roots=model_roots,
            classed_original_data_root=classed_original_data_root,
            enhanced_classed_data_root=enhanced_classed_data_root,
        )
        base_prediction_csvs = {
            model_name: Path(payload["output_csv"])
            for model_name, payload in eval_result.items()
            if isinstance(payload, dict) and payload.get("output_csv")
        }
        routed_root = model_roots.get("routed_expert_ensemble_model_root", "")
        if routed_root:
            eval_result["routed_expert_ensemble"] = _predict_routed_expert_ensemble(
                Path(routed_root),
                aug_csv,
                base_prediction_csvs,
                outputs_root / eval_name / "routed_expert_ensemble.csv",
                apply_residual=False,
            )
            eval_result["routed_expert_residual_ensemble"] = _predict_routed_expert_ensemble(
                Path(routed_root),
                aug_csv,
                base_prediction_csvs,
                outputs_root / eval_name / "routed_expert_residual_ensemble.csv",
                apply_residual=True,
            )
        results[eval_name] = eval_result
    _write_json(outputs_root / "evaluation_summary.json", results)
    return results


def evaluate_saved_experiment(
    output_root: Path,
    experiment_id: str,
    raw_csvs: dict[str, Path],
    *,
    evaluation_tag: str = "external_eval",
) -> dict[str, Any]:
    exp_root = output_root / experiment_id
    run_summary_path = exp_root / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing trained experiment summary: {run_summary_path}")
    run_summary = _read_json(run_summary_path)
    analytical_parameters_json = exp_root / "analytical" / "deploy" / "full_data_parameters.json"
    if not analytical_parameters_json.exists():
        raise FileNotFoundError(f"Missing analytical parameters: {analytical_parameters_json}")
    preprocess_config = dict(run_summary.get("experiment_config", {}).get("preprocess_config", {}))
    results = _evaluate_model_roots_on_raw_csvs(
        raw_csvs=raw_csvs,
        eval_aug_root=exp_root / "re_evaluations" / evaluation_tag / "augmented",
        outputs_root=exp_root / "re_evaluations" / evaluation_tag / "outputs",
        model_roots=dict(run_summary["training_roots"]),
        analytical_parameters_json=analytical_parameters_json,
        preprocess_config=preprocess_config,
    )
    summary = {
        "experiment_id": experiment_id,
        "evaluation_tag": evaluation_tag,
        "raw_csvs": {name: str(path) for name, path in raw_csvs.items()},
        "results": results,
    }
    _write_json(exp_root / "re_evaluations" / evaluation_tag / "summary.json", summary)
    return summary


def resume_experiment_evaluation(output_root: Path, experiment_id: str) -> dict[str, Any]:
    exp_root = output_root / experiment_id
    analytical_parameters_json = exp_root / "analytical" / "deploy" / "full_data_parameters.json"
    if not analytical_parameters_json.exists():
        raise FileNotFoundError(f"Missing analytical parameters: {analytical_parameters_json}")
    training_root = exp_root / "training"
    model_roots = {
        "classed_data_root": str(training_root / "classed_original_data"),
        "original_classed_model_root": str(training_root / "original_classed_mlp"),
        "original_single_model_root": str(training_root / "original_single_mlp"),
        "enhanced_classed_data_root": str(training_root / "enhanced_classed_data"),
        "enhanced_classed_model_root": str(training_root / "enhanced_classed_mlp"),
        "global_model_root": str(training_root / "global_enhanced"),
        "family_model_root": str(training_root / "family_enhanced"),
    }
    residual_single_root = training_root / "analytical_residual_single_mlp"
    if residual_single_root.exists():
        model_roots["analytical_residual_single_model_root"] = str(residual_single_root)
    routed_root = training_root / "routed_expert_ensemble"
    if routed_root.exists():
        model_roots["routed_expert_ensemble_model_root"] = str(routed_root)
        model_roots["routed_expert_residual_model_root"] = str(routed_root)
    missing = [path for path in model_roots.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing training artifacts for resume: {missing}")
    dataset_root = exp_root / "dataset"
    if not dataset_root.exists():
        dataset_root = exp_root / "dataset_train_protocol"
    experiment_config = _experiment_config(experiment_id)
    evaluation_summary = _evaluate_all_models_for_experiment(
        experiment_root=exp_root,
        model_roots=model_roots,
        analytical_parameters_json=analytical_parameters_json,
        preprocess_config=dict(experiment_config),
    )
    summary = {
        "experiment_id": experiment_id,
        "description": EXPERIMENT_SPECS[experiment_id]["description"],
        "experiment_config": experiment_config,
        "dataset_root": str(dataset_root),
        "analytical_root": str(exp_root / "analytical"),
        "augmented_dataset_csv": str(exp_root / "augmented_dataset.csv"),
        "training_roots": model_roots,
        "evaluation_summary": evaluation_summary,
    }
    _write_json(exp_root / "run_summary.json", summary)
    return summary


def run_experiment(output_root: Path, experiment_id: str) -> dict[str, Any]:
    prepared = prepare_experiment_datasets(output_root, experiment_id)
    exp_root = output_root / experiment_id
    dataset_root = Path(prepared["experiments"][experiment_id]["dataset_root"])
    experiment_config = _experiment_config(experiment_id)
    analytical_root = exp_root / "analytical"
    fixed_analytical_params = experiment_config.get("fixed_analytical_params_json")
    if fixed_analytical_params:
        analytical_manifest = _build_fixed_analytical_artifacts(
            dataset_root,
            analytical_root,
            params_json=Path(str(fixed_analytical_params)),
        )
    else:
        analytical_manifest = _train_analytical(dataset_root, analytical_root)
    augmented_csv = _prepare_augmented_dataset(dataset_root, analytical_root, exp_root / "augmented_dataset.csv")

    training_root = exp_root / "training"
    original_payload = _train_original_models(
        dataset_root=dataset_root,
        analytical_root=analytical_root,
        output_root=training_root,
    )
    enhanced_payload = _train_enhanced_models(
        dataset_root=dataset_root,
        analytical_parameters_json=Path(analytical_manifest["artifacts"]["deploy_parameters_json"]),
        classed_root=Path(original_payload["classed_data_root"]),
        augmented_dataset_csv=augmented_csv,
        output_root=training_root,
        preprocess_config=experiment_config,
        base_model_roots=original_payload,
    )
    model_roots = {
        **original_payload,
        **enhanced_payload,
    }
    evaluation_summary = _evaluate_all_models_for_experiment(
        experiment_root=exp_root,
        model_roots=model_roots,
        analytical_parameters_json=Path(analytical_manifest["artifacts"]["deploy_parameters_json"]),
        preprocess_config=experiment_config,
    )
    summary = {
        "experiment_id": experiment_id,
        "description": EXPERIMENT_SPECS[experiment_id]["description"],
        "experiment_config": experiment_config,
        "dataset_root": str(dataset_root),
        "analytical_root": str(analytical_root),
        "augmented_dataset_csv": str(augmented_csv),
        "training_roots": model_roots,
        "evaluation_summary": evaluation_summary,
    }
    _write_json(exp_root / "run_summary.json", summary)
    return summary


def run_all(output_root: Path, experiment_ids: list[str] | None = None) -> dict[str, Any]:
    selected = experiment_ids or list(EXPERIMENT_SPECS.keys())
    overall = {"output_root": str(output_root), "experiments": {}}
    prepare_experiment_datasets(output_root)
    for experiment_id in selected:
        overall["experiments"][experiment_id] = run_experiment(output_root, experiment_id)
    _write_json(output_root / "run_summary.json", overall)
    return overall
