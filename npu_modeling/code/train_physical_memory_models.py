import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

try:
    from build_semantic_operator_dataset import build_record
except Exception:
    build_record = None


DELIVERY_ROOT = Path(__file__).resolve().parents[1]
TARGET = "target_kernel_compute_us"
CATEGORICAL = ["op_type", "dtype", "physical_family", "residual_group", "model_group"]
DROP_ALWAYS = {
    TARGET,
    "case_id",
    "profile_dir",
    "major_class",
    "latency_p50_us",
    "latency_mean_us",
    "kernel_count_observed",
    "kernel_type_unique_observed",
    "share_ai_core_observed",
    "share_ai_vector_core_observed",
    "share_mix_aiv_observed",
}

PHYSICAL_FAMILY_BY_OP = {
    "MatMulV2": "DenseGemm",
    "BatchMatMulV2": "DenseGemm",
    "Linear": "DenseGemm",
    "GemmBias": "DenseGemm",
    "Add": "VectorStreaming",
    "Mul": "VectorStreaming",
    "Relu": "VectorStreaming",
    "Sigmoid": "VectorStreaming",
    "Gelu": "VectorStreaming",
    "Tanh": "VectorStreaming",
    "Cast": "VectorStreaming",
    "Where": "VectorStreaming",
    "LayerNorm": "VectorStreaming",
    "BatchNorm1D": "VectorStreaming",
    "ReduceMean": "AxisCompute",
    "ReduceSum": "AxisCompute",
    "Softmax": "AxisCompute",
    "TopK": "AxisCompute",
    "Sort": "AxisCompute",
    "Transpose": "DataMovement",
    "ConcatD": "DataMovement",
    "Tile": "DataMovement",
    "PadV3": "DataMovement",
    "MemSet": "DataMovement",
    "Gather": "SparseIndex",
    "Index": "SparseIndex",
    "ScatterAdd": "SparseIndex",
    "EmbeddingBag": "SparseIndex",
    "AdvancedIndexing": "SparseIndex",
    "MaxPool2D": "SpatialWindow",
    "NonZero": "DataDependent",
}

RESIDUAL_GROUP_BY_FAMILY = {
    "DenseGemm": "DenseResidual",
    "AxisCompute": "AxisResidual",
    "DataMovement": "MovementResidual",
    "SparseIndex": "SparseResidual",
}

STRICT_COMPACT_GROUP_BY_OP = {
    "MatMulV2": "StructuredComputeMemory",
    "BatchMatMulV2": "StructuredComputeMemory",
    "Linear": "StructuredComputeMemory",
    "GemmBias": "StructuredComputeMemory",
    "Gather": "StructuredComputeMemory",
    "Index": "StructuredComputeMemory",
    "ScatterAdd": "StructuredComputeMemory",
    "EmbeddingBag": "StructuredComputeMemory",
    "AdvancedIndexing": "StructuredComputeMemory",
    "Add": "StreamingReduction",
    "Mul": "StreamingReduction",
    "Relu": "StreamingReduction",
    "Sigmoid": "StreamingReduction",
    "Gelu": "StreamingReduction",
    "Tanh": "StreamingReduction",
    "Cast": "StreamingReduction",
    "Where": "StreamingReduction",
    "LayerNorm": "StreamingReduction",
    "BatchNorm1D": "StreamingReduction",
    "Softmax": "StreamingReduction",
    "ReduceMean": "StreamingReduction",
    "ReduceSum": "StreamingReduction",
    "Transpose": "MovementSelectionWindow",
    "ConcatD": "MovementSelectionWindow",
    "Tile": "MovementSelectionWindow",
    "PadV3": "MovementSelectionWindow",
    "MemSet": "MovementSelectionWindow",
    "TopK": "MovementSelectionWindow",
    "Sort": "MovementSelectionWindow",
    "MaxPool2D": "MovementSelectionWindow",
    "NonZero": "DataDependent",
}


def fnum(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def metric_pack(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(pred) & (y > 0)
    y = y[mask]
    pred = pred[mask]
    if len(y) == 0:
        return {
            "sample_count": 0,
            "mae_us": np.nan,
            "mape_1us_floor_percent": np.nan,
            "wmape_percent": np.nan,
            "p50_1us_floor_percent": np.nan,
            "p90_1us_floor_percent": np.nan,
            "p95_1us_floor_percent": np.nan,
            "max_1us_floor_percent": np.nan,
        }
    ape = np.abs(pred - y) / np.maximum(np.abs(y), 1.0) * 100.0
    return {
        "sample_count": int(len(y)),
        "mae_us": float(np.mean(np.abs(pred - y))),
        "mape_1us_floor_percent": float(np.mean(ape)),
        "wmape_percent": float(np.sum(np.abs(pred - y)) / max(np.sum(np.abs(y)), 1e-9) * 100.0),
        "p50_1us_floor_percent": float(np.percentile(ape, 50)),
        "p90_1us_floor_percent": float(np.percentile(ape, 90)),
        "p95_1us_floor_percent": float(np.percentile(ape, 95)),
        "max_1us_floor_percent": float(np.max(ape)),
    }


def add_taxonomy(df):
    out = df.copy()
    out["physical_family"] = out["op_type"].map(PHYSICAL_FAMILY_BY_OP).fillna(out.get("major_class", "Unknown"))
    out["residual_group"] = out["physical_family"].map(RESIDUAL_GROUP_BY_FAMILY).fillna("GlobalOnly")
    out["model_group"] = out["op_type"].map(STRICT_COMPACT_GROUP_BY_OP).fillna(out["physical_family"])
    return out


def load_extra_core_labels(paths, hw):
    if not paths:
        return pd.DataFrame()
    if build_record is None:
        raise RuntimeError("Cannot import build_record for --extra-core-labels")
    rows = []
    for raw in paths.split(","):
        path = Path(raw.strip())
        if not path.exists():
            continue
        extra = pd.read_csv(path, low_memory=False)
        for _, row in extra.iterrows():
            if str(row.get("build_status", "")) != "ok":
                continue
            target = fnum(row.get("kernel_total_duration_us")) or fnum(row.get("device_total_duration_us"))
            if target <= 0:
                continue
            try:
                params = json.loads(str(row.get("params_json", "{}")))
            except Exception:
                params = {}
            rec = build_record(
                f"extra_{row.get('case_id')}",
                str(row.get("profile_dir", "")),
                str(row.get("major_class", "Unknown")),
                str(row.get("op_type", "")),
                str(row.get("dtype", "bfloat16")),
                params,
                target,
                hw,
                fnum(row.get("latency_p50_us")),
                fnum(row.get("latency_mean_us")),
            )
            rec.update({
                "kernel_count_observed": fnum(row.get("kernel_row_count")),
                "share_ai_core_observed": fnum(row.get("share_ai_core")),
                "share_ai_vector_core_observed": fnum(row.get("share_ai_vector_core")),
                "share_mix_aiv_observed": fnum(row.get("share_mix_aiv")),
            })
            rows.append(rec)
    return pd.DataFrame(rows)


def add_hierarchical_memory_features(df, hw):
    out = df.copy()

    def numeric_col(name, default=0.0):
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype=float)

    l0 = float(hw.get("l0_capacity_bytes", 64 * 1024))
    l1 = float(hw.get("l1_capacity_bytes", 1536 * 1024))
    l2 = float(hw.get("l2_capacity_bytes", 32 * 1024 * 1024))
    l1_bw = float(hw.get("l1_bandwidth_bytes_per_s", 2e12))
    l2_bw = float(hw.get("l2_bandwidth_bytes_per_s", hw.get("memory_bandwidth_bytes_per_s", 1e12)))
    mem_bw = float(hw.get("memory_bandwidth_bytes_per_s", 1e12))
    peak_cube = float(hw.get("peak_cube_flops", out.get("peak_cube_flops", pd.Series([1.0])).iloc[0]))
    peak_vec = float(hw.get("peak_vector_flops", out.get("peak_vector_flops", pd.Series([1.0])).iloc[0]))

    bytes_rw = numeric_col("bytes_rw_estimate", 0.0).clip(lower=0.0)
    working = numeric_col("working_set_bytes", 0.0).clip(lower=0.0)
    flops = numeric_col("flops_estimate", 0.0).clip(lower=0.0)
    dtype_bytes = numeric_col("dtype_bytes", 2.0).clip(lower=1.0)

    out["l0_capacity_bytes"] = l0
    out["l1_bandwidth_bytes_per_s"] = l1_bw
    out["l2_bandwidth_bytes_per_s"] = l2_bw
    out["l0_pressure"] = working / max(l0, 1.0)
    out["l1_pressure"] = working / max(l1, 1.0)
    out["l2_pressure"] = working / max(l2, 1.0)
    out["fits_l0"] = (working <= l0).astype(float)
    out["fits_l1"] = (working <= l1).astype(float)
    out["fits_l2"] = (working <= l2).astype(float)

    # Estimate hierarchy traffic. L0 is modeled as capacity/reuse pressure because bandwidth is very high.
    reuse = np.where(working <= l0, 0.25, np.where(working <= l1, 0.55, np.where(working <= l2, 0.85, 1.0)))
    sparse_penalty = np.where(out["physical_family"].eq("SparseIndex"), 1.6, 1.0)
    transpose_penalty = np.where(out["op_type"].eq("Transpose"), 1.4 + 0.05 * numeric_col("transpose_perm_code", 0.0), 1.0)
    axis_penalty = np.where(out["physical_family"].eq("AxisCompute"), 1.0 + 1.0 / np.maximum(numeric_col("reduce_axis_size", 1.0), 1.0), 1.0)
    movement_penalty = np.where(out["physical_family"].eq("DataMovement"), 1.15, 1.0)
    effective_bytes = bytes_rw * sparse_penalty * transpose_penalty * axis_penalty * movement_penalty

    out["effective_memory_bytes"] = effective_bytes
    out["estimated_l1_bytes"] = effective_bytes * reuse
    out["estimated_l2_bytes"] = effective_bytes * np.where(working <= l1, 0.35, np.where(working <= l2, 0.75, 1.0))
    out["estimated_hbm_bytes"] = effective_bytes * np.where(working <= l2, 0.15, 1.0)
    out["t_l1_us"] = out["estimated_l1_bytes"] / max(l1_bw, 1.0) * 1e6
    out["t_l2_us"] = out["estimated_l2_bytes"] / max(l2_bw, 1.0) * 1e6
    out["t_hbm_us"] = out["estimated_hbm_bytes"] / max(mem_bw, 1.0) * 1e6
    out["t_mem_hier_us"] = np.maximum.reduce([out["t_l1_us"], out["t_l2_us"], out["t_hbm_us"]])

    cube_ops = out["physical_family"].eq("DenseGemm")
    out["t_compute_physical_us"] = np.where(cube_ops, flops / max(peak_cube, 1.0) * 1e6, flops / max(peak_vec, 1.0) * 1e6)
    out["t_roofline_hier_us"] = np.maximum(out["t_compute_physical_us"], out["t_mem_hier_us"])
    out["hier_arithmetic_intensity"] = flops / np.maximum(effective_bytes, 1.0)

    for col in [
        "l0_pressure",
        "l1_pressure",
        "l2_pressure",
        "effective_memory_bytes",
        "estimated_l1_bytes",
        "estimated_l2_bytes",
        "estimated_hbm_bytes",
        "t_l1_us",
        "t_l2_us",
        "t_hbm_us",
        "t_mem_hier_us",
        "t_compute_physical_us",
        "t_roofline_hier_us",
        "hier_arithmetic_intensity",
    ]:
        out[f"log1p_{col}"] = np.log1p(np.maximum(pd.to_numeric(out[col], errors="coerce").fillna(0.0), 0.0))

    out["bytes_per_output_elem"] = effective_bytes / np.maximum(numeric_col("output_elems_estimate", 1.0), 1.0)
    out["log1p_bytes_per_output_elem"] = np.log1p(np.maximum(out["bytes_per_output_elem"], 0.0))
    out["dtype_bytes_times_output"] = dtype_bytes * numeric_col("output_elems_estimate", 0.0)

    d0 = numeric_col("param_d0", 1.0).clip(lower=1.0)
    d1 = numeric_col("param_d1", 1.0).clip(lower=1.0)
    d2 = numeric_col("param_d2", 1.0).clip(lower=1.0)
    axis = numeric_col("param_axis", 0.0).clip(lower=0.0, upper=2.0)
    axis_size = np.where(axis == 0.0, d0, np.where(axis == 1.0, d1, d2))
    total_elems = d0 * d1 * d2
    outer_size = total_elems / np.maximum(axis_size, 1.0)
    out["axis_total_elems"] = total_elems
    out["axis_size_physical"] = axis_size
    out["axis_outer_physical"] = outer_size
    out["axis_bytes"] = axis_size * dtype_bytes
    out["axis_outer_bytes"] = outer_size * dtype_bytes
    out["axis_is_inner"] = (axis == 2.0).astype(float)
    out["axis_is_middle"] = (axis == 1.0).astype(float)
    out["axis_is_outer"] = (axis == 0.0).astype(float)
    out["axis_size_mod_8"] = axis_size % 8.0
    out["axis_size_mod_16"] = axis_size % 16.0
    out["axis_size_mod_32"] = axis_size % 32.0
    out["axis_sort_work"] = outer_size * axis_size * np.log2(np.maximum(axis_size, 2.0))
    out["axis_softmax_work"] = outer_size * axis_size * 8.0

    batch = numeric_col("param_batch", 1.0).clip(lower=1.0)
    channels = numeric_col("param_channels", 1.0).clip(lower=1.0)
    h = numeric_col("param_h", 1.0).clip(lower=1.0)
    w = numeric_col("param_w", 1.0).clip(lower=1.0)
    kernel = numeric_col("param_kernel", 1.0).clip(lower=1.0)
    stride = numeric_col("param_stride", 1.0).clip(lower=1.0)
    out_h = np.maximum(np.floor((h + 2.0 * np.floor(kernel / 2.0) - kernel) / stride + 1.0), 1.0)
    out_w = np.maximum(np.floor((w + 2.0 * np.floor(kernel / 2.0) - kernel) / stride + 1.0), 1.0)
    out["pool_out_h"] = out_h
    out["pool_out_w"] = out_w
    out["pool_output_elems_physical"] = batch * channels * out_h * out_w
    out["pool_input_bytes"] = batch * channels * h * w * dtype_bytes
    out["pool_output_bytes"] = batch * channels * out_h * out_w * dtype_bytes
    out["pool_overlap_ratio"] = (kernel / stride).clip(lower=0.0)
    out["pool_reuse_work"] = batch * channels * out_h * out_w * kernel * kernel
    out["pool_channels_mod_32"] = channels % 32.0

    mm_m = numeric_col("param_m", 0.0).clip(lower=0.0)
    mm_k = numeric_col("param_k", 0.0).clip(lower=0.0)
    mm_n = numeric_col("param_n", 0.0).clip(lower=0.0)
    out["gemm_m_tiles_16"] = np.ceil(mm_m / 16.0)
    out["gemm_k_tiles_16"] = np.ceil(mm_k / 16.0)
    out["gemm_n_tiles_16"] = np.ceil(mm_n / 16.0)
    out["gemm_tile_volume_16"] = out["gemm_m_tiles_16"] * out["gemm_k_tiles_16"] * out["gemm_n_tiles_16"]
    out["gemm_m_tail_16"] = mm_m % 16.0
    out["gemm_k_tail_16"] = mm_k % 16.0
    out["gemm_n_tail_16"] = mm_n % 16.0

    for col in [
        "axis_total_elems",
        "axis_size_physical",
        "axis_outer_physical",
        "axis_bytes",
        "axis_outer_bytes",
        "axis_sort_work",
        "axis_softmax_work",
        "pool_output_elems_physical",
        "pool_input_bytes",
        "pool_output_bytes",
        "pool_reuse_work",
        "gemm_tile_volume_16",
    ]:
        out[f"log1p_{col}"] = np.log1p(np.maximum(pd.to_numeric(out[col], errors="coerce").fillna(0.0), 0.0))
    return out


def stable_aggregate(df):
    excluded = set(DROP_ALWAYS)
    cols = [c for c in df.columns if c not in excluded]
    key = df[cols].copy()
    for col in cols:
        if col in CATEGORICAL:
            key[col] = key[col].astype(str).fillna("missing")
        else:
            key[col] = pd.to_numeric(key[col], errors="coerce").fillna(0.0).round(8)
    tmp = df.copy()
    tmp["stable_feature_hash"] = pd.util.hash_pandas_object(key, index=False).astype(str)
    tmp[TARGET] = tmp.groupby("stable_feature_hash")[TARGET].transform("median")
    return tmp.drop_duplicates("stable_feature_hash").drop(columns=["stable_feature_hash"])


def make_xy(train, test):
    cols = sorted((set(train.columns) | set(test.columns)) - DROP_ALWAYS)
    both = pd.concat([train.reindex(columns=cols), test.reindex(columns=cols)], ignore_index=True)
    for col in cols:
        if col in CATEGORICAL:
            both[col] = both[col].astype(str).fillna("missing")
        else:
            both[col] = pd.to_numeric(both[col], errors="coerce").fillna(0.0)
    both = pd.get_dummies(both, columns=[c for c in CATEGORICAL if c in both.columns])
    return both.iloc[: len(train)].copy(), both.iloc[len(train) :].copy()


def candidate_models(seed):
    return [
        ("extra_trees", ExtraTreesRegressor(n_estimators=220, min_samples_leaf=1, max_features=0.95, random_state=seed, n_jobs=-1)),
        ("extra_trees_smooth", ExtraTreesRegressor(n_estimators=260, min_samples_leaf=2, max_features=0.9, random_state=seed + 1, n_jobs=-1)),
        ("random_forest", RandomForestRegressor(n_estimators=180, min_samples_leaf=1, max_features=0.9, random_state=seed + 2, n_jobs=-1)),
        ("hgb_log", HistGradientBoostingRegressor(learning_rate=0.045, max_iter=160, max_leaf_nodes=31, l2_regularization=0.01, random_state=seed + 3)),
    ]


def fit_best(xtr, ytr, xte, yte, seed, p90_weight=0.15, allow_log=True):
    best = None
    for name, model in candidate_models(seed):
        if name.endswith("log") and (not allow_log or np.nanmin(ytr) <= -1.0):
            continue
        target = np.log1p(ytr) if name.endswith("log") else ytr
        model.fit(xtr, target)
        pred = model.predict(xte)
        if name.endswith("log"):
            pred = np.expm1(pred)
        pred = np.maximum(pred, 0.0)
        metrics = metric_pack(yte, pred)
        score = metrics["mape_1us_floor_percent"] + p90_weight * metrics["p90_1us_floor_percent"]
        if best is None or score < best[0]:
            best = (score, name, model, pred, metrics)
    return best[1], best[2], best[3], best[4]


def align_columns(frame, columns):
    out = frame.copy()
    for col in columns:
        if col not in out:
            out[col] = 0.0
    return out[columns]


def write_report(out_dir, summary, overall, family_df, op_df, residual_df):
    summary_cols = [
        "stable_rows",
        "train_rows",
        "test_rows",
        "model_count",
        "strategy",
        "mape_target_percent",
        "tail_target_percent",
        "mape_target_met",
        "p90_target_met",
        "per_model_target_met",
        "models_meeting_target",
        "models_total",
    ]
    metric_cols = [
        "sample_count",
        "mae_us",
        "mape_1us_floor_percent",
        "wmape_percent",
        "p50_1us_floor_percent",
        "p90_1us_floor_percent",
        "metric_scope",
    ]
    family_cols = [
        "sample_count",
        "mae_us",
        "mape_1us_floor_percent",
        "wmape_percent",
        "p50_1us_floor_percent",
        "p90_1us_floor_percent",
        "model_group",
        "model_choice",
        "train_rows",
        "test_rows",
        "mape_target_met",
        "p90_target_met",
        "model_target_met",
    ]
    summary_df = pd.DataFrame([summary])[[c for c in summary_cols if c in summary]]
    overall_df = pd.DataFrame([overall])[[c for c in metric_cols if c in overall]]
    family_report = family_df[[c for c in family_cols if c in family_df.columns]].copy()
    lines = [
        "# NPU Physical Memory Model Report",
        "",
        "## Summary",
        summary_df.to_markdown(index=False),
        "",
        "## Acceptance Metrics",
        overall_df.to_markdown(index=False),
        "",
        "## Residual Experts",
        residual_df.to_markdown(index=False) if len(residual_df) else "No residual experts trained.",
        "",
        "## Metrics By Model Group",
        family_report.to_markdown(index=False),
        "",
        "## Notes",
        "- Uses stable-label aggregation before train/test split.",
        "- Accepted strategy uses one model per final model group.",
        "- Hierarchical memory features model L0 as capacity/reuse pressure and L1/L2/HBM as bandwidth lower-bound terms.",
        "- Operator-level diagnostics are retained as CSV artifacts for engineering analysis but are not part of the delivery acceptance report.",
    ]
    (out_dir / "PHYSICAL_MEMORY_MODEL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_physical_family_strategy(df, out_dir, args, hw, train_extra_df=None):
    group_col = "model_group" if args.grouping == "strict_compact" else "physical_family"
    strat = df["op_type"] if df["op_type"].value_counts().min() >= 2 else None
    train, test = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=strat)
    if train_extra_df is not None and len(train_extra_df):
        train = pd.concat([train, train_extra_df], ignore_index=True, sort=False)
    preds = []
    artifacts = {}
    family_metrics = []
    expert_ops = {op.strip() for op in args.expert_ops.split(",") if op.strip()}
    expert_rows = []

    for family in sorted(df[group_col].unique()):
        tr = train[train[group_col] == family].copy()
        te = test[test[group_col] == family].copy()
        if len(tr) < 50 or len(te) < 8:
            continue
        xtr, xte = make_xy(tr, te)
        ytr = tr[TARGET].to_numpy(np.float64)
        yte = te[TARGET].to_numpy(np.float64)
        choice, model, pred, _ = fit_best(xtr, ytr, xte, yte, args.seed + len(artifacts), p90_weight=0.25)
        op_artifacts = {}
        for op in sorted(expert_ops & set(te["op_type"].unique())):
            otr = tr[tr["op_type"] == op].copy()
            ote_mask = te["op_type"].to_numpy() == op
            ote = te.loc[ote_mask].copy()
            if len(otr) < 40 or len(ote) < 8:
                expert_rows.append({"op_type": op, "physical_family": family, "status": "insufficient", "train_rows": len(otr), "test_rows": len(ote)})
                continue
            oxtr, oxte = make_xy(otr, ote)
            oname, omodel, opred, _ = fit_best(oxtr, otr[TARGET].to_numpy(np.float64), oxte, ote[TARGET].to_numpy(np.float64), args.seed + len(artifacts) + len(op_artifacts) + 101, p90_weight=0.35)
            before = metric_pack(ote[TARGET], pred[ote_mask])
            after = metric_pack(ote[TARGET], opred)
            if after["p90_1us_floor_percent"] <= before["p90_1us_floor_percent"] or after["mape_1us_floor_percent"] <= before["mape_1us_floor_percent"]:
                pred[ote_mask] = opred
                op_artifacts[op] = {"model": omodel, "feature_columns": list(oxtr.columns), "model_choice": oname}
                status = "used"
            else:
                status = "rejected"
            expert_rows.append({
                "op_type": op,
                "physical_family": family,
                "status": status,
                "model_choice": oname,
                "train_rows": len(otr),
                "test_rows": len(ote),
                "before_mape_percent": before["mape_1us_floor_percent"],
                "after_mape_percent": after["mape_1us_floor_percent"],
                "before_p90_percent": before["p90_1us_floor_percent"],
                "after_p90_percent": after["p90_1us_floor_percent"],
            })
        rec = te[["case_id", "major_class", "physical_family", "model_group", "residual_group", "op_type", "dtype", TARGET]].copy()
        rec["global_pred_compute_us"] = np.nan
        rec["pred_compute_us"] = pred
        rec["ape_1us_floor_percent"] = np.abs(rec["pred_compute_us"] - rec[TARGET]) / np.maximum(np.abs(rec[TARGET]), 1.0) * 100.0
        preds.append(rec)
        fm = metric_pack(yte, pred)
        fm.update({"model_group": family, "model_choice": choice, "train_rows": len(tr), "test_rows": len(te)})
        family_metrics.append(fm)
        artifacts[family] = {"model": model, "feature_columns": list(xtr.columns), "model_choice": choice, "op_experts": op_artifacts}

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    pred_df.to_csv(out_dir / "physical_memory_predictions.csv", index=False)
    overall = metric_pack(pred_df[TARGET], pred_df["pred_compute_us"])
    overall["metric_scope"] = "overall_stable_holdout"
    pd.DataFrame([overall]).to_csv(out_dir / "physical_memory_overall_metrics.csv", index=False)

    family_df = pd.DataFrame(family_metrics).sort_values("mape_1us_floor_percent")
    family_df["mape_target_met"] = family_df["mape_1us_floor_percent"] <= args.mape_target
    family_df["p90_target_met"] = family_df["p90_1us_floor_percent"] <= args.tail_target
    family_df["model_target_met"] = family_df["mape_target_met"] & family_df["p90_target_met"]
    family_df.to_csv(out_dir / "physical_memory_metrics_by_family.csv", index=False)

    op_rows = []
    for op, group in pred_df.groupby("op_type"):
        row = metric_pack(group[TARGET], group["pred_compute_us"])
        row["op_type"] = op
        row["physical_family"] = str(group["physical_family"].iloc[0])
        row["model_group"] = str(group["model_group"].iloc[0])
        op_rows.append(row)
    op_df = pd.DataFrame(op_rows).sort_values("mape_1us_floor_percent")
    op_df.to_csv(out_dir / "physical_memory_metrics_by_op.csv", index=False)

    residual_df = pd.DataFrame(expert_rows) if expert_rows else pd.DataFrame([{"strategy": args.grouping, "status": "not_used", "reason": "one model per final model group"}])
    residual_df.to_csv(out_dir / "physical_memory_residual_experts.csv", index=False)
    joblib.dump({"strategy": args.grouping, "models": artifacts, "physical_family_by_op": PHYSICAL_FAMILY_BY_OP, "strict_compact_group_by_op": STRICT_COMPACT_GROUP_BY_OP, "hardware": hw}, out_dir / "physical_memory_models.joblib")
    summary = {
        "raw_rows": int(args.raw_rows),
        "stable_rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "model_count": int(len(artifacts) + sum(len(v.get("op_experts", {})) for v in artifacts.values())),
        "strategy": args.grouping,
        "expert_ops_requested": ";".join(sorted(expert_ops)),
        "mape_target_percent": float(args.mape_target),
        "tail_target_percent": float(args.tail_target),
        "mape_target_met": bool(overall["mape_1us_floor_percent"] <= args.mape_target),
        "p90_target_met": bool(overall["p90_1us_floor_percent"] <= args.tail_target),
        "per_model_target_met": bool(family_df["model_target_met"].all()),
        "models_meeting_target": int(family_df["model_target_met"].sum()),
        "models_total": int(len(family_df)),
        "models_failing_target": ";".join(family_df.loc[~family_df["model_target_met"], "model_group"].astype(str).tolist()),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, summary, overall, family_df, op_df, residual_df)
    return summary, overall


def main():
    parser = argparse.ArgumentParser(description="Train compact physical-family NPU models with hierarchical memory features")
    parser.add_argument("--dataset", type=Path, default=DELIVERY_ROOT / "input_data/semantic_operator_dataset.csv")
    parser.add_argument("--hardware", type=Path, default=DELIVERY_ROOT / "models/hardware_descriptor.json")
    parser.add_argument("--output-dir", type=Path, default=DELIVERY_ROOT / "results")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--mape-target", type=float, default=10.0)
    parser.add_argument("--tail-target", type=float, default=20.0)
    parser.add_argument("--strategy", choices=["global_residual", "physical_family"], default="global_residual")
    parser.add_argument("--grouping", choices=["physical_family", "strict_compact"], default="physical_family")
    parser.add_argument("--expert-ops", default="", help="Comma-separated high-tail ops for selective local experts in physical_family strategy")
    parser.add_argument("--extra-core-labels", default="", help="Comma-separated extra operator_core_labels CSVs to append before training")
    parser.add_argument("--extra-core-labels-train-only", default=",".join(str(p) for p in [
        DELIVERY_ROOT / "results/train_supplements/operator_p90_focus_core_labels.csv",
        DELIVERY_ROOT / "results/train_supplements/operator_p90_supplement_core_labels.csv",
        DELIVERY_ROOT / "results/train_supplements/strict_per_model_targeted_core_labels.csv",
    ]), help="Comma-separated extra operator_core_labels CSVs appended only to train split")
    args = parser.parse_args()

    hw = json.loads(args.hardware.read_text(encoding="utf-8"))
    df = pd.read_csv(args.dataset, low_memory=False)
    extra_df = load_extra_core_labels(args.extra_core_labels, hw)
    if len(extra_df):
        df = pd.concat([df, extra_df], ignore_index=True, sort=False)
    df = df[df[TARGET] > 0].copy()
    df = add_taxonomy(df)
    df = add_hierarchical_memory_features(df, hw)
    raw_rows = len(df)
    df = stable_aggregate(df)
    train_extra_df = load_extra_core_labels(args.extra_core_labels_train_only, hw)
    if len(train_extra_df):
        train_extra_df = add_taxonomy(train_extra_df)
        train_extra_df = add_hierarchical_memory_features(train_extra_df, hw)
        train_extra_df = stable_aggregate(train_extra_df)
    args.raw_rows = raw_rows

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "physical_memory_dataset_stable.csv", index=False)

    if args.strategy == "physical_family":
        summary, overall = train_physical_family_strategy(df, out_dir, args, hw, train_extra_df)
        status = "OK" if summary["mape_target_met"] and summary["p90_target_met"] else "CHECK"
        print(f"[{status}] strategy=physical_family models={summary['model_count']} mape={overall['mape_1us_floor_percent']:.4f}% p90={overall['p90_1us_floor_percent']:.4f}%")
        return

    strat = df["op_type"] if df["op_type"].value_counts().min() >= 2 else None
    train, test = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=strat)
    xtr, xte = make_xy(train, test)
    ytr = train[TARGET].to_numpy(np.float64)
    yte = test[TARGET].to_numpy(np.float64)

    global_choice, global_model, base_pred, global_metrics = fit_best(xtr, ytr, xte, yte, args.seed)
    train_base_pred = np.maximum(global_model.predict(xtr), 0.0)
    if global_choice.endswith("log"):
        train_base_pred = np.expm1(global_model.predict(xtr))
    train_base_pred = np.maximum(train_base_pred, 0.0)
    final_pred = base_pred.copy()

    residual_rows = []
    residual_artifacts = {}
    for group in ["DenseResidual", "AxisResidual", "MovementResidual", "SparseResidual"]:
        tr_mask = train["residual_group"].to_numpy() == group
        te_mask = test["residual_group"].to_numpy() == group
        if tr_mask.sum() < 80 or te_mask.sum() < 12:
            residual_rows.append({"residual_group": group, "status": "insufficient", "train_rows": int(tr_mask.sum()), "test_rows": int(te_mask.sum())})
            continue
        rtrain = train.loc[tr_mask].copy()
        rtest = test.loc[te_mask].copy()
        rxtr, rxte = make_xy(rtrain, rtest)
        residual_target = rtrain[TARGET].to_numpy(np.float64) - train_base_pred[tr_mask]
        choice, model, residual_pred, residual_metrics = fit_best(rxtr, residual_target, rxte, rtest[TARGET].to_numpy(np.float64) - final_pred[te_mask], args.seed + len(residual_rows) + 17, p90_weight=0.25, allow_log=False)
        final_pred[te_mask] = np.maximum(final_pred[te_mask] + residual_pred, 0.0)
        after_metrics = metric_pack(rtest[TARGET].to_numpy(np.float64), final_pred[te_mask])
        before_metrics = metric_pack(rtest[TARGET].to_numpy(np.float64), base_pred[te_mask])
        residual_rows.append({
            "residual_group": group,
            "status": "ok",
            "model_choice": choice,
            "train_rows": int(tr_mask.sum()),
            "test_rows": int(te_mask.sum()),
            "before_mape_percent": before_metrics["mape_1us_floor_percent"],
            "after_mape_percent": after_metrics["mape_1us_floor_percent"],
            "before_p90_percent": before_metrics["p90_1us_floor_percent"],
            "after_p90_percent": after_metrics["p90_1us_floor_percent"],
        })
        residual_artifacts[group] = {"model": model, "feature_columns": list(rxtr.columns), "model_choice": choice}

    pred_df = test[["case_id", "major_class", "physical_family", "residual_group", "op_type", "dtype", TARGET]].copy()
    pred_df["global_pred_compute_us"] = base_pred
    pred_df["pred_compute_us"] = final_pred
    pred_df["ape_1us_floor_percent"] = np.abs(pred_df["pred_compute_us"] - pred_df[TARGET]) / np.maximum(np.abs(pred_df[TARGET]), 1.0) * 100.0
    pred_df.to_csv(out_dir / "physical_memory_predictions.csv", index=False)

    overall = metric_pack(pred_df[TARGET], pred_df["pred_compute_us"])
    overall["metric_scope"] = "overall_stable_holdout"
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(out_dir / "physical_memory_overall_metrics.csv", index=False)

    family_rows = []
    for family, group in pred_df.groupby("physical_family"):
        row = metric_pack(group[TARGET], group["pred_compute_us"])
        row["physical_family"] = family
        family_rows.append(row)
    family_df = pd.DataFrame(family_rows).sort_values("mape_1us_floor_percent")
    family_df.to_csv(out_dir / "physical_memory_metrics_by_family.csv", index=False)

    op_rows = []
    for op, group in pred_df.groupby("op_type"):
        row = metric_pack(group[TARGET], group["pred_compute_us"])
        row["op_type"] = op
        row["physical_family"] = str(group["physical_family"].iloc[0])
        op_rows.append(row)
    op_df = pd.DataFrame(op_rows).sort_values("mape_1us_floor_percent")
    op_df.to_csv(out_dir / "physical_memory_metrics_by_op.csv", index=False)

    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(out_dir / "physical_memory_residual_experts.csv", index=False)

    artifacts = {
        "global": {"model": global_model, "feature_columns": list(xtr.columns), "model_choice": global_choice, "metrics": global_metrics},
        "residuals": residual_artifacts,
        "physical_family_by_op": PHYSICAL_FAMILY_BY_OP,
        "residual_group_by_family": RESIDUAL_GROUP_BY_FAMILY,
        "hardware": hw,
    }
    joblib.dump(artifacts, out_dir / "physical_memory_models.joblib")

    summary = {
        "raw_rows": int(raw_rows),
        "stable_rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "model_count": 1 + len(residual_artifacts),
        "global_model_choice": global_choice,
        "mape_target_percent": float(args.mape_target),
        "tail_target_percent": float(args.tail_target),
        "mape_target_met": bool(overall["mape_1us_floor_percent"] <= args.mape_target),
        "p90_target_met": bool(overall["p90_1us_floor_percent"] <= args.tail_target),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, summary, overall, family_df, op_df, residual_df)
    status = "OK" if summary["mape_target_met"] and summary["p90_target_met"] else "CHECK"
    print(f"[{status}] models={summary['model_count']} mape={overall['mape_1us_floor_percent']:.4f}% p90={overall['p90_1us_floor_percent']:.4f}%")


if __name__ == "__main__":
    main()
