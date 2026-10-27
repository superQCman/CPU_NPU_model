import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

DELIVERY_ROOT = Path(__file__).resolve().parents[1]

from build_semantic_operator_dataset import build_record  # noqa: E402
from train_physical_memory_models import add_hierarchical_memory_features, add_taxonomy  # noqa: E402


ATEN_TO_OP = {
    "aten::linear": "MatMulV2",
    "aten::bmm": "BatchMatMulV2",
    "aten::relu": "Relu",
    "aten::sigmoid": "Sigmoid",
    "aten::cat": "ConcatD",
    "aten::index": "AdvancedIndexing",
}

OP_TO_FAMILY = {
    "MatMulV2": "DenseLinearAlgebra",
    "BatchMatMulV2": "DenseLinearAlgebra",
    "Relu": "ElementwiseActivation",
    "Sigmoid": "ElementwiseActivation",
    "ConcatD": "TensorTransformMove",
    "AdvancedIndexing": "IndexingSparse",
}


def fnum(v: Any) -> float:
    try:
        if pd.isna(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def parse_shapes(raw: Any) -> List[List[int]]:
    s = str(raw or "").replace('"', "").strip()
    if not s or s.upper() == "N/A":
        return []
    out = []
    for part in s.split(";"):
        nums = re.findall(r"-?\d+", part)
        if nums:
            out.append([int(x) for x in nums])
    return out


def prod(xs: List[int]) -> int:
    v = 1
    for x in xs:
        v *= max(int(x), 1)
    return int(v)


def extract_invocation(row: pd.Series, row_index: int, hw: Dict[str, float]) -> Optional[Dict[str, Any]]:
    name = str(row.get("Name", ""))
    op_type = ATEN_TO_OP.get(name)
    if op_type is None:
        return None
    shapes = parse_shapes(row.get("Input Shapes", ""))
    params: Dict[str, Any] = {}
    if op_type == "MatMulV2":
        if len(shapes) < 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
            return None
        x, w = shapes[0], shapes[1]
        params = {"m": float(prod(x[:-1])), "k": float(x[-1]), "n": float(w[0])}
    elif op_type == "BatchMatMulV2":
        if len(shapes) < 2 or len(shapes[0]) < 3 or len(shapes[1]) < 2:
            return None
        a, b = shapes[0], shapes[1]
        params = {"batch": float(prod(a[:-2])), "m": float(a[-2]), "k": float(a[-1]), "n": float(b[-1])}
    elif op_type in {"Relu", "Sigmoid"}:
        if not shapes:
            return None
        s = shapes[0]
        params = {"rows": float(prod(s[:-1])) if len(s) > 1 else 1.0, "cols": float(s[-1]) if s else 1.0}
    elif op_type == "ConcatD":
        if not shapes:
            return None
        features = [float(prod(shp[1:])) if len(shp) > 1 else float(shp[0]) for shp in shapes[:3] if shp]
        if not features:
            return None
        params = {"batch": float(shapes[0][0]) if shapes[0] else 1.0, "feature_a": features[0], "feature_b": features[1] if len(features) > 1 else 0.0, "feature_c": sum(features[2:]) if len(features) > 2 else 0.0}
    elif op_type == "AdvancedIndexing":
        nums = [int(x) for x in re.findall(r"-?\d+", str(row.get("Input Shapes", "")))]
        if len(nums) < 3:
            return None
        outer, d1, d2 = nums[0], nums[1], nums[2]
        index_count = 36 if len(nums) < 6 else max(nums[-1], 1)
        params = {"outer": float(outer), "d1": float(d1), "d2": float(d2), "index_count": float(index_count), "pattern": "pair_select"}
    rec = build_record(
        case_id="app",
        profile_dir="dlrm_e2e",
        major_class=OP_TO_FAMILY[op_type],
        op_type=op_type,
        dtype="float32",
        params=params,
        target_us=max(fnum(row.get("Device Total Duration(us)")), fnum(row.get("Device Total Duration With AICore(us)"))),
        hw=hw,
        latency_p50=0.0,
        latency_mean=0.0,
    )
    rec.update({
        "operator_row_index": row_index,
        "source_name": name,
        "input_shapes": str(row.get("Input Shapes", "")),
        "real_device_total_us": fnum(row.get("Device Total Duration(us)")),
        "real_device_aicore_us": fnum(row.get("Device Total Duration With AICore(us)")),
        "host_total_us": fnum(row.get("Host Total Duration(us)")),
    })
    return rec


def feature_frame(rows: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    drop = {
        "target_kernel_compute_us", "case_id", "profile_dir", "major_class", "latency_p50_us", "latency_mean_us",
        "kernel_count_observed", "kernel_type_unique_observed", "share_ai_core_observed", "share_ai_vector_core_observed", "share_mix_aiv_observed",
        "operator_row_index", "source_name", "input_shapes", "real_device_total_us", "real_device_aicore_us", "host_total_us", "case_name",
    }
    cols = [c for c in rows.columns if c not in drop]
    x = rows[cols].copy()
    for c in list(x.columns):
        if c in {"op_type", "dtype", "physical_family", "residual_group", "model_group"}:
            x[c] = x[c].astype(str).fillna("missing")
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x = pd.get_dummies(x, columns=[c for c in ["op_type", "dtype", "physical_family", "residual_group", "model_group"] if c in x.columns])
    return x.reindex(columns=feature_columns, fill_value=0.0)


def predict_rows(inv: pd.DataFrame, artifacts: Dict[str, Any], hw: Dict[str, float]) -> pd.DataFrame:
    out = add_taxonomy(inv)
    out = add_hierarchical_memory_features(out, hw)
    out["pred_compute_us"] = np.nan
    out["model_level"] = "none"
    models = artifacts.get("models", {})
    for group, pkg in models.items():
        mask = out["model_group"].astype(str) == group
        if not mask.any():
            continue
        x = feature_frame(out.loc[mask], pkg["feature_columns"])
        pred = pkg["model"].predict(x)
        if str(pkg.get("model_choice", "")).endswith("log"):
            pred = np.expm1(pred)
        out.loc[mask, "pred_compute_us"] = np.maximum(pred, 0.0)
        out.loc[mask, "model_level"] = group
    out["is_predicted"] = np.isfinite(out["pred_compute_us"]).astype(int)
    return out


def find_profiler_outputs(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("case_*/profiling/rank0/**/ASCEND_PROFILER_OUTPUT"))


def metric_pack(values):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return {"count": int(len(vals)), "mean": float(np.mean(vals)), "p50": float(np.percentile(vals, 50)), "p90": float(np.percentile(vals, 90))} if len(vals) else {"count": 0, "mean": np.nan, "p50": np.nan, "p90": np.nan}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(DELIVERY_ROOT))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description="Validate physical-memory model on multi-config DLRM profiles")
    parser.add_argument("--results-dir", type=Path, default=DELIVERY_ROOT / "input_data/dlrm_profiles/results_20260426_135144")
    parser.add_argument("--model-artifact", type=Path, default=DELIVERY_ROOT / "results/physical_memory_models.joblib.gz")
    parser.add_argument("--hardware", type=Path, default=DELIVERY_ROOT / "models/hardware_descriptor.json")
    parser.add_argument("--output-dir", type=Path, default=DELIVERY_ROOT / "results/dlrm_end_to_end")
    args = parser.parse_args()
    hw = json.loads(args.hardware.read_text(encoding="utf-8"))
    artifacts = joblib.load(args.model_artifact)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_summaries = []
    all_inv = []
    for prof in find_profiler_outputs(args.results_dir):
        case_name = prof.parents[3].name if len(prof.parents) >= 4 else prof.parent.name
        op_csv = prof / "operator_details.csv"
        step_csv = prof / "step_trace_time.csv"
        if not op_csv.exists():
            continue
        op_df = pd.read_csv(op_csv)
        has_linear = (op_df["Name"].astype(str) == "aten::linear").any()
        records = []
        for i, row in op_df.iterrows():
            if has_linear and str(row.get("Name", "")) == "aten::addmm":
                continue
            rec = extract_invocation(row, int(i), hw)
            if rec:
                rec["case_name"] = case_name
                records.append(rec)
        inv = pd.DataFrame(records)
        if inv.empty:
            continue
        pred = predict_rows(inv, artifacts, hw)
        all_inv.append(pred)

        stage_real = 0.0
        npu_compute_real = 0.0
        fwd = op_df[op_df["Name"].astype(str) == "DLRM forward"]
        if not fwd.empty:
            stage_real = fnum(fwd["Host Total Duration(us)"].iloc[0])
        if step_csv.exists():
            step_df = pd.read_csv(step_csv)
            if not step_df.empty:
                npu_compute_real = fnum(step_df["Computing"].iloc[0]) if "Computing" in step_df else 0.0
                if stage_real <= 0 and "Stage" in step_df:
                    stage_real = fnum(step_df["Stage"].iloc[0])
        modeled = pred[pred["is_predicted"] == 1].copy()
        modeled_real = float(modeled["real_device_total_us"].sum())
        modeled_pred = float(modeled["pred_compute_us"].sum())
        npu_pred_replace = max(npu_compute_real - modeled_real, 0.0) + modeled_pred if npu_compute_real > 0 else np.nan
        stage_pred_replace = max(stage_real - modeled_real, 0.0) + modeled_pred if stage_real > 0 else np.nan
        case_summaries.append({
            "case_name": case_name,
            "modeled_operator_count": int(len(modeled)),
            "modeled_real_device_total_us": modeled_real,
            "modeled_pred_compute_us": modeled_pred,
            "modeled_sum_ape_percent": abs(modeled_pred - modeled_real) / max(modeled_real, 1e-9) * 100.0,
            "npu_compute_real_us": npu_compute_real,
            "npu_compute_pred_replace_modeled_us": npu_pred_replace,
            "npu_compute_replace_modeled_ape_percent": abs(npu_pred_replace - npu_compute_real) / max(npu_compute_real, 1e-9) * 100.0 if npu_compute_real > 0 else np.nan,
            "stage_real_us": stage_real,
            "stage_pred_replace_modeled_us": stage_pred_replace,
            "stage_replace_modeled_ape_percent": abs(stage_pred_replace - stage_real) / max(stage_real, 1e-9) * 100.0 if stage_real > 0 else np.nan,
        })

    summary = pd.DataFrame(case_summaries)
    inv_all = pd.concat(all_inv, ignore_index=True) if all_inv else pd.DataFrame()
    summary.to_csv(args.output_dir / "dlrm_case_summary.csv", index=False)
    inv_all.to_csv(args.output_dir / "dlrm_invocation_predictions.csv", index=False)

    metrics = []
    for col in ["modeled_sum_ape_percent", "npu_compute_replace_modeled_ape_percent", "stage_replace_modeled_ape_percent"]:
        row = metric_pack(summary[col] if col in summary else [])
        row["metric"] = col
        metrics.append(row)
    metrics_df = pd.DataFrame(metrics)[["metric", "count", "mean", "p50", "p90"]]
    metrics_df.to_csv(args.output_dir / "dlrm_overall_metrics.csv", index=False)
    report_metrics = metrics_df[metrics_df["metric"].isin([
        "npu_compute_replace_modeled_ape_percent",
        "stage_replace_modeled_ape_percent",
    ])].copy()
    by_op = []
    if not inv_all.empty:
        for op, g in inv_all.groupby("op_type"):
            real = pd.to_numeric(g["real_device_total_us"], errors="coerce").fillna(0.0)
            pred = pd.to_numeric(g["pred_compute_us"], errors="coerce").fillna(0.0)
            by_op.append({"op_type": op, "count": int(len(g)), "real_sum_us": float(real.sum()), "pred_sum_us": float(pred.sum()), "sum_ape_percent": float(abs(pred.sum() - real.sum()) / max(real.sum(), 1e-9) * 100.0)})
    by_op_df = pd.DataFrame(by_op).sort_values("sum_ape_percent", ascending=False) if by_op else pd.DataFrame()
    by_op_df.to_csv(args.output_dir / "dlrm_by_op_summary.csv", index=False)
    lines = [
        "# DLRM End-to-End NPU Validation",
        "",
        f"- Source profiles: `{display_path(args.results_dir)}`",
        f"- Cases evaluated: `{len(summary)}`",
        "- DLRM profiles are not used in model training.",
        "- CPU, communication, synchronization, transfer, and unmodeled NPU time remain measured.",
        "- Reported NPU error replaces modeled NPU compute operator invocations with model predictions.",
        "",
        "## Replacement Metrics",
        report_metrics.to_markdown(index=False),
        "",
        "## Notes",
        "- Operator-level and modeled-operator-sum diagnostics are retained as CSV artifacts for engineering analysis but are not part of the delivery acceptance report.",
    ]
    (args.output_dir / "DLRM_END_TO_END_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] DLRM validation written to {args.output_dir}; cases={len(summary)}")


if __name__ == "__main__":
    main()
