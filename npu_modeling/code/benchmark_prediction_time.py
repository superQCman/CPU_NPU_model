import argparse
import gc
import json
import statistics
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from train_physical_memory_models import add_hierarchical_memory_features, add_taxonomy
from validate_dlrm_end_to_end import feature_frame


DELIVERY_ROOT = Path(__file__).resolve().parents[1]


def load_holdout_rows(dataset_path: Path, predictions_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path, low_memory=False)
    prediction_rows = pd.read_csv(predictions_path, low_memory=False)
    holdout_ids = set(prediction_rows["case_id"].astype(str))
    holdout = dataset[dataset["case_id"].astype(str).isin(holdout_ids)].copy()
    if len(holdout) != len(prediction_rows):
        raise ValueError(
            f"Holdout reconstruction failed: dataset matched {len(holdout)} rows, "
            f"predictions contain {len(prediction_rows)} rows"
        )
    return holdout


def run_once(holdout: pd.DataFrame, model_path: Path) -> dict:
    load_start = time_counter()
    artifacts = joblib.load(model_path)
    load_sec = time_counter() - load_start

    feature_start = time_counter()
    enriched = add_taxonomy(holdout)
    enriched = add_hierarchical_memory_features(enriched, artifacts["hardware"])
    grouped_features = {}
    for group, package in artifacts["models"].items():
        rows = enriched[enriched["model_group"].astype(str) == group]
        if len(rows):
            grouped_features[group] = (rows, feature_frame(rows, package["feature_columns"]))
    feature_build_sec = time_counter() - feature_start

    predict_start = time_counter()
    predictions = np.full(len(enriched), np.nan, dtype=np.float64)
    prediction_index = enriched.index.to_numpy()
    position_by_index = {original: position for position, original in enumerate(prediction_index)}
    for group, (_, features) in grouped_features.items():
        package = artifacts["models"][group]
        raw_prediction = package["model"].predict(features)
        if str(package.get("model_choice", "")).endswith("log"):
            raw_prediction = np.expm1(raw_prediction)
        positions = [position_by_index[index] for index in features.index]
        predictions[positions] = np.maximum(raw_prediction, 0.0)
    predict_sec = time_counter() - predict_start
    if not np.isfinite(predictions).all():
        raise ValueError("Model did not produce finite predictions for every holdout row")

    result = {
        "load_sec": float(load_sec),
        "feature_build_sec": float(feature_build_sec),
        "predict_sec": float(predict_sec),
        "feature_build_and_predict_sec": float(feature_build_sec + predict_sec),
        "total_sec": float(load_sec + feature_build_sec + predict_sec),
        "mean_pred_compute_us": float(np.mean(predictions)),
    }
    del artifacts, enriched, grouped_features
    gc.collect()
    return result


def time_counter() -> float:
    return perf_counter()


def write_markdown(output_path: Path, summary: dict) -> None:
    lines = [
        "# Prediction Timing Report",
        "",
        "Timing excludes CSV input loading. It measures artifact loading, static feature construction, and batched model prediction on the delivered stable holdout rows.",
        "",
        "| 指标 | 数值（秒） |",
        "|---|---:|",
        f"| 模型加载耗时 | {summary['load_sec']:.4f} |",
        f"| 静态特征构建耗时 | {summary['feature_build_sec']:.4f} |",
        f"| 批量预测耗时 | {summary['predict_sec']:.4f} |",
        f"| 静态特征构建和批量预测耗时 | {summary['feature_build_and_predict_sec']:.4f} |",
        f"| 总耗时 | {summary['total_sec']:.4f} |",
        "",
        f"- 测试样本数：{summary['rows']}",
        f"- 重复次数：{summary['runs_measured']}",
        "- 汇总方式：中位数。",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark delivered physical-memory model inference time")
    parser.add_argument("--model", type=Path, default=DELIVERY_ROOT / "results/physical_memory_models.joblib.gz")
    parser.add_argument("--dataset", type=Path, default=DELIVERY_ROOT / "results/physical_memory_dataset_stable.csv")
    parser.add_argument("--predictions", type=Path, default=DELIVERY_ROOT / "results/physical_memory_predictions.csv")
    parser.add_argument("--output", type=Path, default=DELIVERY_ROOT / "results/prediction_time_report.json")
    parser.add_argument("--markdown-output", type=Path, default=DELIVERY_ROOT / "results/PREDICTION_TIME_REPORT.md")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    holdout = load_holdout_rows(args.dataset, args.predictions)
    runs = [run_once(holdout, args.model) for _ in range(args.repeats)]

    keys = [
        "load_sec",
        "feature_build_sec",
        "predict_sec",
        "feature_build_and_predict_sec",
        "total_sec",
        "mean_pred_compute_us",
    ]
    summary = {
        "model": str(args.model),
        "input": str(args.predictions),
        "input_scope": "stable_holdout_rows",
        "rows": int(len(holdout)),
        "runs_measured": int(args.repeats),
        "aggregation": "median",
    }
    summary.update({key: float(statistics.median(run[key] for run in runs)) for key in keys})
    summary["runs"] = runs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.markdown_output, summary)
    print(
        f"[OK] rows={summary['rows']} load={summary['load_sec']:.4f}s "
        f"feature_build_and_predict={summary['feature_build_and_predict_sec']:.4f}s total={summary['total_sec']:.4f}s"
    )


if __name__ == "__main__":
    main()
