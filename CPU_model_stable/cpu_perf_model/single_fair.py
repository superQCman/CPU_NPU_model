from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import train_mlp
from .infer_mlp_onnx import run_inference
from .paths import DEFAULT_SINGLE_MODEL_ROOT


FAIR_SINGLE_HIDDEN_LAYERS = (128, 128, 128, 128, 128, 128)
FAIR_SINGLE_BATCH_SIZE = 1024
FAIR_SINGLE_MAX_ITER = 200
FAIR_SINGLE_ALPHA = 1e-4
FAIR_SINGLE_LEARNING_RATE_INIT = 1e-3
FAIR_SINGLE_SEED = 42


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_grouped_feature_pool(classed_data_root: Path) -> dict[str, Any]:
    dataset_summary = _read_json(classed_data_root / "dataset_summary.json")
    categorical_features = list(dataset_summary.get("shared_categorical_features", []))
    per_group = dataset_summary.get("per_model_group_numeric_features", {})
    ordered_numeric: list[str] = []
    for group_name in dataset_summary.get("model_group_order", []):
        for column in per_group.get(group_name, []):
            if column not in ordered_numeric:
                ordered_numeric.append(column)
    if not ordered_numeric:
        raise RuntimeError(f"No grouped numeric feature pool found in {classed_data_root}")
    return {
        "dataset_summary": dataset_summary,
        "numeric_features": ordered_numeric,
        "categorical_features": categorical_features,
    }


def build_fair_single_dataset(classed_data_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_pool = load_grouped_feature_pool(classed_data_root)
    dataset_frame = pd.read_csv(classed_data_root / "classed_dataset_full.csv", low_memory=False)

    dataset_full_csv = output_dir / "dataset_full.csv"
    dataset_frame.to_csv(dataset_full_csv, index=False)
    split_counts: dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        split_df = dataset_frame[dataset_frame["split"] == split_name].copy()
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)
        split_counts[split_name] = int(len(split_df))

    feature_manifest = {
        "feature_dialect": "cpu_perf_model_fair_single_union",
        "numeric_features": feature_pool["numeric_features"],
        "categorical_features": feature_pool["categorical_features"],
        "analytical_base_column": "ana_calib_total_us",
        "residual_target_column": "label_operator_residual_log",
        "feature_pool_description": "Union of grouped-model numeric features plus shared categorical features.",
    }
    feature_columns_json = output_dir / "feature_columns.json"
    feature_columns_json.write_text(json.dumps(feature_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "classed_data_root": str(classed_data_root),
        "output_dir": str(output_dir),
        "numeric_features": feature_pool["numeric_features"],
        "categorical_features": feature_pool["categorical_features"],
        "feature_count": len(feature_pool["numeric_features"]) + len(feature_pool["categorical_features"]),
        "split_row_counts": split_counts,
        "artifacts": {
            "dataset_full_csv": str(dataset_full_csv),
            "train_csv": str(output_dir / "train.csv"),
            "val_csv": str(output_dir / "val.csv"),
            "test_csv": str(output_dir / "test.csv"),
            "feature_columns_json": str(feature_columns_json),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def train_fair_single_model(
    *,
    classed_data_root: Path,
    output_root: Path = DEFAULT_SINGLE_MODEL_ROOT,
    hidden_layers: Sequence[int] = FAIR_SINGLE_HIDDEN_LAYERS,
    batch_size: int = FAIR_SINGLE_BATCH_SIZE,
    max_iter: int = FAIR_SINGLE_MAX_ITER,
    alpha: float = FAIR_SINGLE_ALPHA,
    learning_rate_init: float = FAIR_SINGLE_LEARNING_RATE_INIT,
    seed: int = FAIR_SINGLE_SEED,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / "dataset"
    model_dir = output_root / "model"
    dataset_manifest = build_fair_single_dataset(classed_data_root, dataset_dir)
    model_summary = train_mlp.train_model(
        data_dir=dataset_dir,
        output_dir=model_dir,
        hidden_layers=tuple(int(value) for value in hidden_layers),
        batch_size=int(batch_size),
        max_iter=int(max_iter),
        alpha=float(alpha),
        learning_rate_init=float(learning_rate_init),
        seed=int(seed),
        log_target=True,
        target_mode="direct_us",
        train_device="auto",
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
        "training_hparams": {
            "hidden_layers": list(hidden_layers),
            "batch_size": int(batch_size),
            "max_iter": int(max_iter),
            "alpha": float(alpha),
            "learning_rate_init": float(learning_rate_init),
            "seed": int(seed),
        },
        "model_summary": model_summary,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def infer_fair_single_model(
    *,
    model_root: Path,
    dataset_dir: Path | None = None,
    input_csv: Path | None = None,
    output_dir: Path | None = None,
    provider: str = "auto",
    device_id: int = 0,
    batch_size: int = 4096,
) -> dict[str, Any]:
    if dataset_dir is None and input_csv is None:
        raise ValueError("dataset_dir or input_csv must be provided for fair single inference")
    target_output_dir = output_dir or (model_root.parent / "inference")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    if input_csv is not None:
        payload = run_inference(
            model_dir=model_root,
            input_csv=input_csv,
            output_csv=target_output_dir / "predictions.csv",
            provider=provider,
            device_id=device_id,
            batch_size=batch_size,
            metrics_json=target_output_dir / "metrics.json",
        )
        results["custom"] = payload
    else:
        for split_name in ["train", "val", "test"]:
            payload = run_inference(
                model_dir=model_root,
                input_csv=dataset_dir / f"{split_name}.csv",
                output_csv=target_output_dir / f"predictions_{split_name}.csv",
                provider=provider,
                device_id=device_id,
                batch_size=batch_size,
                metrics_json=target_output_dir / f"metrics_{split_name}.json",
            )
            results[split_name] = payload

    manifest = {
        "model_root": str(model_root),
        "dataset_dir": "" if dataset_dir is None else str(dataset_dir),
        "input_csv": "" if input_csv is None else str(input_csv),
        "output_dir": str(target_output_dir),
        "provider": provider,
        "device_id": int(device_id),
        "batch_size": int(batch_size),
        "results": results,
    }
    (target_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
