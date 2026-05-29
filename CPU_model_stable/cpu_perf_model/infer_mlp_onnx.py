from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_contract import METADATA_COLUMNS, TARGET_COLUMN
from .train_mlp import (
    evaluation_metrics,
    reconstruct_latency_predictions,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exported single-op MLP ONNX inference on CPU or NPU.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing mlp_model.onnx and preprocessor_state.json.",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV file to score. Typically train.csv, val.csv, test.csv, or dataset_full.csv.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="CSV file where predictions will be written.",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "cann", "openvino", "cpu"],
        help="Execution provider preference. auto prefers NPU providers and falls back to CPU.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="NPU device id used by CANNExecutionProvider.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size used when running ONNX inference.",
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional path for saving inference metrics when the target column is present.",
    )
    return parser.parse_args()


def import_onnxruntime() -> tuple[Any, list[str]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX/NPU inference. Install a build with your target provider first."
        ) from exc
    return ort, ort.get_available_providers()


def resolve_provider_chain(
    available_providers: list[str],
    requested_provider: str,
    device_id: int,
) -> tuple[list[Any], str]:
    cpu_chain: list[Any] = ["CPUExecutionProvider"]
    cann_chain: list[Any] = [
        (
            "CANNExecutionProvider",
            {
                "device_id": str(device_id),
                "precision_mode": "force_fp32",
                "op_select_impl_mode": "high_performance",
                "arena_extend_strategy": "kNextPowerOfTwo",
                "enable_cann_graph": "0",
            },
        ),
        "CPUExecutionProvider",
    ]
    openvino_chain: list[Any] = [
        (
            "OpenVINOExecutionProvider",
            {
                "device_type": "NPU",
            },
        ),
        "CPUExecutionProvider",
    ]

    request = requested_provider.strip().lower()
    has_cann = "CANNExecutionProvider" in available_providers
    has_openvino = "OpenVINOExecutionProvider" in available_providers

    if request == "cann":
        if not has_cann:
            raise RuntimeError(
                f"CANNExecutionProvider is not available. available_providers={available_providers}"
            )
        return cann_chain, "CANNExecutionProvider"

    if request == "openvino":
        if not has_openvino:
            raise RuntimeError(
                f"OpenVINOExecutionProvider is not available. available_providers={available_providers}"
            )
        return openvino_chain, "OpenVINOExecutionProvider"

    if request == "cpu":
        return cpu_chain, "CPUExecutionProvider"

    if has_cann:
        return cann_chain, "CANNExecutionProvider"
    if has_openvino:
        return openvino_chain, "OpenVINOExecutionProvider"
    return cpu_chain, "CPUExecutionProvider"


def load_preprocessor_state(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "preprocessor_state.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_runtime_config(model_dir: Path) -> dict[str, Any]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "log_target": True,
            "target_mode": "direct_us",
            "analytical_base_column": "",
            "train_target_column": TARGET_COLUMN,
        }
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "log_target": bool(payload.get("log_target", True)),
        "target_mode": str(payload.get("target_mode", "direct_us")),
        "analytical_base_column": str(payload.get("analytical_base_column", "")),
        "train_target_column": str(payload.get("train_target_column", TARGET_COLUMN)),
    }


def run_session_predictions(
    session: Any,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0,), dtype=np.float32)

    input_name = session.get_inputs()[0].name
    predictions: list[np.ndarray] = []
    for start in range(0, len(features), max(1, int(batch_size))):
        batch = features[start:start + max(1, int(batch_size))].astype(np.float32, copy=False)
        output = session.run(None, {input_name: batch})[0]
        output = np.asarray(output, dtype=np.float32).reshape(-1)
        predictions.append(output)
    return np.concatenate(predictions, axis=0) if predictions else np.empty((0,), dtype=np.float32)


def build_output_frame(
    input_df: pd.DataFrame,
    predictions: np.ndarray,
    *,
    target_mode: str,
    train_target_column: str,
    analytical_base_column: str,
    predicted_model_target: np.ndarray | None,
) -> pd.DataFrame:
    columns = [column for column in METADATA_COLUMNS if column in input_df.columns]
    output_df = input_df[columns].copy()
    output_df["target_mode"] = target_mode
    output_df["pred_us"] = np.clip(predictions, a_min=0.0, a_max=None)
    if analytical_base_column and analytical_base_column in input_df.columns:
        output_df[analytical_base_column] = pd.to_numeric(
            input_df[analytical_base_column],
            errors="coerce",
        )
    if train_target_column != TARGET_COLUMN and train_target_column in input_df.columns:
        output_df[train_target_column] = pd.to_numeric(input_df[train_target_column], errors="coerce")
    if predicted_model_target is not None and train_target_column != TARGET_COLUMN:
        output_df[f"pred_{train_target_column}"] = np.asarray(predicted_model_target, dtype=float)
    if TARGET_COLUMN in input_df.columns:
        target = pd.to_numeric(input_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        output_df["target_us"] = target
        output_df["abs_error_us"] = np.abs(output_df["pred_us"] - output_df["target_us"])
        output_df["ape"] = output_df["abs_error_us"] / np.clip(output_df["target_us"], 1e-9, None)
    return output_df


def run_inference(
    *,
    model_dir: Path,
    input_csv: Path,
    output_csv: Path,
    provider: str = "auto",
    device_id: int = 0,
    batch_size: int = 4096,
    metrics_json: Path | None = None,
) -> dict[str, Any]:
    model_path = model_dir / "mlp_model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    ort, available_providers = import_onnxruntime()
    providers, resolved_provider = resolve_provider_chain(
        available_providers=available_providers,
        requested_provider=provider,
        device_id=device_id,
    )

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)

    preprocessor_state = load_preprocessor_state(model_dir)
    runtime_config = load_model_runtime_config(model_dir)
    input_df = pd.read_csv(input_csv)
    features = transform_features(input_df, preprocessor_state)
    raw_predictions = run_session_predictions(session, features, batch_size=batch_size)
    predictions, predicted_model_target = reconstruct_latency_predictions(
        raw_predictions,
        input_df,
        target_mode=runtime_config["target_mode"],
        log_target=runtime_config["log_target"],
        analytical_base_column=runtime_config["analytical_base_column"],
    )

    output_df = build_output_frame(
        input_df,
        predictions,
        target_mode=runtime_config["target_mode"],
        train_target_column=runtime_config["train_target_column"],
        analytical_base_column=runtime_config["analytical_base_column"],
        predicted_model_target=predicted_model_target,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    metrics_summary = None
    if TARGET_COLUMN in input_df.columns:
        target = pd.to_numeric(input_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        metrics_summary = evaluation_metrics(target, predictions.astype(float))

    payload = {
        "model_dir": str(model_dir),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "requested_provider": provider,
        "resolved_provider": resolved_provider,
        "available_providers": available_providers,
        "metrics": metrics_summary,
    }
    if metrics_json is not None:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with metrics_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    payload = run_inference(
        model_dir=Path(args.model_dir),
        input_csv=Path(args.input_csv),
        output_csv=output_path,
        provider=args.provider,
        device_id=args.device_id,
        batch_size=args.batch_size,
        metrics_json=Path(args.metrics_json) if args.metrics_json else None,
    )
    print(f"output_csv={output_path}")
    print(f"resolved_provider={payload['resolved_provider']}")
    if payload["metrics"] is not None:
        print(f"mae_us={payload['metrics']['mae_us']:.6f}")
        print(f"rmse_us={payload['metrics']['rmse_us']:.6f}")
        print(f"r2={payload['metrics']['r2']:.6f}")


if __name__ == "__main__":
    main()
