from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .feature_contract import (
    ANALYTICAL_RESIDUAL_TARGET_COLUMN,
    BASELINE_CATEGORICAL_FEATURES,
    BASELINE_NUMERIC_FEATURES,
    DEFAULT_FEATURE_DIALECT,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


MISSING_CATEGORICAL_TOKEN = "__missing__"
UNKNOWN_CATEGORICAL_TOKEN = "__unknown__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP regressor on the prepared single-op tables.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing train.csv, val.csv, test.csv, and feature_columns.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the saved model, ONNX export, predictions, and metrics.",
    )
    parser.add_argument(
        "--hidden-layers",
        default="48,48,48,48,48",
        help="Comma-separated MLP hidden sizes.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument("--alpha", type=float, default=1e-4, help="Weight decay used by AdamW.")
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-device",
        default="auto",
        help="PyTorch training device: auto, cpu, cuda, or npu.",
    )
    parser.add_argument(
        "--npu-device-id",
        type=int,
        default=0,
        help="Ascend NPU device id used when train-device resolves to npu.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Stop after this many epochs without validation improvement.",
    )
    parser.add_argument(
        "--disable-log-target",
        action="store_true",
        help="Train directly on microseconds instead of log1p(target).",
    )
    parser.add_argument(
        "--target-mode",
        choices=["direct_us", "analytical_residual"],
        default="direct_us",
        help="Train directly on operator latency or on log(label_us / ana_base_us).",
    )
    parser.add_argument(
        "--disable-onnx-export",
        action="store_true",
        help="Skip exporting the trained PyTorch model to ONNX.",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=17,
        help="ONNX opset used when exporting the MLP.",
    )
    return parser.parse_args()


def parse_hidden_layers(text: str) -> tuple[int, ...]:
    cleaned = [part.strip() for part in str(text).split(",") if part.strip()]
    if not cleaned:
        raise ValueError("At least one hidden layer size is required")
    return tuple(int(part) for part in cleaned)


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _import_torch() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for train_mlp.py. Install it first, for example: pip install torch"
        ) from exc
    torch_npu = None
    if os.environ.get("CPU_PERF_MODEL_ENABLE_NPU", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        try:
            import torch_npu as imported_torch_npu
        except ImportError:
            imported_torch_npu = None
        torch_npu = imported_torch_npu
    return torch, nn, DataLoader, TensorDataset, torch_npu


def load_split_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for split_name in ["train", "val", "test"]:
        path = data_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[split_name] = pd.read_csv(path)
    return tables


def load_feature_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "feature_columns.json"
    if not manifest_path.exists():
        return {
            "feature_dialect": DEFAULT_FEATURE_DIALECT,
            "numeric_features": list(BASELINE_NUMERIC_FEATURES),
            "categorical_features": list(BASELINE_CATEGORICAL_FEATURES),
            "analytical_base_column": "ana_base_us",
            "residual_target_column": ANALYTICAL_RESIDUAL_TARGET_COLUMN,
        }
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("feature_dialect", DEFAULT_FEATURE_DIALECT)
    payload.setdefault("numeric_features", list(BASELINE_NUMERIC_FEATURES))
    payload.setdefault("categorical_features", list(BASELINE_CATEGORICAL_FEATURES))
    payload.setdefault("analytical_base_column", "ana_base_us")
    payload.setdefault("residual_target_column", ANALYTICAL_RESIDUAL_TARGET_COLUMN)
    return payload


def sanitize_categorical_series(series: pd.Series) -> pd.Series:
    text = series.fillna(MISSING_CATEGORICAL_TOKEN).astype(str).str.strip()
    return text.replace({"": MISSING_CATEGORICAL_TOKEN, "nan": MISSING_CATEGORICAL_TOKEN, "None": MISSING_CATEGORICAL_TOKEN})


def configure_torch_npu(torch: Any, torch_npu: Any, npu_device_id: int) -> None:
    if torch_npu is None:
        raise RuntimeError(
            "Requested NPU training, but torch_npu is not installed. "
            "Please install the Ascend PyTorch stack first."
        )
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device_name = f"npu:{int(npu_device_id)}"
    if hasattr(torch_npu.npu, "set_device"):
        torch_npu.npu.set_device(device_name)
    elif hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
        torch.npu.set_device(device_name)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fit_preprocessor_state(
    train_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    numeric_stats: list[dict[str, Any]] = []
    categorical_stats: list[dict[str, Any]] = []
    transformed_feature_names: list[str] = []

    for column in numeric_features:
        if column in train_df.columns:
            values = pd.to_numeric(train_df[column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=train_df.index, dtype=float)
        median = _safe_float(values.median(), 0.0)
        filled = values.fillna(median).astype(float)
        mean = _safe_float(filled.mean(), 0.0)
        std = _safe_float(filled.std(ddof=0), 1.0)
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0
        numeric_stats.append(
            {
                "name": column,
                "median": median,
                "mean": mean,
                "std": std,
            }
        )
        transformed_feature_names.append(column)

    offset = len(numeric_stats)
    for column in categorical_features:
        if column in train_df.columns:
            values = sanitize_categorical_series(train_df[column])
        else:
            values = pd.Series(MISSING_CATEGORICAL_TOKEN, index=train_df.index, dtype=str)
        unique_values = sorted({value for value in values.tolist() if value != UNKNOWN_CATEGORICAL_TOKEN})
        vocabulary = [UNKNOWN_CATEGORICAL_TOKEN, *unique_values]
        categorical_stats.append(
            {
                "name": column,
                "vocabulary": vocabulary,
                "offset": offset,
                "size": len(vocabulary),
            }
        )
        transformed_feature_names.extend([f"{column}={token}" for token in vocabulary])
        offset += len(vocabulary)

    return {
        "version": 1,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "input_dim": offset,
        "transformed_feature_names": transformed_feature_names,
        "missing_categorical_token": MISSING_CATEGORICAL_TOKEN,
        "unknown_categorical_token": UNKNOWN_CATEGORICAL_TOKEN,
    }


def transform_features(df: pd.DataFrame, preprocessor_state: dict[str, Any]) -> np.ndarray:
    rows = len(df)
    input_dim = int(preprocessor_state["input_dim"])
    matrix = np.zeros((rows, input_dim), dtype=np.float32)

    numeric_stats = preprocessor_state["numeric_stats"]
    for index, entry in enumerate(numeric_stats):
        name = str(entry["name"])
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce")
        else:
            values = pd.Series(np.nan, index=df.index, dtype=float)
        median = float(entry["median"])
        mean = float(entry["mean"])
        std = float(entry["std"]) if float(entry["std"]) > 0.0 else 1.0
        normalized = ((values.fillna(median).astype(float) - mean) / std).to_numpy(dtype=np.float32)
        matrix[:, index] = normalized

    for entry in preprocessor_state["categorical_stats"]:
        name = str(entry["name"])
        if name in df.columns:
            values = sanitize_categorical_series(df[name])
        else:
            values = pd.Series(MISSING_CATEGORICAL_TOKEN, index=df.index, dtype=str)
        vocabulary = list(entry["vocabulary"])
        mapping = {token: idx for idx, token in enumerate(vocabulary)}
        offset = int(entry["offset"])
        encoded_indices = [mapping.get(value, 0) for value in values.tolist()]
        row_indices = np.arange(rows, dtype=np.int64)
        matrix[row_indices, offset + np.asarray(encoded_indices, dtype=np.int64)] = 1.0

    return matrix


def transform_target(values: np.ndarray, log_target: bool) -> np.ndarray:
    target = np.asarray(values, dtype=np.float32)
    if not log_target:
        return target
    return np.log1p(np.clip(target, a_min=0.0, a_max=None)).astype(np.float32)


def inverse_transform_target(values: np.ndarray, log_target: bool) -> np.ndarray:
    target = np.asarray(values, dtype=np.float32)
    if not log_target:
        return target
    return np.expm1(np.clip(target, a_min=-20.0, a_max=20.0)).astype(np.float32)


def resolve_target_mode_config(
    feature_manifest: dict[str, Any],
    *,
    target_mode: str,
    log_target_requested: bool,
) -> dict[str, Any]:
    mode = str(target_mode).strip().lower()
    if mode == "direct_us":
        return {
            "target_mode": mode,
            "train_target_column": TARGET_COLUMN,
            "model_log_target": bool(log_target_requested),
            "analytical_base_column": "",
            "prediction_reconstruction": "direct_us",
        }
    if mode == "analytical_residual":
        analytical_base_column = str(feature_manifest.get("analytical_base_column", "ana_base_us"))
        residual_target_column = str(
            feature_manifest.get("residual_target_column", ANALYTICAL_RESIDUAL_TARGET_COLUMN)
        )
        return {
            "target_mode": mode,
            "train_target_column": residual_target_column,
            "model_log_target": False,
            "analytical_base_column": analytical_base_column,
            "prediction_reconstruction": f"{analytical_base_column} * exp({residual_target_column})",
        }
    raise ValueError(f"Unsupported target_mode={target_mode!r}")


def resolve_analytical_base_values(df: pd.DataFrame, analytical_base_column: str) -> np.ndarray:
    if not analytical_base_column or analytical_base_column not in df.columns:
        raise KeyError(
            f"Analytical residual mode requires column {analytical_base_column!r} in the dataset split table"
        )
    return (
        pd.to_numeric(df[analytical_base_column], errors="coerce")
        .fillna(1e-3)
        .clip(lower=1e-3)
        .to_numpy(dtype=np.float32)
    )


def reconstruct_latency_predictions(
    raw_predictions: np.ndarray,
    df: pd.DataFrame,
    *,
    target_mode: str,
    log_target: bool,
    analytical_base_column: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if target_mode == "direct_us":
        predictions = inverse_transform_target(raw_predictions, log_target=log_target)
        return predictions.astype(np.float32), None

    if target_mode == "analytical_residual":
        residual_log = np.asarray(raw_predictions, dtype=np.float32)
        base_us = resolve_analytical_base_values(df, analytical_base_column)
        predictions = base_us * np.exp(np.clip(residual_log, a_min=-20.0, a_max=20.0))
        return predictions.astype(np.float32), residual_log

    raise ValueError(f"Unsupported target_mode={target_mode!r}")


class TorchMLPRegressor:
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], nn_module: Any) -> None:
        layers: list[Any] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn_module.Linear(prev_dim, hidden_dim))
            layers.append(nn_module.ReLU())
            prev_dim = hidden_dim
        layers.append(nn_module.Linear(prev_dim, 1))
        self.network = nn_module.Sequential(*layers)

    def __call__(self, inputs: Any) -> Any:
        outputs = self.network(inputs)
        return outputs.squeeze(-1)

    def parameters(self) -> Any:
        return self.network.parameters()

    def to(self, device: Any) -> "TorchMLPRegressor":
        self.network = self.network.to(device)
        return self

    def train(self) -> None:
        self.network.train()

    def eval(self) -> None:
        self.network.eval()

    def state_dict(self) -> dict[str, Any]:
        return self.network.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.network.load_state_dict(state_dict)


def select_training_device(torch: Any, torch_npu: Any, requested_device: str, npu_device_id: int) -> Any:
    choice = str(requested_device).strip().lower()
    if choice in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch_npu is not None and hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            configure_torch_npu(torch, torch_npu, npu_device_id)
            return torch.device(f"npu:{int(npu_device_id)}")
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested train_device=cuda, but CUDA is not available")
        return torch.device("cuda")
    if choice == "npu":
        configure_torch_npu(torch, torch_npu, npu_device_id)
        if not (hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available()):
            raise RuntimeError("Requested train_device=npu, but PyTorch NPU support is not available after importing torch_npu")
        return torch.device(f"npu:{int(npu_device_id)}")
    return torch.device(choice)


def evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "mae_us": 0.0,
            "rmse_us": 0.0,
            "r2": 0.0,
            "mape": 0.0,
            "median_ape": 0.0,
        }

    clipped_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    denominator = np.clip(y_true, a_min=1e-9, a_max=None)
    rmse = float(np.sqrt(mean_squared_error(y_true, clipped_pred)))
    return {
        "mae_us": float(mean_absolute_error(y_true, clipped_pred)),
        "rmse_us": rmse,
        "r2": float(r2_score(y_true, clipped_pred)),
        "mape": float(np.mean(np.abs(clipped_pred - y_true) / denominator)),
        "median_ape": float(np.median(np.abs(clipped_pred - y_true) / denominator)),
    }


def save_predictions(
    split_name: str,
    df: pd.DataFrame,
    predictions: np.ndarray,
    output_dir: Path,
    *,
    target_mode: str,
    train_target_column: str,
    analytical_base_column: str = "",
    predicted_model_target: np.ndarray | None = None,
) -> dict[str, float]:
    pred = np.clip(predictions, a_min=0.0, a_max=None)
    y_true = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    columns = [column for column in METADATA_COLUMNS if column in df.columns]
    prediction_df = df[columns].copy()
    prediction_df["target_mode"] = target_mode
    prediction_df["target_us"] = y_true
    prediction_df["pred_us"] = pred
    if analytical_base_column and analytical_base_column in df.columns:
        prediction_df[analytical_base_column] = pd.to_numeric(
            df[analytical_base_column],
            errors="coerce",
        )
    if train_target_column != TARGET_COLUMN and train_target_column in df.columns:
        prediction_df[train_target_column] = pd.to_numeric(df[train_target_column], errors="coerce")
    if predicted_model_target is not None and train_target_column != TARGET_COLUMN:
        prediction_df[f"pred_{train_target_column}"] = np.asarray(predicted_model_target, dtype=float)
    prediction_df["abs_error_us"] = np.abs(pred - y_true)
    prediction_df["ape"] = prediction_df["abs_error_us"] / np.clip(prediction_df["target_us"], 1e-9, None)
    prediction_df.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)
    return evaluation_metrics(y_true, pred)


def save_training_history(history: list[dict[str, Any]], output_dir: Path) -> Path:
    history_df = pd.DataFrame(history)
    history_path = output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    return history_path


def plot_training_history(history: list[dict[str, Any]], output_dir: Path) -> str | None:
    plt = _import_matplotlib_pyplot()
    if plt is None or not history:
        return None

    history_df = pd.DataFrame(history)
    if history_df.empty or "epoch" not in history_df.columns:
        return None

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history_df["epoch"], history_df["train_loss"], label="train_loss", linewidth=2)
    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        axis.plot(history_df["epoch"], history_df["val_loss"], label="val_loss", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("MLP Training Loss Curve")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()

    output_path = output_dir / "loss_curve.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return str(output_path)


def _make_tensor_dataset(torch: Any, features: np.ndarray, targets: np.ndarray) -> Any:
    _, _, _, TensorDataset, _ = _import_torch()
    return TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )


def _evaluate_loss(model: TorchMLPRegressor, features: np.ndarray, targets: np.ndarray, device: Any, torch: Any, criterion: Any) -> float | None:
    if len(features) == 0:
        return None
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features.astype(np.float32)).to(device)
        labels = torch.from_numpy(targets.astype(np.float32)).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    return float(loss.detach().cpu().item())


def predict_with_torch(
    model: TorchMLPRegressor,
    features: np.ndarray,
    batch_size: int,
    device: Any,
    torch: Any,
) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0,), dtype=np.float32)

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), max(1, int(batch_size))):
            batch = torch.from_numpy(features[start:start + max(1, int(batch_size))].astype(np.float32)).to(device)
            pred = model(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(pred)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def export_to_onnx(
    model: TorchMLPRegressor,
    input_dim: int,
    onnx_path: Path,
    opset: int,
) -> dict[str, Any]:
    torch, _, _, _, _ = _import_torch()
    model.eval()
    model.to(torch.device("cpu"))
    dummy_input = torch.zeros((1, input_dim), dtype=torch.float32)
    try:
        torch.onnx.export(
            model.network,
            dummy_input,
            str(onnx_path),
            input_names=["features"],
            output_names=["prediction"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "prediction": {0: "batch_size"},
            },
            opset_version=opset,
        )
        return {
            "onnx_path": str(onnx_path),
            "status": "exported",
            "opset": opset,
        }
    except Exception as exc:
        return {
            "onnx_path": str(onnx_path),
            "status": "failed",
            "opset": opset,
            "error": str(exc),
        }


def train_model(
    data_dir: Path,
    output_dir: Path,
    hidden_layers: tuple[int, ...],
    batch_size: int,
    max_iter: int,
    alpha: float,
    learning_rate_init: float,
    seed: int,
    log_target: bool,
    target_mode: str = "direct_us",
    train_device: str = "auto",
    npu_device_id: int = 0,
    early_stopping_patience: int = 12,
    export_onnx: bool = True,
    onnx_opset: int = 17,
) -> dict[str, Any]:
    torch, nn, DataLoader, _, torch_npu = _import_torch()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tables = load_split_tables(data_dir)
    feature_manifest = load_feature_manifest(data_dir)
    target_mode_config = resolve_target_mode_config(
        feature_manifest,
        target_mode=target_mode,
        log_target_requested=log_target,
    )
    train_target_column = str(target_mode_config["train_target_column"])
    model_log_target = bool(target_mode_config["model_log_target"])
    analytical_base_column = str(target_mode_config["analytical_base_column"])
    numeric_features = [
        column
        for column in feature_manifest["numeric_features"]
        if column in tables["train"].columns
    ]
    categorical_features = [
        column
        for column in feature_manifest["categorical_features"]
        if column in tables["train"].columns
    ]
    if not numeric_features and not categorical_features:
        raise RuntimeError("No feature columns were found in the training table")
    for split_name, frame in tables.items():
        if train_target_column not in frame.columns:
            raise KeyError(
                f"Target column {train_target_column!r} is missing from {split_name}.csv. "
                f"Requested target_mode={target_mode!r}."
            )
        if target_mode == "analytical_residual":
            resolve_analytical_base_values(frame, analytical_base_column)

    preprocessor_state = fit_preprocessor_state(
        train_df=tables["train"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    feature_matrices = {
        split_name: transform_features(frame, preprocessor_state)
        for split_name, frame in tables.items()
    }
    target_arrays = {
        split_name: pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float32)
        for split_name, frame in tables.items()
    }
    model_target_arrays = {
        split_name: pd.to_numeric(frame[train_target_column], errors="coerce").to_numpy(dtype=np.float32)
        for split_name, frame in tables.items()
    }
    transformed_targets = {
        split_name: transform_target(targets, log_target=model_log_target)
        for split_name, targets in model_target_arrays.items()
    }

    device = select_training_device(torch, torch_npu, train_device, npu_device_id)
    model = TorchMLPRegressor(
        input_dim=int(preprocessor_state["input_dim"]),
        hidden_layers=hidden_layers,
        nn_module=nn,
    ).to(device)

    train_dataset = _make_tensor_dataset(torch, feature_matrices["train"], transformed_targets["train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, min(int(batch_size), max(1, len(train_dataset)))),
        shuffle=True,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate_init,
        weight_decay=alpha,
        foreach=False,
        fused=False,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = None
    epochs_without_improvement = 0
    train_history: list[dict[str, Any]] = []

    for epoch in range(1, int(max_iter) + 1):
        model.train()
        epoch_losses: list[float] = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss = _evaluate_loss(
            model=model,
            features=feature_matrices["val"],
            targets=transformed_targets["val"],
            device=device,
            torch=torch,
            criterion=criterion,
        )
        score_loss = val_loss if val_loss is not None else train_loss
        improved = best_val_loss is None or score_loss < best_val_loss
        if improved:
            best_val_loss = score_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        train_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if epochs_without_improvement >= max(1, int(early_stopping_patience)):
            break

    model.load_state_dict(best_state)

    metrics: dict[str, Any] = {}
    for split_name, frame in tables.items():
        raw_predictions = predict_with_torch(
            model=model,
            features=feature_matrices[split_name],
            batch_size=batch_size,
            device=device,
            torch=torch,
        )
        predictions, predicted_model_target = reconstruct_latency_predictions(
            raw_predictions,
            frame,
            target_mode=target_mode,
            log_target=model_log_target,
            analytical_base_column=analytical_base_column,
        )
        metrics[split_name] = save_predictions(
            split_name,
            frame,
            predictions,
            output_dir,
            target_mode=target_mode,
            train_target_column=train_target_column,
            analytical_base_column=analytical_base_column,
            predicted_model_target=predicted_model_target,
        )

    model_path = output_dir / "mlp_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_layers": list(hidden_layers),
            "input_dim": int(preprocessor_state["input_dim"]),
            "log_target": model_log_target,
            "target_column": TARGET_COLUMN,
            "train_target_column": train_target_column,
            "target_mode": target_mode,
            "analytical_base_column": analytical_base_column,
        },
        model_path,
    )
    preprocessor_path = output_dir / "preprocessor_state.json"
    with preprocessor_path.open("w", encoding="utf-8") as handle:
        json.dump(preprocessor_state, handle, indent=2, ensure_ascii=False)
    training_history_path = save_training_history(train_history, output_dir)
    loss_curve_path = plot_training_history(train_history, output_dir)

    onnx_export = {
        "status": "skipped",
        "onnx_path": str(output_dir / "mlp_model.onnx"),
        "opset": onnx_opset,
    }
    if export_onnx:
        onnx_export = export_to_onnx(
            model=model,
            input_dim=int(preprocessor_state["input_dim"]),
            onnx_path=output_dir / "mlp_model.onnx",
            opset=onnx_opset,
        )

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "model_type": "pytorch_mlp",
        "feature_dialect": feature_manifest.get("feature_dialect", DEFAULT_FEATURE_DIALECT),
        "target_column": TARGET_COLUMN,
        "train_target_column": train_target_column,
        "target_mode": target_mode,
        "prediction_reconstruction": target_mode_config["prediction_reconstruction"],
        "analytical_base_column": analytical_base_column,
        "feature_count": len(numeric_features) + len(categorical_features),
        "input_dim_after_encoding": int(preprocessor_state["input_dim"]),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "hidden_layers": list(hidden_layers),
        "batch_size": batch_size,
        "max_iter": max_iter,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "seed": seed,
        "train_device": str(device),
        "torch_npu_enabled": bool(torch_npu is not None),
        "npu_device_id": int(npu_device_id),
        "log_target_requested": log_target,
        "log_target": model_log_target,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_trained": len(train_history),
        "training_history": train_history,
        "training_history_tail": train_history[-10:],
        "metrics": metrics,
        "onnx_export": onnx_export,
        "artifacts": {
            "model_pt": str(model_path),
            "preprocessor_state_json": str(preprocessor_path),
            "training_history_csv": str(training_history_path),
            "loss_curve_png": loss_curve_path,
            "metrics_json": str(output_dir / "metrics.json"),
            "predictions_train_csv": str(output_dir / "predictions_train.csv"),
            "predictions_val_csv": str(output_dir / "predictions_val.csv"),
            "predictions_test_csv": str(output_dir / "predictions_test.csv"),
            "model_onnx": str(output_dir / "mlp_model.onnx"),
        },
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    model_manifest = {
        "model_type": summary["model_type"],
        "data_dir": summary["data_dir"],
        "output_dir": summary["output_dir"],
        "feature_dialect": summary["feature_dialect"],
        "target_mode": summary["target_mode"],
        "train_target_column": summary["train_target_column"],
        "analytical_base_column": summary["analytical_base_column"],
        "numeric_features": summary["numeric_features"],
        "categorical_features": summary["categorical_features"],
        "hidden_layers": summary["hidden_layers"],
        "batch_size": summary["batch_size"],
        "max_iter": summary["max_iter"],
        "alpha": summary["alpha"],
        "learning_rate_init": summary["learning_rate_init"],
        "seed": summary["seed"],
        "log_target": summary["log_target"],
        "artifacts": summary["artifacts"],
        "onnx_export": summary["onnx_export"],
    }
    with (output_dir / "model_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(model_manifest, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    summary = train_model(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        hidden_layers=parse_hidden_layers(args.hidden_layers),
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        seed=args.seed,
        log_target=not args.disable_log_target,
        target_mode=args.target_mode,
        train_device=args.train_device,
        npu_device_id=args.npu_device_id,
        early_stopping_patience=args.early_stopping_patience,
        export_onnx=not args.disable_onnx_export,
        onnx_opset=args.onnx_opset,
    )
    print(f"model_pt={summary['artifacts']['model_pt']}")
    print(f"preprocessor_state_json={summary['artifacts']['preprocessor_state_json']}")
    print(f"metrics_json={summary['artifacts']['metrics_json']}")
    print(f"onnx_status={summary['onnx_export']['status']}")
    print(f"model_onnx={summary['artifacts']['model_onnx']}")


if __name__ == "__main__":
    main()
