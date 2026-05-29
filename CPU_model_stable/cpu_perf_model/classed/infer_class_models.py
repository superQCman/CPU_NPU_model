from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..infer_mlp_onnx import run_inference
from .train_class_models import combine_predictions, load_dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped ONNX inference for all classed MLP models and merge the results.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Classed dataset root produced by build_classed_dataset.py.",
    )
    parser.add_argument(
        "--model-root",
        default="",
        help="Directory containing one frozen model subdirectory per model group. Defaults to <data-root>/models.",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "cann", "openvino", "cpu"],
        help="Execution provider preference.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def infer_all_classes(
    *,
    data_root: Path,
    model_root: Path,
    provider: str = "auto",
    device_id: int = 0,
    batch_size: int = 4096,
) -> dict[str, Any]:
    dataset_summary = load_dataset_summary(data_root)
    model_group_order = list(dataset_summary.get("model_group_order", []))
    if not model_group_order:
        raise RuntimeError(f"No model_group_order found in {data_root / 'dataset_summary.json'}")

    inference_models: dict[str, Any] = {}
    for model_group in model_group_order:
        group_model_dir = model_root / model_group
        group_data_dir = data_root / "datasets" / model_group
        split_outputs: dict[str, Any] = {}
        for split_name in ["train", "val", "test"]:
            payload = run_inference(
                model_dir=group_model_dir,
                input_csv=group_data_dir / f"{split_name}.csv",
                output_csv=group_model_dir / f"predictions_{split_name}.csv",
                provider=provider,
                device_id=device_id,
                batch_size=batch_size,
                metrics_json=group_model_dir / f"inference_metrics_{split_name}.json",
            )
            split_outputs[split_name] = payload
        inference_models[model_group] = {
            "model_dir": str(group_model_dir),
            "data_dir": str(group_data_dir),
            "splits": split_outputs,
        }

    combined_payload = combine_predictions(data_root, model_root, inference_models, model_group_order)
    payload = {
        "data_root": str(data_root),
        "model_root": str(model_root),
        "provider": provider,
        "device_id": int(device_id),
        "batch_size": int(batch_size),
        "model_group_order": model_group_order,
        "models": inference_models,
        "combined": combined_payload,
    }
    with (model_root / "inference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    model_root = Path(args.model_root) if args.model_root else data_root / "models"
    payload = infer_all_classes(
        data_root=data_root,
        model_root=model_root,
        provider=args.provider,
        device_id=args.device_id,
        batch_size=args.batch_size,
    )
    print(json.dumps(payload["combined"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
