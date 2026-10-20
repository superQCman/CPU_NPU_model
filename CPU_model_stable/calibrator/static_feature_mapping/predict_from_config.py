from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import lzma
import pickle
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytical_base import run as base_run
from calibrator.ml_semantic_routes import (
    WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
    WHITEBOX_PRIOR_ONLY_ROUTE,
    WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
)
from calibrator.static_feature_contract import augment_static_feature_frame
from calibrator.whitebox_family_routes import (
    fit_whitebox_family_physics_guarded_rf_bundle,
    fit_whitebox_family_prior_bundle,
    fit_whitebox_family_simple_rf_bundle,
    predict_whitebox_family_physics_guarded_rf,
    predict_whitebox_family_prior,
    predict_whitebox_family_simple_rf,
)
from cpu_perf_model.graph_contract import ComboArtifactPaths, ComboSpec
from cpu_perf_model.schedule_engine import build_op_graph, schedule_combo

from calibrator.static_feature_mapping.core import (
    FULL_GRAPH_76,
    MODEL_INPUT_51,
    DlrmStaticConfig,
    build_static_feature_frame,
    build_static_op_shape_table,
)


SUPPORTED_ROUTES = (
    WHITEBOX_PRIOR_ONLY_ROUTE,
    WHITEBOX_PRIOR_SIMPLE_RF_ROUTE,
    WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate static DLRM operator features from top-level config and run "
            "trained whitebox route inference."
        ),
    )
    parser.add_argument("--model-bundle", default="", help="Exported pickle bundle (.pkl), XZ bundle (.pkl.xz), or split XZ manifest (.pkl.xz.parts.json).")
    parser.add_argument("--train-root", default="", help="Legacy mode: split root containing train.csv for on-the-fly fitting.")
    parser.add_argument("--hw-config", default="", help="Legacy mode: whitebox hardware YAML for on-the-fly fitting.")
    parser.add_argument("--route", default=WHITEBOX_PHYSICS_GUARDED_RF_ROUTE, choices=SUPPORTED_ROUTES)
    parser.add_argument("--policy", default="static_config")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-indices-per-lookup", type=int, required=True)
    parser.add_argument("--arch-embedding-size", required=True)
    parser.add_argument("--arch-mlp-bot", required=True)
    parser.add_argument("--arch-mlp-top", default="1600-800-1")
    parser.add_argument("--inter-threads", type=int, required=True)
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not str(args.model_bundle).strip() and (
        not str(args.train_root).strip() or not str(args.hw_config).strip()
    ):
        parser.error("Either --model-bundle, or both --train-root and --hw-config, must be provided.")
    return args


def _load_train_frame(train_root: Path) -> pd.DataFrame:
    train_path = train_root / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found under {train_root}")
    frame = pd.read_csv(train_path, low_memory=False)
    if base_run.TARGET_COLUMN not in frame.columns:
        raise KeyError(f"{train_path} is missing {base_run.TARGET_COLUMN}")
    frame[base_run.TARGET_US_COLUMN] = pd.to_numeric(frame[base_run.TARGET_COLUMN], errors="coerce").fillna(0.0)
    if "row_uid" in frame.columns:
        frame["row_uid"] = frame["row_uid"].astype(str)
    frame = augment_static_feature_frame(frame, dataset_root=train_root)
    frame.attrs["dataset_root"] = str(train_root)
    return frame


def _fit_bundle(route: str, train_root: Path, hw_config: Path, policy: str) -> dict[str, Any]:
    if route == WHITEBOX_PRIOR_ONLY_ROUTE:
        return fit_whitebox_family_prior_bundle(pd.DataFrame(), hw_config, policy=policy)
    train_frame = _load_train_frame(train_root)
    if route == WHITEBOX_PHYSICS_GUARDED_RF_ROUTE:
        return fit_whitebox_family_physics_guarded_rf_bundle(train_frame, hw_config, policy=policy)
    return fit_whitebox_family_simple_rf_bundle(train_frame, hw_config, policy=policy)


def _load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model bundle not found: {path}")
    with ExitStack() as stack:
        if path.name.endswith(".xz.parts.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("format") != "xz_parts_v1" or not manifest.get("parts"):
                raise ValueError(f"invalid split XZ model manifest: {path}")
            compressed = stack.enter_context(tempfile.TemporaryFile())
            for part in manifest["parts"]:
                filename = part["filename"]
                if Path(filename).name != filename:
                    raise ValueError(f"model part must be a sibling file: {filename}")
                with path.with_name(filename).open("rb") as source:
                    shutil.copyfileobj(source, compressed)
            compressed.seek(0)
            handle = stack.enter_context(lzma.open(compressed, "rb"))
        elif path.suffix == ".xz":
            handle = stack.enter_context(lzma.open(path, "rb"))
        else:
            handle = stack.enter_context(path.open("rb"))
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"model bundle must contain a dict, got {type(bundle).__name__}")
    return bundle


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


def _predict_with_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    route = str(bundle.get("route", "")).strip()
    if route == WHITEBOX_PHYSICS_GUARDED_RF_ROUTE:
        return predict_whitebox_family_physics_guarded_rf(bundle, frame)
    if route == WHITEBOX_PRIOR_SIMPLE_RF_ROUTE:
        return predict_whitebox_family_simple_rf(bundle, frame)
    if route == WHITEBOX_PRIOR_ONLY_ROUTE or "prior_bundle" not in bundle:
        return predict_whitebox_family_prior(bundle, frame)
    raise ValueError(f"Unsupported model bundle route: {route!r}")


def _schedule_static_e2e(
    *,
    config: DlrmStaticConfig,
    op_shapes: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    graph = build_op_graph(op_shapes)
    combo_spec = ComboSpec(
        case_id=str(config.case_id),
        combo=str(config.combo),
        batch_size=int(config.batch_size),
        num_indices_per_lookup=int(config.num_indices_per_lookup),
        inter_threads=int(config.inter_threads),
        artifact_paths=ComboArtifactPaths(
            case_id=str(config.case_id),
            combo=str(config.combo),
            shape_csv=Path("static_op_shapes_full_graph_76.csv"),
            profile_dir=Path("."),
            timeline_csv=Path(""),
        ),
    )
    combo_pred = predictions[["op_idx", "pred_us"]].copy()
    combo_pred["op_idx"] = combo_pred["op_idx"].astype(int)
    combo_pred["pred_us"] = pd.to_numeric(combo_pred["pred_us"], errors="coerce").fillna(0.0)
    schedule_result = schedule_combo(combo_spec, graph, combo_pred)

    rows: list[dict[str, Any]] = []
    for span in schedule_result.task_spans:
        rows.append(
            {
                "span_kind": "task",
                "span_id": span.task_id,
                "task_kind": span.task_kind,
                "partition": span.partition,
                "node_idx": "",
                "node_name": "",
                "branch_idx": span.branch_idx if span.branch_idx is not None else "",
                "start_us": float(span.start_us),
                "end_us": float(span.end_us),
                "duration_us": float(span.duration_us),
                "node_indices": "|".join(str(value) for value in span.node_indices),
            }
        )
    for span in schedule_result.node_spans:
        rows.append(
            {
                "span_kind": "node",
                "span_id": str(span.node_idx),
                "task_kind": "op",
                "partition": graph[int(span.node_idx)].partition if int(span.node_idx) in graph else "",
                "node_idx": int(span.node_idx),
                "node_name": span.node_name,
                "branch_idx": graph[int(span.node_idx)].branch_idx if int(span.node_idx) in graph and graph[int(span.node_idx)].branch_idx is not None else "",
                "start_us": float(span.start_us),
                "end_us": float(span.end_us),
                "duration_us": float(span.duration_us),
                "node_indices": str(span.node_idx),
            }
        )
    schedule_frame = pd.DataFrame(rows)
    summary = {
        "e2e_graph_scope": "model_input_51_observable_profiled_nodes",
        "predicted_e2e_us": float(schedule_result.predicted_observed_subgraph_us),
        "predicted_full_graph_76_us": float(schedule_result.predicted_full_graph_us),
        "coverage_ratio": 1.0,
        "full_graph_coverage_ratio": float(schedule_result.coverage.coverage_ratio),
        "observed_count": int(schedule_result.coverage.observed_count),
        "expected_full_graph_count": int(schedule_result.coverage.expected_count),
        "missing_full_graph_count": int(schedule_result.coverage.missing_count),
        "bottom_end_us": float(schedule_result.bottom_end_us),
        "branch_pool_end_us": float(schedule_result.branch_pool_end_us),
        "tail_barrier_us": float(schedule_result.tail_barrier_us),
    }
    return summary, schedule_frame


def _config_payload(config: DlrmStaticConfig) -> dict[str, Any]:
    return {
        "batch_size": int(config.batch_size),
        "num_indices_per_lookup": int(config.num_indices_per_lookup),
        "arch_embedding_size": str(config.arch_embedding_size),
        "arch_mlp_bot": str(config.arch_mlp_bot),
        "arch_mlp_top": str(config.arch_mlp_top),
        "inter_threads": int(config.inter_threads),
        "num_threads": int(config.num_threads),
    }


def predict_from_config(
    *,
    config: DlrmStaticConfig,
    model_bundle_path: Path | None = None,
    train_root: Path | None = None,
    hw_config: Path | None = None,
    route: str = WHITEBOX_PHYSICS_GUARDED_RF_ROUTE,
    policy: str = "static_config",
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    op_shapes = build_static_op_shape_table(config)
    full_graph = build_static_feature_frame(config, graph_scope=FULL_GRAPH_76)
    model_input = build_static_feature_frame(config, graph_scope=MODEL_INPUT_51)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        if model_bundle_path is not None:
            bundle = _load_model_bundle(model_bundle_path)
        else:
            if train_root is None or hw_config is None:
                raise ValueError("train_root and hw_config are required when model_bundle_path is not provided")
            bundle = _fit_bundle(str(route), train_root, hw_config, str(policy))
        prepared, pred = _predict_with_bundle(bundle, model_input)

    predictions = prepared.copy()
    predictions["pred_us"] = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    predictions["pred_total_us"] = predictions["pred_us"]
    e2e_summary, schedule_frame = _schedule_static_e2e(
        config=config,
        op_shapes=op_shapes,
        predictions=predictions,
    )
    bundle_route = str(bundle.get("route", route))
    summary = {
        "route": bundle_route,
        "policy": str(policy),
        "graph_scope": MODEL_INPUT_51,
        "model_input_rows": int(len(model_input)),
        "full_graph_rows": int(len(full_graph)),
        "operator_pred_us_sum": float(predictions["pred_us"].sum()),
        "operator_pred_us_max": float(predictions["pred_us"].max()) if len(predictions) else 0.0,
        "model_sha256": str(bundle.get("model_sha256", "")),
        "fit_data_dependent": bool(bundle.get("fit_data_dependent", False)),
        "model_bundle_path": str(model_bundle_path) if model_bundle_path is not None else "",
        "op_thread_scale_enabled": _bundle_op_thread_scale_enabled(bundle),
        "config": _config_payload(config),
        **e2e_summary,
    }
    config_summary = {
        "config": _config_payload(config),
        "route": bundle_route,
        "model_sha256": str(bundle.get("model_sha256", "")),
        "model_bundle_path": str(model_bundle_path) if model_bundle_path is not None else "",
        "static_inference_contract_version": int(bundle.get("static_inference_contract_version", 0) or 0),
        "export_metadata": bundle.get("export_metadata", {}),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        op_shapes.to_csv(out_dir / "static_op_shapes_full_graph_76.csv", index=False)
        full_graph.to_csv(out_dir / "static_full_graph_76.csv", index=False)
        model_input.to_csv(out_dir / "static_model_input_51.csv", index=False)
        predictions.to_csv(out_dir / "static_operator_predictions.csv", index=False)
        predictions.to_csv(out_dir / "static_predictions.csv", index=False)
        schedule_frame.to_csv(out_dir / "static_e2e_schedule_spans.csv", index=False)
        (out_dir / "static_config.json").write_text(
            json.dumps(config_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / "static_e2e_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / "static_prediction_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return predictions, summary


def main() -> None:
    args = _parse_args()
    config = DlrmStaticConfig(
        batch_size=int(args.batch_size),
        num_indices_per_lookup=int(args.num_indices_per_lookup),
        arch_embedding_size=str(args.arch_embedding_size),
        arch_mlp_bot=str(args.arch_mlp_bot),
        arch_mlp_top=str(args.arch_mlp_top),
        inter_threads=int(args.inter_threads),
        num_threads=int(args.num_threads),
    )
    model_bundle_path = Path(args.model_bundle).resolve() if str(args.model_bundle).strip() else None
    train_root = Path(args.train_root).resolve() if str(args.train_root).strip() else None
    hw_config = Path(args.hw_config).resolve() if str(args.hw_config).strip() else None
    _, summary = predict_from_config(
        config=config,
        model_bundle_path=model_bundle_path,
        train_root=train_root,
        hw_config=hw_config,
        route=str(args.route),
        policy=str(args.policy),
        out_dir=Path(args.out_dir).resolve(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
