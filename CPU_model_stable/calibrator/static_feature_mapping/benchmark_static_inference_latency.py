from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrator.static_feature_mapping.core import (  # noqa: E402
    FULL_GRAPH_76,
    MODEL_INPUT_51,
    DlrmStaticConfig,
    build_static_feature_frame,
    build_static_op_shape_table,
)
from calibrator.static_feature_mapping.predict_from_config import (  # noqa: E402
    _load_model_bundle,
    _predict_with_bundle,
    _schedule_static_e2e,
)


ROUTE = "whitebox_family_prior_plus_physics_guarded_rf_v3"


@dataclass(frozen=True)
class DeviceSpec:
    device_id: str
    bundle_path: Path


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    config: DlrmStaticConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark trained static DLRM config inference latency for the "
            "physics-guarded RF v3 route."
        )
    )
    parser.add_argument("--device1-bundle", default="", help="Device_1 / Kunpeng920 v3 model bundle.")
    parser.add_argument("--device2-bundle", default="", help="Device_2 / S920X20 v3 model bundle.")
    parser.add_argument("--out-dir", default="", help="Directory for benchmark CSV/JSON/Markdown outputs.")
    parser.add_argument("--threshold-seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warm-repeats", type=int, default=0)
    parser.add_argument("--cold-repeats", type=int, default=0)
    parser.add_argument("--bulk-repeats", type=int, default=10)
    parser.add_argument("--sample-counts", default="1,2,4,8,16,32,48,64")
    parser.add_argument("--workload-set", default="dlrm_s_m_l", choices=["dlrm_s_m_l"])
    parser.add_argument("--bulk-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bulk-worker-model-bundle", default="", help=argparse.SUPPRESS)
    parser.add_argument("--bulk-worker-sample-count", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--bulk-worker-output", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def _workloads() -> list[WorkloadSpec]:
    embedding = "-".join(["65536"] * 8)
    return [
        WorkloadSpec(
            "DLRM-S",
            DlrmStaticConfig(
                batch_size=1024,
                num_indices_per_lookup=16,
                arch_embedding_size=embedding,
                arch_mlp_bot="256-1024-256",
                arch_mlp_top="292-1024-1",
                inter_threads=8,
                num_threads=3,
            ),
        ),
        WorkloadSpec(
            "DLRM-M",
            DlrmStaticConfig(
                batch_size=2048,
                num_indices_per_lookup=16,
                arch_embedding_size=embedding,
                arch_mlp_bot="512-2048-512",
                arch_mlp_top="548-2048-1",
                inter_threads=8,
                num_threads=3,
            ),
        ),
        WorkloadSpec(
            "DLRM-L",
            DlrmStaticConfig(
                batch_size=4096,
                num_indices_per_lookup=16,
                arch_embedding_size=embedding,
                arch_mlp_bot="1024-4096-1024",
                arch_mlp_top="1060-4096-1",
                inter_threads=8,
                num_threads=3,
            ),
        ),
    ]


def _parse_sample_counts(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"sample counts must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("--sample-counts must contain at least one positive integer")
    return sorted(dict.fromkeys(values))


def _round_multiple(value: float, multiple: int = 8) -> int:
    return max(int(round(float(value) / float(multiple)) * int(multiple)), int(multiple))


def _bulk_sample_configs(sample_count: int) -> list[WorkloadSpec]:
    templates = _workloads()
    batch_factors = (1.0, 1.125, 0.875, 1.25)
    nip_offsets = (0, 4, -4, 8)
    mlp_factors = (1.0, 1.125, 0.875, 1.25)
    thread_pairs = ((8, 3), (8, 4), (6, 3), (4, 5))
    rows: list[WorkloadSpec] = []
    for idx in range(int(sample_count)):
        template = templates[idx % len(templates)]
        cycle = (idx // len(templates)) % len(batch_factors)
        config = template.config
        bot = [int(part) for part in str(config.arch_mlp_bot).split("-")]
        bot_factor = mlp_factors[cycle]
        dense_dim = _round_multiple(bot[0] * bot_factor, 8)
        bot_hidden = _round_multiple(bot[1] * bot_factor, 16)
        bot_out = _round_multiple(bot[2] * bot_factor, 8)
        top_hidden = _round_multiple(bot_hidden, 16)
        embedding_base = 65536
        embedding = _round_multiple(embedding_base * batch_factors[cycle], 1024)
        inter_threads, num_threads = thread_pairs[cycle]
        sample_config = DlrmStaticConfig(
            batch_size=max(int(round(int(config.batch_size) * batch_factors[cycle])), 1),
            num_indices_per_lookup=max(int(config.num_indices_per_lookup) + int(nip_offsets[cycle]), 1),
            arch_embedding_size="-".join([str(embedding)] * 8),
            arch_mlp_bot=f"{dense_dim}-{bot_hidden}-{bot_out}",
            arch_mlp_top=f"{bot_out + 36}-{top_hidden}-1",
            inter_threads=int(inter_threads),
            num_threads=int(num_threads),
        )
        rows.append(WorkloadSpec(f"{template.workload_id}_sample_{idx:03d}", sample_config))
    return rows


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


def _validate_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model bundle not found: {path}")
    bundle = _load_model_bundle(path)
    route = str(bundle.get("route", "")).strip()
    if route != ROUTE:
        raise ValueError(f"{path} route must be {ROUTE}, got {route!r}")
    return bundle


def _predict_loaded_bundle(bundle: dict[str, Any], config: DlrmStaticConfig) -> dict[str, Any]:
    op_shapes = build_static_op_shape_table(config)
    model_input = build_static_feature_frame(config, graph_scope=MODEL_INPUT_51)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        prepared, pred = _predict_with_bundle(bundle, model_input)
    predictions = prepared.copy()
    predictions["pred_us"] = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    predictions["pred_total_us"] = predictions["pred_us"]
    summary, _ = _schedule_static_e2e(
        config=config,
        op_shapes=op_shapes,
        predictions=predictions,
    )
    return {
        **summary,
        "model_input_rows": int(len(model_input)),
        "full_graph_rows": int(len(build_static_feature_frame(config, graph_scope=FULL_GRAPH_76))),
    }


def _predict_bulk_loaded_bundle(bundle: dict[str, Any], sample_count: int) -> tuple[dict[str, Any], pd.DataFrame]:
    samples = _bulk_sample_configs(int(sample_count))
    model_frames: list[pd.DataFrame] = []
    op_shapes_by_sample: list[pd.DataFrame] = []
    full_graph_rows = 0
    slices: list[tuple[int, int]] = []
    cursor = 0
    for sample_idx, workload in enumerate(samples):
        model_input = build_static_feature_frame(workload.config, graph_scope=MODEL_INPUT_51).copy()
        if "row_uid" in model_input.columns:
            model_input["row_uid"] = model_input["row_uid"].astype(str).map(lambda value: f"sample_{sample_idx:04d}::{value}")
        model_input["bulk_sample_id"] = int(sample_idx)
        model_input["bulk_workload_id"] = str(workload.workload_id)
        model_frames.append(model_input)
        op_shapes_by_sample.append(build_static_op_shape_table(workload.config))
        full_graph_rows += int(len(build_static_feature_frame(workload.config, graph_scope=FULL_GRAPH_76)))
        next_cursor = cursor + int(len(model_input))
        slices.append((cursor, next_cursor))
        cursor = next_cursor

    combined_model_input = pd.concat(model_frames, ignore_index=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        prepared, pred = _predict_with_bundle(bundle, combined_model_input)
    prediction_rows: list[dict[str, Any]] = []
    for sample_idx, ((start, end), workload, op_shapes) in enumerate(zip(slices, samples, op_shapes_by_sample, strict=True)):
        sample_prepared = prepared.iloc[start:end].copy()
        sample_pred = np.asarray(pred[start:end], dtype=float)
        sample_predictions = sample_prepared.copy()
        sample_predictions["pred_us"] = np.clip(sample_pred, a_min=0.0, a_max=None)
        sample_predictions["pred_total_us"] = sample_predictions["pred_us"]
        summary, _ = _schedule_static_e2e(
            config=workload.config,
            op_shapes=op_shapes,
            predictions=sample_predictions,
        )
        prediction_rows.append(
            {
                "sample_id": int(sample_idx),
                "workload_id": str(workload.workload_id),
                "predicted_e2e_us": float(summary["predicted_e2e_us"]),
                "predicted_full_graph_76_us": float(summary["predicted_full_graph_76_us"]),
                "model_input_rows": int(end - start),
                "full_graph_rows": 76,
                **_config_payload(workload.config),
            }
        )
    sample_predictions_frame = pd.DataFrame(prediction_rows)
    summary = {
        "sample_count": int(sample_count),
        "static_model_input_rows": int(len(combined_model_input)),
        "static_full_graph_rows": int(full_graph_rows),
        "predicted_positive": bool(sample_predictions_frame["predicted_e2e_us"].gt(0.0).all()),
        "prediction_count": int(len(sample_predictions_frame)),
        "predicted_e2e_us_sum": float(sample_predictions_frame["predicted_e2e_us"].sum()),
        "predicted_e2e_us_mean": float(sample_predictions_frame["predicted_e2e_us"].mean()),
        "predicted_e2e_us_max": float(sample_predictions_frame["predicted_e2e_us"].max()),
    }
    return summary, sample_predictions_frame


def _run_warm_benchmark(
    *,
    devices: list[DeviceSpec],
    workloads: list[WorkloadSpec],
    warmup: int,
    repeats: int,
    threshold_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    bundle_infos: list[dict[str, Any]] = []
    for device in devices:
        load_start = time.perf_counter()
        bundle = _validate_bundle(device.bundle_path)
        load_s = time.perf_counter() - load_start
        bundle_infos.append(
            {
                "device_id": device.device_id,
                "bundle_path": str(device.bundle_path),
                "bundle_size_bytes": int(device.bundle_path.stat().st_size),
                "bundle_size_mb": float(device.bundle_path.stat().st_size / 1024 / 1024),
                "load_s": float(load_s),
                "route": str(bundle.get("route", "")),
                "model_sha256": str(bundle.get("model_sha256", "")),
            }
        )
        if int(repeats) <= 0:
            continue
        for workload in workloads:
            for _ in range(max(int(warmup), 0)):
                _predict_loaded_bundle(bundle, workload.config)
            for repeat_idx in range(max(int(repeats), 0)):
                start = time.perf_counter()
                status = "ok"
                error = ""
                summary: dict[str, Any] = {}
                try:
                    summary = _predict_loaded_bundle(bundle, workload.config)
                except Exception as exc:  # pragma: no cover - retained for benchmark diagnostics.
                    status = "error"
                    error = repr(exc)
                elapsed_s = time.perf_counter() - start
                rows.append(
                    {
                        "mode": "warm_inprocess",
                        "device_id": device.device_id,
                        "workload_id": workload.workload_id,
                        "repeat_idx": int(repeat_idx),
                        "elapsed_s": float(elapsed_s),
                        "threshold_s": float(threshold_s),
                        "pass_threshold": bool(status == "ok" and elapsed_s <= threshold_s),
                        "status": status,
                        "error": error,
                        "returncode": 0 if status == "ok" else 1,
                        "predicted_e2e_us": summary.get("predicted_e2e_us", np.nan),
                        "static_model_input_rows": summary.get("model_input_rows", np.nan),
                        "static_full_graph_rows": summary.get("full_graph_rows", np.nan),
                        "predicted_positive": bool(float(summary.get("predicted_e2e_us", 0.0) or 0.0) > 0.0),
                        **_config_payload(workload.config),
                    }
                )
    return rows, bundle_infos


def _run_warm_bulk_benchmark(
    *,
    devices: list[DeviceSpec],
    sample_counts: list[int],
    warmup: int,
    repeats: int,
    threshold_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sample_prediction_rows: list[dict[str, Any]] = []
    bundle_infos: list[dict[str, Any]] = []
    for device in devices:
        load_start = time.perf_counter()
        bundle = _validate_bundle(device.bundle_path)
        load_s = time.perf_counter() - load_start
        bundle_infos.append(
            {
                "device_id": device.device_id,
                "bundle_path": str(device.bundle_path),
                "bundle_size_bytes": int(device.bundle_path.stat().st_size),
                "bundle_size_mb": float(device.bundle_path.stat().st_size / 1024 / 1024),
                "load_s": float(load_s),
                "route": str(bundle.get("route", "")),
                "model_sha256": str(bundle.get("model_sha256", "")),
            }
        )
        for sample_count in sample_counts:
            for _ in range(max(int(warmup), 0)):
                _predict_bulk_loaded_bundle(bundle, sample_count)
            for repeat_idx in range(max(int(repeats), 0)):
                start = time.perf_counter()
                status = "ok"
                error = ""
                summary: dict[str, Any] = {}
                predictions = pd.DataFrame()
                try:
                    summary, predictions = _predict_bulk_loaded_bundle(bundle, sample_count)
                except Exception as exc:  # pragma: no cover - retained for benchmark diagnostics.
                    status = "error"
                    error = repr(exc)
                elapsed_s = time.perf_counter() - start
                rows.append(
                    {
                        "mode": "warm_bulk_inprocess",
                        "device_id": device.device_id,
                        "workload_id": "heterogeneous_static_configs",
                        "sample_count": int(sample_count),
                        "repeat_idx": int(repeat_idx),
                        "elapsed_s": float(elapsed_s),
                        "per_sample_s": float(elapsed_s / max(int(sample_count), 1)),
                        "throughput_samples_per_s": float(int(sample_count) / max(elapsed_s, 1e-12)),
                        "threshold_s": float(threshold_s),
                        "pass_threshold": bool(status == "ok" and elapsed_s <= threshold_s),
                        "status": status,
                        "error": error,
                        "returncode": 0 if status == "ok" else 1,
                        "predicted_e2e_us": summary.get("predicted_e2e_us_mean", np.nan),
                        "static_model_input_rows": summary.get("static_model_input_rows", np.nan),
                        "static_full_graph_rows": summary.get("static_full_graph_rows", np.nan),
                        "prediction_count": summary.get("prediction_count", np.nan),
                        "predicted_positive": bool(summary.get("predicted_positive", False)),
                    }
                )
                if status == "ok" and repeat_idx == max(int(repeats), 1) - 1:
                    for record in predictions.to_dict(orient="records"):
                        sample_prediction_rows.append(
                            {
                                "mode": "warm_bulk_inprocess",
                                "device_id": device.device_id,
                                "sample_count": int(sample_count),
                                "repeat_idx": int(repeat_idx),
                                **record,
                            }
                        )
    return rows, sample_prediction_rows, bundle_infos


def _run_bulk_worker(model_bundle: Path, sample_count: int, output_path: Path) -> None:
    bundle = _validate_bundle(model_bundle)
    summary, predictions = _predict_bulk_loaded_bundle(bundle, int(sample_count))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "sample_predictions": predictions.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_cold_cli_benchmark(
    *,
    devices: list[DeviceSpec],
    workloads: list[WorkloadSpec],
    repeats: int,
    out_dir: Path,
    threshold_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predict_script = SCRIPT_PATH.with_name("predict_from_config.py")
    for device in devices:
        for workload in workloads:
            for repeat_idx in range(max(int(repeats), 0)):
                run_out = (
                    out_dir
                    / "cold_cli_outputs"
                    / device.device_id
                    / workload.workload_id
                    / f"repeat_{repeat_idx:03d}"
                )
                cmd = [
                    sys.executable,
                    str(predict_script),
                    "--model-bundle",
                    str(device.bundle_path),
                    "--batch-size",
                    str(int(workload.config.batch_size)),
                    "--num-indices-per-lookup",
                    str(int(workload.config.num_indices_per_lookup)),
                    "--arch-embedding-size",
                    str(workload.config.arch_embedding_size),
                    "--arch-mlp-bot",
                    str(workload.config.arch_mlp_bot),
                    "--arch-mlp-top",
                    str(workload.config.arch_mlp_top),
                    "--inter-threads",
                    str(int(workload.config.inter_threads)),
                    "--num-threads",
                    str(int(workload.config.num_threads)),
                    "--out-dir",
                    str(run_out),
                ]
                start = time.perf_counter()
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                elapsed_s = time.perf_counter() - start
                summary: dict[str, Any] = {}
                summary_path = run_out / "static_e2e_summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except Exception as exc:  # pragma: no cover - retained for benchmark diagnostics.
                        summary = {"summary_read_error": repr(exc)}
                status = "ok" if int(proc.returncode) == 0 else "error"
                rows.append(
                    {
                        "mode": "cold_cli",
                        "device_id": device.device_id,
                        "workload_id": workload.workload_id,
                        "repeat_idx": int(repeat_idx),
                        "elapsed_s": float(elapsed_s),
                        "threshold_s": float(threshold_s),
                        "pass_threshold": bool(status == "ok" and elapsed_s <= threshold_s),
                        "status": status,
                        "error": str(summary.get("summary_read_error", "")) or proc.stderr[-1000:],
                        "returncode": int(proc.returncode),
                        "stdout_bytes": int(len(proc.stdout or "")),
                        "stderr_bytes": int(len(proc.stderr or "")),
                        "output_dir": str(run_out),
                        "output_file_count": int(len(list(run_out.glob("*")))) if run_out.exists() else 0,
                        "predicted_e2e_us": summary.get("predicted_e2e_us", np.nan),
                        "static_model_input_rows": summary.get("model_input_rows", np.nan),
                        "static_full_graph_rows": summary.get("full_graph_rows", np.nan),
                        "predicted_positive": bool(float(summary.get("predicted_e2e_us", 0.0) or 0.0) > 0.0),
                        **_config_payload(workload.config),
                    }
                )
    return rows


def _run_cold_bulk_benchmark(
    *,
    devices: list[DeviceSpec],
    sample_counts: list[int],
    repeats: int,
    out_dir: Path,
    threshold_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sample_prediction_rows: list[dict[str, Any]] = []
    for device in devices:
        for sample_count in sample_counts:
            for repeat_idx in range(max(int(repeats), 0)):
                run_out = (
                    out_dir
                    / "cold_bulk_outputs"
                    / device.device_id
                    / f"samples_{int(sample_count):04d}"
                    / f"repeat_{repeat_idx:03d}"
                )
                worker_summary = run_out / "bulk_worker_summary.json"
                cmd = [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--bulk-worker",
                    "--bulk-worker-model-bundle",
                    str(device.bundle_path),
                    "--bulk-worker-sample-count",
                    str(int(sample_count)),
                    "--bulk-worker-output",
                    str(worker_summary),
                ]
                start = time.perf_counter()
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                elapsed_s = time.perf_counter() - start
                payload: dict[str, Any] = {}
                summary: dict[str, Any] = {}
                predictions: list[dict[str, Any]] = []
                if worker_summary.exists():
                    try:
                        payload = json.loads(worker_summary.read_text(encoding="utf-8"))
                        summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
                        predictions = payload.get("sample_predictions", []) if isinstance(payload.get("sample_predictions", []), list) else []
                    except Exception as exc:  # pragma: no cover - retained for benchmark diagnostics.
                        summary = {"summary_read_error": repr(exc)}
                status = "ok" if int(proc.returncode) == 0 else "error"
                rows.append(
                    {
                        "mode": "cold_bulk_cli",
                        "device_id": device.device_id,
                        "workload_id": "heterogeneous_static_configs",
                        "sample_count": int(sample_count),
                        "repeat_idx": int(repeat_idx),
                        "elapsed_s": float(elapsed_s),
                        "per_sample_s": float(elapsed_s / max(int(sample_count), 1)),
                        "throughput_samples_per_s": float(int(sample_count) / max(elapsed_s, 1e-12)),
                        "threshold_s": float(threshold_s),
                        "pass_threshold": bool(status == "ok" and elapsed_s <= threshold_s),
                        "status": status,
                        "error": str(summary.get("summary_read_error", "")) or proc.stderr[-1000:],
                        "returncode": int(proc.returncode),
                        "stdout_bytes": int(len(proc.stdout or "")),
                        "stderr_bytes": int(len(proc.stderr or "")),
                        "output_dir": str(run_out),
                        "output_file_count": int(len(list(run_out.glob("*")))) if run_out.exists() else 0,
                        "predicted_e2e_us": summary.get("predicted_e2e_us_mean", np.nan),
                        "static_model_input_rows": summary.get("static_model_input_rows", np.nan),
                        "static_full_graph_rows": summary.get("static_full_graph_rows", np.nan),
                        "prediction_count": summary.get("prediction_count", np.nan),
                        "predicted_positive": bool(summary.get("predicted_positive", False)),
                    }
                )
                if status == "ok" and repeat_idx == max(int(repeats), 1) - 1:
                    for record in predictions:
                        sample_prediction_rows.append(
                            {
                                "mode": "cold_bulk_cli",
                                "device_id": device.device_id,
                                "sample_count": int(sample_count),
                                "repeat_idx": int(repeat_idx),
                                **record,
                            }
                        )
    return rows, sample_prediction_rows


def _summarize_runs(runs: pd.DataFrame, threshold_s: float) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    runs = runs.copy()
    if "sample_count" not in runs.columns:
        runs["sample_count"] = 1
    runs["sample_count"] = pd.to_numeric(runs["sample_count"], errors="coerce").fillna(1).astype(int)
    grouped = runs.groupby(["mode", "device_id", "workload_id", "sample_count"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (mode, device_id, workload_id, sample_count), group in grouped:
        elapsed = pd.to_numeric(group["elapsed_s"], errors="coerce").dropna().to_numpy(dtype=float)
        expected_model_rows = 51 * int(sample_count)
        expected_full_rows = 76 * int(sample_count)
        observed_model_rows = pd.to_numeric(group["static_model_input_rows"], errors="coerce")
        observed_full_rows = pd.to_numeric(group["static_full_graph_rows"], errors="coerce")
        rows.append(
            {
                "mode": mode,
                "device_id": device_id,
                "workload_id": workload_id,
                "sample_count": int(sample_count),
                "repeat_count": int(len(group)),
                "mean_s": float(np.mean(elapsed)) if len(elapsed) else np.nan,
                "p50_s": float(np.percentile(elapsed, 50)) if len(elapsed) else np.nan,
                "p90_s": float(np.percentile(elapsed, 90)) if len(elapsed) else np.nan,
                "p95_s": float(np.percentile(elapsed, 95)) if len(elapsed) else np.nan,
                "max_s": float(np.max(elapsed)) if len(elapsed) else np.nan,
                "mean_per_sample_s": float(np.mean(elapsed) / max(int(sample_count), 1)) if len(elapsed) else np.nan,
                "max_per_sample_s": float(np.max(elapsed) / max(int(sample_count), 1)) if len(elapsed) else np.nan,
                "mean_throughput_samples_per_s": float(max(int(sample_count), 1) / np.mean(elapsed)) if len(elapsed) else np.nan,
                "max_throughput_samples_per_s": float(max(int(sample_count), 1) / np.min(elapsed)) if len(elapsed) else np.nan,
                "threshold_s": float(threshold_s),
                "pass_threshold": bool(
                    len(elapsed)
                    and group["status"].eq("ok").all()
                    and group["pass_threshold"].astype(bool).all()
                    and group["predicted_positive"].astype(bool).all()
                    and observed_model_rows.eq(expected_model_rows).all()
                    and observed_full_rows.eq(expected_full_rows).all()
                ),
                "predicted_e2e_us_last": float(
                    pd.to_numeric(group["predicted_e2e_us"], errors="coerce").dropna().iloc[-1]
                )
                if pd.to_numeric(group["predicted_e2e_us"], errors="coerce").dropna().size
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "device_id", "sample_count", "workload_id"]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame[columns].iterrows():
        cells: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plot_bulk_summary(summary: pd.DataFrame, out_dir: Path, threshold_s: float) -> None:
    bulk = summary[summary["mode"].astype(str).isin(["cold_bulk_cli", "warm_bulk_inprocess"])].copy()
    if bulk.empty:
        return
    bulk["sample_count"] = pd.to_numeric(bulk["sample_count"], errors="coerce")
    bulk = bulk.sort_values(["mode", "device_id", "sample_count"])
    colors = {"Device_1_kunpeng920": "#b65a2a", "Device_2_s920x20": "#2f5f9e"}
    linestyles = {"cold_bulk_cli": "--", "warm_bulk_inprocess": "-"}
    labels = {"Device_1_kunpeng920": "Device_1", "Device_2_s920x20": "Device_2"}
    metric_specs = [
        ("max_s", "Batch inference latency (s)", "static_inference_bulk_latency_vs_samples"),
        ("max_per_sample_s", "Amortized latency per sample (s)", "static_inference_bulk_per_sample_latency"),
        ("mean_throughput_samples_per_s", "Throughput (samples/s)", "static_inference_bulk_throughput"),
    ]
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    for metric, ylabel, stem in metric_specs:
        fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
        for (device_id, mode), group in bulk.groupby(["device_id", "mode"], sort=False):
            group = group.sort_values("sample_count")
            ax.plot(
                group["sample_count"],
                group[metric],
                marker="o",
                linewidth=1.5,
                color=colors.get(str(device_id), "#555555"),
                linestyle=linestyles.get(str(mode), "-"),
                label=f"{labels.get(str(device_id), str(device_id))} {mode}",
            )
        if metric == "max_s":
            ax.axhline(float(threshold_s), color="0.35", linestyle=":", linewidth=1.0, label=f"{threshold_s:.0f}s threshold")
        ax.set_xlabel("Input sample count")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(bulk["sample_count"].dropna().unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, color="0.85", linewidth=0.7)
        ax.legend(frameon=False, loc="best")
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
        fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)


def _write_report(
    *,
    out_dir: Path,
    summary: pd.DataFrame,
    manifest: dict[str, Any],
    overall_pass: bool,
) -> None:
    max_cold = summary.loc[summary["mode"].eq("cold_cli"), "max_s"].max() if not summary.empty else np.nan
    max_warm = summary.loc[summary["mode"].eq("warm_inprocess"), "max_s"].max() if not summary.empty else np.nan
    max_cold_bulk = summary.loc[summary["mode"].eq("cold_bulk_cli"), "max_s"].max() if not summary.empty else np.nan
    max_warm_bulk = summary.loc[summary["mode"].eq("warm_bulk_inprocess"), "max_s"].max() if not summary.empty else np.nan
    max_bulk_per_sample = (
        summary.loc[summary["mode"].astype(str).str.contains("bulk", na=False), "max_per_sample_s"].max()
        if not summary.empty and "max_per_sample_s" in summary.columns
        else np.nan
    )
    max_bulk_throughput = (
        summary.loc[summary["mode"].astype(str).str.contains("bulk", na=False), "max_throughput_samples_per_s"].max()
        if not summary.empty and "max_throughput_samples_per_s" in summary.columns
        else np.nan
    )
    lines = [
        "# Static Inference Bulk Latency Benchmark",
        "",
        f"Overall result: {'PASS' if overall_pass else 'FAIL'}",
        "",
        "## Scope",
        "",
        "This benchmark validates the trained static DLRM config inference path. It does not retrain any model.",
        "",
        f"- Threshold: {float(manifest['threshold_seconds']):.3f} seconds",
        f"- Sample counts: {', '.join(str(value) for value in manifest.get('sample_counts', []))}",
        f"- Bulk repeats: {int(manifest.get('bulk_repeats', 0))}",
        f"- Legacy cold CLI repeats: {int(manifest['cold_repeats'])}",
        f"- Legacy warm in-process repeats: {int(manifest['warm_repeats'])}",
        f"- Warmup repeats: {int(manifest['warmup'])}",
        "",
        "## Key Result",
        "",
        f"- Cold bulk CLI max latency: {float(max_cold_bulk):.3f} seconds" if np.isfinite(max_cold_bulk) else "- Cold bulk CLI max latency: n/a",
        f"- Warm bulk in-process max latency: {float(max_warm_bulk):.3f} seconds" if np.isfinite(max_warm_bulk) else "- Warm bulk in-process max latency: n/a",
        f"- Max amortized bulk latency: {float(max_bulk_per_sample):.3f} seconds/sample" if np.isfinite(max_bulk_per_sample) else "- Max amortized bulk latency: n/a",
        f"- Best observed bulk throughput: {float(max_bulk_throughput):.3f} samples/s" if np.isfinite(max_bulk_throughput) else "- Best observed bulk throughput: n/a",
        f"- Legacy cold CLI max latency: {float(max_cold):.3f} seconds" if np.isfinite(max_cold) else "- Legacy cold CLI max latency: n/a",
        f"- Legacy warm in-process max latency: {float(max_warm):.3f} seconds" if np.isfinite(max_warm) else "- Legacy warm in-process max latency: n/a",
        "",
        "## Summary",
        "",
        _markdown_table(
            summary,
            [
                "mode",
                "device_id",
                "sample_count",
                "workload_id",
                "repeat_count",
                "mean_s",
                "p50_s",
                "p90_s",
                "p95_s",
                "max_s",
                "max_per_sample_s",
                "mean_throughput_samples_per_s",
                "pass_threshold",
            ],
        ),
        "",
        "## Bundles",
        "",
    ]
    bundle_rows = pd.DataFrame(manifest.get("bundles", []))
    if not bundle_rows.empty:
        lines.append(
            _markdown_table(
                bundle_rows,
                ["device_id", "bundle_size_mb", "load_s", "model_sha256"],
            )
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `static_inference_latency_runs.csv`: raw repeated measurements.",
            "- `static_inference_latency_summary.csv`: grouped latency statistics.",
            "- `static_inference_latency_manifest.json`: benchmark configuration and environment.",
            "- `static_inference_bulk_latency_runs.csv`: raw bulk inference measurements.",
            "- `static_inference_bulk_latency_summary.csv`: grouped bulk latency statistics.",
            "- `static_inference_bulk_sample_predictions.csv`: per-sample predictions from the last bulk repeat.",
            "- `static_inference_bulk_latency_vs_samples.png/pdf`: batch latency versus input sample count.",
            "- `static_inference_bulk_per_sample_latency.png/pdf`: amortized latency per sample.",
            "- `static_inference_bulk_throughput.png/pdf`: throughput in samples per second.",
            "- `STATIC_INFERENCE_BULK_LATENCY_REPORT.md`: human-readable bulk benchmark report.",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    (out_dir / "STATIC_INFERENCE_BULK_LATENCY_REPORT.md").write_text(report_text, encoding="utf-8")
    # Keep the legacy report filename for existing links and scripts.
    (out_dir / "STATIC_INFERENCE_LATENCY_REPORT.md").write_text(report_text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if bool(args.bulk_worker):
        if not str(args.bulk_worker_model_bundle).strip() or int(args.bulk_worker_sample_count) <= 0 or not str(args.bulk_worker_output).strip():
            raise SystemExit("--bulk-worker requires --bulk-worker-model-bundle, --bulk-worker-sample-count and --bulk-worker-output")
        _run_bulk_worker(
            Path(args.bulk_worker_model_bundle).resolve(),
            int(args.bulk_worker_sample_count),
            Path(args.bulk_worker_output).resolve(),
        )
        return
    if not str(args.device1_bundle).strip() or not str(args.device2_bundle).strip() or not str(args.out_dir).strip():
        raise SystemExit("--device1-bundle, --device2-bundle and --out-dir are required")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_s = float(args.threshold_seconds)
    sample_counts = _parse_sample_counts(str(args.sample_counts))
    devices = [
        DeviceSpec("Device_1_kunpeng920", Path(args.device1_bundle).resolve()),
        DeviceSpec("Device_2_s920x20", Path(args.device2_bundle).resolve()),
    ]
    workloads = _workloads()

    warm_rows, bundle_infos = _run_warm_benchmark(
        devices=devices,
        workloads=workloads,
        warmup=int(args.warmup),
        repeats=int(args.warm_repeats),
        threshold_s=threshold_s,
    )
    cold_rows = _run_cold_cli_benchmark(
        devices=devices,
        workloads=workloads,
        repeats=int(args.cold_repeats),
        out_dir=out_dir,
        threshold_s=threshold_s,
    )
    warm_bulk_rows, warm_bulk_sample_rows, bulk_bundle_infos = _run_warm_bulk_benchmark(
        devices=devices,
        sample_counts=sample_counts,
        warmup=int(args.warmup),
        repeats=int(args.bulk_repeats),
        threshold_s=threshold_s,
    )
    cold_bulk_rows, cold_bulk_sample_rows = _run_cold_bulk_benchmark(
        devices=devices,
        sample_counts=sample_counts,
        repeats=int(args.bulk_repeats),
        out_dir=out_dir,
        threshold_s=threshold_s,
    )
    runs = pd.DataFrame([*warm_rows, *cold_rows])
    bulk_runs = pd.DataFrame([*warm_bulk_rows, *cold_bulk_rows])
    all_runs = pd.DataFrame([*warm_rows, *cold_rows, *warm_bulk_rows, *cold_bulk_rows])
    sample_predictions = pd.DataFrame([*warm_bulk_sample_rows, *cold_bulk_sample_rows])
    summary = _summarize_runs(runs, threshold_s)
    bulk_summary = _summarize_runs(bulk_runs, threshold_s)
    all_summary = _summarize_runs(all_runs, threshold_s)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(SCRIPT_PATH),
        "project_root": str(PROJECT_ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "threshold_seconds": threshold_s,
        "sample_counts": sample_counts,
        "warmup": int(args.warmup),
        "warm_repeats": int(args.warm_repeats),
        "cold_repeats": int(args.cold_repeats),
        "bulk_repeats": int(args.bulk_repeats),
        "workload_set": str(args.workload_set),
        "workloads": [
            {"workload_id": workload.workload_id, **_config_payload(workload.config)}
            for workload in workloads
        ],
        "bundles": bundle_infos or bulk_bundle_infos,
    }

    runs.to_csv(out_dir / "static_inference_latency_runs.csv", index=False)
    summary.to_csv(out_dir / "static_inference_latency_summary.csv", index=False)
    bulk_runs.to_csv(out_dir / "static_inference_bulk_latency_runs.csv", index=False)
    bulk_summary.to_csv(out_dir / "static_inference_bulk_latency_summary.csv", index=False)
    sample_predictions.to_csv(out_dir / "static_inference_bulk_sample_predictions.csv", index=False)
    all_runs.to_csv(out_dir / "static_inference_all_latency_runs.csv", index=False)
    all_summary.to_csv(out_dir / "static_inference_all_latency_summary.csv", index=False)
    (out_dir / "static_inference_latency_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "static_inference_bulk_latency_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot_bulk_summary(bulk_summary, out_dir, threshold_s)
    overall_pass = bool(not all_summary.empty and all_summary["pass_threshold"].astype(bool).all())
    _write_report(out_dir=out_dir, summary=all_summary, manifest=manifest, overall_pass=overall_pass)
    result = {
        "out_dir": str(out_dir),
        "overall_pass": overall_pass,
        "threshold_seconds": threshold_s,
        "max_cold_cli_s": float(all_summary.loc[all_summary["mode"].eq("cold_cli"), "max_s"].max()) if not all_summary.empty else float("nan"),
        "max_warm_inprocess_s": float(all_summary.loc[all_summary["mode"].eq("warm_inprocess"), "max_s"].max()) if not all_summary.empty else float("nan"),
        "max_cold_bulk_cli_s": float(all_summary.loc[all_summary["mode"].eq("cold_bulk_cli"), "max_s"].max()) if not all_summary.empty else float("nan"),
        "max_warm_bulk_inprocess_s": float(all_summary.loc[all_summary["mode"].eq("warm_bulk_inprocess"), "max_s"].max()) if not all_summary.empty else float("nan"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
