from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_loader import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_ORT_ROOT,
    build_combo_specs,
    load_op_shapes_frame,
    load_prediction_frame,
    load_timeline_frame,
)
from .schedule_engine import (
    analyze_embedding_execution,
    build_op_graph,
    compute_mean_batch_span,
    schedule_combo,
)
from .paths import DEFAULT_PIPELINE_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU static pipeline evaluation from grouped single-op predictions.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Path to the single-op artifact root containing combined_predictions_{split}.csv.",
    )
    parser.add_argument(
        "--ort-root",
        type=Path,
        default=DEFAULT_ORT_ROOT,
        help="Path to the ORT repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PIPELINE_ROOT,
        help="Directory where pipeline metrics will be written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Prediction split used for pipeline evaluation.",
    )
    return parser.parse_args()


def _metric_summary(metric_rows: list[dict[str, object]], predicted_key: str, actual_key: str) -> dict[str, object]:
    if not metric_rows:
        return {
            "count": 0,
            "mae_us": None,
            "mape": None,
            "p50_ape": None,
            "p95_ape": None,
            "worst_combo": None,
        }

    frame = pd.DataFrame(metric_rows)
    worst_row = frame.sort_values("abs_error_us", ascending=False).iloc[0]
    return {
        "count": int(len(frame)),
        "mae_us": float(frame["abs_error_us"].mean()),
        "mape": float(frame["ape"].mean()),
        "p50_ape": float(frame["ape"].quantile(0.50)),
        "p95_ape": float(frame["ape"].quantile(0.95)),
        "worst_combo": {
            "case_id": worst_row["case_id"],
            "combo": worst_row["combo"],
            predicted_key: float(worst_row[predicted_key]),
            actual_key: float(worst_row[actual_key]),
            "abs_error_us": float(worst_row["abs_error_us"]),
            "ape": float(worst_row["ape"]),
        },
    }


def _build_sorted_frame(rows: list[dict[str, object]], columns: list[str], sort_keys: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(sort_keys).reset_index(drop=True)


def _format_order(order: tuple[int, ...]) -> str:
    return "->".join(str(item) for item in order)


def run_pipeline_eval(
    *,
    artifact_root: Path,
    ort_root: Path,
    output_dir: Path,
    split: str = "test",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_df = load_prediction_frame(artifact_root, split=split)
    combo_specs = build_combo_specs(prediction_df, ort_root=ort_root)

    full_columns = [
        "metric_scope",
        "case_id",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "expected_count",
        "observed_count",
        "coverage_ratio",
        "predicted_e2e_us",
        "actual_e2e_us",
        "abs_error_us",
        "ape",
        "bottom_end_us",
        "branch_pool_end_us",
        "tail_barrier_us",
    ]
    partial_columns = [
        "metric_scope",
        "case_id",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "expected_count",
        "observed_count",
        "missing_count",
        "coverage_ratio",
        "predicted_observed_subgraph_us",
        "actual_observed_subgraph_us",
        "abs_error_us",
        "ape",
        "missing_node_indices",
        "missing_node_names",
    ]
    embedding_columns = [
        "case_id",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "kept_batch_count",
        "launch_order",
        "matches_fifo",
        "max_gather_concurrency",
        "matches_inter_threads",
        "handoff_gap_mean_us",
        "handoff_gap_min_us",
        "handoff_gap_max_us",
        "tail_start_gap_mean_us",
        "tail_start_gap_min_us",
        "tail_start_gap_max_us",
    ]

    full_rows: list[dict[str, object]] = []
    partial_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []

    for combo_spec in combo_specs:
        combo_rows = prediction_df[
            (prediction_df["case_id"] == combo_spec.case_id)
            & (prediction_df["combo"] == combo_spec.combo)
        ].copy()
        op_shapes_df = load_op_shapes_frame(combo_spec.artifact_paths.shape_csv)
        graph = build_op_graph(op_shapes_df)
        schedule_result = schedule_combo(combo_spec, graph, combo_rows)

        timeline_df = load_timeline_frame(combo_spec.artifact_paths.timeline_csv)
        expected_node_names = [graph[node_idx].node_name for node_idx in schedule_result.expected_node_indices]
        observed_node_names = [graph[node_idx].node_name for node_idx in schedule_result.observed_node_indices]

        actual_full = compute_mean_batch_span(timeline_df, expected_node_names)
        actual_observed = compute_mean_batch_span(timeline_df, observed_node_names)
        embedding_observation = analyze_embedding_execution(timeline_df, combo_spec.inter_threads)
        embedding_rows.append(
            {
                "case_id": combo_spec.case_id,
                "combo": combo_spec.combo,
                "batch_size": combo_spec.batch_size,
                "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                "inter_threads": combo_spec.inter_threads,
                "kept_batch_count": embedding_observation.kept_batch_count,
                "launch_order": _format_order(embedding_observation.representative_launch_order),
                "matches_fifo": embedding_observation.all_kept_batches_match_fifo,
                "max_gather_concurrency": embedding_observation.representative_max_gather_concurrency,
                "matches_inter_threads": embedding_observation.all_kept_batches_match_inter_threads,
                "handoff_gap_mean_us": embedding_observation.handoff_gap_mean_us,
                "handoff_gap_min_us": embedding_observation.handoff_gap_min_us,
                "handoff_gap_max_us": embedding_observation.handoff_gap_max_us,
                "tail_start_gap_mean_us": embedding_observation.tail_start_gap_mean_us,
                "tail_start_gap_min_us": embedding_observation.tail_start_gap_min_us,
                "tail_start_gap_max_us": embedding_observation.tail_start_gap_max_us,
            }
        )

        if schedule_result.coverage.is_full_graph:
            predicted_us = schedule_result.predicted_full_graph_us
            actual_us = actual_full.mean_span_us
            full_rows.append(
                {
                    "metric_scope": "full_graph_e2e",
                    "case_id": combo_spec.case_id,
                    "combo": combo_spec.combo,
                    "batch_size": combo_spec.batch_size,
                    "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                    "inter_threads": combo_spec.inter_threads,
                    "expected_count": schedule_result.coverage.expected_count,
                    "observed_count": schedule_result.coverage.observed_count,
                    "coverage_ratio": schedule_result.coverage.coverage_ratio,
                    "predicted_e2e_us": predicted_us,
                    "actual_e2e_us": actual_us,
                    "abs_error_us": abs(predicted_us - actual_us),
                    "ape": abs(predicted_us - actual_us) / actual_us if actual_us else 0.0,
                    "bottom_end_us": schedule_result.bottom_end_us,
                    "branch_pool_end_us": schedule_result.branch_pool_end_us,
                    "tail_barrier_us": schedule_result.tail_barrier_us,
                }
            )
        else:
            predicted_us = schedule_result.predicted_observed_subgraph_us
            actual_us = actual_observed.mean_span_us
            missing_names = [graph[node_idx].node_name for node_idx in schedule_result.coverage.missing_node_indices]
            partial_rows.append(
                {
                    "metric_scope": "observed_subgraph_non_e2e",
                    "case_id": combo_spec.case_id,
                    "combo": combo_spec.combo,
                    "batch_size": combo_spec.batch_size,
                    "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                    "inter_threads": combo_spec.inter_threads,
                    "expected_count": schedule_result.coverage.expected_count,
                    "observed_count": schedule_result.coverage.observed_count,
                    "missing_count": schedule_result.coverage.missing_count,
                    "coverage_ratio": schedule_result.coverage.coverage_ratio,
                    "predicted_observed_subgraph_us": predicted_us,
                    "actual_observed_subgraph_us": actual_us,
                    "abs_error_us": abs(predicted_us - actual_us),
                    "ape": abs(predicted_us - actual_us) / actual_us if actual_us else 0.0,
                    "missing_node_indices": "|".join(str(node_idx) for node_idx in schedule_result.coverage.missing_node_indices),
                    "missing_node_names": "|".join(missing_names),
                }
            )

    full_df = _build_sorted_frame(full_rows, full_columns, ["case_id", "batch_size", "num_indices_per_lookup"])
    partial_df = _build_sorted_frame(partial_rows, partial_columns, ["case_id", "batch_size", "num_indices_per_lookup"])
    embedding_df = _build_sorted_frame(embedding_rows, embedding_columns, ["case_id", "batch_size", "num_indices_per_lookup"])

    full_csv = output_dir / "full_combo_metrics.csv"
    partial_csv = output_dir / "partial_combo_metrics.csv"
    embedding_csv = output_dir / "embedding_order_check.csv"
    full_df.to_csv(full_csv, index=False)
    partial_df.to_csv(partial_csv, index=False)
    embedding_df.to_csv(embedding_csv, index=False)

    summary = {
        "artifact_root": str(artifact_root),
        "ort_root": str(ort_root),
        "output_dir": str(output_dir),
        "split": split,
        "combo_counts": {
            "total_combo_count": int(len(combo_specs)),
            "full_combo_count": int(len(full_rows)),
            "partial_combo_count": int(len(partial_rows)),
        },
        "full_graph_metrics": _metric_summary(full_rows, "predicted_e2e_us", "actual_e2e_us"),
        "partial_graph_metrics": _metric_summary(
            partial_rows,
            "predicted_observed_subgraph_us",
            "actual_observed_subgraph_us",
        ),
        "embedding_semantics": {
            "all_combos_match_fifo": bool(embedding_df["matches_fifo"].all()) if not embedding_df.empty else None,
            "all_combos_match_inter_threads": bool(embedding_df["matches_inter_threads"].all()) if not embedding_df.empty else None,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "artifact_root": str(artifact_root),
        "ort_root": str(ort_root),
        "output_dir": str(output_dir),
        "split": split,
        "artifacts": {
            "full_combo_metrics_csv": str(full_csv),
            "partial_combo_metrics_csv": str(partial_csv),
            "embedding_order_check_csv": str(embedding_csv),
            "summary_json": str(summary_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_pipeline_eval(
        artifact_root=args.artifact_root.resolve(),
        ort_root=args.ort_root.resolve(),
        output_dir=args.output_dir.resolve(),
        split=args.split,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
