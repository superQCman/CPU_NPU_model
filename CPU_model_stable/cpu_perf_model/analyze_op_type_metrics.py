from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PREDICTION_REQUIRED_COLUMNS = ["row_uid", "target_us", "pred_us"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-op-type metrics on the validation and test prediction tables.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model output directory containing predictions_val.csv, predictions_test.csv, and metrics.json.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional dataset directory containing val.csv and test.csv. Defaults to metrics.json:data_dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <model-dir>/op_type_metrics.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Prediction splits to analyze.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many op types to keep in the worst-case bar chart.",
    )
    parser.add_argument(
        "--min-count-for-ranking",
        type=int,
        default=20,
        help="Minimum samples per op type before it is eligible for worst-case ranking and plots.",
    )
    parser.add_argument(
        "--ranking-metric",
        default="mape",
        choices=["mae_us", "rmse_us", "mape", "median_ape", "p90_ape", "r2"],
        help="Metric used to rank the worst op types in the summary and plot.",
    )
    return parser.parse_args()


def import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def load_metrics_payload(model_dir: Path) -> dict[str, Any]:
    metrics_json = model_dir / "metrics.json"
    if not metrics_json.exists():
        raise FileNotFoundError(metrics_json)
    with metrics_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_dir(model_dir: Path, explicit_data_dir: str) -> Path:
    if explicit_data_dir:
        return Path(explicit_data_dir)

    payload = load_metrics_payload(model_dir)
    data_dir = payload.get("data_dir", "")
    if not data_dir:
        raise RuntimeError(
            "Could not determine data_dir automatically. Pass --data-dir explicitly "
            "or ensure metrics.json contains the training data_dir."
        )
    return Path(str(data_dir))


def load_prediction_table(model_dir: Path, split_name: str) -> pd.DataFrame:
    path = model_dir / f"predictions_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in PREDICTION_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {missing}")
    return frame


def enrich_with_metadata(frame: pd.DataFrame, data_dir: Path, split_name: str) -> pd.DataFrame:
    enriched = frame.copy()
    missing_columns = [column for column in ["op_type", "combo"] if column not in enriched.columns]
    if missing_columns:
        data_csv = data_dir / f"{split_name}.csv"
        if not data_csv.exists():
            raise FileNotFoundError(data_csv)
        lookup = pd.read_csv(data_csv, usecols=["row_uid", "op_type", "combo"])
        merge_columns = ["row_uid"] + missing_columns
        enriched = enriched.merge(
            lookup[merge_columns],
            on="row_uid",
            how="left",
            validate="one_to_one",
        )
    if "op_type" not in enriched.columns:
        raise RuntimeError(f"Failed to resolve op_type for split={split_name}")
    missing_count = int(enriched["op_type"].isna().sum())
    if missing_count > 0:
        raise RuntimeError(
            f"Resolved op_type for split={split_name}, but {missing_count} prediction rows still have missing op_type"
        )
    enriched["op_type"] = enriched["op_type"].fillna("__missing__").astype(str)
    if "combo" not in enriched.columns:
        raise RuntimeError(f"Failed to resolve combo for split={split_name}")
    combo_missing_count = int(enriched["combo"].isna().sum())
    if combo_missing_count > 0:
        raise RuntimeError(
            f"Resolved combo for split={split_name}, but {combo_missing_count} prediction rows still have missing combo"
        )
    enriched["combo"] = enriched["combo"].fillna("__missing__").astype(str)
    return enriched


def safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(r2_score(y_true, y_pred))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "mae_us": 0.0,
            "rmse_us": 0.0,
            "r2": float("nan"),
            "mape": 0.0,
            "median_ape": 0.0,
            "p90_ape": 0.0,
        }

    clipped_pred = np.clip(np.asarray(y_pred, dtype=float), a_min=0.0, a_max=None)
    target = np.asarray(y_true, dtype=float)
    denominator = np.clip(target, a_min=1e-9, a_max=None)
    ape = np.abs(clipped_pred - target) / denominator
    return {
        "mae_us": float(mean_absolute_error(target, clipped_pred)),
        "rmse_us": float(math.sqrt(mean_squared_error(target, clipped_pred))),
        "r2": safe_r2_score(target, clipped_pred),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.percentile(ape, 90)),
    }


def summarize_frame(frame: pd.DataFrame) -> dict[str, float]:
    y_true = pd.to_numeric(frame["target_us"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(frame["pred_us"], errors="coerce").to_numpy(dtype=float)
    metric_values = compute_metrics(y_true, y_pred)
    metric_values.update(
        {
            "row_count": int(len(frame)),
            "target_mean_us": float(np.mean(y_true)) if len(frame) else 0.0,
            "target_median_us": float(np.median(y_true)) if len(frame) else 0.0,
            "pred_mean_us": float(np.mean(np.clip(y_pred, a_min=0.0, a_max=None))) if len(frame) else 0.0,
            "pred_median_us": float(np.median(np.clip(y_pred, a_min=0.0, a_max=None))) if len(frame) else 0.0,
            "bias_mean_us": float(np.mean(np.clip(y_pred, a_min=0.0, a_max=None) - y_true)) if len(frame) else 0.0,
        }
    )
    return metric_values


def combo_op_type_total_duration_weighted_mape(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    work = frame.copy()
    work["target_us"] = pd.to_numeric(work["target_us"], errors="coerce").fillna(0.0)
    work["pred_us"] = pd.to_numeric(work["pred_us"], errors="coerce").fillna(0.0).clip(lower=0.0)
    denominator = work["target_us"].clip(lower=1e-9)
    work["ape"] = (work["pred_us"] - work["target_us"]).abs() / denominator
    grouped = (
        work.groupby(["combo", "op_type"], as_index=False)
        .agg(
            group_total_target_us=("target_us", "sum"),
            group_mape=("ape", "mean"),
        )
    )
    if grouped.empty:
        return 0.0
    weights = grouped["group_total_target_us"].to_numpy(dtype=float)
    values = grouped["group_mape"].to_numpy(dtype=float)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return 0.0
    return float(np.average(values, weights=weights))


def per_op_type_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for op_type, group in frame.groupby("op_type", sort=True, dropna=False):
        summary = summarize_frame(group)
        rows.append(
            {
                "op_type": str(op_type),
                **summary,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["row_count", "mape", "op_type"], ascending=[False, False, True]).reset_index(drop=True)


def _ranking_series(frame: pd.DataFrame, ranking_metric: str) -> pd.Series:
    if ranking_metric == "r2":
        return -pd.to_numeric(frame[ranking_metric], errors="coerce").fillna(-np.inf)
    return pd.to_numeric(frame[ranking_metric], errors="coerce").fillna(-np.inf)


def rank_table(frame: pd.DataFrame, ranking_metric: str, min_count_for_ranking: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    eligible = frame[frame["row_count"] >= int(min_count_for_ranking)].copy()
    if eligible.empty:
        eligible = frame.copy()
    eligible["_ranking_key"] = _ranking_series(eligible, ranking_metric)
    ranked = eligible.sort_values(
        ["_ranking_key", "row_count", "op_type"],
        ascending=[False, False, True],
    ).drop(columns=["_ranking_key"])
    return ranked.reset_index(drop=True)


def plot_top_ranked_bar(
    ranked_df: pd.DataFrame,
    output_png: Path,
    split_name: str,
    ranking_metric: str,
    top_n: int,
) -> str | None:
    plt = import_matplotlib_pyplot()
    if plt is None or ranked_df.empty:
        return None

    top_df = ranked_df.head(max(1, int(top_n))).copy()
    if top_df.empty:
        return None

    values = pd.to_numeric(top_df[ranking_metric], errors="coerce").to_numpy(dtype=float)
    labels = top_df["op_type"].astype(str).tolist()
    row_counts = top_df["row_count"].astype(int).tolist()
    if ranking_metric == "r2":
        values_to_plot = values
        axis_label = "R2"
        title = f"{split_name.upper()} Worst Op Types By R2"
        subtitle_values = values
    else:
        values_to_plot = values * 100.0 if "ape" in ranking_metric or ranking_metric == "mape" else values
        axis_label = f"{ranking_metric} (%)" if ("ape" in ranking_metric or ranking_metric == "mape") else f"{ranking_metric} (us)"
        title = f"{split_name.upper()} Worst Op Types By {ranking_metric}"
        subtitle_values = values_to_plot

    figure_height = min(max(4.5, len(top_df) * 0.5), 10.0)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    y_positions = np.arange(len(top_df))
    axis.barh(y_positions, values_to_plot, color="#d95f02", alpha=0.88)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel(axis_label)
    axis.set_title(title)
    axis.grid(True, axis="x", linestyle="--", alpha=0.35)

    for index, (value, count) in enumerate(zip(subtitle_values, row_counts)):
        text = f"{value:.2f} | n={count}"
        axis.text(value, index, f"  {text}", va="center", fontsize=8)

    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    plt.close(figure)
    return str(output_png)


def make_summary_payload(
    split_name: str,
    merged_frame: pd.DataFrame,
    detail_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    ranking_metric: str,
    min_count_for_ranking: int,
    output_csv: Path,
    ranked_csv: Path,
    plot_png: str | None,
) -> dict[str, Any]:
    overall = summarize_frame(merged_frame)
    overall["combo_op_type_total_duration_weighted_mape"] = combo_op_type_total_duration_weighted_mape(merged_frame)
    summary: dict[str, Any] = {
        "split": split_name,
        "row_count": int(len(merged_frame)),
        "op_type_count": int(detail_df["op_type"].nunique()) if not detail_df.empty else 0,
        "ranking_metric": ranking_metric,
        "min_count_for_ranking": int(min_count_for_ranking),
        "overall_metrics": overall,
        "detail_csv": str(output_csv),
        "ranked_csv": str(ranked_csv),
        "plot_png": plot_png,
    }
    if not ranked_df.empty:
        summary["worst_op_types"] = ranked_df.head(10).to_dict(orient="records")
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    data_dir = resolve_data_dir(model_dir, args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "op_type_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload: dict[str, Any] = {
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "splits": {},
    }

    for split_name in args.splits:
        predictions = load_prediction_table(model_dir, split_name)
        merged = enrich_with_metadata(predictions, data_dir, split_name)
        detail_df = per_op_type_metrics(merged)
        ranked_df = rank_table(detail_df, args.ranking_metric, args.min_count_for_ranking)

        detail_csv = output_dir / f"op_type_metrics_{split_name}.csv"
        ranked_csv = output_dir / f"op_type_metrics_{split_name}.ranked.csv"
        detail_df.to_csv(detail_csv, index=False)
        ranked_df.to_csv(ranked_csv, index=False)

        plot_png = plot_top_ranked_bar(
            ranked_df=ranked_df,
            output_png=output_dir / f"op_type_metrics_{split_name}.top{int(args.top_n)}_{args.ranking_metric}.png",
            split_name=split_name,
            ranking_metric=args.ranking_metric,
            top_n=args.top_n,
        )

        summary_payload["splits"][split_name] = make_summary_payload(
            split_name=split_name,
            merged_frame=merged,
            detail_df=detail_df,
            ranked_df=ranked_df,
            ranking_metric=args.ranking_metric,
            min_count_for_ranking=args.min_count_for_ranking,
            output_csv=detail_csv,
            ranked_csv=ranked_csv,
            plot_png=plot_png,
        )

        print(f"split={split_name}")
        print(f"detail_csv={detail_csv}")
        print(f"ranked_csv={ranked_csv}")
        if plot_png:
            print(f"plot_png={plot_png}")
        if not ranked_df.empty:
            preview = ranked_df[["op_type", "row_count", args.ranking_metric]].head(5)
            print(preview.to_string(index=False))

    summary_json = output_dir / "op_type_metrics_summary.json"
    write_json(summary_json, summary_payload)
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
