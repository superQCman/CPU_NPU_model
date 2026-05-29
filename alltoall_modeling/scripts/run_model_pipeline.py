#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bw_model import ALIGNMENT_HIGH, ALIGNMENT_MID, parse_hccl_output, run_two_stage_piecewise
from src.latency_model import parse_and_compare, run_latency_model
from src.memory_bw_utils import MUL_FACTOR_FUNCS


def _to_float_list(arr):
    return [float(x) for x in np.asarray(arr).tolist()]


def _display_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def group_of_size(size):
    if size % ALIGNMENT_HIGH == 0:
        return "Group1 (128KB)"
    if size % ALIGNMENT_MID == 0:
        return "Group2 (32KB)"
    return "Group3 (unaligned)"


def add_group_column(df):
    d = df.copy()
    d["group"] = d["size"].map(group_of_size)
    return d


def write_hccl_like_file(df, out_file):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        f.write("data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:\n")
        for _, row in df.sort_values("size").iterrows():
            f.write(
                f"{int(row['size']):<17} | {float(row['latency']):<14.6f} | {float(row['alg_bw']):<20.6f} | success\n"
            )


def split_size_holdout(df, test_ratio=0.2):
    d = add_group_column(df).sort_values("size").reset_index(drop=True)
    test_mask = np.zeros(len(d), dtype=bool)

    for group in ["Group1 (128KB)", "Group2 (32KB)", "Group3 (unaligned)"]:
        g_idx = d.index[d["group"] == group].to_numpy()
        if len(g_idx) == 0:
            continue

        uniq_sizes = np.array(sorted(d.loc[g_idx, "size"].unique()))
        if len(uniq_sizes) < 2:
            continue

        stride = max(2, int(round(1.0 / max(test_ratio, 1e-6))))
        candidate = uniq_sizes[::stride]
        if len(candidate) == 0:
            candidate = uniq_sizes[:1]
        if len(candidate) == len(uniq_sizes):
            candidate = uniq_sizes[::2]

        pick = set(int(x) for x in candidate)
        test_mask |= ((d["group"] == group) & d["size"].isin(pick)).to_numpy()

    test_df = d.loc[test_mask, ["size", "latency", "alg_bw"]].copy()
    train_df = d.loc[~test_mask, ["size", "latency", "alg_bw"]].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        raise RuntimeError("size_holdout split failed: empty train or test set")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def summarize_coverage(df, dataset_name):
    d = add_group_column(df)
    rows = []
    all_groups = ["Group1 (128KB)", "Group2 (32KB)", "Group3 (unaligned)"]
    for g in all_groups:
        dg = d[d["group"] == g]
        if len(dg) == 0:
            rows.append(
                {
                    "dataset": dataset_name,
                    "group": g,
                    "samples": 0,
                    "unique_sizes": 0,
                    "min_size": "",
                    "max_size": "",
                    "log2_bins": 0,
                }
            )
            continue
        bins = set(int(np.floor(np.log2(v))) for v in dg["size"].to_numpy(np.float64))
        rows.append(
            {
                "dataset": dataset_name,
                "group": g,
                "samples": int(len(dg)),
                "unique_sizes": int(dg["size"].nunique()),
                "min_size": int(dg["size"].min()),
                "max_size": int(dg["size"].max()),
                "log2_bins": int(len(bins)),
            }
        )
    return rows


def calc_overlap(train_df, test_df):
    train_sizes = set(int(x) for x in train_df["size"].unique())
    test_sizes = set(int(x) for x in test_df["size"].unique())
    return sorted(train_sizes.intersection(test_sizes))


def save_bw_params(path, bw_results, raw_data_path, num_npus, split_mode):
    payload = {
        "model_type": "a2a_bandwidth",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_raw_data": _display_path(raw_data_path),
        "num_npus": int(num_npus),
        "split_mode": split_mode,
        "params": {
            "popt_high_trend": _to_float_list(bw_results["popt_high_trend"]),
            "popt_mid_trend": _to_float_list(bw_results["popt_mid_trend"]),
            "popt_high_final": _to_float_list(bw_results["popt_high_final"]),
            "popt_mid_final": _to_float_list(bw_results["popt_mid_final"]),
            "sat_bw_high": float(bw_results["sat_bw_high"]),
            "sat_bw_mid": float(bw_results["sat_bw_mid"]),
            "sat_bw_low3": float(bw_results["sat_bw_low3"]),
            "sat_size_high": int(bw_results["sat_size_high"]),
            "sat_size_mid": int(bw_results["sat_size_mid"]),
            "split_size": int(bw_results["split_size"]),
            "dip_high": bw_results["dip_high"],
            "dip_mid": bw_results["dip_mid"],
            "dip_low3": bw_results["dip_low3"],
            "low3_drop_size": bw_results["low3_drop_size"],
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_latency_params(path, lat_results, raw_data_path, num_npus, split_mode):
    groups = {}
    for group_name, res in lat_results.items():
        groups[group_name] = {
            "t_s": float(res["t_s"]),
            "m1": int(res["m1"]),
            "m2": int(res["m2"]),
            "bw_max": float(res["bw_max"]),
            "mape": float(res["mape"]),
            "rmse": float(res["rmse"]),
            "has_dip": bool(res["has_dip"]),
            "dip": res["dip"],
            "drop_size": res["drop_size"],
            "spline_x_log2": _to_float_list(res["spl_x"]),
            "spline_residual_us": _to_float_list(res["spl_r"]),
        }

    payload = {
        "model_type": "a2a_latency",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_raw_data": _display_path(raw_data_path),
        "num_npus": int(num_npus),
        "split_mode": split_mode,
        "groups": groups,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_fit_metrics_csv(path, lat_results):
    rows = []
    for group_name, res in lat_results.items():
        rows.append(
            {
                "group": group_name,
                "mape_percent": float(res["mape"]),
                "rmse_us": float(res["rmse"]),
                "t_s_us": float(res["t_s"]),
                "m1_bytes": int(res["m1"]),
                "m2_bytes": int(res["m2"]),
                "bw_max_GBs": float(res["bw_max"]),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_validation_metrics_csv(path, compare_df):
    compare_df.to_csv(path, index=False)


def save_coverage_csv(path, coverage_rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "group", "samples", "unique_sizes", "min_size", "max_size", "log2_bins"],
        )
        writer.writeheader()
        writer.writerows(coverage_rows)


def save_summary_md(path, run_meta, lat_results, compare_df, overlap_sizes, coverage_rows):
    lines = [
        "# A2A Modeling Pipeline Summary",
        "",
        f"- Run time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Split mode: `{run_meta['split_mode']}`",
        f"- Train data: `{run_meta['train_data']}`",
        f"- Test data: `{run_meta['test_data']}`",
        "",
        "## Fit Metrics (Latency)",
        "",
        "| Group | MAPE(%) | RMSE(us) |",
        "|---|---:|---:|",
    ]

    for group_name, res in lat_results.items():
        lines.append(
            f"| {group_name} | {res['mape']:.3f} | {res['rmse']:.3f} |"
        )

    lines += ["", "## Train/Test Leakage Check", ""]
    if len(overlap_sizes) == 0:
        lines.append("- Unique-size overlap: **0** (size-disjoint)")
    else:
        lines.append(f"- Unique-size overlap: **{len(overlap_sizes)}** (not disjoint)")

    lines += ["", "## Coverage Summary", "", "| Dataset | Group | Samples | Unique Sizes | Min Size | Max Size | log2 Bins |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in coverage_rows:
        lines.append(
            f"| {row['dataset']} | {row['group']} | {row['samples']} | {row['unique_sizes']} | {row['min_size']} | {row['max_size']} | {row['log2_bins']} |"
        )

    if compare_df is not None and len(compare_df) > 0:
        mape = float(compare_df["rel_err"].mean())
        rmse = float(np.sqrt(np.mean(compare_df["abs_err"] ** 2)))
        p50 = float(compare_df["rel_err"].quantile(0.50))
        p90 = float(compare_df["rel_err"].quantile(0.90))
        p95 = float(compare_df["rel_err"].quantile(0.95))
        lines += [
            "",
            "## Validation Metrics",
            "",
            f"- Overall MAPE: {mape:.3f}%",
            f"- Overall RMSE: {rmse:.3f} us",
            f"- Relative Error P50/P90/P95: {p50:.3f}% / {p90:.3f}% / {p95:.3f}%",
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Fit A2A models and export artifacts.")
    parser.add_argument("--raw-data", default="data/raw/a2a_32kb_aligned.txt")
    parser.add_argument("--num-npus", type=int, default=8)
    parser.add_argument("--split-mode", choices=["size_holdout", "external_test"], default="size_holdout")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--validation-file", default="data/raw/a2a_test_external.txt")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    raw_data = project_root / args.raw_data
    validation_file = project_root / args.validation_file

    plots_dir = project_root / "artifacts/plots"
    models_dir = project_root / "artifacts/models"
    reports_dir = project_root / "artifacts/reports"
    split_dir = reports_dir / "split_data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    raw_df = parse_hccl_output(str(raw_data))
    if raw_df is None or len(raw_df) == 0:
        raise RuntimeError(f"cannot parse raw data: {raw_data}")

    if args.split_mode == "size_holdout":
        train_df, test_df = split_size_holdout(raw_df, test_ratio=args.test_ratio)
        train_file = split_dir / "train_data_size_holdout.txt"
        test_file = split_dir / "test_data_size_holdout.txt"
        write_hccl_like_file(train_df, train_file)
        write_hccl_like_file(test_df, test_file)
    else:
        train_df = raw_df.copy()
        train_file = raw_data
        test_file = validation_file
        test_df = None
        if not args.skip_validation:
            test_df = parse_hccl_output(str(test_file))
            if test_df is None or len(test_df) == 0:
                raise RuntimeError(f"cannot parse external test data: {test_file}")

    bw_results = run_two_stage_piecewise(
        file_path=str(train_file),
        num_gpus=args.num_npus,
        out_png=str(plots_dir / "bw_model_fit.png"),
        split_ratio=0.03,
    )

    mul_factor = MUL_FACTOR_FUNCS["all_to_all"](args.num_npus)
    train_df_for_lat = train_df.copy()
    train_df_for_lat["bus_bw"] = train_df_for_lat["alg_bw"] * mul_factor

    lat_results = run_latency_model(
        bw_results=bw_results,
        raw_df=train_df_for_lat,
        out_png=str(plots_dir / "latency_model_fit.png"),
        spline_smooth=None,
    )

    save_bw_params(models_dir / "bw_model_params.json", bw_results, train_file, args.num_npus, args.split_mode)
    save_latency_params(models_dir / "latency_model_params.json", lat_results, train_file, args.num_npus, args.split_mode)
    save_fit_metrics_csv(reports_dir / "fit_metrics.csv", lat_results)

    compare_df = None
    if not args.skip_validation:
        compare_df = parse_and_compare(
            result_file=str(test_file),
            lat_results=lat_results,
            out_png=str(plots_dir / "latency_compare.png"),
            num_gpus=args.num_npus,
            max_rows_print=80,
        )
        if compare_df is not None:
            save_validation_metrics_csv(reports_dir / "validation_metrics.csv", compare_df)

    if test_df is None and not args.skip_validation:
        test_df = parse_hccl_output(str(test_file))

    coverage_rows = summarize_coverage(train_df, "train")
    overlap_sizes = []
    if test_df is not None:
        coverage_rows += summarize_coverage(test_df, "test")
        overlap_sizes = calc_overlap(train_df, test_df)
    save_coverage_csv(reports_dir / "coverage_summary.csv", coverage_rows)

    run_meta = {
        "split_mode": args.split_mode,
        "train_data": _display_path(train_file),
        "test_data": _display_path(test_file) if not args.skip_validation else "<skipped>",
    }
    save_summary_md(
        reports_dir / "validation_summary.md",
        run_meta,
        lat_results,
        compare_df,
        overlap_sizes,
        coverage_rows,
    )

    print("[done] artifacts generated under artifacts/models, artifacts/reports, artifacts/plots")


if __name__ == "__main__":
    main()
