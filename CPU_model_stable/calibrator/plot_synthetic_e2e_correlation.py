from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_POINTS = (
    "outputs/cache_residency_worker_share_v3_trends_20260505/"
    "trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload/"
    "synthetic_e2e_trend_sweep_points_route_pred.csv"
)
DEFAULT_BINDING = "calibrator/configs/hw_generalization_bindings_cache_residency_worker_share_20260505.json"
DEFAULT_OUT = (
    "outputs/cache_residency_worker_share_v3_trends_20260505/"
    "trend_figures_synthetic_model_physics_guarded_rf_v3_multiworkload"
)

ALIAS_TO_DEVICE = {
    "kunpeng920_remote": "Device_1",
    "s920x20_local": "Device_2",
}

SOFTWARE_FEATURES = [
    "batch_size",
    "num_indices_per_lookup",
    "embedding_size_total",
    "bottom_mlp_ops_proxy",
    "inter_threads",
    "num_threads",
]

HARDWARE_FEATURES = [
    "core.total_cores",
    "core.cpu_clock",
    "pipeline.fetch_width",
    "pipeline.decode_width",
    "pipeline.rename_width",
    "pipeline.dispatch_width",
    "pipeline.issue_width",
    "pipeline.commit_width",
    "instruction.simd_width_bits",
    "instruction.fp_throughput_per_cycle.vector_sp_add",
    "instruction.fp_throughput_per_cycle.vector_sp_mul",
    "instruction.fp_throughput_per_cycle.vector_sp_fma",
    "instruction.fp_latency_cycles.vector_sp_add",
    "instruction.fp_latency_cycles.vector_sp_mul",
    "instruction.fp_latency_cycles.vector_sp_fma",
    "memory.bandwidth_gb_s_total",
    "memory.copy_bw_gb_s_1t",
    "memory.copy_bw_gb_s_saturated",
    "memory.random_gather_bw_gb_s_1t",
    "memory.random_gather_bw_gb_s_saturated",
    "memory.local_mem_delay_ns",
    "runtime.parallel_launch_overhead_us_intra",
    "runtime.parallel_launch_overhead_us_inter",
    "cache.cacheline_bytes",
    "cache.l1d.size",
    "cache.l2.size",
    "cache.l3_per_die.size",
    "cache.l1d.response_latency_cycles",
    "cache.l2.response_latency_cycles",
    "cache.l3_per_die.response_latency_cycles",
]

TREND_TO_HW_FIELDS = {
    "l1d_size": ["cache.l1d.size"],
    "l2_size": ["cache.l2.size"],
    "l3_size": ["cache.l3_per_die.size"],
    "l1d_latency": ["cache.l1d.response_latency_cycles"],
    "l2_latency": ["cache.l2.response_latency_cycles"],
    "l3_latency": ["cache.l3_per_die.response_latency_cycles"],
    "copy_bw": ["memory.copy_bw_gb_s_1t", "memory.copy_bw_gb_s_saturated"],
    "random_gather_bw": ["memory.random_gather_bw_gb_s_1t", "memory.random_gather_bw_gb_s_saturated"],
    "mem_latency": ["memory.local_mem_delay_ns"],
    "cpu_clock": ["core.cpu_clock"],
    "simd_width": ["instruction.simd_width_bits"],
    "fma_throughput": ["instruction.fp_throughput_per_cycle.vector_sp_fma"],
    "fma_latency": ["instruction.fp_latency_cycles.vector_sp_fma"],
    "pipeline_width": [
        "pipeline.fetch_width",
        "pipeline.decode_width",
        "pipeline.rename_width",
        "pipeline.dispatch_width",
        "pipeline.issue_width",
        "pipeline.commit_width",
    ],
    "full_compute_bundle": [
        "core.cpu_clock",
        "instruction.simd_width_bits",
        "instruction.fp_throughput_per_cycle.vector_sp_fma",
    ],
    "full_memory_bundle": [
        "memory.copy_bw_gb_s_1t",
        "memory.copy_bw_gb_s_saturated",
        "memory.random_gather_bw_gb_s_1t",
        "memory.random_gather_bw_gb_s_saturated",
    ],
}

TREND_ORDER = [
    "active_parallel_workers",
    "l1d_size",
    "l2_size",
    "l3_size",
    "l1d_latency",
    "l2_latency",
    "l3_latency",
    "copy_bw",
    "random_gather_bw",
    "mem_latency",
    "cpu_clock",
    "simd_width",
    "fma_throughput",
    "fma_latency",
    "pipeline_width",
    "full_compute_bundle",
    "full_memory_bundle",
]

TREND_LABELS = {
    "active_parallel_workers": "Active workers",
    "l1d_size": "L1D size",
    "l2_size": "L2 size",
    "l3_size": "L3 size",
    "l1d_latency": "L1D latency",
    "l2_latency": "L2 latency",
    "l3_latency": "L3 latency",
    "copy_bw": "Copy bandwidth",
    "random_gather_bw": "Random gather BW",
    "mem_latency": "DRAM latency",
    "cpu_clock": "CPU frequency",
    "simd_width": "SIMD width",
    "fma_throughput": "FMA throughput",
    "fma_latency": "FMA latency",
    "pipeline_width": "Pipeline width",
    "full_compute_bundle": "Compute bundle",
    "full_memory_bundle": "Memory bundle",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic E2E software/hardware correlation heatmap.")
    parser.add_argument("--points-csv", default=DEFAULT_POINTS)
    parser.add_argument("--binding-config", default=DEFAULT_BINDING)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--target-column", default="pred_us")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset aliases to include; empty keeps all aliases in the points CSV.",
    )
    parser.add_argument(
        "--hide-device-label",
        action="store_true",
        help="Hide Device_1/Device_2 labels on heatmap x axes; useful when plotting a single dataset alias.",
    )
    return parser.parse_args()


def _path_from_config(raw: str, *, base_dir: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _nested_get(payload: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _parse_quantity(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().strip("\"'")
    if not text:
        return float("nan")
    lower = text.lower()
    multiplier = 1.0
    for suffix, factor in (
        ("kib", 1024.0),
        ("mib", 1024.0 * 1024.0),
        ("gib", 1024.0 * 1024.0 * 1024.0),
        ("kb", 1000.0),
        ("mb", 1000.0 * 1000.0),
        ("gb", 1000.0 * 1000.0 * 1000.0),
        ("ghz", 1.0),
        ("mhz", 1.0e-3),
        ("khz", 1.0e-6),
    ):
        if lower.endswith(suffix):
            multiplier = factor
            text = text[: -len(suffix)].strip()
            break
    try:
        return float(text) * multiplier
    except ValueError:
        return float("nan")


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    return parsed if isinstance(parsed, list) else [parsed]


def _parse_dash_numbers(value: Any) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return []
    values: list[float] = []
    for token in text.split("-"):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _embedding_size_total(value: Any) -> float:
    return float(sum(_parse_dash_numbers(value)))


def _bottom_mlp_ops_proxy(value: Any) -> float:
    dims = _parse_dash_numbers(value)
    if len(dims) < 2:
        return float("nan")
    return float(sum(a * b for a, b in zip(dims[:-1], dims[1:])))


def _load_bindings(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    result: dict[str, dict[str, Any]] = {}
    for alias, item in dict(payload.get("datasets", {})).items():
        hw_config = _path_from_config(str(item["hw_config"]), base_dir=base_dir)
        result[str(alias)] = {
            "hw_config": hw_config,
            "actual_hardware": str(item.get("actual_hardware", alias)),
        }
    return result


def _load_hw_values(path: Path) -> dict[str, float]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load hardware config files")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    values: dict[str, float] = {}
    for feature in HARDWARE_FEATURES:
        values[feature] = _parse_quantity(_nested_get(payload, feature))
    return values


def _mutated_hw_values(row: pd.Series, baseline: dict[str, float]) -> dict[str, float]:
    values = dict(baseline)
    fields = TREND_TO_HW_FIELDS.get(str(row.get("trend_key", "")), [])
    mutated = _parse_list(row.get("mutated_value"))
    for field, raw in zip(fields, mutated):
        parsed = _parse_quantity(raw)
        if math.isfinite(parsed):
            values[field] = parsed
    return values


def _prepare_feature_frame(points: pd.DataFrame, bindings: dict[str, dict[str, Any]], target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hw_baselines = {
        alias: _load_hw_values(binding["hw_config"])
        for alias, binding in bindings.items()
        if alias in set(points["dataset_alias"].astype(str))
    }
    rows: list[dict[str, Any]] = []
    for _, row in points.iterrows():
        alias = str(row["dataset_alias"])
        if alias not in hw_baselines:
            continue
        item: dict[str, Any] = {
            "dataset_alias": alias,
            "device_id": ALIAS_TO_DEVICE.get(alias, alias),
            "target": pd.to_numeric(row.get(target_column), errors="coerce"),
        }
        item["batch_size"] = pd.to_numeric(row.get("batch_size"), errors="coerce")
        item["num_indices_per_lookup"] = pd.to_numeric(row.get("num_indices_per_lookup"), errors="coerce")
        item["embedding_size_total"] = _embedding_size_total(row.get("arch_embedding_size"))
        item["bottom_mlp_ops_proxy"] = _bottom_mlp_ops_proxy(row.get("arch_mlp_bot"))
        item["inter_threads"] = pd.to_numeric(row.get("inter_threads"), errors="coerce")
        item["num_threads"] = pd.to_numeric(row.get("num_threads"), errors="coerce")
        for feature, value in _mutated_hw_values(row, hw_baselines[alias]).items():
            item[feature] = value
        rows.append(item)
    feature_frame = pd.DataFrame(rows)
    feature_frame = feature_frame[np.isfinite(feature_frame["target"].to_numpy(dtype=float))].reset_index(drop=True)
    feature_catalog = pd.DataFrame(
        [{"feature_type": "software", "feature": feature} for feature in SOFTWARE_FEATURES]
        + [{"feature_type": "hardware", "feature": feature} for feature in HARDWARE_FEATURES]
    )
    return feature_frame, feature_catalog


def _spearman(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 3 or data["x"].nunique(dropna=True) <= 1 or data["y"].nunique(dropna=True) <= 1:
        return float("nan")
    return float(data["x"].rank(method="average").corr(data["y"].rank(method="average")))


def _loglog_slope(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    data = data[(data["x"] > 0) & (data["y"] > 0)]
    if len(data) < 3 or data["x"].nunique(dropna=True) <= 1 or data["y"].nunique(dropna=True) <= 1:
        return float("nan")
    return float(np.polyfit(np.log(data["x"].to_numpy(dtype=float)), np.log(data["y"].to_numpy(dtype=float)), 1)[0])


def _endpoint_log_elasticity(x: pd.Series, y: pd.Series) -> dict[str, float]:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    data = data[(data["x"] > 0) & (data["y"] > 0)]
    if len(data) < 2 or data["x"].nunique(dropna=True) <= 1:
        return {
            "endpoint_log_elasticity_raw": float("nan"),
            "abs_log_elasticity": float("nan"),
            "elasticity_x_min": float("nan"),
            "elasticity_x_max": float("nan"),
            "elasticity_y_at_x_min": float("nan"),
            "elasticity_y_at_x_max": float("nan"),
        }
    data = data.sort_values("x")
    first = data.iloc[0]
    last = data.iloc[-1]
    denom = math.log(float(last["x"])) - math.log(float(first["x"]))
    if abs(denom) <= 1e-12:
        raw = float("nan")
    else:
        raw = (math.log(float(last["y"])) - math.log(float(first["y"]))) / denom
    return {
        "endpoint_log_elasticity_raw": float(raw),
        "abs_log_elasticity": abs(float(raw)) if math.isfinite(raw) else float("nan"),
        "elasticity_x_min": float(first["x"]),
        "elasticity_x_max": float(last["x"]),
        "elasticity_y_at_x_min": float(first["y"]),
        "elasticity_y_at_x_max": float(last["y"]),
    }


def _device_ids(frame: pd.DataFrame) -> list[str]:
    present = set(frame.get("device_id", pd.Series(dtype=str)).astype(str))
    preferred = [device for device in ["Device_1", "Device_2"] if device in present]
    preferred.extend(sorted(present.difference(preferred)))
    return preferred


def _correlation_tables(feature_frame: pd.DataFrame, feature_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    devices = _device_ids(feature_frame)
    for _, catalog_row in feature_catalog.iterrows():
        feature = str(catalog_row["feature"])
        feature_type = str(catalog_row["feature_type"])
        item: dict[str, Any] = {"feature_type": feature_type, "feature": feature}
        keep = False
        for device in devices:
            part = feature_frame[feature_frame["device_id"].astype(str).eq(device)]
            if feature not in part.columns or part[feature].nunique(dropna=True) <= 1:
                item[device] = np.nan
                skipped_rows.append(
                    {
                        "device_id": device,
                        "feature_type": feature_type,
                        "feature": feature,
                        "reason": "constant_or_missing",
                    }
                )
                continue
            item[device] = _spearman(part[feature], part["target"])
            keep = True
        if keep:
            matrix_rows.append(item)
    matrix = pd.DataFrame(matrix_rows)
    skipped = pd.DataFrame(skipped_rows)
    return matrix, skipped


def _trend_max_change(points: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        points.groupby(["trend_group", "trend_key", "trend_title", "dataset_alias", "workload_variant"], dropna=False)
        .agg(
            min_pred_us=("pred_us", "min"),
            max_pred_us=("pred_us", "max"),
            baseline_pred_us=("baseline_pred_us", "first"),
        )
        .reset_index()
    )
    grouped["change_pct"] = (grouped["max_pred_us"] - grouped["min_pred_us"]) / grouped["min_pred_us"].replace(0.0, np.nan) * 100.0
    grouped["device_id"] = grouped["dataset_alias"].astype(str).map(ALIAS_TO_DEVICE).fillna(grouped["dataset_alias"].astype(str))
    grouped = grouped.sort_values(["trend_group", "trend_key", "device_id", "change_pct"], ascending=[True, True, True, False])
    best = grouped.groupby(["trend_group", "trend_key", "trend_title", "device_id"], as_index=False, dropna=False).head(1)
    return best.reset_index(drop=True)


def _metric_comparison_summary(best: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for device in _device_ids(local):
        for metric, frame, column in (
            ("local_directional_spearman", local, "directional_spearman"),
            ("best_workload_directional_spearman", best, "directional_spearman"),
            ("best_workload_directional_elasticity", best, "directional_elasticity"),
            ("best_workload_abs_log_elasticity", best, "abs_log_elasticity"),
            ("best_workload_change_pct", best, "change_pct"),
        ):
            values = pd.to_numeric(frame.loc[frame["device_id"].astype(str).eq(device), column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "device_id": device,
                    "median": float(values.median()),
                    "mean": float(values.mean()),
                    "p90": float(values.quantile(0.9)),
                    "max": float(values.max()),
                    "count": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def _trend_local_sensitivity(points: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    group_cols = ["trend_group", "trend_key", "trend_title", "dataset_alias", "workload_variant", "relation"]
    for key, group in points.groupby(group_cols, dropna=False):
        trend_group, trend_key, trend_title, dataset_alias, workload_variant, relation = key
        x_col = "active_parallel_workers" if str(trend_key) == "active_parallel_workers" else "relative_parameter"
        x = pd.to_numeric(group.get(x_col), errors="coerce")
        y = pd.to_numeric(group.get(target_column), errors="coerce")
        raw_rho = _spearman(x, y)
        raw_elasticity = _loglog_slope(x, y)
        endpoint_elasticity = _endpoint_log_elasticity(x, y)
        if str(relation) == "capacity":
            directional_rho = -raw_rho if math.isfinite(raw_rho) else float("nan")
            directional_elasticity = -raw_elasticity if math.isfinite(raw_elasticity) else float("nan")
        else:
            directional_rho = raw_rho
            directional_elasticity = raw_elasticity
        min_pred = float(y.min())
        max_pred = float(y.max())
        change_pct = (max_pred - min_pred) / min_pred * 100.0 if min_pred > 0 else float("nan")
        rows.append(
            {
                "trend_group": trend_group,
                "trend_key": trend_key,
                "trend_title": trend_title,
                "dataset_alias": dataset_alias,
                "device_id": ALIAS_TO_DEVICE.get(str(dataset_alias), str(dataset_alias)),
                "workload_variant": workload_variant,
                "relation": relation,
                "x_column": x_col,
                "raw_spearman": raw_rho,
                "directional_spearman": directional_rho,
                "loglog_elasticity_raw": raw_elasticity,
                "directional_elasticity": directional_elasticity,
                **endpoint_elasticity,
                "change_pct": change_pct,
                "n_points": int(len(group)),
                "min_pred_us": min_pred,
                "max_pred_us": max_pred,
            }
        )
    local = pd.DataFrame(rows)
    if local.empty:
        return local, local, pd.DataFrame()
    best_idx = local.groupby(["trend_group", "trend_key", "trend_title", "device_id"], dropna=False)["abs_log_elasticity"].idxmax()
    best = local.loc[best_idx].sort_values(["trend_group", "trend_key", "device_id"]).reset_index(drop=True)
    return local, best, _metric_comparison_summary(best, local)


def _plot_heatmap(matrix: pd.DataFrame, out_dir: Path, *, hide_device_label: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if matrix.empty:
        return
    device_columns = [column for column in matrix.columns if str(column).startswith("Device_")]
    if not device_columns:
        return
    values = matrix[device_columns].to_numpy(dtype=float)
    labels = matrix["feature"].astype(str).tolist()
    fig_h = max(6.0, 0.28 * len(labels) + 1.8)
    fig_w = 7.0 if len(device_columns) > 1 else 5.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(device_columns)))
    ax.set_xticklabels(["" for _ in device_columns] if hide_device_label else device_columns)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Synthetic E2E Latency Spearman Correlation")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            text = "" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Spearman r vs E2E latency")
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_e2e_hw_sw_correlation_heatmap.png", dpi=300)
    fig.savefig(out_dir / "synthetic_e2e_hw_sw_correlation_heatmap.pdf")
    plt.close(fig)


def _plot_local_metric_heatmap(
    best: pd.DataFrame,
    out_dir: Path,
    *,
    metric: str,
    filename: str,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
    hide_device_label: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if best.empty or metric not in best.columns:
        return
    pivot = best.pivot_table(index="trend_key", columns="device_id", values=metric, aggfunc="first").reindex(TREND_ORDER)
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return
    labels = [TREND_LABELS.get(str(item), str(item)) for item in pivot.index]
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.9, max(5.8, 0.34 * len(labels) + 1.8)))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(["" for _ in pivot.columns] if hide_device_label else [str(col) for col in pivot.columns])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if np.isfinite(value):
                ax.text(x, y, format(value, fmt), ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename}.png", dpi=300)
    fig.savefig(out_dir / f"{filename}.pdf")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    points_path = Path(args.points_csv)
    binding_path = Path(args.binding_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(points_path, low_memory=False)
    requested_datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    if requested_datasets:
        points = points.loc[points["dataset_alias"].astype(str).isin(set(requested_datasets))].reset_index(drop=True)
    bindings = _load_bindings(binding_path)
    if requested_datasets:
        bindings = {alias: item for alias, item in bindings.items() if alias in set(requested_datasets)}
    feature_frame, feature_catalog = _prepare_feature_frame(points, bindings, str(args.target_column))
    matrix, skipped = _correlation_tables(feature_frame, feature_catalog)
    max_change = _trend_max_change(points)
    local_sensitivity, best_sensitivity, metric_summary = _trend_local_sensitivity(points, str(args.target_column))

    matrix.to_csv(out_dir / "synthetic_e2e_hw_sw_correlation_matrix.csv", index=False)
    skipped.to_csv(out_dir / "synthetic_e2e_hw_sw_correlation_skipped_constants.csv", index=False)
    max_change.to_csv(out_dir / "synthetic_e2e_trend_max_change_by_device.csv", index=False)
    feature_frame.to_csv(out_dir / "synthetic_e2e_hw_sw_correlation_feature_rows.csv", index=False)
    local_sensitivity.to_csv(out_dir / "synthetic_e2e_hardware_local_sensitivity_by_workload.csv", index=False)
    best_sensitivity.to_csv(out_dir / "synthetic_e2e_hardware_local_sensitivity_best_workload.csv", index=False)
    metric_summary.to_csv(out_dir / "synthetic_e2e_hardware_metric_comparison_summary.csv", index=False)
    _plot_heatmap(matrix, out_dir, hide_device_label=bool(args.hide_device_label))
    _plot_local_metric_heatmap(
        best_sensitivity,
        out_dir,
        metric="directional_spearman",
        filename="synthetic_e2e_hardware_local_directional_spearman_heatmap",
        title="Trend-local Direction-corrected Spearman",
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
        hide_device_label=bool(args.hide_device_label),
    )
    _plot_local_metric_heatmap(
        best_sensitivity,
        out_dir,
        metric="directional_elasticity",
        filename="synthetic_e2e_hardware_directional_elasticity_heatmap",
        title="Direction-corrected Log-log Elasticity",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=None,
        fmt=".2f",
        hide_device_label=bool(args.hide_device_label),
    )
    _plot_local_metric_heatmap(
        best_sensitivity,
        out_dir,
        metric="abs_log_elasticity",
        filename="synthetic_e2e_hardware_abs_log_elasticity_heatmap",
        title="Absolute Log-log Elasticity vs E2E Latency",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=None,
        fmt=".2f",
        hide_device_label=bool(args.hide_device_label),
    )
    _plot_local_metric_heatmap(
        best_sensitivity,
        out_dir,
        metric="change_pct",
        filename="synthetic_e2e_hardware_max_change_pct_heatmap",
        title="Max E2E Latency Change across Sweep (%)",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=None,
        fmt=".1f",
        hide_device_label=bool(args.hide_device_label),
    )

    manifest = {
        "points_csv": str(points_path),
        "binding_config": str(binding_path),
        "datasets": requested_datasets,
        "hide_device_label": bool(args.hide_device_label),
        "target_column": str(args.target_column),
        "rows": int(len(feature_frame)),
        "matrix_rows": int(len(matrix)),
        "skipped_rows": int(len(skipped)),
        "local_sensitivity_rows": int(len(local_sensitivity)),
        "best_sensitivity_rows": int(len(best_sensitivity)),
        "method": "spearman_rank_correlation",
        "hardware_local_sensitivity_method": "trend_local_direction_corrected_spearman_and_loglog_elasticity",
    }
    (out_dir / "synthetic_e2e_hw_sw_correlation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
