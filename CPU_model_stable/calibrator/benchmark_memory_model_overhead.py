#!/usr/bin/env python3
"""Measure CPU overhead of the white-box continuous and random memory modules.

This benchmark measures *model evaluation time*, rather than the latency of a
real memory access.  It reuses the same helpers as the formal white-box prior,
but deliberately excludes CSV loading, static-feature preparation, model
loading, and random-forest inference.  The resulting number therefore answers
one narrow question: how much CPU time is spent evaluating each memory module
after an operator row has already been prepared for prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrator.whitebox_family_routes import (  # noqa: E402
    EXEC_MODE_GEMM,
    EXEC_MODE_INDEXED_BLOCK_COPY,
    EXEC_MODE_PAYLOAD_CONCAT,
    EXEC_MODE_REDUCTION,
    EXEC_MODE_TRANSPOSE,
    _bytes_per_us_from_gb_s,
    add_whitebox_family_prior_columns,
    _branch_contention_share_triplet,
    _copy_bw_triplet_gb_s,
    _latency_parallelism_triplet,
    _latency_per_touch_us_by_level,
    _load_hw_params,
    _prepare_base_columns,
    _random_bw_triplet_gb_s,
    _random_memory_latency_composition_mode,
    _random_memory_latency_tail_alpha,
    _service_from_capacity,
    _worker_capacity_curve_gb_s,
)


DEFAULT_DATASET = REPO_ROOT / (
    "data/data_e2e_full_kunpeng920_20260501_portable/"
    "source_kunpeng920_dataset3_unseen_joint_6d_tuple/e2e_test_raw.csv"
)
DEFAULT_HW_CONFIG = REPO_ROOT / "calibrator/hw_config/kunpeng920_single_numa.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/memory_model_overhead_20260803"
CONTINUOUS_EXEC_MODES = {
    EXEC_MODE_REDUCTION,
    EXEC_MODE_PAYLOAD_CONCAT,
    EXEC_MODE_TRANSPOSE,
    EXEC_MODE_GEMM,
}


def _parse_sample_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        size = int(token)
        if size <= 0:
            raise argparse.ArgumentTypeError("sample sizes must be positive integers")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one sample size is required")
    return sorted(set(sizes))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure CPU wall-clock overhead of white-box continuous-memory and "
            "random-memory equations on prepared DLRM operator rows."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hw-config", type=Path, default=DEFAULT_HW_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sample-sizes",
        type=_parse_sample_sizes,
        default=_parse_sample_sizes("1,76,1024,4096"),
        help="Comma-separated prepared-row counts to measure. Default: 1,76,1024,4096.",
    )
    parser.add_argument("--repeats", type=int, default=200, help="Timed repetitions per module and sample size.")
    parser.add_argument("--warmups", type=int, default=20, help="Untimed warm-up repetitions per module and sample size.")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--verification-rows",
        type=int,
        default=16,
        help="Rows per module used to verify equivalence with the formal prior. Default: 16.",
    )
    parser.add_argument(
        "--skip-equivalence-check",
        action="store_true",
        help="Skip the untimed numerical-equivalence check against the formal white-box output.",
    )
    parser.add_argument(
        "--prior-variant",
        choices=["ort_kernel_v3"],
        default="ort_kernel_v3",
        help="This benchmark implements the ORT-kernel v3 equations only.",
    )
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0 or args.verification_rows <= 0:
        parser.error("--repeats and --verification-rows must be positive; --warmups must be non-negative")
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not args.hw_config.is_file():
        parser.error(f"hardware YAML does not exist: {args.hw_config}")
    return args


def _continuous_transfer_bytes(frame: pd.DataFrame) -> np.ndarray:
    """Return the stream-memory demand used by each supported formal branch.

    ReduceSum v3 has a dedicated read-plus-write field.  Concat and Transpose
    use the maximum of read/write bytes because they are one copy window;
    Gemm/MatMul uses total I/O bytes.  These are the same demand definitions as
    the stream-memory branches in ``add_whitebox_family_prior_columns``.
    """

    mode = frame["whitebox_exec_mode"].astype(str).to_numpy()
    read = pd.to_numeric(frame["exact_read_bytes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    write = pd.to_numeric(frame["exact_write_bytes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    demand = read + write

    reduce_mask = mode == EXEC_MODE_REDUCTION
    if np.any(reduce_mask):
        demand[reduce_mask] = pd.to_numeric(
            frame.loc[reduce_mask, "reduce_ort_stream_bytes"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)

    copy_window_mask = np.isin(mode, [EXEC_MODE_PAYLOAD_CONCAT, EXEC_MODE_TRANSPOSE])
    if np.any(copy_window_mask):
        demand[copy_window_mask] = np.maximum(read[copy_window_mask], write[copy_window_mask])
    reshape_like = pd.to_numeric(
        frame.get("feat_transpose_reshape_like_flag", pd.Series(0.0, index=frame.index)), errors="coerce"
    ).fillna(0.0).to_numpy()
    demand[(mode == EXEC_MODE_TRANSPOSE) & (reshape_like > 0.5)] = 0.0
    return np.maximum(demand, 0.0)


def _continuous_memory_intervals(frame: pd.DataFrame, hw_params: dict[str, Any]) -> np.ndarray:
    """Evaluate the formal stream-memory bandwidth and service-time equations.

    Return one row of lower/center/upper service times per input operator.
    """

    lower_bw, center_bw, upper_bw = _copy_bw_triplet_gb_s(frame, hw_params)
    # ORT ReduceSum uses kernel bandwidth, with the generic branch contention
    # removed and the embedding-private worker count restored.
    reduce_mask = frame["whitebox_exec_mode"].eq(EXEC_MODE_REDUCTION).to_numpy()
    if np.any(reduce_mask):
        workers = pd.to_numeric(frame["effective_worker_count"]).to_numpy(dtype=float)
        threads = pd.to_numeric(frame["num_threads"]).to_numpy(dtype=float)
        private = frame["exec_region_static"].eq("embedding_private").to_numpy()
        kernel_workers = np.where(private, np.maximum(workers, threads), workers)
        current_curve = _worker_capacity_curve_gb_s(
            workers, float(hw_params["copy_bw_gb_s_1t"]),
            float(hw_params["copy_bw_gb_s_saturated"]), hw_params,
        )
        kernel_curve = _worker_capacity_curve_gb_s(
            kernel_workers, float(hw_params["copy_bw_gb_s_1t"]),
            float(hw_params["copy_bw_gb_s_saturated"]), hw_params,
        )
        scale = kernel_curve / np.clip(current_curve, 1e-6, None)
        for bandwidth, share in zip(
            (lower_bw, center_bw, upper_bw),
            _branch_contention_share_triplet(frame, hw_params), strict=True,
        ):
            bandwidth[reduce_mask] *= np.clip(share[reduce_mask], 1.0, None) * scale[reduce_mask]
    lower, center, upper = _service_from_capacity(
        _continuous_transfer_bytes(frame),
        _bytes_per_us_from_gb_s(lower_bw),
        _bytes_per_us_from_gb_s(center_bw),
        _bytes_per_us_from_gb_s(upper_bw),
    )
    return np.column_stack((lower, center, upper))


def _random_memory_intervals(frame: pd.DataFrame, hw_params: dict[str, Any]) -> np.ndarray:
    """Evaluate the formal Gather random-memory branch without its copy tail.

    This is the random head of indexed Gather: cacheline-level source touches,
    random bandwidth, cache-residency latency, and the available miss-level
    parallelism.  The Gather destination copy is intentionally not included
    here because it is a continuous-memory operation measured above.
    """

    random_lower, random_center, random_upper, _, _, residency = _random_bw_triplet_gb_s(frame, hw_params)
    parallel_lower, parallel_center, parallel_upper = _latency_parallelism_triplet(frame, hw_params)
    latency_lower, latency_center, latency_upper = _latency_per_touch_us_by_level(residency, hw_params)

    worker_count = pd.to_numeric(frame["effective_worker_count"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    embedding = frame["submodel_role_static"].astype(str).eq("embedding_branch").to_numpy(dtype=bool)
    num_threads = pd.to_numeric(frame["num_threads"], errors="coerce").fillna(1.0).to_numpy(dtype=float)

    # The ORT-kernel v3 Gather path first removes the contention penalty from
    # the generic random-bandwidth curve, then applies a curve whose worker
    # count matches the embedding kernel.  Reusing this exact adjustment is
    # essential: the normalised random-bandwidth field alone is not the final
    # bandwidth used by the formal Gather random-memory equation.
    share_lower, share_center, share_upper = _branch_contention_share_triplet(frame, hw_params)
    isolated_lower = random_lower * np.clip(share_lower, 1.0, None)
    isolated_center = random_center * np.clip(share_center, 1.0, None)
    isolated_upper = random_upper * np.clip(share_upper, 1.0, None)
    embedding_private = frame["exec_region_static"].astype(str).eq("embedding_private").to_numpy(dtype=bool)
    kernel_workers = np.where(embedding_private, np.maximum(worker_count, num_threads), worker_count)
    current_curve = _worker_capacity_curve_gb_s(
        worker_count,
        float(hw_params["random_gather_bw_gb_s_1t"]),
        float(hw_params["random_gather_bw_gb_s_saturated"]),
        hw_params,
    )
    kernel_curve = _worker_capacity_curve_gb_s(
        kernel_workers,
        float(hw_params["random_gather_bw_gb_s_1t"]),
        float(hw_params["random_gather_bw_gb_s_saturated"]),
        hw_params,
    )
    kernel_scale = kernel_curve / np.clip(current_curve, 1e-6, None)
    random_lower = isolated_lower * kernel_scale
    random_center = isolated_center * kernel_scale
    random_upper = isolated_upper * kernel_scale

    parallel_lower = np.where(embedding, parallel_lower / np.clip(worker_count, 1.0, None), parallel_lower)
    parallel_center = np.where(embedding, parallel_center / np.clip(worker_count, 1.0, None), parallel_center)
    parallel_upper = np.where(embedding, parallel_upper / np.clip(worker_count, 1.0, None), parallel_upper)

    cacheline_bytes = max(float(hw_params["cacheline_bytes"]), 1.0)
    random_head_bytes = np.maximum(
        pd.to_numeric(frame["gather_head_cacheline_touches"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        * cacheline_bytes,
        pd.to_numeric(frame["gather_ort_source_payload_bytes"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )
    head_touches = np.maximum(
        np.maximum(
            pd.to_numeric(frame["gather_effective_head_touch_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            1.0,
        ),
        pd.to_numeric(frame["gather_ort_task_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )

    # A bandwidth-delay-product bound prevents the model from assuming more
    # outstanding cache misses than the target core can sustain.
    for bandwidth, latency, parallel in (
        (random_lower, latency_lower, parallel_lower),
        (random_center, latency_center, parallel_center),
        (random_upper, latency_upper, parallel_upper),
    ):
        bdp_parallelism = _bytes_per_us_from_gb_s(bandwidth) * latency / cacheline_bytes
        parallel[:] = np.maximum(parallel, np.clip(bdp_parallelism, 1.0, None))

    head_bw = (
        random_head_bytes / np.clip(_bytes_per_us_from_gb_s(random_lower), 1e-6, None),
        random_head_bytes / np.clip(_bytes_per_us_from_gb_s(random_center), 1e-6, None),
        random_head_bytes / np.clip(_bytes_per_us_from_gb_s(random_upper), 1e-6, None),
    )
    head_latency = (
        head_touches * latency_lower / np.clip(parallel_lower, 1.0, None),
        head_touches * latency_center / np.clip(parallel_center, 1.0, None),
        head_touches * latency_upper / np.clip(parallel_upper, 1.0, None),
    )

    composition = _random_memory_latency_composition_mode()
    tail_alpha = _random_memory_latency_tail_alpha()
    if composition == "additive":
        result = tuple(a + b for a, b in zip(head_bw, head_latency, strict=True))
    elif tail_alpha > 0.0:
        result = tuple(
            np.maximum(a, b) + tail_alpha * np.minimum(a, b)
            for a, b in zip(head_bw, head_latency, strict=True)
        )
    else:
        result = tuple(np.maximum(a, b) for a, b in zip(head_bw, head_latency, strict=True))
    return np.column_stack(result)


def _continuous_memory_core(frame: pd.DataFrame, hw_params: dict[str, Any]) -> float:
    return float(_continuous_memory_intervals(frame, hw_params).sum())


def _random_memory_core(frame: pd.DataFrame, hw_params: dict[str, Any]) -> float:
    return float(_random_memory_intervals(frame, hw_params).sum())


def _sample_rows(frame: pd.DataFrame, requested_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    actual_rows = min(requested_rows, len(frame))
    if actual_rows == len(frame):
        return frame
    indices = rng.choice(len(frame), size=actual_rows, replace=False)
    return frame.iloc[np.sort(indices)]


def _measure(
    func: Callable[[pd.DataFrame, dict[str, Any]], float],
    frame: pd.DataFrame,
    hw_params: dict[str, Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float], list[float], float]:
    checksum = 0.0
    for _ in range(warmups):
        checksum += func(frame, hw_params)

    elapsed_us: list[float] = []
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        checksum += func(frame, hw_params)
        elapsed_us.append((time.perf_counter_ns() - start_ns) / 1_000.0)

    values = np.asarray(elapsed_us, dtype=float)
    summary = {
        "mean_us": float(values.mean()),
        "median_us": float(np.median(values)),
        "p90_us": float(np.quantile(values, 0.90)),
        "min_us": float(values.min()),
        "max_us": float(values.max()),
        "per_row_mean_us": float(values.mean() / len(frame)),
    }
    return summary, elapsed_us, checksum


def _verify_formal_equivalence(
    raw: pd.DataFrame,
    prepared: pd.DataFrame,
    random_rows: pd.DataFrame,
    continuous_rows: pd.DataFrame,
    hw_params: dict[str, Any],
    *,
    prior_variant: str,
    rows_per_module: int,
) -> dict[str, Any]:
    """Check that the benchmark equations reproduce the formal module outputs.

    The check is intentionally untimed.  A small mixed subset is evaluated by
    the full formal prior and compared with the two extracted module kernels.
    This guards against future drift when the main Gather or stream equations
    are changed.
    """

    if prior_variant != "ort_kernel_v3":
        raise ValueError("benchmark equations support ort_kernel_v3 only")
    random_index = random_rows.index[:rows_per_module]
    continuous_index = continuous_rows.index[:rows_per_module]
    selected_index = random_index.append(continuous_index).unique()
    raw_subset = raw.loc[selected_index].copy()
    prepared_subset = prepared.loc[selected_index].copy()
    formal = add_whitebox_family_prior_columns(
        raw_subset,
        {"hw_params": hw_params, "prior_variant": prior_variant},
    )

    formal_random = formal.loc[
        formal["whitebox_exec_mode"].eq(EXEC_MODE_INDEXED_BLOCK_COPY),
        ["svc_random_mem_lower_us", "svc_random_mem_center_us", "svc_random_mem_upper_us"],
    ].to_numpy(dtype=float)
    formal_continuous = formal.loc[
        formal["whitebox_exec_mode"].isin(CONTINUOUS_EXEC_MODES),
        ["svc_stream_mem_lower_us", "svc_stream_mem_center_us", "svc_stream_mem_upper_us"],
    ].to_numpy(dtype=float)
    benchmark_random = _random_memory_intervals(
        prepared_subset.loc[prepared_subset["whitebox_exec_mode"].eq(EXEC_MODE_INDEXED_BLOCK_COPY)], hw_params
    )
    benchmark_continuous = _continuous_memory_intervals(
        prepared_subset.loc[prepared_subset["whitebox_exec_mode"].isin(CONTINUOUS_EXEC_MODES)], hw_params
    )
    random_abs_diff = np.max(np.abs(benchmark_random - formal_random))
    continuous_abs_diff = np.max(np.abs(benchmark_continuous - formal_continuous))
    passed = bool(
        np.allclose(benchmark_random, formal_random, rtol=1e-12, atol=1e-9)
        and np.allclose(benchmark_continuous, formal_continuous, rtol=1e-12, atol=1e-9)
    )
    details = {
        "status": "passed" if passed else "failed",
        "comparison": "per_row_per_bound_max_absolute_difference",
        "random_rows": int(len(random_index)),
        "continuous_rows": int(len(continuous_index)),
        "random_abs_diff": float(random_abs_diff),
        "continuous_abs_diff": float(continuous_abs_diff),
    }
    if not passed:
        raise RuntimeError(f"benchmark equations do not match the formal prior: {details}")
    return details


def _write_report(out_dir: Path, summary: pd.DataFrame, metadata: dict[str, Any]) -> None:
    equivalence = metadata["formal_equivalence"]
    if equivalence["status"] == "passed":
        equivalence_line = (
            "正式方程一致性校验：passed（随机访存绝对差 "
            f"{equivalence['random_abs_diff']:.3e}，连续访存绝对差 {equivalence['continuous_abs_diff']:.3e}）。"
        )
    else:
        equivalence_line = "正式方程一致性校验：skipped。"
    lines = [
        "# CPU 访存建模开销测量",
        "",
        "该结果是白盒模型在 CPU 上计算访存模块的墙钟时间，不是实际 Cache 或 DRAM 访问延迟。",
        "CSV 读取、静态特征展开、模型加载和随机森林残差预测均不在计时区间内。",
        "",
        "- 连续访存模块：带宽曲线、并发争用修正和连续数据搬运服务时间。",
        "- 随机访存模块：Gather 随机源地址访问的随机带宽、缓存层级延迟、可并行未命中数和时间合成；不含 Gather 输出端连续拷贝。",
        "",
        f"输入数据：`{metadata['dataset']}`",
        f"硬件参数：`{metadata['hw_config']}`",
        f"静态特征准备耗时（未计入结果）：{metadata['static_preparation_seconds']:.3f} s",
        f"重复次数：{metadata['repeats']}，预热次数：{metadata['warmups']}",
        equivalence_line,
        "",
        "| 模块 | 请求行数 | 实测行数 | 平均耗时 (us) | P90 (us) | 单行平均 (us) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {module} | {requested_rows} | {actual_rows} | {mean_us:.2f} | {p90_us:.2f} | {per_row_mean_us:.4f} |".format(
                **row.to_dict()
            )
        )
    lines.extend(
        [
            "",
            "完整重复测量值见 `memory_model_overhead_samples.csv`，运行环境和样本数量见 "
            "`memory_model_overhead_metadata.json`。",
        ]
    )
    (out_dir / "memory_model_overhead_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The preparation stage is required to produce the exact formal input
    # fields, but it is deliberately outside the timed region.
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    raw = pd.read_csv(args.dataset, low_memory=False)
    hw_params = _load_hw_params(args.hw_config, require_measured_whitebox=True)
    prep_start = time.perf_counter()
    prepared = _prepare_base_columns(raw, hw_params, prior_variant=args.prior_variant).copy()
    preparation_seconds = time.perf_counter() - prep_start

    random_rows = prepared.loc[prepared["whitebox_exec_mode"].eq(EXEC_MODE_INDEXED_BLOCK_COPY)].copy()
    continuous_rows = prepared.loc[prepared["whitebox_exec_mode"].isin(CONTINUOUS_EXEC_MODES)].copy()
    if random_rows.empty:
        raise RuntimeError("no indexed Gather rows were found for random-memory measurement")
    if continuous_rows.empty:
        raise RuntimeError("no stream-memory operator rows were found for continuous-memory measurement")
    if args.skip_equivalence_check:
        formal_equivalence: dict[str, Any] = {"status": "skipped", "random_abs_diff": None, "continuous_abs_diff": None}
    else:
        formal_equivalence = _verify_formal_equivalence(
            raw,
            prepared,
            random_rows,
            continuous_rows,
            hw_params,
            prior_variant=args.prior_variant,
            rows_per_module=args.verification_rows,
        )

    rng = np.random.default_rng(args.seed)
    module_specs: list[tuple[str, pd.DataFrame, Callable[[pd.DataFrame, dict[str, Any]], float]]] = [
        ("continuous_memory", continuous_rows, _continuous_memory_core),
        ("random_memory", random_rows, _random_memory_core),
    ]
    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    checksums: dict[str, float] = {}
    for module_name, module_frame, func in module_specs:
        for requested_rows in args.sample_sizes:
            sample = _sample_rows(module_frame, requested_rows, rng)
            metrics, elapsed_us, checksum = _measure(
                func,
                sample,
                hw_params,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            checksums[f"{module_name}_{requested_rows}"] = checksum
            summary_rows.append(
                {
                    "module": module_name,
                    "requested_rows": requested_rows,
                    "actual_rows": len(sample),
                    "available_rows": len(module_frame),
                    "repeats": args.repeats,
                    "warmups": args.warmups,
                    **metrics,
                }
            )
            sample_rows.extend(
                {
                    "module": module_name,
                    "requested_rows": requested_rows,
                    "actual_rows": len(sample),
                    "repeat_index": repeat_index,
                    "elapsed_us": elapsed,
                }
                for repeat_index, elapsed in enumerate(elapsed_us)
            )

    summary = pd.DataFrame(summary_rows)
    samples = pd.DataFrame(sample_rows)
    metadata: dict[str, Any] = {
        "dataset": str(args.dataset.resolve()),
        "hw_config": str(args.hw_config.resolve()),
        "prior_variant": args.prior_variant,
        "seed": args.seed,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "sample_sizes": args.sample_sizes,
        "available_random_rows": len(random_rows),
        "available_continuous_rows": len(continuous_rows),
        "static_preparation_seconds": preparation_seconds,
        "formal_equivalence": formal_equivalence,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "checksums": checksums,
    }
    summary.to_csv(args.out_dir / "memory_model_overhead_summary.csv", index=False)
    samples.to_csv(args.out_dir / "memory_model_overhead_samples.csv", index=False)
    (args.out_dir / "memory_model_overhead_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(args.out_dir, summary, metadata)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"results written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
