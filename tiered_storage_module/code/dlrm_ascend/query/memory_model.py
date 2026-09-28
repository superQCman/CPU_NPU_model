#!/usr/bin/env python3
"""
Memory Access Time Model for DLRM 3-Level Storage

Model equation (latency + bandwidth):
    T = α × n_ops + DataSize / BW

    where:
      T        — total transfer time (seconds)
      α        — per-operation fixed latency (seconds/op), covers startup overhead,
                 queue delay, device wakeup, etc.
      n_ops    — number of discrete access operations (hits or misses)
      DataSize — total bytes transferred (bytes)
      BW       — effective bandwidth (bytes/s)

Four stages modeled:
    0. HBM_to_NPU  — on-device HBM → compute unit
    1. DRAM_to_CPU — host DRAM → GPU via PCIe
    2. SSD_to_DRAM — NVMe SSD → host DRAM (random-read dominated)
    3. DRAM_to_HBM — host DRAM → device HBM during hot-cold migration

Usage:
    # Quick prediction with defaults
    python memory_model.py --hbm-hits 800 --dram-hits 150 --ssd-misses 50 \
        --dim 64000 --migrate-count 10

    # Calibrate from log file + predict
    python memory_model.py --calibrate memory_access_timings.log \
        --hbm-hits 800 --dram-hits 150 --ssd-misses 50 --dim 64000

    # Integration with reuse distance output (JSON)
    python memory_model.py --from-reuse result.json --dim 64000
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 1. Data Structures
# ============================================================

@dataclass
class PhysicalSpec:
    """Physical (theoretical) specification of a memory link."""
    name: str
    bw_gb_s: float        # Peak bandwidth (GB/s)
    latency_us: float     # Per-access fixed latency (μs)
    description: str = ""


@dataclass
class FittedParams:
    """Calibrated parameters for one stage.

    Equation:  T(s) = alpha_us * 1e-6 * n_ops + data_bytes / (bw_gb_s * 1e9)
    """
    alpha_us: float       # Per-operation latency (μs)
    bw_gb_s: float        # Effective bandwidth (GB/s)

    def predict_s(self, n_ops: int, data_bytes: float) -> float:
        """Predict transfer time in seconds."""
        alpha_s = self.alpha_us * 1e-6
        bw_bytes_s = self.bw_gb_s * 1e9
        return alpha_s * n_ops + data_bytes / max(bw_bytes_s, 1.0)  # avoid div by zero


@dataclass
class StageSnapshot:
    """One measurement point from a log snapshot."""
    step: int
    n_ops: int            # Cumulative access count for this stage
    data_bytes: float     # Cumulative bytes for this stage
    time_s: float         # Cumulative wall time for this stage


@dataclass
class CalibrationResult:
    """Result of calibrating one stage."""
    fitted: FittedParams
    r2: float             # Coefficient of determination
    n_samples: int        # Number of snapshots used
    condition: str        # "full" (α+BW) | "bw_only" (fixed α) | "insufficient_data"


# Default physical parameters — calibrated from measured profiling data
#   Values from memory_access_timings.log steady-state (step≥7):
#     HBM:  45-57 GB/s  (gather ops on Ascend NPU, not theoretical 1200)
#     DRAM: 1.2-1.7 GB/s (PCIe host→device, not theoretical 100)
#     SSD:  0.3 GB/s    (NVMe random read effective)
DEFAULT_SPECS: Dict[str, PhysicalSpec] = {
    "HBM_to_NPU": PhysicalSpec(
        name="HBM → NPU",
        bw_gb_s=50.0,           # measured effective BW from profiler
        latency_us=0.02,
        description="On-device HBM gather (profiler: 45-57 GB/s effective)",
    ),
    "DRAM_to_CPU": PhysicalSpec(
        name="DRAM → CPU/GPU",
        bw_gb_s=1.5,            # measured effective BW from profiler
        latency_us=0.03,
        description="Host DRAM to GPU via PCIe (profiler: 1.2-1.7 GB/s effective)",
    ),
    "SSD_to_DRAM": PhysicalSpec(
        name="SSD → DRAM",
        bw_gb_s=0.4,            # calibrated via OLS from 13 measurements
        latency_us=10.0,
        description="NVMe SSD to host DRAM (0.3-0.4 GB/s effective)",
    ),
    "DRAM_to_HBM": PhysicalSpec(
        name="DRAM → HBM",
        bw_gb_s=0.3,            # measured from profiler
        latency_us=5.0,
        description="Host DRAM to device HBM via PCIe during migration",
    ),
}

STAGE_KEYS = ["HBM_to_NPU", "DRAM_to_CPU", "SSD_to_DRAM", "DRAM_to_HBM"]


# ============================================================
# 2. Memory Model
# ============================================================

class ThreeLevelMemoryModel:
    """Latency + bandwidth model for DLRM 3-level storage.

    Model:
        T_stage = α × n_ops + DataSize / BW

    Total memory access time per forward:
        T_total = T_HBM_to_NPU + T_DRAM_to_CPU + T_SSD_to_DRAM + T_DRAM_to_HBM
    """

    def __init__(self, specs: Optional[Dict[str, PhysicalSpec]] = None):
        """
        Args:
            specs: Physical specifications per stage.
                   Uses DEFAULT_SPECS if not provided.
        """
        self.specs = specs or DEFAULT_SPECS
        # Current (possibly calibrated) parameters for each stage
        self.params: Dict[str, FittedParams] = {}
        # Calibration metadata
        self.calibration_results: Dict[str, CalibrationResult] = {}
        self._reset_to_defaults()

    def _reset_to_defaults(self):
        """Reset all parameters to physical defaults."""
        for key, spec in self.specs.items():
            self.params[key] = FittedParams(
                alpha_us=spec.latency_us,
                bw_gb_s=spec.bw_gb_s,
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_forward(
        self,
        hbm_hits: int,
        dram_hits: int,
        ssd_misses: int,
        dim: int,
        migrate_count: int = 0,
        byte_per_elem: int = 4,
    ) -> Dict[str, float]:
        """Predict memory access time for one forward pass.

        Args:
            hbm_hits:       Number of embedding indices found in HBM.
            dram_hits:      Number found in DRAM cache.
            ssd_misses:     Number requiring SSD access.
            dim:            Embedding dimension.
            migrate_count:  Number of embedding rows migrated DRAM→HBM this step.
            byte_per_elem:  Bytes per element (4 for float32, 2 for bf16/fp16).

        Returns:
            Dict mapping stage_key → time in milliseconds.
        """
        bytes_per_vec = dim * byte_per_elem

        stage_inputs = [
            ("HBM_to_NPU",  hbm_hits,      hbm_hits * bytes_per_vec),
            ("DRAM_to_CPU", dram_hits,     dram_hits * bytes_per_vec),
            ("SSD_to_DRAM", ssd_misses,    ssd_misses * bytes_per_vec),
            ("DRAM_to_HBM", migrate_count, migrate_count * bytes_per_vec),
        ]

        result = {}
        total_s = 0.0
        for key, n_ops, data_bytes in stage_inputs:
            if n_ops > 0:
                t_s = self.params[key].predict_s(n_ops, data_bytes)
                result[key] = t_s * 1000.0
                total_s += t_s
            else:
                result[key] = 0.0
        result["TOTAL"] = total_s * 1000.0
        return result

    def predict_forward_with_hit_rates(
        self,
        total_accesses: int,
        hit_rates: Dict[str, float],
        dim: int,
        hbm_size: int = 0,
        migration_ratio: float = 0.0,
        byte_per_elem: int = 4,
    ) -> Dict[str, float]:
        """Predict using hit rates (e.g., from reuse distance analysis).

        Args:
            total_accesses:   Total embedding lookups per forward.
            hit_rates:        {'HBM': 0.8, 'DRAM': 0.15, 'SSD': 0.05}
            dim:              Embedding dimension.
            hbm_size:         Number of HBM rows (for migration estimate).
            migration_ratio:  Fraction of HBM replaced per migration interval.
        """
        hbm_hits = int(total_accesses * hit_rates.get("HBM", 0.0))
        dram_hits = int(total_accesses * hit_rates.get("DRAM", 0.0))
        ssd_misses = int(total_accesses * hit_rates.get("SSD", 0.0))

        migrate_count = int(hbm_size * migration_ratio) if hbm_size > 0 else 0

        return self.predict_forward(
            hbm_hits=hbm_hits,
            dram_hits=dram_hits,
            ssd_misses=ssd_misses,
            dim=dim,
            migrate_count=migrate_count,
            byte_per_elem=byte_per_elem,
        )

    # ------------------------------------------------------------------
    # Calibration via Least-Squares
    # ------------------------------------------------------------------

    def calibrate(
        self,
        snapshots: Dict[str, List[StageSnapshot]],
        force_bw_only: bool = False,
        verbose: bool = True,
    ) -> Dict[str, CalibrationResult]:
        """Calibrate α and BW from per-stage measurement snapshots.

        Model:  T = β1 × n_ops + β2 × data_bytes
                α = β1 (converted to μs)
                BW = 1 / β2 (converted to GB/s)

        The linear regression problem:

            [ n_ops_1  data_bytes_1 ] [ β1 ]   [ T_1 ]
            [ n_ops_2  data_bytes_2 ] [ β2 ] = [ T_2 ]
            [    ...        ...      ]          [ ...  ]

        Solved via OLS. When n_ops and data_bytes are perfectly collinear
        (same dim throughout), we fix α from defaults and fit BW only.

        Args:
            snapshots:    stage_key → list of StageSnapshot (cumulative or per-interval).
            force_bw_only: Force BW-only calibration even if full fit is possible.
            verbose:      Print calibration details.

        Returns:
            Dict mapping stage_key → CalibrationResult.
        """
        self.calibration_results = {}

        for key in STAGE_KEYS:
            ss_list = snapshots.get(key, [])
            if len(ss_list) < 2:
                # Not enough data — keep defaults
                self.calibration_results[key] = CalibrationResult(
                    fitted=self.params[key],
                    r2=0.0,
                    n_samples=len(ss_list),
                    condition="insufficient_data",
                )
                if verbose:
                    print(f"  [{key}] insufficient data ({len(ss_list)} samples), "
                          f"using defaults")
                continue

            # Build design matrix and target vector
            X = np.array([[s.n_ops, s.data_bytes] for s in ss_list], dtype=np.float64)
            y = np.array([s.time_s for s in ss_list], dtype=np.float64)

            # Check collinearity: ratio data_bytes / n_ops should vary
            ratios = np.array([s.data_bytes / max(s.n_ops, 1) for s in ss_list])
            ratio_cv = float(np.std(ratios) / max(np.mean(ratios), 1e-9))  # coefficient of variation

            do_full_fit = (ratio_cv > 0.05) and not force_bw_only

            try:
                if do_full_fit:
                    # ---- Full 2-parameter OLS ----
                    beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
                    alpha_us = float(beta[0] * 1e6)
                    bw_gb_s = float(1.0 / beta[1] / 1e9) if beta[1] > 1e-12 else float("inf")
                    condition = "full"
                else:
                    # ---- BW-only: fix α from defaults, fit BW ----
                    alpha_fixed_s = self.specs[key].latency_us * 1e-6
                    # T - α_fixed × n_ops = data_bytes / BW
                    y_adjusted = y - alpha_fixed_s * X[:, 0]
                    # Simple OLS: y_adjusted = β × data_bytes  →  BW = 1/β
                    X_bw = X[:, 1].reshape(-1, 1)
                    beta_bw, _, _, _ = np.linalg.lstsq(X_bw, y_adjusted, rcond=None)
                    bw_gb_s = float(1.0 / beta_bw[0] / 1e9) if beta_bw[0] > 1e-12 else float("inf")
                    alpha_us = self.specs[key].latency_us
                    # Synthesize beta for R² computation
                    beta = np.array([alpha_us * 1e-6, beta_bw[0].item()])
                    condition = "bw_only"

                # Compute R²
                y_pred = X @ beta
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

                fitted = FittedParams(alpha_us=alpha_us, bw_gb_s=bw_gb_s)

                self.calibration_results[key] = CalibrationResult(
                    fitted=fitted,
                    r2=r2,
                    n_samples=len(ss_list),
                    condition=condition,
                )
                self.params[key] = fitted

                if verbose:
                    old = self.specs[key]
                    print(f"  [{key}] ({condition}, n={len(ss_list)}, R²={r2:.4f})")
                    print(f"      α:  {old.latency_us:.2f} → {alpha_us:.2f} μs/op")
                    print(f"      BW: {old.bw_gb_s:.1f} → {bw_gb_s:.1f} GB/s "
                          f"({'↓' if bw_gb_s < old.bw_gb_s else '↑' if bw_gb_s > old.bw_gb_s else '='} "
                          f"{bw_gb_s / old.bw_gb_s * 100:.0f}% of peak)")

            except np.linalg.LinAlgError:
                self.calibration_results[key] = CalibrationResult(
                    fitted=self.params[key],
                    r2=0.0,
                    n_samples=len(ss_list),
                    condition="insufficient_data",
                )
                if verbose:
                    print(f"  [{key}] calibration failed (linear algebra error)")

        return self.calibration_results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_table(self, predicted: Optional[Dict[str, float]] = None) -> str:
        """Return a human-readable summary table.

        Args:
            predicted: Output of predict_forward() if available.
        """
        lines = []
        sep = "=" * 82
        lines.append(sep)
        lines.append(f"  {'Stage':<16s} {'α (μs/op)':>10s} {'BW (GB/s)':>10s}  "
                     f"{'Time (ms)':>10s}  {'%Total':>8s}")
        lines.append("-" * 82)

        total_ms = predicted.get("TOTAL", 0.0) if predicted else 0.0

        for key in STAGE_KEYS:
            p = self.params[key]
            t_ms = predicted.get(key, 0.0) if predicted else 0.0
            pct = f"{t_ms / total_ms * 100:5.1f}%" if total_ms > 0 else "   N/A"
            lines.append(f"  {key:<16s} {p.alpha_us:10.2f} {p.bw_gb_s:10.1f}  "
                         f"{t_ms:10.3f}  {pct:>8s}")

        lines.append("-" * 82)
        if total_ms > 0:
            lines.append(f"  {'TOTAL':<16s} {'':>10s} {'':>10s}  "
                         f"{total_ms:10.3f}   100.0%")
        lines.append(sep)

        # Breakdown of latency vs bandwidth contribution
        if predicted:
            lines.append(f"\n  Latency vs Bandwidth breakdown:")
            lines.append(f"  {'Stage':<16s} {'T_latency':>10s} {'T_bandwidth':>10s}  "
                         f"{'Latency%':>9s}")
            lines.append("-" * 64)
            for key in STAGE_KEYS:
                if predicted.get(key, 0.0) > 0:
                    p = self.params[key]
                    # We don't have n_ops and data_bytes separately from predict_forward
                    # output alone, but we can approximate from the parameters
                    t_ms = predicted[key]
                    # Approximate: T = α·n + d/BW
                    # For display we'd need the original n_ops, data_bytes
                    # We'll compute latency share from params
                    lines.append(f"  {key:<16s} {'(see detail)':>10s}")
            lines.append("  (Use --verbose for per-op latency/bandwidth split)")

        return "\n".join(lines)

    def detailed_breakdown(
        self,
        hbm_hits: int,
        dram_hits: int,
        ssd_misses: int,
        dim: int,
        migrate_count: int = 0,
        byte_per_elem: int = 4,
    ) -> str:
        """Return a detailed breakdown including latency vs bandwidth split."""
        bytes_per_vec = dim * byte_per_elem

        stage_inputs = [
            ("HBM_to_NPU",  hbm_hits,      hbm_hits * bytes_per_vec),
            ("DRAM_to_CPU", dram_hits,     dram_hits * bytes_per_vec),
            ("SSD_to_DRAM", ssd_misses,    ssd_misses * bytes_per_vec),
            ("DRAM_to_HBM", migrate_count, migrate_count * bytes_per_vec),
        ]

        lines = []
        sep = "=" * 90
        lines.append(sep)
        lines.append(f"  Detailed Breakdown (dim={dim}, float{byte_per_elem*8})")
        lines.append(f"  {'Stage':<16s} {'n_ops':>8s} {'Data':>12s} "
                     f"{'T_lat (ms)':>11s} {'T_bw (ms)':>10s} {'T_total (ms)':>12s} "
                     f"{'BW_eff':>8s}")
        lines.append("-" * 90)

        total_ms = 0.0
        for key, n_ops, data_bytes in stage_inputs:
            if n_ops == 0:
                continue
            p = self.params[key]
            alpha_s = p.alpha_us * 1e-6
            bw_bytes_s = p.bw_gb_s * 1e9

            t_lat_s = alpha_s * n_ops
            t_bw_s = data_bytes / bw_bytes_s
            t_total_s = t_lat_s + t_bw_s
            total_ms += t_total_s * 1000

            # Effective bandwidth actually achieved (including latency effect)
            bw_eff = data_bytes / t_total_s / 1e9 if t_total_s > 0 else 0.0

            lines.append(
                f"  {key:<16s} {n_ops:>8d} {data_bytes/1e6:>9.1f} MB "
                f"{t_lat_s*1000:>11.4f} {t_bw_s*1000:>10.4f} {t_total_s*1000:>12.4f} "
                f"{bw_eff:>6.1f} GB/s"
            )

            if n_ops > 0:
                lat_pct = t_lat_s / t_total_s * 100
                lines.append(f"    → per-op: {t_total_s/n_ops*1e6:.2f} μs "
                             f"(latency {lat_pct:.0f}%, bandwidth {100-lat_pct:.0f}%)")

        lines.append("-" * 90)
        lines.append(f"  {'TOTAL':<16s} {'':>8s} {'':>12s} {'':>11s} {'':>10s} "
                     f"{total_ms:>12.4f}")
        lines.append(sep)

        return "\n".join(lines)


# ============================================================
# 3. Log File Parsing
# ============================================================

def parse_memory_log(log_path: str) -> Dict[str, List[StageSnapshot]]:
    """Parse memory_access_timings.log into per-stage snapshot lists.

    Expected log format (from ThreeLevelEmbedding._print_timing_report):
        [Metrics] step=50 HBM_hit=80.0% DRAM_hit=15.0% SSD_hit=5.0%
          HBM -> NPU cache : 0.123 ms/step, 12.5 MB total, 1200.0 GB/s
          DRAM -> CPU cache: 0.456 ms/step, 45.6 MB total, 100.0 GB/s
          SSD -> DRAM      : 0.789 ms/step, 78.9 MB total, 3.5 GB/s
          DRAM -> HBM      : 0.012 ms/step, 1.2 MB total, 64.0 GB/s
          TOTAL wall time: 1.380 ms/step

    Returns:
        Dict stage_key → list of StageSnapshot (cumulative snapshots).
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    snapshots: Dict[str, List[StageSnapshot]] = {k: [] for k in STAGE_KEYS}

    # Regex patterns
    re_header = re.compile(
        r"\[Metrics\]\s+step=(\d+)\s+"
        r"HBM_hit=([\d.]+)%\s+DRAM_hit=([\d.]+)%\s+SSD_hit=([\d.]+)%"
    )

    # Stage line: "  HBM -> NPU cache : 0.123 ms/step, 12.5 MB total, 1200.0 GB/s"
    re_stage = re.compile(
        r"^\s*(HBM -> NPU cache |DRAM -> CPU cache|SSD -> DRAM\s+|DRAM -> HBM\s+)"
        r":\s+([\d.]+)\s+ms/step,\s+([\d.]+)\s+MB total,\s+([\d.]+)\s+GB/s"
    )

    # Map log labels to stage keys
    label_to_key = {
        "HBM -> NPU cache ": "HBM_to_NPU",
        "DRAM -> CPU cache": "DRAM_to_CPU",
        "SSD -> DRAM      ": "SSD_to_DRAM",
        "DRAM -> HBM      ": "DRAM_to_HBM",
    }

    current_step = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            # Try header match
            hm = re_header.search(line)
            if hm:
                current_step = int(hm.group(1))
                continue

            # Try stage match
            sm = re_stage.match(line)
            if sm:
                label = sm.group(1)
                ms_per_step = float(sm.group(2))
                mb_total = float(sm.group(3))
                # gb_s = float(sm.group(4))  # reported BW (computed by the code)

                key = label_to_key.get(label)
                if key:
                    time_s = ms_per_step * 1e-3 * current_step  # cumulative time
                    snapshots[key].append(StageSnapshot(
                        step=current_step,
                        n_ops=0,          # n_ops not directly in log; will be inferred
                        data_bytes=mb_total * 1e6,
                        time_s=time_s,
                    ))

    return snapshots


# ============================================================
# 4. Integration with Reuse Distance Analyzer
# ============================================================

def load_reuse_distance_result(json_path: str) -> dict:
    """Load a JSON file produced by reuse_distance.py."""
    with open(json_path, "r") as f:
        return json.load(f)


def predict_from_reuse_result(
    model: ThreeLevelMemoryModel,
    reuse_result: dict,
    total_accesses_per_forward: int = 0,
    dim: int = 64000,
    byte_per_elem: int = 4,
    migration_ratio: float = 0.01,
) -> Dict[str, float]:
    """Predict per-forward memory time from reuse distance analysis results.

    The key insight: reuse distance analysis gives hit rate PROPORTIONS
    (e.g. HBM 80%, DRAM 15%, SSD 5%), which reflect the access pattern.
    The absolute data volume depends on how many lookups happen per forward:

        n_ops_stage = total_accesses_per_forward × hit_rate_stage
        DataSize    = n_ops_stage × dim × byte_per_elem

    Args:
        model:                     Calibrated (or default) memory model.
        reuse_result:              JSON dict from reuse_distance.py --output-json.
                                   Keys used: 'hit_rates', 'tier_distribution',
                                   'total_accesses' (auto-detected if K not given).
        total_accesses_per_forward: Total embedding lookups per forward pass.
                                    If 0, auto-reads from reuse_result['total_accesses'].
        dim:                       Embedding dimension.
        byte_per_elem:             4 for float32, 2 for bf16/fp16.
        migration_ratio:           Fraction of HBM rows migrated per interval.

    Returns:
        Dict stage_key → time in ms. Includes 'TOTAL'.
    """
    hit_rates = reuse_result.get("hit_rates", {})
    tier_dist = reuse_result.get("tier_distribution", {})

    if total_accesses_per_forward > 0:
        K = total_accesses_per_forward
    else:
        K = reuse_result.get("total_accesses", 0)
        if K <= 0:
            raise ValueError(
                "total_accesses not found in reuse_result JSON and "
                "--total-accesses not specified. "
                "Use --total-accesses or ensure the JSON was produced by "
                "reuse_distance.py (which writes 'total_accesses')."
            )

    hbm_hits   = int(K * hit_rates.get("HBM", 0.0))
    dram_hits  = int(K * hit_rates.get("DRAM", 0.0))
    ssd_misses = int(K * hit_rates.get("SSD", 0.0))

    # Migration: roughly migration_ratio of HBM rows get replaced each interval
    hbm_size = tier_dist.get("HBM", 0)
    migrate_count = int(hbm_size * migration_ratio) if hbm_size and migration_ratio > 0 else 0

    return model.predict_forward(
        hbm_hits=hbm_hits,
        dram_hits=dram_hits,
        ssd_misses=ssd_misses,
        dim=dim,
        migrate_count=migrate_count,
        byte_per_elem=byte_per_elem,
    )


# ============================================================
# 5. Synthetic Data Generation (for testing / demonstration)
# ============================================================

def generate_synthetic_snapshots(
    n_points: int = 10,
    dim: int = 64000,
    byte_per_elem: int = 4,
    access_range: Tuple[int, int] = (100, 2000),
    seed: int = 42,
) -> Dict[str, List[StageSnapshot]]:
    """Generate synthetic calibration data with noise.

    This simulates what real log data would look like for testing the
    calibration routine. Uses DEFAULT_SPECS to generate "true" times
    with added Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    bytes_per_vec = dim * byte_per_elem

    snapshots: Dict[str, List[StageSnapshot]] = {k: [] for k in STAGE_KEYS}

    # Hit rate assumptions for generating realistic access counts
    # HBM: 80-90%, DRAM: 8-15%, SSD: 2-5%
    for step in range(1, n_points + 1):
        total_access = int(rng.integers(*access_range))
        hbm_hits = int(total_access * rng.uniform(0.80, 0.90))
        dram_hits = int(total_access * rng.uniform(0.08, 0.15))
        ssd_misses = total_access - hbm_hits - dram_hits
        migrate_count = int(rng.integers(5, 30))

        stage_counts = [
            ("HBM_to_NPU",  hbm_hits),
            ("DRAM_to_CPU", dram_hits),
            ("SSD_to_DRAM", ssd_misses),
            ("DRAM_to_HBM", migrate_count),
        ]

        for key, n_ops in stage_counts:
            spec = DEFAULT_SPECS[key]
            # True time from the model
            alpha_s = spec.latency_us * 1e-6
            bw_bytes_s = spec.bw_gb_s * 1e9
            true_time = alpha_s * n_ops + (n_ops * bytes_per_vec) / bw_bytes_s

            # Add 5% Gaussian noise to simulate measurement error
            noisy_time = true_time * (1.0 + rng.normal(0.0, 0.05))

            snapshots[key].append(StageSnapshot(
                step=step,
                n_ops=n_ops,
                data_bytes=n_ops * bytes_per_vec,
                time_s=noisy_time,
            ))

    return snapshots


# ============================================================
# 6. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Memory Access Time Model for DLRM 3-Level Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict with direct access counts
  python memory_model.py --hbm-hits 800 --dram-hits 150 --ssd-misses 50 --dim 64000

  # Calibrate from log and predict
  python memory_model.py --calibrate memory_access_timings.log --hbm-hits 800 --dram-hits 150 --ssd-misses 50 --dim 64000

  # Predict from reuse distance analysis
  python memory_model.py --from-reuse reuse_result.json --total-accesses 1000 --dim 64000

  # Full pipeline: reuse distance → prediction with breakdown
  python memory_model.py --from-reuse reuse_result.json --total-accesses 3200 --dim 64000 --breakdown

  # Test calibration with synthetic data
  python memory_model.py --synthetic
        """,
    )

    # Prediction inputs
    pred_group = parser.add_argument_group("Prediction inputs")
    pred_group.add_argument("--hbm-hits", type=int, default=0,
                            help="Number of HBM hits per forward")
    pred_group.add_argument("--dram-hits", type=int, default=0,
                            help="Number of DRAM hits per forward")
    pred_group.add_argument("--ssd-misses", type=int, default=0,
                            help="Number of SSD misses per forward")
    pred_group.add_argument("--dim", type=int, default=64000,
                            help="Embedding dimension")
    pred_group.add_argument("--migrate-count", type=int, default=0,
                            help="Number of rows migrated DRAM→HBM")
    pred_group.add_argument("--dtype", choices=["float32", "float16", "bf16"],
                            default="float32", help="Element data type")

    # Calibration
    calib_group = parser.add_argument_group("Calibration")
    calib_group.add_argument("--calibrate", type=str, default=None, metavar="LOG",
                             help="Path to memory_access_timings.log for calibration")
    calib_group.add_argument("--force-bw-only", action="store_true",
                             help="Force BW-only calibration (skip full 2-param fit)")

    # From reuse distance
    reuse_group = parser.add_argument_group("Reuse distance integration")
    reuse_group.add_argument("--from-reuse", type=str, default=None, metavar="JSON",
                             help="Load reuse distance analysis result (JSON)")
    reuse_group.add_argument("--total-accesses", type=int, default=0,
                             help="Total embedding lookups per forward "
                                  "(= batch_size × lookups_per_sample × num_tables). "
                                  "Optional: if 0, auto-reads 'total_accesses' from "
                                  "the reuse_result JSON.")
    reuse_group.add_argument("--migration-ratio", type=float, default=0.01,
                             help="Fraction of HBM rows migrated per interval (default 1%%)")

    # Display
    parser.add_argument("--breakdown", action="store_true",
                        help="Show detailed latency vs bandwidth breakdown")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run calibration test with synthetic data")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    byte_per_elem = {"float32": 4, "float16": 2, "bf16": 2}[args.dtype]

    # ---- Create model ----
    model = ThreeLevelMemoryModel()

    # ---- Calibration ----
    if args.calibrate:
        if not args.quiet:
            print(f"\n[Calibration] Parsing log: {args.calibrate}")
        snapshots = parse_memory_log(args.calibrate)
        total_snapshots = sum(len(v) for v in snapshots.values())

        if total_snapshots == 0:
            print(f"  Warning: No valid snapshots found in log file. "
                  f"Using default parameters.")
        else:
            if not args.quiet:
                print(f"  Found {total_snapshots} snapshots across "
                      f"{sum(1 for v in snapshots.values() if v)} stages")
                print()
            model.calibrate(snapshots, force_bw_only=args.force_bw_only,
                           verbose=not args.quiet)
            if not args.quiet:
                print()

    # ---- Synthetic test ----
    if args.synthetic:
        print("[Synthetic] Generating calibration data with known ground truth...\n")
        snapshots = generate_synthetic_snapshots(
            n_points=10,
            dim=args.dim,
            byte_per_elem=byte_per_elem,
        )
        if not args.quiet:
            model.calibrate(snapshots, force_bw_only=args.force_bw_only,
                           verbose=True)
            print("\n[Synthetic] Ground truth vs fitted comparison:\n")

        # Compare fitted vs true
        for key in STAGE_KEYS:
            true = DEFAULT_SPECS[key]
            fitted = model.params[key]
            alpha_err = abs(fitted.alpha_us - true.latency_us) / max(true.latency_us, 1e-9) * 100
            bw_err = abs(fitted.bw_gb_s - true.bw_gb_s) / max(true.bw_gb_s, 1e-9) * 100
            print(f"  {key}:")
            print(f"    α:  {fitted.alpha_us:.2f} vs {true.latency_us:.2f} μs "
                  f"(error {alpha_err:.1f}%)")
            print(f"    BW: {fitted.bw_gb_s:.1f} vs {true.bw_gb_s:.1f} GB/s "
                  f"(error {bw_err:.1f}%)")

    # ---- Prediction ----
    if args.from_reuse:
        reuse_result = load_reuse_distance_result(args.from_reuse)
        predicted = predict_from_reuse_result(
            model, reuse_result,
            total_accesses_per_forward=args.total_accesses,
            dim=args.dim,
            byte_per_elem=byte_per_elem,
            migration_ratio=args.migration_ratio,
        )
        # Store computed n_ops for breakdown display
        hit_rates = reuse_result.get("hit_rates", {})
        tier_dist = reuse_result.get("tier_distribution", {})
        K = args.total_accesses
        _hbm_hits   = int(K * hit_rates.get("HBM", 0.0))
        _dram_hits  = int(K * hit_rates.get("DRAM", 0.0))
        _ssd_misses = int(K * hit_rates.get("SSD", 0.0))
        hbm_size = tier_dist.get("HBM", 0)
        _migrate_count = int(hbm_size * args.migration_ratio) if hbm_size else 0
    elif args.hbm_hits > 0 or args.dram_hits > 0 or args.ssd_misses > 0:
        _hbm_hits   = args.hbm_hits
        _dram_hits  = args.dram_hits
        _ssd_misses = args.ssd_misses
        _migrate_count = args.migrate_count
        predicted = model.predict_forward(
            hbm_hits=_hbm_hits,
            dram_hits=_dram_hits,
            ssd_misses=_ssd_misses,
            dim=args.dim,
            migrate_count=_migrate_count,
            byte_per_elem=byte_per_elem,
        )
    else:
        # No prediction inputs — just show parameter table
        predicted = None
        _hbm_hits = _dram_hits = _ssd_misses = _migrate_count = 0

    if predicted:
        if not args.quiet:
            print()
        print(f"[Prediction] Per-forward memory access time "
              f"(dim={args.dim}, dtype={args.dtype})\n")
        print(model.summary_table(predicted))

        if args.breakdown:
            print()
            print(model.detailed_breakdown(
                hbm_hits=_hbm_hits,
                dram_hits=_dram_hits,
                ssd_misses=_ssd_misses,
                dim=args.dim,
                migrate_count=_migrate_count,
                byte_per_elem=byte_per_elem,
            ))

        # Key insight
        if _hbm_hits > 0 or _dram_hits > 0:
            total_ms = predicted["TOTAL"]
            total_accesses = (_hbm_hits + _dram_hits +
                             _ssd_misses + _migrate_count)
            if total_accesses > 0:
                avg_us_per_access = total_ms * 1000 / total_accesses
                print(f"\n  Average memory access latency: "
                      f"{avg_us_per_access:.2f} μs per embedding lookup")

                # Latency contribution estimate
                total_lat_ms = 0.0
                total_data_gb = 0.0
                bytes_per_vec = args.dim * byte_per_elem
                for key, n_ops in [
                    ("HBM_to_NPU", _hbm_hits),
                    ("DRAM_to_CPU", _dram_hits),
                    ("SSD_to_DRAM", _ssd_misses),
                    ("DRAM_to_HBM", _migrate_count),
                ]:
                    if n_ops > 0:
                        total_lat_ms += model.params[key].alpha_us * 1e-3 * n_ops
                        total_data_gb += n_ops * bytes_per_vec / 1e9
                if predicted["TOTAL"] > 0:
                    lat_share = total_lat_ms / predicted["TOTAL"] * 100
                    print(f"  Latency contribution: {total_lat_ms:.3f} ms "
                          f"({lat_share:.0f}% of total)")
                    print(f"  Bandwidth contribution: "
                          f"{predicted['TOTAL'] - total_lat_ms:.3f} ms "
                          f"({100 - lat_share:.0f}% of total)")
                    print(f"  Total data moved: {total_data_gb:.3f} GB")

    elif not args.synthetic and not args.calibrate:
        parser.print_help()
        print("\nTip: Use --synthetic to test calibration or provide prediction inputs.")


if __name__ == "__main__":
    main()
