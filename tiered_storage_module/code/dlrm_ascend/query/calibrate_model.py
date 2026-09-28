#!/usr/bin/env python3
"""
Calibrate memory model α and BW from measured times + reuse distance predictions.

Usage:
    # First run reuse_distance for all configs:
    for N in 1000 5000 10000 50000 100000 500000 1000000; do
        python reuse_distance.py --num-embeddings $N --trace-length 500000 \
            --th-hbm 1000 --th-dram 10000 --output-json reuse_n${N}.json --seed 42
    done

    # Then calibrate:
    python calibrate_model.py
"""

import json
import sys
import numpy as np

# ============================================================
# 1. Your measured data (from profiling)
# ============================================================
# Format: each row = (num_embeddings, dim, time_SSD_to_DRAM_s, time_DRAM_to_CPU_s,
#                      time_DRAM_to_HBM_s, time_HBM_to_NPU_s)

MEASURED = [
    # N       dim    SSD→DRAM(s)  DRAM→CPU(s)  DRAM→HBM(s)  HBM→NPU(s)
    (1000,    64000, 0.1923,      0.000165,    0.000015,    0.000529),
    (5000,    64000, 0.8598,      0.000189,    0.000020,    0.000547),
    (10000,   64000, 1.6313,      0.000198,    0.000021,    0.000550),
    (50000,   64000, 3.6967,      0.000235,    0.000023,    0.000542),
    (100000,  64000, 4.5805,      0.000234,    0.000025,    0.000583),
    (500000,  64000, 4.8333,      0.000317,    0.000030,    0.000565),
    (1000000, 64000, 4.9065,      0.000167,    0.000026,    0.000620),
]

STAGE_KEYS = ["SSD_to_DRAM", "DRAM_to_CPU", "DRAM_to_HBM", "HBM_to_NPU"]
TOTAL_ACCESSES = 25600  # per forward
BYTE_PER_ELEM = 4       # float32


# ============================================================
# 2. Load reuse distance results and compute n_ops per stage
# ============================================================

def load_reuse_results():
    """Load all reuse distance JSON files, return dict of N -> hit_rates + tier_dist."""
    results = {}
    for n, dim, *_ in MEASURED:
        path = f"reuse_n{n}.json"
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: {path} not found. Run reuse_distance.py first.")
            sys.exit(1)
        results[n] = data
    return results


def compute_n_ops(reuse_data, total_accesses=TOTAL_ACCESSES):
    """Convert hit rates to absolute n_ops per stage."""
    hr = reuse_data["hit_rates"]
    td = reuse_data["tier_distribution"]
    return {
        "HBM_to_NPU":  int(total_accesses * hr["HBM"]),
        "DRAM_to_CPU": int(total_accesses * hr["DRAM"]),
        "SSD_to_DRAM": int(total_accesses * hr["SSD"]),
        "DRAM_to_HBM": int(td["HBM"] * 0.01) if td["HBM"] else 0,
    }


# ============================================================
# 3. OLS calibration: solve T = α × n_ops + data_bytes / BW
# ============================================================

def calibrate_stage(n_ops_list, data_bytes_list, time_s_list, stage_name):
    """
    Fit: T = α × n_ops + data_bytes / BW

    Design matrix: X = [n_ops, data_bytes], target: y = T
    β₁ = α,  β₂ = 1/BW

    If n_ops and data_bytes are perfectly collinear (same dim throughout),
    fall back to BW-only fit with α from known hardware specs.
    """
    if len(time_s_list) < 2:
        return None, "insufficient data"

    X = np.array([[n_ops_list[i], data_bytes_list[i]]
                   for i in range(len(n_ops_list))], dtype=np.float64)
    y = np.array(time_s_list, dtype=np.float64)

    # Check collinearity
    ratios = [data_bytes_list[i] / max(n_ops_list[i], 1)
              for i in range(len(n_ops_list))]
    ratio_cv = float(np.std(ratios) / max(np.mean(ratios), 1e-9))

    if ratio_cv > 0.05:
        # Full 2-parameter fit
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha_us = float(beta[0] * 1e6)
        bw_gb_s = float(1.0 / beta[1] / 1e9) if abs(beta[1]) > 1e-12 else float("inf")
        condition = "full (α + BW)"
    else:
        # BW-only: fix α at hardware estimate
        alpha_fixed = {"SSD_to_DRAM": 10.0, "DRAM_to_CPU": 1.0,
                       "DRAM_to_HBM": 5.0, "HBM_to_NPU": 0.1}
        alpha_fixed_s = alpha_fixed.get(stage_name, 1.0) * 1e-6
        y_adj = y - alpha_fixed_s * X[:, 0]
        X_bw = X[:, 1].reshape(-1, 1)
        beta_bw, _, _, _ = np.linalg.lstsq(X_bw, y_adj, rcond=None)
        bw_gb_s = float(1.0 / beta_bw[0] / 1e9) if abs(beta_bw[0]) > 1e-12 else float("inf")
        alpha_us = alpha_fixed.get(stage_name, 1.0)
        beta = np.array([alpha_us * 1e-6, beta_bw[0].item()])
        condition = "BW-only (α fixed)"

    # R²
    y_pred = X @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "alpha_us": alpha_us,
        "bw_gb_s": bw_gb_s,
        "condition": condition,
        "r2": r2,
        "n_points": len(time_s_list),
        "residuals_pct": [float((y[i] - y_pred[i]) / max(y[i], 1e-9) * 100)
                          for i in range(len(y))],
    }, None


# ============================================================
# 4. Main
# ============================================================

def main():
    print("=" * 85)
    print("  Memory Model Calibration: Measured Times × Reuse Distance")
    print("=" * 85)

    reuse_results = load_reuse_results()

    # Gather per-stage data
    stage_data = {k: {"n_ops": [], "data_bytes": [], "times_s": [], "labels": []}
                  for k in STAGE_KEYS}

    for n, dim, t_ssd, t_dram, t_hbm_mig, t_hbm in MEASURED:
        rd = reuse_results[n]
        ops = compute_n_ops(rd)
        label = f"N={n:,}"

        stage_data["SSD_to_DRAM"]["n_ops"].append(ops["SSD_to_DRAM"])
        stage_data["SSD_to_DRAM"]["data_bytes"].append(ops["SSD_to_DRAM"] * dim * BYTE_PER_ELEM)
        stage_data["SSD_to_DRAM"]["times_s"].append(t_ssd)
        stage_data["SSD_to_DRAM"]["labels"].append(label)

        stage_data["DRAM_to_CPU"]["n_ops"].append(ops["DRAM_to_CPU"])
        stage_data["DRAM_to_CPU"]["data_bytes"].append(ops["DRAM_to_CPU"] * dim * BYTE_PER_ELEM)
        stage_data["DRAM_to_CPU"]["times_s"].append(t_dram)
        stage_data["DRAM_to_CPU"]["labels"].append(label)

        stage_data["DRAM_to_HBM"]["n_ops"].append(ops["DRAM_to_HBM"])
        stage_data["DRAM_to_HBM"]["data_bytes"].append(ops["DRAM_to_HBM"] * dim * BYTE_PER_ELEM)
        stage_data["DRAM_to_HBM"]["times_s"].append(t_hbm_mig)
        stage_data["DRAM_to_HBM"]["labels"].append(label)

        stage_data["HBM_to_NPU"]["n_ops"].append(ops["HBM_to_NPU"])
        stage_data["HBM_to_NPU"]["data_bytes"].append(ops["HBM_to_NPU"] * dim * BYTE_PER_ELEM)
        stage_data["HBM_to_NPU"]["times_s"].append(t_hbm)
        stage_data["HBM_to_NPU"]["labels"].append(label)

    # ---- Calibrate each stage ----
    print("\n[Calibration Results]\n")
    calibrated = {}

    for stage in STAGE_KEYS:
        sd = stage_data[stage]
        # Filter out zero n_ops points
        valid = [(n, d, t, l) for n, d, t, l in
                 zip(sd["n_ops"], sd["data_bytes"], sd["times_s"], sd["labels"])
                 if n > 0 and t > 0]
        if not valid:
            print(f"  {stage}: no valid data points")
            continue

        n_ops_list = [v[0] for v in valid]
        data_list = [v[1] for v in valid]
        time_list = [v[2] for v in valid]
        labels = [v[3] for v in valid]

        result, err = calibrate_stage(n_ops_list, data_list, time_list, stage)
        if result is None:
            print(f"  {stage}: {err}")
            continue

        calibrated[stage] = result
        print(f"  [{stage}]  ({result['condition']}, {result['n_points']} points, R²={result['r2']:.4f})")
        print(f"    α  = {result['alpha_us']:.3f} μs/op")
        print(f"    BW = {result['bw_gb_s']:.1f} GB/s")
        print(f"    {'Point':<12s} {'n_ops':>8s} {'Data':>10s} {'Meas':>8s} {'Pred':>8s} {'Res':>8s}")
        for i in range(len(time_list)):
            pred_s = result["alpha_us"] * 1e-6 * n_ops_list[i] + \
                     data_list[i] / (result["bw_gb_s"] * 1e9)
            print(f"    {labels[i]:<12s} {n_ops_list[i]:>8,} "
                  f"{data_list[i]/1e6:>7.1f} MB "
                  f"{time_list[i]*1000:>7.3f}ms "
                  f"{pred_s*1000:>7.3f}ms "
                  f"{result['residuals_pct'][i]:>+6.1f}%")
        print()

    # ---- Prediction vs Measured comparison table ----
    print("=" * 85)
    print("  Before vs After Calibration: Predicted TOTAL vs Measured TOTAL")
    print("=" * 85)
    print(f"  {'Config':<14s} {'HBM%':>6s} {'DRAM%':>6s} {'SSD%':>6s} "
          f"{'Default(ms)':>12s} {'Calib(ms)':>12s} {'Meas(ms)':>10s} {'Calib err':>10s}")
    print(f"  {'-'*78}")

    for n, dim, t_ssd, t_dram, t_hbm_mig, t_hbm in MEASURED:
        rd = reuse_results[n]
        hr = rd["hit_rates"]
        ops = compute_n_ops(rd)
        meas_total_s = t_ssd + t_dram + t_hbm_mig + t_hbm

        # Default model prediction
        default_params = {
            "HBM_to_NPU":  (0.1, 1200.0),
            "DRAM_to_CPU": (1.0, 100.0),
            "SSD_to_DRAM": (10.0, 3.5),
            "DRAM_to_HBM": (5.0, 64.0),
        }
        default_total_s = 0.0
        calib_total_s = 0.0
        for stage in STAGE_KEYS:
            n_ops = ops[stage]
            if n_ops > 0:
                d_bytes = n_ops * dim * BYTE_PER_ELEM
                # Default
                a_def, bw_def = default_params[stage]
                default_total_s += a_def * 1e-6 * n_ops + d_bytes / (bw_def * 1e9)
                # Calibrated
                if stage in calibrated:
                    c = calibrated[stage]
                    calib_total_s += c["alpha_us"] * 1e-6 * n_ops + d_bytes / (c["bw_gb_s"] * 1e9)

        default_total_ms = default_total_s * 1000
        calib_total_ms = calib_total_s * 1000
        meas_total_ms = meas_total_s * 1000
        calib_err = (calib_total_ms - meas_total_ms) / max(meas_total_ms, 1e-9) * 100

        print(f"  {f'N={n:,}':<14s} {hr['HBM']:5.1%} {hr['DRAM']:5.1%} {hr['SSD']:5.1%} "
              f"{default_total_ms:>12.2f} {calib_total_ms:>12.2f} {meas_total_ms:>10.3f} "
              f"{calib_err:>+9.1f}%")

    print()

    # ---- Summary: calibrated parameters to copy into DEFAULT_SPECS ----
    print("=" * 85)
    print("  Calibrated Parameters (copy into memory_model.py DEFAULT_SPECS)")
    print("=" * 85)
    for stage in STAGE_KEYS:
        if stage in calibrated:
            c = calibrated[stage]
            print(f'  "{stage}": PhysicalSpec(name="...", bw_gb_s={c["bw_gb_s"]:.1f}, '
                  f'latency_us={c["alpha_us"]:.3f}),')
    print()


if __name__ == "__main__":
    main()
