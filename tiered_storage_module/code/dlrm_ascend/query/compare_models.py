#!/usr/bin/env python3
"""
Compare current α+BW model against 13 measured data points.
Also fit pure-bandwidth model (T = DataSize / BW) for reference.
"""

import json
import sys
import subprocess
import numpy as np

# ============================================================
# 1. Measured data
# ============================================================
sizes_gb = np.array([2, 10.24, 20.48, 40.96, 61.44, 102.4, 143.36, 204.8,
                     409.6, 614.4, 1024, 1536, 2048])
times_meas_s = np.array([
    0.057566684484,
    0.061558535695,
    0.066510388255,
    0.111510388255,
    0.116510388255,
    0.121510388255,
    0.126510388255,
    0.136510388255,
    4.7510388255,
    4.78510388255,
    4.895520782471,
    4.925520782471,
    4.945520782471,
])

DIM = 64000
N_TABLES = 8
BYTES_PER_ROW = DIM * 4  # float32
# Total rows per table: total_gb / N_TABLES / bytes_per_row * 1024^3
rows_per_table = (sizes_gb * 1024**3 / N_TABLES / BYTES_PER_ROW).astype(int)

print(f"{'Size(GB)':>10s} {'Rows/table':>12s} {'Meas(s)':>10s}")
print("-" * 36)
for i in range(len(sizes_gb)):
    print(f"{sizes_gb[i]:>10.1f} {rows_per_table[i]:>12,} {times_meas_s[i]:>10.3f}")

# ============================================================
# 2. Pure bandwidth model: T = DataSize / BW  (previous approach)
# ============================================================

def fit_pure_bw(size_gb, time_s, bw_gb_s):
    """Predict time using pure bandwidth model."""
    return size_gb / bw_gb_s

# Fit BW separately for "low" and "high" regimes
mask_low = sizes_gb <= 204.8
mask_high = sizes_gb >= 409.6

# For low regime: linear fit T = size / BW + overhead
X_low = np.column_stack([sizes_gb[mask_low], np.ones(mask_low.sum())])
y_low = times_meas_s[mask_low]
beta_low, _, _, _ = np.linalg.lstsq(X_low, y_low, rcond=None)
bw_low = 1.0 / beta_low[0] if beta_low[0] > 0 else float('inf')  # GB/s
overhead_low = beta_low[1]  # seconds

# For high regime: SSD-dominated, flat
bw_ssd = np.mean(sizes_gb[mask_high]) / np.mean(times_meas_s[mask_high])

print(f"\n[Pure BW model]")
print(f"  Low regime  (≤204.8 GB): BW={bw_low:.0f} GB/s, overhead={overhead_low*1000:.1f} ms")
print(f"  High regime (≥409.6 GB): BW={bw_ssd:.0f} GB/s (SSD dominated)")

# Predict
pred_bw = np.zeros(len(sizes_gb))
pred_bw[mask_low] = sizes_gb[mask_low] / bw_low + overhead_low
pred_bw[mask_high] = sizes_gb[mask_high] / bw_ssd

# ============================================================
# 3. Current α + d/BW model with reuse distance
# ============================================================
from memory_model import ThreeLevelMemoryModel, DEFAULT_SPECS

# Run reuse_distance for each config (capacity-based tier assignment)
# HBM=1%, DRAM=5% of rows (matching default ThreeLevelEmbedding)
HBM_RATIO = 0.01
DRAM_RATIO = 0.05
TOTAL_ACCESSES = 25600  # per forward

pred_ab = np.zeros(len(sizes_gb))

print(f"\n[α + d/BW model] Running reuse_distance for each config...")
for i, (size_gb, n_rows) in enumerate(zip(sizes_gb, rows_per_table)):
    # 容量模式: 按实际 HBM/DRAM 容量百分比分配行数
    hbm_rows = max(1, int(n_rows * HBM_RATIO))
    dram_rows = max(1, int(n_rows * DRAM_RATIO))

    json_path = f"reuse_cap_{n_rows}.json"

    cmd = [
        sys.executable, "reuse_distance.py",
        "--num-embeddings", str(n_rows),
        "--trace-length", str(min(500000, max(50000, n_rows * 10))),
        "--hbm-rows", str(hbm_rows),
        "--dram-rows", str(dram_rows),
        "--output-json", json_path,
        "--seed", "42",
    ]
    subprocess.run(cmd, capture_output=True)

    with open(json_path) as f:
        result = json.load(f)

    hit_rates = result["hit_rates"]
    tier_dist = result["tier_distribution"]

    K = TOTAL_ACCESSES
    hbm_hits   = int(K * hit_rates["HBM"])
    dram_hits  = int(K * hit_rates["DRAM"])
    ssd_misses = int(K * hit_rates["SSD"])
    migrate_count = int(tier_dist.get("HBM", 0) * 0.01) if tier_dist.get("HBM", 0) else 0

    model = ThreeLevelMemoryModel()
    pred = model.predict_forward(
        hbm_hits=hbm_hits,
        dram_hits=dram_hits,
        ssd_misses=ssd_misses,
        dim=DIM,
        migrate_count=migrate_count,
    )
    pred_ab[i] = pred["TOTAL"] / 1000  # ms → s

    print(f"  {size_gb:>6.1f} GB  N={n_rows:>8,}  "
          f"hbm={hbm_rows:>6,} dram={dram_rows:>7,}  "
          f"HBM={hit_rates['HBM']:.1%} DRAM={hit_rates['DRAM']:.1%} SSD={hit_rates['SSD']:.1%}  "
          f"pred={pred_ab[i]:.3f}s  meas={times_meas_s[i]:.3f}s")

# ============================================================
# 4. Comparison table
# ============================================================
print(f"\n{'='*90}")
print(f"  Model Comparison: Measured vs Pure-BW vs α+BW")
print(f"{'='*90}")
print(f"  {'Size(GB)':>10s} {'N/table':>10s} {'Meas(s)':>10s} "
      f"{'BW pred':>10s} {'BW err':>8s} {'α+BW pred':>10s} {'α+BW err':>8s}")
print(f"  {'-'*72}")

for i in range(len(sizes_gb)):
    bw_err = (pred_bw[i] - times_meas_s[i]) / times_meas_s[i] * 100
    ab_err = (pred_ab[i] - times_meas_s[i]) / max(times_meas_s[i], 1e-9) * 100
    print(f"  {sizes_gb[i]:>10.1f} {rows_per_table[i]:>10,} {times_meas_s[i]:>10.4f} "
          f"{pred_bw[i]:>10.4f} {bw_err:>+7.1f}% {pred_ab[i]:>10.4f} {ab_err:>+7.1f}%")

# MAPE
mape_bw = np.mean(np.abs((pred_bw - times_meas_s) / times_meas_s)) * 100
mape_ab = np.mean(np.abs((pred_ab - times_meas_s) / np.maximum(times_meas_s, 0.001))) * 100
print(f"  {'-'*72}")
print(f"  {'MAPE':>10s} {'':>10s} {'':>10s} {'':>10s} {mape_bw:>7.1f}% {'':>10s} {mape_ab:>7.1f}%")

# ============================================================
# 5. Calibrated model
# ============================================================
print(f"\n[Calibrating SSD BW from high-regime data]")
# Use the high-regime data (≥409.6 GB) to calibrate SSD BW
# In α+BW model, total time ≈ T_ssd for large configs
# T_ssd = α_ssd × n_ssd + data_ssd / BW_ssd
# For large configs, n_ssd ≈ K × SSD_hit_rate, data_ssd = n_ssd × DIM × 4

# Get SSD hit information from the largest config
largest_n = rows_per_table[-1]
with open(f"reuse_cap_{largest_n}.json") as f:
    largest_result = json.load(f)

# Direct BW calibration from high regime
ssd_hit_rate = largest_result["hit_rates"]["SSD"]
ssd_n_ops = int(TOTAL_ACCESSES * ssd_hit_rate)
ssd_data = ssd_n_ops * DIM * 4
avg_high_time = np.mean(times_meas_s[mask_high])  # ~4.85s
bw_cal_ssd = ssd_data / (avg_high_time - 10e-6 * ssd_n_ops) / 1e9  # subtract α term

# Or simpler: fit T = α·n + d/BW for SSD directly using high-regime data
# For pure SSD-dominated regime, T ≈ n_ssd × α + d_ssd / BW
ssd_n_ops_arr = np.zeros(mask_high.sum())
ssd_data_arr = np.zeros(mask_high.sum())
for j, idx in enumerate(np.where(mask_high)[0]):
    n_rows = rows_per_table[idx]
    with open(f"reuse_cap_{n_rows}.json") as f:
        r = json.load(f)
    ssd_n_ops_arr[j] = int(TOTAL_ACCESSES * r["hit_rates"]["SSD"])
    ssd_data_arr[j] = ssd_n_ops_arr[j] * DIM * 4

X_ssd = np.column_stack([ssd_n_ops_arr, ssd_data_arr])
y_ssd = times_meas_s[mask_high]
try:
    beta_ssd, _, _, _ = np.linalg.lstsq(X_ssd, y_ssd, rcond=None)
    alpha_cal_ssd = beta_ssd[0] * 1e6  # μs
    bw_cal_ssd = 1.0 / beta_ssd[1] / 1e9 if beta_ssd[1] > 0 else float('inf')
    print(f"  Calibrated SSD: α={alpha_cal_ssd:.2f} μs, BW={bw_cal_ssd:.2f} GB/s")
except:
    print(f"  SSD calibration failed")

print(f"\nDone. Run 'python plot_compare.py' to generate comparison plots.")
