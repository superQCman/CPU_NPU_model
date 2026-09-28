#!/usr/bin/env python3
"""SSD physical model vs fio measurements — two overlay figures (curve + dots).

Fig B: bandwidth vs queue depth (bs = 4 KB)
Fig C: latency vs block size (QD = 1)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ssd_model import (
    BW_MAX_BPS, IOPS_MAX,
    FIO_BS_SWEEP, FIO_QD_SWEEP, latency_us, effective_bw_bps,
)

# ---- palette (light) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_MEAS = "#2a78d6"    # measured (blue)
C_MODEL = "#eb6834"   # model (orange)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_ax(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(BASELINE)
    return ax


# ============ Fig B: bandwidth vs queue depth (bs = 4 KB) ============
qd_data = np.array([d[1] for d in FIO_QD_SWEEP], dtype=float)
bw_meas = np.array([d[4] for d in FIO_QD_SWEEP], dtype=float)

qd_grid = np.logspace(0, np.log10(512), 300)
bw_model = np.array([effective_bw_bps(4096, q) for q in qd_grid]) / (1024 ** 2)

fig1, ax1 = plt.subplots(figsize=(7.2, 5.2))

ax1.scatter(qd_data, bw_meas, s=60, facecolor=C_MEAS, edgecolor=SURFACE,
            linewidth=1.8, zorder=3, label="Measured")
ax1.plot(qd_grid, bw_model, color=C_MODEL, linewidth=2, zorder=2, label="Model")

sat = min(BW_MAX_BPS, IOPS_MAX * 4096) / (1024 ** 2)
ax1.axhline(sat, color=MUTED, linewidth=0.9, linestyle=(0, (4, 4)), zorder=1)
ax1.text(40, sat * 1.12, f"Saturation ≈ {sat:.0f} MiB/s", color=MUTED, fontsize=9)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(1, 512)
ax1.set_ylim(30, 10000)
ax1.set_xlabel("Queue depth QD")
ax1.set_ylabel("Bandwidth (MiB/s)")
ax1.set_title("Bandwidth vs queue depth (bs = 4 KB)", color=INK, fontweight="semibold")
style_ax(ax1)
ax1.legend(frameon=False, loc="upper left", labelcolor=INK2, fontsize=10)

fig1.tight_layout()
fig1.savefig("ssd_model_bw_vs_qd.png", dpi=150)
print("saved -> ssd_model_bw_vs_qd.png")

# ============ Fig C: latency vs block size (QD = 1) ============
bs_data = np.array([d[0] for d in FIO_BS_SWEEP], dtype=float)
lat_meas = np.array([d[2] for d in FIO_BS_SWEEP], dtype=float)

bs_grid = np.logspace(np.log10(512), np.log10(65536), 300)
lat_model = np.array([latency_us(b, 1) for b in bs_grid])

fig2, ax2 = plt.subplots(figsize=(7.2, 5.2))

ax2.scatter(bs_data, lat_meas, s=60, facecolor=C_MEAS, edgecolor=SURFACE,
            linewidth=1.8, zorder=3, label="Measured")
ax2.plot(bs_grid, lat_model, color=C_MODEL, linewidth=2, zorder=2, label="Model")

ax2.set_xscale("log")
ax2.set_xlim(512, 65536)
ax2.set_ylim(75, 145)
ax2.set_xlabel("Block size bs (bytes)")
ax2.set_ylabel("Latency (µs)")
ax2.set_title("Latency vs block size (QD = 1)", color=INK, fontweight="semibold")
style_ax(ax2)
ax2.legend(frameon=False, loc="lower right", labelcolor=INK2, fontsize=10)

fig2.tight_layout()
fig2.savefig("ssd_model_lat_vs_bs.png", dpi=150)
print("saved -> ssd_model_lat_vs_bs.png")

# ============ Fig D: latency vs queue depth (bs = 4 KB) ============
qd_data = np.array([d[1] for d in FIO_QD_SWEEP], dtype=float)
lat_qd_meas = np.array([d[2] for d in FIO_QD_SWEEP], dtype=float)

qd_grid_lin = np.linspace(1, 64, 200)
lat_qd_model = np.array([latency_us(4096, q) for q in qd_grid_lin])

fig3, ax3 = plt.subplots(figsize=(7.2, 5.2))

ax3.scatter(qd_data, lat_qd_meas, s=60, facecolor=C_MEAS, edgecolor=SURFACE,
            linewidth=1.8, zorder=3, label="Measured")
ax3.plot(qd_grid_lin, lat_qd_model, color=C_MODEL, linewidth=2, zorder=2, label="Model")

ax3.set_xlim(0, 64)
ax3.set_ylim(82, 110)
ax3.set_xlabel("Queue depth QD")
ax3.set_ylabel("Latency (µs)")
ax3.set_title("Latency vs queue depth (bs = 4 KB)", color=INK, fontweight="semibold")
style_ax(ax3)
ax3.legend(frameon=False, loc="upper left", labelcolor=INK2, fontsize=10)

fig3.tight_layout()
fig3.savefig("ssd_model_lat_vs_qd.png", dpi=150)
print("saved -> ssd_model_lat_vs_qd.png")
