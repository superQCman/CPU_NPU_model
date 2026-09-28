#!/usr/bin/env python3
"""
读 measure_migration_bw.py 产出的 migration_bw_result.json, 画两张图:
  1. migration_bw_fit.png          PCIe 拟合验证 (实测散点 + 拟合线)
  2. migration_time_prediction.png 不同数据量下的分段迁移时间 (log-log)

用法:
    python plot_migration_bw.py
    python plot_migration_bw.py --result migration_bw_result.json --outdir ./figs
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 调色板 (dataviz 参考色)
# ----------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK_SECONDARY)


def plot_bw_fit(results, outpath, dpi):
    """带宽拟合验证: 三条链路都是实测散点 + 拟合线 (T = α + S/BW)。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(SURFACE)

    series = [("d2h", "HBM->DRAM", BLUE), ("h2d", "DRAM->HBM", ORANGE),
              ("ssd", "SSD->DRAM", AQUA)]
    for direction, name, color in series:
        if direction not in results:
            continue
        r = results[direction]
        x_mb = np.array([s / 1e6 for s in r["sizes_bytes"]])
        y_ms = np.array([t * 1e3 for t in r["times_s"]])
        ax.scatter(x_mb, y_ms, color=color, s=48, edgecolors=SURFACE,
                   linewidths=1.5, zorder=3, label=f"{name} measured")
        xs = np.linspace(x_mb.min(), x_mb.max(), 50)
        ys = (r["alpha_s"] + xs * 1e6 / r["bw_bytes_s"]) * 1e3
        ax.plot(xs, ys, color=color, linewidth=2, linestyle="--",
                label=f"{name} fit (BW={r['bw_bytes_s']/1e9:.1f} GB/s)")

    ax.set_xlabel("data size (MB)")
    ax.set_ylabel("copy time (ms)")
    ax.set_title("Bandwidth fit  T = α + S/BW", color=INK_PRIMARY, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _style_ax(ax)
    fig.savefig(outpath, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] -> {outpath}")


def plot_prediction(results, ssd_bw_bytes_s, outpath, dpi):
    """不同数据量下的分段迁移时间 (log-log, 每条链路一条线)。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(SURFACE)

    vol_mb = np.logspace(0, 4, 50)  # 1 MB ~ 10 GB
    vol_bytes = vol_mb * 1e6

    series = [
        (results["d2h"]["bw_bytes_s"], "HBM->DRAM (PCIe)", BLUE),
        (results["h2d"]["bw_bytes_s"], "DRAM->HBM (PCIe)", ORANGE),
        (ssd_bw_bytes_s, "SSD->DRAM (NVMe)", AQUA),
    ]
    for bw, name, color in series:
        t_ms = vol_bytes / bw * 1e3            # 纯带宽模型 T = DataSize/BW
        ax.plot(vol_mb, t_ms, color=color, linewidth=2,
                label=f"{name}  {bw/1e9:.1f} GB/s")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("data volume (MB)")
    ax.set_ylabel("migration time (ms)")
    ax.set_title("Migration time vs data volume  (T = DataSize / BW)",
                 color=INK_PRIMARY, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _style_ax(ax)
    fig.savefig(outpath, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] -> {outpath}")


def main():
    parser = argparse.ArgumentParser(description="画迁移带宽拟合图 + 迁移时间预测图")
    parser.add_argument("--result", default="migration_bw_result.json",
                        help="measure_migration_bw.py 产出的 JSON")
    parser.add_argument("--outdir", default=".", help="图输出目录")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    with open(args.result) as f:
        data = json.load(f)

    results = {"d2h": data["d2h"], "h2d": data["h2d"]}
    if "ssd" in data:
        results["ssd"] = data["ssd"]
    ssd_bw = data.get("ssd_bw_bytes_s", 7.0e9)

    os.makedirs(args.outdir, exist_ok=True)
    plot_bw_fit(results, os.path.join(args.outdir, "migration_bw_fit.png"), args.dpi)
    plot_prediction(results, ssd_bw,
                    os.path.join(args.outdir, "migration_time_prediction.png"), args.dpi)


if __name__ == "__main__":
    main()
