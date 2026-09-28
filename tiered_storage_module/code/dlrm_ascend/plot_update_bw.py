#!/usr/bin/env python3
"""
读 measure_update_bw.py 产出的 update_bw_result.json, 每条写回链路单独一张图:
  点   = 实测 (sizes_bytes, times_s)
  曲线 = 公式 T = α + S/BW (用拟合出的 alpha_s / bw_bytes_s)

两张图 (一段路径一张):
  1. update_npu2dram_fit.png  NPU->DRAM (device->host 写回, PCIe)
  2. update_dram2ssd_fit.png  DRAM->SSD (NVMe 写)

log-log 坐标, 能同时看清 α 固定开销区 (小尺寸段平坦) 与带宽区 (大尺寸段斜率 1)。

用法:
    python plot_update_bw.py
    python plot_update_bw.py --result update_bw_result.json --outdir ./figs
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ----------------------------------------------------------------------
# 调色板 (dataviz 参考色, 与 plot_migration_bw.py / plot_model_vs_measured.py 一致)
# ----------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # NPU->DRAM
AQUA = "#1baf7a"     # DRAM->SSD


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


def plot_link(r, name, color, outpath, dpi):
    """一张链路一张图: 实测散点 + 模型曲线 (log-log, 刻度标注 log10)。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(SURFACE)

    sizes_bytes = np.array(r["sizes_bytes"], dtype=float)
    times_s = np.array(r["times_s"], dtype=float)
    x_mb = sizes_bytes / 1e6
    y_ms = times_s * 1e3

    # 实测点
    ax.scatter(x_mb, y_ms, color=color, s=56, edgecolors=SURFACE,
               linewidths=1.5, zorder=3, label="measured")

    # 模型曲线 T = α + S/BW
    alpha_s = r["alpha_s"]
    bw = r["bw_bytes_s"]
    lo, hi = x_mb.min(), x_mb.max()
    xs = np.logspace(np.log10(max(lo * 0.5, 1e-3)), np.log10(hi * 2.0), 100)
    ys_ms = (alpha_s + xs * 1e6 / bw) * 1e3
    ax.plot(xs, ys_ms, color=color, linewidth=2, zorder=2, label="model")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # 刻度写成 log10 值 (100 -> 2), 并在坐标轴标签注明取了 log
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{np.log10(v):g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{np.log10(v):g}"))
    ax.set_xlabel("log10(data size / MB)")
    ax.set_ylabel("log10(time / ms)")
    ax.set_title(name, color=INK_PRIMARY, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _style_ax(ax)
    fig.savefig(outpath, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] -> {outpath}")


LARGE_SIZE_MB = 32.0  # 大尺寸段分界 (线性区), 用于分段的平均相对误差


def mape(r):
    """各尺寸点的相对误差 |pred - meas| / meas (%)。"""
    sizes_bytes = np.array(r["sizes_bytes"], dtype=float)
    times_s = np.array(r["times_s"], dtype=float)
    alpha_s = r["alpha_s"]
    bw = r["bw_bytes_s"]
    pred_s = alpha_s + sizes_bytes / bw
    return np.abs(pred_s - times_s) / times_s * 100.0


def main():
    parser = argparse.ArgumentParser(description="画更新写回链路的实测 vs 建模对比图")
    parser.add_argument("--result", default="update_bw_result.json",
                        help="measure_update_bw.py 产出的 JSON")
    parser.add_argument("--outdir", default=".", help="PNG 输出目录")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    with open(args.result) as f:
        data = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    links = [
        ("d2h", "NPU→DRAM (device→host write-back)", BLUE, "update_npu2dram_fit.png"),
        ("ssd", "DRAM→SSD (NVMe write)", AQUA, "update_dram2ssd_fit.png"),
    ]
    for key, name, color, fname in links:
        if key not in data:
            print(f"  [warn] {args.result} 缺少 '{key}', 跳过")
            continue
        r = data[key]
        if not r.get("bw_bytes_s"):
            print(f"  [warn] '{key}' 缺少有效带宽, 跳过")
            continue
        plot_link(r, name, color, os.path.join(args.outdir, fname), args.dpi)

        rel = mape(r)
        mb = np.array(r["sizes_bytes"], dtype=float) / 1e6
        large = mb >= LARGE_SIZE_MB
        print(f"  [{name}] MAPE = {rel.mean():.2f}%  "
              f"(>= {LARGE_SIZE_MB:.0f} MB: {rel[large].mean():.2f}%)")


if __name__ == "__main__":
    main()
