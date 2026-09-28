#!/usr/bin/env python3
"""
建模 vs 实测对比图（单张）：三条迁移链路的模型曲线 + 实测散点。

模型曲线 T = α + S/BW，参数来源按链路区分:
  - HBM->DRAM / DRAM->HBM (PCIe): 读 migration_bw_result.json 里拟合的
      alpha_s / bw_bytes_s (有效 device<->host 拷贝, 大块传输带宽界)。
  - SSD->DRAM (NVMe): 分段函数，分界点 10MB。
      >10MB 用 JSON 拟合 alpha_s / bw_bytes_s (不变);
      <10MB 用小尺寸实测点重新拟合 α_small / BW_small。

三条链路（单次迁移 = 一次大块搬移，n=1）:
  1. HBM->DRAM  (d2h, PCIe device->host)
  2. DRAM->HBM  (h2d, PCIe host->device)
  3. SSD->DRAM  (NVMe 单次随机读)

用法:
  # 1) 先测多组数据量（单次迁移，不同大小），产生迁移拟合结果:
  python measure_migration_bw.py --sizes 1 2 4 8 16 32 64 128 256 512 1024 --repeat 5

  # 2) 再画图:
  python plot_model_vs_measured.py
  python plot_model_vs_measured.py --result migration_bw_result.json --outdir ./figs

注:
  - PCIe 两条直接读拟合结果; SSD->DRAM 按 10MB 分段 (小尺寸重新拟合)。
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
# 调色板 (dataviz 参考色, 与 plot_migration_bw.py / plot_ssd_model.py 一致)
# ----------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # HBM->DRAM (d2h)
ORANGE = "#eb6834"   # DRAM->HBM (h2d)
AQUA = "#1baf7a"     # SSD->DRAM


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


def model_time_ms(size_bytes, alpha_s, bw_bytes_s):
    """解析模型 T = α + S/BW（n=1），返回毫秒。alpha_s 秒, bw_bytes_s 字节/秒。"""
    return (alpha_s + size_bytes / bw_bytes_s) * 1e3


def main():
    parser = argparse.ArgumentParser(description="建模 vs 实测对比图 (单张)")
    parser.add_argument("--result", default="migration_bw_result.json",
                        help="measure_migration_bw.py 产出的 JSON")
    parser.add_argument("--ssd-split-mb", type=float, default=10.0,
                        help="SSD->DRAM 分段分界点 (MB), 默认 10")
    parser.add_argument("--outdir", default=".", help="PNG 输出目录")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    with open(args.result) as f:
        data = json.load(f)

    # 三条路径都读 migration_bw_result.json 的拟合结果 (alpha_s / bw_bytes_s)
    series = [
        ("d2h", "HBM→DRAM (PCIe)", BLUE),
        ("h2d", "DRAM→HBM (PCIe)", ORANGE),
        ("ssd", "SSD→DRAM (NVMe)", AQUA),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.6))
    fig.patch.set_facecolor(SURFACE)

    for direction, name, color in series:
        if direction not in data:
            print(f"  [warn] {args.result} 缺少 '{direction}', 跳过")
            continue
        r = data[direction]

        alpha_s = r.get("alpha_s", 0.0)
        bw_bytes_s = r.get("bw_bytes_s")
        if not bw_bytes_s or not np.isfinite(bw_bytes_s):
            print(f"  [warn] '{direction}' 缺少有效带宽, 跳过")
            continue

        sizes_bytes = np.array(r["sizes_bytes"], dtype=float)
        times_s = np.array(r["times_s"], dtype=float)
        x_mb = sizes_bytes / 1e6
        y_ms = times_s * 1e3
        ax.scatter(x_mb, y_ms, color=color, s=56, edgecolors=SURFACE,
                   linewidths=1.5, zorder=3, label=f"{name} measured")

        lo, hi = x_mb.min(), x_mb.max()

        # ---- SSD->DRAM: 分段函数 (>分界点不变, <分界点重新拟合) ----
        if direction == "ssd":
            split_mb = args.ssd_split_mb
            small = sizes_bytes < split_mb * 1e6
            if small.sum() >= 2:
                slope, intercept = np.polyfit(sizes_bytes[small], times_s[small], 1)
                alpha_small, bw_small = intercept, 1.0 / slope
            else:
                alpha_small, bw_small = alpha_s, bw_bytes_s

            xs_small = np.logspace(np.log10(max(lo * 0.5, 1e-3)), np.log10(split_mb), 60)
            ys_small = model_time_ms(xs_small * 1e6, alpha_small, bw_small)
            ax.plot(xs_small, ys_small, color=color, linewidth=2, linestyle="--",
                    zorder=2)

            xs_large = np.logspace(np.log10(split_mb), np.log10(hi * 2.0), 60)
            ys_large = model_time_ms(xs_large * 1e6, alpha_s, bw_bytes_s)
            ax.plot(xs_large, ys_large, color=color, linewidth=2, zorder=2,
                    label=(f"{name} model [piecewise @{split_mb:.0f}MB] "
                           f"(<{split_mb:.0f}MB: BW={bw_small/1e9:.1f}, "
                           f"α={alpha_small*1e6:.0f}µs | "
                           f">{split_mb:.0f}MB: BW={bw_bytes_s/1e9:.1f}, "
                           f"α={alpha_s*1e6:.0f}µs)"))

            print(f"[{name}]  (piecewise @ {split_mb:.0f} MB)")
            print(f"    <{split_mb:.0f}MB: α={alpha_small*1e6:.2f} µs, "
                  f"BW={bw_small/1e9:.2f} GB/s")
            print(f"    >{split_mb:.0f}MB: α={alpha_s*1e6:.2f} µs, "
                  f"BW={bw_bytes_s/1e9:.2f} GB/s")
        else:
            xs_mb = np.logspace(np.log10(lo * 0.5), np.log10(hi * 2.0), 100)
            ys_ms = model_time_ms(xs_mb * 1e6, alpha_s, bw_bytes_s)
            ax.plot(xs_mb, ys_ms, color=color, linewidth=2, zorder=2,
                    label=f"{name} model [fit] (BW={bw_bytes_s/1e9:.1f} GB/s, "
                          f"α={alpha_s*1e6:.1f} µs)")
            print(f"[{name}]  (model source: fit)")
            print(f"    BW  = {bw_bytes_s/1e9:.2f} GB/s")
            print(f"    α   = {alpha_s*1e6:.2f} µs")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlabel("data size (MB)")
    ax.set_ylabel("single-migration time (ms)")
    ax.set_title("Model vs measured: single-migration time per link",
                 color=INK_PRIMARY, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _style_ax(ax)

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, "model_vs_measured.png")
    fig.savefig(outpath, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save] -> {outpath}")


if __name__ == "__main__":
    main()
