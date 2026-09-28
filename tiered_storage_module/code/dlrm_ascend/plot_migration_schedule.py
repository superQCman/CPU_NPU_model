#!/usr/bin/env python3
"""
可视化三级存储的迁移时间表 (matplotlib 静态图)

读 dlrm_msprof_mtrain.py 的 _log_migration_event 产出的 migration_schedule.log (CSV)，
输出两张 PNG:

  1. migration_schedule.png
     迁移时间表 (模型①: 迁移何时发生 / 每次迁多少)
     横轴 = step (迁移触发步), 每行 = 一个 embedding table,
     每个标记 = 一次迁移事件, 颜色 = 本次迁移数据量 (蓝阶, 越大越深)

  2. migration_duration_vs_volume.png
     数据量 vs 迁移时长 (模型②: 不同数据量所需的迁移时间)
     散点 + 线性拟合, 拟合结果即 T = slope * data + intercept 的标定

用法:
    python plot_migration_schedule.py
    python plot_migration_schedule.py --log migration_schedule.log --outdir ./figs

CSV 列名 (由 _log_migration_event 写入, 各链路分开计时):
    table_id, step, wall_time_s,
    n_hbm_to_dram, t_hbm_to_dram_s,
    n_dram_to_hbm, t_dram_to_hbm_s,
    n_ssd_to_dram, t_ssd_to_dram_s,
    n_dram_to_ssd,
    n_ssd_to_hbm, t_ssd_to_hbm_s,
    data_bytes, total_time_s
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头环境，避免依赖 display
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# ----------------------------------------------------------------------
# 调色板 (dataviz 参考色: light 表面)
# ----------------------------------------------------------------------
SURFACE = "#fcfcfb"          # 图表面
INK_PRIMARY = "#0b0b0b"      # 主文字
INK_SECONDARY = "#52514e"    # 次级文字 (轴标题)
INK_MUTED = "#898781"        # 弱化文字 (刻度)
GRID = "#e1e0d9"             # 网格线 (hairline)
AXIS = "#c3c2b7"             # 基线 / 轴脊线
BLUE = "#2a78d6"             # 类别槽 1 (主色)
ORANGE = "#eb6834"           # 类别槽 2 (拟合线强调色)

# 蓝阶 ramp (sequential: 数据量 magnitude, 浅→深)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


# ----------------------------------------------------------------------
# 读取 CSV
# ----------------------------------------------------------------------
def load_events(log_path):
    events = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                events.append({
                    "table_id": int(r["table_id"]),
                    "step": int(r["step"]),
                    "wall_time_s": float(r["wall_time_s"]),
                    "n_hbm_to_dram": int(r["n_hbm_to_dram"]),
                    "t_hbm_to_dram_s": float(r["t_hbm_to_dram_s"]),
                    "n_dram_to_hbm": int(r["n_dram_to_hbm"]),
                    "t_dram_to_hbm_s": float(r["t_dram_to_hbm_s"]),
                    "n_ssd_to_dram": int(r["n_ssd_to_dram"]),
                    "t_ssd_to_dram_s": float(r["t_ssd_to_dram_s"]),
                    "n_dram_to_ssd": int(r["n_dram_to_ssd"]),
                    "n_ssd_to_hbm": int(r["n_ssd_to_hbm"]),
                    "t_ssd_to_hbm_s": float(r["t_ssd_to_hbm_s"]),
                    "data_bytes": float(r["data_bytes"]),
                    "total_time_s": float(r["total_time_s"]),
                })
            except (KeyError, ValueError) as e:
                print(f"  [warn] 跳过一行 (解析失败 {e}): {r}")
    return events


def _style_ax(ax):
    """统一风格: 浅色表面 / 细灰轴 / hairline 网格 / 弱化刻度。"""
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    # 轴标题用次级 ink
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK_SECONDARY)


# ----------------------------------------------------------------------
# 图① 迁移时间表 (schedule)
# ----------------------------------------------------------------------
def plot_schedule(events, outpath, dpi):
    tables = sorted({e["table_id"] for e in events})
    y_of_table = {tid: i for i, tid in enumerate(tables)}
    n_tables = len(tables)

    data_mb = np.array([e["data_bytes"] / 1e6 for e in events], dtype=float)
    steps = np.array([e["step"] for e in events], dtype=int)
    ys = np.array([y_of_table[e["table_id"]] for e in events], dtype=int)

    vmin, vmax = data_mb.min(), data_mb.max()
    if vmax <= vmin:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(10, max(2.6, 0.55 * n_tables + 1.6)))
    fig.patch.set_facecolor(SURFACE)

    sc = ax.scatter(steps, ys, c=data_mb, cmap=cmap, norm=norm,
                    s=48, edgecolors=SURFACE, linewidths=1.5, zorder=3)

    ax.set_yticks(list(range(n_tables)))
    ax.set_yticklabels([str(t) for t in tables])
    ax.set_xlabel("step (migration trigger)")
    ax.set_ylabel("table_id")
    ax.set_title("Migration schedule (when + how much)", color=INK_PRIMARY, fontsize=12)
    _style_ax(ax)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("data volume (MB)", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED, length=3, width=0.8)
    cbar.outline.set_edgecolor(AXIS)

    fig.savefig(outpath, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [schedule] -> {outpath}  ({len(events)} events, {n_tables} tables)")


# ----------------------------------------------------------------------
# 图② 数据量 vs 迁移时长 (模型②标定)
# ----------------------------------------------------------------------
def plot_duration_vs_volume(events, outpath, dpi):
    data_mb = np.array([e["data_bytes"] / 1e6 for e in events], dtype=float)
    dur_ms = np.array([e["total_time_s"] * 1e3 for e in events], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fig.patch.set_facecolor(SURFACE)

    ax.scatter(data_mb, dur_ms, s=48, color=BLUE,
               edgecolors=SURFACE, linewidths=1.5, zorder=3, label="measured")

    # 线性拟合: T = slope * data + intercept
    if len(data_mb) >= 2:
        slope, intercept = np.polyfit(data_mb, dur_ms, 1)
        pred = slope * data_mb + intercept
        ss_res = float(np.sum((dur_ms - pred) ** 2))
        ss_tot = float(np.sum((dur_ms - dur_ms.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        xs = np.linspace(data_mb.min(), data_mb.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=2,
                label=f"fit: T = {slope:.3f}·MB + {intercept:.3f} ms  (R²={r2:.3f})")

        # 输出到控制台，供直接读
        print(f"  [fit]   T(ms) = {slope:.3f} * data(MB) + {intercept:.3f}   R²={r2:.3f}")

    ax.set_xlabel("data volume (MB)")
    ax.set_ylabel("migration duration (ms)")
    ax.set_title("Migration duration vs data volume", color=INK_PRIMARY, fontsize=12)
    _style_ax(ax)

    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.savefig(outpath, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [duration] -> {outpath}")


# ----------------------------------------------------------------------
# 图③ 各分段 Gantt: 横轴=发生时间, 每链路一行, bar 长=持续时间, 颜色=时长(log)
# ----------------------------------------------------------------------
LINKS = ["HBM->DRAM", "DRAM->HBM", "SSD->DRAM", "SSD->HBM"]
LINK_NCOLS = ["n_hbm_to_dram", "n_dram_to_hbm", "n_ssd_to_dram", "n_ssd_to_hbm"]
LINK_TCOLS = ["t_hbm_to_dram_s", "t_dram_to_hbm_s", "t_ssd_to_dram_s", "t_ssd_to_hbm_s"]


def plot_segment_gantt(events, outpath, dpi):
    """各分段发生时间 + 持续时间 (Gantt)。

    横轴 = wall_time (归一化到 0), 纵轴 = 4 条链路, 每个事件在对应链路上画一条 bar,
    bar 长度 = 该分段持续时间 (设最小可见宽度), 颜色 = 时长 (log 色阶, 便于读短分段)。
    """
    if not events:
        print("  [segments] 无事件, 跳过")
        return

    t0 = min(e["wall_time_s"] for e in events)
    t_span = (max(e["wall_time_s"] for e in events) - t0) or 1.0
    min_w = t_span * 0.0005  # 最小可见宽度 (0.05% 跨度)

    dur_all = [e[t] for e in events for t in LINK_TCOLS if e[t] > 0]
    if not dur_all:
        print("  [segments] 无有效时长, 跳过")
        return
    norm = LogNorm(vmin=min(dur_all), vmax=max(dur_all))
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(12, 4.2))
    fig.patch.set_facecolor(SURFACE)

    n_links = len(LINKS)
    ymap = {i: n_links - 1 - i for i in range(n_links)}  # 第一条链路画在最上

    for e in events:
        x0 = e["wall_time_s"] - t0
        for i, (ncol, tcol) in enumerate(zip(LINK_NCOLS, LINK_TCOLS)):
            if e[ncol] > 0 and e[tcol] > 0:
                w = max(e[tcol], min_w)
                ax.barh(ymap[i], w, left=x0, height=0.62,
                        color=cmap(norm(e[tcol])), edgecolor="none")

    ax.set_yticks(list(ymap.values()))
    ax.set_yticklabels(LINKS)
    ax.set_xlabel("wall time - t0 (s)")
    ax.set_title("Migration segments: occurrence time + duration",
                 color=INK_PRIMARY, fontsize=12)
    _style_ax(ax)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.array(dur_all))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("segment duration (s, log)", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED, length=3, width=0.8)
    cbar.outline.set_edgecolor(AXIS)

    fig.savefig(outpath, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [segments] -> {outpath}")


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="可视化三级存储迁移时间表")
    parser.add_argument("--log", default="migration_schedule.log",
                        help="迁移事件 CSV (由 dlrm_msprof_mtrain.py 产出)")
    parser.add_argument("--outdir", default=".",
                        help="PNG 输出目录")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"[error] 未找到 {args.log}")
        print("        请先跑带迁移日志的 DLRM (dlrm_msprof_mtrain.py 的 "
              "_log_migration_event 会写这个文件)。")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    events = load_events(args.log)
    if not events:
        print(f"[warn] {args.log} 为空，没有迁移事件可画。")
        return 1

    print(f"[plot] 读取 {len(events)} 条迁移事件")
    plot_schedule(events, os.path.join(args.outdir, "migration_schedule.png"), args.dpi)
    plot_duration_vs_volume(
        events, os.path.join(args.outdir, "migration_duration_vs_volume.png"), args.dpi)
    plot_segment_gantt(
        events, os.path.join(args.outdir, "migration_segment_gantt.png"), args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
