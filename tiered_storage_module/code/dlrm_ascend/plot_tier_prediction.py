#!/usr/bin/env python3
"""
DLRM 3-tier storage prediction overview (Zipf analytic / no trace).

Three-panel figure:
  +-------------+-------------+
  | 1) hit rate | 2) time     |
  +-------------+-------------+
  | 3) Zipf CDF cut           |
  +---------------------------+

Hit rates (Zipf closed form, no trace):
  r_HBM = H_{N_HBM}(a) / H_N(a)
  r_DRAM = [H_{N_HBM+N_DRAM}(a) - H_{N_HBM}(a)] / H_N(a)
  r_SSD = 1 - r_HBM - r_DRAM
  where H_M(a) = sum_{j=1}^{M} 1/j^a is the generalized harmonic number.

Time (reuses memory_model.py four-stage alpha/BW, consistent with the main model):
  n_i = r_i * K,  T_i = alpha_i * n_i + n_i * m * b / BW_i,  T_total = sum T_i

Usage:
  python plot_tier_prediction.py
  python plot_tier_prediction.py --alpha 1.0 --n 1000000 --hbm-rows 10000 --dram-rows 90000
  python plot_tier_prediction.py --dim 64000 --k 3200 --outdir ./figs
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from memory_model import ThreeLevelMemoryModel, STAGE_KEYS

# ----------------------------------------------------------------------
# Palette (consistent with plot_model_vs_measured.py etc.)
# ----------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # HBM
ORANGE = "#eb6834"   # DRAM
AQUA = "#1baf7a"     # SSD
VIOLET = "#8e6cc4"   # migration (DRAM->HBM)

# hit-rate tier colors
TIER_COLORS = {"HBM": BLUE, "DRAM": ORANGE, "SSD": AQUA}

# four data paths -> display name / color
STAGE_DISPLAY = {
    "HBM_to_NPU": "HBM->NPU",
    "DRAM_to_CPU": "DRAM->CPU",
    "SSD_to_DRAM": "SSD->DRAM",
    "DRAM_to_HBM": "DRAM->HBM (migrate)",
}
STAGE_COLORS = {
    "HBM_to_NPU": BLUE,
    "DRAM_to_CPU": ORANGE,
    "SSD_to_DRAM": AQUA,
    "DRAM_to_HBM": VIOLET,
}


# ----------------------------------------------------------------------
# Zipf analytic
# ----------------------------------------------------------------------
def zipf_cdf(N: int, alpha: float):
    """Return (j, cdf): j = 1..N, cdf[i] = H_{i+1}(a)/H_N(a)."""
    j = np.arange(1, N + 1, dtype=np.float64)
    w = j ** (-alpha)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return j, cdf


def hit_rates(N: int, n_hbm: int, n_dram: int, alpha: float):
    """Compute three-tier hit rates from Zipf closed form, plus cut indices."""
    _, cdf = zipf_cdf(N, alpha)
    cut1 = min(max(n_hbm, 0), N)
    cut2 = min(max(n_hbm + n_dram, 0), N)
    r_hbm = cdf[cut1 - 1] if cut1 > 0 else 0.0
    r_dram = cdf[cut2 - 1] - r_hbm if cut2 > 0 else 0.0
    r_ssd = 1.0 - r_hbm - r_dram
    return {"HBM": r_hbm, "DRAM": r_dram, "SSD": r_ssd}, cut1, cut2


# ----------------------------------------------------------------------
# Style / helpers
# ----------------------------------------------------------------------
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


def _annotate_segment(ax, left, width, label, threshold=0.06):
    """Label a segment in white text when wide enough; else rely on legend."""
    if width >= threshold:
        ax.text(left + width / 2, 0, label, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")


def _fmt_rows(n):
    """Abbreviate row counts: 1000 -> 1K, 1000000 -> 1M."""
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}K"
    return f"{n:,}"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DLRM 3-tier storage prediction overview")
    # hit rates / Zipf
    parser.add_argument("--alpha", type=float, default=1.0, help="Zipf skew parameter")
    parser.add_argument("--n", type=int, default=1_000_000, help="Total embedding rows N")
    parser.add_argument("--hbm-rows", type=int, default=10_000, help="Rows that fit in HBM")
    parser.add_argument("--dram-rows", type=int, default=90_000, help="Rows that fit in DRAM")
    # time
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension m")
    parser.add_argument("--byte-per-elem", type=int, default=4, help="Bytes per element (float32=4)")
    parser.add_argument("--k", type=int, default=10_240, help="Total lookups per forward K=B*k*T")
    parser.add_argument("--migrate-ratio", type=float, default=0.01,
                        help="Fraction of HBM rows migrated per interval")
    # output
    parser.add_argument("--outdir", default=".", help="PNG output directory")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # ---- 1) hit rates (Zipf closed form) ----
    hr, cut1, cut2 = hit_rates(args.n, args.hbm_rows, args.dram_rows, args.alpha)

    # ---- 2) time (memory_model four stages) ----
    model = ThreeLevelMemoryModel()
    n_hbm = int(round(hr["HBM"] * args.k))
    n_dram = int(round(hr["DRAM"] * args.k))
    n_ssd = int(round(hr["SSD"] * args.k))
    n_migrate = int(round(args.hbm_rows * args.migrate_ratio))
    pred = model.predict_forward(
        hbm_hits=n_hbm, dram_hits=n_dram, ssd_misses=n_ssd,
        dim=args.dim, migrate_count=n_migrate, byte_per_elem=args.byte_per_elem,
    )
    total_ms = pred["TOTAL"]

    # ---- layout: 2 rows x 2 cols, third panel spans both columns ----
    fig = plt.figure(figsize=(13, 8.4))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          hspace=0.45, wspace=0.30, top=0.86)
    ax_hit = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_cdf = fig.add_subplot(gs[1, :])

    # ================= Panel 1: hit-rate stacked bar =================
    left = 0.0
    for t in ["HBM", "DRAM", "SSD"]:
        r = hr[t]
        ax_hit.barh(0, r, left=left, height=0.5, color=TIER_COLORS[t],
                    edgecolor=SURFACE, linewidth=1.0)
        _annotate_segment(ax_hit, left, r, f"{t} {r*100:.1f}%", threshold=0.04)
        left += r
    ax_hit.set_xlim(0, 1)
    ax_hit.set_ylim(-0.55, 0.55)
    ax_hit.set_yticks([])
    ax_hit.set_xticks([0, 0.5, 1.0])
    ax_hit.set_xticklabels(["0%", "50%", "100%"])
    ax_hit.set_title("1) Hit rate (which tier)", color=INK_PRIMARY, fontsize=12)
    _style_ax(ax_hit)

    # ================= Panel 2: time stacked bar =================
    left = 0.0
    for key in STAGE_KEYS:
        share = pred[key] / total_ms if total_ms > 0 else 0.0
        ax_time.barh(0, share, left=left, height=0.42, color=STAGE_COLORS[key],
                     edgecolor=SURFACE, linewidth=1.0)
        if share >= 0.06:
            _annotate_segment(ax_time, left, share,
                              f"{STAGE_DISPLAY[key]} {share*100:.1f}%", threshold=0.06)
        left += share
    ax_time.set_xlim(0, 1)
    ax_time.set_ylim(-0.9, 0.9)
    ax_time.set_yticks([])
    ax_time.set_xticks([0, 0.5, 1.0])
    ax_time.set_xticklabels(["0%", "50%", "100%"])
    ax_time.set_title(f"2) Time breakdown (total {total_ms:.2f} ms)",
                      color=INK_PRIMARY, fontsize=12)
    _style_ax(ax_time)

    # per-stage legend (name + ms + share)
    handles = [mpatches.Patch(color=STAGE_COLORS[k]) for k in STAGE_KEYS]
    labels = [
        f"{STAGE_DISPLAY[k]}  {pred[k]:.3g} ms ({pred[k]/total_ms*100:.1f}%)"
        for k in STAGE_KEYS
    ]
    ax_time.legend(handles, labels, loc="upper left", frameon=False, fontsize=8,
                   labelcolor=INK_SECONDARY, handlelength=1.0, borderaxespad=0.4)

    # ================= Panel 3: Zipf CDF cut =================
    _, cdf = zipf_cdf(args.n, args.alpha)
    xs = np.unique(np.logspace(0, np.log10(args.n), 600).astype(int))
    xs = np.clip(xs, 1, args.n)
    ys = cdf[xs - 1]

    # shade the three tiers
    ax_cdf.axvspan(1, cut1, color=BLUE, alpha=0.12)
    ax_cdf.axvspan(cut1, cut2, color=ORANGE, alpha=0.12)
    ax_cdf.axvspan(cut2, args.n, color=AQUA, alpha=0.12)

    ax_cdf.plot(xs, ys, color=INK_PRIMARY, linewidth=2, zorder=3)

    # capacity cut lines + hit-rate readout lines
    ax_cdf.axvline(cut1, color=BLUE, linestyle="--", linewidth=1.2, zorder=4)
    ax_cdf.axvline(cut2, color=ORANGE, linestyle="--", linewidth=1.2, zorder=4)
    ax_cdf.axhline(hr["HBM"], color=INK_MUTED, linestyle=":", linewidth=1.0, zorder=2)
    ax_cdf.axhline(hr["HBM"] + hr["DRAM"], color=INK_MUTED, linestyle=":",
                   linewidth=1.0, zorder=2)

    # region labels (tier + hit rate + index range), at geometric centers
    c1, c2 = max(cut1, 1), max(cut2, 1)
    ax_cdf.text(np.sqrt(c1), 0.97,
                f"HBM {hr['HBM']*100:.1f}%\n1-{_fmt_rows(cut1)}",
                ha="center", va="top", color=BLUE, fontsize=9, fontweight="bold")
    ax_cdf.text(np.sqrt(c1 * c2), 0.97,
                f"DRAM {hr['DRAM']*100:.1f}%\n{_fmt_rows(cut1)}-{_fmt_rows(cut2)}",
                ha="center", va="top", color=ORANGE, fontsize=9, fontweight="bold")
    ax_cdf.text(np.sqrt(c2 * args.n), 0.97,
                f"SSD {hr['SSD']*100:.1f}%\n{_fmt_rows(cut2)}-{_fmt_rows(args.n)}",
                ha="center", va="top", color=AQUA, fontsize=9, fontweight="bold")

    ax_cdf.set_xscale("log")
    ax_cdf.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax_cdf.set_xlabel("embedding index i (descending frequency)")
    ax_cdf.set_ylabel("cumulative probability P(i)")
    ax_cdf.set_ylim(0, 1.0)
    ax_cdf.set_title(f"3) Zipf CDF cut (alpha={args.alpha})",
                     color=INK_PRIMARY, fontsize=12)
    _style_ax(ax_cdf)

    # ---- overall title: key takeaway ----
    ssd_share = pred["SSD_to_DRAM"] / total_ms * 100 if total_ms > 0 else 0.0
    fig.suptitle(
        f"DLRM 3-tier storage prediction (Zipf alpha={args.alpha}, N={args.n:,}, "
        f"N_HBM={args.hbm_rows:,}, N_DRAM={args.dram_rows:,})\n"
        f"SSD hit only {hr['SSD']*100:.1f}% but consumes {ssd_share:.1f}% of time",
        color=INK_PRIMARY, fontsize=12.5,
    )

    # ---- console summary ----
    print("=" * 62)
    print("  3-tier storage prediction (Zipf analytic / no trace)")
    print("=" * 62)
    print(f"  Hit rates:  HBM {hr['HBM']*100:5.1f}%   "
          f"DRAM {hr['DRAM']*100:5.1f}%   SSD {hr['SSD']*100:5.1f}%")
    print(f"  Access counts: HBM {n_hbm:,}  DRAM {n_dram:,}  SSD {n_ssd:,}  "
          f"migrate {n_migrate:,}")
    print("  Per-stage time (ms):")
    for key in STAGE_KEYS:
        t = pred[key]
        share = t / total_ms * 100 if total_ms > 0 else 0.0
        print(f"    {STAGE_DISPLAY[key]:<18s} {t:8.3f}  ({share:5.1f}%)")
    print(f"    {'TOTAL':<18s} {total_ms:8.3f}  (100.0%)")
    print(f"  -> SSD few hits but dominates time: {ssd_share:.1f}%")
    print("=" * 62)

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, "tier_prediction_overview.png")
    fig.savefig(outpath, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save] -> {outpath}")


if __name__ == "__main__":
    main()
