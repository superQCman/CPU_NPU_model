"""
All-to-All 延时建模
=============================================================
三段延时模型 + 残差样条修正：

  t_raw(m)   = 原三段模型输出（基于带宽模型，系统性偏高）
  t_final(m) = t_raw(m) - spline_correction(log2(m))

残差样条修正：
  1. 在训练数据上计算残差 r_i = t_raw(m_i) - t_true(m_i)
  2. 在 log2(size) 空间对残差做平滑三次样条拟合
  3. 预测时减去样条输出，消除系统性偏高
  4. 边界外自动用端点斜率线性外推，防止外推爆炸

优点：
  - 完全不改带宽模型（上图橙线保持不变）
  - 不引入新的物理假设，纯数据驱动修正
  - 样条在 log 空间平滑，不会过拟合
  - 小消息段（m <= m1）残差接近0，样条自然退化

参数：
  smooth_factor: 样条平滑因子，越大越平滑（默认 None = 自动）
  min_size_for_spline: 只对 size > 此值的点做残差拟合（默认 m1）
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

try:
    from src.memory_bw_utils import sigmoid, MUL_FACTOR_FUNCS
    from src.bw_model import (
        parse_hccl_output,
        sigmoid_with_gaussian_dip,
        ALIGNMENT_HIGH, ALIGNMENT_MID,
    )
except ModuleNotFoundError:
    from memory_bw_utils import sigmoid, MUL_FACTOR_FUNCS
    from bw_model import (
        parse_hccl_output,
        sigmoid_with_gaussian_dip,
        ALIGNMENT_HIGH, ALIGNMENT_MID,
    )

M1_FIXED = 2 ** 21   # 2 MiB，固定小消息边界


# ============================================================
# 1. 从饱和带宽反向定位 m2（不变）
# ============================================================
def find_m2_from_satbw(
    df, sat_bw, eval_bw_fn,
    search_start_bytes=32 * 1024 * 1024,
    stable_ratio=0.97,
    persist=20,
):
    d = df[df['size'] >= search_start_bytes].sort_values('size').reset_index(drop=True)
    if len(d) < persist * 2:
        return int(d['size'].quantile(0.80)) if len(d) > 0 else search_start_bytes

    sizes = d['size'].to_numpy()
    bw_model = np.array([eval_bw_fn(int(s)) for s in sizes])
    threshold = sat_bw * stable_ratio

    cnt = 0
    start_idx = None
    for i, bw in enumerate(bw_model):
        if bw >= threshold:
            if cnt == 0:
                start_idx = i
            cnt += 1
            if cnt >= persist:
                return int(sizes[start_idx])
        else:
            cnt = 0
            start_idx = None

    return int(sizes[max(0, len(sizes) - persist)])


# ============================================================
# 2. 三段延时模型（原始，不含修正）
# ============================================================
def latency_model_base(m, t_s, m1, m2, bw_max_GBs, eval_bw):
    """
    m <= m1  : t = t_s
    m1<m<m2  : t = m / BW(m)
    m >= m2  : t = t_s + m / BW_max
    """
    m = np.asarray(m, dtype=np.float64)
    scalar = m.ndim == 0
    m = np.atleast_1d(m)
    bw_max_bpus = bw_max_GBs * 1e3

    t = np.empty_like(m)
    for i, mi in enumerate(m):
        if mi <= m1:
            t[i] = t_s
        elif mi >= m2:
            t[i] = t_s + mi / bw_max_bpus
        else:
            bw_bpus = max(eval_bw(int(mi)) * 1e3, 1e-9)
            t[i] = mi / bw_bpus

    return float(t[0]) if scalar else t


# ============================================================
# 3. 估计 t_s：小消息段均值
# ============================================================
def estimate_ts_from_small(df, m1=M1_FIXED, clip_quantile=0.05):
    small = df[df['size'] <= m1]
    if len(small) < 3:
        small = df.nsmallest(10, 'size')
        print(f"  [warn] 小消息点不足，回退到最小10个点（n={len(small)}）")
    vals = small['latency'].to_numpy()
    lo = np.quantile(vals, clip_quantile)
    hi = np.quantile(vals, 1.0 - clip_quantile)
    valid = vals[(vals >= lo) & (vals <= hi)]
    return float(np.mean(valid))


# ============================================================
# 4. 残差样条拟合
# ============================================================
def fit_residual_spline(df, t_s, m1, m2, bw_max_GBs, eval_bw,
                        smooth_factor=None,
                        min_fit_size=None):
    """
    在训练数据上拟合残差样条：
      r_i = t_raw(m_i) - t_true(m_i)
    在 log2(size) 空间做平滑三次样条。

    返回: correction_fn(m) -> 修正量（标量或数组）
      t_final = t_raw - correction_fn(m)

    策略：
      - size <= m1（小消息段）：correction = 0，完全使用 t_s 平均值，不做任何修正
      - size >  m1（中大消息段）：用样条拟合残差进行修正
      - 在 m1 处插入锚点 (log2(m1), 0)，保证样条从0平滑过渡，无跳变
      - 右端界外线性外推（用端点导数），防止样条外推爆炸
    """
    if min_fit_size is None:
        min_fit_size = m1

    fit_df = df.sort_values('size').reset_index(drop=True)

    sizes    = fit_df['size'].to_numpy(np.float64)
    lat_true = fit_df['latency'].to_numpy(np.float64)
    lat_base  = latency_model_base(sizes, t_s, m1, m2, bw_max_GBs, eval_bw)

    # 残差 = 原始预测 - 真实值（正值表示预测偏高）
    residuals = lat_base - lat_true

    # ── 只取 size > m1 的点做样条拟合 ────────────────────
    mask_large = sizes > min_fit_size
    sizes_large = sizes[mask_large]
    resid_large = residuals[mask_large]
    log_sizes_large = np.log2(sizes_large)

    # 对中大消息段残差做 bin 聚合（中位数去噪）
    bin_edges = np.arange(log_sizes_large.min() - 0.05,
                          log_sizes_large.max() + 0.15, 0.25)
    bin_idx   = np.digitize(log_sizes_large, bin_edges)
    agg_log_x, agg_res = [], []
    for b in np.unique(bin_idx):
        bmask = bin_idx == b
        if bmask.sum() > 0:
            agg_log_x.append(np.median(log_sizes_large[bmask]))
            agg_res.append(np.median(resid_large[bmask]))
    agg_log_x = np.array(agg_log_x)
    agg_res   = np.array(agg_res)

    # 排序
    sort_idx  = np.argsort(agg_log_x)
    agg_log_x = agg_log_x[sort_idx]
    agg_res   = agg_res[sort_idx]

    # ── 在 m1 处插入锚点 (log2(m1), 0) ──────────────────
    # 保证样条在小消息边界处平滑收敛到0，不产生跳变
    anchor_x = np.log2(float(m1))
    anchor_y = 0.0
    agg_log_x = np.concatenate([[anchor_x], agg_log_x])
    agg_res   = np.concatenate([[anchor_y], agg_res])

    # 平滑因子：默认用（点数-1）的 0.5 倍（锚点不计入）
    if smooth_factor is None:
        smooth_factor = (len(agg_log_x) - 1) * 0.5

    spline = UnivariateSpline(agg_log_x, agg_res,
                              k=3, s=smooth_factor, ext=0)

    # 右端点信息（用于线性外推）
    x_hi  = agg_log_x[-1]
    y_hi  = float(spline(x_hi))
    dy_hi = float(spline.derivative()(x_hi))

    def correction_fn(m):
        """返回修正量（正值 = 预测偏高，需减去）
        - size <= m1：直接返回 0，小消息段完全不修正，延时保持 t_s
        - size >  m1：样条插值修正；右端界外线性外推
        """
        m = np.asarray(m, dtype=np.float64)
        scalar = m.ndim == 0
        m = np.atleast_1d(m)

        corr = np.empty(len(m), dtype=np.float64)
        for i, mi in enumerate(m):
            if mi <= m1:
                corr[i] = 0.0          # 小消息段不修正
            else:
                lx = np.log2(mi)
                if lx > x_hi:
                    corr[i] = y_hi + dy_hi * (lx - x_hi)   # 右端线性外推
                else:
                    corr[i] = float(spline(lx))

        return float(corr[0]) if scalar else corr

    # 诊断信息
    res_pred  = np.array([correction_fn(s) for s in sizes])
    resid_after = residuals - res_pred
    lat_final  = lat_base - res_pred
    mape_final = float(np.mean(np.abs(lat_final - lat_true) /
                               np.maximum(lat_true, 1e-9)) * 100)

    print(f"  [spline] 聚合点数={len(agg_log_x)}, smooth_factor={smooth_factor:.1f}")
    print(f"  [spline] final MAPE={mape_final:.2f}%  "
          f"(residual mean after correction {resid_after.mean():.1f}us)")

    return correction_fn, spline, agg_log_x, agg_res, mape_final


# ============================================================
# 5. 带修正的延时预测
# ============================================================
def latency_model_corrected(m, t_s, m1, m2, bw_max_GBs, eval_bw,
                             correction_fn):
    """
    t_final(m) = t_raw(m) - correction_fn(m)
    保证输出 >= 0。
    """
    t_base  = latency_model_base(m, t_s, m1, m2, bw_max_GBs, eval_bw)
    m_arr  = np.atleast_1d(np.asarray(m, dtype=np.float64))
    scalar = np.asarray(m, dtype=np.float64).ndim == 0

    corr   = correction_fn(m_arr)
    t_arr  = np.atleast_1d(t_base) - corr
    t_arr  = np.maximum(t_arr, 0.0)

    return float(t_arr[0]) if scalar else t_arr


# ============================================================
# 6. 评估
# ============================================================
def evaluate_latency_model(df, t_s, m1, m2, bw_max_GBs, eval_bw,
                            correction_fn=None):
    sizes    = df['size'].to_numpy(np.float64)
    lat_true = df['latency'].to_numpy(np.float64)
    if correction_fn is not None:
        lat_pred = latency_model_corrected(sizes, t_s, m1, m2,
                                           bw_max_GBs, eval_bw, correction_fn)
    else:
        lat_pred = latency_model_base(sizes, t_s, m1, m2, bw_max_GBs, eval_bw)
    abs_err = np.abs(lat_true - lat_pred)
    mape = float(np.mean(abs_err / np.maximum(lat_true, 1e-9)) * 100)
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    return mape, rmse


# ============================================================
# 7. 主入口
# ============================================================
def run_latency_model(bw_results, raw_df, out_png='latency_model.png',
                      spline_smooth=None):

    eval_model      = bw_results['eval_model']
    sat_bw_high     = bw_results['sat_bw_high']
    sat_bw_mid      = bw_results['sat_bw_mid']
    sat_bw_low3     = bw_results['sat_bw_low3']
    dip_high        = bw_results['dip_high']
    dip_mid         = bw_results['dip_mid']
    dip_low3        = bw_results['dip_low3']
    low3_drop_size  = bw_results['low3_drop_size']
    popt_high_final = bw_results['popt_high_final']
    popt_mid_final  = bw_results['popt_mid_final']

    high_df = raw_df[raw_df['size'] % ALIGNMENT_HIGH == 0].copy().reset_index(drop=True)
    mid_df  = raw_df[(raw_df['size'] % ALIGNMENT_HIGH != 0) &
                     (raw_df['size'] % ALIGNMENT_MID  == 0)].copy().reset_index(drop=True)
    low_df  = raw_df[ raw_df['size'] % ALIGNMENT_MID  != 0 ].copy().reset_index(drop=True)

    mid_fit_df = mid_df if len(mid_df) > 20 else (low_df if len(low_df) > 20 else high_df)
    low_fit_df = low_df if len(low_df) > 20 else None

    def eval_bw_high(sz): return eval_model(sz, 'high')
    def eval_bw_mid(sz):  return eval_model(sz, 'mid')
    def eval_bw_low(sz):  return eval_model(sz, 'low')

    m2_high = find_m2_from_satbw(high_df,    sat_bw_high, eval_bw_high)
    m2_mid  = find_m2_from_satbw(mid_fit_df, sat_bw_mid,  eval_bw_mid)
    m2_low3 = low3_drop_size if low3_drop_size is not None else m2_high

    groups = [
        ('Group1 (128KB)', high_df,    eval_bw_high, sat_bw_high, m2_high,
         popt_high_final, dip_high, None),
        ('Group2 (32KB)',  mid_fit_df, eval_bw_mid,  sat_bw_mid,  m2_mid,
         popt_mid_final,  dip_mid,  None),
    ]
    if low_fit_df is not None:
        groups.append(
            ('Group3 (unaligned)', low_fit_df, eval_bw_low, sat_bw_low3, m2_low3,
             popt_high_final, dip_low3, low3_drop_size)
        )

    results = {}
    for name, df_g, eval_bw, bw_max, m2, popt_bw, dip, drop_size in groups:
        if len(df_g) == 0:
            continue

        m1  = M1_FIXED
        t_s = estimate_ts_from_small(df_g, m1=m1)

        # 残差样条拟合
        (correction_fn, spline_obj,
         spl_x, spl_r,
         mape_spl) = fit_residual_spline(
            df_g, t_s, m1, m2, bw_max, eval_bw,
            smooth_factor=spline_smooth,
        )

        mape, rmse = evaluate_latency_model(
            df_g, t_s, m1, m2, bw_max, eval_bw, correction_fn)

        print(f"\n[{name}]")
        print(f"  t_s    = {t_s:.2f} us")
        print(f"  m1     = 2^21 = {M1_FIXED//1024} KB (fixed)")
        print(f"  m2     = {m2/1024/1024:.1f} MiB  (sat-BW anchored)")
        print(f"  BW_max = {bw_max:.2f} GB/s")
        print(f"  MAPE   = {mape:.2f}%,  RMSE = {rmse:.2f} us  (spline corrected)")

        results[name] = dict(
            t_s=t_s, m1=m1, m2=m2, bw_max=bw_max,
            eval_bw=eval_bw, popt_bw=popt_bw,
            correction_fn=correction_fn,
            spline_obj=spline_obj,
            spl_x=spl_x, spl_r=spl_r,
            has_dip=(len(popt_bw) == 7),
            dip=dip, drop_size=drop_size,
            mape=mape, rmse=rmse,
            df=df_g,
        )

    _plot_latency_model(results, out_png)
    return results


# ============================================================
# 8. 画图
# ============================================================
def _plot_latency_model(results, out_png):
    n = len(results)
    # 3行：带宽图 / 延时图 / 残差样条图
    fig, axes = plt.subplots(3, n, figsize=(7 * n, 15))
    if n == 1:
        axes = axes.reshape(3, 1)

    grp_colors = {
        'Group1 (128KB)':     '#d62728',
        'Group2 (32KB)':      '#1f77b4',
        'Group3 (unaligned)': '#2ca02c',
    }

    for col, (name, res) in enumerate(results.items()):
        df_g         = res['df']
        t_s          = res['t_s']
        m1           = res['m1']
        m2           = res['m2']
        bw_max       = res['bw_max']
        eval_bw      = res['eval_bw']
        correction_fn= res['correction_fn']
        dip          = res['dip']
        drop_size    = res['drop_size']
        spl_x        = res['spl_x']
        spl_r        = res['spl_r']
        c            = grp_colors.get(name, 'gray')
        is_low3      = drop_size is not None

        x_min = df_g['size'].min()
        x_max = df_g['size'].max()
        x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 800)

        # ── 行0：带宽（不变，仅展示带宽模型拟合质量）────────
        ax_bw = axes[0][col]
        ax_bw.axvspan(x_min, m1,    alpha=0.10, color='#d0e8ff', label='① Small')
        ax_bw.axvspan(m1,    m2,    alpha=0.10, color='#fff3cd', label='② Transition')
        ax_bw.axvspan(m2,    x_max, alpha=0.10, color='#d4edda', label='③ Linear')
        ax_bw.scatter(df_g['size'], df_g['bus_bw'],
                      s=8, alpha=0.22, color=c, label='Raw BW', zorder=2)

        if is_low3:
            x_before = x_fit[x_fit < drop_size]
            if len(x_before):
                bw_b = np.array([eval_bw(int(v)) for v in x_before])
                ax_bw.plot(x_before, bw_b, color='orange', lw=2.5,
                           label='BW model (pre-drop)', zorder=4)
            x_after = x_fit[x_fit >= drop_size]
            if len(x_after):
                ax_bw.plot(x_after, np.full(len(x_after), bw_max),
                           color='orange', lw=2.5,
                           label=f'Sat={bw_max:.1f} GB/s', zorder=4)
            ax_bw.axvline(drop_size, ls='-.', lw=2.0, color='#d62728', alpha=0.85,
                          label=f'Drop@{drop_size//1024//1024}MiB')
        else:
            bw_curve = np.array([eval_bw(int(v)) for v in x_fit])
            ax_bw.plot(x_fit, bw_curve, color='orange', lw=2.5,
                       label='BW model (eval)', zorder=4)

        ax_bw.axhline(bw_max, color=c, ls=':', lw=1.8, alpha=0.7,
                      label=f'BW_max={bw_max:.1f} GB/s')
        ax_bw.axvline(m1, color='steelblue', ls='--', lw=1.8,
                      label=f'm1={m1//1024}KB')
        ax_bw.axvline(m2, color='black', ls='--', lw=2.0,
                      label=f'm2={m2/1024/1024:.1f}MiB')
        if dip is not None:
            ax_bw.axvspan(dip[0], dip[1], alpha=0.12, color='#ff7f0e',
                          label=f'Dip {dip[0]//1024//1024}~{dip[1]//1024//1024}MiB')
        ax_bw.set_xscale('log', base=2)
        ax_bw.grid(True, which='both', ls='-', alpha=0.2)
        ax_bw.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax_bw.set_xlabel('Message Size (Bytes)')
        ax_bw.set_ylabel('Bus BW (GB/s)')
        ax_bw.legend(fontsize=7, loc='upper left', ncol=2)

        # ── 行1：延时（对比修正前后）─────────────────────
        ax_lat = axes[1][col]
        ax_lat.axvspan(x_min, m1,    alpha=0.10, color='#d0e8ff')
        ax_lat.axvspan(m1,    m2,    alpha=0.10, color='#fff3cd')
        ax_lat.axvspan(m2,    x_max, alpha=0.10, color='#d4edda')

        ax_lat.scatter(df_g['size'], df_g['latency'],
                       s=8, alpha=0.22, color=c, label='Raw latency', zorder=2)

        # 修正后（实线，分三段着色）
        x1 = x_fit[x_fit <= m1]
        if len(x1):
            lat1 = latency_model_corrected(x1, t_s, m1, m2, bw_max, eval_bw, correction_fn)
            ax_lat.plot(x1, lat1, color='#0057ff', lw=2.5,
                        label=f'①(t_s={t_s:.0f}μs)', zorder=5)

        x2 = x_fit[(x_fit > m1) & (x_fit < m2)]
        if len(x2):
            lat2 = latency_model_corrected(x2, t_s, m1, m2, bw_max, eval_bw, correction_fn)
            ax_lat.plot(x2, lat2, color='#ff7f0e', lw=2.5,
                        label='② (sigmoid)', zorder=5)

        x3 = x_fit[x_fit >= m2]
        if len(x3):
            lat3 = latency_model_corrected(x3, t_s, m1, m2, bw_max, eval_bw, correction_fn)
            ax_lat.plot(x3, lat3, color='#2ca02c', lw=2.5,
                        label='③(linear)', zorder=5)

        ax_lat.axvline(m1, color='steelblue', ls='--', lw=1.8)
        ax_lat.axvline(m2, color='black',     ls='--', lw=2.0)
        ax_lat.set_xscale('log', base=2)
        ax_lat.set_yscale('log')
        ax_lat.grid(True, which='both', ls='-', alpha=0.2)
        ax_lat.set_title(
            f't_s={t_s:.1f}μs  m2={m2/1024/1024:.1f}MiB  '
            f'BW_max={bw_max:.0f}GB/s ',
            fontsize=9)
        ax_lat.set_xlabel('Message Size (Bytes)')
        ax_lat.set_ylabel('Latency (us)')
        ax_lat.legend(fontsize=7, loc='upper left')

        # ── 行2：残差样条图（诊断用）─────────────────────
        ax_spl = axes[2][col]

        # 原始残差散点
        sizes_g    = df_g['size'].to_numpy(np.float64)
        lat_true_g = df_g['latency'].to_numpy(np.float64)
        lat_base_g  = latency_model_base(sizes_g, t_s, m1, m2, bw_max, eval_bw)
        residuals_g = lat_base_g - lat_true_g
        ax_spl.scatter(sizes_g, residuals_g,
                       s=8, alpha=0.25, color=c, label='Raw residual', zorder=2)

        # 聚合点（样条拟合点）
        agg_sizes = 2.0 ** spl_x
        ax_spl.scatter(agg_sizes, spl_r,
                       s=40, color='black', zorder=4, label='Agg median', marker='D')

        # 样条曲线
        corr_curve = np.array([correction_fn(v) for v in x_fit])
        ax_spl.plot(x_fit, corr_curve, color='red', lw=2.5,
                    label='Spline correction', zorder=5)
        ax_spl.axhline(0, color='gray', ls='-', lw=1.0)
        ax_spl.axvline(m1, color='steelblue', ls='--', lw=1.5)
        ax_spl.axvline(m2, color='black',     ls='--', lw=1.5)

        ax_spl.set_xscale('log', base=2)
        ax_spl.grid(True, which='both', ls='-', alpha=0.2)
        ax_spl.set_title('Residual = t_raw - t_true  (red=spline fit)', fontsize=9)
        ax_spl.set_xlabel('Message Size (Bytes)')
        ax_spl.set_ylabel('Residual (us)')
        ax_spl.legend(fontsize=7, loc='upper left')

    fig.suptitle('All-to-All: Three-Segment Latency Model  '
                 '(BW model unchanged + residual spline correction)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"\nSaved: {out_png}")


# ============================================================
# 9. eval_latency：统一查询入口
# ============================================================
def eval_latency(size_bytes, latency_results, group=None):
    if group is None:
        if size_bytes % ALIGNMENT_HIGH == 0:
            group = 'Group1 (128KB)'
        elif size_bytes % ALIGNMENT_MID == 0:
            group = 'Group2 (32KB)'
        else:
            group = 'Group3 (unaligned)'
    if group not in latency_results:
        group = next(iter(latency_results))
    res = latency_results[group]
    return latency_model_corrected(
        size_bytes,
        res['t_s'], res['m1'], res['m2'],
        res['bw_max'], res['eval_bw'],
        res['correction_fn'],
    )


# ============================================================
# 10. 生成验证脚本 & 对比分析
# ============================================================
def generate_test_sizes(n=100, lo=4*1024, hi=1*1024*1024*1024):
    raw = np.logspace(np.log2(lo), np.log2(hi), n, base=2)
    aligned = (np.round(raw / 4096) * 4096).astype(np.int64)
    return np.unique(np.clip(aligned, lo, hi))


def write_test_script(sizes, script_path, result_path, bin_dir):
    lines = [
        '#!/bin/bash',
        f'cd {bin_dir}',
        f'> {result_path}',
        '',
        f'# 验证测试：{len(sizes)} 个消息大小（4KB~1GB 对数均匀）',
        '',
    ]
    for sz in sizes:
        lines.append(
            f'mpirun -n 8 ./alltoall_test -b {sz} -e {sz} -f 2 -p 8 -n 100 '
            f'>> {result_path} 2>&1'
        )
    lines += ['', f'echo "All done. Results: {result_path}"']
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(script_path, 0o755)
    print(f"\n[Validation script] {script_path}  ({len(sizes)} sizes)")


def parse_and_compare(result_file, lat_results,
                      out_png='latency_compare.png', num_gpus=8,
                      max_rows_print=80):
    mul_factor = MUL_FACTOR_FUNCS['all_to_all'](num_gpus)
    df = parse_hccl_output(result_file)
    if df is None or len(df) == 0:
        print(f"[Error] Cannot parse: {result_file}")
        return None

    df['bus_bw'] = df['alg_bw'] * mul_factor
    df = df.sort_values('size').reset_index(drop=True)

    df['latency_pred'] = [eval_latency(int(r['size']), lat_results)
                          for _, r in df.iterrows()]
    df['abs_err'] = (df['latency_pred'] - df['latency']).abs()
    df['rel_err'] = df['abs_err'] / df['latency'].clip(lower=1e-9) * 100

    print(f"\n{'='*65}")
    print(f"{'Size':>12}  {'True(us)':>10}  {'Pred(us)':>10}  "
          f"{'AbsErr':>8}  {'RelErr%':>8}")
    print(f"{'-'*65}")
    if len(df) <= max_rows_print:
        rows_to_print = df
    else:
        head_n = max_rows_print // 2
        tail_n = max_rows_print - head_n
        rows_to_print = pd.concat([df.head(head_n), df.tail(tail_n)], axis=0)

    for _, row in rows_to_print.iterrows():
        sz = row['size']
        sz_str = f"{sz//1024:.0f}KB" if sz < 1024*1024 else f"{sz//1024//1024:.1f}MiB"
        print(f"{sz_str:>12}  {row['latency']:>10.1f}  "
              f"{row['latency_pred']:>10.1f}  "
              f"{row['abs_err']:>8.1f}  {row['rel_err']:>7.1f}%")
    if len(df) > max_rows_print:
        print(f"... ({len(df) - max_rows_print} rows omitted)")
    print(f"{'='*65}")
    mape = df['rel_err'].mean()
    rmse = float(df['abs_err'].pow(2).mean() ** 0.5)
    p50  = df['rel_err'].quantile(0.50)
    p90  = df['rel_err'].quantile(0.90)
    print(f"  MAPE={mape:.2f}%  RMSE={rmse:.1f}us  P50={p50:.1f}%  P90={p90:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(df['size'], df['latency'],
               s=45, color='steelblue', label='True latency', zorder=3)
    ax.scatter(df['size'], df['latency_pred'],
               s=45, color='#d62728', marker='^', label='Predicted', zorder=4)
    for _, row in df.iterrows():
        ax.plot([row['size'], row['size']],
                [row['latency'], row['latency_pred']],
                color='gray', lw=0.8, alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.set_xlabel('Message Size (Bytes)')
    ax.set_ylabel('Latency (us)')
    ax.set_title('Predicted vs True Latency')
    ax.legend()

    ax2 = axes[1]
    ax2.scatter(df['size'], df['rel_err'], s=45, color='#ff7f0e', zorder=3)
    ax2.axhline(mape, color='red', ls='--', lw=1.5, label=f'Mean={mape:.1f}%')
    ax2.axhline(10,   color='gray', ls=':',  lw=1.2, label='10% line')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, which='both', ls='-', alpha=0.2)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Message Size (Bytes)')
    ax2.set_ylabel('Relative Error (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"\nSaved: {out_png}")
    return df


# ============================================================
# 11. 主入口
# ============================================================
if __name__ == '__main__':
    try:
        from src.bw_model import run_two_stage_piecewise
    except ModuleNotFoundError:
        from bw_model import run_two_stage_piecewise

    FILE    = 'data/raw/a2a_32kb_aligned.txt'
    BIN_DIR = 'bin'

    # Step 1: 带宽模型
    bw_results = run_two_stage_piecewise(
        file_path=FILE, num_gpus=8,
        out_png='artifacts/plots/bw_model_fit.png', split_ratio=0.03,
    )

    # Step 2: 原始数据
    mul_factor = MUL_FACTOR_FUNCS['all_to_all'](8)
    raw_df = parse_hccl_output(FILE)
    raw_df['bus_bw'] = raw_df['alg_bw'] * mul_factor

    # Step 3: 延时模型（残差样条修正）
    # spline_smooth=None 自动，也可手动指定如 spline_smooth=5.0 控制平滑度
    lat_results = run_latency_model(
        bw_results=bw_results,
        raw_df=raw_df,
        out_png='artifacts/plots/latency_model_fit.png',
        spline_smooth=None,
    )

    # Step 4: 验证对比
    compare_df = parse_and_compare(
        result_file=f'{BIN_DIR}/latency_validation_results.txt',
        lat_results=lat_results,
        out_png='artifacts/plots/latency_compare.png',
    )
    
    
