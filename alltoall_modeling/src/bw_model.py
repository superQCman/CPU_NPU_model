"""
All-to-All 带宽建模
=============================================================
三类数据行为：
  Group1 (128KB对齐):    sigmoid上升 → dip震荡 → 饱和≈115 GB/s
  Group2 (32KB非128KB): sigmoid上升(较慢) → dip震荡 → 饱和≈85 GB/s
  Group3 (完全不对齐):   前段跟Group1走 → dip震荡 → 跌落至≈85 GB/s（Group2饱和）

建模策略：
  Group1: sigmoid趋势 + Gaussian dip，尾部 clip 到 sat_bw_high
  Group2: sigmoid趋势 + Gaussian dip，尾部 clip 到 sat_bw_mid
  Group3: 分两段
    - size < drop_size: 用 Group1 的趋势（两者上升段相同）
    - size >= drop_size: 直接返回 sat_bw_mid（跌落到低饱和）
    同样穿过 dip 时叠加 Gaussian 下降

eval_model(size, alignment_class) 统一入口：
  alignment_class: 'high'(128KB) / 'mid'(32KB) / 'low'(unaligned)
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    from src.memory_bw_utils import sigmoid, MUL_FACTOR_FUNCS
except ModuleNotFoundError:
    from memory_bw_utils import sigmoid, MUL_FACTOR_FUNCS

ALIGNMENT_HIGH = 131072   # 128 KB
ALIGNMENT_MID  = 32768    # 32 KB


# ============================================================
# 1. 解析
# ============================================================
def parse_hccl_output(filename):
    d = {'size': [], 'alg_bw': [], 'latency': []}
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        for line in f:
            if '|' not in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if not parts or not parts[0].isdigit():
                continue
            try:
                d['size'].append(int(parts[0]))
                d['latency'].append(float(parts[1]))
                d['alg_bw'].append(float(parts[2]))
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(d)


# ============================================================
# 2. sigmoid 拟合（log2 size → log10 bw）
# ============================================================
def fit_sigmoid_logbw(df, size_col='size', bw_col='bus_bw',
                      sigma=None, p0=None, bw_cap=None):
    x = np.log2(df[size_col].to_numpy(dtype=np.float64))
    y = np.log10(df[bw_col].to_numpy(dtype=np.float64))

    if bw_cap is not None:
        y = np.minimum(y, np.log10(bw_cap) + 0.01)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    if p0 is None:
        p0 = [max(0.05, y_max - y_min), np.median(x), 0.6, y_min]

    lb = [0.0,  x_min, 1e-4, y_min - 0.8]
    ub = [max(0.5, (y_max - y_min) * 3.0), x_max, 10.0, y_max + 0.8]

    popt, _ = curve_fit(
        sigmoid, x, y, p0=p0, bounds=(lb, ub),
        sigma=sigma,
        absolute_sigma=(sigma is not None),
        method='trf', maxfev=80000
    )
    return popt


# ============================================================
# 3. 自动检测饱和带宽
#    - 排除凹陷区间后，对大消息段取 p85
#    - 对于 Group3，需要检测"跌落后的低饱和"：取最末尾数据
# ============================================================
def detect_saturation_bw(
    df,
    size_col='size', bw_col='bus_bw',
    sat_start_quantile=0.80,
    exclude_dip_range=None,
    tail_pct=85,         # 用 p85 估计饱和（鲁棒排除凹陷）
    min_points=20,
):
    d = df.sort_values(size_col).copy()
    if exclude_dip_range is not None:
        ds, de = exclude_dip_range
        d = d[~((d[size_col] >= ds) & (d[size_col] <= de))]

    size_thresh = float(np.quantile(d[size_col], sat_start_quantile))
    d_large = d[d[size_col] >= size_thresh]
    if len(d_large) < min_points:
        d_large = d.tail(min_points)

    sat_bw = float(np.percentile(d_large[bw_col].to_numpy(), tail_pct))

    # 找首次稳定达到 sat_bw * 0.95 的消息大小
    bw_threshold = sat_bw * 0.95
    win = 10
    d_all = df.sort_values(size_col).copy()
    bw_roll = d_all[bw_col].rolling(win, min_periods=1).median().to_numpy()
    sizes   = d_all[size_col].to_numpy()
    sat_size = int(sizes[-1])   # fallback: 最大消息
    for i, (sz, bw) in enumerate(zip(sizes, bw_roll)):
        if bw >= bw_threshold:
            ahead = bw_roll[i:i+win]
            if len(ahead) >= win // 2 and np.median(ahead) >= bw_threshold:
                sat_size = int(sz)
                break

    return sat_bw, sat_size


# ============================================================
# 4. 检测 Group3 的"跌落点"与"低饱和带宽"
#    策略：纯数据驱动，不依赖趋势模型
#    步骤：
#      1. 对大消息段做滑动中位数平滑
#      2. 找到峰值位置（高平台顶点）
#      3. 从峰值之后，找到带宽持续稳定在某低平台的起始点
#      4. 低平台带宽 = 末尾数据的 p85（排除凹陷噪声）
# ============================================================
def detect_low3_structure(
    low_df,
    sat_bw_mid_ref,          # Group2 饱和带宽参考值（用于验证检测结果合理性）
    search_start_bytes=32 * 1024 * 1024,   # 从 32MiB 开始搜
    smooth_win=25,
    # 跌落判定：峰值后带宽下降超过 peak * peak_drop_ratio 且持续 persist 个点
    peak_drop_ratio=0.08,
    persist=15,
    # 低平台判定：末尾数据稳定性
    tail_quantile=0.70,      # 取末尾 30% 数据判断低饱和
    tail_pct=85,
    min_points=20,
):
    """
    返回: (dip_range, drop_size, sat_bw_low3)
      dip_range  : (dip_start, dip_end) 或 None
      drop_size  : 跌落稳定后的起始 size，之后使用 sat_bw_low3
      sat_bw_low3: Group3 末段稳定带宽（≈ Group2 饱和带宽）
    """
    d = low_df[low_df['size'] >= search_start_bytes].sort_values('size').copy().reset_index(drop=True)
    if len(d) < min_points * 2:
        return None, None, sat_bw_mid_ref

    bw_smooth = pd.Series(d['bus_bw']).rolling(smooth_win, center=True, min_periods=1).median().to_numpy()
    sizes = d['size'].to_numpy()

    # ── 步骤1：找峰值（高平台顶点） ───────────────────────
    peak_idx = int(np.argmax(bw_smooth))
    peak_bw  = float(bw_smooth[peak_idx])

    # ── 步骤2：检测末尾低平台带宽 ─────────────────────────
    # 取消息大小 top (1-tail_quantile) 的数据，计算稳定带宽
    size_tail_thresh = float(np.quantile(sizes, tail_quantile))
    tail_mask = sizes >= size_tail_thresh
    if tail_mask.sum() < min_points:
        tail_mask = np.zeros(len(sizes), dtype=bool)
        tail_mask[-min_points:] = True
    sat_bw_low3 = float(np.percentile(d['bus_bw'].to_numpy()[tail_mask], tail_pct))

    # 验证合理性：低饱和带宽应明显低于峰值（否则不存在跌落）
    if sat_bw_low3 >= peak_bw * 0.92:
        # 没有明显跌落，Group3 直接饱和
        return None, None, sat_bw_low3

    # ── 步骤3：在峰值之后找 dip 起始和跌落稳定点 ──────────
    drop_threshold  = peak_bw * (1.0 - peak_drop_ratio)   # 低于此值算进入凹陷/跌落
    stable_threshold = sat_bw_low3 * 1.05                  # 稳定在低平台的判定上界

    dip_start_idx  = None
    drop_size      = None

    cnt_stable = 0
    for i in range(peak_idx, len(bw_smooth)):
        b = bw_smooth[i]
        # 找凹陷起始：峰值后第一个明显下降点
        if dip_start_idx is None and b < drop_threshold:
            dip_start_idx = i

        # 找跌落后稳定低平台的起始：带宽稳定 <= stable_threshold 且持续 persist 点
        if dip_start_idx is not None and b <= stable_threshold:
            cnt_stable += 1
            if cnt_stable >= persist and drop_size is None:
                drop_size = int(sizes[i - persist + 1])
        else:
            # 若带宽回升到高于 stable_threshold，重置计数（排除短暂回落）
            if dip_start_idx is not None and b > stable_threshold * 1.02:
                cnt_stable = 0

    # ── 步骤4：确定 dip 范围 ─────────────────────────────
    dip_range = None
    if dip_start_idx is not None and drop_size is not None:
        dip_start = int(sizes[dip_start_idx])
        # dip 结束 = 跌落稳定点之前（dip 是从下降开始到低平台稳定前的震荡区间）
        dip_end   = drop_size
        if dip_end > dip_start:
            dip_range = (dip_start, dip_end)

    print(f"[Group3] peak={peak_bw:.1f} GB/s @ {sizes[peak_idx]/1024/1024:.0f} MiB, "
          f"sat_low={sat_bw_low3:.1f} GB/s, "
          f"dip={f'{dip_range[0]//1024//1024}~{dip_range[1]//1024//1024} MiB' if dip_range else 'None'}, "
          f"drop_size={f'{drop_size//1024//1024} MiB' if drop_size else 'None'}")

    return dip_range, drop_size, sat_bw_low3


# ============================================================
# 5. 鲁棒趋势拟合（凹陷点权重=0，大消息段加权锚定饱和）
# ============================================================
def robust_trend_fit_sigmoid(
    df,
    drop_ratio=0.06,
    keep_quantile=0.92,
    max_iter=5,
    sat_bw_anchor=None,
    anchor_weight=5.0,
):
    d = df.sort_values('size').copy().reset_index(drop=True)

    if sat_bw_anchor is None:
        sat_bw_anchor = float(np.percentile(d['bus_bw'].to_numpy(), 90))

    popt = fit_sigmoid_logbw(d, bw_cap=sat_bw_anchor * 1.05)
    x = np.log2(d['size'].to_numpy(np.float64))
    y = np.log10(d['bus_bw'].to_numpy(np.float64))
    bw = d['bus_bw'].to_numpy(np.float64)

    for _ in range(max_iter):
        y_hat = sigmoid(x, *popt)
        bw_hat = 10 ** y_hat
        drop = (bw_hat - bw) / np.maximum(bw_hat, 1e-12)

        dip_mask  = drop > drop_ratio
        keep_mask = drop <= float(np.quantile(drop, keep_quantile))
        main_mask = (~dip_mask) & keep_mask

        if main_mask.sum() < 30:
            break

        w = np.zeros_like(y)
        w[main_mask] = 1.0
        large_mask = x >= np.quantile(x, 0.80)
        w[large_mask & main_mask] *= anchor_weight
        sigma = np.where(w > 0, 1.0 / np.sqrt(w + 1e-9), 1e6)

        popt = fit_sigmoid_logbw(d, sigma=sigma, p0=list(popt),
                                 bw_cap=sat_bw_anchor * 1.05)
    return popt


# ============================================================
# 6. 凹陷检测（基于趋势残差的最长连续负偏段）
# ============================================================
def detect_dip_range(
    df, popt_trend,
    min_size=32 * 1024 * 1024,
    rel_drop=0.06,
    smooth_win=21,
    min_points=30,
):
    d = df[df['size'] >= min_size].sort_values('size').copy()
    if len(d) < min_points:
        return None

    x = np.log2(d['size'].to_numpy(np.float64))
    y = d['bus_bw'].to_numpy(np.float64)
    y_hat = 10 ** sigmoid(x, *popt_trend)
    drop = (y_hat - y) / np.maximum(y_hat, 1e-12)

    if smooth_win >= 3 and smooth_win < len(drop):
        s = pd.Series(drop).rolling(smooth_win, center=True, min_periods=1).median().to_numpy()
    else:
        s = drop

    mask  = s >= rel_drop
    best  = None
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if not m else i
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
            start = None

    if best is None or (best[1] - best[0] + 1) < min_points:
        return None

    return int(d['size'].iloc[best[0]]), int(d['size'].iloc[best[1]])


# ============================================================
# 7. sigmoid + Gaussian dip 拟合
# ============================================================
def sigmoid_with_gaussian_dip(x, L, x0, k, b, A, mu, sig):
    return sigmoid(x, L, x0, k, b) - A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def fit_sigmoid_plus_dip(df, popt_trend, dip_range):
    d = df.sort_values('size').copy()
    x = np.log2(d['size'].to_numpy(np.float64))
    y = np.log10(d['bus_bw'].to_numpy(np.float64))

    dip_start, dip_end = dip_range
    mu0  = np.log2((dip_start + dip_end) / 2.0)
    sig0 = max(0.2, (np.log2(dip_end) - np.log2(dip_start)) / 3.0)

    in_dip = (d['size'] >= dip_start) & (d['size'] <= dip_end)
    xd = np.log2(d.loc[in_dip, 'size'].to_numpy(np.float64))
    yd = np.log10(d.loc[in_dip, 'bus_bw'].to_numpy(np.float64))
    yhatd = sigmoid(xd, *popt_trend)
    A0 = max(0.02, float(np.max(yhatd - yd)))

    p0 = list(popt_trend) + [A0, mu0, sig0]
    y_min, y_max = float(y.min()), float(y.max())
    lb = [0.0, x.min(), 1e-4, y_min - 1.0, 0.0, np.log2(dip_start) - 0.6, 0.05]
    ub = [5.0, x.max(), 10.0, y_max + 1.0, 2.0, np.log2(dip_end)   + 0.6, 3.0]

    popt, _ = curve_fit(
        sigmoid_with_gaussian_dip, x, y,
        p0=p0, bounds=(lb, ub), method='trf', maxfev=200000
    )
    return popt


# ============================================================
# 8. Split 点（两条趋势曲线差异 > ratio 的首点）
# ============================================================
def find_diverge_point_by_ratio(
    popt_high, popt_low,
    size_min, size_max,
    ratio=0.05, persist=64, num_samples=4096,
):
    xs = np.linspace(np.log2(size_min), np.log2(size_max), num_samples)
    bh = 10 ** sigmoid(xs, *popt_high)
    bl = 10 ** sigmoid(xs, *popt_low)
    delta = (bh - bl) / np.maximum(bh, 1e-12)
    cnt = 0
    for i, ok in enumerate(delta >= ratio):
        cnt = (cnt + 1) if ok else 0
        if cnt >= persist:
            return float(xs[i - persist + 1]), int(round(2 ** float(xs[i - persist + 1])))
    return None, None


# ============================================================
# 9. 主入口
# ============================================================
def run_two_stage_piecewise(
    file_path,
    num_gpus=8,
    out_png='two_stage_piecewise.png',
    split_ratio=0.05,
):
    raw_df = parse_hccl_output(file_path)
    if raw_df is None:
        raise FileNotFoundError(file_path)

    mul_factor = MUL_FACTOR_FUNCS['all_to_all'](num_gpus)
    raw_df['bus_bw'] = raw_df['alg_bw'] * mul_factor
    raw_df = raw_df.sort_values('size').reset_index(drop=True)

    # ── 数据分三类 ──────────────────────────────────────────
    high_df = raw_df[raw_df['size'] % ALIGNMENT_HIGH == 0].copy().reset_index(drop=True)
    mid_df  = raw_df[(raw_df['size'] % ALIGNMENT_HIGH != 0) &
                     (raw_df['size'] % ALIGNMENT_MID  == 0)].copy().reset_index(drop=True)
    low_df  = raw_df[ raw_df['size'] % ALIGNMENT_MID  != 0 ].copy().reset_index(drop=True)

    print(f"[Data] Group1(128KB): {len(high_df)}, Group2(32KB only): {len(mid_df)}, "
          f"Group3(unaligned): {len(low_df)}")

    if len(high_df) == 0:
        raise ValueError("Group1 (128KB aligned) is empty.")

    # mid 或 low 若为空则用对方代替（至少保证后续不崩溃）
    mid_fit_df = mid_df if len(mid_df) > 20 else (low_df if len(low_df) > 20 else high_df)
    low_fit_df = low_df if len(low_df) > 20 else None   # Group3 可能不存在

    # ── Step 1: 粗拟合 + 凹陷粗定位 + 饱和带宽估计 ────────
    # Group1
    popt_high_init  = fit_sigmoid_logbw(high_df)
    dip_high_rough  = detect_dip_range(high_df, popt_high_init,
                                       min_size=32*1024*1024, rel_drop=0.03)
    sat_bw_high, sat_size_high = detect_saturation_bw(
        high_df, exclude_dip_range=dip_high_rough)

    # Group2
    popt_mid_init  = fit_sigmoid_logbw(mid_fit_df)
    dip_mid_rough  = detect_dip_range(mid_fit_df, popt_mid_init,
                                      min_size=32*1024*1024, rel_drop=0.04)
    sat_bw_mid, sat_size_mid = detect_saturation_bw(
        mid_fit_df, exclude_dip_range=dip_mid_rough)

    print(f"[Sat] Group1(high)={sat_bw_high:.2f} GB/s  Group2(mid)={sat_bw_mid:.2f} GB/s")

    # ── Step 2: 鲁棒趋势拟合 ───────────────────────────────
    popt_high_trend = robust_trend_fit_sigmoid(
        high_df, drop_ratio=0.04, sat_bw_anchor=sat_bw_high)
    popt_mid_trend  = robust_trend_fit_sigmoid(
        mid_fit_df, drop_ratio=0.06, sat_bw_anchor=sat_bw_mid)

    # ── Step 3: Split 点（Group1 vs Group2 趋势分叉） ──────
    size_min = max(int(raw_df['size'].min()), 4 * 1024 * 1024)
    size_max = int(raw_df['size'].max())
    split_x, split_size = find_diverge_point_by_ratio(
        popt_high_trend, popt_mid_trend,
        size_min=size_min, size_max=size_max,
        ratio=split_ratio, persist=64)
    if split_size is None:
        split_size = 32 * 1024 * 1024
        split_x    = np.log2(split_size)
        print("[Split] Not found; fallback to 32MiB.")
    print(f"[Split] Group1 vs Group2 @ {split_size/1024/1024:.2f} MiB")

    # ── Step 4: 精细凹陷拟合 ───────────────────────────────
    dip_high = detect_dip_range(high_df, popt_high_trend,
                                min_size=32*1024*1024, rel_drop=0.03)
    dip_mid  = detect_dip_range(mid_fit_df, popt_mid_trend,
                                min_size=32*1024*1024, rel_drop=0.04)

    popt_high_final = popt_high_trend
    popt_mid_final  = popt_mid_trend

    if dip_high is not None:
        print(f"[Dip High] {dip_high[0]//1024//1024}~{dip_high[1]//1024//1024} MiB")
        popt_high_final = fit_sigmoid_plus_dip(high_df, popt_high_trend, dip_high)
    if dip_mid is not None:
        print(f"[Dip Mid]  {dip_mid[0]//1024//1024}~{dip_mid[1]//1024//1024} MiB")
        popt_mid_final  = fit_sigmoid_plus_dip(mid_fit_df, popt_mid_trend, dip_mid)

    # ── Step 5: Group3 结构检测（纯数据驱动） ──────────────
    low3_drop_size = None
    dip_low3       = None
    sat_bw_low3    = sat_bw_mid   # 默认用 Group2 饱和带宽
    if low_fit_df is not None:
        dip_low3, low3_drop_size, sat_bw_low3 = detect_low3_structure(
            low_fit_df,
            sat_bw_mid_ref=sat_bw_mid,
            search_start_bytes=32 * 1024 * 1024,
            smooth_win=25,
            peak_drop_ratio=0.08,
            persist=15,
        )

    # ── Step 6: eval_model ────────────────────────────────
    def _pred_sigmoid(size_bytes, popt):
        return float(10 ** sigmoid(np.log2(float(size_bytes)), *popt))

    def _pred_dip(size_bytes, popt):
        return float(10 ** sigmoid_with_gaussian_dip(
            np.log2(float(size_bytes)), *popt))

    def eval_model(size_bytes: int, group: str = None) -> float:
        """
        group: 'high'(128KB) / 'mid'(32KB only) / 'low'(unaligned)
               若为 None，则根据整除性自动判断
        """
        if group is None:
            if size_bytes % ALIGNMENT_HIGH == 0:
                group = 'high'
            elif size_bytes % ALIGNMENT_MID == 0:
                group = 'mid'
            else:
                group = 'low'

        x = np.log2(float(size_bytes))

        # ── Group1 ─────────────────────────────────────────
        if group == 'high':
            if size_bytes < split_size:
                # 上升段统一用 high_trend
                return _pred_sigmoid(size_bytes, popt_high_trend)
            # 饱和段
            if dip_high is not None and dip_high[0] <= size_bytes <= dip_high[1]:
                # 凹陷区间：使用带 dip 的模型，允许下降
                return _pred_dip(size_bytes, popt_high_final)
            # 非凹陷区间：clip 到饱和带宽
            pred = (_pred_dip(size_bytes, popt_high_final)
                    if len(popt_high_final) == 7
                    else _pred_sigmoid(size_bytes, popt_high_final))
            return float(min(pred, sat_bw_high))

        # ── Group2 ─────────────────────────────────────────
        elif group == 'mid':
            if size_bytes < split_size:
                # 小消息阶段与 Group1 趋势一致（由 split_size 决定何时分叉）
                return _pred_sigmoid(size_bytes, popt_high_trend)
            if dip_mid is not None and dip_mid[0] <= size_bytes <= dip_mid[1]:
                return _pred_dip(size_bytes, popt_mid_final)
            pred = (_pred_dip(size_bytes, popt_mid_final)
                    if len(popt_mid_final) == 7
                    else _pred_sigmoid(size_bytes, popt_mid_final))
            return float(min(pred, sat_bw_mid))

        # ── Group3 ─────────────────────────────────────────
        else:  # 'low'
            # 跌落稳定后：直接返回低饱和带宽
            if low3_drop_size is not None and size_bytes >= low3_drop_size:
                return float(sat_bw_low3)

            # 跌落前：完全复用 Group1 最终模型（趋势相同）
            pred = (_pred_dip(size_bytes, popt_high_final)
                    if len(popt_high_final) == 7
                    else _pred_sigmoid(size_bytes, popt_high_trend))
            return float(min(pred, sat_bw_high))

    # ── Step 7: 三张独立子图 ──────────────────────────────
    x_fit = np.logspace(np.log10(raw_df['size'].min()),
                        np.log10(raw_df['size'].max()), 800)
    xlog2 = np.log2(x_fit)

    n_plots = 3 if low_fit_df is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6),
                             sharey=False)
    if n_plots == 2:
        axes = list(axes) + [None]   # 保持索引一致

    # ── 共用的辅助函数 ────────────────────────────────────
    def _add_split_dip_sat(ax, dip, sat_bw, sat_color):
        """画饱和水平线、split 竖线、dip 色块"""
        ax.axhline(sat_bw, color=sat_color, ls=':', lw=1.8, alpha=0.75,
                   label=f'Sat={sat_bw:.1f} GB/s')
        
        if dip is not None:
            ax.axvspan(dip[0], dip[1], alpha=0.10, color=sat_color,
                       label=f'Dip {dip[0]//1024//1024}~{dip[1]//1024//1024} MiB')

    def _fmt(ax, title):
        ax.set_xscale('log', base=2)
        ax.grid(True, which='both', ls='-', alpha=0.2)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Message Size (Bytes)')
        ax.set_ylabel('Bus BW (GB/s)')
        ax.legend(loc='upper left', fontsize=8)

    # ── 子图 1: Group1 (128KB aligned) ───────────────────
    ax1 = axes[0]
    ax1.scatter(high_df['size'], high_df['bus_bw'],
                s=10, alpha=0.25, color='#ff9896', label='Raw G1', zorder=2)
    ax1.plot(x_fit, 10 ** sigmoid(xlog2, *popt_high_trend),
             color='#d62728', ls='--', lw=2.0, label='Trend (robust)', zorder=3)
    if len(popt_high_final) == 7:
        ax1.plot(x_fit, 10 ** sigmoid_with_gaussian_dip(xlog2, *popt_high_final),
                 color='#880000', ls='-', lw=2.2, alpha=0.85, label='Final (+dip)', zorder=4)
    yh = np.array([eval_model(int(v), 'high') for v in x_fit])
    ax1.plot(x_fit, yh, color='orange', lw=2.8, label='Piecewise model', zorder=5)
    _add_split_dip_sat(ax1, dip_high, sat_bw_high, '#d62728')
    _fmt(ax1, f'Group1: 128KB Aligned  (sat≈{sat_bw_high:.0f} GB/s)')

    # ── 子图 2: Group2 (32KB only) ────────────────────────
    ax2 = axes[1]
    ax2.scatter(mid_fit_df['size'], mid_fit_df['bus_bw'],
                s=10, alpha=0.25, color='#aec7e8', label='Raw G2', zorder=2)
    ax2.plot(x_fit, 10 ** sigmoid(xlog2, *popt_mid_trend),
             color='#1f77b4', ls='--', lw=2.0, label='Trend (robust)', zorder=3)
    if len(popt_mid_final) == 7:
        ax2.plot(x_fit, 10 ** sigmoid_with_gaussian_dip(xlog2, *popt_mid_final),
                 color='#003888', ls='-', lw=2.2, alpha=0.85, label='Final (+dip)', zorder=4)
    ym = np.array([eval_model(int(v), 'mid') for v in x_fit])
    ax2.plot(x_fit, ym, color='#ff7f0e', lw=2.8, label='Piecewise model', zorder=5)
    _add_split_dip_sat(ax2, dip_mid, sat_bw_mid, '#1f77b4')
    _fmt(ax2, f'Group2: 32KB-Only Aligned  (sat≈{sat_bw_mid:.0f} GB/s)')

    # ── 子图 3: Group3 (unaligned) ────────────────────────
    if axes[2] is not None and low_fit_df is not None:
        ax3 = axes[2]
        ax3.scatter(low_fit_df['size'], low_fit_df['bus_bw'],
                    s=10, alpha=0.25, color='#98df8a', label='Raw G3', zorder=2)
        # 参考线：G1 趋势（上升段一致）
        ax3.plot(x_fit, 10 ** sigmoid(xlog2, *popt_high_trend),
                 color='#d62728', ls='--', lw=1.6, alpha=0.6,
                 label=f'G1 trend ref (sat≈{sat_bw_high:.0f})', zorder=3)
        # 高饱和参考线
        ax3.axhline(sat_bw_high, color='#d62728', ls=':', lw=1.5, alpha=0.5,
                    label=f'G1 sat ref={sat_bw_high:.1f} GB/s')
        # 低饱和参考线（检测到的 sat_bw_low3）
        ax3.axhline(sat_bw_low3, color='#2ca02c', ls=':', lw=1.8, alpha=0.85,
                    label=f'G3 sat={sat_bw_low3:.1f} GB/s')
        # Piecewise 模型曲线：两段
        #  段1: drop_size 之前 -> 完全复用 Group1 最终模型
        #  段2: drop_size 之后 → 水平线 sat_bw_low3
        drop_s = low3_drop_size

        x_before = x_fit if drop_s is None else x_fit[x_fit < drop_s]
        if len(x_before) > 0:
            yl_before = np.array([eval_model(int(v), 'low') for v in x_before])
            ax3.plot(x_before, yl_before,
                     color='#2ca02c', lw=2.8, label='Piecewise model (G1 ref)', zorder=5)

        if drop_s is not None:
            x_after = x_fit[x_fit >= drop_s]
            ax3.plot(x_after, np.full(len(x_after), sat_bw_low3),
                     color='#2ca02c', lw=2.8, label=f'Sat plateau={sat_bw_low3:.1f} GB/s', zorder=5)
        # Split 线
        
        # Dip 区间（数据驱动检测到的）
        if dip_low3 is not None:
            ax3.axvspan(dip_low3[0], dip_low3[1], alpha=0.12, color='#ff7f0e',
                        label=f'Dip {dip_low3[0]//1024//1024}~{dip_low3[1]//1024//1024} MiB')
        # 跌落稳定竖线
        if low3_drop_size is not None:
            ax3.axvline(low3_drop_size, ls='-.', lw=2.2, color='#d62728', alpha=0.9,
                        label=f'Drop stable @ {low3_drop_size/1024/1024:.0f} MiB')
        _fmt(ax3, f'Group3: Unaligned  (peak→dip→sat≈{sat_bw_low3:.0f} GB/s)')

    fig.suptitle('All-to-All: Per-Group Bandwidth Model', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

    return {
        "popt_high_trend":  popt_high_trend,
        "popt_mid_trend":   popt_mid_trend,
        "popt_high_final":  popt_high_final,
        "popt_mid_final":   popt_mid_final,
        "sat_bw_high":      sat_bw_high,
        "sat_bw_mid":       sat_bw_mid,
        "sat_size_high":    sat_size_high,
        "sat_size_mid":     sat_size_mid,
        "split_size":       split_size,
        "dip_high":         dip_high,
        "dip_mid":          dip_mid,
        "dip_low3":         dip_low3,
        "sat_bw_low3":      sat_bw_low3,
        "low3_drop_size":   low3_drop_size,
        "eval_model":       eval_model,
    }


if __name__ == "__main__":
    run_two_stage_piecewise(
        file_path="data/raw/a2a_32kb_aligned.txt",
        num_gpus=8,
        out_png="artifacts/plots/bw_model_fit.png",
        split_ratio=0.03,
    )
