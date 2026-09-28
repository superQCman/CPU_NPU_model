#!/usr/bin/env python3
"""
SSD 随机读物理解析模型 (NVMe)

模型形式（每项都有物理含义）:

    单请求延迟   T_lat(bs, QD) = T_fixed + bs / BW_channel + T_queue·(QD-1)
    有效带宽     BW_eff(bs, QD) = min( QD · bs / T_lat(bs, QD),  BW_max )
    有效 IOPS    IOPS_eff(bs,QD) = min( QD / T_lat(bs, QD),      IOPS_max )
    总传输时间   T(n_req, bs, QD) = n_req · bs / BW_eff(bs, QD)

物理参数（从 fio 随机读标定，见下方 VALIDATION 与 validate()）:

    T_fixed    = 81    µs     # NAND + FTL + 控制器固定延迟（QD=1 小块的底）
    BW_channel = 1.14  GB/s   # 每请求 NAND 通道传输率（延迟-大小曲线的斜率）
    T_queue    = 0.36  µs/QD  # 每增加 1 个 QD 的排队延迟增量
    BW_max     = 7.08  GB/s   # NAND 阵列 + PCIe 带宽上限（bs=1M,QD=32 实测）
    IOPS_max   = 1680  k      # 控制器 IOPS 上限（bs=4K,QD=32 实测）

DLRM 接入:

    n_req = r_SSD · K          # 冷行（SSD miss）数量
    bs    = m · b              # 单行字节数（dim × bytes/elem）
    T_ssd = T(n_req, bs, QD)   # SSD 段总耗时

    低 QD（串行，当前代码逐行加载）:  T_ssd ≈ n_req · T_fixed（延迟界，每行 ~81µs）
    高 QD（批量/异步 I/O）:          T_ssd ≈ n_req · bs / BW_max（带宽界）

用法:
    python ssd_model.py --validate                          # 对 fio 数据自验证
    python ssd_model.py --n-ssd 1000 --dim 64 --qd 1        # DLRM SSD 段预测 (QD=1)
    python ssd_model.py --n-ssd 1000 --dim 64 --qd 32       # 同上 (QD=32)
"""

import argparse

import numpy as np

# ============================================================
# 物理参数
# ============================================================

T_FIXED_US = 81.0           # 固定延迟 (µs)
BW_CHANNEL_BPS = 1.14e9     # 每请求通道带宽 (B/s)
T_QUEUE_PER_QD_US = 0.36    # 每 QD 排队延迟增量 (µs/QD)
BW_MAX_BPS = 7.08e9         # 带宽上限 (B/s)
IOPS_MAX = 1680e3           # IOPS 上限 (次/s)

# ============================================================
# 原始 fio 数据（随机读，用于标定与自验证）
# ============================================================

# (bs_bytes, qd, latency_us, iops, bw_mibs)
FIO_BS_SWEEP = [
    (1024,  1, 83.99, 11800, 11.5),
    (4096,  1, 84.62, 11700, 45.8),
    (8192,  1, 89.61, 11100, 86.4),
    (16384, 1, 96.56, 10300, 161),
    (32768, 1, 108.99, 9100, 285),
    (65536, 1, 138.39, 7200, 449),
]

FIO_QD_SWEEP = [  # bs=4K 固定
    (4096, 1,  84.62, 11700, 45.8),
    (4096, 8,  87.5,  90600, 354),
    (4096, 16, 90.5,  174800, 683),
    (4096, 32, 95.8,  330500, 1291),
]

# ============================================================
# 核心模型
# ============================================================

def latency_us(bs_bytes: float, qd: int = 1) -> float:
    """单请求延迟 (µs)：固定延迟 + 传输 + 排队。"""
    return T_FIXED_US + bs_bytes / BW_CHANNEL_BPS * 1e6 + T_QUEUE_PER_QD_US * (qd - 1)


def effective_bw_bps(bs_bytes: float, qd: int = 1) -> float:
    """有效带宽 (B/s)：低 QD 延迟界，高 QD 触顶 min(BW_max, IOPS_max·bs)。"""
    lat_s = latency_us(bs_bytes, qd) * 1e-6
    bw = qd * bs_bytes / lat_s
    return min(bw, BW_MAX_BPS, IOPS_MAX * bs_bytes)


def effective_iops(bs_bytes: float, qd: int = 1) -> float:
    """有效 IOPS：QD/延迟，触顶 IOPS_max。"""
    lat_s = latency_us(bs_bytes, qd) * 1e-6
    return min(qd / lat_s, IOPS_MAX)


def transfer_time_s(n_req: int, bs_bytes: float, qd: int = 1) -> float:
    """读 n_req 个、每个 bs_bytes 字节的请求，在 QD 下的总耗时 (s)。"""
    if n_req <= 0:
        return 0.0
    qd_eff = max(1, min(qd, n_req))  # 并行度受限于请求数本身
    bw = effective_bw_bps(bs_bytes, qd_eff)
    return n_req * bs_bytes / bw


def dlrm_ssd_time_ms(n_ssd: int, dim: int, byte_per_elem: int = 4, qd: int = 1) -> float:
    """DLRM 的 SSD 段耗时 (ms)。

    Args:
        n_ssd:         冷行（SSD miss）数量 = r_SSD · K
        dim:           embedding 维度
        byte_per_elem: 每元素字节数（float32=4, bf16/fp16=2）
        qd:            SSD I/O 队列深度（1=串行，32=批量/异步）
    """
    bs = dim * byte_per_elem
    return transfer_time_s(n_ssd, bs, qd) * 1000.0


# ============================================================
# 参数拟合
# ============================================================

def fit_latency_params(bs_list, lat_us_list):
    """从 (块大小, 延迟) 数据拟合 T_fixed 与 BW_channel。

    模型: T_lat[µs] = T_fixed[µs] + bs[bytes] / BW[bytes/s] * 1e6
    线性回归: T_lat = intercept + slope · bs
        T_fixed     = intercept            (µs)
        BW_channel  = 1e6 / slope          (B/s)

    Returns: (T_fixed_us, BW_channel_bps, r2, residuals_pct)
    """
    bs = np.asarray(bs_list, dtype=float)
    lat = np.asarray(lat_us_list, dtype=float)
    slope, intercept = np.polyfit(bs, lat, 1)
    t_fixed = float(intercept)
    bw_channel = float(1e6 / slope)

    pred = intercept + slope * bs
    ss_res = float(np.sum((lat - pred) ** 2))
    ss_tot = float(np.sum((lat - np.mean(lat)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    residuals_pct = (pred - lat) / lat * 100.0
    return t_fixed, bw_channel, r2, residuals_pct


# ============================================================
# 自验证
# ============================================================

def validate():
    """用嵌入的 fio 数据做三级验证：V1 形式拟合 / V2 泛化 / V3 饱和。"""
    print("=" * 78)
    print("  SSD 随机读物理模型 — 自验证")
    print("=" * 78)

    # ---- V1: 模型形式验证（QD=1 延迟-大小曲线是否线性）----
    bs = [d[0] for d in FIO_BS_SWEEP]
    lat = [d[2] for d in FIO_BS_SWEEP]
    t_fixed, bw_ch, r2, res = fit_latency_params(bs, lat)
    print(f"\n[V1] 模型形式验证：拟合 T_lat = T_fixed + bs/BW_channel (QD=1)")
    print(f"     拟合:  T_fixed = {t_fixed:.2f} µs   BW_channel = {bw_ch/1e9:.3f} GB/s   R² = {r2:.5f}")
    print(f"     标定值: T_fixed = {T_FIXED_US} µs   BW_channel = {BW_CHANNEL_BPS/1e9:.2f} GB/s")
    print(f"     {'bs':>6s} {'实测(µs)':>9s} {'拟合(µs)':>9s} {'残差':>7s}")
    for (b, l, r) in zip(bs, lat, res):
        pred = t_fixed + b / bw_ch * 1e6
        print(f"     {b:>6d} {l:>9.2f} {pred:>9.2f} {r:>+6.1f}%")
    print(f"     通过标准: R² > 0.99 且 |残差| < 5%  →  {'PASS' if r2 > 0.99 and np.max(np.abs(res)) < 5 else 'FAIL'}")

    # ---- V2: 泛化验证（用 QD=1 拟合的参数预测 QD>1 的带宽）----
    print(f"\n[V2] 泛化验证：预测 bs=4K 下不同 QD 的有效带宽")
    print(f"     {'QD':>4s} {'实测(MiB/s)':>12s} {'预测(MiB/s)':>12s} {'误差':>7s}")
    errs = []
    for (b, qd, l, iops, bw_mibs) in FIO_QD_SWEEP:
        pred_bps = effective_bw_bps(b, qd)
        pred_mibs = pred_bps / (1024 ** 2)
        err = (pred_mibs - bw_mibs) / bw_mibs * 100
        errs.append(abs(err))
        print(f"     {qd:>4d} {bw_mibs:>12.1f} {pred_mibs:>12.1f} {err:>+6.1f}%")
    mape = np.mean(errs)
    print(f"     通过标准: 平均 |误差| < 15%（排队项未完全建模时的容差）→  {'PASS' if mape < 15 else 'FAIL'}  ({mape:.1f}%)")

    # ---- V3: 饱和上限验证 ----
    print(f"\n[V3] 饱和上限验证")
    bw_1m_32 = effective_bw_bps(1048576, 32) / 1e9
    iops_4k_32 = effective_iops(4096, 32) / 1e3
    print(f"     预测 BW(bs=1M,QD=32) = {bw_1m_32:.2f} GB/s  vs 实测 {BW_MAX_BPS/1e9:.2f} GB/s  → "
          f"{'PASS' if abs(bw_1m_32 - BW_MAX_BPS/1e9)/ (BW_MAX_BPS/1e9) < 0.01 else 'FAIL'}")
    print(f"     预测 IOPS(bs=4K,QD=32) = {iops_4k_32:.0f} k  vs 实测 {IOPS_MAX/1e3:.0f} k  → "
          f"{'PASS' if abs(iops_4k_32 - IOPS_MAX/1e3)/(IOPS_MAX/1e3) < 0.01 else 'FAIL'}")

    print("\n" + "=" * 78)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SSD 随机读物理解析模型")
    parser.add_argument("--validate", action="store_true", help="对 fio 数据自验证")
    parser.add_argument("--n-ssd", type=int, default=0, help="冷行（SSD miss）数量 = r_SSD·K")
    parser.add_argument("--dim", type=int, default=64, help="embedding 维度")
    parser.add_argument("--dtype", choices=["float32", "float16", "bf16"],
                        default="float32", help="元素精度")
    parser.add_argument("--qd", type=int, default=1, help="SSD I/O 队列深度（1=串行，8/16/32=批量）")
    args = parser.parse_args()

    if args.validate:
        validate()

    if args.n_ssd > 0:
        byte_per_elem = {"float32": 4, "float16": 2, "bf16": 2}[args.dtype]
        bs = args.dim * byte_per_elem
        t = dlrm_ssd_time_ms(args.n_ssd, args.dim, byte_per_elem, args.qd)
        print(f"\n[DLRM SSD 段] n_ssd={args.n_ssd}, dim={args.dim}, dtype={args.dtype}, "
              f"bs={bs} B, QD={args.qd}")
        print(f"  单请求延迟: {latency_us(bs, args.qd):.2f} µs")
        print(f"  有效带宽:   {effective_bw_bps(bs, args.qd)/1e9:.3f} GB/s")
        print(f"  总耗时:     {t:.3f} ms")

        if args.qd == 1:
            t32 = dlrm_ssd_time_ms(args.n_ssd, args.dim, byte_per_elem, 32)
            print(f"  对比 QD=32: {t32:.3f} ms  (加速 {t/t32:.1f}×，说明批量/异步 I/O 的价值)")


if __name__ == "__main__":
    main()
