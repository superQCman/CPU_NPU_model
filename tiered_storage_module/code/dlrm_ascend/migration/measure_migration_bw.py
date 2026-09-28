#!/usr/bin/env python3
"""
测量各迁移链路的有效带宽, 验证解析模型 T = α·n + DataSize/BW。

迁移链路 (三级存储):
  1. HBM -> DRAM  (device -> host, PCIe)   —— 淘汰
  2. DRAM -> HBM  (host -> device, PCIe)   —— 晋升
  3. SSD -> DRAM  (NVMe 随机读)            —— 预取 (用 ssd_model.py 的 fio 标定)

关键点:
  - device<->host 拷贝在 NPU/CUDA 上是异步的, time.time() 只捕获派发;
    必须在计时前后各 synchronize 一次, 才能测到真实传输时间。
  - 用"一个大 tensor"批量拷, 摊薄固定开销, 拟合 T = α + S/BW,
    斜率 = 1/BW, 截距 = α。R²≈1 说明"数据量/带宽"这个线性模型成立。
  - SSD 用 fio 随机读 (QD=1, --direct=1) 测单次迁移耗时, 绕开 OS 页缓存;
    需要被测 SSD 上有一个测试文件 (--ssd-file, 建议 fallocate -l 20G 建)。

输出:
  - migration_bw_result.json  测量结果 (BW/α/R²/扫描点), 供 plot_migration_bw.py 画图

用法:
    fallocate -l 20G ~/testfile        # 先建 SSD 测试文件 (一次即可)
    python measure_migration_bw.py --sizes 1 2 4 8 16 32 64 128 256 512 1024
"""

import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch


# ----------------------------------------------------------------------
# 设备工具
# ----------------------------------------------------------------------
def get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------------------------------------------------------------
# 拷贝微基准 (含同步)
# ----------------------------------------------------------------------
def time_bulk_copy(src_device: str, dst_device: str, n_bytes: int, n_repeat: int = 3) -> float:
    """测一次 src->dst 大块拷贝的真实耗时 (s)。"""
    n_floats = max(1, n_bytes // 4)
    src = torch.randn(n_floats, device=src_device)  # 分配在计时区外
    times = []
    for _ in range(n_repeat):
        device_sync()
        t0 = time.time()
        dst = src.to(dst_device)                     # 只有这一步计拷贝
        device_sync()
        t1 = time.time()
        times.append(t1 - t0)
    return float(np.mean(times))


def fit_bw(sizes_bytes, times_s):
    """线性拟合 T = α + S/BW。返回 (BW bytes/s, α s, R²)。"""
    x = np.asarray(sizes_bytes, dtype=np.float64)
    y = np.asarray(times_s, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)   # slope = 1/BW, intercept = α
    bw = 1.0 / slope if slope > 1e-12 else float("inf")
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return bw, intercept, r2


def measure_link(direction: str, sizes_bytes, n_repeat: int) -> dict:
    """扫描多个大小, 拟合某条 PCIe 链路的 BW/α/R²。direction: 'd2h' 或 'h2d'。"""
    if direction == "d2h":          # HBM->DRAM: device -> host
        src, dst = get_device(), "cpu"
    else:                            # DRAM->HBM: host -> device
        src, dst = "cpu", get_device()
    times = [time_bulk_copy(src, dst, nb, n_repeat) for nb in sizes_bytes]
    bw, alpha, r2 = fit_bw(sizes_bytes, times)
    return {"direction": direction, "bw_bytes_s": bw, "alpha_s": alpha,
            "r2": r2, "sizes_bytes": sizes_bytes, "times_s": times}


def _bytes_to_fio_bs(n_bytes):
    """字节 -> fio bs 字符串 (如 1048576 -> '1m', 1073741824 -> '1g')。"""
    for unit, size in (("g", 1024 ** 3), ("m", 1024 ** 2), ("k", 1024)):
        if n_bytes % size == 0:
            return f"{n_bytes // size}{unit}"
    return str(n_bytes)


def _fio_randread_clat_s(bs_bytes, file_path, runtime):
    """用 fio 随机读 (QD=1, --direct=1) 测一次 bs 大小请求的平均完成延迟 (秒)。"""
    cmd = [
        "fio", "--name=fio_ssd", "--rw=randread",
        f"--bs={_bytes_to_fio_bs(bs_bytes)}",
        "--iodepth=1", "--numjobs=1",
        "--size=8G", f"--runtime={runtime}", "--time_based",
        f"--filename={file_path}", "--direct=1",
        "--group_reporting=1", "--output-format=json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fio 失败 (bs={_bytes_to_fio_bs(bs_bytes)}): "
                           f"{proc.stderr.strip()}")
    d = json.loads(proc.stdout)
    for job in d.get("jobs", []):
        rd = job.get("read")
        if rd and "clat_ns" in rd:
            return rd["clat_ns"]["mean"] / 1e9  # ns -> s
    raise RuntimeError(f"fio 输出缺 read.clat_ns (bs={_bytes_to_fio_bs(bs_bytes)})")


def measure_fio_randread(sizes_bytes, file_path="~/testfile", runtime=15) -> dict:
    """用 fio 随机读 (QD=1, --direct=1) 测 SSD->DRAM 单次迁移的有效带宽。

    与 measure_link 返回同结构 (direction='ssd')。每档大小跑一次 fio,
    取平均 clat 作为该档「单次迁移」的耗时 T, 再拟合 T = α + S/BW。

    要求: 被测 SSD 上存在 file_path (建议 fallocate -l 20G 建), 且装了 fio。
    注: 这是真实 NVMe 随机读 (绕开页缓存), 取代原来的 memmap 读。
    """
    file_path = os.path.expanduser(file_path)
    if shutil.which("fio") is None:
        raise RuntimeError("未找到 fio 命令, 请先安装 fio")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"SSD 测试文件不存在: {file_path}\n"
            f"  请先在被测 SSD 上创建, 例如: fallocate -l 20G {file_path}")
    times = [_fio_randread_clat_s(nb, file_path, runtime) for nb in sizes_bytes]
    bw, alpha, r2 = fit_bw(sizes_bytes, times)
    return {"direction": "ssd", "bw_bytes_s": bw, "alpha_s": alpha,
            "r2": r2, "sizes_bytes": sizes_bytes, "times_s": times}


# ----------------------------------------------------------------------
# 解析模型预测
# ----------------------------------------------------------------------
def migration_time_s(n_rows: int, dim: int, byte_per_elem: int, bw_bytes_s: float,
                     alpha_s: float = 0.0) -> float:
    """T = α·n + (n·dim·b)/BW。返回秒。"""
    data_bytes = n_rows * dim * byte_per_elem
    return alpha_s * n_rows + data_bytes / bw_bytes_s


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="测量迁移链路有效带宽 + 验证解析模型")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100, 512],
                        help="扫描的数据大小 (MB)")
    parser.add_argument("--repeat", type=int, default=3, help="每个大小重复次数")
    parser.add_argument("--dim", type=int, default=64000, help="embedding 维度")
    parser.add_argument("--byte-per-elem", type=int, default=4, help="每元素字节 (float32=4)")
    parser.add_argument("--ssd-file", default="~/testfile",
                        help="fio 随机读的测试文件 (被测 SSD 上)")
    parser.add_argument("--ssd-runtime", type=int, default=15,
                        help="每档大小 fio 运行秒数")
    args = parser.parse_args()

    dev = get_device()
    print(f"[device] {dev}")
    if dev == "cpu":
        print("  [warn] 无 NPU/CUDA 设备, PCIe 带宽测不出。")

    sizes_bytes = [s * 1024 * 1024 for s in args.sizes]
    results = {}

    # ---- PCIe 两条链路 ----
    for direction, name in [("d2h", "HBM->DRAM (device->host)"),
                            ("h2d", "DRAM->HBM (host->device)")]:
        r = measure_link(direction, sizes_bytes, args.repeat)
        results[direction] = r
        print(f"\n[{name}]")
        print(f"  BW    = {r['bw_bytes_s']/1e9:.2f} GB/s")
        print(f"  alpha = {r['alpha_s']*1e6:.2f} us")
        print(f"  R²    = {r['r2']:.5f}   (≈1 说明 T=α+S/BW 线性模型成立)")
        for nb, t in zip(r["sizes_bytes"], r["times_s"]):
            print(f"    {nb/1e6:7.0f} MB -> {t*1e3:8.3f} ms")

    # ---- SSD 链路: fio 随机读 (QD=1, --direct=1), 与 PCIe 同样的 sweep+fit ----
    r_ssd = measure_fio_randread(sizes_bytes, args.ssd_file, args.ssd_runtime)
    results["ssd"] = r_ssd
    print(f"\n[SSD->DRAM] (fio 随机读 QD=1 --direct=1)")
    print(f"  BW    = {r_ssd['bw_bytes_s']/1e9:.2f} GB/s")
    print(f"  alpha = {r_ssd['alpha_s']*1e6:.2f} us")
    print(f"  R²    = {r_ssd['r2']:.5f}   (≈1 说明 T=α+S/BW 线性模型成立)")
    for nb, t in zip(r_ssd["sizes_bytes"], r_ssd["times_s"]):
        print(f"    {nb/1e6:7.0f} MB -> {t*1e3:8.3f} ms")

    # ---- 参考: ssd_model.py 的 fio 解析模型 (供 prediction 图用) ----
    ssd_bw = 7.0e9
    try:
        import ssd_model
        bs = args.dim * args.byte_per_elem
        ssd_bw = ssd_model.effective_bw_bps(bs, 32)
        print(f"  [参考 fio] 单行 QD=1: {ssd_model.transfer_time_s(1, bs, 1)*1e3:.3f} ms/row, "
              f"QD=32: {ssd_model.transfer_time_s(1, bs, 32)*1e3:.3f} ms/row, "
              f"有效BW(QD=32)={ssd_bw/1e9:.2f} GB/s")
    except ImportError:
        print("  未找到 ssd_model.py, prediction 图用默认 BW=7 GB/s。")

    # ---- 迁移时间算例 ----
    if dev != "cpu" and "d2h" in results:
        bw_pcie = results["d2h"]["bw_bytes_s"]
        alpha = results["d2h"]["alpha_s"]
        print(f"\n[迁移时间算例] 用 PCIe BW={bw_pcie/1e9:.2f} GB/s 预测 (dim={args.dim})")
        print(f"  {'n_rows':>8s} {'data(MB)':>10s} {'HBM<->DRAM(ms)':>14s}")
        for n in [10, 100, 1000, 10000]:
            data_mb = n * args.dim * args.byte_per_elem / 1e6
            t_ms = migration_time_s(n, args.dim, args.byte_per_elem, bw_pcie, alpha) * 1e3
            print(f"  {n:8d} {data_mb:10.1f} {t_ms:14.3f}")

    # ---- 保存结果 ----
    save = {d: {"bw_bytes_s": r["bw_bytes_s"], "alpha_s": r["alpha_s"], "r2": r["r2"],
                "sizes_bytes": r["sizes_bytes"], "times_s": r["times_s"]}
            for d, r in results.items()}
    save["ssd_bw_bytes_s"] = ssd_bw
    with open("migration_bw_result.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n[save] -> migration_bw_result.json")


if __name__ == "__main__":
    main()
