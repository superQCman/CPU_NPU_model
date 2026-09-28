#!/usr/bin/env python3
"""
测量「更新(写回)」各链路的有效写带宽, 验证解析模型 T = α·n + DataSize/BW。

更新(写回)链路 (训练场景, 自上而下):
  1. NPU -> DRAM  (device -> host, PCIe 写)      —— 温行写回
  2. DRAM -> SSD  (NVMe 随机写)                  —— 脏行批量刷回
  (NPU -> HBM 原地 read-modify-write 在片上完成, 零跨层流量, 不建模)

与迁移建模 (measure_migration_bw.py) 的对应关系:
  - NPU->DRAM 写回 = device->host, 物理链路与迁移的 d2h(淘汰 HBM->DRAM) 相同,
    可用同一份 d2h 带宽; 这里仍单独测一遍, 便于交叉验证写方向与读方向是否一致。
  - DRAM->SSD 写回 = NVMe 随机写。SSD 写 ≪ 读 (写放大 + 更慢的 NAND 编程),
    是训练的主要开销, 必须用 fio randwrite 单独标定, 不能复用 randread。

关键点 (同 measure_migration_bw.py):
  - device<->host 拷贝在 NPU/CUDA 上异步, time.time() 只捕获派发;
    计时前后各 synchronize 一次, 才能测到真实传输时间。
  - 用"一个大 tensor"批量写, 摊薄固定开销, 拟合 T = α + S/BW,
    斜率 = 1/BW, 截距 = α。R²≈1 说明"数据量/带宽"这个线性模型成立。
  - SSD 用 fio randwrite (QD 可调, --direct=1) 测单次写回耗时, 绕开 OS 页缓存。
    注意: fio 写任务的结果在 job["write"].clat_ns, 不同于读任务在 job["read"]。

输出:
  - update_bw_result.json  写链路拟合结果 (BW/α/R²/扫描点), 供 update_model.py 预测

用法:
    fallocate -l 20G ~/testfile        # 先建 SSD 测试文件 (一次即可)
    python measure_update_bw.py --sizes 1 2 4 8 16 32 64 128 256 512 1024
    # 脏行批量刷回场景 (高 QD):
    python measure_update_bw.py --ssd-qd 32
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
# 写回微基准 (含同步)
# ----------------------------------------------------------------------
def time_bulk_copy(src_device: str, dst_device: str, n_bytes: int, n_repeat: int = 3) -> float:
    """测一次 src->dst 大块拷贝的真实耗时 (s)。

    写回 NPU->DRAM 即 device->host 拷贝 (src=device, dst=cpu)。
    """
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
    """扫描多个大小, 拟合某条 PCIe 写链路的 BW/α/R²。direction: 'd2h'。"""
    if direction == "d2h":          # NPU->DRAM: device -> host
        src, dst = get_device(), "cpu"
    else:
        raise ValueError(f"未知 direction: {direction}")
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


def _fio_randwrite_clat_s(bs_bytes, file_path, runtime, qd=1):
    """用 fio 随机写 (QD 可调, --direct=1) 测一次 bs 大小请求的平均完成延迟 (秒)。

    注意: 写任务的结果在 job["write"].clat_ns, 不同于读任务在 job["read"]。
    """
    cmd = [
        "fio", "--name=fio_ssd_write", "--rw=randwrite",
        f"--bs={_bytes_to_fio_bs(bs_bytes)}",
        f"--iodepth={qd}", "--numjobs=1",
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
        wr = job.get("write")
        if wr and "clat_ns" in wr:
            return wr["clat_ns"]["mean"] / 1e9  # ns -> s
    raise RuntimeError(f"fio 输出缺 write.clat_ns (bs={_bytes_to_fio_bs(bs_bytes)})")


def measure_fio_randwrite(sizes_bytes, file_path="~/testfile", runtime=15, qd=1) -> dict:
    """用 fio 随机写 (QD 可调, --direct=1) 测 DRAM->SSD 单次写回的有效带宽。

    与 measure_link 返回同结构 (direction='ssd')。每档大小跑一次 fio,
    取平均 clat 作为该档「单次写回」的耗时 T, 再拟合 T = α + S/BW。

    要求: 被测 SSD 上存在 file_path (建议 fallocate -l 20G 建), 且装了 fio。
    注: randwrite 会真实改写测试文件, 建议单独建一个写测试文件, 不要与读测试混用。
    """
    file_path = os.path.expanduser(file_path)
    if shutil.which("fio") is None:
        raise RuntimeError("未找到 fio 命令, 请先安装 fio")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"SSD 测试文件不存在: {file_path}\n"
            f"  请先在被测 SSD 上创建, 例如: fallocate -l 20G {file_path}")
    times = [_fio_randwrite_clat_s(nb, file_path, runtime, qd) for nb in sizes_bytes]
    bw, alpha, r2 = fit_bw(sizes_bytes, times)
    return {"direction": "ssd", "bw_bytes_s": bw, "alpha_s": alpha,
            "r2": r2, "sizes_bytes": sizes_bytes, "times_s": times}


# ----------------------------------------------------------------------
# 解析模型预测
# ----------------------------------------------------------------------
def update_time_s(n_rows: int, dim: int, byte_per_elem: int, bw_bytes_s: float,
                  alpha_s: float = 0.0, n_batches: int = 1) -> float:
    """T = α·n_batches + (n_rows·dim·b)/BW。返回秒。

    批量写回: α 按拷贝批次计, 不按行计。n_rows 为已去重的总行数, n_batches 为
    打包成的 device->host / NVMe 拷贝次数 (单表一次=1, 多表逐表=表数)。
    """
    data_bytes = n_rows * dim * byte_per_elem
    return alpha_s * n_batches + data_bytes / bw_bytes_s


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="测量更新(写回)链路有效写带宽 + 验证解析模型")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100, 512],
                        help="扫描的数据大小 (MB)")
    parser.add_argument("--repeat", type=int, default=3, help="每个大小重复次数")
    parser.add_argument("--dim", type=int, default=64000, help="embedding 维度")
    parser.add_argument("--byte-per-elem", type=int, default=4, help="每元素字节 (float32=4)")
    parser.add_argument("--ssd-file", default="~/testfile",
                        help="fio 随机写测试文件 (被测 SSD 上, 建议单独建)")
    parser.add_argument("--ssd-runtime", type=int, default=15,
                        help="每档大小 fio 运行秒数")
    parser.add_argument("--ssd-qd", type=int, default=1,
                        help="fio randwrite 队列深度 (1=单行随机写, 8/16/32=批量刷回)")
    args = parser.parse_args()

    dev = get_device()
    print(f"[device] {dev}")
    if dev == "cpu":
        print("  [warn] 无 NPU/CUDA 设备, PCIe 写带宽测不出。")

    sizes_bytes = [s * 1024 * 1024 for s in args.sizes]
    results = {}

    # ---- PCIe 写链路: NPU->DRAM (device->host) ----
    r = measure_link("d2h", sizes_bytes, args.repeat)
    results["d2h"] = r
    print(f"\n[NPU->DRAM] (device->host 写回, PCIe)")
    print(f"  BW    = {r['bw_bytes_s']/1e9:.2f} GB/s")
    print(f"  alpha = {r['alpha_s']*1e6:.2f} us")
    print(f"  R²    = {r['r2']:.5f}   (≈1 说明 T=α+S/BW 线性模型成立)")
    for nb, t in zip(r["sizes_bytes"], r["times_s"]):
        print(f"    {nb/1e6:7.0f} MB -> {t*1e3:8.3f} ms")

    # ---- SSD 写链路: fio 随机写 (QD 可调, --direct=1) ----
    r_ssd = measure_fio_randwrite(sizes_bytes, args.ssd_file, args.ssd_runtime, args.ssd_qd)
    results["ssd"] = r_ssd
    print(f"\n[DRAM->SSD] (fio 随机写 QD={args.ssd_qd} --direct=1)")
    print(f"  BW    = {r_ssd['bw_bytes_s']/1e9:.2f} GB/s")
    print(f"  alpha = {r_ssd['alpha_s']*1e6:.2f} us")
    print(f"  R²    = {r_ssd['r2']:.5f}   (≈1 说明 T=α+S/BW 线性模型成立)")
    for nb, t in zip(r_ssd["sizes_bytes"], r_ssd["times_s"]):
        print(f"    {nb/1e6:7.0f} MB -> {t*1e3:8.3f} ms")

    # ---- 写回时间算例 (批量写回: α 按拷贝批次计, 默认 1 次) ----
    if dev != "cpu" and "d2h" in results:
        bw_pcie = results["d2h"]["bw_bytes_s"]
        alpha = results["d2h"]["alpha_s"]
        print(f"\n[写回时间算例] 批量写回, PCIe BW={bw_pcie/1e9:.2f} GB/s, "
              f"α={alpha*1e6:.0f} µs/批 (dim={args.dim})")
        print(f"  {'n_rows':>8s} {'data(MB)':>10s} {'NPU->DRAM(ms)':>14s}")
        for n in [10, 100, 1000, 10000]:
            data_mb = n * args.dim * args.byte_per_elem / 1e6
            t_ms = update_time_s(n, args.dim, args.byte_per_elem, bw_pcie, alpha,
                                 n_batches=1) * 1e3
            print(f"  {n:8d} {data_mb:10.1f} {t_ms:14.3f}")

    # ---- 保存结果 ----
    save = {d: {"bw_bytes_s": r["bw_bytes_s"], "alpha_s": r["alpha_s"], "r2": r["r2"],
                "sizes_bytes": r["sizes_bytes"], "times_s": r["times_s"]}
            for d, r in results.items()}
    save["ssd_qd"] = args.ssd_qd
    with open("update_bw_result.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n[save] -> update_bw_result.json")


if __name__ == "__main__":
    main()
