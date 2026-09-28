#!/usr/bin/env python3
"""
更新(写回)时间解析模型

模型形式 (与迁移/读路径统一的 T = α·n + DataSize/BW):
    T_update = α·n_batches + n_rows·m·b / BW
      (批量写回: α 只乘拷贝批次 n_batches, 不乘行数)

更新(写回)链路 (训练场景, 自上而下):
  1. NPU -> HBM   原地 read-modify-write —— 片上, 零跨层流量, 忽略
  2. NPU -> DRAM  device->host 写回        —— 温行, PCIe
  3. DRAM -> SSD  脏行批量刷回             —— 冷行, NVMe 写

输入 (CPU 已去重后给出):
  n_dram — 本步需写回 DRAM 的行数 (已去重)
  n_ssd  — 本步需刷回 SSD 的行数 (已去重)
  dim    — embedding 维度
  b      — 单元素字节数 (float32=4, bf16/fp16=2)

写回为批量: α 按「拷贝批次」计, 不按行计。
  n_batches_dram / n_batches_ssd — 打包成的拷贝次数 (单表一次=1, 多表逐表=表数)

输出:
  T_update = T_npu2dram(n_dram) + T_dram2ssd(n_ssd)

链路参数 (BW/α) 优先从 measure_update_bw.py 产出的 update_bw_result.json 加载;
否则用下方 NOMINAL_LINKS 占位默认值 (仅占位, 上线前务必标定)。

用法:
    # 用标定好的 update_bw_result.json 预测
    python update_model.py --n-dram 3000 --n-ssd 700 --dim 64000

    # 指定 JSON
    python update_model.py --n-dram 3000 --n-ssd 700 --dim 64000 \
        --result update_bw_result.json

    # 作为模块 import
    from update_model import UpdateModel
    m = UpdateModel.load()
    print(m.predict(n_dram=3000, n_ssd=700, dim=64000))
"""

import argparse
import json
import os

# ----------------------------------------------------------------------
# 占位默认参数 (仅占位, 用 measure_update_bw.py 标定后覆盖)
# ----------------------------------------------------------------------
# NPU->DRAM: PCIe device->host 写 (物理链路同迁移的 d2h)
# DRAM->SSD: NVMe 随机写 (SSD 写 ≪ 读, 需 randwrite 单独标定)
NOMINAL_LINKS = {
    "d2h": {"name": "NPU->DRAM", "bw_bytes_s": 12.0e9, "alpha_s": 5.0e-6},    # 占位
    "ssd": {"name": "DRAM->SSD", "bw_bytes_s": 2.0e9,  "alpha_s": 100.0e-6},  # 占位
}


def update_time_s(n_rows: int, dim: int, byte_per_elem: int, bw_bytes_s: float,
                  alpha_s: float = 0.0, n_batches: int = 1) -> float:
    """T = α·n_batches + (n_rows·dim·b)/BW。返回秒。

    批量写回: α 按拷贝批次计, 不按行计。n_rows 为已去重的总行数, n_batches 为
    打包成的 device->host / NVMe 拷贝次数 (单表一次=1, 多表逐表=表数)。
    """
    data_bytes = n_rows * dim * byte_per_elem
    return alpha_s * n_batches + data_bytes / max(bw_bytes_s, 1.0)


def load_links(json_path):
    """从 update_bw_result.json 加载 BW/α。缺失键用占位默认值补齐。"""
    links = {k: dict(v) for k, v in NOMINAL_LINKS.items()}
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        for key in links:
            if key in data:
                links[key]["bw_bytes_s"] = data[key].get("bw_bytes_s",
                                                        links[key]["bw_bytes_s"])
                links[key]["alpha_s"] = data[key].get("alpha_s", links[key]["alpha_s"])
    return links


class UpdateModel:
    """更新(写回)时间解析模型。CPU 给已去重行数, 模型返回时间。"""

    def __init__(self, links=None):
        self.links = links or load_links(None)

    @classmethod
    def load(cls, json_path=None):
        return cls(load_links(json_path))

    def predict(self, n_dram: int, n_ssd: int, dim: int, byte_per_elem: int = 4,
                n_batches_dram: int = 1, n_batches_ssd: int = 1):
        """预测一次更新(写回)的总耗时。返回 dict, 含各段 ms 与 TOTAL ms。

        n_dram / n_ssd 为已去重总行数; n_batches_* 为打包成的拷贝次数 (默认各 1 次批量)。
        """
        out = {}
        total_s = 0.0
        for key, (n_rows, n_batches) in (("d2h", (n_dram, n_batches_dram)),
                                         ("ssd", (n_ssd, n_batches_ssd))):
            if n_rows <= 0:
                out[key] = 0.0
                continue
            t_s = update_time_s(n_rows, dim, byte_per_elem,
                                self.links[key]["bw_bytes_s"],
                                self.links[key]["alpha_s"],
                                n_batches=n_batches)
            out[key] = t_s * 1000.0
            total_s += t_s
        out["TOTAL"] = total_s * 1000.0
        return out


def main():
    parser = argparse.ArgumentParser(description="更新(写回)时间解析模型")
    parser.add_argument("--n-dram", type=int, default=0, help="本步写回 DRAM 的行数 (已去重)")
    parser.add_argument("--n-ssd", type=int, default=0, help="本步刷回 SSD 的行数 (已去重)")
    parser.add_argument("--n-batches-dram", type=int, default=1,
                        help="NPU->DRAM 打包成的拷贝次数 (默认 1)")
    parser.add_argument("--n-batches-ssd", type=int, default=1,
                        help="DRAM->SSD 打包成的写批次 (默认 1)")
    parser.add_argument("--dim", type=int, default=64000, help="embedding 维度")
    parser.add_argument("--dtype", choices=["float32", "float16", "bf16"],
                        default="float32", help="元素精度")
    parser.add_argument("--result", default="update_bw_result.json",
                        help="measure_update_bw.py 产出的 JSON (不存在则用占位默认值)")
    args = parser.parse_args()

    byte_per_elem = {"float32": 4, "float16": 2, "bf16": 2}[args.dtype]
    model = UpdateModel.load(args.result)

    print(f"[更新/写回时间] n_dram={args.n_dram}, n_ssd={args.n_ssd}, "
          f"dim={args.dim}, dtype={args.dtype}")
    for key, link in model.links.items():
        src = "标定" if (args.result and os.path.exists(args.result)) else "占位默认"
        print(f"  {link['name']:<10s} BW={link['bw_bytes_s']/1e9:.2f} GB/s, "
              f"α={link['alpha_s']*1e6:.1f} us  ({src})")

    if args.n_dram <= 0 and args.n_ssd <= 0:
        print("\n[提示] 未给定 n-dram / n-ssd, 仅展示链路参数。")
        return

    pred = model.predict(args.n_dram, args.n_ssd, args.dim, byte_per_elem,
                         n_batches_dram=args.n_batches_dram,
                         n_batches_ssd=args.n_batches_ssd)
    print(f"\n  {'段':<12s} {'n_rows':>8s} {'时间(ms)':>10s}")
    print(f"  {'-'*34}")
    for key, (name, n) in (("d2h", ("NPU->DRAM", args.n_dram)),
                           ("ssd", ("DRAM->SSD", args.n_ssd))):
        if n > 0:
            print(f"  {name:<12s} {n:>8,} {pred[key]:>10.3f}")
    print(f"  {'-'*34}")
    print(f"  {'TOTAL':<12s} {'':>8s} {pred['TOTAL']:>10.3f}")


if __name__ == "__main__":
    main()
