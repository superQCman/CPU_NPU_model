#!/usr/bin/env python3
"""
基于重用距离 (Reuse Distance) 的存储层级预测

算法参考 MTrainS 论文 Section 3.2:
  - 维护 LRU 栈模拟访问序列
  - 对每个 Embedding 计算平均重用距离 d̄(e)
  - d̄(e) ≤ THBM → HBM, d̄(e) ≤ TDRAM → DRAM, 其余 → SSD

使用方法:
    python reuse_distance.py --num-embeddings 100000 --trace-length 500000
    python reuse_distance.py --trace-file trace.txt --th-hbm 50 --th-dram 500
"""

import argparse
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


class ReuseDistanceAnalyzer:
    """基于重用距离的存储层级预测器"""

    def __init__(self, access_sequence: List[int], th_hbm: int, th_dram: int):
        """
        Args:
            access_sequence: 访问序列 S = [e1, e2, ..., eN]
            th_hbm: HBM 阈值，平均 RD ≤ th_hbm → HBM
            th_dram: DRAM 阈值，平均 RD ≤ th_dram → DRAM (其余 → SSD)
        """
        self.S = access_sequence
        self.N = len(access_sequence)
        self.th_hbm = th_hbm
        self.th_dram = th_dram

        self.rd_map: Dict[int, List[int]] = defaultdict(list)
        self.tier: Dict[int, str] = {}
        self._avg_rd: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Step 1: 计算重用距离
    # ------------------------------------------------------------------
    def compute_reuse_distances(self, verbose: bool = True) -> Dict[int, List[int]]:
        """
        遍历访问序列，维护 LRU 栈，记录每次重用的距离。

        stack[0] = 栈底 (最久未访问)
        stack[-1] = 栈顶 (最近访问)
        重用距离 = 从栈顶往下数的位置 (栈顶距离为 1)
        """
        stack: List[int] = []
        total = self.N

        t_start = time.time()
        for i, ei in enumerate(self.S):
            if ei not in stack:
                # 首次访问，不入栈 (或者入栈但 RD 记作无穷)
                stack.append(ei)
            else:
                # 重用: 距离 = 栈顶到该元素的位置 (1-indexed)
                pos = len(stack) - 1 - stack.index(ei) + 1  # 栈顶距离
                self.rd_map[ei].append(pos)
                # 移至栈顶
                stack.remove(ei)
                stack.append(ei)

            if verbose and (i + 1) % 100000 == 0:
                elapsed = time.time() - t_start
                pct = (i + 1) / total * 100
                speed = (i + 1) / elapsed
                eta = (total - i - 1) / speed if speed > 0 else 0
                print(f"  [RD] {i + 1:,}/{total:,} ({pct:.0f}%) "
                      f"| stack={len(stack):,} | {speed:.0f} rec/s | ETA {eta:.0f}s")

        if verbose:
            elapsed = time.time() - t_start
            print(f"  [RD] done in {elapsed:.1f}s, {total/elapsed:.0f} rec/s, "
                  f"unique={len(self.rd_map):,}")

        return self.rd_map

    # ------------------------------------------------------------------
    # Step 2: 按平均重用距离分配存储层级
    # ------------------------------------------------------------------
    def assign_tiers(self) -> Dict[int, str]:
        """d̄(e) ≤ THBM → HBM, d̄(e) ≤ TDRAM → DRAM, 其余 → SSD"""
        for e, rd_list in self.rd_map.items():
            if not rd_list:
                self._avg_rd[e] = float("inf")
                self.tier[e] = "SSD"
            else:
                self._avg_rd[e] = float(np.mean(rd_list))
                if self._avg_rd[e] <= self.th_hbm:
                    self.tier[e] = "HBM"
                elif self._avg_rd[e] <= self.th_dram:
                    self.tier[e] = "DRAM"
                else:
                    self.tier[e] = "SSD"

        # 访问序列中出现过但 rd_map 中没有记录的 embedding (只访问了一次)
        all_seen = set(self.S)
        for e in all_seen:
            if e not in self.tier:
                self._avg_rd[e] = float("inf")
                self.tier[e] = "SSD"

        return self.tier

    # ------------------------------------------------------------------
    # Step 3: 统计命中率与数据量分布
    # ------------------------------------------------------------------
    def compute_hit_rates(self) -> Dict[str, float]:
        """按分配好的 tier 回放访问序列，统计各级命中次数"""
        hits = {"HBM": 0, "DRAM": 0, "SSD": 0}
        for ei in self.S:
            tier = self.tier.get(ei, "SSD")
            hits[tier] += 1
        return {k: v / self.N for k, v in hits.items()}

    def compute_tier_distribution(self) -> Dict[str, int]:
        """各级存储的 Embedding 数量"""
        dist = {"HBM": 0, "DRAM": 0, "SSD": 0}
        for tier in self.tier.values():
            dist[tier] += 1
        return dist

    def compute_rd_distribution(self) -> Dict[str, List[float]]:
        """各级存储中 embedding 的平均重用距离分布"""
        rd_by_tier: Dict[str, List[float]] = {"HBM": [], "DRAM": [], "SSD": []}
        for e, tier in self.tier.items():
            if e in self._avg_rd and self._avg_rd[e] != float("inf"):
                rd_by_tier[tier].append(self._avg_rd[e])
        return rd_by_tier

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, verbose: bool = True) -> dict:
        if verbose:
            print(f"Trace: {self.N:,} accesses, "
                  f"thresholds HBM≤{self.th_hbm} DRAM≤{self.th_dram}")

        self.compute_reuse_distances(verbose=verbose)
        self.assign_tiers()

        hit_rates = self.compute_hit_rates()
        tier_dist = self.compute_tier_distribution()
        rd_dist = self.compute_rd_distribution()

        return {
            "tier_assignment": self.tier,
            "avg_rd": self._avg_rd,
            "hit_rates": hit_rates,
            "tier_distribution": tier_dist,
            "rd_distribution": rd_dist,
            "total_accesses": self.N,
            "unique_embeddings": len(set(self.S)),
        }

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------
    @staticmethod
    def print_report(result: dict, th_hbm: int, th_dram: int):
        dist = result["tier_distribution"]
        hr = result["hit_rates"]
        rd_dist = result["rd_distribution"]
        total_uniq = max(result["unique_embeddings"], 1)

        print(f"\n{'='*65}")
        print(f"  Reuse Distance Storage Tier Prediction")
        print(f"{'='*65}")
        print(f"  Total accesses:         {result['total_accesses']:>12,}")
        print(f"  Unique embeddings:      {result['unique_embeddings']:>12,}")
        print(f"  Thresholds:             HBM ≤ {th_hbm},  DRAM ≤ {th_dram}")
        print(f"{'='*65}")

        print(f"\n  Tier Distribution:")
        for tier in ["HBM", "DRAM", "SSD"]:
            n = dist[tier]
            rd_list = rd_dist[tier]
            rd_str = f"avg RD={np.mean(rd_list):.1f}" if rd_list else "avg RD=N/A"
            print(f"    {tier:5s}: {n:>10,} embeddings ({n/total_uniq*100:5.1f}%)  {rd_str}")

        print(f"\n  Hit Rates (simulated):")
        for tier in ["HBM", "DRAM", "SSD"]:
            bar = "█" * int(hr[tier] * 40)
            print(f"    {tier:5s}: {hr[tier]*100:6.2f}%  {bar}")

        print(f"{'='*65}\n")


# ============================================================
# 辅助: 生成合成访问序列
# ============================================================
def generate_zipfian_trace(
    num_embeddings: int,
    trace_length: int,
    alpha: float = 1.0,
    seed: int = 42,
) -> List[int]:
    """生成 Zipfian 分布的访问序列，模拟推荐系统 embedding 访问模式"""
    rng = np.random.default_rng(seed)
    pmf = 1.0 / np.arange(1, num_embeddings + 1, dtype=np.float64) ** alpha
    pmf /= pmf.sum()
    return rng.choice(num_embeddings, size=trace_length, p=pmf).tolist()


def load_trace_from_file(filepath: str) -> List[int]:
    """从文件加载访问序列 (每行一个整数索引)"""
    trace = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trace.append(int(line))
    return trace


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reuse Distance based Storage Tier Prediction"
    )
    # 数据源
    parser.add_argument("--trace-file", type=str, default=None,
                        help="Path to access trace file (one index per line)")
    parser.add_argument("--num-embeddings", type=int, default=100_000,
                        help="Number of unique embeddings (for synthetic trace)")
    parser.add_argument("--trace-length", type=int, default=500_000,
                        help="Length of synthetic access trace")
    parser.add_argument("--zipf-alpha", type=float, default=1.0,
                        help="Zipf distribution skew (higher = more skewed)")
    parser.add_argument("--seed", type=int, default=42)

    # 阈值
    parser.add_argument("--th-hbm", type=int, default=100,
                        help="HBM threshold: avg RD ≤ N → HBM")
    parser.add_argument("--th-dram", type=int, default=1000,
                        help="DRAM threshold: avg RD ≤ N → DRAM (rest → SSD)")

    args = parser.parse_args()

    # 加载或生成访问序列
    if args.trace_file:
        print(f"Loading trace from {args.trace_file} ...")
        trace = load_trace_from_file(args.trace_file)
    else:
        print(f"Generating Zipfian trace "
              f"(N={args.num_embeddings:,}, len={args.trace_length:,}, "
              f"alpha={args.zipf_alpha}) ...")
        trace = generate_zipfian_trace(
            num_embeddings=args.num_embeddings,
            trace_length=args.trace_length,
            alpha=args.zipf_alpha,
            seed=args.seed,
        )

    analyzer = ReuseDistanceAnalyzer(trace, th_hbm=args.th_hbm, th_dram=args.th_dram)
    result = analyzer.run(verbose=True)
    ReuseDistanceAnalyzer.print_report(result, args.th_hbm, args.th_dram)
