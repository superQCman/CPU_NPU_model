#!/usr/bin/env python3
"""
基于重用距离 (Reuse Distance) 的存储层级预测

核心思想: 重用距离能反映数据热度 —— 重用距离越小, 数据越热, 应放更快的存储层。

输入:
  - embedding 访问序列: 来自 dlrm 运行生成的 trace, 无法获取时按 Zipf 分布自动合成
  - 存储容量配置: 各层可容纳的向量行数

重用距离计算 (O(N log N), 哈希表 last + 树状数组 fw):
  维护 last[e] (元素 e 上次出现的时间戳) 和树状数组 fw (位置 t 为 1 当且仅当 t 是
  某元素当前「最后一次出现」的时间戳)。再次访问元素 e 时:
      RD(e) = fw.prefix(t) - fw.prefix(last[e]+1)   # (last[e], t) 之间的不同元素个数
  每个 embedding 取所有重用距离的平均值 d̄(e)。
  例 S=[A,B,A,B,C,B,A] → A 的重用距离 [1,2](均值 1.5), B 的重用距离 [1,1](均值 1)。

两种层级分配模式:
  阈值模式: 某级缓存能装下 d 个向量, d̄(e) ≤ d 即命中该层 (速度快, 但可能超容量)
  容量模式: 按 d̄(e) 升序排序, 依次装满 HBM→DRAM→SSD (较慢, 尽量装满)

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


class FenwickTree:
    """树状数组 (Binary Indexed Tree), 用于 O(log N) 计算重用距离。

    语义: 逻辑数组 fw[0..N-1], fw[t]=1 表示时间戳 t 是某个元素当前「最后一次出现」。
      - add(pos, delta): 单点更新 fw[pos] += delta
      - prefix(k):       前缀和 fw[0..k-1] (前 k 个位置中活跃标记的数量)
    """

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)  # 内部 1-indexed 存储

    def add(self, pos: int, delta: int):
        i = pos + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, k: int) -> int:
        """返回 fw[0..k-1] 的和 (前 k 个时间戳中活跃标记的数量)。"""
        s = 0
        i = k
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s


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
        self.access_count: Dict[int, int] = {}
        self.tier: Dict[int, str] = {}
        self._avg_rd: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Step 1: 计算重用距离 (哈希表 last + 树状数组 fw)
    # ------------------------------------------------------------------
    def compute_reuse_distances(self, verbose: bool = True) -> Dict[int, List[int]]:
        """
        遍历访问序列，用「哈希表 last + 树状数组 fw」计算重用距离，O(N log N)。

        定义: 重用距离 = 同一元素相邻两次访问之间出现的「不同元素」个数。
        例 S=[A,B,A,B,C,B,A] → A 的重用距离 [1,2], B 的重用距离 [1,1]。

        实现:
          - last[e]  元素 e 上一次出现的时间戳 (首次访问则无)
          - fw       树状数组, 位置 t 为 1 当且仅当 t 是当前「最后一次出现」的时间戳
          - 再次访问 e 时: RD = fw.prefix(t) - fw.prefix(last[e]+1)
                           = (last[e], t) 区间内仍活跃的不同元素个数
          - 更新: 清除 last[e] 处的旧标记 (置 0), 在 t 处打新标记 (置 1)
        """
        last: Dict[int, int] = {}
        fw = FenwickTree(self.N)
        total = self.N

        t_start = time.time()
        for t, ei in enumerate(self.S):
            # 记录频次（无论首次还是重用）
            self.access_count[ei] = self.access_count.get(ei, 0) + 1

            if ei in last:
                l = last[ei]
                # (l, t) 之间不同元素个数 = 区间内活跃标记数量
                rd = fw.prefix(t) - fw.prefix(l + 1)
                self.rd_map[ei].append(rd)
                # 清除旧的最后一次出现标记
                fw.add(l, -1)

            # 打上新标记, 并记录本次出现位置
            fw.add(t, 1)
            last[ei] = t

            if verbose and (t + 1) % 100000 == 0:
                elapsed = time.time() - t_start
                pct = (t + 1) / total * 100
                speed = (t + 1) / elapsed
                eta = (total - t - 1) / speed if speed > 0 else 0
                print(f"  [RD] {t + 1:,}/{total:,} ({pct:.0f}%) "
                      f"| unique={len(self.access_count):,} | {speed:.0f} rec/s | ETA {eta:.0f}s")

        if verbose:
            elapsed = time.time() - t_start
            print(f"  [RD] done in {elapsed:.1f}s, {total/elapsed:.0f} rec/s, "
                  f"unique={len(self.access_count):,}")

        return self.rd_map

    # ------------------------------------------------------------------
    # Step 2: 按平均重用距离分配存储层级 (两种模式)
    # ------------------------------------------------------------------
    def assign_tiers(self, hbm_rows: int = None, dram_rows: int = None) -> Dict[int, str]:
        """分配存储层级。

        模式 A (阈值模式, 默认): d̄(e) ≤ THBM → HBM, d̄(e) ≤ TDRAM → DRAM, 其余 → SSD
        模式 B (容量模式): 按 avg_RD 升序，取前 hbm_rows → HBM，再 dram_rows → DRAM，其余 → SSD
        """
        # 先计算所有 embedding 的平均 RD
        for e, rd_list in self.rd_map.items():
            if not rd_list:
                self._avg_rd[e] = float("inf")
            else:
                self._avg_rd[e] = float(np.mean(rd_list))

        # 访问序列中出现过但 rd_map 中没有记录的 embedding (只访问了一次)
        all_seen = set(self.S)
        for e in all_seen:
            if e not in self._avg_rd:
                self._avg_rd[e] = float("inf")

        if hbm_rows is not None and dram_rows is not None:
            # ---- 容量模式: 按 RD 排序，取 top-K ----
            # 只看有 RD 记录且 RD 有限的 embedding
            finite_rd = {e: rd for e, rd in self._avg_rd.items() if rd != float("inf")}
            sorted_emb = sorted(finite_rd.items(), key=lambda x: x[1])  # RD 升序

            assigned = set()
            # Top hbm_rows → HBM
            for e, _ in sorted_emb[:hbm_rows]:
                self.tier[e] = "HBM"
                assigned.add(e)
            # Next dram_rows → DRAM
            for e, _ in sorted_emb[hbm_rows:hbm_rows + dram_rows]:
                self.tier[e] = "DRAM"
                assigned.add(e)
            # 其余有限 RD 的 → SSD
            for e, _ in sorted_emb[hbm_rows + dram_rows:]:
                self.tier[e] = "SSD"
                assigned.add(e)
            # 只有一次访问的 (RD=∞): 模拟 preload 行为
            # 实际 ThreeLevelEmbedding 会预加载索引 [0, hbm_rows+dram_rows) 到 HBM/DRAM
            preload_end = hbm_rows + dram_rows
            for e in all_seen:
                if e not in assigned:
                    if e < hbm_rows:
                        self.tier[e] = "HBM"       # 在 HBM 范围内 → HBM 命中
                    elif e < preload_end:
                        self.tier[e] = "DRAM"      # 在 preload 范围 → DRAM 命中
                    else:
                        self.tier[e] = "SSD"       # 超出 → SSD miss
        else:
            # ---- 阈值模式 (原有逻辑) ----
            for e, avg_rd in self._avg_rd.items():
                if avg_rd == float("inf"):
                    self.tier[e] = "SSD"
                elif avg_rd <= self.th_hbm:
                    self.tier[e] = "HBM"
                elif avg_rd <= self.th_dram:
                    self.tier[e] = "DRAM"
                else:
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

    def compute_hit_rates_from_counts(self) -> Dict[str, float]:
        """基于第一次遍历时记录的访问频次，按 tier 聚合计算命中率。

        与 compute_hit_rates 结果等价，但不需要回放 trace，
        复杂度从 O(N) 降为 O(U)（unique embeddings）。
        """
        hits = {"HBM": 0, "DRAM": 0, "SSD": 0}
        for ei, count in self.access_count.items():
            tier = self.tier.get(ei, "SSD")
            hits[tier] += count
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
    def run(self, hbm_rows: int = None, dram_rows: int = None,
            verbose: bool = True, fast_hit_rate: bool = False) -> dict:
        if verbose:
            if hbm_rows is not None:
                print(f"Trace: {self.N:,} accesses, "
                      f"capacity HBM={hbm_rows:,} DRAM={dram_rows:,}")
            else:
                print(f"Trace: {self.N:,} accesses, "
                      f"thresholds HBM≤{self.th_hbm} DRAM≤{self.th_dram}")

        self.compute_reuse_distances(verbose=verbose)
        self.assign_tiers(hbm_rows=hbm_rows, dram_rows=dram_rows)

        if fast_hit_rate:
            hit_rates = self.compute_hit_rates_from_counts()
        else:
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
            "fast_hit_rate": fast_hit_rate,
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

    # 阈值模式 (传统)
    parser.add_argument("--th-hbm", type=int, default=100,
                        help="[阈值模式] HBM threshold: avg RD ≤ N → HBM")
    parser.add_argument("--th-dram", type=int, default=1000,
                        help="[阈值模式] DRAM threshold: avg RD ≤ N → DRAM")

    # 容量模式 (推荐: 对齐 ThreeLevelEmbedding 的 hbm_ratio/dram_cache_ratio)
    parser.add_argument("--hbm-rows", type=int, default=None,
                        help="[容量模式] HBM 行数，按 avg_RD 升序取 top-K")
    parser.add_argument("--dram-rows", type=int, default=None,
                        help="[容量模式] DRAM 行数，取 HBM 之后的 K 行")

    # 输出
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save analysis result as JSON (for memory_model.py)")
    parser.add_argument("--include-details", action="store_true",
                        help="Include per-embedding tier/avg_rd in JSON (large dicts)")
    parser.add_argument("--fast-hit-rates", action="store_true",
                        help="Use access-frequency aggregation instead of "
                             "replaying trace to compute hit rates. "
                             "Equivalent result, O(U) instead of O(N).")

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

    # 容量模式: 如果指定了 --hbm-rows/--dram-rows，按容量分配
    use_capacity = (args.hbm_rows is not None and args.dram_rows is not None)
    result = analyzer.run(
        hbm_rows=args.hbm_rows if use_capacity else None,
        dram_rows=args.dram_rows if use_capacity else None,
        verbose=True,
        fast_hit_rate=args.fast_hit_rates,
    )
    ReuseDistanceAnalyzer.print_report(result, args.th_hbm, args.th_dram)

    # 导出 JSON 供 memory_model.py 使用
    if args.output_json:
        import json as _json

        # 基础输出：summary 字段
        export = {
            "total_accesses": result["total_accesses"],
            "unique_embeddings": result["unique_embeddings"],
            "hit_rates": result["hit_rates"],
            "tier_distribution": result["tier_distribution"],
            "mode": "capacity" if use_capacity else "threshold",
        }
        if use_capacity:
            export["capacity"] = {"hbm_rows": args.hbm_rows, "dram_rows": args.dram_rows}
        else:
            export["thresholds"] = {"hbm": args.th_hbm, "dram": args.th_dram}

        # RD 分布 (每级存储中 embedding 的平均 RD 列表 → 统计摘要)
        rd_stats = {}
        for tier, rd_list in result["rd_distribution"].items():
            if rd_list:
                rd_stats[tier] = {
                    "count": len(rd_list),
                    "mean": float(np.mean(rd_list)),
                    "median": float(np.median(rd_list)),
                    "min": float(np.min(rd_list)),
                    "max": float(np.max(rd_list)),
                }
            else:
                rd_stats[tier] = {"count": 0, "mean": None}
        export["rd_statistics"] = rd_stats

        # 可选：每个 embedding 的详细 tier / avg_rd
        if args.include_details:
            # int keys → str keys for JSON compatibility
            export["tier_assignment"] = {
                str(k): v for k, v in result["tier_assignment"].items()
            }
            export["avg_rd"] = {
                str(k): (float(v) if v != float("inf") else "inf")
                for k, v in result["avg_rd"].items()
            }

        with open(args.output_json, "w") as f:
            _json.dump(export, f, indent=2)
        print(f"[ReuseDistance] Result saved to {args.output_json}")
