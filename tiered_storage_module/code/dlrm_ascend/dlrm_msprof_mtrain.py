#!/usr/bin/env python3
"""
MTrainS版本DLRM训练脚本 - 简化版
基于论文 "MTrainS: Hierarchical Memory for DLRM Training"
实现三级存储: HBM -> DRAM -> SSD

"""

import torch
import torch.nn as nn
import numpy as np
import threading
import queue
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional
import time  # 引入时间模块

# ============================================================
# MTrainS 三级存储核心实现
# ============================================================

class SSDStorage:
    """SSD存储层"""
    def __init__(self, db_path: str = "./embedding_ssd_cache"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self._storage: Dict[int, torch.Tensor] = {}
        self._lock = threading.RLock()

    def get(self, key: int) -> Optional[torch.Tensor]:
        with self._lock:
            return self._storage.get(key)

    def set(self, key: int, value: torch.Tensor):
        with self._lock:
            self._storage[key] = value.cpu()

    def deferred_init(self, key: int, dim: int) -> torch.Tensor:
        with self._lock:
            if key not in self._storage:
                data = torch.randn(dim) * 0.01
                self._storage[key] = data
                return data
            return self._storage[key]


class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache = OrderedDict()

    def get(self, key: int) -> Optional[torch.Tensor]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: int, value: torch.Tensor):
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def contains(self, key: int) -> bool:
        return key in self._cache


class ThreeLevelEmbedding(nn.Module):
    """三级存储Embedding: HBM -> DRAM -> SSD"""
    _instances = []  # 注册所有实例，用于聚合统计

    @classmethod
    def aggregate_stats(cls):
        """聚合所有实例的统计数据 (避免类变量共享导致的重复计数)"""
        total_times = {}
        total_data = {}
        total_count = 0
        for inst in cls._instances:
            total_count += inst.forward_count
            for k, v in inst.total_forward_times.items():
                total_times[k] = total_times.get(k, 0.0) + v
            for k, v in inst.total_forward_data.items():
                total_data[k] = total_data.get(k, 0.0) + v
        return total_times, total_data, total_count

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hbm_ratio: float = 0.01,
        dram_cache_ratio: float = 0.05,
        migration_interval: int = 100,
        print_interval: int = 50,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 计算各级存储容量
        self.hbm_rows = max(1, int(num_embeddings * hbm_ratio))
        self.dram_rows = max(1, int(num_embeddings * dram_cache_ratio))

        # Level 0: HBM存储 (动态冷热迁移)
        if self.hbm_rows > 0:
            hbm_weight = torch.empty(self.hbm_rows, embedding_dim)
            nn.init.uniform_(hbm_weight, a=-0.01, b=0.01)
            self.hbm_weight = nn.Parameter(hbm_weight, requires_grad=True)
            # 当前 HBM 中存储的逻辑索引 (sorted，用于 searchsorted 定位)
            self.hbm_indices = torch.arange(self.hbm_rows, dtype=torch.long)
        else:
            self.hbm_weight = None
            self.hbm_indices = torch.empty(0, dtype=torch.long)

        # 冷热迁移控制
        self.migration_interval = migration_interval
        self.print_interval = print_interval
        self.access_freq: Dict[int, int] = {}

        # Level 1: DRAM缓存
        self.dram_cache = LRUCache(self.dram_rows)

        # Level 2: SSD存储
        self.ssd_storage = SSDStorage()

        # 统计 — 实例变量，每个 table 独立计数
        self.hbm_hits = 0
        self.dram_hits = 0
        self.ssd_hits = 0
        self.total_access = 0
        self.forward_count = 0
        self.total_forward_times = {"SSD_to_DRAM": 0.0, "DRAM_to_CPU": 0.0, "DRAM_to_HBM": 0.0, "HBM_to_NPU": 0.0, "TOTAL_MEMORY_ACCESS": 0.0}
        self.total_forward_data = {"SSD_to_DRAM": 0.0, "DRAM_to_CPU": 0.0, "DRAM_to_HBM": 0.0, "HBM_to_NPU": 0.0, "TOTAL_MEMORY_ACCESS": 0.0}

        ThreeLevelEmbedding._instances.append(self)

        # 可选预热：把非 HBM 的索引预加载到 DRAM，避免首次 forward 产生 SSD 访问
        # 注意：若 num_embeddings 很大，此操作会占用大量内存并增加初始化时间
        preload_end = min(self.num_embeddings, self.hbm_rows + self.dram_rows)
        for k in range(self.hbm_rows, preload_end):
            vec = self.ssd_storage.deferred_init(k, self.embedding_dim)
            # 存入 DRAM 缓存（放到 CPU）
            self.dram_cache.put(k, vec.cpu())

        print(f"[MTrainS] Table {num_embeddings}: HBM={self.hbm_rows}, DRAM_cache={self.dram_rows}, SSD={num_embeddings - self.hbm_rows}")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        print(f"[MTrainS] Forward called with {len(indices)} indices")
        device = indices.device
        n = len(indices)
        byte_per_elem = 4  # float32

        t_start = time.time()
        self.total_access += n

        output = torch.empty(n, self.embedding_dim, device=device)

        # --- Level 0: HBM 命中 (向量化批量 gather，动态索引) ---
        hbm_idx = self.hbm_indices.to(device)
        hbm_mask = torch.isin(indices, hbm_idx)
        if hbm_mask.any():
            t0 = time.time()
            hbm_logical = indices[hbm_mask]
            hbm_pos = torch.searchsorted(hbm_idx, hbm_logical)
            output[hbm_mask] = self.hbm_weight[hbm_pos]
            elapsed = time.time() - t0
            cnt = hbm_mask.sum().item()
            self.total_forward_times["HBM_to_NPU"] += elapsed
            self.total_forward_data["HBM_to_NPU"] += cnt * self.embedding_dim * byte_per_elem
            self.hbm_hits += cnt

        # --- Level 1+2: DRAM 缓存 / SSD ---
        non_hbm_mask = ~hbm_mask
        if non_hbm_mask.any():
            non_hbm_pos = torch.where(non_hbm_mask)[0]
            non_hbm_idx_list = indices[non_hbm_mask].cpu().tolist()

            dram_hit_pos = []   # index into non_hbm_idx_list
            dram_hit_vecs = []  # embedding vectors (CPU)
            ssd_miss_pos = []   # index into non_hbm_idx_list
            ssd_miss_idx = []   # actual embedding row index

            for i, idx in enumerate(non_hbm_idx_list):
                vec = self.dram_cache.get(idx)
                if vec is not None:
                    dram_hit_pos.append(i)
                    dram_hit_vecs.append(vec)
                else:
                    ssd_miss_pos.append(i)
                    ssd_miss_idx.append(idx)

            # DRAM hits: 批量搬运到 device
            if dram_hit_pos:
                t0 = time.time()
                hit_pos_t = torch.tensor(dram_hit_pos, device=device)
                hit_data = torch.stack(dram_hit_vecs).to(device)
                output[non_hbm_pos[hit_pos_t]] = hit_data
                elapsed = time.time() - t0
                cnt = len(dram_hit_pos)
                self.total_forward_times["DRAM_to_CPU"] += elapsed
                self.total_forward_data["DRAM_to_CPU"] += cnt * self.embedding_dim * byte_per_elem
                self.dram_hits += cnt

            # SSD misses: 逐个加载 (冷数据，量小)
            if ssd_miss_idx:
                t0 = time.time()
                for i, idx in zip(ssd_miss_pos, ssd_miss_idx):
                    data = self.ssd_storage.deferred_init(idx, self.embedding_dim)
                    output[non_hbm_pos[i]] = data.to(device)
                    self.dram_cache.put(idx, data.cpu())
                elapsed = time.time() - t0
                cnt = len(ssd_miss_idx)
                self.total_forward_times["SSD_to_DRAM"] += elapsed
                self.total_forward_data["SSD_to_DRAM"] += cnt * self.embedding_dim * byte_per_elem
                self.ssd_hits += cnt

        # 更新访问频率 (所有索引)
        for idx in indices.cpu().tolist():
            self.access_freq[idx] = self.access_freq.get(idx, 0) + 1

        total_elapsed = time.time() - t_start
        self.total_forward_times["TOTAL_MEMORY_ACCESS"] += total_elapsed
        self.forward_count += 1
        print(f"[MTrainS] Forward {self.forward_count}: Memory access time: {total_elapsed:.3f} s")

        # 打印 4 段时间统计 (独立于迁移，由 print_interval 控制)
        if self.forward_count % self.print_interval == 0:
            print(f"[MTrainS] Forward {self.forward_count}: Total memory access time: {total_elapsed:.3f} s")
            self._print_timing_report()
        self._print_timing_report()

        # 周期性冷热迁移
        if self.forward_count % self.migration_interval == 0:
            print(f"[MTrainS] Forward {self.forward_count}: Starting migration...")
            self._migrate(device)

        self._migrate(device)
        return output

    def _migrate(self, device):
        """按访问频率重新分配 HBM 空间: 驱逐冷数据到 DRAM，提升热数据到 HBM"""
        if not self.access_freq or self.hbm_weight is None:
            return

        # 按频率降序，取 top-K 作为新 HBM 集合
        sorted_items = sorted(self.access_freq.items(), key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, _ in sorted_items[:self.hbm_rows]])

        old_set = set(self.hbm_indices.tolist())
        new_set = set(top_indices)

        evicted = old_set - new_set   # 从 HBM 驱逐 → DRAM
        promoted = new_set - old_set  # 从 DRAM/SSD 晋升 → HBM

        if not evicted and not promoted:
            self._decay_freq()
            return

        old_hbm_idx = self.hbm_indices.to(device)

        # 驱逐: HBM → DRAM
        for logical_idx in evicted:
            pos = torch.searchsorted(
                old_hbm_idx, torch.tensor(logical_idx, device=device)
            )
            self.dram_cache.put(logical_idx, self.hbm_weight[pos].detach().cpu())

        # 构建新的 HBM weight (保持 nn.Parameter 引用)
        new_weight = torch.empty(self.hbm_rows, self.embedding_dim, device=device)
        dram_to_hbm_time = 0.0
        dram_to_hbm_bytes = 0
        byte_per_elem = 4  # float32

        for new_pos, logical_idx in enumerate(top_indices):
            if logical_idx in old_set:
                old_pos = torch.searchsorted(
                    old_hbm_idx, torch.tensor(logical_idx, device=device)
                )
                new_weight[new_pos] = self.hbm_weight[old_pos].detach()
            else:
                vec = self.dram_cache.get(logical_idx)
                if vec is not None:
                    t0 = time.time()
                    new_weight[new_pos] = vec.to(device)
                    dram_to_hbm_time += time.time() - t0
                else:
                    t0 = time.time()
                    new_weight[new_pos] = self.ssd_storage.deferred_init(
                        logical_idx, self.embedding_dim
                    ).to(device)
                    dram_to_hbm_time += time.time() - t0
                dram_to_hbm_bytes += self.embedding_dim * byte_per_elem

        self.total_forward_times["DRAM_to_HBM"] += dram_to_hbm_time
        self.total_forward_data["DRAM_to_HBM"] += dram_to_hbm_bytes

        self.hbm_weight.data.copy_(new_weight)
        self.hbm_indices = torch.tensor(top_indices, dtype=torch.long)

        self._decay_freq()

    def _print_timing_report(self):
        """打印当前 4 段访存时间的统计报告"""
        fc = max(self.forward_count, 1)
        total_acc = max(self.total_access, 1)

        header = (
            f"[Metrics] step={self.forward_count} "
            f"HBM_hit={self.hbm_hits / total_acc:.1%} "
            f"DRAM_hit={self.dram_hits / total_acc:.1%} "
            f"SSD_hit={self.ssd_hits / total_acc:.1%}"
        )
        lines = [header]

        stages = [
            ("HBM_to_NPU",  "HBM -> NPU cache "),
            ("DRAM_to_CPU", "DRAM -> CPU cache"),
            ("SSD_to_DRAM", "SSD -> DRAM      "),
            ("DRAM_to_HBM", "DRAM -> HBM      "),
        ]
        for key, label in stages:
            t = self.total_forward_times[key]
            d = self.total_forward_data[key]
            avg_ms = t / fc * 1000
            bw = d / t / 1e9 if t > 0 else 0
            lines.append(f"  {label}: {avg_ms:.3f} ms/step, {d / 1e6:.1f} MB total, {bw:.1f} GB/s")

        total_ms = self.total_forward_times["TOTAL_MEMORY_ACCESS"] / fc * 1000
        lines.append(f"  TOTAL wall time: {total_ms:.3f} ms/step")
        print("\n".join(lines))

        # Append the same report to memory_access_timings.log
        try:
            with open("memory_access_timings.log", "a") as log_file:
                # include a timestamp for convenience
                log_file.write(f"\n# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                for ln in lines:
                    log_file.write(ln + "\n")
        except Exception as e:
            # don't crash training if logging fails; print a short notice
            print(f"Warning: failed to write memory access log: {e}")

    def _decay_freq(self):
        """衰减访问频率，保留近期热点，淘汰冷记录防止 dict 无限膨胀"""
        new_freq: Dict[int, int] = {}
        for idx, freq in self.access_freq.items():
            decayed = freq // 2
            if decayed > 0:
                new_freq[idx] = decayed
        self.access_freq = new_freq

class MTrainSEmbeddingBag(nn.Module):
    """MTrainS版本的EmbeddingBag"""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 hbm_ratio: float = 0.01, dram_cache_ratio: float = 0.05,
                 migration_interval: int = 100, print_interval: int = 50):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self._embedding = ThreeLevelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hbm_ratio=hbm_ratio,
            dram_cache_ratio=dram_cache_ratio,
            migration_interval=migration_interval,
            print_interval=print_interval,
        )

    @property
    def weight(self):
        # 返回 HBM 权重，作为默认权重
        return self._embedding.hbm_weight


    def forward(self, indices: torch.Tensor, offsets: torch.Tensor, per_sample_weights=None):
        embeddings = self._embedding(indices)

        batch_size = len(offsets) - 1
        output = torch.zeros(batch_size, self.embedding_dim, device=indices.device)

        for i in range(batch_size):
            start = offsets[i].item()
            end = offsets[i+1].item()
            if start < end:
                if per_sample_weights is not None:
                    weights = per_sample_weights[start:end].unsqueeze(1)
                    output[i] = (embeddings[start:end] * weights).sum(dim=0)
                else:
                    output[i] = embeddings[start:end].sum(dim=0)

        return output


# ============================================================
# 导入原始模块并应用猴子补丁
# ============================================================

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings("ignore")

# 导入原始模块
import dlrm_msprof as original
import dlrm_data_pytorch as dp


# # 修复DataLoader问题
# def _fixed_make_random_data_and_loader(args, ln_emb, m_den):
#     """修复后的数据加载器"""
#     import torch

#     # 确保data_size有效
#     if args.data_size <= 0:
#         args.data_size = 100000  # 默认10万样本

#     # 创建随机数据集
#     train_data = dp.RandomDataset(
#         args.data_size,
#         ln_emb,
#         m_den,
#         args.num_indices_per_lookup,
#         args.num_indices_per_lookup_fixed,
#         args.rand_data_dist,
#         args.rand_data_min,
#         args.rand_data_max,
#         args.rand_data_mu,
#         args.rand_data_sigma,
#     )

#     # 根据num_workers设置prefetch_factor
#     prefetch_factor = 2 if args.num_workers > 0 else None
#     persistent_workers = args.num_workers > 0

#     train_loader = torch.utils.data.DataLoader(
#         train_data,
#         batch_size=args.mini_batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=args.use_gpu,
#         prefetch_factor=prefetch_factor,
#         persistent_workers=persistent_workers
#     )

#     # 测试数据
#     test_data = dp.RandomDataset(
#         max(10000, args.data_size // 10),
#         ln_emb,
#         m_den,
#         args.num_indices_per_lookup,
#         args.num_indices_per_lookup_fixed,
#         args.rand_data_dist,
#         args.rand_data_min,
#         args.rand_data_max,
#         args.rand_data_mu,
#         args.rand_data_sigma,
#     )

#     test_loader = torch.utils.data.DataLoader(
#         test_data,
#         batch_size=args.test_mini_batch_size,
#         shuffle=False,
#         num_workers=args.test_num_workers if args.test_num_workers > 0 else 2,
#         pin_memory=args.use_gpu,
#         prefetch_factor=prefetch_factor,
#         persistent_workers=persistent_workers
#     )

#     return train_data, train_loader, test_data, test_loader


# # 替换函数
# dp.make_random_data_and_loader = _fixed_make_random_data_and_loader


# 补丁create_emb方法
_original_create_emb = original.DLRM_Net.create_emb

def _patched_create_emb(self, m, ln, weighted_pooling=None):
    """支持MTrainS的create_emb"""
    print(f"[MTrainS] Creating embedding layers with MTrainS support")
    emb_l = nn.ModuleList()
    v_W_l = []

    # MTrainS配置
    use_mtrain = getattr(self, '_use_mtrain', False)
    hbm_ratio = getattr(self, '_hbm_ratio', 0.01)      # 1%热数据在HBM
    dram_ratio = getattr(self, '_dram_cache_ratio', 0.05)  # 5%在DRAM缓存

    for i in range(0, ln.size):
        if original.ext_dist.my_size > 1:
            if i not in self.local_emb_indices:
                continue
        n = ln[i]

        if use_mtrain:
            # 使用MTrainS三级存储
            EE = MTrainSEmbeddingBag(
                num_embeddings=n,
                embedding_dim=m,
                hbm_ratio=hbm_ratio,
                dram_cache_ratio=dram_ratio
            )
        else:
            # 原始实现
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)

        if weighted_pooling is None:
            v_W_l.append(None)
        else:
            v_W_l.append(torch.ones(n, dtype=torch.float32))
        emb_l.append(EE)

    return emb_l, v_W_l


# 应用补丁
original.DLRM_Net.create_emb = _patched_create_emb


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 解析MTrainS参数
    import argparse

    # 提取MTrainS参数并过滤
    mtrain_args_list = []
    remaining_args = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['--use-mtrain', '--hbm-ratio', '--dram-cache-ratio']:
            mtrain_args_list.append(arg)
            if arg != '--use-mtrain':
                i += 1
                if i < len(sys.argv):
                    mtrain_args_list.append(sys.argv[i])
        else:
            remaining_args.append(arg)
        i += 1

    # 解析MTrainS参数
    use_mtrain = '--use-mtrain' in mtrain_args_list

    hbm_ratio = 0.01
    dram_ratio = 0.05

    for idx, arg in enumerate(mtrain_args_list):
        if arg == '--hbm-ratio' and idx + 1 < len(mtrain_args_list):
            hbm_ratio = float(mtrain_args_list[idx + 1])
        if arg == '--dram-cache-ratio' and idx + 1 < len(mtrain_args_list):
            dram_ratio = float(mtrain_args_list[idx + 1])

    # 设置标志
    if use_mtrain:
        original.DLRM_Net._use_mtrain = True
        original.DLRM_Net._hbm_ratio = hbm_ratio
        original.DLRM_Net._dram_cache_ratio = dram_ratio

        print(f"[MTrainS] Enabled")
        print(f"[MTrainS] HBM ratio: {hbm_ratio:.2%} (hot embeddings)")
        print(f"[MTrainS] DRAM cache ratio: {dram_ratio:.2%}")

    # 添加默认参数
    default_args = ['--data-size', '100000']
    for default in default_args:
        if default not in ' '.join(remaining_args):
            remaining_args.insert(1, default)

    # 运行原始程序
    sys.argv = [sys.argv[0]] + remaining_args
    print(f"[MTrainS] Running with args: {sys.argv}")

    original.run()

    # 程序结束时聚合所有 table 的统计
    agg_times, agg_data, total_forwards = ThreeLevelEmbedding.aggregate_stats()
    if total_forwards > 0:
        avg_times = {k: v / total_forwards for k, v in agg_times.items()}
        avg_data = {k: v / total_forwards for k, v in agg_data.items()}

        with open("memory_access_timings.log", "a") as log_file:
            log_file.write("\n=== Average Time and Data per Forward (aggregated across all tables) ===\n")
            for key in avg_times:
                log_file.write(f"{key}: {avg_times[key]:.12f} s, avg data: {avg_data[key]:.1f} bytes\n")