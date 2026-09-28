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
    def __init__(self, num_embeddings: int, dim: int, path: str = "./embedding_table.bin"):
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.path = path
        shape = (num_embeddings, dim)
        if not os.path.exists(path):
            # 创建稀疏文件: 只设置大小、不物理写满整块磁盘, OS 按需分配磁盘页
            with open(path, "wb") as f:
                f.seek(num_embeddings * dim * 4 - 1)
                f.write(b"\0")
        self._table = np.memmap(path, dtype=np.float32, mode="r+", shape=shape)

    def get_row(self, key: int) -> torch.Tensor:
        """从磁盘读一行 (OS page cache 缓存热页)。返回 CPU float32 tensor。"""
        row = np.array(self._table[key], dtype=np.float32, copy=True)
        return torch.from_numpy(row)

    def set_row(self, key: int, value: torch.Tensor):
        """写回一行到磁盘 (训练场景的更新写回)。"""
        self._table[key] = value.detach().cpu().numpy()


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


# 迁移事件时间表输出文件 (CSV)
MIGRATION_LOG = "migration_schedule.log"


class ThreeLevelEmbedding(nn.Module):
    """三级存储Embedding: HBM -> DRAM -> SSD"""
    _instances = []  # 注册所有实例，用于聚合统计
    _next_table_id = 0  # 给每个表分配唯一 id，用于区分多表日志

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
        gamma: float = 0.9,
        tau_high: float = 2.0,
        tau_low: float = 0.5,
        heat_epsilon: float = 1e-3,
        print_interval: int = 50,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 各级容量 (行数)
        self.hbm_rows = max(1, int(num_embeddings * hbm_ratio))
        self.dram_rows = max(1, int(num_embeddings * dram_cache_ratio))

        # 热度模型参数 (指数衰减 + 阈值分级)
        self.gamma = gamma              # 热度衰减因子 (每步)
        self.tau_high = tau_high        # 热度升破此值 -> 预取晋升 HBM
        self.tau_low = tau_low          # 热度跌破此值 -> 淘汰下沉 SSD
        self.heat_epsilon = heat_epsilon  # 热度低于此值的 SSD 行剪枝
        self.print_interval = print_interval

        # Level 0: HBM (初始占满前 hbm_rows 行, 后续由热度阈值动态调整)
        hbm_weight = torch.empty(self.hbm_rows, embedding_dim)
        nn.init.uniform_(hbm_weight, a=-0.01, b=0.01)
        self.hbm_weight = nn.Parameter(hbm_weight, requires_grad=True)
        self.hbm_indices = torch.arange(self.hbm_rows, dtype=torch.long)  # sorted

        # 热度表: logical_idx -> heat (HBM/DRAM 行 + 近期访问过的 SSD 行)
        self.heat: Dict[int, float] = {}

        # Level 1: DRAM (热度管理缓存, 取代 LRU)
        self.dram_data: Dict[int, torch.Tensor] = {}

        # Level 2: SSD存储
        self.ssd_storage = SSDStorage(num_embeddings, embedding_dim)

        # 统计 — 实例变量，每个 table 独立计数
        self.hbm_hits = 0
        self.dram_hits = 0
        self.ssd_hits = 0
        self.total_access = 0
        self.forward_count = 0
        self.total_forward_times = {"SSD_to_DRAM": 0.0, "DRAM_to_CPU": 0.0, "DRAM_to_HBM": 0.0, "HBM_to_NPU": 0.0, "TOTAL_MEMORY_ACCESS": 0.0}
        self.total_forward_data = {"SSD_to_DRAM": 0.0, "DRAM_to_CPU": 0.0, "DRAM_to_HBM": 0.0, "HBM_to_NPU": 0.0, "TOTAL_MEMORY_ACCESS": 0.0}

        self._table_id = ThreeLevelEmbedding._next_table_id
        ThreeLevelEmbedding._next_table_id += 1
        ThreeLevelEmbedding._instances.append(self)

        # 预热: 前 hbm_rows 行视为初始热行 (heat=tau_high), 次 dram_rows 行预加载到 DRAM (heat=tau_low)
        for k in range(self.hbm_rows):
            self.heat[k] = self.tau_high
        preload_end = min(self.num_embeddings, self.hbm_rows + self.dram_rows)
        for k in range(self.hbm_rows, preload_end):
            vec = self.ssd_storage.get_row(k)
            self.dram_data[k] = vec.cpu()
            self.heat[k] = self.tau_low

        print(f"[MTrainS] Table {num_embeddings}: HBM<={self.hbm_rows}, DRAM<={self.dram_rows}, SSD={num_embeddings} "
              f"(gamma={self.gamma}, tau_high={self.tau_high}, tau_low={self.tau_low})")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        device = indices.device
        n = len(indices)
        byte_per_elem = 4  # float32

        t_start = time.time()
        self.total_access += n

        output = torch.empty(n, self.embedding_dim, device=device)

        # --- Level 0: HBM 命中 (向量化批量 gather) ---
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

        # --- Level 1+2: DRAM / SSD ---
        non_hbm_mask = ~hbm_mask
        if non_hbm_mask.any():
            non_hbm_pos = torch.where(non_hbm_mask)[0]
            non_hbm_idx_list = indices[non_hbm_mask].cpu().tolist()

            dram_hit_pos = []   # index into non_hbm_idx_list
            dram_hit_vecs = []  # embedding vectors (CPU)
            ssd_miss_pos = []   # index into non_hbm_idx_list
            ssd_miss_idx = []   # actual embedding row index

            for i, idx in enumerate(non_hbm_idx_list):
                vec = self.dram_data.get(idx)
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

            # SSD misses: 逐个加载 (冷数据，量小), 只读不算缓存填充 (SSD->DRAM 由迁移第3步统一预取)
            if ssd_miss_idx:
                t0 = time.time()
                for i, idx in zip(ssd_miss_pos, ssd_miss_idx):
                    data = self.ssd_storage.get_row(idx)
                    output[non_hbm_pos[i]] = data.to(device)
                elapsed = time.time() - t0
                cnt = len(ssd_miss_idx)
                self.total_forward_times["SSD_to_DRAM"] += elapsed
                self.total_forward_data["SSD_to_DRAM"] += cnt * self.embedding_dim * byte_per_elem
                self.ssd_hits += cnt

        # --- 热度更新 (指数衰减 + 访问累加) ---
        self._update_heat(indices.cpu().tolist())

        total_elapsed = time.time() - t_start
        self.total_forward_times["TOTAL_MEMORY_ACCESS"] += total_elapsed
        self.forward_count += 1

        # --- 连续热度分级迁移 (淘汰/预取) ---
        self._tier_migration(device)

        if self.forward_count % self.print_interval == 0:
            self._print_timing_report()
        return output

    def _update_heat(self, accessed_list):
        """热度更新: 全局指数衰减 + 访问累加 + 冷 SSD 行剪枝。"""
        for idx in self.heat:
            self.heat[idx] *= self.gamma
        for idx in accessed_list:
            self.heat[idx] = self.heat.get(idx, 0.0) + 1.0
        hbm_set = set(self.hbm_indices.tolist())
        for idx in list(self.heat.keys()):
            if (self.heat[idx] < self.heat_epsilon
                    and idx not in hbm_set and idx not in self.dram_data):
                del self.heat[idx]

    def _tier_migration(self, device):
        """热度阈值驱动的连续淘汰/预取 (逐级迁移, 不越级)。各链路分开计时。"""
        byte_per_elem = 4
        old_hbm_idx = self.hbm_indices
        old_hbm_set = set(old_hbm_idx.tolist())
        hbm_set = set(old_hbm_set)  # 工作副本, 最终成为新 HBM 集合

        # 各链路分开计数/计时
        n_hbm_to_dram = 0; t_hbm_to_dram = 0.0   # 淘汰: device->host 拷贝
        n_ssd_to_dram = 0; t_ssd_to_dram = 0.0   # 预取: SSD 读入 DRAM
        n_dram_to_ssd = 0                        # 下沉: 丢弃缓存副本 (无物理拷贝)

        # 1. 淘汰 HBM -> DRAM: 热度跌破 tau_high (物理拷贝 device->host)
        for idx in list(hbm_set):
            if self.heat.get(idx, 0.0) < self.tau_high:
                t0 = time.time()
                pos = int(torch.searchsorted(old_hbm_idx, torch.tensor(idx)))
                self.dram_data[idx] = self.hbm_weight[pos].detach().cpu()
                t_hbm_to_dram += time.time() - t0
                n_hbm_to_dram += 1
                hbm_set.discard(idx)

        # 2. 淘汰 DRAM -> SSD: 热度跌破 tau_low (丢弃缓存副本, SSD 仍是真源)
        for idx in list(self.dram_data.keys()):
            if self.heat.get(idx, 0.0) < self.tau_low:
                self.dram_data.pop(idx, None)
                self.heat.pop(idx, None)
                n_dram_to_ssd += 1

        # 3. 预取 SSD -> DRAM: 热度升破 tau_low、当前不在 HBM/DRAM 的行
        for idx, h in list(self.heat.items()):
            if h >= self.tau_low and idx not in hbm_set and idx not in self.dram_data:
                t0 = time.time()
                self.dram_data[idx] = self.ssd_storage.get_row(idx)
                t_ssd_to_dram += time.time() - t0
                n_ssd_to_dram += 1

        # 4. 预取 DRAM -> HBM: 热度升破 tau_high (逐级, 只从 DRAM 晋升, 避免 SSD->HBM 越级)
        #    这里只标记候选, 真正的 host->device 拷贝在 _rebuild_hbm 里统计
        for idx in list(self.dram_data.keys()):
            if self.heat.get(idx, 0.0) >= self.tau_high:
                hbm_set.add(idx)

        # 5. HBM 容量约束: 超出 hbm_rows 淘汰最冷 (区分旧 HBM 行与刚预取的行)
        if len(hbm_set) > self.hbm_rows:
            overflow = sorted(hbm_set, key=lambda i: self.heat.get(i, 0.0))[:len(hbm_set) - self.hbm_rows]
            for idx in overflow:
                if idx in old_hbm_set:
                    # 旧 HBM 行: 拷贝回 DRAM
                    t0 = time.time()
                    pos = int(torch.searchsorted(old_hbm_idx, torch.tensor(idx)))
                    self.dram_data[idx] = self.hbm_weight[pos].detach().cpu()
                    t_hbm_to_dram += time.time() - t0
                    n_hbm_to_dram += 1
                # 否则是刚预取的行: 数据本就在 DRAM, 不产生拷贝, 也不计数
                hbm_set.discard(idx)

        # 6. DRAM 容量约束: 超出 dram_rows 淘汰最冷到 SSD (丢弃)
        if len(self.dram_data) > self.dram_rows:
            overflow = sorted(self.dram_data.keys(), key=lambda i: self.heat.get(i, 0.0))[:len(self.dram_data) - self.dram_rows]
            for idx in overflow:
                self.dram_data.pop(idx, None)
                self.heat.pop(idx, None)
                n_dram_to_ssd += 1

        # 无任何物理移动则跳过
        if (hbm_set == old_hbm_set and n_hbm_to_dram == 0
                and n_ssd_to_dram == 0 and n_dram_to_ssd == 0):
            return

        # 7. 重建 HBM (DRAM->HBM 的 host->device 拷贝发生在这里, 顺带统计)
        t_dram_to_hbm, n_dram_to_hbm, t_ssd_to_hbm, n_ssd_to_hbm = \
            self._rebuild_hbm(device, sorted(hbm_set))

        total_time = t_hbm_to_dram + t_dram_to_hbm + t_ssd_to_dram + t_ssd_to_hbm
        data_bytes = (n_hbm_to_dram + n_dram_to_hbm + n_ssd_to_dram + n_ssd_to_hbm) * self.embedding_dim * byte_per_elem

        # 8. 阶段统计 (迁移总开销)
        self.total_forward_times["DRAM_to_HBM"] += total_time
        self.total_forward_data["DRAM_to_HBM"] += data_bytes

        # 9. 记录迁移事件 (各链路分开)
        self._log_migration_event(
            step=self.forward_count,
            n_hbm_to_dram=n_hbm_to_dram, t_hbm_to_dram_s=t_hbm_to_dram,
            n_dram_to_hbm=n_dram_to_hbm, t_dram_to_hbm_s=t_dram_to_hbm,
            n_ssd_to_dram=n_ssd_to_dram, t_ssd_to_dram_s=t_ssd_to_dram,
            n_dram_to_ssd=n_dram_to_ssd,
            n_ssd_to_hbm=n_ssd_to_hbm, t_ssd_to_hbm_s=t_ssd_to_hbm,
            data_bytes=data_bytes,
            total_time_s=total_time,
        )

    def _rebuild_hbm(self, device, new_hbm_sorted):
        """按新的 HBM 逻辑索引集合重建 hbm_weight / hbm_indices。

        Returns: (t_dram_to_hbm, n_dram_to_hbm, t_ssd_to_hbm, n_ssd_to_hbm)
        即晋升进 HBM 的两类拷贝: DRAM->HBM (正常逐级) 与 SSD->HBM (兜底, 通常为 0)。
        """
        old_hbm_idx = self.hbm_indices
        old_hbm_set = set(old_hbm_idx.tolist())
        old_hbm_weight = self.hbm_weight

        new_weight = torch.empty(len(new_hbm_sorted), self.embedding_dim, device=device)
        t_dram_to_hbm = 0.0; n_dram_to_hbm = 0
        t_ssd_to_hbm = 0.0; n_ssd_to_hbm = 0

        for pos, idx in enumerate(new_hbm_sorted):
            if idx in old_hbm_set:
                old_pos = int(torch.searchsorted(old_hbm_idx, torch.tensor(idx)))
                new_weight[pos] = old_hbm_weight[old_pos].detach()
            else:
                # 晋升源应在 DRAM (逐级迁移); 兜底从 SSD 读
                vec = self.dram_data.pop(idx, None)
                t0 = time.time()
                if vec is not None:
                    new_weight[pos] = vec.to(device)          # DRAM -> HBM
                    t_dram_to_hbm += time.time() - t0
                    n_dram_to_hbm += 1
                else:
                    new_weight[pos] = self.ssd_storage.get_row(idx).to(device)  # SSD -> HBM
                    t_ssd_to_hbm += time.time() - t0
                    n_ssd_to_hbm += 1

        # 动态容量: 重建 nn.Parameter (inference-only 场景安全)
        self.hbm_weight = nn.Parameter(new_weight, requires_grad=True)
        self.hbm_indices = torch.tensor(new_hbm_sorted, dtype=torch.long)
        return t_dram_to_hbm, n_dram_to_hbm, t_ssd_to_hbm, n_ssd_to_hbm

    def _log_migration_event(self, step,
                             n_hbm_to_dram, t_hbm_to_dram_s,
                             n_dram_to_hbm, t_dram_to_hbm_s,
                             n_ssd_to_dram, t_ssd_to_dram_s,
                             n_dram_to_ssd,
                             n_ssd_to_hbm, t_ssd_to_hbm_s,
                             data_bytes, total_time_s):
        """把一次迁移事件写入时间表: stdout 打印 + 追加到 CSV (各链路分开计时)。"""
        print(
            f"[Migration] table={self._table_id} step={step} "
            f"hbm->dram={n_hbm_to_dram}({t_hbm_to_dram_s * 1e3:.3f}ms) "
            f"dram->hbm={n_dram_to_hbm}({t_dram_to_hbm_s * 1e3:.3f}ms) "
            f"ssd->dram={n_ssd_to_dram}({t_ssd_to_dram_s * 1e3:.3f}ms) "
            f"dram->ssd={n_dram_to_ssd} "
            f"ssd->hbm={n_ssd_to_hbm}({t_ssd_to_hbm_s * 1e3:.3f}ms) "
            f"total={total_time_s * 1e3:.3f}ms"
        )
        try:
            header = ("table_id,step,wall_time_s,"
                      "n_hbm_to_dram,t_hbm_to_dram_s,"
                      "n_dram_to_hbm,t_dram_to_hbm_s,"
                      "n_ssd_to_dram,t_ssd_to_dram_s,"
                      "n_dram_to_ssd,"
                      "n_ssd_to_hbm,t_ssd_to_hbm_s,"
                      "data_bytes,total_time_s")
            new_file = (not os.path.exists(MIGRATION_LOG)
                        or os.path.getsize(MIGRATION_LOG) == 0)
            with open(MIGRATION_LOG, "a") as f:
                if new_file:
                    f.write(header + "\n")
                f.write(
                    f"{self._table_id},{step},{time.time():.6f},"
                    f"{n_hbm_to_dram},{t_hbm_to_dram_s:.6f},"
                    f"{n_dram_to_hbm},{t_dram_to_hbm_s:.6f},"
                    f"{n_ssd_to_dram},{t_ssd_to_dram_s:.6f},"
                    f"{n_dram_to_ssd},"
                    f"{n_ssd_to_hbm},{t_ssd_to_hbm_s:.6f},"
                    f"{data_bytes:.0f},{total_time_s:.6f}\n"
                )
        except Exception as e:
            print(f"Warning: failed to write migration log: {e}")

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

class MTrainSEmbeddingBag(nn.Module):
    """MTrainS版本的EmbeddingBag"""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 hbm_ratio: float = 0.01, dram_cache_ratio: float = 0.05,
                 gamma: float = 0.9, tau_high: float = 2.0, tau_low: float = 0.5,
                 print_interval: int = 50):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self._embedding = ThreeLevelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hbm_ratio=hbm_ratio,
            dram_cache_ratio=dram_cache_ratio,
            gamma=gamma,
            tau_high=tau_high,
            tau_low=tau_low,
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
