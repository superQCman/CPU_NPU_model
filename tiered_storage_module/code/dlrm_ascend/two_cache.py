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
    """DRAM缓存层，使用LRU策略"""
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


class TwoLevelEmbedding(nn.Module):
    """两级存储Embedding: SSD -> DRAM"""
    total_forward_times = {"SSD_to_DRAM": 0, "DRAM_to_CPU": 0}
    total_forward_data = {"SSD_to_DRAM": 0, "DRAM_to_CPU": 0}
    forward_count = 0  # 记录 forward 调用次数

    def __init__(self, num_embeddings: int, embedding_dim: int, dram_cache_ratio: float = 0.05):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 计算DRAM缓存容量
        self.dram_rows = max(1, int(num_embeddings * dram_cache_ratio))

        # Level 1: DRAM缓存
        self.dram_cache = LRUCache(self.dram_rows)

        # Level 2: SSD存储
        self.ssd_storage = SSDStorage()

        print(f"[TwoLevelEmbedding] Table {num_embeddings}: DRAM_cache={self.dram_rows}, SSD={num_embeddings - self.dram_rows}")

    @property
    def weight(self):
        # 返回 DRAM 权重，作为默认权重
        return self._embedding.dram_cache

    def forward(self, indices: torch.Tensor, offsets: Optional[torch.Tensor] = None, per_sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward method for TwoLevelEmbedding with SSD -> DRAM -> CPU cache hierarchy.
        """
        print(f"Forward called with indices shape: {indices.shape}, offsets shape: {offsets.shape if offsets is not None else 'None'}, per_sample_weights shape: {per_sample_weights.shape if per_sample_weights is not None else 'None'}")

        if per_sample_weights is not None:
            raise NotImplementedError("per_sample_weights is not yet supported in TwoLevelEmbedding.")

        device = indices.device
        indices_list = indices.cpu().tolist()  # Convert indices to a list for processing

        results = []
        ssd_miss = []

        # Timings for memory access
        total_times = {"SSD_to_DRAM": 0, "DRAM_to_CPU": 0}
        total_data = {"SSD_to_DRAM": 0, "DRAM_to_CPU": 0}

        # Process each index
        for idx in indices_list:
            if self.dram_cache.contains(idx):
                # DRAM -> CPU cache access
                start_time = time.time()
                results.append(self.dram_cache.get(idx).to(device))
                end_time = time.time()

                # Record timing and data size
                elapsed_time = end_time - start_time
                data_size = self.embedding_dim * 4  # Assuming float32 (4 bytes per element)
                total_times["DRAM_to_CPU"] += elapsed_time
                total_data["DRAM_to_CPU"] += data_size
            else:
                # SSD -> DRAM access
                ssd_miss.append(idx)

        # Handle SSD misses
        if ssd_miss:
            start_time = time.time()
            for idx in ssd_miss:
                # Load from SSD and cache in DRAM
                data = self.ssd_storage.deferred_init(idx, self.embedding_dim)
                self.dram_cache.put(idx, data.cpu())
                results.append(data.to(device))
            end_time = time.time()

            # Record timing and data size
            elapsed_time = (end_time - start_time) / len(ssd_miss)
            data_size = len(ssd_miss) * self.embedding_dim * 4
            total_times["SSD_to_DRAM"] += elapsed_time * len(ssd_miss)
            total_data["SSD_to_DRAM"] += data_size

        # Write memory access timings to log file
        with open("memory_access_timings.log", "a") as log_file:
            log_file.write("\n=== Memory Access Timings ===\n")
            for key in total_times:
                log_file.write(f"{key}: {total_times[key]:.6f} seconds, Data size: {total_data[key]} bytes\n")

        # Update global statistics
        TwoLevelEmbedding.forward_count += 1
        for key in total_times:
            TwoLevelEmbedding.total_forward_times[key] += total_times[key]
            TwoLevelEmbedding.total_forward_data[key] += total_data[key]

        # If offsets are provided, aggregate results based on offsets
        if offsets is not None:
            # Group embeddings based on offsets
            aggregated_results = []
            for i in range(len(offsets) - 1):
                start = offsets[i].item()
                end = offsets[i + 1].item()
                group = torch.stack(results[start:end])
                aggregated_results.append(group.sum(dim=0))  # Example: sum aggregation

            # 如果最后一组没有被处理，确保处理最后一部分
            if len(offsets) > 0 and offsets[-1].item() < len(results):
                start = offsets[-1].item()
                group = torch.stack(results[start:])
                aggregated_results.append(group.sum(dim=0))

            return torch.stack(aggregated_results).to(device)

        # Return results as a stacked tensor
        return torch.stack(results).to(device)

# 测试代码
if __name__ == "__main__":
    num_embeddings = 100000
    embedding_dim = 128
    dram_cache_ratio = 0.05

    embedding = TwoLevelEmbedding(num_embeddings, embedding_dim, dram_cache_ratio)

    # 模拟访问
    indices = torch.tensor([0, 1, 2, 50000, 99999])
    output = embedding(indices)
    print(output)