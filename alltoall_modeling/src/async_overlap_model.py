"""Async overlap model for All-to-All communication regions.

The model composes a single-operator communication latency prediction with a
compute window that can run concurrently with communication. It is intentionally
small and deterministic so it can be reused by reports and by future profiler
trace parsers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class AsyncOverlapParams:
    """Parameters controlling async communication hiding."""

    eta: float = 0.85
    launch_overhead_us: float = 3.0
    sync_overhead_us: float = 8.0
    overlap_window_scale: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def clamp_eta(eta: float) -> float:
    return max(0.0, min(1.0, float(eta)))


def predict_async_region(
    comm_us: float,
    compute_us: float,
    params: AsyncOverlapParams | None = None,
    pre_us: float = 0.0,
    post_us: float = 0.0,
) -> Dict[str, float]:
    """Predict sync and async region latency.

    Args:
        comm_us: All-to-All latency from the single-operator model.
        compute_us: Compute work that is scheduled between async launch and wait.
        params: Async hiding parameters.
        pre_us: Work before async launch; not hideable.
        post_us: Work after wait; not hideable.

    Returns:
        A dictionary with synchronous baseline, async total, hidden time and
        visible communication. Units are microseconds.
    """

    params = params or AsyncOverlapParams()
    comm_us = max(float(comm_us), 0.0)
    compute_us = max(float(compute_us), 0.0)
    pre_us = max(float(pre_us), 0.0)
    post_us = max(float(post_us), 0.0)

    eta = clamp_eta(params.eta)
    effective_compute_window = max(compute_us * float(params.overlap_window_scale), 0.0)
    overlap_bound = min(comm_us, effective_compute_window)
    hidden_comm_us = eta * overlap_bound
    visible_comm_us = max(comm_us - hidden_comm_us, 0.0)

    synchronous_total_us = pre_us + comm_us + compute_us + post_us
    async_total_us = (
        pre_us
        + compute_us
        + visible_comm_us
        + float(params.launch_overhead_us)
        + float(params.sync_overhead_us)
        + post_us
    )
    speedup = synchronous_total_us / max(async_total_us, 1e-9)
    overlap_ratio = hidden_comm_us / max(overlap_bound, 1e-9) if overlap_bound > 0 else 0.0
    hidden_fraction_of_comm = hidden_comm_us / max(comm_us, 1e-9) if comm_us > 0 else 0.0

    return {
        "comm_us": comm_us,
        "compute_us": compute_us,
        "pre_us": pre_us,
        "post_us": post_us,
        "effective_compute_window_us": effective_compute_window,
        "hidden_comm_us": hidden_comm_us,
        "visible_comm_us": visible_comm_us,
        "synchronous_total_us": synchronous_total_us,
        "async_total_us": async_total_us,
        "speedup": speedup,
        "overlap_ratio": overlap_ratio,
        "hidden_fraction_of_comm": hidden_fraction_of_comm,
    }


def derive_eta(
    comm_us: float,
    compute_us: float,
    async_total_us: float,
    launch_overhead_us: float = 0.0,
    sync_overhead_us: float = 0.0,
    pre_us: float = 0.0,
    post_us: float = 0.0,
    overlap_window_scale: float = 1.0,
) -> float:
    """Estimate eta from a measured async region.

    This is used when profiler data is available. The formula is the inverse of
    ``predict_async_region`` with clamping to [0, 1].
    """

    comm_us = max(float(comm_us), 0.0)
    compute_us = max(float(compute_us), 0.0)
    effective_compute_window = max(compute_us * float(overlap_window_scale), 0.0)
    overlap_bound = min(comm_us, effective_compute_window)
    if overlap_bound <= 0:
        return 0.0

    visible_comm_us = (
        float(async_total_us)
        - max(float(pre_us), 0.0)
        - compute_us
        - max(float(post_us), 0.0)
        - float(launch_overhead_us)
        - float(sync_overhead_us)
    )
    hidden_comm_us = comm_us - visible_comm_us
    return clamp_eta(hidden_comm_us / overlap_bound)
