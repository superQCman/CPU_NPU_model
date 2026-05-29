from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ALIGNMENT_HIGH = 131072
ALIGNMENT_MID = 32768


@dataclass(frozen=True)
class AllToAllPrediction:
    message_bytes: int
    comm_us: float
    compute_window_us: float
    visible_comm_us: float
    hidden_comm_us: float
    hidden_fraction_of_comm: float
    eta: float


class AllToAllAdapter:
    def __init__(self, alltoall_root: str | Path) -> None:
        self.root = Path(alltoall_root)
        models = self.root / "artifacts" / "models"
        self.bw_params = json.loads((models / "bw_model_params.json").read_text(encoding="utf-8"))["params"]
        self.latency_params = json.loads((models / "latency_model_params.json").read_text(encoding="utf-8"))["groups"]
        async_payload = json.loads((models / "dlrm_async_overlap_multiconfig_model_params.json").read_text(encoding="utf-8"))
        self.async_params = async_payload["params"]

    def predict_region(self, message_bytes: int, compute_window_us: float) -> AllToAllPrediction:
        if int(message_bytes) <= 0:
            return AllToAllPrediction(
                message_bytes=0,
                comm_us=0.0,
                compute_window_us=max(float(compute_window_us), 0.0),
                visible_comm_us=0.0,
                hidden_comm_us=0.0,
                hidden_fraction_of_comm=0.0,
                eta=self._clamp(self.async_params.get("eta", 1.0)),
            )
        comm_us = self.predict_latency_us(message_bytes)
        eta = self._clamp(self.async_params.get("eta", 1.0))
        overlap_window = max(float(compute_window_us), 0.0) * float(self.async_params.get("overlap_window_scale", 1.0))
        overlap_bound = min(comm_us, overlap_window)
        hidden = eta * overlap_bound
        visible = max(comm_us - hidden, 0.0)
        return AllToAllPrediction(
            message_bytes=int(message_bytes),
            comm_us=float(comm_us),
            compute_window_us=float(compute_window_us),
            visible_comm_us=float(visible),
            hidden_comm_us=float(hidden),
            hidden_fraction_of_comm=float(hidden / max(comm_us, 1e-9)) if comm_us > 0 else 0.0,
            eta=float(eta),
        )

    def predict_latency_us(self, message_bytes: int) -> float:
        size = max(int(message_bytes), 1)
        group = self._group_for_size(size)
        params = self.latency_params[group]
        t_base = self._latency_base(size, params)
        corr = self._correction(size, params)
        return max(float(t_base - corr), 0.0)

    def _group_for_size(self, size: int) -> str:
        if size % ALIGNMENT_HIGH == 0:
            return "Group1 (128KB)"
        if size % ALIGNMENT_MID == 0:
            return "Group2 (32KB)"
        return "Group3 (unaligned)"

    def _latency_base(self, size: int, params: dict) -> float:
        t_s = float(params["t_s"])
        m1 = int(params["m1"])
        m2 = int(params["m2"])
        bw_max = float(params["bw_max"])
        if size <= m1:
            return t_s
        if size >= m2:
            return t_s + size / max(bw_max * 1e3, 1e-9)
        bw = max(self._eval_bw(size), 1e-9)
        return size / (bw * 1e3)

    def _eval_bw(self, size: int) -> float:
        group = "high" if size % ALIGNMENT_HIGH == 0 else "mid" if size % ALIGNMENT_MID == 0 else "low"
        p = self.bw_params
        split_size = int(p["split_size"])
        sat_high = float(p["sat_bw_high"])
        sat_mid = float(p["sat_bw_mid"])
        sat_low = float(p["sat_bw_low3"])
        high_trend = p["popt_high_trend"]
        high_final = p["popt_high_final"]
        mid_final = p["popt_mid_final"]
        dip_high = p.get("dip_high")
        dip_mid = p.get("dip_mid")
        low_drop = p.get("low3_drop_size")

        if group == "high":
            if size < split_size:
                return self._pred_sigmoid(size, high_trend)
            if dip_high and dip_high[0] <= size <= dip_high[1]:
                return self._pred_dip(size, high_final)
            return min(self._pred_dip(size, high_final), sat_high)
        if group == "mid":
            if size < split_size:
                return self._pred_sigmoid(size, high_trend)
            if dip_mid and dip_mid[0] <= size <= dip_mid[1]:
                return self._pred_dip(size, mid_final)
            return min(self._pred_dip(size, mid_final), sat_mid)
        if low_drop is not None and size >= int(low_drop):
            return sat_low
        return min(self._pred_dip(size, high_final), sat_high)

    def _correction(self, size: int, params: dict) -> float:
        m1 = int(params["m1"])
        if size <= m1:
            return 0.0
        xs = np.asarray(params.get("spline_x_log2", []), dtype=float)
        ys = np.asarray(params.get("spline_residual_us", []), dtype=float)
        if len(xs) == 0:
            return 0.0
        x = float(np.log2(float(size)))
        if x <= xs[0]:
            return float(ys[0])
        if x >= xs[-1]:
            if len(xs) < 2:
                return float(ys[-1])
            slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)
            return float(ys[-1] + slope * (x - xs[-1]))
        return float(np.interp(x, xs, ys))

    def _pred_sigmoid(self, size: int, popt: list[float]) -> float:
        return float(10 ** self._sigmoid(np.log2(float(size)), *popt))

    def _pred_dip(self, size: int, popt: list[float]) -> float:
        x = np.log2(float(size))
        return float(10 ** (self._sigmoid(x, *popt[:4]) - popt[4] * np.exp(-0.5 * ((x - popt[5]) / popt[6]) ** 2)))

    def _sigmoid(self, x: float, l_val: float, x0: float, k_val: float, b_val: float) -> float:
        return float(l_val / (1 + np.exp(-k_val * (x - x0))) + b_val)

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
