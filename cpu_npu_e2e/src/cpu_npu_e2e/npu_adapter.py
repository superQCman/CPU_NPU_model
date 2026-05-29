from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .shapes import GraphNode, infer_gemm_mnk, infer_reduction_axis, product


class NpuLatencyAdapter:
    DROP_COLUMNS = {
        "target_kernel_compute_us",
        "case_id",
        "profile_dir",
        "major_class",
        "latency_p50_us",
        "latency_mean_us",
        "kernel_count_observed",
        "kernel_type_unique_observed",
        "share_ai_core_observed",
        "share_ai_vector_core_observed",
        "share_mix_aiv_observed",
        "operator_row_index",
        "source_name",
        "input_shapes",
        "real_device_total_us",
        "real_device_aicore_us",
        "host_total_us",
        "case_name",
    }

    OP_TO_CLASS = {
        "MatMulV2": "DenseLinearAlgebra",
        "BatchMatMulV2": "DenseLinearAlgebra",
        "GemmBias": "DenseLinearAlgebra",
        "Add": "ElementwiseActivation",
        "Mul": "ElementwiseActivation",
        "Relu": "ElementwiseActivation",
        "Sigmoid": "ElementwiseActivation",
        "ConcatD": "TensorTransformMove",
        "Transpose": "TensorTransformMove",
        "Gather": "IndexingSparse",
        "ReduceSum": "AxisCompute",
    }

    def __init__(self, npu_root: str | Path) -> None:
        self.npu_root = Path(npu_root)
        code_root = self.npu_root / "code"
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        from build_semantic_operator_dataset import build_record  # type: ignore
        from train_physical_memory_models import add_hierarchical_memory_features, add_taxonomy  # type: ignore

        self._build_record = build_record
        self._add_taxonomy = add_taxonomy
        self._add_hierarchical_memory_features = add_hierarchical_memory_features
        self.hardware = json.loads((self.npu_root / "models" / "hardware_descriptor.json").read_text(encoding="utf-8"))
        self.model_artifact_path = self._resolve_model_artifact()
        self.artifacts = joblib.load(self.model_artifact_path)

    def _resolve_model_artifact(self) -> Path:
        candidates = [
            self.npu_root / "results" / "physical_memory_models.joblib",
            self.npu_root / "results" / "physical_memory_models.joblib.gz",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"NPU model artifact not found. Tried: {', '.join(str(path) for path in candidates)}")

    def predict_us(self, node: GraphNode, target_op_type: str) -> tuple[float, str]:
        params = self._params_for_node(node, target_op_type)
        record = self._build_record(
            case_id=f"node_{node.node_idx}",
            profile_dir="cpu_npu_e2e",
            major_class=self.OP_TO_CLASS.get(target_op_type, "Unknown"),
            op_type=target_op_type,
            dtype=self._dtype_for_node(node),
            params=params,
            target_us=0.0,
            hw=self.hardware,
            latency_p50=0.0,
            latency_mean=0.0,
        )
        frame = pd.DataFrame([record])
        out = self._add_taxonomy(frame)
        out = self._add_hierarchical_memory_features(out, self.hardware)
        out["pred_compute_us"] = np.nan
        out["model_level"] = "none"
        for group, pkg in dict(self.artifacts.get("models", {})).items():
            mask = out["model_group"].astype(str).eq(str(group))
            if not mask.any():
                continue
            x = self._feature_frame(out.loc[mask], list(pkg["feature_columns"]))
            pred = pkg["model"].predict(x)
            if str(pkg.get("model_choice", "")).endswith("log"):
                pred = np.expm1(pred)
            out.loc[mask, "pred_compute_us"] = np.maximum(pred, 0.0)
            out.loc[mask, "model_level"] = str(group)
        value = float(out["pred_compute_us"].iloc[0])
        if not np.isfinite(value):
            raise RuntimeError(f"NPU model did not produce a prediction for node {node.node_idx} ({target_op_type})")
        return max(value, 0.0), str(out["model_level"].iloc[0])

    def _feature_frame(self, rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        cols = [col for col in rows.columns if col not in self.DROP_COLUMNS]
        x = rows[cols].copy()
        categorical = {"op_type", "dtype", "physical_family", "residual_group", "model_group"}
        for col in list(x.columns):
            if col in categorical:
                x[col] = x[col].astype(str).fillna("missing")
            else:
                x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
        x = pd.get_dummies(x, columns=[col for col in categorical if col in x.columns])
        return x.reindex(columns=feature_columns, fill_value=0.0)

    def _dtype_for_node(self, node: GraphNode) -> str:
        for spec in node.outputs + node.inputs:
            if spec.dtype:
                return "float32" if spec.dtype.lower() == "float" else spec.dtype
        return "float32"

    def _params_for_node(self, node: GraphNode, op_type: str) -> dict[str, Any]:
        if op_type in {"GemmBias", "MatMulV2"}:
            m, k, n = infer_gemm_mnk(node)
            return {"m": m, "k": k, "n": n}
        if op_type == "Gather":
            table = node.inputs[0].dims if node.inputs else ()
            index_count = float(node.inputs[1].element_count) if len(node.inputs) > 1 else 1.0
            rows = float(table[0]) if len(table) >= 1 else max(index_count, 1.0)
            cols = float(table[1]) if len(table) >= 2 else float(node.outputs[0].dims[-1]) if node.outputs and node.outputs[0].dims else 1.0
            return {"rows": rows, "cols": cols, "index_count": index_count}
        if op_type == "ReduceSum":
            dims = list(node.inputs[0].dims if node.inputs else node.outputs[0].dims if node.outputs else (1,))
            while len(dims) < 3:
                dims.append(1)
            return {"d0": float(dims[0]), "d1": float(dims[1]), "d2": float(dims[2]), "axis": float(infer_reduction_axis(node))}
        if op_type in {"Add", "Mul", "Relu", "Sigmoid"}:
            dims = node.outputs[0].dims if node.outputs else node.inputs[0].dims if node.inputs else (1,)
            cols = float(dims[-1]) if dims else 1.0
            rows = float(product(dims[:-1]) if len(dims) > 1 else 1)
            return {"rows": rows, "cols": cols}
        if op_type == "ConcatD":
            features = []
            for spec in node.inputs[:3]:
                dims = spec.dims
                features.append(float(product(dims[1:]) if len(dims) > 1 else product(dims)))
            while len(features) < 3:
                features.append(0.0)
            batch = float(node.outputs[0].dims[0]) if node.outputs and node.outputs[0].dims else 1.0
            return {"batch": batch, "feature_a": features[0], "feature_b": features[1], "feature_c": features[2]}
        if op_type == "Transpose":
            dims = list(node.inputs[0].dims if node.inputs else node.outputs[0].dims if node.outputs else (1,))
            while len(dims) < 3:
                dims.append(1)
            return {"d0": float(dims[0]), "d1": float(dims[1]), "d2": float(dims[2]), "perm": 0.0}
        raise ValueError(f"unsupported NPU target op_type: {op_type}")
