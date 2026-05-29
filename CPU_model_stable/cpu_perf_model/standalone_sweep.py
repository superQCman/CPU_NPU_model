from __future__ import annotations

import csv
import copy
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_builder import DEFAULT_SPLIT_RATIOS, build_dataset
from .feature_engineering import load_hardware_identity
from .paths import PROJECT_ROOT


DTYPE_TO_TENSOR_PROTO = {
    "float32": 1,
    "uint8": 2,
    "int8": 3,
    "uint16": 4,
    "int16": 5,
    "int32": 6,
    "int64": 7,
    "bool": 9,
    "float16": 10,
    "double": 11,
    "uint32": 12,
    "uint64": 13,
}

OUTPUT_DTYPE_BY_OP = {
    "Shape": "int64",
}

DEFAULT_OPS = (
    "Add",
    "Mul",
    "Relu",
    "Sigmoid",
    "Gemm",
    "MatMul",
    "Concat",
    "Transpose",
    "Flatten",
    "Reshape",
    "Unsqueeze",
    "Shape",
    "ReduceSum",
    "Gather",
)

DEFAULT_STANDALONE_SWEEP_CONFIG_PATH = PROJECT_ROOT / "configs" / "standalone_op_sweep.default.json"
_NUMA_REEXEC_ACTIVE_ENV = "CPU_PERF_MODEL_NUMA_REEXEC_ACTIVE"
_NUMA_NODE_ENV = "CPU_PERF_MODEL_NUMA_NODE"
_NUMA_NUMACTL_ENV = "CPU_PERF_MODEL_NUMACTL_PATH"


@dataclass(frozen=True)
class ValueSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class InitializerSpec:
    name: str
    array: np.ndarray


@dataclass(frozen=True)
class StandaloneOpCase:
    source_name: str
    combo_name: str
    op_type: str
    batch_size: int
    num_indices_per_lookup: int
    num_threads: int
    node_name: str
    graph_inputs: tuple[ValueSpec, ...]
    initializers: tuple[InitializerSpec, ...]
    node_inputs: tuple[str, ...]
    output_name: str
    output_dtype: str
    output_shape: tuple[int, ...]
    attrs: dict[str, Any]
    input_feed_builder: Callable[[], dict[str, np.ndarray]]


def _import_onnx() -> Any:
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise ImportError("onnx is required for standalone operator sweep generation") from exc
    return onnx, helper, numpy_helper


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required for standalone operator sweep profiling") from exc
    return ort


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def current_numa_binding_metadata() -> dict[str, Any]:
    numa_node = os.environ.get(_NUMA_NODE_ENV, "").strip()
    numactl_path = os.environ.get(_NUMA_NUMACTL_ENV, "").strip()
    return {
        "single_numa_enforced": os.environ.get(_NUMA_REEXEC_ACTIVE_ENV) == "1",
        "numa_node": int(numa_node) if numa_node else None,
        "numactl_path": numactl_path or None,
    }


def static_input_feed_builder(*, specs: dict[str, tuple[tuple[int, ...], str, int]]) -> Callable[[], dict[str, np.ndarray]]:
    frozen_specs = {
        name: (tuple(shape), str(dtype), int(seed))
        for name, (shape, dtype, seed) in specs.items()
    }

    def _build() -> dict[str, np.ndarray]:
        return {
            name: input_array(shape, dtype, seed=seed)
            for name, (shape, dtype, seed) in frozen_specs.items()
        }

    return _build


def gather_input_feed_builder(
    *,
    data_shape: tuple[int, ...],
    index_shape: tuple[int, ...],
    table_size: int,
    data_seed: int,
) -> Callable[[], dict[str, np.ndarray]]:
    frozen_data_shape = tuple(data_shape)
    frozen_index_shape = tuple(index_shape)
    frozen_table_size = int(table_size)
    frozen_seed = int(data_seed)

    def _build() -> dict[str, np.ndarray]:
        indices = np.arange(np.prod(frozen_index_shape), dtype=np.int64).reshape(frozen_index_shape) % frozen_table_size
        return {
            "data": input_array(frozen_data_shape, "float32", seed=frozen_seed),
            "indices": indices,
        }

    return _build


def dtype_size(dtype: str) -> int:
    if dtype in {"float32", "int32", "uint32"}:
        return 4
    if dtype in {"float16", "int16", "uint16"}:
        return 2
    if dtype in {"int64", "uint64", "double"}:
        return 8
    if dtype in {"int8", "uint8", "bool"}:
        return 1
    raise ValueError(f"Unsupported dtype: {dtype}")


def shape_to_text(shape: tuple[int, ...]) -> str:
    return json.dumps([int(dim) for dim in shape])


def element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    return int(element_count(shape) * dtype_size(dtype))


def combo_name(op_type: str, params: dict[str, Any]) -> str:
    pieces = [op_type.lower()]
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            value_text = "x".join(str(item) for item in value)
        else:
            value_text = str(value)
        pieces.append(f"{key}{value_text}".replace("/", "_"))
    return "_".join(pieces)


def input_array(shape: tuple[int, ...], dtype: str, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if dtype == "float32":
        return rng.standard_normal(size=shape).astype(np.float32)
    if dtype == "int64":
        return rng.randint(0, 16, size=shape, dtype=np.int64)
    raise ValueError(f"Unsupported input dtype: {dtype}")


def _base_source_name(op_type: str) -> str:
    return f"standalone_{op_type.lower()}_v1"


def _placeholder_arch_metadata() -> tuple[str, str, str]:
    return ("standalone", "standalone", "standalone")


def _import_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML standalone sweep configs") from exc
    return yaml


def _load_structured_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        yaml = _import_yaml()
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported standalone sweep config format: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Standalone sweep config must be a mapping: {path}")
    return data


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _shape_tuple(values: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field_name} must be a non-empty list of integers")
    return tuple(int(item) for item in values)


def _int_list(values: Any, *, field_name: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field_name} must be a non-empty list of integers")
    return [int(item) for item in values]


def _list_of_dicts(config: dict[str, Any], key: str, *, op_type: str) -> list[dict[str, Any]]:
    values = config.get(key)
    if not isinstance(values, list) or not values or not all(isinstance(item, dict) for item in values):
        raise ValueError(f"{op_type} config field {key!r} must be a non-empty list of mappings")
    return values


def _int_values(config: dict[str, Any], key: str, *, op_type: str) -> list[int]:
    return _int_list(config.get(key), field_name=f"{op_type}.{key}")


def _shape_values(config: dict[str, Any], key: str, *, op_type: str) -> list[tuple[int, ...]]:
    values = config.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{op_type} config field {key!r} must be a non-empty list")
    return [_shape_tuple(item, field_name=f"{op_type}.{key}") for item in values]


def _ensure_same_element_count(
    input_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    *,
    op_type: str,
    field_name: str,
) -> None:
    input_elems = element_count(input_shape)
    target_elems = element_count(target_shape)
    if input_elems != target_elems:
        raise ValueError(
            f"{op_type} config {field_name} has mismatched element counts: "
            f"input_shape={list(input_shape)} ({input_elems}) vs "
            f"target_shape={list(target_shape)} ({target_elems})"
        )


def _ensure_axis_in_range(
    axis: int,
    *,
    rank: int,
    op_type: str,
    field_name: str,
    allow_endpoint: bool = False,
) -> None:
    upper_bound = rank if allow_endpoint else rank - 1
    if axis < 0 or axis > upper_bound:
        bound_text = f"[0, {rank}]" if allow_endpoint else f"[0, {rank - 1}]"
        raise ValueError(
            f"{op_type} config {field_name} has invalid axis {axis}; expected within {bound_text}"
        )


def _ensure_axes_in_range(
    axes: tuple[int, ...],
    *,
    rank: int,
    op_type: str,
    field_name: str,
    allow_endpoint: bool = False,
) -> None:
    if not axes:
        raise ValueError(f"{op_type} config {field_name} must contain at least one axis")
    if len(set(axes)) != len(axes):
        raise ValueError(f"{op_type} config {field_name} contains duplicate axes: {list(axes)}")
    for axis in axes:
        _ensure_axis_in_range(
            axis,
            rank=rank,
            op_type=op_type,
            field_name=field_name,
            allow_endpoint=allow_endpoint,
        )


def _ensure_permutation_valid(
    perm: tuple[int, ...],
    *,
    rank: int,
    op_type: str,
    field_name: str,
) -> None:
    if len(perm) != rank:
        raise ValueError(
            f"{op_type} config {field_name} must have length {rank}, got {len(perm)}"
        )
    if sorted(perm) != list(range(rank)):
        raise ValueError(
            f"{op_type} config {field_name} must be a permutation of [0, {rank - 1}], got {list(perm)}"
        )


def _ensure_concat_shapes_compatible(
    input_shapes: list[tuple[int, ...]],
    *,
    axis: int,
    op_type: str,
    field_name: str,
) -> None:
    if len(input_shapes) < 2:
        raise ValueError(f"{op_type} config {field_name} requires at least two inputs")
    rank = len(input_shapes[0])
    _ensure_axis_in_range(axis, rank=rank, op_type=op_type, field_name=f"{field_name}.axis")
    for index, shape in enumerate(input_shapes[1:], start=1):
        if len(shape) != rank:
            raise ValueError(
                f"{op_type} config {field_name} input rank mismatch: "
                f"input0 rank={rank}, input{index} rank={len(shape)}"
            )
        for dim_index, (left_dim, right_dim) in enumerate(zip(input_shapes[0], shape)):
            if dim_index == axis:
                continue
            if left_dim != right_dim:
                raise ValueError(
                    f"{op_type} config {field_name} has incompatible non-concat dimension at axis {dim_index}: "
                    f"{left_dim} vs {right_dim}"
                )


def load_standalone_sweep_config(config_path: Path | None) -> tuple[Path, dict[str, Any]]:
    default_path = DEFAULT_STANDALONE_SWEEP_CONFIG_PATH.resolve()
    config = _load_structured_config(default_path)
    if config_path is not None:
        override_path = config_path.resolve()
        config = _deep_merge(config, _load_structured_config(override_path))
        resolved_path = override_path
    else:
        resolved_path = default_path

    ops_config = config.get("ops")
    if not isinstance(ops_config, dict) or not ops_config:
        raise ValueError("Standalone sweep config must define a non-empty top-level 'ops' mapping")
    return resolved_path, config


def generate_add_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    shape_pairs = _list_of_dicts(op_config, "shape_pairs", op_type="Add")
    for pair_config, threads in product(shape_pairs, thread_values):
        shape = _shape_tuple(pair_config.get("lhs_shape"), field_name="Add.shape_pairs[].lhs_shape")
        rhs = _shape_tuple(pair_config.get("rhs_shape"), field_name="Add.shape_pairs[].rhs_shape")
        params = {"b": shape[0], "i": shape[-1], "r": len(shape), "t": threads}
        combo = combo_name("Add", params)
        lhs_name = "lhs"
        rhs_name = "rhs"
        output_shape = np.broadcast_shapes(shape, rhs)
        cases.append(
            StandaloneOpCase(
                source_name=_base_source_name("Add"),
                combo_name=combo,
                op_type="Add",
                batch_size=int(shape[0]),
                num_indices_per_lookup=int(shape[-1]),
                num_threads=threads,
                node_name="/Add",
                graph_inputs=(
                    ValueSpec(lhs_name, "float32", tuple(shape)),
                    ValueSpec(rhs_name, "float32", tuple(rhs)),
                ),
                initializers=(),
                node_inputs=(lhs_name, rhs_name),
                output_name="y",
                output_dtype="float32",
                output_shape=tuple(int(dim) for dim in output_shape),
                attrs={},
                input_feed_builder=static_input_feed_builder(
                    specs={
                        lhs_name: (tuple(shape), "float32", len(cases) * 7 + 1),
                        rhs_name: (tuple(rhs), "float32", len(cases) * 7 + 2),
                    }
                ),
            )
        )
    return cases


def generate_mul_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    shape_pairs = _list_of_dicts(op_config, "shape_pairs", op_type="Mul")
    for pair_config, threads in product(shape_pairs, thread_values):
        shape = _shape_tuple(pair_config.get("lhs_shape"), field_name="Mul.shape_pairs[].lhs_shape")
        rhs = _shape_tuple(pair_config.get("rhs_shape"), field_name="Mul.shape_pairs[].rhs_shape")
        output_shape = np.broadcast_shapes(shape, rhs)
        params = {"b": shape[0], "i": shape[-1], "r": len(shape), "t": threads}
        combo = combo_name("Mul", params)
        cases.append(
            StandaloneOpCase(
                source_name=_base_source_name("Mul"),
                combo_name=combo,
                op_type="Mul",
                batch_size=int(shape[0]),
                num_indices_per_lookup=int(shape[-1]),
                num_threads=threads,
                node_name="/Mul",
                graph_inputs=(
                    ValueSpec("lhs", "float32", tuple(shape)),
                    ValueSpec("rhs", "float32", tuple(rhs)),
                ),
                initializers=(),
                node_inputs=("lhs", "rhs"),
                output_name="y",
                output_dtype="float32",
                output_shape=tuple(int(dim) for dim in output_shape),
                attrs={},
                input_feed_builder=static_input_feed_builder(
                    specs={
                        "lhs": (tuple(shape), "float32", len(cases) * 11 + 1),
                        "rhs": (tuple(rhs), "float32", len(cases) * 11 + 2),
                    }
                ),
            )
        )
    return cases


def generate_unary_cases(op_type: str, thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    shapes = _shape_values(op_config, "shapes", op_type=op_type)
    for shape, threads in product(shapes, thread_values):
        params = {"b": shape[0], "i": shape[-1], "r": len(shape), "t": threads}
        combo = combo_name(op_type, params)
        cases.append(
            StandaloneOpCase(
                source_name=_base_source_name(op_type),
                combo_name=combo,
                op_type=op_type,
                batch_size=int(shape[0]),
                num_indices_per_lookup=int(shape[-1]),
                num_threads=threads,
                node_name=f"/{op_type}",
                graph_inputs=(ValueSpec("x", "float32", tuple(shape)),),
                initializers=(),
                node_inputs=("x",),
                output_name="y",
                output_dtype="float32",
                output_shape=tuple(shape),
                attrs={},
                input_feed_builder=static_input_feed_builder(
                    specs={"x": (tuple(shape), "float32", len(cases) * 13 + 1)}
                ),
            )
        )
    return cases


def generate_matmul_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    dims = _list_of_dicts(op_config, "dims", op_type="MatMul")
    for dim_config, threads in product(dims, thread_values):
        m = int(dim_config.get("m"))
        k = int(dim_config.get("k"))
        n = int(dim_config.get("n"))
        params = {"m": m, "k": k, "n": n, "t": threads}
        combo = combo_name("MatMul", params)
        cases.append(
            StandaloneOpCase(
                source_name=_base_source_name("MatMul"),
                combo_name=combo,
                op_type="MatMul",
                batch_size=m,
                num_indices_per_lookup=k,
                num_threads=threads,
                node_name="/MatMul",
                graph_inputs=(
                    ValueSpec("lhs", "float32", (m, k)),
                    ValueSpec("rhs", "float32", (k, n)),
                ),
                initializers=(),
                node_inputs=("lhs", "rhs"),
                output_name="y",
                output_dtype="float32",
                output_shape=(m, n),
                attrs={},
                input_feed_builder=static_input_feed_builder(
                    specs={
                        "lhs": ((m, k), "float32", len(cases) * 17 + 1),
                        "rhs": ((k, n), "float32", len(cases) * 17 + 2),
                    }
                ),
            )
        )
    return cases


def generate_gemm_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    dims = _list_of_dicts(op_config, "dims", op_type="Gemm")
    transb_values = _int_values(op_config, "transb_values", op_type="Gemm")
    for dim_config, transb, threads in product(dims, transb_values, thread_values):
        m = int(dim_config.get("m"))
        k = int(dim_config.get("k"))
        n = int(dim_config.get("n"))
        b_shape = (n, k) if transb else (k, n)
        bias = np.ones((n,), dtype=np.float32)
        params = {"m": m, "k": k, "n": n, "tb": transb, "t": threads}
        combo = combo_name("Gemm", params)
        cases.append(
            StandaloneOpCase(
                source_name=_base_source_name("Gemm"),
                combo_name=combo,
                op_type="Gemm",
                batch_size=m,
                num_indices_per_lookup=k,
                num_threads=threads,
                node_name="/Gemm",
                graph_inputs=(
                    ValueSpec("a", "float32", (m, k)),
                    ValueSpec("b", "float32", b_shape),
                ),
                initializers=(InitializerSpec("c", bias),),
                node_inputs=("a", "b", "c"),
                output_name="y",
                output_dtype="float32",
                output_shape=(m, n),
                attrs={"transB": int(transb)},
                input_feed_builder=static_input_feed_builder(
                    specs={
                        "a": ((m, k), "float32", len(cases) * 19 + 1),
                        "b": (b_shape, "float32", len(cases) * 19 + 2),
                    }
                ),
            )
        )
    return cases


def generate_concat_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    configs = _list_of_dicts(op_config, "configs", op_type="Concat")
    input_counts = _int_values(op_config, "input_counts", op_type="Concat")
    for config_index, config in enumerate(configs):
        for input_count in input_counts:
            if input_count not in {2, 3}:
                raise ValueError(
                    f"Concat config input_counts must contain only 2 or 3, got {input_count}"
                )
            for threads in thread_values:
                left_shape = _shape_tuple(config.get("left_shape"), field_name="Concat.configs[].left_shape")
                right_shape = _shape_tuple(config.get("right_shape"), field_name="Concat.configs[].right_shape")
                axis = int(config.get("axis"))
                input_shapes = [left_shape, right_shape]
                if input_count == 3:
                    input_shapes.append(right_shape)
                _ensure_concat_shapes_compatible(
                    input_shapes,
                    axis=axis,
                    op_type="Concat",
                    field_name=f"configs[{config_index}]",
                )
                names = tuple(f"x{i}" for i in range(len(input_shapes)))
                output_dims = list(left_shape)
                output_dims[axis] = sum(shape[axis] for shape in input_shapes)
                params = {"b": left_shape[0], "i": left_shape[-1], "axis": axis, "n": input_count, "t": threads}
                combo = combo_name("Concat", params)
                cases.append(
                    StandaloneOpCase(
                        source_name=_base_source_name("Concat"),
                        combo_name=combo,
                        op_type="Concat",
                        batch_size=int(left_shape[0]),
                        num_indices_per_lookup=int(left_shape[-1]),
                        num_threads=threads,
                        node_name="/Concat",
                        graph_inputs=tuple(ValueSpec(name, "float32", tuple(shape)) for name, shape in zip(names, input_shapes)),
                        initializers=(),
                        node_inputs=names,
                        output_name="y",
                        output_dtype="float32",
                        output_shape=tuple(output_dims),
                        attrs={"axis": axis},
                        input_feed_builder=static_input_feed_builder(
                            specs={
                                name: (tuple(shape), "float32", len(cases) * 23 + index + 1)
                                for index, (name, shape) in enumerate(zip(names, input_shapes))
                            }
                        ),
                    )
                )
    return cases


def generate_transpose_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    configs = _list_of_dicts(op_config, "configs", op_type="Transpose")
    for config_index, config in enumerate(configs):
        for threads in thread_values:
            shape = _shape_tuple(config.get("shape"), field_name="Transpose.configs[].shape")
            perm = tuple(_int_list(config.get("perm"), field_name="Transpose.configs[].perm"))
            _ensure_permutation_valid(
                perm,
                rank=len(shape),
                op_type="Transpose",
                field_name=f"configs[{config_index}].perm",
            )
            output_shape = tuple(shape[index] for index in perm)
            params = {"b": shape[0], "i": shape[-1], "p": "".join(str(x) for x in perm), "t": threads}
            combo = combo_name("Transpose", params)
            cases.append(
                StandaloneOpCase(
                    source_name=_base_source_name("Transpose"),
                    combo_name=combo,
                    op_type="Transpose",
                    batch_size=int(shape[0]),
                    num_indices_per_lookup=int(shape[-1]),
                    num_threads=threads,
                    node_name="/Transpose",
                    graph_inputs=(ValueSpec("x", "float32", tuple(shape)),),
                    initializers=(),
                    node_inputs=("x",),
                    output_name="y",
                    output_dtype="float32",
                    output_shape=output_shape,
                    attrs={"perm": list(perm)},
                    input_feed_builder=static_input_feed_builder(
                        specs={"x": (tuple(shape), "float32", len(cases) * 29 + 1)}
                    ),
                )
            )
    return cases


def generate_flatten_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    configs = _list_of_dicts(op_config, "configs", op_type="Flatten")
    for config_index, config in enumerate(configs):
        for threads in thread_values:
            shape = _shape_tuple(config.get("shape"), field_name="Flatten.configs[].shape")
            axis = int(config.get("axis"))
            _ensure_axis_in_range(
                axis,
                rank=len(shape),
                op_type="Flatten",
                field_name=f"configs[{config_index}].axis",
                allow_endpoint=True,
            )
            left = element_count(shape[:axis])
            right = element_count(shape[axis:])
            params = {"b": shape[0], "i": shape[-1], "a": axis, "t": threads}
            combo = combo_name("Flatten", params)
            cases.append(
                StandaloneOpCase(
                    source_name=_base_source_name("Flatten"),
                    combo_name=combo,
                    op_type="Flatten",
                    batch_size=int(shape[0]),
                    num_indices_per_lookup=int(shape[-1]),
                    num_threads=threads,
                    node_name="/Flatten",
                    graph_inputs=(ValueSpec("x", "float32", tuple(shape)),),
                    initializers=(),
                    node_inputs=("x",),
                    output_name="y",
                    output_dtype="float32",
                    output_shape=(left, right),
                    attrs={"axis": axis},
                    input_feed_builder=static_input_feed_builder(
                        specs={"x": (tuple(shape), "float32", len(cases) * 31 + 1)}
                    ),
                )
            )
    return cases


def generate_reshape_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    configs = _list_of_dicts(op_config, "configs", op_type="Reshape")
    for config_index, (config, threads) in enumerate(product(configs, thread_values)):
        input_shape = _shape_tuple(config.get("input_shape"), field_name="Reshape.configs[].input_shape")
        target_shape = _shape_tuple(config.get("target_shape"), field_name="Reshape.configs[].target_shape")
        _ensure_same_element_count(
            input_shape,
            target_shape,
            op_type="Reshape",
            field_name=f"configs[{config_index % len(configs)}]",
        )
        shape_init = np.asarray(target_shape, dtype=np.int64)
        params = {"b": input_shape[0], "i": input_shape[-1], "r": len(target_shape), "t": threads}
        combo = combo_name("Reshape", params)
        cases.append(
            StandaloneOpCase(
                source_name=_base_source_name("Reshape"),
                combo_name=combo,
                op_type="Reshape",
                batch_size=int(input_shape[0]),
                num_indices_per_lookup=int(input_shape[-1]),
                num_threads=threads,
                node_name="/Reshape",
                graph_inputs=(ValueSpec("x", "float32", tuple(input_shape)),),
                initializers=(InitializerSpec("shape", shape_init),),
                node_inputs=("x", "shape"),
                output_name="y",
                output_dtype="float32",
                output_shape=tuple(target_shape),
                attrs={},
                input_feed_builder=static_input_feed_builder(
                    specs={"x": (tuple(input_shape), "float32", len(cases) * 37 + 1)}
                ),
            )
        )
    return cases


def generate_unsqueeze_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    configs = _list_of_dicts(op_config, "configs", op_type="Unsqueeze")
    for config_index, config in enumerate(configs):
        for threads in thread_values:
            input_shape = _shape_tuple(config.get("input_shape"), field_name="Unsqueeze.configs[].input_shape")
            axes = tuple(_int_list(config.get("axes"), field_name="Unsqueeze.configs[].axes"))
            output_rank = len(input_shape) + len(axes)
            _ensure_axes_in_range(
                axes,
                rank=output_rank,
                op_type="Unsqueeze",
                field_name=f"configs[{config_index}].axes",
            )
            output_shape = list(input_shape)
            for axis in sorted(axes):
                output_shape.insert(axis, 1)
            axes_init = np.asarray(axes, dtype=np.int64)
            params = {"b": input_shape[0], "i": input_shape[-1], "a": "x".join(str(x) for x in axes), "t": threads}
            combo = combo_name("Unsqueeze", params)
            cases.append(
                StandaloneOpCase(
                    source_name=_base_source_name("Unsqueeze"),
                    combo_name=combo,
                    op_type="Unsqueeze",
                    batch_size=int(input_shape[0]),
                    num_indices_per_lookup=int(input_shape[-1]),
                    num_threads=threads,
                    node_name="/Unsqueeze",
                    graph_inputs=(ValueSpec("x", "float32", tuple(input_shape)),),
                    initializers=(InitializerSpec("axes", axes_init),),
                    node_inputs=("x", "axes"),
                    output_name="y",
                    output_dtype="float32",
                    output_shape=tuple(output_shape),
                    attrs={"axes": list(axes)},
                    input_feed_builder=static_input_feed_builder(
                        specs={"x": (tuple(input_shape), "float32", len(cases) * 41 + 1)}
                    ),
                )
            )
    return cases


def generate_shape_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    shapes = _shape_values(op_config, "shapes", op_type="Shape")
    for shape, threads in product(shapes, thread_values):
        params = {"b": shape[0], "i": shape[-1], "r": len(shape), "t": threads}
        combo = combo_name("Shape", params)
        cases.append(
            StandaloneOpCase(
                source_name=_base_source_name("Shape"),
                combo_name=combo,
                op_type="Shape",
                batch_size=int(shape[0]),
                num_indices_per_lookup=int(shape[-1]),
                num_threads=threads,
                node_name="/Shape",
                graph_inputs=(ValueSpec("x", "float32", tuple(shape)),),
                initializers=(),
                node_inputs=("x",),
                output_name="y",
                output_dtype="int64",
                output_shape=(len(shape),),
                attrs={},
                input_feed_builder=static_input_feed_builder(
                    specs={"x": (tuple(shape), "float32", len(cases) * 43 + 1)}
                ),
            )
        )
    return cases


def generate_reducesum_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    configs = _list_of_dicts(op_config, "configs", op_type="ReduceSum")
    for config_index, config in enumerate(configs):
        for threads in thread_values:
            input_shape = _shape_tuple(config.get("input_shape"), field_name="ReduceSum.configs[].input_shape")
            axes = tuple(_int_list(config.get("axes"), field_name="ReduceSum.configs[].axes"))
            keepdims = int(config.get("keepdims"))
            _ensure_axes_in_range(
                axes,
                rank=len(input_shape),
                op_type="ReduceSum",
                field_name=f"configs[{config_index}].axes",
            )
            if keepdims not in {0, 1}:
                raise ValueError(
                    f"ReduceSum config configs[{config_index}].keepdims must be 0 or 1, got {keepdims}"
                )
            axes_set = set(axes)
            output_shape = []
            for index, dim in enumerate(input_shape):
                if index in axes_set:
                    if keepdims:
                        output_shape.append(1)
                else:
                    output_shape.append(dim)
            if not output_shape:
                output_shape = [1]
            axes_init = np.asarray(axes, dtype=np.int64)
            params = {"b": input_shape[0], "i": input_shape[-1], "a": "x".join(str(x) for x in axes), "k": keepdims, "t": threads}
            combo = combo_name("ReduceSum", params)
            cases.append(
                StandaloneOpCase(
                    source_name=_base_source_name("ReduceSum"),
                    combo_name=combo,
                    op_type="ReduceSum",
                    batch_size=int(input_shape[0]),
                    num_indices_per_lookup=int(input_shape[-1]),
                    num_threads=threads,
                    node_name="/ReduceSum",
                    graph_inputs=(ValueSpec("x", "float32", tuple(input_shape)),),
                    initializers=(InitializerSpec("axes", axes_init),),
                    node_inputs=("x", "axes"),
                    output_name="y",
                    output_dtype="float32",
                    output_shape=tuple(output_shape),
                    attrs={"axes": list(axes), "keepdims": int(keepdims)},
                    input_feed_builder=static_input_feed_builder(
                        specs={"x": (tuple(input_shape), "float32", len(cases) * 47 + 1)}
                    ),
                )
            )
    return cases


def generate_gather_cases(thread_values: list[int], op_config: dict[str, Any]) -> list[StandaloneOpCase]:
    cases: list[StandaloneOpCase] = []
    configs = _list_of_dicts(op_config, "configs", op_type="Gather")
    for config, threads in product(configs, thread_values):
        table_size = int(config.get("table_size"))
        dim = int(config.get("dim"))
        batch_size = int(config.get("batch_size"))
        lookup_count = int(config.get("lookup_count"))
        data_shape = (table_size, dim)
        index_shape = (batch_size, lookup_count)
        params = {"b": batch_size, "i": lookup_count, "tbl": table_size, "d": dim, "t": threads}
        combo = combo_name("Gather", params)
        cases.append(
            StandaloneOpCase(
                source_name=_base_source_name("Gather"),
                combo_name=combo,
                op_type="Gather",
                batch_size=batch_size,
                num_indices_per_lookup=lookup_count,
                num_threads=threads,
                node_name="/Gather",
                graph_inputs=(
                    ValueSpec("data", "float32", data_shape),
                    ValueSpec("indices", "int64", index_shape),
                ),
                initializers=(),
                node_inputs=("data", "indices"),
                output_name="y",
                output_dtype="float32",
                output_shape=(batch_size, lookup_count, dim),
                attrs={"axis": 0},
                input_feed_builder=gather_input_feed_builder(
                    data_shape=data_shape,
                    index_shape=index_shape,
                    table_size=table_size,
                    data_seed=len(cases) * 53 + 1,
                ),
            )
        )
    return cases


CASE_GENERATORS = {
    "Add": generate_add_cases,
    "Mul": generate_mul_cases,
    "Relu": lambda thread_values, op_config: generate_unary_cases("Relu", thread_values, op_config),
    "Sigmoid": lambda thread_values, op_config: generate_unary_cases("Sigmoid", thread_values, op_config),
    "MatMul": generate_matmul_cases,
    "Gemm": generate_gemm_cases,
    "Concat": generate_concat_cases,
    "Transpose": generate_transpose_cases,
    "Flatten": generate_flatten_cases,
    "Reshape": generate_reshape_cases,
    "Unsqueeze": generate_unsqueeze_cases,
    "Shape": generate_shape_cases,
    "ReduceSum": generate_reducesum_cases,
    "Gather": generate_gather_cases,
}


def build_cases(
    op_types: list[str],
    thread_values: list[int],
    max_combos_per_op: int,
    sweep_config: dict[str, Any],
) -> dict[str, list[StandaloneOpCase]]:
    cases_by_source: dict[str, list[StandaloneOpCase]] = {}
    op_configs = sweep_config.get("ops", {})
    for op_type in op_types:
        if op_type not in CASE_GENERATORS:
            raise ValueError(f"Unsupported standalone op type: {op_type}")
        op_config = op_configs.get(op_type)
        if not isinstance(op_config, dict):
            raise ValueError(f"Standalone sweep config is missing an 'ops.{op_type}' mapping")
        op_cases = CASE_GENERATORS[op_type](thread_values, op_config)
        if max_combos_per_op > 0:
            op_cases = op_cases[:max_combos_per_op]
        cases_by_source[_base_source_name(op_type)] = op_cases
    return cases_by_source


def save_model(case: StandaloneOpCase, model_path: Path) -> None:
    onnx, helper, numpy_helper = _import_onnx()

    inputs = [
        helper.make_tensor_value_info(spec.name, DTYPE_TO_TENSOR_PROTO[spec.dtype], list(spec.shape))
        for spec in case.graph_inputs
    ]
    outputs = [
        helper.make_tensor_value_info(case.output_name, DTYPE_TO_TENSOR_PROTO[case.output_dtype], list(case.output_shape))
    ]
    node_attrs = dict(case.attrs)
    # Keep structural attrs in feature metadata even when the selected ONNX
    # opset expects them as auxiliary tensor inputs rather than node attributes.
    if case.op_type in {"Unsqueeze", "ReduceSum"}:
        node_attrs.pop("axes", None)
    node = helper.make_node(
        case.op_type,
        list(case.node_inputs),
        [case.output_name],
        name=case.node_name,
        **node_attrs,
    )
    initializers = [numpy_helper.from_array(spec.array, name=spec.name) for spec in case.initializers]
    graph = helper.make_graph([node], case.combo_name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, producer_name="cpu_perf_model_standalone", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 11
    onnx.checker.check_model(model)
    ensure_dir(model_path.parent)
    onnx.save_model(model, model_path)


def build_op_shapes_rows(case: StandaloneOpCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    node_ref = f"0:{case.node_name}:{case.op_type}"
    initializer_lookup = {spec.name: spec for spec in case.initializers}
    for port_idx, input_name in enumerate(case.node_inputs):
        if input_name in initializer_lookup:
            spec = initializer_lookup[input_name]
            producer = "initializer"
            dtype = str(spec.array.dtype)
            shape = tuple(int(dim) for dim in spec.array.shape)
        else:
            spec = next(graph_input for graph_input in case.graph_inputs if graph_input.name == input_name)
            producer = "graph_input"
            dtype = spec.dtype
            shape = spec.shape
        rows.append(
            {
                "node_idx": 0,
                "node_name": case.node_name,
                "op_type": case.op_type,
                "tensor_role": "input",
                "port_idx": port_idx,
                "tensor_name": input_name,
                "dtype": dtype,
                "shape": shape_to_text(shape),
                "is_dynamic": False,
                "shape_source": "static",
                "producer_node": producer,
                "consumer_nodes": node_ref,
            }
        )
    rows.append(
        {
            "node_idx": 0,
            "node_name": case.node_name,
            "op_type": case.op_type,
            "tensor_role": "output",
            "port_idx": 0,
            "tensor_name": case.output_name,
            "dtype": case.output_dtype,
            "shape": shape_to_text(case.output_shape),
            "is_dynamic": False,
            "shape_source": "static",
            "producer_node": node_ref,
            "consumer_nodes": "",
        }
    )
    return rows


def parse_profile_events(profile_json: Path, expected_node_name: str) -> list[dict[str, Any]]:
    events = json.loads(profile_json.read_text(encoding="utf-8"))
    normalized = f"{expected_node_name}_kernel_time"
    records: list[dict[str, Any]] = []
    for event in events:
        if event.get("cat") != "Node":
            continue
        name = str(event.get("name", ""))
        args = event.get("args", {})
        if args.get("provider") != "CPUExecutionProvider":
            continue
        if name not in {expected_node_name, normalized}:
            continue
        records.append(
            {
                "node_name": name,
                "op_name": args.get("op_name", ""),
                "node_index": int(args.get("node_index", 0)),
                "provider": args.get("provider", ""),
                "ts_us": float(event.get("ts", 0.0)),
                "dur_us": float(event.get("dur", 0.0)),
                "output_size": float(args.get("output_size", 0.0) or 0.0),
                "activation_size": float(args.get("activation_size", 0.0) or 0.0),
                "parameter_size": float(args.get("parameter_size", 0.0) or 0.0),
                "input_type_shape": str(args.get("input_type_shape", [])),
                "output_type_shape": str(args.get("output_type_shape", [])),
                "thread_scheduling_stats": args.get("thread_scheduling_stats", {}),
            }
        )
    return sorted(records, key=lambda row: row["ts_us"])


def build_timeline_rows(records: list[dict[str, Any]], case: StandaloneOpCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_idx, record in enumerate(records):
        start_us = float(record["ts_us"])
        dur_us = float(record["dur_us"])
        rows.append(
            {
                "batch_idx": batch_idx,
                "task_name": "standalone",
                "op_name": case.op_type,
                "node_name": record["node_name"],
                "provider": record["provider"],
                "lane": 0,
                "start_us": start_us,
                "end_us": start_us + dur_us,
                "dur_us": dur_us,
                "label": f"standalone:{case.op_type}",
            }
        )
    return rows


def build_single_concurrency_rows(timeline_rows: list[dict[str, Any]], case: StandaloneOpCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in timeline_rows:
        rows.append(
            {
                "start_us": row["start_us"],
                "end_us": row["end_us"],
                "dur_us": row["dur_us"],
                "concurrency": 1,
                "labels": f"standalone:{case.op_type}",
                "tasks": "standalone",
                "providers": "CPUExecutionProvider",
            }
        )
    return rows


def _gather_unique_idx_count(case: StandaloneOpCase) -> float:
    data_spec = next((spec for spec in case.graph_inputs if spec.name == "data"), None)
    index_spec = next((spec for spec in case.graph_inputs if spec.name == "indices"), None)
    if data_spec is None or index_spec is None or not data_spec.shape:
        return 0.0
    total_requests = int(np.prod(index_spec.shape))
    table_size = int(data_spec.shape[0])
    return float(min(table_size, total_requests))


def _gather_working_set_eff_bytes(case: StandaloneOpCase) -> float:
    if case.op_type != "Gather" or not case.graph_inputs:
        return 0.0
    table_spec = case.graph_inputs[0]
    if len(table_spec.shape) < 2:
        return 0.0
    row_bytes = float(element_count(table_spec.shape[1:]) * dtype_size(table_spec.dtype))
    return _gather_unique_idx_count(case) * row_bytes


def build_feature_row(
    case: StandaloneOpCase,
    records: list[dict[str, Any]],
    hardware_identity: dict[str, str],
) -> dict[str, Any]:
    _, arch_mlp_bot, arch_mlp_top = _placeholder_arch_metadata()
    first = records[0] if records else {}
    output_size = float(first.get("output_size", 0.0) or 0.0)
    activation_size = float(first.get("activation_size", 0.0) or 0.0)
    parameter_size = float(first.get("parameter_size", 0.0) or 0.0)
    if output_size <= 0:
        output_size = float(tensor_bytes(case.output_shape, case.output_dtype))
    if activation_size <= 0:
        activation_size = float(
            sum(tensor_bytes(spec.shape, spec.dtype) for spec in case.graph_inputs) + output_size
        )
    if parameter_size <= 0:
        parameter_size = float(sum(spec.array.nbytes for spec in case.initializers))
    avg_dur = float(np.mean([record["dur_us"] for record in records])) if records else 0.0

    return {
        "batch_size": case.batch_size,
        "num_indices_per_lookup": case.num_indices_per_lookup,
        "arch_embedding_size": "standalone",
        "arch_mlp_bot": arch_mlp_bot,
        "arch_mlp_top": arch_mlp_top,
        "node_idx": 0,
        "node_name": case.node_name,
        "op_type": case.op_type,
        "num_threads": case.num_threads,
        "has_cpu_profile": 1,
        "cpu_profile_missing_reason": "",
        "dur_us": avg_dur,
        "output_size": output_size,
        "activation_size": activation_size,
        "parameter_size": parameter_size,
        "hardware_id": str(hardware_identity.get("hardware_id", "unknown_hw")),
        "hardware_family": str(hardware_identity.get("hardware_family", "unknown_family")),
        "hardware_profile_name": str(hardware_identity.get("hardware_profile_name", "unknown_profile")),
        "backend_family": str(hardware_identity.get("backend_family", "ort_cpu")),
        "op_attrs_json": json.dumps(case.attrs, sort_keys=True),
        "attr_axis": case.attrs.get("axis"),
        "attr_perm_json": json.dumps(case.attrs.get("perm", [])) if "perm" in case.attrs else "",
        "attr_axes_json": json.dumps(case.attrs.get("axes", [])) if "axes" in case.attrs else "",
        "attr_keepdims": case.attrs.get("keepdims"),
        "gather_unique_idx_count": _gather_unique_idx_count(case) if case.op_type == "Gather" else np.nan,
        "gather_working_set_eff_bytes": _gather_working_set_eff_bytes(case) if case.op_type == "Gather" else np.nan,
    }


def create_session_options(ort: Any, num_threads: int, profiling: bool) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = profiling
    options.intra_op_num_threads = int(num_threads)
    options.inter_op_num_threads = 1
    if hasattr(ort, "ExecutionMode"):
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if hasattr(ort, "GraphOptimizationLevel"):
        # Keep lightweight shape-only ops visible in the CPU profile instead of
        # letting graph rewrites fold them away before execution.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


def profile_case(case: StandaloneOpCase, model_path: Path, profile_dir: Path, warmup_runs: int, num_batches: int) -> Path:
    ort = _import_onnxruntime()
    input_feed = case.input_feed_builder()

    warmup_session = ort.InferenceSession(
        str(model_path),
        sess_options=create_session_options(ort, case.num_threads, profiling=False),
        providers=["CPUExecutionProvider"],
    )
    for _ in range(max(0, warmup_runs)):
        warmup_session.run(None, input_feed)

    session = ort.InferenceSession(
        str(model_path),
        sess_options=create_session_options(ort, case.num_threads, profiling=True),
        providers=["CPUExecutionProvider"],
    )
    for _ in range(max(1, num_batches)):
        session.run(None, input_feed)
    raw_profile = Path(session.end_profiling())
    target_path = profile_dir / "ort_profile.json"
    ensure_dir(profile_dir)
    if raw_profile.exists():
        shutil.move(str(raw_profile), str(target_path))
    return target_path


def materialize_case(
    case: StandaloneOpCase,
    source_root: Path,
    warmup_runs: int,
    num_batches: int,
    hardware_identity: dict[str, str],
    numa_binding: dict[str, Any],
) -> dict[str, Any]:
    model_path = source_root / "models" / f"{case.combo_name}.onnx"
    feature_csv = source_root / "features" / f"{case.combo_name}.csv"
    op_shapes_csv = source_root / "op_shapes" / f"{case.combo_name}.csv"
    profile_dir = source_root / "onnx_profiles" / case.combo_name
    log_dir = source_root / "logs" / case.combo_name

    save_model(case, model_path)
    write_csv(build_op_shapes_rows(case), op_shapes_csv)
    profile_json = profile_case(case, model_path, profile_dir, warmup_runs=warmup_runs, num_batches=num_batches)
    records = parse_profile_events(profile_json, case.node_name)
    if not records:
        raise RuntimeError(f"No CPU profile node records were recovered for {case.combo_name}")

    timeline_rows = build_timeline_rows(records, case)
    write_csv(timeline_rows, profile_dir / "branch_parallel_op_timeline.csv")
    concurrency_rows = build_single_concurrency_rows(timeline_rows, case)
    write_csv(concurrency_rows, profile_dir / "branch_parallel_concurrency_segments.csv")
    write_csv(concurrency_rows, profile_dir / "branch_parallel_op_concurrency_segments.csv")
    write_csv(records, profile_dir / "cpu_node_events.csv")
    write_csv([build_feature_row(case, records, hardware_identity)], feature_csv)

    ensure_dir(log_dir)
    (log_dir / "run_ort.log").write_text(
        (
            f"Intra threads : {case.num_threads}\n"
            "Provider: CPUExecutionProvider\n"
            f"Single NUMA enforced: {numa_binding.get('single_numa_enforced', False)}\n"
            f"NUMA node: {numa_binding.get('numa_node', '')}\n"
            f"numactl path: {numa_binding.get('numactl_path', '')}\n"
        ),
        encoding="utf-8",
    )
    return {
        "combo_name": case.combo_name,
        "model_path": str(model_path),
        "feature_csv": str(feature_csv),
        "op_shapes_csv": str(op_shapes_csv),
        "profile_dir": str(profile_dir),
        "profile_json": str(profile_json),
    }


def write_source_manifest(output_root: Path, source_roots: dict[str, Path]) -> Path:
    sources = []
    for source_name, source_root in sorted(source_roots.items()):
        sources.append(
            {
                "source_name": source_name,
                "case_id": source_name,
                "source_mode": "standalone",
                "feature_root": str(source_root / "features"),
                "feature_glob": "*.csv",
                "op_shapes_root": str(source_root / "op_shapes"),
                "profile_root": str(source_root / "onnx_profiles"),
                "log_root": str(source_root / "logs"),
            }
        )
    manifest_path = output_root / "standalone_source_manifest.json"
    manifest_path.write_text(json.dumps({"sources": sources}, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def resolve_ops(requested_ops: list[str]) -> list[str]:
    if not requested_ops or requested_ops == ["all"]:
        return list(DEFAULT_OPS)
    resolved: list[str] = []
    seen: set[str] = set()
    for item in requested_ops:
        if item == "all":
            for op in DEFAULT_OPS:
                if op not in seen:
                    resolved.append(op)
                    seen.add(op)
            continue
        match = next((op for op in DEFAULT_OPS if op.lower() == item.lower()), None)
        if match is None:
            raise ValueError(f"Unsupported standalone op type: {item}")
        if match not in seen:
            resolved.append(match)
            seen.add(match)
    return resolved


def run_standalone_sweep(
    *,
    output_root: Path,
    requested_ops: list[str],
    thread_values: list[int],
    sweep_config_path: Path | None,
    max_combos_per_op: int,
    warmup_runs: int,
    num_batches: int,
    dataset_output_dir: Path | None,
    feature_dialect: str,
    hardware_profile_path: Path | None,
    drop_first_profile_batch: bool,
    profile_instability_metric: str,
    profile_instability_threshold: float,
    disable_profile_stability_filter: bool,
) -> dict[str, Any]:
    ops = resolve_ops(requested_ops)
    resolved_sweep_config_path, sweep_config = load_standalone_sweep_config(sweep_config_path)
    cases_by_source = build_cases(
        ops,
        thread_values,
        max_combos_per_op=max_combos_per_op,
        sweep_config=sweep_config,
    )
    ensure_dir(output_root)
    resolved_config_snapshot_path = output_root / "standalone_sweep_config_resolved.json"
    resolved_config_snapshot_path.write_text(
        json.dumps(sweep_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    source_roots: dict[str, Path] = {}
    materialized: dict[str, list[dict[str, Any]]] = {}
    hardware_identity = load_hardware_identity(hardware_profile_path)
    numa_binding = current_numa_binding_metadata()
    for source_name, cases in cases_by_source.items():
        source_root = ensure_dir(output_root / source_name)
        source_roots[source_name] = source_root
        materialized[source_name] = []
        for case in cases:
            materialized[source_name].append(
                materialize_case(
                    case,
                    source_root,
                    warmup_runs=warmup_runs,
                    num_batches=num_batches,
                    hardware_identity=hardware_identity,
                    numa_binding=numa_binding,
                )
            )

    manifest_path = write_source_manifest(output_root, source_roots)
    sweep_summary = {
        "output_root": str(output_root),
        "ops": ops,
        "thread_values": list(thread_values),
        "sweep_config_path": str(resolved_sweep_config_path),
        "resolved_sweep_config_snapshot": str(resolved_config_snapshot_path),
        "max_combos_per_op": int(max_combos_per_op),
        "warmup_runs": int(warmup_runs),
        "num_batches": int(num_batches),
        "numa_binding": numa_binding,
        "source_count": len(source_roots),
        "combo_count_by_source": {source: len(items) for source, items in materialized.items()},
        "source_manifest": str(manifest_path),
    }
    (output_root / "standalone_sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dataset_summary: dict[str, Any] | None = None
    if dataset_output_dir is not None:
        dataset_summary = build_dataset(
            output_dir=dataset_output_dir,
            case_pattern="*",
            selected_case_ids=None,
            source_manifest=manifest_path,
            max_files_per_case=0,
            group_column="sample_group",
            seed=42,
            ratios=dict(DEFAULT_SPLIT_RATIOS),
            feature_dialect=feature_dialect,
            hardware_profile_path=hardware_profile_path,
            drop_first_profile_batch=drop_first_profile_batch,
            profile_instability_metric=profile_instability_metric,
            profile_instability_threshold=profile_instability_threshold,
            disable_profile_stability_filter=disable_profile_stability_filter,
        )

    return {
        "sweep_summary": sweep_summary,
        "source_manifest": str(manifest_path),
        "dataset_summary": dataset_summary,
    }
