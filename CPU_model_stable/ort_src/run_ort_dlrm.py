#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLRM ONNX Runtime 推理脚本
=========================
使用 ORT（ONNX Runtime）对 DLRM 的 ONNX 图进行推理，
支持 CANNExecutionProvider（Ascend NPU）+ CPUExecutionProvider 混合异构执行。

NPU 支持算子自动卸载到 NPU，不支持的算子自动回退到 CPU。

运行方式：
  # 仅 CPU（不需要 CANN ORT）
  python run_ort_dlrm.py --onnx-path ./dlrm_onnx/dlrm_s_pytorch.onnx

  # NPU+CPU 混合
  python run_ort_dlrm.py --onnx-path ./dlrm_onnx/dlrm_s_pytorch.onnx \
      --use-cann --device-id 0

  # 带 profiling
  python run_ort_dlrm.py --onnx-path ./dlrm_onnx/dlrm_s_pytorch.onnx \
      --use-cann --enable-profiling --num-batches 5

环境依赖：
  - onnxruntime（带 CANNExecutionProvider，见 setup_ort_cann.sh）
  - CANN Toolkit 已设置环境变量（source /data/qc/Ascend/ascend-toolkit/set_env.sh）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CANN 环境变量初始化
# 若脚本不通过 set_env.sh 启动，则在这里自动补充
# ─────────────────────────────────────────────────────────────────────────────
_CANN_TOOLKIT_ROOT = Path(
    os.environ.get("ASCEND_TOOLKIT_HOME")
    or os.environ.get("ASCEND_HOME_PATH")
    or "/data/qc/Ascend/ascend-toolkit/latest"
)


def _setup_cann_env() -> bool:
    """
    设置 Ascend CANN 所需的环境变量（LD_LIBRARY_PATH、ASCEND_HOME_PATH 等）。
    返回 True 表示 CANN 库目录存在，False 表示 CANN 未安装。
    """
    root = _CANN_TOOLKIT_ROOT
    if not root.exists():
        return False

    lib64 = str(root / "lib64")
    plugin_ops = str(root / "lib64/plugin/opskernel")
    plugin_nn  = str(root / "lib64/plugin/nnengine")
    import platform
    arch = platform.machine()
    tiling_lib = str(root / f"opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/{arch}")

    # 追加到 LD_LIBRARY_PATH（去重）
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = [lib64, plugin_ops, plugin_nn, tiling_lib]
    existing = set(old_ld.split(":")) if old_ld else set()
    add = [p for p in new_paths if p not in existing and Path(p).exists()]
    if add:
        os.environ["LD_LIBRARY_PATH"] = ":".join(add) + (":" + old_ld if old_ld else "")

    # 其他必需变量
    env_defaults = {
        "ASCEND_TOOLKIT_HOME":   str(root),
        "ASCEND_HOME_PATH":      str(root),
        "ASCEND_AICPU_PATH":     str(root),
        "ASCEND_OPP_PATH":       str(root / "opp"),
        "TOOLCHAIN_HOME":        str(root / "toolkit"),
    }
    for k, v in env_defaults.items():
        if k not in os.environ:
            os.environ[k] = v

    # PYTHONPATH
    py_pkg = str(root / "python/site-packages")
    tbe_path = str(root / "opp/built-in/op_impl/ai_core/tbe")
    old_py = os.environ.get("PYTHONPATH", "")
    add_py = [p for p in [py_pkg, tbe_path] if p not in old_py.split(":") and Path(p).exists()]
    if add_py:
        os.environ["PYTHONPATH"] = ":".join(add_py) + (":" + old_py if old_py else "")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# ONNX 模型预处理（CANN 兼容性补丁）
# ─────────────────────────────────────────────────────────────────────────────

def _patch_model_for_cann(onnx_path: str) -> str:
    """
    预处理 ONNX 模型，修复已知 CANN 运行时兼容性问题：

    问题：
      1. 当前 ORT+CANN 组合下，ONNX Gemm 往往不会被 CANNExecutionProvider 接管，
         但等价的 MatMul + Add 可以正常落到 CANN。
      2. CANN 单算子模式（enable_cann_graph=0）下，当 Mul 的两个输入 shape 不同
         （需要广播）时，CANN 会把该算子内部分解为 BroadcastToD +
         ElementwiseMul，而 BroadcastToD 在 ACL_COMPILE_SYS（预编译库）中
         缺少对应 shape 组合的 kernel，抛出 ACL_ERROR_GE_FAILURE。

    修复（三步）：
      1. Gemm 改写    —— 将 Gemm 重写为等价的 MatMul(+Mul)+Add 链，
                        让线性层默认走 CANN 支持更好的算子路径。
      2. dtype 修复   —— 将所有含 int64 输入的 Mul 节点替换为
                        Cast(→fp32) + Mul(fp32) + Cast(→int64)，
                        避免 int64 BroadcastToD 在预编译库中完全缺失。
      3. shape 修复   —— 对第 2 步得到的 fp32 Mul（或原始 fp32 广播 Mul），
                        通过 onnx.shape_inference 推断各输入 rank；
                        若任意输入 rank < 参考输入 rank（即存在广播维度差），
                        则在 fp32 Mul 之前插入 Shape + Expand 节点，
                        将低 rank 输入显式扩展至高 rank 输入的 shape，
                        使 Mul 退化为等形 element-wise，彻底避免 BroadcastToD。

    返回修改后的临时 onnx 文件路径（若无需修改则返回原路径）。
    """
    try:
        import onnx
        from onnx import TensorProto, numpy_helper, helper, shape_inference as onnx_si
    except ImportError:
        return onnx_path  # 无 onnx 包，跳过预处理

    model = onnx.load(onnx_path)

    def _infer_shapes_if_possible(curr_model: Any) -> Any:
        try:
            return onnx_si.infer_shapes(curr_model)
        except Exception:
            return curr_model

    def _collect_tensor_metadata(curr_graph: Any) -> Tuple[Dict[str, int], Dict[str, list], set[str]]:
        dtype_map_local: Dict[str, int] = {}
        shape_map_local: Dict[str, list] = {}
        known_value_infos: set[str] = set()

        for vi in list(curr_graph.value_info) + list(curr_graph.input) + list(curr_graph.output):
            known_value_infos.add(vi.name)
            if vi.type.HasField("tensor_type"):
                dtype_map_local[vi.name] = vi.type.tensor_type.elem_type
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
                dims: list = []
                for d in vi.type.tensor_type.shape.dim:
                    if d.HasField("dim_value"):
                        dims.append(d.dim_value)
                    elif d.HasField("dim_param"):
                        dims.append(d.dim_param)
                    else:
                        dims.append(None)
                shape_map_local[vi.name] = dims

        for init in curr_graph.initializer:
            dtype_map_local[init.name] = init.data_type
            shape_map_local[init.name] = list(init.dims)

        for node in curr_graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name == "value":
                    dtype_map_local[node.output[0]] = attr.t.data_type
                    shape_map_local[node.output[0]] = list(attr.t.dims)

        return dtype_map_local, shape_map_local, known_value_infos

    model = _infer_shapes_if_possible(model)
    graph = model.graph
    dtype_map, shape_map, known_value_info_names = _collect_tensor_metadata(graph)

    def _is_broadcast_dim(d_src, d_ref) -> bool:
        """判断 d_src 维是否需要广播到 d_ref：d_src==1 且 d_ref≠1（或动态）。"""
        if not isinstance(d_src, int) or d_src != 1:
            return False
        if isinstance(d_ref, int) and d_ref == 1:
            return False  # 两侧都是 1，无需广播
        return True  # d_src==1, d_ref>1 或动态 → 需要广播

    def _needs_expand(shape_src: list, shape_ref: list) -> bool:
        """判断 shape_src 是否需要 Expand 到与 shape_ref 对齐。"""
        if len(shape_src) < len(shape_ref):
            return True  # rank 不足，需要扩展（广播规则：左侧补 1）
        if len(shape_src) != len(shape_ref):
            return False  # shape_ref 更小时不应展开 src（ref 是更大的那个）
        return any(_is_broadcast_dim(ds, dr) for ds, dr in zip(shape_src, shape_ref))

    def _pick_ref_idx(shapes: list) -> int:
        """
        从多个 shape 中选出"最大" shape 的 index 作为广播参考。
        优先选 rank 最大的；rank 相同时选静态维中最大乘积的。
        """
        def _score(s):
            if s is None:
                return (0, 0)
            rank = len(s)
            prod = 1
            for d in s:
                if isinstance(d, int) and d > 0:
                    prod *= d
            return (rank, prod)
        return max(range(len(shapes)), key=lambda i: _score(shapes[i]))

    def _expand_to_ref_if_needed(
        target_nodes: list,
        inp: str,
        ref_inp: str,
        base: str,
        idx: int,
        *,
        prefix: str,
        ref_shape_name: str,
        ref_shape_node_inserted: bool,
        ref_shape: list | None,
        src_shape: list | None,
        elem_type: int = 1,
        force: bool = False,
    ) -> tuple[str, bool]:
        """
        如果 inp 的 shape 需要显式扩展到 ref_shape，则插入 Shape(ref) + Expand(inp, Shape(ref)).
        返回 (最终输入名, 是否插入了新的 Shape 节点)。
        """
        if src_shape is None and not force:
            return inp, ref_shape_node_inserted
        if not force and (ref_shape is None or not _needs_expand(src_shape, ref_shape)):
            return inp, ref_shape_node_inserted

        if not ref_shape_node_inserted:
            shape_node = helper.make_node(
                "Shape",
                inputs=[ref_inp],
                outputs=[ref_shape_name],
                name=f"__cann_patch_{base}_{prefix}_broadcast_shape",
            )
            target_nodes.append(shape_node)
            dtype_map[ref_shape_name] = INT64
            _register_value_info(ref_shape_name, INT64)
            ref_shape_node_inserted = True

        exp_name = f"__cann_patch_{base}_{prefix}_expanded_{idx}"
        exp_node = helper.make_node(
            "Expand",
            inputs=[inp, ref_shape_name],
            outputs=[exp_name],
            name=f"__cann_patch_{base}_{prefix}_expand_node_{idx}",
        )
        target_nodes.append(exp_node)
        dtype_map[exp_name] = elem_type
        if ref_shape is not None:
            shape_map[exp_name] = list(ref_shape)
        _register_value_info(exp_name, elem_type)
        return exp_name, ref_shape_node_inserted

    INT64  = TensorProto.INT64   # 7
    FLOAT  = TensorProto.FLOAT   # 1

    def _register_value_info(name: str, elem_type: int) -> None:
        if not name or name in known_value_info_names:
            return
        graph.value_info.append(helper.make_tensor_value_info(name, elem_type, None))
        known_value_info_names.add(name)

    def _make_scalar_const_node(base: str, suffix: str, value: float) -> Tuple[Any, str]:
        const_name = f"__cann_patch_{base}_{suffix}_const"
        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[const_name],
            name=f"__cann_patch_{base}_{suffix}_const_node",
            value=numpy_helper.from_array(np.array(value, dtype=np.float32), name=const_name),
        )
        dtype_map[const_name] = FLOAT
        shape_map[const_name] = []
        _register_value_info(const_name, FLOAT)
        return const_node, const_name

    def _transpose_shape(shape: list | None) -> list | None:
        if shape is None or len(shape) < 2:
            return shape
        swapped = list(shape)
        swapped[-1], swapped[-2] = swapped[-2], swapped[-1]
        return swapped

    def _infer_matmul_output_shape(lhs_shape: list | None, rhs_shape: list | None) -> list | None:
        """
        保守推断 MatMul 输出 shape。
        这里只需要覆盖 Gemm 改写后的 2D / batched 线性层主路径，目标是给后续
        bias Add 提供一个可靠的广播参考 shape。
        """
        if lhs_shape is None or rhs_shape is None:
            return None
        if len(lhs_shape) == 0 or len(rhs_shape) == 0:
            return None
        if len(lhs_shape) == 1 and len(rhs_shape) == 1:
            return []
        if len(lhs_shape) == 1:
            prefix = []
            tail = rhs_shape[-1] if len(rhs_shape) >= 1 else None
            return prefix + [tail] if tail is not None else None
        if len(rhs_shape) == 1:
            prefix = list(lhs_shape[:-1])
            return prefix if prefix else None
        prefix = list(lhs_shape[:-1])
        tail = rhs_shape[-1]
        if tail is None:
            return None
        return prefix + [tail]

    gemm_nodes = []
    gemm_patch_count = 0
    for node in graph.node:
        if node.op_type != "Gemm":
            gemm_nodes.append(node)
            continue

        base = node.name.replace("/", "_").lstrip("_") or f"gemm_patch_{gemm_patch_count}"
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        trans_a = int(attrs.get("transA", 0))
        trans_b = int(attrs.get("transB", 0))

        lhs_name = node.input[0]
        rhs_name = node.input[1]
        output_name = node.output[0]
        output_dtype = dtype_map.get(output_name, dtype_map.get(lhs_name, FLOAT))
        output_shape = shape_map.get(output_name)
        lhs_shape = shape_map.get(lhs_name)
        rhs_shape = shape_map.get(rhs_name)

        if trans_a:
            lhs_transposed = f"__cann_patch_{base}_transA"
            gemm_nodes.append(
                helper.make_node(
                    "Transpose",
                    inputs=[lhs_name],
                    outputs=[lhs_transposed],
                    name=f"__cann_patch_{base}_transpose_a",
                    perm=[1, 0],
                )
            )
            dtype_map[lhs_transposed] = dtype_map.get(lhs_name, output_dtype)
            shape_map[lhs_transposed] = _transpose_shape(shape_map.get(lhs_name))
            _register_value_info(lhs_transposed, dtype_map[lhs_transposed])
            lhs_name = lhs_transposed

        if trans_b:
            rhs_transposed = f"__cann_patch_{base}_transB"
            gemm_nodes.append(
                helper.make_node(
                    "Transpose",
                    inputs=[rhs_name],
                    outputs=[rhs_transposed],
                    name=f"__cann_patch_{base}_transpose_b",
                    perm=[1, 0],
                )
            )
            dtype_map[rhs_transposed] = dtype_map.get(rhs_name, output_dtype)
            shape_map[rhs_transposed] = _transpose_shape(shape_map.get(rhs_name))
            _register_value_info(rhs_transposed, dtype_map[rhs_transposed])
            rhs_name = rhs_transposed

        has_bias = len(node.input) >= 3 and bool(node.input[2])
        need_alpha_scale = not np.isclose(alpha, 1.0)
        need_bias_add = has_bias and not np.isclose(beta, 0.0)
        matmul_out = output_name if not need_alpha_scale and not need_bias_add else f"__cann_patch_{base}_matmul"

        gemm_nodes.append(
            helper.make_node(
                "MatMul",
                inputs=[lhs_name, rhs_name],
                outputs=[matmul_out],
                name=f"__cann_patch_{base}_matmul",
            )
        )
        dtype_map[matmul_out] = output_dtype
        matmul_shape = list(output_shape) if output_shape is not None else _infer_matmul_output_shape(lhs_shape, rhs_shape)
        if matmul_shape is not None:
            shape_map[matmul_out] = list(matmul_shape)
        _register_value_info(matmul_out, output_dtype)

        current_out = matmul_out
        if need_alpha_scale:
            alpha_node, alpha_const = _make_scalar_const_node(base, "alpha", alpha)
            scaled_out = output_name if not need_bias_add else f"__cann_patch_{base}_matmul_scaled"
            gemm_nodes.append(alpha_node)
            gemm_nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[current_out, alpha_const],
                    outputs=[scaled_out],
                    name=f"__cann_patch_{base}_alpha_mul",
                )
            )
            dtype_map[scaled_out] = output_dtype
            if output_shape is not None:
                shape_map[scaled_out] = list(output_shape)
            _register_value_info(scaled_out, output_dtype)
            current_out = scaled_out

        if need_bias_add:
            bias_name = node.input[2]
            add_rhs = bias_name
            if not np.isclose(beta, 1.0):
                beta_node, beta_const = _make_scalar_const_node(base, "beta", beta)
                scaled_bias = f"__cann_patch_{base}_bias_scaled"
                gemm_nodes.append(beta_node)
                gemm_nodes.append(
                    helper.make_node(
                        "Mul",
                        inputs=[bias_name, beta_const],
                        outputs=[scaled_bias],
                        name=f"__cann_patch_{base}_beta_mul",
                    )
                )
                dtype_map[scaled_bias] = dtype_map.get(bias_name, output_dtype)
                if bias_name in shape_map:
                    shape_map[scaled_bias] = list(shape_map[bias_name])
                _register_value_info(scaled_bias, dtype_map[scaled_bias])
                add_rhs = scaled_bias

            add_rhs_shape = shape_map.get(add_rhs)
            current_shape = shape_map.get(current_out) or output_shape
            add_ref_shape_name = f"__cann_patch_{base}_add_broadcast_shape"
            add_ref_shape_inserted = False
            add_rhs, add_ref_shape_inserted = _expand_to_ref_if_needed(
                gemm_nodes,
                add_rhs,
                current_out,
                base,
                0,
                prefix="add",
                ref_shape_name=add_ref_shape_name,
                ref_shape_node_inserted=add_ref_shape_inserted,
                ref_shape=current_shape,
                src_shape=add_rhs_shape,
                elem_type=dtype_map.get(add_rhs, output_dtype),
                force=True,
            )

            gemm_nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[current_out, add_rhs],
                    outputs=[output_name],
                    name=f"__cann_patch_{base}_add",
                )
            )

        gemm_patch_count += 1

    if gemm_patch_count > 0:
        del graph.node[:]
        graph.node.extend(gemm_nodes)
        model = _infer_shapes_if_possible(model)
        graph = model.graph
        dtype_map, shape_map, known_value_info_names = _collect_tensor_metadata(graph)

    new_nodes = []
    mul_patch_count = 0
    expand_count = 0

    for node in graph.node:
        if node.op_type != "Mul":
            new_nodes.append(node)
            continue

        # 判断是否为 int64 Mul（任意输入是 int64 则认为是 int64 Mul）
        is_int64 = any(
            dtype_map.get(inp, -1) == INT64
            for inp in node.input
        )
        if not is_int64:
            new_nodes.append(node)
            continue

        # ── 步骤 1：插入 Cast(int64→float32) ────────────────────────────────
        base = node.name.replace("/", "_").lstrip("_") or f"mul_patch_{mul_patch_count}"
        cast_fp_inputs = []
        for i, inp in enumerate(node.input):
            cast_out = f"__cann_patch_{base}_castfp_{i}"
            cast_node = helper.make_node(
                "Cast",
                inputs=[inp],
                outputs=[cast_out],
                name=f"__cann_patch_{base}_cast_in_{i}",
                to=FLOAT,
            )
            new_nodes.append(cast_node)
            dtype_map[cast_out] = FLOAT
            # Cast 保持 shape 不变，将原输入的 shape 信息传递给 cast 输出
            if inp in shape_map:
                shape_map[cast_out] = shape_map[inp]
            _register_value_info(cast_out, FLOAT)
            cast_fp_inputs.append(cast_out)

        # ── 步骤 2：检测是否需要显式 Expand 以消除广播维度差 ────────────────
        # CANN 单算子模式下，以下两种情况均会触发 BroadcastToD，
        # 而 BroadcastToD 在 ACL_COMPILE_SYS 预编译库中缺少对应 kernel：
        #   a) rank 不同（如 [N] × []）
        #   b) rank 相同但某维为 1 vs M>1（如 [36] × [1]）
        # 解决方案：对"较小"的输入插入 Shape(ref) + Expand(small, shape_of_ref)，
        # 使 Mul 退化为等形 element-wise，从而避免 BroadcastToD 被调用。
        orig_shapes = [shape_map.get(inp) for inp in cast_fp_inputs]

        # 选最大 shape 作为广播参考
        ref_idx = _pick_ref_idx(orig_shapes)
        ref_inp      = cast_fp_inputs[ref_idx]
        ref_shape    = orig_shapes[ref_idx]

        # 为参考输入生成 Shape 节点（仅在实际需要时才插入，惰性创建）
        ref_shape_node_inserted = False
        ref_shape_name = f"__cann_patch_{base}_broadcast_shape"

        final_fp_inputs = list(cast_fp_inputs)
        for i, (inp, src_shape) in enumerate(zip(cast_fp_inputs, orig_shapes)):
            if i == ref_idx:
                continue  # 参考输入本身不需扩展
            # 判断是否需要 Expand
            if src_shape is None or ref_shape is None:
                continue  # shape 未知，跳过（保守策略；若运行时仍失败可手动调整）
            if not _needs_expand(src_shape, ref_shape):
                continue  # 形状已兼容，无需 Expand

            # 惰性插入 Shape 节点（只插一次）
            if not ref_shape_node_inserted:
                shape_node = helper.make_node(
                    "Shape",
                    inputs=[ref_inp],
                    outputs=[ref_shape_name],
                    name=f"__cann_patch_{base}_shape_node",
                )
                new_nodes.append(shape_node)
                dtype_map[ref_shape_name] = INT64
                _register_value_info(ref_shape_name, INT64)
                ref_shape_node_inserted = True

            exp_name = f"__cann_patch_{base}_expanded_{i}"
            exp_node = helper.make_node(
                "Expand",
                inputs=[inp, ref_shape_name],
                outputs=[exp_name],
                name=f"__cann_patch_{base}_expand_node_{i}",
            )
            new_nodes.append(exp_node)
            dtype_map[exp_name] = FLOAT
            shape_map[exp_name] = ref_shape  # 扩展后 shape = 参考 shape
            _register_value_info(exp_name, FLOAT)
            final_fp_inputs[i] = exp_name
            expand_count += 1

        # ── 步骤 3：插入 fp32 Mul（此时两输入 shape 相同，无 BroadcastToD）──
        mul_fp_out = f"__cann_patch_{base}_mulfp"
        dtype_map[mul_fp_out] = FLOAT
        _register_value_info(mul_fp_out, FLOAT)

        mul_fp_node = helper.make_node(
            "Mul",
            inputs=final_fp_inputs,
            outputs=[mul_fp_out],
            name=f"__cann_patch_{base}_mul_fp",
        )
        new_nodes.append(mul_fp_node)

        # ── 步骤 4：Cast float32 → int64 ────────────────────────────────────
        for orig_out in node.output:
            cast_back_node = helper.make_node(
                "Cast",
                inputs=[mul_fp_out],
                outputs=[orig_out],
                name=f"__cann_patch_{base}_cast_out",
                to=INT64,
            )
            new_nodes.append(cast_back_node)
        mul_patch_count += 1

    if gemm_patch_count == 0 and mul_patch_count == 0:
        return onnx_path  # 无需修改

    # 重建图节点列表
    del graph.node[:]
    graph.node.extend(new_nodes)

    # 写入临时文件
    tmp_path = onnx_path + ".cann_patched.onnx"
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"[PATCH] WARNING: onnx.checker 失败（{e}），继续保存")
    onnx.save(model, tmp_path)
    if gemm_patch_count > 0:
        print(f"[PATCH] 已将 {gemm_patch_count} 个 Gemm 节点改写为 MatMul(+Mul)+Add")
    if mul_patch_count > 0:
        print(
            f"[PATCH] 已将 {mul_patch_count} 个 int64 Mul 节点包装为 fp32 Mul"
            f"（其中 {expand_count} 处额外插入 Expand 以消除广播维度差）"
        )
    print(f"[PATCH] 补丁模型保存到: {tmp_path}")
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# Loop → Gather 替换（EmbeddingBag 向 NPU 迁移）
# ─────────────────────────────────────────────────────────────────────────────

def _infer_bag_size_from_loop(ln: "onnx.NodeProto",  # type: ignore[name-defined]
                               graph: "onnx.GraphProto",  # type: ignore[name-defined]
                               init_map: Dict[str, "onnx.TensorProto"]) -> int:  # type: ignore[name-defined]
    """
    从 ONNX Loop 节点推断 bag_size（即 num-indices-per-lookup）。

    尝试以下三种策略，按优先级依次尝试：
      1. Loop.input[0]（trip_count）是一个 initializer 或 Constant → 直接读取整数值
      2. Loop 消费辅助 Slice 节点：其 ends 减 starts 即为 bag_size
      3. indices_N 的 shape[0] / batch_size_dim 无法确定时回退到 1（保守值）
    """
    # 策略 1：trip_count 在 initializers 中
    if ln.input and ln.input[0]:
        tc_name = ln.input[0]
        if tc_name in init_map:
            import numpy as np
            arr = np.array(init_map[tc_name].int64_data or
                           init_map[tc_name].int32_data or
                           [1], dtype=np.int64)
            if arr.size == 1:
                val = int(arr.flat[0])
                if val > 0:
                    return val
    # 策略 2：遍历消费 Loop 的子图，找 Constant trip_count
    for node in graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == (ln.input[0] if ln.input else ""):
            for attr in node.attribute:
                if attr.name == "value" and attr.t.dims == []:
                    # scalar constant
                    import numpy as np
                    from onnx import numpy_helper
                    arr = numpy_helper.to_array(attr.t)
                    val = int(arr.flat[0])
                    if val > 0:
                        return val
    # 回退：无法推断，返回 -1（由调用方决定行为）
    return -1


def _replace_loop_with_gather(onnx_path: str,
                              override_bag_size: int = 0) -> str:
    """
    将模型中所有 emb_l.N 的 Loop 节点替换为等价的向量化算子序列，
    消除 CPU 顺序迭代开销，使 EmbeddingBag 查表可并行执行。

    背景分析：
      - DLRM 每个 EmbeddingBag 表由 ONNX Loop + Gather + ReduceSum 实现
      - Loop 算子是顺序控制流原语，ORT 逐步执行子图，无法并行
      - Loop 带动的辅助算子（Slice/Unsqueeze/ReduceSum）也构成大量开销

    变换规则（依据 bag_size 自动选择）：

      ── bag_size = 1 ─────────────────────────────────────────────────
        Loop(/emb_l.N/...) → Gather(weight, indices_N, axis=0)
        indices_N shape: [B]  →  output shape: [B, emb_dim]
        原理：bag_size=1 时 ReduceSum 对单行求和等同于恒等，Loop ≡ Gather。

      ── bag_size > 1 ─────────────────────────────────────────────────
        Loop(/emb_l.N/...) → Gather → Reshape → ReduceSum
        变换步骤：
          1. Gather(weight, indices_N, axis=0)
               indices_N shape: [B * bag_size]
               output: [B * bag_size, emb_dim]
          2. Reshape([B * bag_size, emb_dim] → [B, bag_size, emb_dim])
               shape tensor: Constant([B, bag_size, emb_dim])
               注：B 为动态维度，使用 -1 令 Reshape 自动推断
          3. ReduceSum(axis=1, keepdims=0)
               output: [B, emb_dim]  ← 与原 Loop 输出语义完全一致

        关键正确性条件：--num-indices-per-lookup-fixed=True 且所有样本
        的 bag_size 相同（即 indices_N.shape[0] == B * bag_size 是静态的）。
        若 bag_size 动态变化，本替换不适用（保守地跳过该 Loop）。

    override_bag_size : 直接指定 bag_size（对应 --num-indices-per-lookup），
        优先级最高，跳过所有自动推断逻辑。为 0 时自动推断。

    后续图中所有消费 /emb_l.N/Loop_output_0 的节点（通常是 Concat）无需修改。
    原 Loop 的上游辅助节点（offset Slice/Concat/Unsqueeze 等）变成死代码，
    由内置 DCE 逻辑清除。

    返回：转换后模型的临时文件路径（若无可替换 Loop 则返回原路径）。
    """
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
    except ImportError:
        print("[LOOP2GATHER] onnx 未安装，跳过 Loop→Gather 替换")
        return onnx_path

    model = onnx.load(onnx_path)
    graph = model.graph

    # ── 扫描所有 Loop 节点 ──────────────────────────────────────────────────
    loop_nodes = [n for n in graph.node if n.op_type == "Loop"]
    if not loop_nodes:
        print("[LOOP2GATHER] 未发现 Loop 节点，跳过")
        return onnx_path

    # ── 建立辅助查找表 ───────────────────────────────────────────────────────
    # initializer 名 → proto
    init_map: Dict[str, Any] = {init.name: init for init in graph.initializer}

    # 图输入类型 & shape
    graph_input_shape: Dict[str, list] = {}
    for vi in graph.input:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims: list = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append(None)
            graph_input_shape[vi.name] = dims

    # ── 校验并收集可替换的 Loop 节点 ────────────────────────────────────────
    valid_loops: list = []
    for ln in loop_nodes:
        try:
            tbl = int(ln.name.split("emb_l.")[1].split("/")[0])
        except (IndexError, ValueError):
            print(f"[LOOP2GATHER] 跳过非 emb_l 格式 Loop: {ln.name}")
            continue

        weight_name  = f"emb_l.{tbl}.weight"
        indices_name = f"indices_{tbl}"

        has_weight  = weight_name  in init_map
        has_indices = any(inp.name == indices_name for inp in graph.input)
        if not has_weight or not has_indices:
            print(f"[LOOP2GATHER] emb_l.{tbl}: 找不到 {weight_name!r} 或 {indices_name!r}，跳过")
            continue

        # ── 推断 bag_size ──────────────────────────────────────────────────
        # 优先级：外部显式传入 > 从 trip_count 静态推断 > 保守回退到 1
        if override_bag_size > 0:
            bag_size = override_bag_size
            print(f"[LOOP2GATHER] emb_l.{tbl}: 使用外部指定 bag_size={bag_size}")
        else:
            bag_size = _infer_bag_size_from_loop(ln, graph, init_map)
            if bag_size < 1:
                # DLRM 导出时 trip_count 由 offsets Slice 动态计算，无法静态推断
                # 必须通过 --num-indices-per-lookup 显式传入，此处保守回退到 1
                bag_size = 1
                print(
                    f"[LOOP2GATHER] emb_l.{tbl}: 无法从 ONNX 图推断 bag_size，"
                    f"保守回退到 1。\n"
                    f"  若 num-indices-per-lookup > 1，请添加参数 "
                    f"--num-indices-per-lookup=<N> 以启用正确的 Gather+Reshape+ReduceSum 替换。"
                )

        # 获取 embedding 维度（从 weight initializer shape[1]）
        weight_proto = init_map[weight_name]
        emb_dim = int(weight_proto.dims[1]) if len(weight_proto.dims) >= 2 else -1

        valid_loops.append((ln, tbl, weight_name, indices_name, bag_size, emb_dim))

    if not valid_loops:
        print("[LOOP2GATHER] 没有可替换的 Loop 节点")
        return onnx_path

    # ── 构建 output→consumers 映射（DCE 用）────────────────────────────────
    out2consumers: Dict[str, list] = {}
    for node in graph.node:
        for inp in node.input:
            out2consumers.setdefault(inp, []).append(node.name)

    graph_output_names: set = {o.name for o in graph.output}

    loop_names_to_remove: set = set()
    replacement_nodes: list = []   # 插入到图前段的新节点

    INT64 = TensorProto.INT64

    for (ln, tbl, weight_name, indices_name, bag_size, emb_dim) in valid_loops:
        loop_out = ln.output[0]   # 消费方期待的输出张量名
        loop_names_to_remove.add(ln.name)
        base = f"emb_l{tbl}"

        if bag_size == 1:
            # ── 路径 A：bag_size=1，单 Gather 即可 ───────────────────────
            #   Gather(weight, indices_N, axis=0) → loop_out  [B, emb_dim]
            gather_node = helper.make_node(
                "Gather",
                inputs=[weight_name, indices_name],
                outputs=[loop_out],
                name=f"/{base}/Gather_direct",
                axis=0,
            )
            replacement_nodes.append(gather_node)
            print(
                f"[LOOP2GATHER] emb_l.{tbl} (bag_size=1): "
                f"Loop → Gather({weight_name}, {indices_name}) → {loop_out}"
            )
        else:
            # ── 路径 B：bag_size>1，Gather → Reshape → ReduceSum ─────────
            #
            #   indices_N:  [B * bag_size]     （所有样本的索引拼接）
            #
            #   Step 1) Gather(weight, indices_N, axis=0)
            #             → gather_out: [B * bag_size, emb_dim]
            #
            #   Step 2) Reshape(gather_out, [-1, bag_size, emb_dim])
            #             → reshape_out: [B, bag_size, emb_dim]
            #             （-1 表示动态 batch_size，由 Reshape 自动推断）
            #
            #   Step 3) ReduceSum(reshape_out, axes=[1], keepdims=0)
            #             → loop_out: [B, emb_dim]   ← 与原 Loop 输出等价

            gather_out  = f"__{base}_gather_out"
            reshape_out = f"__{base}_reshape_out"
            shape_const = f"__{base}_reshape_shape"

            # -- Step 1: Gather --
            gather_node = helper.make_node(
                "Gather",
                inputs=[weight_name, indices_name],
                outputs=[gather_out],
                name=f"/{base}/Gather",
                axis=0,
            )

            # -- Step 2: shape Constant + Reshape --
            # shape = [-1, bag_size, emb_dim]
            # -1 处理动态 batch_size（Reshape 会正确推断）
            shape_vals = np.array([-1, bag_size, emb_dim], dtype=np.int64)
            shape_const_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[shape_const],
                name=f"/{base}/ReshapeShape",
                value=numpy_helper.from_array(shape_vals, name=shape_const),
            )
            reshape_node = helper.make_node(
                "Reshape",
                inputs=[gather_out, shape_const],
                outputs=[reshape_out],
                name=f"/{base}/Reshape",
            )

            # -- Step 3: ReduceSum(axis=1, keepdims=0) --
            # ONNX opset >= 13：axes 作为第二个输入张量
            # ONNX opset < 13 ：axes 作为属性
            opset_version = 11  # 保守默认
            for opset in model.opset_import:
                if opset.domain == "" or opset.domain == "ai.onnx":
                    opset_version = opset.version
                    break

            if opset_version >= 13:
                axes_name = f"__{base}_reduce_axes"
                axes_const_node = helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    name=f"/{base}/ReduceAxes",
                    value=numpy_helper.from_array(
                        np.array([1], dtype=np.int64), name=axes_name
                    ),
                )
                reduce_node = helper.make_node(
                    "ReduceSum",
                    inputs=[reshape_out, axes_name],
                    outputs=[loop_out],
                    name=f"/{base}/ReduceSum",
                    keepdims=0,
                )
                replacement_nodes.extend([
                    gather_node, shape_const_node, reshape_node,
                    axes_const_node, reduce_node,
                ])
            else:
                reduce_node = helper.make_node(
                    "ReduceSum",
                    inputs=[reshape_out],
                    outputs=[loop_out],
                    name=f"/{base}/ReduceSum",
                    axes=[1],
                    keepdims=0,
                )
                replacement_nodes.extend([
                    gather_node, shape_const_node, reshape_node, reduce_node,
                ])

            print(
                f"[LOOP2GATHER] emb_l.{tbl} (bag_size={bag_size}, emb_dim={emb_dim}): "
                f"Loop → Gather+Reshape+ReduceSum → {loop_out}"
            )

    # ── 死代码删除（保守迭代策略）──────────────────────────────────────────
    removed_set: set = set(loop_names_to_remove)
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if node.name in removed_set:
                continue
            if any(out in graph_output_names for out in node.output):
                continue
            if not node.output:
                continue
            all_dead = all(
                all(c in removed_set for c in out2consumers.get(out, []))
                for out in node.output
            )
            if all_dead:
                removed_set.add(node.name)
                changed = True

    # ── 重建节点列表 ───────────────────────────────────────────────────────
    # replacement_nodes 的所有输入（weight=initializer, indices=graph_input）
    # 在图最开始即可用，安全地插入到列表最前面。
    new_nodes = replacement_nodes + [n for n in graph.node if n.name not in removed_set]

    del graph.node[:]
    graph.node.extend(new_nodes)

    # ── 清理孤立 value_info ────────────────────────────────────────────────
    all_tensor_names: set = (
        {i.name for i in graph.input}
        | {o.name for o in graph.output}
        | {init.name for init in graph.initializer}
        | {o for nd in graph.node for o in nd.output}
    )
    stale_vis = [vi for vi in graph.value_info if vi.name not in all_tensor_names]
    for vi in stale_vis:
        graph.value_info.remove(vi)

    # ── 更新 indices_* 图输入 shape（bag_size>1 时实际长度为 B*bag_size）────────
    # ORT 用 graph input 的 shape 标注预分配 MemcpyFromHost 缓冲区。
    # 若标注仍为原来的 batch_size（如 dim_value=320），则分配 {320,64}，
    # 而 Gather 实际输出 {32000,64}，触发 "Shape mismatch attempting to re-use buffer"。
    # 将 dim[0] 改为符号维度 "indices_total"，ORT 按实际输入动态分配正确大小。
    indices_name_to_bagsize: Dict[str, int] = {
        iname: bs for (_, _, _, iname, bs, _) in valid_loops if bs > 1
    }
    for vi in graph.input:
        bs_override = indices_name_to_bagsize.get(vi.name, 0)
        if bs_override <= 1:
            continue
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            shape = vi.type.tensor_type.shape
            if len(shape.dim) >= 1:
                dim = shape.dim[0]
                dim.ClearField("dim_value")
                dim.ClearField("dim_param")
                dim.dim_param = "indices_total"  # 符号维度，ORT 按实际输入大小动态分配

    # ── 保存 ──────────────────────────────────────────────────────────────
    out_path = onnx_path + ".loop_to_gather.onnx"
    onnx.save(model, out_path)
    dce_count = len(removed_set) - len(loop_names_to_remove)
    n_b1 = sum(1 for (_, _, _, _, bs, _) in valid_loops if bs == 1)
    n_bn = len(valid_loops) - n_b1
    print(
        f"[LOOP2GATHER] 替换完成：{len(valid_loops)} 个 Loop"
        f"（bag_size=1: {n_b1} 个 → Gather；"
        f"bag_size>1: {n_bn} 个 → Gather+Reshape+ReduceSum）"
        f"，额外 DCE 删除 {dce_count} 个死节点"
    )
    print(f"[LOOP2GATHER] 新模型保存到: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 指定算子强制卸载到 CPU
# ─────────────────────────────────────────────────────────────────────────────

# 将 CANN 支持的算子替换为语义等价但 CANN 不支持（走 CPU）的算子。
# 通过探测确认以下替换在此 CANN ORT build 下均落到 CPU：
#   Relu  → LeakyRelu(alpha=0)   （单节点，无需额外 initializer，数学完全等价）
#   Relu  → Clip(x, min=0)       （备选）
#
# 若需扩展更多算子，在 _CPU_REPLACEMENTS 中增加对应转换函数即可。
#
# 背景：CANN EP GetCapability 仅对 CANN 核注册表中存在的算子申领卸载；
#       换成等价但注册表中没有的算子后，该节点会自动落回 CPUExecutionProvider。

def _make_leaky_relu_replacement(
    node: "onnx.NodeProto",  # type: ignore[name-defined]
) -> List["onnx.NodeProto"]:  # type: ignore[name-defined]
    """Relu(x) → LeakyRelu(x, alpha=0.0)，完全等价且保证在 CPU 上执行。"""
    from onnx import helper as _h
    new_node = _h.make_node(
        "LeakyRelu",
        inputs=node.input,
        outputs=node.output,
        name=(node.name + "_cpu") if node.name else "",
        alpha=0.0,
    )
    # 如有 doc_string 也搬过去
    if node.HasField("doc_string"):
        new_node.doc_string = node.doc_string
    return [new_node]


# 算子替换策略表：op_type -> 转换函数(旧节点) -> [新节点列表]
_CPU_REPLACEMENTS: Dict[str, Any] = {
    "Relu": _make_leaky_relu_replacement,
}


def _force_ops_to_cpu(onnx_path: str, op_types: List[str]) -> str:
    """
    将指定算子类型替换为语义等价、但 CANN 不支持（因此落回 CPU）的算子。

    参数
    ----
    onnx_path : 输入 ONNX 文件路径
    op_types  : 需要强制卸载到 CPU 的算子类型列表，例如 ["Relu", "Sigmoid"]

    返回
    ----
    新 ONNX 文件路径（后缀 .cpu_ops.onnx）

    当前支持的算子
    --------------
    - Relu → LeakyRelu(alpha=0)：CANN 无 LeakyRelu 核，自动落 CPU
    """
    import onnx  # 延迟导入，与文件其他 _patch / _replace 函数保持一致

    if not op_types:
        return onnx_path

    # 仅保留已有替换策略的算子，对未知算子给出警告
    targets: List[str] = []
    for t in op_types:
        t = t.strip()
        if not t:
            continue
        if t == "Reshape":
            print(
                "[FORCE_CPU] Reshape 不能仅靠单会话 provider option 稳定下放；"
                "请在 branch-parallel 子模型级别切换到 CPUExecutionProvider。"
            )
            continue
        if t not in _CPU_REPLACEMENTS:
            print(f"[FORCE_CPU] WARNING: 算子 {t!r} 暂无 CPU 替换策略，跳过。"
                  f"  当前支持: {list(_CPU_REPLACEMENTS.keys())}")
        else:
            targets.append(t)

    if not targets:
        return onnx_path

    model = onnx.load(onnx_path)
    graph = model.graph
    target_set = set(targets)

    replaced_counts: Dict[str, int] = {t: 0 for t in targets}
    new_nodes: List[Any] = []

    for node in graph.node:
        if node.op_type in target_set:
            replacer = _CPU_REPLACEMENTS[node.op_type]
            new_nodes.extend(replacer(node))
            replaced_counts[node.op_type] += 1
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    # 验证
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"[FORCE_CPU] WARNING: onnx.checker 失败（{e}），继续保存")

    out_path = onnx_path + ".cpu_ops.onnx"
    onnx.save(model, out_path)

    total = sum(replaced_counts.values())
    detail = ", ".join(f"{t}: {c} 个" for t, c in replaced_counts.items() if c)
    print(f"[FORCE_CPU] 完成：共替换 {total} 个节点（{detail}）")
    print(f"[FORCE_CPU] 新模型保存到: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ORT 加载
# ─────────────────────────────────────────────────────────────────────────────

def _import_onnxruntime():
    """Import onnxruntime，返回 (ort_module, available_providers)。"""
    try:
        import onnxruntime as ort
        return ort, ort.get_available_providers()
    except ImportError:
        print("[ERROR] onnxruntime 未安装。请运行: bash setup_ort_cann.sh")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Session 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_session(
    onnx_path: str,
    use_cann: bool,
    device_id: int,
    enable_profiling: bool,
    profile_dir: str,
    intra_threads: int,
    inter_threads: int,
    disable_graph_optimizations: bool = False,
    replace_loop: bool = True,
    force_cpu_ops: Optional[List[str]] = None,
    bag_size: int = 0,
) -> Tuple[object, List[str], str]:
    """
    构建 ORT InferenceSession。

    Provider 优先级：
      1. CANNExecutionProvider  → NPU（若 use_cann=True 且库可用）
      2. CPUExecutionProvider   → CPU fallback

    ORT 会自动将每个算子分配给第一个支持它的 provider：
      - 若 CANN provider 支持某算子 → 在 NPU 执行
      - 否则                         → 回退到 CPU 执行

    force_cpu_ops : 强制卸载到 CPU 的算子列表，例如 ["Relu"]。
                    Relu 通过语义等价替换实现；
                    Reshape 需要在 branch-parallel 子模型级别切换到 CPUExecutionProvider。
                    仅在 use_cann=True 时有效。
    """
    _setup_cann_env()
    ort, available_providers = _import_onnxruntime()

    # ── 图预处理 ──────────────────────────────────────────────────────────────
    # Loop→Gather 替换是否生效只由 replace_loop 控制，与是否启用 CANN 解耦。
    # 这样即便仅使用 CPUExecutionProvider，也可以显式生成替换后的新 ONNX 图。
    effective_onnx_path = onnx_path

    # ── Session 选项 ──────────────────────────────────────────────────────────
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    if disable_graph_optimizations:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # inter_op 并行执行模式：仅当 inter_op_num_threads > 1 时有效
    # ORT_SEQUENTIAL（默认）：算子按拓扑序逐一执行，inter_op 线程池闲置
    # ORT_PARALLEL      ：无数据依赖的算子（如各 emb_l* Gather）可同时调度
    if inter_threads > 1:
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    # 日志级别：ERROR（3），抑制图优化阶段的 CanUpdateImplicitInputName 等警告
    # 可设为 0(VERBOSE) 1(INFO) 2(WARNING) 3(ERROR) 4(FATAL)
    opts.log_severity_level = 3

    if enable_profiling:
        opts.enable_profiling = True
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        opts.profile_file_prefix = str(Path(profile_dir) / "ort_cann_profile")

    # ── Provider 列表 ─────────────────────────────────────────────────────────
    cann_available = "CANNExecutionProvider" in available_providers

    if use_cann and not cann_available:
        print(
            "[WARN] 请求了 CANNExecutionProvider，但当前 onnxruntime 不包含该 provider。\n"
            "       请运行 bash setup_ort_cann.sh 安装支持 CANN 的 onnxruntime 版本。\n"
            "       本次将仅使用 CPUExecutionProvider。"
        )
        use_cann = False

    if use_cann and cann_available:
        # CANN provider 选项说明：
        #   device_id            : NPU 卡号
        #   precision_mode       : force_fp32 | force_fp16 | allow_mix_precision
        #   op_select_impl_mode  : high_performance | high_precision
        cann_provider_options = {
            "device_id":            str(device_id),
            # force_fp32: 保持与 PyTorch 模型精度一致
            "precision_mode":       "force_fp32",
            "op_select_impl_mode":  "high_performance",
            # 内存池策略：kNextPowerOfTwo（预分配）或 kSameAsRequested
            "arena_extend_strategy": "kNextPowerOfTwo",
            # 禁用 CANN Graph 整图编译模式（默认 1）。
            # 设为 0：CANN 仅处理单算子，不尝试整图编译。
            # 注：若 replace_loop=True，Loop 节点已被替换为 Gather，
            #     此选项主要保留作为保险措施。
            "enable_cann_graph":    "0",
        }
        providers = [
            ("CANNExecutionProvider", cann_provider_options),
            "CPUExecutionProvider",
        ]
        active_providers_str = "CANNExecutionProvider + CPUExecutionProvider (混合执行)"
        # Step 1：修复 int64 Mul → BroadcastToD 在 CANN 上的兼容性问题
        effective_onnx_path = _patch_model_for_cann(effective_onnx_path)
    else:
        providers = ["CPUExecutionProvider"]
        active_providers_str = "CPUExecutionProvider (仅 CPU)"

    # 将 emb_l.N Loop → Gather(emb_l.N.weight, indices_N)
    # 使 EmbeddingBag 查表使用改写后的新图；是否执行只由 replace_loop 控制。
    if replace_loop:
        effective_onnx_path = _replace_loop_with_gather(
            effective_onnx_path, override_bag_size=bag_size
        )

    # 将指定算子替换为 CANN 不支持的等价算子，强制其落回 CPU。
    # 保持在 Loop→Gather 之后执行，与原有图改写顺序一致。
    if use_cann and cann_available and force_cpu_ops:
        effective_onnx_path = _force_ops_to_cpu(effective_onnx_path, force_cpu_ops)

    print(f"\n[ORT] 加载模型: {effective_onnx_path}")
    print(f"[ORT] 使用 provider: {active_providers_str}")

    t0 = time.perf_counter()
    session = ort.InferenceSession(effective_onnx_path, sess_options=opts, providers=providers)
    load_time = (time.perf_counter() - t0) * 1000

    # 打印实际使用的 providers（ORT 内部分配结果）
    actual = session.get_providers()
    print(f"[ORT] 模型加载耗时: {load_time:.1f} ms")
    print(f"[ORT] 实际 providers: {actual}")

    return session, actual, effective_onnx_path


def run_warmup(
    session: object,
    batch_size: int,
    warmup_batches: int,
    onnx_path: Optional[str] = None,
    bag_size: int = 1,
) -> None:
    """执行 warmup，不返回延迟统计。"""
    if warmup_batches <= 0:
        return

    output_names = [o.name for o in session.get_outputs()]
    print(f"\n[INFER] Warmup: {warmup_batches} 次 ...")
    for i in range(warmup_batches):
        feed = generate_inputs(session, batch_size, seed=i,
                               onnx_path=onnx_path, bag_size=bag_size)
        session.run(output_names, feed)
    print("[INFER] Warmup 完成。")


# ─────────────────────────────────────────────────────────────────────────────
# 算子维度 CSV 导出
# ─────────────────────────────────────────────────────────────────────────────

_ONNX_DTYPE_NAME: Dict[int, str] = {
    1: "float32",  2: "uint8",   3: "int8",    4: "uint16",
    5: "int16",    6: "int32",   7: "int64",   8: "string",
    9: "bool",    10: "float16", 11: "float64", 12: "uint32",
   13: "uint64",  14: "complex64", 15: "complex128", 16: "bfloat16",
}


def _collect_runtime_shapes(onnx_path: str, batch_size: int,
                            bag_size: int = 1) -> Dict[str, tuple]:
    """
    将 ONNX 图中所有中间张量临时追加为图输出，用 CPU ORT 跑一次单 batch 推理，
    返回 {tensor_name: (d0, d1, ...)} 的真实运行时 shape 字典。

    bag_size : num-indices-per-lookup，用于正确推算 indices_* 输入的实际长度
               （indices_* 的真实大小为 batch_size * bag_size，而非 batch_size）。

    此函数始终使用 CPUExecutionProvider（不依赖 CANN），可在 CANN 模式下独立调用。
    """
    try:
        import onnx
        from onnx import helper as onnx_helper, TensorProto, ValueInfoProto
        import onnxruntime as _ort
    except ImportError:
        return {}

    model = onnx.load(onnx_path)
    graph = model.graph

    # 收集已有图输出名，避免重复添加
    existing_out = {o.name for o in graph.output}
    initializer_names = {i.name for i in graph.initializer}

    # 把所有节点输出（未在图输出中的）追加为额外输出。
    # 用无类型的裸 ValueInfoProto（不声明 dtype / shape），
    # 避免 ORT 做类型一致性检查（float/int64/bool 混合时不报错）。
    extra_added: List[str] = []
    for node in graph.node:
        for t in node.output:
            if t and t not in existing_out and t not in initializer_names:
                vi = ValueInfoProto()
                vi.name = t
                graph.output.append(vi)
                existing_out.add(t)
                extra_added.append(t)

    import tempfile, os as _os
    tmp_path = onnx_path + ".shape_probe.onnx"
    try:
        onnx.save(model, tmp_path)

        opts = _ort.SessionOptions()
        opts.log_severity_level = 4   # 静默
        probe_sess = _ort.InferenceSession(
            tmp_path, sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        # ── 生成探针输入（安全范围，保证不越界）──────────────────────────────
        # 目标是让模型顺跑一遍以获取 shape，不需要语义正确性。
        # 规则：
        #   * float → 零张量
        #   * offset（EmbeddingBag 偏移）→ 全零单调序列（合法）
        #   * indices_*（稀疏索引）→ 全零，但长度为 batch_size * bag_size
        #     因为 Loop→Gather+Reshape+ReduceSum 变换后 indices_N 的长度
        #     就是原始的 batch_size * bag_size，与 batch_size 不同
        #   * 其他 int64 → 全零
        rng = np.random.default_rng(42)
        feed: Dict[str, np.ndarray] = {}
        indices_total = batch_size * max(1, bag_size)  # indices_* 的真实长度
        for inp in probe_sess.get_inputs():
            is_indices = inp.name.startswith("indices") or inp.name == "indices"
            shape = [
                (indices_total if is_indices else batch_size)
                if not isinstance(d, int) or d <= 0 else d
                for d in inp.shape
            ]
            if "float" in inp.type:
                feed[inp.name] = np.zeros(shape, dtype=np.float32)
            elif "int64" in inp.type or "long" in inp.type:
                feed[inp.name] = np.zeros(shape, dtype=np.int64)
            elif "int32" in inp.type:
                feed[inp.name] = np.zeros(shape, dtype=np.int32)
            else:
                feed[inp.name] = np.zeros(shape, dtype=np.float32)

        all_out_names = [o.name for o in probe_sess.get_outputs()]
        results = probe_sess.run(all_out_names, feed)

        runtime_shapes: Dict[str, tuple] = {}
        for name, arr in zip(all_out_names, results):
            if arr is not None and hasattr(arr, "shape"):
                runtime_shapes[name] = tuple(arr.shape)
        # 图输入同样记录实际 shape
        for name, arr in feed.items():
            runtime_shapes[name] = tuple(arr.shape)

        print(
            f"[CSV] 运行时 shape 捕获完成（batch_size={batch_size}, bag_size={bag_size}），"
            f"共获取 {len(runtime_shapes)} 个张量的真实维度"
        )
        return runtime_shapes

    except Exception as exc:
        print(f"[WARN] 运行时 shape 捕获失败（将回退到静态 shape）: {exc}")
        return {}
    finally:
        try:
            _os.remove(tmp_path)
        except Exception:
            pass


def dump_op_shapes_to_csv(onnx_path: str, csv_path: str,
                          batch_size: Optional[int] = None,
                          bag_size: int = 1) -> None:
    """
    解析 ONNX 图（含 shape inference），将每个算子的每个输入/输出
    的名称、dtype、shape 逐行写入 CSV。

    CSV 列：
      node_idx        — 节点在主图中的序号（0-based）
      node_name       — 节点名称
      op_type         — 算子类型（Gemm / Relu / Mul …）
      tensor_role     — 该张量对于本节点的角色：
                          "input"  = 本节点的输入（由上游节点产生）
                          "output" = 本节点的输出（流向下游节点）
      port_idx        — 端口序号（0-based，同 op_type 文档中的输入/输出编号对应）
      tensor_name     — 张量名称
      dtype           — 数据类型（float32 / int64 …）
      shape           — 维度字符串（优先使用运行时实测值）：
                          [16, 128]  具体整数维度（运行时或静态推断均为整数）
                          scalar     0 维标量
                          unknown    无法推断且无运行时值
                          当 batch_size 参数传入时，unk__N 等符号维也会被替换为实测值
      is_dynamic      — shape 中是否仍含未解析的符号维（runtime 情形下恒为 False）
      shape_source    — shape 的来源：
                          runtime        = 由 CPU 运行时实测（最精确）
                          static         = 静态 shape inference，全为整数，无动态维
                          static_symbolic= 静态推断，含 unk__N / batch_size 等符号维
                          unknown        = 无法推断
      producer_node   — 产生该张量的节点（格式：idx:name:op_type）；
                          graph_input = 模型外部输入；
                          initializer = 权重/常量初始化器
      consumer_nodes  — 消费该张量的所有下游节点，逗号分隔
                        （格式同 producer_node）；
                          graph_output = 模型最终输出；空串表示无后续节点
    """
    try:
        import onnx
        from onnx import shape_inference as onnx_si, TensorProto
    except ImportError:
        print("[WARN] onnx 包未安装，跳过 CSV 导出。")
        return

    model = onnx.load(onnx_path)
    try:
        model = onnx_si.infer_shapes(model)
    except Exception:
        pass  # shape inference 失败（如含 Loop 子图）时继续，shape 为 unknown

    graph = model.graph

    # ── 构建 dtype / shape 查找表 ────────────────────────────────────────────
    dtype_map: Dict[str, int] = {}
    shape_map: Dict[str, list] = {}   # name → List[int | str | None]

    def _register_vi(vi) -> None:
        if not vi.type.HasField("tensor_type"):
            return
        dtype_map[vi.name] = vi.type.tensor_type.elem_type
        if vi.type.tensor_type.HasField("shape"):
            dims: list = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param or "?")
                else:
                    dims.append("?")
            shape_map[vi.name] = dims

    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        _register_vi(vi)
    for init in graph.initializer:
        dtype_map[init.name] = init.data_type
        shape_map[init.name] = list(init.dims)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    dtype_map[node.output[0]] = attr.t.data_type
                    shape_map[node.output[0]] = list(attr.t.dims)

    # ── 构建 producer / consumer 映射 ────────────────────────────────────────
    # graph_input_names: 模型外部输入（不由任何节点产生）
    graph_input_names  = {vi.name for vi in graph.input}
    # graph_output_names: 模型最终输出（不被任何节点消费）
    graph_output_names = {vi.name for vi in graph.output}
    # initializer_names: 权重 / 常量（不由节点产生）
    initializer_names  = {init.name for init in graph.initializer}

    def _node_label(idx: int, node) -> str:
        """节点标签：idx:name:op_type"""
        n = node.name or f"(unnamed_{idx})"
        return f"{idx}:{n}:{node.op_type}"

    # producer_map: tensor_name → label 字符串
    producer_map: Dict[str, str] = {}
    for idx, node in enumerate(graph.node):
        label = _node_label(idx, node)
        for t in node.output:
            if t:
                producer_map[t] = label
    # 图输入 / 初始器没有产生节点
    for name in graph_input_names:
        producer_map.setdefault(name, "graph_input")
    for name in initializer_names:
        producer_map.setdefault(name, "initializer")

    # consumer_map: tensor_name → [label, ...]
    consumer_map: Dict[str, List[str]] = {}
    for idx, node in enumerate(graph.node):
        label = _node_label(idx, node)
        for t in node.input:
            if t:
                consumer_map.setdefault(t, []).append(label)
    # 图输出没有后续节点
    for name in graph_output_names:
        consumer_map.setdefault(name, []).append("graph_output")

    # ── 运行时 shape 捕获（可选）─────────────────────────────────────────────
    runtime_shape_map: Dict[str, tuple] = {}
    if batch_size is not None:
        runtime_shape_map = _collect_runtime_shapes(onnx_path, batch_size,
                                                    bag_size=bag_size)

    # ── shape 字符串格式化 ────────────────────────────────────────────────────
    def _fmt_shape(tensor_name: str):
        """返回 (shape_str, is_dynamic, shape_source)。"""
        # 优先使用运行时实测 shape
        rt = runtime_shape_map.get(tensor_name)
        if rt is not None:
            if len(rt) == 0:
                return "scalar", False, "runtime"
            return "[" + ", ".join(str(d) for d in rt) + "]", False, "runtime"
        # 回退到静态 shape inference
        dims = shape_map.get(tensor_name)
        if dims is None:
            return "unknown", False, "unknown"
        if len(dims) == 0:
            return "scalar", False, "static"
        shape_str  = "[" + ", ".join(str(d) for d in dims) + "]"
        has_symbol = any(not isinstance(d, int) for d in dims)
        source     = "static_symbolic" if has_symbol else "static"
        return shape_str, has_symbol, source

    # ── 逐节点写行 ───────────────────────────────────────────────────────────
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_idx", "node_name", "op_type",
            "tensor_role", "port_idx", "tensor_name",
            "dtype", "shape", "is_dynamic", "shape_source",
            "producer_node", "consumer_nodes",
        ])
        for node_idx, node in enumerate(graph.node):
            node_name  = node.name or f"(unnamed_{node_idx})"
            op_type    = node.op_type
            node_label = _node_label(node_idx, node)

            def _write_ports(tensors, role: str) -> None:
                nonlocal total_rows
                for port_idx, tensor_name in enumerate(tensors):
                    if not tensor_name:          # 可选输入占位空串
                        continue
                    raw_dtype = dtype_map.get(tensor_name)
                    dtype_str = _ONNX_DTYPE_NAME.get(raw_dtype, f"type_{raw_dtype}") \
                                if raw_dtype is not None else "unknown"
                    shape_str, is_dynamic, shape_source = _fmt_shape(tensor_name)
                    producer  = producer_map.get(tensor_name, "unknown")
                    consumers = ", ".join(consumer_map.get(tensor_name, []))
                    writer.writerow([
                        node_idx, node_name, op_type,
                        role, port_idx, tensor_name,
                        dtype_str, shape_str, is_dynamic, shape_source,
                        producer, consumers,
                    ])
                    total_rows += 1

            _write_ports(node.input,  "input")
            _write_ports(node.output, "output")

    print(f"[CSV] 算子维度信息已写入: {csv_path}  （{total_rows} 行，{len(graph.node)} 个节点）")


# ─────────────────────────────────────────────────────────────────────────────
# 输入数据生成
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedding_sizes(onnx_path: str) -> Dict[str, int]:
    """
    从 ONNX 图中为每个 indices_* 输入推断对应 EmbeddingBag 权重表的行数
    （即合法索引的上界，范围为 [0, table_rows)）。

    实现思路：
      对每个 indices_* 图输入，找到消费它的 EmbeddingBag/Gather 节点，
      再找该节点的权重 initializer（shape[0] = 表行数）。
    若无法解析则返回空字典，调用方应回退到默认值。
    """
    try:
        import onnx
    except ImportError:
        return {}

    try:
        model = onnx.load(onnx_path)
    except Exception:
        return {}

    graph = model.graph
    init_shapes: Dict[str, list] = {
        init.name: list(init.dims) for init in graph.initializer
    }

    # 图输入名集合
    graph_input_names = {inp.name for inp in graph.input}

    # 为每个张量名建立「被哪些节点消费」的映射
    consumers: Dict[str, list] = {}
    for node in graph.node:
        for inp_name in node.input:
            consumers.setdefault(inp_name, []).append(node)

    result: Dict[str, int] = {}
    for inp in graph.input:
        name = inp.name
        # 只处理 indices_* 类输入（int64 稀疏索引）
        if not (name.startswith("indices") or name == "indices"):
            continue
        for node in consumers.get(name, []):
            if node.op_type not in ("EmbeddingBag", "Gather", "GatherElements"):
                continue
            # EmbeddingBag: input[0]=weight, input[1]=indices, input[2]=offsets
            # Gather:       input[0]=data（weight）, input[1]=indices
            weight_name = node.input[0] if node.input else None
            if weight_name and weight_name in init_shapes:
                rows = init_shapes[weight_name][0]
                result[name] = int(rows)
                break

    return result


def generate_inputs(
    session: object,
    batch_size: int,
    seed: int = 42,
    onnx_path: Optional[str] = None,
    bag_size: int = 1,
) -> Dict[str, np.ndarray]:
    """
    根据模型输入元信息，生成随机测试数据。

    DLRM 输入：
      dense_x   : float32 (batch_size, dense_dim)
      offsets   : int64   (num_tables, batch_size)   ← EmbeddingBag 偏移
      indices_* : int64   (batch_size * bag_size,)   ← 稀疏索引（所有样本拼接）

    bag_size  : num-indices-per-lookup，每个样本的索引数量。
                Loop→Gather+Reshape+ReduceSum 变换后 indices_* 的真实长度
                为 batch_size * bag_size，而非 batch_size。
    onnx_path（可选）：传入后会从 ONNX 图读取每张 embedding 表的真实行数，
    确保生成的随机索引不越界。当 arch-embedding-size 改变时必须传入。
    """
    rng = np.random.default_rng(seed)
    inputs: Dict[str, np.ndarray] = {}

    # 从 ONNX 图中读取每个 indices_* 对应的表大小（行数）
    emb_sizes: Dict[str, int] = _get_embedding_sizes(onnx_path) if onnx_path else {}

    # indices_* 的真实长度：每样本有 bag_size 个索引，所有样本拼接
    indices_total = batch_size * max(1, bag_size)

    for inp in session.get_inputs():
        name  = inp.name
        dtype = inp.type   # 如 "tensor(float)", "tensor(int64)"
        shape = inp.shape  # 可能含 "batch_size" 等字符串

        # 判断是否为 indices_* 输入（需用 indices_total 而非 batch_size）
        is_indices = name.startswith("indices") or name == "indices"

        # 将动态维度替换为正确的大小
        concrete = []
        for d in shape:
            if isinstance(d, int) and d > 0:
                concrete.append(d)
            else:
                # 动态符号维度
                concrete.append(indices_total if is_indices else batch_size)

        if "float" in dtype:
            data = rng.standard_normal(concrete).astype(np.float32)
        elif "int64" in dtype or "long" in dtype:
            # offsets 需要单调递增且首元素为 0（EmbeddingBag 语义）
            if name == "offsets" or name.startswith("offsets_"):
                offsets = np.zeros(concrete, dtype=np.int64)
                for row in range(concrete[0]):
                    offsets[row] = np.sort(
                        rng.integers(0, concrete[1] + 1, size=concrete[1])
                    )
                    offsets[row][0] = 0
                data = offsets
            else:
                # 稀疏索引：用从 ONNX 读取的真实表大小作上界，
                # 回退值 100（仅当 arch-embedding-size >= 100 时安全）
                table_rows = emb_sizes.get(name, 100)
                data = rng.integers(0, table_rows, size=concrete, dtype=np.int64)
        elif "int32" in dtype:
            data = rng.integers(0, 100, size=concrete, dtype=np.int32)
        else:
            data = rng.standard_normal(concrete).astype(np.float32)

        inputs[name] = data

    return inputs


# ─────────────────────────────────────────────────────────────────────────────
# 算子–设备分布报告
# ─────────────────────────────────────────────────────────────────────────────

def print_device_placement(session: object) -> None:
    """
    打印 ORT 为每个图节点分配的执行 Provider（设备）。

    ORT 内部维护 NodeProvenanceMap，这里通过 enable_profiling 结果
    或 Model Metadata 分析估算 NPU/CPU 分布。
    对于没有 CANN provider 的情况，全部显示为 CPU。
    """
    providers = session.get_providers()
    has_cann = "CANNExecutionProvider" in providers

    print("\n" + "=" * 70)
    print("算子设备分布（每种 op_type 的执行位置）")
    print("=" * 70)

    # ORT 1.x 版本可通过内部 API 获取节点归属；
    # 通用做法是依赖 profiling JSON 文件中的 "provider" 字段。
    if not has_cann:
        print("  [INFO] 当前仅 CPUExecutionProvider 可用，所有算子在 CPU 执行。")
        print("  CANN provider 不可用。安装支持 CANN 的 onnxruntime 后，")
        print("  下列算子将自动卸载到 NPU：")
        print("    Gemm, MatMul, Relu, Sigmoid, Add, Mul, Concat, Transpose")
        print("    以及大部分 element-wise 算子")
        print("  下列算子通常仍在 CPU 执行（CANN 不支持）：")
        print("    Loop, Gather（部分）, Slice, Shape, Constant, Unsqueeze, Reshape")
    else:
        print("  混合执行模式 (CANNExecutionProvider + CPUExecutionProvider):")
        print("  ┌─────────────────────────┬──────────────────────────┐")
        print("  │  NPU (CANN)             │  CPU (fallback)          │")
        print("  ├─────────────────────────┼──────────────────────────┤")
        print("  │  Gemm     Relu          │  Loop     Shape          │")
        print("  │  MatMul   Sigmoid       │  Gather   Slice          │")
        print("  │  Add      Mul           │  Constant Unsqueeze      │")
        print("  │  Concat   Transpose     │  Reshape  Flatten        │")
        print("  │  Flatten  (元素算子)    │  (控制流/索引类算子)     │")
        print("  └─────────────────────────┴──────────────────────────┘")
        print("  注：实际分配以 ORT 运行时与 CANN kernel 注册表为准。")
        print("       启用 --enable-profiling 后可在 JSON 中查看 'provider' 字段。")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# profiling 结果解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_profile_json(profile_dir: str) -> None:
    """
    解析 ORT profiling JSON 文件，按 NPU/CPU 统计耗时分布。
    profiling 文件由 opts.profile_file_prefix 确定。
    """
    import glob
    pattern = str(Path(profile_dir) / "ort_cann_profile*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[PROFILE] 未找到 profiling 文件 ({pattern})")
        return

    latest = files[-1]
    print(f"\n[PROFILE] 解析文件: {latest}")

    with open(latest, "r") as f:
        events = json.load(f)

    # 按 provider 分组统计
    provider_stats: Dict[str, Dict] = {}
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        args = ev.get("args", {})
        provider = args.get("provider", "unknown")
        op_type  = args.get("op_name", ev.get("name", "unknown"))
        dur_us   = ev.get("dur", 0)

        if provider not in provider_stats:
            provider_stats[provider] = {"total_us": 0, "ops": {}}
        provider_stats[provider]["total_us"] += dur_us
        ps = provider_stats[provider]["ops"]
        ps[op_type] = ps.get(op_type, 0) + dur_us

    if not provider_stats:
        print("[PROFILE] profiling 文件中未找到 Node 事件。")
        return

    total_all = sum(v["total_us"] for v in provider_stats.values())
    print(f"\n{'Provider':<35} {'Total (ms)':>12} {'Ratio':>8}")
    print("-" * 60)
    for pv, stats in sorted(provider_stats.items(), key=lambda x: -x[1]["total_us"]):
        ms = stats["total_us"] / 1000
        pct = stats["total_us"] / total_all * 100 if total_all > 0 else 0
        label = "NPU" if "CANN" in pv else "CPU"
        print(f"  {pv:<33} {ms:>10.2f}ms {pct:>7.1f}%  ({label})")
        # top-5 ops
        top5 = sorted(stats["ops"].items(), key=lambda x: -x[1])[:10]
        for op, us in top5:
            print(f"    ├─ {op:<30} {us/1000:>8.2f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# 主推理循环
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    session: object,
    num_batches: int,
    batch_size: int,
    warmup_batches: int,
    onnx_path: Optional[str] = None,
    bag_size: int = 1,
) -> List[float]:
    """
    运行推理，返回每次推理耗时列表（ms）。
    onnx_path 传入后，generate_inputs 将从 ONNX 图读取 embedding 表大小，
    确保随机索引不越界（arch-embedding-size 变动时必须传入）。
    bag_size 用于正确生成 indices_* 输入（大小为 batch_size * bag_size）。
    """
    output_names = [o.name for o in session.get_outputs()]
    run_warmup(session, batch_size, warmup_batches,
               onnx_path=onnx_path, bag_size=bag_size)

    # ── 正式推理 ──────────────────────────────────────────────────────────────
    latencies: List[float] = []
    print(f"\n[INFER] 正式推理: {num_batches} 次，batch_size={batch_size}")
    print(f"        输出张量: {output_names}")

    for i in range(num_batches):
        feed = generate_inputs(session, batch_size, seed=1000 + i,
                               onnx_path=onnx_path, bag_size=bag_size)

        t0 = time.perf_counter()
        outputs = session.run(output_names, feed)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        out_shape = outputs[0].shape if outputs else ()
        print(f"  Batch {i:3d}: {elapsed_ms:7.2f} ms  output shape={out_shape}")

    return latencies


def print_statistics(latencies: List[float]) -> None:
    """打印延迟统计信息。"""
    if not latencies:
        return
    arr = np.array(latencies)
    print("\n" + "=" * 50)
    print("推理延迟统计 (ms)")
    print("=" * 50)
    print(f"  样本数  : {len(arr)}")
    print(f"  均值    : {arr.mean():.2f} ms")
    print(f"  中位数  : {np.median(arr):.2f} ms")
    print(f"  最小值  : {arr.min():.2f} ms")
    print(f"  最大值  : {arr.max():.2f} ms")
    print(f"  P90     : {np.percentile(arr, 90):.2f} ms")
    print(f"  P99     : {np.percentile(arr, 99):.2f} ms")
    print(f"  吞吐量  : {1000 / arr.mean():.1f} batch/s")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DLRM ONNX Runtime 推理（CANN NPU + CPU 混合执行）"
    )
    # 模型
    parser.add_argument(
        "--onnx-path", type=str,
        default="./dlrm_onnx/dlrm_s_pytorch.onnx",
        help="ONNX 模型路径",
    )
    # 执行设备
    parser.add_argument(
        "--use-cann", action="store_true", default=False,
        help="启用 CANNExecutionProvider（NPU 加速）",
    )
    parser.add_argument(
        "--device-id", type=int, default=0,
        help="NPU 卡号（默认 0）",
    )
    # 数据
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--num-batches", type=int, default=10,
                        help="推理次数")
    parser.add_argument("--warmup-batches", type=int, default=3,
                        help="预热次数（不计入统计）")
    # 线程
    parser.add_argument("--intra-threads", type=int, default=8,
                        help="ORT 算子内部并行线程数（CPU 算子用）")
    parser.add_argument("--inter-threads", type=int, default=1,
                        help="ORT 算子间并行线程数")
    # Loop → Gather 变换（独立于 --use-cann 的图改写开关）
    parser.add_argument("--no-replace-loop", action="store_true", default=False,
                        help="禁用 Loop→Gather 替换变换（默认: 启用；"
                             "是否创建替换后的新 ONNX 图仅由该开关控制，"
                             "不依赖 --use-cann）")
    parser.add_argument(
        "--num-indices-per-lookup", type=int, default=0,
        metavar="N",
        help=(
            "每个样本查询的 embedding 索引数（即 DLRM 的 num-indices-per-lookup）。\n"
            "用于 Loop→Gather 替换时指定 bag_size：\n"
            "  N=1 → 单 Gather（每样本 1 个索引）\n"
            "  N>1 → Gather + Reshape + ReduceSum（每样本 N 个索引求均值/和）\n"
            "默认为 0（自动推断，推断失败时保守回退到 1）。\n"
            "建议与导出模型时的 --num-indices-per-lookup 保持一致。"
        ),
    )
    # 强制算子卸载到 CPU
    parser.add_argument(
        "--force-cpu-ops", type=str, default="",
        metavar="OP[,OP,...]",
        help=(
            "强制在 CPU 上执行的算子类型列表，逗号分隔（仅在 --use-cann 时有效）。\n"
            "实现：将目标算子替换为语义等价但 CANN 不支持的算子，令其自动回退到 CPU。\n"
            "当前支持: Relu（→ LeakyRelu(alpha=0)）。\n"
            "示例: --force-cpu-ops Relu\n"
            "      --force-cpu-ops Relu,Sigmoid"
        ),
    )
    # Profiling
    parser.add_argument("--enable-profiling", action="store_true", default=False,
                        help="启用 ORT profiling（输出 JSON）")
    parser.add_argument(
        "--profile-warmup", action="store_true", default=False,
        help="将 warmup 阶段也写入 profiling；默认关闭，即 warmup 不计入 profiling",
    )
    parser.add_argument("--profile-dir", type=str,
                        default="./onnx_operator_analysis",
                        help="Profiling JSON 输出目录")
    parser.add_argument(
        "--disable-graph-optimizations", action="store_true", default=False,
        help="禁用 ORT 图优化与融合，便于对齐原始/改写后的 ONNX 节点与 profiling 结果",
    )
    # 调试
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="打印详细信息")
    # 维度 CSV
    parser.add_argument(
        "--shape-csv", type=str, default="",
        help="将所有算子的输入/输出维度写入该 CSV 文件（默认不写出；"
             "可指定如 ./op_shapes.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 打印环境信息 ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("DLRM ONNX Runtime 推理（NPU + CPU 混合执行）")
    print("=" * 70)
    print(f"  ONNX 模型     : {args.onnx_path}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Batches       : {args.num_batches}（warmup={args.warmup_batches}）")
    print(f"  CANN (NPU)    : {'启用' if args.use_cann else '禁用（仅 CPU）'}")
    print(f"  Replace Loop  : {'启用' if not args.no_replace_loop else '禁用'}")
    if args.use_cann:
        print(f"  NPU device_id : {args.device_id}")
    print(f"  Intra threads : {args.intra_threads}")
    print(f"  Graph Opt     : {'禁用' if args.disable_graph_optimizations else '启用（默认）'}")
    print(f"  Profiling     : {'启用' if args.enable_profiling else '禁用'}")
    if args.enable_profiling:
        print(f"  Profile warmup: {'计入' if args.profile_warmup else '不计入'}")

    # ── CANN 环境检查 ─────────────────────────────────────────────────────────
    cann_found = _setup_cann_env()
    if args.use_cann and not cann_found:
        print(f"\n[WARN] CANN Toolkit 未在 {_CANN_TOOLKIT_ROOT} 找到，")
        print("       NPU 加速将不可用。请确认已安装 CANN Toolkit。")

    # ── 检查 ORT 版本 ─────────────────────────────────────────────────────────
    ort, available_providers = _import_onnxruntime()
    print(f"\n  ORT 版本      : {ort.__version__}")
    print(f"  可用 providers: {available_providers}")

    if args.use_cann and "CANNExecutionProvider" not in available_providers:
        print("\n[WARN] CANNExecutionProvider 不在当前 onnxruntime 中。")
        print("       请运行 `bash setup_ort_cann.sh` 安装支持 CANN 的版本。")

    # ── 构建 Session ──────────────────────────────────────────────────────────
    # 解析 --force-cpu-ops: "Relu,Sigmoid" → ["Relu", "Sigmoid"]
    force_cpu_ops: List[str] = (
        [t.strip() for t in args.force_cpu_ops.split(",") if t.strip()]
        if args.force_cpu_ops else []
    )
    session_kwargs = {
        "onnx_path": args.onnx_path,
        "use_cann": args.use_cann,
        "device_id": args.device_id,
        "profile_dir": args.profile_dir,
        "intra_threads": args.intra_threads,
        "inter_threads": args.inter_threads,
        "disable_graph_optimizations": args.disable_graph_optimizations,
        "replace_loop": not args.no_replace_loop,
        "force_cpu_ops": force_cpu_ops,
        "bag_size": args.num_indices_per_lookup,
    }

    if args.enable_profiling and args.warmup_batches > 0 and not args.profile_warmup:
        print("\n[PROFILE] warmup 不计入 profiling：先执行无 profiling warmup，再重建 profiled session。")
        warmup_session, _, warmup_onnx_path = build_session(
            enable_profiling=False,
            **session_kwargs,
        )
        run_warmup(
            warmup_session,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            onnx_path=warmup_onnx_path,
            bag_size=args.num_indices_per_lookup,
        )
        del warmup_session
        session, actual_providers, effective_onnx_path = build_session(
            enable_profiling=True,
            **session_kwargs,
        )
        effective_warmup_batches = 0
    else:
        session, actual_providers, effective_onnx_path = build_session(
            enable_profiling=args.enable_profiling,
            **session_kwargs,
        )
        effective_warmup_batches = args.warmup_batches

    # ── 模型输入信息 ──────────────────────────────────────────────────────────
    if args.verbose:
        print("\n[MODEL] 输入张量:")
        for inp in session.get_inputs():
            print(f"  {inp.name:<25} dtype={inp.type:<20} shape={inp.shape}")
        print("[MODEL] 输出张量:")
        for out in session.get_outputs():
            print(f"  {out.name:<25} dtype={out.type:<20} shape={out.shape}")

    # ── 算子维度 CSV 导出（可选）────────────────────────────────────────────
    if args.shape_csv:
        dump_op_shapes_to_csv(effective_onnx_path, args.shape_csv,
                              batch_size=args.batch_size,
                              bag_size=args.num_indices_per_lookup)

    # ── 算子设备分布 ──────────────────────────────────────────────────────────
    print_device_placement(session)

    # ── 推理 ─────────────────────────────────────────────────────────────────
    latencies = run_inference(
        session        = session,
        num_batches    = args.num_batches,
        batch_size     = args.batch_size,
        warmup_batches = effective_warmup_batches,
        onnx_path      = effective_onnx_path,
        bag_size       = args.num_indices_per_lookup,
    )

    # ── 统计 ─────────────────────────────────────────────────────────────────
    print_statistics(latencies)

    # ── Profiling 结果解析 ────────────────────────────────────────────────────
    if args.enable_profiling:
        # 结束 session 以触发 profiling 文件写出
        session_prof = getattr(session, "end_profiling", None)
        if callable(session_prof):
            prof_file = session_prof()
            print(f"[PROFILE] profiling 文件: {prof_file}")
        parse_profile_json(args.profile_dir)


if __name__ == "__main__":
    main()
