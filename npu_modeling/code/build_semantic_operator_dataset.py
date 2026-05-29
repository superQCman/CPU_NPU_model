import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


DELIVERY_ROOT = Path(__file__).resolve().parents[1]
DTYPE_BYTES = {"float16": 2.0, "bfloat16": 2.0, "float32": 4.0}
OP_CLASS_MAP = {
    "MatMulV2": "DenseLinearAlgebra",
    "BatchMatMulV2": "DenseLinearAlgebra",
    "Relu": "ElementwiseActivation",
    "Sigmoid": "ElementwiseActivation",
    "ConcatD": "TensorTransformMove",
    "Transpose": "TensorTransformMove",
    "PadV3": "TensorTransformMove",
    "MemSet": "TensorTransformMove",
    "Index": "IndexingSparse",
    "AdvancedIndexing": "IndexingSparse",
}


def load_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_many(paths: str):
    rows = []
    for raw in paths.split(","):
        p = Path(raw.strip())
        if p.exists():
            rows.extend(load_csv(p))
    return rows


def fnum(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def estimate_work(op: str, params: dict, dtype_bytes: float):
    pattern = str(params.get("pattern", ""))
    p = {k: float(v) for k, v in params.items() if isinstance(v, (int, float, bool))}
    inp = out = flops = 0.0
    if op in {"Linear", "GemmBias", "MatMulV2"}:
        m, k, n = p.get("m", 0.0), p.get("k", 0.0), p.get("n", 0.0)
        inp, out = m * k + k * n + n, m * n
        flops = 2.0 * m * k * n
    elif op == "BatchMatMulV2":
        b, m, k, n = p.get("batch", 1.0), p.get("m", 0.0), p.get("k", 0.0), p.get("n", 0.0)
        inp, out = b * m * k + b * k * n, b * m * n
        flops = 2.0 * b * m * k * n
    elif op in {"Add", "Mul", "Gelu", "Tanh", "Where", "Cast", "NonZero", "Relu", "Sigmoid", "MemSet"}:
        out = p.get("rows", 1.0) * p.get("cols", 1.0)
        inp = out * (2.0 if op in {"Add", "Mul", "Where"} else 1.0)
        flops = out * (4.0 if op in {"Gelu", "Tanh"} else 1.0)
    elif op in {"ReduceSum", "ReduceMean", "Softmax", "Sort"}:
        out = p.get("d0", 1.0) * p.get("d1", 1.0) * p.get("d2", 1.0)
        inp = out
        flops = out * (8.0 if op == "Softmax" else max(math.log2(max(out, 2.0)), 1.0) if op == "Sort" else 1.0)
    elif op == "TopK":
        inp = p.get("d0", 1.0) * p.get("d1", 1.0)
        out = p.get("d0", 1.0) * p.get("k", 1.0)
        flops = inp * max(math.log2(max(p.get("k", 1.0), 2.0)), 1.0)
    elif op in {"Reshape", "Tile", "BroadcastTo", "Slice", "Split"}:
        inp = p.get("d0", 1.0) * p.get("d1", 1.0) * p.get("d2", 1.0)
        out = inp * p.get("multiplier", 1.0)
    elif op == "Gather":
        inp = p.get("rows", 1.0) * p.get("cols", 1.0) + p.get("index_count", 1.0)
        out = p.get("index_count", 1.0) * p.get("cols", 1.0)
    elif op == "AdvancedIndexing":
        outer = p.get("outer", 1.0)
        d1 = p.get("d1", 1.0)
        d2 = p.get("d2", 1.0)
        idx_n = p.get("index_count", outer)
        if pattern == "pair_select":
            inp = outer * d1 * d2 + idx_n * 2.0
            out = outer * idx_n
        else:
            inp = outer * d1 * d2 + idx_n * 3.0
            out = idx_n
    elif op == "ScatterAdd":
        inp = p.get("index_count", 1.0) * p.get("cols", 1.0) + p.get("index_count", 1.0)
        out = p.get("rows", 1.0) * p.get("cols", 1.0)
        flops = p.get("index_count", 1.0) * p.get("cols", 1.0)
    elif op == "EmbeddingBag":
        inp = p.get("num_bags", 1.0) * p.get("avg_bag_len", 1.0) * p.get("embedding_dim", 1.0)
        out = p.get("num_bags", 1.0) * p.get("embedding_dim", 1.0)
        flops = inp
    elif op == "LayerNorm":
        out = p.get("batch", 1.0) * p.get("seq", 1.0) * p.get("hidden", 1.0)
        inp = out
        flops = out * 6.0
    elif op == "BatchNorm1D":
        out = p.get("batch", 1.0) * p.get("channels", 1.0) * p.get("length", 1.0)
        inp = out
        flops = out * 4.0
    elif op == "MaxPool2D":
        b, c, h, w = p.get("batch", 1.0), p.get("channels", 1.0), p.get("h", 1.0), p.get("w", 1.0)
        k, s = p.get("kernel", 1.0), max(p.get("stride", 1.0), 1.0)
        oh = max(math.floor((h + 2.0 * math.floor(k / 2.0) - k) / s + 1.0), 1.0)
        ow = max(math.floor((w + 2.0 * math.floor(k / 2.0) - k) / s + 1.0), 1.0)
        inp, out = b * c * h * w, b * c * oh * ow
        flops = out * k * k
        bytes_rw = (inp + out) * dtype_bytes
    elif op == "Index":
        inp = p.get("rows", 1.0) * p.get("cols", 1.0) + p.get("index_count", 1.0)
        out = max(p.get("index_count", 1.0), 1.0) * p.get("cols", 1.0)
    elif op == "ConcatD":
        b = p.get("batch", 1.0)
        feat = p.get("feature_a", 0.0) + p.get("feature_b", 0.0) + p.get("feature_c", 0.0)
        inp = out = b * feat
    elif op in {"Transpose", "PadV3"}:
        if op == "Transpose" and any(k in p for k in ["d0", "d1", "d2"]):
            inp = out = max(p.get("d0", 1.0), 1.0) * max(p.get("d1", 1.0), 1.0) * max(p.get("d2", 1.0), 1.0)
        else:
            inp = out = max(p.get("batch", 1.0), 1.0) * max(p.get("channels", 1.0), 1.0) * max(p.get("width", 1.0), 1.0)
        if op == "PadV3":
            out = max(p.get("batch", 1.0), 1.0) * max(p.get("channels", 1.0), 1.0) * (max(p.get("width", 1.0), 1.0) + p.get("pad_left", 0.0) + p.get("pad_right", 0.0))
    bytes_rw = (inp + out) * dtype_bytes
    working = max(inp, out, 1.0) * dtype_bytes
    return inp, out, flops, bytes_rw, working


def add_param_features(rec, op, params):
    pattern = str(params.get("pattern", ""))
    p = {k: float(v) for k, v in params.items() if isinstance(v, (int, float, bool))}
    for k, v in p.items():
        rec[f"param_{k}"] = v
        rec[f"log1p_param_{k}"] = math.log1p(max(v, 0.0))
    if op in {"Linear", "GemmBias", "MatMulV2"}:
        m, k, n = p.get("m", 0.0), p.get("k", 0.0), p.get("n", 0.0)
        rec.update({"matmul_mn": m * n, "matmul_mk": m * k, "matmul_kn": k * n, "matmul_m_mod_16": m % 16.0, "matmul_k_mod_16": k % 16.0, "matmul_n_mod_16": n % 16.0})
    elif op == "BatchMatMulV2":
        b, m, k, n = p.get("batch", 1.0), p.get("m", 0.0), p.get("k", 0.0), p.get("n", 0.0)
        rec.update({
            "bmm_batch": b,
            "bmm_mn": m * n,
            "bmm_mk": m * k,
            "bmm_kn": k * n,
            "bmm_total_mn": b * m * n,
            "bmm_m_mod_16": m % 16.0,
            "bmm_k_mod_16": k % 16.0,
            "bmm_n_mod_16": n % 16.0,
            "bmm_batch_mod_8": b % 8.0,
        })
    elif op in {"ReduceSum", "ReduceMean", "Softmax", "Sort"}:
        dims = [p.get("d0", 1.0), p.get("d1", 1.0), p.get("d2", 1.0)]
        axis = min(max(int(p.get("axis", 0.0)), 0), 2)
        rec.update({"reduce_axis_size": dims[axis], "reduce_outer_size": math.prod(dims) / max(dims[axis], 1.0), "reduce_axis_log2": math.log2(max(dims[axis], 1.0))})
    elif op == "MaxPool2D":
        rec.update({"pool_window_elems": p.get("kernel", 1.0) ** 2, "pool_stride": p.get("stride", 1.0), "pool_hw": p.get("h", 1.0) * p.get("w", 1.0)})
    elif op in {"Gather", "ScatterAdd"}:
        rec.update({"sparse_index_density": p.get("index_count", 1.0) / max(p.get("rows", 1.0), 1.0), "sparse_payload": p.get("index_count", 1.0) * p.get("cols", 1.0)})
    elif op == "AdvancedIndexing":
        outer = p.get("outer", 1.0)
        d1 = p.get("d1", 1.0)
        d2 = p.get("d2", 1.0)
        idx_n = p.get("index_count", outer)
        rec.update({
            "advanced_index_outer": outer,
            "advanced_index_d1": d1,
            "advanced_index_d2": d2,
            "advanced_index_count": idx_n,
            "advanced_index_volume": outer * d1 * d2,
            "advanced_index_density": idx_n / max(outer * d1 * d2, 1.0),
            "advanced_index_pair_select": 1.0 if pattern == "pair_select" else 0.0,
            "advanced_index_outer_mod_4": outer % 4.0,
            "advanced_index_outer_mod_8": outer % 8.0,
            "advanced_index_outer_mod_16": outer % 16.0,
            "advanced_index_outer_mod_32": outer % 32.0,
            "advanced_index_outer_mod_64": outer % 64.0,
        })
    elif op == "EmbeddingBag":
        rec.update({"emb_total_indices": p.get("num_bags", 1.0) * p.get("avg_bag_len", 1.0), "emb_output_elems": p.get("num_bags", 1.0) * p.get("embedding_dim", 1.0), "emb_dim_mod_32": p.get("embedding_dim", 1.0) % 32.0})
    elif op == "LayerNorm":
        rec.update({"norm_rows": p.get("batch", 1.0) * p.get("seq", 1.0), "norm_hidden": p.get("hidden", 1.0), "norm_hidden_mod_32": p.get("hidden", 1.0) % 32.0})
    elif op in {"Reshape", "Tile", "BroadcastTo", "Slice", "Split"}:
        d0, d1, d2 = p.get("d0", 1.0), p.get("d1", 1.0), p.get("d2", 1.0)
        mult = p.get("multiplier", 1.0)
        rec.update({
            "transform_input_elems": d0 * d1 * d2,
            "transform_multiplier": mult,
            "transform_output_elems": d0 * d1 * d2 * mult,
            "transform_d1_mod_16": d1 % 16.0,
            "transform_d2_mod_16": d2 % 16.0,
        })
    elif op == "ConcatD":
        fa, fb, fc = p.get("feature_a", 0.0), p.get("feature_b", 0.0), p.get("feature_c", 0.0)
        feat_sum = fa + fb + fc
        feat_max = max(fa, fb, fc)
        nonzero = sum(1 for x in [fa, fb, fc] if x > 0)
        rec.update({
            "concat_feature_sum": feat_sum,
            "concat_feature_max": feat_max,
            "concat_feature_min_nonzero": min([x for x in [fa, fb, fc] if x > 0] or [0.0]),
            "concat_nonzero_inputs": float(nonzero),
            "concat_dominant_ratio": feat_max / max(feat_sum, 1.0),
            "concat_sum_mod_32": feat_sum % 32.0,
            "concat_sum_mod_64": feat_sum % 64.0,
        })
    elif op == "Transpose":
        if any(k in p for k in ["d0", "d1", "d2"]):
            d0, d1, d2 = p.get("d0", 1.0), p.get("d1", 1.0), p.get("d2", 1.0)
        else:
            d0, d1, d2 = p.get("batch", 1.0), p.get("channels", 1.0), p.get("width", 1.0)
        rec.update({
            "transpose_volume": d0 * d1 * d2,
            "transpose_d0_mod_16": d0 % 16.0,
            "transpose_d1_mod_16": d1 % 16.0,
            "transpose_d2_mod_16": d2 % 16.0,
            "transpose_channels_mod_16": d1 % 16.0,
            "transpose_width_mod_16": d2 % 16.0,
            "transpose_perm_code": p.get("perm", 0.0),
        })
    elif op == "PadV3":
        width = p.get("width", 1.0)
        pad = p.get("pad_left", 0.0) + p.get("pad_right", 0.0)
        rec.update({"pad_output_width": width + pad, "pad_total": pad, "pad_output_mod_32": (width + pad) % 32.0})
    return rec


def build_record(case_id, profile_dir, major_class, op_type, dtype, params, target_us, hw, latency_p50=0.0, latency_mean=0.0):
    dtype_bytes = DTYPE_BYTES.get(str(dtype).lower(), 2.0 if str(dtype).lower() in {"bf16", "bfloat16"} else 4.0)
    inp, out, flops, bytes_rw, working = estimate_work(op_type, params, dtype_bytes)
    rec = {
        "case_id": case_id,
        "profile_dir": profile_dir,
        "major_class": major_class,
        "op_type": op_type,
        "dtype": dtype,
        "target_kernel_compute_us": target_us,
        "latency_p50_us": latency_p50,
        "latency_mean_us": latency_mean,
        "dtype_bytes": dtype_bytes,
        "input_elems_estimate": inp,
        "output_elems_estimate": out,
        "flops_estimate": flops,
        "bytes_rw_estimate": bytes_rw,
        "working_set_bytes": working,
        "arithmetic_intensity": flops / max(bytes_rw, 1.0),
        "peak_cube_flops": hw["peak_cube_flops"],
        "peak_vector_flops": hw["peak_vector_flops"],
        "memory_capacity_bytes": hw["memory_capacity_bytes"],
        "l1_capacity_bytes": hw["l1_capacity_bytes"],
        "l2_capacity_bytes": hw["l2_capacity_bytes"],
        "memory_bandwidth_bytes_per_s": hw["memory_bandwidth_bytes_per_s"],
        "t_cube_lb_us": flops / max(hw["peak_cube_flops"], 1.0) * 1e6,
        "t_vec_lb_us": flops / max(hw["peak_vector_flops"], 1.0) * 1e6,
        "t_mem_lb_us": bytes_rw / max(hw["memory_bandwidth_bytes_per_s"], 1.0) * 1e6,
        "working_set_over_l1": working / max(hw["l1_capacity_bytes"], 1.0),
        "working_set_over_l2": working / max(hw["l2_capacity_bytes"], 1.0),
        "working_set_over_memory_capacity": working / max(hw["memory_capacity_bytes"], 1.0),
        "kernel_count_observed": 0.0,
        "kernel_type_unique_observed": 0.0,
        "share_ai_core_observed": 0.0,
        "share_ai_vector_core_observed": 0.0,
        "share_mix_aiv_observed": 0.0,
    }
    rec = add_param_features(rec, op_type, params)
    for c in ["input_elems_estimate", "output_elems_estimate", "flops_estimate", "bytes_rw_estimate", "working_set_bytes", "arithmetic_intensity", "t_cube_lb_us", "t_vec_lb_us", "t_mem_lb_us"]:
        rec[f"log1p_{c}"] = math.log1p(max(fnum(rec[c]), 0.0))
    return rec


def main():
    parser = argparse.ArgumentParser(description="Build semantic operator dataset from independent profiler traces")
    parser.add_argument("--metrics", required=True, help="Comma-separated sampling metrics CSVs")
    parser.add_argument("--kernel-map", required=True, help="Comma-separated kernel mapping CSVs")
    parser.add_argument("--old-core-labels", default="", help="Comma-separated legacy 9-op operator_core_labels CSVs")
    parser.add_argument("--hardware", default=DELIVERY_ROOT / "models/hardware_descriptor.json")
    parser.add_argument("--output", default=DELIVERY_ROOT / "input_data/semantic_operator_dataset.csv")
    args = parser.parse_args()
    hw = json.loads(Path(args.hardware).read_text(encoding="utf-8"))
    metrics = {}
    for r in load_many(args.metrics):
        if r.get("status") != "ok" or not r.get("profile_dir"):
            continue
        metrics[(str(r["case_id"]), str(r["profile_dir"]))] = r
    kernels_by_case = defaultdict(list)
    for kr in load_many(args.kernel_map):
        d = fnum(kr.get("duration_us"))
        if d > 0:
            kernels_by_case[(str(kr.get("case_id", "")), str(kr.get("profile_dir", "")))].append(kr)
    rows = []
    for (case_id, profile_dir), krows in sorted(kernels_by_case.items(), key=lambda x: (int(x[0][0]) if str(x[0][0]).isdigit() else 0, x[0][1])):
        mr = metrics.get((case_id, profile_dir))
        if not mr:
            continue
        params = json.loads(mr["params_json"])
        dtype_bytes = DTYPE_BYTES.get(mr["dtype"], 4.0)
        inp, out, flops, bytes_rw, working = estimate_work(mr["op_type"], params, dtype_bytes)
        core_duration = Counter()
        ktype_count = Counter()
        total = 0.0
        for kr in krows:
            d = fnum(kr.get("duration_us"))
            total += d
            core_duration[kr.get("accelerator_core", "unknown")] += d
            ktype_count[kr.get("kernel_type", "unknown")] += 1
        if total <= 0:
            continue
        rec = build_record(case_id, profile_dir, mr["major_class"], mr["op_type"], mr["dtype"], params, total, hw, fnum(mr.get("latency_p50_us")), fnum(mr.get("latency_mean_us")))
        rec.update({"kernel_count_observed": len(krows), "kernel_type_unique_observed": len(ktype_count), "share_ai_core_observed": core_duration["AI_CORE"] / total, "share_ai_vector_core_observed": core_duration["AI_VECTOR_CORE"] / total, "share_mix_aiv_observed": core_duration["MIX_AIV"] / total})
        rows.append(rec)
    for r in load_many(args.old_core_labels) if args.old_core_labels else []:
        if r.get("build_status") != "ok":
            continue
        target = fnum(r.get("kernel_total_duration_us")) or fnum(r.get("device_total_duration_us"))
        if target <= 0:
            continue
        op_type = r.get("op_type", "")
        params = {k: fnum(r.get(k)) for k in ["batch", "m", "k", "n", "rows", "cols", "outer", "d0", "d1", "d2", "perm", "feature_a", "feature_b", "feature_c", "axis", "channels", "width", "pad_left", "pad_right", "index_count"]}
        rec = build_record(f"old_{r.get('case_id')}", r.get("profile_dir", ""), OP_CLASS_MAP.get(op_type, "Unknown"), op_type, r.get("dtype", "bf16"), params, target, hw, fnum(r.get("latency_p50_us")), fnum(r.get("latency_mean_us")))
        rec.update({"kernel_count_observed": fnum(r.get("kernel_row_count")), "share_ai_core_observed": fnum(r.get("share_ai_core")), "share_ai_vector_core_observed": fnum(r.get("share_ai_vector_core")), "share_mix_aiv_observed": fnum(r.get("share_mix_aiv"))})
        rows.append(rec)
    fieldnames = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {len(rows)} semantic operator rows to {out}")


if __name__ == "__main__":
    main()
