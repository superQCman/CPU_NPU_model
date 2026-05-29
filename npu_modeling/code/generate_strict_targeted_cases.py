import argparse
import csv
import json
from pathlib import Path


DELIVERY_ROOT = Path(__file__).resolve().parents[1]


def add(rows, cid, major_class, op_type, dtype, params):
    rows.append({
        "case_id": cid,
        "major_class": major_class,
        "priority": "P0",
        "op_type": op_type,
        "dtype": dtype,
        "params_json": json.dumps(params, separators=(",", ":")),
    })
    return cid + 1


def main():
    parser = argparse.ArgumentParser(description="Generate strict per-model targeted NPU cases")
    parser.add_argument("--output", default=DELIVERY_ROOT / "results/train_supplements/strict_per_model_targeted_cases.csv")
    parser.add_argument("--start-case-id", type=int, default=30000)
    args = parser.parse_args()

    rows = []
    cid = args.start_case_id
    dtypes = ["float16", "bfloat16", "float32"]

    # AxisCompute: cover troublesome axis/outer regimes and large Sort/Softmax cases.
    axis_shapes = [
        (8, 1024, 16), (16, 1024, 8), (32, 1024, 16), (8, 1024, 64),
        (256, 1024, 64), (256, 1024, 128), (8, 1024, 256),
        (64, 512, 64), (128, 256, 128), (256, 128, 256),
    ]
    for dtype in dtypes:
        for d0, d1, d2 in axis_shapes:
            for axis in [0, 1, 2]:
                for op in ["Softmax", "ReduceSum", "ReduceMean", "Sort"]:
                    cid = add(rows, cid, "ReductionSelection", op, dtype, {"d0": d0, "d1": d1, "d2": d2, "axis": axis})
        for d0, d1 in [(64, 1024), (128, 1024), (256, 1024), (512, 512)]:
            for k in [8, 16, 32, 64, 128]:
                cid = add(rows, cid, "ReductionSelection", "TopK", dtype, {"d0": d0, "d1": d1, "k": k})

    # SpatialWindow: expand the kernel/stride/channel regimes around observed p90 failures.
    for dtype in dtypes:
        for batch in [4, 8, 16, 32, 64]:
            for channels in [32, 64, 96, 128, 256]:
                for h, w in [(28, 28), (28, 112), (56, 56), (112, 112)]:
                    for kernel, stride in [(2, 2), (2, 3), (3, 2), (5, 3)]:
                        if len(rows) % 3 == 0:
                            cid = add(rows, cid, "ConvolutionSpatial", "MaxPool2D", dtype, {"batch": batch, "channels": channels, "h": h, "w": w, "kernel": kernel, "stride": stride})

    # DenseGemm: add small/tail and moderate regimes that dominate p90 misses.
    for dtype in dtypes:
        for m, k, n in [(16, 32, 32), (32, 64, 32), (64, 128, 64), (128, 256, 128), (256, 512, 256), (9, 200, 9)]:
            cid = add(rows, cid, "DenseLinearAlgebra", "MatMulV2", dtype, {"m": m, "k": k, "n": n})
            for batch in [1, 4, 16, 64, 128]:
                cid = add(rows, cid, "DenseLinearAlgebra", "BatchMatMulV2", dtype, {"batch": batch, "m": m, "k": k, "n": n})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "major_class", "priority", "op_type", "dtype", "params_json"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {len(rows)} targeted cases -> {out}")


if __name__ == "__main__":
    main()
