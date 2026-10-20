from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fusion import run_fusion, run_fusion_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU/NPU/All-to-All E2E fused latency prediction.")
    parser.add_argument("--cpu-model-root", type=Path, default=Path("../CPU_model_stable"))
    parser.add_argument("--npu-root", type=Path, default=Path("../npu_modeling"))
    parser.add_argument("--alltoall-root", type=Path, default=Path("../alltoall_modeling"))
    parser.add_argument("--routing", type=Path, default=Path("configs/cpu_npu_op_routing.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default"))
    parser.add_argument("--shape-csv", type=Path, default=None, help="Optional explicit op_shapes.csv. Defaults to the first CPU pipeline artifact.")
    parser.add_argument("--cases-config", type=Path, default=None, help="Optional YAML file with multiple E2E shape cases to run and aggregate.")
    parser.add_argument("--cpu-predictions", type=Path, default=None, help="Optional CPU prediction CSV with op_idx/row_uid and pred_us-like column.")
    parser.add_argument("--cpu-model-bundle", type=Path, default=None, help="Optional CPU_model_stable bundle (.pkl, .pkl.xz, or .pkl.xz.parts.json) for formal full_graph_76 CPU inference.")
    parser.add_argument("--cpu-model-num-threads", type=int, default=None, help="Override CPU static model num_threads. Defaults to parsing case_<id>_run_<num_threads>_<inter_threads> from shape path.")
    parser.add_argument("--cpu-model-inter-threads", type=int, default=None, help="Override CPU static model inter_threads. Defaults to parsing case_<id>_run_<num_threads>_<inter_threads> from shape path.")
    parser.add_argument("--cpu-slots", type=int, default=1)
    parser.add_argument("--npu-slots", type=int, default=1)
    parser.add_argument("--no-cpu-static-fallback", action="store_true", help="Require every CPU-routed node to exist in --cpu-predictions or --cpu-model-bundle output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cases_config and args.shape_csv:
        raise SystemExit("--shape-csv cannot be used together with --cases-config")
    if args.cpu_predictions and args.cpu_model_bundle:
        raise SystemExit("--cpu-predictions cannot be used together with --cpu-model-bundle")

    common_args = {
        "cpu_model_root": args.cpu_model_root.resolve(),
        "npu_root": args.npu_root.resolve(),
        "alltoall_root": args.alltoall_root.resolve(),
        "routing_path": args.routing.resolve(),
        "output_dir": args.output_dir.resolve(),
        "cpu_predictions": args.cpu_predictions.resolve() if args.cpu_predictions else None,
        "cpu_model_bundle": args.cpu_model_bundle.resolve() if args.cpu_model_bundle else None,
        "cpu_model_num_threads": args.cpu_model_num_threads,
        "cpu_model_inter_threads": args.cpu_model_inter_threads,
        "cpu_slots": args.cpu_slots,
        "npu_slots": args.npu_slots,
        "allow_cpu_static_fallback": not args.no_cpu_static_fallback,
    }
    if args.cases_config:
        summary = run_fusion_cases(cases_config=args.cases_config.resolve(), **common_args)
    else:
        summary = run_fusion(shape_csv=args.shape_csv.resolve() if args.shape_csv else None, **common_args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
