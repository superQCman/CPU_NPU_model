# CPU/NPU/All-to-All End-to-End Performance Modeling Delivery

This repository collects the CPU, NPU, All-to-All, and fused end-to-end modeling deliverables under one top-level project.

## Directory Overview

| Directory | Purpose |
|---|---|
| `CPU_model_stable/` | CPU-side DLRM graph artifacts, static feature generation, CPU model training/inference code, and CPU modeling documentation. |
| `npu_modeling/` | NPU semantic-operator physical memory model, hardware descriptor, training/inference utilities, and model reports. |
| `alltoall_modeling/` | 8-NPU All-to-All communication latency and asynchronous overlap model artifacts. |
| `cpu_npu_e2e/` | Standalone fusion layer that routes operators to CPU/NPU models, inserts synthetic All-to-All communication, schedules the DAG, and emits end-to-end latency predictions. |

## Fused E2E Model

The final model lives in `cpu_npu_e2e/`. It consumes:

- DLRM `op_shapes.csv` graph/shape artifacts from `CPU_model_stable/`.
- Formal CPU predictions from a `CPU_model_stable` `model_bundles/*.pkl` bundle, or a CPU prediction CSV.
- NPU operator predictions from `npu_modeling/results/physical_memory_models.joblib.gz`.
- All-to-All communication predictions from `alltoall_modeling/artifacts/models/*.json`.

Strict CPU/NPU/All-to-All E2E run:

```bash
cd /data/qc/CPU_NPU/cpu_npu_e2e
python -m cpu_npu_e2e.run_fused_e2e \
  --cases-config configs/e2e_test_cases.yaml \
  --cpu-model-bundle /data/qc/dlrm/CPU_model_stable/outputs_2/outputs/cache_residency_worker_share_v3_full_s920x20_20260505/model_bundles/joint_6d_tuple_source_100_whitebox_family_prior_plus_physics_guarded_rf_v3.pkl \
  --no-cpu-static-fallback \
  --output-dir outputs/e2e_cases_strict_cpu
```

Expected strict-mode checks:

- `summary.json::cpu_static_fallback_used=false`
- all CPU-routed rows in `fused_node_predictions.csv` use `prediction_source=cpu_model_bundle_full_graph_76`
- `routing_coverage.csv`, `alltoall_region_predictions.csv`, and `fused_e2e_combo_metrics.csv` are generated

## Documentation

Primary E2E documentation:

- `cpu_npu_e2e/README.md`
- `cpu_npu_e2e/docs_delivery/USER_GUIDE.md`
- `cpu_npu_e2e/docs_delivery/INTERNAL_PRINCIPLES.md`

Original module documentation is kept inside each subdirectory.

## Artifact Note

GitHub normal Git rejects files larger than 100 MB. The original local NPU model artifact:

```text
npu_modeling/results/physical_memory_models.joblib
```

is intentionally ignored by Git. The repository tracks the compressed equivalent:

```text
npu_modeling/results/physical_memory_models.joblib.gz
```

`cpu_npu_e2e` can load either file.

