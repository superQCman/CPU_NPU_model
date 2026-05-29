# DLRM End-to-End NPU Validation

- Source profiles: `input_data/dlrm_profiles/results_20260426_135144`
- Cases evaluated: `1407`
- DLRM profiles are not used in model training.
- CPU, communication, synchronization, transfer, and unmodeled NPU time remain measured.
- Reported NPU error replaces modeled NPU compute operator invocations with model predictions.

## Replacement Metrics
| metric                                  |   count |       mean |         p50 |          p90 |
|:----------------------------------------|--------:|-----------:|------------:|-------------:|
| npu_compute_replace_modeled_ape_percent |    1407 | 6.75691    | 6.25027     | 13.494       |
| stage_replace_modeled_ape_percent       |    1407 | 0.00044187 | 0.000367075 |  0.000885791 |

## Notes
- Operator-level and modeled-operator-sum diagnostics are retained as CSV artifacts for engineering analysis but are not part of the delivery acceptance report.
