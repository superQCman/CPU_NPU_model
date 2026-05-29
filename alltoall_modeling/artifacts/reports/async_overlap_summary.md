# Async Overlap Modeling Validation Summary

- Run time: 2026-05-19T15:05:52
- Validation mode: replay from measured All-to-All validation results
- Input validation CSV: `artifacts/reports/validation_metrics.csv`
- eta: `0.85`
- launch overhead: `3.0 us`
- sync overhead: `8.0 us`
- compute ratios: `0.0, 0.25, 0.5, 1.0, 2.0, 4.0`

## Overall Metrics

- Samples: 1440
- Async total MAPE: 0.466%
- Async total RMSE: 23.078 us
- Relative Error P50/P90/P95: 0.111% / 0.952% / 2.253%
- Mean observed speedup vs sync baseline: 1.316x
- Mean hidden communication fraction: 0.531

## Metrics By Group

| Group | Samples | MAPE(%) | RMSE(us) | P90 Err(%) | Mean Speedup | Hidden Comm Fraction |
|---|---:|---:|---:|---:|---:|---:|
| Group1 (128KB) | 480 | 0.480 | 23.143 | 0.997 | 1.316 | 0.531 |
| Group2 (32KB) | 480 | 0.376 | 21.658 | 0.849 | 1.316 | 0.531 |
| Group3 (unaligned) | 480 | 0.542 | 24.354 | 0.893 | 1.317 | 0.531 |

## Metrics By Compute Ratio

| compute/comm | Samples | MAPE(%) | RMSE(us) | P90 Err(%) | Mean Speedup | Hidden Comm Fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 240 | 0.797 | 30.343 | 2.183 | 0.993 | 0.000 |
| 0.25 | 240 | 0.769 | 30.343 | 2.105 | 1.197 | 0.212 |
| 0.50 | 240 | 0.742 | 30.343 | 2.032 | 1.386 | 0.425 |
| 1.00 | 240 | 0.405 | 19.801 | 0.777 | 1.728 | 0.850 |
| 2.00 | 240 | 0.056 | 4.551 | 0.153 | 1.391 | 0.850 |
| 4.00 | 240 | 0.029 | 4.551 | 0.079 | 1.203 | 0.850 |

## Interpretation

- This report validates the async overlap formula by replaying measured All-to-All latency points across multiple compute-window sizes.
- The reported prediction error is the propagated error from the single All-to-All latency model into the async region model.
- To calibrate `eta` for a real workload, collect `(T_comm, T_compute_window, T_async_region)` from profiler traces and use `derive_eta` in `src/async_overlap_model.py`.
