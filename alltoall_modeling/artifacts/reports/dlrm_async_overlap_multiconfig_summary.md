# DLRM Multi-Config Async All-to-All Validation Summary

- Run time: 2026-05-26T16:28:03
- Validation mode: real DLRM Ascend profiler step traces across sweep configurations
- Sweep root: `data/profiler/dlrm_sweep`
- Traces found: 1452
- Min communication filter: `1000.0 us`
- Cases used: 1244
- Steps used: 1244

## Observed DLRM Overlap

- Mean communication: 1804397.306 us
- Mean overlappable compute proxy: 255.304 us
- Mean bounded overlapped time: 160.083 us
- Observed eta mean/median: 0.6347 / 1.0000
- Observed eta P10/P90: 0.2306 / 1.0000
- Mean bounded hidden fraction of communication: 0.002451
- Mean bounded speedup over sync comm+compute region: 1.002450x

## Fitted Eta Model Check

- Model eta: 1.0000 (median clipped observed eta)
- Bounded async-region MAPE: 0.027%
- Bounded async-region RMSE: 137.261 us
- Bounded async-region P90 relative error: 0.028%

## By Variant

| Variant | Cases | Mean Comm (us) | Mean Compute Proxy (us) | Median Eta | Bounded Hidden Fraction | Bounded Speedup |
|---|---:|---:|---:|---:|---:|---:|
| case_10 | 311 | 2102387.2 | 269.1 | 0.2593 | 0.002000 | 1.001995x |
| case_7 | 309 | 1624734.4 | 241.5 | 1.0000 | 0.001719 | 1.001719x |
| case_8 | 312 | 1896170.4 | 268.8 | 1.0000 | 0.003720 | 1.003720x |
| case_9 | 312 | 1593524.7 | 241.7 | 0.4536 | 0.002355 | 1.002355x |

## Interpretation

- The 1452-case DLRM sweep confirms that the async All-to-All path was measured across a wide parameter grid, not just one hand-picked run.
- Communication is still orders of magnitude larger than the measured compute proxy, so the model-bounded hidden fraction remains small even when eta clips to 1.0.
- The report uses bounded overlap for final delivery because it matches the model formula `min(Communication, Computing)` and gives a conservative, physically interpretable hidden-communication estimate.
