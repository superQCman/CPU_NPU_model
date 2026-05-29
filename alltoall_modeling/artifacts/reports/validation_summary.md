# A2A Modeling Pipeline Summary

- Run time: 2026-04-17T16:19:37
- Split mode: `size_holdout`
- Train data: `artifacts/reports/split_data/train_data_size_holdout.txt`
- Test data: `artifacts/reports/split_data/test_data_size_holdout.txt`

## Fit Metrics (Latency)

| Group | MAPE(%) | RMSE(us) |
|---|---:|---:|
| Group1 (128KB) | 0.742 | 27.774 |
| Group2 (32KB) | 0.687 | 30.696 |
| Group3 (unaligned) | 0.650 | 27.626 |

## Train/Test Leakage Check

- Unique-size overlap: **0** (size-disjoint)

## Coverage Summary

| Dataset | Group | Samples | Unique Sizes | Min Size | Max Size | log2 Bins |
|---|---|---:|---:|---:|---:|---:|
| train | Group1 (128KB) | 4505 | 4505 | 262144 | 2147483648 | 14 |
| train | Group2 (32KB) | 11059 | 11059 | 65536 | 2146467840 | 15 |
| train | Group3 (unaligned) | 624 | 624 | 1052672 | 816844800 | 10 |
| test | Group1 (128KB) | 1128 | 1127 | 131072 | 2146435072 | 13 |
| test | Group2 (32KB) | 2765 | 2765 | 32768 | 2143322112 | 15 |
| test | Group3 (unaligned) | 157 | 157 | 1024 | 817893376 | 9 |

## Validation Metrics

- Overall MAPE: 0.712%
- Overall RMSE: 30.990 us
- Relative Error P50/P90/P95: 0.345% / 1.596% / 3.078%
