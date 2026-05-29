# NPU Physical Memory Model Report

## Summary
|   stable_rows |   train_rows |   test_rows |   model_count | strategy       |   mape_target_percent |   tail_target_percent | mape_target_met   | p90_target_met   | per_model_target_met   |   models_meeting_target |   models_total |
|--------------:|-------------:|------------:|--------------:|:---------------|----------------------:|----------------------:|:------------------|:-----------------|:-----------------------|------------------------:|---------------:|
|          8666 |        10171 |        1734 |             4 | strict_compact |                    10 |                    20 | True              | True             | True                   |                       4 |              4 |

## Acceptance Metrics
|   sample_count |   mae_us |   mape_1us_floor_percent |   wmape_percent |   p50_1us_floor_percent |   p90_1us_floor_percent | metric_scope           |
|---------------:|---------:|-------------------------:|----------------:|------------------------:|------------------------:|:-----------------------|
|           1734 |  2.88206 |                  8.19086 |         10.3312 |                 5.08197 |                 17.2121 | overall_stable_holdout |

## Residual Experts
| strategy       | status   | reason                          |
|:---------------|:---------|:--------------------------------|
| strict_compact | not_used | one model per final model group |

## Metrics By Model Group
|   sample_count |     mae_us |   mape_1us_floor_percent |   wmape_percent |   p50_1us_floor_percent |   p90_1us_floor_percent | model_group             | model_choice       |   train_rows |   test_rows | mape_target_met   | p90_target_met   | model_target_met   |
|---------------:|-----------:|-------------------------:|----------------:|------------------------:|------------------------:|:------------------------|:-------------------|-------------:|------------:|:------------------|:-----------------|:-------------------|
|             16 | 121.477    |                  3.43592 |        10.2243  |                 1.87203 |                 3.72567 | DataDependent           | extra_trees_smooth |          156 |          16 | True              | True             | True               |
|            567 |   3.80502  |                  7.11843 |        11.0628  |                 4.52741 |                16.2852  | MovementSelectionWindow | random_forest      |         4089 |         567 | True              | True             | True               |
|            352 |   0.812992 |                  7.94678 |         9.16526 |                 4.30351 |                16.0963  | StreamingReduction      | extra_trees        |         2333 |         352 | True              | True             | True               |
|            799 |   0.763744 |                  9.15464 |         9.05574 |                 5.98853 |                19.2637  | StructuredComputeMemory | extra_trees        |         3593 |         799 | True              | True             | True               |

## Notes
- Uses stable-label aggregation before train/test split.
- Accepted strategy uses one model per final model group.
- Hierarchical memory features model L0 as capacity/reuse pressure and L1/L2/HBM as bandwidth lower-bound terms.
- Operator-level diagnostics are retained as CSV artifacts for engineering analysis but are not part of the delivery acceptance report.
