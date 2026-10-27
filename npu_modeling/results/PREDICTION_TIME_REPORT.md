# Prediction Timing Report

Timing excludes CSV input loading. It measures artifact loading, static feature construction, and batched model prediction on the delivered stable holdout rows.

| 指标 | 数值（秒） |
|---|---:|
| 模型加载耗时 | 1.7504 |
| 静态特征构建耗时 | 0.5943 |
| 批量预测耗时 | 0.4863 |
| 静态特征构建和批量预测耗时 | 1.0806 |
| 总耗时 | 2.8310 |

- 测试样本数：1734
- 重复次数：3
- 汇总方式：中位数。
