op_name：算子名（如 Gather、ReduceSum）。
call_count：该算子被调用次数。
total_dur_us：总耗时（dur 累加，单位是 profile 里的 us）。
avg_dur_us：平均单次耗时 = total_dur_us / call_count。
min_dur_us：最短单次耗时。
max_dur_us：最长单次耗时。
avg_actual_threads：平均“实际参与计算”的线程数（活跃子线程 + 主线程若参与）。
min_actual_threads：单次调用中最少实际参与线程数。
max_actual_threads：单次调用中最多实际参与线程数。
thread_count_dist：实际线程数分布，格式 线程数:次数，如 15:63,16:104。
avg_active_sub_threads：平均活跃子线程数（num_run > 0 的子线程）。
main_thread_used_count：主线程实际参与计算（main_Run > 0）的次数。
main_thread_used_pct：主线程参与比例 = main_thread_used_count / call_count * 100%。
avg_main_run：主线程 Run 字段平均值（脚本按原值直接平均）。
avg_main_wait：主线程 Wait 字段平均值。
avg_main_dist：主线程 Distribution 字段平均值。
avg_total_sub_runs：所有子线程 num_run 总和的平均值（每次调用先求和再取平均）。
main_cores：该算子运行时主线程出现过的 CPU 核集合（| 分隔）。
sub_cores：该算子运行时子线程出现过的 CPU 核集合（| 分隔）。

如果要分析 `features_extensible_3_1_selected/` 里的 DynamoRIO 指令特征和 `dur_us` 的相关性，可以直接运行：

```bash
python ORT/onnx_operator_analysis/analyze_dynamorio_feature_latency_correlation.py
```

默认会读取 `ORT/features_extensible_3_1_selected/`，并把表格和图输出到
`ORT/onnx_operator_analysis/artifacts/dynamorio_feature_latency_correlation/`。
