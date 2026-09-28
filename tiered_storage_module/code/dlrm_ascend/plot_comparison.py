import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11

N = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
labels = ["1k", "5k", "10k", "50k", "100k", "500k", "1M"]

# Measured total times (ms)
meas = [193.0, 860.6, 1632.1, 3697.5, 4581.3, 4834.2, 4907.3]

# Predicted total times — calibrated model (ms)
pred_cal = [6.1, 367.8, 969.5, 2973.0, 3877.8, 5705.8, 6367.7]

# Predicted total times — default model (ms)
pred_def = [8.1, 71.9, 152.7, 409.2, 524.1, 754.1, 837.6]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ---- Left: absolute values ----
ax1.plot(N, meas, 'ko-', label='Measured', linewidth=2, markersize=7)
ax1.plot(N, pred_cal, 's--', color='#2196F3', label='Predicted (calibrated)', linewidth=2, markersize=7)
ax1.plot(N, pred_def, '^:', color='#999999', label='Predicted (default)', linewidth=1.5, markersize=6)

ax1.set_xscale('log')
ax1.set_xlabel('Embedding rows per table (N)')
ax1.set_ylabel('Total memory time per forward (ms)')
ax1.set_title('Predicted vs Measured (log scale)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(N)
ax1.set_xticklabels(labels)

# ---- Right: error ratio (pred/meas) ----
ratio_cal = [p/m for p, m in zip(pred_cal, meas)]
ratio_def = [p/m for p, m in zip(pred_def, meas)]

ax2.axhline(y=1.0, color='black', linewidth=1, linestyle='-', alpha=0.3)
ax2.plot(N, ratio_cal, 's-', color='#2196F3', label='Calibrated / Measured', linewidth=2, markersize=7)
ax2.plot(N, ratio_def, '^-', color='#999999', label='Default / Measured', linewidth=1.5, markersize=6)
ax2.set_xscale('log')
ax2.set_xlabel('Embedding rows per table (N)')
ax2.set_ylabel('Predicted / Measured')
ax2.set_title('Prediction Accuracy (1.0 = perfect)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(N)
ax2.set_xticklabels(labels)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.show()
