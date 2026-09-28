import re

with open("memory_access_timings.log", "r") as f:
    text = f.read()

# Parse each [Metrics] block
blocks = text.split("[Metrics]")
total_times = []  # (step, total_ms)

for block in blocks[1:]:  # skip text before first [Metrics]
    # step=N
    step_m = re.search(r"step=(\d+)", block)
    if not step_m:
        continue
    step = int(step_m.group(1))

    # TOTAL wall time: X ms/step
    total_m = re.search(r"TOTAL wall time:\s+([\d.]+)\s+ms/step", block)
    if total_m:
        total_times.append((step, float(total_m.group(1))))

# Average per step
from collections import defaultdict
by_step = defaultdict(list)
for step, t in total_times:
    by_step[step].append(t)

print(f"{'Step':<8s} {'Count':>6s} {'Avg(ms)':>10s} {'Min(ms)':>10s} {'Max(ms)':>10s} {'Std(ms)':>10s}")
print("-" * 58)
all_vals = []
for step in sorted(by_step):
    vals = by_step[step]
    all_vals.extend(vals)
    avg = sum(vals) / len(vals)
    print(f"{step:<8d} {len(vals):>6d} {avg:>10.2f} {min(vals):>10.2f} {max(vals):>10.2f} {(sum((v-avg)**2 for v in vals)/len(vals))**0.5:>10.2f}")

# Overall
overall_avg = sum(all_vals) / len(all_vals)
print("-" * 58)
print(f"{'Overall':<8s} {len(all_vals):>6d} {overall_avg:>10.2f} {min(all_vals):>10.2f} {max(all_vals):>10.2f}")

# Steady-state (skip step 1-2 warmup)
steady = [v for s, v in total_times if s >= 3]
steady_avg = sum(steady) / len(steady)
print(f"\nSteady-state (step>=3): {len(steady)} measurements, avg = {steady_avg:.2f} ms")
print(f"Steady-state (step>=5): {len([v for s,v in total_times if s>=5])} measurements, avg = {sum(v for s,v in total_times if s>=5)/max(len([v for s,v in total_times if s>=5]),1):.2f} ms")
