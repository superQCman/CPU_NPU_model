"""
Extract thread usage statistics for CPU operators from ORT profiling JSON.

For each CPU operator node, extracts:
  - main_thread: core, Distribution, DistributionEnqueue, Run, Wait, WaitRevoke
  - sub_threads: number of sub-threads, per-thread num_run and core
  - Aggregated stats per op_name
"""

import argparse
import json
import csv
import sys
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────
# 1. Load JSON
# ──────────────────────────────────────────────
def load_profile(json_path: str) -> list:
    print(f"Loading {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Total events: {len(data)}")
    return data


# ──────────────────────────────────────────────
# 2. Parse CPU node events
# ──────────────────────────────────────────────
def parse_cpu_nodes(events: list) -> list:
    """
    Filter Node events whose provider == CPUExecutionProvider
    and extract thread scheduling stats.

    Some single-thread ORT profiles emit `thread_scheduling_stats` as an empty
    string instead of a populated dict. In that case we still keep the node so
    downstream datasets retain `dur_us` and shape/size metadata.
    """
    records = []
    skipped = 0
    fallback_without_stats = 0

    for ev in events:
        if ev.get("cat") != "Node":
            continue

        args = ev.get("args", {})
        provider = args.get("provider", "")
        if provider != "CPUExecutionProvider":
            continue

        raw_stats = args.get("thread_scheduling_stats", {})
        if isinstance(raw_stats, dict):
            stats = raw_stats
        else:
            stats = {}

        if stats:
            main = stats.get("main_thread", {})
            sub_threads_raw = stats.get("sub_threads", {})

            # ── main thread fields ──
            main_thread_id    = main.get("thread_id", "")
            main_pool_name    = main.get("thread_pool_name", "")
            main_core         = main.get("core", -1)
            main_block_size   = main.get("block_size", [])
            main_dist         = main.get("Distribution", 0)
            main_dist_enqueue = main.get("DistributionEnqueue", 0)
            main_run          = main.get("Run", 0)
            main_wait         = main.get("Wait", 0)
            main_wait_revoke  = main.get("WaitRevoke", 0)
            main_used         = main_run > 0  # main thread actually computed

            # ── sub-thread fields ──
            num_sub_threads   = len(sub_threads_raw)
            total_sub_runs    = sum(v.get("num_run", 0) for v in sub_threads_raw.values())
            sub_cores         = sorted({v.get("core", -1) for v in sub_threads_raw.values() if v.get("core", -1) >= 0})
            sub_thread_ids    = list(sub_threads_raw.keys())

            # compute per-sub-thread max num_run (useful for utilisation analysis)
            sub_max_runs = max((v.get("num_run", 0) for v in sub_threads_raw.values()), default=0)

            # ── actual active threads ──
            # sub-threads that actually ran at least one task
            active_sub_threads = sum(1 for v in sub_threads_raw.values() if v.get("num_run", 0) > 0)
            # total: active sub-threads + main thread (if it ran)
            actual_threads_used = active_sub_threads + (1 if main_run > 0 else 0)
        else:
            fallback_without_stats += 1
            main_thread_id = ""
            main_pool_name = ""
            main_core = -1
            main_block_size = []
            main_dist = 0
            main_dist_enqueue = 0
            main_run = 0
            main_wait = 0
            main_wait_revoke = 0
            num_sub_threads = 0
            total_sub_runs = 0
            sub_cores = []
            sub_thread_ids = []
            sub_max_runs = 0
            active_sub_threads = 0
            # Keep single-thread runs observable in downstream features even
            # when ORT omitted the scheduling payload.
            main_used = ev.get("dur", 0) > 0
            actual_threads_used = 1 if main_used else 0

        records.append({
            # identification
            "node_name":         ev.get("name", ""),
            "op_name":           args.get("op_name", ""),
            "node_index":        args.get("node_index", ""),
            "provider":          provider,
            # timing
            "ts_us":             ev.get("ts", 0),
            "dur_us":            ev.get("dur", 0),
            # main thread
            "main_thread_id":    main_thread_id,
            "main_pool_name":    main_pool_name,
            "main_core":         main_core,
            "main_block_size":   str(main_block_size),
            "main_Distribution": main_dist,
            "main_DistributionEnqueue": main_dist_enqueue,
            "main_Run":          main_run,
            "main_Wait":         main_wait,
            "main_WaitRevoke":   main_wait_revoke,
            "main_thread_used":  main_used,
            # sub threads
            "num_sub_threads":   num_sub_threads,
            "active_sub_threads": active_sub_threads,
            "actual_threads_used": actual_threads_used,
            "total_sub_runs":    total_sub_runs,
            "sub_max_runs":      sub_max_runs,
            "sub_thread_ids":    "|".join(sub_thread_ids),
            "sub_cores":         "|".join(str(c) for c in sub_cores),
            # shape info
            "input_type_shape":  str(args.get("input_type_shape", [])),
            "output_type_shape": str(args.get("output_type_shape", [])),
            "output_size":       args.get("output_size", ""),
            "activation_size":   args.get("activation_size", ""),
            "parameter_size":    args.get("parameter_size", ""),
        })

    print(f"  CPU Node events extracted : {len(records)}")
    print(f"  Fallback (empty thread stats) : {fallback_without_stats}")
    print(f"  Skipped (no usable data)      : {skipped}")
    return records


# ──────────────────────────────────────────────
# 3. Aggregated stats per op_name
# ──────────────────────────────────────────────
def aggregate_by_op(records: list) -> list:
    """
    Per op_name (e.g. Gemm, Gather, ReduceSum …):
      - call_count
      - total / avg / min / max duration
      - times main_thread was active
      - total / avg sub-thread runs
      - unique cores used
    """
    agg = defaultdict(lambda: {
        "call_count": 0,
        "total_dur_us": 0,
        "min_dur_us": float("inf"),
        "max_dur_us": 0,
        "main_thread_used_count": 0,
        "total_main_run": 0,
        "total_main_wait": 0,
        "total_main_dist": 0,
        "total_active_sub_threads": 0,
        "total_actual_threads_used": 0,
        "min_actual_threads": float("inf"),
        "max_actual_threads": 0,
        "thread_count_dist": defaultdict(int),  # {n_threads: count}
        "total_sub_runs": 0,
        "sub_max_runs_sum": 0,
        "main_cores": set(),
        "sub_cores": set(),
    })

    for r in records:
        op = r["op_name"]
        a  = agg[op]
        a["call_count"]             += 1
        a["total_dur_us"]           += r["dur_us"]
        a["min_dur_us"]              = min(a["min_dur_us"], r["dur_us"])
        a["max_dur_us"]              = max(a["max_dur_us"], r["dur_us"])
        a["main_thread_used_count"] += int(r["main_thread_used"])
        a["total_main_run"]         += r["main_Run"]
        a["total_main_wait"]        += r["main_Wait"]
        a["total_main_dist"]        += r["main_Distribution"]
        a["total_active_sub_threads"] += r["active_sub_threads"]
        a["total_actual_threads_used"] += r["actual_threads_used"]
        a["min_actual_threads"]      = min(a["min_actual_threads"], r["actual_threads_used"])
        a["max_actual_threads"]      = max(a["max_actual_threads"], r["actual_threads_used"])
        a["thread_count_dist"][r["actual_threads_used"]] += 1
        a["total_sub_runs"]         += r["total_sub_runs"]
        a["sub_max_runs_sum"]       += r["sub_max_runs"]
        if r["main_core"] >= 0:
            a["main_cores"].add(r["main_core"])
        for c in r["sub_cores"].split("|"):
            if c:
                a["sub_cores"].add(int(c))

    rows = []
    for op, a in sorted(agg.items(), key=lambda x: -x[1]["total_dur_us"]):
        cnt = a["call_count"]
        # thread count distribution as string: e.g. "3:100,4:4"
        dist_str = ",".join(
            f"{k}:{v}" for k, v in sorted(a["thread_count_dist"].items())
        )
        rows.append({
            "op_name":               op,
            "call_count":            cnt,
            "total_dur_us":          a["total_dur_us"],
            "avg_dur_us":            round(a["total_dur_us"] / cnt, 2) if cnt else 0,
            "min_dur_us":            a["min_dur_us"] if a["min_dur_us"] != float("inf") else 0,
            "max_dur_us":            a["max_dur_us"],
            # ── actual thread usage ──
            "avg_actual_threads":    round(a["total_actual_threads_used"] / cnt, 2) if cnt else 0,
            "min_actual_threads":    a["min_actual_threads"] if a["min_actual_threads"] != float("inf") else 0,
            "max_actual_threads":    a["max_actual_threads"],
            "thread_count_dist":     dist_str,
            "avg_active_sub_threads": round(a["total_active_sub_threads"] / cnt, 2) if cnt else 0,
            # ── main thread ──
            "main_thread_used_count": a["main_thread_used_count"],
            "main_thread_used_pct":  round(100 * a["main_thread_used_count"] / cnt, 1) if cnt else 0,
            "avg_main_run":          round(a["total_main_run"] / cnt, 2) if cnt else 0,
            "avg_main_wait":         round(a["total_main_wait"] / cnt, 2) if cnt else 0,
            "avg_main_dist":         round(a["total_main_dist"] / cnt, 2) if cnt else 0,
            "avg_total_sub_runs":    round(a["total_sub_runs"] / cnt, 2) if cnt else 0,
            "main_cores":            "|".join(str(c) for c in sorted(a["main_cores"])),
            "sub_cores":             "|".join(str(c) for c in sorted(a["sub_cores"])),
        })
    return rows


# ──────────────────────────────────────────────
# 4. Parallel execution detection
# ──────────────────────────────────────────────
def find_parallel_ops(events: list) -> tuple:
    """
    Detect operators that actually executed in parallel (overlapping time windows).

    Algorithm: O(n log n) sweep-line over boundary events (start / end).
    Aggregates at op_name level to avoid O(n²) pair explosion.

    Returns:
      pair_rows   : list of dict — per (op_a, provider_a, op_b, provider_b) pair type,
                    how many times they overlapped and total overlap duration
      group_rows  : list of dict — one row per distinct concurrency-state change,
                    showing all concurrently active op_names at that moment
    """
    # Collect all Node events with timing
    nodes = []
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        ts  = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        if dur <= 0:
            continue
        args = ev.get("args", {})
        nodes.append({
            "op_name":  args.get("op_name", "unknown"),
            "provider": args.get("provider", ""),
            "ts":       ts,
            "end":      ts + dur,
        })

    if not nodes:
        return [], []

    # ── Build boundary event list ────────────────────────────────────────────
    # Each entry: (time, event_type, node_index)
    #   event_type: 0 = start, 1 = end  (start < end for tie-breaking)
    boundaries = []
    for i, n in enumerate(nodes):
        boundaries.append((n["ts"],  0, i))   # start
        boundaries.append((n["end"], 1, i))   # end
    boundaries.sort()

    # ── Sweep line ────────────────────────────────────────────────────────────
    # active_set: set of node indices currently running
    active_set = set()
    # pair_agg: {(op_a, prov_a, op_b, prov_b): {"count": int, "overlap_us": int}}
    pair_agg = defaultdict(lambda: {"count": 0, "overlap_us": 0})
    # group_rows: one row per distinct state change where ≥2 ops are active
    group_rows = []
    prev_time = None

    for time_pt, etype, idx in boundaries:
        # Before updating, record the current active-set state (if ≥2 ops, it's parallel)
        if active_set and prev_time is not None and time_pt > prev_time:
            interval_dur = time_pt - prev_time
            active_list  = [nodes[i] for i in active_set]
            if len(active_list) >= 2:
                # Accumulate pair overlap durations
                sorted_active = sorted(active_list, key=lambda x: x["op_name"])
                for ai in range(len(sorted_active)):
                    for bi in range(ai + 1, len(sorted_active)):
                        a = sorted_active[ai]
                        b = sorted_active[bi]
                        key = (a["op_name"], a["provider"],
                               b["op_name"], b["provider"])
                        pair_agg[key]["count"]      += 1
                        pair_agg[key]["overlap_us"] += interval_dur

                # Record distinct group state (deduplicated by op-name set + count)
                op_key = tuple(sorted(n["op_name"] for n in active_list))
                group_rows.append({
                    "concurrent_count":  len(active_list),
                    "interval_start_us": prev_time,
                    "interval_end_us":   time_pt,
                    "interval_dur_us":   interval_dur,
                    "op_names":    "|".join(n["op_name"]  for n in active_list),
                    "providers":   "|".join(n["provider"] for n in active_list),
                })

        prev_time = time_pt
        if etype == 0:    # start
            active_set.add(idx)
        else:             # end
            active_set.discard(idx)

    # ── Build pair_rows from pair_agg ────────────────────────────────────────
    pair_rows = []
    for (op_a, prov_a, op_b, prov_b), stats in pair_agg.items():
        pair_rows.append({
            "op_a":          op_a,
            "provider_a":    prov_a,
            "op_b":          op_b,
            "provider_b":    prov_b,
            "overlap_count": stats["count"],
            "total_overlap_us": stats["overlap_us"],
        })
    pair_rows.sort(key=lambda x: -x["total_overlap_us"])

    # Deduplicate group_rows by (concurrent_count, op_names), keep longest interval
    best_groups: dict = {}
    for g in group_rows:
        key = (g["concurrent_count"], g["op_names"])
        if key not in best_groups or g["interval_dur_us"] > best_groups[key]["interval_dur_us"]:
            best_groups[key] = g
    group_rows = sorted(best_groups.values(),
                        key=lambda x: (-x["concurrent_count"], -x["interval_dur_us"]))

    return pair_rows, group_rows


def summarize_parallel(pair_rows: list, group_rows: list):
    """Print a concise parallel execution summary."""
    if not pair_rows:
        print("\n[PARALLEL] No overlapping operator executions detected.")
        print("  → inter_op_num_threads is likely 1 (sequential execution)")
        return

    max_concurrent = max((g["concurrent_count"] for g in group_rows), default=1)

    print(f"\n{'='*82}")
    print(f"[PARALLEL] Distinct parallel op-pair types : {len(pair_rows)}")
    print(f"[PARALLEL] Max concurrency observed         : {max_concurrent} ops at once")

    print(f"\nTop 20 parallel op pairs (by total overlap duration):")
    print(f"  {'op_a':<28} {'prov_a':<8} {'op_b':<28} {'prov_b':<8} "
          f"{'overlaps':>8} {'total_us':>10}")
    print(f"  {'-'*84}")
    for r in pair_rows[:20]:
        print(f"  {r['op_a']:<28} {r['provider_a']:<8} {r['op_b']:<28} "
              f"{r['provider_b']:<8} {r['overlap_count']:>8} {r['total_overlap_us']:>10}")

    print(f"\nTop 10 highest-concurrency groups:")
    print(f"  {'#ops':>5}  {'dur_us':>8}  op_names")
    print(f"  {'-'*72}")
    for g in group_rows[:10]:
        ops_short = g["op_names"][:60]
        print(f"  {g['concurrent_count']:>5}  {g['interval_dur_us']:>8}  {ops_short}")
    print(f"{'='*82}")


# ──────────────────────────────────────────────
# 5. Write CSVs
# ──────────────────────────────────────────────
def write_csv(rows: list, path: str):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write a non-empty placeholder so sweep resume checks can treat the
        # stage as completed even when ORT emitted no thread scheduling stats.
        out_path.write_text("\n", encoding="utf-8")
        print(f"  [WARN] No data rows for {path}; wrote placeholder CSV")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {path}  ({len(rows)} rows)")


# ──────────────────────────────────────────────
# 5. Summary report
# ──────────────────────────────────────────────
def print_summary(agg_rows: list):
    print("\n" + "=" * 90)
    print(f"{'op_name':<28} {'calls':>6} {'total_dur(ms)':>14} {'avg_dur(us)':>11} "
          f"{'avg_threads':>11} {'min':>4} {'max':>4} {'thread_count_dist':<25}")
    print("-" * 90)
    for r in agg_rows[:30]:  # top 30 by total duration
        print(f"{r['op_name']:<28} {r['call_count']:>6} "
              f"{r['total_dur_us']/1000:>14.2f} {r['avg_dur_us']:>11.2f} "
              f"{r['avg_actual_threads']:>11.2f} {r['min_actual_threads']:>4} "
              f"{r['max_actual_threads']:>4}  {r['thread_count_dist']:<25}")
    if len(agg_rows) > 30:
        print(f"  ... and {len(agg_rows)-30} more op types")
    print("=" * 90)


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Extract CPU thread usage statistics from an ORT profiling JSON."
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        help="Path to ort_cann_profile_*.json. If omitted, auto-select the latest one.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write the derived CSV files into. Defaults to the script directory.",
    )
    args = parser.parse_args()

    # Accept JSON path as CLI arg or discover the latest in current dir
    if args.json_path:
        json_path = args.json_path
    else:
        candidates = sorted(script_dir.glob("ort_cann_profile_*.json"))
        if not candidates:
            # fallback: search parent
            candidates = sorted(Path(".").glob("ort_cann_profile_*.json"))
        if not candidates:
            print("ERROR: No ort_cann_profile_*.json found. Pass the path as argument.")
            sys.exit(1)
        json_path = str(candidates[-1])
        print(f"Auto-selected: {json_path}")

    stem = Path(json_path).stem  # e.g. ort_cann_profile_2026-03-01_13-13-14
    out_dir = Path(args.out_dir) if args.out_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_detail   = out_dir / f"{stem}_cpu_thread_detail.csv"
    out_agg      = out_dir / f"{stem}_cpu_thread_aggregated.csv"
    out_par_pair = out_dir / f"{stem}_parallel_pairs.csv"
    out_par_grp  = out_dir / f"{stem}_parallel_groups.csv"

    # Pipeline
    events   = load_profile(json_path)
    records  = parse_cpu_nodes(events)
    agg_rows = aggregate_by_op(records)

    print("\n--- Writing detail CSV ---")
    write_csv(records, str(out_detail))

    print("\n--- Writing aggregated CSV ---")
    write_csv(agg_rows, str(out_agg))

    print_summary(agg_rows)

    # ── Parallel execution analysis ─────────────────────────────────────────
    print("\n--- Analyzing parallel operator execution ---")
    parallel_pairs, group_rows = find_parallel_ops(events)
    summarize_parallel(parallel_pairs, group_rows)

    print("\n--- Writing parallel pairs CSV ---")
    write_csv(parallel_pairs, str(out_par_pair))

    print("\n--- Writing parallel groups CSV ---")
    write_csv(group_rows, str(out_par_grp))

    print(f"\nDone. Output files:")
    print(f"  {out_detail}")
    print(f"  {out_agg}")
    print(f"  {out_par_pair}")
    print(f"  {out_par_grp}")


if __name__ == "__main__":
    main()
