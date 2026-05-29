# DLRM Async All-to-All Validation Summary

- Run time: 2026-05-26T16:27:50
- Validation mode: real DLRM Ascend profiler step trace
- Profiler dir: `data/profiler/dlrm_single/ASCEND_PROFILER_OUTPUT`
- Step trace: `step_trace_time.csv`
- Min communication filter: `1000.0 us`
- All-to-All start markers: `80`
- All-to-All wait markers: `80`

## Observed DLRM Overlap

- Steps used: 70
- Mean communication: 48817.287 us
- Mean overlappable compute: 171.177 us
- Mean overlapped time: 23.617 us
- Observed eta mean/median: 0.1380 / 0.1347
- Observed eta P10/P90: 0.1317 / 0.1386
- Mean hidden fraction of communication: 0.000488
- Mean speedup over sync comm+compute region: 1.000487x

## Fitted Eta Model Check

- Model eta: 0.1347 (median observed eta)
- Async-region MAPE: 0.002%
- Async-region RMSE: 4.397 us
- Async-region P90 relative error: 0.002%

## Interpretation

- DLRM does launch All-to-All asynchronously and runs bottom MLP before `wait`, so it is a valid real workload for async hiding validation.
- In this profiled case, communication is tens of milliseconds while the bottom MLP compute window is about hundreds of microseconds, so only a tiny fraction of communication can be hidden.
- The calibrated eta is low in this trace, meaning only a small part of the available bottom-MLP window overlaps the communication timeline.
- The hidden fraction of total communication is even lower because `min(T_comm, T_compute)` is dominated by the short compute window.
