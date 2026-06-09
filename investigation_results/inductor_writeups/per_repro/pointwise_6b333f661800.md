# pointwise_6b333f661800 — QKV Split Layout

## Summary
- **Ratio**: 1.000x (AT_FLOOR)
- **Classification**: BANDWIDTH_BOUND
- **Status**: AT_FLOOR — Inductor matches the oracle

## Benchmark Results
- Oracle: 2.75us
- Compile: 2.75us

## Analysis

Identical performance. This is a very small workload (QKV split producing three `[1, 12, 64, 64]` outputs) where kernel launch overhead dominates. Both approaches achieve the same timing at 2.75us, which is near the minimum kernel launch latency.

## Conclusion

No action needed. Inductor already achieves optimal performance for this QKV split pattern. The workload is launch-bound rather than compute or bandwidth bound.
