# pointwise_37c9e37865c1 — AT_FLOOR (ratio 1.004x)

## Summary
Oracle: `oracle_index_put.py` — scatter-reduce pattern with `index_put(accumulate=True)`
Benchmark: oracle=7.97us, compile=8.0us, ratio=1.004x

## Classification: SCATTER_REDUCE (already at floor)

## Root Cause
The oracle fuses an eq-mask, unsqueeze/broadcast where, zero-fill, and duplicate-preserving
`index_put(accumulate=True)` into a direct scatter-add kernel with Triton atomics.

Inductor already matches this performance (ratio 1.004x). The codegen path for scatter
operations with accumulate=True is handling this pattern efficiently.

## Kernel Count
- Inductor: 1 kernel (fused pointwise + scatter)
- Oracle: 1 kernel

## Config Exploration
No config changes needed — already at performance floor.

## Conclusion
No action needed. Inductor matches oracle performance.
