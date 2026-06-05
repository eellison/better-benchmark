# pointwise_3abc926270f6 — AT_FLOOR (ratio 1.017x)

## Summary
Oracle: `oracle_relu.py` — single relu on float32[128, 768, 1, 1]
Benchmark: oracle=5.79us, compile=5.89us, ratio=1.017x

## Classification: BANDWIDTH_BOUND (at floor)

## Root Cause
This is a single `aten.relu.default` over a materialized `float32[128, 768, 1, 1]` tensor.
The oracle and Inductor both emit equivalent flat Triton load/max/store kernels. There is
no producer, consumer, reduction, scatter, or algebraic dead work beyond the required
output materialization.

## Kernel Count
- Inductor: 1 kernel
- Oracle: 1 kernel

## Config Exploration
No config changes needed — this is bandwidth-bound at the memory floor.

## Conclusion
No action needed. This is a launch-plus-memory floor case.
