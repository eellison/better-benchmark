# pointwise_8389b1fec310 — oracle_relu_pool_dropout_mask

## Summary
**Status**: AT_FLOOR (ratio=0.983x — compile faster)

## Benchmark Results
- Oracle: 120.93 us
- Compile: 118.88 us
- Ratio: 0.983x

## Analysis
Inductor's compiled output is already faster than the oracle. The workload involves
ReLU + pooling + dropout mask generation. Inductor handles this multi-output
stochastic workload optimally.

No investigation needed — Inductor is already at or above oracle performance.

## Classification
AT_FLOOR — Inductor is already optimal.
