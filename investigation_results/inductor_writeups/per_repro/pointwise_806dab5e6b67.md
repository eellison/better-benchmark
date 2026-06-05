# pointwise_806dab5e6b67 — oracle_dropout_residual_add

## Summary
**Status**: AT_FLOOR (ratio=0.998x)

## Benchmark Results
- Oracle: 37.63 us
- Compile: 37.57 us
- Ratio: 0.998x

## Analysis
Inductor is within noise of the oracle (0.2% difference). The workload computes
dropout + residual add. Inductor already fuses these stochastic operations
optimally into a single kernel.

No investigation needed — performance is at floor.

## Classification
AT_FLOOR — no meaningful gap exists.
