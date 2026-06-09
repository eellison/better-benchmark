# pointwise_7e1a654d52d5 — oracle_glu

## Summary
**Status**: AT_FLOOR (ratio=1.018x)

## Benchmark Results
- Oracle: 8.70 us
- Compile: 8.86 us
- Ratio: 1.018x

## Analysis
Inductor is within noise of the oracle (1.8% difference). The workload computes
GLU (Gated Linear Unit) activation on shape [8, 2048, 92]. Inductor's fusion
already handles this optimally.

No investigation needed — performance is at floor.

## Classification
AT_FLOOR — no meaningful gap exists.
