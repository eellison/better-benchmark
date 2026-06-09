# pointwise_7fe0a1e563a5 — oracle_relu_dropout_compare

## Summary
**Status**: AT_FLOOR (ratio=1.002x)

## Benchmark Results
- Oracle: 19.23 us
- Compile: 19.26 us
- Ratio: 1.002x

## Analysis
Inductor is within noise of the oracle (0.2% difference). The workload computes
ReLU + dropout with a comparison mask. Inductor already fuses these stochastic
operations optimally.

No investigation needed — performance is at floor.

## Classification
AT_FLOOR — no meaningful gap exists.
