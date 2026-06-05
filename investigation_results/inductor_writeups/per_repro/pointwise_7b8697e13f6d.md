# pointwise_7b8697e13f6d — oracle_seeded_dropout_add

## Summary
**Status**: AT_FLOOR (ratio=1.006x)

## Benchmark Results
- Oracle: 31.42 us
- Compile: 31.62 us
- Ratio: 1.006x

## Analysis
Inductor is within noise of the oracle (0.6% difference). The workload is a seeded
dropout followed by an add operation. Inductor already fuses these into a single
kernel with proper stochastic handling.

No investigation needed — performance is at floor.

## Classification
AT_FLOOR — no meaningful gap exists.
