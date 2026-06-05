# pointwise_552801b1933a — Seeded Dropout Add

## Summary
- **Repro**: (seeded dropout + add pattern)
- **Oracle**: oracle_seeded_dropout_add.py
- **Ratio**: 0.999x (oracle 31.58us vs compile 31.55us)
- **Status**: AT_FLOOR (compile matches oracle)

## Benchmark Result
The ratio of 0.999x is essentially identical performance. The compiled code perfectly
matches the oracle. No investigation needed.
