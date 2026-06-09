# pointwise_59d71b256016 — GELU Scaled

## Summary
- **Repro**: (gelu + scale pattern)
- **Oracle**: oracle_gelu_scaled.py
- **Ratio**: 0.994x (oracle 10.18us vs compile 10.11us)
- **Status**: AT_FLOOR (compile matches oracle)

## Benchmark Result
The ratio of 0.994x is essentially identical performance. The compiled code perfectly
matches the oracle. No investigation needed.
