# pointwise_4e68825d6848 — Pointwise Tuple

## Summary
- **Repro**: (pointwise tuple pattern)
- **Oracle**: oracle_pointwise_tuple.py
- **Ratio**: 1.013x (oracle 4.96us vs compile 5.02us)
- **Status**: AT_FLOOR (within noise margin)

## Benchmark Result
The ratio of 1.013x is within measurement noise (threshold 1.05x). Inductor's generated
code effectively matches the oracle performance. No investigation needed.
