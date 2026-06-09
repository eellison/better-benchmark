# pointwise_17498cde41e8 — GELU + scale

## Summary
- **Repro**: pointwise_17498cde41e8
- **Ratio**: 1.001 (oracle 46.69us vs compile 46.72us)
- **Status**: AT_FLOOR

## Result
Inductor matches the oracle performance within noise (0.1% gap).
The compiled GELU + scale kernel is already optimal.
