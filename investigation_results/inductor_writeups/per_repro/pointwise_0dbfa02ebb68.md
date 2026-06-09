# pointwise_0dbfa02ebb68 — gelu + pad fusion

## Summary
- **Repro**: pointwise_0dbfa02ebb68
- **Ratio**: 1.004 (oracle 38.02us vs compile 38.18us)
- **Status**: AT_FLOOR

## Result
Inductor matches the oracle performance within noise. No investigation needed.
The compiled kernel is already at parity with the hand-written oracle for this GELU + pad fusion pattern.
