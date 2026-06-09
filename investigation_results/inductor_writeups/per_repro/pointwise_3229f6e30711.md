# pointwise_3229f6e30711 — oracle_dual_dropout_select

## Status: AT_FLOOR

## Benchmark Results
- Oracle: 76.48 us
- Compiled: 77.02 us
- Ratio: 1.007x (within noise)

## Summary
Inductor generates code that is essentially identical in performance to the oracle for this dual dropout select pattern. The gap is within measurement noise (0.7%). No actionable improvement available.

## Root Cause
N/A — at floor.

## Config Exploration
Not needed — already at floor.
