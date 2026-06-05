# pointwise_28fded12b2f4 — oracle_computed_embedding_gather

## Status: AT_FLOOR

## Benchmark Results
- Oracle: 11.81 us
- Compiled: 11.87 us
- Ratio: 1.005x (within noise)

## Summary
Inductor generates code that is essentially identical in performance to the oracle for this computed embedding gather pattern. The gap is within measurement noise (0.5%). No actionable improvement available.

## Root Cause
N/A — at floor.

## Config Exploration
Not needed — already at floor.
