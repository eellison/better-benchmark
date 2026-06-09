# pointwise_226a109e3e14 — oracle_bn_silu_pointwise

## Status: AT_FLOOR

## Benchmark Results
- Oracle: 8.9 us
- Compiled: 9.15 us
- Ratio: 1.029x (within noise, bandwidth bound)

## Summary
Inductor generates code that is within 3% of the oracle for this batch normalization + SiLU pointwise pattern. The gap is within measurement noise and the workload is bandwidth-bound. No actionable improvement available.

## Root Cause
N/A — effectively at floor (bandwidth bound).

## Config Exploration
Not needed — already at floor.
