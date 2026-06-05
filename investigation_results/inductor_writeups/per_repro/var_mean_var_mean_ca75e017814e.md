# var_mean_var_mean_ca75e017814e

## Summary
- **Model**: Inception BN+pool
- **Pattern**: Two per-channel var_mean reductions, running-stat copy_, affine, avg_pool2d
- **Oracle**: 89.06 us
- **Compile (CDT)**: 91.90 us
- **Ratio**: 1.032x (AT_FLOOR)
- **Classification**: AT_FLOOR

## Benchmark Results
Gap is below the 1.05x threshold. Inductor is effectively at parity with the oracle.

## Status: at_floor
No action needed. The 3.2% delta is within measurement noise / at the performance floor.
