# var_mean_var_mean_var_mean_0407b3e7c77f

## Summary
- **Model**: Inception BN+ReLU+AvgPool (triple var_mean pattern)
- **Pattern**: Three training-BN var_mean reductions over [32, 192, 35, 35] with running-stat copy_, affine, ReLU, avg_pool2d
- **Oracle**: 224.93 us
- **Compile (CDT)**: 222.98 us
- **Ratio**: 0.991x (AT_FLOOR)
- **Classification**: AT_FLOOR

## Benchmark Results
Inductor matches or beats the oracle. No gap to investigate.

## Status: at_floor
No action needed. Inductor already achieves optimal performance for this workload.
