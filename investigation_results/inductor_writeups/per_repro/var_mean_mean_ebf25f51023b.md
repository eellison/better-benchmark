# var_mean_mean_ebf25f51023b

## Summary
- **Model**: MobileNetV3 BN + hardswish + spatial mean
- **Pattern**: Per-channel var_mean over [256, 960, 1, 1], running-stat copy_, affine, hardswish, spatial mean
- **Oracle**: 33.63 us
- **Compile (CDT)**: 30.72 us
- **Ratio**: 0.913x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is faster than the oracle (ratio < 1.0). The oracle is suboptimal for this workload.

## Status: bad_oracle
No action needed. Inductor already outperforms this oracle.
