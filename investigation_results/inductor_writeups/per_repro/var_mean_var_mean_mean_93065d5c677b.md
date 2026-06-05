# var_mean_var_mean_mean_93065d5c677b

## Summary
- **Model**: Dual BN ReLU spatial mean
- **Pattern**: Two per-channel var_mean reductions over [128, 2240, 7, 7], running-stat copy_, affine, branch sum, ReLU, spatial mean
- **Oracle**: 36.74 us
- **Compile (CDT)**: 30.66 us
- **Ratio**: 0.834x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is faster than the oracle (ratio < 1.0). The oracle is suboptimal for this workload.

## Status: bad_oracle
No action needed. Inductor already outperforms this oracle.
