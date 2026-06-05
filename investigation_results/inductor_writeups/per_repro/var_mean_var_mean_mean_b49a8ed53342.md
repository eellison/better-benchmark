# var_mean_var_mean_mean_b49a8ed53342

## Summary
- **Model**: Dual BN training spatial mean
- **Pattern**: Two per-channel var_mean reductions over [128, 1408, 7, 7], running-stat copy_, affine, spatial mean
- **Oracle**: 55.04 us
- **Compile (CDT)**: 43.90 us
- **Ratio**: 0.798x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is faster than the oracle (ratio < 1.0). The oracle is suboptimal for this workload.

## Status: bad_oracle
No action needed. Inductor already outperforms this oracle.
