# var_mean_mean_935376ba22dc

## Summary
- **Model**: BN centered spatial mean
- **Pattern**: Per-channel var_mean, running-stat copy_, affine, centered subtraction, spatial mean
- **Oracle**: 31.65 us
- **Compile (CDT)**: 29.50 us
- **Ratio**: 0.932x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is slightly faster than the oracle (ratio < 1.0). No gap exists.

## Status: bad_oracle
No action needed. Inductor already outperforms this oracle.
