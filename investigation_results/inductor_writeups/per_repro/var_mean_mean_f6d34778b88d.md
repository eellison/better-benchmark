# var_mean_mean_f6d34778b88d

## Summary
- **Model**: Dropout LayerNorm spatial mean
- **Pattern**: Dropout + LayerNorm + spatial mean (stochastic ops)
- **Oracle**: 46.82 us
- **Compile (CDT)**: 38.43 us
- **Ratio**: 0.821x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is faster than the oracle (ratio < 1.0). The oracle is suboptimal for this workload, possibly due to the stochastic dropout making the oracle's kernel less efficient than Inductor's approach.

## Status: bad_oracle
No action needed. Inductor already outperforms this oracle.
