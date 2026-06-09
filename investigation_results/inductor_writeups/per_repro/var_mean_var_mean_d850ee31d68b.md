# var_mean_var_mean_d850ee31d68b

## Summary
- **Model**: Dual groupnorm stats
- **Pattern**: Two var_mean reductions for GroupNorm statistics
- **Oracle**: 6.88 us
- **Compile (CDT)**: 6.21 us
- **Ratio**: 0.902x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is faster than the oracle (ratio < 1.0). The oracle is suboptimal for this workload.

## Status: bad_oracle
No action needed. Inductor already outperforms this oracle.
