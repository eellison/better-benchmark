# var_mean_mean_8d6fc761298a

## Summary
- **Model**: MobileNet BN training + ReLU6 + spatial mean + dropout
- **Pattern**: Per-channel var_mean, running-stat copy_, affine, relu6, spatial mean, dropout (stochastic)
- **Oracle**: 39.58 us
- **Compile (CDT)**: 36.80 us
- **Ratio**: 0.930x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is slightly faster than the oracle (ratio < 1.0). The oracle is suboptimal, possibly due to the stochastic dropout complicating the fused kernel or suboptimal tiling for the shape.

## Status: bad_oracle
No action needed. Inductor already outperforms this oracle.
