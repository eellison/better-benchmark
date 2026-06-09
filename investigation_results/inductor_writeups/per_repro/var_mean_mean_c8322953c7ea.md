# var_mean_mean_c8322953c7ea

## Summary
- **Model**: BN training + ReLU + spatial mean
- **Pattern**: Per-channel var_mean over [512, 1024, 7, 7](?), running-stat copy_, affine, ReLU, spatial mean
- **Oracle**: 101.28 us
- **Compile (CDT)**: 47.01 us
- **Ratio**: 0.464x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is over 2x faster than the oracle (ratio = 0.464). The oracle is severely suboptimal for this workload -- likely due to the large batch size (512) making the channel-per-CTA approach impractical (too many elements per channel for register tiling).

## Status: bad_oracle
No action needed. Inductor already significantly outperforms this oracle.
