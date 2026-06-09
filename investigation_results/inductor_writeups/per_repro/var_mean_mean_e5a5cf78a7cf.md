# var_mean_mean_e5a5cf78a7cf

## Summary
- **Model**: BN training + SiLU + spatial mean
- **Pattern**: Per-channel var_mean over [128, 640, 7, 7](?), running-stat copy_, affine, silu, spatial mean
- **Oracle**: 49.06 us
- **Compile (CDT)**: 21.70 us
- **Ratio**: 0.442x (BAD_ORACLE)
- **Classification**: BAD_ORACLE

## Benchmark Results
Inductor compile is over 2x faster than the oracle (ratio = 0.442). The oracle is severely suboptimal for this workload -- likely due to poor tile choices or excessive register usage in the channel-per-CTA approach at this shape.

## Status: bad_oracle
No action needed. Inductor already significantly outperforms this oracle.
