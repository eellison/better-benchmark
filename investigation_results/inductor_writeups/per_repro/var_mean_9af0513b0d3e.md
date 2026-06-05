# var_mean_9af0513b0d3e

## Summary
- **Model**: Dropout + residual + LayerNorm + transpose
- **Pattern**: dropout + residual add + var_mean + normalize + affine + transpose
- **Ratio**: 1.01x (oracle 12.83us vs compile 12.96us)
- **Classification**: AT_FLOOR (no gap)

## Result
Inductor essentially matches the oracle (1% gap is within noise). No investigation needed.
Contains stochastic ops (dropout).
