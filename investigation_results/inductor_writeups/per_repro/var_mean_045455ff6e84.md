# var_mean_045455ff6e84

## Summary
- **Model**: Embedding + LayerNorm
- **Pattern**: var_mean + normalize + affine
- **Ratio**: 0.953x (oracle 8.13us vs compile 7.74us)
- **Classification**: AT_FLOOR (no gap)

## Result
Inductor already matches or beats the oracle (ratio < 1.0). No investigation needed.
The oracle is slightly slower than compile, indicating Inductor's codegen is already optimal for this pattern.
