# var_mean_e3c259686c7b

## Summary
- **Model**: BERT embedding + dropout + LayerNorm
- **Pattern**: embedding add + dropout + var_mean + normalize + affine
- **Ratio**: 0.965x (oracle 21.31us vs compile 20.58us)
- **Classification**: AT_FLOOR (no gap)

## Result
Inductor already matches the oracle (within 3.5%). No investigation needed.
Contains stochastic ops (dropout), which are handled correctly.
