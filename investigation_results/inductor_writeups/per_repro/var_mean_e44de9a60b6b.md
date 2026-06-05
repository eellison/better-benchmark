# var_mean_e44de9a60b6b

## Summary
- **Model**: DistilBERT embedding + dropout + LayerNorm
- **Pattern**: embedding add + dropout + var_mean + normalize + affine
- **Ratio**: 0.932x (oracle 28.54us vs compile 26.59us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle. No investigation needed.
Contains stochastic ops (dropout).
