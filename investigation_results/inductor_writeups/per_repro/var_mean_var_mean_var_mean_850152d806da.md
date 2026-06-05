# var_mean_var_mean_var_mean_850152d806da

## Summary
- **Model**: Inception BN + ReLU + AvgPool
- **Pattern**: multiple var_mean (3 branches) + normalize + affine + ReLU + avgpool
- **Ratio**: 0.873x (oracle 128.67us vs compile 112.38us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle by 12.7%. No investigation needed.
