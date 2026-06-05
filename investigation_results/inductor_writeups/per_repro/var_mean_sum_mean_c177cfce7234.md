# var_mean_sum_mean_c177cfce7234

## Summary
- **Model**: ResNeSt training BN pair mean
- **Pattern**: dual var_mean + sum + mean (batch norm training pair)
- **Ratio**: 0.882x (oracle 21.38us vs compile 18.85us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle by 11.8%. No investigation needed.
