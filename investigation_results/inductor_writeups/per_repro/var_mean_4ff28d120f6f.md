# var_mean_4ff28d120f6f

## Summary
- **Model**: GroupNorm + residual
- **Pattern**: var_mean + group norm + residual add
- **Ratio**: 0.949x (oracle 6.3us vs compile 5.98us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle. No investigation needed.
