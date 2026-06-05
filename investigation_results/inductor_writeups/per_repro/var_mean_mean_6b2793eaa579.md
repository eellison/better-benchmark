# var_mean_mean_6b2793eaa579

## Summary
- **Model**: MobileNet BN training + HardSwish + mean
- **Pattern**: var_mean + BN affine + hardswish + spatial mean
- **Ratio**: 0.688x (oracle 48.83us vs compile 33.6us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle by 31.2%. No investigation needed.
