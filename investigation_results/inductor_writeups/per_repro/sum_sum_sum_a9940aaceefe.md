# sum_sum_sum_a9940aaceefe

## Summary
- **Model**: Blenderbot dropout + LayerNorm backward
- **Pattern**: dropout + LN backward column reductions
- **Ratio**: 0.890x (oracle 86.02us vs compile 76.54us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle by 11%. No investigation needed.
