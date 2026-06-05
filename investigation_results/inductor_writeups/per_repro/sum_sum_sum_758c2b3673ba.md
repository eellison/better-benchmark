# sum_sum_sum_758c2b3673ba

## Summary
- **Model**: Swin window reduction fusion
- **Pattern**: window attention backward + sum reductions
- **Ratio**: 0.545x (oracle 265.15us vs compile 144.38us)
- **Classification**: BAD_ORACLE (oracle significantly slower than compile)

## Result
Inductor already beats the oracle by 45.5%. The oracle's manual Triton kernels are suboptimal for this workload.
No investigation needed.
