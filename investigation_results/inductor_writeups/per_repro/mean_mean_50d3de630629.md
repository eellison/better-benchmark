# mean_mean_50d3de630629 - Qwen RoPE RMSNorm

## Benchmark Result
- Oracle: 33.6 us
- Compile: 28.61 us
- Ratio: 0.851x
- Status: BAD_ORACLE

## Root Cause
The oracle for this Qwen RoPE + RMSNorm pattern is significantly slower than Inductor's compiled version. Inductor's fused reduction kernels with coordinate descent tuning outperform the hand-written oracle.

## Kernel Count
- Inductor outperforms the oracle, kernel structure not a concern.

## Config Exploration
No investigation needed - Inductor already wins by 15%.

## Conclusion
BAD_ORACLE - compiled code is faster. No fix needed.
