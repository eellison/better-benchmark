# mean_mean_6cb471f41ecf - T5 Shifted Dropout RMSNorm

## Benchmark Result
- Oracle: 42.85 us
- Compile: 38.91 us
- Ratio: 0.908x
- Status: BAD_ORACLE

## Root Cause
The oracle for T5 shifted dropout + RMSNorm is slower than Inductor's compiled output. The stochastic operations and RMSNorm reduction are handled efficiently by Inductor's existing reduction codegen.

## Kernel Count
- Inductor outperforms oracle, so kernel count is not a concern.

## Config Exploration
No investigation needed - Inductor already wins.

## Conclusion
BAD_ORACLE - compiled code is ~9% faster. No fix needed.
