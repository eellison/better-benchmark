# mean_mean_1b98d81214e6 - MT5 Dropout RMSNorm Dual

## Benchmark Result
- Oracle: 29.34 us
- Compile: 26.37 us
- Ratio: 0.899x
- Status: BAD_ORACLE

## Root Cause
The oracle for this MT5 dual-RMSNorm + dropout pattern is slower than Inductor's compiled output. The oracle likely has suboptimal kernel configuration or unnecessary intermediate steps that Inductor's autotuning avoids.

## Kernel Count
- Inductor outperforms the oracle here, so kernel structure is not a concern.

## Config Exploration
No investigation needed - Inductor already wins.

## Conclusion
BAD_ORACLE - the oracle is 10% slower than compiled. Inductor handles this pattern well. No fix needed.
