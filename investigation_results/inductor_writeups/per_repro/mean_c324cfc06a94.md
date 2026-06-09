# mean_c324cfc06a94 - BN SiLU Spatial Mean


## Measured Timings
- Oracle: 15.87 us
- Compile (CDT): 11.71 us
- Ratio: 0.74x

## Benchmark Results
- Oracle: slower than compiled (ratio: 0.826x - BAD_ORACLE)

## Classification
BAD_ORACLE - compiled Inductor is already faster than the oracle

## Root Cause
No performance gap to investigate. The compiled Inductor code outperforms the oracle by ~17%. The oracle kernel is suboptimal for this workload.

## Actionable
No - Inductor already achieves better performance than the oracle target.
