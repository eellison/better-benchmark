# var_mean_4a0c3a2ad6fb


## Measured Timings
- Oracle: 29.31 us
- Compile (CDT): 44.93 us
- Ratio: 1.53x

## Classification: BENCH_FAIL

## Current Result

- Oracle path: `repros/canonical/var_mean_4a0c3a2ad6fb/oracle_bn_channel_shuffle.py`
- Status: `bench_fail`

## Diagnosis

The oracle benchmark failed to run (likely a runtime error during CUDAGraph capture or shape mismatch). The bn channel shuffle pattern oracle requires debugging.

## Config exploration results
- N/A (benchmark failure).
