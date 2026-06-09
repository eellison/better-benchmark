# var_mean_67e607b197fd


## Measured Timings
- Oracle: 15.20 us
- Compile (CDT): 15.33 us
- Ratio: 1.01x

## Classification: BENCH_FAIL

## Current Result

- Oracle path: `repros/canonical/var_mean_67e607b197fd/oracle_mobilevit_residual_layernorm_fold.py`
- Status: `bench_fail`

## Diagnosis

The oracle benchmark failed to run (likely a runtime error during CUDAGraph capture or shape mismatch). The mobilevit residual layernorm fold pattern oracle requires debugging.

## Config exploration results
- N/A (benchmark failure).
