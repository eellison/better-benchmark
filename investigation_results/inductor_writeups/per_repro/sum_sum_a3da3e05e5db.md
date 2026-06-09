# sum_sum_a3da3e05e5db


## Measured Timings
- Oracle: 448.26 us
- Compile (CDT): 672.61 us
- Ratio: 1.50x

## Classification: CHECK_FAIL

## Current Result

- Oracle path: `repros/canonical/sum_sum_a3da3e05e5db/oracle_bn_hardswish_backward_full_scope.py`
- Correctness: FAIL
- Status: `check_fail`

## Diagnosis

The oracle fails the correctness check against the eager reference. This oracle needs to be regenerated or debugged -- the Triton kernel produces incorrect output for the bn hardswish backward full scope pattern.

## Config exploration results
- N/A (correctness failure).
