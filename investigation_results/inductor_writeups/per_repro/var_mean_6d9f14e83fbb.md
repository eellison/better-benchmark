# var_mean_6d9f14e83fbb


## Measured Timings
- Oracle: 25.38 us
- Compile (CDT): 90.91 us
- Ratio: 3.58x

## Classification: CHECK_FAIL

## Current Result

- Oracle path: `repros/canonical/var_mean_6d9f14e83fbb/oracle_embedding_layernorm_mask.py`
- Correctness: FAIL
- Status: `check_fail`

## Diagnosis

The oracle fails the correctness check against the eager reference. This oracle needs to be regenerated or debugged -- the Triton kernel produces incorrect output for the embedding layernorm mask pattern.

## Config exploration results
- N/A (correctness failure).
