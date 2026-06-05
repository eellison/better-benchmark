# var_mean_2f98fd23fbea

## Classification: CHECK_FAIL

## Current Result

- Oracle path: `repros/canonical/var_mean_2f98fd23fbea/oracle_channel_layernorm.py`
- Correctness: FAIL
- Status: `check_fail`

## Diagnosis

The oracle fails the correctness check against the eager reference. This oracle needs to be regenerated or debugged -- the Triton kernel produces incorrect output for the channel layernorm pattern.

## Config exploration results
- N/A (correctness failure).
