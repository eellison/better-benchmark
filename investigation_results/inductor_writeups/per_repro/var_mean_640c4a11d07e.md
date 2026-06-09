# var_mean_640c4a11d07e


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 38.30 us
- Ratio: N/A

## Classification: CHECK_FAIL

## Current Result

- Oracle path: `repros/canonical/var_mean_640c4a11d07e/oracle_swin_droppath_layernorm.py`
- Status: `check_fail` (CUDA driver error: no kernel image available for this device)

## Diagnosis

The oracle kernel was compiled for a different CUDA compute capability than the current GPU. The kernel binary is not compatible with this hardware. Cannot benchmark.

## Recommendation

Recompile the oracle kernel for the current GPU architecture (B200).
