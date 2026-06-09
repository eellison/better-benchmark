# var_mean_26f78838cb62


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): not available
- Ratio: N/A

## Classification: CHECK_FAIL

## Current Result

- Oracle path: `repros/canonical/var_mean_26f78838cb62/oracle_beit_patch_layernorm.py`
- Status: `check_fail` (CUDA driver error: no kernel image available for this device)

## Diagnosis

The oracle kernel was compiled for a different CUDA compute capability than the current GPU. The kernel binary is not compatible with this hardware. Cannot benchmark.

## Recommendation

Recompile the oracle kernel for the current GPU architecture (B200).
