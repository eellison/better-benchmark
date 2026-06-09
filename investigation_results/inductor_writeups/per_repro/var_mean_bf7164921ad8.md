# var_mean_bf7164921ad8


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 39.78 us
- Ratio: N/A

## Classification: BROKEN_ORACLE

## Current Result

- Family: `dropout_residual_layernorm`
- Oracle path: `repros/canonical/var_mean_bf7164921ad8/oracle_dropout_residual_layernorm.py`
- Correctness: FAIL (CUDA driver error: no kernel image is available for execution on the device)
- Status: `broken_oracle`

## Diagnosis

The oracle fails with a CUDA compatibility error: "CUDA driver error: no kernel image is available for execution on the device". This indicates the oracle's pre-compiled Triton kernel was built for a different GPU architecture than the current hardware.

The oracle is the same family as var_mean_b31c33bdc684 (dropout_residual_layernorm). Based on the docstring classification of BANDWIDTH_BOUND, this is likely at or near floor once the kernel compatibility issue is resolved.

## Config exploration results
- Unable to test -- oracle kernel is incompatible with current GPU hardware.

## Recommendation

Recompile the oracle kernel for the current GPU architecture, or mark as broken until hardware-specific builds are available.
