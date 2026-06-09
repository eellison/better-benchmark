# mean_var_mean_c9089e261699


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 15.20 us
- Ratio: N/A

## Classification: BROKEN_ORACLE

## Root cause: The oracle kernel (oracle_spatial_mean_layernorm.py) fails with "CUDA driver error: no kernel image is available for execution on the device." The oracle was compiled for a different GPU architecture than the current hardware.

## Status: broken_oracle (cannot measure gap)

## Details
- Error: RuntimeError: CUDA driver error: no kernel image is available for execution on the device
- The oracle's pre-compiled kernel targets a different compute capability.
- No investigation possible until oracle is recompiled for the current GPU.
