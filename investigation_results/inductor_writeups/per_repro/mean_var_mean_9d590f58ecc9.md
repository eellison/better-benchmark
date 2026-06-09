# mean_var_mean_9d590f58ecc9


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 14.02 us
- Ratio: N/A

## Classification: BROKEN_ORACLE

## Root cause: The oracle kernel (oracle_spatial_mean_layernorm_channels_last.py) fails with "CUDA driver error: no kernel image is available for execution on the device". The oracle was compiled for a different GPU architecture than the current hardware.

## Status: broken_oracle (cannot measure gap)

## Details
- Error: RuntimeError: CUDA driver error: no kernel image is available for execution on the device
- This is an architecture mismatch in the precompiled Triton kernel, not an Inductor issue.
- No investigation possible until oracle is recompiled for this GPU.
