# sum_sum_sum_b35553d96630


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 77.76 us
- Ratio: N/A

## Classification: SCHEDULER_FUSION

## Oracle: oracle_swin_layernorm_bwd_splitk.py

## Measurements

- Compiled: N/A (unable to bench due to kernel incompatibility)
- Oracle: N/A
- Ratio: N/A
- Oracle correctness: FAIL (CUDA kernel image unavailable for execution on device)

## Diagnosis

The oracle (swin_layernorm_bwd_splitk) fuses the complete Swin layernorm-backward-style region by combining the window-unpartition clone, two per-row 256-wide reductions, the final gradient-input backing store, and both per-channel reduction partials into one Triton pass, then finalizes both channel sums in one Triton reduction kernel. Inductor emits a generic mix-order reduction that materializes a workspace and lets the two final channel reductions run as separate wrapper-level sums.

The oracle kernel was compiled for a different GPU architecture than the current test device. The kernel binary is not compatible with this GPU (likely compiled for sm_90 but running on a different compute capability).

## Config Exploration

Unable to run config exploration due to CUDA kernel architecture mismatch.

## Status: blocked_infrastructure (kernel arch mismatch)
