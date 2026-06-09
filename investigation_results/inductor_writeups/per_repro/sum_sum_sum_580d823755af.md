# sum_sum_sum_580d823755af


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 65.18 us
- Ratio: N/A

## Compile: ERROR, Oracle: ERROR, Gap: N/A

## Classification: BROKEN_ORACLE

## Root Cause

The oracle fails during CUDA graph capture with error:
"RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned."

The oracle attempts to call `.detach().cpu().tolist()` on a tensor during graph capture, which is incompatible with CUDAGraph-based benchmarking.

## Status: broken_oracle

## Details
- Model: cooperative split-K
- Shape: [512] f32 reductions + [512, 25088] f32 side output + [512] f32
- The oracle has a bug: it performs CPU-GPU data transfer (fmod.detach().cpu().tolist()) inside the forward path that is captured by CUDAGraph
- Cannot benchmark until oracle is fixed to avoid CPU transfers during forward
- No investigation possible without functioning oracle
