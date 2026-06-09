# pointwise_e796eba4dc47 — Longformer Chunk Layout

## Status: BAD_ORACLE (ratio = 0.689x)

## Oracle Description
The oracle fuses bias add, Longformer head split/permutation, symmetric constant pad, overlapping as_strided clone, and final contiguous chunk layout into one Triton materialization kernel that writes directly to the output f32[384,768,64] tensor.

## Classification: SCHEDULER_FUSION (oracle slower than compile)

## Benchmark Results
- Oracle: 45.82 us
- Compiled: 31.58 us
- Ratio: 0.689x (compile is MUCH FASTER)

## Root Cause
The oracle attempts to fuse overlapping-window materialization with padding and bias addition in one kernel. This is a large output (384*768*64 = ~19M elements). The oracle's complex affine indexing for the overlapping as_strided pattern (with window_size=768, window_step=256, pad_before=256) likely causes:
1. Poor cache utilization due to overlapping reads across windows
2. Complex branching for padding boundary conditions
3. Redundant computation that Inductor avoids by materializing intermediate results

Inductor's multi-kernel approach with simpler individual kernels achieves much better throughput.

## Kernel Count
- Oracle: 1 kernel (complex overlapping window indexing)
- Inductor: likely 2-3 kernels (significantly faster)

## Config Exploration
No config changes needed — compile already significantly outperforms oracle.

## Conclusion
Oracle is suboptimal. The overlapping window pattern with padding is better handled by Inductor's separate-kernel approach, which achieves better memory access patterns despite additional kernel launches.
