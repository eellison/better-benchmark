# pointwise_e4cfa8694326 — ShuffleNet BN Channel Shuffle

## Status: BAD_ORACLE (ratio = 0.794x)

## Oracle Description
The oracle fuses the full ShuffleNet inference graph: BN (mean/var/affine), ReLU, cat with a passthrough branch, view/permute/clone channel shuffle — all into one Triton kernel writing the final contiguous f16 output in shuffled channel layout.

## Classification: SCHEDULER_FUSION (oracle slower than compile)

## Benchmark Results
- Oracle: 40.16 us
- Compiled: 31.9 us
- Ratio: 0.794x (compile is MUCH FASTER)

## Root Cause
Despite the oracle's ambitious single-kernel fusion of BN + ReLU + cat + channel shuffle, the resulting kernel is significantly slower than Inductor's multi-kernel approach. This is likely because:
1. The oracle's complex indexing (handling both passthrough and BN branches with shuffled output coordinates) causes poor memory access patterns
2. Inductor's separate kernels benefit from better data locality and simpler index arithmetic per kernel
3. The oracle has stochastic ops (noted in check output), suggesting dropout or BN training mode that complicates the fusion

## Kernel Count
- Oracle: 1 kernel (slower due to complex indexing)
- Inductor: likely 2-3 kernels (faster overall due to simpler access patterns)

## Config Exploration
No config changes needed — compile already significantly outperforms oracle.

## Conclusion
Oracle is suboptimal. The aggressive fusion strategy hurts performance due to complex indexing patterns. Inductor's approach of separate simpler kernels is superior here.
