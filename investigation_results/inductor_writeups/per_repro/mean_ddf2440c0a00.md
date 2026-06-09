# mean_ddf2440c0a00 - MobileNet BN+ReLU6+Spatial Mean (1.115x)


## Measured Timings
- Oracle: 16.13 us
- Compile (CDT): 13.28 us
- Ratio: 0.82x

## Classification
BANDWIDTH_BOUND (tiling heuristic gap)

## Root Cause
The repro computes BN-inference affine over fp16 [128,1280,7,7], clamps to [0,6]
(ReLU6), converts to fp16, then spatially averages each 7x7 tile to produce
[128,1280] output.

Inductor already fuses all ops into a single persistent reduction kernel with
RBLOCK=64 (covering the 49 spatial elements). Each thread block processes one
(batch, channel) pair independently.

The oracle uses a 3D tiling scheme: BLOCK_BATCH=16 x BLOCK_CHANNELS=4 x BLOCK_HW=64,
processing 16*4=64 (batch,channel) pairs per thread block with a shared spatial
reduction. This gives better SM utilization because:
- 163840 work items / 64 = 2560 blocks (Inductor) vs fewer, larger blocks (oracle)
- The oracle amortizes per-channel parameter loads across 16 batch elements

## Kernel Count
- Oracle: 1 kernel
- Inductor: 1 kernel

Both are single-kernel solutions. The gap is purely in tiling strategy.

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| coordinate_descent + combo_kernels | 17.98 | Default compiled |
| max_autotune | 54.56 | Worse due to autotune overhead / wrong config |
| Oracle | 16.13 | Better tiling for small spatial dims |

## Why Inductor Cannot Fix This Today
Inductor's reduction heuristic assigns one (batch, channel) pair per thread block for
persistent reductions. When the reduction dimension is very small (49 elements), this
under-utilizes each thread block. The oracle packs multiple (batch, channel) pairs into
one block, effectively creating a "multi-row" persistent reduction.

A potential fix would be to detect cases where:
- Reduction dimension is small (< WARP_SIZE)
- Outer dimension is large (163840 here)
And automatically apply multi-row reduction tiling.

## File References
- `torch/_inductor/choices.py` - reduction strategy selection
- `torch/_inductor/codegen/triton.py` - persistent reduction codegen

## Status
Design doc only. The 1.115x gap is relatively small and the fix requires changes
to the reduction tiling heuristic that could affect many other cases. Would benefit
from a broader study of small-spatial-dim reduction patterns.
