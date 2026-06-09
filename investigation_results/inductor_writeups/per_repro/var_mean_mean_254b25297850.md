# var_mean_mean_254b25297850

## Summary
- **Model**: timm_swin_base_patch4_window7_224 (inference)
- **Pattern**: residual add -> reshape -> LayerNorm (var_mean over [3]) -> affine -> spatial mean over [1,2]
- **Ratio**: 1.229x (oracle 29.57us vs compile 36.35us)
- **Classification**: SCHEDULER_FUSION (LayerNorm + downstream spatial mean)

## Root Cause

The oracle fuses the residual add, per-row LayerNorm statistics, affine transform, and 7x7 spatial mean into two tightly coupled kernels:
1. `_row_layernorm_stats_kernel` - computes per-row (6272 rows) mean and invstd over 1024 channels
2. `_layernorm_spatial_mean_kernel` - reads the same input, applies normalization using stored stats, applies affine weight/bias, and accumulates the spatial mean (over 49 positions) directly into the [128, 1024] output

Inductor generates 2 kernels:
1. `triton_per_fused_add_var_mean_view_0` - residual add + var_mean per row (xnumel=6272, r0_numel=1024)
2. `triton_per_fused_add_mean_mul_rsqrt_sub_var_mean_view_1` - reads input, loads per-row stats, normalizes, applies affine, reduces spatial dim (xnumel=131072 = 128*1024 channels, r0_numel=49 spatial positions)

The gap is that Inductor's second kernel has 131072 work items each reducing over 49 elements. This means 131072 separate thread blocks (or tiles), each doing a tiny 49-element reduction with a 64-element RBLOCK. The oracle instead tiles by (batch, channel_block) with the 49 spatial positions as a dense inner loop (BLOCK_S=64), letting it process multiple channels per block and achieving much better SM occupancy and memory throughput.

## Kernel Count
- **Inductor**: 2 kernels
- **Oracle**: 2 kernels

## Config Exploration
`combo_kernels=True` and `coordinate_descent_tuning=True` already enabled. The kernel count is optimal (2). The issue is purely in the tiling/grid strategy of the second kernel.

## Design Doc

The 23% gap is a **tiling inefficiency** in the second kernel. The problem:
- 131072 outer elements (batch*channels = 128*1024) with only 49 inner reduction elements
- Each thread block does almost nothing (sum of 49 floats)
- Launch overhead and memory access patterns are suboptimal

The oracle's approach:
- Grid: (128 batches, cdiv(1024, BLOCK_C)) where BLOCK_C is autotuned (16/32/64)
- Each block handles BLOCK_C channels across all 49 spatial positions
- This groups the work into fewer, larger blocks with better data reuse

### What enhancement is needed
The scheduler/codegen should recognize when a downstream reduction has a very small reduction dimension (e.g., 49 spatial positions from a 7x7 feature map) and the outer dimension is very large. In this case, it should consider:
1. Grouping multiple outer elements (channels) into a single block, with the tiny reduction fully unrolled
2. The mean/invstd statistics are loaded per spatial position (49 values) rather than per channel, so the access pattern should tile channels as the fast dimension

This is a tiling heuristic in the codegen that should prefer "wide" blocks (multiple channels per block) when the reduction dimension is tiny.

### Affected files
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (tiling for tiny reductions)
- `/tmp/pytorch-work/torch/_inductor/choices.py` (persistent reduction threshold)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (could fuse the two kernels into one if the stats were computed inline, but the 1024-element LayerNorm stats reduction prevents this)

### Affected repro count
This pattern (LayerNorm + spatial mean) is specific to Swin Transformer's forward_features. Estimated 3-5 repros share this exact structure across different Swin variants and input sizes.
