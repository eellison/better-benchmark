# var_mean_mean_3a508adb31ac

## Summary
- **Model**: timm_mobilenetv2_100 (training)
- **Pattern**: BN-training var_mean -> affine -> ReLU6 -> spatial mean -> running stat copy_
- **Ratio**: 1.139x (oracle 22.11us vs compile 25.18us)
- **Classification**: SCHEDULER_FUSION (training-BN + activation + spatial pool)

## Root Cause

The oracle computes training-BN statistics, running-stat copy_ side effects, affine ReLU6, and the final spatial mean in a tiled channel-specialized kernel that avoids materializing the full [128, 1280, 7, 7] normalized+activated intermediate. The oracle uses three kernels (partial stats, finalize stats, BN-relu6-spatial-mean) but the final kernel reads x once, normalizes, applies affine+ReLU6, and accumulates the spatial mean directly.

Inductor generates 3 kernels:
1. `triton_red_fused_var_mean_0` - computes var/mean statistics via reduction over [0,2,3] dims
2. `triton_per_fused_add_copy__mul_squeeze_var_mean_1` - finalizes stats + running stat updates
3. `triton_per_fused_add_clamp_max_clamp_min_mean_mul_rsqrt_sub_unsqueeze_var_mean_2` - fuses normalize, affine, ReLU6, and spatial mean into one persistent kernel

The issue is that Inductor's kernel #3 already fuses normalize+affine+ReLU6+spatial-mean, which is good. The gap comes from the fact that the oracle's tiled channel-specialized layout (one channel per program) avoids re-reading the statistics per element and allows more efficient register reuse for the accumulation. Inductor's kernel tiles by (batch * channels) in the outer dimension with spatial (49 elements) as the reduction, loading broadcast statistics redundantly per-row.

## Kernel Count
- **Inductor**: 3 kernels
- **Oracle**: 3 kernels (partial_stats, finalize_stats, bn_relu6_spatial_mean)

## Config Exploration
The current Inductor config with `combo_kernels=True` and `coordinate_descent_tuning=True` already achieves a reasonable 3-kernel split. The gap is in the per-channel tiling strategy of the final kernel.

## Design Doc

The ~14% gap is attributable to Inductor's tiling choice in the final fused kernel. The oracle tiles by channel (one thread block per channel, iterating over all batch*spatial elements), which:
1. Loads weight/bias/mean/invstd once per channel into registers
2. Streams through the batch*spatial elements for that channel contiguously

Inductor tiles by (batch, channel) with spatial as the inner reduction, meaning each thread block handles one (batch, channel) pair and reduces over 49 elements. This is a 49-element persistent reduction which is tiny and leaves many SMs underutilized.

### What enhancement is needed
The scheduler/codegen should recognize patterns where:
- A channel-wise reduction (var_mean over [0,2,3]) feeds into a per-element normalization
- Followed by a spatial reduction (mean over [-1,-2]) 
- The spatial dimension is small (49 elements)

In this case, it should consider tiling by channel (outer) with batch*spatial as the inner tile, allowing per-channel scalar broadcasts to be hoisted. This is a codegen/tiling heuristic enhancement in `torch/_inductor/codegen/triton.py` or `choices.py`.

### Affected files
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (tiling decisions)
- `/tmp/pytorch-work/torch/_inductor/choices.py` (reduction strategy for small spatial dims)

### Affected repro count
This pattern (training-BN + activation + spatial pool) appears in multiple MobileNet and EfficientNet variants. Estimated 5-10 repros share this exact structure.
