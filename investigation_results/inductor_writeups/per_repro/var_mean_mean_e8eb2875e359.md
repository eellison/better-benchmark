# var_mean_mean_e8eb2875e359

## Summary
- **Model**: timm_mobilevit_s (training)
- **Pattern**: BN-training var_mean -> affine -> SiLU -> spatial mean -> running stat copy_
- **Ratio**: 1.382x (oracle 14.75us vs compile 20.38us)
- **Classification**: SCHEDULER_FUSION (training-BN + SiLU + spatial pool)

## Root Cause

The oracle computes the entire training-BN + SiLU + spatial mean scope in a single channel-specialized Triton kernel. Each program handles one channel (640 total), iterates over all 128 batch elements and 64 spatial positions (8x8), computes channel statistics (sum, sum_sq), derives mean/var/invstd, updates running stats in-place, normalizes, applies affine + SiLU activation, and accumulates the spatial mean into the [128, 640] output.

Inductor generates 2 kernels:
1. `triton_red_fused_add_copy__mul_squeeze_var_mean_0` - a split reduction that computes var_mean over [0,2,3] (xnumel=640, r0_numel=8192=128*64) plus running stat updates
2. `triton_per_fused_add_div_exp_mean_mul_neg_rsqrt_sub_unsqueeze_var_mean_1` - normalizes, applies affine+SiLU, reduces spatial mean (xnumel=81920=128*640, r0_numel=64)

The gap is the same tiling pattern as the other cases: the second kernel has 81920 outer elements each reducing over 64 spatial positions. Each thread block does a tiny 64-element reduction. The oracle instead has only 640 thread blocks (one per channel), each doing the full computation for that channel across all batch*spatial elements.

The larger gap here (38%) vs the 3a508adb31ac case (14%) is because:
1. The spatial dimension is 64 (8x8) vs 49 (7x7) - slightly larger but still very small
2. The input is [128, 640, 8, 8] = 41.9MB total traffic. The oracle reads it once for both stats and norm+pool. Inductor reads it in kernel #1 for stats AND again in kernel #2 for norm+pool - a 2x read amplification on the main activation tensor.

## Kernel Count
- **Inductor**: 2 kernels
- **Oracle**: 1 kernel (!)

The critical difference: the oracle computes statistics AND the normalized+pooled output in ONE pass over the data, while Inductor requires two separate passes.

## Config Exploration
`combo_kernels=True` and `coordinate_descent_tuning=True` already enabled. The key structural limitation is that Inductor's scheduler cannot fuse a reduction (var_mean over [0,2,3]) with a downstream per-element operation that consumes the reduction result and then does another reduction (mean over [-1,-2]). These are inherently separate kernels in Inductor's current model.

## Design Doc

The 38% gap has two components:
1. **2x memory read amplification**: The input [128, 640, 8, 8] is read once for stats and once for normalize+pool. The oracle reads it once.
2. **Tiling inefficiency in kernel #2**: 81920 tiny reductions over 64 elements each.

### What enhancement is needed

**Primary**: A new "channel-specialized reduction template" that can compute:
- Per-channel statistics (reduction over batch*spatial) 
- Then immediately normalize, activate, and reduce (spatial mean)
- All in one kernel, reading the input only once

This requires the scheduler to recognize the pattern:
```
var_mean(x, [0,2,3]) -> normalize(x, stats) -> activation -> mean(result, [-1,-2])
```
and emit a single channel-tiled kernel where each thread block:
1. Iterates over batch*spatial for one channel
2. Accumulates sum/sum_sq for statistics
3. Derives mean/var
4. In a second pass (or same pass with online algorithm), normalizes and accumulates spatial means

This is fundamentally a **two-pass single-kernel** pattern (or Welford-based single-pass). The scheduler needs to express "use the reduction result immediately in the same kernel" rather than materializing it to global memory.

### Affected files
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion of producer reduction -> consumer that re-reads same input)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (channel-specialized tiling)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (representing two-pass reductions)

### Affected repro count
This pattern (training-BN + activation + spatial pool) with the channel-specialized single-kernel opportunity appears in MobileViT, MobileNet, EfficientNet training. Estimated 8-12 repros share this structure. The specific case of var_mean_mean_3a508adb31ac is the same pattern but with ReLU6 instead of SiLU and has a smaller gap (14%) likely due to different shape balance.
