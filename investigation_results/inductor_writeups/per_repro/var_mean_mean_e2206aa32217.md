# var_mean_mean_e2206aa32217

## Summary
- **Model**: timm_mobilenetv3 (training) / MobileNetV3 BN+ReLU+spatial_mean
- **Pattern**: training batch-norm (var_mean + running stats + affine) -> ReLU -> spatial mean -> [512, 120, 1, 1]
- **Ratio**: 1.215x (oracle 94.14us vs compile 114.37us)
- **Classification**: BN_TRAINING_DOWNSTREAM_FUSION

## Root Cause

The repro computes batch normalization in training mode with fused downstream operations:
- Input: `[512, 120, 28, 28]` f32 (N, C, H, W layout)
- Compute var_mean per channel over [N, H, W] = 512*784 = 401408 elements/channel
- Update running_mean and running_var (copy_ side effects)
- Apply affine transform: `(x - mean) * invstd * weight + bias`
- ReLU activation
- Spatial mean: mean over [H, W] producing [512, 120, 1, 1]

The oracle uses 2 kernels:
1. `_partial_stats_kernel` [(64, 120)]: Split each channel's batch dimension into 64 blocks of 8 samples. Each block computes partial sum and sum-of-squares over 8*784=6272 elements per channel.
2. `_bn_relu_spatial_mean_from_partials_kernel` [(64, 120)]: Each block finalizes the channel statistics from partials, updates running stats (only block 0 does the copy_), then normalizes+affine+ReLU for its 8 samples, and computes the spatial mean (sum over H*W / HW) per sample, writing directly to the [512, 120, 1, 1] output.

Inductor generates 2 kernels:
1. `triton_red_fused_add_copy__mul_squeeze_var_mean_0`: Reduction kernel computing var_mean + running stat updates (xnumel=120, r0_numel=401408)
2. `triton_per_fused_add_mean_mul_relu_rsqrt_sub_unsqueeze_var_mean_1`: Per-row kernel doing normalize+affine+ReLU+spatial_mean (xnumel=61440=512*120, r0_numel=784)

The 21.5% gap comes from:
1. **Under-parallelized BN statistics reduction**: With xnumel=120 channels and r0_numel=401408, only 120 blocks reduce over 401408 elements each. On 148 SMs, this leaves 28 SMs idle, and each block does a very long sequential loop. The oracle uses 64*120=7680 blocks.
2. **Second kernel tiling**: The per-element kernel (61440 work items, each reducing 784 spatial elements) has good parallelism but each block does a tiny 784-element reduction. The oracle instead tiles by (batch_block, channel) and computes the full spatial mean in a single tile of 8*784=6272 elements, achieving better register reuse.

## Kernel Count
- **Inductor**: 2 kernels
- **Oracle**: 2 kernels

## Config Exploration
- `combo_kernels=True`: Does not help; kernels have different grid shapes
- `coordinate_descent_tuning=True`: Helps tune block sizes but cannot fix the reduction parallelism
- `triton.multi_kernel=2`: 264.70us (much worse, likely persistent reduction fails with large rnumel=401408)
- `triton.multi_kernel=3`: 117.91us (slightly worse than default 114.37us)
- Best config is default: 114.37us. No multi_kernel mode helps.
- The first kernel (120 blocks, 401408 reduction) is the bottleneck. The second kernel has adequate parallelism (61440 blocks).

## Design Doc

The 22% gap combines two issues:

### Issue 1: Under-parallelized BN statistics (primary, ~15%)
With 120 channels on 148 SMs, the var_mean reduction is under-parallel. The fix is cooperative split-K for the batch-norm statistics computation.

### Issue 2: BN + downstream spatial mean fusion (secondary, ~7%)
The oracle fuses normalize+affine+ReLU+spatial_mean into the same kernel that reads the partials, avoiding a full re-read of the [512, 120, 28, 28] tensor. Inductor's second kernel re-reads the input (which was already read in kernel 1 for statistics).

**Fix path:**
1. **Split-K BN stats**: In `choices.py`, when computing BN training statistics (var_mean over [N,H,W]) with channel count < SM_count, split the N*H*W reduction domain across multiple blocks per channel.
2. **Fuse norm epilogue with spatial pool**: After finalization, the oracle reads the input one more time to normalize+pool. If the statistics kernel could pass partial results to the norm+pool kernel, only one full data read is needed (currently two).

**Relevant files:**
- `/tmp/pytorch-work/torch/_inductor/choices.py`: BN training reduction strategy
- `/tmp/pytorch-work/torch/_inductor/kernel/norm.py`: batch-norm template
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion of BN output with downstream pool

## Affected Repro Count
Training batch-norm with downstream spatial operations (mean, max_pool) appears in 10+ MobileNet/EfficientNet/ResNet training repros. The BN split-K fix would benefit all training BN repros where channel_count < SM_count.
