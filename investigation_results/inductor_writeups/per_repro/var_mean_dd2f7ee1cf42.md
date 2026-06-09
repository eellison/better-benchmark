# var_mean_dd2f7ee1cf42

## Compile: 27.81us, Oracle: 22.43us, Gap: 1.240x

## Diagnosis: SCHEDULER_FUSION

## Root cause: The oracle computes the complete Visformer training-BatchNorm scope in one channel-tiled Triton kernel: fused add producer ([128,768,7,7] + [128,768,7,7]), per-channel var_mean over N/H/W dimensions (reduction axes [0,2,3] with N*H*W=128*7*7=6272 elements per channel), invstd output, affine normalized output, centered output, and in-place running-stat EMA updates -- all from one channel-reduction schedule.

Inductor already produces 1 kernel (`triton_red_fused_add_copy__mul_rsqrt_squeeze_sub_unsqueeze_var_mean_0`) that handles all 5 outputs. The 24% gap comes from the oracle's tighter channel-parallel reduction strategy for the C=768, N*H*W=6272 shape -- it tiles by channel and reduces over the spatial+batch dimensions more efficiently by using a fixed ELEMENTS_PER_CHANNEL=6272 persistent reduction, avoiding Inductor's generic multi-block reduction pattern for this medium-sized reduction domain.

## Kernel count
- Inductor: 1 kernel (fused BN + add + copy_ + multiple outputs)
- Oracle: 1 kernel (channel-tiled BN with inline add)

## Config exploration results
- multi_kernel=0: 47.50us, multi_kernel=1: 49.99us, multi_kernel=2: 51.40us, multi_kernel=3: 52.36us
- No config helps; multi_kernel makes it worse
- coord_descent_tuning already enabled

## Fix path: This is a reduction strategy issue for BN-training with per-channel reduction over N*H*W=6272. The oracle uses one thread-block per channel (768 blocks), each reducing 6272 elements persistently. Inductor's generic red_fused approach uses a different parallelism decomposition.

The fix is in `choices.py` or the BN norm template:
1. For per-channel BN training with moderate N*H*W (fits in ~8K elements), use a persistent channel-parallel reduction
2. Each block processes one channel's full N*H*W spatial elements
3. Inline the add producer and emit all outputs (invstd, affine, centered, running stats) from the same block

## Status: design_doc

## Details
- Model: timm_timm_visformer_small_train_train_000
- Pattern: add([128,768,7,7], [128,768,7,7]) -> var_mean([0,2,3], correction=0) -> BN-training(C=768, eps=1e-5) -> {invstd[768], affine_out[128,768,7,7], centered[128,768,7,7], running_mean EMA, running_var EMA}
- Shapes: Input [128, 768, 7, 7] f32, reduction domain N*H*W=6272 per channel
- 5 outputs: squeeze(rsqrt)[768], affine[128,768,7,7], centered[128,768,7,7], copy_(running_mean)[768], copy_(running_var)[768]
- The running stat EMA is: new_mean = 0.1*batch_mean + 0.9*old_mean
- File references: /tmp/pytorch-work/torch/_inductor/kernel/norm.py (BN training template), /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy for spatial dims)
