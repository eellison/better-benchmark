# sum_sum_sum_718a025684ba

## Summary
- **Model**: timm_ghostnet_100_train_001
- **Pattern**: dual batch-norm backward (80-channel full + 40-channel slice) with split-K reduction
- **Ratio**: 2.019x (oracle 76.54us vs compile 154.53us)
- **Classification**: COOPERATIVE_SPLIT_K

## Root Cause

The repro computes batch-norm backward for a GhostNet model with two interleaved BN branches:
- Full branch: 80 channels over [512, 80, 14, 14] (reduction over batch*H*W = 100352 elements per channel)
- Slice branch: 40 channels from channels [40:80] of the same tensor

The oracle splits this into 5 kernels using a cooperative split-K strategy:
1. `_partial_reduce_kernel` [80, nblocks]: Each of 80 channels is split across `nblocks` (ceil(100352/1024)=98) blocks. Each block computes partial sums for both the full and slice branches simultaneously.
2. `_finalize_full_kernel` [80]: Sums the 98 partial sums per full channel, derives mean_term, coeff, fused_weight.
3. `_finalize_slice_kernel` [40]: Same for the slice branch.
4. `_pointwise_full_kernel`: Writes the full gradient tensor using finalized coefficients.
5. `_pointwise_slice_kernel`: Writes the slice gradient tensor with channels-last layout.

Inductor emits a single combo kernel with only 80+40=120 thread blocks, each reducing over 100352 elements sequentially. With only 120 blocks on 148 SMs, the GPU is severely underutilized. The oracle's split-K approach launches 80*98=7840 blocks for the reduction phase, achieving ~53x more parallelism.

## What Inductor Cannot Do Today

1. **Cooperative split-K for channel reductions**: When xnumel (80 channels) is much less than SM count (148), Inductor should split the reduction domain to increase parallelism. The `should_use_cooperative_reduction` heuristic in `choices.py` does not trigger here because the combo kernel merging happens after the cooperative reduction decision.
2. **Multi-output split-K with shared producers**: The full and slice branches share input data (channels [40:80] overlap). The oracle exploits this by computing both branches' partial sums in one pass. Inductor cannot represent this shared-data multi-output split-K pattern.
3. **Layout-changing epilogue fusion**: The slice output uses channels-last layout `(CHANNELS_SLICE*HW, 1, W*CHANNELS_SLICE, CHANNELS_SLICE)` but shares a producer with the contiguous full output. Inductor materializes the layout change separately.

## Kernel Count
- **Inductor**: 1 combo kernel (80+40=120 blocks, each looping over 100352 elements)
- **Oracle**: 5 kernels (7840 reduction blocks + 80 finalize + 40 finalize + pointwise full + pointwise slice)

## Config Exploration
- `combo_kernels=True`: Already enabled, produces the combo kernel with 120 blocks
- `coordinate_descent_tuning=True`: Already enabled, helps with block sizes but cannot fix the parallelism gap
- `triton.multi_kernel=1/2/3`: Does not help; the issue is structural (not enough blocks)

## Design Doc

The 2x gap is a **cooperative split-K parallelism** issue. With 80 channels and 100352-element reductions, only 80 blocks occupy 148 SMs. The fix requires:

1. **Detect underutilized-SM pattern**: In `choices.py`, when `xnumel * num_combo_subkernels < SM_count` and `rnumel` is large, split the reduction domain.
2. **Emit split-K reduction + finalization**: Each channel gets `ceil(rnumel / BLOCK_K)` blocks. A separate finalization kernel sums the partials.
3. **Fuse overlapping branches**: When two reductions share input data (slice is a subset of full), compute both in one split-K pass.

**Relevant files:**
- `/tmp/pytorch-work/torch/_inductor/choices.py`: `should_use_cooperative_reduction`
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: combo kernel formation (happens after individual reduction strategy is chosen)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: split-K codegen templates

## Affected Repro Count
This COOPERATIVE_SPLIT_K pattern with channel reductions over large spatial domains appears in 15+ repros across GhostNet, ResNet, Swin, and other architectures with BN-backward on high spatial resolution. The fix would benefit all `sum([0,2,3])` reductions where channel count < SM count.
