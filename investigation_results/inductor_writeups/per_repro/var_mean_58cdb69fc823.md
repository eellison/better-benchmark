# var_mean_58cdb69fc823

## Summary
- **Model**: Instance normalization with reflect padding
- **Pattern**: var_mean over spatial dims -> normalize -> ReLU -> reflect pad [256,256] -> [262,262]
- **Ratio**: 1.246x (oracle 17.18us vs compile 21.41us)
- **Classification**: REDUCTION_EPILOGUE_REREAD

## Root Cause

The repro computes instance normalization with reflected padding:
- Input: `[1, 64, 256, 256]` f32 contiguous
- Compute: var_mean per channel over spatial [256, 256] = 65536 elements
- Normalize: `(x - mean) * rsqrt(var + eps)`
- ReLU activation
- Reflect pad with padding=3: output [1, 64, 262, 262]

The oracle uses 3 kernels:
1. `_partial_channel_stats_kernel` [64 * 64 = 4096]: Split each channel's 65536 elements into 64 blocks of 1024, compute partial sum and sum-of-squares
2. `_final_channel_stats_kernel` [64]: Finalize mean and invstd per channel from 64 partials
3. `_reflect_norm_relu_kernel` [ceil(64*262*262/512)]: In a single pointwise pass, compute reflected input coordinates, normalize, ReLU, and write output

Inductor generates 2 kernels:
1. `triton_red_fused_var_mean_0`: Reduction kernel computing var_mean per channel (xnumel=64, r0_numel=65536)
2. `triton_poi_fused__unsafe_index_abs_add_iota_mul_relu_rsqrt_sub_var_mean_1`: Pointwise kernel doing normalize + ReLU + reflected indexing

The 25% gap comes from:
1. **Under-parallelized reduction**: With xnumel=64 and r0_numel=65536, only 64 thread blocks reduce over 65536 elements each. On 148 SMs, this leaves 84 SMs idle. The oracle uses 4096 blocks (64 channels * 64 blocks/channel) for the reduction, achieving 64x more parallelism.
2. **Reflected indexing overhead**: Minor additional cost from the reflected index computation in the pointwise kernel (already efficient with `_unsafe_index` fusion).

The dominant issue is the reduction parallelism: 64 blocks on 148 SMs means 57% of compute is wasted during the var_mean pass.

## Kernel Count
- **Inductor**: 2 kernels (64-block reduction + pointwise)
- **Oracle**: 3 kernels (4096-block split reduction + 64-block finalization + pointwise)

## Config Exploration
- `combo_kernels=True`: Doesn't help (single reduction kernel)
- `coordinate_descent_tuning=True`: Helps tune the pointwise block size but not the reduction parallelism
- `triton.multi_kernel=2`: 23.54us (worse than default 21.41us)
- `triton.multi_kernel=3`: 20.16us (slightly better than default, but still 1.17x vs oracle 17.18us)
- Best config (multi_kernel=3, looped reduction): 20.16us, ratio 1.17x. Gap reduced but not closed.
- The `should_use_cooperative_reduction` in `choices.py` should trigger here (xnumel=64 < SM_count=148) but apparently does not, or the cooperative mode does not split enough

## Design Doc

The 25% gap is a **split-K parallelism** issue on the var_mean reduction. With only 64 channels and 65536-element reductions, the reduction phase severely underutilizes the GPU.

**Fix path:**
1. In `choices.py`, ensure `should_use_cooperative_reduction` triggers when `xnumel < SM_count` and `rnumel > threshold` (e.g., rnumel > 4096)
2. Split each channel's reduction into `ceil(65536 / BLOCK_K)` blocks, compute partial Welford statistics
3. A finalization kernel of 64 blocks merges the partials

The reflected-index epilogue fusion is already good (Inductor fuses normalize+ReLU+reflect into one pointwise). The gap is purely in the reduction phase.

**Relevant files:**
- `/tmp/pytorch-work/torch/_inductor/choices.py`: `should_use_cooperative_reduction` heuristic
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: Welford split-K template

## Affected Repro Count
Instance normalization reductions with small channel count (< 128) appear in 5+ style-transfer and segmentation model repros. The cooperative reduction fix would benefit all small-xnumel, large-rnumel reductions.
