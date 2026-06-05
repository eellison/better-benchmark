# amax_sum_any_64da678446f9

## Compile: 194.34us, Oracle: 182.05us, Gap: 1.067x

## Classification: ONLINE_SOFTMAX_DROPOUT_EPILOGUE

## Root Cause

The oracle fuses the complete DistilBERT attention softmax/dropout (tautological iota>=0 mask folded to zero bias, [3072,128,128] -> [256,12,128,128] view, stable softmax with all-minus-inf row zeroing, Inductor RNG dropout from seed tensor, expand/view, and final non-contiguous [3072,128,128] transpose view) into a single persistent online-softmax kernel.

Inductor already fuses this into a single persistent kernel (`triton_per_fused_add_any_div_eq_exp_expand_full_ge_gt_inductor_lookup_seed_inductor_random_logical_not_mul_prepare_softmax_online_sub_view_where_0`). The 6.7% gap comes from the trailing permute(0,2,1) output layout that the oracle writes directly in its kernel epilogue, while Inductor may need a separate layout pass or less optimal output write pattern.

## Kernel Count
- Oracle: 1 kernel (mask-fold + softmax + row-guard + dropout + output transpose)
- Inductor: 1 kernel (same computation, natural output layout)

## Config Exploration
| Config | Result |
|--------|--------|
| combo_kernels + CDT | 194.34 us (1.067x) |

Single kernel produced regardless of config since Inductor already achieves full fusion.

## Status: near_floor

The 6.7% gap is minor and related to output layout epilogue. Inductor's fused persistent reduction already handles the softmax+dropout in a single pass. The oracle has a slight edge from writing the transposed output directly.

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: persistent reduction kernel
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`: constant folding could eliminate tautological iota>=0 mask

## Details
- Model: hf_DistilBertForMaskedLM (train)
- Shape: [256, 12, 128, 128] attention
- Pattern: iota>=0 (always true) mask -> add -> amax -> softmax -> all-masked row guard -> dropout -> permute
- Stochastic: yes (inductor_random dropout)
- The iota>=0 mask is tautological and could be constant-folded away
