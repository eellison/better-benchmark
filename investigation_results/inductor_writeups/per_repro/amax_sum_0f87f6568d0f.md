# amax_sum_0f87f6568d0f

## Compile: 206.53us, Oracle: 195.36us, Gap: 1.057x

## Classification: ONLINE_SOFTMAX_DROPOUT_EPILOGUE

## Root Cause

The oracle fuses the complete DeBERTa masked attention softmax/dropout (view [192,512,512] -> [8,24,512,512], broadcast bool mask with scalar fill, stable last-dim softmax, seed-index-70 tl.rand dropout and scale, and final [192,512,512] transposed output view) into a single persistent row-softmax Triton kernel.

Inductor already fuses this into a single kernel (`triton_red_fused_div_exp_gt_inductor_lookup_seed_inductor_random_mul_prepare_softmax_online_sub_view_where_0`). The remaining 5.7% gap is from the trailing permute(0,2,1) layout change on the output that the oracle handles in the same kernel epilogue vs Inductor writing in the natural layout.

## Kernel Count
- Oracle: 1 kernel (softmax + dropout + output transpose)
- Inductor: 1 kernel (softmax + dropout, natural output layout)

## Config Exploration
| Config | Result |
|--------|--------|
| combo_kernels + CDT | 206.53 us (1.057x) |

All configs tested produce the same single-kernel plan since Inductor already fuses everything except the trailing layout epilogue.

## Status: near_floor

The 5.7% gap is marginal and comes from the output transpose layout. Inductor's single kernel is already very close to optimal. The oracle has a slight advantage by writing the transposed output directly in its kernel epilogue rather than requiring a separate copy.

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: the fused reduction kernel already includes the softmax+dropout
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion succeeds for the computation, but the output layout permute may trigger a separate copy

## Details
- Model: hf_DebertaV2ForMaskedLM (train)
- Shape: [8, 24, 512, 512] attention with mask
- Pattern: view -> where(mask) -> amax -> sub -> exp -> sum -> div -> dropout -> permute
- Stochastic: yes (inductor_random dropout)
