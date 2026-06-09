# amax_sum_c79194c2980f

## Compile: 369.54us, Oracle: 339.65us, Gap: 1.088x

## Diagnosis: ONLINE_SOFTMAX_DROPOUT_EPILOGUE

## Root cause: The oracle fuses the complete T5 attention softmax/dropout (view [96,1024,1024] -> [8,12,1024,1024], fp32 bias add, stable last-dimension softmax, seed-index-119 Inductor RNG dropout, 1/(1-0.1) scale, expand/view, and final [96,1024,1024] transposed output layout via permute(0,2,1)) into a single persistent row-softmax Triton kernel. Inductor already fuses this into a single kernel (`triton_per_fused_add_div_exp_gt_mul_prepare_softmax_online_sub_view_0`) but the trailing permute(0,2,1) output layout change is handled separately or suboptimally, adding ~8.8% overhead vs the oracle which writes directly in the transposed layout.

## Kernel count
- Inductor: 1 kernel (fused softmax + dropout, natural output layout)
- Oracle: 1 kernel (softmax + dropout + output transpose in epilogue)

## Config exploration results
| Config | Compile (us) | Ratio |
|--------|-------------|-------|
| combo + CDT + multi_kernel=1 | 371.19 | 1.093x |
| combo + CDT + multi_kernel=2 | 392.26 | 1.155x |
| combo + CDT + multi_kernel=3 | 371.25 | 1.093x |

No config closes the gap. multi_kernel=2 (force persistent) is slightly worse.

## Classification: ONLINE_SOFTMAX_DROPOUT_EPILOGUE

Same mechanism as amax_sum_0f87f6568d0f (DeBERTa, 5.7% gap). The oracle sinks the output transpose into the kernel epilogue store, writing directly in the permuted layout. Inductor's codegen writes in the natural [BH, Q, K] layout and requires a separate or suboptimal copy for the trailing permute(0,2,1).

## Fix path
Extend Inductor's output layout scheduling to detect when a fused reduction kernel's sole consumer is a layout permute, and sink the permuted store indices into the kernel epilogue. File: `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (store emission), `/tmp/pytorch-work/torch/_inductor/scheduler.py` (output layout propagation).

Related repros: amax_sum_0f87f6568d0f (same pattern, DeBERTa 512x512), amax_sum_any_878cbcb8e8c2 (same T5 pattern but at floor)

## Status: design_doc

## Details
- Model: torchbench_hf_T5_base_train_000
- Pattern: view -> add -> amax -> sub -> exp -> sum -> div -> inductor_random -> gt -> mul -> mul -> expand -> view -> permute
- Shape: [96, 1024, 1024] attention (K=1024, 96K rows total)
- Stochastic: yes (inductor_random dropout with seed index 119)
- The gap is small (8.8%) and comes entirely from the trailing layout permute handling
