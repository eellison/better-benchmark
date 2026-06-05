# var_mean_e53e1ab6f33f

## Compile: 11.87us, Oracle: 9.12us, Gap: 1.302x

## Diagnosis: CAT_MATERIALIZATION_IN_BN_TRAINING

## Root cause: Same pattern as var_mean_e603cd44b9cd. The oracle computes the full DenseNet two-input channel-concat training-BatchNorm-ReLU scope directly from the concat sources, including per-channel var_mean, invstd and saved-mean side outputs, in-place running mean/variance copy_ returns, and affine ReLU activation without materializing the logical cat.

Inductor generates a single fused kernel (triton_red_fused_add_cat_copy__mul_relu_rsqrt_squeeze_sub_unsqueeze_var_mean) that handles the cat inline, but with complex branch logic for selecting the correct source buffer within the reduction loop. The oracle uses a clean channel-tiled approach where each CTA knows which source tensor to read from.

## Kernel count
- Inductor: 1 kernel (red_fused_add_cat_copy__mul_relu_rsqrt_squeeze_sub_unsqueeze_var_mean)
- Oracle: 1 kernel (channel-tiled BN-train + cat from two sources + ReLU)

## Config exploration results
- multi_kernel=1 (default): 11.87us (ratio 1.302x)
- multi_kernel=2: 11.68us (ratio 1.267x) - no improvement
- multi_kernel=3: 11.68us (ratio 1.259x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: CAT_MATERIALIZATION_IN_BN_TRAINING

Same root cause as var_mean_e603cd44b9cd but with only 2 concat inputs instead of 4. The oracle's channel-tiled approach with direct source tensor indexing outperforms Inductor's generic branch-heavy cat inlining in the reduction loop.

## Fix path
Same as var_mean_e603cd44b9cd: channel-tiled BN-training codegen that resolves source tensor selection at the CTA level rather than in the inner reduction loop. File: `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`

## Status: design_doc
