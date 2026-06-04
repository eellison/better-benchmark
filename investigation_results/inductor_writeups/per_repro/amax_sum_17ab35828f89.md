# amax_sum_17ab35828f89

## Compile: 59.5us, Oracle: 36.2us, Gap: 1.64x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor lowers the Swin relative-position-bias attention softmax as a generic fused reduction kernel that replays the indexed [169,32] bias-table lookup expression across the softmax loops instead of hoisting the window-invariant [32,49,49] bias and using a dedicated small-row persistent softmax template.

## Fix path: Add an Inductor lowering for Swin-style relative-position attention softmax that hoists or pre-materializes the bias-table lookup, fuses the score add and normalization, and uses a persistent small-row template for the 49-element rows.

## Status: design_todo

## Details

- Model: timm_swin_base_patch4_window7_224 (infer+train, 2 shapes)
- Pattern: amax+sum reduction (attention softmax with relative position bias)
- Ops: reshape x4, index, permute, clone, unsqueeze, add, amax, sub, exp, sum, div, expand
- Shape: [4096,49,49] -> [128,32,49,49] reshape, [169,32] bias gather via [49,49] index -> softmax -> [4096,49,49]
- Row length is only 49 elements -- a persistent kernel can keep everything in registers.
- The 1.64x gap is primarily from redundant recomputation of the bias expression and suboptimal block scheduling for small rows.
