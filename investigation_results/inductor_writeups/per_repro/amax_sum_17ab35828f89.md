# amax_sum_17ab35828f89

## Compile: 30.4us, Oracle: 34.4us, Gap: 0.88x (compile beats oracle)

**Previous**: Compile: 59.5us, Oracle: 36.2us, Gap: 1.64x

## Diagnosis: CLOSED by inline_recomputable_producers (f58d2545cd2)

## Root cause (historical): Inductor lowers the Swin relative-position-bias attention softmax as a generic fused reduction kernel that replays the indexed [169,32] bias-table lookup expression across the softmax loops instead of hoisting the window-invariant [32,49,49] bias and using a dedicated small-row persistent softmax template.

## Fix: inline_recomputable_producers extension

The extension to `inline_recomputable_producers` that handles cheap non-broadcast
producers closed this gap completely. Compile now beats the oracle (0.88x ratio).
Re-measured 2026-06-08.

## Status: closed

## Details

- Model: timm_swin_base_patch4_window7_224 (infer+train, 2 shapes)
- Pattern: amax+sum reduction (attention softmax with relative position bias)
- Ops: reshape x4, index, permute, clone, unsqueeze, add, amax, sub, exp, sum, div, expand
- Shape: [4096,49,49] -> [128,32,49,49] reshape, [169,32] bias gather via [49,49] index -> softmax -> [4096,49,49]
- Row length is only 49 elements -- a persistent kernel can keep everything in registers.
