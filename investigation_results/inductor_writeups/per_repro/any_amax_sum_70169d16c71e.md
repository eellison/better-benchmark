# any_amax_sum_70169d16c71e

## Classification: MASKED_SOFTMAX_PATTERN

## Pattern

M2M100 masked attention softmax with real broadcast mask: [1024, 128, 128] scores
viewed as [64, 16, 128, 128], stride-zero broadcast [64, 1, 128, 128] bool mask
-> 0/-inf bias, any(eq(-inf)) all-masked-row guard, stable softmax (amax, sub,
exp, sum, div), zero fill for all-masked rows, expand, view to [1024, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 25.66 | 1.000 |
| torch.compile (cd=True, combo) | 29.41 | 1.146 |
| torch.compile (cd=True, combo, mk=2) | 33.10 | 1.290 |
| torch.compile (cd=True, combo, mk=3) | 30.09 | 1.173 |

Best compile config: default (cd=True, combo) at 29.41 us.
Ratio: **1.146x**

Correctness: PASS (shape=[1024, 128, 128] f32, max_diff=5.96e-08)

## Kernel Count

- Oracle: 1 kernel (fused mask-softmax with stride-zero mask optimization)
- Inductor: 1 kernel (fused persistent, same kernel count)

## Root Cause

Both produce 1 kernel. The gap comes from:
1. The oracle exploits the stride-zero mask structure (loads mask once per row
   group instead of per element)
2. Inductor's fused kernel handles the mask as a generic pointwise operation,
   redundantly loading/computing mask values
3. The any() reduction for all-masked-row detection is computed separately from
   the amax reduction - the oracle combines them in a single pass

The 1.15x gap represents the overhead of not recognizing the masked-softmax
as a semantic template with optimized mask handling.

## Fix Direction

An Inductor pattern for all-masked-safe attention softmax would:
1. Detect the any(eq(-inf)) + amax + sum + div pattern
2. Exploit stride-zero broadcast masks with efficient loading
3. Combine the any and amax reductions into a single pass

File: `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`

## Status: DESIGN_DOC - masked-softmax template pattern needed
