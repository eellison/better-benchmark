# any_amax_sum_7a4583e34c43

## Classification: MASKED_SOFTMAX_PATTERN

## Pattern

MobileBERT masked attention softmax with real broadcast mask: [1024, 128, 128]
scores viewed as [256, 4, 128, 128], stride-zero broadcast [256, 1, 128, 128]
bool mask -> 0/-inf bias, any(eq(-inf)) all-masked-row guard, stable softmax,
zero fill for all-masked rows, expand, view to [1024, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | ~24.3 | 1.000 |
| torch.compile (cd=True, combo) baseline | ~28.1 | 1.155 |
| torch.compile (cd=True, combo, any_elim) | ~27.5 | 1.13 |
| torch.compile (cd=True, combo, mk=2) | ~33.1 | 1.37 |
| torch.compile (cd=True, combo, mk=3) | ~30.1 | 1.24 |

Best compile config: default (cd=True, combo, any_elim) at ~27.5us.
Ratio with fix: **~1.13x** (was 1.155x)

Correctness: PASS (shape=[1024, 128, 128] f32, max_diff=5.96e-08)

## Kernel Count

- Oracle: 1 kernel (fused mask-softmax with broadcast mask optimization)
- Inductor: 1 kernel (fused persistent)

## Root Cause

Both produce 1 kernel. The remaining gap comes from:
1. The oracle exploits stride-zero mask as a row-level predicate: loads one bool
   per row and skips ALL computation (loads, exp, reductions) for masked rows.
   Inductor always loads and processes all 128 elements per row.
2. Double-exp computation: prepare_softmax_online computes exp(x-max) inside the
   reduction for sum_exp, then the epilogue recomputes exp(x-max) for the output.
   The oracle computes exp only once using exp2 (native hardware instruction).
3. The oracle uses tl.exp2(x * LOG2E) which maps to a single PTX instruction,
   while Inductor uses libdevice.exp (software implementation).

## Fix Applied

**masked_softmax_any_elimination** pass (commit 2247db0f5ab in /tmp/pytorch-work):
- Eliminates the redundant `any(logical_not(eq(add(x, where(mask, 0, -inf)), -inf)), dim=-1)`
  reduction by recognizing that when mask has stride-zero along the reduction dim,
  the result is trivially equal to the mask value itself.
- Saves one warp-shuffle reduction in the fused persistent kernel.
- Improvement: ~0.6us on compile time (~2-3%).

File: `/tmp/pytorch-work/torch/_inductor/fx_passes/masked_softmax_any_elimination.py`
Config: `torch._inductor.config.masked_softmax_any_elimination = True` (default)

## Remaining Gap Analysis

The remaining ~13% gap requires architectural changes:
1. **Double-exp elimination**: The prepare_softmax_online pattern creates separate
   reduction (max+sum_exp) and epilogue (sub+exp+div) passes. In persistent mode,
   the exp computed inside the reduction could be cached and reused for the epilogue.
   File: `/tmp/pytorch-work/torch/_inductor/codegen/simd.py:1402` (prepare_softmax_twopass_fallback)
2. **Conditional row-skip**: The oracle checks the mask first and completely skips
   score loading and all computation for fully-masked rows. This would require
   predicated execution support in Triton codegen (if-else per row tile).
3. **exp2 vs libdevice.exp**: The oracle uses native exp2 hardware instruction.
   Inductor uses libdevice.exp unless use_fast_math=True.

## Status: PARTIAL_FIX - any-reduction eliminated, remaining gap from double-exp and conditional skip
