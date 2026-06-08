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
| Oracle | ~24.3 | 1.000 |
| torch.compile (cd=True, combo) baseline | ~28.1 | 1.154 |
| torch.compile (cd=True, combo, any_elim) | ~27.5 | 1.13 |
| torch.compile (cd=True, combo, mk=2) | ~33.1 | 1.290 |
| torch.compile (cd=True, combo, mk=3) | ~30.1 | 1.173 |

Best compile config: default (cd=True, combo, any_elim) at ~27.5us.
Ratio with fix: **~1.13x** (was 1.154x)

Correctness: PASS (shape=[1024, 128, 128] f32, max_diff=2.98e-08)

## Kernel Count

- Oracle: 1 kernel (fused mask-softmax with stride-zero mask optimization)
- Inductor: 1 kernel (fused persistent, same kernel count)

## Root Cause

Both produce 1 kernel. The gap comes from:
1. The oracle exploits stride-zero mask as a row-level predicate, completely
   skipping score loads and all computation for fully-masked rows.
2. Double-exp computation: prepare_softmax_twopass_fallback computes exp inside
   the reduction (for sum_exp) then recomputes exp in the epilogue (for output).
3. The oracle uses tl.exp2 (native PTX) vs libdevice.exp.

## Fix Applied

**masked_softmax_any_elimination** pass (commit 2247db0f5ab in /tmp/pytorch-work):
- Eliminates the redundant any() reduction by recognizing that when mask has
  stride-zero along the reduction dim, `any(x != -inf, dim)` trivially equals
  the mask value itself.
- Saves one warp-shuffle reduction per row in the fused persistent kernel.
- Consistent ~0.6us improvement on compile time (28.1us -> 27.5us).

File: `/tmp/pytorch-work/torch/_inductor/fx_passes/masked_softmax_any_elimination.py`
Config: `torch._inductor.config.masked_softmax_any_elimination = True` (default)

## Remaining Gap Analysis

Same as any_amax_sum_7a4583e34c43. The remaining ~13% gap requires:
1. **Double-exp elimination** in persistent reduction epilogue
2. **Conditional row-skip** for fully-masked rows
3. **exp2 usage** instead of libdevice.exp

## Status: PARTIAL_FIX - any-reduction eliminated, remaining gap from double-exp and conditional skip
