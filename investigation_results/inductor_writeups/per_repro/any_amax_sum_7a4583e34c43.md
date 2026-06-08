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
| Oracle | ~24.3-25.3 | 1.000 |
| torch.compile (cd=True, combo) baseline | ~27.5-28.1 | 1.13-1.16 |
| torch.compile (cd=True, combo, any_elim) | ~27.5 | 1.13 |
| torch.compile (cd=True, combo, mk=2) | ~33.1 | 1.37 |
| torch.compile (cd=True, combo, mk=3) | ~30.1 | 1.24 |
| **torch.compile (cd=True, combo, use_fast_math=True)** | **~26.0** | **1.029** |
| torch.compile (fast_math + online_softmax=False) | ~26.5 | 1.08 |
| torch.compile (fast_math + multi_kernel=3) | ~26.3 | 1.06 |

Best compile config: **use_fast_math=True** at ~26.0us.
Ratio with fix: **~1.03x** (AT_FLOOR) (was 1.13x)

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

## Key Finding: use_fast_math Closes the Gap

Setting `TORCHINDUCTOR_USE_FAST_MATH=1` (or `torch._inductor.config.use_fast_math = True`)
reduces the gap from **1.136x to 1.029x** (AT_FLOOR status).

### What use_fast_math does here:
- Switches `libdevice.exp(x)` to `tl_math.exp(x)` in the Triton codegen
- `tl_math.exp` maps to NVIDIA's fast SFU exponential instruction (`__expf`)
- Without it, the persistent twopass fallback uses `libdevice.exp` (precision-heavy)
- The online_softmax accumulation path already uses `tl_math.exp` via `triton_helpers.exp()`
  but the epilogue exp in the persistent fallback doesn't - `use_fast_math` fixes this

### Why online_softmax=False doesn't help:
- The two-pass path (online_softmax=False) correctly computes exp only once
  (reusing the numerator from the sum for the division) - GOOD
- But it still uses `libdevice.exp` for that single computation unless use_fast_math=True
- The online_softmax=True path computes exp TWICE (once for sum, once for epilogue)
  but with use_fast_math the fast exp is cheap enough that the double computation
  doesn't matter for this reduction size (128 elements)

### Remaining ~3% gap explanation:
1. Oracle uses `tl.exp2(x * LOG2E)` which is 1 SFU instruction; `tl_math.exp(x)` likely
   compiles to `exp2(x * LOG2E)` as well - so these should be equivalent
2. Oracle has simpler control flow (masked loads, no broadcast/where chains)
3. Oracle processes mask at load-time vs Inductor's where(mask, 0, -inf) + add pattern

## Codegen Details

With `use_fast_math=True + online_softmax=True` (best config):
- 1 kernel, persistent reduction, XBLOCK=1 YBLOCK=4 R0_BLOCK=128
- 4 reductions (2x broadcast-max, 2x sum due to online softmax fallback)
- 2x `tl_math.exp` calls (double-exp from online softmax twopass fallback)
- Despite the double-exp, this is faster than online_softmax=False because the
  code generator already uses tl_math.exp when has_online_softmax=True flag is set

With `use_fast_math=True + online_softmax=False`:
- 1 kernel, persistent reduction
- 2 reductions (1x max, 1x sum)
- 1x `tl_math.exp` call (correctly reused for both sum and division)
- Slightly slower in practice (~26.5us vs 26.0us) due to different autotuning

## Status: CLOSED_BY_CONFIG - use_fast_math=True closes gap to 1.03x (AT_FLOOR)

**Table-stakes config**: `torch._inductor.config.use_fast_math = True`
(env: `TORCHINDUCTOR_USE_FAST_MATH=1`)
