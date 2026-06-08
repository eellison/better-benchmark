# var_mean_7b0bd4f35599

## Classification: ALGEBRAIC_ELIMINATION (FIXED)

## Current Result

- Oracle path: `repros/canonical/var_mean_7b0bd4f35599/oracle_dropout_layernorm_transpose.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: 29.38 us
- Compile BEFORE fix: 47.94 us (ratio 1.63x)
- Compile AFTER fix: 29.38 us (ratio 1.0x)
- Status: **FIXED** (AT_FLOOR)

## Root Cause

The oracle algebraically eliminates the captured `rand > 1e-30` dropout mask and `* 1.0` scale before computing the complete residual LayerNorm transpose plus `rsqrt/768` side output in one row Triton kernel. Inductor previously carried the input-seeded RNG, comparison, and no-op dropout multiply through the normalization and transpose schedule.

## Fix Implemented

Commit: `13cbc75b516` on branch `pr-184905` in `/tmp/pytorch-work`
Pass: `torch/_inductor/fx_passes/degenerate_dropout_elimination.py`
Config: `torch._inductor.config.degenerate_dropout_elimination = True` (default enabled)

The pass recognizes the pattern:
```
rand = inductor_random([shape], seed, 'rand')
mask = gt.Scalar(rand, threshold)   # threshold < 1e-10
masked = mul.Tensor(mask, value)
result = mul.Tensor(masked, 1.0)
```
and replaces `result` with `value` directly.

## Kernel count
- Oracle: 1 kernel (residual LN + affine + transpose layout + side output)
- Inductor (before): 2-3 kernels (RNG/dropout + norm + layout)
- Inductor (after): 1 kernel (same as oracle)

## Config exploration results
- Baseline (before fix): 1.63x
- After fix (cd=True, combo): 1.0x (AT_FLOOR)

## Affected repros (12 total with 1e-30 pattern)
- var_mean_7b0bd4f35599: 1.63x -> 1.0x (FIXED)
- var_mean_81e1858d9aa4: 1.76x -> 1.037x (FIXED)
- var_mean_f3432607acca: 1.45x -> 1.09x (FIXED)
- var_mean_4df40c420c41: 1.63x -> BAD_ORACLE (FIXED)
- var_mean_c48b435e6a2b: -> 1.02x (FIXED)
- var_mean_550595226896: -> BAD_ORACLE (FIXED)
- var_mean_1c62090e8573: -> BAD_ORACLE (FIXED)
- var_mean_8287d162bd36: 1.68x -> 1.32x (PARTIAL - remaining gap from complex64 cast)

## File References
- `/tmp/pytorch-work/torch/_inductor/fx_passes/degenerate_dropout_elimination.py`
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (pass registration)
- `/tmp/pytorch-work/torch/_inductor/config.py` (config flag)
