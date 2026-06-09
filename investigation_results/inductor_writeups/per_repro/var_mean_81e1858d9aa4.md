# var_mean_81e1858d9aa4

## Classification: ALGEBRAIC_ELIMINATION

## Current Result (FIXED)

- Oracle path: `repros/canonical/var_mean_81e1858d9aa4/oracle_dropout_elim_layernorm_side.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: 18.34 us
- Compile BEFORE fix: 32.38 us (ratio 1.760)
- Compile AFTER fix: 18.18 us (ratio 0.991)
- Status: **FIXED** (AT_FLOOR)

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 32.38 | 1.760 |
| combo+mk=2 | 28.13 | 1.529 |
| combo+mk=3 | 28.16 | 1.536 |
| Oracle | 18.40 | 1.000 |

mk=2 helps moderately but the gap remains very large.

## Root Cause

The oracle folds the captured seed-index-34 `rand > 1e-30` dropout mask and `* 1.0` scale to identity, then computes the complete Longformer training bias-add, residual LayerNorm, affine epilogue, final [8192,768] view, and rsqrt/768 side output in one fixed-hidden Triton row-normalization kernel.

Inductor keeps the seeded RNG/mask/mul producer live and schedules it around the norm template. Its algebraic simplifier does not canonicalize near-zero-threshold Inductor dropout masks (where the mask is `rand > 1e-30`, which is always true in practice) before normalization scheduling. This means extra computation (RNG + comparison + multiply by 1.0) runs for every element even though the dropout has no effect.

## Kernel count
- Oracle: 1 kernel (bias + residual + LN + affine + side output)
- Inductor: 2+ kernels (RNG/dropout + norm template)

## Fix Implemented

Commit: `84a2a5cf6ae` on branch `pr-184905` in `/tmp/pytorch-work`
Pass: `torch/_inductor/fx_passes/degenerate_dropout_elimination.py`
Config: `torch._inductor.config.degenerate_dropout_elimination = True` (default enabled)

The pass detects the pattern `gt.Scalar(inductor_random(..., 'rand'), threshold)` where
threshold < 1e-10, followed by `mul(mask, value) * 1.0`, and replaces it with just `value`.
This eliminates the unnecessary RNG + comparison + multiply operations, allowing the
LayerNorm template to receive the simplified bias/residual expression directly.
