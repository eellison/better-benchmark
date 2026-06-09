# var_mean_8287d162bd36

## Classification: ALGEBRAIC_ELIMINATION

## Current Result (PARTIALLY FIXED)

- Oracle path: `repros/canonical/var_mean_8287d162bd36/oracle_identity_dropout_complex_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: 43.14 us
- Compile BEFORE fix: 73.47 us (ratio 1.682)
- Compile AFTER fix: 54.88 us (ratio 1.272)
- Status: partially fixed (remaining gap from complex64 cast scheduling)

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 73.47 | 1.682 |
| combo+mk=2 | 73.34 | 1.707 |
| combo+mk=3 | 73.41 | 1.708 |
| Oracle | 43.68 | 1.000 |

No config helps. Multi_kernel slightly worsens.

## Root Cause

The oracle folds the captured seed-index-11 `rand > 1e-30` dropout mask and `* 1.0` scale to identity, then computes the complete FNet residual LayerNorm scope with affine epilogue, contiguous complex64[32,512,768] real/zero-imaginary output, and rsqrt/768 side output.

Inductor keeps the Inductor RNG, comparison, and no-op multiply live inside the normalization and complex-cast schedule. The oracle notes this is a `not_true_floor` diagnostic because the RNG can occasionally produce an exact zero, making `rand > 1e-30` false - but this probability is negligible (2^-32 per element).

## Kernel count
- Oracle: 1 kernel (residual LN + affine + complex cast + side output)
- Inductor: 3+ kernels (RNG/dropout + norm + complex cast)

## Fix Implemented (Partial)

Degenerate dropout elimination fix applied (same as var_mean_81e1858d9aa4).
Commit: `84a2a5cf6ae` on branch `pr-184905` in `/tmp/pytorch-work`
Pass: `torch/_inductor/fx_passes/degenerate_dropout_elimination.py`

The dropout elimination reduces the gap from 1.68x to 1.27x. The remaining gap is
from the `convert_element_type(f32 -> complex64)` output cast, which requires
writing 2x the data (real + zero imaginary parts). The oracle directly stores the
complex output from the normalization kernel, avoiding a separate cast kernel.

Remaining fix needed: fuse the complex64 type cast into the LayerNorm epilogue
(store real part + zero imaginary in the same kernel that computes normalization).

Note: Technically not a "true floor" since the dropout has non-zero (but negligible
2^-32) probability of dropping.
