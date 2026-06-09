# var_mean_ba48651e0c04

## Classification: SWIN_SHIFTED_WINDOW_LAYERNORM

## Current Result

- Family: `swin_droppath_shift_layernorm`
- Oracle path: `repros/canonical/var_mean_ba48651e0c04/oracle_swin_droppath_shift_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `29.54 us`
- `torch.compile coordinate_descent_tuning=True`: `32.45 us`
- Ratio: 1.099
- Best config: `combo+mk=2`: `33.93 us`; default best at 32.45 us
- Status: `real_gap` (small)

## Diagnosis

The oracle computes the complete Swin window-reverse, iota-indexed cyclic shift, seed-index-8 drop-path residual LayerNorm scope in one shape-specialized Triton row kernel. It includes population var_mean over hidden=512, affine epilogue, final flatten, and rsqrt/512 side output, while fusing Inductor RNG (seeded random mask for drop-path) and nontrivial window-reverse index computation.

Inductor materializes the seeded random mask and shifted window-reverse producer around a generic var_mean normalization schedule.

## Config exploration results

| Config | Time (us) | Ratio vs oracle |
|--------|-----------|-----------------|
| Default (cd=True) | 32.45 | 1.099 |
| combo+mk=2 | 33.93 | 1.149 (worse) |
| combo+mk=3 | 38.45 | 1.302 (worse) |
| fast_math | 35.21 | 1.192 (worse) |
| Oracle | 29.54 | 1.000 |

No config closes the gap. Multi-kernel variants and fast_math actually make it worse. The default compile is already the best Inductor can do.

## Root cause

Inductor's scheduler treats Inductor RNG (seeded random for drop-path) and nontrivial window-reverse/index producers as materialization boundaries for normalization reductions. The oracle inlines all of these into a single fixed-hidden row reduction. The gap is small (~10%) because most of the work is the 512-wide reduction itself, not the producer materialization overhead.

This is the same SWIN_SHIFTED_WINDOW_LAYERNORM pattern seen in var_mean_00824117c097 and other Swin repros.

## Kernel count
- Oracle: 1 kernel (fused drop-path + window-reverse + LN + side output)
- Inductor: 2+ kernels (RNG/window-reverse materialized, then norm template)

## Recommendation

Same fix as other Swin shifted-window LayerNorm repros: the scheduler needs to allow fixed-shape seeded RNG and window-reverse/index producers to fuse into the LayerNorm template while preserving Inductor RNG offsets, iota indexing, and side-output semantics. Low priority given the small 10% gap.
