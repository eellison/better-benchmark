# var_mean_f3f130d2784c

## Classification: SWIN_SHIFTED_WINDOW_LAYERNORM

## Current Result

- Oracle path: `repros/canonical/var_mean_f3f130d2784c/oracle_swin_dropout_layernorm_window.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: 41.92 us
- Compile (cd=True): 56.22 us
- Ratio: 1.341
- Best config: combo+mk=2: 55.36 us (ratio 1.320)
- Status: real_gap

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 56.22 | 1.341 |
| combo+mk=2 | 55.36 | 1.320 |
| combo+mk=3 | 56.13 | 1.340 |
| Oracle | 41.92 | 1.000 |

mk=2 helps slightly but gap remains significant.

## Root Cause

The oracle computes the complete Swin dropout-residual LayerNorm plus shifted-window partition scope, including:
- Seed-index 39 Inductor dropout
- fp32 hidden-size-512 population var_mean
- Affine LayerNorm
- Live rsqrt/512 side output
- Final contiguous [25088,512] window layout (via cyclic shift + partition)

Inductor materializes the RNG vector, stores mean/variance side buffers, writes an indexed [128,14,14,512] intermediate, then launches a separate layout clone. The normalization/pointwise scheduler does not sink the structured index plus window-partition clone into the normalization epilogue or inline the batchwise RNG producer.

## Kernel count
- Oracle: 1 kernel (fused dropout + LN + window partition)
- Inductor: 3+ kernels (RNG + norm + layout)

## Recommendation

Same family as var_mean_00824117c097. Requires a new guarded Swin shifted-window LayerNorm lowering that fuses seeded batch dropout, affine LayerNorm side outputs, advanced indexing, and fixed window-partition layout stores into one scheduled epilogue around the row-statistics kernel.

File: `torch/_inductor/scheduler.py` (norm template + window partition fusion)
File: `torch/_inductor/fx_passes/` (Swin window pattern matching)
