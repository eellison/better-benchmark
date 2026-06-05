# var_mean_f3432607acca

## Classification: ALGEBRAIC_ELIMINATION

## Current Result (FIXED)

- Oracle path: `repros/canonical/var_mean_f3432607acca/oracle_identity_dropout_layernorm_permute.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: 19.14 us
- Compile BEFORE fix: 27.94 us (ratio 1.450)
- Compile AFTER fix: 18.14 us (ratio 0.948, Inductor beats oracle)
- Status: **FIXED** (BAD_ORACLE - Inductor now faster)

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 27.94 | 1.450 |
| combo+mk=2 | 27.90 | 1.441 |
| combo+mk=3 | 28.00 | 1.456 |
| Oracle | 19.26 | 1.000 |

No config closes the gap.

## Root Cause

The oracle computes the complete Longformer training residual LayerNorm scope, including:
- [8192,768] -> [8,1024,768] view
- Seed-index-32 degenerate `rand > 1e-30` dropout elimination (folded to identity)
- Residual add
- Population var_mean over hidden size 768, eps=1e-5
- Affine epilogue
- Final `permute(1,0,2).clone().view([8192,768])` layout
- rsqrt/768 side output

Inductor keeps the input-seeded RNG, comparison, and no-op multiply live through normalization scheduling and then handles the sequence-major output layout generically. Its algebraic simplifier does not prove near-zero-threshold unit-scale Inductor dropout masks are identity before norm-template and layout codegen.

## Kernel count
- Oracle: 1 kernel (residual LN + affine + permute layout + side output)
- Inductor: 2-3 kernels (RNG/dropout + norm + layout)

## Fix Implemented

Same fix as var_mean_81e1858d9aa4.
Commit: `84a2a5cf6ae` on branch `pr-184905` in `/tmp/pytorch-work`
Pass: `torch/_inductor/fx_passes/degenerate_dropout_elimination.py`
Config: `torch._inductor.config.degenerate_dropout_elimination = True` (default enabled)

After eliminating the degenerate dropout, Inductor's normal scheduling handles the
residual LayerNorm + permute layout efficiently enough to beat the oracle.
