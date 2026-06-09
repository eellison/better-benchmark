# var_mean_a0eecc057fbc

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Oracle path: `repros/canonical/var_mean_a0eecc057fbc/oracle_layernorm_index.py`
- Correctness: PASS
- Oracle: `6.72 us`
- `torch.compile coordinate_descent_tuning=True`: `7.87 us`
- Ratio: 1.171x
- Status: gap detected

## Root Cause

Oracle for var_mean_a0eecc057fbc

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the
full Repro.forward contract by reducing all 64 rows of
`add_91 + view(addmm_47)` to produce the returned `rsqrt(var + eps) / 768`
tensor, while applying the affine LayerNorm epilogue onl

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.171x
- multi_kernel=2: 1.110x
- multi_kernel=3: 1.110x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires an algebraic simplification pass to eliminate dead/identity operations before scheduling.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
