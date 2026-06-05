# var_mean_3dd5772aac72

## Classification: BANDWIDTH_BOUND

## Current Result

- Oracle path: `repros/canonical/var_mean_3dd5772aac72/oracle_residual_layernorm_aliases.py`
- Correctness: PASS
- Oracle: `7.23 us`
- `torch.compile coordinate_descent_tuning=True`: `8.03 us`
- Ratio: 1.111x
- Status: gap detected

## Root Cause

Oracle for residual_layernorm_aliases pattern

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.111x
- multi_kernel=2: 0.992x
- multi_kernel=3: 0.977x
- **multi_kernel=3 CLOSES the gap** (ratio drops below 1.05x)
- coordinate_descent_tuning=True already enabled in baseline

## Status: resolved_by_config (multi_kernel=3)

## Fix Direction

This requires improved codegen to match the oracle's memory access pattern.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
