# var_mean_d30f3928376b

## Classification: BANDWIDTH_BOUND

## Current Result

- Oracle path: `repros/canonical/var_mean_d30f3928376b/oracle_training_bn_silu.py`
- Correctness: PASS
- Oracle: `10.5 us`
- `torch.compile coordinate_descent_tuning=True`: `11.65 us`
- Ratio: 1.11x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete training-BatchNorm plus affine SiLU scope for f32 `[128,160,8,8]`, including per-channel `var_mean(correction=0)`, in-place running mean/variance `copy_` aliases, and the returned contiguous activation, using a stats/

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.11x
- multi_kernel=2: 0.909x
- multi_kernel=3: 0.901x
- **multi_kernel=3 CLOSES the gap** (ratio drops below 1.05x)
- coordinate_descent_tuning=True already enabled in baseline

## Status: resolved_by_config (multi_kernel=3)

## Fix Direction

This requires improved codegen to match the oracle's memory access pattern.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
