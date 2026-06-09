# var_mean_68f956ca0908

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_68f956ca0908/oracle_bn_silu_cat.py`
- Correctness: PASS
- Oracle: `9.79 us`
- `torch.compile coordinate_descent_tuning=True`: `15.36 us`
- Ratio: 1.569x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT training BatchNorm-SiLU-channel-cat scope in one per-channel Triton program, including population `var_mean` over `[128,8,8]`, `rsqrt(var + 1e-5)`, in-place running-stat `copy_` updates, affine normalizatio

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.569x
- multi_kernel=2: 1.409x
- multi_kernel=3: 1.414x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
