# var_mean_var_mean_var_mean_5d12643e9b78

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_var_mean_var_mean_5d12643e9b78/oracle_quad_bn_relu_cat.py`
- Correctness: PASS
- Oracle: `112.45 us`
- `torch.compile coordinate_descent_tuning=True`: `126.82 us`
- Ratio: 1.128x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete four-branch Inception training-BatchNorm scope by grouping the sibling per-channel population var_mean reductions, eps=1e-3 affine ReLU epilogues, eight in-place running-stat copy_ outputs, and the final [128,768,17,17] channel-cat.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.128x
- multi_kernel=2: 1.127x
- multi_kernel=3: 1.128x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
