# var_mean_525fd66a1241

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_525fd66a1241/oracle_bn_skip_shuffle.py`
- Correctness: PASS
- Oracle: `30.46 us`
- `torch.compile coordinate_descent_tuning=True`: `39.81 us`
- Ratio: 1.307x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Shufflenet training-BatchNorm plus skip-branch channel shuffle scope, including the BN var_mean reduction, running mean/variance copy_ side effects, affine ReLU epilogue, cat/view/permute/clone/view layout transform,

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.307x
- multi_kernel=2: 1.267x
- multi_kernel=3: 1.269x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
