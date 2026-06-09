# var_mean_cf1ffb1a79d8

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_cf1ffb1a79d8/oracle_avgpool_bn_train_relu.py`
- Correctness: PASS
- Oracle: `12.29 us`
- `torch.compile coordinate_descent_tuning=True`: `22.21 us`
- Ratio: 1.807x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle folds the fixed 2x2 stride-2 avg_pool2d into the channel training-BatchNorm reduction, updates the running mean/variance copy_ outputs in place, returns invstd and mean side outputs, and writes the affine ReLU tensor in one Triton launch

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.807x
- multi_kernel=2: 1.680x
- multi_kernel=3: 1.801x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
