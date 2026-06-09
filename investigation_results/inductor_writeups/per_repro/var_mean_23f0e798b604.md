# var_mean_23f0e798b604

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_23f0e798b604/oracle_weight_standardization.py`
- Correctness: PASS
- Oracle: `9.98 us`
- `torch.compile coordinate_descent_tuning=True`: `12.42 us`
- Ratio: 1.244x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full NFNet weight-standardization scope in one Triton kernel, reading the original strided `[768,128,3,3]` weight, reducing mean/variance over the reshaped `[1,768,1152]` logical rows, applying `rsqrt(var + eps) * gain * scal

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.244x
- multi_kernel=2: 1.183x
- multi_kernel=3: 1.195x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
