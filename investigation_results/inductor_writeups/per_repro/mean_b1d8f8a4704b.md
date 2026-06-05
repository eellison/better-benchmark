# mean_b1d8f8a4704b

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/mean_b1d8f8a4704b/oracle_nfnet_gelu_spatial_mean_f32.py`
- Correctness: PASS
- Oracle: `27.62 us`
- `torch.compile coordinate_descent_tuning=True`: `36.61 us`
- Ratio: 1.326x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet exact-erf GELU scale and 6x6 spatial mean for f32 [128,3072,6,6] in one Triton reduction kernel that writes the final contiguous f32 [128,3072] view directly, whereas Inductor currently lowers the activation through separate scheduling.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.326x
- multi_kernel=2: 1.295x
- multi_kernel=3: 1.313x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
