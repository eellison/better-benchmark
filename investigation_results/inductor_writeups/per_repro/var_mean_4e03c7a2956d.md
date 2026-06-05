# var_mean_4e03c7a2956d

## Classification: COOPERATIVE_SPLIT_K

## Current Result

- Oracle path: `repros/canonical/var_mean_4e03c7a2956d/oracle_bn_silu_pad.py`
- Correctness: PASS
- Oracle: `65.25 us`
- `torch.compile coordinate_descent_tuning=True`: `95.1 us`
- Ratio: 1.458x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete EfficientNet training-BatchNorm plus affine SiLU and constant-pad scope by splitting the per-channel population var_mean reduction over N*H*W, finalizing mean/invstd and the two running-stat copy_ updates, then wr

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.458x
- multi_kernel=2: 1.458x
- multi_kernel=3: 1.457x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires a cooperative split-K reduction strategy for the channel statistics.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
