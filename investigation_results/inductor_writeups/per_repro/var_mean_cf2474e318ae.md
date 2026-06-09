# var_mean_cf2474e318ae

## Classification: COOPERATIVE_SPLIT_K

## Current Result

- Oracle path: `repros/canonical/var_mean_cf2474e318ae/oracle_training_bn_silu_pad.py`
- Correctness: PASS
- Oracle: `59.1 us`
- `torch.compile coordinate_descent_tuning=True`: `106.4 us`
- Ratio: 1.8x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete EfficientNet training-BatchNorm, running-stat side effects, affine SiLU, and bottom/right zero-pad scope for f32 `[128,240,28,28]` using a split-K channel-statistics reduction plus a coalesced padded epilogue, whe

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.8x
- multi_kernel=2: 1.800x
- multi_kernel=3: 1.802x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires a cooperative split-K reduction strategy for the channel statistics.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
