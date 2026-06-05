# pointwise_9d609b482adf

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/pointwise_9d609b482adf/oracle_fused_bn_silu_pad.py`
- Correctness: PASS
- Oracle: `50.91 us`
- `torch.compile coordinate_descent_tuning=True`: `81.7 us`
- Ratio: 1.605x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete per-channel BN-inference affine, SiLU-style divide by exp denominator, and right/bottom constant-pad store in one layout-aware Triton kernel, whereas Inductor currently lowers this repro as a producer pointwise kernel feeding a separate pad stencil consumer.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.605x
- multi_kernel=2: 1.606x
- multi_kernel=3: 1.607x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
