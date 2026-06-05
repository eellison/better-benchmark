# sum_sum_sum_508eb468b8d9

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_508eb468b8d9/oracle_convnextv2_grn_full_scope.py`
- Correctness: PASS
- Oracle: `118.69 us`
- `torch.compile coordinate_descent_tuning=True`: `229.28 us`
- Ratio: 1.932x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 GRN backward-style scope from Repro.forward by folding the input add, the strided NCHW-to-NHWC logical loads, the broadcasted producer, the two per-pixel channel reductions, and all output stores into one kernel.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.932x
- multi_kernel=2: 1.943x
- multi_kernel=3: 1.933x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
