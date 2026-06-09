# pointwise_352e97c8de09

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/pointwise_352e97c8de09/oracle_rotary_layout_fusion.py`
- Correctness: PASS
- Oracle: `12.1 us`
- `torch.compile coordinate_descent_tuning=True`: `13.18 us`
- Ratio: 1.09x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full rotary embedding scope for Q and repeated KV into one layout-aware Triton kernel, preserving Q's non-contiguous permuted output stride while materializing the KV expand/clone/view output directly in contiguous repeated-head layout.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.09x
- multi_kernel=2: 1.099x
- multi_kernel=3: 1.091x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
