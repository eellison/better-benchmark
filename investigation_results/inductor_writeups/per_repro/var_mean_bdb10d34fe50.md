# var_mean_bdb10d34fe50

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_bdb10d34fe50/oracle_whisper_layernorm.py`
- Correctness: PASS
- Oracle: `16.06 us`
- `torch.compile coordinate_descent_tuning=True`: `19.01 us`
- Ratio: 1.183x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Whisper residual-add, fp16 clamp/canonicalization, contiguous-clone-equivalent row source, fp32 population var_mean, eps=1e-5 affine LayerNorm, fp16 cast, and final `[rows, hidden]` view in one Triton row kernel that

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.183x
- multi_kernel=2: 1.250x
- multi_kernel=3: 1.255x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
