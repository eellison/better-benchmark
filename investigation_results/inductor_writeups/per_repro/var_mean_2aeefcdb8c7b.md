# var_mean_2aeefcdb8c7b

## Classification: BANDWIDTH_BOUND

## Current Result

- Oracle path: `repros/canonical/var_mean_2aeefcdb8c7b/oracle_weight_standardization.py`
- Correctness: PASS
- Oracle: `10.02 us`
- `torch.compile coordinate_descent_tuning=True`: `13.02 us`
- Ratio: 1.3x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full NFNet inference weight-standardization scope in one Triton kernel, reading the original strided `[768,128,3,3]` weight, reducing mean/variance over each reshaped `[1,768,1152]` row, applying `rsqrt(var + 1e-5) * gain * 0.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.3x
- multi_kernel=2: 1.214x
- multi_kernel=3: 1.204x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires improved codegen to match the oracle's memory access pattern.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
