# pointwise_1f7516b9892a

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/pointwise_1f7516b9892a/oracle_relu_maxpool_mask.py`
- Correctness: PASS
- Oracle: `311.17 us`
- `torch.compile coordinate_descent_tuning=True`: `335.78 us`
- Ratio: 1.079x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete AlexNet training ReLU plus low-memory 3x3 stride-2 maxpool-with-offsets plus full ReLU<=0 mask scope by sinking ReLU into the stencil maxpool kernel and materializing the bool mask directly from the original input.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.079x
- multi_kernel=2: 1.073x
- multi_kernel=3: 1.079x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires scheduler-level fusion improvements to inline producers into reduction consumers.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
