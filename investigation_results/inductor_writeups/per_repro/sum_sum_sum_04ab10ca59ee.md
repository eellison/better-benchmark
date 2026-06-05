# sum_sum_sum_04ab10ca59ee

## Classification: SCATTER_REDUCE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_04ab10ca59ee/oracle_mt5_embedding_scatter_reduce.py`
- Correctness: PASS
- Oracle: `242.66 us`
- `torch.compile coordinate_descent_tuning=True`: `462.78 us`
- Ratio: 1.907x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full MT5 dual RMSNorm/dropout backward tuple by copying the live vocabulary-gradient base once, computing each row producer once per branch, accumulating both hidden-column reductions, and directly adding valid indexed contributions.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.907x
- multi_kernel=2: 1.920x
- multi_kernel=3: 1.922x
- multi_kernel does NOT help (gap persists)
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires a scatter-reduce optimization to avoid materializing full dense intermediates.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
