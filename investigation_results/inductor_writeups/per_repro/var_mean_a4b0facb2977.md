# var_mean_a4b0facb2977

## Classification: NEW_PATTERN

## Current Result

- Oracle path: `repros/canonical/var_mean_a4b0facb2977/oracle_embedding_layernorm.py`
- Correctness: PASS
- Oracle: `7.42 us`
- `torch.compile coordinate_descent_tuning=True`: `7.81 us`
- Ratio: 1.052x
- Status: gap detected

## Root Cause

Oracle for var_mean_a4b0facb2977

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete
RoBERTa embedding assembly and LayerNorm in one Triton row kernel, including the
word embedding lookup, cumsum/mask-derived position ids, token-type gather,
token-type and position embedd

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.052x
- multi_kernel=2: 1.030x
- multi_kernel=3: 1.051x
- **multi_kernel=2 CLOSES the gap** (ratio drops below 1.05x)
- coordinate_descent_tuning=True already enabled in baseline

## Status: resolved_by_config (multi_kernel=2)

## Fix Direction

This requires a specialized template for this embedding/norm pattern.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
