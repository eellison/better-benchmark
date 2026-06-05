# var_mean_bd299b8b72c8

## Classification: NEW_PATTERN

## Current Result

- Oracle path: `repros/canonical/var_mean_bd299b8b72c8/oracle_gptneo_embedding_layernorm_mask.py`
- Correctness: PASS
- Oracle: `18.27 us`
- `torch.compile coordinate_descent_tuning=True`: `19.26 us`
- Ratio: 1.054x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo token embedding plus generated position embedding, hidden-size-2048 LayerNorm affine epilogue, three aliasing [4096,2048] views, and all-false adjacent-position mask in one Triton row kernel, whereas Inductor curr

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.054x
- multi_kernel=2: 1.002x
- multi_kernel=3: 1.049x
- **multi_kernel=2 CLOSES the gap** (ratio drops below 1.05x)
- coordinate_descent_tuning=True already enabled in baseline

## Status: resolved_by_config (multi_kernel=2)

## Fix Direction

This requires a specialized template for this embedding/norm pattern.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
