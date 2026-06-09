# var_mean_ac3c4a9016c7

## Classification: NEW_PATTERN

## Current Result

- Oracle path: `repros/canonical/var_mean_ac3c4a9016c7/oracle_bert_embedding_layernorm.py`
- Correctness: PASS
- Oracle: `27.52 us`
- `torch.compile coordinate_descent_tuning=True`: `32.51 us`
- Ratio: 1.181x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BERT word/token-type/position embedding assembly, fp32 hidden-size-768 population var_mean, eps=1e-12 affine LayerNorm, and three aliasing [16384, 768] view outputs in one fixed-hidden Triton row kernel, whereas Inductor

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 1.181x
- multi_kernel=2: 1.112x
- multi_kernel=3: 1.148x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires a specialized template for this embedding/norm pattern.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
