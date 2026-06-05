# sum_sum_ceb449ea29e4

## Classification: SCATTER_REDUCE

## Current Result

- Oracle path: `repros/canonical/sum_sum_ceb449ea29e4/oracle_llama_embedding_scatter_reduce.py`
- Correctness: PASS
- Oracle: `208.67 us`
- `torch.compile coordinate_descent_tuning=True`: `459.68 us`
- Ratio: 2.203x
- Status: gap detected

## Root Cause

Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Llama embedding/layernorm-backward tuple by copying the incoming bf16 vocabulary gradient once, then directly reducing the 2048 token-row contributors into that final output while also accumulating the sibling hidden reduction.

## Kernel Analysis

The oracle achieves better performance by fusing more operations into fewer kernel launches and/or using a more efficient reduction strategy than Inductor's generic scheduling.

## Config exploration results
- Baseline (multi_kernel=1): 2.203x
- multi_kernel=2: 2.211x
- multi_kernel=3: 2.181x
- multi_kernel partially helps but does not close the gap
- coordinate_descent_tuning=True already enabled in baseline

## Status: design_doc

## Fix Direction

This requires a scatter-reduce optimization to avoid materializing full dense intermediates.

## File References
- /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
- /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
- /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm templates)
