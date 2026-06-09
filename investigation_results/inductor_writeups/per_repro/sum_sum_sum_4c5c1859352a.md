# sum_sum_sum_4c5c1859352a

## Summary

- Model: BN-backward style grouped reduction (add/le/where producer)
- Oracle: `oracle_multi_output_reduction.py`
- Classification: ALGEBRAIC_ELIMINATION
- Ratio: 1.464x (oracle 9.86us, compile 14.43us)
- Kernel count: Inductor multiple kernels, Oracle 1 fused grouped multi-output reduction

## Root Cause

The oracle computes the complete 3-output graph by streaming the added-and-masked producer once per `(batch, two-channel group)`, keeping `sum(where)` and `sum(where * arg52_1)` over each 8x8 spatial block in registers, and deriving the BN-backward-style full tensor epilogue plus both returned channel vectors from those summaries.

Inductor currently schedules the add/le/where producer, sibling spatial reductions, dependent grouped reductions, full tensor epilogue, and final channel reductions as separate generic regions over materialized intermediates.

The 1.464x gap comes from:
1. Materializing the intermediate masked tensor
2. Not recognizing the view-threaded dependent reduction chain
3. Cannot preserve the two base spatial summaries across all consumers

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 14.43 | 1.464x |
| multi_kernel=2 | 36.54 | 3.706x (WORSE) |
| multi_kernel=3 | 41.95 | 4.254x (WORSE) |

multi_kernel=2/3 dramatically worsen performance for this small kernel. Default CDT is the best config. The gap is purely structural.

## Fix Assessment

**Design doc** -- requires ALGEBRAIC_ELIMINATION rewrite pass.

### What's needed:
Add a guarded rewrite for this grouped BN-backward-style pattern that lowers the shared spatial summaries and dependent tensor/vector epilogues as one fused multi-output reduction plan. The algebraic simplifier needs to recognize the view-threaded dependent reduction chain and preserve the two base spatial summaries across all consumers.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (pattern recognition)
- `torch/_inductor/scheduler.py` (grouped multi-output reduction plan)
