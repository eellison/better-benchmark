# sum_sum_sum_cd47d7ca4703

## Summary

- Model: Longformer embedding backward (scatter-reduce)
- Oracle: `oracle_longformer_embedding_backward_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.718x (oracle 79.74us, compile 136.99us)
- Kernel count: Inductor multiple kernels, Oracle fused multi-destination scatter-reduce

## Root Cause

The oracle computes the complete Longformer backward-like scope, including both hidden-column reductions and all three duplicate-index accumulate=True embedding scatter-add outputs with the captured mask/where semantics, by emitting the row math once, accumulating hidden-column reductions, and atomically scattering each masked embedding gradient directly into its destination row.

Inductor schedules the rowwise producer, reductions, masks, and generic index_put scatter-adds as separate kernels, materializing large intermediates.

The 1.718x gap comes from:
1. Materializing the rowwise producer for separate consumption
2. Three separate scatter kernels for position/segment/vocab tables
3. Cannot keep row-local hidden reductions live while feeding multiple sibling reductions and indexed accumulator destinations

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 136.99 | 1.718x |
| multi_kernel=2 | 134.30 | 1.685x |
| multi_kernel=3 | 134.10 | 1.682x |

multi_kernel=2/3 provide negligible improvement (~2%). The gap is purely structural.

## Fix Assessment

**Design doc** -- requires SCATTER_REDUCE multi-destination embedding-backward lowering.

### What's needed:
Add a multi-destination embedding-backward scatter-reduce lowering that emits the row math once, accumulates hidden-column reductions, and atomically scatters each masked embedding gradient directly into its destination row.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (embedding scatter-reduce pattern)
- `torch/_inductor/scheduler.py` (fusion across scatter boundaries)
- `torch/_inductor/codegen/triton.py` (multi-destination atomic accumulation)
