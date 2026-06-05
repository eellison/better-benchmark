# sum_sum_sum_8ffe5090cf87

## Summary

- Model: ALBERT embedding backward (scatter-reduce)
- Oracle: `oracle_embedding_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.606x (oracle 25.98us, compile 41.73us)
- Kernel count: Inductor multiple kernels, Oracle fused scatter-reduce

## Root Cause

The oracle computes the complete ALBERT embedding-backward return tuple by fusing the shared rowwise hidden producer, both `[128]` sibling reductions, and all three duplicate-index `index_put(accumulate=True)` scatter-add outputs (512-row, 2-row, and 30000-row destinations).

Inductor materializes the `[8, 512, 128]` producer and schedules the two reductions plus the scatter outputs as separate generic kernels.

The 1.606x gap comes from:
1. Materializing the intermediate producer
2. Separate generic scatter kernels for each embedding table destination
3. Cannot share one row-local reduction producer across multiple indexed accumulation destinations

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 41.73 | 1.606x |
| multi_kernel=2 | error (multi_kernel cache_file_path) | N/A |
| multi_kernel=3 | error (multi_kernel cache_file_path) | N/A |

multi_kernel=2/3 both error for this workload. Default CDT is the only working config.

## Fix Assessment

**Design doc** -- requires SCATTER_REDUCE embedding-backward lowering.

### What's needed:
Add an embedding-backward scatter-reduce lowering that keeps the row algebra in registers, emits the hidden-column reductions, and atomically accumulates every indexed embedding-gradient output directly.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (scatter-reduce pattern)
- `torch/_inductor/scheduler.py` (fusion across scatter boundaries)
- `torch/_inductor/codegen/triton.py` (atomic accumulation codegen)
