# sum_sum_sum_6a2ad206248c

## Summary

- Model: Electra embedding backward (scatter-reduce)
- Oracle: `oracle_embedding_backward_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 2.006x (oracle 89.86us, compile 180.22us)
- Kernel count: Inductor multiple kernels, Oracle fused scatter-reduce kernels

## Root Cause

The oracle computes the complete Electra backward-like expression by recomputing the fixed hidden-size row formula inside Triton kernels and feeding its results directly into the two hidden reductions, the sequence and type-id accumulate scatters, and the vocabulary embedding-gradient accumulate scatter added to the incoming gradient tensor.

Inductor lowers the decomposed graph through separate reductions, where materializations, index_put accumulations, and a final add. It cannot share a row reduction epilogue across multiple duplicate-index accumulate=True index_put users.

The 2.006x gap comes from:
1. Materializing the full intermediate producer tensor
2. Running separate kernels for each index_put scatter destination
3. Not recognizing zero-initialized index_put accumulations fed by row-reduction epilogues

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 180.22 | 2.006x |
| multi_kernel=2 | error (multi_kernel cache_file_path) | N/A |
| multi_kernel=3 | 169.60 | 1.888x |

multi_kernel=3 provides marginal improvement (~6%) but doesn't close the 2x gap. The structural scatter-reduce issue dominates.

## Fix Assessment

**Design doc** -- requires SCATTER_REDUCE embedding-backward lowering.

### What's needed:
Add a gather/reduce/scatter-aware lowering that recognizes zero-initialized index_put accumulations fed by row-reduction epilogues and emits direct atomic/reduction kernels for all sibling scatter and reduction outputs.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (scatter-reduce pattern)
- `torch/_inductor/scheduler.py` (fusion across scatter boundaries)
- `torch/_inductor/codegen/triton.py` (atomic accumulation codegen)
