# sum_sum_sum_dbc10f22635c

## Summary

- Model: ConvNeXt-style multi-output reduction (scatter/expand/div/sum tail)
- Oracle: `oracle_multi_output.py`
- Classification: ALGEBRAIC_ELIMINATION
- Ratio: 1.325x (oracle 10.05us, compile 13.31us)
- Kernel count: Inductor multiple kernels, Oracle 1 fused single-pass kernel

## Root Cause

The oracle computes the complete scope with one Triton pass over the original inputs, including the two row-local reductions that feed the final per-channel reduction and the full/scatter/expand/div/sum tail folded into a direct accumulation over the pre-expanded values.

Inductor lowers the decomposed graph as generic pointwise, sibling reductions, and the scatter/expand reduction tail separately. It does not prove that the dense as_strided_scatter plus zero-stride expand divided by 49 and reduced over the expanded dimensions is equivalent to reducing the source values directly.

The 1.325x gap comes from:
1. Not eliminating the scatter/expand/div/sum tail via algebraic simplification
2. Separate kernels for dependent row reductions and sibling channel reductions
3. Cannot fuse the multi-output plan into one pass

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 13.31 | 1.325x |
| multi_kernel=2 | error (multi_kernel cache_file_path) | N/A |
| multi_kernel=3 | error (multi_kernel cache_file_path) | N/A |

multi_kernel=2/3 both error for this workload. Default CDT is the only working config.

## Fix Assessment

**Design doc** -- requires ALGEBRAIC_ELIMINATION of dense scatter-expand-reduce tail.

### What's needed:
Add shape/stride-aware elimination of the dense scatter-expand-reduce tail and lower the remaining dependent multi-output reduction as one plan. The simplifier needs to prove that `as_strided_scatter + zero-stride expand / N + sum(expand_dims)` is equivalent to dividing the source by N directly.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (scatter-expand-reduce canonicalization)
- `torch/_inductor/scheduler.py` (multi-output reduction plan)
