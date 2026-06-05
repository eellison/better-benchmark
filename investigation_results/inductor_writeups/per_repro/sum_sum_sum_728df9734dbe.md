# sum_sum_sum_728df9734dbe

## Summary

- Model: GPT-J layernorm-backward (multi-output reduction)
- Oracle: `oracle_multi_output_reduction.py`
- Classification: MULTI_OUTPUT_SHARED_REDUCTION
- Ratio: 1.148x (oracle 11.49us, compile 13.18us)
- Kernel count: Inductor multiple kernels, Oracle 1 fused two-phase layernorm-backward kernel

## Root Cause

The oracle computes the full GPT-J layernorm-backward multi-output region with row-local reductions feeding the transpose side output and three column reductions in one kernel.

Inductor schedules the row reductions, transpose materialization, and later vector reductions as separate generic kernels. It cannot build a dependent multi-output reduction plan that reuses row-reduction scalars while materializing a side-layout output and reducing it by column.

The 1.148x gap comes from:
1. Separate kernel for row reductions vs column reductions
2. Materializing the transposed intermediate between phases
3. Cannot reuse row-reduction scalars across side-output store and column reduction

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 13.18 | 1.148x |
| multi_kernel=2 | 24.38 | 2.122x (WORSE) |
| multi_kernel=3 | 24.61 | 2.142x (WORSE) |

multi_kernel=2/3 make things much worse. Default CDT is the best config.

## Fix Assessment

**Design doc** -- requires SCHEDULER_FUSION enhancement for coordinated two-phase layernorm-backward reduction template.

### What's needed:
Teach Inductor to form a coordinated two-phase layernorm-backward reduction template with fused side-output stores and sibling column reductions. The scheduler needs a dependent multi-output reduction plan that reuses row-reduction scalars while materializing a side-layout output and reducing it by column.

### Files:
- `torch/_inductor/scheduler.py` (dependent multi-output reduction plan)
- `torch/_inductor/codegen/triton.py` (two-phase reduction codegen)
