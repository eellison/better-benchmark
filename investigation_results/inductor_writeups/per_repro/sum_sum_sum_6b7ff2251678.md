# sum_sum_sum_6b7ff2251678

## Summary

- Model: ConvNeXt layernorm-backward (multi-output)
- Oracle: `oracle_multi_output.py`
- Classification: MULTI_OUTPUT_SHARED_REDUCTION
- Ratio: 1.694x (oracle 7.84us, compile 13.28us)
- Kernel count: Inductor multiple kernels, Oracle 1 fused row-local multi-output reduction

## Root Cause

The oracle computes the full ConvNeXt layernorm-backward-shaped graph in one row-local multi-output Triton reduction, including:
1. Reshape/permute pointwise work
2. Two per-row reductions
3. Two sibling channel reductions
4. The as_strided_scatter/expand/div path feeding the third channel reduction

Inductor emits separate kernels for the sibling channel reductions, zero-fill, row-local layernorm-backward pointwise store, and two-stage expanded view reduction.

The 1.694x gap comes from:
1. Materializing intermediates between row reductions and channel reductions
2. No single node for dependent row reductions that feed compatible channel reductions
3. Cannot keep row-local scalars in registers while accumulating all compatible channel outputs

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 13.28 | 1.694x |
| multi_kernel=2 | 30.08 | 3.837x (WORSE) |
| multi_kernel=3 | 28.16 | 3.592x (WORSE) |

multi_kernel=2/3 make things significantly worse for this small kernel -- the overhead of persistent/looped reduction strategies exceeds the benefit. Default CDT is the best config.

## Fix Assessment

**Design doc** -- requires SCHEDULER_FUSION enhancement for dependent multi-output reduction scheduling.

### What's needed:
Add dependent multi-output reduction scheduling that can keep row-local scalars in registers while accumulating all compatible channel outputs without materializing the structured intermediate. The scheduler needs to recognize that row-reduction results feed both a structured view-reduction consumer and sibling channel reductions.

### Files:
- `torch/_inductor/scheduler.py` (dependent multi-output reduction plan)
- `torch/_inductor/codegen/triton.py` (row-local scalar reuse codegen)
