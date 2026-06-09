# sum_sum_sum_3eab263249bb

## Summary

- Model: Swin Transformer relative-position-bias backward (scatter-reduce)
- Oracle: `oracle_relative_position_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.197x (oracle 97.34us, compile 116.51us)
- Kernel count: Inductor multiple kernels, Oracle 1 fused scatter-reduce kernel

## Root Cause

The oracle computes the full three-output Swin relative-position-bias backward with one vectorized Triton producer per `(channel, row, batch tile)` that emits the `[8192, 49, 49]` softmax-backward side output and atomically accumulates both tile batch-reduction partials into duplicate `[169, 16]` relative-position buckets.

Inductor lowers the two `sum(dim=0) -> permute/view -> index_put(accumulate=True)` branches and the `mul/sum/neg/fma/view` branch as separate generic reduction, layout, scatter, and pointwise kernels over materialized intermediates.

The 1.197x gap comes from:
1. Separate batch reduction kernels followed by separate scatter kernels
2. Not recognizing the duplicate-index relative-position `index_put(accumulate=True)` as structured scatter-reduce
3. Cannot share the rowwise softmax-backward producer with required side-output stores

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 116.51 | 1.197x |
| multi_kernel=2 | 135.17 | 1.389x (WORSE) |
| multi_kernel=3 | 134.27 | 1.380x (WORSE) |

multi_kernel=2/3 make things worse. Default CDT is the best config. The gap is from missing structured scatter-reduce fusion.

## Fix Assessment

**Design doc** -- requires SCATTER_REDUCE structured relative-position lowering.

### What's needed:
Add a structured relative-position scatter-reduce lowering that fuses batch reductions with indexed accumulation and emits any full-tensor side output from the same producer. The scheduler/codegen needs to recognize duplicate-index `index_put(accumulate=True)` as a structured scatter-reduce pattern.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (scatter-reduce pattern)
- `torch/_inductor/scheduler.py` (fusion across scatter boundaries)
- `torch/_inductor/codegen/triton.py` (atomic accumulation codegen)
