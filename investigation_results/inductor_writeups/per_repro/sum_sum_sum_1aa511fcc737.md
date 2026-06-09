# sum_sum_sum_1aa511fcc737

## Summary

- Model: RepVGG (chained batch-norm backward)
- Oracle: `oracle_full_scope_bn_chain.py`
- Classification: REDUCTION_EPILOGUE_REREAD
- Ratio: 1.184x (oracle 328.7us, compile 389.1us)
- Kernel count: Inductor 1 kernel (combo), Oracle 1 kernel (full-scope)

## Root Cause

The repro computes a chained RepVGG batch-norm-backward scope. The oracle shares the first channel reductions, materializes the masked residual producer once, shares the downstream sibling channel reductions, and emits both full tensor gradients plus all vector outputs in one integrated kernel.

Inductor also emits 1 combo kernel but it is less efficient because:
1. The first BN-backward epilogue computes channel sums, then the downstream BN-backward needs to re-read the full input to compute the dependent gradient
2. The sibling channel reductions (for the chained BN) are computed with suboptimal memory access patterns
3. The combo kernel doesn't share intermediate channel summaries across the two BN stages

The gap is moderate (1.184x) because combo_kernels does successfully fuse everything into one launch -- the remaining overhead is from suboptimal internal tiling and redundant memory traffic within the single kernel.

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT (default) | 389.1 |
| multi_kernel=2 | 507.9 |
| multi_kernel=3 | 852.9 |

No config helps. multi_kernel=2/3 are significantly worse, suggesting the combo kernel (default) is already the best available strategy. The remaining gap is within-kernel efficiency.

## Fix Assessment

**Design doc** -- requires codegen improvement for chained BN-backward patterns.

### What's needed:
The combo kernel codegen needs to recognize that when two BN-backward stages are chained (first BN output feeds second BN input), the channel summaries from the first stage can be kept in registers/shared memory and reused by the second stage without re-reading the intermediate tensor.

This is a codegen-level optimization within the existing combo kernel framework, not a scheduling change.

### Difficulty: Medium
The fix would involve detecting chained reduction patterns within a combo kernel and optimizing register reuse between sub-kernels.

## Relevant Files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: combo kernel sub-kernel codegen
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: combo kernel fusion scoring
