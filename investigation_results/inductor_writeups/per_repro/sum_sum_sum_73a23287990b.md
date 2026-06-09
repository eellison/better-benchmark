# sum_sum_sum_73a23287990b

## Summary

- Model: timm_beit_base_patch16_224_train
- Oracle: `oracle_relative_position_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.494x (oracle 502us, compile 750us)
- Kernel count: Inductor 4 kernels, Oracle 1 kernel

## Root Cause

The repro computes 12 repeated branches of:
`[128, 12, 197, 197] -> sum(dim=0) -> slice_scatter -> constant_pad_nd -> squeeze -> permute -> reshape -> index_put(accumulate=True) -> [732, 12]`

Inductor lowers each branch as a separate batch reduction kernel followed by a scatter kernel, totaling many kernel launches. The oracle fuses the batch sum and scatter-reduce into a single Triton kernel that:
1. Reduces [128, 12, 197, 197] over the batch dimension (dim=0) in tiles
2. Directly atomically accumulates the resulting [197, 197, 12] partial sums into the [732, 12] output buckets using the relative position index

The key insight is that the slice_scatter/constant_pad_nd/squeeze/permute/reshape chain between the reduction and the index_put is algebraically a no-op view transformation -- it just reinterprets the shape without moving data. The oracle recognizes this and folds it into the scatter addressing.

## What Inductor Cannot Do Today

1. **Fuse reduction + scatter**: The scheduler does not recognize that a `sum(dim=0)` feeding into `index_put(accumulate=True)` can be fused into a single atomic scatter-reduce kernel.
2. **Fold view chain**: The `slice_scatter -> pad -> squeeze -> permute -> reshape` sequence is lowered as real operations rather than being recognized as a structured no-op feeding a scatter.
3. **Batch sibling branches**: Inductor cannot recognize that 12 identical branches with different inputs/indices can share a single kernel dispatch.

## Fix Path

**SCATTER_REDUCE lowering**: Add an FX pass or scheduler pattern that detects:
- `sum(dim=batch_dim)` -> view chain -> `index_put(accumulate=True)`
- Lowers this to a fused scatter-reduce kernel using atomic accumulation

**Relevant files:**
- `torch/_inductor/fx_passes/post_grad.py`: Pattern matching for scatter-reduce fusion
- `torch/_inductor/scheduler.py`: Score function for reduction-scatter fusion
- `torch/_inductor/lowering.py`: index_put lowering to expose scatter semantics

## Config Exploration

Standard configs (combo_kernels, coordinate_descent_tuning) do not close this gap because the fundamental issue is missing fusion between reduction and scatter operations, not tuning within existing kernels.

## Design Doc

The fix requires a new structured lowering that:
1. Detects zero-fill + reduction + view-chain + index_put(accumulate) pattern
2. Generates a Triton kernel that tiles the reduction dimension, computes partial sums, and atomically accumulates into output buckets
3. Optionally batches identical sibling branches into one grid launch

This pattern appears in BEiT relative position bias backward (12 attention heads x 12 layers = 144 instances per model). High impact if fixed.
