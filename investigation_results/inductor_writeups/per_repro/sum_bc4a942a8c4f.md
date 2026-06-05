# sum_bc4a942a8c4f — SCATTER_REDUCE

## Summary
- **Model**: torchbench_alexnet_train_001
- **Pattern**: MaxPool-backward scatter_add + boolean where + per-channel sum reduction
- **Oracle**: `oracle_maxpool_scatter_sum.py`
- **Ratio**: 1.351x (oracle 122.56us vs compile 165.63us)
- **Classification**: SCATTER_REDUCE

## Root Cause

The oracle avoids materializing the dense `[262144, 169]` scatter target entirely. Instead, it directly:
1. Iterates over the maxpool offset domain (per-channel, tiled into 4096-element blocks)
2. For each offset element, computes the target index from the low-memory maxpool offsets
3. Applies the boolean mask and accumulates into per-channel partial sums
4. Finalizes the partial sums in a tiny second kernel

Inductor materializes the full computation pipeline:
1. `triton_poi_fused_..._0`: Zero-initializes the dense `[262144, 169]` scatter target (44M elements of float32 = 176MB write)
2. `triton_poi_fused_..._1`: Computes maxpool offset-to-index conversion and performs scatter_add into the dense buffer
3. `triton_red_fused_sum_view_where_2`: Reads the dense buffer, applies where-mask, and reduces to `[256]` channel sums

The dense scatter buffer is 262144 x 169 = ~44M float32 values (176MB), which must be both written (zero-init) and read (for the sum). The oracle skips this entirely by reasoning about which elements would be non-zero in the scatter and directly computing their contribution to the final sum.

## Kernel Count
- **Oracle**: 2 kernels (partials + finalize)
- **Inductor**: 3 kernels (zero-init + scatter + where-reduce)

## Analysis

The 35.1% gap comes from:
1. **Eliminated memory traffic**: The oracle never materializes the 176MB dense scatter buffer. It directly maps offsets to their target positions and accumulates.
2. **Fewer kernel launches**: 2 vs 3 kernels
3. **Better data locality**: The oracle processes data per-channel in blocks, keeping the working set small

The fundamental issue is that Inductor's lowering of `scatter_add` always materializes the dense target buffer, even when the only consumer is a reduction that could be computed directly from the sparse source data. The optimizer does not recognize that `zero-init -> scatter_add -> where -> sum` can be algebraically simplified to a direct gather-mask-reduce over the source domain.

## Config Exploration
- `coordinate_descent_tuning = True` + `combo_kernels = True`: 165.63us (3 kernels)
- `triton.multi_kernel = 3`: no structural improvement (same kernel count)

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/lowering.py`: `scatter_add` lowering always creates dense target
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`: would need a new pass to detect scatter_add -> reduce pattern
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: cannot fuse across scatter_add materialization

## Design Doc

**Why Inductor cannot fix this today**: The `scatter_add` op is lowered to a dense buffer write. There is no FX pass that recognizes the pattern `full(0) -> scatter_add -> [view] -> where -> sum` as equivalent to a direct "gather-mask-reduce" over the source domain. This would require:

1. **Pattern recognition**: An FX pass in `post_grad.py` that identifies when a scatter_add's output is only consumed by a reduction (possibly with intervening view/where ops)
2. **Algebraic rewrite**: Replace the scatter_add + sum with a direct accumulation: for each source element, compute its target position, check the mask at that position, and accumulate into the output reduction
3. **Maxpool-specific lowering**: Recognize that `_low_memory_max_pool_offsets_to_indices` produces structured offsets that map 1:1 from source to scattered positions

**Needed enhancement**: Add a "scatter-reduce fusion" pass that detects `scatter_add(...) -> reduce(...)` chains and rewrites them into direct gather-reduce kernels that never materialize the intermediate dense buffer. This would save the 176MB round-trip and reduce kernel count.

**Affected repro count**: This pattern appears in multiple AlexNet/VGG maxpool-backward repros (sum_8a66186d1ffe shares the same oracle_maxpool_scatter_sum classification, though that one shows BAD_ORACLE on current hardware).
