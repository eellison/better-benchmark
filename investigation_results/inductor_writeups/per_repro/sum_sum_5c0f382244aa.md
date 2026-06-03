# sum_sum_5c0f382244aa

## Classification: COOPERATIVE_SPLIT_K

## Oracle: oracle_cooperative_split_k.py

## Model: torchbench_densenet121_train_001

## Measurements

- Compiled (default): 43.8 us
- Compiled (cd): 37.7 us
- Oracle (cooperative split-K): 34.8 us
- Oracle correctness: PASS (atol=2e-2, rtol=1e-2)

## Diagnosis

The oracle uses a two-phase cooperative split-K approach for DenseNet batch-norm-backward channel
summaries over a shared (N=64, H=14, W=14) reduction domain with C=448 channels:

1. **Phase 1** (`_masked_bn_dual_reduce_split_k_kernel`): Tiles the N*H*W=12544 spatial dimension
   across K-blocks (BLOCK_K=512), computing partial sums for `sum(where_self)` and
   `sum(where_self * centered)` per channel, accumulated via `tl.atomic_add` into C-sized output
   buffers. This gives full spatial parallelism across the reduction domain.

2. **Phase 2** (`_final_sum_and_slice_epilogue_kernel`): Reads the finalized channel summaries,
   computes `sum_where_centered * rsqrt` for the 448-element sum output, and simultaneously
   fuses the BN input-gradient computation for the last-32-channel slice (channels 416:448) with
   the 18 upstream gradient slice additions, writing both outputs in a single pass.

Inductor currently schedules the sibling channel reductions, the masked BN-backward producer, and
the sliced add epilogue as separate generic reduction and pointwise kernels over materialized
intermediates. The gap (1.08x with cd tuning, 1.26x with default) comes from:
- Materializing intermediate where_self and centered tensors to DRAM
- Running separate reduction kernels for the two channel sums
- Separate pointwise kernel for the 18-way slice addition + BN gradient epilogue

The oracle avoids all intermediate materialization by fusing everything into two cooperative kernels.

## Gap Analysis

- Best compile (cd): 37.7 us
- Oracle: 34.8 us
- Gap: 1.08x (small but real)
- Default compile gap: 1.26x

The gap is modest with coordinate_descent_tuning enabled. The oracle demonstrates the value of
cooperative split-K for this shape (large spatial reduction, moderate channel count), but Inductor
with cd tuning already gets close. This is a lower-priority COOPERATIVE_SPLIT_K case.

## Inductor Fix Path

- Teach the scheduler to recognize compatible multi-output BN-backward reductions (sum + sum*centered)
  sharing the same (N,H,W) reduction domain and fuse them with atomic split-K accumulation.
- Fuse the dependent sliced epilogue (18 upstream slice additions + BN gradient) into the
  finalization pass rather than scheduling it as a separate pointwise kernel.
- Gate on shape: beneficial when reduction_size (N*H*W) >> output_size (C) and multiple compatible
  reductions share the same domain.

## Status

COOPERATIVE_SPLIT_K -- needs_work (1.08x gap with cd, 1.26x gap default)

## Done Criteria

- Inductor fuses sibling BN-backward reductions into cooperative split-K with epilogue fusion.
- Gap closes to <1.05x vs oracle, or documented as shape-specific limitation.
