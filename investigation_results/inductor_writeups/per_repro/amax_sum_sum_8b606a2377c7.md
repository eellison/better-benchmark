# amax_sum_sum_8b606a2377c7

## Classification: `SCHEDULER_FUSION`

## Pattern

ResNeSt split-attention backward tail: avg_pool2d_backward + BN-affine/ReLU gate +
radix-2 softmax backward (fma) + spatial reduction + channel reduction

- Model: torchbench_timm_resnest_train
- Shape: grad_pool f32[32, 128, 28, 28], bn_input f32[32, 256, 56, 56], logits f32[32, 256, 1, 1]
- Output: f32[256] (channel gradient)
- Oracle: `oracle_resnest_split_attention_backward.py`

## Measurements

| Metric | Value |
|--------|-------|
| Oracle | 96.13 us |
| Compile (best, CD tuning) | 109.41 us |
| Ratio | 1.138x |
| Kernel count (Inductor) | 3 |
| Kernel count (Oracle) | 2 |
| Status | GOOD |

## Diagnosis

### Oracle structure (2 kernels)

1. **`_partial_spatial_sums_kernel`** [grid: (13, 32, 128)]: For each tile of HW,
   each batch, and each group channel:
   - Computes avg_pool2d_backward inline (scatter-add from 28x28 pooled gradients)
   - Applies BN-affine + ReLU gate to the two radix branches
   - Multiplies pool_backward * relu and spatially reduces to partial sums
   - Stores `partial[tile, batch, radix, channel]`

2. **`_final_softmax_backward_reduce_kernel`** [grid: (128,)]: For each group channel:
   - Reduces partials across tiles and batches
   - Computes radix-2 softmax backward (exp, div, fma)
   - Reduces across batch dimension to produce final [256] output

### Inductor structure (3 kernels)

1. **`triton_poi_fused_avg_pool2d_backward_0`**: Pointwise kernel that computes the
   full avg_pool2d_backward into a f32[32, 128, 56, 56] intermediate (~51 MB)

2. **`triton_red_fused_..._sum_1`**: Reduction kernel that reads the materialized
   pool backward, applies BN-affine + ReLU, multiplies with the expanded pool backward,
   and reduces spatially to get [32, 2, 128, 1, 1] partial sums

3. **`triton_per_fused_..._softmax_..._sum_2`**: Persistent reduction that computes
   the radix-2 softmax backward and reduces over batch to [256]

### Root cause

The key inefficiency is **kernel 1 materializing the full avg_pool2d_backward
intermediate**. The oracle never writes this 51 MB buffer to HBM - it computes the
pool backward on-the-fly within each spatial tile of the second kernel. This saves
one full read+write of the [32, 128, 56, 56] tensor.

The Inductor scheduler cannot fuse the avg_pool2d_backward (a scatter/pointwise op)
with the subsequent spatial reduction because:
- The pool backward has complex indirect indexing (each output pixel gathers from
  multiple pooled pixels)
- The scheduler sees it as a pointwise op with different iteration order than the
  reduction consumer
- There's no mechanism to inline a "backwards pool" computation within a reduction
  kernel's inner loop

### Secondary factors

- The oracle uses a tiled approach (13 HW tiles x 32 batch x 128 channels) that
  naturally parallelizes across all 3 dimensions, while Inductor's reduction kernel
  iterates sequentially over the full 56*56=3136 spatial elements
- The oracle avoids writing and re-reading the 51 MB intermediate

## Config Exploration

| Config | Time (us) | Notes |
|--------|-----------|-------|
| CD tuning + combo_kernels (default) | 109.41 | baseline (3 kernels) |
| + multi_kernel=3 | ~328 | much worse |

No config change can eliminate the materialized pool-backward intermediate.

## File/Line References

- avg_pool2d_backward lowering: `/tmp/pytorch-work/torch/_inductor/lowering.py:6577`
- Scheduler fusion decisions: `/tmp/pytorch-work/torch/_inductor/scheduler.py`
- The pool backward kernel uses complex indirect indexing that prevents fusion with
  downstream reductions

## Inductor Closure Path

**Design doc**: Closing this gap requires the scheduler to recognize that:

1. A pointwise producer (avg_pool2d_backward) feeds exclusively into a spatial
   reduction consumer
2. The producer can be inlined into the consumer's inner loop (computed on-the-fly
   per spatial tile)
3. This eliminates the intermediate buffer materialization

This is a specialized form of "inline pointwise producer into reduction consumer"
which the scheduler partially supports for simple pointwise ops, but NOT for
avg_pool2d_backward due to its complex scatter-gather indexing pattern.

The fix would be a scheduler enhancement to detect when a pointwise-with-indirect-indexing
producer has a single reduction consumer and can be profitably inlined, trading compute
for memory bandwidth.

## Related Repros

- `amax_sum_sum_b942094d64da`: ResNeSt split-attention forward (same family, similar
  scheduler fusion gap)
- `amax_sum_sum_b9c8bc430873`: ResNeSt split-attention forward (already at floor for
  this shape)
