# pointwise_f33e2464d32f


## Measured Timings
- Oracle: 5.79 us
- Compile (CDT): 5.79 us
- Ratio: 1.00x

Full-scope oracle: `repros/canonical/pointwise_f33e2464d32f/oracle_maxpool_norm.py`.

## Root Cause

Classification: SCHEDULER_FUSION (realize_hint blocks fusion).

The oracle fuses per-channel affine normalization (BN-inference), ReLU, and 2x1 stride-2 maxpool with int8 offsets into a single Triton kernel. Inductor emits two kernels:
1. Kernel 0: `triton_poi_fused_add_mul_reciprocal_relu_sub_unsqueeze_0` -- computes BN+ReLU, writes 65536 elements to an intermediate buffer.
2. Kernel 1: `triton_poi_fused__low_memory_max_pool_with_offsets_...` -- reads from that buffer, computes max and argmax over window [2,1], writes 32768 pooled values + 32768 int8 offsets.

The root cause is `x.realize_hint()` at `/tmp/pytorch-work/torch/_inductor/lowering.py:5560` in `_max_pool_with_offsets()`. This forces the BN+ReLU output to materialize before the pooling reduction, preventing fusion.

## Kernel Count
- Oracle: 1 kernel (fused norm+ReLU+maxpool+offsets)
- Inductor: 2 kernels (pointwise BN+ReLU, then separate maxpool)

## Config Exploration
- `coordinate_descent_tuning=True`: compile_us=7.04 (default config)
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: does not help (still 2 kernels)

## Measurements
- `python repros/canonical/pointwise_f33e2464d32f/oracle_maxpool_norm.py --check`: PASS (stochastic output 0 skipped, output 1 exact int8 match).
- `python repros/canonical/pointwise_f33e2464d32f/oracle_maxpool_norm.py --bench`: oracle_us=5.79, compile_us=7.04, ratio=1.215, status=GOOD.

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/lowering.py:5560` -- `x.realize_hint()` in `_max_pool_with_offsets` blocks fusion of pointwise producer into maxpool consumer.
- `/tmp/pytorch-work/torch/_inductor/lowering.py:5595-5620` -- The Reduction.create calls for max and argmax that define the pooling.
- `/tmp/pytorch-work/torch/_inductor/lowering.py:5647` -- `config.patch(unroll_reductions_threshold=25)` in `_low_memory_max_pool_with_offsets` unrolls the reduction (kernel_size=[2,1] has product 2 < 25), so it becomes a Pointwise node. But `realize_hint` already forced materialization.

## Design Doc

**Why it cannot be fixed today**: The `realize_hint()` exists because maxpool accesses the input with shifted indices (stencil pattern). The scheduler's `can_fuse_pointwise_into_reduction` path does not handle the case where the "reduction" is actually an unrolled comparison over a small window accessing the producer's output at multiple spatial offsets. Removing `realize_hint()` unconditionally could cause exponential code blowup for large kernel windows.

**What enhancement is needed**: A scheduler-level optimization that detects when:
1. The producer is a pure pointwise kernel (BN+ReLU)
2. The consumer is a small-window maxpool (kernel product <= threshold, e.g. <= 4)
3. The stencil access pattern only reads adjacent spatial elements from the producer

In this case, the scheduler should allow inlining the producer computation into the consumer, recomputing it for each stencil tap (2x recomputation for kernel=[2,1] is cheaper than the intermediate buffer write+read round-trip of 65536 * 4 bytes = 256 KB).

**Affected repro count**: This pattern (BN/affine + ReLU fused with small-window maxpool) appears in ResNet and DocTR models. At least `pointwise_f33e2464d32f` (DocTR) and `pointwise_f2df04089ff4` (ResNet, already AT_FLOOR for larger shapes) are affected.
