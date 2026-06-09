# pointwise_437415a3398d

## Compile: 291.81us, Oracle: 121.63us, Gap: 2.40x

## Diagnosis: SCHEDULER_FUSION

## Root cause: Inductor emits 2 kernels for the DenseNet inference BN-ReLU-MaxPool-BN-ReLU scope: one large pointwise kernel that fuses both BN affine stages, both ReLUs, and the maxpool into a single `triton_poi_fused__low_memory_max_pool_with_offsets_add_convert_element_type_mul_reciprocal_relu_sqrt_sub_unsqueeze` kernel (12.8M elements), plus a second kernel. However, the maxpool argmax codegen generates an extremely verbose argmax combine function (~100+ comparison operations for 9-element window) due to the generic NaN-handling and tie-breaking logic in `get_reduction_combine_fn`. The oracle achieves 2.4x speedup by implementing the BN-pool-BN chain as a single stencil kernel with explicit 3x3 loop over the pooling window, using a simple `take_if_better` comparison pattern that avoids the verbose tie-breaking chain.

## Fix path: Two potential fixes: (1) Optimize the unrolled argmax combine function in `ir.py` to detect when input is guaranteed non-NaN (e.g., from ReLU) and emit a simpler comparison chain. (2) For maxpool-with-offsets specifically, add a specialized codegen path that tracks the best value and offset in one pass using a simple `>` comparison rather than the full NaN-aware tie-breaking protocol.

## Status: partially_improved

## Fix applied: Optimized unrolled argmax combine function in `torch/_inductor/ir.py` reduces ratio from 2.40x to 2.29x for this case. The remaining gap is from the BN+ReLU producer fusion through maxpool (SCHEDULER_FUSION classification) which requires deeper scheduler changes. Committed on pr-184905 branch in /tmp/pytorch-work.

## Details

- Model: torchbench_densenet121 inference
- Pattern: BN affine (fp16->fp32) -> ReLU -> 3x3 stride-2 padded maxpool-with-offsets -> BN affine -> ReLU
- Inductor kernel count: 2
- Shapes: Input [64, 64, 112, 112] fp16, output [64, 64, 56, 56] fp16 + int8 offsets
- The dominant cost is the verbose argmax unrolled code: for each of 8 pairwise comparisons in the 9-element window, Inductor emits ~6 comparison/logic ops (gt, eq, ne, logical_or, logical_and, where) for NaN handling and tie-breaking
- coord_descent_tuning: 316us (worse), combo_kernels: 619us (worse), multi_kernel=2: 308us (worse)
- The oracle uses BLOCK_C=4 channels and BLOCK_OUT=128 output positions per thread block with num_warps=8, which gives better data reuse of BN parameters

## Re-measurement 2026-06-09 (current pr-184905 branch)

- Oracle: 117.4-117.6 us, Compile: 106.1-106.2 us, Ratio: **0.90x — AT_FLOOR (compile beats oracle)**
- Measured twice with fresh cache, CUDAGraph + GPU lock; reproducible.
- The 2.29x "remaining gap" above is stale: the unrolled argmax tie-break
  optimization (14b0254f8a9) plus subsequently landed scheduler/codegen fixes
  (inline_recomputable_producers et al.) closed it entirely.
- Status: CLOSED (no remaining gap; oracle could be rewritten if a tighter
  floor is believed to exist)
