# sum_sum_b7f94adef30f
## Oracle: oracle_multi_output_reduction (GhostNet BN-backward dual reduction)
## Compile: 175.2us, Oracle: 103.3us, Gap: 1.70x
## Diagnosis: COOPERATIVE_SPLIT_K
## What the oracle does differently: Uses a split-K dual-accumulator Triton kernel where many row/spatial tiles contribute partials for both sum(add_tensor_1) and sum(add_tensor_1 * centered) via atomic accumulation before one fused epilogue pass.
## Inductor fix: COOPERATIVE_SPLIT_K support for compatible multi-output reductions -- the scheduler rejects this small-output (C=72), large-reduction (N*H*W=512*28*28=401408) BN-backward shape under the no-split threshold, scheduling sibling reductions with insufficient reduction-dimension parallelism.
