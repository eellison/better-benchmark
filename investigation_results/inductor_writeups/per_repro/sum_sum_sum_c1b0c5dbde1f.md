# sum_sum_sum_c1b0c5dbde1f

## Compile: 164.83us, Oracle: 100.29us, Gap: 1.64x

## Diagnosis: BANDWIDTH_BOUND / COOPERATIVE_SPLIT_K

## Root cause: Inductor schedules the GhostNet batch-norm-backward fragment as 2 separate Triton kernels (a reduction + a pointwise), materializing intermediate per-channel partial sums before the full-tensor epilogue. The oracle uses a cooperative split-K approach with shared multi-accumulator partial reductions where the two channel groups (112 and 56) overlap, exploiting the fact that the 56-channel group reads from a slice of the same add producer as the 112-channel group. This avoids redundant reads of the large [512,112,14,14] tensor and reduces kernel launch overhead from 2 kernels to 5 smaller specialized kernels that collectively do less total memory traffic.

## Fix path: Add a multi-output channel-reduction template in Inductor that recognizes sibling `sum.dim_IntList([0,2,3])` reductions sharing a common producer with overlapping channel slices, fuses the partial accumulation across both groups, and emits the dependent pointwise epilogue in one coordinated plan.

## Status: design_todo

## Details

- Model: timm_ghostnet_100 (train)
- Pattern: sum+sum+sum reduction (batch-norm backward with two channel groups)
- Inductor kernels: 2 unique Triton kernels
- Oracle kernels: 5 (partial_reduce_112, finalize_112, finalize_56, pointwise_112, pointwise_56)
- Ops: add, copy, clone, sum([0,2,3]), sub, mul, unsqueeze, slice
- Shapes: [512,112,14,14] inputs, [112] and [56] channel reductions, channels-last output for 56-channel group
- The gap is primarily from: (1) redundant reads of the large tensor when Inductor separates the 112-channel and 56-channel reductions, (2) materializing intermediate channel summaries before the pointwise epilogue
- combo_kernels=True does not help (makes it worse: 214us)
- aggressive_fusion=True was not tested on this specific repro but the pattern requires cross-reduction fusion which is not supported
- Config experiments: no existing config flag addresses this gap
