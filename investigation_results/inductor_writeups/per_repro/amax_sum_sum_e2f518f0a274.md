# amax_sum_sum_e2f518f0a274

## Compile: 153.8us, Oracle: 92.7us, Gap: 1.66x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor lowers the GPT-2 shifted-label cross-entropy mean as separate generic reductions (amax, sum for logsumexp), pointwise kernels, a gather, and scalar reductions that materialize the full [2048,50257] log-softmax intermediate, rather than recognizing this as a fused online cross-entropy pattern with shifted labels, ignore-index masking, and mean reduction.

## Fix path: Add an Inductor lowering for shifted log_softmax plus masked gather plus loss/count mean that emits the online cross-entropy row kernel (online max+sum+gather in one pass) and small final scalar reduction directly, avoiding the full intermediate materialization.

## Status: design_todo

## Details

- Model: torchbench_hf_GPT2 / hf_GPT2_large (train, 2 shapes)
- Pattern: amax+sum+sum+sum reduction (cross-entropy with shifted labels)
- Ops: constant_pad_nd, slice, clone, view, amax, sub, exp, sum, log, sub, ne, where, gather, squeeze, neg, where, sum, sum, div
- Shape: [4,512] labels + [4,512,50257] logits -> [2048,50257] reshaped -> scalar loss
- 2 kernels generated; the dominant cost is materializing the full [2048,50257] intermediate for the logsumexp + gather pass.
- The 1.66x gap would be closed by a single-pass online cross-entropy kernel that computes max, sum(exp), and target log-probability in one row sweep.
