# sum_sum_sum_be2a8911a995
## Compile: 224us, Oracle: 165us, Gap: 1.36x
## Diagnosis: SCATTER_REDUCE
## Root cause: Inductor lowers the two sum-permute-view-index_put(accumulate=True) branches and the softmax-backward side output as separate reduction, layout, scatter, and pointwise kernels; the oracle fuses batch reductions with indexed accumulation into duplicate [169,8] relative-position buckets while writing the [16384,49,49] side output.
## Fix path: Structured relative-position scatter-reduce: fuse batch reductions with indexed accumulation and emit complete side-output from same producer. Borderline gap (1.36x, 59us).
## Status: borderline
