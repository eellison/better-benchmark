# sum_sum_sum_821bf10276d6
## Compile: 207us, Oracle: 90us, Gap: 2.30x
## Diagnosis: SCATTER_REDUCE
## Root cause: Inductor lowers two sum(dim=0)->permute/reshape->index_put(accumulate=True) branches and the mul/sum/neg/fma/reshape branch as separate reduction, layout, scatter, and pointwise kernels over 5 kernel launches; the oracle fuses batch reductions with indexed accumulation into duplicate [169,16] relative-position buckets while emitting the [8192,49,49] softmax-backward tensor.
## Fix path: Structured relative-position scatter-reduce: fuse batch reductions with indexed accumulation and emit full-tensor side output from same producer. Largest gap in batch (2.30x, 117us).
## Status: design_todo
