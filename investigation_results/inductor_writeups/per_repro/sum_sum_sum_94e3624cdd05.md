# sum_sum_sum_94e3624cdd05
## Compile: 92us, Oracle: 79us, Gap: 1.17x
## Diagnosis: SCATTER_REDUCE
## Root cause: Inductor materializes the [8, 512, 768] layernorm-gradient producer and lowers the sum and index_put(accumulate=True) consumers as separate generic reduction and scatter kernels; it has no structured scatter-reduce template that shares one rowwise reduction producer across sibling column reductions and multiple indexed accumulation destinations.
## Fix path: Add an embedding-backward scatter-reduce lowering that emits the rowwise layernorm math once, accumulates the column reductions, and writes the position and word embedding scatter-add outputs directly in one fused kernel.
## Status: design_todo
