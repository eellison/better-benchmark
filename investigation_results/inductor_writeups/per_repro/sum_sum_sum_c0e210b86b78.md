# sum_sum_sum_c0e210b86b78
## Compile: 78us, Oracle: 40us, Gap: 1.94x
## Diagnosis: SCATTER_REDUCE
## Root cause: Inductor lowers the NanoGPT layernorm-backward row reductions, pointwise producer, two index_put(accumulate=True) scatter consumers, and dense outer-product add as generic scheduled kernels around materialized intermediates; it does not model a shared rowwise reduction producer feeding multiple structured row scatter-reduces plus a dense add epilogue (tokens=64, hidden=768, vocab=50304).
## Fix path: Add an embedding-backward scatter-reduce lowering that computes each row producer once, accumulates duplicate row indices directly into every destination, and fuses compatible sibling column reductions and dense side-output initialization in one kernel.
## Status: design_todo
