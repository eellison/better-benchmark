# sum_sum_sum_ae9b17407681
## Compile: 122.9us, Oracle: 72.1us, Gap: 1.70x
## Diagnosis: scatter_reduce (embedding-backward scatter-reduce)
## Root cause: Inductor materializes the [4,512,768] gradient producer and lowers the two index_put(accumulate=True) embedding updates plus sibling column reductions as generic scheduled kernels, without sharing the rowwise layernorm-backward producer across multiple indexed accumulation outputs and column reductions in a single fused pass.
## Fix path: Add an Inductor structured scatter-reduce template for embedding-backward that shares one rowwise reduction producer across multiple indexed accumulation outputs and column reductions, emitting shared reduction/scatter epilogues without materializing the intermediate.
## Status: design_todo
