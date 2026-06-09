# sum_sum_sum_812de8ed33fc
## Compile: 121.1us, Oracle: 76.5us, Gap: 1.58x
## Diagnosis: scatter_reduce (BERT-large embedding/layernorm-backward)
## Root cause: Inductor materializes the rowwise gradient and lowers the two hidden reductions and three index_put(accumulate=True) embedding-gradient outputs (position, segment, vocabulary) as separate scheduled kernels, without a structured scatter-reduce template that shares row-reduction producers across column reductions and multiple indexed accumulator destinations.
## Fix path: Add scheduler/codegen support for fused embedding-backward scatter-reduce kernels that keep the row layernorm algebra in registers, emit sibling hidden reductions, and atomically accumulate position, segment, and vocabulary gradients directly.
## Status: design_todo
