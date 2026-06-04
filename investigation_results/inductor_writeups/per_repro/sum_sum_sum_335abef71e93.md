# sum_sum_sum_335abef71e93
## Compile: 54.1us, Oracle: 37.8us, Gap: 1.43x
## Diagnosis: cooperative_split_k
## Root cause: Inductor schedules the Whisper layer-norm backward row reductions, pointwise gradient epilogue/clone/permute, and sibling sum([0,1])/sum([0]) column reductions as separate generic kernels, materializing intermediates between them instead of fusing row-local reductions with cooperative column accumulators.
## Fix path: Cooperative split-K multi-output reduction template that tiles rows, keeps row-local reductions in registers, stores a transposed side output, and cooperatively accumulates column-reduction partials across tiles.
## Status: design_todo
