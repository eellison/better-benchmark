# sum_sum_sum_43d9d6c4646b
## Compile: 53.6us, Oracle: 33.2us, Gap: 1.61x
## Diagnosis: cooperative_split_k
## Root cause: Inductor schedules the Swin layer-norm-backward row reductions, non-contiguous [1024, 6272] transposed side-output store, and two sibling sum([0,1,2]) column reductions as separate generic pointwise/reduction kernels over materialized intermediates, lacking a coordinated multi-output reduction template.
## Fix path: Cooperative split-K multi-output reduction template that keeps row-local reductions, a materialized transposed side output, and sibling column accumulators in one coordinated producer with partial finalization across row tiles.
## Status: design_todo
