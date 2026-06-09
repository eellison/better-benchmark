# sum_sum_sum_56cdc50216ea
## Compile: 52.1us, Oracle: 32.8us, Gap: 1.59x
## Diagnosis: cooperative_split_k
## Root cause: Inductor schedules the Swin layer-norm-backward row reductions, non-contiguous [1024, 6272] transposed side-output store, and two sibling sum([0,1,2]) column reductions as separate generic pointwise/reduction kernels instead of tiling rows and cooperatively accumulating column partials from a shared producer.
## Fix path: Cooperative split-K multi-output reduction template that keeps row-local reductions, a materialized transposed side output, and sibling column accumulators in one coordinated producer with partial finalization.
## Status: design_todo
