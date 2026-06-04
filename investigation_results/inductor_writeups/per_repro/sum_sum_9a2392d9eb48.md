# sum_sum_9a2392d9eb48
## Compile: 86us, Oracle: 62us, Gap: 1.38x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor schedules the DenseNet batch-norm-backward channel reductions (N=64, C=320, H=28, W=28), masked producer, vector epilogue, and sliced residual-add epilogue (last-32-channel slice) as separate generic reduction and pointwise kernels with materialized intermediates; it lacks a cooperative split-K multi-output template that coordinates compatible channel reductions with a dependent sliced side-output epilogue.
## Fix path: Teach Inductor to split compatible small-output channel reductions across the reduced (N, H, W) domain, combine partial summaries, and fuse the downstream vector and sliced tensor epilogues with the live upstream slice additions.
## Status: design_todo
