# sum_sum_3d76bb8bc6b6
## Compile: 49us, Oracle: 33us, Gap: 1.52x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor schedules the DenseNet batch-norm-backward sibling channel reductions (over N=128, H=32, W=32), dependent broadcast arithmetic, and sliced add output as separate generic reduction/pointwise kernels over materialized intermediates; it has no cooperative split-K multi-output template that coordinates compatible small-output channel reductions with a dependent sliced side-output epilogue.
## Fix path: Teach Inductor to split compatible BN-backward channel reductions across the reduced (N, H, W) domain, combine partial summaries, and fuse the downstream vector and sliced tensor epilogues without materializing the full intermediate producer.
## Status: design_todo
