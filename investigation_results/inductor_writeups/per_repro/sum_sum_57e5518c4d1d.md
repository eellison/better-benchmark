# sum_sum_57e5518c4d1d

## Compile: 183.3us, Oracle: 87.9us, Gap: 2.09x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: Inductor schedules the DenseNet batch-norm-backward channel reductions (sum over [N,H,W]=[64,56,56]) and the dependent sliced residual-add epilogue as separate generic reduction/pointwise kernels over materialized intermediates, rather than coordinating the compatible channel reductions with a cooperative split-K template that finalizes summaries and writes the sliced side outputs in one epilogue.

## Fix path: Teach Inductor to split compatible BN channel reductions across the reduced N/H/W dimension, combine partial summaries, and fuse the downstream vector and slice epilogues with the upstream slice additions. This is the largest gap in this batch (2.09x).

## Status: design_todo

## Details

- Model: torchbench_densenet121 (train, 1 shape)
- Pattern: sum+sum reduction (BN backward with DenseNet residual slices)
- Ops: slice x6, add x5, le, where, sum x2, sub x3, mul x9, unsqueeze x9
- Shape: [64,96,56,56] + multiple [64,{128-256},56,56] sliced inputs -> [96] channel sums + full tensor output
- 2 kernels (combo_persistent) but the materialization of multiple large intermediate slices and separate reduction passes dominates runtime.
- This is the highest-gap repro in this batch and a strong candidate for cooperative split-K optimization.
