# sum_sum_e036cf2d1ccb Investigation

## Summary
- **Model**: hf_MT5ForConditionalGeneration_train_001
- **Ratio**: 1.077x (oracle: 17.02us, compile: 18.34us)
- **Classification**: COOPERATIVE_SPLIT_K

## Root Cause

The oracle computes the full MT5 dropout/layer-norm-backward return tuple by row-tiling the `[4096, 512]` producer, sharing the row-local reduction needed by the transposed `[512, 4096]` gradient output, and emitting partial column sums for the returned `[512]` vector from the same element pass.

Inductor schedules the row reduction, dependent pointwise epilogue, transposed side-output write, and sibling column reduction as separate generic kernels over materialized intermediates.

Shape: [4096, 512] with a row reduction (dim=2, 512 elements) and a column reduction (dim=[0,1], 4096 elements).

## Kernel Count
- **Oracle**: 2 kernels (row-tiled producer with partials + finalizer)
- **Inductor**: Multiple kernels (reduction + transpose + column reduction)

## Config Exploration
The gap is very small (1.077x, only 1.3us absolute). coordinate_descent_tuning=True already brings Inductor close to optimal. This is borderline AT_FLOOR.

## Key Files
- `/tmp/pytorch-work/torch/_inductor/choices.py` - cooperative split-K strategy
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - row-tiled codegen

## Design Doc

The gap is marginal (1.077x). The oracle's advantage is in sharing one element load pass across the transpose store, row reduction, and column partial accumulation. Inductor's generic scheduling separates these, causing one extra read of the [4096, 512] intermediate. Given the tiny absolute gap (1.3us), this is low priority but would benefit from a cooperative split-K multi-output template that coordinates row-local ops with column accumulators.
