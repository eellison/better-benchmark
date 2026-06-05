# sum_sum_b16afac198fb Investigation

## Summary
- **Model**: torchbench_densenet121_train_001
- **Ratio**: 1.796x (oracle: 122.24us, compile: 219.58us)
- **Classification**: SCHEDULER_FUSION

## Root Cause

DenseNet BN-backward tail pattern. The oracle fuses:
1. ReLU-mask `where` application
2. Channel centering (sub mean)
3. Two sibling channel reductions (sum of masked values, sum of masked*centered values)
4. Only materializes the returned slice channels 224:255 of the BN-backward pointwise epilogue

Inductor cannot form a full-scope multi-output reduction template that shares the masked/centered producer across different reduction expressions and sinks the downstream slice so only returned channels are materialized.

Shape: [64, 256, 56, 56] with reduction over [0, 2, 3] (200704 elements per channel). Only 256 channels means only 256 thread blocks in Inductor's approach.

## Kernel Count
- **Oracle**: 3 kernels (split-K partial reduce, finalize stats, write sliced output)
- **Inductor**: 1 kernel with 256 blocks, massive per-block reduction

## Config Exploration
Same as sum_sum_67d7103962e7 - combo_kernels and coordinate_descent_tuning already enabled but the issue is reduction parallelism and multi-output scheduling.

## Key Files
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - multi-output reduction fusion
- `/tmp/pytorch-work/torch/_inductor/choices.py` - cooperative reduction threshold

## Design Doc

Same family as sum_sum_67d7103962e7. With 256 channels and 200,704 elements per channel reduction, Inductor gets reasonable but not optimal parallelism. The oracle additionally exploits the slice-limited output (only channels 224:255 of the full [64,256,56,56] result are returned), avoiding computation and memory writes for 224 unused channels in the epilogue.

Fix requires:
1. Cooperative split-K for the reduction phase
2. Multi-output reduction template for sibling reductions sharing a producer
3. Slice-aware epilogue generation (skip channels not in the returned slice)
