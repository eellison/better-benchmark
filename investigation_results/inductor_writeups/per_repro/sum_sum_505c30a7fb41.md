# sum_sum_505c30a7fb41

## Compile: 188.42us, Oracle: 103.36us, Gap: 1.823x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: Same structural pattern as sum_sum_02744d87feff but with C=192 channels and fewer upstream residual slices (2 vs 4). Inductor emits 2 kernels: a fused reduction processing the dual per-channel sums over the full 200704-element N/H/W domain at 1 CTA per channel, and a pointwise for the sliced residual-add. The oracle uses cooperative split-K with BLOCK_K=1024 tiles across the spatial domain using atomic partial-sum accumulation (196 CTAs per channel-tile), then fuses the BN input-gradient epilogue and the 2-way residual slice addition into a second kernel. The 1.82x gap is larger here because with C=192 channels, Inductor's reduction kernel has even more serialized work per CTA (192 channels * 200704 elements vs the oracle's highly parallelized spatial tiling).

## Fix path: Same as sum_sum_02744d87feff -- teach Inductor cooperative split-K for BN-backward multi-output reductions where the reduction domain (N*H*W) vastly exceeds the output size (C channels). The scheduler should split the 200704-element reduction across ~196 CTAs per channel-tile, accumulate via atomics, then fuse the epilogue.

## Status: design_todo

## Details

- Model: torchbench_densenet121_train_001
- Pattern: sum+sum reduction (BN backward: where_self sum + where_self*centered sum, C=192)
- Shape: [64, 192, 56, 56] inputs, reduction over dims [0, 2, 3] -> [192] outputs
- Reduction domain: N*H*W = 200704 elements per channel
- Inductor kernels: 2 (fused reduction + sliced residual pointwise)
- Oracle kernels: 2 (cooperative split-K reduce + fused epilogue)
- Config exploration: coordinate_descent_tuning (189.3us), combo+multi3 (189.2us) -- no config improvement.
- This is the same DenseNet BN-backward family as 02744d87feff; a single cooperative split-K scheduler enhancement would fix both.
- File references: /tmp/pytorch-work/torch/_inductor/scheduler.py, /tmp/pytorch-work/torch/_inductor/choices.py (should_use_cooperative_reduction)
