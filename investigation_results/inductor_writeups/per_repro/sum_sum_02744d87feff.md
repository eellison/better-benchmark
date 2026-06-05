# sum_sum_02744d87feff

## Compile: 147.36us, Oracle: 88.0us, Gap: 1.675x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: Inductor emits 2 kernels (one fused reduction kernel computing dual per-channel sums over the N/H/W domain, one pointwise for the sliced residual-add epilogue) for the DenseNet-121 BN-backward tail. The oracle uses cooperative split-K to tile the large N*H*W=200704 reduction across multiple CTAs per channel using atomic_add partial accumulators, then fuses the downstream BN input-gradient and the 4-way residual slice addition into a single epilogue kernel. Inductor's reduction kernel is bandwidth-limited by the large reduction domain (128 channels x 200704 elements), while the oracle parallelizes across spatial tiles and avoids materializing the full [64,128,56,56] intermediate for the pointwise epilogue.

## Fix path: Teach Inductor's scheduler/codegen to emit cooperative split-K multi-output reduction templates for BN-backward-like patterns where: (1) multiple sibling reductions share the same N/H/W reduction domain, (2) reduction outputs feed both a small vector epilogue and a sliced pointwise epilogue over the same input data. The scheduler should detect that the reduction domain (200704) greatly exceeds the output domain (128 channels), tile across spatial with atomic partial-sum accumulation, and fuse the epilogue that reads both the accumulated sums and the original data into a dependent pointwise pass.

## Status: design_todo

## Details

- Model: torchbench_densenet121_train_001
- Pattern: sum+sum reduction (BN backward: where_self sum + where_self*centered sum)
- Shape: [64, 128, 56, 56] inputs, reduction over dims [0, 2, 3] -> [128] outputs
- Reduction domain: N*H*W = 64*56*56 = 200704 elements per channel
- Inductor kernels: 2 (fused reduction + sliced residual pointwise)
- Oracle kernels: 2 (cooperative split-K reduce + epilogue combining reduce results with slice-add)
- The oracle's split-K with BLOCK_K=1024 launches ceil(200704/1024)=196 CTAs per channel-tile, enabling much higher occupancy and memory throughput than Inductor's single-CTA-per-channel reduction.
- Config exploration: coordinate_descent_tuning (147.3us), combo_kernels+multi_kernel=3 (147.3us) -- no config closes the gap.
- File references: /tmp/pytorch-work/torch/_inductor/scheduler.py (can_fuse, score_fusion), /tmp/pytorch-work/torch/_inductor/choices.py (should_use_cooperative_reduction)
