# var_mean_f8560e5925aa - DenseNet Cat+BN+ReLU Training

## Benchmark Results
- Oracle: 14.66 us
- Compiled: 22.46 us
- Ratio: 1.533x (oracle wins)

## Classification
SCHEDULER_FUSION - Persistent reduction with materialized intermediate

## Root Cause

The oracle computes the full DenseNet training BN (cat from 7 inputs -> var_mean reduction -> running stat updates -> affine+ReLU) in a single kernel with **one pass** over the data. It uses BLOCK_M=4096 which covers all 3136 elements per channel (64 batches * 7*7=49 spatial) in registers. After computing the reduction, it immediately applies normalization and writes the final ReLU output.

Inductor generates 1 kernel but with **two passes**:
1. First loop: reads all 7 cat sources, computes welford reduction for var_mean, **materializes the concatenated tensor to global memory** (buf0/buf5 shape [64, 704, 7, 7] = 9.5 MB)
2. Second loop: re-reads the materialized cat from global memory, subtracts mean, multiplies by invstd*weight+bias, applies ReLU, stores result

The oracle eliminates the write+read of the intermediate cat tensor entirely. For 704 channels * 3136 elements * 4 bytes = ~8.8 MB written then read = ~17.6 MB extra memory traffic.

## Kernel Count
- Oracle: 1 kernel (single pass, persistent)
- Inductor: 1 kernel (two-pass with global memory intermediate)

## Config Exploration
- `coordinate_descent_tuning=True`: no improvement (already used)
- `combo_kernels=True, combo_kernel_per_subkernel_blocks=True, triton.multi_kernel=3`: same performance (~22 us)
- The issue is structural: Inductor's reduction codegen always uses the two-pass pattern when the reduction is too large for a persistent reduction block

## Why Inductor Cannot Do This Today

Inductor's Triton reduction codegen (in `choices.py` `should_use_persistent_reduction`) decides whether a reduction can fit in registers. For r0_numel=3136 with XBLOCK > 1 (needed for parallelism across 704 channels), the total register pressure is XBLOCK * R0_BLOCK which exceeds the persistent threshold. So it falls back to the two-pass pattern that stores intermediates to global memory.

The oracle sidesteps this by using XBLOCK=1 (one channel per program) with BLOCK_M=4096 >= 3136, making the entire reduction persistent in a single thread block.

## Design Doc: What Enhancement Is Needed

**Fix location**: `/tmp/pytorch-work/torch/_inductor/choices.py` (persistent reduction heuristic) and `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (reduction codegen)

**Enhancement**: When the reduction dimension is small enough to fit in registers (e.g. <= 4096 elements), and the kernel has a "reduce then reuse" pattern (var_mean followed by normalization of the same data), the scheduler should consider a persistent single-pass strategy with XBLOCK=1. This avoids materializing the intermediate and halves memory traffic.

Alternatively, the scheduler could detect this "two-pass same data" pattern and keep the data in shared memory instead of writing to global memory on the first pass.

**Affected repro count**: At least 2 in this batch (var_mean_f8560e5925aa, var_mean_bff30f530d6e), likely many more DenseNet BN-training patterns in the corpus.
