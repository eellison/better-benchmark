# var_mean_bff30f530d6e - DenseNet BN Train Cat (5 inputs)

## Benchmark Results
- Oracle: 14.27 us
- Compiled: 19.78 us
- Ratio: 1.386x (oracle wins)

## Classification
SCHEDULER_FUSION - Persistent reduction with materialized intermediate

## Root Cause

Same structural issue as var_mean_f8560e5925aa. This is another DenseNet training BN pattern with fewer branches (5 inputs: 1x512 + 4x32 = 640 channels) but identical computation structure: cat -> var_mean -> running stat updates -> affine + ReLU.

The oracle uses BLOCK_K=4096 with one channel per program (grid=(640,)), keeping all 3136 elements per channel in registers for a single-pass persistent reduction.

Inductor generates 1 kernel with two passes:
1. First pass: reads 5 cat sources, computes welford, **materializes cat to global memory** (buf0/buf5 shape [64, 640, 7, 7])
2. Second pass: re-reads cat from global memory, applies normalization+ReLU

Extra memory traffic from intermediate materialization: 640 channels * 3136 elements * 4 bytes * 2 (write+read) = ~15.4 MB

## Kernel Count
- Oracle: 1 kernel (single pass)
- Inductor: 1 kernel (two-pass with global memory round-trip)

## Config Exploration
- All standard configs tested (coordinate_descent_tuning, combo_kernels, multi_kernel=3): no improvement
- The gap is structural, not a tuning issue

## Why Inductor Cannot Do This Today

Same root cause as var_mean_f8560e5925aa. Inductor's persistent reduction threshold rejects r0_numel=3136 when XBLOCK > 1. The optimal strategy (XBLOCK=1, RBLOCK=4096) is not considered because the scheduler prefers higher XBLOCK for occupancy, trading off the intermediate materialization cost.

## Design Doc

See var_mean_f8560e5925aa.md for full design doc. Same fix applies here.

**Fix location**: `/tmp/pytorch-work/torch/_inductor/choices.py` (persistent reduction threshold)

**Key insight**: For training BN patterns where the cat+reduction+normalization+epilogue forms a chain, the XBLOCK=1 persistent strategy wins because it eliminates an entire buffer materialization (~15 MB round-trip), which more than compensates for lower SM occupancy.
