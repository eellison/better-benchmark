# sum_sum_5c61feb6340e

## Status: NO GAP (BAD_ORACLE)

## Measurement
- Oracle: 28.45 us
- Compiled: 24.45 us
- Ratio: 0.859x (oracle is ~1.16x SLOWER than compiled)

## Pattern
DenseNet-121 batch norm backward with 23-branch dense block slice-add epilogue. Pattern hash `5c61feb6340e`, from `torchbench_densenet121_train_001`.
Shape: 23 branch tensors from [64,320,14,14] to [64,1024,14,14], sliced to channel [256:288], plus BN backward statistics over [64,288,14,14].
Outputs: [288] channel statistics tensor and [64,32,14,14] sliced epilogue.

## What the Oracle Does
Two custom Triton kernels:
1. `_reduce_sums_kernel`: per-channel reduction (288 CTAs) computing sum(where) and sum(where*centered) across [N,H,W], emitting the [288] output directly
2. `_slice_epilogue_kernel`: pointwise kernel loading from all 23 branch tensors (channel-sliced) plus reading the reduction results back, computing BN backward epilogue

## Root Cause of BAD_ORACLE
Inductor's compiled output already handles this workload more efficiently. The oracle's two-kernel approach (first reduce, then pointwise epilogue that re-reads results) does not beat Inductor's fused scheduling. At this shape, Inductor likely fuses the reduction and pointwise epilogue better, or its auto-tuned kernel configs provide better occupancy/memory bandwidth utilization for the 23-input slice-add pattern.

The oracle's approach of launching 288 CTAs for the reduction (one per channel, each reducing 12544 elements) is likely suboptimal compared to Inductor's tiled approach that can process multiple channels per CTA.

## Classification: NO_GAP (BAD_ORACLE)

No Inductor fix needed. Compile already outperforms oracle. No investigation required.
