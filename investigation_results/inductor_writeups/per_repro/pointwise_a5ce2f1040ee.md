# pointwise_a5ce2f1040ee — Cat BN ReLU (DenseNet121, 672 channels)

## Summary
- **Model**: torchbench_densenet121_infer_000
- **Classification**: SCHEDULER_FUSION
- **Ratio**: 1.051x (oracle 13.92us vs compile 14.62us)
- **Status**: At threshold (marginal)

## Root Cause

Same pattern as pointwise_b8e2463e8d92 but with 6 source tensors concatenated along the channel dimension ([64,512,7,7] + 5x[64,32,7,7] = [64,672,7,7]) followed by BN-inference affine -> fp16 cast -> ReLU.

The oracle virtualizes the cat and reads from 6 source tensors with channel-range selection. Inductor materializes the cat intermediate.

## Kernel Count
- **Inductor**: 2 kernels (cat materialization + BN/cast/ReLU)
- **Oracle**: 1 kernel (virtual cat + fused)

## Config Exploration

No config helps. Same scheduler fusion limitation as pointwise_b8e2463e8d92.

## What the Oracle Does

Single kernel with channel-range source selection across 6 input tensors, applying BN affine in fp32, casting to fp16, and ReLU.

## Fix Location

Same as pointwise_b8e2463e8d92:
- **File**: `/tmp/pytorch-work/torch/_inductor/scheduler.py`
- **Enhancement needed**: Virtual cat fusion into downstream pointwise consumers.

## Design Doc

The gap is smaller here (1.051x vs 1.125x for the 608-channel variant) because the larger tensor size makes the kernel more memory-bound, amortizing the cost of the extra cat kernel launch. The pattern is identical to pointwise_b8e2463e8d92 and would be fixed by the same scheduler enhancement.

**Affected patterns**: Same as pointwise_b8e2463e8d92 (DenseNet family).
