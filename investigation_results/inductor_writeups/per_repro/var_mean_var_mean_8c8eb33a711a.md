# var_mean_var_mean_8c8eb33a711a

## Summary
- **Model**: torchbench_timm_resnest_train_000 (ResNeSt dual BN + ReLU + avg_pool)
- **Pattern**: Two per-channel var_mean reductions over [32, 1024, 14, 14], running-stat copy_, affine, branch sum, ReLU, 2x2 avg_pool2d (ceil_mode)
- **Oracle**: 32.77 us (1 kernel: fused channel-per-CTA with stats + normalize + pool)
- **Compile (CDT)**: 44.61 us (2 kernels)
- **Ratio**: 1.361x
- **Classification**: REDUCTION_EPILOGUE_REREAD

## Root Cause

The oracle computes everything in a single channel-specialized Triton kernel: per-channel statistics (over N*H*W=6272 elements), running-stat updates, affine normalization, branch sum, ReLU, and the 2x2 average pooling -- all in one kernel launch per channel. It reads the input data once for statistics, then re-reads selective elements for the pooling output, but avoids materializing the full normalized activation tensor.

Inductor generates 2 kernels:
1. A reduction kernel computing per-channel var_mean statistics over [0,2,3] dims
2. A pointwise/reduction kernel that normalizes, applies affine, sums branches, ReLU, and pools

The 36% gap comes from:
1. **Extra kernel launch** - the oracle does everything in 1 kernel (1024 CTAs), while Inductor needs 2 kernel launches
2. **Intermediate buffer materialization** - between the two kernels, Inductor must store the mean/var statistics to global memory and reload them
3. **Suboptimal tiling** - for the pooling kernel, Inductor tiles across the full output while the oracle tiles by channel with the spatial positions as an inner loop

With `multi_kernel=3`, Inductor achieves 34.87us (ratio 1.064x vs oracle), nearly closing the gap by improving tile selection in the second kernel. However, the fundamental 2-kernel vs 1-kernel overhead remains.

## Kernel Count
- Oracle: 1 kernel (fused stats + normalize + pool per channel)
- Inductor: 2 kernels (reduction for stats, then normalize + pool)

## Config Exploration
| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| CDT (default) | 44.61 | 1.361x |
| CDT + combo + multi_kernel=2 | 46.89 | 1.431x |
| CDT + combo + multi_kernel=3 | 34.87 | 1.064x |

**multi_kernel=3 nearly closes the gap** (1.064x residual, within noise margin).

## Design Doc

The remaining 6% gap with multi_kernel=3 is due to the 2-kernel overhead. To fully match the oracle, Inductor would need to fuse the per-channel BN statistics reduction with the normalization epilogue and downstream avg_pool into a single kernel. This requires:

1. The scheduler recognizing that a per-channel reduction (var_mean over [0,2,3]) followed by an element-wise normalization epilogue that re-reads the same input can be fused into one kernel with a "compute stats then re-read" pattern
2. The fused kernel must also incorporate the avg_pool consumer (stencil pattern with 2x2 windows)

This is a known limitation: Inductor's norm template does not fuse the stats-computation reduction with the downstream normalization + pooling consumer in a single schedule.

### Relevant files
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion decisions for reduction + epilogue
- `/tmp/pytorch-work/torch/_inductor/kernel/norm.py` - norm template that could incorporate pooling
- `/tmp/pytorch-work/torch/_inductor/choices.py` - tile selection for multi_kernel

### Affected repros
This "dual BN + ReLU + pool" pattern appears in multiple ResNeSt/ResNet variants. The exact same root cause (REDUCTION_EPILOGUE_REREAD with stencil pool consumer) appears in `var_mean_var_mean_ca75e017814e` (already at floor) and other resnest repros.

## Status: design_doc (mostly resolved by multi_kernel=3, residual 6% from 2-kernel overhead)
