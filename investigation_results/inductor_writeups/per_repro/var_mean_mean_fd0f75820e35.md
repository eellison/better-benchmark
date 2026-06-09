# var_mean_mean_fd0f75820e35

## Summary
- **Model**: torchbench_timm_efficientnet_train_000 (EfficientNet BN + swish + spatial mean)
- **Pattern**: Per-channel var_mean over [32, 1280, 7, 7], running-stat copy_, affine, swish (x*sigmoid(x)), spatial mean -> [32, 1280]
- **Oracle**: 17.41 us (1 kernel: fused stats + normalize + swish + spatial mean per channel)
- **Compile (CDT)**: 20.13 us (2 kernels)
- **Ratio**: 1.156x
- **Classification**: REDUCTION_EPILOGUE_REREAD

## Root Cause

The oracle computes everything in a single channel-specialized kernel with grid=(1280,):
- Loads all 1568 elements (32*49) for the channel
- Computes mean/var statistics in a single pass
- Updates running stats in-place
- Normalizes, applies affine, computes swish, and accumulates spatial mean
- All without materializing any intermediate buffer

Inductor generates 2 kernels:
1. A reduction kernel computing per-channel var_mean statistics (reduction over [0,2,3])
2. A pointwise+reduction kernel that normalizes, applies affine, computes swish, and reduces spatial dims

The 15.6% gap comes from:
1. **Extra kernel launch** overhead (2 launches vs 1)
2. **Intermediate stats buffer** written/read between kernels
3. **Input re-read** in the second kernel (1568 elements per channel read again for normalization)

With `multi_kernel=3`, Inductor achieves 18.47us (ratio 1.061x), nearly closing the gap through better tile optimization in the second kernel.

## Kernel Count
- Oracle: 1 kernel (persistent per-channel fused kernel)
- Inductor: 2 kernels (reduction + normalize/swish/pool)

## Config Exploration
| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| CDT (default) | 20.13 | 1.156x |
| CDT + combo + multi_kernel=2 | 20.59 | 1.183x |
| CDT + combo + multi_kernel=3 | 18.47 | 1.061x |

**multi_kernel=3 nearly closes the gap** (residual 6% from 2-kernel overhead).

## Design Doc

The fundamental issue is that Inductor cannot fuse a per-channel statistics reduction (var_mean over [0,2,3] with 1568 elements) with its normalization epilogue in a single kernel. The oracle achieves this by:
- Using a persistent approach: all 32*49=1568 elements fit in registers (BLOCK_N=32, BLOCK_HW=64)
- Computing stats, then immediately re-using the data for normalization + swish + spatial pool

For Inductor to match, it needs a norm template that:
1. Detects small per-channel element counts (1568 = 32*49 easily fits in registers)
2. Emits a persistent single-kernel that computes stats and epilogue without intermediate stores
3. Handles the running-stat mutation side effects within the same kernel

### Relevant files
- `/tmp/pytorch-work/torch/_inductor/kernel/norm.py` - norm template (could add persistent BN-training variant)
- `/tmp/pytorch-work/torch/_inductor/choices.py` - persistent reduction threshold
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion of reduction + epilogue

### Affected repros
This pattern (BN-training + activation + spatial mean over small spatial dims 7x7) is common in EfficientNet/MobileNet final layers. Similar repros: var_mean_mean_e5a5cf78a7cf, var_mean_mean_ebf25f51023b, var_mean_mean_8d6fc761298a.

## Status: design_doc (mostly resolved by multi_kernel=3, residual 6% from 2-kernel overhead)
