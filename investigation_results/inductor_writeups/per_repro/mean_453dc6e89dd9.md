# mean_453dc6e89dd9

## Summary
- **Model**: torchbench_phlippe_densenet (inference)
- **Pattern**: cat([conv_16ch, cat_168ch], dim=1) -> BN-inference (sub, rsqrt, mul, add) -> ReLU -> spatial mean -> view
- **Ratio**: 1.119x (oracle 6.21us vs compile 6.94us)
- **Classification**: SCHEDULER_FUSION (cat inlining into downstream reduction)

## Root Cause

The oracle fuses the channel concatenation, BN-inference normalization, ReLU, and spatial mean into a single kernel that reads directly from the two concat source tensors (conv [128,16,4,4] and cat [128,168,4,4]) without materializing the concatenated [128,184,4,4] intermediate. Each thread block handles a tile of (batch, channels) pairs, selects the correct source based on channel index (< 16 uses conv, >= 16 uses cat), normalizes, activates, and reduces the 16 spatial elements directly into the [128,184] output.

Inductor generates 1 kernel:
`triton_per_fused_add_cat_mean_mul_reciprocal_relu_sqrt_sub_unsqueeze_0` - a single fused kernel that handles cat+BN+ReLU+mean

Surprisingly, Inductor already fuses everything into a single kernel! The gap is smaller (12%) and is likely due to:
1. **Cat materialization within the kernel**: Even though it's one kernel, Inductor may materialize the cat result in shared memory or registers before the reduction, whereas the oracle uses conditional loads (tl.where to select source)
2. **Tiling differences**: The oracle uses a specialized (BLOCK_BATCH=4, BLOCK_CHANNELS=16, BLOCK_HW=16) tile shape optimized for this exact problem size, while Inductor uses generic autotuned parameters

## Kernel Count
- **Inductor**: 1 kernel
- **Oracle**: 1 kernel

## Config Exploration
Both `combo_kernels=True` and `coordinate_descent_tuning=True` are enabled. Inductor achieves the optimal kernel count. The 12% gap is purely in code quality / tiling efficiency within the single kernel.

## Design Doc

The gap is small (12%) and the kernel count is already optimal. The remaining difference is in:

1. **Concat inlining strategy**: The oracle uses `tl.where(c < 16, conv_values, cat_values)` to conditionally load from two different source pointers based on channel index. Inductor may handle the cat differently (e.g., computing the concatenated offset arithmetically).

2. **Tiling for tiny spatial dims**: With only 16 spatial elements (4x4), the reduction is trivial. The oracle's tile shape (4 batches x 16 channels x 16 spatial) keeps everything in registers. Inductor's persistent reduction with R0_BLOCK may have different occupancy characteristics.

### What enhancement is needed
This is a minor tiling/codegen quality issue rather than a fundamental scheduler limitation. The enhancement would be:
- When `aten.cat` feeds directly into a reduction consumer with very small spatial dimensions, the codegen could emit conditional loads (virtual concat) rather than materializing the concatenated tensor, even within a fused kernel.
- Better autotuning of tile shapes for very small reduction dimensions (16 elements)

### Affected files
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (cat lowering within fused kernels)
- `/tmp/pytorch-work/torch/_inductor/lowering.py` (how cat is represented in IR for fusion)

### Affected repro count
DenseNet's dense connection pattern (many small cats feeding into BN) is specific to this architecture. Estimated 2-4 repros share this exact pattern.
