# mean_27e29b31cd8d

## Summary
- **Model**: timm_tf_efficientnet_b0 (inference)
- **Pattern**: BN-inference (sub, sqrt, reciprocal, mul, add) -> SiLU -> spatial mean [128,1280,7,7] -> view [128,1280]
- **Ratio**: 1.053x (oracle 21.63us vs compile 22.78us)
- **Classification**: SCHEDULER_FUSION (borderline)
- **Status**: Minimal actionable gap

## Benchmark Results
- Oracle: 21.63 us
- Compiled: 22.78 us
- Ratio: 1.053x

## Root Cause

The oracle emits a single autotuned Triton kernel that fuses BN-inference affine, SiLU activation (`x / (exp(-x) + 1)`), and 7x7 spatial mean into one pass. It tiles over (batch, channel) pairs with block sizes like BLOCK_B=8/16, BLOCK_C=2/4, reading 49 spatial elements per (b,c) pair.

Inductor also emits a single fused kernel (`triton_per_fused_add_div_exp_mean_mul_neg_reciprocal_sqrt_sub_unsqueeze_0`) that handles the entire BN+SiLU+mean computation. The kernel count is identical (1 vs 1).

The 5.3% gap is borderline and likely comes from:
1. **Tiling differences**: The oracle's autotuned 3D tiling (batch x channel x spatial) may have better occupancy than Inductor's 1D row-major persistent reduction
2. **BN algebra redundancy**: Same issue as mean_148ad2fc17ca - sqrt/reciprocal recomputed per row instead of hoisted

## Kernel Count
- **Inductor**: 1 kernel (fully fused)
- **Oracle**: 1 kernel (fully fused with autotune)

## Config Exploration
- `combo_kernels=True` + `coordinate_descent_tuning=True` already enabled
- Inductor achieves optimal kernel count; gap is pure codegen quality

## Design Doc

The gap is borderline (5.3%) and the kernel count is already optimal. The residual difference is the same BN-inference algebraic issue seen in mean_148ad2fc17ca: per-channel sqrt/reciprocal recomputed per batch row rather than hoisted. The fix is the same algebraic canonicalization pass.

### Affected Files
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (BN algebraic canonicalization)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (loop-invariant hoisting)

### Affected Repro Count
Same family as mean_148ad2fc17ca and other BN-inference + spatial mean patterns. Estimated 10-20 repros.
