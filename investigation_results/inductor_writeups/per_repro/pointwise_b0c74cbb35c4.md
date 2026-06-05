# pointwise_b0c74cbb35c4 — Dual BN Cat Add (GhostNet)

## Summary
- **Model**: timm_ghostnet_100_infer_000
- **Classification**: ALGEBRAIC_ELIMINATION
- **Ratio**: 1.055x (oracle 18.11us vs compile 19.10us)
- **Status**: Marginal gap (just at threshold)

## Root Cause

The oracle folds both BN-inference affine branches (sub-mean, mul-invstd, mul-weight, add-bias) into per-channel scale/shift coefficients, then writes the `[512, 160, 7, 7]` output tensor in one pass. It also virtualizes the channel-cat by selecting the appropriate source tensor based on channel index. Inductor emits the full chain of sqrt/reciprocal/broadcast affine operations in the loop body.

The gap is marginal because Inductor already fuses the entire scope into one kernel -- it just does more ALU work per element (computing sqrt, reciprocal at runtime rather than precomputing per-channel coefficients).

## Kernel Count
- **Inductor**: 1 kernel (fully fused pointwise)
- **Oracle**: 1 kernel (with precomputed coefficients)

## Config Exploration

combo_kernels and coordinate_descent_tuning are already enabled. The issue is not fusion but instruction efficiency.

## What the Oracle Does

The oracle precomputes:
- `scale = weight * rsqrt(var + eps)`
- `shift = bias - mean * scale`

Then applies `output = input * scale[c] + shift[c]` per element, which is 2 FLOPs vs the ~5 FLOPs Inductor emits (sub, add_eps, sqrt, reciprocal, mul, mul, add).

For a memory-bound kernel at this size (7x7 spatial), the ALU savings are minimal, explaining the small 1.055x gap.

## Fix Location

- **File**: `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`
- **Enhancement needed**: BN-inference affine folding pass that canonicalizes the sub/sqrt/reciprocal/mul/mul/add chain to precomputed scale+shift coefficients. This is a well-known optimization but Inductor currently emits the full chain.

## Design Doc

The fix is straightforward but the expected benefit is marginal for memory-bound kernels. The BN coefficient folding is most beneficial when:
1. The spatial dimensions are large (amortizing coefficient computation)
2. Multiple consumers share the same BN parameters

For this repro at [512, 160, 7, 7], the kernel is heavily memory-bound and the extra ALU is essentially free. The 1.055x gap may be within measurement noise.

**Affected patterns**: All BN-inference patterns in GhostNet, ResNet, DenseNet, etc.
