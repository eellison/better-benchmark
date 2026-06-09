# pointwise_5a066f66776e — BN + SiLU + Cat Fusion

## Summary
- **Ratio**: 1.164x (oracle: 11.3us, compile: 13.15us)
- **Classification**: ALGEBRAIC_ELIMINATION
- **Oracle kernels**: 2 (fold_affine + bn_silu_cat fused writer)
- **Inductor kernels**: 2 (broadcast-parameter kernel + generic pointwise/cat kernel)

## Root Cause

The oracle pre-computes per-channel `scale = weight / sqrt(var + eps)` and `shift = bias - mean * scale` in a small 1-block kernel, then fuses `conv * scale + shift -> SiLU -> cat(residual, silu_out)` into a single output-tiled kernel that writes directly into the final dense `[128, 320, 8, 8]` layout.

Inductor emits two kernels but keeps the unfactored `(x - mean) * rsqrt(var + eps) * weight + bias` arithmetic in the main pointwise loop. The issue is that Inductor's pointwise fusion does not algebraically fold broadcasted normalization parameters into a simpler `scale/shift` form before scheduling the fused cat writer. This means every output element computes 4 broadcast loads (mean, var, weight, bias) and the full normalization chain, rather than 2 loads (scale, shift) and a single FMA.

## Config Exploration

The standard configs (combo_kernels, coordinate_descent_tuning) are already active. The gap is not about kernel count or fusion decisions -- both produce 2 kernels. The gap is about algebraic simplification within the fused kernel.

## Design Doc

**Why it cannot be fixed today**: Inductor's pointwise codegen does not have an algebraic simplification pass that detects the `(x - broadcast_const) * broadcast_const * broadcast_const + broadcast_const` pattern and rewrites it into `x * scale + shift` with precomputed per-channel constants.

**What enhancement is needed**: An FX pass (in `torch/_inductor/fx_passes/`) that:
1. Detects the canonical BN-inference affine pattern: `(tensor - mean[C]) * rsqrt(var[C] + eps) * weight[C] + bias[C]`
2. Rewrites it to `tensor * scale[C] + shift[C]` where scale/shift are constant-folded or emitted as a tiny prologue kernel
3. This reduces broadcast loads from 4 to 2 per element and eliminates the sub/rsqrt/mul chain

**Affected files**: `torch/_inductor/fx_passes/post_grad.py` (pass registration), new pass file for BN affine folding

**Affected repro count**: This pattern appears in any model using BatchNorm in inference mode (ResNet, EfficientNet, MobileNet, etc.), likely 10+ repros in the corpus.
