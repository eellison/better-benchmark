# pointwise_42ab411e21f6 — ALGEBRAIC_ELIMINATION (ratio 1.931x)

## Summary
Oracle: `oracle_repvgg_bn_sum_relu.py` — Three-branch RepVGG BN-inference sum + ReLU
Benchmark: oracle=24.19us, compile=46.72us, ratio=1.931x
Model: `timm_repvgg_a2_infer_000`
Shape: 3 activations of `[128, 384, 14, 14]` f32 + 12 channel params of `[384]` f32

## Root Cause

The oracle folds three BN-inference branches into per-channel scale/shift coefficients:
```
scale_i = weight_i * rsqrt(var_i + eps)
shift_i = bias_i - mean_i * scale_i
y = x0*scale0 + x1*scale1 + x2*scale2 + (shift0 + shift1 + shift2)
output = relu(y)
```

This hoists 12 channel-parameter loads and all normalization math (sqrt, reciprocal, sub)
outside the spatial loop, computing them once per channel block. It also uses NaN-aware
masking to skip activation loads for channels with invalid (NaN) coefficients.

Inductor generates a single flat Grid1D kernel that:
1. Loads all 15 inputs per element (3 activations + 12 channel params)
2. Recomputes `sqrt + reciprocal + mul*1` three times per element
3. Uses `((xindex // 196) % 384)` integer division per element for channel indexing
4. Total: 15 loads + 3 sqrt + 3 reciprocal + many FMA ops per element

The oracle's 3D tiling (N=128, C_blocks, HW_blocks=196) loads channel params once per
C_block and reuses them across 196 spatial positions, saving ~12x redundant loads per
spatial element.

## Kernel Count
- Inductor: 1 kernel (flat Grid1D, xnumel=9633792)
- Oracle: 1 kernel (3D grid: N x C_blocks x HW_blocks, with autotuned BLOCK_C/BLOCK_HW)

## Config Exploration

| Config | Time (us) | Ratio to Oracle |
|--------|-----------|----------------|
| Default (coord_descent + combo_kernels) | 46.72 | 1.931x |
| + multi_kernel=3 | 48.84 | 2.019x |
| + combo_kernel_per_subkernel_blocks | 65.08 | 2.690x |
| + prefer_nd_tiling | 131.06 | 5.417x |
| + max_autotune_pointwise | 32.03 | 1.324x |

**max_autotune_pointwise reduces the gap from 1.93x to 1.32x** but cannot fully close it.
The remaining 32% gap is due to redundant channel-parameter math (sqrt/reciprocal) that
cannot be eliminated by tiling alone — it requires algebraic folding.

prefer_nd_tiling catastrophically worsens performance (5.4x), likely by choosing a bad
2D decomposition that breaks coalescing for the 3-activation case.

## File/Line References
- Tiling selection: `/tmp/pytorch-work/torch/_inductor/codegen/simd.py:4271` (`select_tiling`)
- FX passes location: `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`
- Pointwise codegen: `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`
- Config: `/tmp/pytorch-work/torch/_inductor/config.py`

## Design Doc

**Why it happens**: Inductor's pointwise codegen treats broadcast [C] parameters as ordinary
loads with `evict_last` eviction and recomputes `sqrt(var+eps)` and `reciprocal` for every
spatial element. With 3 branches x 4 params x 196 spatial positions, this results in:
- 12 * 196 = 2352 redundant channel-param loads per batch element (vs 12 in oracle)
- 3 * 196 = 588 redundant sqrt+reciprocal per batch element (vs 3 in oracle)

**What the oracle does differently**:
1. **Algebraic folding**: Pre-computes `scale = weight * rsqrt(var + eps)` and
   `shift = bias - mean * scale` as channel-only vectors, reducing per-element work
   to 3 FMAs + addition + relu
2. **Channel-tiled iteration**: 3D grid loads channel coefficients once per C_block
   and broadcasts to spatial tile via `[:, None]`
3. **Dead load elimination**: Skips activation loads for NaN channels (minor)

**Required Inductor enhancement (two-part)**:

1. **BN-affine algebraic folding pass** (`fx_passes/`): Recognize the pattern
   `(x - mean) * reciprocal(sqrt(var + eps)) * weight + bias` and fold to
   `x * scale + shift` where `scale = weight * rsqrt(var + eps)`, `shift = bias - mean * scale`.
   This eliminates the per-element sqrt/reciprocal. For multi-branch patterns, fold all
   branches' shifts into a single additive constant.

2. **Broadcast-aware tiling heuristic** (`codegen/simd.py`): When a kernel has multiple
   broadcast loads from a shared dimension (here [384] broadcast into [128,384,14,14]),
   prefer a tiling that tiles along that broadcast dimension to maximize register reuse.
   The current flat 1D tiling with `evict_last` relies on L1 cache hits but cannot match
   explicit register-level reuse.

**Impact**: This pattern (multi-branch BN sum + activation) appears in RepVGG, ResNet
skip connections, and DenseNet. Fixing it could improve all multi-branch BN inference workloads.

**Partial mitigation**: `max_autotune_pointwise = True` reduces the gap to 1.32x by finding
a better XBLOCK that improves cache behavior, but cannot eliminate the algebraic redundancy.
