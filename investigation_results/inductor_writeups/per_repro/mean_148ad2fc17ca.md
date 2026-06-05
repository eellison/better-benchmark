# mean_148ad2fc17ca

## Summary
- **Model**: torchbench_mnasnet1_0 (inference)
- **Pattern**: BN-inference (sub, sqrt, reciprocal, mul, add) -> fp16 cast -> ReLU -> spatial mean [256,1280,7,7] -> [256,1280]
- **Ratio**: 1.329x (oracle 22.14us vs compile 29.44us)
- **Classification**: ALGEBRAIC_ELIMINATION
- **Status**: Actionable

## Benchmark Results
- Oracle: 22.14 us
- Compiled: 29.44 us
- Ratio: 1.329x

## Root Cause

The oracle precomputes per-channel BN affine parameters (scale = weight * rsqrt(var + eps), shift = bias - mean * scale) once in a small kernel, then applies them in the main reduction kernel. The main kernel loads the precomputed scale/shift per channel and applies: `y = x * scale + shift; relu(fp16(y)); mean over 7x7`.

Inductor emits a single fused persistent reduction kernel that recomputes the BN algebra (sqrt, reciprocal, mul) for every (batch, channel) row. With 327,680 rows (256*1280), each row independently computes:
- `tmp2.to(f32)` (mean)
- `tmp5.to(f32) + 1e-5` (var + eps)
- `sqrt(var + eps)`
- `1.0 / sqrt(...)` (reciprocal)
- `reciprocal * 1.0` (unnecessary mul by 1)

The per-channel parameters (mean, var, weight, bias) are only 1280 elements each, but they are recomputed 256 times (once per batch element) within the same kernel. The oracle hoists this to a separate precompute pass.

Key inefficiencies in the Inductor kernel:
1. **Redundant sqrt/reciprocal**: computed 327,680 times instead of 1,280 times
2. **mul by 1.0**: `reciprocal * 1.0` is a no-op but costs an instruction per element
3. **No per-channel parameter hoisting**: the BN affine algebra could be precomputed

## Kernel Count
- **Inductor**: 1 kernel (triton_per_fused_add_convert_element_type_mean_mul_reciprocal_relu_sqrt_sub_unsqueeze_0)
- **Oracle**: 2 kernels (affine precompute + tiled BN+ReLU+mean)

## Config Exploration
- `combo_kernels=True` + `coordinate_descent_tuning=True` already enabled
- The single-kernel fusion is correct (no fusion failure), the issue is algebraic

## Design Doc

### Why Inductor Cannot Do This Today

Inductor's codegen does not have a mechanism to hoist batch-invariant BN-inference algebra (sqrt/reciprocal/scale/shift) out of the reduction loop body. The per-channel parameters are broadcast into each (batch, channel) row, but the transcendental ops (sqrt, reciprocal) are recomputed per row rather than being precomputed and reused.

### Enhancement Needed

1. **Algebraic hoisting in reduction codegen** (`/tmp/pytorch-work/torch/_inductor/codegen/triton.py`): When a reduction kernel has loop-invariant expressions that depend only on a subset of the output dimensions (here: channel), hoist them outside the per-row computation. This is analogous to loop-invariant code motion (LICM) but at the Triton kernel level.

2. **BN-inference canonicalization** (`/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`): Recognize the `(x - mean) * reciprocal(sqrt(var + eps)) * weight + bias` pattern and rewrite to `x * scale + shift` where scale/shift are precomputed. This eliminates the sqrt/reciprocal from the hot loop entirely.

3. **Eliminate mul-by-1** (`/tmp/pytorch-work/torch/_inductor/lowering.py`): The `reciprocal_default * 1` pattern should be simplified away.

### Affected Files
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (BN algebraic canonicalization)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (loop-invariant hoisting)
- `/tmp/pytorch-work/torch/_inductor/lowering.py` (mul-by-1 elimination)

### Affected Repro Count
Many BN-inference + spatial mean repros share this pattern. Other affected repros include mean_cf12f53df7dd (MobileNetV3, ratio 1.208x), and likely all `bn_relu_spatial_mean` and `bn_silu_spatial_mean` variants in the corpus (estimated 10-20 repros).
