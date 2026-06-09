# pointwise_f564ff7f54c2


## Measured Timings
- Oracle: 20.29 us
- Compile (CDT): 24.10 us
- Ratio: 1.19x

Full-scope oracle: `repros/canonical/pointwise_f564ff7f54c2/oracle_bn_hardswish_pointwise.py`.

## Root Cause

Classification: ALGEBRAIC_ELIMINATION (redundant per-element broadcast recomputation).

The oracle pre-computes per-channel `scale = weight / sqrt(var + eps)` and `shift = bias - mean * scale` in a tiny 960-element kernel, then uses these pre-computed values in the main elementwise kernel (12M elements) to compute `affine = x * scale[c] + shift[c]` followed by hard-swish. This avoids redundant rsqrt computation.

Inductor emits a single kernel `triton_poi_fused_add_clamp_max_clamp_min_convert_element_type_div_mul_reciprocal_sub_unsqueeze_0` that, for every element of the 12,042,240-element tensor, reloads the per-channel mean/var/weight/bias vectors and recomputes `rsqrt(var + eps)` per element. While the rsqrt values are the same across the spatial dimensions (256 * 7 * 7 = 12544 elements share the same channel), the computation is repeated 12544 times per channel.

The oracle's approach:
1. Kernel 1 (tiny): compute `scale[c]` and `shift[c]` for 960 channels
2. Kernel 2 (main): `out[n,c,h,w] = hardswish(x[n,c,h,w] * scale[c] + shift[c])`

Inductor's approach:
1. Single kernel: `out[n,c,h,w] = hardswish((x[n,c,h,w] - mean[c]) * rsqrt(var[c] + eps) * weight[c] + bias[c])`

The single-kernel approach loads 4 broadcast vectors per element (mean, var, weight, bias) vs 2 (scale, shift) and performs extra rsqrt + multiply operations. On a 12M element tensor with 960 channels and HW=49, the redundant work is significant.

## Kernel Count
- Oracle: 2 kernels (tiny per-channel affine params, then main BN-hardswish)
- Inductor: 1 kernel (all ops fused but with redundant broadcast recomputation)

## Config Exploration
- `coordinate_descent_tuning=True`: compile_us=25.44
- `combo_kernels=True,...,triton.multi_kernel=3`: does not help (single kernel with same redundant recomputation)

## Measurements
- `python repros/canonical/pointwise_f564ff7f54c2/oracle_bn_hardswish_pointwise.py --check`: PASS, output 0 values shape `[256, 960, 7, 7]`, dtype `torch.float16`, max_finite_diff=6.25e-02, NaN mask matches.
- `python repros/canonical/pointwise_f564ff7f54c2/oracle_bn_hardswish_pointwise.py --bench`: oracle_us=20.29, compile_us=25.44, ratio=1.254, status=GOOD.

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` -- The pointwise codegen loads broadcast vectors with `eviction_policy='evict_last'` (correctly identified as broadcast), but does not extract the invariant rsqrt subexpression.
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` -- Where a BN-affine factoring pass should be registered.
- `/tmp/pytorch-work/torch/_inductor/fx_passes/linear_reduction_elimination.py` -- Existing broadcast-aware pass infrastructure that could be extended.

## Design Doc

**Why it cannot be fixed today**: Inductor's pointwise codegen correctly fuses all ops into one kernel (which is good for avoiding intermediate buffer materialization). However, it lacks an algebraic simplification pass that recognizes when a broadcast-invariant subexpression (the BN affine: `rsqrt(var+eps)*weight` and `bias - mean*rsqrt(var+eps)*weight`) can be factored out and pre-computed in a separate tiny kernel. The existing `linear_reduction_elimination` pass works on reduction dimensions, not broadcast dimensions in pointwise graphs.

**What enhancement is needed**: An FX pass (or codegen-level optimization) that:
1. Identifies subexpressions in a pointwise kernel that are invariant across broadcast dimensions (i.e., they only depend on the channel dimension, not spatial dims)
2. When the spatial expansion factor (N*H*W / C) exceeds a threshold (e.g., > 64), hoists the invariant subexpression into a pre-computation step
3. Rewrites the main kernel to load the pre-computed scale/shift vectors instead of the raw mean/var/weight/bias vectors

This is essentially "loop-invariant code motion" at the kernel level: the rsqrt computation is loop-invariant over the spatial dimensions but is currently recomputed in the inner loop.

**Estimated benefit**: 1.254x speedup for MobileNetV3 BN+hardswish layers. The pattern (BN-inference affine with broadcast params feeding a nonlinearity) is extremely common in CNN inference.

**Affected repro count**: MobileNetV3 and similar architectures with BN + hard-swish. This specific pattern (`convert_element_type -> sub -> rsqrt -> mul -> add -> hardswish`) appears in multiple MobileNet variants.
