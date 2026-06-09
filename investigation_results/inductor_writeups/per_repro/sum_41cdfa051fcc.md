# sum_41cdfa051fcc — SCATTER_REDUCE

## Summary
- **Model**: timm_dm_nfnet_f0_train
- **Pattern**: average-pool backward + GELU derivative + channel reduction
- **Ratio**: 1.085x
- **Status**: Moderate gap — scatter-reduce fusion opportunity

## Operation
Takes `[128, 3072]` f32 pooled gradient (mm output) and `[128, 3072, 6, 6]` f32 activation:
1. Views mm as `[128, 3072, 1, 1]` and expands to `[128, 3072, 6, 6]` via as_strided_scatter/expand (avgpool backward = divide by 36)
2. Multiplies by GELU derivative: `0.5*(1+erf(x/sqrt2)) + x * gaussian_pdf(x)` applied to activation
3. Reduces via `sum(dim=[0, 2, 3])` to produce `[3072]` channel sum

## Kernel Count
- Inductor: 1 kernel (fused reduction)
- Oracle: 1 kernel (fused reduction)

## Benchmark
- Oracle: 36.70 us
- Compiled: 39.84 us
- Ratio: 1.085x

## Root Cause Analysis

Both Inductor and the oracle produce a single fused reduction kernel. The 8.5% gap is in
the reduction efficiency:

Inductor's kernel `triton_red_fused_add_div_erf_exp_expand_mul_squeeze_sum_view_0` computes:
- The as_strided_scatter/expand pattern (avgpool backward broadcast)
- GELU derivative computation (erf + exp + multiplies)
- Channel reduction over `[0, 2, 3]` dims (reducing 128*6*6 = 4608 elements per channel)

The oracle's kernel likely uses a more optimized reduction strategy for the specific
`[128, 3072, 6, 6]` -> `[3072]` reduction shape (reducing over dims 0,2,3 with inner dim 6*6=36
being small).

**Likely cause**: The reduction loop bound (r0_numel = 4608 = 128*36) may not tile optimally
with Inductor's default heuristics. The oracle may use a specialized tiling that exploits the
factored structure (128 batches x 36 spatial elements).

## Why Inductor Generates Slightly Suboptimal Code

The reduction dimension (4608) is not a power of 2, which means Triton's reduction blocks
will have wasted lanes or require multiple passes. The oracle can hard-code optimal block sizes
for this specific shape.

## Fix Direction

This is a minor codegen/autotuning gap rather than a missing fusion. Possible improvements:
- Better `coordinate_descent_tuning` exploration for non-power-of-2 reduction dimensions
- Factored reduction: reduce spatial (36) first as an inner loop, then reduce batch (128)

## Conclusion
The gap is small (8.5%) and Inductor already achieves single-kernel fusion. This is a
codegen quality / autotuning gap rather than a structural fusion issue.
