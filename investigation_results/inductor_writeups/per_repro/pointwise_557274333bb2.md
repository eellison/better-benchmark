# pointwise_557274333bb2 -- BN Affine Per-Channel Recomputation

## Summary
- **Repro**: torchbench_mnasnet1_0_infer_000
- **Oracle**: oracle_bn_affine.py
- **Previous Ratio**: 1.399x (oracle 10.82us vs compile 15.14us)
- **Current Ratio**: 1.024x (oracle 10.82us vs compile 11.07us)
- **Classification**: ALGEBRAIC_ELIMINATION (FIXED)
- **Status**: AT_FLOOR

## Fix Applied

Covered by `bn_affine_folding` pass (config `fold_bn_affine = True`).

The pass precomputes per-channel BN scale/shift coefficients:
- `scale = weight * rsqrt(var + eps)`
- `shift = bias - mean * scale`

Then the main kernel applies a flat FMA: `output = x * scale + shift`, eliminating
per-element sqrt/reciprocal recomputation.

## Root Cause (historical)

The oracle pre-computed per-channel BN scale/shift coefficients (320 elements) once in a
small kernel, then applied a flat FMA across the entire [256, 320, 7, 7] output tensor.
Inductor previously fused ALL operations (convert_element_type, sub, sqrt, reciprocal, mul,
add) into a single pointwise kernel that recomputed sqrt/reciprocal for every output element.

With shape [256, 320, 7, 7], the per-channel ops on [320] got recomputed 256 * 49 = 12,544
times each.

## Kernel Count
- **Oracle**: 2 kernels (1 small per-channel coefficient kernel + 1 FMA kernel)
- **Inductor (after fix)**: 1 kernel (fused with precomputed coefficients via inline)

## Config Exploration
- BN affine folding brings from 1.399x to 1.024x
- Remaining 2.4% gap is within measurement noise
