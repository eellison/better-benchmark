# pointwise_849c8a7e937a - BN Affine + Residual (1.075x)

## Classification
NEW_PATTERN (minor codegen gap)

## Root Cause
The repro computes BN-inference affine (normalize, scale, shift) plus residual add
for fp32 [128,160,7,7] tensor. The oracle uses a flat 1D kernel with simple channel
index extraction (`offsets // (H*W) % C`).

Inductor already fuses everything into a single pointwise kernel. The difference is
in indexing overhead: Inductor uses `x1 = ((xindex // 49) % 160)` for channel
extraction, which is semantically identical but the integer division by 49 (non-power-
of-2) generates more expensive modular arithmetic compared to the oracle's simpler
pattern.

The gap (1.075x) is very small and within noise for many runs.

## Kernel Count
- Oracle: 1 kernel
- Inductor: 1 kernel

Both are already fully fused single-kernel solutions.

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| coordinate_descent + combo_kernels | 9.12 | Default compiled |
| Oracle | 8.48 | Marginally faster indexing |

## Why Inductor Cannot Fix This Today
The 1.075x gap comes from integer division overhead in the index calculation. For
NCHW layout with non-power-of-2 spatial dims (7x7=49), Inductor must compute
`xindex // 49 % 160` which involves expensive integer division. The oracle achieves
slightly better performance by using the same computation but with optimized
constexpr handling by Triton.

This is fundamentally a minor codegen efficiency issue, not a scheduler or fusion
problem. No fix is needed for this magnitude of gap.

## File References
- `torch/_inductor/codegen/triton.py` - index calculation codegen

## Status
Closed - gap is at noise floor (1.075x). No fix warranted.
