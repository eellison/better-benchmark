# mean_7639bfb9be38 - Dual BN ReLU Spatial Mean (RegNet)

## Status: FIXED (2026-06-09)

Fix implemented on branch pr-184905 (B200 measurements):
- Before (rsqrt canonicalization + fold_bn_affine disabled): oracle 11.87 us, compile 16.06 us, ratio 1.353x
- rsqrt canonicalization only (fold_bn_affine off): compile 12.93 us, ratio 1.08x
- After (defaults, both passes on): oracle 11.97 us, compile 11.87 us, ratio 0.992x (AT_FLOOR)
- Correctness: PASS (max_finite_diff 6.25e-02 within fp16 tolerance, NaN masks match)

Implementation: post-grad graph patterns `reciprocal(sqrt(x)) -> rsqrt(x)` and
`div(1, sqrt(x)) -> rsqrt(x)` in `torch/_inductor/fx_passes/post_grad.py`,
gated behind new config flag `config.rsqrt_canonicalization`
(env `TORCHINDUCTOR_RSQRT_CANONICALIZATION`, default True). The remaining gap is
closed by the existing `fold_bn_affine` pass (BN affine folding), which depends on
the rsqrt-canonicalized form.

Commits: a73d1583b34 (patterns), 6703f38fa2d (config gate + verification).

Regression spot-check (defaults vs gate off, no slowdown, correctness PASS):
- var_mean_598830735cf6: 1.387x vs 1.395x
- mean_a3a8e8f5cc86: 1.088x vs 1.130x
- var_mean_0854c7adb682: 1.045x vs 1.038x

## Benchmark Results (original, before fix)
- Oracle: 12.16 us
- Compiled: 17.12 us
- Ratio: 1.408x (oracle wins)

## Classification
ALGEBRAIC_ELIMINATION - rsqrt canonicalization + dual BN redundant computation

## Root Cause

The oracle computes the full dual RegNet BN-inference affine (two separate convolution outputs with their own BN params), their element-wise add, ReLU, and 7x7 spatial mean in one reduction kernel. It uses a folded `rsqrt` affine form and stores only the final fp16 [32, 2240] output.

Inductor generates a fused kernel that handles the same operations but:
1. Uses `sqrt` followed by `reciprocal` for both BN variance terms (2x2=4 transcendental ops per row vs oracle's 2 rsqrt ops)
2. The reduction over 49 spatial elements (7x7) means these BN computations run 32*2240=71680 times
3. Each row computes BN for both branches, adds, applies ReLU, then reduces - the oracle does this with fewer instructions per element

The input tensors are f16[32, 2240, 7, 7] (NCHW contiguous). The oracle tiles by (batch*channel) rows, with each row reducing 49 elements.

## Kernel Count
- Oracle: 1 kernel (dual BN-affine + add + ReLU + spatial mean)
- Inductor: 1 kernel (same fusion, different arithmetic)

## Config Exploration
- `combo_kernels = True`: no effect (already fused into one kernel)
- `coordinate_descent_tuning = True`: may help block sizes but not arithmetic
- The gap (1.408x) is significant and primarily algebraic

## Why Inductor Cannot Do This Today

The primary issue is **algebraic**: for each of the two BN branches, Inductor emits:
```
sqrt(var + eps) -> reciprocal -> multiply
```
instead of:
```
rsqrt(var + eps) -> multiply
```

This doubles the transcendental instruction count (4 ops vs 2). On 71680 rows, each saving 2 transcendental ops (one per BN branch), the total savings are 143360 fewer transcendental instructions.

Additionally, the oracle may pre-compute the scale = weight * rsqrt(var+eps) and bias' = bias - mean * scale as a folded affine, reducing the per-element work to a single multiply-add per branch rather than sub + mul + mul + add.

## Design Doc

**Fix location**: `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`

**Enhancement needed**:
1. Add algebraic rewrite: `reciprocal(sqrt(x))` -> `rsqrt(x)`
2. Optionally: fold BN inference to `x * (weight * rsqrt(var+eps)) + (bias - mean * weight * rsqrt(var+eps))` reducing per-element work to fma

**Affected repro count**: This is the highest-ratio case in this chunk (1.408x). The dual-BN pattern appears in RegNet/ResNet residual blocks. Combined with single-BN variants, likely 10+ repros benefit from rsqrt canonicalization.
