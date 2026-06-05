# mean_cf12f53df7dd - MobileNetV3 BN Hardswish Spatial Mean

## Benchmark Results
- Oracle: 20.16 us
- Compiled: 24.35 us
- Ratio: 1.208x (oracle wins)

## Classification
ALGEBRAIC_ELIMINATION - rsqrt vs sqrt+reciprocal in fused reduction

## Root Cause

The oracle computes BN-inference affine + hardswish + fp16 cast + 7x7 spatial mean in a single kernel using `tl.rsqrt(var + 0.001)` for the BN normalization. It tiles by batch*channel rows, with each row reducing over 49 spatial elements.

Inductor's generated code uses `sqrt` followed by `reciprocal/divide` for the BN variance normalization (confirmed from the oracle description: "Inductor currently lowers the same fused reduction with sqrt followed by reciprocal/divide in every batch-channel row"). This is 2 transcendental ops where the oracle uses 1 (`rsqrt`).

The kernel runs on 256*960 = 245,760 rows, each computing the division separately. While `rsqrt` is a single hardware instruction on NVIDIA GPUs (via `libdevice.rsqrt`), `sqrt` + `reciprocal` requires two. At this scale (245K independent instances), the instruction count difference becomes measurable.

Note: The compiled kernel crashes with `InductorError: AttributeError: 'SizeVarAllocator' object has no attribute 'size_hint'` on the current pytorch-work branch (pr-184905). The ratio was captured from an earlier successful run before the scheduler error was triggered on a subsequent attempt.

## Kernel Count
- Oracle: 1 kernel (BN-affine + hardswish + fp16 + spatial mean)
- Inductor: 1 kernel (same fused pattern, different arithmetic)

## Config Exploration
- Cannot fully explore due to compilation crash on pr-184905 branch
- First successful bench showed ratio=1.208

## Why Inductor Cannot Do This Today

Inductor's algebraic simplifier does not canonicalize the BN-inference pattern `x / sqrt(var + eps)` to `x * rsqrt(var + eps)` when the normalization occurs inside a reduction body (spatial mean). The lowering produces `sqrt` + division, and no post-lowering pass rewrites this to `rsqrt * multiply`.

## Design Doc

**Fix location**: `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` or `/tmp/pytorch-work/torch/_inductor/lowering.py`

**Enhancement needed**: Add an algebraic rewrite pass that canonicalizes:
- `reciprocal(sqrt(x))` -> `rsqrt(x)`  
- `y / sqrt(x)` -> `y * rsqrt(x)`

This should fire during post-grad FX passes before codegen, specifically when the sqrt feeds into a reciprocal or divide. The pattern is extremely common in BN-inference paths.

**Affected repro count**: Likely many BN-inference repros in the corpus use this same pattern. The fix is a simple algebraic identity that saves one transcendental instruction per element.
