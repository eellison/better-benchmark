# pointwise_bc30b7a4c3fd: Dual BN Affine Chain Folding

## Summary

- **Repro**: `repros/canonical/pointwise_bc30b7a4c3fd/`
- **Model**: timm visformer_small inference
- **Classification**: ALGEBRAIC_ELIMINATION
- **Oracle**: 13.95 us | **Compile**: 25.76 us | **Ratio**: 1.85x
- **Tensor shape**: `[128, 768, 7, 7]` (4.82M elements, ~19 MB)

## The Pattern

The repro computes two sequential BN-inference affine chains with a residual add between them:

```
BN1: y1 = (x - mean1) * rsqrt(var1 + eps) * weight1 + bias1
Residual: y2 = y1 + residual   (residual is [1, 768, 7, 7])
BN2: result = (y2 - mean2) * rsqrt(var2 + eps) * weight2 + bias2
```

## The Algebraic Identity

Expanding and collecting terms:
```
scale1 = rsqrt(var1 + eps) * weight1            # [768]
shift1 = bias1 - mean1 * scale1                 # [768]
scale2 = rsqrt(var2 + eps) * weight2            # [768]
x_scale = scale1 * scale2                       # [768]
residual_scale = scale2                         # [768]
shift = (shift1 - mean2) * scale2 + bias2       # [768]

result = x * x_scale + residual * residual_scale + shift
```

This reduces the per-element computation from ~15 ops (including 2x sqrt, 2x reciprocal) to 3 FMAs.

## Root Cause Analysis

### Why Inductor is slow (25.76 us)

Inductor correctly fuses everything into a single pointwise kernel, but that kernel:
1. Loads 9 per-channel parameters + x + residual = 11 loads per element
2. Computes sqrt, reciprocal (expensive transcendental ops) per-element, even though they only depend on the channel index
3. Executes ~15 arithmetic operations per element

The kernel is **memory-bound** (arithmetic intensity ~2 FLOP/byte), but achieves only 45% of peak memory bandwidth because the excessive instruction count and register pressure from the transcendental ops prevent the GPU from saturating the memory pipeline.

### Why the oracle is fast (13.95 us)

The oracle:
1. Precomputes combined coefficients in a tiny [3, 768] buffer (separate kernel, negligible cost)
2. The main kernel does only: 2 loads (x + residual) + 3 coefficient loads + 2 muls + 2 adds + 1 store
3. Achieves 82% of peak memory bandwidth due to minimal instruction pressure

### Performance breakdown (measured)

| Version | Time (us) | Notes |
|---------|-----------|-------|
| Inductor baseline (1D tiling) | 28.76 | Single fused kernel, flat iteration |
| Inductor with 2D tiling (`prefer_nd_tiling`) | 25.65 | Better coalescing, same compute |
| FX-rewritten (precomputed coeffs) | 24.98 | Inductor re-fuses coefficients back! |
| Pure FMA kernel (forced precompute) | 17.02 | Inductor's best with precomputed coefficients |
| Oracle (custom Triton) | 13.95 | Optimal autotuned kernel |

## Key Finding: Inductor Re-Fuses Precomputed Coefficients

When we write an FX-level rewrite that precomputes the combined [768] coefficients, Inductor's scheduler sees that these small intermediates are only used by the main kernel and **re-inlines them** into the large pointwise kernel. This is because the fusion heuristic treats small per-channel ops (rsqrt) as "free" to recompute, not accounting for:
- They execute 4.8M times (once per output element) instead of 768 times
- rsqrt is an expensive transcendental instruction
- The extra instructions destroy memory bandwidth utilization

## Fix Strategy

This requires a **two-part fix**:

### Part 1: FX Pass (BN-affine coefficient folding)
A post-grad FX pass that pattern-matches the BN inference affine chain:
```
(x - mean) * rsqrt(var + eps) * weight + bias
```
and rewrites chained instances into precomputed combined coefficients. The pass should:
- Detect serial BN-affine chains (with optional residual adds between them)
- Algebraically combine scale/bias into per-channel coefficients
- Insert explicit small-tensor coefficient computation nodes in the graph

### Part 2: Scheduler Cost Model Fix (prevent re-inlining)
The scheduler's fusion decision must account for **recomputation amplification**: when a node produces a small tensor (e.g., [768]) but its consumer has many more elements (e.g., [128, 768, 7, 7] = 6272x amplification), re-computing expensive ops (rsqrt, sqrt, exp) inside the consumer kernel is costly. The fix could be:
- Option A: Mark precomputed coefficients with `no_fuse_buffer_names` in the FX pass lowering
- Option B: Add a cost penalty to the fusion score when the producer node contains transcendental ops and the numel ratio (consumer/producer) exceeds a threshold
- Option C: After lowering, call `.realize()` on the coefficient buffers to force materialization

### Implementation Location
- FX pass: `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (or new file `bn_affine_folding.py`)
- Scheduler fix: `/tmp/pytorch-work/torch/_inductor/scheduler.py` around `_score_fusion_memory_for_can_fuse`

## How Many Repros Would Benefit

At least 5 repros in this corpus have the same BN-affine folding opportunity:
- `pointwise_bc30b7a4c3fd` (this one) - dual BN affine + residual
- `pointwise_ffe1ef7f99e0` - single BN affine
- `pointwise_b0c74cbb35c4` - dual BN + cat + add
- `pointwise_42ab411e21f6` - RepVGG three-branch BN sum + ReLU
- `pointwise_5a066f66776e` - BN + SiLU + cat

All are classified ALGEBRAIC_ELIMINATION with the same root cause: channel-invariant BN coefficient computation redundantly evaluated per-element.

## Alternative Approaches Considered

### 2D Tiling + Triton LICM (does NOT close the gap)

With `config.triton.prefer_nd_tiling = True`, Inductor generates a 2D kernel (Y=98304=128*768, X=49=7*7). In theory, per-channel computations (sqrt, reciprocal) should be loop-invariant across the X dimension and hoistable by Triton's LICM. In practice, this only yields 12% improvement (25.65us vs 28.76us) because:
- Y = batch * channel, so the per-channel values still vary along Y (they vary with channel index y%768)
- Triton's LICM cannot hoist ops whose inputs depend on program_id in any dimension
- The thread block tiles both Y and X simultaneously; per-channel ops depend on yindex

### Codegen-Level Hoisting (not currently possible)

Inductor's Triton codegen has no mechanism to hoist arbitrary compute out of the main loop body. The `prologue` mechanism is specific to TMA block pointer descriptors. A codegen-level fix would require:
- Detecting that certain loads/computations depend only on a subset of the iteration dimensions
- Emitting them in a separate code block before the main iteration
- This is equivalent to what Triton's LLVM backend LICM should do, but doesn't for cross-block-dimension dependencies

## Conclusion

This gap has two viable fix paths:

**Path 1 (FX pass + scheduler cost model)**: Precompute combined BN coefficients at graph level and fix the scheduler to not re-inline them. Expected improvement: 1.5x (from 26us to ~17us). The remaining gap to oracle (17us vs 14us) requires autotuning.

**Path 2 (Scheduler-only, broader impact)**: Fix the scheduler's fusion cost model to penalize recomputing transcendental ops (rsqrt, sqrt, exp) when the amplification ratio (consumer_numel / producer_numel) exceeds a threshold. This would naturally prevent the [768] coefficient ops from being fused into the [128, 768, 7, 7] main kernel. This approach benefits all BN-affine patterns without per-pattern FX rewrites.

The most impactful single fix would be the scheduler cost model change (Path 2), as it addresses the general pattern of "expensive per-channel ops redundantly computed per-element in a fused kernel."
