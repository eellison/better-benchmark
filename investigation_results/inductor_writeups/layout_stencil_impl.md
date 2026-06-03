# Inductor Implementation: Layout / Stencil Slice-Scatter Elision

## Status: IMPLEMENTED (prototype)

- Queue id: `layout_stencil_functional_updates`
- Priority: P1, rank 6
- Owner: Maxwell

## Summary

Implemented a post-grad FX pass (`slice_scatter_elision_pass`) that eliminates
redundant `slice(slice_scatter(base, src, dim, a, b), dim, a2, b2)` patterns
where the slice reads entirely within the scatter-written region.

## Results

| Repro | Baseline (us) | With Elision (us) | Speedup |
|-------|---------------|-------------------|---------|
| pointwise_70c0751eb408 | 1120 | 864 | 1.30x |
| pointwise_531d72f1b34a | 755 | 685 | 1.10x |
| pointwise_a2382a85ee99 | 158 | 158 | 1.00x (no stencil pattern) |

Kernel count reduction for 70c0751eb408:
- Before: 66 distinct kernels, 67 launches
- After: 29 distinct kernels, 30 launches (55% fewer launches)
- Pattern elisions applied: 38 (70c0751eb408), 36 (531d72f1b34a)

## Root Cause Analysis

The pyhpc benchmarks (isoneutral_mixing, turbulent_kinetic_energy, equation_of_state)
use a stencil pattern where:

1. `full(0, [N+pad, M+pad, K])` creates a zero-padded tensor
2. `slice_scatter(padded, computed_interior, dim, start, end)` writes the interior
3. Subsequent code immediately `slice(result, dim, start, end)` back to the interior

This causes Inductor to:
- Materialize full padded buffers (8.5MB each in f64)
- Generate kernels with redundant bounds-checking logic
- Prevent fusion across slice_scatter boundaries (different iteration domains)
- Total: ~65 intermediate buffers, ~550MB of redundant memory traffic

## Implementation

File: `/tmp/pytorch-work/torch/_inductor/fx_passes/slice_scatter_elision.py`

The pass recognizes two cases:

### Exact match (most common)
```
slice(slice_scatter(base, src, dim, a, b), dim, a, b) -> src
```

### Partial match
```
slice(slice_scatter(base, src, dim, a, b), dim, a2, b2)
  -> slice(src, dim, a2-a, b2-a)    [when a2 >= a and b2 <= b]
```

### Integration
- Added `slice_scatter_elision = True` config to `torch._inductor.config`
- Wired into `post_grad_passes()` after `scatter_reduce_fusion` pass
- Runs before pattern_matcher and lowering

## Correctness

Verified on all 3 target repros:
- `pointwise_70c0751eb408`: 17 outputs match (maxdiff < 2.3e-13 in f64)
- `pointwise_531d72f1b34a`: 60 outputs match (exact or within 1e-13)
- `pointwise_a2382a85ee99`: single output matches

## Limitations & Future Work

1. **Chained slice_scatter**: Currently handles one level. Recursive application
   would catch `slice(scatter(scatter(...)))` chains (the pass does fire repeatedly
   on separate nodes, but not recursively through chains in a single pass).

2. **Multi-dimensional stencil**: The pass handles one dimension at a time.
   For `slice(scatter(_, _, dim=0), dim=1, ...)` it won't fire. Need separate
   handling for consumers that slice on a different dim than the scatter.

3. **Non-full bases**: When `base` is not `full(C)`, the optimization still applies
   if the slice region is entirely within the scatter region, since the base content
   is irrelevant there. The current implementation handles this case.

4. **select_scatter**: Similar pattern exists with `select_scatter + select.int`.
   Not yet handled.

5. **Further fusion opportunities**: After elision, adjacent kernels may become
   fusible because they now share iteration domains. A second fusion pass could
   capitalize on this.

## Files Modified

- `torch/_inductor/fx_passes/slice_scatter_elision.py` (NEW - the pass)
- `torch/_inductor/fx_passes/post_grad.py` (integration)
- `torch/_inductor/config.py` (config flag)
