# pointwise_78e5e6e20190

## Classification: KV_CACHE_SLICE_SCATTER

## Current Result

- Oracle path: `repros/canonical/pointwise_78e5e6e20190/oracle_kv_cache_head_layout.py`
- Correctness: PASS
- Oracle: 6.91 us
- Compile (cd=True, partial_elision=True): 8.90 us
- Ratio: 1.29x
- Status: GOOD (improved from 1.49x with small-scatter FX pass)

## Root Cause

The oracle computes the Llama inference KV-cache update and key head-layout materialization in one Triton launch. It writes the [0:32, 1:33, :, :] cache slice from the packed real view while directly producing the fresh contiguous [B*H, D, S] = [256, 64, 33] attention-key output.

Inductor (with fix) produces 2 kernels:
1. K0 (540672 elements, 2D tiling): fused consumer - conditionally loads from view_as_real_15 (for seq>=1) or cache (for seq=0), permutes, stores to output
2. K1 (524288 elements): mutation - loads from view_as_real_15, stores to cache at seq offset

## Kernel Count

- Oracle: 1 kernel (fused cache write + head-layout output)
- Inductor (with fix): 2 kernels (fused consumer + mutation)
- Inductor (without fix): 2 kernels (mutation + unfused permute/clone reading post-mutation buffer)

## Fix Implemented: Small slice_scatter rewrite (FX pass)

Commit: `2199f90f782` on `/tmp/pytorch-work` branch `pr-184905`

The `_partial_overlap_elision` pass in `slice_scatter_elision.py` rewrites:
```
slice(slice_scatter(base, src, dim, 1, 33), dim, 0, 33)
```
to:
```
slice_scatter(slice(base, dim, 0, 33), src, dim, 1, 33)
```

This creates a Pointwise node of size [32, 33, 8, 64] that:
- Conditionally loads from src (positions 1-32) or base (position 0)
- Fuses with downstream permute+clone into one kernel with 2D tiling

Key improvement: the 2D tiling (ynumel=16384, xnumel=33) provides better memory coalescing than the original 1D tiling (xnumel=540672) which had scattered access through the large stride-524288 cache buffer.

## Config Exploration

| Config | Ratio |
|--------|-------|
| Default (cd=True, partial_elision=True) | 1.29x |
| partial_elision=False (original) | 1.49x |
| multi_kernel=2 | no improvement (pointwise) |
| multi_kernel=3 | no improvement (pointwise) |

## Remaining Gap Analysis

The 1.29x gap requires scheduler-level mutation fusion:
- K0 reads cache[0:32, 0:1] and src; K1 reads src and writes cache[0:32, 1:33]
- These are non-overlapping but the scheduler sees same-buffer access as a dependency
- Fusing into one kernel with dual stores would save one kernel launch + avoid double-reading src

## File/Line References

- Config: `torch/_inductor/config.py` (slice_scatter_partial_elision)
- FX pass: `torch/_inductor/fx_passes/slice_scatter_elision.py` (_partial_overlap_elision)
- Reinplace: `torch/_inductor/fx_passes/reinplace.py` (any_use_of_views_after_node)

## Affected Repros

This pattern affects all Llama-style KV-cache inference workloads:
- pointwise_78e5e6e20190: 1.49x -> 1.29x (this repro, 12% improvement)
- pointwise_5c9390c2182e: ~1.34x (same pattern, similar gap)
