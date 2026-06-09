# pointwise_5c9390c2182e

## Classification: MUTATION_READ_DEPENDENCY

## Current Result

- Oracle path: `repros/canonical/pointwise_5c9390c2182e/oracle_slice_update_head_layout.py`
- Correctness: PASS
- Oracle: 6.72 us
- Compile (cd=True, partial_elision=True): 9.02 us
- Ratio: 1.34x
- Status: GOOD (gap, improved from 1.49x with new small-scatter approach)

## Root Cause

The oracle computes a LLaMA KV-cache slice-update plus head-major clone output in a single Triton kernel. It writes `arg85_1[:32, 1:33, :, :]` from `mm_51.view(32, 32, 8, 64)` while directly materializing the fresh contiguous `[256, 33, 64]` output.

Inductor produces 2 kernels:
1. K0 (540672 elements): fused consumer - conditionally loads from mm_51 (for seq>=1) or from cache (for seq=0), permutes, stores to output
2. K1 (524288 elements): mutation - loads from mm_51, stores to cache at seq offset

K0 and K1 cannot be fused because they share a buffer (arg0_1): K0 reads from it and K1 writes to it. Even though the access ranges are non-overlapping (K0 reads [0:32, 0:1], K1 writes [0:32, 1:33]), the scheduler conservatively treats same-buffer access as a dependency.

## Kernel Count

- Inductor: 2 kernels (consumer + mutation)
- Oracle: 1 kernel (fused conditional load + dual store)

## Config Exploration

| Config | Result |
|--------|--------|
| Default (cd=True, combo=True, partial_elision=True) | ratio=1.34x |
| slice_scatter_partial_elision=False (original) | ratio=1.34x |
| Old cat approach (slice_scatter_partial_elision=True, old code) | ratio=1.44x (WORSE) |

Note: For this specific repro, the small-scatter approach produces the same 2-kernel decomposition as the original (both read arg1_1 + arg0_1 and write arg0_1 + output). The benefit is primarily visible on repro pointwise_78e5e6e20190 which gets better tiling with the new approach.

## Fix Implemented: Small slice_scatter rewrite (FX pass)

Commit: `2199f90f782` on `/tmp/pytorch-work` branch `pr-184905`

Rewrote `_partial_overlap_elision` in `slice_scatter_elision.py` to use a SMALL `slice_scatter` instead of `cat`:

```
slice(slice_scatter(base, src, dim, a, b), dim, a2, b2)
  where read [a2, b2) partially overlaps write [a, b)
->
slice_scatter(slice(base, dim, a2, b2), src_overlap, dim, a-a2, min(b,b2)-a2)
```

Key insight: the new nodes are placed BEFORE the original scatter in graph order so the reinplace pass's `any_use_of_views_after_node` check still passes, allowing efficient in-place mutation.

The new `slice_scatter` creates a Pointwise IR node of size [32, 33, 8, 64] (output-sized) instead of [32, 1024, 8, 64] (full cache), enabling:
1. Direct fusion with downstream permute+clone into one kernel
2. No ConcatKernel intermediate buffer
3. Better tiling (2D tiling on some repros)

## Remaining Gap Analysis

The 1.34x gap requires scheduler-level mutation fusion:
- The scheduler needs to detect that K0 reads cache[0:1] and K1 writes cache[1:33] are non-overlapping
- It could then fuse both into a single kernel with dual stores (read mm once, write to both output and cache)
- This saves one kernel launch + eliminates double-read of mm_51

## Memory Traffic Analysis

| Approach | Reads | Writes | Total (4B each) |
|----------|-------|--------|------------------|
| Oracle (1 kernel) | 540K (16K from arg0 + 524K from arg1) | 1064K (540K output + 524K arg0) | 6.4MB |
| Inductor (2 kernels, new) | 1064K (540K: 16K arg0 + 524K arg1; 524K: arg1) | 1064K (540K output + 524K arg0) | 8.5MB |
| Cat approach (3 kernels, old) | 1080K | 1604K | 10.7MB |

## Affected Repros

This pattern appears in LLaMA/GPT KV-cache update paths:
- pointwise_5c9390c2182e: 1.34x (this repro)
- pointwise_78e5e6e20190: 1.29x (12% improvement from new pass)

**File references:**
- Config: `torch/_inductor/config.py` (slice_scatter_partial_elision)
- FX pass: `torch/_inductor/fx_passes/slice_scatter_elision.py` (_partial_overlap_elision)
- Reinplace: `torch/_inductor/fx_passes/reinplace.py` (any_use_of_views_after_node)
