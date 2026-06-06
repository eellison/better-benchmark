# pointwise_5c9390c2182e

## Classification: MUTATION_READ_DEPENDENCY

## Current Result

- Oracle path: `repros/canonical/pointwise_5c9390c2182e/oracle_slice_update_head_layout.py`
- Correctness: PASS
- Oracle: 6.91 us
- Compile (cd=True): 9.02 us
- Ratio: 1.306
- Status: GOOD (gap)

## Root Cause

The oracle computes a LLaMA KV-cache slice-update plus head-major clone output in a single Triton kernel. It writes `arg85_1[:32, 1:33, :, :]` from `mm_51.view(32, 32, 8, 64)` while directly materializing the fresh contiguous `[256, 33, 64]` output.

Inductor produces 2 kernels:
1. K0 (524288 elements): loads from arg1_1 (mm_51), stores to arg0_1 at seq offset +512 (the in-place cache update)
2. K1 (540672 elements): loads from arg0_1 (including the just-written data), permutes, stores to output

K1 depends on K0 via a RAW (read-after-write) dependency on arg0_1. The scheduler cannot fuse them because K1 reads data that K0 just wrote. The oracle avoids this by conditionally loading from mm_51 (for seq>=1) or arg0_1 (for seq=0) directly, eliminating the round-trip through arg0_1.

## Kernel Count

- Inductor: 2 kernels (mutation + permute/clone)
- Oracle: 1 kernel (fused conditional load + dual store)

## Config Exploration

| Config | Result |
|--------|--------|
| Default (cd=True, combo=True) | ratio=1.306 |
| max_autotune=True | ratio=1.301 |
| multi_kernel=2 | no improvement (pointwise, not reduction) |
| multi_kernel=3 | no improvement (pointwise, not reduction) |
| slice_scatter_partial_elision=True | ratio=1.44 (WORSE - adds cat buffer overhead) |

## Investigation Details

### FX-level partial elision pass (implemented but disabled)

Added `_partial_overlap_elision` to `slice_scatter_elision.py`. The pass recognizes:
```
slice(slice_scatter(base, src, dim, start, end), dim, read_start, read_end)
```
where `read_start < start` (read extends before scatter write region).

Rewrites to:
```
cat([clone(slice(base, dim, read_start, start)), src], dim)
```

This correctly breaks the RAW dependency and allows the reinplace pass to optimize the mutation. However, it introduces a ConcatKernel that materializes the cat buffer, adding extra memory traffic (16K clone + 524K extra write to cat buffer + 540K read from cat buffer). Net result: 3 kernels and WORSE performance.

### Why the reinplace pass needed special handling

The slice+clone of the base must be placed BEFORE the scatter in topological order AND the clone must break storage aliasing. Otherwise, the reinplace pass's `any_use_of_views_after_node` check sees the base slice as a conflicting view use and falls back to a full-buffer functional scatter (33M elements instead of 524K).

### The correct fix: scheduler-level mutation fusion

The true fix requires the scheduler to recognize the pattern:
- Producer (mutation): iterates over [32, 32, 8, 64], loads from src, stores to buf at offset
- Consumer: iterates over [32, 8, 33, 64], loads from buf, permutes, stores to output

And fuse them into:
- Single kernel iterating over [32, 8, 33, 64] (consumer's space):
  - For indices in producer's write range: load from src (arg1_1), store to BOTH output AND buf (mutation)
  - For indices outside write range: load from buf (arg0_1), store to output only

This requires:
1. Detecting that the consumer's read range partially overlaps with the producer's write range
2. Computing the index remapping from consumer's iteration space to producer's source
3. Creating a fused inner_fn with conditional logic and dual stores

**File references:**
- Config: `torch/_inductor/config.py` (slice_scatter_partial_elision flag)
- FX pass: `torch/_inductor/fx_passes/slice_scatter_elision.py` (_partial_overlap_elision)
- Scheduler: `torch/_inductor/scheduler.py` (inline_recomputable_producers area - needs new mutation_fusion pass)

## Memory Traffic Analysis

| Approach | Reads | Writes | Total (4B each) |
|----------|-------|--------|------------------|
| Oracle (1 kernel) | 540K (16K from arg0 + 524K from arg1) | 1064K (540K output + 524K arg0) | 6.4MB |
| Inductor (2 kernels) | 1064K (524K arg1 + 540K arg0) | 1064K (524K arg0 + 540K output) | 8.5MB |
| Cat approach (3 kernels) | 1080K | 1604K | 10.7MB |

The oracle saves 2MB by avoiding the re-read of 524K elements from arg0_1 that were just written.

## Affected Repros

This pattern appears in LLaMA/GPT KV-cache update paths. At least 1 repro confirmed here. The pattern is: `copy_to_slice + read_overlapping_region + permute + clone`.
