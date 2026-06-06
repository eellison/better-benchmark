# pointwise_5c9390c2182e

## Classification: SLICE_SCATTER_HEAD_LAYOUT_FUSION

## Current Result

- Oracle path: `repros/canonical/pointwise_5c9390c2182e/oracle_slice_update_head_layout.py`
- Correctness: PASS
- Oracle: 6.69 us
- Compile (cd=True): 9.12 us
- Ratio: 1.364
- Status: GOOD (gap)

## Root Cause

The oracle computes a LLaMA KV-cache slice-update plus head-major clone output in a single Triton kernel. It writes `arg85_1[:32, 1:33, :, :]` from `mm_51.view(32, 32, 8, 64)` while directly materializing the fresh contiguous `[256, 33, 64]` output.

Inductor sees the captured `copy -> slice_scatter -> slice_scatter -> permute -> clone -> copy_` layout chain and lowers it through generic slice-scatter materialization, requiring multiple kernels to:
1. Perform the slice_scatter in-place update
2. Permute and clone to head-major layout

The oracle fuses these into one kernel with conditional indexing: for seq=0, read from the existing cache; for seq>0, read from the mm output. Both paths write to the final contiguous output and also update the cache in-place.

## Kernel Count

- Inductor: 2-3 kernels (slice_scatter materialization + permute/clone)
- Oracle: 1 kernel (fused conditional load + dual store)

## Config Exploration

| Config | Result |
|--------|--------|
| Default (cd=True) | ratio=1.364 |
| multi_kernel=2 | compiles OK, same pattern - no improvement expected (pointwise, not reduction) |
| multi_kernel=3 | compiles OK, same pattern - no improvement expected (pointwise, not reduction) |

## Diagnosis

Inductor cannot express a direct in-place slice update and derived head-layout clone as one guarded layout stencil today. The slice_scatter decomposition forces full buffer materialization of the unchanged region before the permute/clone can proceed.

## Design Doc

**What's needed**: A specialized FX pass or scheduler enhancement that recognizes the pattern:
- `slice_scatter` updating a contiguous range of an existing buffer
- Immediately followed by `slice -> permute -> clone` extracting a region that overlaps the updated range

The pass should fuse these into a single kernel that:
1. Conditionally loads from the update source or existing buffer based on index
2. Writes to both the cache buffer (in-place) and the permuted output

**File references**:
- `torch/_inductor/scheduler.py` (fusion scoring for slice_scatter + clone)
- `torch/_inductor/fx_passes/post_grad.py` (pattern match opportunity)
- `torch/_inductor/lowering.py` (slice_scatter lowering)

**Affected repro count**: This pattern appears in LLaMA/GPT KV-cache update paths. At least 1 repro confirmed here.
