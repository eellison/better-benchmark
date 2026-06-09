# pointwise_4ea254f5c392

## Classification: SCALED_TRANSPOSE_LAYOUT_COPY

## Current Result

- Oracle path: `repros/canonical/pointwise_4ea254f5c392/oracle_scaled_layout_transpose.py`
- Correctness: PASS
- Oracle: 20.86 us
- Compile (cd=True): 22.98 us
- Ratio: 1.101
- Status: GOOD (gap)

## Root Cause

The oracle computes a scaled transpose in one kernel: `input[512, 512, 64]` -> view to `[8, 64, 512, 64]` -> mul 0.354 -> permute `[0,2,1,3]` -> clone to contiguous `[8, 512, 64, 64]` -> view to `[4096, 4096]` -> transpose to `[4096, 4096]` stride `[1, 4096]`.

The final output is `[4096, 4096]` with stride `[1, 4096]` (column-major). The oracle does this in one kernel: read input, apply scale, write to transposed output layout.

Inductor likely generates separate kernels for:
1. The scale + permute (pointwise with layout change)
2. The final transpose materialization (or uses a non-fused copy)

The output layout requirement (stride `[1, 4096]` i.e. column-major) means a full transpose must occur, and Inductor may not fuse the mul_scalar into that transpose kernel efficiently.

## Kernel Count

- Inductor: 1-2 kernels (scale + transpose, or fused but with suboptimal tiling)
- Oracle: 1 kernel (fused scale + transpose with optimal tiling)

## Config Exploration

| Config | Result |
|--------|--------|
| Default (cd=True) | ratio=1.101 |
| multi_kernel=2 | N/A (pointwise, not reduction) |
| multi_kernel=3 | N/A (pointwise, not reduction) |

## Diagnosis

The gap is relatively small (10.1%). Inductor does fuse pointwise ops into layout copies, but the specific combination of:
1. View reshape (non-trivial index remapping)
2. Scalar multiply
3. Permute + clone to specific non-contiguous output stride

may result in suboptimal tiling for the transpose. The oracle can tile specifically for the 64x64 head-dim blocks which gives better memory coalescing.

## Design Doc

**What's needed**: Better tiling heuristics for fused pointwise+transpose kernels. When the dominant cost is a transpose (large matrix with non-contiguous output), the codegen should select tile dimensions that match the transpose block size for optimal memory coalescing.

**File references**:
- `torch/_inductor/codegen/triton.py` (transpose tiling selection)
- `torch/_inductor/codegen/triton_heuristics.py` (tile size heuristics)

**Affected repro count**: At least 1 confirmed here. Similar scaled-transpose patterns likely appear in other attention head materializations.
