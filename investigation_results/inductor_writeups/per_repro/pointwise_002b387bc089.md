# pointwise_002b387bc089

## Classification: PERMUTE_COPY_FLAT_TILING

## Current Result

- Oracle path: `repros/canonical/pointwise_002b387bc089/oracle_layout_copy.py`
- Correctness: PASS
- Oracle: 24.38 us
- Compile (cd=True): 29.50 us
- Ratio: 1.21
- Status: GOOD (oracle wins)

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| default (cd=True, combo=True) | 29.50 |
| multi_kernel=1 | 31.60 |
| multi_kernel=2 | 59.96 |
| multi_kernel=3 | 31.27 |

No config closes the gap.

## Root Cause

The repro is a pure layout copy: `permute([0,2,1,3]) -> clone(contiguous) -> view` on a `[128, 12, 198, 64]` tensor (DeiT attention head rearrangement).

**Oracle approach**: 2D tiling with BLOCK_ROWS=4 rows at a time, processing 512+256 columns per row. Each row corresponds to a (batch, seq) pair, and the oracle computes head/dim indices explicitly. This gives good memory coalescing: adjacent threads process adjacent columns in the output (contiguous stores) and read with known strides.

**Inductor approach**: 1D flat tiling with modular arithmetic decomposition (`x0 = xindex % 64`, `x1 = (xindex//64) % 12`, etc.). The load pattern is `in_ptr0 + (x0 + 64*x2 + 12672*x1 + 152064*x3)` which produces scattered reads when the head dimension (x1, stride 12672) varies across adjacent elements.

Both produce exactly 1 kernel with 1 load and 1 store per element (same memory traffic). The difference is purely in memory access pattern quality - the oracle's 2D tiling gives better L2 hit rates and coalescing for this specific permute pattern.

## Kernel Count

- Inductor: 1 kernel (triton_poi_fused_clone_permute_0)
- Oracle: 1 kernel (_layout_copy_kernel)

## Design Issue

Inductor's pointwise codegen uses a flat 1D linearization for all permute-clone patterns. For transpose-heavy layouts (permuting the inner dimensions), a 2D tiling approach that tiles along output rows and processes full column widths per tile would give better coalescing. This requires the scheduler/codegen to detect "permute-clone" patterns and select a 2D grid with row-major output writes.

**Relevant files**:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (tiling strategy selection)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (layout detection for permute patterns)

## Conclusion

Real gap due to suboptimal tiling strategy for permute-clone kernels. The fix would require teaching Inductor's triton codegen to use 2D tiling (row-major output, explicit index math per dimension) when the operation is a pure layout copy with transposed inner dimensions.
