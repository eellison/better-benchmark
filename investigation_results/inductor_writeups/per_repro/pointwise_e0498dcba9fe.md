# pointwise_e0498dcba9fe

## Classification: BROADCAST_AFFINE_FLAT_TILING

## Current Result

- Oracle path: `repros/canonical/pointwise_e0498dcba9fe/oracle_row_affine.py`
- Model: hf_MobileBertForMaskedLM_infer_000
- Correctness: PASS
- Oracle: 14.21 us (CUDA graph replay: 8.21 us)
- Compile (cd=True, combo=True): 35.78 us (CUDA graph replay: 30.84 us)
- Ratio: 2.518
- Status: GOOD (significant gap)

## Config Exploration

| Config | Compile (us) | Ratio | Status |
|--------|-------------|-------|--------|
| Default (cd=True, combo=True) | 35.78 | 2.518 | GOOD |
| multi_kernel=2 | 35.78 | 2.404 | GOOD |
| multi_kernel=3 | 35.78 | 2.524 | GOOD |
| use_fast_math=True | 35.74 | 2.516 | GOOD |
| prefer_nd_tiling=True | 35.78 | 2.535 | GOOD |

No config closes the gap. The compile time is constant at ~35.78us across all configs.

## Root Cause Analysis

The repro computes a MobileBERT residual affine transform on a [32768, 512] tensor:
```
output = (addmm + residual) * scale + bias
```
where scale and bias are [512] vectors broadcast across 32768 rows.

**Oracle approach**: 2D autotuned kernel with BLOCK_M rows x BLOCK_N=512 columns. Processes entire rows at a time, loading scale[0:512] and bias[0:512] once per tile and broadcasting across BLOCK_M rows. Autotuned configs include (1x128, 2x128, 4x128, 8x128, 1x512, 2x512, 4x512) with varying warps. The 2D access pattern gives perfect coalescing for contiguous row-major memory.

**Inductor approach**: 1D flat tiling with `xnumel = 16777216`. Uses `x0 = (xindex % 512)` for column index and `x2 = xindex` for flat offset. Loads broadcast parameters with `in_ptr2 + (x0)` and `in_ptr3 + (x0)`. The kernel is correct and fused (1 kernel), but the 1D linearization means:
1. Adjacent threads in a warp may cross row boundaries, causing sub-optimal broadcast reuse
2. The modular arithmetic adds instruction overhead
3. The XBLOCK tile size doesn't align to the 512-column boundary, so broadcast loads don't get optimal register reuse within a tile

**Why the gap is large (2.5x)**: On B200 with 126MB L2 cache, the working set (64MB input + 64MB residual + 64MB output = 192MB) partially fits in L2 during CUDA graph replay. The oracle's row-aligned tiling achieves much better L2 hit rates because it processes complete rows sequentially, keeping scale/bias in registers and accessing the large tensors in a streaming pattern. Inductor's flat 1D access pattern interleaves elements from different rows within a tile, causing more L2 thrashing.

## Kernel Count

- Inductor: 1 kernel (triton_poi_fused_add_mul_view_0)
- Oracle: 1 kernel (_row_affine_kernel)

Both are already single-kernel solutions. The gap is purely in tiling strategy.

## Design Issue

Inductor's pointwise codegen always uses 1D flat linearization. For broadcast-heavy patterns (where a small vector is broadcast across many rows), a 2D tiling approach that processes full rows per tile would:
1. Load broadcast parameters once into registers per tile
2. Achieve perfect memory coalescing for row-major data
3. Avoid modular arithmetic for column index extraction

The `prefer_nd_tiling` flag does not help here because the generated 2D kernel still doesn't get autotuned tile sizes optimized for this access pattern.

**What's needed**: A broadcast-aware pointwise template that detects when inputs have different ranks (e.g., [M, N] and [N]) and tiles in 2D with the broadcast dimension as the inner tile, ensuring the broadcast vector loads once per outer tile iteration. This is essentially the "row affine template" pattern.

## File References

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - pointwise tiling strategy
- `/tmp/pytorch-work/torch/_inductor/ir.py` - broadcast detection for tiling decisions
- `/tmp/pytorch-work/torch/_inductor/codegen/triton_utils.py` - grid/block selection

## Status

Open - genuine 2.5x gap that persists across all standard configs. Requires broadcast-aware 2D pointwise tiling template in Inductor codegen.
