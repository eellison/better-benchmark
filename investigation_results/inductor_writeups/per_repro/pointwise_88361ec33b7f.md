# pointwise_88361ec33b7f — oracle_cat_div

## Summary
**Status**: GOOD (ratio=1.08x)
**Classification**: BANDWIDTH_BOUND / minor indexing overhead

## Benchmark Results
- Oracle: 6.43 us
- Compile: 6.94 us
- Ratio: 1.08x (oracle is 8% faster)

## Root Cause

The oracle computes a cat(bmm[32,1,1] reshaped to [32,1], mm[32,32000]) / 0.07
in one row-tiled Triton kernel with explicit column-block scheduling and a simple
branch (cols == 0 selects bmm, else mm). The oracle uses `triton.autotune` with
4 configs varying BLOCK_COLS (512/1024/2048/4096).

Inductor generates a single fused kernel (`triton_poi_fused_cat_div_permute_unsqueeze_view_0`)
that correctly fuses all view/permute/cat/div ops. Both approaches produce 1 kernel.

The 8% gap comes from:
1. **Indexing overhead**: Inductor uses a 1D grid over all 1,024,032 elements and
   computes `x0 = xindex % 32001` and `x1 = xindex // 32001` per element, whereas
   the oracle uses a 2D grid `(ROWS, cdiv(OUT_COLS, BLOCK_COLS))` which avoids
   the integer division entirely.
2. **Non-power-of-2 stride**: The output has stride (32001, 1) which means the
   modulo operation is expensive (not a power of 2).

## Kernel Count
- Inductor: 1 kernel
- Oracle: 1 kernel

## Config Exploration
- `coordinate_descent_tuning = True`: no improvement (already single kernel)
- `combo_kernels = True`: no impact (already single kernel)
- `max_autotune = True`: no improvement

## Inductor Code Analysis

Inductor's kernel (from `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`):
```
x0 = (xindex % 32001)   # expensive non-power-of-2 modulo
x1 = xindex // 32001    # expensive non-power-of-2 division
```

Oracle's kernel avoids this entirely:
```
row = tl.program_id(0)           # free
cols = block * BLOCK_COLS + arange  # simple addition
```

## Design Doc

This is a minor gap (8%) caused by Inductor's generic 1D tiling strategy for
pointwise kernels with non-power-of-2 inner dimensions. A potential fix would be:

**Enhancement needed**: When the output tensor has a non-power-of-2 inner dimension
and the kernel is cat-like (selecting from different sources based on column index),
Inductor could emit a 2D grid where program_id(0) = row and program_id(1) = column
block, avoiding the expensive integer division/modulo.

**File**: `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (grid selection logic)
**File**: `/tmp/pytorch-work/torch/_inductor/runtime/triton_heuristics.py` (tiling heuristics)

However, the 8% gap on a 7us kernel (0.5us absolute) may not justify the
complexity of a specialized 2D cat tiling pass. This is close to bandwidth floor.

## Affected Repros
This pattern (cat with non-power-of-2 output + scalar div) is relatively uncommon.
The gap is small and at the noise threshold for many benchmarking setups.
