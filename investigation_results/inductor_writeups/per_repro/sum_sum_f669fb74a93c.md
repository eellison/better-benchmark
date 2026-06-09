# sum_sum_f669fb74a93c

## Summary

- Oracle: `oracle_full_scope_masked_broadcast_sum.py`
- Classification: MULTI_OUTPUT_SHARED_REDUCTION
- Ratio: 1.34x (oracle 11.4us, compile 15.3us)
- Kernel count: Inductor 4 kernels, Oracle 1 kernel

## Root Cause

The repro computes a masked broadcast-product followed by multiple reductions. Inputs:
- `arg52_1`: bool mask [2048, 1]
- `arg54_1`: float row values [2048, 1]
- `arg13_1`: float col values [1, 1024]
- `arg51_1`: float gate [2048, 1024]

Operations:
1. Mask rows: `where(mask, 0, row_values)` -> store row base + row sum
2. Broadcast product: `masked_rows * col_values` gated by `relu_gate > 0`
3. Store full product [2048, 1024] + column sum [1024]

The oracle does all of this in a single column-tiled kernel: each tile loads the full 2048 rows (persistent), computes the masked rows, broadcasts with the column slice, gates, stores the product tile, and accumulates the column sum. The row sum is computed once in tile 0.

Inductor splits this into 4 kernels because:
1. The row reduction (sum over 2048 rows) and the column reduction (sum over 2048 rows for each column) are separate scheduling regions
2. The full product store is a separate pointwise kernel
3. The scheduler cannot recognize that the broadcast producer is cheap enough to recompute per column tile

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT (default) | 15.3 |
| multi_kernel=2 | 60.9 |
| multi_kernel=3 | 61.4 |

No config closes the gap. multi_kernel=2/3 make it significantly worse (likely because they force all reductions to one mode unsuitable for this mixed-output pattern).

## Fix Assessment

**Design doc** -- requires scheduler enhancement to fuse:
- A scalar row reduction (sum of 2048 values)
- A 2D product store [2048, 1024]
- A column reduction [1024] (sum over axis 0 of the product)

All sharing the same masked-row computation. The scheduler would need to recognize that the broadcast producer (`masked_rows[:, None] * col_values[None, :]`) is a cheap virtual intermediate that can be computed per-tile without materialization.

## Relevant Files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: multi-output reduction scheduling
- `/tmp/pytorch-work/torch/_inductor/ir.py`: realize_hint for broadcast intermediates
