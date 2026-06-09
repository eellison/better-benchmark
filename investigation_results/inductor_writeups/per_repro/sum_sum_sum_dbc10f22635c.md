# sum_sum_sum_dbc10f22635c

## Summary

- Model: ConvNeXt-style multi-output reduction (scatter/expand/div/sum tail)
- Oracle: `oracle_multi_output.py`
- Classification: ALGEBRAIC_ELIMINATION (FIXED)
- Previous Ratio: 1.325x (oracle 10.05us, compile 13.31us)
- Current Ratio: 0.815x (Inductor BEATS oracle: compile 8.9us vs oracle 10.91us)
- Status: **FIXED** (BAD_ORACLE - Inductor now faster)

## Fix

Covered by TWO existing FX passes working together:
1. `as_strided_scatter_elision` - eliminates the full(0) + as_strided_scatter + as_strided chain
2. `expand_sum_elision` - eliminates sum(expand(x, [128,640,7,7]) / 49, [0,2,3]) -> sum(x, [0])

These were already committed:
- as_strided_scatter_elision: committed in earlier work
- expand_sum_elision: commit `c33b0e78618`

## Root Cause (historical)

The oracle computed the complete scope with one Triton pass including the scatter/expand/div/sum tail folded into a direct accumulation. Inductor now achieves the same via algebraic simplification: the scatter-expand-reduce tail is proven equivalent to a simple channel reduction.

## Kernel Count (after fix)
- Oracle: 1 fused single-pass kernel
- Inductor: 2 kernels (but total faster due to better tiling/autotuning)
