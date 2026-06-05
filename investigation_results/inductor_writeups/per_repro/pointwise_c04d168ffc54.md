# pointwise_c04d168ffc54 - RoPE Stencil (1.177x)

## Classification
SCHEDULER_FUSION

## Root Cause
The repro computes LLaVA RoPE (rotary position embedding) for two input matrices
(mm0, mm1) producing two outputs with non-contiguous strides [1,32,512,128] with
stride (2097152, 128, 4096, 1).

The oracle uses a specialized 2D grid (seq_pos x head_block) kernel that:
1. Loads per-position frequency values once
2. Computes cos/sin once per position
3. For each head, loads both x and rotate_half(x) with direct offset arithmetic
4. Writes both outputs (out0, out1) from the same kernel

Inductor already fuses this into 1 kernel with both outputs, which is good. However,
it uses conditional loads (`tl.where(dim < 64, load_upper, load_lower)`) for the
rotate_half pattern, which introduces masking overhead. It also uses a 1D grid over
the full 2M elements rather than a 2D grid that would give better cache locality
for the frequency loads.

## Kernel Count
- Oracle: 1 kernel (2D grid: SEQ x ceil(HEADS/BLOCK_H))
- Inductor: 1 kernel (1D grid over 2097152 elements)

Both produce 2 outputs - Inductor's fusion is correct.

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| coordinate_descent + combo | 11.49 | Default compiled output |
| multi_kernel=2 | 39.01 | Worse - multi_kernel doesn't help here |
| Oracle | 9.76 | Specialized stencil indexing |

## Why Inductor Cannot Fix This Today
The gap comes from two codegen quality issues:
1. **rotate_half lowering**: The `cat([-x[..., n:], x[..., :n]])` pattern gets
   lowered to conditional loads with `tl.where`, adding branch overhead. A
   pattern-matched rotate_half -> direct offset arithmetic rewrite would help.
2. **Grid dimensionality**: Inductor uses a 1D grid for all pointwise kernels.
   A 2D grid with (position, head_group) would allow frequency values to be loaded
   once per position and reused across heads in the same block.

## File References
- `torch/_inductor/codegen/triton.py` - 1D grid pointwise codegen
- `torch/_inductor/lowering.py` - cat/slice lowering that produces conditional loads

## Status
Design doc only. The rotate_half pattern recognition requires an FX pass that detects
`cat(neg(slice_hi), slice_lo)` and replaces it with direct index arithmetic. This
would benefit all RoPE implementations across transformer models.
