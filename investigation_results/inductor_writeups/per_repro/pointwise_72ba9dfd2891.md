# pointwise_72ba9dfd2891 — oracle_batchnorm_silu

## Summary
Oracle: 13.92us | Compiled: 16.03us (default), 16.38us (CDT only) | Ratio: 1.152x | Status: GOOD

## Classification: SCHEDULER_FUSION (channel-broadcast tiling)

## Root cause
The oracle uses a 3D grid `(N, cdiv(C, BLOCK_C), cdiv(HW, BLOCK_S))` to compute batch normalization + SiLU for a [64, 1152, 7, 7] fp16 tensor. The channel-spatial decomposition means:
1. Per-channel parameters (mean, var, weight, bias) are loaded once per `(n, c_block)` program and reused across the entire spatial tile
2. Indexing is trivial affine arithmetic: `(n * C + c) * HW + hw`
3. The autotuned configs explore different BLOCK_C/BLOCK_S ratios (4x64, 8x64, 16x32, 8x128)

Inductor generates a 1D flattened pointwise kernel over 3,612,672 elements. It uses:
- `x1 = ((xindex // 49) % 1152)` for channel extraction (integer division + modulo)
- Channel parameters loaded with `evict_last` policy (good), but the div/mod overhead remains
- No explicit spatial tiling or parameter reuse across the spatial dimension

The performance gap (~15%) comes from:
1. Integer div/mod overhead for index decomposition in the hot loop
2. Suboptimal occupancy/tiling compared to the oracle's explicit 3D decomposition
3. The oracle's autotune finds the best BLOCK_C/BLOCK_S ratio while Inductor only tunes XBLOCK

## Kernel count
- Inductor: 1 kernel (fused pointwise 1D)
- Oracle: 1 kernel (3D grid, autotuned)

## Config exploration
| Config | Time (us) | Ratio |
|--------|-----------|-------|
| coordinate_descent_tuning=True, combo_kernels=True (default bench) | 16.03 | 1.152x |
| coordinate_descent_tuning=True only | 16.38 | 1.177x |
| multi_kernel=3, combo_kernels=True | 17.51 | 1.258x |

None of the config explorations improve over the default. The gap is codegen-level (1D vs 3D tiling), not a config issue.

## File/line references
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py:257` — block types: XBLOCK, YBLOCK, ZBLOCK; pointwise only uses XBLOCK
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py:149-150` — tiling score thresholds
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` — fusion scoring

## Design doc: Why it can't be fixed today

Inductor's pointwise codegen uses a 1D flattened iteration space (XBLOCK only) for all pointwise operations. The oracle's advantage comes from explicit 3D tiling that:
1. Extracts the channel dimension as its own grid axis, enabling parameter reuse without div/mod
2. Uses a 2D tile (BLOCK_C x BLOCK_S) that maps naturally to the NCHW layout
3. Autotunes the channel/spatial tile ratio

**What's needed**: A codegen enhancement to detect NCHW channel-broadcast pointwise patterns (where per-channel parameters are broadcast across spatial dims) and emit a multi-dimensional grid with:
- One axis for batch
- One axis for channel blocks
- One axis for spatial blocks

This would eliminate the `// 49 % 1152` index decomposition in the hot path and enable explicit parameter reuse. The key detection heuristic: when a pointwise kernel has inputs with shapes like [C] broadcast to [N,C,H,W], the channel dimension should be a separate tiling axis.

**Gap magnitude**: 15% - moderate priority. This pattern is common in EfficientNet and other NCHW inference workloads with batch normalization.
