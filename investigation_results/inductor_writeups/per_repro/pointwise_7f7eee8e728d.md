# pointwise_7f7eee8e728d

## Classification: BANDWIDTH_BOUND_LAYOUT_COPY

## Current Result

- Oracle path: `repros/canonical/pointwise_7f7eee8e728d/oracle_padded_transpose.py`
- Correctness: PASS
- Oracle: 30.34 us
- Compile (cd=True): 36.42 us
- Ratio: 1.200
- Status: GOOD

## Root Cause

The oracle computes a padded transpose as one tiled layout-transform kernel that writes the final contiguous [768, 30524] result including zero pad columns. Inductor's compiled version handles the same operation but with slightly less efficient tiling for the combined transpose+pad pattern.

## Kernel Count

- Oracle: 1 kernel (fused transpose + pad)
- Inductor: 1-2 kernels (transpose then pad, or single kernel with less optimal tiling)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.200 |
| multi_kernel=2 | 1.053 |
| multi_kernel=3 | 1.050 |

multi_kernel=3 effectively closes the gap (ratio 1.050, within noise of AT_FLOOR). The persistent reduction / alternate tiling strategy from multi_kernel helps the layout transform hit better memory access patterns.

## File/Line References

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: tiling strategy for transpose kernels
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: layout-copy scheduling

## Design Doc

This gap is effectively closed by multi_kernel=3 (ratio drops to 1.05, at floor). The remaining gap at default config is a tiling quality issue for combined transpose+pad patterns. The oracle's manual tiling is slightly more cache-friendly than Inductor's auto-selected tile dimensions for this specific shape.

Best config: multi_kernel=3 (closes gap to floor).
