# sum_7953a8b0bbba - oracle_sum_permute

## Classification
SCHEDULER_FUSION

## Benchmark Results
- Oracle: 7.81 us (1 kernel)
- Inductor: 11.74 us (3 kernels)
- Ratio: 1.504x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT | 11.74 |
| combo + multi_kernel=3 + per_subkernel_blocks | 29.29 |
| combo + multi3 + cooperative_reductions | 29.85 |

Config changes made the problem worse (multi_kernel overhead for small kernels). The base config with combo_kernels is already the best Inductor achieves.

## Root Cause

The repro (from hf_Whisper train) does:
1. Takes [8, 256] input
2. unsqueeze -> expand [8, 1500, 256] -> div by 1500 -> view [12000, 256]
3. Returns permute [256, 12000] as output 0
4. Returns sum(dim=0) [256] as output 1

The oracle fuses these into ONE kernel: it streams through the [12000, 256] data once, writing both the transposed [256, 12000] output AND accumulating the [256] column sum in the same pass.

Inductor generates 3 kernels:
1. `triton_poi_fused_div_expand_unsqueeze_0` - materializes the expanded/divided tensor
2. `triton_red_fused_div_expand_sum_unsqueeze_view_1` - partial reduction
3. `triton_red_fused_div_expand_sum_unsqueeze_view_2` - final reduction

The fundamental problem is that Inductor cannot fuse a **layout-changing materialization** (the permute output, which requires writing in transposed order) with a **sibling reduction** (the sum over the same data). The scheduler treats these as separate consumers of the same producer.

## Kernel Count
- Oracle: 1 kernel
- Inductor: 3 kernels

## Fix Assessment
**Design doc** - Cannot be fixed with config changes alone.

### What's needed:
The scheduler needs a "multi-output materializing reduction" pattern: when a producer is consumed by both a layout-changing store (permute/transpose) AND a reduction, emit a single kernel that iterates over the producer in reduction-friendly order, writing the transposed output AND accumulating partials.

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion decisions (can_fuse, score_fusion)
- `/tmp/pytorch-work/torch/_inductor/ir.py` - realize_hint blocks fusion when layout change is required
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - multi-output kernel emission

### Why it's hard:
The permute output requires data to be stored in [256, 12000] layout (stride=(12000,1) effectively), while the reduction sums along dim=0 of [12000, 256]. These have conflicting iteration orders. The oracle solves this by tiling [BLOCK_ROWS, BLOCK_COLS] and writing transposed tiles, which works for this shape but requires the codegen to recognize the compatible tiling opportunity.

### Affected repro count:
This pattern (layout-copy + sibling reduction) likely affects sum_b8db5e701976 as well (same structural pattern).
