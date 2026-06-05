# sum_sum_sum_17d1ea77dce0

## Summary

- Model: MobileViT (patch layout reductions)
- Oracle: `oracle_mobilevit_layout_reductions.py`
- Classification: MOBILEVIT_LAYOUT_REDUCTION
- Ratio: 1.775x (oracle 171.7us, compile 304.9us)
- Kernel count: Inductor 7 kernels, Oracle fewer (fused layout + reduction)

## Root Cause

The repro computes the MobileViT 2x2 patch layout transformation followed by multiple reductions. Input is `[128, 144, 32, 32]` NCHW that gets reshaped through:
`clone -> view [294912, 2, 16, 2] -> permute [294912, 16, 2, 2] -> clone -> view [294912, 64] -> permute -> clone -> view [512, 256, 144]`

This layout pipeline produces a `[512, 256, 144]` tensor from which:
1. Per-row hidden reductions are computed (sum over dim=1 -> [512, 144] or similar)
2. Column reductions produce [144] vectors
3. A full [144, 131072] side view is materialized for downstream use

Inductor lowers the clone/view/permute chain as separate copy kernels, then schedules the reductions independently. The 7 kernels include layout copies + separate reduction passes.

The oracle maps rows directly back to source NCHW indices (computing the permute/reshape algebraically), fuses the row reductions with the layout transform, and emits both reduction outputs and the materialized side tensor from fewer passes.

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT (default) | 304.9 |
| multi_kernel=2 | (failed/slow) |
| multi_kernel=3 | (failed/slow) |

No standard config closes the gap. The issue is structural: Inductor cannot eliminate the layout copies or fuse through them.

## Fix Assessment

**Design doc** -- requires a layout-fusion optimization.

### What's needed:
1. **Algebraic layout folding**: Recognize that `clone -> view -> permute -> clone -> view -> permute -> clone -> view` is a pure index remap. Instead of materializing each intermediate, compute the composite index transformation and apply it directly in the reduction kernel.

2. **Through-layout reduction**: The scheduler needs to recognize that reductions over the final layout can be expressed as reductions over the source layout with transformed indices, eliminating the need to materialize the transposed intermediate.

### Difficulty: High
The MobileViT patch-unfold pattern involves multiple non-trivial permutations. Teaching the scheduler to "see through" this chain to the original NCHW source requires significant IR analysis.

## Relevant Files

- `/tmp/pytorch-work/torch/_inductor/ir.py`: View/Permute IR nodes
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: layout copy fusion
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`: layout simplification passes

## Affected Repro Count

MobileViT-specific pattern. Likely affects a small number of repros (MobileViT variants only).
