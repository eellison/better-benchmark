# sum_sum_sum_3e04a44dd549 - GhostNet Two-Branch BN Backward (Cooperative Split-K)

## Classification: COOPERATIVE_SPLIT_K

## Benchmark Results
- Oracle: 44.93 us
- Inductor compiled: 66.4 us
- Ratio: 1.478x (oracle is 1.48x faster)

## Kernel Count
- Inductor: 1 kernel (single fused reduction)
- Oracle: 3 kernels (split-K reduce + full BN epilogue + slice BN epilogue)

## Root Cause

The repro computes the GhostNet two-branch batch-norm-backward for shapes:
- Full branch: [512, 160, 7, 7] (add + clone + copy + BN-backward)
- Slice branch: [512, 80, 7, 7] (slice of the full at channels [80:160])

Both branches need:
1. Channel sum: sum(x, [0,2,3]) -> [C] vector
2. Channel dot: sum(x * centered, [0,2,3]) -> [C] vector
3. BN-backward epilogue using those sums to produce full/slice output tensors + scale grad vectors

The oracle exploits the shared producer (clone_3 + getitem_63 = add) between both branches:
- First kernel: cooperative split-K reduction with atomic accumulation, computing all 4 channel sums (full_sum, full_dot, slice_sum, slice_dot) in one pass, sharing the add computation
- Second/third kernels: BN-backward epilogues that read the finalized sums and write the output tensors in appropriate memory formats

Inductor fuses everything into a single kernel but:
1. Cannot share the add producer across both branches efficiently within one reduction schedule
2. The single kernel must handle both the full [160] and sliced [80] channel reductions simultaneously, leading to suboptimal memory access patterns
3. The epilogue (writing full [512,160,7,7] contiguous + [512,80,7,7] channels-last) is bandwidth-heavy and would benefit from separation

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| coordinate_descent + combo_kernels | 66.4 | Default compiled |
| Oracle (split-K + dual epilogue) | 44.93 | 1.48x faster |

## What Inductor Needs (Design Doc)

**Enhancement needed**: Cooperative split-K multi-output reduction template in scheduler/codegen.

The fix would:
1. Detect sibling channel reductions over shared producers with slice relationships
2. Emit a cooperative split-K pass with atomic_add coordination that computes both branch sums simultaneously
3. Separate the bandwidth-heavy epilogue stores into their own kernels with appropriate tiling per output layout

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - detect shared-producer sibling reductions with slice
- `/tmp/pytorch-work/torch/_inductor/choices.py` - cooperative reduction strategy selection
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - atomic split-K template

**Affected repro count**: The GhostNet/DenseNet two-branch BN-backward with shared producers appears across multiple sum_sum_sum repros (likely 3-5 in corpus).
