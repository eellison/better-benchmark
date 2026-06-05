# var_mean_var_mean_40a0055bb26e - oracle_bn_shuffle

## Classification
SCHEDULER_FUSION

## Benchmark Results
- Oracle: 45.22 us (1 kernel)
- Inductor: 53.02 us (1 kernel)
- Ratio: 1.173x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT | 53.02 |
| combo + multi_kernel=3 + per_subkernel_blocks | 39.15 |
| combo + multi3 + coop_red | same range |

**multi_kernel=3 helps significantly** (53.02 -> 39.15 us), bringing Inductor *below* the oracle. This suggests the issue is autotuning tile sizes rather than a fundamental fusion gap.

## Root Cause

The repro (from Shufflenet) computes dual training-BatchNorm with channel-shuffle:
- Two per-channel var_mean reductions over [512, 232, 7, 7]
- Running stat updates (copy_)
- Affine epilogues
- cat/view/permute/clone/view channel shuffle
- Split return

Inductor already fuses this into 1 kernel (same as oracle!). The performance gap is purely from **tile size selection** - the oracle hand-picks an optimal BLOCK_K for the reduction dimension, while Inductor's autotuner with default CDT finds a suboptimal configuration.

With `multi_kernel=3` (which tries multiple tile configs and picks the best at runtime), Inductor achieves 39.15 us which is **better** than the oracle's 45.22 us.

## Kernel Count
- Oracle: 1 kernel
- Inductor: 1 kernel (same fusion achieved!)

## Fix Assessment
**Resolved by config** - `triton.multi_kernel=3` closes the gap and exceeds oracle performance.

The remaining delta with default config is an autotuning quality issue, not a codegen/scheduling gap. The scheduler correctly fuses everything into one kernel. The fix is ensuring CDT or multi_kernel finds the optimal tile sizes.

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/choices.py` - reduction strategy/tile selection
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - tile heuristics

### Status: No code fix needed
multi_kernel=3 (an allowed table-stakes config) fully resolves this gap.
