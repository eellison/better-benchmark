# var_mean_9e94237b5daf

## Classification
SCHEDULER_FUSION (BN_AFFINE_RECOMPUTATION residual)

## Root Cause
The oracle computes the complete MobileNetV3 BN-training + hardswish path in a single
per-channel kernel: var_mean reduction, running stat updates (copy_), affine transform,
and hardswish activation for shape [512, 960, 7, 7] (rnumel=25088, xnumel=960).

Inductor also fuses into 1 kernel (after fixing the crash bugs), but the generated
kernel's tiling is less optimal than the oracle's per-channel layout with BLOCK=32768.

## Bug Fix (primary deliverable)
Two crash bugs in `scheduler.py` prevented compilation of this repro:

1. **RecursionError in `rename()`** (line 4525): `mutation_renames` could form cycles,
   causing infinite recursion. Fixed by converting recursive `rename()` to iterative
   with a visited set for cycle detection.

2. **KeyError in `compute_ancestors`/`compute_input_distances`** (lines 4945, 4970):
   After `inline_recomputable_producers` removes nodes and re-runs dependency
   computation, some `unmet_dependencies` reference buffers whose defining ops are
   either removed (DCE'd), self-referential, or appear later in the topological order.
   Fixed by gracefully skipping deps that resolve to ops not yet processed.

## Kernel Count
- Oracle: 1 kernel (per-channel, BLOCK=32768)
- Inductor: 1 kernel (fused reduction + pointwise)

## Config Exploration
| Config | Ratio |
|--------|-------|
| default (coord_descent=True) | 1.259x |
| multi_kernel=1 | 1.193x |
| multi_kernel=2 | 1.194x |
| multi_kernel=3 | 1.193x |

All multi_kernel settings give approximately 1.19x. The remaining gap is kernel quality
(tiling strategy), not fusion.

## Fix Implemented
- Commit: `c465f1751c6` on branch `pr-184905` in `/tmp/pytorch-work`
- Before: RecursionError / KeyError crash (could not compile)
- After: Compiles successfully, ratio ~1.19-1.26x depending on config

## Files Modified
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` lines 4524-4533, 4937-4955, 4958-4987

## Residual Gap (1.19x)
The remaining gap is a kernel quality issue: the oracle tiles by channel (960 CTAs, each
processing 25088 elements) which gives perfect data locality. Inductor's tiling uses a
different decomposition. This is classified as BN_AFFINE_RECOMPUTATION / persistent
reduction threshold interaction -- the per-channel tiling with large BLOCK_R is not
selected by the Inductor heuristics.
