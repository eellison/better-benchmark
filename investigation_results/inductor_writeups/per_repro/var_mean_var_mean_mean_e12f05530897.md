# var_mean_var_mean_mean_e12f05530897

## Summary
- **Model**: timm_repvgg_a2_train_000 (RepVGG dual BN + ReLU + spatial mean)
- **Pattern**: Two per-channel var_mean reductions over [128, 1408, 7, 7], running-stat copy_, affine, branch sum, ReLU, spatial mean -> [128, 1408]
- **Oracle**: 60.29 us (2 kernels: partial stats + finalize/normalize/pool)
- **Compile (CDT)**: 65.25 us (2 kernels)
- **Ratio**: 1.082x
- **Classification**: PERSISTENT_THRESHOLD

## Root Cause

The oracle uses a split-K approach with 8 partial-sum blocks per channel to compute dual BN statistics, then a second kernel finalizes stats, normalizes, applies affine, sums branches, ReLU, and spatial mean. Inductor generates 2 kernels as well (same kernel count), but with default CDT the tile sizes are suboptimal for the reduction over 6272 elements per channel (128*7*7).

The gap is purely an autotuning/tile-selection issue, not a fusion gap. With `multi_kernel=3`, Inductor achieves 41.45us which is **faster** than the oracle's 60.29us.

## Kernel Count
- Oracle: 2 kernels
- Inductor: 2 kernels (same fusion achieved)

## Config Exploration
| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| CDT (default) | 65.25 | 1.082x |
| CDT + combo + multi_kernel=2 | ~43 | 0.71x |
| CDT + combo + multi_kernel=3 | 41.45 | 0.69x |

**multi_kernel=3 fully resolves this gap** -- Inductor beats the oracle by 31%.

## Fix Assessment
**Resolved by config** - `triton.multi_kernel=3` (a table-stakes allowed config) not only closes the 8% gap but exceeds oracle performance. The remaining delta with default config is an autotuning quality issue in tile size selection for the 6272-element per-channel reduction.

### Relevant files
- `/tmp/pytorch-work/torch/_inductor/choices.py` - reduction strategy/tile selection
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - tile heuristics

## Status: resolved_by_config (multi_kernel=3)
