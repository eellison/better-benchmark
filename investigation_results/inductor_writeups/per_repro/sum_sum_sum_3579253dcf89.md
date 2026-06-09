# sum_sum_sum_3579253dcf89

## Summary

- Model: RepVGG-A2 (timm_repvgg_a2_train)
- Oracle: `oracle_shared_bn_dual_reduction.py`
- Classification: BANDWIDTH_BOUND
- Ratio: 1.213x (oracle 704.7us, compile 855.1us)

## Root Cause

The repro computes a RepVGG BN-backward-style fragment with one shared masked producer (ReLU backward via `where`), one three-accumulator channel reduction (`sum(where)`, `sum(where * centered1)`, `sum(where * centered2)`), both returned scale-gradient vectors, and both dense BN epilogues over [128, 64, 112, 112] f32 activations.

The oracle fuses the masked producer and all three reductions into a single pass, reading each activation element once and accumulating three channel-wise partial sums simultaneously. Inductor separates the masked pointwise from the reductions and runs each reduction as a separate kernel, causing redundant re-reads of the large [128, 64, 112, 112] activations (~392MB total per tensor).

The 21.3% gap comes from Inductor's inability to fuse the shared masked producer with the multi-accumulator reduction in one kernel launch.

## Config Exploration

| Config | Compile Time (us) |
|--------|-------------------|
| baseline (CDT) | 855.1 |
| multi_kernel=2 | CRASH (ND reduction assert) |
| multi_kernel=3 | CRASH (ND reduction assert) |

multi_kernel=2/3 trigger an Inductor bug: `AssertionError: Expected ND reduction size ({'r0_': 65536}) to have 131072 elements`. This is a separate Inductor bug in persistent_reduction config generation for this spatial reduction shape [128, 64, 112, 112] -> [64].

## Fix Assessment

**Scheduler fusion** -- Teach Inductor to fuse a pointwise producer (the masked `where`) with multiple same-axis channel reductions over [N, C, H, W] -> [C] into a single multi-accumulator kernel.

### Additional bug found
multi_kernel=2/3 triggers `_get_nd_reduction_numels` assertion failure for spatial reductions. This should be filed separately.

### Difficulty: Medium-High
The reductions share the same producer but Inductor's current scheduler cannot group them when they have different elementwise pre-processing (plain sum vs product-sum).

## Status: open_gap
