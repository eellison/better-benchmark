# sum_sum_sum_92327e661e73

## Summary

- Model: Adv-Inception channels-last BN-backward (6 branches)
- Oracle: `oracle_cooperative_split_k.py`
- Classification: COOPERATIVE_SPLIT_K
- Ratio: 1.871x (oracle 59.14us, compile 110.62us)
- Kernel count: Inductor many kernels, Oracle 1 split-K reduction + 1 epilogue

## Root Cause

The oracle computes the complete channels-last Adv-Inception `mm / 64` fanout plus all six ReLU-gated batch-norm-backward branch reductions and dependent tensor/vector epilogues with one coalesced split-K Triton reduction followed by one channels-last epilogue.

Inductor currently schedules the expanded `mm`, sliced views, ReLU masks, twelve `sum([0, 2, 3])` reductions, and BN-backward epilogues as separate pointwise/reduction kernels over materialized intermediates.

The 1.87x gap comes from:
1. Multiple separate reduction kernels for each branch
2. No cooperative template to coordinate disjoint channels-last channel slices
3. Cannot reuse the shared sliced `mm / 64` producer across branches

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 110.62 | 1.871x |
| multi_kernel=2 | 100.29 | 1.696x |
| multi_kernel=3 | 105.44 | 1.783x |

multi_kernel=2 provides ~9% improvement (bringing ratio to 1.70x) but doesn't close the gap. The structural issue is the lack of cooperative multi-output reduction across disjoint channel slices.

## Fix Assessment

**Design doc** -- requires COOPERATIVE_SPLIT_K scheduler enhancement.

### What's needed:
Teach Inductor to split compatible `(N, H, W)` channel reductions, reuse the shared sliced `mm / 64` producer, and fuse each branch's gradient/vector epilogue while preserving channels-last layout. This requires a cooperative multi-output reduction template that coordinates disjoint channels-last channel slices with finalized per-channel summaries feeding full-tensor side-output epilogues.

### Files:
- `torch/_inductor/scheduler.py` (multi-branch reduction fusion)
- `torch/_inductor/choices.py` (cooperative reduction heuristic)
- `torch/_inductor/codegen/triton.py` (channels-last split-K codegen)
