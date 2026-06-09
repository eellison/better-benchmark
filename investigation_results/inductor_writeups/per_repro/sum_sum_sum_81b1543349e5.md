# sum_sum_sum_81b1543349e5

## Summary

- Model: timm_swin_base_patch4_window7_224_train
- Oracle: `oracle_cooperative_split_k.py`
- Classification: COOPERATIVE_SPLIT_K
- Ratio: 1.075x (oracle 43us, compile 46us) -- borderline
- Kernel count: Inductor 6 kernels, Oracle 1 kernel (cooperative)

## Root Cause

The repro is a Swin Transformer layer-norm-backward + residual add + drop_path + transposed side output:
- Input: `mm_11 [6272, 1024]` reshaped through window partition
- LayerNorm backward: row reductions for `sum(dim=-1)` producing row scalars
- Residual add + drop_path: mask multiplication and scaling
- Returns: 2 x `[1024]` column sums, 1 x `[1024, 6272]` transposed, 1 x `[1024]` final column sum

Inductor emits 6 kernels including MixOrderReduction finalization passes. The oracle uses cooperative split-K to handle all in one launch.

## Current Status

Previous measurement showed compile beating oracle (ratio=0.80x). Current measurement shows marginal oracle advantage (ratio=1.075x). This is hardware/run-variance sensitive and near the noise floor. The gap is likely due to Inductor needing 6 kernel launches vs 1, but each kernel is well-tuned.

## Fix Path (if gap persists)

Same COOPERATIVE_SPLIT_K template as sum_sum_sum_117551af918e: fuse row reductions, dependent epilogue, and column accumulators into one coordinated kernel. The smaller M dimension (6272 vs 16384) makes this less impactful.

## Status: BORDERLINE (ratio barely above 1.05)
