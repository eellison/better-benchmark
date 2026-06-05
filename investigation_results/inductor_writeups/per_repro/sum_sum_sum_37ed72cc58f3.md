# sum_sum_sum_37ed72cc58f3

## Summary

- Model: Swin Transformer (window-reverse indexed layernorm-backward / drop-path)
- Oracle: `oracle_cooperative_split_k.py`
- Classification: COOPERATIVE_SPLIT_K
- Ratio: 2.025x (oracle 119.87us, compile 242.69us)
- Kernel count: Inductor multiple kernels, Oracle 1 fused cooperative split-K kernel

## Root Cause

The repro computes the Swin window-reverse/indexed layernorm-backward/drop-path backward. The oracle applies a dynamic height/width index gather, reduces each 256-wide row for the input-gradient epilogue, writes a `[256, 100352]` transposed side output, and accumulates three returned `[256]` column reductions from the same row-tiled producer in one coordinated split-K Triton kernel.

Inductor separates the window layout clone/indexing, row reductions, residual/drop-path pointwise epilogue, transposed side-output store, and sibling reductions into separate generic kernels over materialized intermediates.

The 2.025x gap comes from:
1. Multiple kernel launches with intermediate materialization
2. No cooperative split-K multi-output reduction template
3. Cannot keep dynamic indexed layout reconstruction, row-local reductions, a dependent transposed side output, and sibling column accumulators in one producer

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 242.69 | 2.025x |
| multi_kernel=2 | 243.58 | 2.033x |
| multi_kernel=3 | 243.58 | 2.033x |

No config helps -- the structural scheduling gap dominates.

## Fix Assessment

**Design doc** -- requires COOPERATIVE_SPLIT_K scheduler enhancement.

### What's needed:
Teach Inductor to split compatible row-tiled reductions across the indexed window producer, fuse the dependent epilogue/store, and finalize the sibling channel sums together. The scheduler needs a cooperative multi-output reduction template that coordinates dynamic indexed layout reconstruction with finalized per-channel summaries feeding full-tensor side-output epilogues.

### Files:
- `torch/_inductor/scheduler.py` (can_fuse, score_fusion)
- `torch/_inductor/choices.py` (cooperative reduction heuristic)
- `torch/_inductor/codegen/triton.py` (split-K codegen)
