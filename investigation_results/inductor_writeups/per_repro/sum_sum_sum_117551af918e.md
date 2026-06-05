# sum_sum_sum_117551af918e

## Summary

- Model: hf_DistillGPT2_train (also GPT2, GPT2_large -- 4 models total)
- Oracle: `oracle_cooperative_split_k.py`
- Classification: COOPERATIVE_SPLIT_K
- Ratio: 1.481x (oracle 59us, compile 88us)
- Kernel count: Inductor 3 kernels, Oracle 1 kernel (cooperative)

## Root Cause

The repro is a layer-norm-backward + dropout-mask pattern from GPT2 training:
- Input: `mm_1 [16384, 768]` (matmul output), plus layernorm state and a bool mask
- Computes: `add -> mul(weight) -> [row_sum, weighted_row_sum] -> sub/sub -> mul(scale) -> [convert_bool_mask * result]`
- Returns: 3 x `[768]` column sums (sum over dim=[0,1] of different expressions)

Inductor emits 3 kernels:
1. `triton_per_fused_add_mul_sum_view_0`: Row-local reductions for the layernorm backward
2. `triton_red_fused_add_convert_element_type_mul_sub_sum_view_1`: Partial column reduction of the masked gradient
3. `triton_red_fused_2`: Finalization / remaining column reductions

The oracle uses a cooperative split-K approach: row tiles compute row-local statistics and simultaneously accumulate column sums via cooperative partial buffers. One kernel launch handles all three output reductions by:
1. Each row tile computes the two row-sum scalars needed for layernorm backward
2. Same tile immediately applies the dropout mask and accumulates into 3 column-sum partials
3. A final tile-level reduction sums the partials

## What Inductor Cannot Do Today

1. **Multi-output cooperative reduction**: Inductor cannot emit a single kernel that performs row reductions (dim=-1) and simultaneously accumulates column reductions (dim=0) from the same data pass.
2. **Dependent reduction fusion**: The column sum of `(masked_gradient)` depends on the row-level layernorm backward computation. Inductor schedules these as separate kernels because the column reduction depends on the full pointwise epilogue.
3. **Split-K with shared producers**: The scheduler has no template for "compute row stats, apply epilogue, cooperatively reduce columns" in one launch.

## Fix Path

**COOPERATIVE_SPLIT_K template**: Teach Inductor's `choices.py` or scheduler to emit a split-K kernel where:
- Each block processes a tile of rows
- Block-local row reductions drive a pointwise epilogue
- Column accumulators are cooperatively reduced across blocks

**Relevant files:**
- `torch/_inductor/choices.py`: `should_use_cooperative_reduction` heuristic
- `torch/_inductor/scheduler.py`: Multi-output reduction scheduling
- `torch/_inductor/codegen/triton.py`: Cooperative reduction codegen

## Config Exploration

`coordinate_descent_tuning=True` and `combo_kernels=True` do not help because the issue is structural (missing multi-axis reduction fusion), not tuning.

## Affected Repro Count

This pattern (hash 117551af918e) appears in 4 models: DistillGPT2, GPT2, GPT2_large variants. The cooperative split-K fix would benefit all layernorm-backward-with-dependent-column-reduction patterns.
