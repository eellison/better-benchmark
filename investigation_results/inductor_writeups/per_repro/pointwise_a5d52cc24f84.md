# pointwise_a5d52cc24f84 — Embedding with Index Offset

## Summary
- **Model**: vllm_facebook_opt-125m_001
- **Classification**: ALGEBRAIC_ELIMINATION
- **Ratio**: 1.094x (oracle 6.11us vs compile 6.69us)
- **Status**: Minor gap

## Root Cause

The repro computes `add(arg0_1, 2)` followed by `embedding(arg1_1, add_result)`. Inductor materializes the int64 add result as an intermediate tensor before launching the embedding gather. The oracle folds the constant offset (+2) directly into the embedding table address computation, eliminating the intermediate i64 tensor.

## Kernel Count
- **Inductor**: 2 kernels (1 for int64 add, 1 for embedding gather)
- **Oracle**: 1 kernel (offset folded into gather address)

## Config Exploration

No config helps here. This requires an algebraic rewrite pass.

## What the Oracle Does

The `_embedding_offset_kernel` loads the index from `ids_ptr`, adds `index_offset` (=2) inline, then uses the result as the row index into the embedding table. No intermediate tensor is materialized for the shifted indices.

## Fix Location

- **File**: `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` or a new pass file
- **Enhancement needed**: Add a pattern-matching pass that detects `add.Tensor(int64_indices, constant) -> embedding.default(table, shifted_indices)` and folds the constant offset into the embedding lowering. The embedding codegen in `lowering.py` would need to accept an optional index offset parameter.

## Design Doc

This is a relatively simple algebraic elimination:
1. Detect that an int64 add with a scalar constant feeds directly into an embedding gather
2. Eliminate the add node and pass the offset to the embedding lowering
3. The generated gather kernel adds the offset inline

The gap is small (1.094x) because the int64 add kernel is very cheap (just element-wise addition on indices), and the embedding gather is already memory-bound. The main savings come from eliminating one kernel launch and the intermediate i64 buffer write/read.

**Affected patterns**: Any model that uses offset embeddings (OPT, some BERT variants with position_ids + 2).
