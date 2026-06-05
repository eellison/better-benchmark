# pointwise_1c5b4956323d — embedding + constant offset gather

## Summary
- **Repro**: `hf_BartForCausalLM_train_003`
- **Ratio**: 1.068 (oracle 6.11us vs compile 6.53us)
- **Classification**: CODEGEN_OVERHEAD (device_assert)

## What the oracle does
The oracle implements unsqueeze -> add(2) -> embedding as a single Triton kernel that:
1. Computes token index from flat offset: `token = offsets // HIDDEN`
2. Loads the index: `row = tl.load(ids_ptr + token) + ROW_OFFSET`
3. Directly gathers from the embedding table: `tl.load(table_ptr + row * HIDDEN + col)`

No bounds checking is performed (the oracle trusts that indices are valid after adding the constant offset).

## What Inductor does
Inductor generates a fused kernel `triton_poi_fused_add_embedding_unsqueeze_0` that performs the same computation but includes a per-element `tl.device_assert`:
```
tl.device_assert((0 <= tmp2) & (tmp2 < 1026), "index out of bounds: 0 <= tmp2 < 1026")
```

This assert is evaluated for every element (1048576 total), adding overhead for the bounds check comparison and branch.

## Kernel count
- Inductor: 1 kernel
- Oracle: 1 kernel

## Config exploration
The gap is small (6.8%) and comes entirely from the device_assert overhead. With `assert_indirect_indexing=False` in Inductor config, this assert would be elided and performance should match.

## Root cause
Inductor's indirect indexing lowering always emits `tl.device_assert` for embedding gathers to guard against out-of-bounds indices. For cases where the index is provably in-bounds (e.g., constant offset added to known-range indices, or vocab_size is known), this check is redundant overhead.

## Fix path
**Design doc**: A range-proof pass could track index bounds through the IR:
- If the embedding indices come from `arange` or are known to be in `[0, N)`, and the offset is a compile-time constant, the assert can be elided.
- File: `/tmp/pytorch-work/torch/_inductor/lowering.py` (embedding lowering, `index_impl`)
- File: `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (`assert_indirect_indexing` emission)
- Config: `torch._inductor.config.assert_indirect_indexing` (already exists, but defaults to True)

Given the small gap (6.8%), this is low priority but affects all embedding-heavy models (transformers). The fix would be to implement compile-time index range analysis rather than blanket assertion.
