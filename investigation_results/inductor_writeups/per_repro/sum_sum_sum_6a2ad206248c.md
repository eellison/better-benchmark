# sum_sum_sum_6a2ad206248c (hf_ElectraForCausalLM_train_001)


## Measured Timings
- Oracle: 85.54 us
- Compile (CDT): 164.74 us
- Ratio: 1.93x

## Classification: REDUCTION_EPILOGUE_REREAD + SCATTER_REDUCE

## Summary

| Metric | Value |
|--------|-------|
| Baseline gap (before any fixes) | 2.01x |
| After scatter_add_into_fusion only | 1.98x |
| After + reduce_scatter_distribution | 1.93x (default config) |
| Best config (combo_kernels + multi_kernel=2) | 1.91x |
| Oracle kernels | 3 |
| Inductor kernels (after fixes) | 3 |

## Root Cause

This is an Electra model backward pass computing:
1. Layer-norm backward producing gradient `mul_tensor_6` [64, 512, 128]
2. Three `index_put(full(0), [idx], values, accumulate=True)` scatters from that gradient:
   - Sequence position scatter: [512, 128] (requires batch-dim reduction first)
   - Type-token scatter: [2, 128]
   - Vocabulary embedding scatter: [30522, 128]
3. Two global reductions: sum_dim_int_list_2/3 producing [128] vectors
4. Final: `add(mm_1, index_put_default_2)` - add existing gradient to vocabulary scatter

The oracle avoids:
- Materializing the [30522, 128] zeros buffer by initializing from mm_1 directly
- Materializing `mul_tensor_6` by recomputing layer-norm backward inline in each scatter kernel
- Separate reduction kernels by accumulating via atomic_add per-sequence-position
- The 2-pass reduction-epilogue pattern by using a different parallelization strategy (grid over seq, loop over batch in registers)

## Fixes Implemented

### Fix 1: scatter_add_into_fusion (previously committed)

**Commit**: e034b931eab in /tmp/pytorch-work (branch pr-184905)

Rewrites: `add(A, index_put(zeros, idx, val, accumulate=True))` -> `index_put(A, idx, val, accumulate=True)`

Eliminates the full(0) initialization of [30522, 128] and the separate add kernel.

### Fix 2: reduce_scatter_distribution (new)

**Commit**: 5e89fd4dc38 in /tmp/pytorch-work (branch pr-184905)

**Pass**: `config.reduce_scatter_distribution = True` (default enabled)

Rewrites: `sum(x, [0]) -> where(mask, fill, sum) -> index_put(zeros, [idx], val, acc=True)`
Into: `index_put(zeros, [idx_expanded], where(mask, fill/B, x), acc=True)`

Key algebraic identity: the accumulate=True in index_put implicitly performs the batch summation. For masked positions, scattering fill/B for each of B batch elements accumulates to fill (correct).

This eliminates:
- The explicit batch-dim sum reduction ([64, 512, 128] -> [1, 512, 128])
- The 16MB write of mul_tensor_6 to global memory (buf6)
- The 16MB read of mul_tensor_6 in the final reduction kernel
- The batch-reduction-then-scatter combo sub-kernel

**Files**:
- `/tmp/pytorch-work/torch/_inductor/config.py` (new config flag)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (pass registration)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/scatter_reduce_fusion.py` (implementation)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Baseline (no fixes) | 2.01x |
| scatter_add_into_fusion only | 1.98x |
| + reduce_scatter_distribution (default) | 1.93x |
| + combo_kernels | 1.91x |
| + combo_kernels + multi_kernel=2 | 1.91x |
| + combo_kernels + multi_kernel=0 | 1.89x (best, noisy) |

## Remaining Gap (1.91-1.93x)

The remaining gap is fundamentally due to **REDUCTION_EPILOGUE_REREAD**: the scheduler/codegen generates a 2-pass kernel structure for the LN-backward computation:

1. **Pass 1**: Reads mm_148, bool_mask, gamma, arg104. Reduces over hidden dim (128) to compute per-row statistics (sum_p, sum_prod). Also computes partial outer reductions (sum2/sum3) with mix-order fusion.

2. **Pass 2**: Re-reads mm_148, bool_mask, gamma, arg104 to compute the full gradient using the reduction results from pass 1. Performs 3 atomic_adds (token scatter, type scatter, embedding scatter).

The oracle avoids this by:
- Using a **different parallelization strategy**: grid over sequence positions (512), with batch accumulated in a register loop (64 iterations)
- Reading each row's data only ONCE: the hidden dim (128) fits in a single BLOCK, so the reduction and epilogue share registers
- Doing only 1 atomic per sequence position for token scatter (vs 64 in Inductor)

### Why this can't be fixed with existing scheduler/codegen:

1. **Reduction-epilogue re-read**: The codegen emits two loops over the reduction dimension. With small R0_BLOCK (< r0_numel), data is re-read from memory. With persistent reduction (R0_BLOCK = r0_numel = 128), data stays in registers and the "re-read" is free. The autotuner sometimes selects persistent, sometimes looped.

2. **Atomic contention**: The token scatter has 64 atomics per destination (one per batch element). The oracle accumulates in registers first, then does 1 atomic. This would require the scheduler to recognize the batch-loop-accumulate pattern and emit a fundamentally different kernel structure.

3. **Parallelization axis mismatch**: Inductor parallelizes over all 32768 rows. The oracle parallelizes over 512 sequence positions with an inner batch loop. The latter enables register accumulation and fewer atomics but requires a different grid structure that the scheduler doesn't generate.

### What enhancement is needed:

A "batch-loop scatter" codegen template that recognizes when:
- Multiple scatter consumers share the same source tensor
- The scatter index is broadcast over one dimension (batch)
- An inner loop over that dimension with register accumulation would reduce atomic pressure

This would require new scheduler logic to select a "grid-over-inner-dim, loop-over-outer-dim" strategy when the output involves scatter-adds that don't vary over the outer dimension.

## Affected Repro Count

This pattern (LN-backward + multi-scatter) appears in:
- ElectraForCausalLM backward variants
- Other transformer models with tied embeddings and token-type embeddings
