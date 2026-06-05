# sum_sum_sum_6a2ad206248c (hf_ElectraForCausalLM_train_001)

## Classification: SCATTER_REDUCE (embedding backward)

## Summary

| Metric | Value |
|--------|-------|
| Baseline gap | 2.01x |
| Best with fix | 1.91x (scatter_add_into + multi_kernel=3) |
| Default config gap | 1.98x (scatter_add_into only) |
| Oracle kernels | 3 |
| Inductor kernels (before) | 5 |
| Inductor kernels (after fix) | 3 |

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

## Fix Implemented

**Commit**: e034b931eab in /tmp/pytorch-work (branch pr-184905)

**Pass**: `config.scatter_add_into_fusion = True` (default enabled)

Rewrites: `add(A, index_put(zeros, idx, val, accumulate=True))` -> `index_put(A, idx, val, accumulate=True)`

This eliminates:
- The full(0) initialization of the [30522, 128] buffer
- The separate add kernel that reads both mm_1 and the scatter result

**Files**:
- `/tmp/pytorch-work/torch/_inductor/config.py` (new config flag)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (pass registration)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/scatter_reduce_fusion.py` (implementation)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Baseline (no fix) | 2.01x |
| scatter_add_into_fusion (default) | 1.98x |
| + multi_kernel=2 | 1.92x |
| + multi_kernel=3 | 1.91x |

## Remaining Gap (1.91x with best config)

The remaining gap is due to:
1. **REDUCTION_EPILOGUE_REREAD**: `mul_tensor_6` [32768, 128] must be materialized because it has 3 scatter consumers + the batch-sum consumer. The oracle recomputes layer-norm backward inline per scatter kernel.
2. **MULTI_OUTPUT_SHARED_REDUCTION**: `sum_dim_int_list_2/3` (reducing [32768, 128] -> [128]) cannot be fused with the main kernel that already iterates all rows because the main kernel's reduction dimension is the hidden (128) dim, not the batch*seq (32768) dim.

These require scheduler-level enhancements:
- Recomputation heuristic (cheaper to recompute than materialize when consumers are scatter operations)
- Cooperative atomic reduction (accumulate global sums via atomic_add during scatter iteration)
