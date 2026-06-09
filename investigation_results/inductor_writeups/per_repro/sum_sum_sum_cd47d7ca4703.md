# sum_sum_sum_cd47d7ca4703 (hf_AllenaiLongformerBase_train_006)


## Measured Timings
- Oracle: 77.47 us
- Compile (CDT): 132.93 us
- Ratio: 1.72x

## Classification: SCATTER_REDUCE (embedding backward)

## Summary

| Metric | Value |
|--------|-------|
| Baseline gap | 1.72x |
| Best config | 1.71x (multi_kernel=3) |
| scatter_add_into impact | None (no add-after-scatter pattern) |
| Oracle kernels | 3 |
| Inductor kernels | 4 |

## Root Cause

Same family as Electra/Albert but without the `add(mm_1, scatter)` final pattern.
Returns raw index_put results directly. The pass `scatter_add_into_fusion` does not
apply here because there is no `add(existing_grad, scattered)` step.

The gap is due to:
1. Materializing `mul_tensor_6` [8, 1024, 768] for 3 scatter consumers + batch-sum
2. Separate batch-reduction kernel before the sequence position scatter
3. Separate reduction kernels for sum_dim_int_list_2/3

## Config Exploration

| Config | Ratio |
|--------|-------|
| Baseline | 1.72x |
| scatter_add_into_fusion | 1.72x (no pattern match) |
| multi_kernel=3 | 1.71x |
| scatter_reduce_fusion (env) | 1.72x (no pattern match for this type) |

## Remaining Gap (1.71x)

Entirely due to REDUCTION_EPILOGUE_REREAD + MULTI_OUTPUT_SHARED_REDUCTION.
The oracle recomputes layer-norm backward inline in each of its 3 Triton kernels,
avoiding the materialization of the [8, 1024, 768] intermediate.

Requires scheduler-level recomputation heuristic to close.
