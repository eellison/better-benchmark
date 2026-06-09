# sum_sum_sum_83ddb84cc830 (hf_MobileBertForMaskedLM_train_014)


## Measured Timings
- Oracle: 114.27 us
- Compile (CDT): 185.25 us
- Ratio: 1.62x

## Classification: SCATTER_REDUCE (embedding backward)

## Summary

| Metric | Value |
|--------|-------|
| Baseline gap | 1.73x |
| Best config | 1.73x (no fix applies) |
| scatter_add_into impact | None (no add-after-scatter pattern) |
| Oracle kernels | 3 |
| Inductor kernels | 5 |

## Root Cause

MobileBERT backward with shared `[256, 128, 512]` intermediate (= [batch*seq, 512]).
Unlike Electra/Albert, does NOT have `add(existing_grad, scatter)` - returns raw scatter
results plus a permuted side output. The scatter_add_into_fusion pass does not apply.

The gap is due to:
1. Materializing the full [256, 128, 512] intermediate (= 16M elements)
2. Separate scatter kernels for type-token [2, 512] and position [512, 512] outputs
3. Separate global reductions for sum_dim_int_list outputs
4. Additional permute side output forces materialization

## Config Exploration

| Config | Ratio |
|--------|-------|
| Baseline | 1.73x |
| scatter_add_into_fusion | 1.73x (no match) |
| multi_kernel=2 | 1.95x (worse) |
| multi_kernel=3 | 1.96x (worse) |

multi_kernel hurts this workload. Default CDT is optimal.

## Remaining Gap (1.73x)

Entirely structural - requires REDUCTION_EPILOGUE_REREAD fix:
- Scheduler recomputation heuristic to avoid materializing [256, 128, 512]
- Multi-destination atomic scatter template that iterates rows once and writes all outputs
- The permute side output (`permute_default: [512, 32768]`) adds complexity
