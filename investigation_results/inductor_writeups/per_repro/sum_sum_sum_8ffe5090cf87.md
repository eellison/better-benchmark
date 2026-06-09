# sum_sum_sum_8ffe5090cf87 (hf_AlbertForMaskedLM_train_001)


## Measured Timings
- Oracle: 25.47 us
- Compile (CDT): 35.74 us
- Ratio: 1.40x

## Classification: SCATTER_REDUCE (embedding backward)

## Summary

| Metric | Value |
|--------|-------|
| Baseline gap | 1.64x |
| Best with fix | 1.36x (scatter_add_into + multi_kernel=3) |
| Default config gap | 1.53x (scatter_add_into only) |
| Oracle kernels | 3 |
| Inductor kernels (before) | 5 |
| Inductor kernels (after fix) | 3 |

## Root Cause

Same pattern as sum_sum_sum_6a2ad206248c (Electra) but smaller tensors ([8, 512, 128], vocab=30000).
Layer-norm backward -> 3 scatters + 2 global reductions + add(mm_1, vocab_scatter).

## Fix Implemented

**Commit**: e034b931eab in /tmp/pytorch-work (branch pr-184905)

**Pass**: `config.scatter_add_into_fusion = True` (default enabled)

Rewrites: `add(A, index_put(zeros, idx, val, accumulate=True))` -> `index_put(A, idx, val, accumulate=True)`

Eliminates the full(0) init of [30000, 128] and the separate add kernel.

## Config Exploration

| Config | Ratio |
|--------|-------|
| Baseline (no fix) | 1.64x |
| scatter_add_into_fusion (default) | 1.53x |
| + multi_kernel=3 | 1.36x |

## Remaining Gap (1.36x with best config)

Same as Electra: REDUCTION_EPILOGUE_REREAD for the intermediate producer and
MULTI_OUTPUT_SHARED_REDUCTION for the global [128] sums. Requires scheduler-level
recomputation heuristic or cooperative atomic reduction.
