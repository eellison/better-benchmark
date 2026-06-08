# sum_sum_sum_a431ef798d5d

## Classification: COOPERATIVE_SPLIT_K

## Oracle: oracle_cooperative_split_k.py

## Model: ViT layer-norm-backward with transposed gradient output

## Measurements

- Compiled (default + combo + cd): 77.8 us
- Compiled (multi_kernel=1): 85.4 us
- Compiled (multi_kernel=2): 84.5 us
- Compiled (multi_kernel=3): 85.6 us
- Oracle: 67.0 us
- Ratio: 1.161x
- Oracle correctness: PASS

## Diagnosis

The oracle computes the full ViT layer-norm-backward tuple using one row-tiled producer to share the LN-backward row reductions, write the returned non-contiguous `[768, 32768]` gradient transpose, and emit split-K partials for the three returned `[768]` column sums. Inductor schedules the row reductions, gradient materialization/permute, and sibling column reductions as separate generic pointwise and reduction kernels.

## Config exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (combo + cd) | 77.8 | 1.161x |
| multi_kernel=1 | 85.4 | 1.275x |
| multi_kernel=2 | 84.5 | 1.261x |
| multi_kernel=3 | 85.6 | 1.277x |

The default combo+cd config is actually the best. Multi_kernel makes it worse, likely because the graph needs combo_kernels fusion more than persistent reduction tuning.

## Root cause

Inductor lacks a cooperative split-K multi-output reduction template that can combine:
1. Row-local LN-backward reductions
2. Materialized transposed side-output store ([768, 32768])
3. Multiple small-output column accumulators ([768] each)

The oracle coordinates all of these in one producer/finalizer pair, avoiding intermediate materialization of the [32768, 768] producer tensor.

## Fix path

- Teach the scheduler to recognize compatible layer-norm-backward column reductions sharing the same `[N*S, H]` domain
- Fuse with split-K accumulation and dependent transposed side store
- File: `torch/_inductor/scheduler.py` (fusion across transpose + reduction boundaries)
- File: `torch/_inductor/choices.py` (cooperative split-K threshold)

## Status: needs_work (1.16x gap, multi_kernel does not help)

## Re-measurement (2026-06-08)

- Oracle: 65.31 us
- Compiled: 73.6 us
- Ratio: 1.127x (improved from 1.161x)

The split-K improvements reduced the gap slightly (from 16.1% to 12.7%). The remaining gap is
still the multi-output reduction fusion pattern -- sharing row-local reductions with transposed
side-output store and column accumulators in one kernel pair.
