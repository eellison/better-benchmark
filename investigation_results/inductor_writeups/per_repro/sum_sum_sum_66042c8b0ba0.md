# sum_sum_sum_66042c8b0ba0

## Summary

- Model: RoBERTa embedding/layernorm-backward (scatter-reduce)
- Oracle: `oracle_roberta_embedding_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.718x (oracle 275.26us, compile 472.96us)
- Kernel count: Inductor multiple kernels, Oracle 1 fused scatter-reduce kernel

## Root Cause

The oracle computes the full RoBERTa embedding/layernorm-backward row producer once, including the two hidden reductions and all three `index_put(accumulate=True)` scatter-add outputs with the vocabulary add, whereas Inductor materializes the `[4,512,768]` producer and schedules the sibling sums, sentinel `where` masks, zero fills, three indexed accumulators, and vocabulary add as separate generic kernels.

The 1.718x gap comes from:
1. Materializing the full `[4,512,768]` intermediate
2. Running 3 separate scatter-add kernels for position/segment/vocab gradients
3. Not recognizing that row-local layernorm reduction scalars can stay live while feeding multiple indexed accumulator destinations

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 472.96 | 1.718x |
| multi_kernel=2 | 469.95 | 1.707x |
| multi_kernel=3 | 471.14 | 1.712x |

No meaningful improvement from multi_kernel configs. The structural scatter-reduce fusion gap dominates.

## Fix Assessment

**Design doc** -- requires SCATTER_REDUCE embedding-backward lowering.

### What's needed:
Add an embedding-backward scatter-reduce lowering that fuses the rowwise producer with hidden reductions, sentinel guards, indexed atomic accumulation, and vocabulary-gradient accumulation for the full return tuple. The scheduler/codegen needs a structured scatter-reduce template that keeps row-local layernorm reduction scalars live while feeding multiple indexed accumulator destinations of different sizes plus sibling hidden reductions.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (scatter-reduce pattern recognition)
- `torch/_inductor/scheduler.py` (fusion across scatter boundaries)
- `torch/_inductor/codegen/triton.py` (atomic accumulation codegen)
