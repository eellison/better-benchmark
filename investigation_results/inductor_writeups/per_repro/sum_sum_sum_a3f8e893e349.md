# sum_sum_sum_a3f8e893e349

## Summary

- Model: LayoutLM layernorm/dropout backward embedding (scatter-reduce)
- Oracle: `oracle_layoutlm_embedding_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.076x (oracle 255.01us, compile 274.30us)
- Kernel count: Inductor multiple kernels, Oracle fused multi-destination scatter-reduce

## Root Cause

The oracle computes the complete LayoutLM layernorm/dropout backward return tuple, including both hidden-column reductions, all indexed embedding-gradient accumulators, and the vocabulary-gradient add into `mm_1[:-2]`, by recomputing the shared row producer once and scattering it directly.

Inductor materializes the rowwise producer, masks, generic `index_put(accumulate=True)` scatter intermediates, pairwise scatter adds, and final slice add as separate scheduled work.

The 1.076x gap is modest and comes from:
1. Extra kernel launch overhead for separate scatter passes
2. Materializing the rowwise producer intermediate
3. Not keeping row-local reductions live while feeding scatter targets

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 274.30 | 1.076x |
| multi_kernel=2 | 265.90 | 1.043x |
| multi_kernel=3 | 269.00 | 1.055x |

multi_kernel=2 reduces the gap slightly to 1.043x (below the 1.05 threshold). The gap is marginal and partially addressable with config.

## Fix Assessment

**Near floor with config** -- multi_kernel=2 brings the ratio to 1.043x, below the significance threshold. The remaining structural scatter-reduce gap is minimal for this workload.

### Files:
- `torch/_inductor/choices.py` (heuristic could prefer mk=2 for this shape)
