# sum_sum_sum_72d5ffa26b42

## Summary

- Model: Longformer layer-norm-backward/dropout embedding backward (scatter-reduce)
- Oracle: `oracle_longformer_embedding_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.183x (oracle 56.29us, compile 66.59us)
- Kernel count: Inductor multiple kernels, Oracle fused scatter-reduce template

## Root Cause

The oracle computes the complete Longformer layer-norm-backward/dropout return tuple, including both `[768]` hidden reductions and all three duplicate-index `index_put(accumulate=True)` embedding scatter outputs, by keeping the row producer live while feeding multiple accumulator destinations.

Inductor materializes the rowwise layer-norm backward producer and lowers the sibling sums plus three indexed scatters as separate generic kernels.

The 1.183x gap comes from:
1. Materializing the rowwise producer intermediate
2. Running separate scatter kernels for position/segment/vocab destinations
3. Not keeping row-local reductions live while feeding multiple scatter targets

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 66.59 | 1.183x |
| multi_kernel=2 | 57.60 | 1.023x |
| multi_kernel=3 | 55.90 | 0.993x |

**multi_kernel=3 closes the gap entirely** (ratio drops below 1.0x). This suggests Inductor's looped reduction strategy is already sufficient for this workload shape when properly configured.

## Fix Assessment

**Partially addressable with config** -- multi_kernel=3 (looped reduction) closes the gap to at-floor. The remaining structural scatter-reduce improvement is not needed for this shape.

Note: This suggests the default heuristic for choosing between persistent vs looped reductions may be suboptimal for this workload. The `choices.py` heuristic could be improved to prefer looped reduction for similar shapes.

### Files:
- `torch/_inductor/choices.py` (persistent vs looped reduction heuristic)
