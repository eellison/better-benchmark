# sum_sum_sum_9e60972dd685

## Summary

- Model: GPT-J layer-norm-backward + index_put (bandwidth-bound)
- Oracle: `oracle_multi_output_index_put.py`
- Classification: BANDWIDTH_BOUND
- Ratio: 1.083x (oracle 121.79us, compile 131.87us)
- Status: Near floor, bandwidth-bound workload

## Root Cause

The oracle computes the full GPT-J layer-norm-backward-shaped multi-output graph including the dense `[50400,4096]` `index_put(accumulate=True)` output. However, the oracle itself classifies this as BANDWIDTH_BOUND: the returned index_put tensor is a real dense contiguous output whose zero-fill/write dominates the runtime (~826 MB materialization).

Even a scatter-reduce lowering for the row-local producer cannot remove the required dense materialization. The 1.083x gap is marginal and dominated by memory bandwidth.

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 131.87 | 1.083x |
| multi_kernel=2 | error (multi_kernel cache_file_path) | N/A |
| multi_kernel=3 | 124.10 | 1.019x |

multi_kernel=3 reduces the gap to essentially at-floor (1.019x).

## Fix Assessment

**Near floor / BANDWIDTH_BOUND** -- The gap is marginal (8.3%) and dominated by mandatory memory bandwidth for the 826 MB output materialization. multi_kernel=3 closes most of the remaining gap. No structural fix needed.

### Files:
- `torch/_inductor/choices.py` (looped vs persistent heuristic could prefer mk=3 here)
