# sum_sum_sum_b8d91626261d

## Summary

- Model: MT5 attention backward (scatter-reduce)
- Oracle: `oracle_mt5_attention_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.237x (oracle 96.00us, compile 118.72us)
- Kernel count: Inductor multiple kernels, Oracle fused attention scatter-reduce

## Root Cause

The oracle computes the complete four-output MT5 attention-backward tuple by fusing each rowwise softmax-backward producer with the returned dense dscore side output and the duplicate-index relative-position bucket accumulation.

Inductor materializes the dense dscore branches, separately reduces residual-plus-dscore tensors over batch, permutes/clones the `[128, 128, 6]` bucket values, and lowers both `index_put(accumulate=True)` operations as generic high-contention scatter kernels.

The 1.237x gap comes from:
1. Materializing dense dscore branches
2. Separate batch reductions and permute/clone operations
3. Generic high-contention scatter kernels instead of structured accumulation
4. Not modeling the rowwise softmax-backward producer feeding both side-output and scatter-reduce

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 118.72 | 1.237x |
| multi_kernel=2 | 90.40 | 0.942x |
| multi_kernel=3 | 89.60 | 0.933x |

**multi_kernel=3 closes the gap entirely** (ratio drops below 1.0x). multi_kernel=2 also closes it. This indicates the default persistent reduction heuristic is choosing a suboptimal strategy for this workload.

## Fix Assessment

**Addressable with config** -- multi_kernel=2 or multi_kernel=3 both close the gap completely. The issue is purely a heuristic problem in `choices.py` for choosing the reduction strategy.

### What's needed:
Improve the reduction strategy heuristic in `choices.py` to prefer looped/persistent reduction for this MT5 attention shape, which would eliminate the gap without structural changes.

### Files:
- `torch/_inductor/choices.py` (persistent threshold / reduction strategy heuristic)
