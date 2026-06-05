# sum_sum_sum_8dc0ac508cce

## Classification: SCATTER_REDUCE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_8dc0ac508cce/oracle_swin_scatter_reduce.py`
- Correctness: PASS
- Oracle: 195.52 us
- Compile (cd=True): 379.87 us
- Ratio: 1.943
- Best config: combo+mk=2: 368.54 us (ratio 1.895)
- Status: real_gap (LARGE)

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 379.87 | 1.943 |
| combo+mk=2 | 368.54 | 1.895 |
| combo+mk=3 | 367.49 | 1.888 |
| Oracle | 195.52 | 1.000 |

No config comes close to closing this nearly 2x gap.

## Root Cause

The oracle computes the complete Swin attention backward return tuple by reducing the two duplicate-index relative-position `index_put(accumulate=True)` outputs directly into their final [169, 8] buffers while the softmax-gradient producer writes the required [16384, 49, 49] tensor.

Inductor currently lowers the three large reductions, permutes/reshapes, softmax-gradient pointwise work, and duplicate-index scatters as separate generic scheduled regions. The scheduler/codegen does not recognize this multi-output rowwise softmax-backward producer feeding a structured duplicate-index scatter-reduce epilogue.

## Kernel count
- Oracle: 2 kernels (zero + fused softmax-backward + scatter-reduce)
- Inductor: 5+ kernels (separate reductions, permutes, scatters)

## Recommendation

This is the same SCATTER_REDUCE pattern documented for sum_sum_sum_041aaca79573 and the 87-repro Swin family. Requires a lowering that keeps the rowwise reduction producer in registers and accumulates duplicate relative-position rows directly into the final dense outputs.

File: `torch/_inductor/fx_passes/` (scatter_reduce fusion pass)
File: `torch/_inductor/scheduler.py` (structured scatter epilogue)
