# sum_8a66186d1ffe — BAD_ORACLE

## Summary
- **Model**: torchbench_alexnet_train_001
- **Pattern**: MaxPool-backward scatter_add + where + per-channel sum
- **Oracle**: `oracle_maxpool_scatter_sum.py`
- **Ratio**: 0.698x (oracle 216.96us vs compile 151.36us)
- **Classification**: BAD_ORACLE

## Result

The oracle is slower than Inductor's compiled output on this hardware. The scatter-reduce approach used by the oracle (which avoids dense buffer materialization) does not provide a speedup for this particular shape configuration `[512]` output channels. The overhead of the oracle's gather-mask-reduce tiling strategy exceeds the memory savings for this workload size.

No investigation needed — Inductor already wins.
