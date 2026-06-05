# sum_sum_sum_a190850c7f80

## Summary

- Model: MaxPool + BN + scatter-reduce (contiguous variant)
- Oracle: `oracle_maxpool_bn_scatter_reduce_contiguous.py`
- Classification: BAD_ORACLE
- Ratio: 0.795x (oracle 380.03us, compile 301.95us)
- Status: Oracle is slower than Inductor compile

## Root Cause

The oracle is slower than Inductor's compiled output by ~21%. The oracle's fused maxpool/BN/scatter-reduce contiguous approach has suboptimal memory access patterns or register pressure for this shape. Inductor's generic scheduled work with autotuned kernels outperforms the hand-written fusion.

## Config Exploration

Not needed -- oracle is the slower baseline.

## Fix Assessment

**No fix needed / BAD_ORACLE** -- Inductor already outperforms the oracle. The oracle should be revised.
