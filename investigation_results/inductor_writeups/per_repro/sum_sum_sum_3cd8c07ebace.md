# sum_sum_sum_3cd8c07ebace

## Summary

- Model: MaxPool + BN + scatter-reduce
- Oracle: `oracle_maxpool_bn_scatter_reduce.py`
- Classification: BAD_ORACLE
- Ratio: 0.788x (oracle 603.97us, compile 476.06us)
- Status: Oracle is slower than Inductor compile

## Root Cause

The oracle is slower than Inductor's compiled output by ~21%. The oracle's fused maxpool/BN/scatter-reduce approach does not provide benefit over Inductor's generic lowering for this workload. The oracle likely has suboptimal memory access patterns or excessive register pressure from fusing too many operations.

## Config Exploration

Not needed -- oracle is the slower baseline.

## Fix Assessment

**No fix needed / BAD_ORACLE** -- Inductor already exceeds the oracle's performance for this pattern. The oracle should be revised or this repro should be marked as having no known optimization opportunity.
