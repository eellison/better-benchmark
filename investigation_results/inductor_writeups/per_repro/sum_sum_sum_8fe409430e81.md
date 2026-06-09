# sum_sum_sum_8fe409430e81

## Summary

- Model: Qwen row scatter-reduce (RMSNorm backward)
- Oracle: `oracle_qwen_row_scatter_reduce.py`
- Classification: BAD_ORACLE
- Ratio: 0.533x (oracle 184.10us, compile 98.18us)
- Status: Oracle is significantly slower than Inductor compile

## Root Cause

The oracle is nearly 2x slower than Inductor's compiled output. The oracle's fused Qwen row scatter-reduce approach (bfloat16 workload with RMSNorm backward) has suboptimal performance for this shape. Inductor's generic lowering with its autotuned reduction strategy significantly outperforms the hand-written oracle.

This likely indicates the oracle has register pressure or memory access pattern issues for the `[128, 2048]` bfloat16 workload with `[2048, 16384]` intermediate.

## Config Exploration

Not needed -- oracle is the much slower baseline.

## Fix Assessment

**No fix needed / BAD_ORACLE** -- Inductor already significantly outperforms the oracle. The oracle should be revised or this repro marked as having no optimization opportunity.
