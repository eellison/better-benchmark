# amax_sum_any_e20fbe84102b

## Summary

- Model: Electra attention softmax/dropout
- Oracle: `oracle_full_attention_softmax_dropout.py`
- Classification: AT_FLOOR (stochastic, NEW_PATTERN)
- Ratio: 1.018x (oracle 233.12us, compile 237.31us)
- Status: Inductor matches oracle performance within noise

## Root Cause

No meaningful performance gap (1.8%). The oracle computes attention softmax/dropout with a fused persistent online-softmax template including tautological mask removal and row-all-masked guard. However, Inductor already achieves near-identical performance with its generic decomposed lowering.

The oracle classifies this as NEW_PATTERN (attention softmax/dropout lowering) but the actual measured gap is negligible. Exact stochastic equality is not established (dropout-dependent output skipped in comparison).

## Config Exploration

Not needed -- already at floor.

## Fix Assessment

**No fix needed** -- Inductor already achieves oracle-level performance for this pattern.
