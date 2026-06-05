# sum_sum_sum_cd459367a1c3

## Summary

- Model: BERT embedding backward (scatter-reduce)
- Oracle: `oracle_bert_embedding_scatter_reduce.py`
- Classification: AT_FLOOR
- Ratio: 0.985x (oracle 220.22us, compile 216.93us)
- Status: Inductor matches or exceeds oracle performance

## Root Cause

No meaningful performance gap. Inductor's compiled output is actually slightly faster than the oracle (1.5% faster). For this particular BERT embedding backward shape, Inductor's generic decomposed lowering with coordinate descent tuning already matches the oracle's proposed scatter-reduce fusion.

## Config Exploration

Not needed -- already at/below floor.

## Fix Assessment

**No fix needed** -- Inductor already achieves oracle-level (or better) performance for this pattern at this shape.
