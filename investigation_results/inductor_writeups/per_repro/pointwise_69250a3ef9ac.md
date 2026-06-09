# pointwise_69250a3ef9ac

## Summary

- Model: BERT GELU + dropout (layout)
- Oracle: `oracle_gelu_dropout_layout.py`
- Classification: BAD_ORACLE (BANDWIDTH_BOUND)
- Ratio: 0.948x (oracle 194.08us, compile 184.06us)
- Status: Oracle is slower than Inductor compile

## Root Cause

The oracle is slower than Inductor's compiled output by ~5%. The oracle itself classifies this as BANDWIDTH_BOUND: the scope is dominated by mandatory f32 read, exact-erf math, RNG/dropout mask, and f32 output store. Inductor already lowers this to the same practical one-pass pointwise envelope, so there's no optimization opportunity.

The stochastic output means exact comparison is skipped, making this "not_true_floor" by the oracle's own assessment.

## Config Exploration

Not needed -- oracle is the slower baseline.

## Fix Assessment

**No fix needed / BAD_ORACLE** -- Inductor already matches or exceeds the oracle for this bandwidth-bound stochastic pointwise pattern.
